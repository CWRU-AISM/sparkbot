"""Closed-loop passes and the LLM last-resort rung of run_task."""

import copy
import logging
import time

import yaml

from spark_real.control.success_predicates import UNVERIFIED
from spark_real.control.success_verifier import task_success
from spark_real.pipeline_execution import CACHE_PLAN_SOURCES, build_detection_details
from spark_real.planning.plan_annotate import annotate_for_planner

logger = logging.getLogger(__name__)


class ClosedLoopMixin:
    """Outer passes after the first tree, and the one-shot LLM replan."""

    def _closed_loop_passes(
        self, instruction, prompts, score, exec_results, execute, plan_source, bt_hash
    ):
        """Re-perceive, re-plan and re-execute until the goal holds.

        Returns (all execution results, targets still unhandled or None).
        """
        # Closed-loop outer passes (generic, all families). Re-perceive the
        # scene, decide what targets remain unhandled, re-plan over the
        # fresh detections, and re-execute until the goal holds or the pass
        # budget is spent. Already-handled objects stay closed: the
        # destination filter in execute() drops items inside the receptacle
        # and the executor's _placed_labels persist across passes, so the
        # re-plan never re-visits a node already taken care of. Newly
        # perturbed objects (added utensil, swapped receptacle) are picked
        # up because each pass detects fresh.
        all_exec_results = list(exec_results)
        max_passes = max(1, int(getattr(self.config, "max_task_passes", 1)))
        did_execute = bool(exec_results) and execute and self._robot is not None
        last_score = score
        prev_unhandled = None
        stale = 0
        pass_idx = 1
        while did_execute and pass_idx < max_passes:
            if getattr(self._executor, "_abort", False):
                logger.info("[closed-loop] abort flag set; stopping passes")
                break
            settle = float(getattr(self.config, "closed_loop_settle_s", 0.8))
            if settle > 0:
                time.sleep(settle)
            handled = set(getattr(self._executor, "_placed_labels", set()) or set())
            with self.activity("detecting"):
                progress = self.evaluate_scene_progress(last_score, prompts, handled_labels=handled)
            unhandled = progress["unhandled"]
            if not unhandled:
                logger.info(
                    "[closed-loop] goal holds after pass %d " "(handled=%s, destinations=%s)",
                    pass_idx,
                    progress["handled"],
                    progress["destinations"],
                )
                break
            # No-progress guard: if the unhandled count is not shrinking,
            # stop rather than loop on an unreachable or mis-detected target.
            n = len(unhandled)
            if prev_unhandled is not None and n >= prev_unhandled:
                stale += 1
            else:
                stale = 0
            prev_unhandled = n
            if stale >= 2:
                logger.warning(
                    "[closed-loop] no progress over 2 passes " "(%d unhandled: %s); stopping",
                    n,
                    unhandled,
                )
                break
            logger.info(
                "[closed-loop] pass %d/%d: %d target(s) still "
                "unhandled %s -> re-planning over fresh scene",
                pass_idx + 1,
                max_passes,
                n,
                unhandled,
            )
            dets2 = progress["detections"]
            if not dets2:
                break
            details2 = build_detection_details(dets2)
            # Re-plan. When the first pass was cache-served, reuse the SAME
            # tree rather than re-resolving: the single-shot UI pin was
            # already consumed by pass 1, so a re-resolve here would fall
            # through to the LLM mid-run.
            if bt_hash and plan_source in CACHE_PLAN_SOURCES:
                score2 = copy.deepcopy(self._bt_library.get_score(bt_hash) or last_score)
                logger.info(
                    "[closed-loop] reusing cache-served BT %s for pass %d "
                    "(no re-resolve, no LLM call)",
                    bt_hash,
                    pass_idx + 1,
                )
            else:
                with self.activity("planning"):
                    score2 = self.plan(instruction, dets2, detection_details=details2)
            logger.info(
                "[closed-loop] re-plan:\n%s",
                yaml.dump(score2, default_flow_style=False),
            )
            results2 = self.execute(score2, dets2)
            all_exec_results.extend(results2)
            last_score = score2
            pass_idx += 1
            if not results2:
                break

        exec_results = all_exec_results
        goal_unhandled = None
        if max_passes > 1 and did_execute:
            logger.info(
                "[closed-loop] completed %d pass(es), %d total actions",
                pass_idx,
                len(exec_results),
            )
            # Final ground-truth check: the last pass executed actions that were
            # never re-perceived, so confirm the goal physically holds before
            # reporting success. A target still outside the destination means
            # the task failed regardless of what the per-action verify claimed.
            settle = float(getattr(self.config, "closed_loop_settle_s", 0.8))
            if settle > 0:
                time.sleep(settle)
            handled = set(getattr(self._executor, "_placed_labels", set()) or set())
            with self.activity("detecting"):
                final_progress = self.evaluate_scene_progress(
                    last_score, prompts, handled_labels=handled
                )
            goal_unhandled = final_progress["unhandled"]
            if goal_unhandled:
                logger.warning(
                    "[closed-loop] goal NOT met: %d target(s) still outside " "destination %s",
                    len(goal_unhandled),
                    goal_unhandled,
                )
            else:
                logger.info("[closed-loop] goal met: all targets handled")
        return exec_results, goal_unhandled

    def _llm_last_resort(
        self, instruction, prompts, success, verify_status, outcome, goal_unhandled, exec_results
    ):
        """Terminal recovery rung: adjudicate, else replan once with the failure.

        Returns (success, verify_status, outcome); extends exec_results.
        """
        # LLM REPLAN -- THE TERMINAL RUNG, at most once per task. The recovery
        # ladder below this point is deliberately LLM-free (re-bind ->
        # search_keypoint -> predicate-gated replay); this rung exists because
        # a ladder that ends in "give up" wastes the one planner call that
        # could have fixed a genuinely wrong PLAN. The affordable place for an
        # in-loop VLM call (cf. ReKep, 2409.01652) is HERE, after everything
        # cheap has been exhausted, with the failure evidence in hand.
        # Gated on real evidence (verify FAIL or the
        # closed-loop goal check), never on an abstain: replanning on no
        # evidence is how a held object gets dropped on a guess.
        if (
            not success
            and bool(getattr(self.config, "llm_last_resort", True))
            and (verify_status == "fail" or goal_unhandled)
            and self._planner is not None
        ):
            try:
                why = (
                    f"verify={verify_status}"
                    + (f"; unplaced targets: {sorted(goal_unhandled)}"
                       if goal_unhandled else "")
                    + (f"; reason: {outcome.reason}"
                       if outcome is not None and outcome.reason else "")
                )
                logger.warning(
                    "[llm-last-resort] all LLM-free recovery exhausted (%s); "
                    "consulting the planner ONCE with the failure evidence",
                    why,
                )
                retry_instruction = (
                    f"{instruction}\n\nIMPORTANT: a previous attempt at this "
                    f"task just FAILED ({why}). The scene below is the CURRENT "
                    "state -- some steps may already be partially done. Plan "
                    "only what remains, and prefer a DIFFERENT approach to "
                    "whatever failed."
                )
                with self.activity("detecting"):
                    captures2 = self.capture()
                    dets2 = self.merge_detections(
                        self.detect(captures2, prompts)
                    )
                cam2 = "birdview" if "birdview" in captures2 else next(iter(captures2))
                raw_views = [
                    (c, captures2[c].get("rgb"))
                    for c in ("birdview", "sideview")
                    if c in captures2 and captures2[c].get("rgb") is not None
                ] or [(cam2, captures2[cam2].get("rgb"))]

                # STEP 1 -- ADJUDICATE before assuming failure. The verifier's
                # false negatives are a first-class failure mode (a correct
                # placement can be reported failed), so the terminal rung's
                # first question is "is this actually
                # fine?", asked on CLEAN frames -- overlay paint sits exactly
                # on the objects whose state is in dispute. A confident YES
                # here means no motion at all: the worst outcome of replanning
                # a completed task is the robot un-completing it.
                proprio = getattr(self._executor, "_release_proprio", None) or {}
                proprio_hard_fail = proprio.get("verdict") == "fail"
                judged = self._planner.assess_completion(
                    instruction, raw_views, evidence=why
                )
                if (
                    judged["complete"]
                    and judged["confidence"] >= 0.7
                    and not proprio_hard_fail
                ):
                    # The arm's own evidence outranks the LLM: a hard proprio
                    # fail (jaws never opened / commanded tilt dropped) means
                    # the pretty picture is of a task that did not execute.
                    success = True
                    verify_status = "pass"
                    logger.warning(
                        "[llm-last-resort] Gemini adjudicated COMPLETE "
                        "(conf %.2f): %s -- overriding verify=%s, no replan",
                        judged["confidence"], judged["reason"], why,
                    )
                else:
                    if judged["complete"]:
                        logger.info(
                            "[llm-last-resort] adjudication said complete but "
                            "was overruled (conf %.2f, proprio_fail=%s)",
                            judged["confidence"], proprio_hard_fail,
                        )
                    # STEP 2 -- REPLAN. Both image forms go to the planner:
                    # the CLEAN frame so nothing is painted over, and the
                    # ANNOTATED frame because the plan must reference the
                    # numbered keypoints -- seeing the failure and naming the
                    # fix are different needs.
                    scene_views3 = list(raw_views)
                    if raw_views[0][1] is not None:
                        scene_views3.append(
                            (raw_views[0][0] + " (annotated keypoints)",
                             annotate_for_planner(raw_views[0][1], dets2))
                        )
                    with self.activity("planning"):
                        score3 = self.plan(
                            retry_instruction, dets2, scene_image=scene_views3,
                            scene_camera=raw_views[0][0],
                            # a fresh sample, never the cached tree that failed
                            temperature=0.4,
                        )
                    results3 = self.execute(score3, dets2)
                    outcome = getattr(self._executor, "verify_outcome", None)
                    success = task_success(outcome)
                    verify_status = (
                        outcome.status if outcome is not None else UNVERIFIED
                    )
                    logger.info(
                        "[llm-last-resort] replan executed: verify=%s",
                        verify_status,
                    )
                    exec_results.extend(results3 or [])
            except Exception as exc:  # noqa: BLE001 - the rung must not mask the original failure
                logger.warning("[llm-last-resort] replan failed: %s", exc)
        return success, verify_status, outcome
