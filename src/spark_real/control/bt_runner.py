"""Score execution entry point and the flat action loop with legato look-ahead."""

import logging
import threading
from typing import List

import numpy as np

from spark_real.control import execution_recovery
from spark_real.control.executor_types import (
    AbortRequested,
    ExecutionResult,
)
from spark_real.control.trajectory import TrajectoryRecorder

logger = logging.getLogger(__name__)


class BtRunnerMixin:
    """Score execution entry point and the flat action loop with legato look-ahead."""

    def execute_score(self, score: dict) -> List[ExecutionResult]:
        self._results = []
        # NOT an unconditional `self._abort = False`: a stop pressed after this
        # task was armed (during capture -> detect -> plan) must survive into
        # the run rather than be wiped by it.
        self._abort = self._abort_epoch != self._armed_epoch
        if self._abort:
            logger.warning(
                "execute_score: a stop was requested after this task was "
                "armed (epoch %d != %d); honouring it instead of starting",
                self._abort_epoch,
                self._armed_epoch,
            )
        self._running = True
        self._recorder = TrajectoryRecorder(self.robot)

        tree = score.get("tree", score)
        actions = self._flatten_tree(tree)
        # Recovery budget is scoped to ONE execute_score call: a closed-loop
        # pass gets its own six re-attempts, but a single pass can never spend
        # more however deeply the plan nests retry inside fallback.
        self._recovery_attempts_used = 0
        # Post-task verification and the replay path both want a plain list of
        # leaves: every leaf of every branch.
        leaf_actions = self._leaf_actions(actions)
        logger.info(
            "Executing %d actions (%d leaves)", len(actions), len(leaf_actions)
        )

        aborted = False
        try:
            try:
                self._run_actions(actions)
            except AbortRequested:
                logger.warning("Execution aborted by user")
                self._results.append(
                    ExecutionResult(
                        action_type="abort", success=False, message="Aborted by user"
                    )
                )
                aborted = True
                self._stop_robot()

            if self._pipeline is not None and not aborted:
                try:
                    self._run_post_task_verification(score, leaf_actions)
                except AbortRequested:
                    logger.warning("Post-task verification aborted by user")
                    aborted = True

            try:
                self._recorder.save()
            except Exception as e:
                logger.warning("Failed to save trajectory: %s", e)

            return self._results
        finally:
            self._running = False

    def _is_legato_blendable(self, action: dict) -> bool:
        """
        Return True if this action can be a legato note (vs barrier).
        """
        if action is None:
            return False
        atype = action.get("type", action.get("name", ""))
        if atype not in self._LEGATO_BLENDABLE:
            return False
        if atype == "move_to_keypoint" and not self._holding:
            return False
        return True

    @staticmethod
    def _peek_release_tilt(actions):
        """Params of the first ``release`` leaf in ``actions``, else None.

        Depth-first through selector/retry markers, in execution order. Lets
        the PLACE TRANSPORT know what dip the release will ask for: a tilt
        past the positive pitch budget (the wrist-camera clearance, 40 deg) is
        only achievable when the approach yaw already put the dip on the
        negative-pitch side (90 deg budget), see ReleaseMixin._dip_side_orient.
        """
        for node in actions or []:
            if not isinstance(node, dict):
                continue
            if node.get("_bt_selector"):
                for branch in node.get("branches") or []:
                    hit = BtRunnerMixin._peek_release_tilt(branch)
                    if hit is not None:
                        return hit
                continue
            if node.get("_bt_retry"):
                hit = BtRunnerMixin._peek_release_tilt(node.get("branch"))
                if hit is not None:
                    return hit
                continue
            if node.get("type", node.get("name", "")) == "release":
                return node.get("params", {}) or {}
        return None

    @staticmethod
    def _peek_place_label(actions):
        """Label of the next PLACE destination in ``actions``, else None.

        Sibling of _peek_release_tilt: it lets the grasp choose between the
        two equivalent grasp yaws knowing the destination. A parallel gripper
        closes identically at yaw and yaw+180, but the choice fixes which way
        the held object points, and therefore the yaw the place will need
        (which can land past the +-100 deg wrist limit and seat the tool
        END-FOR-END).
        """
        for node in actions or []:
            if not isinstance(node, dict):
                continue
            if node.get("_bt_selector"):
                for branch in node.get("branches") or []:
                    hit = BtRunnerMixin._peek_place_label(branch)
                    if hit is not None:
                        return hit
                continue
            if node.get("_bt_retry"):
                hit = BtRunnerMixin._peek_place_label(node.get("branch"))
                if hit is not None:
                    return hit
                continue
            params = node.get("params", {}) or {}
            kind = node.get("type", node.get("name", ""))
            if kind == "place_in_slot":
                return params.get("container_label")
            if kind == "move_to_keypoint":
                # The transport that delivers to the place is the FIRST
                # keypoint move after the grasp; the grasp's own approach has
                # already been consumed by the time this look-ahead runs.
                return params.get("keypoint_label")
        return None

    def _run_actions(self, actions):
        i = 0
        while i < len(actions):
            action = actions[i]
            # Refresh the release look-ahead every step so it can neither go
            # stale (a consumed release must stop steering later transports)
            # nor be missed (legato phrases and recovery both re-enter here).
            self._pending_release_tilt = self._peek_release_tilt(actions[i:])
            # Skip the current node: the NEXT destination is what matters.
            self._pending_place_label = self._peek_place_label(actions[i + 1:])
            if self._abort:
                logger.warning("Execution aborted by user")
                self._results.append(
                    ExecutionResult(
                        action_type="abort", success=False, message="Aborted by user"
                    )
                )
                break

            # Control-flow markers are handled first: they are not
            # dispatchable leaves, they own their children's failure handling,
            # and the flat path's recovery must not double up on theirs.
            if action.get("_bt_selector") or action.get("_bt_retry"):
                if action.get("_bt_selector"):
                    result = self._run_selector(action)
                else:
                    result = self._run_retry(action)
                self._results.append(result)
                self._last_action_failed = not bool(result.success)
                if not result.success:
                    i = self._skip_failed_cycle(actions, i)
                    continue
                i += 1
                continue

            # Legato look-ahead
            if self._is_legato_blendable(action):
                run = [action]
                j = i + 1
                while j < len(actions) and self._is_legato_blendable(actions[j]):
                    run.append(actions[j])
                    j += 1
                if len(run) >= 2:
                    logger.info(
                        "[%d-%d/%d] legato phrase: %s",
                        i + 1,
                        j,
                        len(actions),
                        [a.get("type") for a in run],
                    )
                    ok = self._execute_legato_phrase(run, i, len(actions))
                    if ok:
                        i = j
                        continue

            action_type = action.get("type", action.get("name", "unknown"))
            params = action.get("params", {})
            logger.info("[%d/%d] %s %s", i + 1, len(actions), action_type, params)

            if action_type == "move_to_keypoint":
                kp = params.get("keypoint_label", "")
                label = f"move_to {kp}" if kp else action_type
            elif action_type == "move_relative":
                label = "move_rel"
            else:
                label = action_type
            self._recorder.set_action_label(label)
            self._recorder.mark_transition(label)
            self._recorder.record()

            # Re-detect before approaching when previous action failed
            if (
                action_type in ("move_to_keypoint", "grasp_se3")
                and i > 0
                and not self._holding
                and self._pipeline is not None
                and getattr(self, "_last_action_failed", False)
            ):
                kp_label = params.get("keypoint_label", "")
                if kp_label and kp_label not in self._placed_labels:
                    self._join_redetect()
                    self._redetect_single(kp_label)

            if action_type == "move_to_keypoint" and not self._holding:
                self._last_pick_label = params.get("keypoint_label", "")
            elif action_type == "grasp_se3" and not self._holding:
                self._last_pick_label = params.get("keypoint_label", "")

            result = self._dispatch_action(action_type, params)
            self._results.append(result)
            self._last_action_failed = not bool(result.success)

            if result.success and action_type == "release":
                if (
                    self._last_pick_label
                    and self._pipeline is not None
                    and (
                        getattr(self, "strict_placement_verify", False)
                        # The per-release two-anchor verdict + rebind also
                        # rides this capture (verification.release_verdict).
                        or execution_recovery.release_verdict_enabled(self)
                    )
                ):
                    try:
                        execution_recovery.verify_placement(
                            self, self._last_pick_label, self._last_place_label
                        )
                    except Exception as e:
                        logger.warning("Post-release verification error: %s", e)
                if self._last_pick_label:
                    self._placed_labels.add(self._last_pick_label)
                    logger.info("Marked '%s' as placed", self._last_pick_label)
                    self._last_pick_label = ""
                if self._pipeline is not None:
                    remaining = actions[i + 1 :]
                    has_next_pick = any(
                        a.get("type") in ("move_to_keypoint", "grasp_se3")
                        for a in remaining
                    )
                    if has_next_pick:
                        # Never two re-detect threads at once.
                        self._join_redetect()
                        self._redetect_thread = threading.Thread(
                            target=self._redetect_all,
                            args=(actions, i),
                            daemon=True,
                        )
                        self._redetect_thread.start()
                        logger.info(
                            "Inter-grasp: skipping home, going direct to next pick"
                        )

            if not result.success:
                logger.warning("Action %d failed: %s", i + 1, result.message)
                msg_lower = (result.message or "").lower()
                if any(
                    k in msg_lower
                    for k in ("reflex", "singular", "discontinuity", "control")
                ):
                    try:
                        self._snapshot_failure_state(action_type, params, result)
                    except Exception as _snap_e:
                        logger.warning("Failure snapshot skipped: %s", _snap_e)
                recovery = self._attempt_recovery(
                    action_type, params, result, actions, i
                )
                if recovery is not None and recovery.success:
                    self._results[-1] = recovery
                    i += 1
                    continue
                logger.warning("Action failed, resetting")
                self._check_abort()
                still_holding = False
                if self._holding:
                    try:
                        still_holding = self._verify_grasp()
                        logger.info(
                            "Recovery: force-verify says %s; " "%s open the gripper",
                            "HOLDING" if still_holding else "EMPTY",
                            "skipping" if still_holding else "going to",
                        )
                    except Exception as _vexc:
                        logger.warning("Recovery verify failed: %s", _vexc)
                if not still_holding:
                    self.robot.open_gripper()
                    self._abort_sleep(0.3)
                    self._holding = False
                cur = self._get_current_position()
                if cur[2] < 0.45:
                    cur[2] = min(cur[2] + 0.10, 0.45)
                    self._move_to(cur, self.current_orientation())
                # Post-failure J5 escape from singularity
                try:
                    q_now = np.asarray(self.robot.get_joint_positions(), dtype=float)
                    if abs(q_now[4]) < 0.5:
                        q_escape = q_now[:7].copy()
                        q_escape[4] = -0.8
                        logger.info(
                            "Recovery: J5=%.3f near singular; "
                            "escape-nudge to -0.8 before next cycle",
                            float(q_now[4]),
                        )
                        self.robot.move_to_joint_config(
                            q_escape.tolist(), velocity=self.velocity * 0.5
                        )
                        try:
                            self._wait_stationary_after_servo()
                        except Exception:
                            pass
                except Exception as _e:
                    logger.debug("Recovery J5 escape skipped: %s", _e)
                # Retry-once on grasp failure
                if action_type in ("grasp", "grasp_se3"):
                    kp = params.get("keypoint_label", "")
                    if not hasattr(self, "_grasp_retry_count"):
                        self._grasp_retry_count = {}
                    retries = self._grasp_retry_count.get(kp, 0) if kp else 0
                    if kp and retries < 1:
                        self._grasp_retry_count[kp] = retries + 1
                        logger.info(
                            "[retry] grasp '%s' failed; retrying once with "
                            "fresh detection (attempt %d)",
                            kp,
                            retries + 2,
                        )
                        continue
                    if kp:
                        logger.info(
                            "[retry] grasp '%s' failed twice; giving up, "
                            "skipping cycle",
                            kp,
                        )
                    while i + 1 < len(actions):
                        next_type = actions[i + 1].get("type", "")
                        if next_type in ("grasp", "grasp_se3"):
                            break
                        i += 1
                        self._results.append(
                            ExecutionResult(
                                action_type=next_type,
                                success=False,
                                message="Skipped (prior grasp failed)",
                            )
                        )
                    logger.info("Skipped to next grasp cycle")

            i += 1
