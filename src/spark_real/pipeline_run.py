"""
Full task-run orchestration for the SPARK real-robot pipeline.

Houses the closed-loop ``run_task`` driver that ties capture, SAM3
detection, Gemini planning, and robot execution together. RunMixin is
mixed into SPARKRealPipeline alongside ExecutionMixin, so the cross-mixin
self.plan() / self.execute() / self.evaluate_scene_progress() calls
resolve against the assembled class via MRO.
"""

import logging
import re
import time
from datetime import datetime
from typing import List, Optional

import numpy as np
import yaml

from spark_real.control.success_predicates import UNVERIFIED
from spark_real.control.success_verifier import reset_run_state, task_success
from spark_real.pipeline_execution import (
    build_detection_details,
    verify_block_labels,
)
from spark_real.planning.plan_annotate import annotate_for_planner
from spark_real.pipeline_io import user_aborted
from spark_real.pipeline_closed_loop import ClosedLoopMixin
from spark_real.pipeline_types import TaskResult

logger = logging.getLogger(__name__)


# Full capture -> detect -> plan -> execute orchestration.
# Mixed into the pipeline class.
class RunMixin(ClosedLoopMixin):

    def run_task(
        self,
        instruction: str,
        prompts: List[str] = None,
        execute: bool = True,
        save_to_library: Optional[bool] = None,
    ) -> TaskResult:
        """
        Full pipeline: capture -> detect -> plan -> execute.

        When ``config.max_task_passes`` > 1 the pipeline runs closed-loop:
        after each behavior tree it re-perceives the scene, decides which
        targets still need handling (generic, any task), and re-plans over
        the fresh detections until the goal holds or the pass budget is
        spent. Single-pass behavior (max_task_passes == 1) is unchanged.
        """
        t0 = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info("Task: %s", instruction)

        # Clear the previous run's verdict HERE, not in execute(). A dry run,
        # a no-robot run or a capture failure never reaches execute(), and the
        # verdict read at the end of this method would then be the LAST run's
        # -- reporting a stale `pass` as this run's success.
        reset_run_state(getattr(self, "_executor", None))

        # Capture
        logger.info("[1/5] Capturing images...")
        with self.activity("capturing"):
            captures = self.capture()
        if not captures:
            return TaskResult(
                instruction=instruction,
                timestamp=timestamp,
                detections=[],
                plan={},
                execution_results=[],
                success=False,
                duration=time.time() - t0,
            )

        saved_paths = {}
        if self.config.save_captures:
            saved_paths = self._save_captures(captures, timestamp)

        # Detect: two-phase approach:
        #   Phase 1: If no prompts given, ask Gemini to look at the scene
        #            and decide what to detect.
        #   Phase 2: Run SAM3 with those prompts, then send annotated
        #            results back to Gemini for planning.
        # Collect EVERY fixed view, not the first one that exists. A single
        # camera foreshortens: an elongated tool lying away from that image
        # plane reads far rounder than it is. The API takes several images in
        # one request, so send all and let the model cross-check.
        # scene_cam stays the primary (first) view; the annotation overlay
        # is defined against exactly one camera.
        scene_img_raw = None
        scene_cam = None
        scene_raw_views = []
        for cam_name in ["birdview", "sideview"]:
            cam_data = captures.get(cam_name)
            if cam_data and cam_data.get("rgb") is not None:
                scene_raw_views.append((cam_name, cam_data["rgb"]))
                if scene_img_raw is None:
                    scene_img_raw = cam_data["rgb"]
                    scene_cam = cam_name

        use_multi_instance = False

        # Resolve the cached BT ONCE, up front. Two things ride on it:
        #   - the SAM3 detection prompts are harvested from the tree's own
        #     labels, which skips the Gemini prompt-generation call so a
        #     cache-only run is genuinely offline, and
        #   - plan() below is handed the SAME match, so the harvest and the
        #     executed tree can never disagree about which BT this run is.
        # In cache_only mode a miss raises BTCacheMiss out of run_task
        # rather than silently falling through to the planner.
        _plan_mode = self.plan_mode()
        cached_match = None
        if self._bt_library is not None:
            cached_match = self.resolve_cached_bt(instruction)
        if prompts is None and cached_match is not None:
            harvested = self._harvest_cached_prompts(cached_match)
            if harvested:
                prompts = harvested
                use_multi_instance = True  # treat as batch (matches the LLM path)
        # Task-prompt registry (configs/tasks/*.yaml): the task's frozen
        # perception contract -- pinned vocabulary, declared object counts,
        # and a deterministic geometric instance ordering. Sits ahead of BOTH
        # the regex extractor and Gemini because it is the only prompt source
        # that is identical run to run; generate_prompts runs at temperature
        # 0.3, so the prompt VOCABULARY itself drifts, which is exactly the
        # non-repeatability a demo corpus must not have.
        #
        # A miss is not an error: unregistered instructions fall through to
        # the paths below unchanged.
        task_spec = self.task_spec(instruction)
        # Tier-2 success override (spec section 1.5). Set even when the task supplies no
        # prompts, and CLEARED otherwise so the previous task's predicate does
        # not leak into this one. The planner's own `verify:` still wins.
        if self._executor is not None:
            self._executor._task_verify_block = (
                dict(task_spec.verify) if task_spec is not None and task_spec.verify else None
            )
            if task_spec is not None and task_spec.verify:
                logger.info(
                    "[2/5] task registry %r: verify block from %s",
                    task_spec.task,
                    task_spec.source_path,
                )
        if prompts is None and task_spec is not None:
            prompts = list(task_spec.prompts)
            use_multi_instance = True
            logger.info(
                "[2/5] task registry %r: prompts %s (frozen in version "
                "control -- no LLM call)",
                task_spec.task,
                prompts,
            )
        if prompts is None and _plan_mode == "cache_only":
            # Strict offline: never fall through to the planner for prompts.
            # The regex extractor is a legitimate offline fallback; Gemini is
            # not. The task registry above covers the demo tasks without
            # either; this is the backstop for an unregistered instruction.
            prompts = self._extract_prompts(instruction)
            logger.info(
                "[2/5] plan_mode=cache_only: no cached BT labels to harvest; "
                "using regex prompts %s (no LLM call)",
                prompts,
            )
        if prompts is None:
            if self._planner is not None and scene_img_raw is not None:
                logger.info("[2/5] Asking Gemini for detection prompts...")
                try:
                    with self.activity("planning"):
                        prompts = self._planner.generate_prompts(instruction, scene_img_raw)
                    use_multi_instance = True  # Gemini prompts -> batch task -> multi-instance
                    logger.info("  Gemini prompts: %s", prompts)
                except Exception as e:
                    logger.warning(
                        "  Gemini prompt generation failed: %s, falling back to regex",
                        e,
                    )
                    prompts = self._extract_prompts(instruction)
            else:
                prompts = self._extract_prompts(instruction)

        # SAM3 detection: multi-instance only for Gemini-generated prompts (batch tasks).
        # User-provided prompts get top-1 per prompt (specific objects).
        logger.info(
            "[3/5] Detecting objects: %s (multi_instance=%s)",
            prompts,
            use_multi_instance,
        )
        with self.activity("detecting"):
            if task_spec is not None:
                # Registered task: detect under its contract. Adds the count
                # gate (with the relax-then-alt-prompts escalation) and the
                # declared instance ordering, so "blue block 1" is the same
                # physical block every run and a cached BT keeps binding.
                # Raises PromptCountMismatch when the gate fails and the
                # task's policy is `abort` -- deliberately fatal, because a
                # mislabeled episode poisons a training set silently.
                (
                    detections,
                    all_detections,
                    _resolution,
                    task_spec,
                    prompts,
                ) = self.detect_for_task(
                    captures, instruction, spec=task_spec, prompts=prompts
                )
            else:
                all_detections = self.detect(
                    captures, prompts, multi_instance=use_multi_instance
                )
                detections = self.merge_detections(all_detections)
            # Enrich container-like detections with slot poses + world-
            # frame major-axis orientation so the place_in_slot primitive
            # can drop heterogeneous items into distinct slots with
            # the gripper yaw aligned to the container's long axis.
            try:
                self._enrich_with_slots(detections, captures)
            except Exception as _e:
                logger.warning("slot enrichment failed (continuing without slots): %s", _e)
        logger.info(
            "  Found %d unique objects (%d total across cameras)",
            len(detections),
            len(all_detections),
        )
        for d in detections:
            pos_str = (
                f"({d.position_3d[0]:.3f}, {d.position_3d[1]:.3f}, {d.position_3d[2]:.3f})"
                if d.position_3d is not None
                else "N/A"
            )
            logger.info(
                "  - %s: conf=%.3f, pos=%s, cam=%s",
                d.label,
                d.confidence,
                pos_str,
                getattr(d, "camera", "?"),
            )

        # Plan: send Gemini the annotated image + detection details
        # so it can filter false positives and generate the plan.
        logger.info("[4/5] Generating plan...")
        scene_img_annotated = None
        if scene_img_raw is not None:
            cam_dets = [d for d in all_detections if getattr(d, "camera", None) == scene_cam]
            # Planner-specific overlay: adds the OBB rectangle and major-axis
            # arrow on top of the tint+contour, so the model can SEE whether an
            # object has a meaningful long axis. _annotate_image stays as-is
            # for the UI and the detection routes.
            scene_img_annotated = annotate_for_planner(
                scene_img_raw, cam_dets if cam_dets else detections
            )

        # Annotate each view against ITS OWN camera's detections -- an overlay
        # drawn from another camera's pixels would be worse than no overlay.
        scene_views = []
        for _cam, _raw in scene_raw_views:
            _dets = [d for d in all_detections if getattr(d, "camera", None) == _cam]
            if not _dets:
                continue
            try:
                scene_views.append((_cam, annotate_for_planner(_raw, _dets)))
            except Exception as _exc:  # noqa: BLE001
                logger.warning("[plan] could not annotate %s: %s", _cam, _exc)
        if len(scene_views) > 1:
            logger.info(
                "[plan] sending %d camera views to the planner: %s",
                len(scene_views),
                ", ".join(c for c, _ in scene_views),
            )
        elif scene_img_annotated is not None:
            scene_views = [(scene_cam, scene_img_annotated)]

        detection_details = build_detection_details(detections)

        with self.activity("planning"):
            score = self.plan(
                instruction,
                detections,
                scene_image=(scene_views if scene_views else scene_img_annotated),
                detection_details=detection_details,
                scene_camera=scene_cam,
            )
        plan_source = getattr(self, "_last_plan_source", None) or "llm"
        bt_hash = getattr(self, "_last_bt_hash", None)
        logger.info(
            "  Plan (source=%s, bt=%s):\n%s",
            plan_source,
            bt_hash or "-",
            yaml.dump(score, default_flow_style=False),
        )

        exec_results = []
        if execute and self._robot is not None:
            logger.info("[5/5] Executing on robot...")
            exec_results = self.execute(score, detections)
        else:
            logger.info("[5/5] Skipping execution (dry run or no robot)")

        exec_results, goal_unhandled = self._closed_loop_passes(
            instruction, prompts, score, exec_results, execute, plan_source, bt_hash
        )

        # Task success is the verifier's verdict and nothing else. An
        # ExecutionResult says a primitive RAN, not that the task was
        # achieved: a "last action succeeded" fallback banks a fork dropped
        # beside the tray as a success.
        outcome = getattr(self._executor, "verify_outcome", None) if self._executor else None
        success = task_success(outcome)
        verify_status = outcome.status if outcome is not None else UNVERIFIED
        # Closed-loop ground truth can only ever REMOVE a pass: a target still
        # outside the destination means the task did not succeed, whatever the
        # predicate said.
        if goal_unhandled:
            if success:
                logger.warning(
                    "Verifier said pass but %d target(s) are still outside the "
                    "destination %s; downgrading to fail",
                    len(goal_unhandled),
                    goal_unhandled,
                )
            success = False
            verify_status = "fail"

        success, verify_status, outcome = self._llm_last_resort(
            instruction, prompts, success, verify_status, outcome, goal_unhandled, exec_results
        )
        duration = time.time() - t0

        result = TaskResult(
            instruction=instruction,
            timestamp=timestamp,
            detections=[
                {
                    "label": d.label,
                    "confidence": d.confidence,
                    "position_3d": (
                        d.position_3d.tolist()
                        if isinstance(d.position_3d, np.ndarray)
                        else d.position_3d
                    ),
                    "centroid_2d": d.centroid_2d,
                    "depth_meters": d.depth_meters,
                    "orientation_angle": d.orientation_angle,
                    "aspect_ratio": d.aspect_ratio,
                    "obb_minor_m": float(getattr(d, "obb_minor_m", 0.0) or 0.0),
                }
                for d in detections
            ],
            plan=score,
            execution_results=[
                {
                    "action": r.action_type,
                    "success": r.success,
                    "message": r.message,
                    "duration": r.duration,
                }
                for r in exec_results
            ],
            success=success,
            duration=duration,
            captures=saved_paths,
        )

        # Plan provenance for the run record.
        result.plan_source = plan_source
        result.bt_hash = bt_hash
        result.label_resolutions = dict(getattr(self, "_last_label_resolutions", None) or {})
        # The verdict itself, so /scores and the recorded episode can show WHY
        # a run passed or failed rather than just a bare boolean.
        result.verify = outcome.to_dict() if outcome is not None else None
        result.verify_status = verify_status

        self._task_history.append(result)
        self._save_result(result, timestamp)
        # Automatic success feedback. ``executed`` is the guard that keeps a
        # dry run out of the cache; ``plan_source``/``bt_hash`` route a
        # cache-served run to bump()+alias instead of minting a duplicate.
        self._maybe_save_bt(
            instruction,
            score,
            detections,
            success,
            save_to_library=save_to_library,
            executed=bool(exec_results),
            plan_source=plan_source,
            bt_hash=bt_hash,
            user_aborted=user_aborted(exec_results),
            verify_status=verify_status,
        )
        logger.info(
            "Task complete (%.1fs, verify=%s, success=%s, plan_source=%s)",
            duration,
            verify_status,
            success,
            plan_source,
        )
        return result

    def _harvest_cached_prompts(self, cached_match):
        """SAM3 prompts from a cached BT's labels and verify block, or None.

        BTs carry instance-suffixed labels ("knife handle 1"); SAM3 wants the
        de-instanced prompt, so trailing " <N>" is stripped and the list
        deduplicated in insertion order.
        """
        cached_instr = cached_match.entry.instruction
        cached_score = cached_match.score
        if cached_score is not None:
            # BTs carry instance-suffixed labels like "knife handle 1",
            # "knife handle 2". SAM3 wants the de-instanced prompt
            # ("knife handle") and produces those numbered labels
            # itself in multi-instance mode. Strip trailing " <N>"
            # and dedupe while preserving insertion order so the
            # prompt list is deterministic.
            seen = set()
            cached_labels = []

            def _harvest(node):
                if not isinstance(node, dict):
                    return
                p = node.get("params") or {}
                # Pull any label-shaped param so the harvested
                # prompt set covers every label the BT will look up
                # at runtime (place_in_slot's container_label, e.g.
                # "tray"), not just the grasp's keypoint_label.
                for k in (
                    "keypoint_label",
                    "container_label",
                    "target_label",
                    "area_label",
                    "source_label",
                ):
                    lbl = p.get(k)
                    if not lbl:
                        continue
                    base = re.sub(r"\s+\d+$", "", lbl).strip()
                    if base and base not in seen:
                        seen.add(base)
                        cached_labels.append(base)
                for child in node.get("children") or []:
                    _harvest(child)

            _harvest((cached_score or {}).get("tree", {}))
            # Predicate labels sit outside the tree. Without them a
            # cache-served run detects only what it manipulates, and the
            # verifier abstains in every camera -> `unverified` forever.
            for lbl in sorted(verify_block_labels(cached_score)):
                base = re.sub(r"\s+\d+$", "", lbl).strip()
                if base and base not in seen:
                    seen.add(base)
                    cached_labels.append(base)
            if cached_labels:
                logger.info(
                    "[2/5] cache %s: harvested prompts %s from cached BT "
                    "(matched %r) -- no prompt-generation LLM call",
                    cached_match.source,
                    cached_labels,
                    cached_instr,
                )
            return cached_labels or None
        return None
