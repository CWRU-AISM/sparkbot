import logging
import re
import time
from typing import Dict, List

import numpy as np

from spark_core.sam3 import ObjectDetection
from spark_real.bt_label_resolver import LabelResolvingDetectionMap
from spark_real.control import success_verifier
from spark_real.control.safe_robot import Obstacle, SafeRobot
from spark_real.control.score_executor import ExecutionResult
from spark_real.perception.detection_fusion import fusion_map_fields
from spark_real.pipeline_plan import PlanMixin  # noqa: F401
from spark_real.plan_inputs import (  # noqa: F401
    CACHE_PLAN_SOURCES,
    LEGACY_DISABLE_LLM_ENV,
    PLAN_MODE_ENV,
    VERIFY_LABEL_KEYS,
    _bt_frozen,
    build_detection_details,
    verify_block_labels,
)

logger = logging.getLogger(__name__)


# Plan, execute, and full run_task orchestration.
# Mixed into the pipeline class.
class ExecutionMixin(PlanMixin):


    def execute(self, score: Dict, detections: List[ObjectDetection]) -> List[ExecutionResult]:
        """
        Execute a SPARK score on the robot.
        """
        if self._executor is None:
            logger.warning("No robot connected; cannot execute")
            return []

        # Clear the previous run's verdict, gates and release witness. The
        # executor outlives a task, so without this a stale `pass` from the
        # last run would be read as this run's success.
        success_verifier.reset_run_state(self._executor)

        # Apply planner label corrections to detections before building map.
        #
        # This pop is why a cached score must be hashed AFTER execution:
        # label_corrections is per-run bookkeeping that RENAMES live
        # detections, so replaying it blind in a different scene renames the
        # wrong object. bt_library strips the key before hashing, and
        # _maybe_save_bt stores the post-pop score, so the served hash and
        # the stored hash agree.
        corrections = score.pop("label_corrections", None) or {}
        if corrections:
            logger.info("Applying %d label correction(s) from the planner:", len(corrections))
            correction_map = {str(k): str(v) for k, v in corrections.items()}
            for det in detections:
                if det.label in correction_map:
                    new_label = correction_map[det.label]
                    logger.info("  %s -> %s", det.label, new_label)
                    det.label = new_label

        # Label-drift tolerant map: a cached BT that says "blue block 2" or
        # "grey block" still binds when this run's SAM3 pass produced
        # "blue block" / "gray block". Exact hits behave identically to a
        # plain dict; see bt_label_resolver.
        detection_map = LabelResolvingDetectionMap()
        for det in detections:
            if det.position_3d is not None:
                detection_map[det.label] = {
                    "position_3d": (
                        det.position_3d.tolist()
                        if isinstance(det.position_3d, np.ndarray)
                        else det.position_3d
                    ),
                    "confidence": det.confidence,
                    "orientation_angle": det.orientation_angle,
                    "aspect_ratio": det.aspect_ratio,
                    # OBB short-axis length in METERS. Powers the
                    # mask-derived grasp-width prior in score_executor so
                    # the executor can synthesize a target_width for
                    # objects with no per-label calibration. 0.0 means no
                    # depth available, so the executor falls back to default.
                    "obb_minor_m": float(getattr(det, "obb_minor_m", 0.0) or 0.0),
                    # True when the mask still trips a quality trigger after
                    # reprompting. Forces the `auto` grasp strategy top-down.
                    "low_quality": bool(getattr(det, "low_quality", False)),
                    "reprompt_attempts": int(getattr(det, "reprompt_attempts", 0) or 0),
                    # Slot detection fields, populated by
                    # _enrich_with_slots for container labels (tray,
                    # plate, dish, bowl, drawer, dustpan). place_in_slot
                    # reads these to compute slot pose + gripper yaw.
                    "world_major_axis_rad": (
                        float(getattr(det, "world_major_axis_rad", None))
                        if getattr(det, "world_major_axis_rad", None) is not None
                        else None
                    ),
                    "slots": (
                        [
                            {
                                **s,
                                "world_xyz": (
                                    s["world_xyz"].tolist()
                                    if isinstance(s.get("world_xyz"), np.ndarray)
                                    else s.get("world_xyz")
                                ),
                            }
                            for s in (getattr(det, "slots", None) or [])
                        ]
                        if getattr(det, "slots", None)
                        else None
                    ),
                    "_mask": det.mask,
                    "_camera": getattr(det, "camera", None),
                }
                # Detector-fusion stamps (fused_confidence, axis_trust). Same
                # emit-only-when-measured convention as obb_confidence below;
                # detection_fusion.fusion_map_fields owns the key names.
                detection_map[det.label].update(fusion_map_fields(det))
                # How trustworthy the OBB axes are. Emitted ONLY when actually
                # measured: the grasp resolver treats an ABSENT key as unknown
                # and a PRESENT low value as a veto, so publishing a default
                # 0.0 for depth-less detections would kill every oriented grasp.
                _obb_conf = float(getattr(det, "obb_confidence", 0.0) or 0.0)
                if _obb_conf > 0.0:
                    detection_map[det.label]["obb_confidence"] = _obb_conf
                # Colour-free role names published by the task prompt
                # registry (e.g. "same color block 1"). Registered as
                # aliases rather than as extra dict entries so the
                # destination filter and the obstacle push still enumerate
                # only real detections. Without this a cached BT that
                # addresses a role cannot resolve it at all.
                detection_map.register_aliases(
                    getattr(det, "role_labels", None), det.label
                )

        # Container-as-destination filter:
        # If the score sends grasped items INTO a container (the
        # container shows up as a move_to_keypoint AFTER a grasp), then
        # any other detections that overlap the container's XY footprint
        # are already-placed items and should be dropped from grasp
        # targets. For OUT-OF-container tasks (no grasp-to-container flow),
        # we keep all detections so the user can grasp items inside.
        destination_labels = set()
        try:
            # Flatten the score (handles nested sequence trees and flat
            # action lists) so container/receptacle targets are found
            # regardless of how the planner nested them.
            actions = self._flatten_score_actions(score)
            seen_grasp = False
            for act in actions:
                atype = act.get("type", "") if isinstance(act, dict) else ""
                params = act.get("params", {}) if isinstance(act, dict) else {}
                # Explicit container/receptacle params on any primitive
                # (place_in_slot container_label, sweep/pour target_label,
                # bimanual dustpan_label) are always destinations.
                for key in ("container_label", "target_label", "dustpan_label"):
                    lbl = params.get(key)
                    if lbl:
                        destination_labels.add(lbl)
                if atype.startswith("grasp") or atype == "pick":
                    seen_grasp = True
                elif atype == "move_to_keypoint" and seen_grasp:
                    kp = params.get("keypoint_label")
                    if kp:
                        destination_labels.add(kp)
                elif atype in ("release", "drop", "place"):
                    seen_grasp = False
        except Exception as _e:
            logger.debug("destination-label scan skipped: %s", _e)

        # Labels the plan itself references must NEVER be filtered: a
        # mug's own handle sits inside the mug's footprint, so a put-back
        # move_to_keypoint that makes the mug a destination would otherwise
        # drop 'mug handle' as already-placed cargo and kill the grasp.
        referenced_labels = set()
        try:
            for act in self._flatten_score_actions(score):
                params = act.get("params", {}) if isinstance(act, dict) else {}
                for key in (
                    "keypoint_label",
                    "container_label",
                    "target_label",
                    "source_label",
                    "dustpan_label",
                ):
                    lbl = params.get(key)
                    if lbl:
                        referenced_labels.add(str(lbl))
                for lbl in str(params.get("object_labels", "") or "").split(","):
                    if lbl.strip():
                        referenced_labels.add(lbl.strip())
            # A predicate's subject must survive the destination filter too:
            # filtering it away leaves the verifier with nothing to detect.
            referenced_labels |= verify_block_labels(score)
        except Exception:
            pass

        for dest_label in destination_labels:
            dest_det = detection_map.get(dest_label)
            if dest_det is None or dest_det.get("position_3d") is None:
                continue
            dest_pos = np.asarray(dest_det["position_3d"], dtype=float)
            dest_minor = float(dest_det.get("obb_minor_m") or 0.0)
            dest_radius = max(0.12, dest_minor * 0.6)
            to_drop = []
            for label, det in detection_map.items():
                if label == dest_label or label in referenced_labels:
                    continue
                pos = det.get("position_3d")
                if pos is None:
                    continue
                d_xy = float(np.linalg.norm(np.asarray(pos[:2], dtype=float) - dest_pos[:2]))
                if d_xy < dest_radius:
                    to_drop.append((label, d_xy))
            for label, d_xy in to_drop:
                logger.info(
                    "Destination filter ('%s'): dropping %s " "(d_xy=%.3fm < dest_radius=%.3fm)",
                    dest_label,
                    label,
                    d_xy,
                    dest_radius,
                )
                detection_map.pop(label, None)

        # Hand the destination labels to the executor so that
        # execution-time re-detection can apply the same filter
        # (planner's filter only runs once, at plan time; without
        # this, re-detect after a place can find the just-placed
        # object and retarget the gripper back to the container).
        try:
            self._executor._destination_labels = set(destination_labels)
        except Exception:
            pass

        self._executor.update_detections(detection_map)
        # Push detected objects as obstacles into SafeRobot for path safety
        if isinstance(self._robot, SafeRobot):
            obstacles = []
            for label, det in detection_map.items():
                pos = det.get("position_3d")
                if pos is not None:
                    obstacles.append(
                        Obstacle(
                            position=np.array(pos),
                            radius=0.05,
                            timestamp=time.monotonic(),
                        )
                    )
            self._robot._obstacle_map.update(obstacles)
            logger.info("Pushed %d detection obstacles to SafeRobot", len(obstacles))
        try:
            results = self._executor.execute_score(score)
        finally:
            # Which cached-BT labels had to be re-bound to differently-named
            # detections this run. Surfaced in the run result so a silent
            # mis-binding is auditable after the fact.
            self._last_label_resolutions = dict(detection_map.resolutions)
            if self._last_label_resolutions:
                logger.info(
                    "label resolver re-bound %d cached label(s): %s",
                    len(self._last_label_resolutions),
                    self._last_label_resolutions,
                )
        return results

    @staticmethod
    def _flatten_score_actions(score) -> List[dict]:
        """
        Flatten a SPARK score (tree of sequences/leaves, or a flat
        actions/children list) into the ordered list of leaf actions.

        Used by the closed-loop destination scan; kept independent of the
        executor so it works in dry runs with no robot connected.
        """
        if not isinstance(score, dict):
            return []
        out: List[dict] = []

        def walk(node):
            if not isinstance(node, dict):
                return
            children = node.get("children")
            if children:
                for child in children:
                    walk(child)
            else:
                out.append(node)

        if score.get("actions"):
            for act in score["actions"]:
                walk(act)
        else:
            walk(score.get("tree", score))
        return out

    def _closed_loop_destinations(self, score) -> set:
        """
        Labels that act as a destination/receptacle in this score.

        Walks the flattened tree and collects container-like targets across
        every primitive that has one: a move_to_keypoint that follows a
        grasp (pick-place), place_in_slot's container_label, sweep / scrub /
        pour's target_label, and the bimanual place/sweep dustpan_label.
        These are where objects are meant to END UP, so the progress check
        treats an object sitting in one of them as handled.
        """
        dests = set()
        seen_grasp = False
        for act in self._flatten_score_actions(score):
            if not isinstance(act, dict):
                continue
            atype = act.get("type", act.get("name", ""))
            params = act.get("params", {}) or {}
            for key in ("container_label", "target_label", "dustpan_label"):
                lbl = params.get(key)
                if lbl:
                    dests.add(str(lbl))
            if atype.startswith("grasp") or atype == "pick":
                seen_grasp = True
            elif atype == "move_to_keypoint" and seen_grasp:
                kp = params.get("keypoint_label")
                if kp:
                    dests.add(str(kp))
            elif atype in ("release", "drop", "place"):
                seen_grasp = False
        return dests

    @staticmethod
    def _base_label(label) -> str:
        """
        Strip a trailing instance number ("fork 2" -> "fork") so a target
        re-detected with fresh numbering still matches the planner's label.
        """
        return re.sub(r"\s+\d+$", "", str(label)).strip()

    def evaluate_scene_progress(self, score, prompts, handled_labels=None, captures=None) -> Dict:
        """
        Re-perceive the scene and report which targets still need work.

        Generic across tasks: a "target" is any detected object whose label
        is not a destination (receptacle / dustpan / container). A target is
        HANDLED when it sits inside a destination zone or its base label is
        in handled_labels (the executor's placed set). Everything else is
        unhandled work the closed loop should keep acting on, for example a
        re-pick of a dropped fork, a re-sweep of an object that missed the
        dustpan, or a lift of an item that did not move. Newly perturbed
        objects (added utensil, swapped receptacle) appear here on their own
        because the detection is run fresh.

        Returns {"unhandled": [labels], "handled": [labels],
                 "destinations": [labels], "detections": [ObjectDetection]}.
        """
        handled_labels = set(handled_labels or set())
        if captures is None:
            captures = self.capture()
        empty = {"unhandled": [], "handled": [], "destinations": [], "detections": []}
        if not captures:
            return empty

        destinations = self._closed_loop_destinations(score)
        dest_base = {self._base_label(d) for d in destinations}
        dets = self.merge_detections(self.detect(captures, prompts, multi_instance=True))

        floor_r = float(getattr(self.config, "closed_loop_clear_radius_m", 0.12))
        zones = []
        for det in dets:
            if det.position_3d is None:
                continue
            if self._base_label(det.label) in dest_base:
                minor = float(getattr(det, "obb_minor_m", 0.0) or 0.0)
                zones.append(
                    (
                        np.asarray(det.position_3d[:2], dtype=float),
                        max(floor_r, minor * 0.6),
                    )
                )

        handled_base = {self._base_label(h) for h in handled_labels}
        handled, unhandled = [], []
        for det in dets:
            if self._base_label(det.label) in dest_base:
                continue  # the receptacle itself is not a target
            # Handled = executor placed-set OR physically inside a destination
            # zone. The zone test alone is too strict: zones are circles of
            # radius max(0.12, obb_minor*0.6) around the destination CENTROID,
            # so an item correctly placed near the end of an elongated tray
            # sits outside every zone and would be re-picked forever.
            is_handled = det.label in handled_labels or self._base_label(det.label) in handled_base
            if not is_handled and det.position_3d is not None:
                xy = np.asarray(det.position_3d[:2], dtype=float)
                for zc, zr in zones:
                    if float(np.linalg.norm(xy - zc)) < zr:
                        is_handled = True
                        break
            (handled if is_handled else unhandled).append(det.label)

        return {
            "unhandled": unhandled,
            "handled": handled,
            "destinations": sorted(destinations),
            "detections": dets,
        }
