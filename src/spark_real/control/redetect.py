"""Re-detection of the scene or one label, and the pre-approach rebind."""

import logging

import numpy as np

from spark_core.binding import (
    _labels_match,
)
from spark_core.util import det_field
from spark_real.control.container_geometry import (  # noqa: E402,F401
    TRAY_FALLBACK_HALF_EXTENT_M,
    container_region,
    xy_inside_container,
)
from spark_real.control.sticky import _held_label_of, sticky_adopt, sticky_settings

logger = logging.getLogger(__name__)


def redetect_all(executor, actions, current_index):
    """Re-detect the next pick label so the next fold/grasp uses a fresh pose.

    Handles two pick shapes:
      * Pick-place:  move_to_keypoint(X) -> grasp(...) ...
      * SE(3) fold:  grasp_se3(keypoint_label=X) directly (the skill does
        the approach internally, so there's no move_to_keypoint before it).
    The cloth-fold sequence shifts the cloth state on every release, so
    re-detecting before the next grasp_se3 is the high-value path for folds.
    """
    remaining = actions[current_index + 1 :]
    next_label = None
    for j, a in enumerate(remaining):
        atype = a.get("type")
        params = a.get("params", {})
        label = params.get("keypoint_label", "")
        if atype == "grasp_se3" and label and label not in executor._placed_labels:
            next_label = label
            break
        if atype == "move_to_keypoint":
            next_actions = remaining[j + 1 : j + 3]
            is_pick = any(na.get("type") == "grasp" for na in next_actions)
            if label and is_pick and label not in executor._placed_labels:
                next_label = label
                break

    if not next_label:
        return

    # Redetecting during inter-grasp transit lets birdview see the arm
    # mid-motion and SAM3 latch onto phantom masks. Two guards:
    #   1. wait until the arm is stationary and near home before capturing
    #   2. reject a new position that teleported > REDETECT_MAX_MOVE_M from
    #      the planner anchor (a real object move stays within a few cm).
    REDETECT_MAX_MOVE_M = 0.10
    try:
        # Best-effort stationary wait; skip silently if no such helper.
        if hasattr(executor, "_wait_stationary_after_servo"):
            executor._wait_stationary_after_servo()
        # Best-effort "arm near home?" gate so a mid-transit arm doesn't
        # occlude birdview; skipped if joints can't be read.
        try:
            home_q = getattr(executor.robot, "HOME_CONFIG", None)
            cur_q = executor.robot.get_joint_positions()
            if home_q is not None and cur_q is not None:
                home_a = np.asarray(home_q, dtype=float)[:7]
                cur_a = np.asarray(cur_q, dtype=float)[:7]
                d_home = float(np.max(np.abs(cur_a - home_a)))
                if d_home > 0.3:
                    logger.info(
                        "Re-detect deferred for '%s': arm not at home "
                        "(max |dq|=%.2f rad > 0.30); using stale "
                        "detection. Will retry from grasp_se3's own "
                        "pre-approach redetect when arm reaches hover.",
                        next_label,
                        d_home,
                    )
                    return
        except Exception:
            pass

    except Exception:
        pass

    logger.info("Re-detecting next pick target: %s", next_label)
    try:
        captures = executor._pipeline.capture()
        new_dets = executor._pipeline.detect(captures, prompts=[next_label])

        # Build destination-exclusion zones (e.g. the tray) so a detection
        # inside the destination (likely the object just placed) is not
        # retargeted as the next pickup. Mirrors pipeline.py's planning-time
        # destination filter, applied here at execution time.
        dest_zones = []
        for dest_label in getattr(executor, "_destination_labels", set()) or set():
            dest_det = executor.detection_map.get(dest_label)
            if not dest_det or not dest_det.get("position_3d"):
                continue
            dest_pos_xy = np.asarray(dest_det["position_3d"][:2], dtype=float)
            dest_minor = float(dest_det.get("obb_minor_m") or 0.0)
            dest_zones.append((dest_label, dest_pos_xy, max(0.12, dest_minor * 0.6)))

        cfg = sticky_settings(executor)
        if cfg["enabled"]:
            # Gated adoption path: keep the destination-zone exclusion (placed
            # objects must not become the next pickup), then dedup + held
            # suppression + ambiguity hold + never-adopt-out-of-gate.
            filtered = []
            for det in new_dets:
                lbl = det.label if hasattr(det, "label") else det.get("label", "")
                pos = det_field(det, "position_3d")
                if not lbl or pos is None:
                    continue
                cand_xy = np.asarray(pos[:2], dtype=float)
                if any(
                    np.linalg.norm(cand_xy - dest_xy) < dest_r
                    for _dl, dest_xy, dest_r in dest_zones
                ):
                    continue
                filtered.append(det)
            sticky_adopt(executor, next_label, filtered, context="post-release")
            return

        # Pick the candidate closest to the planner's anchor that is not
        # inside any destination zone; if none remain outside, keep the
        # existing map entry (the planner anchor is the best guess).
        old_pos = executor.detection_map.get(next_label, {}).get("position_3d")
        old_xy = np.asarray(old_pos[:2], dtype=float) if old_pos is not None else None
        best_det = None
        best_dist = float("inf")
        for det in new_dets:
            lbl = det.label if hasattr(det, "label") else det.get("label", "")
            pos = (
                det.position_3d
                if hasattr(det, "position_3d")
                else det.get("position_3d")
            )
            if not lbl or pos is None:
                continue
            cand_xy = np.asarray(pos[:2], dtype=float)
            # Skip if inside any destination zone (placed objects).
            in_dest = False
            for dest_label, dest_xy, dest_r in dest_zones:
                if np.linalg.norm(cand_xy - dest_xy) < dest_r:
                    logger.info(
                        "Re-detect dropped '%s' at [%.3f,%.3f] "
                        "(inside '%s' destination zone, likely a placed item)",
                        lbl,
                        pos[0],
                        pos[1],
                        dest_label,
                    )
                    in_dest = True
                    break
            if in_dest:
                continue
            # Prefer closest to planner's anchor; if no anchor,
            # accept first valid.
            if old_xy is not None:
                d = float(np.linalg.norm(cand_xy - old_xy))
                if d < best_dist:
                    best_dist = d
                    best_det = det
            elif best_det is None:
                best_det = det

        if best_det is None:
            logger.warning(
                "Re-detect for '%s': no candidate outside "
                "destination zones; keeping old position",
                next_label,
            )
            return
        pos = (
            best_det.position_3d
            if hasattr(best_det, "position_3d")
            else best_det.get("position_3d")
        )
        # Reject a re-detect that teleports the target: a real move between
        # trial start and inter-grasp transit is a few cm at most, so a jump
        # of hundreds of mm is a phantom/wrong-instance detection. Trust the
        # planner anchor and let grasp_se3's pre-approach redetect refine.
        if old_xy is not None and best_dist > REDETECT_MAX_MOVE_M:
            logger.warning(
                "Re-detect '%s' REJECTED: candidate at [%.3f,%.3f] is "
                "%.1fmm from planner anchor (>%.0fmm threshold). "
                "Likely a phantom / wrong-instance detection. Keeping "
                "old position.",
                next_label,
                pos[0],
                pos[1],
                best_dist * 1000,
                REDETECT_MAX_MOVE_M * 1000,
            )
            return
        orient = getattr(best_det, "orientation_angle", 0.0)
        ar = getattr(best_det, "aspect_ratio", 1.0)
        executor.detection_map[next_label] = {
            "position_3d": list(pos),
            "orientation_angle": orient,
            "aspect_ratio": ar,
        }
        logger.info(
            "Re-detect '%s' -> [%.3f,%.3f,%.3f]%s",
            next_label,
            pos[0],
            pos[1],
            pos[2],
            (
                f" (moved {best_dist * 1000:.1f}mm)"
                if old_xy is not None and best_dist != float("inf")
                else ""
            ),
        )
    except Exception as e:
        logger.warning("Re-detection failed: %s", e)


def rebind_settings(executor) -> dict:
    """perception.rebind block with defaults (enabled by default)."""
    out = {"enabled": True, "log_drift_cm": 2.0}
    try:
        profile = getattr(getattr(executor, "_pipeline", None), "profile", None)
        raw = getattr(profile, "raw", None) or {}
        block = (raw.get("perception") or {}).get("rebind") or {}
        if isinstance(block, dict):
            out.update({k: block[k] for k in out if k in block})
    except Exception:  # noqa: BLE001
        pass
    return out


def rebind_before_approach(executor, label):
    """Re-detect the bound label before every approach so a target moved
    between plan approval and the approach becomes a silent parameter update
    on the same tree: no re-plan, no LLM, no failed attempt first.

    Adoption is delegated to redetect_single (sticky gating handles held-object
    suppression, ambiguity hold, never-adopt-out-of-gate). This wrapper only
    decides whether to run it and reports the drift it corrected.
    """
    cfg = rebind_settings(executor)
    if not cfg["enabled"] or not label:
        return
    old = None
    try:
        d = (executor.detection_map or {}).get(label) or {}
        p = d.get("position_3d")
        if p is not None:
            old = np.array(p, dtype=float)
    except Exception:  # noqa: BLE001
        pass
    try:
        redetect_single(executor, label)
    except Exception as exc:  # noqa: BLE001 - a failed rebind keeps the old binding
        logger.warning("[rebind] '%s' re-detect failed (%s); keeping the "
                       "approved binding", label, exc)
        return
    try:
        d = (executor.detection_map or {}).get(label) or {}
        p = d.get("position_3d")
        if old is not None and p is not None:
            drift_cm = float(np.linalg.norm(np.array(p)[:2] - old[:2])) * 100.0
            if drift_cm >= float(cfg["log_drift_cm"]):
                logger.info(
                    "[rebind] '%s' moved %.1f cm since approval; approach "
                    "retargeted in place (no replan)", label, drift_cm,
                )
    except Exception:  # noqa: BLE001
        pass


def redetect_single(executor, label):
    logger.info("Pre-approach re-detection for '%s'", label)
    cfg = sticky_settings(executor)
    if cfg["enabled"]:
        held = _held_label_of(executor)
        if held and _labels_match(label, held):
            logger.info(
                "Sticky[pre-approach]: '%s' is HELD; skipping re-detect "
                "entirely", label
            )
            return
    try:
        base_prompt = label.rsplit(" ", 1)[0] if label[-1].isdigit() else label
        captures = executor._pipeline.capture()
        new_dets = executor._pipeline.detect(captures, prompts=[base_prompt])
        merged = executor._pipeline.merge_detections(new_dets)

        if cfg["enabled"]:
            # Gated adoption: dedup + held suppression + ambiguity hold +
            # never-adopt-out-of-gate (the legacy path below adopts the
            # nearest candidate unconditionally at any distance).
            sticky_adopt(executor, label, merged, context="pre-approach")
            return

        old_pos = executor.detection_map.get(label, {}).get("position_3d")
        best_det = None
        best_dist = float("inf")

        for det in merged:
            pos = (
                det.position_3d
                if hasattr(det, "position_3d")
                else det.get("position_3d")
            )
            if pos is None:
                continue
            if old_pos is not None:
                dist = np.linalg.norm(np.array(pos)[:2] - np.array(old_pos)[:2])
                if dist < best_dist:
                    best_dist = dist
                    best_det = det
            elif best_det is None:
                best_det = det

        if best_det is not None:
            pos = (
                best_det.position_3d
                if hasattr(best_det, "position_3d")
                else best_det.get("position_3d")
            )
            orient = getattr(best_det, "orientation_angle", 0.0)
            ar = getattr(best_det, "aspect_ratio", 1.0)
            executor.detection_map[label] = {
                "position_3d": list(pos),
                "orientation_angle": orient,
                "aspect_ratio": ar,
            }
            if old_pos is not None:
                logger.info(
                    "'%s' updated: [%.3f,%.3f,%.3f] " "(moved %.1fmm, %d candidates)",
                    label,
                    *executor.detection_map[label]["position_3d"],
                    best_dist * 1000,
                    len(merged),
                )
        else:
            logger.warning("'%s' not found, keeping old position", label)
    except Exception as e:
        logger.warning("Re-detection for '%s' failed: %s", label, e)
