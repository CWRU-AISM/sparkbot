"""
grasp_se3 skill. The release ships the OBB-aligned top-down strategy only
(grasp_top_down.py); the EquiGraspFlow and side-grasp strategies are not
included, so every request runs top-down. The decorator params stay as
the frozen planner prompt lists them.
"""

import logging
import time

import numpy as np

from spark_real.skills.registry import spark_skill
from spark_real.skills.primitives import _result
from spark_real.skills.grasp_top_down import grasp_top_down as _grasp_top_down

logger = logging.getLogger(__name__)


@spark_skill(
    name="grasp_se3",
    description=(
        "OBB-aligned top-down grasp on a detected object (the 6-DOF "
        "strategies are not in this release)"
    ),
    params={
        "keypoint_label": str,
        "n_candidates": int,
        "prefer_side": bool,
        "prefer_angled": bool,
        "grip_end": str,
        "pre_grasp_distance": float,
        "force": float,
        "target_width": float,
        "strategy": str,
        "top_down": bool,
        "object_height_m": float,
        "approach_dir_x": float,
        "approach_dir_y": float,
        "approach_pitch_rad": float,
        "pitch_rad": float,
        "angle": float,
        "grasp_height_fraction": float,
        "offset_x": float,
        "offset_y": float,
        "offset_z": float,
    },
)
def grasp_se3(executor, params: dict):
    """Top-down OBB-aligned grasp on a detected object."""
    t0 = time.time()

    # Wait for any background redetection to finish
    _redetect_thread = getattr(executor, "_redetect_thread", None)
    if _redetect_thread is not None and _redetect_thread.is_alive():
        _redetect_thread.join(timeout=5.0)
        executor._redetect_thread = None

    label = params.get("keypoint_label", "")
    force = params.get("force", 60)
    target_width = params.get("target_width", None)
    strategy = params.get("strategy") or (
        "top_down" if params.get("top_down", False) else "se3"
    )
    if strategy != "top_down":
        logger.warning(
            "grasp_se3 '%s': strategy %r is not in this release; using top_down",
            label,
            strategy,
        )

    logger.info("grasp_se3 '%s': looking up detection...", label)
    det = executor.detection_map.get(label)
    if det is None:
        return _result(
            "grasp_se3",
            False,
            f"Object '{label}' not found in detections",
            time.time() - t0,
        )

    # Hard guard: never grasp an object that already sits inside the tray /
    # container, on first-pass grasps and any closed-loop repass. The target
    # xy is read from the live detection_map here.
    try:
        # Imported here to break a circular import: execution_recovery
        # imports spark_real.skills.registry at module load.
        from spark_real.control.execution_recovery import grasp_target_in_container

        _in = grasp_target_in_container(executor, label)
    except Exception:
        _in = None
    if _in is not None:
        executor._placed_labels.add(label)
        logger.info(
            "grasp_se3 '%s': already in container '%s', skipping grasp", label, _in
        )
        return _result(
            "grasp_se3",
            True,
            f"'{label}' already in container '{_in}', skipping",
            time.time() - t0,
        )

    # Optional explicit grasp-point offset from the detection centroid, in
    # base frame meters. Lets a BT grasp a specific spot on a large object
    # (e.g. a brush handle's midpoint) while keeping the validated hover ->
    # OBB yaw -> gated descend -> squeeze pipeline. Applied to a copy so
    # the live detection_map keeps the true centroid.
    _off = np.array(
        [
            float(params.get("offset_x", 0.0) or 0.0),
            float(params.get("offset_y", 0.0) or 0.0),
            float(params.get("offset_z", 0.0) or 0.0),
        ]
    )
    if np.any(_off != 0.0):
        det = dict(det)
        _p = np.asarray(det.get("position_3d"), dtype=float) + _off
        det["position_3d"] = _p.tolist()
        logger.info(
            "grasp_se3 '%s': grasp offset (%+.3f,%+.3f,%+.3f) -> " "(%.3f,%.3f,%.3f)",
            label,
            *_off,
            *_p,
        )

    return _grasp_top_down(
        executor,
        label,
        det,
        force,
        target_width,
        t0,
        grip_mode=params.get("grip_mode", "force"),
    )
