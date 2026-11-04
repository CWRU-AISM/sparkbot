"""
Single-arm cloth folding skills the planner composes (ported from the FR3
T-shirt fold procedure).

    cloth_fold_single  pinch a corner/edge (stepped retries), lift, arc to the
                       mirror position across the fold axis (or land_point),
                       lay it down, release, retract.
                       post: pinched before the swing, jaw open after
    cloth_fold         T-shirt: +Y sleeve and -Y sleeve folded past the shirt
                       center, re-perceive, hem folded up to the collar; each
                       step is a cloth_fold_single.     post: every step's

Positions come from the detection map (sleeve, shirt, hem and collar
labels); heights and margins from the plan, else the family `task_motion:`
block, else the defaults here.
"""

from __future__ import annotations

import logging
import math
import time

import numpy as np

from spark_real.skills import task_motion as tm
from spark_real.skills.primitives import _result
from spark_real.skills.registry import spark_skill

logger = logging.getLogger(__name__)

FOLD = {
    "approach_height": 0.08,
    "lift_height": 0.08,  # pinch-to-arc lift above the grasp (m)
    "arc_peak": 0.030,
    "place_height": 0.03,  # landing height above the grasp z (m)
    "arc_steps": 12,
    "force": 30.0,
    "past_center_m": 0.08,  # sleeves land this far past the shirt center line
    "grasp_dz": -0.02,  # cloth: pinch below the detected surface point
}


def _f(ex, params, key):
    v = tm.cfg(ex, key, params)
    return FOLD[key] if v is None else v


def _redetect(ex, label: str) -> None:
    """Refresh one label from the cameras when a pipeline is attached."""
    if getattr(ex, "_pipeline", None) is None or not label:
        return
    try:
        from spark_real.control.redetect import redetect_single

        redetect_single(ex, label)
    except Exception as exc:  # noqa: BLE001 - keep the previous detection
        logger.warning("re-detect of %r failed: %s", label, exc)


@spark_skill(
    "cloth_fold_single",
    "Pinch a cloth corner, lift, arc it across the fold axis, lay it down, release.",
    {"corner_label": str, "anchor_label": str, "anchor_midpoint": list, "land_point": list,
     "lift_height": float, "place_height": float, "force": float, "arc_steps": int,
     "approach_height": float, "release_after": bool, "yaw_deg": float},
)
def cloth_fold_single(ex, params: dict):
    t0 = time.time()
    corner = tm.det_xyz(ex, params.get("corner_label", "")) if params.get("corner_point") is None \
        else np.asarray(params["corner_point"], dtype=float)
    if params.get("land_point") is not None:
        land = np.asarray(params["land_point"], dtype=float)
    else:
        anchor = np.asarray(params["anchor_midpoint"], dtype=float) if params.get("anchor_midpoint") is not None \
            else tm.det_xyz(ex, params.get("anchor_label", ""))
        land = 2 * anchor - corner  # mirror across the fold axis
    orient = tm.yawed(ex, math.radians(float(params.get("yaw_deg", 0.0))))
    pick_params = {
        "force": _f(ex, params, "force"),
        "approach_dz": _f(ex, params, "approach_height"),
        "grasp_dz": _f(ex, params, "grasp_dz"),
        "lift_dz": _f(ex, params, "lift_height"),
        "target_width": params.get("grip_width"),
    }
    if not tm.pick(ex, corner, orient, pick_params):
        ex.robot.open_gripper()
        return _result("cloth_fold_single", False, "pinch failed after retries", time.time() - t0)
    start = tm.tcp(ex)
    land_z = corner[2] + float(_f(ex, params, "grasp_dz")) + float(_f(ex, params, "place_height"))
    tm.arc(ex, start, land, start[2], float(_f(ex, params, "arc_peak")), orient,
           n=int(_f(ex, params, "arc_steps")), land_z=land_z, keep_x=bool(params.get("keep_x", False)))
    if params.get("release_after", True):
        ex.robot.open_gripper()
        ex._holding = False
        end = tm.tcp(ex)
        ex._move_to([end[0], end[1], end[2] + float(_f(ex, params, "lift_height"))], orient)
    opened = not params.get("release_after", True) or not ex.robot.is_object_detected()
    return _result("cloth_fold_single", opened, f"folded to {np.round(land[:2], 3).tolist()}", time.time() - t0)


@spark_skill(
    "cloth_fold",
    "T-shirt fold: both sleeves past the center line, then hem up to the collar.",
    {"instruction": str, "sleeve_labels": list, "shirt_label": str, "hem_label": str,
     "collar_label": str, "past_center_m": float},
)
def cloth_fold(ex, params: dict):
    t0 = time.time()
    sleeves = params.get("sleeve_labels") or ["left sleeve", "right sleeve"]
    shirt = params.get("shirt_label", "shirt")
    hem = params.get("hem_label", "hem")
    center = tm.det_xyz(ex, shirt)
    past = float(_f(ex, params, "past_center_m"))
    pts = sorted(((s, tm.det_xyz(ex, s)) for s in sleeves), key=lambda sp: -sp[1][1])
    steps = []
    for label, p in pts:  # +Y sleeve first; each lands past the center line
        side = 1.0 if p[1] > center[1] else -1.0
        land = [p[0], center[1] - side * past, p[2]]
        r = cloth_fold_single(ex, {**params, "corner_label": label, "land_point": land, "keep_x": True,
                                   "yaw_deg": 90.0})
        steps.append((label, r.success))
        if not r.success:
            return _result("cloth_fold", False, f"{label} fold failed: {r.message}", time.time() - t0)
    for label in (hem, shirt, params.get("collar_label")):
        _redetect(ex, label)
    hem_p = tm.det_xyz(ex, hem)
    center = tm.det_xyz(ex, shirt)
    collar = tm.det_xyz(ex, params["collar_label"]) if params.get("collar_label") else 2 * center - hem_p
    r = cloth_fold_single(ex, {**params, "corner_label": hem, "land_point": [collar[0], center[1], hem_p[2]],
                               "yaw_deg": 0.0})
    steps.append((hem, r.success))
    return _result("cloth_fold", all(ok for _, ok in steps), f"steps={steps}", time.time() - t0)
