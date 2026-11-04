"""
bimanual_shirt_fold: the two-arm T-shirt fold as a chain of the typed
bimanual primitives (skills/bimanual_primitives.py), with parameters from
perception, ported from the rig's two-arm fold procedure:

    sleeves  bimanual_grasp_points (sleeve tips) -> bimanual_lift_together
             -> bimanual_arc_drape to the shirt center line
    re-perceive the hem, shirt and collar
    hem      bimanual_grasp_points (hem corners) -> bimanual_lift_together
             -> bimanual_arc_drape up to the collar

Each step's post-condition gates the next; the skill succeeds only when all
of them pass. Points are in the right arm's frame (the scene frame); the
sleeve or corner with the larger y goes to the left arm.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from spark_real.skills import bimanual_primitives as bp
from spark_real.skills.primitives import _result
from spark_real.skills.registry import spark_skill

logger = logging.getLogger(__name__)


def _xyz(ex, label):
    det = ex.detection_map.get(label or "")
    if det is None or det.get("position_3d") is None:
        raise KeyError(f"{label!r} is not in the detection map")
    return np.asarray(det["position_3d"], dtype=float)[:3]


def _by_arm(a, b):
    """{'left': point with larger y, 'right': the other} (right-arm frame)."""
    return {"left": a, "right": b} if a[1] >= b[1] else {"left": b, "right": a}


def _refresh(ex, labels):
    if getattr(ex, "_pipeline", None) is None:
        return
    try:
        from spark_real.control.redetect import redetect_single

        for label in labels:
            if label:
                redetect_single(ex, label)
        ex.update_detections(ex.detection_map)
    except Exception as exc:  # noqa: BLE001 - keep the previous detections
        logger.warning("bimanual_shirt_fold re-detect failed: %s", exc)


def _phase(ex, name, grab, land, params, lift_dz):
    for step, kwargs in (
        ("bimanual_grasp_points", {"left_point": list(grab["left"]), "right_point": list(grab["right"])}),
        ("bimanual_lift_together", {"lift_dz": lift_dz}),
        ("bimanual_arc_drape", {"left_point": list(land["left"]), "right_point": list(land["right"]),
                                "arc_peak": params.get("arc_peak"), "land_z": params.get("land_z")}),
    ):
        res = getattr(bp, step)(ex, {k: v for k, v in kwargs.items() if v is not None})
        if not res.success:
            return f"{name}: {step} failed ({res.message})"
    return ""


@spark_skill(
    "bimanual_shirt_fold",
    "Two-arm T-shirt fold: sleeves to the center line, re-perceive, hem up to the collar.",
    {"instruction": str, "sleeve_labels": list, "shirt_label": str, "hem_label": str,
     "collar_label": str, "past_center_m": float, "hem_half_width_m": float},
)
def bimanual_shirt_fold(ex, params: dict):
    t0 = time.time()
    if getattr(ex, "motion", None) is None:
        return _result("bimanual_shirt_fold", False, "runs only on the bimanual executor")
    sleeves = params.get("sleeve_labels") or ["left sleeve", "right sleeve"]
    shirt, hem = params.get("shirt_label", "shirt"), params.get("hem_label", "hem")
    past = float(params.get("past_center_m", 0.0))
    center = _xyz(ex, shirt)
    grab = _by_arm(_xyz(ex, sleeves[0]), _xyz(ex, sleeves[1]))
    # Each sleeve lands past the center line, on the far side of it.
    land = {a: [p[0], center[1] - past if a == "left" else center[1] + past, p[2]] for a, p in grab.items()}
    lift = float(params.get("sleeve_lift_dz", ex.motion.cfg.get("lift_dz", 0.06)))
    err = _phase(ex, "sleeves", grab, land, params, lift)
    if err:
        return _result("bimanual_shirt_fold", False, err, time.time() - t0)

    _refresh(ex, [hem, shirt, params.get("collar_label")])
    hem_p, center = _xyz(ex, hem), _xyz(ex, shirt)
    half = float(params.get("hem_half_width_m", 0.12))
    grab = _by_arm(hem_p + [0, half, 0], hem_p - [0, half, 0])
    collar = _xyz(ex, params["collar_label"]) if params.get("collar_label") else 2 * center - hem_p
    land = {a: [collar[0], p[1], p[2]] for a, p in grab.items()}
    err = _phase(ex, "hem", grab, land, params, float(params.get("hem_lift_dz", 0.13)))
    return _result("bimanual_shirt_fold", not err, err or "sleeves and hem folded", time.time() - t0)
