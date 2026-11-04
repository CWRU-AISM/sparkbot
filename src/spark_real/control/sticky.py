"""Sticky re-detect adoption: never rewrite a held label, gate by distance, hold on ambiguity."""

import logging
import os

import numpy as np

from spark_core.binding import (
    STICKY_AMBIGUOUS,
    STICKY_ASSOCIATED,
    STICKY_NO_CANDIDATES,
    STICKY_OUT_OF_GATE,
    _labels_match,
    dedup_candidate_indices,
    is_self_occluded,
    sticky_associate,
)
from spark_core.util import det_field
from spark_real.control.container_geometry import (  # noqa: E402,F401
    TRAY_FALLBACK_HALF_EXTENT_M,
    container_region,
    xy_inside_container,
)

logger = logging.getLogger(__name__)

_STICKY_DEFAULTS = {
    "enabled": False,
    "gate_m": 0.12,
    "ambiguity_sep_m": 0.03,
    "dedup_radius_m": 0.02,
}


def sticky_settings(executor) -> dict:
    """The perception.sticky block with defaults; env SPARK_STICKY overrides
    just the enable bit."""
    out = dict(_STICKY_DEFAULTS)
    try:
        profile = getattr(getattr(executor, "_pipeline", None), "profile", None)
        raw = getattr(profile, "raw", None) or {}
        block = (raw.get("perception") or {}).get("sticky") or {}
        if isinstance(block, dict):
            for k in out:
                if block.get(k) is not None:
                    out[k] = block[k]
    except Exception:  # noqa: BLE001 - config read must never break recovery
        pass
    env = os.environ.get("SPARK_STICKY")
    if env is not None:
        out["enabled"] = env.strip() not in ("", "0", "false", "no")
    out["enabled"] = bool(out["enabled"])
    for k in ("gate_m", "ambiguity_sep_m", "dedup_radius_m"):
        out[k] = float(out[k])
    return out


def _measure_pos(det):
    """Same-source measurement position: the primary-camera backprojection
    when stamped, else the fused position."""
    pos = det_field(det, "position_agentview")
    if pos is None:
        pos = det_field(det, "position_3d")
    return None if pos is None else np.asarray(pos, dtype=float).reshape(-1)[:3]


def _held_label_of(executor) -> str:
    if not getattr(executor, "_holding", False):
        return ""
    return getattr(executor, "_last_pick_label", "") or getattr(
        executor, "_active_grasp_label", ""
    ) or ""


def sticky_adopt(executor, label, candidates, *, context="") -> bool:
    """Gate-checked adoption of fresh ``candidates`` for ``label`` into
    executor.detection_map. Returns True iff the map entry was updated.

    Every non-adopt branch leaves detection_map UNTOUCHED (the existing
    keep-old-position behavior) and says why.
    """
    cfg = sticky_settings(executor)
    det_map = getattr(executor, "detection_map", None)
    if det_map is None:
        return False

    held = _held_label_of(executor)
    if held and _labels_match(label, held):
        logger.info(
            "Sticky[%s]: '%s' is HELD; camera evidence cannot relocate an "
            "object inside the jaws -- binding kept", context, label
        )
        return False

    entry = det_map.get(label) or {}
    anchor = _measure_pos(entry)
    fused_old = det_field(entry, "position_3d")
    fused_old = (
        None if fused_old is None
        else np.asarray(fused_old, dtype=float).reshape(-1)[:3]
    )

    positions = [_measure_pos(c) for c in candidates]
    confs = [float(det_field(c, "confidence", 0.0) or 0.0) for c in candidates]
    kept = dedup_candidate_indices(
        positions, confs, merge_radius_m=cfg["dedup_radius_m"]
    )
    cands = [candidates[i] for i in kept]
    cand_pos = [positions[i] for i in kept]

    if anchor is None:
        # No prior position: nothing to gate against; adopt the highest-
        # confidence deduped candidate (first bind).
        if not cands:
            logger.info("Sticky[%s]: '%s' no candidates; binding kept",
                        context, label)
            return False
        best_i = int(np.argmax([float(det_field(c, "confidence", 0.0) or 0.0)
                                for c in cands]))
        _write_adopted(executor, label, cands[best_i], cand_pos[best_i],
                       cand_pos[best_i], context, note="first-bind")
        return True

    res = sticky_associate(
        anchor, cand_pos,
        gate_m=cfg["gate_m"], ambiguity_sep_m=cfg["ambiguity_sep_m"],
    )
    if res.status == STICKY_ASSOCIATED:
        c = cands[res.chosen_index]
        fresh = cand_pos[res.chosen_index]
        base = fused_old if fused_old is not None else anchor
        new_pos = base + (fresh - anchor)  # delta application, same-source
        _write_adopted(executor, label, c, new_pos, fresh, context,
                       note=f"d={res.chosen_dist_m * 100:.1f}cm")
        return True
    if res.status == STICKY_AMBIGUOUS:
        logger.info(
            "Sticky[%s]: '%s' AMBIGUOUS (two in-gate candidates within "
            "%.0fmm of each other); binding kept",
            context, label, cfg["ambiguity_sep_m"] * 1000,
        )
        return False
    if res.status == STICKY_OUT_OF_GATE:
        # Self-occlusion hold: the EE hovering over the last bound position
        # blinds birdview, so an out-of-gate re-association at that moment is
        # occlusion, not a scene change. Radii are sim-tuned pending rig
        # re-measurement.
        ee = None
        try:
            ee = executor._get_current_position()
        except Exception:  # noqa: BLE001
            pass
        if ee is not None and is_self_occluded(anchor, ee):
            logger.info(
                "Sticky[%s]: '%s' out-of-gate while EE hovers over it "
                "(self-occlusion); binding kept", context, label
            )
        else:
            logger.info(
                "Sticky[%s]: '%s' nearest candidate %.1fcm away "
                "(> gate %.0fcm); NOT adopted, binding kept",
                context, label,
                (res.chosen_dist_m or 0.0) * 100, cfg["gate_m"] * 100,
            )
        return False
    logger.info("Sticky[%s]: '%s' %s; binding kept",
                context, label,
                "no candidates" if res.status == STICKY_NO_CANDIDATES
                else res.status)
    return False


def _write_adopted(executor, label, det, fused_pos, measured_pos, context,
                   note=""):
    old = executor.detection_map.get(label, {}) or {}
    executor.detection_map[label] = {
        "position_3d": [float(v) for v in np.asarray(fused_pos).reshape(-1)[:3]],
        "position_agentview": [
            float(v) for v in np.asarray(measured_pos).reshape(-1)[:3]
        ],
        "orientation_angle": det_field(
            det, "orientation_angle", det_field(old, "orientation_angle", 0.0)
        ),
        "aspect_ratio": det_field(
            det, "aspect_ratio", det_field(old, "aspect_ratio", 1.0)
        ),
    }
    logger.info(
        "Sticky[%s]: '%s' adopted -> [%.3f,%.3f,%.3f] %s",
        context, label, *executor.detection_map[label]["position_3d"], note,
    )
