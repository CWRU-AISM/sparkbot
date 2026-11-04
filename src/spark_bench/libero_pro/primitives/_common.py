"""
Shared helpers used by multiple primitive handlers.

* ``fuzzy_get_det`` - exact / substring / word-overlap label match.
* ``approach_descend`` - three-stage descend (high -> mid -> at) used by
  bowl- and plate-rim picks.
* ``nudge_toward_base`` - small Cartesian shift toward the robot base
  when a grasp residual is too large.
* ``scene_front_axis`` - the scene's 'front' direction from the robot base.
* ``mask_points_3d`` - backprojected world points of a detection's mask.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


__all__ = [
    'fuzzy_get_det',
    'approach_descend',
    'nudge_toward_base',
    'scene_front_axis',
    'mask_points_3d',
]


def fuzzy_get_det(det_map: dict, label: str):
    """
    Return ``det_map[label]`` with substring / word-overlap fallback.
    """
    if not label:
        return None
    det = det_map.get(label)
    if det is not None and getattr(det, 'position_3d', None) is not None:
        return det
    label_lower = label.lower()
    label_words = set(label_lower.split())
    best_score, best_det = 0, None
    for dkey, dval in det_map.items():
        if getattr(dval, 'position_3d', None) is None:
            continue
        dkey_lower = dkey.lower()
        if label_lower in dkey_lower or dkey_lower in label_lower:
            return dval
        dkey_words = set(dkey_lower.split())
        overlap = len(label_words & dkey_words)
        if overlap > best_score:
            best_score = overlap
            best_det = dval
    return best_det


def approach_descend(executor, target: np.ndarray, *,
                      hi: float = 0.10, mid: float = 0.02,
                      high_steps: int = 100, mid_steps: int = 100,
                      final_steps: int = 150) -> None:
    """
    Three-stage descend: high above -> just above -> at target.

    Used for bowl/plate rim approaches where a vertical drop catches the
    rim instead of the body.
    """
    high = target.copy()
    high[2] += hi
    just = target.copy()
    just[2] += mid
    executor._move_to(high, True, steps=high_steps)
    executor._move_to(just, True, steps=mid_steps)
    executor._move_to(target, True, steps=final_steps)


def nudge_toward_base(executor, target: np.ndarray, mag: float = 0.025) -> None:
    """
    Shift the target a few cm toward the robot base and re-attempt move.
    """
    robot_base_xy = np.array([-0.6, 0.0])
    to_base = robot_base_xy - target[:2]
    to_base /= (np.linalg.norm(to_base) + 1e-8)
    nudged = target.copy()
    nudged[:2] += to_base * mag
    executor._move_to(nudged, True, steps=150)


def scene_front_axis(executor) -> np.ndarray:
    """
    Unit XY direction of the scene's 'front': from the robot base toward the table centre, snapped to
    the dominant world axis. From the robot's own base pose; no scene model read, no task knowledge.
    """
    base = None
    if getattr(executor, 'robot_base_world', None) is not None:
        base = np.asarray(executor.robot_base_world, dtype=float)[:2]
    if base is None:
        base = np.array([-0.6, 0.0])
    v = -base
    if abs(v[0]) >= abs(v[1]):
        return np.array([np.sign(v[0]) or 1.0, 0.0])
    return np.array([0.0, np.sign(v[1]) or 1.0])


def mask_points_3d(executor, det) -> Optional[np.ndarray]:
    """
    Backproject a detection's mask into world points (same path as ``keypoint._do_pick``). None when
    mask, depth or SAM3 is absent, or the mask has 20 points or fewer.
    """
    mask = getattr(det, 'mask', None)
    if executor.sam3 is None or executor.depth is None or mask is None:
        return None
    try:
        pts = executor.sam3._backproject_mask(
            mask, executor.depth, executor.cam_pos, executor.cam_mat,
            executor.cam_fovy, executor.cam_w, executor.cam_h)
        if pts is not None and len(pts) > 20:
            return np.asarray(pts)
    except Exception:
        pass
    return None
