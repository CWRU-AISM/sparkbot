"""
Shared motion helpers for the single-arm task skills (tool_use.py, cloth.py).

Ported from the FR3 task procedures: top-down pick with a stepped descent
retry (close, read the jaw width, open, step down), sinusoidal arc moves,
and circular paths. Every move goes through the executor's _move_to (joint
IK path on the FR3, firmware IK on the UR10e), so workspace clipping, the
table floor and abort apply. Positions come from the detection map; numbers
not given in the plan come from the family config `task_motion:` block, then
from DEFAULTS.
"""

from __future__ import annotations

import logging
import math
import time
from typing import List, Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from spark_real.config import family_block

logger = logging.getLogger(__name__)

DEFAULTS = {
    "approach_dz": 0.08,  # hover above the object before the descent (m)
    "grasp_dz": 0.003,  # grasp height above the detected centroid (m)
    "retry_step_m": 0.003,  # extra descent per failed close (m)
    "grasp_retries": 4,
    "held_width_m": 0.001,  # jaw width above this after a close = something held
    "lift_dz": 0.10,
    "slow_vel": 0.08,
}


def cfg(executor, key: str, params: Optional[dict] = None):
    """A tuning value: plan param, else family `task_motion:` block, else DEFAULTS."""
    if params and params.get(key) is not None:
        return params[key]
    pipeline = getattr(executor, "_pipeline", None)
    family = getattr(getattr(pipeline, "config", None), "robot_family", "") or ""
    block = family_block(getattr(executor, "_profile", None), family, "task_motion") if family else {}
    return block.get(key, DEFAULTS.get(key))


def det_xyz(executor, label: str) -> np.ndarray:
    det = executor.detection_map.get(label or "")
    if det is None or det.get("position_3d") is None:
        raise KeyError(f"{label!r} is not in the detection map")
    return np.asarray(det["position_3d"], dtype=float)[:3]


def labels(value) -> List[str]:
    """A label list from a list or a comma-separated string."""
    if isinstance(value, str):
        return [s.strip() for s in value.split(",") if s.strip()]
    return [str(s) for s in (value or [])]


def yawed(executor, yaw_rad: float) -> List[float]:
    """The executor's top-down grasp rotvec turned by yaw about base Z."""
    base = Rotation.from_rotvec(executor.GRASP_ORIENTATION)
    return (Rotation.from_euler("z", yaw_rad) * base).as_rotvec().tolist()


def tcp(executor) -> np.ndarray:
    return np.asarray(executor._get_current_position(), dtype=float)[:3]


def jaw_width(executor) -> float:
    try:
        return float(executor.robot.get_gripper_width())
    except Exception:  # noqa: BLE001 - an unreadable width counts as empty
        return 0.0


def holding(executor, params: Optional[dict] = None) -> bool:
    """Post-condition of a pick: the jaws stopped short of closing."""
    return jaw_width(executor) > float(cfg(executor, "held_width_m", params))


def pick(executor, xyz: Sequence[float], orient, params: dict) -> bool:
    """Approach, descend, close; on an empty close open and step down, up to
    `grasp_retries` times; lift. Returns the holding post-condition."""
    p = np.asarray(xyz, dtype=float)
    force = float(params.get("force", 40.0))
    executor.robot.open_gripper()
    executor._move_to(p + [0, 0, float(cfg(executor, "approach_dz", params))], orient)
    z = p[2] + float(cfg(executor, "grasp_dz", params))
    vel = float(cfg(executor, "slow_vel", params))
    held = False
    for attempt in range(int(cfg(executor, "grasp_retries", params))):
        executor._check_abort()
        executor._move_to([p[0], p[1], z], orient, velocity=vel)
        if params.get("target_width") is not None and hasattr(executor.robot, "grasp_to_width"):
            executor.robot.grasp_to_width(float(params["target_width"]), force=force)
        else:
            executor.robot.close_gripper(force=force)
        time.sleep(0.3)
        if holding(executor, params):
            held = True
            break
        logger.info("pick attempt %d empty at z=%.3f; stepping down", attempt + 1, z)
        executor.robot.open_gripper()
        z -= float(cfg(executor, "retry_step_m", params))
    executor._holding = held
    executor._move_to([p[0], p[1], z + float(cfg(executor, "lift_dz", params))], orient)
    return held and holding(executor, params)


def arc(executor, start: Sequence[float], end: Sequence[float], lift_z: float, peak_m: float,
        orient, n: int = 12, land_z: Optional[float] = None, keep_x: bool = False, vel: float = 0.11):
    """Sinusoidal arc at lift_z from start to end xy; the last waypoint lands at land_z."""
    for i in range(1, n + 1):
        executor._check_abort()
        t = i / n
        x = start[0] if keep_x else start[0] + t * (end[0] - start[0])
        y = start[1] + t * (end[1] - start[1])
        z = lift_z + peak_m * math.sin(t * math.pi)
        if land_z is not None and i == n:
            z = land_z
        executor._move_to([x, y, z], orient, velocity=vel)


def circle_points(center: Sequence[float], radius: float, steps: int, pattern: str = "circle") -> List[np.ndarray]:
    """One cycle of a scrub pattern around center (circle, figure8 or lines)."""
    c = np.asarray(center, dtype=float)
    pts = []
    for i in range(steps):
        a = 2 * math.pi * i / steps
        if pattern == "figure8":
            off = [radius * math.sin(a), radius * math.sin(a) * math.cos(a), 0.0]
        elif pattern == "lines":
            off = [radius * (1 if (i // max(1, steps // 4)) % 2 else -1), radius * (2 * i / steps - 1), 0.0]
        else:
            off = [radius * math.cos(a), radius * math.sin(a), 0.0]
        pts.append(c + off)
    return pts
