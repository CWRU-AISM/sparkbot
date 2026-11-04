"""Injection phases and the per-(task, trial) displacement schedule.

The schedule depends only on (task, trial), so every method sees the same displacement.
"""
from __future__ import annotations

import zlib
from enum import Enum

import numpy as np


class Phase(str, Enum):
    POST_PLAN = 'post_plan'              # right after planning, before execution
    MID_APPROACH = 'mid_approach'        # EE has covered half its start distance to the pick target
    POST_GRASP_PLAN = 'post_grasp_plan'  # first close command on the action channel


# 8 compass unit vectors in the table plane.
_S = float(np.sqrt(0.5))
DIRECTION_TABLE: tuple = (
    (1.0, 0.0), (_S, _S), (0.0, 1.0), (-_S, _S),
    (-1.0, 0.0), (-_S, -_S), (0.0, -1.0), (_S, -_S),
)


def schedule_seed(task_name: str, trial: int) -> int:
    """Deterministic 32-bit seed per (task, trial). crc32, not the per-process salted hash()."""
    return zlib.crc32(f'{task_name}|{int(trial)}'.encode('utf-8')) & 0xFFFFFFFF


def displacement_direction(task_name: str, trial: int) -> np.ndarray:
    """Unit XY direction for this (task, trial); the magnitude sweep reuses it."""
    idx = schedule_seed(task_name, trial) % len(DIRECTION_TABLE)
    return np.asarray(DIRECTION_TABLE[idx], dtype=float)


def displacement_vector(task_name: str, trial: int, magnitude_cm: float) -> np.ndarray:
    """(dx, dy, 0) in meters for this (task, trial, magnitude)."""
    d = displacement_direction(task_name, trial) * (float(magnitude_cm) / 100.0)
    return np.array([d[0], d[1], 0.0], dtype=float)
