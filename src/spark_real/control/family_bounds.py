"""
ScoreExecutor bounds for the Franka families (franka, bimanual_franka).

Reads the family config: the `workspace:` box (bimanual: `workspace_<arm>`
for a per-arm executor), `table_z_floor`, and the `control:` block's
`grasp_orientation` (default: tool Z down, rotvec [pi, 0, 0]) and
`max_velocity` (a cap on the executor's Cartesian velocity).
"""

import logging
import math

import numpy as np

from spark_real.config import family_block

logger = logging.getLogger(__name__)

FRANKA_FAMILIES = ("franka", "bimanual_franka")
FRANKA_REACH_M = 0.855


def apply_franka_bounds(executor, family: str) -> None:
    raw = family_block(getattr(executor, "_profile", None), family)
    arm = getattr(executor, "arm", None)
    ws = raw.get(f"workspace_{arm}" if arm else "workspace") or {}
    if ws:
        executor.WORKSPACE_MIN = np.array(
            [float(ws.get("x_min", -0.85)), float(ws.get("y_min", -0.85)), float(ws.get("z_min", -0.05))]
        )
        executor.WORKSPACE_MAX = np.array(
            [float(ws.get("x_max", 0.85)), float(ws.get("y_max", 0.85)), float(ws.get("z_max", 1.20))]
        )
        executor.MAX_REACH = float(ws.get("reach_max", FRANKA_REACH_M))
    if raw.get("table_z_floor") is not None:
        executor.TABLE_Z_FLOOR = float(raw["table_z_floor"])
    control = raw.get("control") or {}
    grasp = control.get("grasp_orientation")
    executor.GRASP_ORIENTATION = [float(v) for v in grasp] if grasp else [math.pi, 0.0, 0.0]
    if control.get("max_velocity") is not None:
        executor.velocity = min(float(executor.velocity), float(control["max_velocity"]))
    logger.info(
        "ScoreExecutor %s%s bounds: min=%s max=%s reach=%.3f table_z_floor=%.3f grasp=%s v=%.2f",
        family,
        f"[{arm}]" if arm else "",
        np.round(executor.WORKSPACE_MIN, 3).tolist(),
        np.round(executor.WORKSPACE_MAX, 3).tolist(),
        executor.MAX_REACH,
        executor.TABLE_Z_FLOOR,
        executor.GRASP_ORIENTATION,
        executor.velocity,
    )
