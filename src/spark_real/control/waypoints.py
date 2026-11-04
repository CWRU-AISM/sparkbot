"""
UR10e blended joint path: one URScript program of ``movej(..., r=)`` rows.

Leaves of a motion phrase become ``JointRow`` entries; the executor emits
them as ONE URScript program over the primary socket, which is the only
blend mechanism immune to the rtde_c control-script lifecycle.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Why one URScript program instead of rtde_c.moveJ(path):
#   * every move on this rig already goes out over the raw URScript socket
#     (ur10e_driver._send_script, port 30002) because rtde_c.moveJ silently
#     no-ops once a speedl / gripper script has stopped its control script,
#     and _send_script itself calls rtde_c.stopScript() before every move.
#   * consecutive _send_script("movej(...)") calls do NOT blend: each send
#     REPLACES the running program, so send #2 cancels move #1 mid-flight.
#     Blending therefore requires all rows inside ONE program in ONE send.
#     That shape is already proven here by the gripper publisher, which
#     uploads a multi-line def/end block plus its call in a single send.
#
# Emitted shape (last row always r=0, so the terminal waypoint is a true
# stop and the executor's exact joint-arrival test still applies):
#
#   def spark_blend_path():
#     write_output_integer_register(15, 1000)
#     movej([...],a=1.4,v=1.05,r=0.05)
#     write_output_integer_register(15, 1001)
#     movej([...],a=1.4,v=1.05,r=0)
#     write_output_integer_register(15, 1002)
#   end
#   spark_blend_path()
#
# The register writes are a mid-path progress signal. They are inside the
# same program, so they cannot cancel anything (unlike a fresh send), and
# they need no control script. Without them a stall inside a multi-second
# path would only surface as a silent full-path timeout (the measured 15.0 s /
# 1817-speedl grind into a mechanical stop). "Register not advancing AND TCP
# still" is the stall signature.

# 12/13/14 belong to the gripper publisher (_POS/_OBJ/_SEQ_REGISTER).
UR_PROGRESS_REGISTER = 15
# value = epoch * STRIDE + row_index, so a register left over from the
# PREVIOUS path can never be mistaken for progress on this one.
UR_PROGRESS_STRIDE = 1000
UR_MAX_EPOCH = 200
UR_BLEND_FUNC = "spark_blend_path"

# A junction blend radius may not exceed this fraction of either adjoining
# segment. r_i <= 0.45*min(L_i, L_i+1) also guarantees non-overlap of
# consecutive blend regions (r_i + r_i+1 <= 0.9 * L_i+1). PolyScope does no
# validation (rtde_control.Path.toScriptCode() accepted 0.05 blends on
# ~0.1 rad segments), so the policing is here.
UR_BLEND_SEG_FRAC = 0.45
# Below this a blend is not worth the corner-cut; emit a full stop instead.
UR_BLEND_MIN_M = 0.01
# Two rows whose joint targets agree this closely are the same pose: the row
# is a no-op and is dropped rather than blended (this is _transport_to's
# first lift, which is ~0 m long right after a move_relative(dz=+0.20)).
UR_ROW_MERGE_RAD = 0.01
# Reject a phrase whose IK jumps this far on one row: an elbow/wrist branch
# flip mid-path becomes one continuous blended wrench instead of a discrete,
# visible reconfiguration between two stopped moves.
UR_MAX_ROW_DQ_RAD = 2.6


@dataclass
class JointRow:
    """One ``movej`` row of a blended URScript path.

    Attributes:
        q: (6,) joint target [rad].
        position: (3,) TCP position this row commands, base frame [m]. Only
            used to size blend radii (``r`` is metres of TCP path even for
            movej) and to budget the wait; never sent.
        velocity: joint velocity [rad/s] -> ``v=``.
        acceleration: joint acceleration [rad/s^2] -> ``a=``.
        blend: blend radius [m] -> ``r=``. 0 means decelerate to a stop.
        label: free text for logs.
    """

    q: np.ndarray
    position: np.ndarray
    velocity: float
    acceleration: float
    blend: float = 0.0
    label: str = ""

    def __post_init__(self):
        self.q = np.asarray(self.q, dtype=float).reshape(-1)
        self.position = np.asarray(self.position, dtype=float).reshape(3)


def _fmt_num(x: float) -> str:
    """Format like ur_rtde's own PathEntry emitter (C++ ostream default: %g)."""
    return f"{float(x):g}"


def format_movej_row(q: Sequence[float], velocity: float, acceleration: float, blend: float) -> str:
    """One ``movej([...],a=,v=,r=)`` line.

    Byte-identical to ``rtde_control.PathEntry(MoveJ, PositionJoints,
    [*q, v, a, r]).toScriptCode()`` minus its leading tab (asserted by
    tests/test_blend_path.py), so the wire contract is ur_rtde's.
    """
    qs = ",".join(_fmt_num(v) for v in list(q)[:6])
    return (
        f"movej([{qs}],a={_fmt_num(acceleration)}," f"v={_fmt_num(velocity)},r={_fmt_num(blend)})"
    )


def resolve_blend_radii(
    start_position: Sequence[float],
    rows: List[JointRow],
    radius_m: float,
    final_radius_m: float = 0.0,
    min_radius_m: float = UR_BLEND_MIN_M,
    merge_rad: float = UR_ROW_MERGE_RAD,
    start_q: Optional[Sequence[float]] = None,
) -> tuple:
    """Drop no-op rows and size every junction blend. Mutates ``rows``' blends.

    ``radius_m`` is the requested junction radius; each junction is clamped to
    ``UR_BLEND_SEG_FRAC`` of BOTH adjoining segments, and anything below
    ``min_radius_m`` becomes a full stop. The LAST row is always r=0;
    ur_rtde's own emitter does the same, and the executor's arrival test
    depends on that terminal waypoint being a true stop.

    ``final_radius_m`` caps the junction leading INTO the last row separately,
    so a phrase that ends in a grasp descent can keep that descent essentially
    vertical (a 5 cm corner-cut into a 15 cm descent is not wanted) while the
    transit junctions before it stay generous.

    Returns (kept_rows, dropped_labels).
    """
    kept: List[JointRow] = []
    dropped: List[str] = []
    prev_q = None if start_q is None else np.asarray(start_q, dtype=float).reshape(-1)
    for i, row in enumerate(rows):
        last = i == len(rows) - 1
        if (
            prev_q is not None
            and not last
            and prev_q.size >= row.q.size
            and float(np.max(np.abs(row.q - prev_q[: row.q.size]))) < merge_rad
        ):
            # Same pose as the current or previous target: a no-op row.
            dropped.append(row.label or f"row{i}")
            continue
        kept.append(row)
        prev_q = row.q
    if not kept:
        return kept, dropped

    pts = [np.asarray(start_position, dtype=float).reshape(3)] + [r.position for r in kept]
    seg = [float(np.linalg.norm(pts[i + 1] - pts[i])) for i in range(len(kept))]
    for i, row in enumerate(kept):
        if i == len(kept) - 1:
            row.blend = 0.0  # terminal waypoint is a true stop
            continue
        want = radius_m
        if i == len(kept) - 2:
            want = min(want, final_radius_m) if final_radius_m > 0 else want
        r = min(want, UR_BLEND_SEG_FRAC * seg[i], UR_BLEND_SEG_FRAC * seg[i + 1])
        row.blend = float(r) if r >= min_radius_m else 0.0
    return kept, dropped


def build_ur_blend_program(
    rows: List[JointRow],
    epoch: int,
    progress_register: int = UR_PROGRESS_REGISTER,
    func_name: str = UR_BLEND_FUNC,
) -> str:
    """Wrap ``rows`` into one uploadable URScript program.

    ``epoch`` namespaces the progress register so a value left by the previous
    path is not read as progress on this one. Row k is *entered* when the
    register reads ``epoch*STRIDE + k``; ``+ len(rows)`` means the program ran
    to the end.
    """
    if not rows:
        raise ValueError("build_ur_blend_program: no rows")
    base = int(epoch) * UR_PROGRESS_STRIDE
    lines = [f"def {func_name}():"]
    for k, row in enumerate(rows):
        if progress_register is not None:
            lines.append(f"  write_output_integer_register({progress_register}, {base + k})")
        lines.append("  " + format_movej_row(row.q, row.velocity, row.acceleration, row.blend))
    if progress_register is not None:
        lines.append(f"  write_output_integer_register({progress_register}, {base + len(rows)})")
    lines.append("end")
    lines.append(f"{func_name}()")
    return "\n".join(lines) + "\n"


def blend_progress_value(epoch: int, row_index: int) -> int:
    """The register value written on entering ``row_index`` of ``epoch``."""
    return int(epoch) * UR_PROGRESS_STRIDE + int(row_index)


def decode_blend_progress(value, epoch: int, n_rows: int):
    """Row index from a register read, or None if it is not this path's epoch."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None
    base = int(epoch) * UR_PROGRESS_STRIDE
    if base <= v <= base + n_rows:
        return v - base
    return None


def joint_move_seconds(dq_max: float, velocity: float, acceleration: float) -> float:
    """Trapezoidal duration of one movej, from the leading joint's travel.

    Used only to BUDGET the wait (3x this plus the start lag), never to
    decide arrival.
    """
    d = abs(float(dq_max))
    v = max(float(velocity), 1e-6)
    a = max(float(acceleration), 1e-6)
    if d <= v * v / a:
        return 2.0 * math.sqrt(d / a)
    return d / v + v / a
