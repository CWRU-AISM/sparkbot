"""Joint and Cartesian speed envelope: one source for every movej and movel the executor emits."""

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

# Joint / Cartesian speed envelope: ONE source for every movej and movel this
# executor emits. The URScript defaults movej(q, a=1.4, v=1.05) and
# movel(pose, a=1.2, v=0.25) are 50% of the base/shoulder rating and 33% of
# the wrist rating, not UR10e capabilities.
#
# Rated maxima, UR10e tech sheet (base + shoulder 120 deg/s, elbow + wrist
# 1/2/3 180 deg/s):
#   https://www.universal-robots.com/manuals/EN/HTML/SW5_25/Content/
#   prod-usr-man/complianceUR10e/H_g5_sections/appendix_g5/tech_spec_sheet.htm
UR10E_JOINT_VEL_RATED_RAD_S = (2.094, 2.094, 3.142, 3.142, 3.142, 3.142)

# SAFETY MARGIN, stated explicitly: command at most 85% of the rated joint
# speed (15% headroom). The headroom is not decoration: the installed pendant
# safety configuration has its own Joint Speed Limit, and exceeding THAT is a
# protective stop (C150/C152), not a faster move. 85% also leaves room for the
# pendant Stopping Time / Stopping Distance limits, which throttle commanded
# speed so the arm can always stop inside them.
JOINT_VEL_MARGIN = 0.85

# UR publishes NO per-joint acceleration rating: movej "scales torques to be
# within hardware limits" itself, so this is a REQUEST ceiling, not a hardware
# number. 3.5 rad/s^2 is 2.5x the URScript default of 1.4. Acceleration is the
# dominant lever here (a trapezoid is triangular for any move shorter than
# v^2/a, and most primitives on this task are short), but very high acceleration combined with
# blends is UR's named trigger for C173 joint-torque-overload, so this is
# deliberately not pushed further without rig evidence.
JOINT_ACC_CAP_RAD_S2 = 3.5

# Never crawl: a tiny commanded velocity makes large reconfigurations take
# forever.
JOINT_VEL_FLOOR_RAD_S = 0.3

# The Cartesian `velocity` knob (PipelineConfig.velocity, default 0.25 m/s) is
# what the whole executor is parameterised by. This is the value of that knob
# which maps to the FULL leading-axis joint cap; below it the joint cap scales
# down proportionally.
JOINT_VEL_REF_LINEAR_M_S = 0.25

# movel is Cartesian. UR10e tool speed: 4 m/s max, "approx 1 m/s" typical in
# the tech-sheet table. Cap at half the typical figure.
LINEAR_VEL_CAP_M_S = 0.5

LINEAR_ACC_CAP_M_S2 = 1.5

# Ramp-in knob: SPARK_JOINT_SPEED_FRACTION scales the whole envelope (v and a
# together). 1.0 = the envelope above; ~0.6 reproduces a 1.05 rad/s velocity.
_SPEED_FRACTION_ENV = "SPARK_JOINT_SPEED_FRACTION"

# SPARK_LEGACY_JOINT_LIMITS restores the legacy clamps below exactly
# (one-env-var rollback).
_LEGACY_LIMITS_ENV = "SPARK_LEGACY_JOINT_LIMITS"

_LEGACY = {
    "vel_scale": 6.0,
    "vel_cap": 1.05,
    "acc_cap": 1.4,
    "pose_vel_scale": 3.0,
    "pose_vel_cap": 1.0,
    "lin_vel_cap": 0.25,
    "lin_acc_cap": 1.2,
}


def legacy_joint_limits() -> bool:
    """True when SPARK_LEGACY_JOINT_LIMITS restores the legacy clamps."""
    return os.environ.get(_LEGACY_LIMITS_ENV, "0") not in ("0", "false", "False", "")


def joint_speed_fraction() -> float:
    """Fraction of the whole speed envelope to command (0 < f <= 1)."""
    raw = os.environ.get(_SPEED_FRACTION_ENV)
    if raw is None:
        return 1.0
    try:
        return float(np.clip(float(raw), 0.05, 1.0))
    except ValueError:
        logger.warning("%s=%r ignored", _SPEED_FRACTION_ENV, raw)
        return 1.0


def leading_axis_vel_cap(dq) -> float:
    """Highest movej ``v`` that keeps EVERY joint inside its own rating.

    URScript movej applies ``v``/``a`` to the LEADING axis (the largest
    |dq|); every other joint is time-scaled to finish together, so joint i
    actually runs at ``v * dq_i / L``. A single global cap therefore has to be
    the slowest joint's rating (2.094 rad/s), but on a wrist-dominated move
    the base is barely turning and there is 50% more headroom available. The
    exact condition is ``v * dq_i / L <= rated_i`` for all i, i.e.
    ``v <= min_i(rated_i * L / dq_i)``.

    Falls back to the most conservative rating when dq is unknown or zero.
    """
    dq = np.abs(np.asarray(dq, dtype=float).reshape(-1))
    rated = np.asarray(UR10E_JOINT_VEL_RATED_RAD_S[: dq.size], dtype=float)
    if dq.size == 0 or rated.size != dq.size:
        return JOINT_VEL_MARGIN * min(UR10E_JOINT_VEL_RATED_RAD_S)
    lead = float(dq.max())
    if lead <= 1e-9:
        return JOINT_VEL_MARGIN * min(UR10E_JOINT_VEL_RATED_RAD_S)
    moving = dq > 1e-9
    cap = float(np.min(rated[moving] * lead / dq[moving]))
    return JOINT_VEL_MARGIN * cap


def resolve_joint_limits(velocity, dq=None, pose_fallback=False):
    """Map the Cartesian ``velocity`` knob to one movej ``(v, a)`` in rad, s.

    ``dq`` is the per-joint travel of the move about to be issued; passing it
    unlocks the wrist headroom described in leading_axis_vel_cap. Omit it and
    the cap collapses to the slowest joint's rating, which is what a move with
    no solved joint target has to assume.

    ``pose_fallback`` marks the legacy Cartesian path (_movej_to_pose), whose
    legacy clamps (velocity*3 capped at 1.0) differ from the IK path's
    (velocity*6 capped at 1.05). It only changes anything under
    SPARK_LEGACY_JOINT_LIMITS.
    """
    if legacy_joint_limits():
        if pose_fallback:
            vel_j = float(
                min(float(velocity) * _LEGACY["pose_vel_scale"], _LEGACY["pose_vel_cap"])
            )
        else:
            vel_j = float(
                np.clip(
                    float(velocity) * _LEGACY["vel_scale"],
                    JOINT_VEL_FLOOR_RAD_S,
                    _LEGACY["vel_cap"],
                )
            )
        return vel_j, float(min(vel_j * 2.0, _LEGACY["acc_cap"]))
    frac = joint_speed_fraction()
    cap = leading_axis_vel_cap(dq if dq is not None else []) * frac
    ref = max(JOINT_VEL_REF_LINEAR_M_S, 1e-6)
    vel_j = float(np.clip(float(velocity) / ref * cap, JOINT_VEL_FLOOR_RAD_S, cap))
    acc_j = float(min(vel_j * 2.0, JOINT_ACC_CAP_RAD_S2 * frac))
    return vel_j, acc_j


def resolve_linear_limits(velocity):
    """movel ``(v, a)`` in m/s, m/s^2 for the legacy Cartesian fallback."""
    if legacy_joint_limits():
        vel_l = float(min(float(velocity), _LEGACY["lin_vel_cap"]))
        return vel_l, float(min(max(vel_l, 0.05) * 4.0, _LEGACY["lin_acc_cap"]))
    frac = joint_speed_fraction()
    vel_l = float(min(float(velocity), LINEAR_VEL_CAP_M_S * frac))
    acc_l = float(min(max(vel_l, 0.05) * 4.0, LINEAR_ACC_CAP_M_S2 * frac))
    return vel_l, acc_l
