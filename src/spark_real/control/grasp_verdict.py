"""Grasp verdict types, force thresholds and gripper-call helpers."""

import time
from dataclasses import dataclass, field
from typing import Optional


def _call_force_publish(fn):
    """Ask for a forced publish, tolerating drivers without the kwarg."""
    try:
        return fn(force=True)
    except TypeError:
        return fn()


def _call_maybe_passive(fn, publish: bool):
    """
    Call a gripper getter, honouring ``publish`` when the driver supports it.

    Drivers without a ``publish`` kwarg get the plain call.
    """
    if publish:
        return fn()
    try:
        return fn(publish=False)
    except TypeError:
        return fn()


# Force-verify default (N) when neither config nor env sets one. The
# close-on-air transient on this rig is ~0.6 N. A DEFAULT, not a floor:
# grasp.force_empty_n is authoritative and may lower it.
GRASP_FORCE_EMPTY_DEFAULT_N = 2.0

# Extra downward pull (N) at which the confirming lift CORROBORATES "holding".
# Corroboration only, it can never veto. Config: grasp.lift_delta_min_n.
GRASP_LIFT_DELTA_MIN_N = 0.5

# Width-targeted grasp bounds (m).
GRASP_WIDTH_MIN_M = 0.004

GRASP_WIDTH_MAX_M = 0.080


@dataclass
class ForceVerdict:
    """What TCP force says about the jaws, and the evidence behind it."""

    peak_n: float
    lift_delta_n: float
    lift_corroborates: bool
    holding: bool


def force_verdict(
    clamp_mag: float,
    clamp_dz: float,
    lift_mag: float,
    lift_dz: float,
    empty_n: float,
    lift_delta_min_n: float,
) -> ForceVerdict:
    """Decide "force says holding" from the clamp + confirming-lift readings.

    Pure and module-level so the decision is testable without a robot.

    ``peak`` folds in BOTH readings: a light object (silverware is ~0.2 N)
    shows up in the clamp reaction, not in its own weight, so lift_delta
    CORROBORATES but never vetoes. Closed-on-air is caught by ``peak``: the
    ~0.6 N transient stays below empty_n.
    """
    peak = max(abs(clamp_mag), abs(lift_mag), abs(clamp_dz), abs(lift_dz))
    delta = float(clamp_dz - lift_dz)
    return ForceVerdict(
        peak_n=float(peak),
        lift_delta_n=delta,
        lift_corroborates=delta >= float(lift_delta_min_n),
        holding=float(peak) >= float(empty_n),
    )


@dataclass
class GraspVerdict:
    """Structured result of the G1 grasp gate, for the success verifier.

    `held` is the executor's verdict; the rest is the evidence behind it.
    """

    held: bool
    source: str = ""
    gobj: Optional[bool] = None
    gripper_pos: Optional[int] = None
    force_peak_n: float = 0.0
    force_empty_n: float = 0.0
    lift_delta_n: Optional[float] = None
    lift_corroborates: Optional[bool] = None
    label: str = ""
    strategy: str = ""
    # Typed outcome (control.grasp_outcome vocabulary): 'secured' /
    # 'empty_close' / 'slip' / 'unknown'. Stamped by _record_grasp_verdict
    # when the producer did not set it; upgraded to 'slip' by
    # note_transport_drop when a SECURED grip later reads lost.
    outcome: Optional[str] = None
    t: float = field(default_factory=time.time)


# Sub-part words a planner appends when it wants the jaws on a specific part.
# "screwdriver handle" -> "screwdriver"; the parent carries the shape that says
# which way the tool points.
_SUBPART_WORDS = ("handle", "grip", "blade", "tip", "head", "shaft", "prongs", "tines")


def _parent_object_label(label, detection_map):
    """The whole-object label for a grasped sub-part, if it was also detected."""
    if not label or not detection_map:
        return None
    parts = str(label).split()
    if len(parts) < 2 or parts[-1].lower() not in _SUBPART_WORDS:
        return None
    parent = " ".join(parts[:-1])
    return parent if parent in detection_map else None
