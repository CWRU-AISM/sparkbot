"""Proprioceptive verification gates: grasp (G1), transport (G2), release witness (G3)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spark_real.control.success_predicates import (
    container_obb_xy,
    detection_xyz,
    obb_contains,
)

logger = logging.getLogger(__name__)

GATE_NAMES = ("grasp", "transport", "release")


@dataclass
class ReleaseWitness:
    held_before: bool
    held_after: bool
    tcp_xyz: Tuple[float, float, float]
    container_label: Optional[str] = None
    inside_region: Optional[bool] = None
    tcp_dz_to_container: Optional[float] = None
    # Jaw evidence behind held_after. ``released`` is the gate's verdict:
    # True = the jaws demonstrably opened, False = they did not, None = no
    # usable read (abstain). The raw numbers are logged so the next rig run
    # calibrates GRIPPER_RELEASED_MAX_POS / GRIPPER_RELEASE_MIN_TRAVEL.
    released: Optional[bool] = None
    jaw_pos_before: Optional[float] = None
    jaw_pos_after: Optional[float] = None
    obj_after: Optional[bool] = None
    confirm_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "held_before": bool(self.held_before),
            "held_after": bool(self.held_after),
            "tcp_xyz": [float(v) for v in self.tcp_xyz],
            "container_label": self.container_label,
            "inside_region": self.inside_region,
            "tcp_dz_to_container": self.tcp_dz_to_container,
            "released": self.released,
            "jaw_pos_before": self.jaw_pos_before,
            "jaw_pos_after": self.jaw_pos_after,
            "obj_after": self.obj_after,
            "confirm_s": round(float(self.confirm_s), 3),
        }


def _gates(executor) -> Dict[str, Optional[bool]]:
    flags = getattr(executor, "_verify_gates", None)
    if not isinstance(flags, dict):
        flags = {}
        try:
            executor._verify_gates = flags
        except Exception:
            pass
    return flags


def set_gate(executor, name: str, value: Optional[bool], detail: str = "") -> None:
    if executor is None or name not in GATE_NAMES:
        return
    _gates(executor)[name] = value
    if value is False:
        logger.warning("Verify gate '%s' FAILED%s", name, f": {detail}" if detail else "")


def record_grasp_verdict(executor, holding: bool, detail: str = "", verdict: Any = None) -> None:
    """G1: called from the grasp verify path with its structured verdict.

    ``verdict`` is the executor's own richer record (control.executor_grasp's
    GraspVerdict) when it has one; it is stored verbatim so the ASPIRE trace
    keeps the evidence behind the boolean.
    """
    set_gate(executor, "grasp", bool(holding), detail)
    if holding:
        # A successful re-grasp clears an earlier transport drop.
        if _gates(executor).get("transport") is False:
            set_gate(executor, "transport", True, "re-grasp after drop")
    try:
        executor._last_grasp_verdict = (
            verdict if verdict is not None else {"holding": bool(holding), "detail": detail}
        )
    except Exception:
        pass


def _verdict_holding(verdict: Any) -> Optional[bool]:
    """Read the boolean out of either verdict shape (dict or GraspVerdict)."""
    if verdict is None:
        return None
    if isinstance(verdict, dict):
        val = verdict.get("holding")
    else:
        val = getattr(verdict, "held", None)
    return None if val is None else bool(val)


def note_transport_drop(executor, waypoint: str = "") -> None:
    """G2: called when a stationary-waypoint grip check says DROPPED.

    Also upgrades the stored grasp verdict's typed outcome to SLIP when the
    grasp had classified SECURED: a secured close followed by a lost grip
    at a stationary waypoint is the Robotiq analog of the sim's
    aperture-collapse SLIP. Additive metadata only.
    """
    set_gate(executor, "transport", False, f"drop at {waypoint}")
    try:
        from spark_core.grasp_outcome import (
            GraspOutcome,
            slip_after_secured,
        )

        v = getattr(executor, "_last_grasp_verdict", None)
        prev_raw = (
            v.get("outcome") if isinstance(v, dict)
            else getattr(v, "outcome", None)
        )
        prev = GraspOutcome(prev_raw) if prev_raw else None
        slip = slip_after_secured(prev, False)
        if slip is not None:
            if isinstance(v, dict):
                v["outcome"] = slip.value
            else:
                v.outcome = slip.value
            logger.warning(
                "Grasp outcome: SECURED -> SLIP (drop at %s)", waypoint
            )
    except Exception:  # noqa: BLE001 - annotation must never break the gate
        pass


def reset_run_state(executor, task_scope: bool = True) -> None:
    """Drop every verdict/gate carried by the previous run.

    An executor instance outlives a task, so without this the next run would
    inherit the last run's outcome, gates and release witness, and a stale
    ``pass`` would be read as this run's success.

    ``task_scope`` also clears the executor's own per-task bookkeeping
    (_placed_labels, _holding, the grasp retry budget, ...) via
    ScoreExecutor.reset_task_state. Pass False from a CLOSED-LOOP PASS
    boundary: the passes of one task share that state (_placed_labels is what
    tells pass 3 that pass 1 already handled an object), so clearing it there
    would make the loop re-pick an object already sitting in the receptacle.
    """
    if executor is None:
        return
    for attr, value in (
        ("verify_outcome", None),
        ("_verify_gates", {}),
        ("_release_witness", None),
        ("_last_grasp_verdict", None),
    ):
        try:
            setattr(executor, attr, value)
        except Exception:
            pass
    if not task_scope:
        return
    reset = getattr(executor, "reset_task_state", None)
    if callable(reset):
        try:
            reset()
        except Exception as exc:
            logger.warning("reset_run_state: per-task reset failed: %s", exc)


def _read_holding(executor) -> Optional[bool]:
    """Best available live grip state (gObj-primary when the rig exposes it)."""
    try:
        if hasattr(executor, "_grip_intact") and executor._gripper_type() == "robotiq_2f85":
            return bool(executor._grip_intact())
    except Exception as exc:
        logger.warning("Release witness: grip read failed: %s", exc)
    held = getattr(executor, "_holding", None)
    return None if held is None else bool(held)


def _is_robotiq(executor) -> bool:
    try:
        return executor._gripper_type() == "robotiq_2f85"
    except Exception:
        return False


def _has_jaw_source(executor) -> bool:
    """Does the driver expose any jaw state (register, position or width)?"""
    if _is_robotiq(executor):
        return True
    robot = getattr(executor, "robot", None)
    return any(
        hasattr(robot, name) for name in ("get_gripper_position", "get_gripper_width")
    )


def _jaw_sample(executor) -> Tuple[Optional[bool], Optional[float]]:
    """One fresh jaw sample: ``(object_flag, jaw_pos)``, 0=open .. 255=closed.

    Any driver: the executor's _get_gripper_position rescales a non-Robotiq
    position (or width) onto the Robotiq count scale, so the open-side
    thresholds below apply to every gripper. ``(None, None)`` only when the
    driver has no position source or the read failed. On the Robotiq,
    _read_gripper_state forces a publish and waits for the controller to
    acknowledge it, so this is a current sample, not a stale register.
    """
    try:
        obj, pos = executor._read_gripper_state()
    except Exception as exc:
        logger.warning("Release witness: jaw read failed: %s", exc)
        return None, None
    return (
        None if obj is None else bool(obj),
        None if pos is None else float(pos),
    )


def _jaws_opened(
    executor,
    pos_before: Optional[float],
    pos_after: Optional[float],
    obj_after: Optional[bool],
) -> Optional[bool]:
    """Did the jaws open? None when there is no usable position read.

    Reaching the open stop is conclusive on its own: a gObj that still reads
    contact there is a stale flag, not an object. Short of the stop, gObj=1
    means the jaws were halted BY something, so only travel with gObj clear
    counts. The travel clause carries the thin object, whose jaws legitimately
    stop before the absolute threshold.
    """
    if pos_after is None:
        return None
    max_pos = float(getattr(executor, "GRIPPER_RELEASED_MAX_POS", 60.0))
    min_travel = float(getattr(executor, "GRIPPER_RELEASE_MIN_TRAVEL", 40.0))
    if pos_after <= max_pos:
        return True
    if obj_after is True:
        return False
    if pos_before is not None and (pos_before - pos_after) >= min_travel:
        return True
    return False


def _confirm_jaws_open(
    executor, pos_before: Optional[float]
) -> Tuple[Optional[bool], Optional[bool], Optional[float], float]:
    """Poll for the jaws to read open. ``(opened, gObj, pos, waited_s)``.

    Only safe once the open URScript has finished: every sample uploads a
    program of its own, which would cancel an rq_move_and_wait still running.
    The caller's settle guarantees that. Costs nothing on a normal release;
    the first sample already reads open.
    """
    timeout = float(getattr(executor, "RELEASE_CONFIRM_TIMEOUT_S", 0.6))
    t0 = time.monotonic()
    obj, pos = _jaw_sample(executor)
    opened = _jaws_opened(executor, pos_before, pos, obj)
    while opened is False and (time.monotonic() - t0) < timeout:
        time.sleep(0.05)
        obj, pos = _jaw_sample(executor)
        opened = _jaws_opened(executor, pos_before, pos, obj)
    return opened, obj, pos, time.monotonic() - t0


def begin_release_witness(executor) -> Dict[str, Any]:
    """G3 part 1: call immediately BEFORE opening the gripper."""
    state = {"held_before": _read_holding(executor)}
    _, state["jaw_pos_before"] = _jaw_sample(executor)
    try:
        state["tcp_xyz"] = [float(v) for v in executor._get_current_position()[:3]]
    except Exception:
        state["tcp_xyz"] = [0.0, 0.0, 0.0]
    return state


def finish_release_witness(
    executor, state: Dict[str, Any], container_label: Optional[str] = None
) -> Optional[ReleaseWitness]:
    """G3 part 2: call after the open + settle. Stores the witness on the executor.

    ``held_after`` must NOT be read with _grip_intact. That predicate answers
    "are the jaws still spread on something" after a CLOSE, and its position
    clause (pos < GRIPPER_FULLY_CLOSED = holding) is True for every OPEN
    gripper (pos ~0 on a released 2F-85).
    Post-open the meaningful question is the opposite one: did the jaws reach
    open, or did something stop them.
    """
    try:
        pos_before = state.get("jaw_pos_before")
        opened, obj_after, pos_after, waited = _confirm_jaws_open(executor, pos_before)
        if opened is not None:
            released = opened
        elif _has_jaw_source(executor):
            # The rig HAS a jaw source and it did not read: abstain. Falling
            # back to _grip_intact here would answer "DROPPED" from the same
            # unreadable state and pass the gate on no evidence at all.
            released = None
        else:
            # No jaw state at all: the executor's own flag is the only signal.
            held = _read_holding(executor)
            released = None if held is None else (not held)
        witness = ReleaseWitness(
            held_before=bool(state.get("held_before")),
            held_after=(released is False),
            tcp_xyz=tuple(state.get("tcp_xyz") or (0.0, 0.0, 0.0)),
            container_label=container_label,
            released=released,
            jaw_pos_before=pos_before,
            jaw_pos_after=pos_after,
            obj_after=obj_after,
            confirm_s=waited,
        )
        label = container_label or getattr(executor, "_last_place_label", "") or ""
        det = (getattr(executor, "detection_map", None) or {}).get(label)
        obb = container_obb_xy(det) if det is not None else None
        if obb is not None:
            witness.container_label = label
            witness.inside_region = bool(
                obb_contains(np.asarray(witness.tcp_xyz[:2], dtype=float), obb, 0.0)
            )
            cxyz = detection_xyz(det)
            if cxyz is not None:
                witness.tcp_dz_to_container = float(witness.tcp_xyz[2] - cxyz[2])
        executor._release_witness = witness
        open_max = getattr(executor, "GRIPPER_RELEASED_MAX_POS", 60)
        min_travel = getattr(executor, "GRIPPER_RELEASE_MIN_TRAVEL", 40)
        detail = (
            f"held_before={witness.held_before} released={released} "
            f"jaw {pos_before} -> {pos_after} "
            f"(open<={open_max}, travel>={min_travel}) "
            f"gObj_after={obj_after} confirm={waited:.2f}s"
        )
        if released is None:
            # Abstain rather than invent a verdict: collect_gates reads a
            # missing gate as "the primitive never reported", and a fabricated
            # failure here would fail every task on a rig with no jaw register.
            logger.warning("Release witness: no grip evidence, abstaining (%s)", detail)
        else:
            set_gate(executor, "release", bool(witness.held_before and released), detail)
        logger.info("Release witness: %s", witness.to_dict())
        return witness
    except Exception as exc:
        logger.warning("Release witness failed: %s", exc)
        return None


def collect_gates(executor) -> Tuple[Dict[str, bool], List[str]]:
    """Resolve the three gates to booleans plus the reasons any of them failed.

    An unknown gate reads True (the primitive never ran); only explicit
    negative evidence fails, and the vision path still has to pass.
    """
    flags = dict(_gates(executor))
    if "grasp" not in flags:
        held = _verdict_holding(getattr(executor, "_last_grasp_verdict", None))
        if held is not None:
            flags["grasp"] = held
        else:
            results = [
                r
                for r in (getattr(executor, "_results", None) or [])
                if getattr(r, "action_type", "") == "grasp"
            ]
            if results:
                flags["grasp"] = bool(results[-1].success)
    witness = getattr(executor, "_release_witness", None)
    if "release" not in flags and isinstance(witness, ReleaseWitness):
        # Same verdict finish_release_witness computed, never a re-derivation:
        # released=None means it abstained, and re-deriving from held_after
        # would resurrect the false negative it exists to avoid.
        if witness.released is not None:
            flags["release"] = bool(witness.held_before and witness.released)

    gates, failures = {}, []
    for name in GATE_NAMES:
        val = flags.get(name)
        gates[name] = True if val is None else bool(val)
        if val is False:
            failures.append(name)
    return gates, failures
