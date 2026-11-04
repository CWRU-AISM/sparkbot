"""Helpers shared by post-task verification and the retry replay: evidence, freshness, labels."""

from typing import Any, List, Optional, Sequence, Tuple

from spark_real.control.success_predicates import FAIL, VerifyOutcome

# Pre-retry retreat. The lift is vertical only and happens before any lateral
# transit, so the tool clears the container it just failed over.
RETRY_LIFT_M = 0.10

RETRY_MIN_CLEAR_Z = 0.05

# Action types that name a keypoint the replay will drive to.
_KEYPOINT_ACTIONS = ("move_to_keypoint", "grasp_se3")


def _keypoint_label(action: Any) -> str:
    if not isinstance(action, dict):
        return ""
    params = action.get("params") or {}
    return params.get("keypoint_label", "") or ""


def _replay_labels(actions: Sequence[dict]) -> List[str]:
    """Keypoint labels a replay segment drives to, in order, de-duplicated."""
    out: List[str] = []
    for action in actions or ():
        if not isinstance(action, dict):
            continue
        if action.get("type", action.get("name", "")) not in _KEYPOINT_ACTIONS:
            continue
        label = _keypoint_label(action)
        if label and label not in out:
            out.append(label)
    return out


def _detection_fingerprint(detection_map: Any, label: str) -> Optional[Tuple]:
    """``(entry, xyz)`` for ``label`` in the detection map, or ``None``.

    The entry OBJECT is kept, not its ``id``: holding the reference is what
    stops a freed dict's address being recycled by its replacement and read
    back as "unchanged".
    """
    get = getattr(detection_map, "get", None)
    if not callable(get):
        return None
    entry = get(label)
    if entry is None:
        return None
    if hasattr(entry, "get"):
        pos = entry.get("position_3d")
    else:
        pos = getattr(entry, "position_3d", None)
    try:
        xyz = None if pos is None else tuple(float(v) for v in pos)
    except (TypeError, ValueError):
        xyz = None
    return (entry, xyz)


def _is_fresh(before: Optional[Tuple], after: Optional[Tuple]) -> bool:
    """Did a re-detection actually write something for this label?

    ``execution_recovery.redetect_single`` leaves the stale entry untouched
    when the detector finds nothing ("not found, keeping old position"), which
    is otherwise indistinguishable from a successful re-read.
    """
    if after is None:
        return False
    if before is None:
        return True
    return after[0] is not before[0] or after[1] != before[1]


def _fail_has_evidence(outcome: VerifyOutcome) -> bool:
    """Did this ``fail`` come from evidence, or was it synthesised?

    ``verification.on_missing: fail`` turns "no predicate derivable" (a
    structural abstain, no capture, no vote) into a fail. Replaying a pick
    cycle off that opens the gripper on the strength of nothing at all, which
    is the same "abstain moves the robot" bug the retry gate exists to stop.
    A failed gate or a camera voting fail is evidence; nothing else is.
    """
    if any(v is False for v in (outcome.gates or {}).values()):
        return True
    return any(getattr(v, "vote", "") == FAIL for v in (outcome.votes or ()))
