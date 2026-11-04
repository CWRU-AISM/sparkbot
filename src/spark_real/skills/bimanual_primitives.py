"""
Typed bimanual primitives the planner composes (BimanualScoreExecutor only).

Each primitive runs on the rig's motion path (control/bimanual_motion.py:
PyRoki IK to joint moves on the per-arm drivers, the registered jaws) and
returns success only when its post-condition holds:

    pick_with_arm         approach, descend, close, lift   post: arm holding
    place_with_arm        open, retract                    post: jaw open
    move_to_keypoint_arm  jaw tip to a detection + offset  post: move issued (IK solved)
    grasp_arm / release_arm  close / open in place         post: holding / open
    bimanual_grasp_points both arms pinch two points       post: both holding (fails on air)
    bimanual_lift_together synchronized lift               post: both still holding
    bimanual_arc_drape    arc to land points, release      post: jaws open
    handoff               giver -> receiver at a point     post: receiver holding, giver open
    bimanual_lift         co-grasp both sides, lift        post: both holding
    bimanual_place        co-held object down onto target  post: both jaws open

Scene points (detections, meeting_point) are in the right arm's base frame;
each arm's targets are mapped into its own frame from calibration. Heights
(grasp_z, approach_z, ...) are in the arm's own frame, table near z = 0.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Dict, Optional, Sequence

import numpy as np

from spark_real.skills.primitives import _result
from spark_real.skills.registry import spark_skill

logger = logging.getLogger(__name__)

ARMS = ("left", "right")
HEIGHTS = {  # arm-frame jaw-tip heights (m) from the rig procedures
    "approach_z": 0.05,
    "grasp_z": -0.015,
    "lift_dz": 0.06,
    "safe_z": 0.10,
    "arc_peak": 0.040,
    "land_z": 0.025,
}

BIMANUAL_PRIMITIVES = (
    "pick_with_arm", "place_with_arm", "move_to_keypoint_arm", "grasp_arm", "release_arm",
    "bimanual_grasp_points", "bimanual_lift_together", "bimanual_arc_drape",
    "handoff", "bimanual_lift", "bimanual_place",
)


def _h(executor, params: dict, key: str) -> float:
    if params.get(key) is not None:
        return float(params[key])
    return float(executor.motion.cfg.get(key, HEIGHTS[key]))


def _require(executor, name: str):
    if getattr(executor, "motion", None) is None:
        return _result(name, False, f"{name} runs only on the bimanual executor")
    return None


def _arm(params: dict) -> str:
    arm = params.get("arm")
    if arm not in ARMS:
        raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")
    return arm


def _point(executor, arm: str, label: Optional[str] = None, point=None):
    """(xyz in ``arm``'s frame, yaw in that frame or None) of a label or a right-frame point."""
    yaw = None
    if point is None:
        det = executor.detection_map.get(label or "")
        if det is None or det.get("position_3d") is None:
            raise KeyError(f"{label!r} is not in the detection map")
        point = det["position_3d"]
        if det.get("world_major_axis_rad") is not None:
            yaw = float(det["world_major_axis_rad"])
    p = executor.motion.to_arm(arm, point)
    if yaw is not None and arm == "left":
        T = executor.motion.T_right_to_left
        yaw += math.atan2(T[1, 0], T[0, 0])
    return p, yaw


def _offset(params: dict) -> np.ndarray:
    return np.array([float(params.get(k, 0.0)) for k in ("offset_x", "offset_y", "offset_z")])


def _done(name: str, t0: float, ok: bool, msg: str):
    return _result(name, ok, msg, time.time() - t0)


def _set_holding(executor, arm: str, value: bool):
    executor.state.holding[arm] = value


@spark_skill("move_to_keypoint_arm", "Move one arm's jaw tip to a detected keypoint plus offset.",
             {"arm": str, "keypoint_label": str, "offset_x": float, "offset_y": float, "offset_z": float})
def move_to_keypoint_arm(executor, params: dict):
    t0 = time.time()
    if (err := _require(executor, "move_to_keypoint_arm")):
        return err
    arm = _arm(params)
    p, yaw = _point(executor, arm, params.get("keypoint_label"))
    target = p + _offset(params)
    # A held object keeps the jaw's current yaw; an empty jaw aligns to the target.
    ok = executor.motion.move(arm, target, yaw_rad=None if executor.state.holding[arm] else yaw)
    executor.state.last_target[arm] = target
    return _done("move_to_keypoint_arm", t0, ok, f"{arm} -> {np.round(target, 3).tolist()}")


@spark_skill("pick_with_arm", "One arm picks a detected object: approach, descend, close, lift.",
             {"arm": str, "keypoint_label": str, "grasp_z": float, "force": float})
def pick_with_arm(executor, params: dict):
    t0 = time.time()
    if (err := _require(executor, "pick_with_arm")):
        return err
    m = executor.motion
    arm = _arm(params)
    p, yaw = _point(executor, arm, params.get("keypoint_label"))
    p = p + _offset(params)
    m.open([arm])
    steps = [
        ([p[0], p[1], _h(executor, params, "approach_z")], m.cfg["move_vel"]),
        ([p[0], p[1], _h(executor, params, "grasp_z")], m.cfg["approach_vel"]),
    ]
    for target, vel in steps:
        if not m.move(arm, target, vel, yaw_rad=yaw):
            return _done("pick_with_arm", t0, False, f"{arm} IK failed at {np.round(target, 3).tolist()}")
    m.close([arm], force_ma=params.get("force_ma"))
    lift = [p[0], p[1], _h(executor, params, "grasp_z") + _h(executor, params, "lift_dz")]
    m.move(arm, lift, yaw_rad=yaw)
    executor.state.last_target[arm] = np.asarray(lift)
    held = m.holding(arm)
    _set_holding(executor, arm, held)
    executor.state.last_pick_label[arm] = params.get("keypoint_label", "")
    return _done("pick_with_arm", t0, held, f"{arm} holding={held} raw={m.jaw_raw(arm)}")


@spark_skill("place_with_arm", "Open one arm's jaw where it is and retract upward.",
             {"arm": str, "safe_z": float})
def place_with_arm(executor, params: dict):
    t0 = time.time()
    if (err := _require(executor, "place_with_arm")):
        return err
    m = executor.motion
    arm = _arm(params)
    m.open([arm])
    _set_holding(executor, arm, False)
    last = executor.state.last_target.get(arm)
    if last is not None:
        m.move(arm, [last[0], last[1], _h(executor, params, "safe_z")])
    ok = m.is_open(arm)
    return _done("place_with_arm", t0, ok, f"{arm} open={ok}")


@spark_skill("grasp_arm", "Close one arm's jaw in place.", {"arm": str, "force_ma": int})
def grasp_arm(executor, params: dict):
    t0 = time.time()
    if (err := _require(executor, "grasp_arm")):
        return err
    arm = _arm(params)
    executor.motion.close([arm], force_ma=params.get("force_ma"))
    held = executor.motion.holding(arm)
    _set_holding(executor, arm, held)
    return _done("grasp_arm", t0, held, f"{arm} holding={held}")


@spark_skill("release_arm", "Open one arm's jaw in place.", {"arm": str})
def release_arm(executor, params: dict):
    t0 = time.time()
    if (err := _require(executor, "release_arm")):
        return err
    arm = _arm(params)
    executor.motion.open([arm])
    _set_holding(executor, arm, False)
    ok = executor.motion.is_open(arm)
    return _done("release_arm", t0, ok, f"{arm} open={ok}")


def _two_points(executor, params: dict) -> Dict[str, tuple]:
    """Per-arm (point, yaw) from left_label/right_label or left_point/right_point."""
    out = {}
    for arm in ARMS:
        p, yaw = _point(executor, arm, params.get(f"{arm}_label"), params.get(f"{arm}_point"))
        if params.get(f"{arm}_yaw_deg") is not None:
            yaw = math.radians(float(params[f"{arm}_yaw_deg"]))
        out[arm] = (p + _offset(params), yaw)
    return out


@spark_skill("bimanual_grasp_points", "Both arms pinch two points together (fails if a jaw closes on air).",
             {"left_label": str, "right_label": str, "left_point": list, "right_point": list,
              "grasp_z": float})
def bimanual_grasp_points(executor, params: dict):
    t0 = time.time()
    if (err := _require(executor, "bimanual_grasp_points")):
        return err
    m = executor.motion
    pts = _two_points(executor, params)
    yaws = {a: y for a, (_, y) in pts.items()}
    m.open(ARMS)
    for key, vel in (("approach_z", m.cfg["move_vel"]), ("grasp_z", m.cfg["approach_vel"])):
        z = _h(executor, params, key)
        if not m.move_both({a: [p[0], p[1], z] for a, (p, _) in pts.items()}, vel, yaws):
            return _done("bimanual_grasp_points", t0, False, f"move to {key} failed")
    m.close(ARMS, force_ma=params.get("force_ma"))
    z = _h(executor, params, "grasp_z")
    for a, (p, _) in pts.items():
        executor.state.last_target[a] = np.array([p[0], p[1], z])
        executor.state.last_yaw[a] = yaws[a]
    held = {a: m.holding(a) for a in ARMS}
    for a in ARMS:
        _set_holding(executor, a, held[a])
    return _done("bimanual_grasp_points", t0, all(held.values()),
                 f"holding={held} raw={ {a: m.jaw_raw(a) for a in ARMS} }")


@spark_skill("bimanual_lift_together", "Lift both arms together by lift_dz.", {"lift_dz": float})
def bimanual_lift_together(executor, params: dict):
    t0 = time.time()
    if (err := _require(executor, "bimanual_lift_together")):
        return err
    m = executor.motion
    if any(executor.state.last_target.get(a) is None for a in ARMS):
        return _done("bimanual_lift_together", t0, False, "no previous two-arm grasp to lift from")
    dz = _h(executor, params, "lift_dz")
    targets = {a: executor.state.last_target[a] + [0.0, 0.0, dz] for a in ARMS}
    ok = m.move_both(targets, m.cfg["move_vel"], dict(executor.state.last_yaw))
    for a in ARMS:
        executor.state.last_target[a] = targets[a]
    held = all(m.holding(a) for a in ARMS)
    return _done("bimanual_lift_together", t0, ok and held, f"moved={ok} still_holding={held}")


@spark_skill("bimanual_arc_drape", "Arc the held cloth to land points (one or both arms), release, retract.",
             {"arms": list, "left_label": str, "right_label": str, "left_point": list,
              "right_point": list, "arc_peak": float, "land_z": float})
def bimanual_arc_drape(executor, params: dict):
    t0 = time.time()
    if (err := _require(executor, "bimanual_arc_drape")):
        return err
    m = executor.motion
    arms = [a for a in (params.get("arms") or ARMS) if a in ARMS]
    starts, ends = {}, {}
    for a in arms:
        if executor.state.last_target.get(a) is None:
            return _done("bimanual_arc_drape", t0, False, f"{a} arm holds nothing to drape")
        starts[a] = executor.state.last_target[a]
        ends[a], _ = _point(executor, a, params.get(f"{a}_label"), params.get(f"{a}_point"))
    lift_z = max(float(s[2]) for s in starts.values())
    ok = m.arc(starts, ends, lift_z, _h(executor, params, "arc_peak"),
               n=int(params.get("n_arc", 10)), land_z=_h(executor, params, "land_z"),
               yaws={a: executor.state.last_yaw.get(a) for a in arms})
    m.open(arms)
    for a in arms:
        _set_holding(executor, a, False)
        end = ends[a]
        m.move(a, [end[0], end[1], _h(executor, params, "safe_z")])
        executor.state.last_target[a] = None
    opened = all(m.is_open(a) for a in arms)
    return _done("bimanual_arc_drape", t0, ok and opened, f"arc={ok} jaws_open={opened}")


@spark_skill("handoff", "Pass a held object from_arm -> to_arm at meeting_point (right-arm frame).",
             {"from_arm": str, "to_arm": str, "meeting_point": list, "approach_offset": float,
              "force_ma": int})
def handoff(executor, params: dict):
    t0 = time.time()
    if (err := _require(executor, "handoff")):
        return err
    m = executor.motion
    giver, receiver = params.get("from_arm"), params.get("to_arm")
    if {giver, receiver} != set(ARMS):
        return _done("handoff", t0, False, f"invalid arms {giver}/{receiver}")
    if not executor.state.holding[giver]:
        return _done("handoff", t0, False, f"{giver} arm is not holding anything")
    meet = np.asarray(params.get("meeting_point", [0.45, -0.30, 0.25]), dtype=float)
    off = float(params.get("approach_offset", 0.08))
    side = {"left": +1.0, "right": -1.0}  # left approaches from +y of the right frame
    pre = {a: m.to_arm(a, meet + [0.0, side[a] * off, 0.0]) for a in ARMS}
    meet_pts = {a: m.to_arm(a, meet + [0.0, side[a] * 0.02, 0.0 if a == receiver else 0.02]) for a in ARMS}
    if not m.move_both(pre, m.cfg["move_vel"]):
        return _done("handoff", t0, False, "pre-meeting move failed")
    # Converging: the jaws must come within the inter-arm floor, so this
    # phase runs slowly and without the clearance gate.
    if not m.move_both(meet_pts, m.cfg["approach_vel"], check_clearance=False):
        return _done("handoff", t0, False, "converge move failed")
    m.close([receiver], force_ma=params.get("force_ma"))
    if not m.holding(receiver):
        return _done("handoff", t0, False, f"{receiver} jaw did not close on the object")
    time.sleep(0.3)  # receiver holds before the giver lets go
    m.open([giver])
    _set_holding(executor, receiver, True)
    _set_holding(executor, giver, False)
    m.move_both({a: meet_pts[a] + [0.0, 0.0, 0.08] for a in ARMS}, m.cfg["move_vel"], check_clearance=False)
    ok = m.holding(receiver) and m.is_open(giver)
    return _done("handoff", t0, ok, f"{giver}->{receiver} receiver_holding={m.holding(receiver)}")


@spark_skill("bimanual_lift", "Both arms grasp opposite sides (+/-y) of an object and lift it together.",
             {"keypoint_label": str, "object_width": float, "lift_height": float, "force_ma": int})
def bimanual_lift(executor, params: dict):
    t0 = time.time()
    if (err := _require(executor, "bimanual_lift")):
        return err
    det = executor.detection_map.get(params.get("keypoint_label") or "")
    if det is None or det.get("position_3d") is None:
        return _done("bimanual_lift", t0, False, f"{params.get('keypoint_label')!r} not detected")
    c = np.asarray(det["position_3d"], dtype=float)[:3]
    half = float(params.get("object_width", 0.20)) / 2.0
    side = {"left": +1.0, "right": -1.0}
    pts = {a: list(c + [0.0, side[a] * half, 0.0]) for a in ARMS}
    res = bimanual_grasp_points(executor, {**params, "left_point": pts["left"], "right_point": pts["right"],
                                            "grasp_z": params.get("grasp_z")})
    if not res.success:
        return _done("bimanual_lift", t0, False, f"co-grasp failed: {res.message}")
    res = bimanual_lift_together(executor, {"lift_dz": params.get("lift_height", 0.10)})
    return _done("bimanual_lift", t0, res.success, res.message)


@spark_skill("bimanual_place", "Lower a co-held object onto target_label and release both jaws.",
             {"target_label": str, "place_offset_z": float})
def bimanual_place(executor, params: dict):
    t0 = time.time()
    if (err := _require(executor, "bimanual_place")):
        return err
    m = executor.motion
    if not all(executor.state.holding[a] for a in ARMS):
        return _done("bimanual_place", t0, False, "both arms must be holding (bimanual_lift first)")
    target = {a: _point(executor, a, params.get("target_label"))[0] for a in ARMS}
    last = {a: executor.state.last_target[a] for a in ARMS}
    # Keep the grasp geometry: shift both jaws by the same delta, which is
    # the target minus the jaws' midpoint (in the right arm's frame).
    mid_r = 0.5 * (_to_right(m, "left", last["left"]) + np.asarray(last["right"], dtype=float))
    delta_r = np.asarray(target["right"]) - mid_r
    z = float(target["right"][2]) + float(params.get("place_offset_z", 0.06))
    goals = {}
    for a in ARMS:
        g_r = _to_right(m, a, last[a]) + delta_r
        g_r[2] = z
        goals[a] = m.to_arm(a, g_r)
    ok = m.move_both(goals, m.cfg["approach_vel"])
    m.open(ARMS)
    for a in ARMS:
        _set_holding(executor, a, False)
    m.move_both({a: goals[a] + [0.0, 0.0, 0.10] for a in ARMS}, m.cfg["move_vel"])
    opened = all(m.is_open(a) for a in ARMS)
    return _done("bimanual_place", t0, ok and opened, f"moved={ok} jaws_open={opened}")


def _to_right(m, arm: str, p: Sequence[float]) -> np.ndarray:
    p = np.asarray(p, dtype=float)[:3]
    if arm == "right":
        return p
    return (np.linalg.inv(m.T_right_to_left) @ np.append(p, 1.0))[:3]
