"""
Typed bimanual primitives on a fake rig: per-arm frames from calibration,
two-arm synchronized motion, jaw post-conditions, inter-arm gate, and the
executor dispatch. The fake IK returns the target as the "joints", so each
recorded joint move is the arm-frame flange target. No hardware.

Usage:
    PYTHONPATH=src python -m pytest src/spark_real/tests/test_bimanual_primitives.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_bimanual_executor_surface import _make  # noqa: E402

import spark_real.control.bimanual_motion as bm  # noqa: E402


class _Planner:
    def solve(self, arm, target_position_base, target_wxyz_base, prev_cfg):
        return np.concatenate([target_position_base, target_wxyz_base])


class _ArmDriver:
    def __init__(self):
        self.moves = []

    def get_joint_positions(self):
        return [0.0] * 7

    def move_to_joint_config(self, q, velocity=None):
        self.moves.append(np.asarray(q[:3]))


class _Jaw:
    GRIPPER_TYPE = "ssg48"

    def __init__(self, closed_raw=180):
        self.raw = 0.0
        self.closed_raw = closed_raw
        self.packs = []

    def send_pack_locked(self, pos, *rest):
        self.packs.append(pos)
        self.raw = float(self.closed_raw if pos else 0)

    def raw_position_fresh(self):
        return self.raw


class _Grippers:
    def __init__(self, left, right):
        self.jaws = {"left": left, "right": right}

    def for_arm(self, arm):
        return self.jaws[arm]


T_R2L = np.eye(4)
T_R2L[:3, 3] = [0.0, 0.8, 0.0]  # left base 0.8 m along -y of the right base


def _rig(closed_raw=180, gate=None):
    ex = _make()
    ex.T_right_to_left = T_R2L
    ex.driver = type("D", (), {})()
    ex.driver._drivers = {"left": _ArmDriver(), "right": _ArmDriver()}
    ex.driver.grippers = _Grippers(_Jaw(closed_raw), _Jaw(closed_raw))
    ex.motion.driver = ex.driver
    ex.motion._planner = _Planner()
    ex.motion.cfg.update(close_settle_s=0.0, open_settle_s=0.0)
    ex.motion.clearance = gate
    ex.update_detections({
        "fork": {"position_3d": [0.40, -0.10, 0.0]},
        "knife": {"position_3d": [0.40, 0.30, 0.0]},
        "tray": {"position_3d": [0.50, 0.10, 0.02]},
    })
    return ex


def _leaf(ex, ntype, **params):
    return ex._run_leaf({"type": ntype, "params": params})


def test_pick_with_arm_runs_in_that_arms_calibrated_frame():
    ex = _rig()
    assert _leaf(ex, "pick_with_arm", arm="left", keypoint_label="knife")
    moves = ex.driver._drivers["left"].moves
    # knife at right-frame (0.40, 0.30) is (0.40, 1.10) in the left frame;
    # approach 0.05, grasp -0.015, lift +0.06, each +0.103 to the flange.
    assert np.allclose([m[:2] for m in moves], [[0.40, 1.10]] * 3)
    assert np.allclose([m[2] for m in moves], [0.153, 0.088, 0.148])
    assert ex.driver.grippers.jaws["left"].packs == [0, 255]
    assert ex.state.holding["left"] and not ex.driver._drivers["right"].moves
    assert ex._results[-1].action_type == "pick_with_arm[left]"


def test_grasp_on_air_fails_the_post_condition():
    ex = _rig(closed_raw=255)
    assert _leaf(ex, "pick_with_arm", arm="right", keypoint_label="fork") is False
    assert "holding=False" in ex._results[-1].message


def test_two_arm_fold_chain():
    ex = _rig()
    assert _leaf(ex, "bimanual_grasp_points", left_label="knife", right_label="fork")
    assert _leaf(ex, "bimanual_lift_together", lift_dz=0.05)
    left, right = (ex.driver._drivers[a].moves for a in ("left", "right"))
    assert np.allclose(right[-1], [0.40, -0.10, -0.015 + 0.05 + 0.103])
    assert np.allclose(left[-1], [0.40, 1.10, -0.015 + 0.05 + 0.103])
    assert _leaf(ex, "bimanual_arc_drape", left_label="tray", right_label="tray", n_arc=4)
    assert not any(ex.state.holding.values())
    # The arc landed on the tray in each arm's frame, then retracted to safe_z.
    assert np.allclose(right[-2], [0.50, 0.10, 0.025 + 0.103])
    assert np.allclose(left[-2], [0.50, 0.90, 0.025 + 0.103])
    assert all(j.raw == 0 for j in ex.driver.grippers.jaws.values())


def test_handoff_receiver_closes_before_the_giver_opens():
    ex = _rig()
    assert _leaf(ex, "pick_with_arm", arm="left", keypoint_label="knife")
    order = []
    for arm, jaw in ex.driver.grippers.jaws.items():
        jaw.send_pack_locked = (lambda f, a: lambda pos, *r: (order.append((a, pos)), f(pos, *r)))(
            jaw.send_pack_locked, arm)
    assert _leaf(ex, "handoff", from_arm="left", to_arm="right", meeting_point=[0.45, 0.35, 0.25])
    assert order == [("right", 255), ("left", 0)]
    assert ex.state.holding == {"left": False, "right": True}


def test_handoff_requires_the_giver_to_hold():
    ex = _rig()
    assert _leaf(ex, "handoff", from_arm="left", to_arm="right") is False


def test_bimanual_lift_then_place_on_target():
    ex = _rig()
    assert _leaf(ex, "bimanual_lift", keypoint_label="tray", object_width=0.20, lift_height=0.10)
    assert all(ex.state.holding.values())
    assert _leaf(ex, "bimanual_place", target_label="tray", place_offset_z=0.05)
    right_down = ex.driver._drivers["right"].moves[-2]
    assert np.allclose(right_down, [0.50, 0.0, 0.02 + 0.05 + 0.103])  # -half width in y
    assert not any(ex.state.holding.values())


def test_inter_arm_gate_refuses_a_target():
    ex = _rig(gate=lambda arm, xyz: xyz[1] < 1.0)
    assert _leaf(ex, "move_to_keypoint_arm", arm="left", keypoint_label="knife") is False
    assert ex.driver._drivers["left"].moves == []


def test_missing_label_and_unshipped_primitive_fail_clearly():
    ex = _rig()
    assert _leaf(ex, "pick_with_arm", arm="right", keypoint_label="spoon") is False
    assert "spoon" in ex._results[-1].message
    assert _leaf(ex, "bimanual_shirt_fold") is False  # no shirt detected
    assert "shirt" in ex._results[-1].message
    assert _leaf(ex, "bimanual_pants_fold") is False
    assert "not in this release" in ex._results[-1].message


def test_left_frame_needs_calibration():
    m = bm.BimanualMotion(None, None)
    with pytest.raises(RuntimeError, match="calibration"):
        m.to_arm("left", [0, 0, 0])
    assert np.allclose(m.to_arm("right", [1, 2, 3]), [1, 2, 3])


def test_yaw_branch_is_fixed_per_arm():
    m = bm.BimanualMotion(None, None)
    a = m.orientation("right", 0.5)
    b = m.orientation("right", 0.5 + np.pi)
    assert np.allclose(np.abs(a), np.abs(b))  # 180-degree ambiguity resolved to one branch
