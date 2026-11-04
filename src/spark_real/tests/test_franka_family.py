"""
Franka single-arm family without hardware: config parsing, the factory with a
fake franky, gripper-type dispatch, Franka Hand commands, executor bounds and
the FR3 joint-IK path.

Usage:
    PYTHONPATH=src python -m pytest src/spark_real/tests/test_franka_family.py -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_selector_semantics import _FakePipeline, _MockRobot  # noqa: E402

import spark_real.control.executor_ik as executor_ik  # noqa: E402
import spark_real.grippers.franka_hand as fh  # noqa: E402
import spark_real.robots.franka.franka_driver as fd  # noqa: E402
from spark_real.grippers import GripperMount  # noqa: E402
from spark_real.config import SparkConfig, load_profile  # noqa: E402
from spark_real.control.score_executor import ScoreExecutor  # noqa: E402
from spark_real.robots.factory import make_robot_driver  # noqa: E402

IP = "192.0.2.1"


class _FakeHand:
    max_width = 0.08
    width = 0.03
    is_grasped = True

    def __init__(self, ip):
        self.calls = []

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *a: self.calls.append((name, a)) or True


class _FakeFrankyRobot:
    has_errors = False

    def __init__(self, ip, realtime_config=None):
        self.collision = None

    def set_collision_behavior(self, *args):
        self.collision = args


@pytest.fixture
def fake_franky(monkeypatch):
    monkeypatch.setattr(fd, "HAS_FRANKY", True)
    monkeypatch.setattr(fd, "Robot", _FakeFrankyRobot, raising=False)
    monkeypatch.setattr(fh, "_FrankyGripper", _FakeHand)
    monkeypatch.setattr(fd, "RealtimeConfig", SimpleNamespace(Ignore=None), raising=False)


def _profile(**kw):
    return load_profile(SparkConfig(family="franka", **kw))


def test_profile_reads_the_franka_config():
    prof = _profile(ip=IP)
    kwargs = prof.to_pipeline_kwargs()
    assert kwargs["robot_family"] == "franka"
    assert kwargs["robot_model"] == "Franka FR3"
    assert kwargs["robot_ip"] == IP
    assert kwargs["wrist_tool_offset_xyz"] == [0.0, 0.06, -0.05]
    blocks = prof.driver_blocks()
    assert blocks["gripper"]["type"] == "franka_hand"
    assert blocks["robot"]["backend"] == "franky"
    assert len(blocks["collision"]["joint_torque_nom_lower"]) == 7
    assert prof.home_config()[4] == pytest.approx(-0.15)


def test_factory_mounts_the_configured_hand(fake_franky):
    prof = _profile()
    drv = make_robot_driver("franka", IP, **prof.driver_blocks())
    assert isinstance(drv, GripperMount) and isinstance(drv._robot, fd.FrankaDriver)
    assert drv.GRIPPER_TYPE == "franka_hand" and drv.robot_family == "franka"
    assert drv.HOME_CONFIG == prof.home_config()
    drv.connect()
    assert isinstance(drv._jaw._hand, _FakeHand)
    # Eight per-axis lists: acceleration + nominal thresholds.
    assert len(drv._robot._robot.collision) == 8


@pytest.mark.parametrize("gtype,err", [("robotiq_2f85", RuntimeError), ("parallel_jaw", ValueError)])
def test_gripper_without_a_driver_or_transport_raises_clearly(fake_franky, gtype, err):
    with pytest.raises(err, match="gripper.type"):
        make_robot_driver("franka", IP, gripper={"type": gtype})


def test_franka_hand_commands(fake_franky):
    drv = make_robot_driver("franka", IP, gripper={"type": "franka_hand", "speed": 0.05, "force": 40.0})
    drv.connect()
    hand = drv._jaw._hand
    drv.open_gripper()
    drv.close_gripper()
    assert drv.grasp_to_width(0.02) is True
    assert drv.is_object_detected() is True
    drv.set_gripper_position(0.5)
    assert ("open", (0.05,)) in hand.calls
    assert ("grasp", (0.0, 0.05, 40.0, 0.08, 0.08)) in hand.calls
    assert ("grasp", (0.02, 0.05, 40.0, 0.08, 0.08)) in hand.calls
    assert ("move_async", (0.04, 0.05)) in hand.calls
    assert drv.get_gripper_width() == pytest.approx(0.03)
    assert drv.get_gripper_position() == pytest.approx((1 - 0.03 / 0.08) * 255)


def _franka_executor(velocity=0.25):
    return ScoreExecutor(_MockRobot(), detection_map={}, velocity=velocity, pipeline=_FakePipeline())


def test_executor_bounds_come_from_the_franka_config():
    ex = _franka_executor()
    assert ex.GRASP_ORIENTATION == pytest.approx([3.14159, 0.0, 0.0])
    assert ex.velocity == pytest.approx(0.10)  # control.max_velocity
    assert ex.TABLE_Z_FLOOR == pytest.approx(-0.025)
    assert ex.MAX_REACH == pytest.approx(0.855)
    assert ex.WORKSPACE_MIN[2] == pytest.approx(-0.040)
    assert _franka_executor(velocity=0.05).velocity == pytest.approx(0.05)


def test_movej_via_ik_uses_fr3_ik_and_joint_velocity(monkeypatch):
    robot = _MockRobot()
    moves = []
    robot.move_to_joint_config = lambda q, velocity=None, **kw: moves.append((list(q), velocity))
    ex = ScoreExecutor(robot, detection_map={}, velocity=0.10, pipeline=_FakePipeline())
    q0 = np.asarray(robot.get_joint_positions(), dtype=float)
    q1 = q0 + np.array([0.4, 0, 0, 0, 0, 0, 0])
    seen = {}

    def _solve(pos, orient, q_seed):
        seen["pos"] = pos
        return q1

    monkeypatch.setattr(executor_ik, "solve_ik_fr3", _solve)
    ex._movej_via_ik([0.4, 0.0, 0.3], [math.pi, 0.0, 0.0], ex.velocity)
    assert np.allclose(seen["pos"], [0.4, 0.0, 0.3])
    (q_cmd, vel_j), = moves
    assert np.allclose(q_cmd, q1)
    dq = np.abs(q1 - q0)
    assert vel_j == pytest.approx(executor_ik.resolve_joint_limits(0.10, dq=dq)[0])


def test_movej_via_ik_raises_when_fr3_ik_fails(monkeypatch):
    ex = ScoreExecutor(_MockRobot(), detection_map={}, velocity=0.1, pipeline=_FakePipeline())
    monkeypatch.setattr(executor_ik, "solve_ik_fr3", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="FR3 IK failed"):
        ex._movej_via_ik([0.4, 0.0, 0.3], [math.pi, 0.0, 0.0], 0.1)
