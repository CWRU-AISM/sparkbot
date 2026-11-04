"""
Gripper and camera driver registries: any family can name any registered
type; an unregistered type or an unusable transport raises with a clear
message. No device is opened.

Usage:
    PYTHONPATH=src python -m pytest src/spark_real/tests/test_gripper_registry.py -q
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

import spark_real.grippers.franka_hand as fh
from spark_real.grippers import (
    GripperMount,
    arm_gripper_cfg,
    make_dual_gripper,
    make_gripper,
    mount_gripper,
)
from spark_real.perception.camera_registry import build_camera_registry, camera_type


class _Arm:
    robot_family = "franka"
    SUPPORTS_URSCRIPT = False
    robot_ip = "192.0.2.5"

    def __init__(self):
        self.linear = []

    def get_tcp_pose(self):
        return [0.4, 0.0, 0.3, math.pi, 0.0, 0.0]

    def move_linear(self, pose, **kw):
        self.linear.append(list(pose))

    def get_observation(self):
        return {"tcp_pose": self.get_tcp_pose()}


class _URArm(_Arm):
    SUPPORTS_URSCRIPT = True
    GRIPPER_TYPE = "robotiq_2f85"

    def close_gripper(self, **kw):
        return True


class _Jaw:
    GRIPPER_TYPE = "ssg48"
    config = SimpleNamespace(width_open_m=0.09, width_closed_m=0.0)

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *a, **k: self.calls.append((name, a, k)) or 0.04


@pytest.mark.parametrize("cfg", [{}, {"type": "parallel_jaw"}, {"type": "dynamixel"}, {"type": "ssg48"}])
def test_unregistered_type_or_missing_wiring_is_value_error(cfg):
    with pytest.raises(ValueError):
        make_gripper(cfg, _Arm())


def test_robotiq_is_native_on_a_ur_and_has_no_transport_elsewhere():
    ur = _URArm()
    assert mount_gripper(ur, {"type": "robotiq_2f85"}) is ur
    with pytest.raises(RuntimeError, match="URScript"):
        make_gripper({"type": "robotiq_2f85"}, _Arm())


def test_jaws_mount_on_any_arm():
    # A Dynamixel jaw on a UR10e and an SSG-48 on a Franka: same registry.
    ur = mount_gripper(_URArm(), {"type": "dynamixel", "id": 3, "port": "/dev/x"})
    assert isinstance(ur, GripperMount) and ur.GRIPPER_TYPE == "dynamixel"
    assert ur._jaw.config.motor_id == 3 and ur.SUPPORTS_URSCRIPT is True  # arm unchanged
    fr = mount_gripper(_Arm(), {"type": "ssg48", "channel": "/dev/y"})
    assert fr.GRIPPER_TYPE == "ssg48" and fr.robot_family == "franka"


def test_franka_hand_needs_franky_and_defaults_to_the_host_ip(monkeypatch):
    monkeypatch.setattr(fh, "_FrankyGripper", None)
    with pytest.raises(RuntimeError, match="franky"):
        make_gripper({"type": "franka_hand"}, _Arm())
    monkeypatch.setattr(fh, "_FrankyGripper", object)
    g = make_gripper({"type": "franka_hand", "force": 40}, _Arm())
    assert g.config.ip == "192.0.2.5" and g.config.force == 40.0


def test_mount_routes_the_gripper_api_and_offsets_the_tcp():
    jaw = _Jaw()
    m = GripperMount(_Arm(), jaw, tcp_offset_m=0.1)
    m.close_gripper(force=20)
    m.set_gripper_position(0.5)
    assert [c[0] for c in jaw.calls] == ["close", "move_to_width"]
    assert jaw.calls[1][1][0] == pytest.approx(0.045)
    assert np.allclose(m.get_tcp_pose()[:3], [0.4, 0.0, 0.2])  # tool Z down
    m.move_linear([0.5, 0.0, 0.1, math.pi, 0.0, 0.0])
    assert np.allclose(m._robot.linear[-1][:3], [0.5, 0.0, 0.2])


def test_dual_gripper_merges_per_arm_blocks():
    cfg = {"type": "ssg48", "max_force_ma": 900, "left": {"channel": "/dev/a"},
           "right": {"channel": "/dev/b", "max_force_ma": 700}}
    assert arm_gripper_cfg(cfg, "right")["max_force_ma"] == 700
    dual = make_dual_gripper(cfg)
    assert (dual.left.config.channel, dual.right.config.channel) == ("/dev/a", "/dev/b")
    assert (dual.left.config.max_force_ma, dual.right.config.max_force_ma) == (900, 700)
    with pytest.raises(RuntimeError):
        make_dual_gripper({"type": "robotiq_2f85"})


def test_camera_registry_rejects_unknown_types_and_missing_sdks():
    with pytest.raises(ValueError, match="no driver"):
        camera_type({"role": "x", "type": "kinectv1"})
    assert camera_type({"role": "x", "type": "uvc"}) == "webcam"
    # No ZED SDK on the test host: the transport cannot work, so it raises
    # before any camera opens.
    pytest.importorskip("cv2")
    try:
        import pyzed.sl  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="pyzed"):
            build_camera_registry([{"role": "external", "type": "zed_mini"}])


def test_camera_without_a_device_is_left_out_not_simulated():
    reg = build_camera_registry([{"role": "wrist_left", "type": "webcam", "arm": "left", "device": None}])
    assert reg.roles == []
