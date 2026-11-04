"""
Bimanual family without hardware: config parsing, gripper dispatch, the
driver's per-arm GripperMount (jaw routing and jaw-tip TCP), camera roster checks, the
calibration loader and the executor's per-arm frames and leaf mapping.

Usage:
    PYTHONPATH=src python -m pytest src/spark_real/tests/test_bimanual_family.py -q
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_bimanual_executor_surface import _make  # noqa: E402

import spark_real.pipeline_init_bimanual as pib  # noqa: E402
from spark_real.config import SparkConfig, load_profile  # noqa: E402
from spark_real.perception.camera_registry import camera_type, spec  # noqa: E402
from spark_real.robots.bimanual_franka import BimanualFrankaDriver  # noqa: E402


class _Jaw:
    GRIPPER_TYPE = "ssg48"

    def __init__(self):
        self.calls = []
        self.config = type("C", (), {"width_open_m": 0.09, "width_closed_m": 0.0})()

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *a, **k: self.calls.append((name, a, k)) or 0.04


class _Arm:
    """Per-arm driver double: flange pointing down at (0.4, 0, 0.3)."""

    def __init__(self):
        self.flange = [0.4, 0.0, 0.3, math.pi, 0.0, 0.0]
        self.linear = []

    def get_tcp_pose(self):
        return list(self.flange)

    def move_linear(self, pose, **kw):
        self.linear.append(list(pose))

    def get_observation(self):
        return {"tcp_pose": self.get_tcp_pose()}

    def get_joint_positions(self):
        return [0.0] * 7

    def get_tcp_force(self):
        return [0.0] * 6


class _Dual:
    GRIPPER_TYPE = "ssg48"

    def __init__(self):
        self.jaws = {"left": _Jaw(), "right": _Jaw()}

    def for_arm(self, arm):
        return self.jaws[arm]


def _driver(jaw_offset=0.1275):
    drv = BimanualFrankaDriver("192.0.2.21", "192.0.2.22", _Dual(), backend="bamboo", jaw_offset_m=jaw_offset)
    drv._drivers = {"left": _Arm(), "right": _Arm()}
    return drv


def test_profile_reads_the_bimanual_config():
    prof = load_profile(SparkConfig(family="bimanual"))
    assert prof.family == "bimanual_franka"
    kwargs = prof.to_pipeline_kwargs()
    assert kwargs["use_kinect"] is False and kwargs["use_realsense"] is False
    assert kwargs["robot_model"] == "Bimanual Franka"
    assert prof.gripper()["type"] == "ssg48"
    assert prof.driver_blocks()["robot"]["backend"] == "bamboo"
    assert len(prof.home_config()) == 14
    assert load_profile(SparkConfig(family="bimanual", ip="192.0.2.9")).resolved_ip() == "192.0.2.9"
    roles = spec(prof)
    assert [camera_type(roles[r]) for r in ("external", "wrist_left", "wrist_right")] == [
        "zed_mini", "webcam", "webcam"]


def test_arm_view_routes_gripper_to_its_jaw():
    drv = _driver()
    left = drv.for_arm("left")
    assert left.GRIPPER_TYPE == "ssg48" and left.robot_family == "bimanual_franka"
    left.close_gripper(force=20)
    left.set_gripper_position(0.5)
    assert left.get_gripper_width() == pytest.approx(0.04)
    jaw = drv.grippers.jaws["left"]
    assert [c[0] for c in jaw.calls] == ["close", "move_to_width", "get_present_width"]
    assert jaw.calls[1][1][0] == pytest.approx(0.045)  # half of the 90 mm stroke
    assert drv.grippers.jaws["right"].calls == []


def test_arm_view_tcp_is_the_jaw_tip():
    drv = _driver(jaw_offset=0.1)
    view = drv.for_arm("right")
    # Tool Z points down, so the tip is 0.1 m below the flange.
    assert np.allclose(view.get_tcp_pose()[:3], [0.4, 0.0, 0.2])
    view.move_linear([0.5, 0.1, 0.05, math.pi, 0.0, 0.0])
    assert np.allclose(drv._drivers["right"].linear[-1][:3], [0.5, 0.1, 0.15])
    obs = drv.get_observation("right")
    assert np.allclose(obs["tcp_pose"][:3], [0.4, 0.0, 0.3])  # flange, for anchor_tap
    assert np.allclose(obs["jaw_tip_pose"][:3], [0.4, 0.0, 0.2])


def test_calibration_loader(tmp_path):
    T_r = np.eye(4)
    T_r[:3, 3] = [0.1, 0.2, 0.3]
    T_l = np.eye(4)
    T_l[:3, 3] = [0.1, -0.6, 0.3]
    path = tmp_path / "calibration_bimanual.json"
    path.write_text(json.dumps({"arms": {"right": {"T_zed_to_base": T_r.tolist()},
                                         "left": {"T_zed_to_base": T_l.tolist()}}}))
    T_zed_r, T_r2l = pib.load_bimanual_calibration(path)
    assert np.allclose(T_zed_r, T_r)
    assert np.allclose(T_r2l[:3, 3], [0.0, -0.8, 0.0])
    assert pib.load_bimanual_calibration(tmp_path / "missing.json") is None


def test_left_arm_gets_the_scene_in_its_own_frame():
    ex = _make()
    T = np.eye(4)
    T[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]  # yaw +90 deg
    T[:3, 3] = [0.0, 1.0, 0.0]
    ex.T_right_to_left = T
    ex.update_detections({"tray": {"position_3d": [0.5, 0.0, 0.0], "world_major_axis_rad": 0.0,
                                   "slots": [{"world_xyz": [0.5, 0.1, 0.0]}]}})
    left = ex._arm_executors["left"].detection_map["tray"]
    assert np.allclose(left["position_3d"], [0.0, 1.5, 0.0])
    assert left["world_major_axis_rad"] == pytest.approx(math.pi / 2)
    assert np.allclose(left["slots"][0]["world_xyz"], [-0.1, 1.5, 0.0])
    assert ex._arm_executors["right"].detection_map["tray"]["position_3d"] == [0.5, 0.0, 0.0]


def test_execute_refuses_without_calibration():
    ex = _make()
    ex.T_right_to_left = None
    results = ex.execute_score({"tree": {"type": "wait", "params": {"arm": "left", "duration": 0.0}}})
    assert len(results) == 1 and not results[0].success
    assert "calibration" in results[0].message


def test_pipeline_task_fields_reach_both_arms():
    ex = _make()
    ex._destination_labels = {"tray"}
    ex._task_verify_block = {"enabled": True}
    for sub in ex._arm_executors.values():
        assert sub._destination_labels == {"tray"}
        assert sub._task_verify_block == {"enabled": True}
