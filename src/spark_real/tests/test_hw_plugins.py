"""Camera and gripper drivers named by class path, and Kinect roles from the `cameras:` roster."""
import sys
import types

from spark_real.grippers import make_gripper
from spark_real.perception.camera_base import CameraConfig
from spark_real.perception.camera_registry import assign_kinect_roles, build_camera_registry


class FakeCam:
    def __init__(self, url, width=320, height=240):
        self.url = url
        self.config = CameraConfig(width=width, height=height)
        self.opened = False

    def open(self):
        self.opened = True


class FakeJaw:
    def __init__(self, port, speed=1.0):
        self.port, self.speed = port, speed


sys.modules["my_rig"] = types.SimpleNamespace(FakeCam=FakeCam, FakeJaw=FakeJaw)


def test_plugin_camera_gets_its_keys_and_opens():
    reg = build_camera_registry([{"role": "top", "type": "my_rig:FakeCam", "url": "rtsp://x", "width": 640, "arm": "left"}])
    cam = reg["top"].device
    assert cam.opened and cam.url == "rtsp://x" and cam.config.width == 640
    assert reg["top"].kind == "my_rig:FakeCam" and reg.wrist_for_arm == {"left": "top"}


def test_plugin_gripper_gets_its_keys():
    jaw = make_gripper({"type": "my_rig:FakeJaw", "port": "/dev/ttyUSB3", "tcp_offset_m": 0.1})
    assert jaw.port == "/dev/ttyUSB3" and jaw.GRIPPER_TYPE == "FakeJaw"


ROSTER = [
    {"role": "birdview", "type": "kinect", "serial": "B"},
    {"role": "sideview", "type": "kinect", "serial": "S", "sync": "master"},
]


def test_kinect_roles_follow_roster_serials_and_master_opens_last():
    assert assign_kinect_roles({0: "S", 1: "B"}, ROSTER, use_hw_sync=True) == [
        (1, "birdview", "SUBORDINATE"), (0, "sideview", "MASTER")]


def test_kinects_without_serials_fill_roles_in_roster_order():
    assert assign_kinect_roles({0: "X"}, [{"role": "sideview", "type": "kinect"}]) == [(0, "sideview", "STANDALONE")]
    assert assign_kinect_roles({0: None, 1: None}, []) == [(0, "sideview", "STANDALONE"), (1, "birdview", "STANDALONE")]


def test_master_serial_override_wins():
    plan = assign_kinect_roles({0: "S", 1: "B"}, ROSTER, master_serial="B", use_hw_sync=True)
    assert dict((d, (r, s)) for d, r, s in plan) == {1: ("sideview", "MASTER"), 0: ("birdview", "SUBORDINATE")}
