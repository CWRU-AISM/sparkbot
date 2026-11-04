"""spec(profile) returns the ur10e camera roster by role without opening hardware."""

from spark_real.config import SparkConfig, load_profile
from spark_real.perception.camera_registry import spec


def test_ur10e_camera_spec():
    s = spec(load_profile(SparkConfig(family="ur10e")))
    assert set(s) == {"birdview", "sideview", "wrist"}, set(s)
    assert s["birdview"]["type"] == "kinect"
    assert s["sideview"]["type"] == "kinect"
    assert s["wrist"]["type"] == "realsense"
