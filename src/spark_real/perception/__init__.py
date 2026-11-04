"""SPARK real-robot perception: cameras, registry, fusion and prompt contracts."""

from spark_core.sam3 import ObjectDetection, SPARKPerception
from spark_real.perception.camera_base import CameraConfig, USBCamera
from spark_real.perception.kinect import AzureKinectCamera
from spark_real.perception.realsense import RealSenseCamera

__all__ = [
    "ObjectDetection",
    "SPARKPerception",
    "AzureKinectCamera",
    "CameraConfig",
    "RealSenseCamera",
    "USBCamera",
]
