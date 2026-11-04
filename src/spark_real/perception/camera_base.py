"""Camera basics: the process-wide USB lifecycle lock, teardown tracking, CameraConfig and a plain USB camera."""

import os
import threading
import time as _time
from typing import Tuple

import numpy as np

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
try:
    import yaml
except ImportError:
    yaml = None


# One process-wide lock serializing EVERY USB lifecycle transition (device
# open, close, hardware_reset, bus re-enumeration) across all camera classes.
# Steady-state streaming is fine; concurrent lifecycle churn corrupts the
# xHCI controller's TRB rings (kernel logs "Event dma ... not part of TD")
# and can hard-kill the host. RLock so a composite transition (e.g.
# /api/kinect_reset closing then reopening) can nest the per-device close
# under its own hold.
USB_LIFECYCLE_LOCK = threading.RLock()

# Count of device teardowns whose SDK call has been ABANDONED but may still
# be executing. close() runs device.stop() on a worker and gives up waiting
# after 4 s, releasing USB_LIFECYCLE_LOCK while libk4a may still be issuing
# SET_INTERFACE, endpoint halts and hub traffic on the shared controller.
# Anything that would touch the bus (uhubctl power-cycle, sysfs deauthorize,
# hardware_reset, a fresh open) must refuse while this is non-zero.
# Incremented before the release worker starts, decremented by the worker
# itself, so it stays set for as long as the SDK call really runs.
_TEARDOWN_LOCK = threading.Lock()

_TEARDOWN_IN_FLIGHT = 0


def teardown_in_flight() -> bool:
    """True while an abandoned-but-still-running device teardown may be
    driving USB traffic. See _TEARDOWN_IN_FLIGHT."""
    with _TEARDOWN_LOCK:
        return _TEARDOWN_IN_FLIGHT > 0


def _teardown_begin() -> None:
    global _TEARDOWN_IN_FLIGHT
    with _TEARDOWN_LOCK:
        _TEARDOWN_IN_FLIGHT += 1


def _teardown_end() -> None:
    global _TEARDOWN_IN_FLIGHT
    with _TEARDOWN_LOCK:
        _TEARDOWN_IN_FLIGHT = max(0, _TEARDOWN_IN_FLIGHT - 1)


def await_teardown_clear(timeout: float = 15.0, poll: float = 0.1) -> bool:
    """
    Block until no abandoned teardown is in flight. Returns False on timeout.

    Callers about to touch the bus (open, reset, power-cycle) use this
    instead of assuming USB_LIFECYCLE_LOCK covered the teardown -- it does
    not once close() has stopped waiting for libk4a.
    """
    deadline = _time.monotonic() + timeout
    while teardown_in_flight():
        if _time.monotonic() >= deadline:
            return False
        _time.sleep(poll)
    return True


def usb_reset_disabled() -> bool:
    """
    True (reset paths OFF) unless SPARK_ALLOW_USB_RESET is set.

    Every software-initiated USB reset path (the Kinect capture loop's
    bounded device stop/start, the RealSense hardware_reset recovery, and
    /api/kinect_reset) fires unattended, precisely when the controller is
    ALREADY misbehaving (40 consecutive capture failures, 25 consecutive
    frame timeouts), and reset traffic on a sick controller precedes the
    ring desync in the kernel log. A degraded wrist camera is strictly
    better than a dead host.

    SPARK_NO_USB_RESET still wins if set. Read at call time, not import time.
    """
    def _set(name: str) -> bool:
        # kill switch: any value except empty/0/false counts, so a typo cannot re-enable resets
        return os.environ.get(name, "") not in ("", "0", "false", "False")
    if _set("SPARK_NO_USB_RESET"):
        return True
    return not _set("SPARK_ALLOW_USB_RESET")


class CameraConfig:
    """
    Camera intrinsics and extrinsics.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fx: float = 615.0,
        fy: float = 615.0,
        cx: float = 320.0,
        cy: float = 240.0,
        extrinsic: np.ndarray = None,
    ):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.extrinsic = extrinsic if extrinsic is not None else np.eye(4)

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        """
        3x3 camera intrinsic matrix.
        """
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

    def backproject(self, u: float, v: float, depth: float) -> np.ndarray:
        """
        Backproject pixel + depth to 3D camera frame.
        """
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        return np.array([x, y, depth])

    def to_world(self, point_camera: np.ndarray) -> np.ndarray:
        """
        Transform point from camera frame to world frame.
        """
        p_hom = np.append(point_camera, 1.0)
        return (self.extrinsic @ p_hom)[:3]

    @classmethod
    def from_yaml(cls, path: str) -> "CameraConfig":
        """
        Load from YAML calibration file.
        """
        if yaml is None:
            raise RuntimeError("PyYAML not installed; CameraConfig.from_yaml unavailable")

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            width=data.get("width", 640),
            height=data.get("height", 480),
            fx=data["fx"],
            fy=data["fy"],
            cx=data["cx"],
            cy=data["cy"],
            extrinsic=(
                np.array(data["extrinsic"]).reshape(4, 4)
                if "extrinsic" in data
                else None
            ),
        )


class USBCamera:
    """
    Simple USB camera via OpenCV.
    """

    def __init__(self, device_id: int = 0, config: CameraConfig = None):
        self.device_id = device_id
        self.config = config or CameraConfig()
        self._cap = None

    def open(self):
        self._cap = cv2.VideoCapture(self.device_id)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.device_id}")

    def read(self, **kwargs) -> Tuple[np.ndarray, None]:
        """
        Returns (rgb_image, None). No depth from a USB camera; accepts and
        ignores depth/ir kwargs so registry-wide read loops (streaming) can
        call every camera uniformly.
        """
        if self._cap is None:
            return None, None
        ret, frame = self._cap.read()
        if not ret:
            return None, None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), None

    def close(self):
        if self._cap:
            self._cap.release()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()
