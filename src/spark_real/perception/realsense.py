"""Intel RealSense camera with a capture thread and bounded USB recovery."""

import logging
import os
import threading
import time as _time
from typing import Optional, Tuple

import numpy as np

try:
    import pyrealsense2 as rs

    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False
from spark_real.perception.camera_base import (
    USB_LIFECYCLE_LOCK,
    CameraConfig,
    teardown_in_flight,
    usb_reset_disabled,
)

logger = logging.getLogger(__name__)

# Shared librealsense context. On the RSUSB backend (mandatory on this host;
# see /etc/modprobe.d/blacklist-uvcvideo.conf) every rs.context() re-inits
# libusb and re-walks the bus, so fresh contexts per call multiply
# enumeration traffic. One context per process, created on first use.
_RS_CONTEXT = None


def _get_rs_context():
    global _RS_CONTEXT
    if _RS_CONTEXT is None:
        _RS_CONTEXT = rs.context()
    return _RS_CONTEXT


class RealSenseCamera:
    """
    Intel RealSense D435/D455 RGB-D camera with background capture thread.
    """

    # hardware_reset recovery is bounded PER PROCESS (class-level counter):
    # a reset storm on a wedged link is a prime xHCI-corruption aggravator on
    # this host, where the D435i must run direct to the motherboard and a
    # reset is anything but cheap. Gaps between attempts back off 30/60 s.
    MAX_RECOVER_ATTEMPTS = 3
    RECOVER_BACKOFF_BASE_S = 30.0
    CLOSE_JOIN_TIMEOUT_S = 10.0
    _recover_attempts_total = 0
    _last_recover_ts = None

    def __init__(
        self, serial: str = None, config: CameraConfig = None, color_only: bool = False
    ):
        if not HAS_REALSENSE:
            raise ImportError("pyrealsense2 not installed. pip install pyrealsense2")
        self.serial = serial
        self.config = config or CameraConfig()
        # Stream color only (skip depth) to roughly halve the SuperSpeed UVC
        # bandwidth: the D435i color+depth+IMU triplet on a marginal link
        # desyncs the xHCI event ring ("Event dma ... not part of TD" on the
        # video endpoint) and the stream wedges. The wrist is a 2D camera, so
        # depth is dead weight.
        self._color_only = bool(color_only)
        # Env override (reversible test hook): SPARK_WRIST_COLOR_ONLY=0 re-enables
        # depth without touching config, e.g. to trial wrist-depth refinement.
        _env = os.environ.get("SPARK_WRIST_COLOR_ONLY")
        if _env is not None:
            self._color_only = _env not in ("0", "false", "False", "")
        self._pipeline = None
        self._align = None
        self._latest_rgb = None
        self._latest_depth = None
        self._lock = __import__("threading").Lock()
        self._thread = None
        self._running = False
        # Set once at the start of shutdown (request_stop). The capture loop
        # and recovery path poll it so close() can interrupt a recovery
        # mid-flight instead of racing it.
        self._closing = threading.Event()
        self._reset_disabled_logged = False
        self._recover_exhausted_logged = False
        # Teardown reentrancy state, same rationale as AzureKinectCamera:
        # pipeline.shutdown(), /api/kinect_reset, the SIGTERM handler and
        # atexit can all reach close() and none of them coordinate. Without
        # this, two closers each call rs.pipeline.stop() on the SAME
        # librealsense handle.
        self._close_lock = threading.Lock()
        self._closed = False
        self._close_in_progress = False

    def open(self):
        # Probe the USB link before configuring streams: D435i firmware needs
        # SuperSpeed for the color+depth+IMU triplet, so a USB 2.x link falls
        # back to color-only at low fps for calibration and 2D detection.
        # The whole open (probe + stream start) is a lifecycle transition and
        # is serialized against every other open/close/reset in the process.
        with USB_LIFECYCLE_LOCK:
            usb_type = "3.2"
            try:
                ctx = _get_rs_context()
                for d in ctx.query_devices():
                    if (
                        not self.serial
                        or d.get_info(rs.camera_info.serial_number) == self.serial
                    ):
                        usb_type = d.get_info(rs.camera_info.usb_type_descriptor)
                        break
            except Exception:
                pass
            usb2_mode = usb_type.startswith("2.")
            if usb2_mode:
                logger.warning(
                    "RealSense link is USB %s (not SuperSpeed): depth disabled, "
                    "color forced to 6 fps. Replace the cable with a SuperSpeed-"
                    "rated USB-A/C to USB-C to restore depth + 30 fps.",
                    usb_type,
                )

            self._usb2_mode = usb2_mode
            self._start_pipeline()

        # Start background capture thread
        self._running = True
        self._thread = __import__("threading").Thread(
            target=self._capture_loop, daemon=True
        )
        self._thread.start()

    def _start_pipeline(self):
        """
        Build and start the stream pipeline.

        Shared by open() and the capture-loop recovery path, so it must be
        safe to call repeatedly.
        """
        self._pipeline = rs.pipeline()
        rs_config = rs.config()
        if self.serial:
            rs_config.enable_device(self.serial)
        color_fps = 6 if self._usb2_mode else 15
        rs_config.enable_stream(
            rs.stream.color,
            self.config.width,
            self.config.height,
            rs.format.rgb8,
            color_fps,
        )
        self._depth_enabled = False
        if not self._usb2_mode and not self._color_only:
            try:
                rs_config.enable_stream(
                    rs.stream.depth,
                    self.config.width,
                    self.config.height,
                    rs.format.z16,
                    15,
                )
                self._depth_enabled = True
            except Exception:
                pass
        profile = self._pipeline.start(rs_config)
        if self._depth_enabled:
            self._align = rs.align(rs.stream.color)
        else:
            self._align = None

        intrinsics = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        self.config.fx = intrinsics.fx
        self.config.fy = intrinsics.fy
        self.config.cx = intrinsics.ppx
        self.config.cy = intrinsics.ppy

        # Drop warmup frames so the first read() is usable: early frames are
        # underexposed while auto-exposure settles and depth comes back nearly
        # empty. Tolerate early timeouts so a flaky link cannot wedge open.
        for _ in range(30):
            if self._closing.is_set():
                break
            try:
                self._pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                break

    def _recovery_allowed(self) -> bool:
        """
        Gate on hardware_reset recovery: kill-switch, per-process attempt
        bound, and exponential backoff between attempts.

        Returns False (with one-time logging) instead of raising so the
        capture loop can keep serving whatever frames still arrive; a
        disallowed recovery means the wrist camera degrades to no-frames
        rather than generating reset traffic on the shared controller.
        """
        if self._closing.is_set() or not self._running:
            return False
        if teardown_in_flight():
            # A Kinect stop() is still running on the shared controller even
            # though USB_LIFECYCLE_LOCK is free. A hardware_reset now is
            # exactly the overlapping-lifecycle pattern the lock exists to prevent.
            logger.warning(
                "RealSense: teardown in flight on another camera; skipping "
                "hardware_reset recovery"
            )
            return False
        if usb_reset_disabled():
            if not self._reset_disabled_logged:
                logger.warning(
                    "USB reset paths disabled (SPARK_ALLOW_USB_RESET unset): "
                    "RealSense hardware_reset recovery off; wrist camera "
                    "degrades to no-frames until replug/restart"
                )
                self._reset_disabled_logged = True
            return False
        cls = RealSenseCamera
        if cls._recover_attempts_total >= self.MAX_RECOVER_ATTEMPTS:
            if not self._recover_exhausted_logged:
                logger.warning(
                    "RealSense: %d/%d hardware_reset attempts used this "
                    "process; giving up on reset recovery (wrist camera "
                    "degraded). Replug or restart to retry.",
                    cls._recover_attempts_total,
                    self.MAX_RECOVER_ATTEMPTS,
                )
                self._recover_exhausted_logged = True
            return False
        if cls._last_recover_ts is not None:
            # Gap after attempt N is base * 2**(N-1): 30 s, 60 s, ...
            gap = self.RECOVER_BACKOFF_BASE_S * (
                2.0 ** (cls._recover_attempts_total - 1)
            )
            if _time.monotonic() - cls._last_recover_ts < gap:
                return False
        return True

    def _try_recover(self):
        """
        Recover a wedged D435i via hardware_reset.

        The camera can wedge into frame-timeout storms (USB link up, zero
        frames); hardware_reset clears it short of replugging. Stop the dead
        pipeline, reset, wait for re-enumeration, restart streams.

        Bounded (MAX_RECOVER_ATTEMPTS per process, exponential backoff) and
        serialized under USB_LIFECYCLE_LOCK: a reset must never overlap a
        Kinect open/close on the same xHCI controller. Honors the
        reset kill-switch via _recovery_allowed(), and aborts
        early if close() is racing it (self._closing).
        """
        if not self._recovery_allowed():
            return False
        with USB_LIFECYCLE_LOCK:
            if self._closing.is_set():
                return False
            cls = RealSenseCamera
            cls._recover_attempts_total += 1
            cls._last_recover_ts = _time.monotonic()
            logger.warning(
                "RealSense: no frames, attempting hardware_reset recovery "
                "(attempt %d/%d this process)",
                cls._recover_attempts_total,
                self.MAX_RECOVER_ATTEMPTS,
            )
            try:
                self._pipeline.stop()
            except Exception:
                pass
            try:
                ctx = _get_rs_context()
                for d in ctx.query_devices():
                    if (
                        not self.serial
                        or d.get_info(rs.camera_info.serial_number) == self.serial
                    ):
                        d.hardware_reset()
                        break
                # Re-enumeration takes ~4-5 s. Event-wait so close() can
                # interrupt instead of racing a sleeping recovery.
                if self._closing.wait(6.0):
                    return False
            except Exception as exc:
                logger.warning("RealSense hardware_reset failed: %s", exc)
            # Re-probe the link speed before re-opening: the camera may have
            # been replugged onto a different port since open() (USB2 at boot,
            # SuperSpeed now, or vice versa), and _usb2_mode decides whether
            # depth streams are enabled.
            try:
                ctx = _get_rs_context()
                for d in ctx.query_devices():
                    if (
                        not self.serial
                        or d.get_info(rs.camera_info.serial_number) == self.serial
                    ):
                        usb_type = d.get_info(rs.camera_info.usb_type_descriptor)
                        new_usb2 = usb_type.startswith("2.")
                        if new_usb2 != self._usb2_mode:
                            logger.info(
                                "RealSense link changed: usb2_mode %s -> %s",
                                self._usb2_mode,
                                new_usb2,
                            )
                            self._usb2_mode = new_usb2
                        break
            except Exception:
                pass
            if self._closing.is_set():
                return False
            try:
                self._start_pipeline()
                logger.info("RealSense: recovered after hardware_reset")
                return True
            except Exception as exc:
                logger.warning("RealSense re-open after reset failed: %s", exc)
                return False

    def _capture_loop(self):
        """
        Background thread: continuously grab frames, keep latest.
        """
        frame_count = 0
        fail_count = 0
        consec_fails = 0
        while self._running:
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=500)
                if self._align:
                    aligned = self._align.process(frames)
                    color_frame = aligned.get_color_frame()
                    depth_frame = aligned.get_depth_frame()
                else:
                    color_frame = frames.get_color_frame()
                    depth_frame = None
                if color_frame:
                    rgb = np.asanyarray(color_frame.get_data()).copy()
                    depth = None
                    if depth_frame:
                        depth = (
                            np.asanyarray(depth_frame.get_data()).astype(np.float32)
                            / 1000.0
                        )
                    with self._lock:
                        self._latest_rgb = rgb
                        self._latest_depth = depth
                    frame_count += 1
                    consec_fails = 0
                    if frame_count % 100 == 0:
                        logger.debug("RealSense: %d frames captured", frame_count)
                _time.sleep(0.01)
            except RuntimeError as e:
                fail_count += 1
                consec_fails += 1
                if fail_count <= 3 or fail_count % 50 == 0:
                    logger.warning("RealSense frame timeout #%d: %s", fail_count, e)
                # Self-heal: a long run of consecutive misses means the link
                # is wedged, not flaky, so try to recover. _recovery_allowed
                # enforces the kill-switch, the per-process attempt bound,
                # and exponential backoff between attempts.
                if consec_fails >= 25 and self._recovery_allowed():
                    if self._try_recover():
                        consec_fails = 0
                # Event-wait instead of sleep so close() interrupts promptly.
                self._closing.wait(0.1)

    def read(self, **kwargs) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Returns latest cached (rgb, depth). Non-blocking.
        """
        with self._lock:
            return self._latest_rgb, self._latest_depth

    def request_stop(self):
        """
        Non-blocking quiesce: stop the capture loop and abort any in-flight
        recovery at its next checkpoint. Called by pipeline.shutdown() BEFORE
        the Kinects close, so a hardware_reset can never fire while another
        camera on the same controller is mid-close. close() completes the
        teardown afterwards; request_stop() alone touches no USB state.
        """
        self._running = False
        self._closing.set()

    def close(self):
        self.request_stop()
        # LOCK ORDER. _close_lock is held ONLY around the flag transitions,
        # never across _close_locked, which joins the capture thread and
        # then takes USB_LIFECYCLE_LOCK. Nesting _close_lock ->
        # USB_LIFECYCLE_LOCK would invert against
        # /api/kinect_reset._kinect_reset_locked, which already holds
        # USB_LIFECYCLE_LOCK when it reaches a camera close.
        with self._close_lock:
            if self._closed:
                return
            if self._close_in_progress:
                logger.warning(
                    "RealSense: close() already in progress on another "
                    "thread; not racing it"
                )
                return
            self._close_in_progress = True
        try:
            self._close_locked()
        finally:
            with self._close_lock:
                self._close_in_progress = False

    def _close_locked(self):
        """Body of close(); exactly one thread is in here at a time."""
        t = self._thread
        if t is not None:
            # JOIN BEFORE CLOSE: the capture thread owns the pipeline handle
            # (wait_for_frames, recovery). _closing interrupts its sleeps and
            # any recovery at the next checkpoint, so this join is normally
            # near-instant.
            t.join(timeout=self.CLOSE_JOIN_TIMEOUT_S)
            if t.is_alive():
                logger.warning(
                    "RealSense capture thread did not exit within %.0fs; "
                    "SKIPPING pipeline.stop() rather than stopping the "
                    "stream out from under a live reader (that race "
                    "corrupts xHCI state). Device releases when the "
                    "process exits.",
                    self.CLOSE_JOIN_TIMEOUT_S,
                )
                return
            self._thread = None
        pipe = self._pipeline
        if pipe is not None:
            # Drop the reference BEFORE stopping so nothing can hand the same
            # handle to a second stop() even if this one raises.
            self._pipeline = None
            with USB_LIFECYCLE_LOCK:
                try:
                    pipe.stop()
                except Exception:
                    pass
        self._closed = True

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()
