"""Azure Kinect frame reads and the bounded close (capture thread joined before device stop)."""

import logging
import threading as _threading
import time as _time

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
from spark_real.perception.camera_base import (
    USB_LIFECYCLE_LOCK,
    _teardown_begin,
    _teardown_end,
)

logger = logging.getLogger(__name__)


class KinectIOMixin:
    """Azure Kinect frame reads and the bounded close (capture thread joined before device stop)."""

    def set_ir_enabled(self, enabled: bool) -> None:
        """
        Turn the IR preview image on/off at runtime (see want_ir).

        Takes effect on the next captured frame; a consumer that has just
        enabled it may get ir=None for one frame period. Cheap and
        idempotent, so callers can set it on every request.
        """
        self._want_ir = bool(enabled)

    def last_frame_age_s(self):
        """Age of the newest cached frame in seconds, or None before the
        first frame. Surfaced into pipeline.get_status so the UI can show a
        dead camera instead of a frozen picture."""
        with self._lock:
            ts = self._latest_frame_ts
        if not ts:
            return None
        return max(0.0, _time.time() - ts)

    def read(self, depth: bool = True, ir: bool = False):
        """
        Returns (rgb, depth) by default; (rgb, depth, ir) when ir=True.

        Served from the single-producer capture thread's cached frame
        under self._lock. Multiple consumers (streamer, perception,
        CBF) can call this concurrently without entering pyk4a from
        more than one thread.

        Args:
            depth: If False, callers receive None in the depth slot
                regardless of what the capture thread cached.
            ir: If True, return (rgb, depth, ir) instead of (rgb, depth).
        """
        # Wait briefly for the first frame to arrive (open() may have
        # just spawned the thread; subordinate cameras only start
        # producing once master is firing).
        deadline = _time.time() + 2.0
        while _time.time() < deadline:
            with self._lock:
                # STALENESS GATE: never serve a cached frame older than
                # STALE_S. A wedged device (USB drop, depth-engine stall)
                # otherwise freezes the scene image silently and the arm
                # moves on stale coordinates. Within the deadline, keep
                # polling in case the capture thread recovers.
                _fresh = (
                    self._latest_frame_ts > 0
                    and (_time.time() - self._latest_frame_ts) <= self.STALE_S
                )
                if self._latest_rgb is not None and _fresh:
                    self._stale_logged = False
                    # Deferred rectification: the capture thread stores raw
                    # frames; the first consumer of a frame pays one remap and
                    # the result is cached for everyone else.
                    if (
                        self._rect_maps is not None
                        and not getattr(self, "_latest_rectified", True)
                        and self._latest_rgb.shape[1] == self._rect_maps[0].shape[1]
                    ):
                        m1, m2 = self._rect_maps
                        self._latest_rgb = cv2.remap(
                            self._latest_rgb, m1, m2, cv2.INTER_LINEAR
                        )
                        if self._latest_depth is not None:
                            # Nearest neighbour: never blend depths across
                            # an object edge.
                            self._latest_depth = cv2.remap(
                                self._latest_depth, m1, m2, cv2.INTER_NEAREST
                            )
                        self._latest_rectified = True
                    rgb = self._latest_rgb
                    depth_m = self._latest_depth if depth else None
                    ir_img = self._latest_ir
                    if ir:
                        return rgb, depth_m, ir_img
                    return rgb, depth_m
            _time.sleep(0.005)
        # First-frame timeout OR stale cache; treat as no signal. Logged once
        # per staleness episode so a 15 Hz stream doesn't flood the log.
        with self._lock:
            if self._latest_rgb is not None and not self._stale_logged:
                self._stale_logged = True
                age = _time.time() - self._latest_frame_ts
                logger.error(
                    "Kinect %d: cached frame is %.1fs old (> %.1fs); serving "
                    "NO frame instead of a frozen scene. Device is likely "
                    "wedged; see /api/kinect_reset.",
                    self.device_id,
                    age,
                    self.STALE_S,
                )
        return (None, None, None) if ir else (None, None)

    def close(self) -> bool:
        """
        Release the device. Returns True ONLY on a full, clean release.

        Returns False when the capture thread would not exit, or when the
        SDK's stop() had to be abandoned -- i.e. when a live thread may
        still be doing libusb I/O on this device. Callers that are about to
        power-cycle, deauthorize or reopen the device MUST check this;
        doing any of those to a device with a live capture thread is the
        concurrent-teardown pattern behind the xHCI ring corruption.

        A failed close is RETRYABLE: _closed is set only after a clean
        release, so a later close() can try again.
        """
        # Reentrancy guard: SIGTERM handler + atexit + on_event("shutdown")
        # may all race to close the same device. One closer at a time; a
        # completed close short-circuits, an in-flight one reports failure
        # rather than tearing down the same handle twice.
        with self._close_lock:
            if self._closed:
                return True
            if self._close_in_progress:
                logger.warning(
                    "Kinect %d: close() already in progress on another "
                    "thread; reporting failure rather than racing it",
                    self.device_id,
                )
                return False
            self._close_in_progress = True
        ok = False  # an exception out of _close_locked is a failed close
        try:
            ok = self._close_locked()
        finally:
            with self._close_lock:
                self._close_in_progress = False
                if ok:
                    self._closed = True
        return ok

    def _close_locked(self) -> bool:
        """Body of close(); exactly one thread is in here at a time."""
        ok = True
        self._stop_flag = True
        t = self._thread
        if t is not None:
            # JOIN BEFORE CLOSE: the capture thread owns the device handle
            # (get_capture + decoders). get_capture polls with a 250 ms
            # timeout, so a healthy thread exits almost immediately; give a
            # grace retry, and if it STILL will not die, libk4a is wedged
            # and stopping the device from this thread would race the
            # in-flight get_capture -- the exact concurrent-teardown pattern
            # behind the xHCI TRB corruption. Leave the handle to process
            # exit instead.
            t.join(timeout=self.JOIN_TIMEOUT_S)
            if t.is_alive():
                logger.warning(
                    "Kinect %d capture thread still running after %.0fs; "
                    "waiting up to %.0fs more before releasing the device",
                    self.device_id,
                    self.JOIN_TIMEOUT_S,
                    self.JOIN_GRACE_S,
                )
                t.join(timeout=self.JOIN_GRACE_S)
            if t.is_alive():
                logger.warning(
                    "Kinect %d capture thread did not exit; NOT stopping "
                    "the device from another thread (would race libk4a "
                    "mid-capture). Device releases at process exit; the "
                    "depth MCU may need a power-cycle on next launch.",
                    self.device_id,
                )
                # FAILED close: the handle is still open and a live thread
                # may still be inside libusb on it. Report it so callers do
                # not power-cycle or reopen underneath that thread, and
                # leave _closed unset so a later attempt can retry.
                return False
            self._thread = None

        device = self._device
        if device is None:
            return ok
        self._device = None

        # Run device.stop() on a worker so a hung call inside libk4a
        # can't wedge the whole shutdown path. pyk4a's stop() already
        # calls _device_close() internally; calling close() again
        # raises "Device is not opened", so it is not called here. The stop itself is a
        # lifecycle transition: hold USB_LIFECYCLE_LOCK for its bounded
        # window so it cannot overlap a RealSense reset or another open.
        # _teardown_begin/_teardown_end bracket the SDK call itself, not the
        # wait for it: when the 4 s budget expires USB_LIFECYCLE_LOCK is dropped
        # while device.stop() may still be issuing SET_INTERFACE / endpoint
        # halts / hub traffic. teardown_in_flight() stays True until the
        # worker really returns, and every bus-touching path checks it.
        def _release():
            try:
                device.stop()
            except Exception as exc:
                logger.warning("Kinect %d stop failed: %s", self.device_id, exc)
            finally:
                _teardown_end()

        _teardown_begin()
        with USB_LIFECYCLE_LOCK:
            t = _threading.Thread(
                target=_release, daemon=True, name=f"kinect{self.device_id}-release"
            )
            t.start()
            t.join(timeout=self.RELEASE_JOIN_TIMEOUT_S)
            if t.is_alive():
                logger.warning(
                    "Kinect %d stop/close exceeded %.0fs, abandoning the wait. "
                    "teardown_in_flight() stays True until libk4a returns, so "
                    "no reset/power-cycle/open can overlap it. Depth MCU may "
                    "need a reset on next launch.",
                    self.device_id,
                    self.RELEASE_JOIN_TIMEOUT_S,
                )
                ok = False
        return ok
