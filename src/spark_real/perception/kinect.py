"""Azure Kinect camera: open/close with sync modes, GIL release and a capture thread."""

import logging
import os
import threading
import time as _time

import numpy as np

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
from spark_real.perception.camera_base import (
    USB_LIFECYCLE_LOCK,
    CameraConfig,
    await_teardown_clear,
    usb_reset_disabled,
)
from spark_real.perception.kinect_io import KinectIOMixin  # noqa: F401

logger = logging.getLogger(__name__)

# pyk4a is imported inside AzureKinectCamera.open() (not at module top)
# to avoid a glibc tpp assertion on PREEMPT_RT kernels when other native
# libs (scipy BLAS) are loaded first. This is a load-ordering
# hazard on the realtime-kernel rig, not a circular import.
HAS_K4A = False

pyk4a = None

PyK4A = None

K4AConfig = None


class K4ATimeoutException(Exception):
    pass


def _kinect_release_gil() -> bool:
    """Whether Kinect device calls should drop the GIL while they block.

    Default True. Set SPARK_KINECT_HOLD_GIL=1 to restore pyk4a's stock
    behaviour without editing config -- an escape hatch, not a tuning knob.
    """
    return os.environ.get("SPARK_KINECT_HOLD_GIL", "").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    )


def _apply_release_gil(device, release_gil: bool) -> None:
    """Clear pyk4a's `thread_safe` flag so blocking k4a calls drop the GIL.

    pyk4a's flag is INVERTED relative to how it reads: in k4a_module.c every
    entry point releases the GIL (PyEval_SaveThread) only when
    `thread_safe == 0`, so pyk4a's default `thread_safe=True` means "hold
    the GIL for the whole call".

    _capture_loop blocks in `device.get_capture(timeout=250)` for ~66 ms at
    15 fps. Holding the GIL across that wait freezes the WHOLE interpreter:
    with two Kinects running the capture threads hold the GIL ~99% of the
    time, and a CPU-bound SAM3 detect (0.20 s of CPU for a 6-prompt detect)
    takes 20-50 s of wall time (measured slowdown 22x with one capture
    thread, 241x with two).

    Clearing the flag is safe because after open() returns only the capture
    thread touches the device, and close() joins that thread before it stops
    the device (and refuses to stop it at all if the join fails). pyk4a's
    thread_safe does not cover the capture decoders or the shared
    transformation_handle anyway.

    Capture/calibration objects inherit the flag from the device at creation,
    so setting it here also covers capture.color and capture.transformed_depth.
    """
    if not release_gil:
        return
    try:
        device.thread_safe = False
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "could not clear pyk4a thread_safe (%s); Kinect calls will hold "
            "the GIL and SAM3 detects will be 10-100x slower",
            exc,
        )


class AzureKinectCamera(KinectIOMixin):
    """
    Azure Kinect DK RGB-D camera.
    """

    # close(): how long to wait for the capture thread before the grace
    # retry, and the grace itself. If the thread outlives both, the device
    # is NOT stopped from another thread (see close()).
    JOIN_TIMEOUT_S = 2.0
    JOIN_GRACE_S = 6.0
    # How long close() waits for libk4a's device.stop() before it gives up
    # waiting (the call itself keeps running; see teardown_in_flight).
    RELEASE_JOIN_TIMEOUT_S = 4.0

    def __init__(
        self,
        device_id: int = 0,
        config: CameraConfig = None,
        color_resolution: str = "1080P",
        depth_mode: str = "NFOV_UNBINNED",
        sync_mode: str = "STANDALONE",
        subordinate_delay_usec: int = 0,
        camera_fps: int = 30,
        pre_started_device=None,
        want_depth: bool = True,
        want_ir: bool = False,
        release_gil: bool = None,
    ):
        """
        Args:
            device_id: Device index (0 for first Kinect).
            config: Camera intrinsics/extrinsics config.
            color_resolution: "720P", "1080P", "1440P", "2160P".
            depth_mode: "NFOV_UNBINNED", "NFOV_2X2BINNED",
                "WFOV_UNBINNED", "WFOV_2X2BINNED".
            sync_mode: "STANDALONE" | "MASTER" | "SUBORDINATE".  When two
                Kinects are wired together via the 3.5mm sync cable, set
                one to MASTER and the other to SUBORDINATE.  Subordinates
                wait for the master trigger and only emit one frame per
                trigger, which dramatically reduces aggregate USB
                bandwidth versus two free-running STANDALONE devices.
                Open subordinates FIRST then master because the SDK
                requires the slave already be waiting for the trigger
                before the master starts firing.
            subordinate_delay_usec: For SUBORDINATE only.  Phase offset
                relative to master pulse; 0 means stagger as tightly as
                the depth-laser duty cycle allows (~160 us).
            camera_fps: 5, 15, or 30. WFOV_UNBINNED caps at 15 in
                hardware; higher requests are clamped.
            want_depth: Decode transformed_depth in the capture loop.
                Leave True for any camera whose depth is consumed (both
                Kinects). Setting it False stops the k4a
                transformation engine running for that camera and stops
                the per-frame colour-sized depth allocation.
            release_gil: Drop the GIL while blocking k4a calls run (see
                _apply_release_gil). None takes the process default, which
                is True unless SPARK_KINECT_HOLD_GIL is set. Leave it True:
                False is pyk4a's stock behaviour and it freezes the whole
                interpreter for the duration of every get_capture wait.
            want_ir: Build the IR preview image in the capture loop.
                Default False: it is consumed only by the ir=True
                preview/detect mode. Call set_ir_enabled(True) to turn it
                on at runtime.

        NOTE: neither flag changes what is on the USB wire. Depth mode is
        fixed at device-open time, so the depth stream is transmitted at
        the configured fps regardless. These flags cut host-side CPU, GPU
        transformation dispatches and allocation churn only. The wire is
        controlled by resolution / fps / depth_mode.
        """
        self.device_id = device_id
        self.config = config or CameraConfig(width=1920, height=1080)
        self._color_resolution = color_resolution
        self._depth_mode = depth_mode
        self._sync_mode = (sync_mode or "STANDALONE").upper()
        self._subordinate_delay_usec = int(subordinate_delay_usec)
        # WFOV_UNBINNED is the only mode that physically caps at 15 fps.
        if depth_mode == "WFOV_UNBINNED" and camera_fps > 15:
            camera_fps = 15
        self._camera_fps = int(camera_fps)
        self._device = None
        self._pre_started = pre_started_device
        self._warmup_frames = 30
        self._want_depth = bool(want_depth)
        self._want_ir = bool(want_ir)
        self._release_gil = _kinect_release_gil() if release_gil is None else bool(release_gil)

        # Dedicated capture-thread state. Multiple FastAPI executor threads can
        # call read() concurrently, but pyk4a's thread_safe flag does not cover
        # the capture property decoders or the shared transformation_handle, so
        # concurrent device access can abort the process from libk4a's C++
        # threads. Serialise into one producer thread that owns the device and
        # caches the latest RGB/depth/IR under a lock for read(). Mirrors
        # RealSenseCamera above.
        self._lock = threading.Lock()
        self._thread = None
        self._stop_flag = False
        self._latest_rgb = None
        self._latest_depth = None
        self._latest_ir = None
        self._latest_frame_ts = 0.0
        # Staleness gate state: once the cached frame ages past STALE_S,
        # read() refuses to serve it (logged once per staleness episode).
        self._stale_logged = False
        # Bounded in-process capture-loop restarts (see _capture_loop).
        self._loop_restarts = 0
        # Teardown reentrancy state. Built HERE, not lazily in close(): a
        # lazy `hasattr` check is itself a race, and two first-time closers
        # (SIGTERM handler and on_event("shutdown"), or /api/kinect_reset
        # and atexit) would each build a SEPARATE Lock and both run
        # device.stop() on the same libk4a handle.
        self._close_lock = threading.Lock()
        self._closed = False
        self._close_in_progress = False

    # Max age (seconds) of the cached frame that read() will serve. A wedged
    # device otherwise serves the SAME frozen frame forever: perception, the
    # executor's redetect and the wrist servo all silently operate on a stale
    # scene and the arm moves on stale coordinates. Env-tunable.
    STALE_S = float(os.environ.get("SPARK_KINECT_STALE_S", "2.0"))
    # Consecutive get_capture failures before the loop attempts one bounded
    # in-process device restart (mirrors the RealSense _try_recover pattern).
    _FAIL_RESTART_THRESHOLD = 40
    _MAX_LOOP_RESTARTS = 2


    def open(self):
        """
        Start the Azure Kinect device and update intrinsics.
        """
        # USB_LIFECYCLE_LOCK is necessary but NOT sufficient: an abandoned
        # device.stop() can still be running with the lock released, and
        # opening a device while libk4a is mid-teardown on the same
        # controller is the overlap this whole subsystem exists to prevent.
        # Checked before anything else so no SDK state is built first.
        if not await_teardown_clear():
            raise RuntimeError(
                f"Kinect {self.device_id}: refusing to open while a device "
                "teardown is still in flight (libk4a stop() has not "
                "returned). Restart the server rather than layering an open "
                "on top of an unfinished close."
            )
        global pyk4a, PyK4A, K4AConfig, HAS_K4A, K4ATimeoutException
        if not HAS_K4A:
            import pyk4a as _k4a

            pyk4a = _k4a
            PyK4A = _k4a.PyK4A
            K4AConfig = _k4a.Config
            from pyk4a.errors import K4ATimeoutException as _kte

            K4ATimeoutException = _kte
            HAS_K4A = True
        res_map = {
            "720P": pyk4a.ColorResolution.RES_720P,
            "1080P": pyk4a.ColorResolution.RES_1080P,
            "1440P": pyk4a.ColorResolution.RES_1440P,
            "2160P": pyk4a.ColorResolution.RES_2160P,
        }
        depth_map = {
            "NFOV_UNBINNED": pyk4a.DepthMode.NFOV_UNBINNED,
            "NFOV_2X2BINNED": pyk4a.DepthMode.NFOV_2X2BINNED,
            "WFOV_UNBINNED": pyk4a.DepthMode.WFOV_UNBINNED,
            "WFOV_2X2BINNED": pyk4a.DepthMode.WFOV_2X2BINNED,
        }
        fps_map = {
            5: pyk4a.FPS.FPS_5,
            15: pyk4a.FPS.FPS_15,
            30: pyk4a.FPS.FPS_30,
        }
        sync_map = {
            "STANDALONE": pyk4a.WiredSyncMode.STANDALONE,
            "MASTER": pyk4a.WiredSyncMode.MASTER,
            "SUBORDINATE": pyk4a.WiredSyncMode.SUBORDINATE,
        }
        k4a_config = K4AConfig(
            color_resolution=res_map.get(
                self._color_resolution, pyk4a.ColorResolution.RES_720P
            ),
            depth_mode=depth_map.get(self._depth_mode, pyk4a.DepthMode.NFOV_UNBINNED),
            camera_fps=fps_map.get(self._camera_fps, pyk4a.FPS.FPS_15),
            synchronized_images_only=True,
            wired_sync_mode=sync_map.get(
                self._sync_mode, pyk4a.WiredSyncMode.STANDALONE
            ),
            subordinate_delay_off_master_usec=(
                self._subordinate_delay_usec if self._sync_mode == "SUBORDINATE" else 0
            ),
        )
        # Device bring-up (start + warmup + calibration read) is a lifecycle
        # transition: serialize it against every other camera open/close/
        # reset in the process (notably RealSense hardware_reset recovery).
        with USB_LIFECYCLE_LOCK:
            if self._pre_started is not None:
                self._device = self._pre_started
                self._pre_started = None
            else:
                self._device = PyK4A(k4a_config, device_id=self.device_id)
                self._device.start()

            # Do this before the first get_capture below, and do it here
            # rather than at the PyK4A() call so it also covers a device
            # handed in pre-started (pipeline_init/server open those).
            _apply_release_gil(self._device, self._release_gil)

            # Warm up auto-exposure. Subordinates are armed but produce no
            # frames until the master starts firing triggers; calling
            # get_capture() on them here would block forever. Skip warmup
            # in that case and let the caller do a deferred warmup after
            # the master is up.
            if self._sync_mode != "SUBORDINATE":
                for _ in range(self._warmup_frames):
                    self._device.get_capture()

            # Update intrinsics from calibration. Calibration is read from
            # the device factory data and does not require a captured frame.
            cal = self._device.calibration
            K = cal.get_camera_matrix(pyk4a.CalibrationType.COLOR)
            self.config.fx = float(K[0, 0])
            self.config.fy = float(K[1, 1])
            self.config.cx = float(K[0, 2])
            self.config.cy = float(K[1, 2])

            # Update resolution from first capture, but skip on subordinate
            # for the same reason as the warmup loop.
            if self._sync_mode != "SUBORDINATE":
                test_cap = self._device.get_capture()
                if test_cap.color is not None:
                    self.config.height, self.config.width = test_cap.color.shape[:2]

        if self._sync_mode == "SUBORDINATE":
            # Subordinate resolution matches master's; the calibration
            # block above already set fx/fy/cx/cy from the device's
            # factory intrinsics, which is what downstream code needs.
            color_res_map = {
                "720P": (1280, 720),
                "1080P": (1920, 1080),
                "1440P": (2560, 1440),
                "2160P": (3840, 2160),
            }
            w, h = color_res_map.get(self._color_resolution, (1920, 1080))
            self.config.width, self.config.height = w, h

        # The SDK serves images in DISTORTED pixel space (per k4a docs "2D
        # coordinates always refer to the distorted image"), but every
        # downstream consumer here deprojects with a pure pinhole model.
        # Rectify RGB + depth once in the capture thread so the pinhole
        # math is exact everywhere. The LUT is built the way the SDK's own
        # undistort example does it (k4a_calibration_3d_to_2d per ray);
        # the maps reproject onto the SAME K, so fx/fy/cx/cy stay valid.
        self._rect_maps = None
        try:
            dist = cal.get_distortion_coefficients(pyk4a.CalibrationType.COLOR)
            if np.any(np.abs(np.asarray(dist)) > 1e-9):
                self._rect_maps = self._build_rect_maps(
                    cal,
                    K,
                    np.asarray(dist, dtype=np.float64),
                    (self.config.width, self.config.height),
                )
        except Exception as exc:
            logger.warning(
                "Kinect %d: distortion rectification unavailable (%s); "
                "captures stay in distorted pixel space",
                self.device_id,
                exc,
            )

        # All single-threaded device setup is done; from this point on,
        # ONLY the capture thread touches self._device.
        self._stop_flag = False
        self._latest_frame_ts = _time.time()
        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name=f"kinect{self.device_id}-capture",
        )
        self._thread.start()

    def _build_rect_maps(self, cal, K, dist, size):
        """
        Undistortion LUT built the way the SDK's undistort example does:
        for every pixel of the target pinhole image, cast the ray through K
        and ask the SDK (k4a_calibration_3d_to_2d via pyk4a) where that ray
        lands in the real, distorted image. The field is smooth, so it is
        sampled every 8 px and bilinearly upsampled; cv2 only executes the
        remap. The OpenCV Brown-Conrady map is computed as a cross-check
        (disagreement logged) and fills the few fringe pixels the SDK marks
        invalid. Returns CV_16SC2 maps for fast remap.
        """
        w, h = size
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        step = 8
        us = np.linspace(0.0, w - 1.0, int(np.ceil(w / step)) + 1)
        vs = np.linspace(0.0, h - 1.0, int(np.ceil(h / step)) + 1)
        mx = np.full((len(vs), len(us)), np.nan, np.float32)
        my = np.full((len(vs), len(us)), np.nan, np.float32)
        for j, v in enumerate(vs):
            ry = (v - cy) / fy
            for i, u in enumerate(us):
                rx = (u - cx) / fx
                try:
                    du, dv = cal.convert_3d_to_2d(
                        (rx * 1000.0, ry * 1000.0, 1000.0), pyk4a.CalibrationType.COLOR
                    )
                    mx[j, i] = du
                    my[j, i] = dv
                except Exception:
                    pass  # ray outside the calibrated FOV; cv2 fills below
        map_x = cv2.resize(mx, size, interpolation=cv2.INTER_LINEAR)
        map_y = cv2.resize(my, size, interpolation=cv2.INTER_LINEAR)
        c1, c2 = cv2.initUndistortRectifyMap(K, dist, None, K, size, cv2.CV_32FC1)
        ok = np.isfinite(map_x) & np.isfinite(map_y)
        if ok.any():
            dd = np.hypot(map_x[ok] - c1[ok], map_y[ok] - c2[ok])
            logger.info(
                "Kinect %d rectification: SDK-built LUT (%.0f%% px), OpenCV "
                "model agrees to %.2f px max / %.2f px median",
                self.device_id,
                100.0 * ok.mean(),
                float(dd.max()),
                float(np.median(dd)),
            )
        map_x = np.where(ok, map_x, c1).astype(np.float32)
        map_y = np.where(ok, map_y, c2).astype(np.float32)
        gx, gy = np.meshgrid(
            np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
        )
        disp = np.hypot(map_x - gx, map_y - gy)
        logger.info(
            "Kinect %d color distortion: max correction %.1f px, median "
            "%.1f px; rectifying all captures",
            self.device_id,
            float(disp.max()),
            float(np.median(disp)),
        )
        m1, m2 = cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)
        return m1, m2

    def _capture_loop(self):
        """
        Single-producer capture loop.

        Polls get_capture with a short timeout so the loop can react
        to close(); pulls every consumer-relevant image off the
        capture handle inside this thread (so the C++ lazy decoders
        and the shared transformation_handle are never touched from
        anywhere else); stashes copies into latest_* under the lock.
        Consumers read those copies via read(); they never see the
        underlying capture handle.

        QUEUE DRAIN: after each blocking get_capture, non-blocking-poll
        the device for additional buffered frames and skip ahead to the
        newest. The K4A SDK maintains an internal frame queue; if a
        consumer stalls this thread the SDK keeps queueing frames, and
        processing them one-at-a-time falls further behind until the SDK
        drops frames and wedges the device. Draining at the C level keeps
        the loop on the latest frame. ``timeout_ms=0`` polls without blocking.
        """
        device = self._device
        consecutive_failures = 0
        while not self._stop_flag:
            try:
                # 250ms is short enough for close() to be responsive
                # and long enough that no frame is missed at 30fps.
                capture = device.get_capture(timeout=250)
                consecutive_failures = 0
            except K4ATimeoutException:
                continue
            except Exception as exc:
                consecutive_failures += 1
                logger.warning("Kinect %d get_capture failed: %s", self.device_id, exc)
                # Bounded in-process recovery, mirroring RealSense
                # _try_recover: after a run of hard failures, stop+start the
                # device once or twice under USB_LIFECYCLE_LOCK. Honors the
                # reset kill-switch (OFF by default): a device restart
                # re-runs SET_INTERFACE bandwidth negotiation while the OTHER
                # Kinect's isochronous reservation is live on the same
                # controller.
                if (
                    consecutive_failures >= self._FAIL_RESTART_THRESHOLD
                    and self._loop_restarts < self._MAX_LOOP_RESTARTS
                    and not usb_reset_disabled()
                ):
                    self._loop_restarts += 1
                    logger.error(
                        "Kinect %d: %d consecutive capture failures; bounded "
                        "device restart %d/%d",
                        self.device_id,
                        consecutive_failures,
                        self._loop_restarts,
                        self._MAX_LOOP_RESTARTS,
                    )
                    try:
                        with USB_LIFECYCLE_LOCK:
                            if self._stop_flag:
                                break
                            try:
                                device.stop()
                            except Exception:  # noqa: BLE001
                                pass
                            _time.sleep(1.0)
                            device.start()
                        consecutive_failures = 0
                        continue
                    except Exception as rexc:  # noqa: BLE001
                        logger.error(
                            "Kinect %d device restart failed: %s",
                            self.device_id,
                            rexc,
                        )
                _time.sleep(0.05)
                continue
            # Non-blocking drain: skip ahead to the newest buffered frame
            # if the consumer stalled this loop. Intermediate capture
            # handles are discarded; only the latest survives to decode.
            drained = 0
            while True:
                try:
                    newer = device.get_capture(timeout=0)
                except K4ATimeoutException:
                    break
                except Exception:
                    break
                if newer is None:
                    break
                capture = newer
                drained += 1
                if drained >= 30:  # cap so a runaway stall doesn't spin forever
                    break
            if drained >= 3:
                logger.debug(
                    "Kinect %d drained %d backlogged frames (consumer " "fell behind)",
                    self.device_id,
                    drained,
                )
            try:
                color = capture.color
                if color is None:
                    continue
                rgb = color[:, :, :3][:, :, ::-1].copy()
                # transformed_depth runs the k4a transformation engine (a GPU
                # dispatch) and allocates a colour-sized uint16 buffer per
                # frame. Gated so a camera whose depth nobody consumes stops
                # paying for it; default stays on because sideview depth is
                # what grasp height comes from.
                depth_m = None
                if self._want_depth:
                    tdepth = capture.transformed_depth
                    if tdepth is not None:
                        depth_m = tdepth.astype(np.float32) / 1000.0
                # Rectification is deferred to read(): remapping every captured
                # frame here starves the velocity-servo loop of CPU, and most
                # frames are never consumed.
                #
                # IR is OFF by default; it is consumed only in the ir=True
                # preview/detect mode. Consumers call set_ir_enabled(True) and
                # the next frames carry it.
                ir_rgb = None
                if self._want_ir:
                    ir_raw = getattr(capture, "ir", None)
                    if ir_raw is not None and ir_raw.size > 0:
                        ir_f = ir_raw.astype(np.float32)
                        hi = max(float(np.percentile(ir_f, 99.5)), 200.0)
                        ir_stretched = np.clip(ir_f / hi, 0.0, 1.0) ** 0.6
                        ir_norm = (ir_stretched * 255.0).astype(np.uint8)
                        ir_rgb = np.stack([ir_norm, ir_norm, ir_norm], axis=-1)
            except Exception as exc:
                logger.warning("Kinect %d decode failed: %s", self.device_id, exc)
                continue
            finally:
                # Drop the handle NOW, not at the next rebind: pyk4a hands
                # back a PyCapsule with a k4a_capture_release destructor, and
                # holding it across the next blocking 250 ms get_capture keeps
                # a buffer out of libk4a's bounded internal pool.
                capture = None
            with self._lock:
                self._latest_rgb = rgb
                self._latest_depth = depth_m
                self._latest_ir = ir_rgb
                self._latest_rectified = False
                self._latest_frame_ts = _time.time()


    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()
