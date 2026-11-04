import json
import logging
import time

import numpy as np

from spark_real.config import calibrations_dir
from spark_real.perception.kinect import AzureKinectCamera
from spark_real.perception.camera_registry import assign_kinect_roles, registry_from_open_devices
from spark_real.perception.camera_registry import (
    PIPELINE_OPENED,
    build_camera_registry,
    camera_type,
    spec as camera_spec,
)
from spark_real.perception.camera_registry import CameraEntry
from spark_real.robots.factory import make_robot_driver
from spark_real.control.score_executor import ScoreExecutor
from spark_real.control.safe_robot import SafeRobot, SafetyConfig
from spark_real.calibration import CameraCalibration

logger = logging.getLogger(__name__)

try:
    import pyk4a

    HAS_PYK4A = True
except ImportError:
    HAS_PYK4A = False


# Camera, calibration, and robot bring-up for SPARKRealPipeline.
# Mixed into the pipeline class; uses attributes set in __init__.
class InitMixin:

    def _resolve_wrist_tool_offset(self) -> np.ndarray:
        """
        Build the 4x4 TCP-to-camera transform.

        Priority order:
          1. Explicit ``wrist_tool_offset_xyz`` in PipelineConfig
             (translation-only override).
          2. Measured hand-eye calibration at
             ``<calibrations_dir>/handeye_wrist.json``: the FULL 4x4.
          3. A placeholder translation (identity rotation).
        """
        offset_xyz = self.config.wrist_tool_offset_xyz
        if offset_xyz is not None:
            T = np.eye(4)
            T[0, 3] = float(offset_xyz[0])
            T[1, 3] = float(offset_xyz[1])
            T[2, 3] = float(offset_xyz[2])
            return T

        handeye_path = calibrations_dir() / "handeye_wrist.json"
        if handeye_path.exists():
            try:
                data = json.loads(handeye_path.read_text())
                # Accept both old AX=XB ("transform_4x4") and new RGB-D
                # Procrustes ("T_cam_to_base_4x4") output formats.
                T_key = (
                    "T_cam_to_base_4x4"
                    if "T_cam_to_base_4x4" in data
                    else "transform_4x4"
                )
                T = np.array(data[T_key], dtype=np.float64)
                if T.shape != (4, 4):
                    raise ValueError(f"{T_key} must be 4x4, got {T.shape}")
                # Stash the per-camera depth correction so it can be applied
                # in capture() once _realsense_cal is created. We can't set
                # _realsense_cal.depth_scale/offset here because that cal
                # object doesn't exist yet at pipeline init time.
                self._wrist_depth_scale = float(data.get("depth_scale_correction", 1.0))
                self._wrist_depth_offset = float(
                    data.get("depth_offset_correction", 0.0)
                )
                logger.info(
                    "wrist tool_offset: loaded from handeye_wrist.json "
                    "(rmse=%.1fmm, depth_scale=%.4f, depth_offset=%.1fmm, "
                    "pairs/poses=%d)",
                    float(data.get("rmse_mm", data.get("residual_mm", 0.0))),
                    self._wrist_depth_scale,
                    self._wrist_depth_offset * 1000.0,
                    int(data.get("num_pairs", data.get("num_poses", 0))),
                )
                return T
            except Exception as exc:
                logger.warning(
                    "wrist tool_offset: failed to load handeye_wrist.json "
                    "(%s); using default tool offset.",
                    exc,
                )

        offset_xyz = [0.05, 0.00, -0.10]
        T = np.eye(4)
        T[0, 3] = float(offset_xyz[0])
        T[1, 3] = float(offset_xyz[1])
        T[2, 3] = float(offset_xyz[2])
        return T

    def _init_kinects(self):
        """
        Initialize Azure Kinect cameras.

        When ``config.kinect_use_hw_sync`` is True (3.5mm sync cable
        connected between two Kinects), the **subordinate** is opened
        FIRST in SUBORDINATE mode (it then waits silently for the master
        trigger), THEN the master is opened in MASTER mode and starts
        firing pulses. This is mandatory per the K4A SDK and also keeps
        USB bandwidth manageable when both Kinects share one host
        controller: the subordinate doesn't free-run.

        Without the sync cable (standalone), both Kinects run free and
        bandwidth must come from separate USB host controllers.
        """
        if not HAS_PYK4A:
            logger.warning("pyk4a not available, skipping Kinect init")
            return

        num_kinects = pyk4a.connected_device_count()
        logger.info("Found %d Azure Kinect device(s)", num_kinects)

        master_serial = self.config.kinect_master_serial
        use_hw_sync = bool(getattr(self.config, "kinect_use_hw_sync", False))
        serial_map = {}  # serial -> device_id
        for dev_id in range(num_kinects):
            try:
                dev = pyk4a.PyK4A(device_id=dev_id)
                dev.open()
                serial_map[dev.serial] = dev_id
                dev.close()
            except Exception:
                pass
        logger.info("Camera serials: %s", serial_map)

        roster = self.profile.cameras() if self.profile is not None else []
        serials = {d: None for d in range(num_kinects)}
        serials.update({d: sn for sn, d in serial_map.items()})
        plan = assign_kinect_roles(serials, roster, master_serial, use_hw_sync)
        logger.info("Kinect open plan (device, role, sync): %s", plan)

        for dev_id, name, sync_mode in plan:
            serial = serials.get(dev_id)
            logger.info(
                "Opening Kinect %d as %s (SN=%s, sync=%s)...",
                dev_id,
                name,
                serial or "?",
                sync_mode,
            )
            try:
                # Subordinate delay depends on the master's depth laser
                # pulse width, which differs by depth mode. NFOV pulses
                # are ~125us so 160us margin is fine; WFOV pulses are
                # nearly an order of magnitude longer, so use 1600us to
                # avoid cross-illumination causing dropped frames and
                # blocking open() forever in warmup.
                sub_delay = 0
                if sync_mode == "SUBORDINATE":
                    if self.config.kinect_depth_mode.startswith("WFOV"):
                        sub_delay = 1600
                    else:
                        sub_delay = 160
                cam = AzureKinectCamera(
                    device_id=dev_id,
                    color_resolution=self.config.kinect_resolution,
                    depth_mode=self.config.kinect_depth_mode,
                    sync_mode=sync_mode,
                    subordinate_delay_usec=sub_delay,
                    camera_fps=self.config.kinect_fps,
                )
                for attempt in range(2):
                    try:
                        cam.open()
                        break
                    except Exception:
                        if attempt == 0:
                            logger.warning(
                                "Kinect %d failed, retrying in 2s...", dev_id
                            )
                            time.sleep(2)
                        else:
                            raise
                # From here on `cam` is open. Any exception before the
                # outer except catches it would leak the handle, so
                # close it explicitly on failure.
                try:
                    cal = CameraCalibration(
                        name=name,
                        width=cam.config.width,
                        height=cam.config.height,
                        fx=cam.config.fx,
                        fy=cam.config.fy,
                        cx=cam.config.cx,
                        cy=cam.config.cy,
                        # Carry any config-level extrinsic across instead of
                        # silently dropping it: the fallback for a camera whose
                        # config carries an extrinsic but has no hand-eye file
                        # (_load_handeye_calibrations() overwrites cal.extrinsic
                        # when one exists).
                        extrinsic=getattr(cam.config, "extrinsic", None),
                    )
                    if name == "sideview":
                        self._kinect = cam
                        self._kinect_cal = cal
                    else:
                        self._kinect2 = cam
                        self._kinect2_cal = cal
                except Exception:
                    try:
                        cam.close()
                    except Exception:
                        pass
                    raise
                logger.info(
                    "Kinect %d (%s) ready (%dx%d, fx=%.1f)",
                    dev_id,
                    name,
                    cal.width,
                    cal.height,
                    cal.fx,
                )
                # Stagger camera starts to avoid USB contention
                time.sleep(3)
            except Exception as e:
                logger.warning("Kinect %d failed: %s", dev_id, e)

    def _load_handeye_calibrations(self):
        """
        Apply saved hand-eye calibrations to the live Kinect cals.

        spark_real/calibration/handeye.py writes the cam-to-base 4x4 transforms
        to ``<calibrations_dir>/handeye_{birdview,sideview,wrist}.json``.
        Without this loader the Kinect cals stay at identity and detect()
        backprojects every pixel ~1 m below the table (the OpenGL fallback
        path triggers because ``intrinsic_matrix`` is gated on
        ``not np.allclose(extrinsic, np.eye(4))``).

        Wrist is intentionally skipped here: its extrinsic is recomputed
        every frame from the live TCP pose in ``capture()`` so a fixed
        cam-to-base load would be wrong the instant the arm moves.
        """
        cal_dir = calibrations_dir()
        targets = [
            ("birdview", "handeye_birdview.json", self._kinect2_cal),
            ("sideview", "handeye_sideview.json", self._kinect_cal),
        ]
        for name, fname, cal in targets:
            if cal is None:
                continue
            path = cal_dir / fname
            if not path.exists():
                logger.info("hand-eye: no %s file at %s", name, path)
                continue
            try:
                data = json.loads(path.read_text())
                # Accept either format:
                #   {transform_4x4: [[...]], ...}              (legacy
                #     AX=XB hand-eye output)
                #   {T_cam_to_base_4x4: [[...]],
                #    depth_scale_correction: 0.85, ...}        (new
                #     RGB-D Procrustes output from spark_calibrate.py)
                T_key = (
                    "T_cam_to_base_4x4"
                    if "T_cam_to_base_4x4" in data
                    else "transform_4x4"
                )
                T = np.array(data[T_key], dtype=np.float64)
                if T.shape != (4, 4):
                    raise ValueError(f"{T_key} must be 4x4, got {T.shape}")
                cal.extrinsic = T
                depth_scale = float(data.get("depth_scale_correction", 1.0))
                depth_offset = float(data.get("depth_offset_correction", 0.0))
                cal.depth_scale = depth_scale
                cal.depth_offset = depth_offset
                logger.info(
                    "hand-eye: applied %s extrinsic from %s "
                    "(residual=%.1fmm, depth_scale=%.4f, depth_offset=%.1fmm, "
                    "poses/corners=%d)",
                    name,
                    fname,
                    float(data.get("rmse_mm", data.get("residual_mm", 0.0))),
                    depth_scale,
                    depth_offset * 1000.0,
                    int(data.get("num_pairs", data.get("num_poses", 0))),
                )
            except Exception as exc:
                logger.warning(
                    "hand-eye: failed to load %s from %s: %s", name, fname, exc
                )

    def _build_unified_camera_registry(self):
        # Wrap the already-open single-arm cameras in one role-keyed registry.
        # The dual-Kinect open in _init_kinects() carries master/subordinate
        # sync, depth-mode, retry, and stagger semantics that depend on
        # runtime device enumeration, so do NOT re-open through the
        # config-driven builder here: wrap the live devices and their
        # calibrated CameraCalibration objects so the registry entries and the
        # _kinect / _kinect2 / _realsense slots reference the same objects.
        # The bimanual rig opens its whole roster from config instead.
        if (getattr(self.config, "robot_family", "") or "").lower() == "bimanual_franka":
            self._init_bimanual_cameras()
            return

        entries = []
        # sideview = master Kinect (_kinect); birdview = subordinate (_kinect2);
        # wrist = RealSense (_realsense). This is the slot<->role mapping from
        # _init_kinects (camera_registry.assign_kinect_roles).
        slot_roles = [
            ("sideview", self._kinect, self._kinect_cal, "kinect"),
            ("birdview", self._kinect2, self._kinect2_cal, "kinect"),
            ("wrist", self._realsense, self._realsense_cal, "realsense"),
        ]
        for role, device, cal, kind in slot_roles:
            if device is None or cal is None:
                continue
            entries.append(
                CameraEntry(
                    role=role,
                    kind=kind,
                    device=device,
                    calibration=cal,
                    serial=getattr(device, "serial", "") or "",
                )
            )

        # `sideview` (master Kinect) is the workspace anchor for single-arm
        # rigs, matching camera_registry._resolve_primary_role.
        primary = (
            "sideview"
            if any(e.role == "sideview" for e in entries)
            else (entries[0].role if entries else "")
        )
        self._camera_registry = registry_from_open_devices(
            entries, primary_role=primary
        )

        # Re-point the legacy slots FROM the registry (identity reassignments;
        # the entries wrap the same device + cal objects).
        sv = self._camera_registry.get("sideview")
        bv = self._camera_registry.get("birdview")
        wr = self._camera_registry.get("wrist")
        if sv is not None:
            self._kinect = sv.device
            self._kinect_cal = sv.calibration
        if bv is not None:
            self._kinect2 = bv.device
            self._kinect2_cal = bv.calibration
        if wr is not None:
            self._realsense = wr.device
            self._realsense_cal = wr.calibration
        # Every other roster entry (e.g. a ZED or a webcam on this rig) opens
        # through the camera driver registry and joins the same registry.
        if self.profile is not None:
            extra = [
                dict(e, type=camera_type(e)) for e in camera_spec(self.profile).values()
                if (e["role"], camera_type(e)) not in PIPELINE_OPENED and e.get("enabled", True) is not False
            ]
            build_camera_registry(extra, registry=self._camera_registry)
        logger.info(
            "Unified camera registry built: roles=%s primary=%s",
            self._camera_registry.roles,
            self._camera_registry.primary_role,
        )

    def _init_robot(self):
        """
        Connect the robot through ``make_robot_driver`` with the family
        config's gripper/robot/collision blocks, then build the executor.

        ur10e: SafeRobot (CBF filter on, force off) around the driver.
        franka: the bare driver; the CBF clamps direct joint moves
        unpredictably, so the libfranka collision thresholds and the
        executor's workspace box protect the arm. bimanual_franka:
        pipeline_init_bimanual. A connection failure leaves the pipeline
        perception-only; a config error (unknown family, gripper type or
        backend) raises.
        """
        # Tear down a previous driver before reassigning self._robot so a
        # reconnect never leaves two control sockets open.
        if getattr(self, "_robot", None) is not None:
            try:
                self._robot.disconnect()
            except Exception as exc:
                logger.warning(
                    "previous robot driver disconnect raised: %s "
                    "(continuing with reconnect)",
                    exc,
                )
            self._robot = None

        family = (getattr(self.config, "robot_family", "ur10e") or "ur10e").lower()
        try:
            if family == "bimanual_franka":
                self._init_bimanual_franka()
                return
            blocks = self.profile.driver_blocks() if self.profile is not None else {}
            self._robot = make_robot_driver(family, self.config.robot_ip, frequency=None, **blocks)
            self._robot.connect()
            if family == "ur10e":
                # The SafeRobot box comes from the same control: block the
                # ScoreExecutor reads, so the two boxes stay one source.
                ctrl = (self.profile.control() if self.profile is not None else {}) or {}
                ws_min = ctrl.get("workspace_min", [-1.1, -0.5, -0.27])
                ws_max = ctrl.get("workspace_max", [-0.5, 0.7, 0.50])
                safety_cfg = SafetyConfig(
                    vel_lin_max=self.config.velocity + 0.05,
                    ws_min=np.array([float(v) for v in ws_min]),
                    ws_max=np.array([float(v) for v in ws_max]),
                )
                self._robot = SafeRobot(
                    self._robot,
                    enable_obstacles=False,  # obstacles set from detection map
                    enable_force=False,  # noisy F/T causes jiggling
                    config=safety_cfg,
                )
            self._executor = ScoreExecutor(
                robot=self._robot,
                velocity=self.config.velocity,
                safe_height=self.config.safe_height,
                pipeline=self,
                strict_placement_verify=self.config.strict_placement_verify,
                grasp_calibration=self._grasp_calibration,
            )
            self._executor._profile = self.profile
            self._executor._apply_family_workspace_bounds()
            logger.info(
                "%s connected at %s (gripper %s, SafeRobot %s)",
                family,
                self.config.robot_ip,
                self._robot.GRIPPER_TYPE,
                "on" if family == "ur10e" else "off",
            )
        except ValueError:
            self._robot = None
            self._executor = None
            raise
        except Exception as e:
            logger.warning(
                "Robot connection failed (%s): %s (running perception-only)",
                family,
                e,
            )
            self._robot = None
            self._executor = None
