"""
Bimanual Franka bring-up for SPARKRealPipeline (mixed in next to InitMixin).

Cameras: every entry of the config `cameras:` roster opens by its `type`
into a role-keyed registry (pipeline.camera_registry). The bimanual
calibration (<calibrations_dir>/calibration_bimanual.json, written by
spark_real.calibration.bimanual_anchor_tap) sets the external camera's
extrinsic to the right arm's base frame and supplies T_right_to_left.

Robot: make_robot_driver builds the BimanualFrankaDriver from the
`robot:` and `grippers:` blocks; BimanualSafeRobot adds the per-arm boxes
(`workspace_left` / `workspace_right`) and the inter-arm distance gate, in
the calibrated frames when the calibration exists; the
BimanualScoreExecutor runs the typed bimanual primitives on the rig motion
path (`bimanual_motion:` block) and other arm-tagged primitives on two
per-arm ScoreExecutors.
"""

import json
import logging

import numpy as np

from spark_real.config import calibrations_dir, family_block
from spark_real.control.bimanual_executor import BimanualScoreExecutor
from spark_real.control.bimanual_safe_robot import BaseFrames, BimanualSafeRobot, InterArmConfig
from spark_real.control.safe_robot import SafetyConfig
from spark_real.perception.camera_registry import build_camera_registry
from spark_real.robots.factory import make_robot_driver

logger = logging.getLogger(__name__)

BIMANUAL_CALIBRATION = "calibration_bimanual.json"


def load_bimanual_calibration(path=None):
    """(T_zed_to_right_base, T_right_to_left) from the calibration file, or None."""
    path = path or calibrations_dir() / BIMANUAL_CALIBRATION
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    arms = data["arms"]
    T_r = np.asarray(arms["right"]["T_zed_to_base"], dtype=float).reshape(4, 4)
    T_l = np.asarray(arms["left"]["T_zed_to_base"], dtype=float).reshape(4, 4)
    return T_r, T_l @ np.linalg.inv(T_r)


def _arm_safety(ws: dict, max_linear_velocity: float) -> SafetyConfig:
    return SafetyConfig(
        vel_lin_max=max_linear_velocity + 0.05,
        ws_min=np.array([ws.get("x_min", -0.85), ws.get("y_min", -0.85), ws.get("z_min", 0.0)], dtype=float),
        ws_max=np.array([ws.get("x_max", 0.85), ws.get("y_max", 0.85), ws.get("z_max", 1.20)], dtype=float),
        reach_max=float(ws.get("reach_max", 0.855)),
        eps_singularity=0.0,  # the UR elbow check does not apply to a Franka
        eta_kinematic=float(ws.get("eta_kinematic", 5.0)),
    )


class BimanualInitMixin:

    def _init_bimanual_cameras(self):
        """Open the config roster; map external/wrist roles onto the slots."""
        cfg = family_block(self.profile, "bimanual_franka")
        self.camera_registry = build_camera_registry(cfg.get("cameras") or [], primary_role="external")
        self._T_right_to_left = None
        ext = self.camera_registry.get("external")
        try:
            cal = load_bimanual_calibration()
        except Exception as exc:  # noqa: BLE001 - a bad file must not stop bring-up
            logger.error("bimanual calibration unreadable: %s", exc)
            cal = None
        if cal is None:
            logger.warning(
                "no %s in %s: detections stay in the camera frame and the bimanual "
                "executor refuses to run; calibrate with spark_real.calibration.bimanual_anchor_tap",
                BIMANUAL_CALIBRATION, calibrations_dir(),
            )
        else:
            T_zed_to_right, self._T_right_to_left = cal
            if ext is not None:
                ext.calibration.extrinsic = T_zed_to_right
            logger.info("bimanual calibration loaded: external camera -> right arm base")
        # Slots read by the single-arm capture path: the external camera is
        # the scene camera; the right wrist stands in for "wrist".
        wl, wr = self.camera_registry.get("wrist_left"), self.camera_registry.get("wrist_right")
        wrist = wr or wl
        self._kinect = ext.device if ext else None
        self._kinect_cal = ext.calibration if ext else None
        self._kinect2 = None
        self._kinect2_cal = None
        self._realsense = wrist.device if wrist else None
        self._realsense_cal = wrist.calibration if wrist else None
        self._wrist_for_arm = dict(self.camera_registry.wrist_for_arm)

    def _init_bimanual_franka(self):
        """Driver + BimanualSafeRobot + BimanualScoreExecutor from the profile."""
        cfg = family_block(self.profile, "bimanual_franka")
        robot_cfg = cfg.get("robot") or {}
        blocks = self.profile.driver_blocks() if self.profile is not None else {
            "gripper": cfg.get("grippers") or {}, "robot": robot_cfg, "collision": cfg.get("collision_behavior") or {},
        }
        driver = make_robot_driver("bimanual_franka", self.config.robot_ip, **blocks)
        driver.connect()
        self._raw_driver = driver

        vmax = float(robot_cfg.get("max_linear_velocity", 0.15))
        inter = cfg.get("inter_arm") or {}
        T_r2l = getattr(self, "_T_right_to_left", None)
        if T_r2l is not None:
            # Calibrated frames: the right arm's base is the shared frame.
            bases = BaseFrames(T_left_world=np.linalg.inv(T_r2l), T_right_world=np.eye(4))
        else:
            logger.warning("no bimanual calibration: the inter-arm gate uses robot.base_left/right "
                           "from the config, and the executor refuses scores until calibrated")
            try:
                bases = BaseFrames.from_yaml_block(robot_cfg)
            except ValueError as exc:
                logger.warning("bimanual base offsets missing (%s); identity bases", exc)
                bases = None
        self._safe = BimanualSafeRobot(
            driver,
            _arm_safety(cfg.get("workspace_left") or {}, vmax),
            _arm_safety(cfg.get("workspace_right") or {}, vmax),
            InterArmConfig(
                dist_hard_m=float(inter.get("dist_hard_m", 0.12)),
                dist_soft_m=float(inter.get("dist_soft_m", 0.20)),
                eta_kinematic=float(inter.get("eta_kinematic", 8.0)),
            ),
            bases=bases,
        )
        self._robot = self._safe
        self._executor = BimanualScoreExecutor(
            self._safe,
            velocity=self.config.velocity,
            pipeline=self,
            strict_placement_verify=self.config.strict_placement_verify,
            T_right_to_left=T_r2l,
            motion_cfg=cfg.get("bimanual_motion") or {},
        )
        self._executor._profile = self.profile
        # The per-arm executors applied bounds in __init__ before _profile
        # existed; re-apply so each reads its workspace_<arm> box, the table
        # floor and the grasp orientation from the profile.
        for ex in self._executor._arm_executors.values():
            ex._profile = self.profile
            ex._apply_family_workspace_bounds()
        logger.info(
            "Bimanual Franka connected (backend=%s, grippers=%s, left=%s, right=%s)",
            driver.backend, driver.GRIPPER_TYPE, driver.left_ip, driver.right_ip,
        )
