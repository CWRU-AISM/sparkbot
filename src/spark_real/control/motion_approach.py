"""Approach to a grasp target and the Cartesian servo move."""

import logging

import numpy as np
from scipy.spatial.transform import Rotation

from spark_real.control.grasp_strategy import (
    resolve_grasp_orientation,
)

logger = logging.getLogger(__name__)


class ApproachMixin:
    """Approach to a grasp target and the Cartesian servo move."""

    def _approach_target(self, target: np.ndarray, detection: dict = None, params: dict = None):
        """
        Approach target: lift, position above, wrist refine, servo descend.

        `params` are the plan node's params; only grasp_strategy /
        grasp_yaw_deg are read. Omitting them is the auto route.
        """
        self._check_abort()
        if not self._holding:
            self._ensure_jaws_open("pre-approach")
        # Orientation source is the planner's `grasp_strategy` when it named
        # one, otherwise the auto route (AR gate + input-validity floors).
        orient, strategy = resolve_grasp_orientation(
            params or {}, detection, self, context="approach"
        )
        self._active_grasp_strategy = strategy
        self._active_grasp_label = self._last_keypoint_label or ""
        adjusted = target.copy()
        _gc_label = self._last_keypoint_label or ""
        _gc_learned = (
            self.grasp_calibration.learned_offset(_gc_label)
            if self.grasp_calibration is not None
            else None
        )
        _gc_initial = _gc_learned if _gc_learned is not None else self.GRIPPER_OPEN_Z_OFFSET
        adjusted[2] += _gc_initial
        if _gc_learned is not None:
            logger.info(
                "Grasp learned-offset for %r: %.3f m (%d samples)",
                _gc_label,
                _gc_learned,
                self.grasp_calibration._stats.get(
                    _gc_label.lower().rstrip("0123456789 "), type("_", (), {"n": 0})()
                ).n,
            )
        self._grasp_perception_target_z = float(target[2])
        # Per-pick reset: a stale close-height from the PREVIOUS pick must
        # never feed this pick's release give-back.
        self._actual_grasp_tcp_z = None

        current = self._get_current_position()
        safe_z = self._safe_clearance_z(current, adjusted)
        transit = []
        if np.linalg.norm(adjusted - current) > 0.5:
            lift = current.copy()
            lift[2] = safe_z
            transit.append((lift, list(self.GRASP_ORIENTATION), None, "lift"))
            above = adjusted.copy()
            above[2] = safe_z
            transit.append((above, list(self.GRASP_ORIENTATION), None, "above"))
        else:
            above = adjusted.copy()
            above[2] = max(adjusted[2] + 0.10, safe_z)
            transit.append((above, list(self.GRASP_ORIENTATION), None, "above"))

        label = self._last_keypoint_label or ""
        if self._blend_enabled():
            # The descent target only exists AFTER a wrist capture, so a phrase
            # can only absorb it when refinement is off (the default here).
            refine_pending = bool(
                self._wrist_refine_enabled and label and self._wrist_refine_mode() != "off"
            )
            phrase = list(transit)
            with_descent = not refine_pending
            if with_descent:
                phrase.append(
                    (
                        adjusted.copy(),
                        list(orient),
                        self.velocity * 0.9,
                        "grasp_descent",
                    )
                )
            if self._blend_transit(phrase, phrase="approach"):
                if with_descent:
                    # Terminal row was r=0: the arm IS stopped at the grasp pose.
                    return
                transit = []
        for pos_i, ori_i, _v, _lbl in transit:
            self._move_to(pos_i, ori_i)

        # Wrist-camera refinement, gated by control.wrist_refine (off by
        # default); when off the servo goes straight to the birdview/sideview
        # target.
        if label:
            refined = self._refine_with_wrist(
                label, adjusted, z_plane=self._grasp_perception_target_z
            )
            if refined is not None:
                new_pos = np.array(refined["position_3d"])
                xy_diff = np.linalg.norm(new_pos[:2] - adjusted[:2])
                if xy_diff < 0.05:
                    adjusted[:2] = new_pos[:2]
                    # ray_plane's Z IS the prior it was handed; re-applying it
                    # would add the gripper offset twice. Only the depth path
                    # can correct Z.
                    if refined.get("source") != "ray_plane" and (
                        abs(new_pos[2] - adjusted[2]) < 0.03
                    ):
                        adjusted[2] = new_pos[2] + self.GRIPPER_OPEN_Z_OFFSET
                    logger.info("Wrist XY refine: delta=%.3fm", xy_diff)
                else:
                    logger.info("Wrist XY too far (%.3fm), keeping birdview", xy_diff)

                # Re-resolve with the wrist's own AR/angle, keeping the source
                # detection's confidence: same strategy rules, fresher geometry.
                img_deg = np.rad2deg(refined["orientation_angle"])
                yaw_deg = (img_deg % 180) - self.WRIST_CAM_OFFSET_DEG
                if yaw_deg > 90:
                    yaw_deg -= 180
                if yaw_deg < -90:
                    yaw_deg += 180
                wrist_det = dict(detection or {})
                wrist_det["aspect_ratio"] = refined["aspect_ratio"]
                wrist_det["orientation_angle"] = refined["orientation_angle"]
                wrist_orient, wrist_strategy = resolve_grasp_orientation(
                    params or {},
                    wrist_det,
                    self,
                    yaw_override_rad=np.deg2rad(yaw_deg),
                    context="wrist-refine",
                )
                if wrist_strategy == "obb":
                    orient = wrist_orient
                    self._active_grasp_strategy = wrist_strategy
                    logger.info(
                        "Wrist grasp: img=%.1f deg -> yaw=%.1f deg (ar=%.1f)",
                        img_deg,
                        yaw_deg,
                        refined["aspect_ratio"],
                    )

        # Persist the (possibly OBB-yawed) orientation so the grasp primitive's
        # descent/clamp/lift reuse it instead of reverting to GRASP_ORIENTATION
        # right before the jaws close. See _grasp_v2.
        self._active_grasp_orient = list(orient)
        # Remember which WAY the object points, so the place can put its heavy
        # end where the receptacle's heavy end is (a PCA axis is only a line).
        self._held_axis_sign = None
        self._held_axis_img = None
        # Clear the mask WITH the sign, or an approach without a detection
        # (recovery's retract_retry) registers the PREVIOUS pick's silhouette.
        self._held_mask = None
        try:
            _m = detection.get("_mask") if isinstance(detection, dict) else None
            if _m is not None:
                from spark_core.mask_obb import _pca_obb
                from spark_real.perception.mask_features import heavy_end_sign

                _a, _ar, _, _ = _pca_obb(np.asarray(_m))
                _sgn = heavy_end_sign(np.asarray(_m), _a)
                self._held_axis_img = float(_a)
                self._held_axis_sign = float(_sgn)
                self._held_mask = np.asarray(_m)
                logger.info(
                    "[held-axis] image axis %.1f deg, heavy end %s (ar %.2f)",
                    np.rad2deg(_a),
                    {1.0: "+", -1.0: "-", 0.0: "UNKNOWN"}.get(_sgn, "?"),
                    _ar,
                )
        except Exception as exc:  # noqa: BLE001 - never let this stop a grasp
            logger.warning("[held-axis] could not measure: %s", exc)
        # Final oriented descent as a JOINT move (pyroki IK -> movej): smooth,
        # fast, hits the yaw exactly. The Cartesian PD servo is jittery near
        # convergence and slow; fall back to it only if the joint-IK path is
        # unavailable or fails.

        if self._robot_family() == "ur10e":
            try:
                self._movej_via_ik(adjusted, orient, self.velocity * 0.9)
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[approach] joint-IK descend failed (%s); Cartesian-servo fallback",
                    exc,
                )
        self._servo_to(adjusted, orient, velocity=self.velocity * 0.9)

    def _servo_to(self, position: np.ndarray, orientation: list, velocity: float = 0.15) -> bool:
        """
        Servo to a target pose with precise orientation control.

        Returns False when the servo STALLED (gave up against an unreachable
        target, no movel retry issued, TCP parked where it got stuck). True
        otherwise (converged, or a timeout the movel fallback finished). The
        place descent must not treat a stall as an arrival.
        """
        self._check_abort()
        target_pose = list(position) + list(orientation)
        self._servo.max_vel_linear = min(velocity, 0.25)
        converged = self._servo.move_to_pose(target_pose, velocity=velocity)
        self._check_abort()
        stalled = False
        if not converged:
            # A STALL means the servo held a command into an unreachable
            # target. A movel to that same pose would drive into the identical
            # obstruction, risking a protective stop. Accept the current pose
            # and let the caller's verification decide. A TIMEOUT is different:
            # the servo was still closing, so movel can finish the job.
            if getattr(self._servo, "last_exit", "") == "stalled":
                stalled = True
                logger.warning(
                    "Servo stalled at pos_err=%.4f ori_err=%.4f; NOT retrying "
                    "with movel (same unreachable target)",
                    self._servo.last_pos_err,
                    self._servo.last_ori_err,
                )
            else:
                logger.warning("Servo did not converge, falling back to movel")
                self._move_to_linear(position, orientation, velocity=velocity)
        # Hand the controller back before anything else uploads a program.
        # The servo streams servoL commands and OWNS the controller until it is
        # stopped; a gripper command is a URScript upload, and uploading one
        # while the servo still holds the channel silently loses it (jaw
        # register 175.95 -> 175.95, zero travel).
        try:
            if hasattr(self.robot, "servo_stop"):
                self.robot.servo_stop()
        except Exception as exc:  # noqa: BLE001 - never let cleanup stop motion
            logger.warning("servo_stop after servo failed: %s", exc)

        actual = self._servo._get_tcp_pose()
        actual_rotvec = actual[3:6]
        R_actual = Rotation.from_rotvec(actual_rotvec).as_matrix()
        tool_x_world = R_actual @ np.array([1, 0, 0])
        tool_y_world = R_actual @ np.array([0, 1, 0])
        logger.info(
            "Servo result: tool_X=%.1f deg, tool_Y=%.1f deg (closing axis)",
            np.rad2deg(np.arctan2(tool_x_world[1], tool_x_world[0])),
            np.rad2deg(np.arctan2(tool_y_world[1], tool_y_world[0])),
        )
        return not stalled
