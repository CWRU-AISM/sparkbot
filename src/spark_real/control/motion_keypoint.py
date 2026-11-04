"""move_to_keypoint and the orientation helpers it uses."""

import logging
import time

import numpy as np
from scipy.spatial.transform import Rotation

from spark_real.config import family_block
from spark_real.control.execution_recovery import grasp_target_in_container
from spark_real.control.executor_types import (
    ExecutionResult,
)

logger = logging.getLogger(__name__)


class KeypointMixin:
    """move_to_keypoint and the orientation helpers it uses."""

    def current_orientation(self, fallback=None):
        """The orientation the arm is actually holding, or `fallback`.

        Translations must not re-pick a wrist angle: passing the fixed
        GRASP_ORIENTATION into a pure move UNWINDS whatever yaw the task had
        established.

        `obs.get("tcp_pose") or obs.get(...)` raises on a numpy array, so the
        None comparisons are explicit.
        """
        try:
            obs = self.robot.get_observation()
            tcp = obs.get("tcp_pose")
            if tcp is None:
                tcp = obs.get("tcp_pos")
            if tcp is not None and len(tcp) >= 6:
                return list(np.asarray(tcp[3:6], dtype=float))
        except Exception as exc:  # noqa: BLE001
            logger.warning("current_orientation: read failed (%s)", exc)
        return fallback if fallback is not None else list(self.GRASP_ORIENTATION)

    def _nearest_symmetric_yaw(self, yaw: float) -> float:
        """
        Pick yaw or yaw+-pi, whichever needs the least wrist travel.

        A two-jaw gripper grasps identically at yaw and yaw+pi, so <=90 deg
        of travel always suffices. Candidates stay inside MAX_YAW_OFFSET; on
        any read failure the canonical yaw passes through unchanged.
        """
        try:
            tcp = np.asarray(self.robot.get_tcp_pose(), dtype=float)
            R_cur = Rotation.from_rotvec(tcp[3:6])
            R_base = Rotation.from_rotvec(np.asarray(self.GRASP_ORIENTATION, dtype=float))
            rel = (R_cur * R_base.inv()).as_matrix()
            cur_yaw = float(np.arctan2(rel[1, 0], rel[0, 0]))
        except Exception:
            return yaw
        best = yaw
        for cand in (yaw - np.pi, yaw + np.pi):
            if abs(cand) <= self.MAX_YAW_OFFSET and abs(cand - cur_yaw) < abs(best - cur_yaw):
                best = cand
        if best != yaw:
            logger.info(
                "[yaw-symmetry] %.1f deg -> %.1f deg (current %.1f, travel "
                "%.1f instead of %.1f deg)",
                np.rad2deg(yaw),
                np.rad2deg(best),
                np.rad2deg(cur_yaw),
                np.rad2deg(abs(best - cur_yaw)),
                np.rad2deg(abs(yaw - cur_yaw)),
            )
        return best

    def _move_to_keypoint(self, params: dict, t0: float) -> ExecutionResult:
        self._check_abort()
        label = params.get("keypoint_label", "")
        self._last_keypoint_label = label
        # offset_z default 0.0: move_to_keypoint takes the gripper TO the
        # keypoint so the following grasp closes on it (the grasp descent-retry
        # only spans ~7.5cm). The planner emits explicit offsets (0 picks,
        # 0.04+ places); this only governs plans/BTs that omit the field.
        offset = np.array(
            [
                params.get("offset_x", 0.0),
                params.get("offset_y", 0.0),
                params.get("offset_z", 0.0),
            ]
        )

        det = self.detection_map.get(label)
        if det is None:
            return ExecutionResult(
                action_type="move_to_keypoint",
                success=False,
                message=f"Object '{label}' not found in detections",
                duration=time.time() - t0,
            )

        # Hard guard for pick approaches: if this object already sits inside
        # the tray / container, skip the pick entirely (it is done). Only
        # applies when not holding so placement moves into the container are
        # never blocked.
        if not self._holding:
            try:
                _in = grasp_target_in_container(self, label)
            except Exception:
                _in = None
            # Only skip the pick if this run placed the object into the
            # container (a re-pick guard for stacking recovery). An object that
            # STARTED inside a container is not in _placed_labels and must be
            # picked normally.
            if _in is not None and label in self._placed_labels:
                self._placed_labels.add(label)
                logger.info(
                    "move_to_keypoint '%s': already in container " "'%s', skipping pick approach",
                    label,
                    _in,
                )
                return ExecutionResult(
                    action_type="move_to_keypoint",
                    success=True,
                    message=f"'{label}' already in container '{_in}', " f"skipping",
                    duration=time.time() - t0,
                )

        pos = np.array(det["position_3d"])
        if pos[2] < self.TABLE_Z_FLOOR:
            logger.warning("Detection '%s' Z=%.3f below floor, clamping", label, pos[2])
            pos[2] = self.TABLE_Z_FLOOR
        target = pos + offset

        if self._holding:
            # PLACE. Recompute the Z from the container's MEASURED rim and
            # interior floor plus the held object's overhang; the plan's
            # offset_z survives as an upper bound, and an unmeasurable
            # container leaves it untouched (see control/release_height.py).
            # XY is never touched here.
            target[2] = self._place_release_z(float(target[2]), det, params)
            delivered = self._transport_to(
                target, target_detection=det, target_label=label, params=params
            )
            if not self._holding:
                return ExecutionResult(
                    action_type="move_to_keypoint",
                    success=False,
                    message=f"Object lost during transport to '{label}'",
                    duration=time.time() - t0,
                )
            if not delivered:
                # Still holding, but not over the target. A failed action:
                # executor_core runs recovery, and the reset path does NOT open
                # the gripper while it is still holding.
                record = getattr(self, "_place_arrival_record", None) or {}
                return ExecutionResult(
                    action_type="move_to_keypoint",
                    success=False,
                    message=(
                        f"Did not arrive over '{label}': "
                        f"{record.get('detail', 'transport stalled short')}"
                    ),
                    duration=time.time() - t0,
                )
        else:
            self._approach_target(target, detection=det, params=params)

        return ExecutionResult(
            action_type="move_to_keypoint",
            success=True,
            message=f"Moved to '{label}' at {target.tolist()}",
            duration=time.time() - t0,
        )

    def _oriented_grasp(self, yaw_offset: float = 0.0) -> list:
        """Post-rotate GRASP_ORIENTATION by Rz(yaw) about world Z.

        The closing direction ends up perpendicular to the OBB major axis.
        """
        clamped = np.clip(yaw_offset, -self.MAX_YAW_OFFSET, self.MAX_YAW_OFFSET)
        if abs(clamped - yaw_offset) > 0.01:
            logger.warning(
                "_oriented_grasp: yaw clamped from %.1f to %.1f deg",
                np.rad2deg(yaw_offset),
                np.rad2deg(clamped),
            )
        logger.info("_oriented_grasp: yaw_offset=%.1f deg", np.rad2deg(clamped))
        base = Rotation.from_rotvec(self.GRASP_ORIENTATION)
        yaw = Rotation.from_euler("z", clamped)
        return (yaw * base).as_rotvec().tolist()

    def _safe_clearance_z(self, start: np.ndarray, end: np.ndarray) -> float:
        """
        Safety lift 15 cm above the object/end Z.
        """
        base_z = end[2] + 0.15
        if not self.detection_map:
            return base_z
        max_obstacle_z = base_z
        for label, det in self.detection_map.items():
            pos = det.get("position_3d")
            if pos is None:
                continue
            obs = np.array(pos)
            path_dir = end[:2] - start[:2]
            path_len = np.linalg.norm(path_dir)
            if path_len < 0.01:
                continue
            path_unit = path_dir / path_len
            to_obs = obs[:2] - start[:2]
            proj = np.clip(np.dot(to_obs, path_unit), 0, path_len)
            closest = start[:2] + proj * path_unit
            dist_xy = np.linalg.norm(obs[:2] - closest)
            if dist_xy < self.OBSTACLE_XY_RADIUS:
                obstacle_top = obs[2] + self.CLEARANCE_MARGIN
                if obstacle_top > max_obstacle_z:
                    max_obstacle_z = obstacle_top
                    logger.info(
                        "Clearance: '%s' at z=%.3f, clearing at z=%.3f",
                        label,
                        obs[2],
                        obstacle_top,
                    )
        return max(max_obstacle_z, base_z)

    def _control_config(self) -> dict:
        """The resolved `control:` block (profile.raw preferred, else family YAML).

        Returns {} on any missing pipeline/config/file.
        """
        cfg = getattr(self._pipeline, "config", None) if self._pipeline else None
        if cfg is None:
            return {}
        family = (getattr(cfg, "robot_family", "ur10e") or "ur10e").lower()
        return family_block(getattr(self._pipeline, "profile", None), family, "control")
