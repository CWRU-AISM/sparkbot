"""The grasp primitive: strategy resolution, close and retries."""

import logging
import os
import time

import numpy as np
from scipy.spatial.transform import Rotation

from spark_core.grasp_outcome import GraspOutcome
from spark_real.control.executor_types import ExecutionResult
from spark_real.control.grasp_strategy import (
    resolve_grasp_orientation,
    resolve_strategy,
)
from spark_real.control.grasp_verdict import GraspVerdict, _parent_object_label

logger = logging.getLogger(__name__)


class GraspExecMixin:
    """The grasp primitive: strategy resolution, close and retries."""

    def _resolve_grasp_node_strategy(self, params: dict) -> str:
        """Settle the strategy for THIS grasp and set _active_grasp_orient.

        `_active_grasp_orient` is cleared first and then set explicitly, so a
        grasp reached without a preceding approach (recovery re-entry,
        grasp_top_down, grasp_se3_flow) cannot inherit the PREVIOUS object's yaw.
        """
        approach_orient = getattr(self, "_active_grasp_orient", None)
        approach_strategy = getattr(self, "_active_grasp_strategy", None)
        approach_label = getattr(self, "_active_grasp_label", None)
        self._active_grasp_orient = None

        label = self._last_keypoint_label or ""
        det = self.detection_map.get(label) if label else None
        # An approach for a DIFFERENT object (or none at all) is stale.
        stale = approach_label != label
        node_asked = bool(params) and (
            "grasp_strategy" in params or "grasp_yaw_deg" in params
        )
        if node_asked or stale or not approach_strategy:
            strategy, reason = resolve_strategy(params or {}, det, self)
        else:
            strategy, reason = approach_strategy, "from approach"

        if strategy == "obb":
            # Reuse the approach's composed yaw when it is fresh (already
            # cable-limit checked there); otherwise re-derive from the detection.
            orient = None if (stale or node_asked) else approach_orient
            if orient is None:
                orient, used = resolve_grasp_orientation(
                    params or {}, det, self, context="grasp"
                )
                strategy = used
                orient = orient if used == "obb" else None
            self._active_grasp_orient = list(orient) if orient is not None else None
        self._active_grasp_strategy = strategy
        self._active_grasp_label = label
        # Record the held object's MASK on the path the grasp actually takes;
        # without it the downstream direction tests fall back to symmetric_yaw's
        # arbitrary mod-180 choice, which seats a tool end-for-end.
        self._held_mask = None
        self._held_axis_img = None
        self._held_axis_sign = None
        try:
            # Direction must be measured on the WHOLE tool, not the grasped
            # sub-part: a handle is symmetric end-for-end, the asymmetry lives
            # in the shaft. Grasp 'knife handle', orient by 'knife'.
            _m = det.get("_mask") if isinstance(det, dict) else None
            _src = label or "?"
            _parent = _parent_object_label(label, getattr(self, "detection_map", None))
            if _parent is not None:
                _pm = (self.detection_map.get(_parent) or {}).get("_mask")
                if _pm is not None:
                    _m, _src = _pm, _parent
            if _m is not None:
                from spark_core.mask_obb import _pca_obb
                from spark_real.perception.mask_features import heavy_end_sign
                _a, _ar, _, _ = _pca_obb(np.asarray(_m))
                self._held_mask = np.asarray(_m)
                self._held_axis_img = float(_a)
                self._held_axis_sign = float(heavy_end_sign(np.asarray(_m), _a))
                logger.info(
                    "[held-axis] '%s' image axis %.1f deg, ar %.2f, heavy end %s",
                    _src, np.rad2deg(_a), _ar,
                    {1.0: "+", -1.0: "-", 0.0: "UNKNOWN"}.get(
                        self._held_axis_sign, "?"),
                )
        except Exception as exc:  # noqa: BLE001 - never let this stop a grasp
            logger.warning("[held-axis] could not record the held mask: %s", exc)
        logger.info("[grasp-strategy] grasp node: %s (%s)", strategy, reason)
        return strategy

    def _grasp(self, params: dict, t0: float) -> ExecutionResult:
        """
        Close gripper and verify an object was grasped.
        """
        self._check_abort()
        # Sets _active_grasp_orient for this grasp.
        self._resolve_grasp_node_strategy(params or {})
        # Grasp v2 (mid-height grasp + force verify) is ROBOTIQ-ONLY: it exists
        # for the UR/Robotiq rig whose gripper position/OBJ register reads are
        # stale. Other grippers keep the legacy path below.
        # SPARK_GRASP_V2=0 also reverts.
        if (
            os.environ.get("SPARK_GRASP_V2", "1") == "1"
            and self._gripper_type() == "robotiq_2f85"
        ):
            return self._grasp_v2(params, t0)
        label = self._last_keypoint_label or ""
        det = self.detection_map.get(label) if label else None

        # Priority chain: BT > mask-derived > legacy default.
        target_width = None
        force = None
        source = None

        bt_tw = params.get("target_width", None)
        if bt_tw is not None:
            target_width = float(bt_tw)
            bt_f = params.get("force", None)
            if bt_f is not None:
                force = float(bt_f)
            source = "bt"

        if target_width is None and det is not None:
            try:
                mask_prior = self._derive_target_width_from_detection(det)
            except Exception as exc:
                logger.warning("Mask-derived prior raised: %s", exc)
                mask_prior = None
            if mask_prior is not None:
                target_width = float(mask_prior[0])
                bt_f = params.get("force", None)
                force = float(bt_f) if bt_f is not None else float(mask_prior[1])
                source = f"mask:{mask_prior[2]}"

        if force is None:
            force = float(params.get("force", 50))
        if source is None:
            source = "legacy"

        logger.info(
            "Grasp params: target_width=%s force=%.1f source=%s label=%r",
            f"{target_width:.4f}m" if target_width is not None else "none",
            force,
            source,
            label,
        )

        self._last_grasp_target_width = (
            float(target_width) if target_width is not None else None
        )

        # Capture pre-grasp wrench baseline
        try:
            if hasattr(self.robot, "get_tcp_force"):
                ft0 = np.asarray(self.robot.get_tcp_force(), dtype=float)
                self._pre_grasp_force = ft0[:6].copy() if ft0.size >= 6 else np.zeros(6)
                self._pre_grasp_force_magnitude = float(np.linalg.norm(ft0[:3]))
            else:
                self._pre_grasp_force = np.zeros(6)
                self._pre_grasp_force_magnitude = 0.0
        except Exception:
            self._pre_grasp_force = np.zeros(6)
            self._pre_grasp_force_magnitude = 0.0
        logger.info(
            "Pre-grasp wrench: F=(%.2f,%.2f,%.2f) N, |F|=%.2f",
            self._pre_grasp_force[0],
            self._pre_grasp_force[1],
            self._pre_grasp_force[2],
            self._pre_grasp_force_magnitude,
        )

        try:
            tcp_pose = self._servo._get_tcp_pose()
            R_tcp = Rotation.from_rotvec(tcp_pose[3:6]).as_matrix()
            tx = R_tcp @ [1, 0, 0]
            ty = R_tcp @ [0, 1, 0]
            logger.info(
                "Pre-grasp orientation: tool_X=%.1f deg, tool_Y=%.1f deg",
                np.rad2deg(np.arctan2(tx[1], tx[0])),
                np.rad2deg(np.arctan2(ty[1], ty[0])),
            )
        except Exception:
            pass

        self._gripper_squeeze(force=force, speed=80, settle=1.5)
        self._robotiq_resqueeze(speed=50, force=100, settle=1.0)

        def _verdict(held: bool, source: str, pos, outcome: GraspOutcome) -> None:
            # G1 record for the success verifier, same shape as _grasp_v2's;
            # no extra register read on the exit path.
            self._record_grasp_verdict(
                GraspVerdict(
                    held=held,
                    source=source,
                    gripper_pos=pos,
                    force_empty_n=self._force_empty_n(),
                    label=self._last_keypoint_label or "",
                    strategy=getattr(self, "_active_grasp_strategy", "") or "",
                    outcome=outcome.value,
                )
            )

        grasp_tcp = self._get_current_position()
        logger.info(
            "Grasp TCP: [%.3f, %.3f, %.3f] (z=%.3f)",
            grasp_tcp[0],
            grasp_tcp[1],
            grasp_tcp[2],
            grasp_tcp[2],
        )

        if self._verify_grasp():
            if self.grasp_calibration is not None and self._last_keypoint_label:
                _final_z = self._get_current_position()[2]
                _delta_z = _final_z - self._grasp_perception_target_z
                self.grasp_calibration.record_success(
                    self._last_keypoint_label, _delta_z, n_retries=0
                )
        else:
            # Descent retry: try going lower in 1.5cm steps
            DESCENT_RETRIES = 5
            DESCENT_STEP = 0.015
            FLOOR_MARGIN = 0.002
            floor_z = self.TABLE_Z_FLOOR + FLOOR_MARGIN
            grasped = False
            hit_floor = False
            for retry in range(DESCENT_RETRIES):
                self._check_abort()
                self.robot.open_gripper()
                self._abort_sleep(0.3)
                current = self._get_current_position()
                target_z = current[2] - DESCENT_STEP
                if target_z <= floor_z:
                    target_z = floor_z
                    hit_floor = True
                    logger.warning(
                        "Grasp retry %d/%d: descent clamped to floor "
                        "z=%.3f (table=%.3f, margin=%.0fmm)",
                        retry + 1,
                        DESCENT_RETRIES,
                        target_z,
                        self.TABLE_Z_FLOOR,
                        FLOOR_MARGIN * 1000,
                    )
                current[2] = target_z
                logger.info(
                    "Grasp retry %d/%d: descending to z=%.3f",
                    retry + 1,
                    DESCENT_RETRIES,
                    current[2],
                )
                self._servo_to(
                    current, self.GRASP_ORIENTATION, velocity=self.velocity * 0.3
                )
                self._gripper_squeeze(force=force, speed=80, settle=1.0)
                # Force re-squeeze: the width-targeted _gripper_squeeze only
                # POSITIONS the jaws to the target width (never clamps), so a
                # retry would stop ~3.7cm wide on the block without gripping it.
                self._robotiq_resqueeze(speed=50, force=100, settle=1.0)
                if self._verify_grasp():
                    grasped = True
                    if self.grasp_calibration is not None and self._last_keypoint_label:
                        _final_z = self._get_current_position()[2]
                        _delta_z = _final_z - self._grasp_perception_target_z
                        self.grasp_calibration.record_success(
                            self._last_keypoint_label, _delta_z, n_retries=retry + 1
                        )
                    break
                if hit_floor:
                    break
            if not grasped:
                logger.warning(
                    "Grasp failed after %d retries at z=%.3f%s",
                    DESCENT_RETRIES,
                    self._get_current_position()[2],
                    " (reached floor)" if hit_floor else "",
                )
                self._holding = False
                _verdict(False, "legacy_retries_exhausted", None, GraspOutcome.EMPTY_CLOSE)
                return ExecutionResult(
                    action_type="grasp",
                    success=False,
                    message=(
                        f"Grasp failed after {DESCENT_RETRIES} "
                        f"descent retries" + (" (reached floor)" if hit_floor else "")
                    ),
                    duration=time.time() - t0,
                )

        # Post-lift verification
        logger.info("Grasp check passed, lifting 3cm to verify hold")
        self._holding = True
        current = self._get_current_position()
        lift_pos = current.copy()
        lift_pos[2] += 0.03
        self._move_to(lift_pos, self.GRASP_ORIENTATION, velocity=self.velocity * 0.3)
        self._abort_sleep(0.3)

        pre_pos = self._get_gripper_position()
        if self._gripper_type() == "robotiq_2f85":
            self._gripper_squeeze(force=100, speed=50, settle=0.5)
        else:
            self._abort_sleep(0.5)
        post_pos = self._get_gripper_position()

        if pre_pos is not None and post_pos is not None:
            logger.info("Post-lift grip check: pre=%d post=%d", pre_pos, post_pos)
            if post_pos >= self.GRIPPER_FULLY_CLOSED:
                logger.warning("Object lost during lift (pos=%d)", post_pos)
                self._holding = False
                _verdict(False, "legacy_lift_lost", post_pos, GraspOutcome.EMPTY_CLOSE)
                return ExecutionResult(
                    action_type="grasp",
                    success=False,
                    message=f"Object lost during lift (pos {post_pos})",
                    duration=time.time() - t0,
                )

        _verdict(True, "legacy_verify", post_pos, GraspOutcome.SECURED)
        return ExecutionResult(
            action_type="grasp",
            success=True,
            message=f"Grasped with force={force} (verified, pos={post_pos})",
            duration=time.time() - t0,
        )
