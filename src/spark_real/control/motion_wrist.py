"""Wrist-camera refinement of a target position (depth or ray-plane)."""

import logging
import os
from typing import Optional

import cv2
import numpy as np
from PIL import Image as PILImage

from spark_real.calibration import compute_wrist_camera_extrinsic
from spark_real.calibration_table import load_table_plane
from spark_real.perception.wrist_ray import intersect_ray_plane, pixel_ray_base

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)


class WristRefineMixin:
    """Wrist-camera refinement of a target position (depth or ray-plane)."""

    def _read_wrist_refine_mode_config(self) -> Optional[str]:
        """control.wrist_refine_mode from the profile, else the family YAML."""
        return self._control_config().get("wrist_refine_mode")

    def _wrist_refine_mode(self) -> str:
        """Resolve the wrist refinement mode: off | depth | ray_plane.

        ray_plane (default) needs RGB only, so it works with the wrist D435i
        opened color-only (streaming color+depth+IMU wedges the USB event
        ring on this rig). depth uses mask-median-depth backprojection. Cached
        per executor; SPARK_WRIST_REFINE_MODE overrides.
        """
        cached = getattr(self, "_wrist_refine_mode_cached", None)
        if cached is not None:
            return cached
        # _wrist_refine_mode_cfg is set by executor_core; fall back to reading
        # it here so this works standalone.
        mode = getattr(self, "_wrist_refine_mode_cfg", None)
        if mode is None:
            mode = self._read_wrist_refine_mode_config()
        mode = os.environ.get("SPARK_WRIST_REFINE_MODE") or mode
        mode = str(mode or self.WRIST_REFINE_MODE_DEFAULT).strip().lower()
        if mode not in self.WRIST_REFINE_MODES:
            logger.warning(
                "Unknown wrist_refine_mode %r; using %s",
                mode,
                self.WRIST_REFINE_MODE_DEFAULT,
            )
            mode = self.WRIST_REFINE_MODE_DEFAULT
        self._wrist_refine_mode_cached = mode
        logger.info("Wrist refine mode: %s", mode)
        return mode

    def _wrist_plane_z(
        self,
        coarse_target: np.ndarray,
        cam_z: float,
        z_plane: Optional[float] = None,
        tcp_z: Optional[float] = None,
    ) -> tuple:
        """Pick the horizontal plane the wrist ray is intersected with.

        Priority: explicit object-Z prior > the coarse target's own Z (the
        Kinect Z is reliable here; the wrist is wanted for lateral fixes) >
        the calibrated table surface > TABLE_Z_FLOOR (itself config-fed via
        control.table_z_floor).

        A prior is only taken if it clears WRIST_RAY_MIN_STANDOFF_M below BOTH
        the camera (a grazing ray runs away laterally) and the TCP (a "target"
        at the gripper's own height is not an object: search_keypoint passes
        its TCP waypoint as the coarse target, and the wrist cam sits above
        the TCP, so the camera test alone would wave it through).

        Returns (z, source).
        """
        coarse = np.asarray(coarse_target, dtype=float)
        for z, src in ((z_plane, "object_prior"), (coarse[2], "coarse_target")):
            if z is None:
                continue
            z = float(z)
            if cam_z - z <= self.WRIST_RAY_MIN_STANDOFF_M:
                continue
            if tcp_z is not None and tcp_z - z <= self.WRIST_RAY_MIN_STANDOFF_M:
                continue
            return z, src
        cal = load_table_plane(self._robot_family())
        if cal is not None:
            return float(cal["surface_z"]), "table_cal"
        return float(self.TABLE_Z_FLOOR), "table_z_floor"

    def _wrist_pos_from_depth(self, rs_cal, depth, mask, cx_det, cy_det, label):
        """Depth path: median depth inside the mask, backprojected."""
        if depth is None:
            logger.info(
                "Wrist refine: no depth stream (camera is color-only); "
                "use control.wrist_refine_mode: ray_plane"
            )
            return None, None
        mask_depths = depth[mask > 0]
        valid = mask_depths[(mask_depths > 0.01) & (mask_depths < 1.0)]
        if len(valid) < 5:
            logger.info("Wrist refine: no valid depth for '%s'", label)
            return None, None
        d_val = float(np.median(valid))
        x_cam = (cx_det - rs_cal.cx) * d_val / rs_cal.fx
        y_cam = (cy_det - rs_cal.cy) * d_val / rs_cal.fy
        pos = rs_cal.rotation_matrix @ np.array([x_cam, y_cam, d_val]) + rs_cal.position
        return pos, "depth=%.3f" % d_val

    def _wrist_pos_from_ray(
        self, rs_cal, cx_det, cy_det, coarse_target, z_plane, label, tcp_z=None
    ):
        """RGB-only path: mask-centroid ray x horizontal plane."""
        origin, direction = pixel_ray_base(cx_det, cy_det, rs_cal, rs_cal.extrinsic)
        plane_z, src = self._wrist_plane_z(coarse_target, float(origin[2]), z_plane, tcp_z)
        pos = intersect_ray_plane(origin, direction, plane_z)
        if pos is None:
            logger.info(
                "Wrist refine: ray for '%s' misses plane z=%.3f (%s) -- "
                "parallel or pointing away",
                label,
                plane_z,
                src,
            )
            return None, None
        return pos, "plane_z=%.3f (%s)" % (plane_z, src)

    def _refine_with_wrist(
        self,
        label: str,
        coarse_target: np.ndarray,
        z_plane: Optional[float] = None,
    ) -> Optional[dict]:
        """Capture wrist image, run SAM3, compute OBB orientation in tool frame.

        Returns a dict with refined position_3d, orientation_angle,
        aspect_ratio and the source mode, or None on failure. Gated off by
        default via control.wrist_refine; when off this returns None and never
        touches the camera. control.wrist_refine_mode picks how the 3D point
        is recovered (see _wrist_refine_mode). z_plane is the caller's
        object-height prior for ray_plane.
        """
        if not self._wrist_refine_enabled:
            return None
        if self._pipeline is None:
            return None
        mode = self._wrist_refine_mode()
        if mode == "off":
            return None
        rs = self._pipeline._realsense
        rs_cal = self._pipeline._realsense_cal
        perception = self._pipeline._perception
        if rs is None or rs_cal is None or perception is None:
            return None

        try:
            if torch is None:
                return None

            rgb, depth = rs.read()
            if rgb is None:
                return None

            tcp_pose = self._servo._get_tcp_pose()
            rs_cal.extrinsic = compute_wrist_camera_extrinsic(
                tcp_pose=tcp_pose,
                tool_offset=self._pipeline._wrist_tool_offset,
            )

            sam3 = perception._sam3
            if sam3 is None:
                return None
            pil_img = PILImage.fromarray(rgb)
            state = sam3.set_image(pil_img)
            state = sam3.set_text_prompt(prompt=label, state=state)
            masks = state.get("masks", torch.tensor([]))
            scores = state.get("scores", torch.tensor([]))
            if masks.numel() == 0:
                logger.info("Wrist refine: '%s' not found", label)
                return None

            best_idx = int(scores.argmax())
            mask = masks[best_idx].cpu().numpy().squeeze()
            ys, xs = np.where(mask > 0)
            if len(xs) < 20:
                return None

            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            wrist_angle = 0.0
            aspect_ratio = 1.0
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if len(largest) >= 5:
                    rect = cv2.minAreaRect(largest)
                    box = cv2.boxPoints(rect)
                    s1 = box[1] - box[0]
                    s2 = box[2] - box[1]
                    l1, l2 = np.linalg.norm(s1), np.linalg.norm(s2)
                    major = s1 if l1 >= l2 else s2
                    long, short = max(l1, l2), min(l1, l2)
                    if short > 0:
                        aspect_ratio = float(long / short)
                    wrist_angle = float(np.arctan2(major[1], major[0]))

            cx_det = float(xs.mean())
            cy_det = float(ys.mean())
            if mode == "depth":
                pos_world, detail = self._wrist_pos_from_depth(
                    rs_cal, depth, mask, cx_det, cy_det, label
                )
            else:
                pos_world, detail = self._wrist_pos_from_ray(
                    rs_cal,
                    cx_det,
                    cy_det,
                    coarse_target,
                    z_plane,
                    label,
                    tcp_z=float(tcp_pose[2]),
                )
            if pos_world is None:
                return None

            logger.info(
                "Wrist refine '%s' [%s]: img_angle=%.1f deg, ar=%.1f, %s, "
                "pos=(%.3f, %.3f, %.3f)",
                label,
                mode,
                np.rad2deg(wrist_angle),
                aspect_ratio,
                detail,
                pos_world[0],
                pos_world[1],
                pos_world[2],
            )
            return {
                "position_3d": pos_world,
                "orientation_angle": wrist_angle,
                "aspect_ratio": aspect_ratio,
                "source": mode,
            }
        except Exception as e:
            logger.warning("Wrist refine failed: %s", e)
            return None
