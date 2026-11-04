"""Camera capture, per-camera SAM3 detection and the annotated planner image."""

import logging
import time
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image, ImageDraw

from spark_core.sam3 import ObjectDetection
from spark_real.calibration import compute_wrist_camera_extrinsic
from spark_real.utils.fonts import load_font

logger = logging.getLogger(__name__)


class CaptureMixin:
    """Camera capture, per-camera SAM3 detection and the annotated planner image."""

    def table_z_floor(self) -> float:
        """Work-surface Z in the base frame for perception plausibility checks.
        config.table_height > executor TABLE_Z_FLOOR > profile YAML > class literal.
        """
        floor = getattr(getattr(self, "config", None), "table_height", None)
        if floor is None:
            floor = getattr(getattr(self, "_executor", None), "TABLE_Z_FLOOR", None)
        if floor is None:
            profile = getattr(self, "profile", None)
            if profile is not None:
                floor = profile.control().get("table_z_floor", profile.raw.get("table_z_floor"))
        if floor is None:
            from spark_real.control.executor_core import ScoreExecutorCore

            floor = ScoreExecutorCore.TABLE_Z_FLOOR
        return float(floor)

    def _detect_camera(
        self, rgb, depth, cal, prompts, multi_instance_set, secondary_score
    ) -> List[ObjectDetection]:
        """One camera, one SAM3 pass. Extracted so the ASPIRE reprompt can run
        the SAME detection code the first pass ran."""
        if depth is not None and self.config.use_hardware_depth:
            K = None
            table_z = None
            if not np.allclose(cal.extrinsic, np.eye(4)):
                K = cal.intrinsic_matrix
                table_z = self.table_z_floor()
            return self._perception._detect_with_rendered_depth(
                rgb=rgb,
                depth=depth,
                prompts=prompts,
                cam_pos=cal.position,
                cam_mat=cal.rotation_matrix,
                cam_fovy=cal.fovy_degrees,
                w=cal.width,
                h=cal.height,
                multi_instance_prompts=multi_instance_set,
                intrinsic_matrix=K,
                table_height=table_z,
                secondary_score=secondary_score,
            )
        return self._perception.detect(
            rgb=rgb,
            prompts=prompts,
            cam_pos=cal.position,
            cam_mat=cal.rotation_matrix,
            cam_fovy=cal.fovy_degrees,
        )

    def capture(self) -> Dict[str, Dict]:
        """
        Capture RGB + depth from all active cameras.

        Returns dict of camera_name -> {"rgb": ndarray, "depth": ndarray, "calibration": CameraCalibration}
        """
        captures = {}
        # Per-camera device-read and lock-wait costs. A stream client holding
        # a Kinect read lock shows up here as *_lock, distinct from a slow
        # sensor (*_read); without the split the two are indistinguishable.
        _cap_waits = {}
        _t_capture = time.perf_counter()

        def _scaled(depth, cal):
            """
            Apply the per-camera depth correction z = z*scale + offset
            (sensor bias baked in by the RGB-D Procrustes hand-eye). Depth is
            in meters here, so depth_offset is in meters. The math lives on
            CameraCalibration.correct_depth so the streaming capture path
            (routes/streaming.py) applies the identical correction.
            """
            if depth is None or cal is None:
                return depth
            return cal.correct_depth(depth)

        if self._kinect is not None:
            _t_lock = time.perf_counter()
            with self._kinect_read_lock:
                _cap_waits["sideview_lock"] = time.perf_counter() - _t_lock
                _t_read = time.perf_counter()
                rgb, depth = self._kinect.read()
                _cap_waits["sideview_read"] = time.perf_counter() - _t_read
            if rgb is not None:
                depth = _scaled(depth, self._kinect_cal)
                captures["sideview"] = {
                    "rgb": rgb,
                    "depth": depth,
                    "calibration": self._kinect_cal,
                }

        if self._kinect2 is not None:
            _t_lock = time.perf_counter()
            with self._kinect2_read_lock:
                _cap_waits["birdview_lock"] = time.perf_counter() - _t_lock
                _t_read = time.perf_counter()
                rgb, depth = self._kinect2.read()
                _cap_waits["birdview_read"] = time.perf_counter() - _t_read
            if rgb is not None:
                depth = _scaled(depth, self._kinect2_cal)
                captures["birdview"] = {
                    "rgb": rgb,
                    "depth": depth,
                    "calibration": self._kinect2_cal,
                }

        if self._realsense is not None:
            _t_lock = time.perf_counter()
            with self._realsense_read_lock:
                _cap_waits["wrist_lock"] = time.perf_counter() - _t_lock
                _t_read = time.perf_counter()
                rgb, depth = self._realsense.read()
                _cap_waits["wrist_read"] = time.perf_counter() - _t_read
            if rgb is not None:
                depth = _scaled(depth, self._realsense_cal)
                # Update wrist cam extrinsic from robot FK
                if self._robot is not None and self._realsense_cal is not None:
                    try:
                        obs = self._robot.get_observation()
                        tcp = obs["tcp_pose"]
                        self._realsense_cal.extrinsic = compute_wrist_camera_extrinsic(
                            tcp_pose=tcp,
                            tool_offset=self._wrist_tool_offset,
                        )
                    except Exception:
                        pass
                captures["wrist"] = {
                    "rgb": rgb,
                    "depth": depth,
                    "calibration": self._realsense_cal,
                }

        # Stash the frames for consumers that run after perception but need
        # this run's pixels. Read-only.
        self._last_captures = captures
        # When these frames were taken. Verification reuses them inside a
        # configured freshness window instead of re-grabbing every camera; see
        # control/verify_scope.captures_for_verify.
        self._last_captures_t = time.time()

        _cap_total = time.perf_counter() - _t_capture
        logger.info(
            "[detect-timing] capture: %d camera(s) in %.2fs | %s",
            len(captures),
            _cap_total,
            " ".join(f"{k}={v:.2f}s" for k, v in sorted(_cap_waits.items()))
            or "no cameras",
        )

        return captures

    def detect(
        self,
        captures: Dict,
        prompts: List[str],
        multi_instance: bool = False,
        multi_instance_prompts=None,
        secondary_score=None,
        use_fusion: bool = True,
    ) -> List[ObjectDetection]:
        """
        Run SAM3 detection on captured images with hardware depth.

        Supports multi-instance detection: if SAM3 finds multiple masks
        for a prompt (e.g., 3 forks), they are numbered ("fork 1", "fork 2",
        "fork 3") so the planner and executor can address each one.

        `multi_instance` promotes every prompt; `multi_instance_prompts`
        promotes only the named ones, which is what a task spec supplies so
        that a container prompt does not start returning phantom instances
        just because a utensil prompt needs to. `secondary_score` overrides
        the score a non-best mask must clear to count as another instance.

        Returns all per-camera detections (for visualization across all tiles).
        Use detect_merged() for the executor (one best per label).
        """
        all_detections = []
        detect_cams = {k: v for k, v in captures.items()}
        # Resolved once per detect: an enabled-but-unusable proposer raises out
        # of here rather than degrading to a detector that sees nothing.
        # `use_fusion=False` skips the gate entirely, ASPIRE reprompts included
        # (extra SAM3 passes per detection); the verify path asks for this
        # because no predicate reads a fusion field. See control/verify_scope.
        gate = self.detection_gate() if use_fusion else None

        # Sort: calibrated cameras first (better 3D positions)
        cam_order = sorted(
            detect_cams.keys(),
            key=lambda c: (
                0 if not np.allclose(detect_cams[c]["calibration"].extrinsic, np.eye(4)) else 1
            ),
        )

        # Multi-instance: return all confident masks per prompt (batch tasks).
        # Single-instance: top-1 per prompt (user-specified objects).
        if multi_instance_prompts is not None:
            multi_instance_set = set(multi_instance_prompts)
        else:
            multi_instance_set = set(prompts) if multi_instance else set()

        for cam_name in cam_order:
            data = detect_cams[cam_name]
            rgb = data["rgb"]
            depth = data["depth"]
            cal = data["calibration"]

            # A camera without depth (e.g. wrist that opened on a USB2
            # link) must not fall through to the monocular-depth path:
            # the DepthAnything3 fallback raises when it is not installed
            # and kills the whole detect.
            if depth is None and self.config.use_hardware_depth:
                logger.warning("detect: skipping '%s' (no hardware depth)", cam_name)
                continue

            _t_cam = time.perf_counter()
            detections = self._detect_camera(
                rgb, depth, cal, prompts, multi_instance_set, secondary_score
            )
            _t_detect = time.perf_counter() - _t_cam

            for det in detections:
                det.camera = cam_name
            # THE FUSION SEAM. Stamps fused_confidence / axis_trust /
            # low_quality onto every detection before it can reach the merge,
            # the planner or the executor.
            _t_fuse = time.perf_counter()
            if gate is not None and detections:
                self._apply_fusion(gate, detections, rgb, depth, cal, cam_name, secondary_score)
            _t_fuse = time.perf_counter() - _t_fuse
            # One line per camera, always. The fusion gate's own lines are
            # emitted only after gate.apply() returns, so without this a slow
            # gate pass (31 s per camera has been seen) is indistinguishable
            # from a slow detect.
            logger.info(
                "[detect-timing] camera '%s': detect=%.2fs fusion=%.2fs "
                "total=%.2fs (%d det)",
                cam_name,
                _t_detect,
                _t_fuse,
                _t_detect + _t_fuse,
                len(detections),
            )
            all_detections.extend(detections)

        # Filter out physically impossible detections (below table surface)
        TABLE_Z_MIN = -0.28
        before = len(all_detections)
        all_detections = [
            d for d in all_detections if d.position_3d is None or d.position_3d[2] > TABLE_Z_MIN
        ]
        if len(all_detections) < before:
            logger.warning(
                "Filtered %d detections below table (Z < %.2f)",
                before - len(all_detections),
                TABLE_Z_MIN,
            )

        # Don't number yet; numbering happens after cross-camera merge
        logger.info("Detected %d total across %d cameras", len(all_detections), len(captures))
        return all_detections

    def _is_camera_calibrated(self, cam_name: str) -> bool:
        """
        Check if a camera has been calibrated (non-identity extrinsic).
        """
        cal = None
        if cam_name == "sideview":
            cal = self._kinect_cal
        elif cam_name == "birdview":
            cal = self._kinect2_cal
        elif cam_name == "wrist":
            cal = self._realsense_cal
        if cal is None:
            return False
        return not np.allclose(cal.extrinsic, np.eye(4))

    @staticmethod
    def _annotate_image(
        rgb: np.ndarray, detections: list, depth: np.ndarray = None, calibration=None
    ) -> np.ndarray:
        """
        Draw clean detection mask overlays and labels on an image.

        Shows ONLY:
          - Semi-transparent colored mask with contour outline
          - Label text with black outline for readability

        No bounding boxes, OBBs, axes, arrows, crosshairs, or coordinates.
        The ``depth`` and ``calibration`` parameters are accepted for
        API compatibility but no longer affect the mask overlay.
        """
        vis = rgb.copy()
        colors = [
            (255, 60, 60),
            (60, 220, 120),
            (60, 120, 255),
            (255, 200, 40),
            (220, 60, 220),
            (60, 220, 220),
            (255, 140, 40),
            (140, 255, 40),
            (40, 140, 255),
        ]

        for i, det in enumerate(detections):
            c = np.array(colors[i % len(colors)])
            c_tuple = tuple(int(x) for x in c)
            mask = getattr(det, "mask", None)
            if mask is not None and mask.shape == vis.shape[:2]:
                # Semi-transparent mask tint
                vis[mask > 0] = (vis[mask > 0] * 0.55 + c * 0.45).astype(np.uint8)
                # Contour outline
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(vis, contours, -1, c_tuple, 2)

        pil = Image.fromarray(vis)
        draw = ImageDraw.Draw(pil)
        font = load_font(16)

        for i, det in enumerate(detections):
            cx, cy = det.centroid_2d if det.centroid_2d else (0, 0)
            c = colors[i % len(colors)]
            label_text = det.label
            tx, ty = int(cx) + 14, int(cy) - 10
            # Black outline for readability against busy backgrounds
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    draw.text((tx + dx, ty + dy), label_text, fill=(0, 0, 0), font=font)
            draw.text((tx, ty), label_text, fill=c, font=font)

        return np.array(pil)
