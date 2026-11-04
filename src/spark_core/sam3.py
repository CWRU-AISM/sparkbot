"""SAM3 perception: text/point-prompted masks and metric 3D positions for each detection.

The sam3 package must be importable (pip install -e of the pinned checkout); a missing or broken
install raises at load_models, never a silent perception-free run. Depth: the caller's aligned
metric depth (sim_detect.detect_rendered, the benchmark path) or DA3 monocular depth (detect).
"""

import contextlib
import functools
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from PIL import Image

from spark_core.mask_obb import (
    _cloud_median_world,
    _compute_orientation,
    _pca_obb,
    _sigma_n,
    _top_layer_mask,
    _world_xy_pca_obb,
    mask_height_profile,
)

logger = logging.getLogger(__name__)

# facebook/sam3 checkpoint revision the reported runs used (sam3 itself would load whatever main is).
SAM3_CKPT_REVISION = os.environ.get("SAM3_CKPT_REVISION", "3c879f39826c281e95690f02c7821c4de09afae7")

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    _SAM3_IMPORT_ERROR = None
except ImportError as _exc:  # raised at load_models with the cause attached
    build_sam3_image_model = None
    Sam3Processor = None
    _SAM3_IMPORT_ERROR = _exc


@dataclass
class ObjectDetection:
    """
    Detected object with 3D position.
    """

    label: str
    confidence: float
    centroid_2d: tuple
    mask_area: int
    depth_meters: float = 0.0
    position_3d: Optional[np.ndarray] = None
    bbox: Optional[tuple] = None
    mask: Optional[np.ndarray] = None
    camera: Optional[str] = None
    orientation_angle: float = 0.0
    aspect_ratio: float = 1.0
    obb_minor_m: float = 0.0  # OBB short-axis length in meters (drives grasp width prior)
    slots: Optional[list] = None  # slot poses for container detections (tray, plate, etc.)
    world_major_axis_rad: Optional[float] = None  # major-axis angle (radians from world +X)
    # Ray-plane orientation: the MASK's major axis intersected with the table
    # plane, in world frame. Kept alongside orientation_angle because the two
    # fail in opposite conditions. This one needs a crisp silhouette and ONE
    # centroid depth; the point-cloud OBB that usually overwrites
    # orientation_angle needs good depth over the WHOLE object. On thin
    # specular things -- silverware, tools -- the silhouette is excellent and
    # the point cloud is noise, so this is the axis to grasp by.
    plane_orientation_angle: Optional[float] = None
    plane_aspect_ratio: Optional[float] = None
    # (hue_deg, saturation, value, hue_conf) sampled over the eroded mask.
    # Populated opportunistically; None when the source RGB was unavailable.
    # Consumed by the prompt registry's optional hsv_cluster disambiguation.
    hsv_median: Optional[tuple] = None
    # Colour-free role aliases assigned by the prompt registry, e.g.
    # "same color block 1". A cached behaviour tree addresses these instead of
    # a literal colour when the staged object identity varies between episodes.
    role_labels: Optional[list] = None
    # How well-defined the OBB axes are, [0,1]. A round or soft object scores
    # near 0 even when its aspect_ratio reads high off backprojection noise, so
    # the grasp resolver can refuse a meaningless yaw. 0.0 means NOT MEASURED
    # (no depth), which the resolver treats as unknown, not as a veto.
    obb_confidence: float = 0.0
    # World-Z top and bottom of this mask's depth cloud, and how many valid
    # depth pixels backed them (see mask_obb.mask_height_profile). For a
    # CONTAINER these are its rim and its interior floor; for anything else,
    # simply the object's top and bottom. None / 0 means NOT MEASURED -- the
    # release-height computation falls back to the plan rather than guess.
    rim_z_m: Optional[float] = None
    interior_z_m: Optional[float] = None
    height_samples: int = 0
    # Agentview-only backprojected centroid, kept alongside the (possibly
    # wrist-fused) position_3d.  Same-source displacement measurement: a
    # later agentview-only re-detection must be compared against THIS, not
    # against the fused position - the two cameras disagree by a systematic
    # 1-2 cm on some objects, which otherwise reads as a phantom 'moved'.
    # None when the detection never had an agentview backprojection.
    position_agentview: Optional[np.ndarray] = None
    # ASPIRE reprompt bookkeeping (perception/detection conditioning).
    reprompt_attempts: int = 0
    # True when the surviving mask still trips a quality trigger. Forces the
    # `auto` grasp strategy to top-down and is reported to the planner.
    low_quality: bool = False


# torch.cuda.OutOfMemoryError (torch >= 1.13). None on older builds, in
# which case OOM containment is skipped rather than mis-catching.
_CUDA_OOM = getattr(torch.cuda, "OutOfMemoryError", None)


# A lock wait longer than this is a real contention event worth a log line.
_LOCK_WAIT_WARN_S = 0.25


class StageTimer:
    """Accumulates per-stage wall time for one detect, for a single log line.

    CUDA is async, so any stage that ends on GPU work synchronizes before the
    clock stops; otherwise the backbone's cost lands on whichever later stage
    first reads a tensor back to the host.
    """

    __slots__ = ("totals", "counts", "_sync")

    def __init__(self, sync_cuda=True):
        self.totals = {}
        self.counts = {}
        # SPARK_DETECT_TIMING_SYNC=0 keeps the stage lines and drops the CUDA
        # syncs, at the cost of async GPU work being attributed to a later
        # stage.
        if os.environ.get("SPARK_DETECT_TIMING_SYNC", "1").strip() == "0":
            sync_cuda = False
        self._sync = sync_cuda and torch.cuda.is_available()

    @contextlib.contextmanager
    def stage(self, name):
        if self._sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self._sync:
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            self.totals[name] = self.totals.get(name, 0.0) + dt
            self.counts[name] = self.counts.get(name, 0) + 1

    def total(self):
        return sum(self.totals.values())

    def describe(self):
        """"stage=total_s(xN)" for every stage, slowest first."""
        if not self.totals:
            return "no stages timed"
        parts = [
            f"{k}={v:.2f}s(x{self.counts[k]})"
            for k, v in sorted(self.totals.items(), key=lambda kv: -kv[1])
        ]
        return " ".join(parts)


def _serialized(fn):
    """Hold the instance's _detect_lock for the whole call + contain CUDA OOM.

    Serialization: the SAM3 predictor is STATEFUL (set_image then predict);
    two threads interleaving set_image/predict compute masks against the
    wrong image (the arm then servos to a phantom pose), and concurrent
    access to one CUDA model context can raise device-side asserts that
    poison the CUDA context. The executor's post-release re-detect thread
    runs concurrently with foreground detects, so every SAM3 entry point
    serializes here.

    OOM containment: the server shares its GPU with other jobs (and the
    Azure Kinect depth engine). empty_cache + one retry, then a CLEAR
    failure the caller can report.
    """

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        # Time the WAIT separately from the work: a detect slow because it sat
        # behind another thread's lock looks the same from outside as one slow
        # on the GPU. Only a wait worth acting on is logged.
        _t_wait = time.perf_counter()
        with self._detect_lock:
            _waited = time.perf_counter() - _t_wait
            if _waited > _LOCK_WAIT_WARN_S:
                logger.warning(
                    "[detect-timing] %s waited %.2fs for _detect_lock "
                    "(another detect/redetect thread held it)",
                    fn.__name__,
                    _waited,
                )
            if _CUDA_OOM is None:
                return fn(self, *args, **kwargs)
            try:
                return fn(self, *args, **kwargs)
            except _CUDA_OOM:
                logger.warning(
                    "CUDA OOM in %s; empty_cache + one retry "
                    "(is another GPU job running beside the server?)",
                    fn.__name__,
                )
                torch.cuda.empty_cache()
                try:
                    return fn(self, *args, **kwargs)
                except _CUDA_OOM as exc:
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        f"SAM3/DA3 {fn.__name__} failed: CUDA out of memory "
                        "even after empty_cache + retry. Free GPU memory "
                        "(stop competing GPU jobs) and re-run the detect."
                    ) from exc

    return wrapper


class SPARKPerception:
    """
    SAM3 + DA3 perception module.

    Detects objects from text prompts and estimates their 3D world positions
    using monocular metric depth.
    """

    def __init__(
        self,
        sam3_threshold: float = 0.05,
        da3_model: str = "depth-anything/DA3METRIC-LARGE",
    ):
        self.sam3_threshold = sam3_threshold
        self.da3_model_name = da3_model
        self._sam3 = None
        self._da3 = None
        # Serializes every SAM3/DA3 forward pass (see _serialized). RLock so
        # a locked entry point may call another locked helper.
        self._detect_lock = threading.RLock()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_models(self, load_da3: bool = True):
        """Load SAM3 and optionally DA3."""
        if self._sam3 is None:
            if _SAM3_IMPORT_ERROR is not None:
                raise ImportError(
                    "sam3 is not importable; install the pinned checkout with pip install -e"
                ) from _SAM3_IMPORT_ERROR
            logger.info("Loading SAM3...")
            from huggingface_hub import hf_hub_download
            ckpt = hf_hub_download("facebook/sam3", "sam3.pt", revision=SAM3_CKPT_REVISION)
            model = build_sam3_image_model(checkpoint_path=ckpt)
            self._sam3 = Sam3Processor(model)
            self._sam3.set_confidence_threshold(self.sam3_threshold)
            logger.info("SAM3 ready")
        if load_da3 and self._da3 is None:
            from depth_anything_3.api import DepthAnything3

            logger.info("Loading DA3 metric depth...")
            self._da3 = DepthAnything3.from_pretrained(self.da3_model_name)
            self._da3 = self._da3.to(self.device).eval()
            logger.info("DA3 ready")

    def set_point_prompt(self, x: float, y: float, state: dict, label: int = 1) -> dict:
        """
        Segment an object via a single point click using SAM2 decoder.

        Args:
            x: Pixel x-coordinate.
            y: Pixel y-coordinate.
            state: State dict previously returned by ``_sam3.set_image()``.
            label: 1 = foreground, 0 = background.

        Returns:
            Updated state dict with masks, scores, boxes (same format as
            ``set_text_prompt``).
        """
        self.load_models(load_da3=False)
        with self._detect_lock:
            return self._sam3.set_point_prompt(x, y, state, label=label)

    @_serialized
    def detect(
        self,
        rgb: np.ndarray,
        prompts: list,
        cam_pos=None,
        cam_mat=None,
        cam_fovy: float = 45.0,
    ) -> list:
        """
        Detect objects and estimate 3D positions using DA3 depth.
        """
        self.load_models()
        pil_img = Image.fromarray(rgb)
        h, w = rgb.shape[:2]

        # Run DA3 with focal scaling
        process_res = 504
        timer = StageTimer()
        _t_all = time.perf_counter()
        with timer.stage("da3_depth"):
            with torch.no_grad():
                da3_pred = self._da3.inference(
                    image=[pil_img], process_res=process_res
                )
                depth_map = da3_pred.depth[0]
            actual_f = h / (2 * np.tan(np.deg2rad(cam_fovy) / 2))
            da3_scale = process_res / max(w, h)
            depth_map = depth_map * (actual_f * da3_scale / 300.0)
            depth_resized = np.array(
                Image.fromarray(depth_map).resize((w, h), Image.BILINEAR)
            )

        detections = []
        # Encode the image ONCE and reuse it for every prompt: set_image() runs
        # the full vision backbone (1008x1008); set_text_prompt() only runs the
        # text encoder + grounding head against the cached backbone_out.
        # set_image() is deterministic and carries no prompt state.
        #
        # SAM3 needs bf16 autocast here, else the vision neck mixes
        # BFloat16 activations with Float weights.
        with timer.stage("encode"):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                base_state = self._sam3.set_image(pil_img)

        for prompt in prompts:
            # Shallow copy per prompt so each prompt's masks/scores land on its
            # own dict while the (expensive, image-only) backbone_out is shared.
            with timer.stage("grounding"):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    state = self._sam3.set_text_prompt(
                        prompt=prompt, state={**base_state}
                    )
            masks = state.get("masks", torch.tensor([]))
            scores = state.get("scores", torch.tensor([]))
            if masks.numel() == 0:
                continue

            n_confident = int((scores > 0.30).sum())
            sigma = _sigma_n(n_confident)
            best_idx = int(scores.argmax())
            score = float(scores[best_idx]) * sigma
            mask = masks[best_idx].cpu().numpy().squeeze()
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                continue

            cx, cy = float(xs.mean()), float(ys.mean())
            bbox = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
            obb_mask = _top_layer_mask(mask, depth_resized)
            img_angle_rad, aspect_ratio, short_px, long_px = _pca_obb(obb_mask)
            _wxy_orient, _wxy_ar, _wxy_mj_m, _wxy_mn_m, _wxy_conf = _world_xy_pca_obb(
                mask,
                depth_resized,
                cam_pos,
                cam_mat,
                fovy_deg=cam_fovy,
                use_opencv=False,
                return_confidence=True,
            )

            mask_depths = depth_resized[mask > 0]
            valid = mask_depths[mask_depths > 0.01]
            metric_depth = (
                float(np.median(valid))
                if len(valid) > 0
                else float(depth_resized[int(cy), int(cx)])
            )

            orientation_angle = _compute_orientation(
                mask,
                depth_resized,
                cam_mat,
                cam_pos,
                cam_fovy,
                cx,
                cy,
                h,
                w,
                img_angle_rad,
                long_px,
                metric_depth,
            )

            obb_minor_m = 0.0
            if short_px > 0 and metric_depth > 0.01:
                f_minor = h / (2 * np.tan(np.deg2rad(cam_fovy) / 2)) if cam_fovy else h
                if f_minor > 0:
                    obb_minor_m = float(short_px * metric_depth / f_minor)

            # Preserve the ray-plane answer BEFORE the point-cloud OBB
            # overwrites it: the override ignores _wxy_conf, and grasp_strategy
            # falls back to these when the point-cloud OBB is untrustworthy.
            plane_orientation_angle = float(orientation_angle)
            plane_aspect_ratio = float(aspect_ratio)

            if _wxy_orient is not None:
                orientation_angle = _wxy_orient
                aspect_ratio = float(_wxy_ar)
                obb_minor_m = float(_wxy_mn_m)

            det = ObjectDetection(
                label=prompt,
                confidence=score,
                centroid_2d=(cx, cy),
                mask_area=len(xs),
                depth_meters=metric_depth,
                bbox=bbox,
                mask=mask,
                orientation_angle=orientation_angle,
                plane_orientation_angle=plane_orientation_angle,
                plane_aspect_ratio=plane_aspect_ratio,
                aspect_ratio=aspect_ratio,
                obb_minor_m=obb_minor_m,
                obb_confidence=float(_wxy_conf or 0.0),
            )

            if cam_pos is not None and cam_mat is not None:
                centroid_w, _ = _cloud_median_world(
                    mask,
                    depth_resized,
                    cam_mat,
                    cam_pos,
                    fovy_deg=cam_fovy,
                    use_opencv=False,
                )
                if centroid_w is not None:
                    det.position_3d = centroid_w
                else:
                    f = h / (2 * np.tan(np.deg2rad(cam_fovy) / 2))
                    x_cam = (cx - w / 2) * metric_depth / f
                    y_cam = -(cy - h / 2) * metric_depth / f
                    z_cam = -metric_depth
                    det.position_3d = (
                        cam_mat @ np.array([x_cam, y_cam, z_cam]) + cam_pos
                    )
                # Same rim/interior profile as the hardware-depth path. DA3 is
                # monocular metric depth, so the absolute scale is softer here
                # -- the release-height clamps are what keep that honest.
                _profile = mask_height_profile(
                    mask,
                    depth_resized,
                    cam_pos,
                    cam_mat,
                    fovy_deg=cam_fovy,
                    use_opencv=False,
                )
                if _profile is not None:
                    det.rim_z_m = _profile["rim_z"]
                    det.interior_z_m = _profile["interior_z"]
                    det.height_samples = _profile["n_valid"]
            detections.append(det)

        # "geometry" is the residual: total wall time minus the stages timed
        # above, so nothing unaccounted for disappears.
        _elapsed = time.perf_counter() - _t_all
        logger.info(
            "[detect-timing] da3 %dx%d: %d prompt(s) -> %d det(s) in %.2fs | "
            "%s geometry+other=%.2fs",
            w,
            h,
            len(prompts),
            len(detections),
            _elapsed,
            timer.describe(),
            max(_elapsed - timer.total(), 0.0),
        )
        return detections

    # Score a non-best mask must clear to be considered a second instance of
    # the same prompt. A task whose second object is reliably dimmer or more
    # occluded than the first can relax this via the prompt registry's
    # fallback.relax_secondary_score, rather than lowering it globally and
    # admitting spurious instances into every other task.
    SECONDARY_MASK_SCORE = 0.50

    def _detect_with_rendered_depth(self, *args, **kwargs):
        """sim_detect.detect_rendered on this instance (the name the timing wrapper hooks)."""
        from spark_core.sim_detect import detect_rendered

        return detect_rendered(self, *args, **kwargs)

    @staticmethod
    def _backproject_mask(mask, depth, cam_pos, cam_mat, fovy, w, h):
        from spark_core.sim_detect import backproject_mask

        return backproject_mask(mask, depth, cam_pos, cam_mat, fovy, w, h)
