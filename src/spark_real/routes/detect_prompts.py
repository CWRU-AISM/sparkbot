"""Click and box SAM3 prompts on one consistent frame, back-projected to the base frame."""

import logging

import numpy as np
import torch
from PIL import Image

from spark_core.mask_obb import _cloud_median_world, _pca_obb
from spark_core.sam3 import ObjectDetection
from spark_real.routes import state

logger = logging.getLogger(__name__)


def _snapshot_frame(pipeline):
    """
    One consistent ``(rgb, depth, cam_name)`` for a click/box detect.

    Returns the last published frame, or captures one and publishes it if
    nothing has been published yet. The publish also carries the camera name,
    so the very first click after a restart deprojects through the right
    extrinsics.

    Returns ``(None, None, None)`` when there is nothing to capture.
    """
    rgb, depth, cam_name = state.get_last_frame()
    if rgb is not None:
        return rgb, depth, cam_name
    captures = pipeline.capture()
    if not captures:
        return None, None, None
    cam_key = state.active_camera
    if cam_key not in captures:
        cam_key = next(iter(captures))
    cam = captures[cam_key]
    state.set_last_frame(cam["rgb"], cam["depth"], cam_key)
    return state.get_last_frame()


def _cal_for_camera(pipeline, cam_name):
    """The CameraCalibration for a slot name, or None (sim / unknown slot)."""
    if pipeline._kinect_cal and cam_name == "sideview":
        return pipeline._kinect_cal
    if pipeline._kinect2_cal and cam_name == "birdview":
        return pipeline._kinect2_cal
    if pipeline._realsense_cal and cam_name == "wrist":
        return pipeline._realsense_cal
    return None


def _backproject_prompt_mask(pipeline, mask, depth, cam_name, h, w):
    """Median world position + robust depth for a click/box mask.

    Deprojects through mask_geometry._cloud_median_world with
    use_opencv=True, the same convention and code the text-detect path
    uses (_process_single_mask on hardware depth). The hand-eye extrinsic
    is solved for the OpenCV pinhole convention (y down, z_c = +d);
    deprojecting in the OpenGL/MuJoCo convention (y up, z_c = -d) moves
    every birdview annotation from the table (z = -0.270 in base) to
    roughly the camera's slant range (z = +1.52..1.57). Sharing
    _cloud_median_world also buys the top-surface Z logic (95th percentile
    when the mask straddles table + object), so click/box and text
    detections of the same object agree in all three axes.

    Returns ``(position_3d | None, d_val, cal | None)``.
    """
    cal = _cal_for_camera(pipeline, cam_name)
    if depth is None or cal is None:
        return None, 0.0, cal
    fx = cal.fx if cal.fx > 0 else h / (2 * np.tan(np.deg2rad(cal.fovy_degrees) / 2))
    fy = cal.fy if cal.fy > 0 else fx
    cx_intr = cal.cx if cal.cx > 0 else w / 2
    cy_intr = cal.cy if cal.cy > 0 else h / 2
    position_3d, d_val = _cloud_median_world(
        mask,
        depth,
        cal.rotation_matrix,
        cal.position,
        fx=fx,
        fy=fy,
        cx_k=cx_intr,
        cy_k=cy_intr,
        use_opencv=True,
    )
    return position_3d, d_val, cal


def _run_point_prompt(pipeline, rgb, depth, px_x, px_y, label="", cam_name=None):
    """
    Run SAM3 with a true point prompt on the image.

    Uses the SAM2 instance predictor path (predict_inst) for precise
    click-to-segment.  Returns a list with 0 or 1 ObjectDetection.

    ``cam_name`` names the camera ``rgb``/``depth`` came from and selects the
    calibration to deproject through. It is a parameter, not a re-read of
    state.last_cam_name, so a concurrent stream publish cannot swap it
    mid-call (see routes/state.set_last_frame).
    """
    perception = pipeline._perception
    perception.load_models(load_da3=False)
    h, w = rgb.shape[:2]
    pil_img = Image.fromarray(rgb)
    # The SAM3 predictor is stateful; serialize set_image + prompt against
    # any concurrent pipeline.detect() (executor re-detect thread, stream
    # detects) via the shared perception lock.
    with perception._detect_lock:
        sam3_state = perception._sam3.set_image(pil_img)

        try:
            sam3_state = perception.set_point_prompt(
                px_x,
                px_y,
                sam3_state,
                label=1,
            )
        except Exception as exc:
            logger.warning("Point prompt failed (%s), will fall back to box", exc)
            return []  # caller should fall back

    masks = sam3_state.get("masks", torch.tensor([]))
    scores = sam3_state.get("scores", torch.tensor([]))
    if masks.numel() == 0:
        return []

    best_idx = int(scores.argmax())
    score = float(scores[best_idx])
    mask = masks[best_idx].cpu().numpy().squeeze()
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return []

    cx_det, cy_det = float(xs.mean()), float(ys.mean())

    # Backproject with depth -- shared with _run_box_prompt and, more
    # importantly, with the text-detect convention (see helper docstring).
    position_3d, d_val, cal = _backproject_prompt_mask(
        pipeline, mask, depth, cam_name, h, w
    )

    img_angle_rad, aspect_ratio, short_px, _long_px = _pca_obb(mask > 0)
    orientation_angle = float(img_angle_rad) if aspect_ratio > 1.3 else 0.0

    obb_minor_m = 0.0
    if short_px > 0 and d_val > 0.01 and cal is not None:
        f_minor = (
            cal.fx if cal.fx > 0 else h / (2 * np.tan(np.deg2rad(cal.fovy_degrees) / 2))
        )
        if f_minor > 0:
            obb_minor_m = float(short_px * d_val / f_minor)

    det = ObjectDetection(
        label=label or "selected",
        confidence=score,
        centroid_2d=(cx_det, cy_det),
        mask_area=len(xs),
        depth_meters=d_val,
        position_3d=position_3d,
        orientation_angle=orientation_angle,
        aspect_ratio=aspect_ratio,
        obb_minor_m=obb_minor_m,
    )
    det.mask = mask
    return [det]


def _run_box_prompt(pipeline, rgb, depth, x1, y1, x2, y2, label="", cam_name=None):
    """
    Run SAM3 with a box prompt on the image.

    ``cam_name`` selects the calibration; see _run_point_prompt for why it is
    a parameter rather than a re-read of state.last_cam_name.
    """
    perception = pipeline._perception
    perception.load_models(load_da3=False)
    h, w = rgb.shape[:2]
    pil_img = Image.fromarray(rgb)

    bx = (x1 + x2) / 2.0 / w
    by = (y1 + y2) / 2.0 / h
    bw = abs(x2 - x1) / w
    bh = abs(y2 - y1) / h

    # Stateful predictor: serialize set_image + geometric prompt against
    # concurrent detects (see _run_point_prompt).
    with perception._detect_lock:
        sam3_state = perception._sam3.set_image(pil_img)
        perception._sam3.reset_all_prompts(sam3_state)
        sam3_state = perception._sam3.add_geometric_prompt(
            state=sam3_state,
            box=[bx, by, bw, bh],
            label=True,
        )

    masks = sam3_state.get("masks", torch.tensor([]))
    scores = sam3_state.get("scores", torch.tensor([]))
    if masks.numel() == 0:
        return []

    best_idx = int(scores.argmax())
    score = float(scores[best_idx])
    mask = masks[best_idx].cpu().numpy().squeeze()
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return []

    cx_det, cy_det = float(xs.mean()), float(ys.mean())

    # Backproject with depth -- shared helper, same convention as text
    # detect (see _backproject_prompt_mask).
    position_3d, d_val, cal = _backproject_prompt_mask(
        pipeline, mask, depth, cam_name, h, w
    )

    img_angle_rad, aspect_ratio, short_px, _long_px = _pca_obb(mask > 0)
    orientation_angle = float(img_angle_rad) if aspect_ratio > 1.3 else 0.0

    obb_minor_m = 0.0
    if short_px > 0 and d_val > 0.01 and cal is not None:
        f_minor = (
            cal.fx if cal.fx > 0 else h / (2 * np.tan(np.deg2rad(cal.fovy_degrees) / 2))
        )
        if f_minor > 0:
            obb_minor_m = float(short_px * d_val / f_minor)

    det = ObjectDetection(
        label=label or "selected",
        confidence=score,
        centroid_2d=(cx_det, cy_det),
        mask_area=len(xs),
        depth_meters=d_val,
        position_3d=position_3d,
        orientation_angle=orientation_angle,
        aspect_ratio=aspect_ratio,
        obb_minor_m=obb_minor_m,
    )
    det.mask = mask
    return [det]


def _save_anchor_if_requested(pipeline, task, label, camera, box_norm):
    """Persist an operator annotation for `task`, or return None.

    Stored NORMALISED so it survives a resolution change, and only when the
    operator explicitly asked -- a one-off rescue click must not silently
    become permanent state. The anchor can later only ADD a box-prompted
    candidate; it can never force a detection past the quality/count gates.
    """
    if not task or not str(task).strip():
        return None
    store = getattr(pipeline, "_annotation_anchors", None)
    if store is None:
        logger.warning("annotation anchor requested but no store on the pipeline")
        return None
    try:
        anchor = store.save(task, label or "object", camera, box_norm)
        return {"task": str(task), **anchor}
    except Exception as exc:  # noqa: BLE001 - never fail a detect over this
        logger.warning("could not save annotation anchor: %s", exc)
        return None
