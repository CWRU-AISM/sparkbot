"""Detection on simulator renders (or any aligned RGB + metric depth): SAM3 masks backprojected
through the depth image, one ObjectDetection per kept mask, then cross-label dedup."""

import logging

import numpy as np
import torch
from PIL import Image

from spark_core import dedup
from spark_core.mask_obb import _sigma_n
from spark_core.mask_process import process_single_mask, select_mask_indices
from spark_core.sam3 import StageTimer, _serialized

logger = logging.getLogger(__name__)

# Image-space centroid distance below which two masks in ONE camera are
# CANDIDATES for being the same object; dedup.same_object_2d then decides with
# mask IoU whether they actually are.
_DEDUP_PX = 30


def backproject_mask(mask, depth, cam_pos, cam_mat, fovy, w, h):
    """World points of the mask pixels with depth in (0.01, 10) m (OpenGL camera convention)."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.empty((0, 3))
    f = h / (2 * np.tan(np.deg2rad(fovy) / 2))
    mask_depths = depth[mask > 0]
    valid = (mask_depths > 0.01) & (mask_depths < 10)
    if valid.sum() == 0:
        return np.empty((0, 3))
    vx, vy, vd = (
        xs[valid].astype(np.float64),
        ys[valid].astype(np.float64),
        mask_depths[valid],
    )
    pts_cam = np.stack([(vx - w / 2) * vd / f, -(vy - h / 2) * vd / f, -vd], axis=1)
    return (cam_mat @ pts_cam.T).T + cam_pos


@_serialized
def detect_rendered(
    perception,
    rgb,
    depth,
    prompts,
    cam_pos,
    cam_mat,
    cam_fovy,
    w,
    h,
    multi_instance_prompts=None,
    intrinsic_matrix=None,
    table_height=None,
    secondary_score=None,
):
    """
    SAM3 + hardware/rendered depth for exact 3D positions. `perception` is a SPARKPerception.
    """
    perception.load_models(load_da3=False)
    multi_instance_prompts = multi_instance_prompts or set()
    pil_img = Image.fromarray(rgb)

    use_opencv = intrinsic_matrix is not None
    if use_opencv:
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        cx_k = intrinsic_matrix[0, 2]
        cy_k = intrinsic_matrix[1, 2]
    else:
        fx = fy = cx_k = cy_k = None
    f = h / (2 * np.tan(np.deg2rad(cam_fovy) / 2))

    detections = []
    timer = StageTimer()
    # Encode the image once, reuse for every prompt. bf16 autocast is
    # required by SAM3 here (mixed-dtype crash without it).
    with timer.stage("encode"):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            base_state = perception._sam3.set_image(pil_img)

    for prompt in prompts:
        with timer.stage("grounding"):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                state = perception._sam3.set_text_prompt(
                    prompt=prompt, state={**base_state}
                )
        masks = state.get("masks", torch.tensor([]))
        scores = state.get("scores", torch.tensor([]))
        if masks.numel() == 0:
            continue

        sigma = _sigma_n(int((scores > 0.30).sum()))
        best_idx = int(scores.argmax())
        floor = (perception.SECONDARY_MASK_SCORE if secondary_score is None
                 else float(secondary_score))
        indices = select_mask_indices(
            prompt, masks, scores, best_idx, multi_instance_prompts, floor)

        for mask_idx in indices:
            with timer.stage("geometry"):
                det = process_single_mask(
                    prompt,
                    masks[mask_idx],
                    float(scores[mask_idx]) * sigma,
                    depth,
                    cam_pos,
                    cam_mat,
                    cam_fovy,
                    w,
                    h,
                    fx,
                    fy,
                    cx_k,
                    cy_k,
                    f,
                    use_opencv,
                    intrinsic_matrix,
                    table_height,
                    rgb=rgb,
                )
            if det is not None:
                detections.append(det)

    logger.info(
        "[detect-timing] hw-depth %dx%d: %d prompt(s) -> %d mask(s) in "
        "%.2fs | %s",
        w,
        h,
        len(prompts),
        len(detections),
        timer.total(),
        timer.describe(),
    )

    # Cross-label dedup. Two prompts that resolved to the SAME mask are
    # one object; two prompts whose masks merely overlap because one thing
    # is sitting inside the other are two. dedup.same_object_2d draws that
    # line with mask IoU -- centroid distance alone deletes the placed
    # object out of its container. See perception/dedup.py.
    aware = dedup.containment_aware()
    deduped = []
    for det in detections:
        duplicate = False
        for i, existing in enumerate(deduped):
            if det.label == existing.label:
                continue
            # Part-of guard: never dedup a part against its parent
            # ('mug handle' vs 'mug'). The parent mask can contain the
            # part, so their centroids collide and the part is lost.
            if det.label in existing.label or existing.label in det.label:
                continue
            if dedup.same_object_2d(det, existing, _DEDUP_PX, aware=aware):
                if det.confidence > existing.confidence:
                    deduped[i] = det
                duplicate = True
                break
        if not duplicate:
            deduped.append(det)
    return deduped
