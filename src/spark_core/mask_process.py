"""Per-mask processing for depth-backed SAM3 detections: instance selection, mask -> ObjectDetection,
punch-through correction for translucent objects."""

import logging

import cv2
import numpy as np

from spark_core.mask_obb import (
    _cloud_median_world,
    _compute_orientation,
    _pca_obb,
    _top_layer_mask,
    _world_xy_pca_obb,
    mask_color_stats,
    mask_height_profile,
)
from spark_core.sam3 import ObjectDetection

logger = logging.getLogger(__name__)


def select_mask_indices(prompt, masks, scores, best_idx, multi_instance_prompts, floor):
    """
    Select which mask indices to process (NMS for multi-instance). `floor` is the score a
    non-best mask must clear to count as a second instance.
    """
    if prompt not in multi_instance_prompts:
        return [best_idx]
    candidates = [best_idx] + sorted(
        [
            i
            for i in range(len(scores))
            if i != best_idx and float(scores[i]) > floor
        ],
        key=lambda i: float(scores[i]),
        reverse=True,
    )
    kept_masks = []
    indices = []
    for ci in candidates:
        m = masks[ci].cpu().numpy().squeeze().astype(bool)
        m_ys, m_xs = np.where(m)
        m_cx, m_cy = float(m_xs.mean()), float(m_ys.mean())
        suppress = False
        for km in kept_masks:
            km_ys, km_xs = np.where(km)
            if (
                np.sqrt(
                    (m_cx - float(km_xs.mean())) ** 2
                    + (m_cy - float(km_ys.mean())) ** 2
                )
                < 30
            ):
                suppress = True
                break
            inter = np.logical_and(m, km).sum()
            union = np.logical_or(m, km).sum()
            iou = inter / union if union > 0 else 0
            iomin = (
                inter / min(m.sum(), km.sum()) if min(m.sum(), km.sum()) > 0 else 0
            )
            if iou > 0.3 or iomin > 0.5:
                suppress = True
                break
        if not suppress:
            indices.append(ci)
            kept_masks.append(m)
    return indices


def process_single_mask(
    prompt,
    mask_tensor,
    score,
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
    rgb=None,
):
    """
    Process a single SAM3 mask into an ObjectDetection.

    `rgb` is optional and used only to sample the mask's median colour;
    omitting it leaves `hsv_median` None and costs nothing else.
    """
    mask = mask_tensor.cpu().numpy().squeeze()
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    cx, cy = float(xs.mean()), float(ys.mean())
    point_world, d_val = _cloud_median_world(
        mask,
        depth,
        cam_mat,
        cam_pos,
        fx=fx,
        fy=fy,
        cx_k=cx_k,
        cy_k=cy_k,
        fovy_deg=cam_fovy,
        use_opencv=use_opencv,
    )

    if point_world is None:
        ui, vi = int(cx), int(cy)
        if 0 <= vi < h and 0 <= ui < w and depth[vi, ui] > 0.01:
            d_val = float(depth[vi, ui])
            if use_opencv:
                x_cam = (cx - cx_k) * d_val / fx
                y_cam = (cy - cy_k) * d_val / fy
                z_cam = d_val
            else:
                x_cam = (cx - w / 2) * d_val / f
                y_cam = -(cy - h / 2) * d_val / f
                z_cam = -d_val
            point_world = cam_mat @ np.array([x_cam, y_cam, z_cam]) + cam_pos
        else:
            return None

    # Punch-through correction for translucent/concave objects
    point_world = punch_through_correction(
        prompt,
        mask,
        depth,
        point_world,
        d_val,
        xs,
        ys,
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
        table_height,
    )

    # Orientation + OBB
    obb_mask_e = _top_layer_mask(mask, depth)
    img_angle_rad, ar, short_px_e, long_px_e = _pca_obb(obb_mask_e)
    orient_angle = img_angle_rad

    _wxy = _world_xy_pca_obb(
        mask,
        depth,
        cam_pos,
        cam_mat,
        return_confidence=True,
        fx=fx,
        fy=fy,
        cx_k=cx_k,
        cy_k=cy_k,
        fovy_deg=cam_fovy,
        use_opencv=use_opencv,
    )

    if long_px_e > 0 and d_val and d_val > 0.01:
        fx_b = (
            float(intrinsic_matrix[0, 0])
            if (use_opencv and intrinsic_matrix is not None)
            else (
                h / (2 * np.tan(np.deg2rad(cam_fovy) / 2)) if cam_fovy else float(h)
            )
        )
        fy_b = (
            float(intrinsic_matrix[1, 1])
            if (use_opencv and intrinsic_matrix is not None)
            else fx_b
        )
        cx_b = (
            float(intrinsic_matrix[0, 2])
            if (use_opencv and intrinsic_matrix is not None)
            else w / 2.0
        )
        cy_b = (
            float(intrinsic_matrix[1, 2])
            if (use_opencv and intrinsic_matrix is not None)
            else h / 2.0
        )
        orient_angle = _compute_orientation(
            mask,
            depth,
            cam_mat,
            cam_pos,
            cam_fovy,
            cx,
            cy,
            h,
            w,
            img_angle_rad,
            long_px_e,
            d_val,
            use_opencv,
            fx_b,
            fy_b,
            cx_b,
            cy_b,
        )

    obb_minor_m_e = 0.0
    if short_px_e > 0 and d_val > 0.01:
        f_for_minor = (
            float(intrinsic_matrix[1, 1])
            if (intrinsic_matrix is not None)
            else (
                h / (2 * np.tan(np.deg2rad(cam_fovy) / 2)) if cam_fovy else float(h)
            )
        )
        if f_for_minor > 0:
            obb_minor_m_e = float(short_px_e * d_val / f_for_minor)

    # Preserve the ray-plane answer BEFORE the point-cloud OBB overwrites
    # it (same guard as detect()); the grasp fallback reads these.
    plane_orientation_angle = (
        float(orient_angle) if orient_angle is not None else None
    )
    plane_aspect_ratio = float(ar) if ar is not None else None

    obb_conf_e = 0.0
    if _wxy[0] is not None:
        orient_angle = _wxy[0]
        ar = float(_wxy[1])
        obb_minor_m_e = float(_wxy[3])
        obb_conf_e = float(_wxy[4] or 0.0)

    hsv = mask_color_stats(rgb, mask > 0) if rgb is not None else None

    # World-Z top/bottom of the same depth cloud the position came from.
    # A container's rim and interior floor; the release height is computed
    # from these instead of a planner constant (control.release_height).
    _profile = mask_height_profile(
        mask,
        depth,
        cam_pos,
        cam_mat,
        fx=fx,
        fy=fy,
        cx_k=cx_k,
        cy_k=cy_k,
        fovy_deg=cam_fovy,
        use_opencv=use_opencv,
    )

    return ObjectDetection(
        label=prompt,
        confidence=score,
        centroid_2d=(cx, cy),
        mask_area=len(xs),
        depth_meters=d_val,
        position_3d=point_world,
        bbox=(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())),
        mask=mask,
        orientation_angle=orient_angle,
        aspect_ratio=ar,
        plane_orientation_angle=plane_orientation_angle,
        plane_aspect_ratio=plane_aspect_ratio,
        obb_minor_m=obb_minor_m_e,
        obb_confidence=obb_conf_e,
        hsv_median=hsv,
        rim_z_m=_profile["rim_z"] if _profile else None,
        interior_z_m=_profile["interior_z"] if _profile else None,
        height_samples=_profile["n_valid"] if _profile else 0,
    )


def punch_through_correction(
    prompt,
    mask,
    depth,
    point_world,
    d_val,
    xs,
    ys,
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
    table_height,
):
    """
    Correct Z for translucent objects where depth punches through to table.

    table_height is the work surface Z in the same world frame as the
    camera pose; None means the caller has no table, and no correction.
    """
    if point_world is None or table_height is None or len(xs) < 30:
        return point_world
    _valid_d = mask & (depth > 0.01)
    if _valid_d.sum() < 30:
        return point_world
    _vy, _vx = np.where(_valid_d)
    _vd = depth[_vy, _vx]
    if use_opencv:
        _vxc = (_vx - cx_k) * _vd / fx
        _vyc = (_vy - cy_k) * _vd / fy
        _vzc = _vd
    else:
        _vxc = (_vx - w / 2) * _vd / f
        _vyc = -(_vy - h / 2) * _vd / f
        _vzc = -_vd
    _pts_world_v = (cam_mat @ np.column_stack([_vxc, _vyc, _vzc]).T).T + cam_pos
    _wz = _pts_world_v[:, 2]
    _wz_p05, _wz_p50, _wz_p95 = np.percentile(_wz, [5, 50, 95])
    _wz_spread = float(_wz_p95 - _wz_p05)
    _bbox_h_px = float(ys.max() - ys.min())
    _table_z = float(table_height)
    _flat = _wz_spread < 0.040
    _on_table = _wz_p50 < (_table_z + 0.030)
    _tall_bbox = _bbox_h_px > 40
    _bbox_w_px = float(xs.max() - xs.min())
    # Use the oriented minor axis, not the axis-aligned bbox: an
    # elongated utensil lying diagonally spans a wide box in both image
    # axes and would wrongly pass the chunky gate. minAreaRect gives the
    # true object width regardless of yaw, so thin cutlery fails the gate
    # while a real translucent cup still passes.
    try:
        _rect = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
        _short_px = float(min(_rect[1])) or min(_bbox_h_px, _bbox_w_px)
    except Exception:
        _short_px = min(_bbox_h_px, _bbox_w_px)
    _fy_m = fy if (use_opencv and fy is not None) else f
    _obj_minor_m = _short_px * d_val / _fy_m if _fy_m and _fy_m > 0 else 0.0
    _chunky = _obj_minor_m > 0.040
    _img_ar = _bbox_h_px / max(_bbox_w_px, 1.0)
    _tall_shape = _img_ar > 1.8
    if _flat and _on_table and _tall_bbox and _chunky and _tall_shape:
        _fy_h = fy if (use_opencv and fy is not None) else f
        _est_h_m = max(float(_bbox_h_px * d_val / _fy_h), 0.050)
        _new_z = _table_z + _est_h_m / 2.0
        logger.info(
            "[punch-through] '%s': spread=%.3fm median=%.3f "
            "table=%.3f bbox_h=%.0fpx minor=%.3fm -> Z %.3f -> %.3f",
            prompt,
            _wz_spread,
            _wz_p50,
            _table_z,
            _bbox_h_px,
            _obj_minor_m,
            point_world[2],
            _new_z,
        )
        return np.array([float(point_world[0]), float(point_world[1]), _new_z])
    return point_world
