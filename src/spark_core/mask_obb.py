"""
Free geometry helpers for SPARK mask-to-3D perception.

Pure numpy/cv2 functions extracted from spark_perception. They turn SAM3
masks plus depth and camera extrinsics into world-frame positions and
oriented bounding boxes. No model state, no torch.

Contents:
    _sigma_n                 cardinality penalty (disabled, returns 1.0)
    _cloud_median_world      backproject a mask to its per-axis world median
    _top_layer_mask          keep only the top-surface depth layer of a mask
    _world_xy_pca_obb        world-XY footprint OBB via backprojection + PCA
    _pca_obb                 image-frame OBB via PCA
    _ray_intersect_table     intersect a pixel ray with a world z-plane
    _compute_orientation     world-frame orientation via ray-plane projection
    mask_color_stats         median HSV + hue concentration over a mask
    mask_height_profile      world-Z top/bottom of a mask's depth cloud
"""

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

# Fraction of a uniform extent covered by the 5th-95th percentile band.
_TRIM_COVERAGE = 0.90

# Aspect ratio at which the PCA major axis counts as fully resolved. Below it
# the two eigenvalues are close and the reported angle is partly arbitrary.
_AXIS_CONF_AR_REF = 2.0

# mask_height_profile: world-Z percentiles standing in for a mask's top and
# bottom surface. Asymmetric on purpose -- see that function's docstring.
_RIM_PCT = 95.0
_INTERIOR_PCT = 20.0
# Below this many valid depth pixels a height profile is noise, not a surface.
_HEIGHT_PROFILE_MIN_PIXELS = 50


def _sigma_n(n_candidates: int) -> float:
    """
    Cardinality penalty disabled; returns 1.0 unconditionally.
    """
    return 1.0


def _cloud_median_world(
    mask,
    depth,
    cam_mat,
    cam_pos,
    fx=None,
    fy=None,
    cx_k=None,
    cy_k=None,
    fovy_deg=None,
    use_opencv=False,
    min_depth=0.01,
    max_depth=10.0,
):
    """
    Backproject every mask pixel to 3D, return per-axis median.
    """
    h, w = mask.shape
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None, 0.0
    depths = depth[ys, xs].astype(np.float64)
    valid = (depths > min_depth) & (depths < max_depth)
    if valid.sum() < 3:
        return None, 0.0
    xs = xs[valid].astype(np.float64)
    ys = ys[valid].astype(np.float64)
    depths = depths[valid]

    if use_opencv:
        if fx is None or fy is None or cx_k is None or cy_k is None:
            return None, 0.0
        x_cam = (xs - cx_k) * depths / fx
        y_cam = (ys - cy_k) * depths / fy
        z_cam = depths
    else:
        if fovy_deg is None:
            return None, 0.0
        f = h / (2 * np.tan(np.deg2rad(fovy_deg) / 2))
        x_cam = (xs - w / 2) * depths / f
        y_cam = -(ys - h / 2) * depths / f
        z_cam = -depths

    pts_cam = np.column_stack([x_cam, y_cam, z_cam])
    pts_world = (cam_mat @ pts_cam.T).T + cam_pos

    # Z: use 95th percentile when bimodal (table + object surface)
    z_all = pts_world[:, 2]
    z_range = float(np.percentile(z_all, 95) - np.percentile(z_all, 5))
    if z_range > 0.04:
        top_z = float(np.percentile(z_all, 95))
        top_mask = z_all > (top_z - 0.03)
        if top_mask.sum() >= 3:
            centroid_xy = np.median(pts_world[top_mask, :2], axis=0)
            centroid = np.array([centroid_xy[0], centroid_xy[1], top_z])
        else:
            centroid = np.median(pts_world, axis=0)
            centroid[2] = top_z
    else:
        centroid = np.median(pts_world, axis=0)
    d_val = float(np.percentile(depths, 25))
    return centroid, d_val


def _top_layer_mask(
    mask: np.ndarray, depth: np.ndarray, layer_m: float = 0.02, min_pixels: int = 30
) -> np.ndarray:
    """
    Filter mask to its top-surface layer for OBB computation.
    """
    if mask is None or depth is None:
        return mask
    ys, xs = np.where(mask > 0)
    if len(xs) < min_pixels:
        return mask
    depths = depth[ys, xs]
    valid = depths > 0.01
    if int(valid.sum()) < min_pixels:
        return mask
    valid_depths = depths[valid]
    top_d = float(np.percentile(valid_depths, 10))
    layer_keep = valid & (depths <= top_d + layer_m)
    if int(layer_keep.sum()) < min_pixels:
        return mask
    filtered = np.zeros_like(mask)
    filtered[ys[layer_keep], xs[layer_keep]] = mask[ys[layer_keep], xs[layer_keep]]
    return filtered


def _trimmed_span(proj: np.ndarray) -> float:
    """5th-95th percentile span, rescaled to a full-extent equivalent.

    max()-min() over backprojected depth is a RANGE statistic: two stray
    pixels set it (a plushie read aspect ratio 3.14 that way). The 5-95 band
    of a uniform footprint covers 90% of its extent, so divide by 0.9 to keep
    obb_minor_m comparable with the untrimmed values the grasp width prior
    and container extents are tuned against.
    """
    lo, hi = np.percentile(proj, [5.0, 95.0])
    return float((hi - lo) / _TRIM_COVERAGE)


def _world_xy_pca_obb(
    mask,
    depth,
    cam_pos,
    cam_mat,
    fx=None,
    fy=None,
    cx_k=None,
    cy_k=None,
    fovy_deg=None,
    use_opencv=False,
    min_pixels=30,
    return_confidence=False,
):
    """
    World-XY footprint OBB via backprojection + PCA.

    Returns (orient_world_rad, ar, mj_len_m, mn_len_m), or with
    ``return_confidence`` a 5th element obb_confidence in [0,1]. All None on
    failure.
    """
    _fail = (None, None, None, None, None) if return_confidence else (None,) * 4
    if mask is None or depth is None or cam_mat is None or cam_pos is None:
        return _fail
    ys, xs = np.where(mask > 0)
    if len(xs) < min_pixels:
        return _fail
    depths = depth[ys, xs].astype(np.float64)
    valid = depths > 0.01
    if int(valid.sum()) < min_pixels:
        return _fail
    xs_v, ys_v, d_v = (
        xs[valid].astype(np.float64),
        ys[valid].astype(np.float64),
        depths[valid],
    )
    h, w = mask.shape
    if use_opencv:
        if fx is None or fy is None or cx_k is None or cy_k is None:
            return _fail
        x_cam = (xs_v - cx_k) * d_v / fx
        y_cam = (ys_v - cy_k) * d_v / fy
        z_cam = d_v
    else:
        if fovy_deg is None:
            return _fail
        f = h / (2 * np.tan(np.deg2rad(fovy_deg) / 2))
        x_cam = (xs_v - w / 2) * d_v / f
        y_cam = -(ys_v - h / 2) * d_v / f
        z_cam = -d_v
    pts_cam = np.column_stack([x_cam, y_cam, z_cam])
    pts_world = (cam_mat @ pts_cam.T).T + cam_pos
    # Footprint from the object's TOP surface layer, not the full mask. For a tall
    # object the camera sees the top AND a side face; the side backprojects into
    # the XY footprint and stretches one axis, inflating the AR (a cube read ~1.4
    # full-mask vs ~1.0 from its top layer -> spurious yaw on symmetric cubes).
    # Taking the top band gives the true top-down footprint: cubes ~1.0, 2:1
    # blocks ~2.0. Tunable via SPARK_OBB_TOP_BAND_M; falls back to the full mask
    # if the top band is too sparse.
    z_all = pts_world[:, 2]
    top_band = float(os.environ.get("SPARK_OBB_TOP_BAND_M", "0.008"))
    full_xy = pts_world[:, :2]
    _sel = z_all >= (float(z_all.max()) - top_band)
    band_ok = int(_sel.sum()) >= min_pixels
    if band_ok:
        band_xy = pts_world[_sel, :2]
        # A band whose XY spread EXCEEDS the full mask's is not a top surface:
        # it is depth noise smeared across the footprint. Reject it.
        band_spread = float(np.hypot(*band_xy.std(axis=0)))
        full_spread = float(np.hypot(*full_xy.std(axis=0)))
        if full_spread > 1e-9 and band_spread > 1.5 * full_spread:
            logger.debug(
                "[obb] top band spread %.3f > 1.5x full %.3f; using full mask",
                band_spread, full_spread,
            )
            band_ok = False
    xy = pts_world[_sel, :2] if band_ok else full_xy
    mean_xy = xy.mean(axis=0)
    cov = np.cov((xy - mean_xy).T)
    if cov.ndim != 2 or cov.shape != (2, 2):
        return _fail
    eigvals, eigvecs = np.linalg.eigh(cov)
    mj, mn = eigvecs[:, 1], eigvecs[:, 0]
    proj_mj = (xy - mean_xy) @ mj
    proj_mn = (xy - mean_xy) @ mn
    mj_len_m = _trimmed_span(proj_mj)
    mn_len_m = _trimmed_span(proj_mn)
    if mj_len_m <= 0 or mn_len_m <= 0:
        return _fail
    # Trimming can reorder the two axes on a near-square footprint; keep
    # "major" meaning the longer one (same guard as resolve_slot_direction).
    if mj_len_m < mn_len_m:
        mj, mn = mn, mj
        mj_len_m, mn_len_m = mn_len_m, mj_len_m
    orient = float(np.arctan2(mj[1], mj[0]))
    if not return_confidence:
        return orient, mj_len_m / mn_len_m, mj_len_m, mn_len_m

    # Confidence has two independent factors, both needed:
    #   band quality   - a flat, well-populated top band means the footprint
    #                    came from a real top surface, not smeared depth;
    #   axis definition - PCA on a near-circular footprint returns an ARBITRARY
    #                    major axis (eigenvalues tie), so a round dome must
    #                    score low however clean its depth is.
    band_z = pts_world[_sel, 2] if band_ok else z_all
    z_std = float(np.std(band_z))
    n_band = int(_sel.sum()) if band_ok else int(len(z_all))
    band_q = float(
        np.clip(1.0 - z_std / max(top_band, 1e-6), 0.0, 1.0)
        * min(1.0, n_band / 200.0)
    )
    ar = mj_len_m / mn_len_m
    axis_q = float(np.clip((ar - 1.0) / (_AXIS_CONF_AR_REF - 1.0), 0.0, 1.0))
    return orient, ar, mj_len_m, mn_len_m, band_q * axis_q


def _pca_obb(mask: np.ndarray):
    """
    Return (img_angle_rad, aspect_ratio, short_px, long_px) via PCA.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < 5:
        return 0.0, 1.0, 0.0, 0.0
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    mean = pts.mean(axis=0)
    cov = np.cov(pts - mean, rowvar=False)
    if cov.ndim == 0 or cov.shape != (2, 2):
        return 0.0, 1.0, 0.0, 0.0
    eigvals, eigvecs = np.linalg.eigh(cov)
    major_dir, minor_dir = eigvecs[:, 1], eigvecs[:, 0]
    centered = pts - mean
    long_px = float((centered @ major_dir).max() - (centered @ major_dir).min())
    short_px = float((centered @ minor_dir).max() - (centered @ minor_dir).min())
    if short_px <= 0:
        return float(np.arctan2(major_dir[1], major_dir[0])), 1.0, 0.0, long_px
    return (
        float(np.arctan2(major_dir[1], major_dir[0])),
        long_px / short_px,
        short_px,
        long_px,
    )


# heavy_end_sign: below this mass imbalance between the two axis halves the
# shape is called symmetric and the sign abstains (0.0). Sized against noise:
# a uniform rectangle of N pixels has imbalance sd ~ 1/sqrt(N) (~0.014 at
# N=5000), while a synthetic screwdriver silhouette (handle 2.5x the shaft
# width over ~40% of the length) measures ~0.3. See
# tests/test_mask_direction.py.


def _ray_intersect_table(u, v, f, cx_img, cy_img, cam_mat, cam_pos, z_plane, use_opencv=False):
    """
    Intersect camera ray through pixel (u,v) with world plane z=z_plane.
    """
    if use_opencv:
        dir_cam = np.array([(u - cx_img) / f, (v - cy_img) / f, 1.0])
    else:
        dir_cam = np.array([(u - cx_img) / f, -(v - cy_img) / f, -1.0])
    dir_world = cam_mat @ dir_cam
    denom = dir_world[2]
    if abs(denom) < 1e-9:
        return None
    t = (z_plane - cam_pos[2]) / denom
    if t <= 0:
        return None
    return cam_pos + t * dir_world


def _compute_orientation(
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
    long_px,
    metric_depth,
    use_opencv=False,
    fx_b=None,
    fy_b=None,
    cx_b=None,
    cy_b=None,
):
    """
    Compute world-frame orientation via ray-plane intersection.
    """
    if long_px <= 0 or metric_depth <= 0.01 or cam_mat is None or cam_pos is None:
        return 0.0
    du = np.cos(img_angle_rad)
    dv = np.sin(img_angle_rad)
    if fx_b is None:
        fx_b = h / (2 * np.tan(np.deg2rad(cam_fovy) / 2)) if cam_fovy else float(h)
    if fy_b is None:
        fy_b = fx_b
    if cx_b is None:
        cx_b = w / 2.0
    if cy_b is None:
        cy_b = h / 2.0
    half_step = float(long_px) * 0.4

    # Get table_z from centroid backprojection
    if use_opencv:
        _cx_c = (cx - cx_b) * metric_depth / fx_b
        _cy_c = (cy - cy_b) * metric_depth / fy_b
        _cz_c = metric_depth
    else:
        _cx_c = (cx - cx_b) * metric_depth / fx_b
        _cy_c = -(cy - cy_b) * metric_depth / fy_b
        _cz_c = -metric_depth
    centroid_world = cam_mat @ np.array([_cx_c, _cy_c, _cz_c]) + cam_pos
    table_z = float(centroid_world[2])

    p1 = _ray_intersect_table(
        cx + half_step * du,
        cy + half_step * dv,
        fx_b,
        cx_b,
        cy_b,
        cam_mat,
        cam_pos,
        table_z,
        use_opencv,
    )
    p2 = _ray_intersect_table(
        cx - half_step * du,
        cy - half_step * dv,
        fx_b,
        cx_b,
        cy_b,
        cam_mat,
        cam_pos,
        table_z,
        use_opencv,
    )
    if p1 is not None and p2 is not None:
        dir_xy = p1[:2] - p2[:2]
        return float(np.arctan2(dir_xy[1], dir_xy[0]))
    return 0.0


def _erode_mask(mask, iterations):
    """Binary erosion by `iterations` steps of a 3x3 cross.

    Shrinking the mask before sampling colour matters: a SAM3 mask tracks
    the object silhouette, so its outermost ring of pixels is antialiased
    against the background and drags the median hue toward the table.
    """
    out = np.asarray(mask, dtype=bool)
    for _ in range(max(0, int(iterations))):
        shrunk = out.copy()
        shrunk[1:, :] &= out[:-1, :]
        shrunk[:-1, :] &= out[1:, :]
        shrunk[:, 1:] &= out[:, :-1]
        shrunk[:, :-1] &= out[:, 1:]
        shrunk[0, :] = False
        shrunk[-1, :] = False
        shrunk[:, 0] = False
        shrunk[:, -1] = False
        if not shrunk.any():
            return out  # eroding further would erase a thin object entirely
        out = shrunk
    return out


def mask_color_stats(rgb, mask, erode_px=3, max_samples=20000):
    """Median HSV of the pixels under `mask`, plus a hue-concentration score.

    Returns ``(hue_deg, saturation, value, hue_conf)`` or ``None`` when the
    mask is empty. Hue is degrees in [0, 360); saturation and value are on
    the 0-255 scale OpenCV uses, so thresholds are comparable with the rest
    of the codebase. ``hue_conf`` is the resultant length of the
    saturation-weighted circular mean, in [0, 1]: near 1 for a uniformly
    coloured object, near 0 for a grey or multicoloured one.

    Hue is averaged circularly rather than with a plain median because hue
    wraps -- red pixels straddling 0/360 would otherwise average to cyan.
    Weighting by saturation keeps near-grey pixels, whose hue is numerically
    unstable, from steering the result.

    Used only by the optional ``disambiguate.hsv_cluster`` path in the task
    prompt registry; the shipped task set resolves on counts and geometry
    alone, so this stays off by default.
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim > 2:
        m = m.squeeze()
    if not m.any():
        return None
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] < 3 or arr.shape[:2] != m.shape:
        return None

    eroded = _erode_mask(m, erode_px)
    px = arr[eroded][:, :3].astype(np.float64)
    if px.size == 0:
        return None
    if px.shape[0] > max_samples:
        # Deterministic stride, not a random sample: the same mask must give
        # the same colour on every run or instance ordering stops being stable.
        px = px[:: max(1, px.shape[0] // max_samples)]

    px /= 255.0
    mx = px.max(axis=1)
    mn = px.min(axis=1)
    delta = mx - mn

    sat = np.where(mx > 0, delta / np.maximum(mx, 1e-12), 0.0)

    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    hue = np.zeros_like(mx)
    safe = delta > 1e-12
    idx = safe & (mx == r)
    hue[idx] = ((g[idx] - b[idx]) / delta[idx]) % 6.0
    idx = safe & (mx == g) & (mx != r)
    hue[idx] = (b[idx] - r[idx]) / delta[idx] + 2.0
    idx = safe & (mx == b) & (mx != r) & (mx != g)
    hue[idx] = (r[idx] - g[idx]) / delta[idx] + 4.0
    hue *= 60.0

    rad = np.deg2rad(hue)
    wsum = float(sat.sum())
    if wsum > 1e-9:
        cx = float((sat * np.cos(rad)).sum()) / wsum
        cy = float((sat * np.sin(rad)).sum()) / wsum
        hue_deg = float(np.rad2deg(np.arctan2(cy, cx)) % 360.0)
        hue_conf = float(min(1.0, np.hypot(cx, cy)))
    else:
        hue_deg = 0.0
        hue_conf = 0.0

    return (
        hue_deg,
        float(np.median(sat) * 255.0),
        float(np.median(mx) * 255.0),
        hue_conf,
    )


def mask_height_profile(
    mask,
    depth,
    cam_pos,
    cam_mat,
    fx=None,
    fy=None,
    cx_k=None,
    cy_k=None,
    fovy_deg=None,
    use_opencv=False,
    min_pixels=_HEIGHT_PROFILE_MIN_PIXELS,
    min_depth=0.01,
    max_depth=10.0,
):
    """World-Z extremes of a mask's depth cloud: the TOP and BOTTOM surfaces.

    Returns ``{"rim_z", "interior_z", "n_valid"}`` or ``None`` when the mask
    carries too little valid depth to say anything. Nothing is invented: no
    depth means no answer, and the caller falls back.

    For a CONTAINER these two numbers are its rim and its interior floor. A
    bowl's mask is bimodal in world Z -- the rim ring projects high, the
    interior floor low -- which is the same bimodality ``_cloud_median_world``
    already exploits when it reports the 95th percentile as the object top.
    For any other object they are simply its top and its bottom, which is what
    the release-height computation needs for the object being CARRIED (see
    ``control.release_height``).

    Percentiles, not min/max: a handful of dropout or edge-bleed pixels sets
    the extremes and would report a rim that does not exist. The interior uses
    a deliberately HIGH percentile (_INTERIOR_PCT, 20th) rather than a
    symmetric one, because an interior read too LOW lowers the release, and
    the safe error here is to read it high.
    """
    if mask is None or depth is None or cam_mat is None or cam_pos is None:
        return None
    ys, xs = np.where(mask > 0)
    if len(xs) < min_pixels:
        return None
    depths = depth[ys, xs].astype(np.float64)
    valid = (depths > min_depth) & (depths < max_depth)
    n_valid = int(valid.sum())
    if n_valid < min_pixels:
        return None
    xs_v = xs[valid].astype(np.float64)
    ys_v = ys[valid].astype(np.float64)
    d_v = depths[valid]

    h, w = mask.shape
    if use_opencv:
        if fx is None or fy is None or cx_k is None or cy_k is None:
            return None
        x_cam = (xs_v - cx_k) * d_v / fx
        y_cam = (ys_v - cy_k) * d_v / fy
        z_cam = d_v
    else:
        if fovy_deg is None:
            return None
        f = h / (2 * np.tan(np.deg2rad(fovy_deg) / 2))
        x_cam = (xs_v - w / 2) * d_v / f
        y_cam = -(ys_v - h / 2) * d_v / f
        z_cam = -d_v
    pts_world = (cam_mat @ np.column_stack([x_cam, y_cam, z_cam]).T).T + cam_pos
    wz = pts_world[:, 2]
    rim_z, interior_z = np.percentile(wz, [_RIM_PCT, _INTERIOR_PCT])
    return {
        "rim_z": float(rim_z),
        "interior_z": float(min(interior_z, rim_z)),
        "n_valid": n_valid,
    }
