"""Mask shape features used only by the real-robot executor: which end of an elongated mask is the
heavy end, the rotation that registers a held mask onto a target cut-out, and the direction a
placed utensil must lie along in a slotted tray."""

import logging

import cv2
import numpy as np

from spark_core.mask_obb import _pca_obb
from spark_core.util import det_field

logger = logging.getLogger(__name__)

# A near-square world-XY OBB (aspect_ratio below this) has an untrustworthy
# major axis: PCA can swap the major/minor eigenvectors when the two spreads
# are close, so the reported world OBB angle may actually be the SHORT axis.
# For the 3-D cutlery tray this fired (world PCA reported ~103 deg / AR~1.3-1.8,
# the tray's short axis mislabelled as major). Below this threshold do NOT
# trust the world OBB major; derive slot direction from a more reliable source.
TRAY_OBB_AR_TRUST = 1.5

_HEAVY_END_IMBALANCE_GATE = 0.08
_HEAVY_END_MIN_PIXELS = 50


def _largest_component(mask8: np.ndarray) -> np.ndarray:
    """The largest 8-connected component of a binary mask.

    SAM3 masks fragment, and satellite blobs drag the extent midpoint of a
    DIRECTED measurement toward a confidently wrong sign. The body of the
    object is the largest component; measure that.
    """
    n, lab = cv2.connectedComponents(mask8, connectivity=8)
    if n <= 2:  # background + at most one component
        return mask8
    counts = np.bincount(lab.ravel())
    counts[0] = 0  # background
    return (lab == int(np.argmax(counts))).astype(np.uint8)


def heavy_end_sign(mask: np.ndarray, axis_rad: float) -> float:
    """Which END of `axis_rad` the mask's fat half lies toward: +1.0 / -1.0.

    A PCA axis is a LINE -- identical under a 180 deg flip -- so it cannot say
    which end of a screwdriver is the tip. This breaks that tie with the only
    directed signal a silhouette has: one end being fatter than the other.
    Pixels are projected onto the axis direction and split at the midpoint of
    the (percentile-trimmed) extent; whichever half holds more mask pixels is
    the heavy end. +1.0 means it lies toward +(cos axis, sin axis) in image
    coords (x right, y down), -1.0 the opposite end.

    Returns 0.0 -- "too symmetric to call" -- below _HEAVY_END_IMBALANCE_GATE,
    and every consumer must treat 0.0 as an abstention, never a direction. A
    bare handle mask is symmetric end-for-end, so the direction must be
    measured on the WHOLE tool (see _record_held_axis's parent-label upgrade).
    """
    m8 = _largest_component((np.asarray(mask) > 0).astype(np.uint8))
    ys, xs = np.where(m8 > 0)
    if len(xs) < _HEAVY_END_MIN_PIXELS:
        return 0.0
    # The caller's axis usually comes from _pca_obb over the FULL mask. When
    # fragments dragged that axis away from the body's own, a sign measured
    # along it describes the fragments, not the object. If the two axes
    # disagree by more than 45 deg the premise is broken: abstain.
    ang_cc, _, _, _ = _pca_obb(m8)
    d_ax = abs(ang_cc - float(axis_rad)) % np.pi
    if min(d_ax, np.pi - d_ax) > np.pi / 4:
        return 0.0
    proj = xs * float(np.cos(axis_rad)) + ys * float(np.sin(axis_rad))
    # Trim the extent, not the mass: a few speck pixels must not drag the
    # midpoint, but every real pixel still votes on which half is heavier.
    lo, hi = np.percentile(proj, [2.0, 98.0])
    if hi - lo < 1e-6:
        return 0.0
    mid = 0.5 * (lo + hi)
    n_pos = int(np.count_nonzero(proj > mid))
    n_neg = int(np.count_nonzero(proj < mid))
    total = n_pos + n_neg
    if total == 0:
        return 0.0
    imbalance = (n_pos - n_neg) / float(total)
    if abs(imbalance) < _HEAVY_END_IMBALANCE_GATE:
        return 0.0
    return 1.0 if imbalance > 0 else -1.0


# best_fit_rotation resamples both masks onto this square canvas. 192 px keeps
# a full 360-sweep of warps under ~10 ms while leaving a ~2 px quantization,
# far under the 90-deg-class margins the consumers gate on.
_FIT_CANVAS_PX = 192
_FIT_STEP_DEG = 3.0
_FIT_MIN_PIXELS = 50


def best_fit_rotation(held_mask: np.ndarray, target_mask: np.ndarray):
    """Image rotation that best registers `held_mask` onto `target_mask`.

    Returns (angle_rad, best_iou, margin):
      angle_rad  signed image-frame rotation in (-pi, pi]. Rotating the held
                 silhouette by +angle (the same convention as _pca_obb angles:
                 atan2 in image coords, x right, y down) best overlays it on
                 the target, i.e. angle ~= target_axis - held_axis when the
                 shapes are elongated.
      best_iou   IoU at that rotation, both masks centroid-centered and drawn
                 at a SHARED scale -- relative size is real signal (a cutout
                 slightly larger than its tool still overlaps well; a bowl
                 twice its size does not).
      margin     best_iou minus the best IoU among rotations >= 90 deg away.
                 This is the DIRECTED part of the answer: for a directional
                 shape in a directional recess the 180-off candidate scores
                 visibly worse and the margin is large; for a round or
                 twofold-symmetric pair every rotation ties and the margin
                 collapses toward 0 -- which is the honest "shape does not
                 decide" answer the JIGSAW_MIN_MARGIN gate turns into an
                 abstention (executor_motion, gate 0.05).

    (0.0, 0.0, 0.0) when either mask is degenerate (< _FIT_MIN_PIXELS).
    """
    # Largest component only, for the same reason as heavy_end_sign: a
    # satellite blob shifts the centroid and radius, and the registration then
    # scores rotations of the wrong center.
    h8 = _largest_component((np.asarray(held_mask) > 0).astype(np.uint8))
    t8 = _largest_component((np.asarray(target_mask) > 0).astype(np.uint8))
    if int(h8.sum()) < _FIT_MIN_PIXELS or int(t8.sum()) < _FIT_MIN_PIXELS:
        return 0.0, 0.0, 0.0

    c = _FIT_CANVAS_PX
    ctr = (c - 1) / 2.0

    def _centroid_and_radius(m):
        ys, xs = np.where(m > 0)
        cx, cy = float(xs.mean()), float(ys.mean())
        r = float(np.sqrt(((xs - cx) ** 2 + (ys - cy) ** 2).max()))
        return cx, cy, max(r, 1.0)

    hx, hy, hr = _centroid_and_radius(h8)
    tx, ty, tr = _centroid_and_radius(t8)
    # ONE scale for both, sized so the larger just fits under every rotation.
    s = (c / 2.0 - 2.0) / max(hr, tr)

    def _to_canvas(m, cx, cy):
        M = np.array([[s, 0.0, ctr - s * cx], [0.0, s, ctr - s * cy]])
        return cv2.warpAffine(m, M, (c, c), flags=cv2.INTER_NEAREST)

    held_c = _to_canvas(h8, hx, hy)
    tgt_c = _to_canvas(t8, tx, ty).astype(bool)
    tgt_area = int(tgt_c.sum())

    angles = np.deg2rad(np.arange(0.0, 360.0, _FIT_STEP_DEG))
    ious = np.zeros(len(angles))
    for i, a in enumerate(angles):
        ca, sa = float(np.cos(a)), float(np.sin(a))
        # Forward map [[c,-s],[s,c]] about the canvas center: a pixel at image
        # angle b moves to b+a, matching _pca_obb's atan2 convention, so the
        # returned rotation composes directly with axis angles.
        M = np.array(
            [[ca, -sa, ctr - ca * ctr + sa * ctr],
             [sa, ca, ctr - sa * ctr - ca * ctr]]
        )
        rot = cv2.warpAffine(held_c, M, (c, c), flags=cv2.INTER_NEAREST).astype(bool)
        inter = int(np.count_nonzero(rot & tgt_c))
        union = int(rot.sum()) + tgt_area - inter
        ious[i] = inter / union if union > 0 else 0.0

    best = int(np.argmax(ious))
    best_iou = float(ious[best])
    # Circular distance to the winner; rivals live >= 90 deg away.
    d = np.abs(np.rad2deg(angles) - np.rad2deg(angles[best]))
    d = np.minimum(d, 360.0 - d)
    far = d >= 90.0
    rival = float(ious[far].max()) if far.any() else 0.0
    ang = float((angles[best] + np.pi) % (2 * np.pi) - np.pi)
    return ang, best_iou, best_iou - rival


def _pca_major_angle(points_xy):
    """World-frame angle (rad) of the dominant-spread axis of 2D points.

    Used on detected slot CENTROIDS: parallel-compartment trays array their
    slots along the tray's LONG axis, so the centroids' principal axis IS the
    tray long axis. Returns None if fewer than 2 distinct points.
    """
    pts = np.asarray(points_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return None
    mean = pts.mean(axis=0)
    centered = pts - mean
    if float(np.max(np.linalg.norm(centered, axis=1))) < 1e-4:
        return None  # all points coincident
    cov = np.cov(centered, rowvar=False)
    if np.asarray(cov).shape != (2, 2):
        return None
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, 1]  # largest eigenvalue = dominant spread
    return float(np.arctan2(major[1], major[0]))


def resolve_slot_direction(det, container_label=""):
    """World angle (rad) a placed utensil must lie ALONG for a slotted container.

    Returns (slot_direction_rad, source_str). Shared by place_in_slot and the
    generic-place path (executor_motion._transport_to) so BOTH align to the
    slots, not the axis-swapped container OBB whose major axis is perpendicular
    to the slots. `det` may be a dict (detection_map) or an ObjectDetection.

    A parallel-compartment tray arrays its slots along the tray's LONG (major)
    axis, and each slot (hence each utensil dropped into it) is elongated
    PERPENDICULAR to that array, along the tray's SHORT (minor) axis. So:
    slot_direction = tray_long_axis + 90 deg (hardware-validated on the real
    tray).

    The tray_long_axis must come from a source that CANNOT axis-swap:
      1. Detected slot CENTROIDS (world frame, slot_detector "normals" mode):
         their principal axis is the true array = tray long axis. Swap-immune
         (real world positions, not the near-square tray OBB). PREFERRED.
      2. The mask's minimum-area rectangle: a rectangle's angle is carried by
         its outline, so no elongation gate is needed.
      3. World OBB major axis, ONLY when aspect_ratio >= TRAY_OBB_AR_TRUST;
         a near-square tray's world OBB may report its SHORT axis as "major".
      4. Dense image-frame mask minAreaRect / PCA fallback (top-down birdview
         image angle maps to world).
      5. No reliable source: keep the tray-OBB perpendicular but WARN.
    """
    # 1. Slot centroids (world frame), real (normals) detections only.
    _slots = det_field(det, "slots", None)
    if _slots:
        centroids = []
        has_real = False
        for s in _slots:
            w = s.get("world_xyz") if hasattr(s, "get") else getattr(s, "world_xyz", None)
            mode = s.get("mode", "") if hasattr(s, "get") else getattr(s, "mode", "")
            if w is None:
                continue
            centroids.append(np.asarray(w, dtype=float)[:2])
            if str(mode) == "normals":
                has_real = True
        if has_real and len(centroids) >= 2:
            array_axis = _pca_major_angle(centroids)
            if array_axis is not None:
                return (
                    array_axis + np.pi / 2.0,
                    f"slot_centroids(n={len(centroids)})",
                )

    # 2. The mask's MINIMUM-AREA RECTANGLE. A tray is a rectangle, so its
    # direction lives in its OUTLINE and needs no elongation gate; the
    # mass-distribution estimators below degrade as a shape approaches square.
    # PCA is pulled by every mask irregularity (a lip on one edge, a shadow,
    # the utensils already in the tray): on the real tray it reported image AR
    # 1.36 then 1.46 against a measured world extent of 0.348 x 0.217 (=1.60)
    # and swung 25 deg between runs (-78.3 deg placed the spoon correctly,
    # -53.1 deg laid it across its slots). minAreaRect keys off the four sides.
    _m = det_field(det, "_mask", None)
    if _m is None:
        _m = det_field(det, "mask", None)
    if _m is not None:
        try:
            _m8 = (np.asarray(_m) > 0).astype(np.uint8)
            _cnts, _ = cv2.findContours(
                _m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if _cnts:
                (_cx, _cy), (_w, _h), _deg = cv2.minAreaRect(
                    max(_cnts, key=cv2.contourArea)
                )
                # minAreaRect's angle names the WIDTH side; take the long one.
                _long = _deg if _w >= _h else _deg + 90.0
                _ar = max(_w, _h) / max(min(_w, _h), 1e-6)
                return (
                    float(np.deg2rad(_long)) + np.pi / 2.0,
                    f"image_minrect_perp(AR={_ar:.2f})",
                )
        except Exception as exc:  # noqa: BLE001 - fall through to the old rungs
            logger.warning("[slot_direction] minAreaRect failed: %s", exc)

    # 3. World OBB major axis, only when AR is trustworthy.
    tray_ar = float(det_field(det, "aspect_ratio", 1.0) or 1.0)
    if 1e-6 < tray_ar < 1.0:
        tray_ar = 1.0 / tray_ar
    world_major = det_field(det, "world_major_axis_rad", None)
    if world_major is None:
        world_major = det_field(det, "orientation_angle", None)
    if world_major is not None and tray_ar >= TRAY_OBB_AR_TRUST:
        return (
            float(world_major) + np.pi / 2.0,
            f"world_obb_perp(AR={tray_ar:.2f})",
        )

    # 4. Dense image-frame mask PCA (kept as a fallback).
    mask = det_field(det, "_mask", None)
    if mask is None:
        mask = det_field(det, "mask", None)
    if mask is not None:
        try:
            # minAreaRect, not PCA: a rectangle's angle is carried by its
            # OUTLINE, not its mass distribution (see rung 2).
            m8 = (np.asarray(mask) > 0).astype(np.uint8)
            cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                (_cx, _cy), (_w, _h), _deg = cv2.minAreaRect(
                    max(cnts, key=cv2.contourArea)
                )
                # minAreaRect's angle names the width side; take the LONG side.
                long_deg = _deg if _w >= _h else _deg + 90.0
                rect_ar = (max(_w, _h) / max(min(_w, _h), 1e-6))
                return (
                    float(np.deg2rad(long_deg)) + np.pi / 2.0,
                    f"image_minrect_perp(AR={rect_ar:.2f})",
                )
            img_major, img_ar, _, _ = _pca_obb(np.asarray(mask))
            return (
                float(img_major) + np.pi / 2.0,
                f"image_pca_perp(AR={img_ar:.2f})",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[slot_direction] image-PCA slot axis failed: %s", exc)

    # 5. No reliable source: tray-OBB perpendicular per the geometry above, WARN.
    logger.warning(
        "[slot_direction] no reliable slot-direction source for '%s' "
        "(no real slots, world OBB AR<%.2f, no mask); using tray-OBB "
        "perpendicular; placement yaw may be unreliable",
        container_label,
        TRAY_OBB_AR_TRUST,
    )
    fallback = float(world_major) if world_major is not None else 0.0
    return fallback + np.pi / 2.0, "tray_obb_perp(fallback)"
