"""Mask shape features used only by the real-robot executor: which end of an elongated mask is the
heavy end, the rotation that registers a held mask onto a target cut-out, and the direction a
placed utensil must lie along in a slotted tray."""

import logging

import cv2
import numpy as np

from spark_core.mask_obb import _pca_obb
from spark_core.util import det_field
from spark_real.perception.slot_detector import SLOT_AXIS_MIN_AR, slot_axis_consensus

logger = logging.getLogger(__name__)

# Below this a tray's world OBB axis is too close to square to name its long side.
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


def resolve_slot_direction(det, container_label="", slot_idx=None):
    """World angle (rad) a placed utensil must lie ALONG, and its source.

    Shared by place_in_slot and the generic place path. `det` may be a dict
    (detection_map) or an ObjectDetection.

      1. This slot's own floor, when it is elongated (mixed-compartment trays).
      2. The consensus of all measured slot floors.
      3. No floor measured (dividers below the depth resolution): the
         container's long axis. A tray whose slots run across its short side
         needs the planner's grasp_yaw_deg here.
    """
    # ponytail: step 3 assumes slots along the long side; read dividers from RGB if a rig needs otherwise.
    measured = [s for s in (det_field(det, "slots", None) or []) if s.get("mode") == "normals"]
    for s in measured:
        if s.get("slot_idx") == slot_idx and s.get("axis_ar", 0.0) >= SLOT_AXIS_MIN_AR:
            return float(s["axis_rad"]), f"slot_{slot_idx}_floor(AR={s['axis_ar']:.2f})"
    axis = slot_axis_consensus(measured)
    if axis is not None:
        return axis, f"slot_floors(n={len(measured)})"

    world_major = det_field(det, "world_major_axis_rad", None)
    if world_major is None:
        world_major = det_field(det, "orientation_angle", None)
    tray_ar = float(det_field(det, "aspect_ratio", 1.0) or 1.0)
    if world_major is not None and 1e-6 < tray_ar < 1.0:  # the stored "major" is the short axis
        world_major, tray_ar = float(world_major) + np.pi / 2.0, 1.0 / tray_ar
    if world_major is not None and tray_ar >= TRAY_OBB_AR_TRUST:
        return float(world_major), f"container_long_axis(AR={tray_ar:.2f})"
    plane_axis = det_field(det, "plane_orientation_angle", None)
    if plane_axis is not None:
        return float(plane_axis), "container_long_axis(ray_plane)"
    logger.warning(
        "[slot_direction] no reliable direction for '%s' (AR %.2f); placement yaw may be wrong",
        container_label,
        tray_ar,
    )
    return float(world_major or 0.0), "container_obb(low_confidence)"


