"""Drawer geometry from perception for ``open_drawer``: the requested level, handle clusters per level,
the level's height band on the cabinet face, the pull direction, and the camera check of how far a
drawer came out.
"""
from __future__ import annotations

from typing import Optional

import mujoco
import numpy as np

from spark_bench.libero_pro.perception import _drawer_part_of_cabinet, redetect_agentview
from spark_bench.libero_pro.primitives.keypoint import _top_surface

_LEVELS = {
    'top': ('top', 'upper'),
    'middle': ('middle', 'center'),
    'bottom': ('bottom', 'lower'),
}


def level_from(params: dict, instruction: str) -> Optional[str]:
    text = ' '.join([str(params.get('level', '') or ''),
                     str(params.get('joint_name', '') or ''),
                     str(params.get('keypoint_label', '') or ''),
                     instruction or '']).lower()
    for lvl, kws in _LEVELS.items():
        if any(k in text for k in kws):
            return lvl
    return None


def bar_top_z(det, pos: np.ndarray) -> float:
    """
    World Z of the handle bar's TOP surface, not its cloud centre.

    ``position_3d`` is the mask's representative point, whose vertical
    meaning varies by perception lineage (near surface vs cloud median; a
    ~1 cm bias on a ~2 cm bar), and the descend gate below is a 1.2 cm
    window, so the gate has to be referenced to a surface.

    ``ObjectDetection.rim_z_m`` (95th percentile of the mask's world Z) is
    preferred when present.  Falls back to ``position_3d[2]`` (never BELOW
    it) when it is absent.
    """
    z = float(pos[2])
    rim = getattr(det, 'rim_z_m', None)
    if rim is None:
        return z
    try:
        rim = float(rim)
    except (TypeError, ValueError):
        return z
    # A rim below the reported centre means the profile latched something
    # other than this bar - keep the centre.
    return max(z, rim)


def in_reach(executor, det_map: dict, reach: float = 0.85) -> dict:
    """Detections within the arm's horizontal reach of its own base (proprioception, not scene state). A
    LIBERO-90 kitchen has other handled furniture 1.2 m away; its bars became the 'top' drawer."""
    try:
        b = mujoco.mj_name2id(executor.model, mujoco.mjtObj.mjOBJ_BODY, 'robot0_base')
        base = np.asarray(executor.data.xpos[b], float)[:2]
    except Exception:
        return det_map
    return {k: d for k, d in det_map.items() if getattr(d, 'position_3d', None) is None
            or float(np.linalg.norm(np.asarray(d.position_3d, float)[:2] - base)) <= reach}


def cluster_handles(det_map: dict) -> list[tuple[str, np.ndarray, float, float]]:
    """
    One representative (label, pos, conf, top_z) per drawer level,
    sorted top-down.

    SAM3 returns many overlapping low-confidence masks per real handle;
    cluster by Z (+-3.5 cm) keeping the highest-confidence member.
    ``top_z`` is the bar's top surface (see :func:`bar_top_z`); the
    clustering itself still runs on ``position_3d`` so level separation is
    unaffected.
    """
    # A handle is part of a cabinet: drop 'handle' masks more than 30 cm (xy) from every cabinet detection
    # (a stray one 1.4 m away became the 'bottom' bar on a LIBERO-90 kitchen scene).
    cabs = [np.asarray(d.position_3d, float)[:2] for k, d in det_map.items()
            if 'cabinet' in k.lower() and getattr(d, 'position_3d', None) is not None]
    cands = []
    for label, det in det_map.items():
        if getattr(det, 'position_3d', None) is None:
            continue
        low = label.lower()
        if 'handle' not in low:
            continue
        if cabs and min(np.linalg.norm(np.asarray(det.position_3d, float)[:2] - c) for c in cabs) > 0.30:
            continue
        conf = float(getattr(det, 'confidence', 0.0) or 0.0)
        cands.append((label, det.position_3d.copy(), conf,
                      bar_top_z(det, det.position_3d)))
    clusters: list[dict] = []
    for label, pos, conf, top_z in sorted(cands, key=lambda c: -c[2]):
        placed = False
        for cl in clusters:
            if abs(cl['pos'][2] - pos[2]) < 0.035:
                placed = True
                break
        if not placed:
            clusters.append({'label': label, 'pos': pos, 'conf': conf,
                             'top_z': top_z})
    out = [(c['label'], c['pos'], c['conf'], c['top_z']) for c in clusters]
    out.sort(key=lambda c: -c[1][2])
    return out


def level_band_z(executor, level, clusters) -> Optional[float]:
    """Height of the named drawer level from the cabinet's own mask. The drawers divide the face into equal
    bands, so the top bar sits half a band below the cabinet top: band = 2 * (top - top bar). The top edge
    is reliable; the bottom often is not (in a swapped layout the lower third was hidden). Without a bar in
    the top band, equal thirds of the visible face. None without a level, a cabinet mask, or depth."""
    if level not in ('top', 'middle', 'bottom'):
        return None
    sam3, depth = getattr(executor, 'sam3', None), getattr(executor, 'depth', None)
    cab = [d for k, d in in_reach(executor, executor.det_map).items()
           if 'cabinet' in k.lower() and getattr(d, 'mask', None) is not None]
    if sam3 is None or depth is None or not cab:
        return None
    try:
        pts = sam3._backproject_mask(cab[0].mask, depth, executor.cam_pos, executor.cam_mat,
                                     executor.cam_fovy, executor.cam_w, executor.cam_h)
    except Exception:
        return None
    if pts is None or len(pts) < 100:
        return None
    lo, hi = float(np.percentile(pts[:, 2], 2)), float(np.percentile(pts[:, 2], 98))
    n = max(3, len(clusters))
    k = {'top': 0, 'middle': n // 2, 'bottom': n - 1}[level]
    band = (hi - lo) / n
    tops = [float(c[1][2]) for c in clusters if 0.0 < hi - float(c[1][2]) < band]
    if tops:
        band = 2.0 * (hi - max(tops))
        return max(tops) - k * band
    return hi - (k + 0.5) * band


def handle_advance(executor, handle_pos, pull) -> Optional[float]:
    """Camera check of how far the drawer at this level has come out: the largest advance along the pull
    of a handle at the handle's height. None when it cannot be checked (no SAM3, or no handle seen)."""
    if getattr(executor, 'sam3', None) is None:
        return None
    try:
        fresh = redetect_agentview(executor.env, executor.sam3, ['drawer handle'], executor.cfg,
                                   all_instances=True, render_only=True)
    except Exception:
        return None
    dets = [d for d in (getattr(fresh, 'dets', None) or []) if getattr(d, 'position_3d', None) is not None]
    if not dets:
        return None
    merged = {k: v for k, v in executor.det_map.items() if 'cabinet' in k.lower()}
    merged.update({f'drawer handle {i}': d for i, d in enumerate(dets)})
    same = [c for c in cluster_handles(merged) if abs(float(c[1][2]) - float(handle_pos[2])) < 0.03]
    if not same:
        return None
    adv = max(float(np.dot((c[1] - handle_pos)[:2], pull[:2])) for c in same)
    print(f"[open_drawer] handle advanced {adv*1000:.0f}mm along the pull")
    return adv


def ensure_cabinet(executor) -> None:
    """The level band, the handle filter and the pull direction need the cabinet; the plan's prompts may
    not include it ('drawer handle', 'drawer' only). Detect it on the current frame when missing, then
    re-point a stray 'drawer' at it."""
    if any('cabinet' in k.lower() for k in in_reach(executor, executor.det_map)):
        return
    if getattr(executor, 'sam3', None) is None:
        return
    try:
        fresh = redetect_agentview(executor.env, executor.sam3, ['wooden cabinet', 'cabinet'], executor.cfg,
                                   render_only=True)
    except Exception:
        return
    if fresh is None:
        return
    for k, v in (fresh.det_map or {}).items():
        if 'cabinet' in k.lower() and getattr(v, 'position_3d', None) is not None:
            executor.det_map[k] = v
    _drawer_part_of_cabinet(executor.det_map)


def pull_dir(executor, handle_pos: np.ndarray) -> np.ndarray:
    """
    Horizontal pull direction (unit, z=0).
    """
    cab = None
    best = (-1, 0.0)
    for label, det in in_reach(executor, executor.det_map).items():
        low = label.lower()
        if 'handle' in low:
            continue
        if ('drawer' in low or 'cabinet' in low) and \
                getattr(det, 'position_3d', None) is not None:
            # a cabinet over a bare 'drawer' (SAM3's 'drawer' landed on the wine rack); its top-surface
            # centre over the mask centroid, which sits on the front face (the centroid-to-handle vector
            # came out sideways and the pull went along -x for a +y drawer)
            rank = (1 if 'cabinet' in low else 0, float(getattr(det, 'confidence', 0.0) or 0.0))
            if rank > best:
                top = _top_surface(executor, det) if 'cabinet' in low else None
                cab, best = (top if top is not None else det.position_3d), rank
    if cab is not None:
        v = handle_pos[:2] - cab[:2]
        n = float(np.linalg.norm(v))
        if n > 0.02:
            # Dominant-axis snap: drawers slide along one axis; the raw
            # cabinet-to-handle vector carries centroid noise.
            if abs(v[0]) >= abs(v[1]):
                return np.array([np.sign(v[0]) or 1.0, 0.0, 0.0])
            return np.array([0.0, np.sign(v[1]) or 1.0, 0.0])
    # Fallback: toward the table centre, dominant axis.
    v = -handle_pos[:2]
    if abs(v[0]) >= abs(v[1]):
        return np.array([np.sign(v[0]) or 1.0, 0.0, 0.0])
    return np.array([0.0, np.sign(v[1]) or 1.0, 0.0])

