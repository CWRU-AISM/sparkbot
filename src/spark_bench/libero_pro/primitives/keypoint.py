"""
``move_to_keypoint``: the pick strategies (bowl rim, plate rim, top-down) and the matching
place.

All entry points take ``(executor, params)``; helpers take the executor
as first arg and pull state (``holding``, ``last_pick_*``) from it.
"""
from __future__ import annotations

import copy
import os
from typing import Optional

import numpy as np

from spark_bench.libero_pro.primitives._common import (
    approach_descend,
    fuzzy_get_det,
    nudge_toward_base,
)
from spark_bench.libero_pro.motion import step_env

__all__ = ['handle']


_FLAT_PLACE_KW = ('plate', 'stove', 'cabinet', 'rack', 'dish')


def handle(executor, params: dict) -> None:
    """
    ``move_to_keypoint`` - pick or place depending on ``executor.holding``.
    """
    label = params.get('keypoint_label', '') or ''
    det = fuzzy_get_det(executor.det_map, label)
    if det is None or det.position_3d is None:
        return
    target = det.position_3d.copy()
    target += np.array([params.get('offset_x', 0),
                         params.get('offset_y', 0),
                         params.get('offset_z', 0)])

    lbl = label.lower()
    is_bowl = 'bowl' in lbl
    is_plate = ('plate' in lbl or 'dish' in lbl) and not is_bowl
    # Wipe targets (dirt/stain/marks) are not graspable: move to them with closed jaws, no pick.
    is_wipe_target = any(kw in lbl for kw in ('dirt', 'marks', 'stain', 'spill', 'smudge'))
    target[2] = max(target[2], 0.005)

    if executor.holding:
        top = (_top_surface(executor, det)
               if ' top' in f" {(getattr(executor, 'instruction', '') or '').lower()}" else None)
        if top is not None:
            # A tall place target (cabinet: mask spans its front face) has its centroid on the front, low;
            # releasing there dropped the wine bottle off the front edge (goal 2 position). Aim at the
            # centre of the surface it will rest on.
            target = top + np.array([params.get('offset_x', 0), params.get('offset_y', 0),
                                     params.get('offset_z', 0)])
        _do_place(executor, target, lbl)
    elif is_wipe_target:
        executor._move_to(target, False, steps=400)
    else:
        if PICK_REGROUND:
            shift = _reground_pick(executor, label, target)
            if shift is not None:
                target[:2] += shift
                det = _shifted(det, shift)
        _do_pick(executor, label, det, target, lbl, is_bowl, is_plate)


# Place


def _top_surface(executor, det) -> Optional[np.ndarray]:
    """Centre of the top surface of a tall (> 10 cm) place target from its mask's backprojected points:
    xy mean of the points within 2 cm of the 95th-percentile height, at that height. None when the target
    is short (plate, bowl, basket: the centroid already is the surface) or the mask is unavailable."""
    mask = getattr(det, 'mask', None)
    if executor.sam3 is None or executor.depth is None or mask is None:
        return None
    try:
        pts = executor.sam3._backproject_mask(mask, executor.depth, executor.cam_pos, executor.cam_mat,
                                              executor.cam_fovy, executor.cam_w, executor.cam_h)
    except Exception:
        return None
    if pts is None or len(pts) < 50:
        return None
    z_hi = float(np.percentile(pts[:, 2], 95))
    if z_hi - float(np.percentile(pts[:, 2], 5)) < 0.10:
        return None
    sel = pts[pts[:, 2] > z_hi - 0.02]
    return np.array([float(sel[:, 0].mean()), float(sel[:, 1].mean()), z_hi])


def _do_place(executor, target: np.ndarray, lbl: str) -> None:
    place_target = target.copy()
    # Remember where we intend to land the held object - the post-release
    # event capture verifies the placement against this.
    executor.last_place_target = target.copy()
    # Drawer-aware placement: after open_drawer ran, a place whose label
    # references the drawer/handle/cabinet should target the pulled-out
    # TRAY interior (tracked by the primitive), not the stale detection of
    # the closed-drawer handle.
    tray = getattr(executor, 'last_drawer_tray_pos', None)
    if tray is not None and any(k in lbl for k in
                                 ('drawer', 'handle', 'cabinet')):
        place_target = np.array([tray[0], tray[1], tray[2] + 0.10])
        # The post-release verification must check the tray landing
        # point, not the stale handle detection.
        executor.last_place_target = place_target.copy()
        ee = executor._ee()
        safe_z = max(float(ee[2]) + 0.05, float(place_target[2]) + 0.15)
        lift = ee.copy()
        lift[2] = safe_z
        executor._move_to(lift, False, steps=300)
        above = place_target.copy()
        above[2] = safe_z
        executor._move_to(above, False, steps=500)
        executor._move_to(place_target, False, steps=300)
        return

    if hasattr(executor.env, '_grip_offset'):
        # The bowl-rim grip leaves the bowl's centroid at EE - grip_offset_world
        # (gripper sits +Y of the rim).  To land the bowl centroid ON the
        # target XY, the EE must be at target + grip_offset.
        place_target[:2] += executor.env._grip_offset[:2]
    held_height = getattr(executor.env, '_held_obj_height', 0.0)
    is_flat_place = any(kw in lbl for kw in _FLAT_PLACE_KW)
    if held_height > 0.05 and is_flat_place:
        safe_drop = max(held_height * 2.0 + 0.05, 0.20)
        place_target[2] = target[2] + safe_drop
    else:
        # Settle-crawl compensation.  The settle-to-contact crawl in
        # _do_top_down_pick grips the object N cm LOWER than the detected
        # keypoint, so the object sits N cm HIGHER at any commanded release
        # height.  The plan's place offset is authored against the un-crawled
        # grip, so give the crawl depth back at release.  Not applied on the
        # flat-place branch above, which already sets an absolute drop height
        # from the object's own extent.
        place_target[2] -= float(getattr(executor, 'last_crawl_dz', 0.0) or 0.0)
    # Three-phase transport (diagonal motion would drag the held bowl
    # through stove/plate edges):
    #   1. LIFT vertically at current XY to a safe clearance,
    #   2. XY-TRANSPORT horizontally at that height,
    #   3. DESCEND vertically to place_target.
    ee = executor._ee()
    safe_z = max(float(ee[2]) + 0.05,
                 float(place_target[2]) + 0.20)
    # Step 1: lift in place
    if ee[2] < safe_z - 0.02:
        lift = ee.copy()
        lift[2] = safe_z
        executor._move_to(lift, False, steps=400)
    # Step 2: XY transit at safe Z, monitored so a moved container is
    # caught mid-flight rather than after the descend.
    above = place_target.copy()
    above[2] = safe_z
    executor._move_to_monitored(lbl, above, False, steps=600)
    # Step 3: descend to place
    executor._move_to(place_target, False, steps=400)


# Pick

# Re-ground the pick target from above with the wrist camera before the grasp (SPARK_PICK_REGROUND=1, on in the
# paper's Table 1 and strict runs): the agentview mask of a tall item is mostly its front face, so its centroid
# sits ~2 cm toward the camera.
PICK_REGROUND = os.environ.get('SPARK_PICK_REGROUND', '0') == '1'
# SPARK_PICK_REGROUND_GATE_M: largest accepted re-ground shift. In static scenes (LIBERO-PRO) nothing moves before
# the grasp and the real fixes are <= 2 cm, while an 18 cm gate let a missed target bind a look-alike 8-15 cm away
# (dev: object position 7 milk 6/6 -> 1/6); the paper's runs use 0.05. 0.18 covers targets relocated at approach.
PICK_REGROUND_GATE_M = float(os.environ.get('SPARK_PICK_REGROUND_GATE_M', '0.18'))


def _wrist_fix(executor, lbl: str, near_xy: np.ndarray):
    """xy of the wrist camera's instance of ``lbl`` nearest ``near_xy``, or None."""
    from spark_bench.libero_pro.perception import redetect_wrist
    fresh = redetect_wrist(executor.env, executor.sam3, executor.prompts_for_refresh, executor.cfg,
                           all_instances=True)
    cands = [d for d in (fresh.dets if fresh is not None else [])
             if d.position_3d is not None and fuzzy_get_det({d.label: d}, lbl) is not None]
    if not cands:
        return None
    xy = np.array([np.asarray(d.position_3d, float)[:2] for d in cands])
    return xy[int(np.argmin(np.linalg.norm(xy - near_xy, axis=1)))]


def _reground_pick(executor, lbl: str, target: np.ndarray):
    """xy shift for the pick target: from a hover above the bound position, the wrist camera's instance of
    ``lbl`` nearest to the binding (not the best-scored one: two bowls may be in view). The arm's approach can
    itself set the scene moving, so up to 4 fixes repeat 8 steps apart until two agree within 1 cm.
    None without a fix within PICK_REGROUND_GATE_M."""
    if executor.sam3 is None or not executor.prompts_for_refresh:
        return None
    hover = np.asarray(target, float).copy()
    hover[2] += 0.10
    executor._move_to(hover, True, steps=150, tol=0.02)   # a view point: no need to settle to 5 mm
    rec = {'label': lbl, 'fixes': []}
    executor.trial_meta.setdefault('pick_reground', []).append(rec)
    near, fix = np.asarray(target, float)[:2], None
    for k in range(4):
        if k:
            for _ in range(8):
                step_env(executor.env, np.array([0, 0, 0, 0, 0, 0, -1.0]))
        try:
            f = _wrist_fix(executor, lbl, near)
        except Exception as e:  # noqa: BLE001  (no fix: keep the binding)
            rec['err'] = repr(e)[:120]
            break
        if f is None:
            break
        rec['fixes'].append([round(float(x) * 100, 1) for x in f - target[:2]])
        still = fix is not None and float(np.linalg.norm(f - fix)) < 0.01
        fix, near = f, f
        if still:
            break
    if fix is None:
        return None
    shift = fix - np.asarray(target, float)[:2]
    rec['shift_cm'] = [round(float(x) * 100, 1) for x in shift]
    if float(np.linalg.norm(shift)) > PICK_REGROUND_GATE_M:
        rec['rejected'] = True
        return None
    return shift


def _shifted(det, shift):
    """A copy of ``det`` moved by ``shift`` in xy (its mask stays the agentview's)."""
    d = copy.copy(det)
    d.position_3d = np.asarray(det.position_3d, float).copy()
    d.position_3d[:2] += shift
    return d


def _do_pick(executor, label: str, det, target: np.ndarray,
              lbl: str, is_bowl: bool, is_plate: bool) -> None:
    # Remember pick context for grip-offset resolution at grasp time.
    executor.last_pick_label = label
    executor.last_pick_det_pos = (
        det.position_3d.copy() if det.position_3d is not None else None)
    executor.env._held_obj_height = 0.0
    # Reset per pick: only the top-down path crawls, and a stale depth from
    # an earlier pick must never bias this object's release height.
    executor.last_crawl_dz = 0.0

    pts = None
    mask = getattr(det, 'mask', None)
    if (executor.sam3 is not None and executor.depth is not None
            and mask is not None):
        pts = executor.sam3._backproject_mask(
            mask, executor.depth, executor.cam_pos, executor.cam_mat,
            executor.cam_fovy, executor.cam_w, executor.cam_h)
        if len(pts) > 50:
            # Held height, read by _do_place's flat-target drop.
            executor.env._held_obj_height = float(pts[:, 2].max() - pts[:, 2].min())

    if is_bowl:
        _do_bowl_rim_pinch(executor, target)
    elif is_plate:
        _do_plate_rim_pinch(executor, target, pts=pts)
    else:
        _do_top_down_pick(executor, target)


def _do_bowl_rim_pinch(executor, target: np.ndarray) -> None:
    rim_target = target.copy()
    rim_target[1] += 0.025
    rim_target[2] -= 0.03
    rim_target[2] = max(rim_target[2], 0.005)
    approach_descend(executor, rim_target, hi=0.10, mid=0.02,
                      mid_steps=100, final_steps=150)
    ee = executor._ee()
    if np.linalg.norm(rim_target - ee) > 0.015:
        nudge_toward_base(executor, rim_target, mag=0.025)
    # Held bowl height, so _do_place's flat-target drop fires (akita black bowl, typical).
    executor.env._held_obj_height = 0.05


def _do_plate_rim_pinch(executor, target: np.ndarray,
                          pts: Optional[np.ndarray] = None) -> None:
    cfg = executor.cfg
    # Rim point from the PERCEIVED extent, not a fixed offset: the
    # jaws separate along world Y, so the pinch point must sit ON the
    # rim circle along +Y (a fixed +4 cm offset lands INSIDE a ~9 cm
    # plate and closes on the flat face).  Fingertips descend to just
    # above the plate's BASE so the pads span the rim lip, not brush
    # its top.
    # Pinch at the rim SHOULDER (~70% of the perceived radius), not the
    # outer edge: a plate's outer lip is a shallow wedge that squirts
    # out of a vertical parallel-jaw close, but the dish wall where the
    # slope is steepest (for LIBERO's plate also where the thin vertical
    # collision ring lives, r~=0.048 of r_outer~=0.069) presents
    # near-vertical contact faces that the jaws hold reliably.
    # Single close + immediate lift: a verify-and-resweep (probe lift +
    # aperture read + retry) leaks grip every extra second and a second
    # close cycle pops the wedged wall out.
    rim_dy = 0.045
    rim_z = max(target[2] - 0.018, 0.005)
    if pts is not None and len(pts) > 50:
        y_ext = float(np.percentile(pts[:, 1], 98)) - float(target[1])
        if y_ext > 0.02:
            rim_dy = float(np.clip(0.70 * y_ext, 0.02, 0.08))
        z_lo = float(np.percentile(pts[:, 2], 5))
        rim_z = max(z_lo + 0.004, 0.005)
    rim_target = target.copy()
    rim_target[1] += rim_dy
    rim_target[2] = rim_z
    approach_descend(executor, rim_target, hi=0.14, mid=0.03,
                      mid_steps=150, final_steps=200, high_steps=150)
    if cfg.verbose:
        print(f"[plate pinch] target=("
              f"{rim_target[0]:.3f},{rim_target[1]:.3f},{rim_target[2]:.3f}) "
              f"rim_dy={rim_dy:.3f}")


def _do_top_down_pick(executor, target: np.ndarray) -> None:
    above = target.copy()
    above[2] += 0.08
    # Monitored approach: re-check the scene between chunks so a mid-
    # flight displacement is caught in ~0.7 s instead of at pre-close.
    executor._move_to_monitored(
        getattr(executor, 'last_pick_label', '') or '',
        above, True, steps=150)
    residual = executor._move_to(target, True, steps=200)
    # Settle-to-contact: the detected keypoint is
    # the TOP surface the camera saw - especially for wrist-only
    # detections, whose backprojected z is the object's lid.  Closing at
    # that height pinches air above tall bodies (cans / cartons: fingers
    # shut to ~1 mm just over the shoulder and the object never moves).
    # Crawl down with the jaws open until the descent stalls (palm or
    # finger contact with the object / table resists the push) or a
    # bounded extra depth, then let the grasp primitive close at BODY
    # height.
    # The crawl MUST regulate XY while it pushes down.  A bare
    # ``action[2] = -1`` leaves the OSC x/y deltas at zero, which is not a
    # position hold: robosuite re-anchors the controller goal to the
    # current pose every step, so the arm sags along its reach axis under
    # the downward push.  Measured on goal/pos-t2: 5-6 crawl steps drifted
    # the EE +2.0 to +2.3 cm in +x (away from the base) off a wine-bottle
    # neck that had been reached to within 1 mm, and the jaws then shut on
    # air (aperture 1 mm) on 29 of 30 attempts.  Holding the pre-crawl XY
    # with the same proportional term ``move_to`` uses keeps the descent
    # vertical.
    hold_xy = executor._ee()[:2].copy()
    start_z = float(executor._ee()[2])
    prev_z, stall = start_z, 0
    for _ in range(50):
        ee_now = executor._ee()
        action = np.zeros(7)
        action[:2] = np.clip((hold_xy - ee_now[:2]) * 10.0 / 0.05, -1.0, 1.0)
        action[2] = -1.0
        action[6] = -1.0
        if step_env(executor.env, action):
            break
        z = float(executor._ee()[2])
        if start_z - z >= 0.035:
            break
        if prev_z - z < 5e-4:
            stall += 1
            if stall >= 3:
                break
        else:
            stall = 0
        prev_z = z
    # How far the crawl actually took us below the detected keypoint.  The
    # place path gives this back so the crawl does not silently raise the
    # release height (see _do_place).
    executor.last_crawl_dz = max(0.0, start_z - float(executor._ee()[2]))
    if residual > 0.015:
        nudge_toward_base(executor, target, mag=0.025)
