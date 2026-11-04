"""
``open_drawer``: perception-driven drawer pull.

1. Handle from SAM3 handle detections clustered by height, level from the instruction (top / middle /
   bottom), checked against the level's band on the cabinet face (``drawer_geometry``).
2. Pull direction from perception: cabinet body to handle, dominant axis.
3. Lower (shadowed) drawers: a frontal hook, a fingertip behind the bar, pitched below horizontal. The
   top-down pinch is blocked for them by the hand: pinching the middle bar puts the gripper body at
   z 0.16 to 0.21, which meets the top drawer's bar at 0.1757 to 0.1921 (wooden_cabinet.xml).
4. Top drawer, or a frontal hook that did not open the drawer: top-down bar pinch with the jaws along
   the pull axis, then a closed-loop pull in substeps with re-grasp rounds.

Afterwards ``executor.last_drawer_tray_pos`` holds the pulled-out tray interior, so a following
place-into-drawer targets it.
"""
from __future__ import annotations

import math

import numpy as np

from spark_bench.libero_pro.ik import quat_from_approach
from spark_bench.libero_pro.perception import redetect_agentview
from spark_bench.libero_pro.primitives._common import fuzzy_get_det
from spark_bench.libero_pro.primitives.drawer_geometry import (
    bar_top_z,
    cluster_handles,
    ensure_cabinet,
    handle_advance,
    in_reach,
    level_band_z,
    level_from,
    pull_dir,
)

__all__ = ['handle_open_drawer']

# Frontal-hook parameters, chosen on LIBERO-90 drawer scenes (the training split; real SAM3 bars, no
# oracle), not from demonstrations. Bottom drawer, task 6: pitch 34 / 60 deg x hook -10/5/11/25 mm:
# 0,1,4,2 / 0,1,4,4 of 4. The top drawer is not hooked: the frontal hook never opened it (stalls at 3-5 mm,
# 18 / 35 mm ride over the bar) and the top-down pinch opened it 14/15. (The perceived bar sits ~7 mm
# above the bar centre: SAM3 sees its upper face.)
_PITCH_DEG = 60.0       # approach pitched this far below horizontal
_ENTRY_DZ = 0.025       # hook point above the perceived bar
# Pull, task 6: seat bias 0 / 6 / 12 mm x 8 / 16 steps: 0,0 / 4,4 / 4,4 of 4.
_PULL_LEN = 0.16        # pull length, in _PULL_STEPS equal steps
_PULL_STEPS = 8
_SEAT = 0.012           # bias along the approach while pulling, keeps the fingertip seated
# A pull waypoint aims 12 mm into the bar, so it never arrives within the default 5 mm and each segment
# ran to the stall exit. Arrive within this instead.
_PULL_TOL = 0.015
# Top-down descend offsets along the pull. LIBERO-90 top drawer (task 7) and the PRO goal-3 scenes: only
# 0.036 reaches bar depth; 0.028 / 0.020 / 0.012 all stop 3 cm high. Try it first.
_DESCEND_OFFSETS = (0.036, 0.028, 0.02, 0.012)
_REGRASP_OFFSETS = (0.02, 0.028, 0.012, 0.036)


def _best_close_sign(executor, pos, approach, close):
    """Return whichever of +/-close leaves the arm furthest inside its joint limits."""
    try:
        model = executor.model
        ids = [model.joint(f'robot0_joint{i + 1}').id for i in range(7)]
        lo = [float(model.jnt_range[j][0]) for j in ids]
        hi = [float(model.jnt_range[j][1]) for j in ids]
    except Exception:
        return close
    best, best_slack = close, -1e9
    for sign in (1.0, -1.0):
        cand = close * sign
        try:
            q = executor._ik6(pos, quat_from_approach(approach, cand))
        except Exception:
            continue
        if q is None:
            continue
        slack = min(min(q[i] - lo[i], hi[i] - q[i]) for i in range(7))
        if slack > best_slack:
            best, best_slack = cand, slack
    return best


def _frontal_pose(executor, handle_pos, pull) -> bool:
    """Open a lower drawer with a fingertip hooked behind the bar, end to end (hook and pull).

    True only when the pull ran and tracked, not merely when the arm reached the bar. Drives
    _move_to_pose (OSC with the orientation channel, tracks 1 to 5 mm). The gripper stays open; the
    approach is pitched _PITCH_DEG below horizontal.
    """
    mtp = executor._move_to_pose
    pull_v = np.asarray(pull, dtype=float)
    pull_v = pull_v / (np.linalg.norm(pull_v) + 1e-9)
    approach = -pull_v * math.cos(math.radians(_PITCH_DEG)) \
        + np.array([0.0, 0.0, -math.sin(math.radians(_PITCH_DEG))])
    finger_close = np.cross(pull_v, np.array([0.0, 0.0, 1.0]))
    entry_off = np.array([0.0, 0.0, _ENTRY_DZ])
    approach = approach / (np.linalg.norm(approach) + 1e-9)
    n = np.linalg.norm(finger_close)
    finger_close = (finger_close / n) if n > 1e-6 else np.array([1.0, 0.0, 0.0])
    finger_close = _best_close_sign(executor, handle_pos, approach, finger_close)

    ez = approach
    ex = finger_close - np.dot(finger_close, ez) * ez
    ex = ex / (np.linalg.norm(ex) + 1e-9)
    ey = np.cross(ez, ex)
    R = np.column_stack([ex, ey, ez])

    hook = np.asarray(handle_pos, dtype=float) + entry_off
    executor._gripper(True, steps=40)          # open, and it stays open
    enter = hook + pull_v * 0.10 + np.array([0.0, 0.0, 0.06])
    mtp(enter, R, True, steps=280)
    pe, re_ = mtp(hook, R, True, steps=240)
    ee = np.asarray(executor._ee(), dtype=float)
    print(f"[open_drawer][FRONTAL] hook target=({hook[0]:.3f},{hook[1]:.3f},{hook[2]:.3f}) "
          f"reached=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) err={pe * 1000:.0f}mm rot={re_:.2f}")
    if pe > 0.03:
        print(f"[open_drawer][FRONTAL] hook unreachable ({pe*1000:.0f}mm); falling back to top-down")
        return False

    seat = ez * _SEAT
    start_p = hook.copy()
    opened = False
    for i in range(1, _PULL_STEPS + 1):
        tgt = start_p + pull_v * (_PULL_LEN * i / _PULL_STEPS) + seat
        mtp(tgt, R, True, steps=70, pos_tol=_PULL_TOL)
        if executor._goal_satisfied():
            opened = True
            break
    ee = np.asarray(executor._ee(), dtype=float)
    advanced = float(np.dot((ee - start_p)[:2], pull_v[:2]))
    print(f"[open_drawer][FRONTAL] pull advanced={advanced*1000:.0f}mm opened={opened}")
    if not (opened or advanced >= 0.12):
        return False
    # Clear the drawer face so a compound task's next action starts clean.
    out = np.asarray(executor._ee(), dtype=float) + pull_v * 0.05
    out[2] += 0.10
    mtp(out, R, True, steps=160)
    return True


def _choose_handle(executor, params, level, verbose):
    """(handle_pos, bar_top_z, clusters) for the requested level, or (None, None, clusters)."""
    clusters = cluster_handles(in_reach(executor, executor.det_map))
    handle_pos = top_z = None
    if clusters:
        if level == 'bottom':
            chosen = clusters[-1]
        elif level == 'middle':
            chosen = clusters[1] if len(clusters) >= 2 else clusters[0]
        else:
            chosen = clusters[0]
        handle_pos, top_z = chosen[1], chosen[3]
        band = level_band_z(executor, level, clusters)
        if band is not None and abs(float(handle_pos[2]) - band) > 0.03:
            # The detected bars do not include the requested level (SAM3 found 1-2 of 3): drawers divide
            # the cabinet face into equal bands, so take that band's height.
            near = min(clusters, key=lambda c: abs(float(c[1][2]) - band))
            if abs(float(near[1][2]) - band) <= 0.03:
                handle_pos, top_z = near[1], near[3]
            else:
                handle_pos = np.array([handle_pos[0], handle_pos[1], band])
                top_z = band
            if verbose:
                print(f"[open_drawer] level {level}: band z={band:.3f} -> handle z={handle_pos[2]:.3f}")
    if handle_pos is None:
        det = fuzzy_get_det(executor.det_map, params.get('keypoint_label', '') or '')
        if det is not None and det.position_3d is not None:
            handle_pos = det.position_3d.copy()
            top_z = bar_top_z(det, handle_pos)
    if handle_pos is not None and top_z is None:
        top_z = float(handle_pos[2])
    return handle_pos, top_z, clusters


def _descend(executor, handle_pos, pull, d_off):
    """Open jaws above the bar, offset d_off along the pull, then down to bar depth."""
    grasp_pt = handle_pos + pull * d_off
    above = grasp_pt.copy()
    above[2] = handle_pos[2] + 0.08
    executor._move_to(above, True, steps=200)
    tgt = grasp_pt.copy()
    tgt[2] = max(handle_pos[2] - 0.005, 0.02)
    executor._move_to(tgt, True, steps=250)
    return tgt, executor._ee()


def _pinch(executor, handle_pos, pull, top_z, offsets, gate_dz, verbose=False) -> None:
    """Top-down bar pinch: the jaws separate along the pull, the TCP sits ahead of the bar so the rear
    finger enters the bar-to-face slot. The slot leaves ~+-3 mm against ~5 mm of detection noise, so
    sweep the offsets until the fingertips reach the bar top + gate_dz; when none does, re-descend at
    the deepest one. Then a regulated close."""
    ok = False
    best = None    # (reached_z, d_off)
    for d_off in offsets:
        at, ee_g = _descend(executor, handle_pos, pull, d_off)
        ok = ee_g[2] < top_z + gate_dz
        if verbose:
            print(f"[open_drawer] descend d_off={d_off:.3f} reached=({ee_g[0]:.3f},{ee_g[1]:.3f},"
                  f"{ee_g[2]:.3f}) target=({at[0]:.3f},{at[1]:.3f},{at[2]:.3f}) "
                  f"gate={top_z + gate_dz:.3f} ok={ok}")
        if ok:
            break
        if best is None or ee_g[2] < best[0]:
            best = (float(ee_g[2]), d_off)
    if not ok and best is not None and best[1] != offsets[-1]:
        if verbose:
            print(f"[open_drawer] no offset cleared the gate; committing to the deepest one "
                  f"d_off={best[1]:.3f} (z={best[0]:.3f})")
        _descend(executor, handle_pos, pull, best[1])
    executor._gripper_hold(False, steps=200)


def _regrasp_handle(executor, handle_pos, top_z, ee_f, pull):
    """Handle position after a slipped pull: re-detected at its level, else dead-reckoned along the pull
    from where the wedge slipped. Returns (handle_pos, bar_top_z)."""
    try:
        if executor.sam3 is not None and executor.prompts_for_refresh:
            fresh = redetect_agentview(executor.env, executor.sam3, executor.prompts_for_refresh,
                                       executor.cfg)
            if fresh is not None and fresh.det_map:
                for k, v in fresh.det_map.items():
                    executor.det_map[k] = v
                cl2 = cluster_handles(executor.det_map)
                if cl2:
                    near = min(cl2, key=lambda c: abs(c[1][2] - handle_pos[2]))
                    return near[1], near[3]
    except Exception:
        pass
    adv = float(np.dot((ee_f - handle_pos)[:2], pull[:2]))
    new_handle = handle_pos + pull * max(min(adv, 0.10), 0.0)
    new_handle[2] = handle_pos[2]
    return new_handle, top_z


def _topdown_open(executor, handle_pos, pull, top_z, verbose):
    """Pinch, then pull in 2 cm substeps with a downward bias (the slip mode is the wedge riding up over
    the bar); a slipped pull re-grasps at the handle's new position, at most 4 rounds. Returns
    (handle_pos, ee at the end of the last pull)."""
    _pinch(executor, handle_pos, pull, top_z, _DESCEND_OFFSETS, 0.010, verbose)
    opened = False
    ee_f = executor._ee()
    for grasp_round in range(4):
        start = executor._ee()
        total, seg = 0.20, 0.02
        t = 0.0
        while t < total:
            t = min(t + seg, total)
            wp = start + pull * t
            wp[2] = start[2] - 0.012
            executor._move_to(wp, False, steps=120, tol=_PULL_TOL)   # wp presses 12 mm into the bar
            if executor._goal_satisfied():
                opened = True
                break
            ee = executor._ee()
            if float(np.linalg.norm(ee[:2] - wp[:2])) > 0.07:
                if verbose:
                    print(f"[open_drawer] pull lost tracking at t={t:.2f} "
                          f"ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f})")
                break
        ee_f = executor._ee()
        advanced = float(np.dot((ee_f - start)[:2], pull[:2]))
        if verbose:
            print(f"[open_drawer] round={grasp_round} pull end t={t:.2f} advanced={advanced:.3f} "
                  f"opened={opened} ee=({ee_f[0]:.3f},{ee_f[1]:.3f},{ee_f[2]:.3f})")
        # A full-length tracked pull means the drawer travelled with it (~16 cm travel < 20 cm pull);
        # redundant rounds would exhaust the episode horizon.
        if opened or advanced >= 0.15 or grasp_round == 3 or advanced < 0.01:
            break
        executor._gripper_hold(True, steps=40)
        lift = executor._ee()
        lift[2] += 0.08
        executor._move_to(lift, True, steps=150)
        handle_pos, top_z = _regrasp_handle(executor, handle_pos, top_z, ee_f, pull)
        if verbose:
            print(f"[open_drawer] re-grasp at=({handle_pos[0]:.3f},{handle_pos[1]:.3f},{handle_pos[2]:.3f})")
        _pinch(executor, handle_pos, pull, top_z, _REGRASP_OFFSETS, 0.007)
    return handle_pos, ee_f


def handle_open_drawer(executor, params: dict) -> None:
    verbose = bool(getattr(executor.cfg, 'verbose', False))
    level = level_from(params, getattr(executor, 'instruction', '') or '')
    ensure_cabinet(executor)
    handle_pos, top_z, clusters = _choose_handle(executor, params, level, verbose)
    if handle_pos is None:
        if verbose:
            print("[open_drawer] no handle detection; skipping")
        return
    pull = pull_dir(executor, handle_pos)
    if verbose:
        print(f"[open_drawer] level={level} handle=({handle_pos[0]:.3f},{handle_pos[1]:.3f},"
              f"{handle_pos[2]:.3f}) bar_top_z={top_z:.3f} pull=({pull[0]:.2f},{pull[1]:.2f}) "
              f"clusters={[(lb, round(p[2], 3)) for lb, p, _, _ in clusters]}")

    if level != 'top':
        # The pull can run its full length with the fingertip riding over the bar, so check with the
        # camera that the handle moved (LIBERO needs 14 of 16 cm), else hook once 12 mm lower.
        # Re-hooking a half-open drawer pushed it back in, so each height gets one pull.
        for dz in (0.0, -0.012):
            hp = handle_pos.copy()
            hp[2] += dz
            if not _frontal_pose(executor, hp, pull):
                continue
            adv = handle_advance(executor, handle_pos, pull)
            if adv is None or adv >= 0.13:
                print(f"[open_drawer][FRONTAL] drawer opened (hook dz={dz*1000:+.0f}mm, "
                      f"advance {'unchecked' if adv is None else f'{adv*1000:.0f}mm'})")
                return
        print("[open_drawer][FRONTAL] frontal hook did not open the drawer; using top-down")

    handle_pos, ee_f = _topdown_open(executor, handle_pos, pull, top_z, verbose)

    # Release and clear the drawer front.
    executor._gripper_hold(True, steps=50)
    executor.holding = False
    back = ee_f + pull * 0.05
    back[2] = ee_f[2] + 0.15
    executor._move_to(back, True, steps=200)

    # Tray interior for a follow-up place-into-drawer: from where the handle ended (camera, else the
    # hand's advance capped at the 16 cm travel), not where the hand ended (the jaws slid ~5 cm past the
    # bar and the object was released over the drawer's front lip).
    adv = handle_advance(executor, handle_pos, pull)
    if adv is None or adv < 0.02:
        adv = min(max(float(np.dot((ee_f - handle_pos)[:2], pull[:2])), 0.0), 0.16)
    executor.last_drawer_tray_pos = np.array([
        handle_pos[0] + pull[0] * (adv - 0.09),
        handle_pos[1] + pull[1] * (adv - 0.09),
        handle_pos[2] - 0.02,
    ])
    # The drawer (and everything near it) has moved: force re-perception.
    executor._last_perception_t = 0.0
