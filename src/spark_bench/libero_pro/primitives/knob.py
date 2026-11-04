"""Knob turn from perception only (``rotate`` / ``screw`` / ``turn_knob``).

Everything about the knob comes from the SAM3 mask and depth: the fin (the only part a parallel jaw can
pinch) is the top band of the mask's 3D points, its long axis is that band's principal axis, and the
hinge is taken under the fin's centre. The turn is a fixed wrist spin; success is checked with the camera
(the fin's angle before vs after), never with the hinge state.

BDDL goal predicate (LIBERO ``Turnon`` / ``Turnoff``): FlatStove turn-on range [0.5, 2.1] rad, turn-off
[-0.005, 0]; knob hinge range [-0.005, 2.1].
"""
from __future__ import annotations

from typing import Optional

import mujoco
import numpy as np

from spark_bench.libero_pro.primitives._common import fuzzy_get_det, mask_points_3d, scene_front_axis

__all__ = ['handle_rotate', 'handle_turn_knob']


def _Rz(a: float) -> np.ndarray:
    c, sn = float(np.cos(a)), float(np.sin(a))
    return np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])


_KNOB_FIN_BAND = 0.012     # m below the highest mask point that counts as fin
_KNOB_SPIN = 2.05          # rad of commanded wrist spin (the fin slips ~1 rad in the jaws)
_KNOB_ON_SIGN = 1.0        # 'turn on' = counter-clockwise seen from above (LIBERO-90 stove scenes)


def _knob_fin(executor, det) -> Optional[tuple]:
    """(centre_xyz, long_axis_xy) of the knob fin from the detection's mask points, or None."""
    pts = mask_points_3d(executor, det)
    if pts is None:
        return None
    top = float(np.percentile(pts[:, 2], 98))
    fin = pts[pts[:, 2] >= top - _KNOB_FIN_BAND]
    if len(fin) < 8:
        return None
    xy = fin[:, :2] - fin[:, :2].mean(axis=0)
    evals, evecs = np.linalg.eigh(xy.T @ xy)
    ax = evecs[:, int(np.argmax(evals))]
    return np.array([*fin[:, :2].mean(axis=0), top]), ax / float(np.linalg.norm(ax))


def _knob_pick(executor, fresh, near: np.ndarray, fovy: float, r_xy: float = 0.05,
               r_z: float = 0.04) -> Optional[tuple]:
    # r_xy beyond 5 cm means ``near`` is the stove, not the knob: the knob is then the tallest fin on
    # it (burner rings sit ~3 cm lower), not the one nearest the stove's centre
    """Fin of the detection in ``fresh`` nearest ``near`` (within r_xy in xy, r_z in z), backprojected
    against ``fresh``'s own frame."""
    if fresh is None:
        return None
    saved = executor.depth, executor.cam_pos, executor.cam_mat, executor.cam_fovy
    best = None
    try:
        executor.depth, executor.cam_pos, executor.cam_mat = fresh.depth, fresh.cam.pos, fresh.cam.mat
        executor.cam_fovy = fovy
        for det in (fresh.dets or list(fresh.det_map.values())):
            f = _knob_fin(executor, det)
            if f is None:
                continue
            dxy = float(np.linalg.norm(f[0][:2] - near[:2]))
            key = -float(f[0][2]) if r_xy > 0.05 else dxy
            if dxy < r_xy and abs(float(f[0][2] - near[2])) < r_z and (best is None or key < best[0]):
                best = (key, f)
    except Exception:
        return None
    finally:
        executor.depth, executor.cam_pos, executor.cam_mat, executor.cam_fovy = saved
    return None if best is None else best[1]


def _knob_redetect(executor, label: str, near: np.ndarray, r_xy: float = 0.05, r_z: float = 0.04):
    """Fin of the knob near ``near`` measured on a fresh frame: agentview with the arm moved clear
    (lifted and pulled back toward the base), else the wrist camera from 20 cm above the knob."""
    if getattr(executor, 'sam3', None) is None:
        return None
    from spark_bench.libero_pro.perception import redetect_agentview, redetect_wrist
    # generic phrases too: a dark knob in front of the dark robot base can miss 'stove knob' (real-PRO
    # goal 7 task scene); the position gate around ``near`` keeps drawer handles and the like out
    prompts = list(dict.fromkeys([label] + list(_KNOB_PROMPTS)))
    ee = executor._ee().copy()
    back = np.asarray(executor.robot_base_world, float)[:2] - ee[:2] \
        if getattr(executor, 'robot_base_world', None) is not None else np.array([-1.0, 0.0])
    clear = ee.copy()
    clear[:2] += 0.15 * back / (float(np.linalg.norm(back)) or 1.0)
    clear[2] = max(ee[2], float(near[2]) + 0.25)
    executor._move_to(clear, True, steps=80)
    try:
        fresh = redetect_agentview(executor.env, executor.sam3, prompts, executor.cfg,
                                   all_instances=True, render_only=True, secondary_score=0.05)
        fin = _knob_pick(executor, fresh, near, executor.cam_fovy, r_xy, r_z)
        if fin is not None:
            return fin
        executor._move_to(np.array([near[0], near[1], near[2] + 0.20]), True, steps=100)
        fresh = redetect_wrist(executor.env, executor.sam3, prompts, executor.cfg, all_instances=True)
        fovy = float(executor.model.cam_fovy[executor.env.sim.model.camera_name2id('robot0_eye_in_hand')])
        return _knob_pick(executor, fresh, near, fovy, r_xy, r_z)
    except Exception:
        return None


def _spin_knob(executor, centre, long_ax, sign: float, verbose=False, spin: float = _KNOB_SPIN) -> None:
    """Clamp the fin across its thickness at the hub and spin the wrist by sign * spin.

    Position first, then joint 7 pre-rolled to the grasp yaw, then the posed approach: a posed move
    straight from a far wrist yaw can swing the arm away (knob close to the robot base, real-PRO
    goal 7 task scene)."""
    m, d = executor.model, executor.data
    close_ax = np.array([-long_ax[1], long_ax[0], 0.0])
    grip = np.array([centre[0], centre[1], centre[2] - 0.010])   # grip site ~11 mm above the fingertips
    above = grip + np.array([0.0, 0.0, 0.06])
    for _ in range(3):   # a long transit (the plan's approach went elsewhere) needs more than one leg
        executor._move_to(above, True, steps=120)
        if float(np.linalg.norm(executor._ee() - above)) < 0.03:
            break
    # Two admissible grasp yaws (jaw symmetry): least wrist travel, unless joint7 would pass its limit
    # over the spin.
    cur = d.site_xmat[executor.ee_site].reshape(3, 3)
    j7 = float(executor._q_now()[6])
    j7_lim = float(np.min(np.abs(m.jnt_range[executor.joint_ids[6]])))
    s7 = -float(np.sign(cur[2, 2])) or 1.0
    cands = []
    for s in (1.0, -1.0):
        x = s * close_ax
        z = np.array([0.0, 0.0, -1.0])
        R = np.column_stack([x, np.cross(z, x), z])
        q4 = np.zeros(4)
        v3 = np.zeros(3)
        mujoco.mju_mat2Quat(q4, (R @ cur.T).flatten())
        mujoco.mju_quat2Vel(v3, q4, 1.0)
        yaw = float(v3[2])
        cands.append((abs(j7 + s7 * (yaw + sign * spin)) > j7_lim, abs(yaw), R, yaw))
    _, _, R_grasp, yaw = min(cands, key=lambda t: (t[0], t[1]))
    if verbose:
        print(f"[knob] fin centre={np.round(centre, 3)} long_ax={np.round(long_ax, 2)} sign={sign:+.0f}")
    q = executor._q_now().copy()
    q[6] = float(np.clip(j7 + s7 * yaw, -j7_lim, j7_lim))
    executor._joint_move(q, gripper_open=True, steps=80)
    executor._move_to_pose(above, R_grasp, True, steps=60)
    executor._move_to_pose(grip, R_grasp, True, steps=90)
    executor._gripper_hold(False, steps=60)
    executor.holding = True
    if verbose:
        print(f"[knob] at grip: ee={np.round(executor._ee(), 3)} want={np.round(grip, 3)}")
    step = 0.25
    for k in range(1, int(np.ceil(spin / step)) + 1):
        executor._move_to_pose(grip, _Rz(sign * min(spin, k * step)) @ R_grasp, False, steps=35)
    executor._gripper_hold(True, steps=40)   # open in place: a drifting release flicks the knob off its stop
    executor.holding = False
    up = executor._ee().copy()
    up[2] += 0.05
    executor._move_to(up, True, steps=40)


def _fin_off_angle(executor, long_ax: np.ndarray) -> float:
    """Angle (0..pi/2) between the fin and the front-back axis; the fin lies front-back when off
    (every LIBERO-90 stove scene)."""
    c = abs(float(np.dot(long_ax, scene_front_axis(executor))))
    return float(np.arccos(np.clip(c, 0.0, 1.0)))


_KNOB_ON_ANGLE = 0.60      # rad off front-back that counts as on (the stove lights from ~0.5)
_KNOB_OFF_ANGLE = 0.15     # rad off front-back that counts as off
_KNOB_OFF_SPIN = 2.8       # rad of commanded spin when turning off
_KNOB_PROMPTS = ('knob', 'black knob', 'lever')
_KNOB_STOVE_R = 0.30       # m: a knob detection farther than this from the stove is another object


def handle_rotate(executor, params: dict) -> None:
    """
    Turn a detected knob on or off (the planner's ``rotate`` / ``screw`` / ``turn_knob``).

    Perception only: the fin pose comes from the SAM3 mask and depth on a fresh agentview frame with the
    arm moved clear; the direction from the instruction ('turn off' reverses); after each spin the fin
    is re-measured and the turn repeats (re-grip) until the fin's angle from front-back says on / off,
    at most three spins.  ``direction`` / ``angle`` params of older plans are accepted and ignored.
    """
    label = params.get('keypoint_label', '') or params.get('target_label', '') or 'stove knob'
    verbose = bool(getattr(executor.cfg, 'verbose', False))
    # the knob does not move until turned: anchor on the plan-time detection (a refresh with the arm
    # over the knob can rebind the label to another object)
    det = fuzzy_get_det(getattr(executor, 'det_map_plan', None) or executor.det_map, label)
    if det is None or getattr(det, 'position_3d', None) is None:
        if verbose:
            print(f"[rotate] no detection for '{label}'")
        return
    near = np.asarray(det.position_3d, float)
    r_xy, r_z = 0.05, 0.04
    plan = getattr(executor, 'det_map_plan', None) or executor.det_map
    stoves = [plan[k] for k in plan if 'stove' in k.lower() and 'knob' not in k.lower()
              and 'burner' not in k.lower() and getattr(plan[k], 'position_3d', None) is not None]
    stove = min(stoves, key=lambda v: float(v.position_3d[2])) if stoves else None   # sits on the table
    if stove is not None and ('knob' not in str(getattr(det, 'label', label)).lower()
                              or float(np.linalg.norm(near[:2] - stove.position_3d[:2])) > _KNOB_STOVE_R):
        # no knob detection (the label fell back to the stove) or one away from the stove: look for
        # the knob around the stove
        if verbose:
            print(f"[rotate] '{label}' at {np.round(near, 3)} is off the stove; searching around the stove")
        near = np.asarray(stove.position_3d, float)
        r_xy, r_z = _KNOB_STOVE_R, 0.08
    off = any(s in executor.instruction.lower() for s in ('turn off', 'off the', 'switch off'))
    sign = -_KNOB_ON_SIGN if off else _KNOB_ON_SIGN
    fin = _knob_redetect(executor, label, near, r_xy, r_z)
    for spin in range(3):
        if fin is None:
            if verbose:
                print(f"[rotate] no knob fin seen near {np.round(near, 3)}")
            return
        ang = _fin_off_angle(executor, fin[1])
        if verbose:
            print(f"[rotate] fin {ang:.2f} rad off front-back ({'off' if off else 'on'} wanted)")
        # 'off' means pressed onto the stop, finer than the camera resolves: always drive into it once
        if (spin > 0 and ang <= _KNOB_OFF_ANGLE) if off else (ang >= _KNOB_ON_ANGLE):
            return
        # 'off' is the knob on its stop: over-drive into it (the fin slips ~0.3-1 rad in the jaws)
        _spin_knob(executor, fin[0], fin[1], sign, verbose, spin=_KNOB_OFF_SPIN if off else _KNOB_SPIN)
        fin = _knob_redetect(executor, label, fin[0])


def handle_turn_knob(executor, params: dict) -> None:
    """
    Thin alias: route legacy ``turn_knob`` BTs to ``handle_rotate``.

    Cached BTs that already set ``keypoint_label`` pass through unchanged.
    Legacy BTs with no params default to whatever knob/stove label SAM3
    actually produced in this scene.
    """
    mapped = dict(params or {})
    mapped.setdefault('angle_rad', 1.8)
    if not mapped.get('keypoint_label'):
        # Prefer the canonical multiphase concept label; if SAM3 emitted
        # only 'stove' or 'flat stove', fuzzy_get_det inside handle_rotate
        # picks that up via word-overlap.
        for cand in ('stove knob', 'knob', 'stove'):
            if cand in executor.det_map:
                mapped['keypoint_label'] = cand
                break
        else:
            mapped['keypoint_label'] = 'stove knob'
    handle_rotate(executor, mapped)

