"""``push_object``: push a source object toward a target region, with re-perception between pushes."""
from __future__ import annotations

from typing import Optional

import numpy as np

from spark_bench.libero_pro.motion import step_env
from spark_bench.libero_pro.primitives._common import fuzzy_get_det, mask_points_3d, scene_front_axis

__all__ = ['handle_push_object']


def _directional_offset(instruction, src_xy, tgt_xy,
                        front_axis=np.array([1.0, 0.0])):
    """
    Apply directional modifier from instruction to shift the effective target.

    LIBERO ``push_*_to_the_<direction>_of_<landmark>`` tasks have a goal
    region that is offset from the landmark body centroid by ~0.15-0.36 m.
    Without this shift, ``push_object`` aims at the body centroid and the
    plate stops short of (or past) the actual goal region.

    ``front_axis`` is the scene-front direction from
    :func:`scene_front_axis` (away from the robot base, dominant axis).
    The magnitude remains a nominal aim - the closed-loop goal polling in
    ``handle_push_object`` supplies the actual stop condition.

    Returns the shifted target xy (2,). If no modifier is found, returns
    ``tgt_xy`` unchanged.
    """
    inst = instruction.lower() if instruction else ""
    if "in front of" in inst or "front of" in inst:
        # 0.36 m matches the landmark-to-goal-region delta for the stove
        # "front of" tasks; the SAM3 detection lands near the body centroid,
        # so a smaller shift undershoots the goal x-band.
        return tgt_xy + front_axis * 0.36
    if "behind" in inst:
        # Behind = toward the robot base.
        return tgt_xy - front_axis * 0.20
    if "left of" in inst:
        return tgt_xy + np.array([0.0, +0.15])
    if "right of" in inst:
        return tgt_xy + np.array([0.0, -0.15])
    return tgt_xy


def _refetch_source(executor, label: str) -> Optional[np.ndarray]:
    """
    Re-run agentview detection and return the source object's fresh 3D
    position (None when perception is unavailable / label not re-seen).
    Merges freshest-wins into det_map, mirroring the executor's gate.
    """
    if executor.sam3 is None or not getattr(executor, 'prompts_for_refresh', None):
        return None
    try:
        from spark_bench.libero_pro.perception import redetect_agentview
        fresh = redetect_agentview(executor.env, executor.sam3,
                                    executor.prompts_for_refresh, executor.cfg)
        if fresh is None or not fresh.det_map:
            return None
        for k, v in fresh.det_map.items():
            executor.det_map[k] = v
        d = fuzzy_get_det(executor.det_map, label)
        if d is not None and getattr(d, 'position_3d', None) is not None:
            return d.position_3d.copy()
    except Exception:
        pass
    return None


def handle_push_object(executor, params: dict) -> None:
    """
    Push a source object toward a (optional) target region.

    Two modes:
      * Trajectory mode (preferred): ``target_label`` is supplied AND its
        position_3d is in ``executor.det_map``.  Push direction = unit
        vector from source -> target XY; distance = source-to-target
        norm clamped to <= 0.40 m so a far target body doesn't drive the
        wrist out of reach in one sweep.
      * Legacy mode: ``push_direction`` + ``push_distance`` (hardcoded by
        LLM) when ``target_label`` is missing or not detected.
    """
    det = fuzzy_get_det(executor.det_map, params.get('keypoint_label', '') or '')
    if det is None or det.position_3d is None:
        return
    obj_pos = det.position_3d.copy()

    push_dir: Optional[np.ndarray] = None
    push_dist: Optional[float] = None
    target_label = params.get('target_label', '') or ''
    if target_label:
        tgt_det = fuzzy_get_det(executor.det_map, target_label)
        if tgt_det is not None and getattr(tgt_det, 'position_3d', None) is not None:
            tgt_pos = tgt_det.position_3d.copy()
            # Apply directional modifier from instruction (e.g. "to the
            # front of the stove"): the LIBERO goal region is offset from
            # the landmark body centroid, so the effective target shifts.
            instruction = getattr(executor, 'instruction', '') or ''
            tgt_xy_pre = tgt_pos[:2].copy()
            front_axis = scene_front_axis(executor)
            tgt_xy_shifted = _directional_offset(
                instruction, obj_pos[:2], tgt_pos[:2],
                front_axis=front_axis)
            # Edge-anchored refinement for front/behind modifiers: the SAM3
            # landmark centroid varies by ~20 cm depending on which part of
            # the fixture (or an object sitting on it) the mask latched, so
            # a fixed centroid offset is unreliable.  The landmark's extent
            # ALONG the front axis is stable across those latchings - anchor
            # the aim to the mask's front/back edge instead.  The closed-
            # loop goal polling downstream still supplies the true stop.
            il_ = (instruction or '').lower()
            tgt_pts = mask_points_3d(executor, tgt_det)
            if tgt_pts is not None and ('front of' in il_ or 'behind' in il_):
                proj = tgt_pts[:, 0] * front_axis[0] + tgt_pts[:, 1] * front_axis[1]
                if 'behind' in il_:
                    edge = float(np.percentile(proj, 5)) - 0.10
                else:
                    edge = float(np.percentile(proj, 95)) + 0.10
                perp = np.array([-front_axis[1], front_axis[0]])
                perp_c = float(np.dot(tgt_pos[:2], perp))
                tgt_xy_shifted = front_axis * edge + perp * perp_c
            tgt_pos[0] = tgt_xy_shifted[0]
            tgt_pos[1] = tgt_xy_shifted[1]
            if bool(getattr(executor.cfg, 'verbose', False)):
                print(f"[push] tgt_pre=({tgt_xy_pre[0]:.3f},"
                      f"{tgt_xy_pre[1]:.3f}) "
                      f"tgt_post=({tgt_xy_shifted[0]:.3f},"
                      f"{tgt_xy_shifted[1]:.3f}) "
                      f"(directional shift applied)")
            # Compute push direction in the XY-plane only - pushing across
            # tabletop, not vertical. Keep Z from the source.
            delta = np.array([tgt_pos[0] - obj_pos[0],
                              tgt_pos[1] - obj_pos[1],
                              0.0])
            norm = float(np.linalg.norm(delta))
            if norm > 1e-3:
                push_dir = delta / norm
                # Push slightly past the source-to-target midpoint toward
                # the target.  No overshoot multiplier: the plate slides
                # roughly 1:1 with the gripper in LIBERO physics, and an
                # overshoot factor pushes the gripper toward the workspace
                # edge (J6 singularity).  Cap at 0.50 m so a far body doesn't
                # drive the wrist out of reach in one sweep.
                push_dist = float(np.clip(norm * 1.0, 0.02, 0.50))

    if push_dir is None:
        push_dir = np.array(params.get('push_direction', [0, -1, 0]),
                             dtype=float)
        # Re-normalize if the LLM emitted a non-unit vector.
        n = float(np.linalg.norm(push_dir))
        if n > 1e-6:
            push_dir = push_dir / n
        push_dist = float(params.get('push_distance', 0.15))
        # Safety net: a drawer slides ONLY along Y. The LIBERO cabinet
        # sits at Y~+0.3 with the drawer face sliding out to Y~+0.15 when
        # open; closing means pushing TOWARD the cabinet body which is
        # +Y. If the planner emitted -Y or +X/+Z for a drawer keypoint
        # while the instruction has "close", override to [0,+1,0] and
        # bump distance so contact actually moves the slide joint.
        kpl = (params.get('keypoint_label', '') or '').lower()
        instr = (getattr(executor, 'instruction', '') or '').lower()
        if ('drawer' in kpl or 'cabinet' in kpl) and 'close' in instr:
            if push_dir[1] < 0.7:
                if bool(getattr(executor.cfg, 'verbose', False)):
                    print(f"[push] DRAWER-CLOSE override: dir "
                          f"{push_dir.tolist()} -> [0,+1,0]")
                push_dir = np.array([0.0, 1.0, 0.0])
            push_dist = max(push_dist, 0.30)

    verbose = bool(getattr(executor.cfg, 'verbose', False))
    src_pts = mask_points_3d(executor, det)
    # Contact height: below the perceived top surface so the fingers strike
    # the side wall of the object rather than skimming its top edge - but
    # NOT so low that the fingertips press into the tabletop (det.position_3d
    # is the mask's backprojected TOP surface; for a ~2 cm plate, top-minus-
    # 2cm IS the table, and the resulting contact force stalls OSC several
    # cm off the commanded line).
    if src_pts is not None:
        z_lo = float(np.percentile(src_pts[:, 2], 5))
        z_hi = float(np.percentile(src_pts[:, 2], 95))
        z_contact = max(z_lo + 0.006, z_hi - 0.015)
    else:
        z_contact = obj_pos[2] - 0.012

    aim_xy = obj_pos[:2] + push_dir[:2] * push_dist

    def _r_for(dir_xy: np.ndarray) -> float:
        """
        Source half-extent along ``dir_xy``, from the INITIAL mask cloud
        against the INITIAL centroid.  Must not be recomputed against a
        moved centre with the stale cloud.
        """
        if src_pts is None:
            return 0.05
        proj = src_pts[:, 0] * dir_xy[0] + src_pts[:, 1] * dir_xy[1]
        proj_c = float(obj_pos[0] * dir_xy[0] + obj_pos[1] * dir_xy[1])
        # p2 (not p5): partial masks under-estimate the radius and a
        # too-short offset lands the fingertips ON the rim.
        return max(proj_c - float(np.percentile(proj, 2)), 0.02)

    def _bd_for(dir_xy: np.ndarray, center: np.ndarray = None) -> float:
        return _r_for(dir_xy) + 0.04

    # Direct push with a horizontal slide-in approach (vertical descents
    # stall on rim edges and overhangs in this suite's rack/cabinet
    # clutter; the slide-in at contact height glides under them).
    # Phase 1: descend at the CURRENT XY to contact height; Phase 2: slide
    # horizontally to the behind-spot; Phase 3: push in SHORT SUBSTEPS
    # polling the BDDL goal between substeps, the same signal
    # ``LiberoExecutor.run`` uses to short-circuit between primitives.
    # Closed-loop polling turns an open-loop distance guess (the goal
    # region's offset from the perceived landmark centroid varies ~20 cm
    # with which part of the landmark SAM3 latched) into
    # push-until-in-region, and stops before the object is shoved out the
    # far side of a narrow goal band.  A slide-in that stalls slightly
    # short (gentle contact with the source's near rim) is benign: the
    # push starts from first contact.
    # Servo-push: up to 4 attempts of (approach -> bounded push -> re-
    # perceive).  A single long push has no feedback on the OBJECT (the
    # plate can lag the fingertip by >30 cm and squirt off the push line
    # when contact lands off-centre); between attempts the source is
    # re-detected and re-aimed.
    ee_pre = executor._ee()
    start_xy = ee_pre[:2].copy()
    src_label = params.get('keypoint_label', '') or ''
    goal_hit = False
    cur_obj = obj_pos.copy()
    if target_label and any(k in src_label.lower() for k in ('plate', 'dish', 'tray')):
        # Open, flat item: drag it from INSIDE. Fingertips go down at its centre just above the floor and
        # move toward the aim; the far inner rim catches them. Pushing from behind needed clear table
        # there, and on LIBERO goal task 5 the behind-spot is the bowl (the descent stalled 18 cm up and
        # the sweep missed the plate; an earlier slide-in plowed the bowl into the wine bottle).
        rem = aim_xy - obj_pos[:2]
        rem_n = float(np.linalg.norm(rem))
        if rem_n > 0.02:
            pdir = np.array([rem[0] / rem_n, rem[1] / rem_n, 0.0])
            r_in = max(_r_for(pdir[:2]) - 0.015, 0.04)
            # fingertips below the rim top (+6 mm over the plate base): at +12 mm they rode over the rim
            z_in = (float(np.percentile(src_pts[:, 2], 20)) if src_pts is not None else obj_pos[2]) + 0.006
            centre = np.array([obj_pos[0], obj_pos[1], z_in])
            above = centre.copy()
            above[2] = z_in + 0.12
            executor._move_to(above, True, steps=250)
            executor._move_to(centre, True, steps=250)
            total = rem_n + r_in + 0.03   # the plate lagged 1-5 cm short of the aim; the goal band is 8 cm
            seg, traveled = 0.025, 0.0
            while traveled < total:
                traveled = min(traveled + seg, total)
                executor._move_to(centre + pdir * traveled, True, steps=80)
                if executor._goal_satisfied():
                    goal_hit = True
                    break
            if verbose:
                ee_f = executor._ee()
                print(f"[push] inside-drag aim=({aim_xy[0]:.3f},{aim_xy[1]:.3f}) r_in={r_in:.3f} z={z_in:.3f} "
                      f"end=({ee_f[0]:.3f},{ee_f[1]:.3f},{ee_f[2]:.3f}) traveled={traveled:.3f}/{total:.3f}")
            ee = executor._ee()
            up = ee.copy()
            up[2] += 0.10
            executor._move_to(up, True, steps=100)
            for _ in range(40):
                if step_env(executor.env, np.array([0, 0, 0, 0, 0, 0, -1.0])):
                    break
            if hasattr(executor, '_last_perception_t'):
                executor._last_perception_t = 0.0
            return
    for attempt in range(4):
        rem = aim_xy - cur_obj[:2]
        rem_n = float(np.linalg.norm(rem))
        if rem_n < 0.02:
            break
        pdir = np.array([rem[0] / rem_n, rem[1] / rem_n, 0.0])
        bd = _bd_for(pdir[:2])
        r_est = _r_for(pdir[:2])
        behind = cur_obj - pdir * bd
        behind[2] = z_contact
        # Approach.  Attempt 0: descend at the trial-start staging XY, then
        # slide in horizontally at contact height (glides under overhangs
        # that stall vertical descents).  Depth-clearance staging regressed:
        # cluttered swap layouts leave no reliably-clear staging spot.
        # Attempts >= 1: the object has vacated the clutter pocket and sits
        # on open table; a slide-in from the start XY would plow straight
        # through it, so approach from ABOVE the new behind-spot instead.
        if attempt == 0 and not target_label:
            stage = np.array([start_xy[0], start_xy[1], z_contact])
            executor._move_to(stage, True, steps=300)
            # Detour if the straight slide-in passes within the source's
            # footprint (it would shove the source the wrong way).  Bow
            # the path outward around the source.
            ab = behind[:2] - stage[:2]
            ab_n2 = float(np.dot(ab, ab))
            if ab_n2 > 1e-9:
                t = float(np.clip(np.dot(cur_obj[:2] - stage[:2], ab) / ab_n2,
                                   0.0, 1.0))
                cp = stage[:2] + t * ab
                dvec = cp - cur_obj[:2]
                dist = float(np.linalg.norm(dvec))
                clear = r_est + 0.06
                if dist < clear:
                    away = (dvec / dist if dist > 1e-6
                             else np.array([-ab[1], ab[0]]) / np.sqrt(ab_n2))
                    detour = np.array([cp[0] + away[0] * (clear - dist),
                                        cp[1] + away[1] * (clear - dist),
                                        z_contact])
                    executor._move_to(detour, True, steps=250)
            executor._move_to(behind, True, steps=400)
        else:
            above_b = behind.copy()
            above_b[2] = behind[2] + 0.12
            executor._move_to(above_b, True, steps=250)
            executor._move_to(behind, True, steps=300)
        if verbose:
            ee_b = executor._ee()
            print(f"[push] a{attempt} behind=({behind[0]:.3f},{behind[1]:.3f},"
                  f"{behind[2]:.3f}) reached=({ee_b[0]:.3f},{ee_b[1]:.3f},"
                  f"{ee_b[2]:.3f}) aim=({aim_xy[0]:.3f},{aim_xy[1]:.3f}) "
                  f"rem={rem_n:.3f}")
        # Bounded push toward the aim: object travel capped per attempt so
        # a mis-tracking push cannot carry the object far off-line before
        # the next re-perception.
        obj_travel = min(rem_n, 0.22)
        # +0.11 margin: the disc lags/squirms 2-4 cm per attempt; the goal
        # poll every 2.5 cm makes over-length pushes safe in an 8 cm band.
        total = bd + obj_travel + 0.11
        seg = 0.025
        traveled = 0.0
        while traveled < total:
            traveled = min(traveled + seg, total)
            wp = behind + pdir * traveled
            executor._move_to(wp, True, steps=80)
            if executor._goal_satisfied():
                goal_hit = True
                break
        if verbose:
            ee_f = executor._ee()
            print(f"[push] a{attempt} end=({ee_f[0]:.3f},{ee_f[1]:.3f},"
                  f"{ee_f[2]:.3f}) traveled={traveled:.3f}/{total:.3f} "
                  f"goal_hit={goal_hit}")
        ee = executor._ee()
        up = ee.copy()
        up[2] += 0.10
        executor._move_to(up, True, steps=100)
        if goal_hit:
            break
        # Dead-reckon estimate (~70% slip along the push line) as a sanity
        # anchor for the re-detection: a post-push SAM3 'plate' latch can
        # jump to a lookalike (burner plate) - reject refetches > 12 cm
        # from where physics says the object can be.
        reckon = cur_obj + pdir * (0.7 * max(traveled - bd, 0.0))
        fresh = _refetch_source(executor, src_label)
        if (fresh is not None
                and float(np.linalg.norm(fresh[:2] - reckon[:2])) < 0.12):
            cur_obj = fresh
            cur_obj[2] = obj_pos[2]
        else:
            cur_obj = reckon
    # Settle: idle for ~40 steps so the pushed object (and any object
    # it landed on, e.g. plate-on-stove-front) stops sliding before any
    # downstream BDDL check.  Without this, the runner's post-execute
    # check_success() may fire while the plate is still mid-slide and
    # report False, triggering a recovery push that knocks the plate
    # back out of the goal region (task 5).
    settle_action = np.zeros(7)
    settle_action[6] = 1.0 if executor.holding else -1.0
    for _ in range(40):
        if step_env(executor.env, settle_action):
            break
    # Force perception refresh: the pushed source object (and possibly
    # the target it landed on) has translated by up to 50 cm.  Cached
    # masks at the pre-push XY are stale for any subsequent keypoint
    # action.
    if hasattr(executor, '_last_perception_t'):
        executor._last_perception_t = 0.0

