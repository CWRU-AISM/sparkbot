"""One LIBERO episode: detect, plan, execute, then recover from the robot's own evidence.

The trial is scored by one read of the simulator's success predicate after the last action. Under
SPARK_NO_ORACLE (the default, see fair/__init__.py) nothing before that read consults the predicate.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

import mujoco
import numpy as np

from spark_core.attribution import Layer, attribute_failure
from spark_core.binding import compute_scene_diff, fuzzy_key
from spark_bench.fair.config import FairConfig
from spark_bench.fair.planning import _plan, _run_label_validator
from spark_bench.libero_pro.executor import _flatten_tree, execute_on_libero
from spark_bench.libero_pro.perception import (
    detect_scene,
    get_sam3,
    redetect_agentview,
    relation_bind_pick,
    select_prompts,
    wrist_verify_picks,
)

log = logging.getLogger(__name__)


def _execute(env, score, det, instruction: str, cfg: FairConfig, sam3,
              prompts: Optional[list] = None,
              trial_meta: Optional[dict] = None) -> None:
    """
    Common execute-via-the-executor call (used by primary + recovery paths).

    ``prompts`` (when given) is the active SAM3 detection prompt list.  The
    executor only reads it when ``cfg.freshness_timeout_s`` is set; in that
    case it re-runs SAM3 with the same vocabulary between primitives to
    bound perception staleness (DynamicVLA LAAS analog).

    ``trial_meta`` receives the executor's per-trial event records
    (grasp_outcomes, capture_events, scene_diffs, retarget_events).
    """
    execute_on_libero(env, score, det.det_map, cfg,
                       sam3=sam3, depth=det.depth,
                       cam_pos=det.cam.pos, cam_mat=det.cam.mat,
                       cam_fovy=det.cam.fovy,
                       cam_w=cfg.cam_width, cam_h=cfg.cam_height,
                       instruction=instruction,
                       prompts_for_refresh=prompts,
                       trial_meta=trial_meta)


def _oracle_gate(env) -> bool:
    """Mid-episode read of the simulator's success predicate, gated.

    Under ``SPARK_NO_ORACLE=1`` every CONTROL read of ``env.check_success``
    returns False: the robot never learns from the simulator that it is done,
    that it should stop pushing, or that a retry is unnecessary. Scoring is
    untouched; the trial's outcome is still the official predicate, read once
    after the last action. This exists to MEASURE what the mid-episode reads
    are worth, because the paper criticises Zetta for exactly this access.
    """
    if _no_oracle():
        return False
    try:
        return bool(env.check_success())
    except Exception:
        return False


def _no_oracle() -> bool:
    return os.environ.get('SPARK_NO_ORACLE', '').strip() in ('1', 'true', 'yes')


def _attempt_ok(env, trial_meta: Optional[dict], n_grasps_before: int = 0) -> bool:
    """Whether the runner treats an execution attempt as done (decides recovery).

    Under SPARK_NO_ORACLE the robot's own evidence decides: the attempt failed only if the executor
    aborted on a scene diff or the attempt's last grasp came up empty or slipped. Otherwise (assisted
    runs) the simulator predicate decides, as before.
    """
    # SPARK_ORACLE_RETRY=1 (the RATs / CaP-Agent0 protocol of Table 1): the success check after an attempt
    # decides the retry, while every other mid-episode read stays gated by SPARK_NO_ORACLE.
    if not _no_oracle() or os.environ.get('SPARK_ORACLE_RETRY') == '1':
        try:
            return bool(env.check_success())
        except Exception:
            return False
    tm = trial_meta if trial_meta is not None else {}
    abort = tm.pop('scene_abort', None)
    if abort is not None:
        tm.setdefault('scene_aborts', []).append(abort)
        return False
    g = (tm.get('grasp_outcomes') or [])[n_grasps_before:]
    return not (g and g[-1].get('outcome') in ('empty_close', 'slip'))


def _settle(env, steps: int = 20) -> None:
    """Hold still for ~1 s before a post-execution success read: the read used to fire the instant
    the plan ended, while a released item was still dropping into its container (butter: the box
    was in the basket in the wrist frames, the read said no, and the recovery re-plan took it back
    out)."""
    try:
        for _ in range(steps):
            env.step(np.zeros(7))
    except Exception:
        pass


def _retract_arm(env, model, data, ee_site: int) -> None:
    """
    Lift the arm 10 cm before re-detecting (recovery loop helper).
    """
    mujoco.mj_forward(model, data)
    ee = (data.site_xpos[ee_site].copy() if ee_site >= 0
          else np.array([0.0, 0.0, 0.3]))
    lift = ee.copy(); lift[2] = max(ee[2] + 0.10, 0.3)
    for _ in range(100):
        try:
            curr = env.env.robots[0].controller.ee_pos
        except Exception:
            break
        err = lift - curr
        if np.linalg.norm(err) < 0.01:
            break
        a = np.zeros(7); a[:3] = np.clip(err * 8 / 0.05, -1, 1); a[6] = -1
        try:
            env.step(a)
        except Exception:
            break


def _find_ee_site(model) -> int:
    for s in range(model.nsite):
        sn = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, s) or '').lower()
        if 'grip_site' in sn:
            return s
    return -1


def _score_primary_label(score) -> str:
    """
    First ``move_to_keypoint`` label in the flattened score - the pick
    target, i.e. the object whose binding stability decides whether a
    failure is a perception problem.
    """
    try:
        for act in _flatten_tree((score or {}).get('tree', {})):
            lbl = (act.get('params', {}) or {}).get('keypoint_label')
            if lbl:
                return str(lbl)
    except Exception:
        pass
    return ''


def _layered_recovery(env, model, data, cfg: FairConfig, sam3,
                        score, det, instruction: str, prompts,
                        rgb, pick_hint: str, place_hint: str,
                        picks, places, pick_counts,
                        trial_meta: Optional[dict], stash_bt,
                        max_attempts: int = 3) -> bool:
    """
    ANCHOR-style minimum-responsible-layer recovery ladder.

    Per attempt: retract, re-check the goal, re-detect, diff the fresh
    detections against the previous binding, then let
    :func:`attribute_failure` pick the repair rung:

    * PERCEPTION -> re-bind (execute the SAME score against the fresh
      detections; no replan).
    * EXECUTION  -> local retry of the score (detections were stable, the
      motion failed); counts toward ``cfg.max_local_retries``.
    * PLAN       -> tier-3 replan via the configured planner (unavailable
      under --no-gemini: the ladder stops there).

    Attribution decisions land in trial_meta['recovery_attributions'],
    each stamped with ``t`` (decision) and ``t_resume`` (re-execution
    start) for the adaptation-latency metric.
    """
    meta = trial_meta if trial_meta is not None else {}
    ee_site = _find_ee_site(model)
    target_label = _score_primary_label(score) or pick_hint
    det_cur, score_cur = det, score
    local_retries = 0

    for attempt in range(max_attempts):
        _retract_arm(env, model, data, ee_site)
        if _oracle_gate(env):
            return True

        det2 = redetect_agentview(env, sam3, prompts, cfg)
        if det2 is not None and det2.det_map:
            # Label retention: the recovery frame is agentview-only,
            # so labels the bind frame got from the wrist camera (or
            # that are occluded right now) would vanish from det_map
            # and downstream fuzzy matching would re-aim at a lookalike
            # sharing a word. Merge freshest-wins per label, retaining
            # last-known entries for labels the fresh frame did not
            # produce (same semantics as the executor's freshness merge).
            merged = dict(det_cur.det_map)
            merged.update(det2.det_map)
            det2.det_map = merged
        # Recovery re-bind is the other moment where the wrong twin instance can be latched.
        if det2 is not None and det2.det_map:
            try:
                relation_bind_pick(det2, target_label or pick_hint, instruction, cfg,
                                   sam3=sam3, trial_meta=trial_meta)
            except Exception as e:  # noqa: BLE001 - binding help must never break a trial
                log.warning("relation bind at recovery failed: %s", e)

        sd = None
        if det2 is None or not det2.det_map:
            verify = {'predicate_ok': False, 'detection_missing': True}
        else:
            # Diff against where execution LEFT the scene when the
            # executor published it (self-caused changes re-bound), not
            # the plan-time binding - the object the robot just placed
            # must not read as an external 'moved'.
            final_binding = meta.get('final_binding')
            if final_binding:
                bound = {k: (np.asarray(v, dtype=float)
                             if v is not None else None)
                         for k, v in final_binding.items()}
            else:
                bound = {k: getattr(d, 'position_3d', None)
                         for k, d in det_cur.det_map.items()}
            current = {k: getattr(d, 'position_3d', None)
                       for k, d in det2.det_map.items()}
            mujoco.mj_forward(model, data)
            grip = (data.site_xpos[ee_site].copy() if ee_site >= 0 else None)
            sd = compute_scene_diff(
                bound, current, target_label=target_label or None,
                gripper_pos=grip,
                moved_threshold_m=float(getattr(
                    cfg, 'scene_diff_moved_cm', 1.5)) / 100.0,
                self_radius_m=float(getattr(
                    cfg, 'scene_diff_self_radius_cm', 12.0)) / 100.0)
            key = (fuzzy_key(det2.det_map, target_label)
                   if target_label else None)
            conf = (getattr(det2.det_map[key], 'confidence', None)
                    if key is not None else None)
            last_grasp = (meta.get('grasp_outcomes') or [{}])[-1]
            verify = {'predicate_ok': False,
                        'detection_missing': key is None,
                        'detection_confidence': conf,
                        'grasp_outcome': last_grasp.get('outcome')}

        layer = attribute_failure(
            verify, sd, local_retries,
            max_local_retries=int(getattr(cfg, 'max_local_retries', 1)))
        rec = {'attempt': attempt, 'layer': layer.value,
                'target_label': target_label,
                'diff_status': getattr(sd, 'target_status', None),
                'detection_confidence': verify.get('detection_confidence'),
                't': time.time()}
        meta.setdefault('recovery_attributions', []).append(rec)
        if cfg.verbose:
            log.info("recovery attempt %d: layer=%s target=%r diff=%s", attempt, layer.value, target_label,
                     rec['diff_status'])

        if layer is Layer.PLAN:
            if cfg.no_gemini:
                rec['action'] = 'replan_unavailable'
                return False
            if det2 is not None and det2.det_map:
                det_cur = det2
            score2 = _plan(cfg, instruction, det_cur.det_map,
                             getattr(det_cur, 'rgb', rgb),
                             pick_hint, place_hint, picks=picks,
                             places=places, pick_counts=pick_counts,
                             prompts=prompts)
            if not score2:
                rec['action'] = 'replan_failed'
                continue
            score2 = _run_label_validator(
                cfg, score2, det_cur.det_map, instruction,
                getattr(det_cur, 'rgb', rgb), pick_hint,
                place_hint, trial_meta=trial_meta,
                slot_key='validator_actions_recovery',
                picks=picks, places=places, pick_counts=pick_counts)
            stash_bt(trial_meta, score2, key='bt_yaml_recovery')
            score_cur = score2
            rec['action'] = 'replan'
        elif layer is Layer.PERCEPTION:
            if det2 is None or not det2.det_map:
                rec['action'] = 'redetect_failed'
                continue
            det_cur = det2
            rec['action'] = 'rebind'
        else:  # EXECUTION
            local_retries += 1
            if det2 is not None and det2.det_map:
                det_cur = det2  # stable anyway; retry off the fresh frame
            rec['action'] = 'local_retry'

        rec['t_resume'] = time.time()
        n_g = len(meta.get('grasp_outcomes') or [])
        _execute(env, score_cur, det_cur, instruction, cfg, sam3,
                  prompts=prompts, trial_meta=trial_meta)
        # Assisted runs: the predicate, as before. SPARK_NO_ORACLE: the robot's own evidence, so a
        # retry that worked is not re-executed on the solved scene.
        if _attempt_ok(env, meta, n_g):
            return True
    return False


def run_spark_on_libero_env(env, prompts: list, instruction: str,
                              pick_hint: str, place_hint: str,
                              cfg: FairConfig,
                              trial_meta: Optional[dict] = None,
                              picks: Optional[list] = None,
                              places: Optional[list] = None,
                              pick_counts: Optional[list] = None,
                              post_plan_hook=None) -> bool:
    """
    Run the SPARK pipeline once on a LIBERO env.

    Pipeline: render -> SAM3 detect -> plan -> execute -> score. On failure, retract + re-detect +
    re-plan + re-execute (twice), or the layered recovery when cfg.layered_recovery is set.

    If ``trial_meta`` is supplied (a dict), per-trial diagnostics are
    written into it: ``bt_yaml`` (raw planner YAML for the first plan
    attempt), ``bt_yaml_recovery`` (list of YAML strings from recovery
    re-plans), and ``planner`` (which planner emitted the BT).  The dict
    is mutated in place - additive only, no schema break for callers
    that pass nothing.  The executor adds its own event records when the
    adaptive-execution flags are on (grasp_outcomes, capture_events,
    scene_diffs, retarget_events, recovery_attributions).

    ``post_plan_hook(env, score, det)`` (when given) fires once after
    planning + label validation, before execution - the LIBERO-Dyn
    perturbation injection point.
    """
    model = env.sim.model._model
    data = env.sim.data._data
    mujoco.mj_forward(model, data)

    def _stash_bt(meta: Optional[dict], score, key: str = 'bt_yaml') -> None:
        if meta is None or not isinstance(score, dict):
            return
        raw = score.get('__raw_yaml')
        planner = score.get('__planner')
        if key == 'bt_yaml':
            if raw is not None:
                meta['bt_yaml'] = raw
            if planner is not None:
                meta['planner'] = planner
        else:
            meta.setdefault('bt_yaml_recovery', [])
            meta['bt_yaml_recovery'].append({
                'planner': planner, 'bt_yaml': raw})

    # Early-out: goal already satisfied. Under SPARK_NO_ORACLE the robot must act and earn the final read.
    if _oracle_gate(env):
        return True

    obs, _, _, _ = env.step(np.zeros(7))
    rgb_raw = obs.get('agentview_image')
    if rgb_raw is None:
        return False
    rgb = rgb_raw[::-1].copy()

    prompts = select_prompts(rgb, instruction, cfg, prompts)

    # detect_scene reads the instruction for the compound multi-instance rule ("both moka pots").
    cfg._current_instruction = instruction

    sam3 = get_sam3()
    det = detect_scene(env, sam3, prompts, obs, cfg)
    if getattr(cfg, 'wrist_verify', False):
        try:
            wrist_verify_picks(env, sam3, det, picks or [pick_hint], cfg, trial_meta=trial_meta)
        except Exception as e:  # noqa: BLE001 - verification must never break a trial
            log.warning("wrist verify failed: %s", e)
    if not det.det_map:
        log.info("no detections")
        return False

    try:
        relation_bind_pick(det, pick_hint, instruction, cfg, sam3=sam3, trial_meta=trial_meta)
    except Exception as e:  # noqa: BLE001
        log.warning("relation bind failed: %s", e)

    score = _plan(cfg, instruction, det.det_map, rgb, pick_hint, place_hint,
                  picks=picks, places=places, pick_counts=pick_counts, prompts=prompts)
    if score is None:
        return False
    # Snap or one-retry any label that is not in det_map before the executor sees it.
    score = _run_label_validator(
        cfg, score, det.det_map, instruction, rgb, pick_hint, place_hint, trial_meta=trial_meta,
        slot_key='validator_actions', picks=picks, places=places, pick_counts=pick_counts)
    _stash_bt(trial_meta, score, key='bt_yaml')

    # LIBERO-Dyn hook: scripted scene changes after planning has bound its keypoints.
    if post_plan_hook is not None:
        try:
            post_plan_hook(env, score, det)
        except Exception as e:  # noqa: BLE001
            log.warning("post_plan_hook failed: %s", e)

    _execute(env, score, det, instruction, cfg, sam3, prompts=prompts, trial_meta=trial_meta)
    _settle(env)
    if trial_meta is None:
        trial_meta = {}
    success = _attempt_ok(env, trial_meta)

    # Lowest-responsible-layer recovery (ANCHOR pattern; opt-in via
    # cfg.layered_recovery): attribute the failure to perception /
    # execution / plan and repair at that layer instead of always
    # replanning.  Works under --no-gemini too (the plan rung is simply
    # unavailable there).
    if not success and getattr(cfg, 'layered_recovery', False):
        success = _layered_recovery(
            env, model, data, cfg, sam3, score, det, instruction, prompts,
            rgb, pick_hint, place_hint,
            picks, places, pick_counts, trial_meta, _stash_bt)

    # Legacy blind recovery: retract + re-detect + re-plan + re-execute
    # (up to 2 tries).
    elif not success and not cfg.no_gemini:
        ee_site = _find_ee_site(model)
        for _ in range(2):
            _retract_arm(env, model, data, ee_site)
            # Re-check goal predicate AFTER retract. For push tasks the
            # plate may still be sliding when the post-execute check fires;
            # once the arm lifts and the sim settles the predicate often
            # flips True, and a further recovery push would knock it out.
            if _oracle_gate(env):
                success = True
                break
            det2 = redetect_agentview(env, sam3, prompts, cfg)
            if det2 is None:
                continue
            score2 = _plan(cfg, instruction, det2.det_map, det2.rgb,
                            pick_hint, place_hint,
                            picks=picks, places=places,
                            pick_counts=pick_counts, prompts=prompts)
            if not score2:
                continue
            score2 = _run_label_validator(
                cfg, score2, det2.det_map, instruction, det2.rgb,
                pick_hint, place_hint,
                trial_meta=trial_meta,
                slot_key='validator_actions_recovery',
                picks=picks, places=places, pick_counts=pick_counts)
            _stash_bt(trial_meta, score2, key='bt_yaml_recovery')
            n_g = len(trial_meta.get('grasp_outcomes') or [])
            _execute(env, score2, det2, instruction, cfg, sam3,
                      prompts=prompts, trial_meta=trial_meta)
            _settle(env)
            success = _attempt_ok(env, trial_meta, n_g)
            if success:
                break

    if _no_oracle():
        # the one scoring read; nothing above consulted the predicate
        _settle(env)
        success = bool(env.check_success())

    return success
