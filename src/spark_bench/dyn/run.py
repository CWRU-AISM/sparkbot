"""LIBERO-Dyn runner: scripted displacement of the pick target (LIBERO-Displace).

    python -m spark_bench.dyn.run --suite object --task-id 0 --no-gemini \
        --phases post_plan,mid_approach,post_grasp_plan --magnitudes-cm 0,2,5,10 --slide-steps 40 \
        --scene-diff --layered-recovery          # the verify-and-adapt arm; see scripts/run_item.sh

Each trial runs the fair runner once; a post-plan hook starts the displacement.
"""
from __future__ import annotations

import json
import logging
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from spark_bench.dyn.bodies import displace_body_qpos, ee_reader, resolve_pick_target_pos
from spark_bench.dyn.inject import PerturbationInjector
from spark_bench.dyn.metrics import compute_latency_metrics
from spark_bench.dyn.schedule import Phase, displacement_vector
from spark_bench.fair.config import FairConfig

log = logging.getLogger(__name__)


@dataclass
class DynConfig(FairConfig):
    """FairConfig plus the LIBERO-Dyn sweep; ``num_trials`` trials per (phase, magnitude) cell."""
    task_id: int = 0
    """Task index within the base suite."""
    phases: str = 'post_plan,mid_approach,post_grasp_plan'
    """Comma-separated injection phases."""
    magnitudes_cm: str = '0,2,5,10'
    """Comma-separated displacement magnitudes (cm)."""
    dyn_output: str = ''
    """JSON output path (default results/dyn/dyn_<suite>_task<id>.json)."""
    slide_steps: int = 0
    """0: one instantaneous qpos write; N > 1: the displacement spread over N env steps (a slide)."""


def _open_env(cfg: DynConfig):
    """(env, task, init_states, bddl_path, base_suite)."""
    from spark_bench.fair.env import benchmark, load_libero_env
    base_suite = f'libero_{cfg.suite}'
    task = benchmark.get_benchmark_dict()[base_suite]().get_task(cfg.task_id)
    env, _task, init_states, bddl_path = load_libero_env(base_suite, cfg.task_id, cfg)
    # Retargets and recovery re-executions need more than the stock 2000 steps.
    try:
        env.env.horizon = max(int(getattr(env.env, 'horizon', 2000)), 6000)
    except Exception:
        pass
    return env, task, init_states, bddl_path, base_suite


def _reset(env, init_states, trial: int) -> None:
    env.seed(trial)
    env.reset()
    if init_states is not None and len(init_states) > 0:
        env.set_init_state(init_states[trial % len(init_states)])
        env.reset()
    for _ in range(10):
        env.step(np.zeros(7))


def _make_hook(cfg, pick_hint, phase, delta, meta, injectors: list):
    """post_plan_hook(env, score, det) for one trial."""
    def _hook(env, score, det):
        if float(np.linalg.norm(delta)) < 1e-9:
            return  # magnitude 0: control cell
        n_slide = max(int(cfg.slide_steps), 0)
        step_delta = (delta / n_slide) if n_slide > 1 else delta

        def displace():
            return displace_body_qpos(env, pick_hint, step_delta)

        if phase == Phase.POST_PLAN and n_slide <= 1:
            meta['t_perturb'] = time.time()
            body = displace()
            meta['perturb_phase'] = phase.value
            if body:
                meta['perturbed_body'] = body
            return
        if phase == Phase.POST_PLAN:
            inj = PerturbationInjector(env, phase, displace_fn=displace, meta=meta,
                                       slide_steps=n_slide, fire_on_first_step=True)
        else:
            inj = PerturbationInjector(env, phase, displace_fn=displace,
                                       pick_target_pos=resolve_pick_target_pos(det.det_map, pick_hint, score),
                                       get_ee_pos=ee_reader(env), meta=meta, slide_steps=n_slide)
        inj.install()
        injectors.append(inj)
    return _hook


def run_trial(cfg, env, init_states, task, task_info, phase, mag, trial) -> dict:
    from spark_bench.fair.execution import run_spark_on_libero_env
    pick_hint = task_info.get('pick', '') or ''
    meta: dict = {'trial': trial, 'phase': phase.value, 'magnitude_cm': mag}
    injectors: list[PerturbationInjector] = []
    success = False
    try:
        _reset(env, init_states, trial)
        delta = displacement_vector(task.name, trial, mag)
        meta['direction'] = [round(float(x), 4) for x in delta[:2]]
        success = run_spark_on_libero_env(
            env, list(task_info['prompts']), task_info['instruction'], pick_hint,
            task_info.get('place', ''), cfg, trial_meta=meta,
            picks=task_info.get('picks') or None, places=task_info.get('places') or None,
            pick_counts=task_info.get('pick_counts') or None,
            post_plan_hook=_make_hook(cfg, pick_hint, phase, delta, meta, injectors))
    except Exception as e:
        meta['error'] = str(e)
        if cfg.verbose:
            traceback.print_exc()
    finally:
        for inj in injectors:
            inj.uninstall()
    meta['success'] = bool(success)
    meta['perturb_fired'] = 't_perturb' in meta
    meta.update(compute_latency_metrics(meta))
    return meta


def run_dyn(cfg: DynConfig) -> dict:
    from spark_bench.libero_pro.bddl import get_task_prompts_for_suite
    env, task, init_states, bddl_path, base_suite = _open_env(cfg)
    task_info = get_task_prompts_for_suite(base_suite, cfg.task_id, task, bddl_path=bddl_path)
    phases = [Phase(p.strip()) for p in cfg.phases.split(',') if p.strip()]
    mags = [float(m) for m in cfg.magnitudes_cm.split(',') if m.strip()]
    log.info('LIBERO-Dyn: %s phases=%s mags=%s cm trials/cell=%d', task.name,
             [p.value for p in phases], mags, cfg.num_trials)
    cells: list[dict] = []
    for phase in phases:
        for mag in mags:
            trials = []
            for trial_i in range(cfg.num_trials):
                trial = trial_i + int(cfg.trial_offset)
                m = run_trial(cfg, env, init_states, task, task_info, phase, mag, trial)
                trials.append(m)
                log.info('%s %gcm T%d success=%s fired=%s det=%s adapt=%s by=%s/%s', phase.value, mag, trial,
                         m['success'], m['perturb_fired'], m.get('detection_latency_s'),
                         m.get('adaptation_latency_s'), m.get('detected_by'), m.get('adapted_by'))
            cells.append({'phase': phase.value, 'magnitude_cm': mag, 'trials': trials,
                          'success_rate': sum(t['success'] for t in trials) / len(trials) if trials else 0.0})
    try:
        env.close()
    except Exception:
        pass
    result = {
        'protocol': 'libero_dyn',
        'suite': cfg.suite,
        'task_id': cfg.task_id,
        'task_name': task.name,
        'instruction': task_info['instruction'],
        'num_trials': cfg.num_trials,
        'config': {
            'event_captures': cfg.event_captures,
            'scene_diff': cfg.scene_diff,
            'layered_recovery': cfg.layered_recovery,
            'retarget_max_cm': cfg.retarget_max_cm,
            'no_gemini': cfg.no_gemini,
            'slide_steps': cfg.slide_steps,
        },
        'cells': cells,
    }
    out_path = (Path(cfg.dyn_output) if cfg.dyn_output else
                Path('results/dyn') / f'dyn_{cfg.suite}_task{cfg.task_id}.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))
    log.info('wrote %s', out_path)
    return result


def main() -> None:
    import tyro
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run_dyn(tyro.cli(DynConfig))


if __name__ == '__main__':
    main()
