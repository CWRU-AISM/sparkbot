"""
SPARK on LIBERO-PRO: one suite under one perturbation, scored by LIBERO's own success predicate.

Each trial resets the LIBERO ``OffScreenRenderEnv`` to one of the task's 50 init states, runs the SPARK
pipeline (SAM3 detection, planner, executor, recovery) and reads ``check_success()`` once at the end.

    python -m spark_bench.run_libero_pro --suite object --perturbation position --num-trials 5

Writes <output_dir>/<suite>.json, with a <suite>.partial.json checkpoint after every task that a rerun
with the same arguments resumes from.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import tyro

from spark_bench.fair import timing
from spark_bench.fair.config import FairConfig
from spark_bench.fair.env import ProSuite, check_upstream_libero, get_perturbation_suite, make_env
from spark_bench.fair.execution import run_spark_on_libero_env
from spark_bench.libero_pro.bddl import get_task_prompts_for_suite, language_only_task_info

log = logging.getLogger(__name__)


def _load_checkpoint(path: Path, cfg: FairConfig) -> list:
    """Task details banked by an earlier process with the same trial count."""
    if not path.exists():
        return []
    try:
        prev = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        log.warning("checkpoint %s unreadable, fresh start: %s", path, e)
        return []
    if prev.get('num_trials') != cfg.num_trials or prev.get('trial_offset', 0) != cfg.trial_offset:
        return []
    banked = prev.get('perturbations', {}).get(cfg.perturbation, {}).get('task_details', [])
    log.info("checkpoint %s: %d tasks banked", path, len(banked))
    return banked


def _write(path: Path, cfg: FairConfig, task_details: list) -> None:
    """Atomic write, so a crash mid-write cannot corrupt the file."""
    rates = [d['success_rate'] for d in task_details]
    out = {'suite': cfg.suite, 'num_trials': cfg.num_trials, 'trial_offset': cfg.trial_offset,
           'perturbations': {cfg.perturbation: {
               'per_task': rates, 'average': float(np.mean(rates)) if rates else 0.0,
               'task_details': task_details}}}
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(out, indent=2))
    os.replace(tmp, path)


def run_trial(env, init_states, trial: int, task_info: dict, cfg: FairConfig) -> dict:
    """One episode; returns its trial_meta (success, timing, planner and executor records)."""
    trial_meta: dict = {'trial': trial}
    idx = trial + cfg.trial_offset
    t0 = s0 = None
    try:
        env.seed(idx)
        env.reset()
        # No reset after this (LIBERO's own eval order): a reset re-samples the placements from the seeded
        # RNG, 0.8-3.4 cm off the official initial state.
        env.set_init_state(init_states[idx % len(init_states)])
        for _ in range(10):
            env.step(np.zeros(7))
        timing.reset()
        t0, s0 = time.perf_counter(), int(env.env.timestep)
        success = run_spark_on_libero_env(
            env, task_info['prompts'], task_info['instruction'],
            task_info.get('pick', ''), task_info.get('place', ''), cfg, trial_meta=trial_meta,
            picks=task_info.get('picks') or None, places=task_info.get('places') or None,
            pick_counts=task_info.get('pick_counts') or None)
        trial_meta['success'] = bool(success)
    except Exception as e:  # noqa: BLE001 - one broken trial is a failed trial, not a failed run
        log.warning("trial %d error: %s", trial, e)
        ok = False
        if 'terminated episode' in str(e):
            # the episode clock ran out: score the final state (the one terminal read)
            try:
                ok = bool(env.check_success())
            except Exception:
                pass
        trial_meta['success'] = ok
        trial_meta['error'] = str(e)
    if t0 is not None:   # episode start (after the settle) to the scoring read
        steps = int(env.env.timestep) - s0
        trial_meta.update(timing.snapshot(), t_wall_s=round(time.perf_counter() - t0, 2), sim_steps=steps,
                          sim_time_s=round(steps / float(env.env.control_freq), 2))
    return trial_meta


def run_suite(cfg: FairConfig) -> None:
    base_suite = f"libero_{cfg.suite}"
    pert_suite = get_perturbation_suite(base_suite, cfg.perturbation)
    check_upstream_libero(pert_suite)
    task_suite = ProSuite(base_suite, pert_suite)
    log.info("%s from %s: %d tasks x %d trials", pert_suite,
             os.path.dirname(task_suite.get_task_bddl_file_path(0)), task_suite.n_tasks, cfg.num_trials)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    partial = out_dir / f'{cfg.suite}.partial.json'
    task_details = _load_checkpoint(partial, cfg)
    done = {d.get('task_id') for d in task_details}

    for task_id in range(task_suite.n_tasks):
        if task_id in done or (cfg.only_task_id >= 0 and task_id != cfg.only_task_id):
            continue
        task = task_suite.get_task(task_id)
        init_states = task_suite.get_task_init_states(task_id)
        bddl_path = task_suite.get_task_bddl_file_path(task_id)
        task_info = get_task_prompts_for_suite(base_suite, task_id, task, bddl_path=bddl_path)
        if cfg.language_only:
            # keep only the instruction; drop the BDDL-derived objects and curated phrases
            task_info = language_only_task_info(task_info['instruction'])
            cfg.use_gemini_prompts = True
        detail = {'task_id': task_id, 'trial_offset': cfg.trial_offset, 'task_name': task.name,
                  'instruction': task_info['instruction'], 'prompts': task_info['prompts'],
                  'pick': task_info.get('pick', ''), 'place': task_info.get('place', '')}
        try:
            env = make_env(bddl_path, cfg)
        except Exception as e:  # noqa: BLE001 - recorded in the result, the other tasks still run
            log.error("task %d: env load failed: %s", task_id, e)
            task_details.append({**detail, 'success_rate': 0, 'error': str(e)})
            continue
        metas = []
        for trial in range(cfg.num_trials):
            metas.append(run_trial(env, init_states, trial, task_info, cfg))
            print("ok" if metas[-1]['success'] else "*", end="", flush=True)
        try:
            env.close()
        except Exception:
            pass
        wins = [bool(m['success']) for m in metas]
        rate = sum(wins) / len(wins) if wins else 0.0
        task_details.append({**detail, 'success_rate': rate, 'trials': wins, 'trial_meta': metas})
        print(f" {task.name[:50]} {rate:.0%} ({sum(wins)}/{len(wins)})", flush=True)
        _write(partial, cfg, task_details)

    task_details.sort(key=lambda d: d['task_id'])
    _write(out_dir / f'{cfg.suite}.json', cfg, task_details)
    rates = [d['success_rate'] for d in task_details]
    log.info("%s %s: %.1f%% over %d tasks", cfg.suite, cfg.perturbation,
             100 * float(np.mean(rates)) if rates else 0.0, len(rates))


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    cfg = tyro.cli(FairConfig)
    timing.install()
    run_suite(cfg)


if __name__ == '__main__':
    main()
