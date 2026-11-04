"""
LIBERO-PRO tables from run_libero_pro result JSONs: per-cell success, the development / held-out / all-four
macros, task-clustered 95% intervals and per-trial timing.

    python -m spark_bench.summarize results/libero_pro [more paths] [--ci] [--suite-means] [--timing] [--cells]

Each path is searched recursively. A directory holding ``items.txt`` and ``out/<i>/`` is a campaign queue:
line i of items.txt (``suite pert task offset n arm``) names the trials in out/<i>/, and with a ``done/``
folder only items marked done count. Any other ``*.json`` (not ``*.partial.json``) is a runner result whose
task details carry ``task_id`` and ``trial_offset``. Trials are keyed (suite, perturbation, task, init state);
a trial seen twice counts once.

Every cell is ten tasks at many init states, so its trials are not independent draws: trials within a task
share a scene, a prompt and a primitive path, and per-task rates vary widely. The intervals therefore
resample whole tasks (task-clustered percentile bootstrap), not trials.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
from pathlib import Path

BOOT = 10000
SEED = 20260827
SUITES = ('object', 'goal', 'spatial')
PERTS = ('position', 'task', 'language', 'object')
MACROS = (('development (position+task)', PERTS[:2]), ('held-out (language+object)', PERTS[2:]),
          ('all four', PERTS))
TIMING = ('t_wall_s', 't_plan_s', 'n_llm_calls', 't_perception_s', 'sim_time_s')


def _trials_of(result: dict, suite: str, task_id=None, offset=None):
    """(key, trial_meta) for every trial in one runner result. ``task_id`` / ``offset`` override the
    values in the file (campaign items)."""
    for pert, cell in result.get('perturbations', {}).items():
        for td in cell.get('task_details', []):
            if 'error' in td:   # the env did not load: no trials ran
                continue
            t = td.get('task_id') if task_id is None else task_id
            off = td.get('trial_offset', 0) if offset is None else offset
            if t is None:
                raise ValueError(f"{suite}/{pert}: result has no task_id; pass its campaign dir instead")
            for tm in td.get('trial_meta', []):
                yield (suite, pert, int(t), int(off) + int(tm['trial'])), tm


def load(paths) -> tuple[dict, int]:
    """All trials under ``paths`` keyed (suite, pert, task, init). Returns (trials, duplicates)."""
    trials, dups = {}, 0

    def add(items):
        nonlocal dups
        for k, tm in items:
            dups += k in trials
            trials.setdefault(k, tm)

    for root in map(Path, paths):
        if root.is_file():
            d = json.loads(root.read_text())
            add(_trials_of(d, d['suite']))
            continue
        camps = {p.parent for p in root.rglob('items.txt') if (p.parent / 'out').is_dir()}
        for camp in sorted(camps):
            items = (camp / 'items.txt').read_text().split('\n')
            done = camp / 'done'
            for out in sorted((camp / 'out').iterdir()):
                if done.is_dir() and not (done / out.name).exists():
                    continue
                files = [f for f in out.glob('*.json') if not f.name.endswith('.partial.json')]
                if not files:
                    continue
                suite, pert, task, off, _n, _arm = items[int(out.name)].split()
                d = json.loads(files[0].read_text())
                add((k, tm) for k, tm in _trials_of(d, suite, task, off) if k[1] == pert)
        for f in sorted(root.rglob('*.json')):
            if f.name.endswith('.partial.json') or any(c in f.parents for c in camps):
                continue
            d = json.loads(f.read_text())
            if isinstance(d, dict) and 'perturbations' in d and 'suite' in d:
                add(_trials_of(d, d['suite']))
    return trials, dups


def cells(trials: dict) -> dict:
    """(suite, pert) -> [(successes, trials) per task id]."""
    by_task = collections.defaultdict(lambda: [0, 0])
    for (s, p, t, _), tm in trials.items():
        by_task[(s, p, t)][0] += bool(tm.get('success'))
        by_task[(s, p, t)][1] += 1
    out = collections.defaultdict(list)
    for (s, p, t) in sorted(by_task):
        out[(s, p)].append(tuple(by_task[(s, p, t)]))
    return out


def rate(tasks) -> float:
    """Mean of per-task success rates (equal to the pooled rate when every task has the same trial count)."""
    return sum(k / n for k, n in tasks) / len(tasks)


def cluster_boot(tasks, reps: int = BOOT, seed: int = SEED) -> tuple[float, float]:
    """95% percentile interval of the pooled rate, resampling whole tasks with replacement."""
    rng = random.Random(seed)
    n = len(tasks)
    means = []
    for _ in range(reps):
        pick = [tasks[rng.randrange(n)] for _ in range(n)]
        means.append(sum(k for k, _ in pick) / sum(m for _, m in pick))
    means.sort()
    return means[int(0.025 * reps)], means[int(0.975 * reps)]


def macro_boot(cell_tasks: list, reps: int = BOOT, seed: int = SEED) -> tuple[float, float]:
    """95% interval of a macro average over cells, resampling tasks within each cell independently."""
    rng = random.Random(seed)
    out = []
    for _ in range(reps):
        acc = []
        for c in cell_tasks:
            pick = [c[rng.randrange(len(c))] for _ in range(len(c))]
            acc.append(sum(k for k, _ in pick) / sum(m for _, m in pick))
        out.append(sum(acc) / len(acc))
    out.sort()
    return out[int(0.025 * reps)], out[int(0.975 * reps)]


def _q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def report(trials: dict, dups: int, ci=False, suite_means=False, timing=False, show_cells=False,
           reps: int = BOOT) -> list[str]:
    lines = []
    c = cells(trials)
    timeouts = sum('error' in tm for tm in trials.values())
    lines.append(f"trials: {len(trials)} (run twice: {dups}, episode timeouts: {timeouts})")
    pct = lambda lo_hi: f"[{100 * lo_hi[0]:.1f}, {100 * lo_hi[1]:.1f}]"
    for s in SUITES:
        for p in PERTS:
            if (s, p) not in c:
                continue
            tasks = c[(s, p)]
            row = f"{s:8s} {p:9s} {100 * rate(tasks):5.1f}"
            if ci:
                row += f"  {pct(cluster_boot(tasks, reps))}"
            ns = [n for _, n in tasks]
            lines.append(row + f"  tasks {len(tasks)}, trials/task {min(ns)}-{max(ns)}")
    for name, ps in MACROS:
        keys = [(s, p) for s in SUITES for p in ps if (s, p) in c]
        if not keys:
            continue
        row = f"macro {name}: {100 * sum(rate(c[k]) for k in keys) / len(keys):.1f}"
        if ci:
            row += f" {pct(macro_boot([c[k] for k in keys], reps))}"
        lines.append(row + ('' if len(keys) == len(SUITES) * len(ps) else f" ({len(keys)} cells)"))
    lines.append('perturbation means: ' + ', '.join(
        f"{p} {100 * statistics.mean(rate(c[(s, p)]) for s in SUITES if (s, p) in c):.1f}"
        for p in PERTS if any((s, p) in c for s in SUITES)))
    if suite_means:
        lines.append('suite means: ' + ', '.join(
            f"{s} {100 * statistics.mean(rate(c[(s, p)]) for p in PERTS if (s, p) in c):.1f}"
            for s in SUITES if any((s, p) in c for p in PERTS)))
    if timing:
        for suite in SUITES + ('all',):
            rs = [tm for (s, *_), tm in trials.items() if suite in ('all', s) and 't_wall_s' in tm]
            if not rs:
                continue
            parts = [f"{suite:8s} n={len(rs)}"]
            for key in TIMING:
                xs = [r[key] for r in rs if r.get(key) is not None]
                if xs:
                    parts.append(f"{key} med {statistics.median(xs):.1f} p90 {_q(xs, .9):.1f}")
            lines.append(' | '.join(parts))
    if show_cells:
        for (s, p), tasks in sorted(c.items()):
            lines.append(f"{s} {p} " + ' '.join(f"t{i}:{k}/{n}" for i, (k, n) in enumerate(tasks)))
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('paths', nargs='+', help='result dirs or files (searched recursively)')
    ap.add_argument('--ci', action='store_true', help='task-clustered 95%% intervals per cell and macro')
    ap.add_argument('--suite-means', action='store_true', help='mean over perturbations per suite')
    ap.add_argument('--timing', action='store_true', help='median / p90 of the per-trial timing fields')
    ap.add_argument('--cells', action='store_true', help='successes / trials per task')
    ap.add_argument('--reps', type=int, default=BOOT, help='bootstrap resamples')
    a = ap.parse_args(argv)
    trials, dups = load(a.paths)
    print('\n'.join(report(trials, dups, a.ci, a.suite_means, a.timing, a.cells, a.reps)))


if __name__ == '__main__':
    main()
