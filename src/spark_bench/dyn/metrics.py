"""Per-trial latency metrics and the LIBERO-Displace summary.

Summary CLI (paper numbers N10-N12):

    python -m spark_bench.dyn.metrics results/dyn

Every dyn JSON under the paths is read and grouped by arm (config.scene_diff gives adapt, else off) and
displacement (cm).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def compute_latency_metrics(trial_meta: dict) -> dict:
    """Detection and adaptation latency from the trial_meta event timeline.

    detection_latency_s: t_perturb to the earliest flag, either a scene-diff record with target_status
    moved/missing or a telemetry empty_close/slip vote (``detected_by`` says which).
    adaptation_latency_s: flag to resumed execution (a retarget's t_resume, else a recovery attribution's).
    """
    out = {
        'detection_latency_s': None,
        'adaptation_latency_s': None,
        'detected_by': None,
        'adapted_by': None,
        'recovery_used': bool(trial_meta.get('recovery_attributions') or trial_meta.get('bt_yaml_recovery')),
    }
    t_p = trial_meta.get('t_perturb')
    if t_p is None:
        return out
    candidates: list[tuple[float, str]] = []
    for rec in trial_meta.get('scene_diffs', []) or []:
        if rec.get('target_status') in ('moved', 'missing') and float(rec.get('t', 0.0)) >= t_p:
            candidates.append((float(rec['t']), 'scene_diff'))
            break
    for rec in trial_meta.get('grasp_outcomes', []) or []:
        if rec.get('outcome') in ('empty_close', 'slip') and float(rec.get('t', 0.0)) >= t_p:
            candidates.append((float(rec['t']), 'telemetry'))
            break
    if not candidates:
        return out
    t_flag, out['detected_by'] = min(candidates)
    out['detection_latency_s'] = t_flag - t_p

    for ev in trial_meta.get('retarget_events', []) or []:
        tf = float(ev.get('t_flag', 0.0))
        tr = ev.get('t_resume')
        if tr is not None and tf >= t_p:
            out['adaptation_latency_s'] = float(tr) - tf
            out['adapted_by'] = 'retarget'
            return out
    for rec in trial_meta.get('recovery_attributions', []) or []:
        tr = rec.get('t_resume')
        if tr is not None and float(tr) >= t_flag:
            out['adaptation_latency_s'] = float(tr) - t_flag
            out['adapted_by'] = f"recovery_{rec.get('layer', '?')}"
            return out
    return out


def load_runs(paths) -> dict:
    """{(kind, arm, x, file, cell): successes} for dyn JSONs: kind displace, arm adapt (scene diff on)
    or off, x = displacement (cm)."""
    runs = {}
    for p in paths:
        p = Path(p)
        for f in ([p] if p.is_file() else sorted(p.rglob('*.json'))):
            r = json.loads(f.read_text())
            cfg = r.get('config', {})
            if r.get('protocol') != 'libero_dyn':
                continue
            arm = 'adapt' if cfg.get('scene_diff') else 'off'
            for i, c in enumerate(r.get('cells', [])):
                key = ('displace', arm, float(c['magnitude_cm']), str(f), i)
                runs[key] = [bool(t.get('success')) for t in c.get('trials', [])]
    return runs


def summarize(runs: dict, kind_of) -> dict:
    """{(kind, arm): {x: (successes, trials)}}; ``kind_of(key)`` gives (kind, arm, x)."""
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for key, succ in runs.items():
        kind, arm, x = kind_of(key)
        agg[(kind, arm)][x][0] += sum(succ)
        agg[(kind, arm)][x][1] += len(succ)
    return {k: {s: tuple(v) for s, v in sorted(by.items())} for k, by in sorted(agg.items())}


def _print(table: dict) -> None:
    for (kind, arm), by in table.items():
        cells = ' / '.join(f'{100 * s / n:.0f}' for s, n in by.values())
        xs = '/'.join(f'{x:g}' for x in by)
        counts = ' '.join(f'{s}/{n}' for s, n in by.values())
        print(f'{kind} {arm}: {cells} % at {xs} cm displacement ({counts})')


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description='LIBERO-Displace success per arm and displacement.')
    ap.add_argument('paths', nargs='*', help='dyn JSON files or directories')
    a = ap.parse_args(argv)
    _print(summarize(load_runs(a.paths), lambda k: k[:3]))


if __name__ == '__main__':
    main()
