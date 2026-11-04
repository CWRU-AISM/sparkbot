"""Parity summary of one record.sh part, and the old-vs-new comparison.

    python summarize.py <part_dir>                    -> JSON on stdout
    python summarize.py --compare <old.json> <new.json>  exit 1 on any difference
"""
import glob
import json
import os
import sys


def summarize(d: str) -> dict:
    trials = {}
    for f in sorted(glob.glob(os.path.join(d, 't*.json'))):
        for pert in json.load(open(f))['perturbations'].values():
            for td in pert['task_details']:
                trials[td['task_name']] = [
                    {'success': bool(m.get('success')), 'sim_steps': m.get('sim_steps'),
                     'error': m.get('error')} for m in td.get('trial_meta', [])]
    llm = os.path.join(d, 'llm.jsonl')
    requests = [json.loads(line) for line in open(llm)] if os.path.exists(llm) else []
    return {'trials': trials, 'requests': requests}


def compare(old: dict, new: dict) -> list:
    diffs = []
    for task in sorted(set(old['trials']) | set(new['trials'])):
        a, b = old['trials'].get(task), new['trials'].get(task)
        if a != b:
            diffs.append(f'trials {task}: {a} != {b}')
    if len(old['requests']) != len(new['requests']):
        diffs.append(f"request count {len(old['requests'])} != {len(new['requests'])}")
    for i, (a, b) in enumerate(zip(old['requests'], new['requests'])):
        if a['body_sha256'] != b['body_sha256']:
            parts = [j for j, (p, q) in enumerate(zip(a['parts'], b['parts'])) if p != q]
            diffs.append(f'request {i}: body differs, parts {parts}')
    return diffs


if __name__ == '__main__':
    if sys.argv[1] == '--compare':
        diffs = compare(json.load(open(sys.argv[2])), json.load(open(sys.argv[3])))
        print('\n'.join(diffs) or 'identical')
        sys.exit(1 if diffs else 0)
    print(json.dumps(summarize(sys.argv[1]), indent=1))
