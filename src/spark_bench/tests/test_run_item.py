"""scripts/run_item.sh builds the runner command of each repro item (python is stubbed on PATH)."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

FAKE_PY = f"""#!/bin/bash
[ "$1" = - ] && exec {sys.executable} "$@"
echo "NO_ORACLE=${{SPARK_NO_ORACLE-unset}} $* OR=${{SPARK_ORACLE_RETRY-}} PH=${{SPARK_PHRASES_V2-}} RG=${{SPARK_PICK_REGROUND-}} RGM=${{SPARK_PICK_REGROUND_GATE_M-}}" >> "$CALLS"
args=("$@")
for i in "${{!args[@]}}"; do
  case "${{args[$i]}}" in
    --dyn-output) echo '{{}}' > "${{args[$((i+1))]}}";;
    --output-dir) echo '{{"perturbations": {{}}}}' > "${{args[$((i+1))]}}/object.json";;
  esac
done
"""


def _run_item(tmp_path, *item, **leak):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / 'python').write_text(FAKE_PY)
    (bin_dir / 'python').chmod(0o755)
    calls = tmp_path / 'calls.txt'
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", CALLS=str(calls))
    for k in ('SPARK_NO_ORACLE', 'SPARK_ORACLE_RETRY', 'SPARK_PHRASES_V2', 'SPARK_PICK_REGROUND',
              'SPARK_PICK_REGROUND_GATE_M'):
        env.pop(k, None)
    env.update(leak)
    subprocess.run(['bash', str(ROOT / 'scripts' / 'run_item.sh'), str(tmp_path / 'out'), *item],
                   env=env, check=True, capture_output=True, text=True)
    return calls.read_text().split()


def test_displace_arms_match_the_paper_runs(tmp_path):
    on = _run_item(tmp_path, 'displace', 'object', '2', 'on')
    assert on[0] == 'NO_ORACLE=0'
    assert {'--scene-diff', '--layered-recovery', '--no-gemini'} <= set(on)
    assert '--event-captures' not in on
    assert on[on.index('--slide-steps') + 1] == '40'
    assert on[on.index('--phases') + 1] == 'post_plan,mid_approach,post_grasp_plan'
    off = _run_item(tmp_path, 'displace', 'object', '2', 'off')[len(on):]
    assert off[0] == 'NO_ORACLE=0'
    assert not {'--scene-diff', '--layered-recovery', '--event-captures'} & set(off)


def test_headline_item_runs_one_trial_without_oracle_override(tmp_path):
    call = _run_item(tmp_path, 'object', 'position', '3', '7', '1', 'wvrb')
    assert call[0] == 'NO_ORACLE=unset'   # fair/__init__ then defaults it to 1
    assert {'--no-bddl-hints', '--adaptive-prompts', '--wrist-verify', '--relation-bind'} <= set(call)
    assert call[call.index('--only-task-id') + 1] == '3'
    assert call[call.index('--trial-offset') + 1] == '7'
    assert call[call.index('--num-trials') + 1] == '1'


def test_paper_arms_set_their_env(tmp_path):
    t1 = _run_item(tmp_path, 'object', 'position', '0', '0', '1', 'wvrb_or_ph_rgs')
    assert {'--wrist-verify', '--relation-bind', 'OR=1', 'PH=1', 'RG=1', 'RGM=0.05'} <= set(t1)
    strict = _run_item(tmp_path, 'object', 'position', '1', '0', '1', 'wvrb_ph_rgs',
                       SPARK_ORACLE_RETRY='1')[len(t1):]   # a shell export does not leak into the arm
    assert {'OR=', 'PH=1', 'RG=1', 'RGM=0.05'} <= set(strict)
    plain = _run_item(tmp_path, 'object', 'position', '2', '0', '1', 'wvrb')[len(t1) + len(strict):]
    assert {'OR=', 'PH=', 'RG=', 'RGM='} <= set(plain)


def test_strict_displace_arms(tmp_path):
    slide = _run_item(tmp_path, 'displace', 'object', '0', 'onec_st')
    assert slide[0] == 'NO_ORACLE=1'
    assert {'--scene-diff', '--layered-recovery', '--event-captures'} <= set(slide)
    assert slide[slide.index('--slide-steps') + 1] == '40'
    inst = _run_item(tmp_path, 'displace', 'object', '0', 'off_st_inst')[len(slide):]
    assert inst[0] == 'NO_ORACLE=1'
    assert not {'--scene-diff', '--layered-recovery', '--event-captures'} & set(inst)
    assert inst[inst.index('--slide-steps') + 1] == '0'
