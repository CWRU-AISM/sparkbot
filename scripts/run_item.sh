#!/bin/bash
# Run one campaign item (one line of a repro/ item list) into <out_root>/<key>/.
# Usage: bash scripts/run_item.sh <out_root> <item line words...>
# Item forms:
#   <suite> <perturbation> <task_id> <trial_offset> <n_trials> <arm>     LIBERO-PRO
#     arm tokens: wv --wrist-verify, rb --relation-bind, _or SPARK_ORACLE_RETRY=1 (the success check after
#     an attempt decides the retry), _ph SPARK_PHRASES_V2=1, _rg SPARK_PICK_REGROUND=1, _rgs also
#     SPARK_PICK_REGROUND_GATE_M=0.05. Paper Table 1: wvrb_or_ph_rgs; strict: wvrb_ph_rgs.
#   displace <suite> <task_id> <arm>                                     fixed-plan displacement
#     arm on|onec|off: on --scene-diff --layered-recovery, onec also --event-captures, off neither;
#     suffix _st: strict (SPARK_NO_ORACLE=1, only the robot's own checks start a recovery);
#     suffix _inst: instantaneous displacement (one state write) instead of the 40-step slide.
# The item directory is the claim (atomic mkdir), so any number of these may run at once, on one or
# many machines sharing <out_root>. An item that finishes without a valid result is moved to
# <out_root>/failed/ and runs again on the next pass.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
OUT=${1:?out_root}; shift
KEY=$(echo "$*" | tr ' :' '__')
DIR=$OUT/$KEY
mkdir -p "$OUT"
mkdir "$DIR" 2>/dev/null || exit 0   # done or claimed elsewhere
# interrupted: release the claim so the next pass runs the item again
trap 'mkdir -p "$OUT/failed"; mv "$DIR" "$OUT/failed/$KEY.$(date +%s)"; exit 130' INT TERM
LOG=$DIR/log.txt
cd "$SPARK_ROOT"

case "$1" in
  displace)
    # Scripted plans (no planner); 3 injection phases x 0/2/5/10 cm. Without _st the executor's
    # goal-satisfied check and the recovery gate read the simulator predicate, as in the paper's
    # per-attempt-check rows (their runs predate SPARK_NO_ORACLE in this runner).
    read -r _ suite task arm <<< "$*"
    extra=(); case "$arm" in on*) extra=(--scene-diff --layered-recovery);; esac
    case "$arm" in onec*) extra+=(--event-captures);; esac
    orc=0; case "$arm" in *_st*) orc=1;; esac
    slide=40; case "$arm" in *_inst*) slide=0;; esac
    SPARK_NO_ORACLE=$orc python -m spark_bench.dyn.run --suite "$suite" --task-id "$task" --no-gemini --num-trials 5 \
      --trial-offset 0 --phases post_plan,mid_approach,post_grasp_plan --magnitudes-cm 0,2,5,10 \
      --slide-steps "$slide" --dyn-output "$DIR/dyn.json" "${extra[@]}" > "$LOG" 2>&1
    RESULT=$DIR/dyn.json;;
  *)
    read -r suite pert task off n arm <<< "$*"
    extra=(); case "$arm" in *wv*) extra+=(--wrist-verify);; esac; case "$arm" in *rb*) extra+=(--relation-bind);; esac
    ienv=(); case "$arm" in *_or*) ienv+=(SPARK_ORACLE_RETRY=1);; esac; case "$arm" in *_ph*) ienv+=(SPARK_PHRASES_V2=1);; esac
    case "$arm" in *_rg*) ienv+=(SPARK_PICK_REGROUND=1);; esac; case "$arm" in *_rgs*) ienv+=(SPARK_PICK_REGROUND_GATE_M=0.05);; esac
    # The arm alone sets these four flags: a value exported in the caller's shell must not leak in.
    env -u SPARK_ORACLE_RETRY -u SPARK_PHRASES_V2 -u SPARK_PICK_REGROUND -u SPARK_PICK_REGROUND_GATE_M "${ienv[@]}" \
      bash src/spark_bench/run_headline_libero_pro.sh "$suite" "$pert" "$n" --only-task-id "$task" \
      --trial-offset "$off" "${extra[@]}" --output-dir "$DIR" > "$LOG" 2>&1
    RESULT=$DIR/$suite.json;;
esac

# Not a result: no output file, a GPU memory failure anywhere in the log, a planner error, or a
# task-level "error" (the env failed to build). A trial that ran out of steps is a failed trial, not this.
ok=1
[ -s "$RESULT" ] || ok=0
grep -qE "CUDA OOM|CUDA error|out of memory|\[planner error\]" "$LOG" && ok=0
if [ $ok = 1 ] && [ "$1" != displace ]; then
  python - "$RESULT" <<'PY' || ok=0
import json, sys
r = json.load(open(sys.argv[1]))
sys.exit(any("error" in td for v in r.get("perturbations", {}).values() for td in v["task_details"]))
PY
fi
if [ $ok = 0 ]; then
  mkdir -p "$OUT/failed"
  mv "$DIR" "$OUT/failed/$KEY.$(date +%s)"
  echo "[item] FAILED $KEY (log in $OUT/failed/)" >&2
  exit 1
fi
echo "[item] done $KEY"
