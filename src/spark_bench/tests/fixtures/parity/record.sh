#!/bin/bash
# Refactor parity record (OPEN_SOURCE_PLAN 6.2 steps 1-2). Run from a repo root in the protocol env, then
# compare each <out_dir>/<part>/summary.json with base_0fd07d2/<part>.json (summarize.py --compare). The
# committed baseline is reconcile-jul07 0fd07d2 on one RTX 4090 (openvla_env); WP0 b09be75 matched it
# exactly. A part with a CUDA out-of-memory error in its trial records is invalid: rerun it.
# The fixture env had opencv 4.12; with the pinned opencv 5.0 the wrist-verify crop-grid images
# (cv2.resize, cv2.putText) differ, so the requests part shows 5 image diffs (requests 1, 4, 8, 13, 16)
# there. Compare code versions in one env.
#   bash src/spark_bench/tests/fixtures/parity/record.sh <out_dir> <part>
# part: nollm_a | nollm_b (object/position tasks 0-4 | 5-9, 3 trials, scripted plan, no LLM)
#       requests (object/position tasks 0-4, 1 trial, headline flags, fake LLM recording every request)
# RUNNER: python module of the LIBERO-PRO runner (it moved in WP0).
# sam3 must be importable (pip install -e of the pinned checkout, or its root on PYTHONPATH); since WP1
# spark_core.sam3 no longer adds a checkout to sys.path.
set -eu
OUT=$1; PART=$2
RUNNER=${RUNNER:-spark_bench.run_libero_pro}
HERE=$(cd "$(dirname "$0")" && pwd)
PORT=${PORT:-18765}
mkdir -p "$OUT/$PART"
python "$HERE/fake_llm.py" $PORT "$OUT/$PART/llm.jsonl" & FAKE=$!
trap 'kill $FAKE' EXIT
export PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH}
export SPARK_LLM_BACKEND=openai OPENAI_BASE_URL=http://127.0.0.1:$PORT/v1 OPENAI_API_KEY=EMPTY SPARK_GEMINI_MODEL=fake
export SPARK_GEMINI_CACHE=0 SPARK_NO_ORACLE=1 SPARK_BT_FROZEN=1 SPARK_REQUIRE_PLANNER=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
for _ in $(seq 30); do curl -s -o /dev/null localhost:$PORT && break; sleep 1; done
kill -0 $FAKE || { echo "fake LLM did not start on port $PORT"; exit 1; }
run() { python -m $RUNNER --suite object --perturbation position --output-dir "$OUT/$PART" --no-bddl-hints \
          --relation-bind "$@"; }
keep() { mv "$OUT/$PART/object.json" "$OUT/$PART/t$1.json"; rm -f "$OUT/$PART/object.partial.json"; }
case $PART in
  nollm_a) for t in 0 1 2 3 4; do run --num-trials 3 --only-task-id $t --no-gemini; keep $t; done ;;
  nollm_b) for t in 5 6 7 8 9; do run --num-trials 3 --only-task-id $t --no-gemini; keep $t; done ;;
  requests) for t in 0 1 2 3 4; do run --num-trials 1 --only-task-id $t --adaptive-prompts --wrist-verify; keep $t; done ;;
esac
python "$HERE/summarize.py" "$OUT/$PART" > "$OUT/$PART/summary.json"
