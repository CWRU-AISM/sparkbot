#!/bin/bash
# The LIBERO-PRO headline protocol as one command:
#   src/spark_bench/run_headline_libero_pro.sh <suite> <perturbation> [num_trials] [extra runner args...]
# suite: object | goal | spatial; perturbation: position | task | language | object.
# Needs the protocol env (upstream LIBERO, sam3, LIBERO_CONFIG_PATH pointing at the LIBERO-PRO data) and the
# planner served on an OpenAI-compatible endpoint (scripts/serve_planner.sh):
#   SPARK_LLM_BACKEND=openai OPENAI_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=EMPTY
# The runner never reads the simulator's success predicate before the final scoring read, plans every trial
# fresh, and fails a trial on a planner error.
set -eu
SUITE=${1:?suite: object|goal|spatial}
PERT=${2:?perturbation: position|task|language|object}
TRIALS=${3:-50}
shift 3 2>/dev/null || shift $#
export SPARK_LLM_BACKEND=${SPARK_LLM_BACKEND:-openai}   # model: SPARK_LLM_MODEL, default the Qwen model in spark_core/llm.py
export MUJOCO_GL=${MUJOCO_GL:-egl} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
# --adaptive-prompts: the VLM proposes 3 phrasings per object from the instruction and SAM3 keeps the one with
# a single confident detection. --wrist-verify and --relation-bind: re-ground weak picks from the wrist image,
# and choose among twin pick targets by the instruction's spatial relation.
exec python -m spark_bench.run_libero_pro --suite "$SUITE" --perturbation "$PERT" \
  --num-trials "$TRIALS" --no-bddl-hints --adaptive-prompts --wrist-verify --relation-bind "$@"
