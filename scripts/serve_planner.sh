#!/bin/bash
# Serve the planner (Qwen3.8-27B W4A16, pinned revision) with vLLM on $PORT (default 8000), with the
# flags the reported runs used. Runs in the foreground; start it in its own terminal or with &.
# Usage: bash scripts/serve_planner.sh [extra vllm args...]
# Env: VLLM_MEM (GPU memory fraction, default 0.55; 0.62 on a 40 GB GPU, 0.30 on 141 GB), MAX_SEQS (default 8).
# vLLM pins its own torch, so it runs in a separate cached uv environment, never the sim one.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
deactivate 2>/dev/null || true
# Node-local compile caches: several processes compiling into one shared-filesystem cache fail
# with "Stale file handle".
CACHE=$(dirname "$(mktemp -u)")/spark_vllm_${USER:-user}   # $TMPDIR, else the system temp dir
export TRITON_CACHE_DIR=$CACHE/triton VLLM_CACHE_ROOT=$CACHE/vllm TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor
mkdir -p "$CACHE"

exec uv run --no-project --isolated --python 3.12 \
  --with vllm==0.30.0 --with transformers==5.17.0 --with "huggingface-hub>=1.5,<2" \
  python -m vllm.entrypoints.openai.api_server \
  --model "$QWEN_MODEL" --revision "$QWEN_REVISION" --tokenizer-revision "$QWEN_REVISION" --served-model-name "$SPARK_LLM_MODEL" \
  --host 127.0.0.1 --port "${PORT:-8000}" \
  --max-model-len 32768 --max-num-seqs "${MAX_SEQS:-8}" --gpu-memory-utilization "${VLLM_MEM:-0.55}" \
  --limit-mm-per-prompt '{"image": 4}' --default-chat-template-kwargs '{"enable_thinking": false}' "$@"
