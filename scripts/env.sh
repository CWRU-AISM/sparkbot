# Sourced by every script here (and by hand before running the bench scripts directly):
#   source scripts/env.sh
# Sets the protocol environment. Every value can be overridden by exporting it first.
SPARK_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export SPARK_ROOT

# The uv environment from scripts/setup.sh, unless one is already active.
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$SPARK_ROOT/.venv/bin/activate" ]; then
  source "$SPARK_ROOT/.venv/bin/activate"
fi

# Pins. The planner weights, SAM3 checkpoint and LIBERO-PRO data the reported runs used.
export QWEN_MODEL=${QWEN_MODEL:-MIRALABS/Qwen3.8-27B-W4A16-AutoRound}
export QWEN_REVISION=${QWEN_REVISION:-a6754efcde1ea362616353759f0c97507775a5a9}
export SAM3_CKPT_REVISION=${SAM3_CKPT_REVISION:-3c879f39826c281e95690f02c7821c4de09afae7}

# LIBERO reads its config from here, never from the per-user default (setup.sh writes it).
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-$SPARK_ROOT/.libero}
# `libero` is imported from the checkout root as the runs did (its setup.py installs no package).
export PYTHONPATH=$SPARK_ROOT/third_party/LIBERO${PYTHONPATH:+:$PYTHONPATH}
# Packages in the user site directory must not shadow the pinned ones.
export PYTHONNOUSERSITE=1

# Headless rendering.
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-${CUDA_VISIBLE_DEVICES%%,*}}
fi

# Planner: the vLLM server from serve_planner.sh.
export SPARK_LLM_BACKEND=${SPARK_LLM_BACKEND:-openai}
export SPARK_LLM_MODEL=${SPARK_LLM_MODEL:-$QWEN_MODEL}
export OPENAI_BASE_URL=${OPENAI_BASE_URL:-http://127.0.0.1:${PORT:-8000}/v1}
export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
