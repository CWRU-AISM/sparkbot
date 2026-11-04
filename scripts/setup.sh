#!/bin/bash
# One-time setup: the uv environment (.venv) with the pinned sim stack and the patched SAM3 submodule,
# the LIBERO-PRO perturbed suites at the revision the reported runs used, and the LIBERO config.
# Safe to re-run.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$SPARK_ROOT"

LIBERO_PRO_DATA_REV=3a33dc4cfbea7ff5eac6764e9c557500972b6259   # HF dataset zhouxueyang/LIBERO-Pro
# sha256 over bddl_files, init_files, envs, assets of the LIBERO-PRO tree the runs read
LIBERO_PRO_TREE_SHA=165b381759ea2ddd6f13be4c37844e50fe3158c1cfc5525c24c17a4f3e594c31
PRO=$SPARK_ROOT/third_party/LIBERO-PRO/libero/libero

git submodule update --init
# sam3 imports pkg_resources, which setuptools >= 81 removed.
git -C third_party/sam3 apply --reverse --check ../../scripts/sam3.patch 2>/dev/null \
  || git -C third_party/sam3 apply ../../scripts/sam3.patch

echo "[setup] uv sync"
uv sync --extra bench --extra dev
source .venv/bin/activate

echo "[setup] LIBERO-PRO perturbed suites at $LIBERO_PRO_DATA_REV"
python - "$PRO" "$LIBERO_PRO_DATA_REV" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("zhouxueyang/LIBERO-Pro", repo_type="dataset", revision=sys.argv[2],
                  allow_patterns=["bddl_files/*", "init_files/*"], local_dir=sys.argv[1])
PY
got=$(cd "$PRO" && find bddl_files init_files envs assets -type f ! -path '*__pycache__*' \
      | LC_ALL=C sort | xargs -d '\n' sha256sum | sha256sum | cut -d' ' -f1)
if [ "$got" != "$LIBERO_PRO_TREE_SHA" ]; then
  echo "[setup] LIBERO-PRO data fingerprint $got does not match the runs' $LIBERO_PRO_TREE_SHA" >&2
  exit 1
fi

mkdir -p "$LIBERO_CONFIG_PATH" third_party/LIBERO-PRO/libero/datasets
cat > "$LIBERO_CONFIG_PATH/config.yaml" <<EOF2
benchmark_root: $PRO
bddl_files: $PRO/bddl_files
init_states: $PRO/init_files
datasets: $SPARK_ROOT/third_party/LIBERO-PRO/libero/datasets
assets: $PRO/assets
EOF2

python - <<'PY'
import libero, mujoco, numpy, robosuite, torch
import sam3.model_builder  # noqa: F401
import spark_core.sam3  # noqa: F401
import spark_bench.run_libero_pro  # noqa: F401
path = list(libero.__path__)[0]
assert "LIBERO-PRO" not in path, f"libero resolves to LIBERO-PRO ({path}); the protocol uses upstream LIBERO"
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "| mujoco", mujoco.__version__,
      "| robosuite", robosuite.__version__, "| numpy", numpy.__version__)
PY
echo "[setup] done. Next: hf auth login (accept the facebook/sam3 license first), then bash scripts/smoke.sh"
