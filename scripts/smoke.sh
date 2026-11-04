#!/bin/bash
# End-to-end smoke test on one GPU (about 15 minutes). Stops at the first failure.
# Needs: scripts/setup.sh done, hf auth login done, and the planner server running
# (scripts/serve_planner.sh) for steps 5-6.
# Usage: bash scripts/smoke.sh [out_dir]   (default out/smoke; cleared first)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$SPARK_ROOT"
OUT=${1:-$SPARK_ROOT/out/smoke}
rm -rf "$OUT"; mkdir -p "$OUT"
step() { echo; echo "[smoke] $*"; }

step "1/6 unit tests"
# unit tests run without the planner variables env.sh sets
env -u SPARK_LLM_BACKEND -u OPENAI_BASE_URL -u OPENAI_API_KEY \
  python -m pytest src/spark_core/tests src/spark_bench/tests -q -m "not slow" -p no:cacheprovider

step "2/6 imports, LIBERO-PRO data, headless GL"
python - <<'PY'
import os
import libero
from libero.libero import benchmark, get_libero_path
import mujoco
import sam3.model_builder  # noqa: F401  a broken sam3 install fails here, not mid-run
path = list(libero.__path__)[0]
assert "LIBERO-PRO" not in path, f"libero resolves to LIBERO-PRO ({path}); use upstream LIBERO"
task = benchmark.get_benchmark_dict()["libero_object"]().get_task(0)
bddl = os.path.join(get_libero_path("bddl_files"), "libero_object_swap", task.bddl_file)
init = os.path.join(get_libero_path("init_states"), "libero_object_swap", task.init_states_file)
for f in (bddl, init):
    assert os.path.isfile(f), f"missing LIBERO-PRO file {f}"
print("libero", path)
print("perturbed bddl", bddl)
print("perturbed init", init)
ctx = mujoco.GLContext(64, 64)
ctx.make_current()
ctx.free()
print("EGL context ok")
PY

step "3/6 SAM3 checkpoint (gated: accept the license at https://huggingface.co/facebook/sam3)"
python - <<'PY'
import os
from huggingface_hub import hf_hub_download
pin = os.environ["SAM3_CKPT_REVISION"]   # spark_core.sam3 loads this revision, not main
try:
    hf_hub_download("facebook/sam3", "sam3.pt", revision=pin)
except Exception as e:
    raise SystemExit(f"SAM3 checkpoint download failed ({type(e).__name__}: {e}).\n"
                     "Accept the license at https://huggingface.co/facebook/sam3, then run: hf auth login")
print("sam3 checkpoint", pin)
PY

step "4/6 LLM-free trial (scripted plan): env, LIBERO-PRO suite, SAM3, executor"
python -m spark_bench.run_libero_pro --suite object --perturbation position --num-trials 1 \
  --only-task-id 0 --no-gemini --output-dir "$OUT/nollm" 2>&1 | tail -5
python - "$OUT/nollm/object.json" <<'PY'
import json, sys
for v in json.load(open(sys.argv[1]))["perturbations"].values():
    for td in v["task_details"]:
        assert "error" not in td, f"trial error: {td['error']}"
        print(f"scripted {td['task_name']}: success rate {td['success_rate']:.2f}")
PY

step "5/6 planner at $OPENAI_BASE_URL (waits up to 10 min for the server to load)"
for _ in $(seq 60); do
  curl -sf --max-time 5 "$OPENAI_BASE_URL/models" | grep -qF "\"$SPARK_LLM_MODEL\"" && break
  sleep 10
done
python - <<'PY'
from spark_core.planner import SPARKPlanner
score = SPARKPlanner().generate_score("put the ketchup in the basket", keypoint_labels=["ketchup", "basket"])
assert isinstance(score, dict) and score.get("tree"), f"no plan tree in planner reply: {score}"
print("plan root:", score["tree"].get("type"))
PY

step "6/6 two headline trials (object position, task 1)"
bash src/spark_bench/run_headline_libero_pro.sh object position 2 --only-task-id 1 \
  --wrist-verify --relation-bind --output-dir "$OUT/headline" > "$OUT/headline.log" 2>&1 || true

python - "$OUT" <<'PY'
import json, os, sys
out = sys.argv[1]
assert "[planner error]" not in open(os.path.join(out, "headline.log")).read(), "planner error in headline.log"
r = json.load(open(os.path.join(out, "headline", "object.json")))
for pert, v in r["perturbations"].items():
    for td in v["task_details"]:
        assert "error" not in td, f"headline task error: {td['error']}"
        print(f"headline {pert} {td['task_name']}: success rate {td['success_rate']:.2f}")
PY
echo
echo "[smoke] passed (success counts above are informational; two trials are noise)"
