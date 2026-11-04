#!/bin/bash
# Re-run one LIBERO-PRO grid of the paper, item by item (5 trials per item), exactly as its item list.
# Needs the planner server (scripts/serve_planner.sh).
# Usage: bash scripts/repro_libero_pro.sh [table1|heldout|strict] [--suites "object goal spatial"]
#                                         [--perts "position task"] [--arm <arm>] [--out <dir>]
#   table1   repro/libero_pro_table1_items.txt   position + task, arm wvrb_or_ph_rgs (default)
#   heldout  repro/libero_pro_heldout_items.txt  language + object, arm wvrb_or_ph_rgs
#   strict   repro/libero_pro_strict_items.txt   position + task, arm wvrb_ph_rgs
# --suites / --perts keep only those items; --arm replaces every item's arm (tokens in
# scripts/run_item.sh). Default --out: results/libero_pro/<grid>.
# WORKERS=n runs n items at once on this machine (default 1). More GPUs or machines: start this
# script once per GPU (CUDA_VISIBLE_DEVICES=i) with the same --out; they share the queue. Re-running
# skips finished items and retries failed ones.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
GRID=table1; SUITES="object goal spatial"; PERTS="position task language object"; OUT=; ARM=
while [ $# -gt 0 ]; do
  case "$1" in
    table1|heldout|strict) GRID=$1; shift;;
    --suites) SUITES=$2; shift 2;;
    --perts) PERTS=$2; shift 2;;
    --out) OUT=$2; shift 2;;
    --arm) ARM=$2; shift 2;;
    *) echo "unknown argument $1" >&2; exit 2;;
  esac
done
OUT=${OUT:-$SPARK_ROOT/results/libero_pro/$GRID}
curl -sf --max-time 5 "$OPENAI_BASE_URL/models" > /dev/null \
  || { echo "planner not reachable at $OPENAI_BASE_URL (start scripts/serve_planner.sh)" >&2; exit 1; }

awk -v S=" $SUITES " -v P=" $PERTS " -v A="$ARM" 'index(S, " " $1 " ") && index(P, " " $2 " ") {if (A != "") $6 = A; print}' \
    "$SPARK_ROOT/repro/libero_pro_${GRID}_items.txt" \
  | xargs -P "${WORKERS:-1}" -L1 bash "$SPARK_ROOT/scripts/run_item.sh" "$OUT"

n=$(find "$OUT" -mindepth 2 -maxdepth 2 -name '*.json' ! -name '*partial*' ! -path "$OUT/failed/*" | wc -l)
echo "[repro] $n item results in $OUT"
echo "[repro] summary: python -m spark_bench.summarize $OUT --cells --ci --timing --suite-means"
