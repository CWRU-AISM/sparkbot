#!/bin/bash
# Re-run the LIBERO-Displace numbers (no planner in the loop).
# Usage: bash scripts/repro_dyn.sh displace <arm> [out_root]      (out_root default results/dyn)
#   displace <arm>        N10-N12, LIBERO-Dyn: fixed scripted plan, object displaced 0/2/5/10 cm mid-episode, 3 object
#                         tasks x 3 phases x 5 initial states. Arms (scripts/run_item.sh): on|off slide with
#                         the per-attempt success check; onec_st|off_st slide, strict; onec_st_inst|off_st_inst
#                         one-step displacement, strict. dyn.metrics groups displace runs only by adapt vs off,
#                         so give each block (slide, slide strict, instantaneous strict) its own out_root.
# WORKERS=n and re-running behave as in repro_libero_pro.sh.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
KIND=${1:?kind: displace}; ARM=${2:?arm}; OUT=${3:-$SPARK_ROOT/results/dyn}
case "$KIND:$ARM" in
  displace:*) items=$(printf 'displace object %s '"$ARM"'\n' 0 1 2);;
  *) echo "unknown kind/arm $KIND $ARM" >&2; exit 2;;
esac
echo "$items" | xargs -P "${WORKERS:-1}" -L1 bash "$SPARK_ROOT/scripts/run_item.sh" "$OUT"
echo "[repro] results in $OUT/*/dyn.json"
