#!/usr/bin/env bash
# Camera benchmark
#   bash scripts/run_camera.sh all                    # all 7 datasets, 65 scenes
#   bash scripts/run_camera.sh Sintel [scene ...]     # one dataset: Sintel Replica TUM Syn4D NVIDIA Dycheck IMC
# Inputs are read from datasets/<Dataset>/<scene>/ and never written: each scene is solved in
# results/camera/<Dataset>/<scene>/ (links to the inputs + the solver outputs), its metrics go to metrics.json there.
# At the end the average of every dataset with results is printed and written to results/camera/averages.txt.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-python}
GPU=${GPU:-0}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
declare -A SCENES=(
  [Sintel]="alley_1 alley_2 ambush_2 ambush_4 ambush_5 ambush_6 ambush_7 bamboo_1 bamboo_2 bandage_1 bandage_2 cave_2 cave_4 market_2 market_5 market_6 mountain_1 shaman_2 shaman_3 sleeping_1 sleeping_2 temple_2 temple_3"
  [Replica]="office1_seq1 office1_seq2 office2_seq1 office2_seq2 room1_seq1 room1_seq2 room2_seq1 room2_seq2"
  [TUM]="tum1 tum2 tum3 tum4 tum5 tum6"
  [Syn4D]="bear1 bear2 bear3 bear4 casa1 casa2 casa3 casa4"
  [NVIDIA]="Balloon1 Balloon2 Jumping Playground Skating Truck Umbrella"
  [Dycheck]="apple block paper-windmill space-out spin teddy wheel"
  [IMC]="gate_00 gate_01 rome_00 rome_04 temple_00 temple_01"
)

solve() {
  local cat=$1 scene=$2
  local src=$ROOT/datasets/$cat/$scene out=$ROOT/results/camera/$cat/$scene
  [ -d "$src" ] || { echo "missing $src (see datasets/README.md)"; return 1; }
  rm -rf "$out"; mkdir -p "$out/outputs"
  for e in "$src"/*; do [ "$(basename "$e")" = outputs ] || ln -s "$e" "$out/"; done
  for e in "$src"/outputs/*; do ln -s "$e" "$out/outputs/"; done
  local depth=da3 flags="--no-da3-scale --json"
  [ "$cat" = IMC ] && { depth=unidepth; flags="--use-ceres --no-visualize --json"; }
  local t0=$SECONDS
  (cd "$ROOT/Python" && XM_DATASET_ROOT=$out XM_DATASET_CAT=$cat XM_DATASET_NAME=$scene XM_DEPTH_SOURCE=$depth \
     CUDA_VISIBLE_DEVICES=$GPU $PY static_solver.py $flags) > "$out/solve.log" 2>&1
  $PY - "$out/solve.log" "$out/metrics.json" $((SECONDS - t0)) <<'EOF'
import json, sys
log, out, secs = sys.argv[1:]
r = json.loads([l for l in open(log) if l.startswith("RESULT_JSON")][-1].split(":", 1)[1])
m = {k: r[k] for k in ("ATE_T", "RPE_T", "RPE_R", "ATE_R")}
json.dump(dict(m, seconds=int(secs)), open(out, "w"), indent=1)
print(" ".join(f"{k} {v:.4f}" for k, v in m.items()))
EOF
}

if [ "${1:-}" = all ]; then
  set -- ; for c in Sintel Replica TUM Syn4D NVIDIA Dycheck IMC; do for s in ${SCENES[$c]}; do set -- "$@" "$c/$s"; done; done
elif [ -n "${1:-}" ]; then
  cat=$1; shift
  [ -n "${SCENES[$cat]:-}" ] || { echo "unknown dataset $cat"; exit 1; }
  [ $# -gt 0 ] || set -- ${SCENES[$cat]}
  set -- $(for s in "$@"; do echo "$cat/$s"; done)
else
  awk 'NR > 1 { if (/^#/) print; else exit }' "$0"; exit 1
fi
$PY -c "import pyceres, pycolmap, teaserpp_python" 2>/dev/null || {
  echo "WARNING: pyceres / pycolmap / teaserpp_python not importable with $PY (needed for the Ceres refinement and"
  echo "TEASER++ alignment). Install them as in the README or set PYTHON to the right interpreter. Stopping."
  exit 1
}
for cs in "$@"; do
  echo -n "${cs}: "
  solve "${cs%%/*}" "${cs#*/}" || echo "FAILED (see results/camera/$cs/solve.log)"
done
$PY - "$ROOT/results/camera" <<'EOF'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
lines = ["dataset  scenes  ATE-T  RPE-T  RPE-R"]
for d in ("Sintel", "Replica", "TUM", "Syn4D", "NVIDIA", "Dycheck", "IMC"):
    ms = [json.loads(f.read_text()) for f in sorted((root / d).glob("*/metrics.json"))]
    if ms:
        lines.append(f"{d:8s} {len(ms):6d}  " + "  ".join(f"{sum(m[k] for m in ms) / len(ms):.3f}" for k in ("ATE_T", "RPE_T", "RPE_R")))
print("\n".join(lines))
(root / "averages.txt").write_text("\n".join(lines) + "\n")
EOF
