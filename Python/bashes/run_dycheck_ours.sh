#!/usr/bin/env bash
# Camera-accuracy "Ours" (XM-SfM) on the DyCheck iPhone dataset.
# Per scene: DA3 depth -> static TAPNext tracking -> XM static solver.
# Config is data-driven from Python/configs/dycheck.json (via utils.definitions).
#
# Usage:  ./run_dycheck_ours.sh [scene ...]      # default: all 7 scenes
#   NODEPTH=1  skip depth (reuse outputs/depth_anything_3)
#   NOTRACK=1  skip tracking (reuse outputs/output_tapnet_da3.npz)
set -e
source "$HOME/miniconda3/etc/profile.d/conda.sh"
cd "$(dirname "${BASH_SOURCE[0]}")/.."   # -> Python/
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

SCENES=("$@")
if [ ${#SCENES[@]} -eq 0 ]; then
  SCENES=(apple block paper-windmill space-out spin teddy wheel)
fi

OUT="dycheck_ours_results.jsonl"; touch "$OUT"

for scene in "${SCENES[@]}"; do
  export XM_DATASET_CAT=Dycheck XM_DATASET_NAME="$scene" XM_DEPTH_SOURCE=da3
  echo "==================== $scene ===================="

  if [ "${NODEPTH:-0}" != "1" ]; then
    conda activate da3
    python get_depths_clean.py
  fi

  if [ "${NOTRACK:-0}" != "1" ]; then
    conda activate tapnet
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python static_tracking_online.py
  fi

  conda activate XM
  echo "solve Dycheck/$scene"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python static_solver.py --no-da3-scale --json \
    | grep "^RESULT_JSON" \
    | sed "s#^RESULT_JSON: #{\"scene\":\"$scene\",\"r\":#; s#\$#}#" >> "$OUT"
  echo "---- last result ----"; tail -1 "$OUT"
done
echo "done -> Python/$OUT"
