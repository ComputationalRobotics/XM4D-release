#!/usr/bin/env bash
# Reproduce the "Ours" accuracy numbers in Notes/data.tex (release configuration).
#
# Per-dataset config is fully data-driven from Python/configs/*.json (selected by
# XM_DATASET_CAT/NAME via utils.definitions), including the GT-vs-DA3 intrinsics
# choice (USE_DA3_INTRINSICS):
#   depth      : Depth-Anything-3   (XM_DEPTH_SOURCE=da3)
#   intrinsics : DA3-predicted for Sintel/Replica/TUM; GT for Syn4D  (per config)
#   masks      : SAM2 masks/ dir (GT masks for synthetic Syn4D); dilated per
#                MASK_DILATE_PX (+10px, expand) for the static query exclusion + filter
#   params     : per-dataset tuned values in Python/configs/*.json
#   solver     : static_solver.py --no-da3-scale
#
# Determinism: OMP/MKL pinned to 1. A few hard large-motion Sintel scenes
# (cave_2, temple_3, ambush_6) carry ~2-3% run-to-run noise from multi-threaded Ceres.
#
# Usage:  RETRACK=1 ./run_release_ours.sh   # regenerate tracks then solve (GPU)
#         ./run_release_ours.sh             # solve on existing tracks (default)
set -e
source "$HOME/miniconda3/etc/profile.d/conda.sh"
cd "$(dirname "${BASH_SOURCE[0]}")/.."   # -> Python/
RETRACK="${RETRACK:-0}"
OUT="release_ours_results.jsonl"; : > "$OUT"

declare -A SCENES=(
  [Sintel]="alley_1 alley_2 ambush_2 ambush_4 ambush_5 ambush_6 ambush_7 bamboo_1 bamboo_2 bandage_1 bandage_2 cave_2 cave_4 market_2 market_5 market_6 mountain_1 shaman_2 shaman_3 sleeping_1 sleeping_2 temple_2 temple_3"
  [Replica]="office1_seq1 office1_seq2 office2_seq1 office2_seq2 room1_seq1 room1_seq2 room2_seq1 room2_seq2"
  [Syn4D]="bear1 bear2 bear3 bear4 casa1 casa2 casa3 casa4"
  [TUM]="tum1 tum2 tum3 tum4 tum5 tum6"
)

for cat in Sintel Replica Syn4D TUM; do
  for scene in ${SCENES[$cat]}; do
    export XM_DATASET_CAT="$cat" XM_DATASET_NAME="$scene" XM_DEPTH_SOURCE=da3

    if [ "$RETRACK" = "1" ]; then
      conda activate tapnet
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python static_tracking_online.py
    fi

    conda activate XM
    echo "solve ${cat}/${scene}"
    # intrinsics/params come from configs/*.json (no env override needed)
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python static_solver.py --no-da3-scale --json \
      | grep "^RESULT_JSON" | sed "s#^RESULT_JSON: #{\"cat\":\"$cat\",\"scene\":\"$scene\",\"r\":#; s#\$#}#" >> "$OUT"
  done
done
echo "done -> Python/$OUT"
