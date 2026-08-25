#!/usr/bin/env bash

DATASET_CATEGORY="Replica"  # Replica or Sintel
DATASET_ROOT="../datasets/${DATASET_CATEGORY}"

SCENES=(
    "office1_seq1"
    "office1_seq2"
    "office2_seq1"
    "office2_seq2"
    "room1_seq1"
    "room1_seq2"
    "room2_seq1"
    "room2_seq2"
)

# SCENES=(
#     "alley_1"
#     "alley_2"
#     "ambush_2"
#     "ambush_4"
#     "ambush_5"
#     "ambush_6"
#     "ambush_7"
#     "bamboo_1"
#     "bamboo_2"
#     "bandage_1"
#     "bandage_2"
#     "cave_2"
#     "cave_4"
#     "market_2"
#     "market_5"
#     "market_6"
#     "mountain_1"
#     "shaman_2"
#     "shaman_3"
#     "sleeping_1"
#     "sleeping_2"
#     "temple_2"
#     "temple_3"
# )

# SCENES=(
#     "bear_0"
#     "bear_1"
#     "bear_2"
#     "bear_3"
# )

# SCENES=(
#     "bear1"
#     "bear2"
#     "bear3"
#     "bear4"
#     "casa1"
#     "casa2"
#     "casa3"
#     "casa4"
# )

# SCENES=(
#     "Capture4_clip01"
#     "Capture1_clip01_new"
#     "Capture2_clip01_new"
#     "Capture3_clip01_new"
#     "Capture4_clip01_new"
# )

# SCENES=(
#     "tum1"
#     "tum2"
#     "tum3"
#     "tum4"
#     "tum5"
#     "tum6"
# )

for scene in "${SCENES[@]}"; do
    scene_dir="${DATASET_ROOT}/${scene}"
    sparse_dir="${scene_dir}/output_colmap/sparse"
    echo "Evaluating ${scene_dir}"
    if [[ ! -d "${scene_dir}" ]]; then
        echo "Skipping missing scene: ${scene_dir}" >&2
        continue
    fi
    python ./eval_colmap.py \
        --scene-dir "${scene_dir}" \
        --sparse-dir "${sparse_dir}" \
        --gt-type "${DATASET_CATEGORY,,}" \
        --show-alignment
done