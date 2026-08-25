#!/usr/bin/env bash
DATASETS="Lab"
SCENES=(
    "Capture1_clip01_new"
    "Capture2_clip01_new"
    "Capture3_clip01_new"
    "Capture4_clip01_new"
)

conda activate da3
for scene in "${SCENES[@]}"; do
    echo "Processing ${DATASETS}/${scene}"
    export XM_DATASET_CAT="${DATASETS}"
    export XM_DATASET_NAME="${scene}"
    python get_depths_clean.py 

conda activate tapnet
for scene in "${SCENES[@]}"; do
    echo "Processing ${DATASETS}/${scene}"
    export XM_DATASET_CAT="${DATASETS}"
    export XM_DATASET_NAME="${scene}"
    python static_tracking_online.py 
    python dynamic_tracking_online_segmented.py

conda activate XM
python static_solver.py