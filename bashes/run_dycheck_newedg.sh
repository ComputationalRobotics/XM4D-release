#!/bin/bash
# DyCheck FULL pipeline, self-contained in this repo (no intermediates imported from elsewhere):
# regenerates depth, depth variants (lidar_filled active), static tracking+solve, dynamic tracking,
# EDG (new parameters in configs/dycheck.json), 4DGS training, and the standard TTO eval
# (estimate_ours_render_tto, novel-app base; split0 is NVIDIA-only and NOT used here).
# Dataset assets required per scene: images/ masks/ cameras/ depth/ (GT lidar) novel/ [frame_map.txt].
# Usage: bash bashes/run_dycheck_newedg.sh <scene> [gpu]
set -eu
SCENE=${1:?scene}; GPU=${2:-0}
cd "$(dirname "$0")/../Python"
export XM_DATASET_CAT=Dycheck XM_DATASET_NAME=$SCENE XM_DEPTH_VARIANT=lidar_filled
export CUDA_VISIBLE_DEVICES=$GPU CUDA_DEVICE_ORDER=PCI_BUS_ID EDG_NO_VIZ=1
SD="$(cd .. && pwd)/datasets/Dycheck/$SCENE"

# FAST MODE: depth / static tracks / camera solve / FORWARD dynamic tracks are pre-seeded by the
# caller (tracks_dynamic_backward.npz intentionally absent -> test_extend_tracks merges forward-only).
# Pipeline reruns from track merging onward with the new EDG parameters.
conda run -n XM     --no-capture-output python test_extend_tracks.py --headless
conda run -n XM     --no-capture-output python optimize_dynamic_tracks.py
# NOTE: no EDG_old -- build_edg selects the EDG; complete_edg_table writes the new solve into the
# seeded results_all.npz base (non-EDG keys carried over, EDG keys fully replaced).
conda run -n XM     --no-capture-output python build_edg.py --gt-depth
conda run -n XM     --no-capture-output python EDG_test_fast.py --merged --seg-len 10 --n-seg 2 --n-runs 50
conda run -n tapnet --no-capture-output python complete_edg_table.py --merged

PREP_LOG=$(mktemp)
conda run -n XM --no-capture-output python train_4dgs_edg.py --dyn-source edg-good \
  --gsplat-result-dir "$SD/outputs/gsplat_newedg" | tee "$PREP_LOG"
TRAIN=$(grep -A1 "Recommended gsplat 4DGS training command" "$PREP_LOG" | tail -1)
TRAIN=$(echo "$TRAIN" | sed "s/CUDA_VISIBLE_DEVICES=[0-9]*/CUDA_VISIBLE_DEVICES=$GPU/")
conda run -n gsplat --no-capture-output bash -c "$TRAIN --disable-viewer"

OUT=$SD/outputs/ours_render_eval_newedg
conda run -n gsplat --no-capture-output python estimate_ours_render_tto.py \
  --scene-dir "$SD" --gaussians "$SD/outputs/gsplat_newedg/ckpts/final_rank0.pt" \
  --camera-source colmap --metric-mask alpha --color-calib none --novel-align tto \
  --use-covisible-mask --save-images \
  --output-dir "$OUT" --output-json "$OUT/metrics_tto.json"
python3 -c "
import json; d=json.load(open('$OUT/metrics_tto.json'))['novel']
print('DYCHECK-NEWEDG[$SCENE] mPSNR:', round(d['mPSNR'],3))"
