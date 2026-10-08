#!/bin/bash
# NVIDIA Dynamic Scenes pipeline (per scene), reproducing the 26.90-mean configuration:
#   dyndense prep (--dyn-min-points 100000 --dyn-jitter 0.25) + configs/nvidia.json
#   + first-frame-split eval (estimate_nvidia.py, --novel-app split0).
# Envs: tracking runs in `tapnet`, solver/EDG/prep in `XM`, training/eval in `gsplat`.
# Usage: bash bashes/run_nvidia.sh <scene> [gpu]      (e.g. bash bashes/run_nvidia.sh Skating 0)
set -eu
SCENE=${1:?scene name, e.g. Skating}
GPU=${2:-0}
cd "$(dirname "$0")/../Python"
export XM_DATASET_CAT=NVIDIA XM_DATASET_NAME=$SCENE XM_DEPTH_SOURCE=da3
export CUDA_VISIBLE_DEVICES=$GPU CUDA_DEVICE_ORDER=PCI_BUS_ID
SD="$(cd .. && pwd)/datasets/NVIDIA/$SCENE"

conda run -n da3 --no-capture-output python get_depths_clean.py
conda run -n tapnet --no-capture-output python static_tracking_online.py
conda run -n XM  --no-capture-output python static_solver.py
# dynamic 2D tracks: MoSca CoTracker (offline) injected via DYN_EXTERNAL_TRACKS_NPZ, lifted with our depth+poses
DYN_EXTERNAL_TRACKS_NPZ=$SD/outputs_mosca/dynamic_dep=metric3d_cotracker_tap.npz \
conda run -n tapnet --no-capture-output python dynamic_tracking.py
conda run -n XM  --no-capture-output python test_extend_tracks.py --headless
conda run -n XM  --no-capture-output python optimize_dynamic_tracks.py --input tracks_dynamic_merged.npz
EDG_NO_VIZ=1 conda run -n XM --no-capture-output python EDG_old.py
conda run -n XM  --no-capture-output python build_edg.py --merged --horizon-only --no-2d
conda run -n XM  --no-capture-output python EDG_test_fast.py --merged --seg-len 6 --n-seg 2 --n-runs 50
conda run -n tapnet --no-capture-output python complete_edg_table.py --merged

# 4DGS dataset prep (dyndense) -> capture the recommended train command, force factor 2 + this GPU
PREP_LOG=$(mktemp)
conda run -n XM --no-capture-output python train_4dgs_edg.py --dyn-source edg-good \
  --dyn-min-points 100000 --dyn-jitter 0.25 | tee "$PREP_LOG"
TRAIN=$(grep -A1 "Recommended gsplat 4DGS training command" "$PREP_LOG" | tail -1)
TRAIN=${TRAIN//--data-factor 1/--data-factor 2}
TRAIN=$(echo "$TRAIN" | sed "s/CUDA_VISIBLE_DEVICES=[0-9]*/CUDA_VISIBLE_DEVICES=$GPU/")
python3 - <<EOF
import cv2, glob, os
gs = "$SD/outputs/gs_edg_target_0000"
os.makedirs(gs+"/images_2", exist_ok=True)
for p in sorted(glob.glob(gs+"/images/*")):
    im = cv2.imread(p)
    cv2.imwrite(gs+"/images_2/"+os.path.basename(p),
                cv2.resize(im, (im.shape[1]//2, im.shape[0]//2), interpolation=cv2.INTER_AREA))
EOF
conda run -n gsplat --no-capture-output bash -c "$TRAIN --steps_scaler 1.0 --disable-viewer"

# headline eval: fixed-v000, TTO, first-frame-split appearance
conda run -n gsplat --no-capture-output python estimate_nvidia.py
echo "DONE $SCENE -> $SD/outputs/nvidia_eval/summary.json"
