#!/bin/bash
# NVIDIA Dynamic Scenes pipeline, SEEDED variant of run_nvidia.sh: reuses the shipped deterministic
# inputs (depth_anything_3 outputs, output_tapnet_da3.npz static tracks, result_XM.npz camera solve)
# and reruns everything from dynamic tracking onward -- the tight-reproduction path (the from-raw path
# is +-up-to-2dB on solver-sensitive scenes because TAPNext inference is nondeterministic).
# Recipe = the 26.90-mean configuration: MoSca CoTracker dynamic tracks, EDG_old seed, dyndense prep,
# factor-2 training, estimate_nvidia (fixed-v000, TTO, --novel-app split0). Legacy EDG behaviour is
# pinned in configs/nvidia.json (kabsch, no ARAP prune, bad-node 0.3, off-mask filter disabled).
# Usage: bash bashes/run_nvidia_seeded.sh <scene> [gpu]
set -euo pipefail
SCENE=${1:?scene}; GPU=${2:-0}
TAG=${RUN_TAG:-}   # optional suffix for the result dirs
cd "$(dirname "$0")/../Python"
REPO="$(cd .. && pwd)"; SD="$REPO/datasets/NVIDIA/$SCENE"
export XM_DATASET_CAT=NVIDIA XM_DATASET_NAME=$SCENE XM_DEPTH_SOURCE=da3
export CUDA_VISIBLE_DEVICES=$GPU CUDA_DEVICE_ORDER=PCI_BUS_ID EDG_NO_VIZ=1
run() { local n=$1; shift; local t0=$(date +%s); "$@"; echo "STAGE $SCENE $n $(( $(date +%s)-t0 ))s"; }

run dyn_track bash -c "DYN_EXTERNAL_TRACKS_NPZ=$SD/outputs_mosca/dynamic_dep=metric3d_cotracker_tap.npz \
    conda run -n tapnet --no-capture-output python dynamic_tracking.py"
run merge     conda run -n XM     --no-capture-output python test_extend_tracks.py --headless
run smooth    conda run -n XM     --no-capture-output python optimize_dynamic_tracks.py --input tracks_dynamic_merged.npz
run edg_old   conda run -n XM     --no-capture-output python EDG_old.py
run build_edg conda run -n XM     --no-capture-output python build_edg.py --merged --horizon-only --no-2d
run edg_solve conda run -n XM     --no-capture-output python EDG_test_fast.py --merged --seg-len 6 --n-seg 2 --n-runs 50
run complete  conda run -n tapnet --no-capture-output python complete_edg_table.py --merged

PREP_LOG=$(mktemp)
run dataset conda run -n XM --no-capture-output python train_4dgs_edg.py --dyn-source edg-good \
    --dyn-min-points 100000 --dyn-jitter 0.25 --gsplat-result-dir "$SD/outputs/gsplat_edg_target_0000${TAG}" \
    | tee "$PREP_LOG" > /dev/null
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
run train conda run -n gsplat --no-capture-output bash -c "$TRAIN --steps_scaler 1.0 --disable-viewer"
run eval  conda run -n gsplat --no-capture-output python estimate_nvidia.py \
    --gaussians "$SD/outputs/gsplat_edg_target_0000${TAG}/ckpts/final_rank0.pt" \
    --output-dir "$SD/outputs/nvidia_eval${TAG}" --output-json "$SD/outputs/nvidia_eval${TAG}/summary.json"
python3 -c "
import json; d=json.load(open('$SD/outputs/nvidia_eval${TAG}/summary.json'))
k=[x for x in ('novel_psnr','mPSNR','psnr') if x in d]
print(f\"RELDONE NVIDIA $SCENE {json.dumps({x: d[x] for x in list(d)[:6]})}\")"
