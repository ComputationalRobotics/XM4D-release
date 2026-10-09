#!/bin/bash
# DyCheck FAST pipeline (2026-10 recipe, self-contained in this repo). Reproduces the 19.47-mean
# configuration: CoWTracker+TAPNext merge (unattached pool, no in-merge smoothing), lift-only 3D,
# EDG with rotation sync + ARAP pruning, completion with node-2D weight (configs/dycheck.json),
# 25k-step 4DGS (steps-scaler 25/35), TTO eval with covisible mask.
# Seeded per scene (deterministic repro): depths/, result_XM.npz, tracks_dynamic_only.npz, masks/.
# From-scratch instead: run get_depths_clean -> build_depth_variants -> static_tracking_online ->
# static_solver -> dynamic_tracking first (see Notes/PIPELINE.md; TAPNext stages are nondeterministic).
# Usage: bash bashes/run_dycheck_fast.sh <scene> [gpu] [steps_scaler]
set -euo pipefail
SCENE=${1:?scene}; GPU=${2:-0}; SCALER=${3:-0.7142857142857143}
TAG=${RUN_TAG:-}   # optional suffix for the result dirs (keeps earlier runs intact)
cd "$(dirname "$0")/../Python"
REPO="$(cd .. && pwd)"; SD="$REPO/datasets/Dycheck/$SCENE"; O="$SD/outputs"
export XM_DATASET_CAT=Dycheck XM_DATASET_NAME=$SCENE XM_DEPTH_VARIANT=lidar_filled
export CUDA_VISIBLE_DEVICES=$GPU CUDA_DEVICE_ORDER=PCI_BUS_ID EDG_NO_VIZ=1
run() { local n=$1; shift; local t0=$(date +%s); "$@"; echo "STAGE $SCENE $n $(( $(date +%s)-t0 ))s"; }

run merge     conda run -n XM     --no-capture-output python cow_attach_tracks.py --reuse-skeleton --no-video --no-optimize
run lift      conda run -n XM     --no-capture-output python optimize_dynamic_tracks.py --input tracks_dynamic_merged.npz --iters 0 --output tracks_dynamic_merged_lift.npz
run build_edg conda run -n XM     --no-capture-output python build_edg.py --gt-depth --tracks tracks_dynamic_merged_lift.npz
run edg_solve conda run -n XM     --no-capture-output python EDG_test_fast.py --tracks tracks_dynamic_merged_lift.npz
run complete  conda run -n tapnet --no-capture-output python complete_edg_table.py --tracks tracks_dynamic_merged_lift.npz
run dataset   conda run -n XM     --no-capture-output python train_4dgs_edg.py --dyn-source edg-good --dyn-lift-per-frame
run train     conda run -n gsplat --no-capture-output python "$REPO/deps/gsplat/examples/simple_trainer.py" default \
    --data-dir "$O/gs_edg_target_0000" --result-dir "$O/gsplat_fast${TAG}" --data-factor 1 \
    --edg-results "$O/results_all.npz" --edg-target-frame 0 --max-steps 35000 --steps-scaler "$SCALER" --disable-viewer
rm -f "$O"/gsplat_fast${TAG}/ckpts/ckpt_*.pt
run eval      conda run -n gsplat --no-capture-output python estimate_ours_render_tto.py \
    --gaussians "$O/gsplat_fast${TAG}/ckpts/final_rank0.pt" --output-dir "$O/ours_render_eval_fast${TAG}" --use-covisible-mask
python3 -c "
import json; d=json.load(open('$O/ours_render_eval_fast${TAG}/metrics_tto.json'))['novel']
print(f\"RELDONE Dycheck $SCENE mPSNR {d['mPSNR']:.3f} mSSIM {d['mSSIM']:.3f} mLPIPS {d['mLPIPS']:.3f}\")"
