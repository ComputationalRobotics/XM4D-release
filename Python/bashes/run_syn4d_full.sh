#!/usr/bin/env bash
# Reproducible full XM4D pipeline for Syn4D scenes.
# Per scene (selected via XM_DATASET_CAT/NAME, read by utils/definitions.py):
#   [depth] -> static tracking -> dynamic tracking -> static solver (camera)
#   -> align prior -> EDG solve -> train_4dgs_edg (gs dataset) -> 4DGS simple_trainer
#
# Usage:
#   bash Python/bashes/run_syn4d_full.sh                 # all 8 scenes
#   SCENES="bear1" bash Python/bashes/run_syn4d_full.sh  # one scene
# Env toggles:
#   RUN_DEPTH=1   also regenerate da3 depth (default 0: reuse existing, deterministic)
#   RUN_TRAIN=0   skip the 4DGS training stage
set -uo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
PYDIR="$HOME/4DReconstruction/Python"
TRAINER="$HOME/4DReconstruction/deps/gsplat/examples/simple_trainer.py"
ROOT="$HOME/4DReconstruction/datasets/Syn4D"
cd "$PYDIR"

SCENES="${SCENES:-bear1 bear2 bear3 bear4 casa1 casa2 casa3 casa4}"
RUN_DEPTH="${RUN_DEPTH:-0}"
RUN_TRAIN="${RUN_TRAIN:-1}"
CUDA="${CUDA_VISIBLE_DEVICES:-0}"
MASTER="${MASTER:-/tmp/claude-1000/-home-fenchel-4DReconstruction/16de01fe-ac0e-42da-b34b-4cc3df2f99b3/scratchpad/syn4d_full.master.log}"
: > "$MASTER"

export DISABLE_OPEN3D_WINDOWS=1 MOSCA_DISABLE_OPEN3D=1
export CUDA_VISIBLE_DEVICES="$CUDA"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

stage() {  # stage <name> -- <cmd...>
    local name="$1"; shift; [ "$1" = "--" ] && shift
    local t0=$(date +%s)
    echo "  [$SCENE] $name start $(date -Iseconds)" | tee -a "$MASTER"
    "$@" > "$SLOG/$name.log" 2>&1
    local rc=$?
    local dt=$(( $(date +%s) - t0 ))
    echo "  [$SCENE] $name rc=$rc ${dt}s" | tee -a "$MASTER"
    return $rc
}

for SCENE in $SCENES; do
    export XM_DATASET_CAT=Syn4D XM_DATASET_NAME="$SCENE" XM_DEPTH_SOURCE=da3
    SLOG="$ROOT/$SCENE/outputs/logs/full_rerun"
    mkdir -p "$SLOG"
    echo "===== $SCENE | start $(date -Iseconds) =====" | tee -a "$MASTER"

    if [ "$RUN_DEPTH" = "1" ]; then
        conda activate da3; stage depth -- python get_depths_clean.py || true
    fi

    conda activate tapnet
    stage static_tracking  -- python static_tracking_online.py || { echo "  [$SCENE] ABORT (static_tracking)" | tee -a "$MASTER"; continue; }
    stage dynamic_tracking -- python dynamic_tracking.py       || { echo "  [$SCENE] ABORT (dynamic_tracking)" | tee -a "$MASTER"; continue; }

    conda activate XM
    stage static_solver -- python static_solver.py || { echo "  [$SCENE] ABORT (static_solver)" | tee -a "$MASTER"; continue; }
    stage align_prior   -- python align_pointcloud_prior.py \
        --frame-index 0 --max-components 1 \
        --prior-coordinate-space auto --camera-pose-source auto --no-show-open3d \
        || { echo "  [$SCENE] ABORT (align_prior)" | tee -a "$MASTER"; continue; }
    stage edg -- python EDG_prior.py           || { echo "  [$SCENE] ABORT (edg)" | tee -a "$MASTER"; continue; }
    stage train_4dgs -- python train_4dgs_edg.py || { echo "  [$SCENE] ABORT (train_4dgs)" | tee -a "$MASTER"; continue; }

    if [ "$RUN_TRAIN" = "1" ]; then
        conda activate gsplat
        DATA="$ROOT/$SCENE/outputs/gs_edg_target_0000"
        RES="$ROOT/$SCENE/outputs/gsplat_edg_target_0000"
        EDG="$ROOT/$SCENE/outputs/results_all.npz"
        stage simple_trainer -- python "$TRAINER" default \
            --data-dir "$DATA" --result-dir "$RES" --data-factor 1 \
            --edg-results "$EDG" --edg-target-frame 0 --disable-viewer \
            || { echo "  [$SCENE] ABORT (simple_trainer)" | tee -a "$MASTER"; continue; }
    fi
    echo "===== $SCENE | done $(date -Iseconds) =====" | tee -a "$MASTER"
done
echo "ALL DONE $(date -Iseconds)" | tee -a "$MASTER"
