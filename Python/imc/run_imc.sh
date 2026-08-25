#!/bin/bash
# End-to-end IMC XM-SfM preprocessing for one scene.
# Produces <scene>/outputs/output_tapnet_unidepth.npz (+ edges/weights/landmarks.npy)
# consumable by Python/static_solver.py with:
#   XM_DATASET_CAT=IMC XM_DATASET_NAME=<scene> XM_DEPTH_SOURCE=unidepth
#
# Usage: bash run_imc.sh <scene> [intrinsics] [weight]
#   scene:      name under datasets/IMC/ (e.g. gate_00) or an absolute path
#   intrinsics: gt | glomap | unidepth   (default: glomap) -- camera used for the
#               UniDepth depth AND the 2D->3D lift. With `gt`, GT intrinsics are
#               also injected into COLMAP/GLOMAP matching; otherwise matching
#               estimates them. Re-running with a different value redoes stages 2-3
#               (and stage1 only when intrinsics=gt, since matching changes).
#   weight:     uniform | confidence    (default: uniform) -- per-observation
#               solver weight; `confidence` uses UniDepth's per-pixel confidence.
set -e
SCENE_ARG="${1:?usage: run_imc.sh <scene-name-or-path> [gt|glomap|unidepth] [uniform|confidence]}"
INTR="${2:-glomap}"
WEIGHT="${3:-uniform}"
if [ -d "$SCENE_ARG" ]; then SCENE="$SCENE_ARG"
else SCENE="/home/fenchel/4DReconstruction/datasets/IMC/$SCENE_ARG"; fi
HERE="$(cd "$(dirname "$0")" && pwd)"
source /home/fenchel/miniconda3/etc/profile.d/conda.sh

STAGE1_ARGS=()
if [ "$INTR" = "gt" ]; then STAGE1_ARGS+=(--gt-intrinsics); fi

echo "### [$SCENE] stage1: COLMAP (GPU) + GLOMAP view graph  (intrinsics=$INTR)"
conda run -n XM python "$HERE/stage1_colmap_glomap.py" "$SCENE" "${STAGE1_ARGS[@]}"

echo "### [$SCENE] stage2: UniDepth depth (GPU), intrinsics=$INTR"
conda run -n unidepth python "$HERE/stage2_unidepth.py" "$SCENE" --intrinsics "$INTR"

echo "### [$SCENE] stage3: lift to solver npz, intrinsics=$INTR weight=$WEIGHT"
conda run -n XM python "$HERE/stage3_lift.py" "$SCENE" --intrinsics "$INTR" --weight "$WEIGHT"

echo "### done ($INTR intrinsics, $WEIGHT weight): $SCENE/outputs/output_tapnet_unidepth.npz"
