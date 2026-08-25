#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export DISABLE_OPEN3D_WINDOWS="${DISABLE_OPEN3D_WINDOWS:-1}"
export MOSCA_DISABLE_OPEN3D="${MOSCA_DISABLE_OPEN3D:-1}"

SCENE_DIR="${1:-${SCENE_DIR:-${REPO_ROOT}/datasets/Syn4D/bear3}}"
SPLIT="${SPLIT:-novel}"
FRAME_INDEX="${FRAME_INDEX:-50}"
NOVEL_ID="${NOVEL_ID:-4}"
OUT_DIR="${OUT_DIR:-${SCENE_DIR}/outputs/depth_visualization_aligned}"

GT_SCALE="${GT_SCALE:-6553.5}"
MASK_THRESHOLD="${MASK_THRESHOLD:-127.0}"
ALPHA_THRESHOLD="${ALPHA_THRESHOLD:-0.9}"
MIN_DEPTH="${MIN_DEPTH:-1e-6}"
ERROR_CLIP="${ERROR_CLIP:-0.5}"
SAVE_DOWNSCALE="${SAVE_DOWNSCALE:-0.5}"
METHODS="${METHODS:-ours da3_origin da3_render mosca spatracker}"
DEPTH_CORRECTION_RANGE="${DEPTH_CORRECTION_RANGE:-0.2}"
SPATRACKER_DEPTH_CORRECTION_RANGE="${SPATRACKER_DEPTH_CORRECTION_RANGE:-0.2}"
DEPTH_CORRECTION_SAMPLES="${DEPTH_CORRECTION_SAMPLES:-401}"
STATIC_ONLY="${STATIC_ONLY:-0}"
SPATRACKER_RESULT_NPZ="${SPATRACKER_RESULT_NPZ:-${SCENE_DIR}/outputs_spatracker/spatracker_rgb_dynamic_adaptive_input/results/result.npz}"
SPATRACKER_STATIC_POINT_RADIUS="${SPATRACKER_STATIC_POINT_RADIUS:-1}"
SPATRACKER_DYNAMIC_POINT_RADIUS="${SPATRACKER_DYNAMIC_POINT_RADIUS:-5}"
SPATRACKER_ALPHA_DENSITY_RADIUS="${SPATRACKER_ALPHA_DENSITY_RADIUS:-20}"
SPATRACKER_ALPHA_DENSITY_SCALE="${SPATRACKER_ALPHA_DENSITY_SCALE:-1.0}"
SPATRACKER_FILL_HOLES="${SPATRACKER_FILL_HOLES:-1}"
SPATRACKER_FILL_HOLE_RADIUS="${SPATRACKER_FILL_HOLE_RADIUS:-5}"
SPATRACKER_STATIC_POINT_STRIDE="${SPATRACKER_STATIC_POINT_STRIDE:-1}"
SPATRACKER_STATIC_FRAME_STRIDE="${SPATRACKER_STATIC_FRAME_STRIDE:-1}"
SPATRACKER_STATIC_MAX_POINTS="${SPATRACKER_STATIC_MAX_POINTS:-0}"
SPATRACKER_MAX_DYNAMIC_POINTS="${SPATRACKER_MAX_DYNAMIC_POINTS:-0}"
SPATRACKER_VISIBILITY_THRESHOLD="${SPATRACKER_VISIBILITY_THRESHOLD:-0.5}"
SPATRACKER_USE_GT_INTRINSICS="${SPATRACKER_USE_GT_INTRINSICS:-0}"
XM_ENV="${XM_ENV:-XM}"

COMMON_ARGS=(
    --scene-dir "${SCENE_DIR}"
    --split "${SPLIT}"
    --frame-index "${FRAME_INDEX}"
    --novel-id "${NOVEL_ID}"
    --output-dir "${OUT_DIR}"
    --gt-scale "${GT_SCALE}"
    --mask-threshold "${MASK_THRESHOLD}"
    --alpha-threshold "${ALPHA_THRESHOLD}"
    --min-depth "${MIN_DEPTH}"
    --error-clip "${ERROR_CLIP}"
    --save-downscale "${SAVE_DOWNSCALE}"
    --depth-correction-range "${DEPTH_CORRECTION_RANGE}"
    --spatracker-depth-correction-range "${SPATRACKER_DEPTH_CORRECTION_RANGE}"
    --depth-correction-samples "${DEPTH_CORRECTION_SAMPLES}"
    --spatracker-static-point-radius "${SPATRACKER_STATIC_POINT_RADIUS}"
    --spatracker-dynamic-point-radius "${SPATRACKER_DYNAMIC_POINT_RADIUS}"
    --spatracker-alpha-density-radius "${SPATRACKER_ALPHA_DENSITY_RADIUS}"
    --spatracker-alpha-density-scale "${SPATRACKER_ALPHA_DENSITY_SCALE}"
    --spatracker-fill-hole-radius "${SPATRACKER_FILL_HOLE_RADIUS}"
    --spatracker-static-point-stride "${SPATRACKER_STATIC_POINT_STRIDE}"
    --spatracker-static-frame-stride "${SPATRACKER_STATIC_FRAME_STRIDE}"
    --spatracker-static-max-points "${SPATRACKER_STATIC_MAX_POINTS}"
    --spatracker-max-dynamic-points "${SPATRACKER_MAX_DYNAMIC_POINTS}"
    --spatracker-visibility-threshold "${SPATRACKER_VISIBILITY_THRESHOLD}"
    --methods ${METHODS}
)

if [[ -n "${SPATRACKER_RESULT_NPZ}" ]]; then
    COMMON_ARGS+=(--spatracker-result-npz "${SPATRACKER_RESULT_NPZ}")
fi

if [[ "${STATIC_ONLY}" == "1" || "${STATIC_ONLY}" == "true" ]]; then
    COMMON_ARGS+=(--static-only)
fi

if [[ "${SPATRACKER_USE_GT_INTRINSICS}" == "0" || "${SPATRACKER_USE_GT_INTRINSICS}" == "false" ]]; then
    COMMON_ARGS+=(--no-spatracker-use-gt-intrinsics)
else
    COMMON_ARGS+=(--spatracker-use-gt-intrinsics)
fi

if [[ "${SPATRACKER_FILL_HOLES}" == "0" || "${SPATRACKER_FILL_HOLES}" == "false" ]]; then
    COMMON_ARGS+=(--no-spatracker-fill-holes)
else
    COMMON_ARGS+=(--spatracker-fill-holes)
fi

echo "[depth-vis] scene=${SCENE_DIR}"
echo "[depth-vis] split=${SPLIT} frame=${FRAME_INDEX} novel=${NOVEL_ID}"
echo "[depth-vis] output=${OUT_DIR}"
echo "[depth-vis] methods=${METHODS}"
echo "[depth-vis] metric=mean(abs(pred-gt)/gt), viewer depth convention"
echo "[depth-vis] train0 correction range=${DEPTH_CORRECTION_RANGE} samples=${DEPTH_CORRECTION_SAMPLES}"
echo "[depth-vis] spatracker correction range=${SPATRACKER_DEPTH_CORRECTION_RANGE}"
echo "[depth-vis] static_only=${STATIC_ONLY}"
echo "[depth-vis] spatracker_raw_result=${SPATRACKER_RESULT_NPZ:-auto}"
echo "[depth-vis] spatracker radii static=${SPATRACKER_STATIC_POINT_RADIUS} dynamic=${SPATRACKER_DYNAMIC_POINT_RADIUS}"
echo "[depth-vis] spatracker_fill_holes=${SPATRACKER_FILL_HOLES} radius=${SPATRACKER_FILL_HOLE_RADIUS}"

conda run --no-capture-output -n "${XM_ENV}" python -u "${SCRIPT_DIR}/visualize_depth_comparison_aligned.py" \
    "${COMMON_ARGS[@]}"

echo "[depth-vis] done: ${OUT_DIR}"
