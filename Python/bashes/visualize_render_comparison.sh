#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MOSCA_DIR="${REPO_ROOT}/deps/MoSca"

export DISABLE_OPEN3D_WINDOWS="${DISABLE_OPEN3D_WINDOWS:-1}"
export MOSCA_DISABLE_OPEN3D="${MOSCA_DISABLE_OPEN3D:-1}"

SCENE_DIR="${1:-${SCENE_DIR:-${REPO_ROOT}/datasets/Syn4D/casa3}}"
SCENE_DIR="$(cd "${SCENE_DIR}" && pwd)"
SPLIT="${SPLIT:-novel}"
FRAME_INDEX="${FRAME_INDEX:-50}"
NOVEL_ID="${NOVEL_ID:-3}"
if [[ "${SPLIT}" == "novel" ]]; then
    TARGET_LABEL="novel$(printf '%06d' "${NOVEL_ID}")"
else
    TARGET_LABEL="frame$(printf '%06d' "${FRAME_INDEX}")"
fi
OUT_DIR="${OUT_DIR:-${SCENE_DIR}/outputs/render_visualization}"
mkdir -p "${OUT_DIR}"
OUT_DIR="$(cd "${OUT_DIR}" && pwd)"
TMP_DIR="${TMP_DIR:-${OUT_DIR}/tmp}"
mkdir -p "${TMP_DIR}"
TMP_DIR="$(cd "${TMP_DIR}" && pwd)"
OUT_PNG="${OUT_PNG:-${OUT_DIR}/${TARGET_LABEL}_ours_mosca_render_comparison.png}"
OUT_JSON="${OUT_JSON:-${OUT_DIR}/${TARGET_LABEL}_ours_mosca_render_comparison.json}"

OURS_ENV="${OURS_ENV:-gsplat}"
MOSCA_ENV="${MOSCA_ENV:-mosca}"
COMPOSE_ENV="${COMPOSE_ENV:-${OURS_ENV}}"

OURS_CAMERA_SOURCE="${OURS_CAMERA_SOURCE:-colmap}"
OURS_ALPHA_THRESHOLD="${OURS_ALPHA_THRESHOLD-0.8}"
OURS_RENDER_SCALE="${OURS_RENDER_SCALE:-1.0}"
OURS_EXTRA_ARGS="${OURS_EXTRA_ARGS:-}"

MOSCA_RENDER_ALPHA_TH="${MOSCA_RENDER_ALPHA_TH:-0.1}"
MOSCA_EXTRA_ARGS="${MOSCA_EXTRA_ARGS:-}"

ERROR_CLIP="${ERROR_CLIP:-0.25}"
RENDER_HOLE_FILL="${RENDER_HOLE_FILL:-white}"
case "${RENDER_HOLE_FILL}" in
    gt|raw|black|white) ;;
    *)
        echo "[render-viz] invalid RENDER_HOLE_FILL=${RENDER_HOLE_FILL}; use gt, raw, black, or white" >&2
        exit 2
        ;;
esac

OURS_NPZ="${TMP_DIR}/${TARGET_LABEL}_ours_render.npz"
MOSCA_NPZ="${TMP_DIR}/${TARGET_LABEL}_mosca_render.npz"

echo "[render-viz] scene=${SCENE_DIR}"
echo "[render-viz] split=${SPLIT}"
echo "[render-viz] frame=${FRAME_INDEX}"
echo "[render-viz] novel=${NOVEL_ID}"
echo "[render-viz] output=${OUT_PNG}"
echo "[render-viz] envs ours=${OURS_ENV} mosca=${MOSCA_ENV} compose=${COMPOSE_ENV}"
echo "[render-viz] render_hole_fill=${RENDER_HOLE_FILL}"

echo "[render-viz] checking ours env=${OURS_ENV}"
conda run -n "${OURS_ENV}" python - <<'PY'
import cv2  # noqa: F401
import rich  # noqa: F401
import torch  # noqa: F401
PY

echo "[render-viz] checking mosca env=${MOSCA_ENV}"
conda run -n "${MOSCA_ENV}" python - <<'PY'
import imageio  # noqa: F401
import numpy  # noqa: F401
import torch  # noqa: F401
PY

ours_args=(
    "${SCRIPT_DIR}/export_ours_render_frame.py"
    --scene-dir "${SCENE_DIR}"
    --split "${SPLIT}"
    --frame-index "${FRAME_INDEX}"
    --novel-id "${NOVEL_ID}"
    --output-npz "${OURS_NPZ}"
    --camera-source "${OURS_CAMERA_SOURCE}"
    --alpha-threshold "${OURS_ALPHA_THRESHOLD}"
    --render-scale "${OURS_RENDER_SCALE}"
)
if [[ -n "${OURS_EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    ours_extra=( ${OURS_EXTRA_ARGS} )
    ours_args+=("${ours_extra[@]}")
fi

echo "[render-viz] rendering ours with env=${OURS_ENV}"
conda run -n "${OURS_ENV}" python -u "${ours_args[@]}"

mosca_args=(
    "export_mosca_render_frame.py"
    --scene-path "${SCENE_DIR}"
    --split "${SPLIT}"
    --frame-index "${FRAME_INDEX}"
    --novel-id "${NOVEL_ID}"
    --output-npz "${MOSCA_NPZ}"
    --render-alpha-th "${MOSCA_RENDER_ALPHA_TH}"
)
if [[ -n "${MOSCA_EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    mosca_extra=( ${MOSCA_EXTRA_ARGS} )
    mosca_args+=("${mosca_extra[@]}")
fi

echo "[render-viz] rendering mosca with env=${MOSCA_ENV}"
(
    cd "${MOSCA_DIR}"
    conda run -n "${MOSCA_ENV}" python -u "${mosca_args[@]}"
)

echo "[render-viz] composing comparison"
conda run -n "${COMPOSE_ENV}" python -u "${SCRIPT_DIR}/compose_render_comparison.py" \
    --ours-npz "${OURS_NPZ}" \
    --mosca-npz "${MOSCA_NPZ}" \
    --output-png "${OUT_PNG}" \
    --output-json "${OUT_JSON}" \
    --frame-index "${FRAME_INDEX}" \
    --error-clip "${ERROR_CLIP}" \
    --hole-fill "${RENDER_HOLE_FILL}"

echo "[render-viz] wrote ${OUT_PNG}"
