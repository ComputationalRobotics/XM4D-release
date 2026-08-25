#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SCENE_DIR="${1:?Usage: bash Python/bashes/run_spatracker_dynamic_adaptive.sh /path/to/scene}"
SCENE_DIR="$(cd "${SCENE_DIR}" && pwd)"
VIDEO_NAME="${VIDEO_NAME:-$(basename "${SCENE_DIR}")}"
RUN_ROOT="${RUN_ROOT:-${SCENE_DIR}/outputs_spatracker}"
OUT_NAME="${OUT_NAME:-spatracker_rgb_dynamic_adaptive_input}"
INPUT_DIR="${INPUT_DIR:-${RUN_ROOT}/${OUT_NAME}}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${RUN_ROOT}/spatracker_eval}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${LOG_DIR}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/run_${RUN_ID}.log}"
RUNTIME_JSON="${RUNTIME_JSON:-${LOG_DIR}/runtime_${RUN_ID}.json}"
START_EPOCH="$(date +%s)"
START_ISO="$(date -Iseconds)"

finish() {
    local status=$?
    local end_epoch end_iso duration
    end_epoch="$(date +%s)"
    end_iso="$(date -Iseconds)"
    duration=$((end_epoch - START_EPOCH))
    printf '{\n  "status": %d,\n  "start": "%s",\n  "end": "%s",\n  "duration_seconds": %d,\n  "scene_dir": "%s",\n  "run_root": "%s",\n  "input_dir": "%s",\n  "eval_output_dir": "%s",\n  "log": "%s"\n}\n' \
        "${status}" "${START_ISO}" "${end_iso}" "${duration}" "${SCENE_DIR}" "${RUN_ROOT}" "${INPUT_DIR}" "${EVAL_OUTPUT_DIR}" "${RUN_LOG}" > "${RUNTIME_JSON}"
    echo "[spatracker-adaptive] runtime_seconds=${duration}"
    echo "[spatracker-adaptive] log=${RUN_LOG}"
    echo "[spatracker-adaptive] runtime=${RUNTIME_JSON}"
}
trap finish EXIT
exec > >(tee -a "${RUN_LOG}") 2>&1

ENV="SpaTrack2"
FPS="${FPS:-1}"
PREP_FPS="${PREP_FPS:-10}"
MAX_SIDE="${MAX_SIDE:-0}"
GRID_SIZE="${GRID_SIZE:-360}"
FIRST_QUERY_POINTS="${FIRST_QUERY_POINTS:-800}"
LATER_QUERY_POINTS="${LATER_QUERY_POINTS:-800}"
DEFAULT_VO_POINTS="${FIRST_QUERY_POINTS}"
if (( LATER_QUERY_POINTS > DEFAULT_VO_POINTS )); then
    DEFAULT_VO_POINTS="${LATER_QUERY_POINTS}"
fi
VO_POINTS="${VO_POINTS:-${DEFAULT_VO_POINTS}}"
ADAPTIVE_COVERAGE_RADIUS="${ADAPTIVE_COVERAGE_RADIUS:-0}"
MAX_QUERY_POINTS="${MAX_QUERY_POINTS:-0}"
MASK_THRESHOLD="${MASK_THRESHOLD:-127}"
EVAL_GT_SCALE="${EVAL_GT_SCALE:-6553.5}"
EVAL_POINT_RADIUS="${EVAL_POINT_RADIUS:-5}"
EVAL_ALPHA_DENSITY_RADIUS="${EVAL_ALPHA_DENSITY_RADIUS:-1}"
EVAL_ALPHA_THRESHOLD="${EVAL_ALPHA_THRESHOLD:-0.05}"
EVAL_STATIC_MAX_POINTS="${EVAL_STATIC_MAX_POINTS:-200000}"
EVAL_STATIC_POINT_STRIDE="${EVAL_STATIC_POINT_STRIDE:-0}"
EVAL_POINT_STRIDE="${EVAL_POINT_STRIDE:-0}"
EVAL_USE_GT_INTRINSICS="${EVAL_USE_GT_INTRINSICS:-0}"
EVAL_INTRINSICS_ARG="--use-gt-intrinsics"
if [[ "${EVAL_USE_GT_INTRINSICS}" != "1" && "${EVAL_USE_GT_INTRINSICS}" != "true" ]]; then
    EVAL_INTRINSICS_ARG="--no-use-gt-intrinsics"
fi
FIRST_RESULT_NAME="result_first.npz"
LATER_RESULT_NAME="result_later.npz"
MERGED_RESULT_NAME="result.npz"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
CLEAN_RESULTS="${CLEAN_RESULTS:-1}"

echo "[spatracker-adaptive] scene=${SCENE_DIR}"
echo "[spatracker-adaptive] run_root=${RUN_ROOT}"
echo "[spatracker-adaptive] input=${INPUT_DIR}"
echo "[spatracker-adaptive] eval_output=${EVAL_OUTPUT_DIR}"
echo "[spatracker-adaptive] log=${RUN_LOG}"
echo "[spatracker-adaptive] video=${VIDEO_NAME}"
echo "[spatracker-adaptive] grid_size=${GRID_SIZE} vo_points=${VO_POINTS}"
echo "[spatracker-adaptive] first_query_points=${FIRST_QUERY_POINTS} later_query_points=${LATER_QUERY_POINTS} max_query_points=${MAX_QUERY_POINTS}"
echo "[spatracker-adaptive] eval_gt_scale=${EVAL_GT_SCALE}"

conda run -n "${ENV}" python "${SCRIPT_DIR}/prepare_spatracker_scene.py" \
    --scene-dir "${SCENE_DIR}" \
    --output-dir "${INPUT_DIR}" \
    --video-name "${VIDEO_NAME}" \
    --fps "${PREP_FPS}" \
    --max-side "${MAX_SIDE}" \
    --write-mp4 \
    --write-masks \
    --write-gt-camera \
    --no-write-rgbd

if [[ "${CLEAN_RESULTS}" == "1" || "${CLEAN_RESULTS}" == "true" ]]; then
    echo "[spatracker-adaptive] cleaning stale SpaTracker result npz files"
    rm -f \
        "${INPUT_DIR}/results/${FIRST_RESULT_NAME}" \
        "${INPUT_DIR}/results/${LATER_RESULT_NAME}" \
        "${INPUT_DIR}/results/result_second.npz" \
        "${INPUT_DIR}/results/results_first.npz" \
        "${INPUT_DIR}/results/results_later.npz" \
        "${INPUT_DIR}/results/results_second.npz" \
        "${INPUT_DIR}/results/${MERGED_RESULT_NAME}"
fi

run_spatracker_pass() {
    local query_mode="$1"
    local result_name="$2"
    local first_cap="$3"
    local later_cap="$4"
    echo "[spatracker-adaptive] pass=${query_mode} result=${result_name}"
    pushd "${REPO_ROOT}/deps/SpaTrackerV2" >/dev/null
    conda run -n "${ENV}" python -u inference.py \
        --data_type RGB \
        --data_dir "${INPUT_DIR}" \
        --video_name "${VIDEO_NAME}" \
        --track_mode offline \
        --fps "${FPS}" \
        --grid_size "${GRID_SIZE}" \
        --vo_points "${VO_POINTS}" \
        --query_mode "${query_mode}" \
        --query_dynamic_masks \
        --mask_threshold "${MASK_THRESHOLD}" \
        --first_query_points "${first_cap}" \
        --later_query_points "${later_cap}" \
        --adaptive_coverage_radius "${ADAPTIVE_COVERAGE_RADIUS}" \
        --max_query_points "${MAX_QUERY_POINTS}" \
        --gt_camera_npz "${INPUT_DIR}/${VIDEO_NAME}_gt_camera.npz" \
        --result_name "${result_name}"
    popd >/dev/null
}

run_spatracker_pass "first_only" "${FIRST_RESULT_NAME}" "${FIRST_QUERY_POINTS}" 0
run_spatracker_pass "later_only" "${LATER_RESULT_NAME}" 0 "${LATER_QUERY_POINTS}"

RESULT_NPZ="${INPUT_DIR}/results/result.npz"
echo "[spatracker-adaptive] merging two passes -> ${RESULT_NPZ}"
python - <<PY
import numpy as np
from pathlib import Path
out = Path("${INPUT_DIR}") / "results"
def first_existing(names):
    for name in names:
        path = out / name
        if path.is_file():
            return path
    raise FileNotFoundError("None of these SpaTracker split outputs exist: " + ", ".join(str(out / name) for name in names))

first_path = first_existing(["${FIRST_RESULT_NAME}", "results_first.npz", "first.npz"])
later_path = first_existing(["${LATER_RESULT_NAME}", "result_second.npz", "results_later.npz", "results_second.npz", "second.npz"])
merged_path = out / "${MERGED_RESULT_NAME}"
first = np.load(first_path, allow_pickle=True)
later = np.load(later_path, allow_pickle=True)
merged = {k: first[k] for k in first.files}
for key in ("coords", "visibs", "conf_pred"):
    if key in first and key in later:
        merged[key] = np.concatenate([first[key], later[key]], axis=1)
if "queries_xyt" in first and "queries_xyt" in later:
    merged["queries_xyt"] = np.concatenate([first["queries_xyt"], later["queries_xyt"]], axis=0)
if "query_counts_per_frame" in first and "query_counts_per_frame" in later:
    merged["query_counts_per_frame"] = first["query_counts_per_frame"] + later["query_counts_per_frame"]
if "query_candidate_counts_per_frame" in first:
    merged["query_candidate_counts_per_frame"] = first["query_candidate_counts_per_frame"]
merged["query_mode"] = np.array("two_pass_first_then_later")
merged["first_pass_result"] = np.array(str(first_path))
merged["later_pass_result"] = np.array(str(later_path))
merged["first_query_points"] = np.array(${FIRST_QUERY_POINTS}, dtype=np.int64)
merged["later_query_points"] = np.array(${LATER_QUERY_POINTS}, dtype=np.int64)
np.savez(merged_path, **merged)
d = np.load(merged_path, allow_pickle=True)
print("[spatracker-adaptive] first coords", first["coords"].shape)
print("[spatracker-adaptive] later coords", later["coords"].shape)
print("[spatracker-adaptive] merged coords", d["coords"].shape)
print("[spatracker-adaptive] merged queries_xyt", d["queries_xyt"].shape)
if "query_counts_per_frame" in d:
    q = d["query_counts_per_frame"]
    print("[spatracker-adaptive] merged query_counts min/mean/max", int(q.min()), float(q.mean()), int(q.max()))
PY

echo "[spatracker-adaptive] rendering/evaluating SpaTracker train+novel depth -> ${EVAL_OUTPUT_DIR}"
conda run -n "${XM_ENV:-XM}" python "${SCRIPT_DIR}/eval_spatracker.py" \
    --scene-dir "${SCENE_DIR}" \
    --result-npz "${RESULT_NPZ}" \
    --output-dir "${EVAL_OUTPUT_DIR}" \
    --gt-scale "${EVAL_GT_SCALE}" \
    --point-radius "${EVAL_POINT_RADIUS}" \
    --alpha-density-radius "${EVAL_ALPHA_DENSITY_RADIUS}" \
    --alpha-threshold "${EVAL_ALPHA_THRESHOLD}" \
    --static-max-points "${EVAL_STATIC_MAX_POINTS}" \
    --static-point-stride "${EVAL_STATIC_POINT_STRIDE}" \
    --point-stride "${EVAL_POINT_STRIDE}" \
    --mask-threshold "${MASK_THRESHOLD}" \
    "${EVAL_INTRINSICS_ARG}" \
    --json
