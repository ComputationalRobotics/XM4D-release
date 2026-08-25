#!/usr/bin/env bash
set -euo pipefail

DATASET_CATEGORY="${DATASET_CATEGORY:-Replica}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${SCRIPT_DIR}/../datasets/${DATASET_CATEGORY}"
VGGT_SUBDIR="${VGGT_SUBDIR:-output_vggt}"
VGGT_EVAL_SUBDIR="${VGGT_EVAL_SUBDIR:-output_vggt}"
POINT_STRIDE="${POINT_STRIDE:-4}"
DYNAMIC_POINT_STRIDE="${DYNAMIC_POINT_STRIDE:-1}"
STATIC_FRAME_STRIDE="${STATIC_FRAME_STRIDE:-1}"
STATIC_MAX_POINTS="${STATIC_MAX_POINTS:-2000000}"
MAX_POINTS="${MAX_POINTS:-0}"
POINT_RADIUS="${POINT_RADIUS:-1}"
ALPHA_DENSITY_RADIUS="${ALPHA_DENSITY_RADIUS:-1}"
ALPHA_THRESHOLD="${ALPHA_THRESHOLD:-0.05}"
GT_SCALE="${GT_SCALE:-6553.5}"
XM_ENV="${XM_ENV:-}"

# SCENES=(
#     "bear1"
# )

SCENES=(
    "office1_seq1"
    "office1_seq2"
    "office2_seq1"
    "office2_seq2"
    "room1_seq1"
    "room1_seq2"
    "room2_seq1"
    "room2_seq2"
)

# SCENES=(
#     "tum1"
#     "tum2"
#     "tum3"
#     "tum4"
#     "tum5"
#     "tum6"
# )

RESULT_FILES=()

run_python() {
    if [[ -n "${XM_ENV}" ]]; then
        conda run -n "${XM_ENV}" python -u "$@"
    else
        python -u "$@"
    fi
}

for scene in "${SCENES[@]}"; do
    scene_dir="${DATASET_ROOT}/${scene}"
    vggt_dir="${scene_dir}/${VGGT_SUBDIR}"
    eval_dir="${scene_dir}/${VGGT_EVAL_SUBDIR}"
    dense_npz="${vggt_dir}/dense_point_cloud/dense_point_cloud.npz"
    sparse_dir="${vggt_dir}/sparse"

    if [[ ! -d "${vggt_dir}" && "${scene}" =~ [0-9]$ && "${scene}" != *"_"* ]]; then
        alt_scene="${scene%?}_${scene: -1}"
        alt_vggt_dir="${DATASET_ROOT}/${alt_scene}/${VGGT_SUBDIR}"
        if [[ -d "${alt_vggt_dir}" ]]; then
            echo "Using VGGT output from ${alt_vggt_dir} for ${scene}."
            vggt_dir="${alt_vggt_dir}"
            eval_dir="${DATASET_ROOT}/${alt_scene}/${VGGT_EVAL_SUBDIR}"
            dense_npz="${vggt_dir}/dense_point_cloud/dense_point_cloud.npz"
            sparse_dir="${vggt_dir}/sparse"
        fi
    fi

    train_output_json="${eval_dir}/train_render_depth_eval.json"
    train_vis_out="${eval_dir}/train_render_depth_eval_vis.png"
    novel_output_json="${eval_dir}/novel_depth_eval.json"
    novel_vis_out="${eval_dir}/novel_depth_eval_vis.png"

    echo "Evaluating VGGT dense rendered depth for ${scene_dir}"
    if [[ ! -d "${scene_dir}" ]]; then
        echo "Skipping missing scene: ${scene_dir}" >&2
        continue
    fi
    if [[ ! -f "${dense_npz}" ]]; then
        echo "Skipping missing dense point cloud: ${dense_npz}" >&2
        continue
    fi
    if [[ ! -d "${sparse_dir}" ]]; then
        echo "Skipping missing VGGT sparse cameras: ${sparse_dir}" >&2
        continue
    fi

    if [[ -d "${scene_dir}/depths" ]]; then
        echo "Evaluating VGGT rendered train-view depth for ${scene_dir}"
        run_python "${SCRIPT_DIR}/eval_depth_vggt_dense.py" \
            --eval-split train \
            --scene-dir "${scene_dir}" \
            --vggt-dir "${vggt_dir}" \
            --dense-npz "${dense_npz}" \
            --sparse-dir "${sparse_dir}" \
            --gt-depth-dir "${scene_dir}/depths" \
            --gt-scale "${GT_SCALE}" \
            --point-stride "${POINT_STRIDE}" \
            --dynamic-point-stride "${DYNAMIC_POINT_STRIDE}" \
            --static-frame-stride "${STATIC_FRAME_STRIDE}" \
            --static-max-points "${STATIC_MAX_POINTS}" \
            --max-points "${MAX_POINTS}" \
            --point-radius "${POINT_RADIUS}" \
            --alpha-density-radius "${ALPHA_DENSITY_RADIUS}" \
            --alpha-threshold "${ALPHA_THRESHOLD}" \
            --output-json "${train_output_json}" \
            --vis-out "${train_vis_out}" \
            --rendered-depth-dir "${eval_dir}/train_render_depth_arrays" \
            --depth-png-dir "${eval_dir}/train_render_depth_png" \
            --json

        RESULT_FILES+=("${train_output_json}")
    else
        echo "Skipping training-view eval (missing ${scene_dir}/depths)." >&2
    fi

    if [[ -d "${scene_dir}/novel" ]]; then
        echo "Evaluating VGGT rendered novel-view depth for ${scene_dir}"
        cmd=(
            "${SCRIPT_DIR}/eval_depth_vggt_dense.py"
            --eval-split novel
            --scene-dir "${scene_dir}"
            --vggt-dir "${vggt_dir}"
            --dense-npz "${dense_npz}"
            --sparse-dir "${sparse_dir}"
            --novel-dir "${scene_dir}/novel"
            --gt-scale "${GT_SCALE}"
            --point-stride "${POINT_STRIDE}"
            --dynamic-point-stride "${DYNAMIC_POINT_STRIDE}"
            --static-frame-stride "${STATIC_FRAME_STRIDE}"
            --static-max-points "${STATIC_MAX_POINTS}"
            --max-points "${MAX_POINTS}"
            --point-radius "${POINT_RADIUS}"
            --alpha-density-radius "${ALPHA_DENSITY_RADIUS}"
            --alpha-threshold "${ALPHA_THRESHOLD}"
            --output-json "${novel_output_json}"
            --vis-out "${novel_vis_out}"
            --rendered-depth-dir "${eval_dir}/novel_render_depth_arrays"
            --depth-png-dir "${eval_dir}/novel_depth_png"
            --json
        )
        run_python "${cmd[@]}"
        RESULT_FILES+=("${novel_output_json}")
    else
        echo "Skipping novel-view eval (missing ${scene_dir}/novel)." >&2
    fi
done

echo
echo "VGGT dense rendered depth summary:"
python - "${RESULT_FILES[@]}" <<'PY'
import json
import sys
from pathlib import Path

by_scene = {}
for arg in sys.argv[1:]:
    path = Path(arg)
    if not path.is_file():
        continue
    data = json.loads(path.read_text())
    scene = Path(data["scene_dir"]).name
    slot = "render_train" if data.get("eval_split") == "train" else "novel"
    by_scene.setdefault(scene, {})[slot] = data.get("l1_avg_frame", data.get("l1"))

print("| Scene | VGGT render train | VGGT novel |")
print("| --- | --- | --- |")
for scene in sorted(by_scene):
    row = by_scene[scene]
    def fmt(key):
        val = row.get(key)
        return "-" if val is None else f"{float(val):.4f}"
    print(f"| {scene} | {fmt('render_train')} | {fmt('novel')} |")
PY
