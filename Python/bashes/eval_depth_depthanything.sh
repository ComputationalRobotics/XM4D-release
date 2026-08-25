#!/usr/bin/env bash
set -euo pipefail

DATASET_CATEGORY="Syn4D"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${SCRIPT_DIR}/../datasets/${DATASET_CATEGORY}"
EST_SUBDIR="outputs/depth_anything_3"
NORMALIZE_STAT="median"
POINT_STRIDE="4"
POINT_RADIUS="1"
ALPHA_DENSITY_RADIUS="1"

SCENES=(
    "bear1"
    "bear2"
    "bear3"
    "bear4"
    "casa1"
    "casa2"
    "casa3"
    "casa4"
)

RESULT_FILES=()

for scene in "${SCENES[@]}"; do
    scene_dir="${DATASET_ROOT}/${scene}"
    est_dir="${scene_dir}/${EST_SUBDIR}"
    output_json="${est_dir}/novel_depth_eval.json"
    vis_out="${est_dir}/novel_depth_eval_vis.png"
    train_output_json="${est_dir}/depth_eval.json"
    train_vis_out="${est_dir}/depth_eval_vis.png"
    train_render_output_json="${est_dir}/train_render_depth_eval.json"
    train_render_vis_out="${est_dir}/train_render_depth_eval_vis.png"

    echo "Evaluating DepthAnything train-view depth for ${scene_dir}"
    if [[ -d "${scene_dir}/depths" ]]; then
        python -u "${SCRIPT_DIR}/eval_depth.py" \
            --scene-dir "${scene_dir}" \
            --est-depth-dir "${est_dir}" \
            --normalize-stat "${NORMALIZE_STAT}" \
            --normalizer-scope "depthanything_train_dataset_shared" \
            --output-json "${train_output_json}" \
            --vis-out "${train_vis_out}" \
            --json

        RESULT_FILES+=("${train_output_json}")
        read -r gt_normalizer est_normalizer < <(
            python - "${train_output_json}" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
print(data["gt_normalizer"], data["est_normalizer"])
PY
        )
    else
        echo "Skipping training-view eval (missing ${scene_dir}/depths)." >&2
        gt_normalizer=""
        est_normalizer=""
    fi

    echo "Evaluating DepthAnything rendered train-view depth for ${scene_dir}"
    if [[ -n "${gt_normalizer}" && -n "${est_normalizer}" ]]; then
        python -u "${SCRIPT_DIR}/eval_depth_depthanything.py" \
            --eval-split train \
            --scene-dir "${scene_dir}" \
            --da3-depth-dir "${est_dir}" \
            --gt-depth-dir "${scene_dir}/depths" \
            --normalize-stat "${NORMALIZE_STAT}" \
            --point-stride "${POINT_STRIDE}" \
            --point-radius "${POINT_RADIUS}" \
            --alpha-density-radius "${ALPHA_DENSITY_RADIUS}" \
            --output-json "${train_render_output_json}" \
            --vis-out "${train_render_vis_out}" \
            --depth-png-dir "${est_dir}/train_render_depth_png" \
            --gt-normalizer "${gt_normalizer}" \
            --est-normalizer "${est_normalizer}" \
            --normalizer-scope "depthanything_train_dataset_shared" \
            --json

        RESULT_FILES+=("${train_render_output_json}")
    else
        echo "Skipping rendered training-view eval (missing direct train normalizers)." >&2
    fi

    echo "Evaluating DepthAnything novel-view depth for ${scene_dir}"
    if [[ ! -d "${scene_dir}" ]]; then
        echo "Skipping missing scene: ${scene_dir}" >&2
        continue
    fi
    if [[ ! -d "${est_dir}" ]]; then
        echo "Skipping missing estimate dir: ${est_dir}" >&2
        continue
    fi

    cmd=(
        python -u "${SCRIPT_DIR}/eval_depth_depthanything.py"
        --scene-dir "${scene_dir}" \
        --da3-depth-dir "${est_dir}" \
        --novel-dir "${scene_dir}/novel" \
        --normalize-stat "${NORMALIZE_STAT}" \
        --point-stride "${POINT_STRIDE}" \
        --point-radius "${POINT_RADIUS}" \
        --alpha-density-radius "${ALPHA_DENSITY_RADIUS}" \
        --output-json "${output_json}" \
        --vis-out "${vis_out}" \
        --depth-png-dir "${est_dir}/novel_depth_png" \
        --json
    )
    if [[ -n "${gt_normalizer}" && -n "${est_normalizer}" ]]; then
        cmd+=(
            --gt-normalizer "${gt_normalizer}"
            --est-normalizer "${est_normalizer}"
            --normalizer-scope "depthanything_train_dataset_shared"
        )
    fi
    "${cmd[@]}"

    RESULT_FILES+=("${output_json}")
done

echo
echo "DepthAnything depth summary:"
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
    slot = {
        "depth_eval.json": "direct_train",
        "train_render_depth_eval.json": "render_train",
        "novel_depth_eval.json": "novel",
    }.get(path.name, path.stem)
    by_scene.setdefault(scene, {})[slot] = data.get("l1")

print("| Scene | DA direct train | DA render train | DA novel |")
print("| --- | --- | --- | --- |")
for scene in sorted(by_scene):
    row = by_scene[scene]
    def fmt(key):
        val = row.get(key)
        return "-" if val is None else f"{float(val):.4f}"
    print(f"| {scene} | {fmt('direct_train')} | {fmt('render_train')} | {fmt('novel')} |")
PY
