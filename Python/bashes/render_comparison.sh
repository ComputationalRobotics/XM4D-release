#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MOSCA_DIR="${REPO_ROOT}/deps/MoSca"

export DISABLE_OPEN3D_WINDOWS="${DISABLE_OPEN3D_WINDOWS:-1}"
export MOSCA_DISABLE_OPEN3D="${MOSCA_DISABLE_OPEN3D:-1}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/datasets/Syn4D}"
SCENES="${SCENES:-bear1 bear2 bear3 bear4 casa1 casa2 casa3 casa4}"

OURS_ENV="${OURS_ENV:-gsplat}"
MOSCA_ENV="${MOSCA_ENV:-mosca}"
RUN_OURS="${RUN_OURS:-1}"
RUN_MOSCA="${RUN_MOSCA:-1}"
FORCE="${FORCE:-0}"

FRAME_STRIDE="${FRAME_STRIDE:-1}"
MAX_VIEWS="${MAX_VIEWS:-0}"
SAVE_IMAGES="${SAVE_IMAGES:-0}"

OURS_CAMERA_SOURCE="${OURS_CAMERA_SOURCE:-xm}"
OURS_VIEW_SPLIT="${OURS_VIEW_SPLIT:-train}"
OURS_METRIC_MASK="${OURS_METRIC_MASK:-alpha}"
OURS_ALPHA_THRESHOLD="${OURS_ALPHA_THRESHOLD:-0.8}"
OURS_RENDER_SCALE="${OURS_RENDER_SCALE:-1.0}"
OURS_EXTRA_ARGS="${OURS_EXTRA_ARGS:-}"

MOSCA_RENDER_ALPHA_TH="${MOSCA_RENDER_ALPHA_TH:-0.001}"
MOSCA_EXTRA_ARGS="${MOSCA_EXTRA_ARGS:-}"

SUMMARY_TSV="${SUMMARY_TSV:-${DATA_ROOT}/render_comparison_summary.tsv}"

run_py() {
    local env_name="$1"
    shift
    conda run -n "${env_name}" python -u "$@"
}

run_ours() {
    local scene_dir="$1"
    local out_dir="${scene_dir}/outputs/ours_render_eval"
    local out_json="${out_dir}/ours_render_metrics_summary.json"
    if [[ "${FORCE}" != "1" && -f "${out_json}" ]]; then
        echo "[render-compare] ours skip existing ${out_json}"
        return
    fi

    local args=(
        "${SCRIPT_DIR}/estimate_ours_render.py"
        --scene-dir "${scene_dir}"
        --camera-source "${OURS_CAMERA_SOURCE}"
        --output-dir "${out_dir}"
        --output-json "${out_json}"
        --view-split "${OURS_VIEW_SPLIT}"
        --frame-stride "${FRAME_STRIDE}"
        --max-views "${MAX_VIEWS}"
        --metric-mask "${OURS_METRIC_MASK}"
        --alpha-threshold "${OURS_ALPHA_THRESHOLD}"
        --render-scale "${OURS_RENDER_SCALE}"
        --json
    )
    if [[ "${SAVE_IMAGES}" == "1" || "${SAVE_IMAGES}" == "true" ]]; then
        args+=(--save-images)
    fi
    if [[ -n "${OURS_EXTRA_ARGS}" ]]; then
        # shellcheck disable=SC2206
        local extra=( ${OURS_EXTRA_ARGS} )
        args+=("${extra[@]}")
    fi

    echo "[render-compare] ours ${scene_dir}"
    run_py "${OURS_ENV}" "${args[@]}"
}

run_mosca() {
    local scene_dir="$1"
    local out_dir="${scene_dir}/outputs_mosca/render_eval"
    local out_json="${out_dir}/render_metrics_summary.json"
    if [[ "${FORCE}" != "1" && -f "${out_json}" ]]; then
        echo "[render-compare] mosca skip existing ${out_json}"
        return
    fi

    local scene_abs
    scene_abs="$(cd "${scene_dir}" && pwd)"
    local args=(
        "estimate_mosca_render.py"
        --scene-path "${scene_abs}"
        --output-dir "${out_dir}"
        --render-alpha-th "${MOSCA_RENDER_ALPHA_TH}"
        --frame-stride "${FRAME_STRIDE}"
        --max-views "${MAX_VIEWS}"
    )
    if [[ "${SAVE_IMAGES}" == "1" || "${SAVE_IMAGES}" == "true" ]]; then
        args+=(--save-images)
    fi
    if [[ -n "${MOSCA_EXTRA_ARGS}" ]]; then
        # shellcheck disable=SC2206
        local extra=( ${MOSCA_EXTRA_ARGS} )
        args+=("${extra[@]}")
    fi

    echo "[render-compare] mosca ${scene_dir}"
    (
        cd "${MOSCA_DIR}"
        run_py "${MOSCA_ENV}" "${args[@]}"
    )
}

write_summary() {
    python - "${DATA_ROOT}" "${SUMMARY_TSV}" ${SCENES} <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

data_root = Path(sys.argv[1])
summary_tsv = Path(sys.argv[2])
scenes = sys.argv[3:]


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def triplet(data: dict[str, Any], split: str) -> str:
    row = data.get(split) or {}
    if not row or row.get("mPSNR") is None:
        return "N/A"
    return f"{float(row['mPSNR']):.2f}/{float(row['mSSIM']):.3f}/{float(row['mLPIPS']):.3f}"


def views(data: dict[str, Any], split: str) -> str:
    row = data.get(split) or {}
    value = row.get("views")
    return "N/A" if value is None else str(int(value))


header = [
    "Scene",
    "Ours train PSNR/SSIM/LPIPS",
    "MoSca train PSNR/SSIM/LPIPS",
    "Ours novel PSNR/SSIM/LPIPS",
    "MoSca novel PSNR/SSIM/LPIPS",
    "Ours train views",
    "MoSca train views",
    "Ours novel views",
    "MoSca novel views",
]
lines = ["\t".join(header)]
for scene in scenes:
    scene_dir = data_root / scene
    ours = load(scene_dir / "outputs" / "ours_render_eval" / "ours_render_metrics_summary.json")
    mosca = load(scene_dir / "outputs_mosca" / "render_eval" / "render_metrics_summary.json")
    lines.append(
        "\t".join(
            [
                scene,
                triplet(ours, "training"),
                triplet(mosca, "training"),
                triplet(ours, "novel"),
                triplet(mosca, "novel"),
                views(ours, "training"),
                views(mosca, "training"),
                views(ours, "novel"),
                views(mosca, "novel"),
            ]
        )
    )

summary_tsv.parent.mkdir(parents=True, exist_ok=True)
summary_tsv.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[render-compare] wrote {summary_tsv}")
PY
}

for scene in ${SCENES}; do
    scene_dir="${DATA_ROOT}/${scene}"
    if [[ ! -d "${scene_dir}" ]]; then
        echo "[render-compare] skip missing scene ${scene_dir}" >&2
        continue
    fi
    if [[ "${RUN_OURS}" == "1" || "${RUN_OURS}" == "true" ]]; then
        run_ours "${scene_dir}"
    fi
    if [[ "${RUN_MOSCA}" == "1" || "${RUN_MOSCA}" == "true" ]]; then
        run_mosca "${scene_dir}"
    fi
done

write_summary
