#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export DISABLE_OPEN3D_WINDOWS="${DISABLE_OPEN3D_WINDOWS:-1}"
export MOSCA_DISABLE_OPEN3D="${MOSCA_DISABLE_OPEN3D:-1}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/datasets/Syn4D}"
SCENES="${SCENES:-bear1 bear2 bear3 bear4 casa1 casa2 casa3 casa4}"
TRAIN_FRAME_INDEX="${TRAIN_FRAME_INDEX:-0}"
NOVEL_ID="${NOVEL_ID:-0}"
GT_SCALE="${GT_SCALE:-6553.5}"
MASK_THRESHOLD="${MASK_THRESHOLD:-127.0}"
ALPHA_THRESHOLD="${ALPHA_THRESHOLD:-0.05}"
MIN_DEPTH="${MIN_DEPTH:-1e-6}"
ERROR_CLIP="${ERROR_CLIP:-0.5}"
METHODS="${METHODS:-ours da3_origin da3_render mosca spatracker}"
DEPTH_CORRECTION_RANGE="${DEPTH_CORRECTION_RANGE:-0.2}"
DEPTH_CORRECTION_SAMPLES="${DEPTH_CORRECTION_SAMPLES:-401}"
STATIC_ONLY="${STATIC_ONLY:-0}"
OUT_ROOT="${OUT_ROOT:-${DATA_ROOT}/depth_comparison_aligned}"
SUMMARY_TSV="${SUMMARY_TSV:-${OUT_ROOT}/depth_l1_summary.tsv}"
XM_ENV="${XM_ENV:-XM}"

run_one() {
    local scene="$1"
    local split="$2"
    local scene_dir="${DATA_ROOT}/${scene}"
    local out_dir="${OUT_ROOT}/${scene}"
    local args=(
        --scene-dir "${scene_dir}"
        --split "${split}"
        --frame-index "${TRAIN_FRAME_INDEX}"
        --novel-id "${NOVEL_ID}"
        --output-dir "${out_dir}"
        --gt-scale "${GT_SCALE}"
        --mask-threshold "${MASK_THRESHOLD}"
        --alpha-threshold "${ALPHA_THRESHOLD}"
        --min-depth "${MIN_DEPTH}"
        --error-clip "${ERROR_CLIP}"
        --depth-correction-range "${DEPTH_CORRECTION_RANGE}"
        --depth-correction-samples "${DEPTH_CORRECTION_SAMPLES}"
        --methods ${METHODS}
    )
    if [[ "${STATIC_ONLY}" == "1" || "${STATIC_ONLY}" == "true" ]]; then
        args+=(--static-only)
    fi
    echo "[depth-table] ${scene} ${split}"
    conda run -n "${XM_ENV}" python "${SCRIPT_DIR}/visualize_depth_comparison_aligned.py" "${args[@]}"
}

mkdir -p "${OUT_ROOT}"

for scene in ${SCENES}; do
    run_one "${scene}" train
    run_one "${scene}" novel
done

python - "$OUT_ROOT" "$SUMMARY_TSV" "$TRAIN_FRAME_INDEX" "$NOVEL_ID" $SCENES <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
summary_tsv = Path(sys.argv[2])
train_frame = int(sys.argv[3])
novel_id = int(sys.argv[4])
scenes = sys.argv[5:]

def metric(scene: str, label: str, method: str) -> str:
    path = out_root / scene / f"{label}_aligned_depth_metrics.json"
    if not path.is_file():
        return "N/A"
    data = json.loads(path.read_text())
    value = data.get("methods", {}).get(method, {}).get("relative_l1")
    return "N/A" if value is None else f"{float(value):.4f}"

header = [
    "Scene",
    "Ours train depth",
    "MoSca train depth",
    "DA train depth",
    "DA train-render depth",
    "Spa train depth",
    "Ours novel depth",
    "MoSca novel depth",
    "DA novel depth",
    "Spa novel depth",
]
rows = ["\t".join(header)]
train_label = f"train_frame{train_frame:06d}"
novel_label = f"novel{novel_id:06d}"
for scene in scenes:
    rows.append(
        "\t".join(
            [
                scene,
                metric(scene, train_label, "ours"),
                metric(scene, train_label, "mosca"),
                metric(scene, train_label, "da3_origin"),
                metric(scene, train_label, "da3_render"),
                metric(scene, train_label, "spatracker"),
                metric(scene, novel_label, "ours"),
                metric(scene, novel_label, "mosca"),
                metric(scene, novel_label, "da3_render"),
                metric(scene, novel_label, "spatracker"),
            ]
        )
    )

summary_tsv.parent.mkdir(parents=True, exist_ok=True)
summary_tsv.write_text("\n".join(rows) + "\n")
print(f"[depth-table] wrote {summary_tsv}")
PY
