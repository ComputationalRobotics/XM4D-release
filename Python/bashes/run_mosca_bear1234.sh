#!/usr/bin/env bash
set -euo pipefail

MOScA_DIR="/home/fenchel/4DReconstruction/deps/MoSca"
DATA_ROOT="/home/fenchel/4DReconstruction/datasets/Syn4D"
PREP_CFG="${MOScA_DIR}/profile/sintel/sintel_prep.yaml"
FIT_CFG="${MOScA_DIR}/profile/sintel/sintel_fit.yaml"

scenes=(bear1 bear2 bear3 bear4)

run_scene() {
    local scene="$1"
    local ws="${DATA_ROOT}/${scene}"
    local out_ws="${ws}/outputs_mosca"
    local log_file="${out_ws}/mosca_run.txt"
    local runtime_json="${out_ws}/mosca_runtime.json"

    if [[ ! -d "${ws}" ]]; then
        echo "[WARN] Missing scene folder: ${ws}" >&2
        return 0
    fi

    mkdir -p "${out_ws}"

    local start_epoch
    start_epoch=$(date +%s)
    {
        echo "===== ${scene} | start $(date -Is) ====="
        echo "Workspace: ${ws}"
        echo "Precompute config: ${PREP_CFG}"
        echo "Reconstruct config: ${FIT_CFG}"
        echo
        cd "${MOScA_DIR}"
        python mosca_precompute.py --cfg "${PREP_CFG}" --ws "${ws}"
        python mosca_reconstruct.py --cfg "${FIT_CFG}" --ws "${out_ws}"
        echo "===== ${scene} | end $(date -Is) ====="
        echo
    } 2>&1 | tee "${log_file}"

    local end_epoch
    end_epoch=$(date +%s)

    SCENE_NAME="${scene}" \
    START_EPOCH="${start_epoch}" \
    END_EPOCH="${end_epoch}" \
    RUNTIME_JSON="${runtime_json}" \
    python - <<'PY'
import json
import os
import time

scene = os.environ.get("SCENE_NAME")
start_epoch = int(os.environ.get("START_EPOCH", "0"))
end_epoch = int(os.environ.get("END_EPOCH", "0"))
runtime_path = os.environ.get("RUNTIME_JSON")

payload = {
    "scene": scene,
    "start_epoch": start_epoch,
    "end_epoch": end_epoch,
    "duration_sec": max(0, end_epoch - start_epoch),
    "start_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(start_epoch)) if start_epoch else None,
    "end_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(end_epoch)) if end_epoch else None,
}

    if runtime_path:
    os.makedirs(os.path.dirname(runtime_path), exist_ok=True)
    with open(runtime_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
PY
}

for scene in "${scenes[@]}"; do
    run_scene "${scene}"
done
