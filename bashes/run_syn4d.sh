#!/bin/bash
# Syn4D pipeline (per scene), reproducing the data.tex configuration (Ours Train avg 34.62 / Novel avg
# 19.02 + camera accuracy): forward-only merge -> smooth -> build_edg --gt-depth -> EDG_test_fast
# (seg 10 x 2 x 50, seeded) -> complete (writes into the seeded results_all.npz base) -> 4DGS (tuned
# config from configs/syn4d_*.json via the recommended command) -> novel eval (TTO, NO covisible mask)
# + camera accuracy (eval_xm --gt-type syn4d). Legacy EDG behaviour is pinned in the syn4d jsons
# (EDG_ROT_MODE kabsch, EDG_ARAP_MARK_MAX -1, EDG_COMPLETE_BAD_NODE_FRAC 0.3, DYN_OFFMASK_MAX_FRAC -1).
# Seeded per scene: output_tapnet_da3.npz, result_XM.npz, tracks_dynamic_only.npz, results_all.npz, depths.
# Usage: bash bashes/run_syn4d.sh <scene> [gpu]
set -euo pipefail
SCENE=${1:?scene}; GPU=${2:-0}
TAG=${RUN_TAG:-}   # optional suffix for the result dirs
cd "$(dirname "$0")/../Python"
REPO="$(cd .. && pwd)"; SD="$REPO/datasets/Syn4D/$SCENE"; O="$SD/outputs"
export XM_DATASET_CAT=Syn4D XM_DATASET_NAME=$SCENE
export CUDA_VISIBLE_DEVICES=$GPU CUDA_DEVICE_ORDER=PCI_BUS_ID EDG_NO_VIZ=1
run() { local n=$1; shift; local t0=$(date +%s); "$@"; echo "STAGE $SCENE $n $(( $(date +%s)-t0 ))s"; }

run merge     conda run -n XM     --no-capture-output python test_extend_tracks.py --headless
run smooth    conda run -n XM     --no-capture-output python optimize_dynamic_tracks.py --input tracks_dynamic_merged.npz
run build_edg conda run -n XM     --no-capture-output python build_edg.py --gt-depth
run edg_solve conda run -n XM     --no-capture-output python EDG_test_fast.py --merged --seg-len 10 --n-seg 2 --n-runs 50
run complete  conda run -n tapnet --no-capture-output python complete_edg_table.py --merged

PREP_LOG=$(mktemp)
run dataset conda run -n XM --no-capture-output python train_4dgs_edg.py --dyn-source edg-good \
    --dyn-lift-per-frame --gsplat-result-dir "$O/gsplat_rel${TAG}" | tee "$PREP_LOG" > /dev/null
TRAIN=$(grep -A1 "Recommended gsplat 4DGS training command" "$PREP_LOG" | tail -1)
TRAIN=$(echo "$TRAIN" | sed "s/CUDA_VISIBLE_DEVICES=[0-9]*/CUDA_VISIBLE_DEVICES=$GPU/")
run train conda run -n gsplat --no-capture-output bash -c "$TRAIN --disable-viewer"
rm -f "$O"/gsplat_rel${TAG}/ckpts/ckpt_*.pt 2>/dev/null || true

OUT=$O/ours_render_eval_rel${TAG}
run eval conda run -n gsplat --no-capture-output python estimate_ours_render_tto.py \
    --gaussians "$O/gsplat_rel${TAG}/ckpts/final_rank0.pt" --no-use-covisible-mask \
    --output-dir "$OUT" --output-json "$OUT/metrics_tto.json"
run accuracy conda run -n XM --no-capture-output python eval_xm.py --gt-type syn4d \
    --scene-dir "$SD" --sparse-dir "$O/result_XM.npz" --json \
    > "$O/eval_xm_rel${TAG}.txt" 2>&1 || true
python3 - <<PYEOF
import json, glob, re
nov=json.load(open("$OUT/metrics_tto.json"))["novel"]
tr=None
vals=glob.glob("$O/gsplat_rel${TAG}/stats/val_step*.json")
if vals:
    best=max(vals, key=lambda q: int(re.search(r"step(\d+)", q).group(1)))   # numeric, not lexical
    tr=json.load(open(best)).get("psnr")
acc=open("$O/eval_xm_rel${TAG}.txt").read() if __import__("os").path.exists("$O/eval_xm_rel${TAG}.txt") else ""
m=re.search(r"ATE[-_ ]?T[^0-9]*([0-9.eE+-]+)", acc)
print(f"RELDONE Syn4D $SCENE novel_mPSNR {nov['mPSNR']:.3f} train_PSNR {tr if tr is None else round(tr,2)} "
      f"ATE_T {m.group(1) if m else 'see eval_xm_rel${TAG}.txt'}")
PYEOF
