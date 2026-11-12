#!/usr/bin/env bash
# Rendering benchmark (EDG + 4D Gaussians), from the camera solve of scripts/run_camera.sh on
#   bash scripts/run_render.sh all                    # Syn4D, NVIDIA, Dycheck: 22 scenes
#   bash scripts/run_render.sh Dycheck [scene ...]    # one dataset: Syn4D NVIDIA Dycheck
# Run it in the xm4d env (EDG stages); the 4D Gaussian training and evaluation run in the xm4d-gs env (GS_ENV).
# Inputs are datasets/<Dataset>/<scene>/ (never written) and results/camera/<Dataset>/<scene>/outputs/result_XM.npz:
# each scene runs in results/render/<Dataset>/<scene>/ (links to the inputs, copies of the files a stage rewrites,
# all outputs); its novel-view metrics go to metrics.json there. At the end the dataset averages are printed and
# written to results/render/averages.txt.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-python}
GS_ENV=${GS_ENV:-xm4d-gs}
GPU=${GPU:-0}
TRAINER=$ROOT/build/gsplat_examples/simple_trainer.py
[ -f "$TRAINER" ] || { echo "missing $TRAINER: run scripts/setup_gsplat.sh first"; exit 1; }
export CUDA_VISIBLE_DEVICES=$GPU CUDA_DEVICE_ORDER=PCI_BUS_ID EDG_NO_VIZ=1
declare -A SCENES=(
  [Syn4D]="bear1 bear2 bear3 bear4 casa1 casa2 casa3 casa4"
  [NVIDIA]="Balloon1 Balloon2 Jumping Playground Skating Truck Umbrella"
  [Dycheck]="apple block paper-windmill space-out spin teddy wheel"
)
gs() { conda run -n "$GS_ENV" --no-capture-output "$@"; }
run() { local n=$1; shift; local t0=$SECONDS; "$@" || { echo "STAGE $n FAILED"; exit 1; }; echo "STAGE $n $((SECONDS - t0))s"; }

# results/render/<Dataset>/<scene>: inputs linked, except the files a stage rewrites in place, which are copied
# (train_4dgs_edg copies the active depth into depth_anything_3/), plus a copy of the camera solve.
setup_scene() {
  local src=$1 out=$2 cam=$3
  rm -rf "$out"; mkdir -p "$out/outputs/depth_anything_3"
  cp "$cam" "$out/outputs/result_XM.npz"
  for e in "$src"/*; do [ "$(basename "$e")" = outputs ] || ln -s "$e" "$out/"; done
  for e in "$src"/outputs/*; do
    case $(basename "$e") in
      depth_anything_3) ;;
      output_tapnet_da3.npz|tracks_dynamic_merged.npz)
        cp "$e" "$out/outputs/" ;;
      *) ln -s "$e" "$out/outputs/" ;;
    esac
  done
  for e in "$src"/outputs/depth_anything_3/*; do
    if [ "$(basename "$e")" = depth_maps.npy ]; then cp "$e" "$out/outputs/depth_anything_3/"
    else ln -s "$e" "$out/outputs/depth_anything_3/"; fi
  done
}

# The training command that train_4dgs_edg.py recommends for this scene (Syn4D, NVIDIA).
recommended_train() { grep -A1 "Recommended gsplat 4DGS training command" "$1" | tail -1 | sed "s/CUDA_VISIBLE_DEVICES=[0-9]* //"; }

render_syn4d() {
  local O=$1/outputs log=$1/prepare.log
  run smooth    $PY optimize_dynamic_tracks.py --input tracks_dynamic_merged.npz
  run edg_old   $PY EDG_old.py
  run build_edg $PY build_edg.py --gt-depth
  run edg_solve $PY EDG_test_fast.py --merged --seg-len 10 --n-seg 2 --n-runs 50
  run complete  $PY complete_edg_table.py --merged
  run dataset   $PY train_4dgs_edg.py --dyn-source edg-good --dyn-lift-per-frame --gsplat-result-dir "$O/gsplat" > "$log"
  run train     gs bash -c "$(recommended_train "$log") --disable-viewer"
  rm -f "$O"/gsplat/ckpts/ckpt_*.pt
  run eval      gs python estimate_ours_render_tto.py --gaussians "$O/gsplat/ckpts/final_rank0.pt" \
                  --no-use-covisible-mask --output-dir "$O/render_eval" --output-json "$O/render_eval/metrics_tto.json"
}

render_nvidia() {
  local O=$1/outputs log=$1/prepare.log
  run smooth    $PY optimize_dynamic_tracks.py --input tracks_dynamic_merged.npz
  run edg_old   $PY EDG_old.py
  run build_edg $PY build_edg.py --merged --horizon-only --no-2d
  run edg_solve $PY EDG_test_fast.py --merged --seg-len 6 --n-seg 2 --n-runs 50
  run complete  $PY complete_edg_table.py --merged
  run dataset   $PY train_4dgs_edg.py --dyn-source edg-good --dyn-min-points 100000 --dyn-jitter 0.25 \
                  --gsplat-result-dir "$O/gsplat" > "$log"
  # NVIDIA trains at half resolution: images_2/ next to images/
  $PY - "$O/gs_edg_target_0000" <<'EOF'
import cv2, glob, os, sys
gs = sys.argv[1]
os.makedirs(gs + "/images_2", exist_ok=True)
for p in sorted(glob.glob(gs + "/images/*")):
    im = cv2.imread(p)
    cv2.imwrite(gs + "/images_2/" + os.path.basename(p),
                cv2.resize(im, (im.shape[1] // 2, im.shape[0] // 2), interpolation=cv2.INTER_AREA))
EOF
  local train; train=$(recommended_train "$log"); train=${train//--data-factor 1/--data-factor 2}
  run train     gs bash -c "$train --steps_scaler 1.0 --disable-viewer"
  run eval      gs python estimate_nvidia.py --gaussians "$O/gsplat/ckpts/final_rank0.pt" \
                  --output-dir "$O/render_eval" --output-json "$O/render_eval/metrics_tto.json"
}

render_dycheck() {
  local O=$1/outputs
  export XM_DEPTH_VARIANT=lidar_filled
  run lift      $PY optimize_dynamic_tracks.py --input tracks_dynamic_merged.npz --iters 0 --output tracks_dynamic_merged_lift.npz
  run edg_old   $PY EDG_old.py
  run build_edg $PY build_edg.py --gt-depth --tracks tracks_dynamic_merged_lift.npz
  run edg_solve $PY EDG_test_fast.py --tracks tracks_dynamic_merged_lift.npz
  run complete  $PY complete_edg_table.py --tracks tracks_dynamic_merged_lift.npz
  run dataset   $PY train_4dgs_edg.py --dyn-source edg-good --dyn-lift-per-frame > "$1/prepare.log"
  # 25k of 35k steps (steps scaler 25/35)
  run train     gs python "$TRAINER" default --data-dir "$O/gs_edg_target_0000" --result-dir "$O/gsplat" \
                  --data-factor 1 --edg-results "$O/results_all.npz" --edg-target-frame 0 \
                  --max-steps 35000 --steps-scaler 0.7142857142857143 --disable-viewer
  rm -f "$O"/gsplat/ckpts/ckpt_*.pt
  run eval      gs python estimate_ours_render_tto.py --gaussians "$O/gsplat/ckpts/final_rank0.pt" \
                  --output-dir "$O/render_eval" --use-covisible-mask
  unset XM_DEPTH_VARIANT
}

render() {
  local cat=$1 scene=$2
  local src=$ROOT/datasets/$cat/$scene out=$ROOT/results/render/$cat/$scene
  local cam=$ROOT/results/camera/$cat/$scene/outputs/result_XM.npz
  [ -d "$src" ] || { echo "missing $src (see datasets/README.md)"; return 1; }
  [ -f "$cam" ] || { echo "missing $cam: run scripts/run_camera.sh $cat $scene first"; return 1; }
  setup_scene "$src" "$out" "$cam"
  local t0=$SECONDS
  (cd "$ROOT/Python" && export XM_DATASET_ROOT=$out XM_DATASET_CAT=$cat XM_DATASET_NAME=$scene &&
     case $cat in Syn4D) render_syn4d "$out" ;; NVIDIA) render_nvidia "$out" ;; Dycheck) render_dycheck "$out" ;; esac
  ) > "$out/render.log" 2>&1
  $PY - "$out/outputs/render_eval/metrics_tto.json" "$out/metrics.json" $((SECONDS - t0)) <<'EOF'
import json, sys
src, out, secs = sys.argv[1:]
d = json.load(open(src))
d = d.get("novel", d)
m = {k: d[k] for k in ("mPSNR", "mSSIM", "mLPIPS")}
json.dump(dict(m, seconds=int(secs)), open(out, "w"), indent=1)
print(" ".join(f"{k} {v:.3f}" for k, v in m.items()))
EOF
}

if [ "${1:-}" = all ]; then
  set -- ; for c in Syn4D NVIDIA Dycheck; do for s in ${SCENES[$c]}; do set -- "$@" "$c/$s"; done; done
elif [ -n "${1:-}" ]; then
  cat=$1; shift
  [ -n "${SCENES[$cat]:-}" ] || { echo "unknown dataset $cat"; exit 1; }
  [ $# -gt 0 ] || set -- ${SCENES[$cat]}
  set -- $(for s in "$@"; do echo "$cat/$s"; done)
else
  sed -n '2,9p' "$0"; exit 1
fi
for cs in "$@"; do
  echo -n "${cs}: "
  render "${cs%%/*}" "${cs#*/}" || echo "FAILED (see results/render/$cs/render.log)"
done
$PY - "$ROOT/results/render" <<'EOF'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
lines = ["dataset  scenes  PSNR    SSIM   LPIPS"]
for d in ("Syn4D", "NVIDIA", "Dycheck"):
    ms = [json.loads(f.read_text()) for f in sorted((root / d).glob("*/metrics.json"))]
    if ms:
        a = {k: sum(m[k] for m in ms) / len(ms) for k in ("mPSNR", "mSSIM", "mLPIPS")}
        lines.append(f"{d:8s} {len(ms):6d}  {a['mPSNR']:.2f}  {a['mSSIM']:.3f}  {a['mLPIPS']:.3f}")
print("\n".join(lines))
(root / "averages.txt").write_text("\n".join(lines) + "\n")
EOF
