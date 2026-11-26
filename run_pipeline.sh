#!/bin/bash
# Steps 1-3 of the README in one command. For a video, the depth (1) and the 2D tracks (2) do not need each other:
# with two GPUs they run side by side (1 on the first, 2 on the second), then the solver (3) on the first; with one GPU
# one after the other. Photos are matched with step 1's intrinsics, so 2 waits for 1. Step 1 is Depth Anything 3 or
# UniDepth V2 (DEPTH_MODEL), step 2 the video tracker or, for "input": "photos", the feature matching
# (match_photos.py). The steps and their settings are the README's; only their timing overlaps.
# usage: bash run_pipeline.sh              scene and parameters: config.json; logs: outputs/logs/run_*.log
#   CUDA_VISIBLE_DEVICES                   the GPUs to use (default: all; the first two are used)
#   DA3_ENV UNIDEPTH_ENV TAPNET_ENV XM_ENV conda environments (default da3, unidepth, tapnet, XM; xm4d for da3, tapnet
#                                          and XM with the one-environment install)
#   SEQUENTIAL=1                           one step after the other
set -eo pipefail
cd "$(dirname "$0")/Python"
# numpy asks the kernel for huge pages for big arrays; on a machine with fragmented memory every such request can stall
# for seconds in memory compaction. Same results.
export NUMPY_MADVISE_HUGEPAGE=${NUMPY_MADVISE_HUGEPAGE:-0}
source "$(conda info --base)/etc/profile.d/conda.sh"
TAP=${TAPNET_ENV:-tapnet}; XM=${XM_ENV:-XM}
mapfile -t INFO < <(conda activate "$XM" && python -c \
  "import utils.definitions as d; print(d.depth_model); print(int(d.photos)); print(d.output_dir)" | tail -n 3)
MODEL=${INFO[0]}; PHOTOS=${INFO[1]}; OUT=${INFO[2]}
if [ "$MODEL" = unidepth ]; then DEPTH_ENV=${UNIDEPTH_ENV:-unidepth}; DEPTH_PY=get_depths_unidepth.py
else DEPTH_ENV=${DA3_ENV:-da3}; DEPTH_PY=get_depths_clean.py; fi
if [ "$PHOTOS" = 1 ]; then TRACK_ENV=$XM; TRACK_PY=match_photos.py
else TRACK_ENV=$TAP; TRACK_PY=static_tracking_online.py; fi
mkdir -p "$OUT/logs"
if [ -n "${CUDA_VISIBLE_DEVICES+set}" ]; then IFS=, read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
else mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader 2> /dev/null | grep -E '^[0-9]+$'); fi
GA=${GPUS[0]:-}; GB=${GPUS[1]:-$GA}            # no GPU found: the steps keep the environment's CUDA_VISIBLE_DEVICES
T0=$SECONDS
step() {   # <name> <gpu> <env> <script>: one step in its conda environment, its output in outputs/logs/run_<name>.log
  local name=$1 gpu=$2 env=$3 script=$4 t=$SECONDS
  echo "[run] $name started at $((t - T0)) s"
  if ( conda activate "$env" && { [ -z "$gpu" ] || export CUDA_VISIBLE_DEVICES=$gpu; } && exec python "$script" ) \
       > "$OUT/logs/run_$name.log" 2>&1 < /dev/null; then
    echo "[run] $name done in $((SECONDS - t)) s"
  else
    echo "[run] $name FAILED, last lines of $OUT/logs/run_$name.log:"; tail -n 20 "$OUT/logs/run_$name.log"; return 1
  fi
}
step 0_masks "$GA" "$XM" check_masks.py   # merge the object masks now: step 1 would merge them while step 2 reads them
if [ "${SEQUENTIAL:-0}" = 1 ] || [ "$GA" = "$GB" ] || [ "$PHOTOS" = 1 ]; then
  step 1_depth "$GA" "$DEPTH_ENV" "$DEPTH_PY"
  step 2_tracks "$GA" "$TRACK_ENV" "$TRACK_PY"
else
  killtree() { local c; for c in $(pgrep -P "$1"); do killtree "$c"; done; kill "$1" 2> /dev/null || true; }
  trap 'killtree $TRACKS' EXIT
  step 2_tracks "$GB" "$TRACK_ENV" "$TRACK_PY" & TRACKS=$!
  step 1_depth "$GA" "$DEPTH_ENV" "$DEPTH_PY"
  wait $TRACKS
  trap - EXIT
fi
step 3_solver "$GA" "$XM" static_solver.py
echo "[run] steps 1-3 done in $((SECONDS - T0)) s: cameras in $OUT/result_XM.npz and $OUT/colmap/"
