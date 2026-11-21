#!/bin/bash
# Steps 1-5 of the README in one command, with the steps that do not need each other's output run side by side:
#   two GPUs: the depth (1) then the static solve (3) on the first, the static tracks (2a) then the dynamic tracks (2b)
#             on the second (2b waits for the depth only for its 3D lift);
#   one GPU:  1, 2a, then the static solve (mostly CPU) next to 2b.
# Steps 4 and 5 follow. The steps and their settings are the README's; only their timing overlaps. (Ceres in step 3
# stops after CERES_MAX_SOLVE_TIME s, which a busier machine reaches at fewer iterations on very long videos.) With
# "no_masks": true in config.json (no moving object) it runs steps 1, 2a and 3 only.
# usage: bash run_pipeline.sh              scene and parameters: config.json; logs: outputs/logs/run_*.log
#   CUDA_VISIBLE_DEVICES          the GPUs to use (default: all; the first two are used)
#   DA3_ENV TAPNET_ENV XM_ENV     conda environments (default da3, tapnet, XM; xm4d for all three with the
#                                 one-environment install)
#   SEQUENTIAL=1                  one step after the other, as listed in the README
set -eo pipefail
cd "$(dirname "$0")/Python"
# numpy asks the kernel for huge pages for big arrays; on a machine with fragmented memory every such request can stall
# for seconds in memory compaction. Same results.
export NUMPY_MADVISE_HUGEPAGE=${NUMPY_MADVISE_HUGEPAGE:-0}
source "$(conda info --base)/etc/profile.d/conda.sh"
DA3=${DA3_ENV:-da3}; TAP=${TAPNET_ENV:-tapnet}; XM=${XM_ENV:-XM}
INFO=$(conda activate "$XM" && python -c "import utils.definitions as d; print(int(bool(d.active_config.get('no_masks')))); print(d.output_dir)" | tail -n 2)
NO_MASKS=$(head -n 1 <<< "$INFO"); OUT=$(tail -n 1 <<< "$INFO")
mkdir -p "$OUT/logs"
if [ -n "${CUDA_VISIBLE_DEVICES+set}" ]; then IFS=, read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
else mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader 2> /dev/null | grep -E '^[0-9]+$'); fi
GA=${GPUS[0]:-}; GB=${GPUS[1]:-$GA}            # no GPU found: the steps keep the environment's CUDA_VISIBLE_DEVICES
DEPTH_DONE=$OUT/logs/.depth_done; STATIC_DONE=$OUT/logs/.static_tracks_done
rm -f "$DEPTH_DONE" "$DEPTH_DONE.failed" "$STATIC_DONE"
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
depth() { if step 1_depth "$GA" "$DA3" get_depths_clean.py; then touch "$DEPTH_DONE"; else touch "$DEPTH_DONE.failed"; return 1; fi; }
dynamic() { [ "$NO_MASKS" = 1 ] || step 2b_dynamic "$1" "$TAP" dynamic_tracking.py; }   # no moving object: nothing to track
step 0_masks "$GA" "$XM" check_masks.py      # merge the object masks now: step 1 would merge them while 2a reads them
killtree() { local c; for c in $(pgrep -P "$1"); do killtree "$c"; done; kill "$1" 2> /dev/null || true; }
cleanup() { local j; for j in $(jobs -p); do killtree "$j"; done; }
if [ "${SEQUENTIAL:-0}" = 1 ]; then
  depth
  step 2a_static "$GA" "$TAP" static_tracking_online.py
  dynamic "$GA"
  step 3_solver "$GA" "$XM" static_solver.py
elif [ "$GA" != "$GB" ]; then
  trap cleanup EXIT
  ( step 2a_static "$GB" "$TAP" static_tracking_online.py && touch "$STATIC_DONE" &&
    XM_WAIT_FOR_DEPTH=$DEPTH_DONE dynamic "$GB" ) & TRACKS=$!
  depth
  while [ ! -f "$STATIC_DONE" ]; do kill -0 $TRACKS 2> /dev/null || { wait $TRACKS; exit 1; }; sleep 1; done
  step 3_solver "$GA" "$XM" static_solver.py
  wait $TRACKS
  trap - EXIT
else
  depth
  step 2a_static "$GA" "$TAP" static_tracking_online.py
  trap cleanup EXIT
  dynamic "$GA" & DYN=$!
  step 3_solver "$GA" "$XM" static_solver.py
  wait $DYN
  trap - EXIT
fi
if [ "$NO_MASKS" = 1 ]; then
  rm -f "$DEPTH_DONE" "$STATIC_DONE"
  echo "[run] \"no_masks\": true, no moving object: steps 1-3 done in $((SECONDS - T0)) s; steps 4-5 need moving objects"; exit 0
fi
step 4a_smooth "$GA" "$XM" smooth_dynamic_tracks.py
step 4b_topology "$GA" "$XM" edg_topology.py
step 4c_edg "$GA" "$XM" EDG_test_fast.py
step 4d_complete "$GA" "$TAP" complete_edg_table.py
echo "[run] steps 1-4 done in $((SECONDS - T0)) s"
step 5_data "$GA" "$XM" train_4dgs_edg.py
rm -f "$DEPTH_DONE" "$STATIC_DONE"
echo "[run] steps 1-5 done in $((SECONDS - T0)) s; step 6: the training command at the end of $OUT/logs/run_5_data.log"
