#!/usr/bin/env bash
# Run MoSca (monocular dynamic reconstruction) camera-pose estimation on IMC
# scenes and evaluate with the same metric as the other IMC baselines.
#
# MoSca is a monocular casual-VIDEO method: precompute = UniDepth + SpaTracker
# long-term tracks + RAFT optical flow between consecutive frames; fit = bundle
# adjustment with camera-trajectory smoothness priors + FOV search. IMC scenes
# are unordered internet photos (different camera/resolution per image, static
# scene), so like MegaSAM this is an out-of-domain stress test. We reuse the
# uniform 1024x768 resize (datasets/IMC_msam), give it all-zero dynamic masks
# (IMC is static) and the representative GT intrinsic (most charitable, matches
# COLMAP/GLOMAP/Ours GT-fixed treatment).
#
# Camera-only path: lite_moca_reconstruct.py (BA + COLMAP export, no 3DGS).
# MoSca exports c2w-convention COLMAP -> we convert to standard w2c before eval.
#
# Usage: Python/imc/run_all_mosca.sh [scene ...]   (default: all 6)
set -u -o pipefail

REPO=/home/fenchel/4DReconstruction
MOSCA=$REPO/deps/MoSca
ENV=mosca
PREP_CFG=./profile/sintel/sintel_prep.yaml
FIT_CFG=./profile/sintel/sintel_fit.yaml
LOG=$REPO/datasets/IMC/mosca.log
SLOGD=$REPO/datasets/IMC/mosca_stage_logs
mkdir -p "$SLOGD"

SCENES=("$@")
if [ ${#SCENES[@]} -eq 0 ]; then
  SCENES=(gate_00 gate_01 rome_00 rome_04 temple_00 temple_01)
fi

echo "===== MoSca IMC run: ${SCENES[*]} =====" | tee -a "$LOG"

for seq in "${SCENES[@]}"; do
  ws=$REPO/datasets/IMC_msam/$seq
  slog=$SLOGD/$seq
  echo "----- $seq -----" | tee -a "$LOG"

  # 0. masks (all-zero, 1024x768) + representative intrinsic
  conda run -n XM python "$REPO/Python/imc/prep_mosca_scene.py" "$seq" > "$slog.prep.log" 2>&1
  read -r FX FY CX CY < "$ws/mosca_intrinsic.txt"
  echo "  intrinsic fx=$FX fy=$FY cx=$CX cy=$CY" | tee -a "$LOG"
  K_ARGS="--gt_fx=$FX --gt_fy=$FY --gt_cx=$CX --gt_cy=$CY"

  # 1. precompute (UniDepth + SpaTracker + RAFT); static scene -> skip dynamic resample
  ( cd "$MOSCA" && conda run -n "$ENV" python mosca_precompute.py \
      --cfg "$PREP_CFG" --ws "$ws" --skip_dynamic_resample $K_ARGS ) \
      > "$slog.precompute.log" 2>&1
  echo "  precompute exit=$?" | tee -a "$LOG"

  # 2. reconstruct camera poses (lite: BA + COLMAP export)
  #    ba_epi_th relaxed 3e-5 -> 0.01: the default is so strict that 0 tracks are
  #    static on IMC (all classified dynamic) and it crashes at FOV search. 0.01
  #    admits the epipolar-consistent tracks so BA runs (result is still garbage).
  t0=$(date +%s)
  ( cd "$MOSCA" && conda run -n "$ENV" python lite_moca_reconstruct.py \
      --cfg "$FIT_CFG" --ws "$ws/outputs_mosca" $K_ARGS --ba_epi_th=0.01 ) \
      > "$slog.reconstruct.log" 2>&1
  rc=$?; t1=$(date +%s)
  echo "  reconstruct exit=$rc time=$((t1-t0))s" | tee -a "$LOG"

  # 3. convert c2w -> standard w2c, then eval vs original GT
  SRC=$ws/outputs_mosca/output_mosca/sparse/0
  STD=$ws/outputs_mosca/output_mosca/sparse_w2c
  conda run -n XM python "$REPO/Python/imc/fix_mosca_colmap.py" "$SRC" "$STD" > "$slog.fix.log" 2>&1
  ( cd "$REPO/Python" && conda run -n XM python eval_colmap.py \
      --scene-dir "$REPO/datasets/IMC/$seq" --sparse-dir "$STD" \
      --gt-type imc --json ) > "$slog.eval.log" 2>&1
  grep -E "RESULT_JSON" "$slog.eval.log" | tee -a "$LOG" || \
      { echo "  EVAL FAILED:"; tail -3 "$slog.eval.log" | tee -a "$LOG"; }
done

echo "===== MoSca IMC done =====" | tee -a "$LOG"
