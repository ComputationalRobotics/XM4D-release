#!/usr/bin/env bash
# Run MegaSAM (camera pose) on IMC scenes and evaluate with the same metric as
# the other IMC baselines (eval_colmap.py --gt-type imc).
#
# MegaSAM is a casual-VIDEO SfM method (DROID-SLAM backbone + optical flow
# between consecutive frames, single shared intrinsic). IMC scenes are unordered
# internet photos with a DIFFERENT camera and resolution per image, so this is an
# out-of-domain stress test. To even run, every frame must share one resolution,
# so we first resize each scene's images to a uniform 1024x768 (parallel root
# datasets/IMC_msam). Poses are evaluated against the ORIGINAL IMC GT by filename.
#
# Stages per scene (mega_sam_new env, run from the mega-sam repo):
#   0. resize images                 -> datasets/IMC_msam/<scene>/images
#   1. DepthAnything mono-disparity  -> Depth-Anything/video_visualization/<scene>
#   2. UniDepth metric depth + fov   -> UniDepth/outputs/<scene>
#   3. DROID camera tracking         -> datasets/IMC_msam/<scene>/output_megasam/sparse
#   4. eval_colmap.py (vs orig GT)   -> ATE-T / RPE-T / RPE-R
#
# Usage: Python/imc/run_all_megasam.sh [scene ...]   (default: all 6)
set -u -o pipefail

REPO=/home/fenchel/4DReconstruction
MEGASAM=/home/fenchel/mega-sam
ENV=mega_sam_new
DATA_PATH=$REPO/datasets/IMC_msam
CKPT=checkpoints/megasam_final.pth
LOG=$REPO/datasets/IMC/megasam.log
STAGE_LOG_DIR=$REPO/datasets/IMC/megasam_stage_logs
mkdir -p "$STAGE_LOG_DIR"

SCENES=("$@")
if [ ${#SCENES[@]} -eq 0 ]; then
  SCENES=(gate_00 gate_01 rome_00 rome_04 temple_00 temple_01)
fi

echo "===== MegaSAM IMC run: ${SCENES[*]} =====" | tee -a "$LOG"

for seq in "${SCENES[@]}"; do
  scene_dir=$DATA_PATH/$seq
  slog=$STAGE_LOG_DIR/$seq
  echo "----- $seq -----" | tee -a "$LOG"

  # 0. Resize to uniform resolution (parallel root)
  conda run -n XM python "$REPO/Python/imc/resize_for_megasam.py" "$seq" \
      > "$slog.resize.log" 2>&1 && tail -1 "$slog.resize.log" | tee -a "$LOG"

  # 1. DepthAnything mono-disparity
  ( cd "$MEGASAM" && conda run -n "$ENV" python Depth-Anything/run_videos.py \
      --encoder vitl --load-from Depth-Anything/checkpoints/depth_anything_vitl14.pth \
      --img-path "$scene_dir/images" \
      --outdir "Depth-Anything/video_visualization/$seq" ) > "$slog.depthanything.log" 2>&1
  echo "  depthanything exit=$? ($(ls $MEGASAM/Depth-Anything/video_visualization/$seq 2>/dev/null | wc -l) files)" | tee -a "$LOG"

  # 2. UniDepth metric depth (+fov)
  ( cd "$MEGASAM" && PYTHONPATH="$MEGASAM/UniDepth" conda run -n "$ENV" python \
      UniDepth/scripts/demo_mega-sam.py --scene-name "$seq" \
      --img-path "$scene_dir/images" --outdir UniDepth/outputs ) > "$slog.unidepth.log" 2>&1
  echo "  unidepth exit=$? ($(ls $MEGASAM/UniDepth/outputs/$seq 2>/dev/null | wc -l) files)" | tee -a "$LOG"

  # 3. DROID camera tracking (optimize focal, single shared intrinsic)
  ( cd "$MEGASAM" && CUDA_VISIBLE_DEVICES=0 conda run -n "$ENV" python \
      camera_tracking_scripts/test_sintel_new.py --datapath="$DATA_PATH" \
      --weights="$CKPT" --scene_name "$seq" \
      --mono_depth_path "$MEGASAM/Depth-Anything/video_visualization" \
      --metric_depth_path "$MEGASAM/UniDepth/outputs" \
      --disable_vis --opt_focal ) > "$slog.tracking.log" 2>&1
  echo "  tracking exit=$? $(grep -oE 'tracking.*time_s[^,]*' $scene_dir/output_megasam/logs.json 2>/dev/null | head -1)" | tee -a "$LOG"

  # 4. Evaluate against ORIGINAL GT (name-based matching)
  ( cd "$REPO/Python" && conda run -n XM python eval_colmap.py \
      --scene-dir "$REPO/datasets/IMC/$seq" \
      --sparse-dir "$scene_dir/output_megasam/sparse" \
      --gt-type imc --json ) > "$slog.eval.log" 2>&1
  grep -E "RESULT_JSON" "$slog.eval.log" | tee -a "$LOG" || \
      { echo "  EVAL FAILED:"; tail -3 "$slog.eval.log" | tee -a "$LOG"; }
done

echo "===== MegaSAM IMC done =====" | tee -a "$LOG"
