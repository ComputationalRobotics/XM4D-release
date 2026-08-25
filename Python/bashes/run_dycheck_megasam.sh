#!/usr/bin/env bash
# MegaSAM (casual-video SfM) camera-pose baseline on the DyCheck iPhone dataset.
# DyCheck is exactly MegaSAM's domain: ordered monocular video from one moving
# camera at a uniform resolution (360x480), so NO resize is needed (unlike IMC).
#
# Per scene (env mega_sam_new, run from ~/mega-sam):
#   1. DepthAnything mono-disparity  -> Depth-Anything/video_visualization/<scene>
#   2. UniDepth metric depth + fov   -> UniDepth/outputs/<scene>
#   3. DROID camera tracking (--opt_focal, shared intrinsic)
#        -> datasets/Dycheck/<scene>/output_megasam/sparse  (COLMAP text model)
#   4. eval_colmap.py --gt-type dycheck
#
# Usage: Python/bashes/run_dycheck_megasam.sh [scene ...]   (default: all 7)
set -u -o pipefail

REPO=/home/fenchel/4DReconstruction
MEGASAM=/home/fenchel/mega-sam
ENV=mega_sam_new
DATA_PATH=$REPO/datasets/Dycheck
CKPT=checkpoints/megasam_final.pth
GPU="${CUDA_VISIBLE_DEVICES:-0}"
LOG=$REPO/datasets/Dycheck/megasam.log
STAGE_LOG_DIR=$REPO/datasets/Dycheck/megasam_stage_logs
mkdir -p "$STAGE_LOG_DIR"

SCENES=("$@")
if [ ${#SCENES[@]} -eq 0 ]; then
  SCENES=(apple block paper-windmill space-out spin teddy wheel)
fi

echo "===== MegaSAM DyCheck run (GPU $GPU): ${SCENES[*]} =====" | tee -a "$LOG"

for seq in "${SCENES[@]}"; do
  scene_dir=$DATA_PATH/$seq
  slog=$STAGE_LOG_DIR/$seq
  echo "----- $seq -----" | tee -a "$LOG"

  # 1. DepthAnything mono-disparity
  ( cd "$MEGASAM" && CUDA_VISIBLE_DEVICES=$GPU conda run -n "$ENV" python Depth-Anything/run_videos.py \
      --encoder vitl --load-from Depth-Anything/checkpoints/depth_anything_vitl14.pth \
      --img-path "$scene_dir/images" \
      --outdir "Depth-Anything/video_visualization/$seq" ) > "$slog.depthanything.log" 2>&1
  echo "  depthanything exit=$? ($(ls $MEGASAM/Depth-Anything/video_visualization/$seq 2>/dev/null | wc -l) files)" | tee -a "$LOG"

  # 2. UniDepth metric depth (+fov)
  ( cd "$MEGASAM" && CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$MEGASAM/UniDepth" conda run -n "$ENV" python \
      UniDepth/scripts/demo_mega-sam.py --scene-name "$seq" \
      --img-path "$scene_dir/images" --outdir UniDepth/outputs ) > "$slog.unidepth.log" 2>&1
  echo "  unidepth exit=$? ($(ls $MEGASAM/UniDepth/outputs/$seq 2>/dev/null | wc -l) files)" | tee -a "$LOG"

  # 3. DROID camera tracking (optimize a shared focal length)
  ( cd "$MEGASAM" && CUDA_VISIBLE_DEVICES=$GPU conda run -n "$ENV" python \
      camera_tracking_scripts/test_sintel_new.py --datapath="$DATA_PATH" \
      --weights="$CKPT" --scene_name "$seq" \
      --mono_depth_path "$MEGASAM/Depth-Anything/video_visualization" \
      --metric_depth_path "$MEGASAM/UniDepth/outputs" \
      --disable_vis --opt_focal ) > "$slog.tracking.log" 2>&1
  echo "  tracking exit=$? $(grep -oE 'tracking.*time_s[^,]*' $scene_dir/output_megasam/logs.json 2>/dev/null | head -1)" | tee -a "$LOG"

  # 4. Evaluate (name-based matching vs DyCheck GT)
  ( cd "$REPO/Python" && conda run -n XM python eval_colmap.py \
      --scene-dir "$scene_dir" \
      --sparse-dir "$scene_dir/output_megasam/sparse" \
      --gt-type dycheck --json ) > "$slog.eval.log" 2>&1
  grep -E "RESULT_JSON" "$slog.eval.log" | tee -a "$LOG" || \
      { echo "  EVAL FAILED:"; tail -3 "$slog.eval.log" | tee -a "$LOG"; }
done

echo "===== MegaSAM DyCheck done =====" | tee -a "$LOG"
