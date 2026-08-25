#!/usr/bin/env bash
# VGGT (feed-forward SfM) camera-pose baseline on the DyCheck iPhone dataset.
# VGGT does global self-attention over ALL frames at once, so long sequences OOM.
# We first try res 378 (IMC's working point). If a scene still OOMs, retry with a
# uniform frame subsample (kept sorted; eval matches by name on the subset) until
# it fits, and print SUBSAMPLED so the reduction is not silent.
#
# Eval: eval_colmap.py --gt-type dycheck (camera-center metric, intrinsic-agnostic,
# so VGGT predicting its own intrinsics is fine).
#
# Usage: Python/bashes/run_dycheck_vggt.sh [scene ...]   (default: all 7)
set -u -o pipefail
source /home/fenchel/miniconda3/etc/profile.d/conda.sh
REPO=/home/fenchel/4DReconstruction
RES="${RES:-378}"
GPU="${CUDA_VISIBLE_DEVICES:-1}"
LOG=$REPO/datasets/Dycheck/vggt.log
SUBSAMPLES=(1 2 3 4)   # progressive frame stride if OOM

SCENES=("$@")
if [ ${#SCENES[@]} -eq 0 ]; then
  SCENES=(apple block paper-windmill space-out spin teddy wheel)
fi

echo "===== VGGT DyCheck run (GPU $GPU, res $RES): ${SCENES[*]} =====" | tee -a "$LOG"

for S in "${SCENES[@]}"; do
  scene=$REPO/datasets/Dycheck/$S
  out=$scene/output_vggt
  ok=0
  for stride in "${SUBSAMPLES[@]}"; do
    rm -rf "$out"
    if [ "$stride" -gt 1 ]; then
      # build a strided image subset; run_replica.py reads <scene_dir>/images/*
      sub=$scene/vggt_frames_s$stride
      rm -rf "$sub"; mkdir -p "$sub/images"
      i=0
      for f in $(ls "$scene/images"/*.png | sort); do
        if [ $((i % stride)) -eq 0 ]; then ln -s "$f" "$sub/images/$(basename "$f")"; fi
        i=$((i+1))
      done
      img_arg="--scene_dir=$sub"
      nsub=$(ls "$sub/images" | wc -l)
    else
      img_arg="--scene_dir=$scene"
      nsub=$(ls "$scene/images" | wc -l)
    fi
    t0=$(date +%s.%N)
    ( cd $REPO/deps/vggt && CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      conda run -n vggt python run_replica.py $img_arg --output_dir=$out --resolution $RES ) \
      > "$scene/vggt.log" 2>&1
    rc=$?
    vt=$(echo "$(date +%s.%N) - $t0" | bc)
    if [ $rc -eq 0 ] && [ -d "$out/sparse" ]; then
      tag="stride=$stride nframes=$nsub"
      [ "$stride" -gt 1 ] && tag="SUBSAMPLED $tag"
      echo -n "$S VGGT ($tag, ${vt%.*}s): " | tee -a "$LOG"
      conda run -n XM python $REPO/Python/eval_colmap.py --scene-dir $scene \
        --sparse-dir $out/sparse --gt-type dycheck --json 2>/dev/null \
        | grep RESULT_JSON | sed 's/.*RESULT_JSON: //' | tee -a "$LOG"
      ok=1; break
    elif grep -qiE "out of memory|CUDA out of memory|OOM" "$scene/vggt.log"; then
      echo "$S OOM at stride=$stride (nframes=$nsub) -> retrying with larger stride" | tee -a "$LOG"
      continue
    else
      echo "$S VGGT FAILED (stride=$stride, rc=$rc):" | tee -a "$LOG"; tail -3 "$scene/vggt.log" | tee -a "$LOG"
      break
    fi
  done
  [ $ok -eq 0 ] && echo "$S VGGT UNRESOLVED (even at max stride)" | tee -a "$LOG"
done
echo "===== VGGT DyCheck done =====" | tee -a "$LOG"
