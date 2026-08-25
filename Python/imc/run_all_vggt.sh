#!/bin/bash
# VGGT (feed-forward) on IMC, ALL 200 frames. VGGT OOMs at res 518 on 200 imgs, so run at
# reduced model resolution 378 (multiple of 14) which fits 200 frames with ~unchanged accuracy.
# Eval with eval_colmap.py (same metric) against full GT.
source /home/fenchel/miniconda3/etc/profile.d/conda.sh
REPO=/home/fenchel/4DReconstruction; RES=378
for S in gate_00 gate_01 rome_00 rome_04 temple_00 temple_01; do
  scene=$REPO/datasets/IMC/$S; out=$scene/output_vggt
  rm -rf $out
  t0=$(date +%s.%N)
  ( cd $REPO/deps/vggt && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    conda run -n vggt python run_replica.py --scene_dir=$scene --output_dir=$out --resolution $RES ) > $scene/outputs/vggt.log 2>&1
  vt=$(echo "$(date +%s.%N) - $t0"|bc)
  echo -n "$S VGGT (200f@$RES, solver=$(printf %.1f $vt)s): "
  conda run -n XM python $REPO/Python/eval_colmap.py --scene-dir $scene --sparse-dir $out/sparse --gt-type imc --json 2>/dev/null | grep RESULT_JSON | sed 's/.*RESULT_JSON: //' | python3 -c "import sys,json;d=json.load(sys.stdin);print('ATE_T=%.4f RPE_T=%.4f RPE_R=%.4f matched=%d'%(d['ATE_T'],d['RPE_T'],d['RPE_R'],d['matched_frames']))" 2>/dev/null || echo "eval fail"
done
echo ALL_VGGT_DONE
