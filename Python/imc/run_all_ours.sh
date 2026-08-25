#!/bin/bash
# Ours (XM-SfM) on the 6 IMC scenes: full pipeline (preprocess timing) + static_solver
# (solver timing) + pose eval. gt intrinsics + confidence weights (XM2_KEEP_PERCENTILE from imc.json).
source /home/fenchel/miniconda3/etc/profile.d/conda.sh
cd /home/fenchel/4DReconstruction
SCENES=(gate_00 gate_01 rome_00 rome_04 temple_00 temple_01)
for S in "${SCENES[@]}"; do
  scene=datasets/IMC/$S
  echo "==================== $S ===================="
  t0=$(date +%s)
  bash Python/imc/run_imc.sh $S gt confidence > $scene/outputs/ours_pipeline.log 2>&1
  pre=$(( $(date +%s) - t0 ))
  echo "$S ours_preprocess_wall=${pre}s"
  ( cd Python && XM_DATASET_CAT=IMC XM_DATASET_NAME=$S XM_DEPTH_SOURCE=unidepth \
      conda run -n XM python static_solver.py --no-use-ceres --no-visualize --json ) \
      > $scene/outputs/ours_solver.log 2>&1
  echo -n "$S ours solve: "; grep RESULT_JSON $scene/outputs/ours_solver.log | sed 's/.*RESULT_JSON: //' \
      | python3 -c "import sys,json; d=json.load(sys.stdin); print('solve_time=%.1f total_time=%.1f'%(d['solve_time'],d['total_time']))" 2>/dev/null || echo "solver parse failed"
  echo -n "$S ours eval: "
  conda run -n XM python Python/eval_xm.py --scene-dir $scene --sparse-dir $scene/outputs/result_XM.npz --gt-type imc --json 2>/dev/null | grep RESULT_JSON | sed 's/RESULT_JSON: //'
done
echo ALL_OURS_DONE
