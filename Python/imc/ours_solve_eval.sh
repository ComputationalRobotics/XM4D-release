#!/bin/bash
# Ours: ensure npz exists (rome_04 needs pipeline), run static_solver (solver timing),
# eval poses with eval_xm. Config from imc.json (+ any XM_* env overrides passed in).
source /home/fenchel/miniconda3/etc/profile.d/conda.sh
cd /home/fenchel/4DReconstruction
SCENES=(gate_00 gate_01 rome_00 rome_04 temple_00 temple_01)
for S in "${SCENES[@]}"; do
  scene=datasets/IMC/$S
  echo "==================== $S ===================="
  if [ ! -f $scene/outputs/output_tapnet_unidepth.npz ]; then
    echo "[pipeline] building npz for $S"
    bash Python/imc/run_imc.sh $S gt confidence > $scene/outputs/ours_pipeline.log 2>&1
  fi
  ( cd Python && XM_DATASET_CAT=IMC XM_DATASET_NAME=$S XM_DEPTH_SOURCE=unidepth $XM_EXTRA_ENV \
      conda run -n XM python static_solver.py --no-use-ceres --no-visualize --json ) \
      > $scene/outputs/ours_solver.log 2>&1
  st=$(grep RESULT_JSON $scene/outputs/ours_solver.log | sed 's/.*RESULT_JSON: //' | python3 -c "import sys,json;d=json.load(sys.stdin);print('%.1f'%d['solve_time'])" 2>/dev/null)
  echo "$S ours solve_time=${st}s"
  echo -n "$S ours eval: "
  conda run -n XM python Python/eval_xm.py --scene-dir $scene --sparse-dir $scene/outputs/result_XM.npz --gt-type imc --json 2>/dev/null | grep -E "Matched frames|RESULT_JSON" | tr '\n' ' '; echo
done
echo ALL_OURS_DONE
