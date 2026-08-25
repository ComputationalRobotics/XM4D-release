#!/bin/bash
# Ours final config: RANK=3 (config), gt intrinsics + confidence, Ceres refinement ON.
# Reports static_solver's internal (correctly-aligned) ATE/RPE + solve+ceres time.
source /home/fenchel/miniconda3/etc/profile.d/conda.sh
cd /home/fenchel/4DReconstruction/Python
for S in gate_00 gate_01 rome_00 rome_04 temple_00 temple_01; do
  log=../datasets/IMC/$S/outputs/ours_final.log
  XM_DATASET_CAT=IMC XM_DATASET_NAME=$S XM_DEPTH_SOURCE=unidepth \
    conda run -n XM python static_solver.py --use-ceres --no-visualize --json > $log 2>&1
  echo -n "$S ours: "
  grep RESULT_JSON $log | sed 's/.*RESULT_JSON: //' | python3 -c "import sys,json;d=json.load(sys.stdin);print('ATE_T=%.4f ATE_R=%.4f RPE_T=%.4f RPE_R=%.4f solve=%.1f ceres=%.1f solver_total=%.1f'%(d['ATE_T'],d['ATE_R'],d['RPE_T'],d['RPE_R'],d['solve_time'],d['ceres_time'],d['solve_time']+d['ceres_time']))" 2>/dev/null || { echo "FAIL"; tail -5 $log | grep -vi warn; }
done
echo OURS_FINAL_DONE
