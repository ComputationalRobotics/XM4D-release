#!/bin/bash
# COLMAP + GLOMAP baselines on the 6 IMC scenes, with eval. Clean sequential timing.
source /home/fenchel/miniconda3/etc/profile.d/conda.sh
cd /home/fenchel/4DReconstruction
SCENES=(gate_00 gate_01 rome_00 rome_04 temple_00 temple_01)
for S in "${SCENES[@]}"; do
  scene=datasets/IMC/$S
  echo "==================== $S ===================="
  ts=$(date +%s)
  conda run -n XM python Python/imc/bench_baselines.py $scene > $scene/outputs/bench.log 2>&1
  echo "$S bench wall=$(( $(date +%s) - ts ))s  times=$(cat $scene/outputs/baseline/times.json 2>/dev/null)"
  for M in colmap glomap; do
    model=$scene/outputs/baseline/$M/0
    if [ -d "$model" ]; then
      echo -n "$S $M: "
      conda run -n XM python Python/eval_colmap.py --scene-dir $scene --sparse-dir $model --gt-type imc --json 2>/dev/null | grep RESULT_JSON | sed 's/RESULT_JSON: //'
    else echo "$S $M: NO MODEL"; fi
  done
done
echo ALL_BASELINES_DONE
