#!/bin/bash
# Preprocess all remaining IMC scenes: GT intrinsics + confidence-weighted observations.
# Sequential (GLOMAP writes a shared ./assets/tempdata path).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
SCENES=(gate_01 rome_00 rome_01 temple_00 temple_01)
for S in "${SCENES[@]}"; do
  echo "======================== $S ========================"
  LOG="/home/fenchel/4DReconstruction/datasets/IMC/$S/outputs/run_imc.log"
  mkdir -p "$(dirname "$LOG")"
  if bash "$HERE/run_imc.sh" "$S" gt confidence > "$LOG" 2>&1; then
    grep -iE "num_observations|frames_with_points|intrinsics_source|weight_mode|done \(" "$LOG" | tail -6
    echo "OK $S"
  else
    echo "FAILED $S (see $LOG)"; tail -8 "$LOG"
  fi
done
echo ALL_IMC_DONE
