#!/bin/bash
# Corrected baselines: COLMAP + GLOMAP with GT intrinsics held FIXED (no refinement),
# matching Ours (which uses fixed GT intrinsics). Reuses existing feature DB.
source /home/fenchel/miniconda3/etc/profile.d/conda.sh
cd /home/fenchel/4DReconstruction
E=glomap-official
for S in gate_00 gate_01 rome_00 rome_04 temple_00 temple_01; do
  scene=datasets/IMC/$S; db=$scene/outputs/baseline/database.db; img=$scene/images
  echo "==================== $S ===================="
  # COLMAP fixed-K
  cout=$scene/outputs/baseline/colmap_fixK
  if [ ! -d $cout/0 ]; then
    rm -rf $cout; mkdir -p $cout
    t=$(date +%s.%N)
    conda run -n $E colmap mapper --database_path $db --image_path $img --output_path $cout \
      --Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_principal_point 0 --Mapper.ba_refine_extra_params 0 \
      > $scene/outputs/colmap_fixK.log 2>&1
    echo "  colmap_fixK_time=$(echo "$(date +%s.%N) - $t"|bc)s"
  fi
  echo -n "  COLMAP-fixK: "; conda run -n XM python Python/eval_colmap.py --scene-dir $scene --sparse-dir $cout/0 --gt-type imc --json 2>/dev/null | grep RESULT_JSON | sed 's/.*RESULT_JSON: //' | python3 -c "import sys,json;d=json.load(sys.stdin);print('ATE_T=%.4f RPE_T=%.4f RPE_R=%.4f'%(d['ATE_T'],d['RPE_T'],d['RPE_R']))"
  # GLOMAP fixed-K
  gout=$scene/outputs/baseline/glomap_fixK
  rm -rf $gout; mkdir -p $gout
  t=$(date +%s.%N)
  conda run -n $E glomap mapper --database_path $db --image_path $img --output_path $gout \
    --TrackEstablishment.max_num_view_per_track 1000000 \
    --skip_view_graph_calibration 1 --BundleAdjustment.optimize_intrinsics 0 \
    > $scene/outputs/glomap_fixK.log 2>&1
  echo "  glomap_fixK_time=$(echo "$(date +%s.%N) - $t"|bc)s"
  echo -n "  GLOMAP-fixK: "; conda run -n XM python Python/eval_colmap.py --scene-dir $scene --sparse-dir $gout/0 --gt-type imc --json 2>/dev/null | grep RESULT_JSON | sed 's/.*RESULT_JSON: //' | python3 -c "import sys,json;d=json.load(sys.stdin);print('ATE_T=%.4f RPE_T=%.4f RPE_R=%.4f'%(d['ATE_T'],d['RPE_T'],d['RPE_R']))"
done
echo CORRECT_DONE
