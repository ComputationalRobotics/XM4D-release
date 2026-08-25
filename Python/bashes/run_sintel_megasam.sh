conda activate mosca

scene_names=(
  alley_1
  alley_2
  ambush_2
  ambush_4
  ambush_5
  ambush_6
  ambush_7
  bamboo_1
  bamboo_2
  bandage_1
  bandage_2
  cave_2
  cave_4
  market_2
  market_5
  market_6
  mountain_1
  shaman_2
  shaman_3
  sleeping_1
  sleeping_2
  temple_2
  temple_3
)

data_root="../datasets/Sintel"
prep_cfg="./profile/sintel/sintel_prep.yaml"
fit_cfg="./profile/sintel/sintel_fit.yaml"

num_scenes=${#scene_names[@]}
echo "Total scenes: ${num_scenes}"

for scene in "${scene_names[@]}"; do
  python eval_replica_colmap_vs_gt.py --scene-dir "${data_root}/${scene}" --sparse-dir "${data_root}/${scene}/output_mosca/sparse/0"
done
echo "All done."