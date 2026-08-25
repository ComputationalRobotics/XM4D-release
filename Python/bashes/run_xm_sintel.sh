#!/usr/bin/env bash
set -e

# Ensure conda is available in non-interactive shells
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
	# shellcheck source=/dev/null
	source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
	# shellcheck source=/dev/null
	source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
	echo "Could not find conda.sh. Please install conda or update run_preprocess.sh." >&2
	exit 1
fi

#!/usr/bin/env bash
DATASETS="NVIDIA"
# SCENES=(
#     "alley_1"
#     "alley_2"
#     "ambush_2"
#     "ambush_4"
#     "ambush_5"
#     "ambush_6"
#     "ambush_7"
#     "bamboo_1"
#     "bamboo_2"
#     "bandage_1"
#     "bandage_2"
#     "cave_2"
#     "cave_4"
#     "market_2"
#     "market_5"
#     "market_6"
#     "mountain_1"
#     "shaman_2"
#     "shaman_3"
#     "sleeping_1"
#     "sleeping_2"
#     "temple_2"
#     "temple_3"
# )

# SCENES=(
#     "office1_seq1"
#     "office1_seq2"
#     "office2_seq1"
#     "office2_seq2"
#     "room1_seq1"
#     "room1_seq2"
#     "room2_seq1"
#     "room2_seq2"
# )


# SCENES=(
#     "tum1"
#     "tum2"
#     "tum3"
#     "tum4"
#     "tum5"
#     "tum6"
# )


SCENES=(
    "Balloon1"
    "Balloon2"
    "Jumping"
    "Playground"
    "Skating"
    "Truck"
    "Umbrella"
)


for scene in "${SCENES[@]}"; do
    echo "Processing ${DATASETS}/${scene}"
    export XM_DATASET_CAT="${DATASETS}"
    export XM_DATASET_NAME="${scene}"
    # conda activate da3
    # python get_depths_clean.py
    # conda activate tapnet
    # python static_tracking_online.py
    conda activate XM
    python static_solver.py --no-da3-scale
done

# conda activate XM
# python static_solver.py