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
conda activate mosca

cd /home/fenchel/4DReconstruction/deps/MoSca/

DATASET="Sintel"
SCENES=(
    # "alley_1"
    # "alley_2"
    "ambush_2"
    "ambush_4"
    "ambush_5"
    "ambush_6"
    "ambush_7"
    "bamboo_1"
    "bamboo_2"
    "bandage_1"
    "bandage_2"
    "cave_2"
    "cave_4"
    "market_2"
    "market_5"
    "market_6"
    "mountain_1"
    "shaman_2"
    "shaman_3"
    "sleeping_1"
    "sleeping_2"
    "temple_2"
    "temple_3"
)

for scene in "${SCENES[@]}"; do
    echo "Processing ${DATASET}/${scene}"
    python mosca_precompute.py --cfg ./profile/sintel/sintel_prep_old.yaml --ws ../../datasets/${DATASET}/${scene}
    python lite_moca_reconstruct.py --cfg ./profile/sintel/sintel_fit_old.yaml --ws ../../datasets/${DATASET}/${scene}/outputs_mosca
done