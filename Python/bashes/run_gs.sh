


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

conda activate gsplat

CUDA_VISIBLE_DEVICES=0 python /home/fenchel/4DReconstruction/deps/gsplat/examples/simple_trainer.py default --data-dir /home/fenchel/4DReconstruction/datasets/Syn4D/bear4/outputs/gs_edg_target_0000 --result-dir /home/fenchel/4DReconstruction/datasets/Syn4D/bear4/outputs/gsplat_edg_target_0000 --data-factor 1 --edg-results /home/fenchel/4DReconstruction/datasets/Syn4D/bear4/outputs/results_all.npz --edg-target-frame 0

CUDA_VISIBLE_DEVICES=0 python /home/fenchel/4DReconstruction/deps/gsplat/examples/simple_trainer.py default --data-dir /home/fenchel/4DReconstruction/datasets/Syn4D/casa1/outputs/gs_edg_target_0000 --result-dir /home/fenchel/4DReconstruction/datasets/Syn4D/casa1/outputs/gsplat_edg_target_0000 --data-factor 1 --edg-results /home/fenchel/4DReconstruction/datasets/Syn4D/casa1/outputs/results_all.npz --edg-target-frame 0

CUDA_VISIBLE_DEVICES=0 python /home/fenchel/4DReconstruction/deps/gsplat/examples/simple_trainer.py default --data-dir /home/fenchel/4DReconstruction/datasets/Syn4D/casa2/outputs/gs_edg_target_0000 --result-dir /home/fenchel/4DReconstruction/datasets/Syn4D/casa2/outputs/gsplat_edg_target_0000 --data-factor 1 --edg-results /home/fenchel/4DReconstruction/datasets/Syn4D/casa2/outputs/results_all.npz --edg-target-frame 0

CUDA_VISIBLE_DEVICES=0 python /home/fenchel/4DReconstruction/deps/gsplat/examples/simple_trainer.py default --data-dir /home/fenchel/4DReconstruction/datasets/Syn4D/casa3/outputs/gs_edg_target_0000 --result-dir /home/fenchel/4DReconstruction/datasets/Syn4D/casa3/outputs/gsplat_edg_target_0000 --data-factor 1 --edg-results /home/fenchel/4DReconstruction/datasets/Syn4D/casa3/outputs/results_all.npz --edg-target-frame 0