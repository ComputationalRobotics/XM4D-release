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

conda activate da3
python get_depths_clean.py 

conda activate tapnet
python static_tracking_online.py 
python dynamic_tracking.py

conda activate XM
python static_solver.py

ALIGN_FRAME_INDEX="${ALIGN_FRAME_INDEX:-0}"
ALIGN_MAX_COMPONENTS="${ALIGN_MAX_COMPONENTS:-1}"
ALIGN_PRIOR_COORDINATE_SPACE="${ALIGN_PRIOR_COORDINATE_SPACE:-auto}"
ALIGN_PRIOR_IN_CAMERA_FRAME="${ALIGN_PRIOR_IN_CAMERA_FRAME:-false}"
ALIGN_CAMERA_POSE_SOURCE="${ALIGN_CAMERA_POSE_SOURCE:-auto}"
ALIGN_SHOW_OPEN3D="${ALIGN_SHOW_OPEN3D:-true}"

ALIGN_ARGS=(
	--frame-index "${ALIGN_FRAME_INDEX}"
	--max-components "${ALIGN_MAX_COMPONENTS}"
	--prior-coordinate-space "${ALIGN_PRIOR_COORDINATE_SPACE}"
	--camera-pose-source "${ALIGN_CAMERA_POSE_SOURCE}"
)

if [ -n "${ALIGN_PRIOR_PATH:-}" ]; then
	ALIGN_ARGS+=(--prior-path "${ALIGN_PRIOR_PATH}")
fi

if [ "${ALIGN_PRIOR_IN_CAMERA_FRAME}" = "1" ] || [ "${ALIGN_PRIOR_IN_CAMERA_FRAME,,}" = "true" ]; then
	ALIGN_ARGS+=(--prior-in-camera-frame)
fi

if [ "${ALIGN_SHOW_OPEN3D}" = "0" ] || [ "${ALIGN_SHOW_OPEN3D,,}" = "false" ]; then
	ALIGN_ARGS+=(--no-show-open3d)
fi

python align_pointcloud_prior.py "${ALIGN_ARGS[@]}"
python EDG_prior.py

# python EDG_old.py

python train_4dgs_edg.py



