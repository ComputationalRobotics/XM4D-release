#!/bin/bash
set -e

# init conda
source "$HOME/miniconda3/etc/profile.d/conda.sh"

cd "$HOME/4DReconstruction/Python"

videoname="${1:-}"
if [[ -z "$videoname" ]]; then
  echo "Usage: $0 <videoname>"
  exit 1
fi

# run main pipeline
conda activate XM
## reconstruct_3d.py: only largest connected component per frame
## reconstruct_3d_v2.py: use everything
# python reconstruct_3d.py --videoname ${videoname}
# same but skips frames with <400 points, and uses interpolation to fill in missing frames:
python reconstruct_3d_v3.py --videoname ${videoname}
# reconstruct 4D video:
#### python reconstruct_4d_video.py --videoname ${videoname}
# python visualize.py --videoname train --which dyn --fps 20
# to visualize EDG nodes:
# python visualize.py --videoname train --which edg --fps 20
#### python visualize_merged.py --videoname ${videoname} --outputs_root ../deps/tapnet/outputs --fps 5
# python visualize_merged_multiframe.py --videoname ${videoname} --outputs_root ../deps/tapnet/outputs --fps 5

# python visualize_merged_improved.py --videoname ${videoname} --outputs_root ../deps/tapnet/outputs --fps 5
conda deactivate

# Note: ####: removed only because not needed for evaluation of camera poses. otherwise add