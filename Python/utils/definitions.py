import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

# ---- the scene and every parameter: config.json at the repo root ----
# "scene" is a folder in datasets/ (or an absolute path) holding the video frames in images/. The other keys are the
# tracking / solver / training parameters, read through utils.dataset_config.
CONFIG_PATH = REPO_ROOT / "config.json"
active_config = json.loads(CONFIG_PATH.read_text())
rootpath = Path(active_config["scene"])
if not rootpath.is_absolute():
    rootpath = REPO_ROOT / "datasets" / rootpath
dataset_name = rootpath.name
print(f"[config] {CONFIG_PATH} -> scene {rootpath}")

output_dir = (rootpath / "outputs").absolute()
tracks_dynamics_smoothed_path = rootpath / "outputs" / "tracks_dynamic_only_smoothed.npz"   # smooth_dynamic_tracks.py


def dynamic_tracks_for_edg():
    """The moving points' tracks the deformation-graph steps read: the smoothed ones (smooth_dynamic_tracks.py)."""
    return tracks_dynamics_smoothed_path
video_dir = rootpath / "images"
mask_dir = rootpath / "masks"

da3_depth_dir = rootpath / "outputs" / "depth_anything_3"

# frames: .jpg, or .png when the scene has no .jpg frames
image_pattern = "*.jpg" if any(video_dir.glob("*.jpg")) else "*.png"
mask_pattern = "*.png"
