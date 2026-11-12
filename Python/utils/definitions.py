import os
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

# ---- the scene and every parameter: config.json at the repo root ----
# "scene" is a folder in datasets/ (or an absolute path) holding the video frames in images/. The other keys are the
# tracking / solver / training parameters, read through utils.dataset_config (an environment variable of the same
# name still overrides a parameter).
CONFIG_PATH = REPO_ROOT / "config.json"
active_config = json.loads(CONFIG_PATH.read_text())
active_config_name = CONFIG_PATH.name
rootpath = Path(active_config["scene"])
if not rootpath.is_absolute():
    rootpath = REPO_ROOT / "datasets" / rootpath
dataset_cat = "Custom"   # own video: no ground truth; the intrinsics come from Depth Anything 3
dataset_name = rootpath.name
depth_source = os.environ.get("XM_DEPTH_SOURCE", "da3")
print(f"[config] {CONFIG_PATH} -> scene {rootpath}")

output_dir = (rootpath / "outputs").absolute()
tracks_dynamics_path = rootpath / "outputs" / "tracks_dynamic_only.npz"
video_dir = rootpath / "images"
mask_dir = rootpath / os.environ.get("XM_MASK_SUBDIR", "masks")

gt_depth_dir = rootpath / "depths"

da3_depth_dir = rootpath / "outputs" / "depth_anything_3"
vggt_output_dir = rootpath / "output_vggt"
vggt_depth_dir = vggt_output_dir / "depths"
vggt_conf_dir = vggt_output_dir / "depth_conf"
spatracker_output_dir = rootpath / "outputs_spatracker"
spatracker_depth_dir = spatracker_output_dir / "spatracker_eval" / "training_depth"
unidepth_output_root = Path(
    os.environ.get("XM_UNIDEPTH_OUTPUT_ROOT", str(Path.home() / "mega-sam" / "UniDepth" / "outputs"))
)
unidepth_depth_dir = Path(os.environ.get("XM_UNIDEPTH_DEPTH_DIR", str(unidepth_output_root / dataset_name)))

# ---- ACTIVE depth variant (single source of truth for which depth everything lifts from) ----
# One of the folders built by build_depth_variants.py under outputs/depths/<variant>/ :
#   raw_da3 | aligned_da3 | lidar | lidar_filled
# Default: aligned_da3 (raw DepthAnything-3 per-frame scale+shift aligned to LiDAR). Override: XM_DEPTH_VARIANT.
# Falls back to the da3 slot (depth_anything_3/) if the variant catalog hasn't been built for this scene yet.
depth_variant = os.environ.get("XM_DEPTH_VARIANT", "aligned_da3")
_variant_dir = rootpath / "outputs" / "depths" / depth_variant
active_depth_dir = _variant_dir if (_variant_dir / "depth_maps.npy").exists() else da3_depth_dir
active_depth_maps = active_depth_dir / "depth_maps.npy"        # (T,H,W) THE depth to lift from
active_intrinsics = active_depth_dir / "intrinsics.npy"        # K matching that depth (GT for Dycheck)
print(f"[depth] variant={depth_variant} -> {active_depth_dir}"
      f"{'' if active_depth_dir == _variant_dir else '  (variant not built; using da3 slot)'}")

segment_length = 10  # number of frames per segment for tracking

# frames: .jpg, or .png when the scene has no .jpg frames
image_pattern = "*.jpg" if any(video_dir.glob("*.jpg")) else "*.png"
mask_pattern = "*.png"
