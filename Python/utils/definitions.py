import os
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MPI_SINTEL_TRAINING_ROOT = Path(
    os.environ.get(
        "XM_MPI_SINTEL_ROOT",
        REPO_ROOT / "datasets" / "MPI-Sintel-depth-training-20150305" / "training",
    )
)

dataset_cat = os.environ.get("XM_DATASET_CAT", "Dycheck")  # "Sintel", "DAVIS", "TUM", "Replica", "Syn4D", "Lab", "IMC", "NVIDIA" or "Custom"
dataset_name = os.environ.get("XM_DATASET_NAME", "paper-windmill")  # e.g. "alley_2" for Sintel, "bear" for DAVIS, "rgbd_dataset_freiburg1_desk" for TUM
depth_source = os.environ.get("XM_DEPTH_SOURCE", "da3")

# ---- per-dataset parameter config (Python/configs/*.json) ----
# Selected from dataset_cat/dataset_name above, so changing the dataset here
# automatically switches the tracking+solver parameters. Syn4D splits by scene
# prefix (casa* vs bear*). dataset_config.py exposes these values (env vars win).
CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _config_key(cat, name):
    if cat == "Syn4D":
        return "syn4d_casa" if str(name).lower().startswith("casa") else "syn4d_bear"
    return str(cat).lower()


active_config_name = _config_key(dataset_cat, dataset_name)


def _scene_config_name(cat, name):
    """Optional per-SCENE override file, layered on top of the category config:
    configs/<cat>_<scene>.json (e.g. custom_cup.json). Lets a single hard scene carry its
    own tuned params without disturbing sibling scenes that share the category config."""
    safe = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(name).lower())
    return f"{str(cat).lower()}_{safe}"


scene_config_name = _scene_config_name(dataset_cat, dataset_name)


def _load_active_config():
    cfg = {}
    loaded = []
    for fname in ("default.json", f"{active_config_name}.json", f"{scene_config_name}.json"):
        path = CONFIG_DIR / fname
        if path.exists():
            cfg.update(json.loads(path.read_text()))
            loaded.append(fname)
    return cfg, loaded


active_config, _loaded_cfgs = _load_active_config()
print(f"[config] {dataset_cat}/{dataset_name} -> {' + '.join(_loaded_cfgs)}")

rootpath = Path(
    os.environ.get(
        "XM_DATASET_ROOT",
        str(REPO_ROOT / "datasets" / dataset_cat / dataset_name),
    )
)

output_dir = (rootpath / "outputs").absolute()
tracks_dynamics_path = rootpath / "outputs" / "tracks_dynamic_only.npz"
video_dir = rootpath / "images"
mask_dir = rootpath / os.environ.get("XM_MASK_SUBDIR", "masks")

if dataset_cat == "Sintel":
    _local_gt_depth_dir = rootpath / "depths"
    _fallback_gt_depth_dir = MPI_SINTEL_TRAINING_ROOT / "depth" / dataset_name
    gt_depth_dir = _local_gt_depth_dir if _local_gt_depth_dir.exists() else _fallback_gt_depth_dir
else:
    gt_depth_dir = rootpath / "depths"

da3_depth_dir = rootpath / "outputs" / "depth"
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
# Falls back to the da3 slot (depth/) if the variant catalog hasn't been built for this scene yet.
depth_variant = os.environ.get("XM_DEPTH_VARIANT", "aligned_da3")
_variant_dir = rootpath / "outputs" / "depths" / depth_variant
active_depth_dir = _variant_dir if (_variant_dir / "depth_maps.npy").exists() else da3_depth_dir
active_depth_maps = active_depth_dir / "depth_maps.npy"        # (T,H,W) THE depth to lift from
active_intrinsics = active_depth_dir / "intrinsics.npy"        # K matching that depth (GT for Dycheck)
print(f"[depth] variant={depth_variant} -> {active_depth_dir}"
      f"{'' if active_depth_dir == _variant_dir else '  (variant not built; using da3 slot)'}")

segment_length = 10  # number of frames per segment for tracking

if dataset_cat == "DAVIS":
    image_pattern = "*.jpg"
    mask_pattern = "*.png"
elif dataset_cat == "TUM":
    image_pattern = "*.png"
    mask_pattern = "*.png"
elif dataset_cat == "Sintel":
    image_pattern = "*.png"
    mask_pattern = "*.png"
elif dataset_cat == "Replica":
    image_pattern = "*.jpg"
    mask_pattern = "*.png"
elif dataset_cat == "Syn4D":
    image_pattern = "*.jpg"
    mask_pattern = "*.png"
elif dataset_cat == "Lab":
    image_pattern = "*.jpg"
    mask_pattern = "*.png"
elif dataset_cat == "IMC":
    image_pattern = "*.jpg"
    mask_pattern = "*.png"
elif dataset_cat == "NVIDIA":
    image_pattern = "*.png"
    mask_pattern = "*.png"
elif dataset_cat == "Dycheck":
    image_pattern = "*.png"
    mask_pattern = "*.png"
elif dataset_cat == "Custom":
    image_pattern = "*.jpg"
    mask_pattern = "*.png"
