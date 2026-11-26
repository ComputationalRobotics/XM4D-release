import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

# ---- the scene and every parameter: config.json at the repo root ----
# "scene" is a folder in datasets/ (or an absolute path) holding the frames or photos in images/. "input" says which:
# "video" (the frames of one video, sorted by the number in the file name) or "photos" (an unordered photo collection).
# The other keys are the depth / tracking / solver parameters, read through utils.dataset_config.
CONFIG_PATH = REPO_ROOT / "config.json"
active_config = json.loads(CONFIG_PATH.read_text())
rootpath = Path(active_config["scene"])
if not rootpath.is_absolute():
    rootpath = REPO_ROOT / "datasets" / rootpath
dataset_name = rootpath.name
print(f"[config] {CONFIG_PATH} -> scene {rootpath}")

input_mode = active_config.get("input", "video")
if input_mode not in ("video", "photos"):
    raise ValueError(f'config.json "input" must be "video" or "photos", got {input_mode!r}')
photos = input_mode == "photos"

output_dir = (rootpath / "outputs").absolute()
video_dir = rootpath / "images"
mask_dir = rootpath / "masks"

# step 1's output: outputs/depth_anything_3/ (DEPTH_MODEL "da3") or outputs/unidepth/ ("unidepth")
DEPTH_MODELS = {"da3": "depth_anything_3", "unidepth": "unidepth"}
depth_model = active_config.get("DEPTH_MODEL", "da3")
if depth_model not in DEPTH_MODELS:
    raise ValueError(f'config.json "DEPTH_MODEL" must be one of {sorted(DEPTH_MODELS)}, got {depth_model!r}')
depth_dir = rootpath / "outputs" / DEPTH_MODELS[depth_model]
# photos: one file per photo, <depth_dir>/photos/<photo name>.npz with its depth, confidence (at the photo's size) and K
photo_depth_dir = depth_dir / "photos"
photo_tracks_path = output_dir / "photo_tracks.npz"   # step 2 for photos (match_photos.py)

# frames: .jpg, or .png when the scene has no .jpg frames
image_pattern = "*.jpg" if any(video_dir.glob("*.jpg")) else "*.png"
mask_pattern = "*.png"


def photo_paths():
    """The photos of a photo collection (input "photos"): every .jpg/.jpeg/.png in images/, sorted by name. A photo's
    index in this list is its camera index in the solver's result."""
    return sorted(p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"))
