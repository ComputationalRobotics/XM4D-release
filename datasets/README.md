# Preprocessed inputs for the camera benchmark

Download link: **TODO** (≈23 GB, 65 scenes). Unpack it here so that `datasets/<Dataset>/<scene>/` exists for each scene
below. `scripts/run_camera.sh` only reads this folder; every solve is written to `results/`.

Every scene holds the solver inputs and the ground truth that scoring needs, nothing else. The learned-model
preprocessing (depth estimation, 2D tracking) is already done. These inputs are exactly the ones behind the paper's
camera-accuracy numbers.

## Per-scene contents

| dataset | scenes | ground truth | solver inputs (`outputs/`) |
|---|---|---|---|
| Sintel | 23 | `cameras/*.cam` | `output_tapnet_da3.npz`, `depth_anything_3/{depth_maps,conf_maps,intrinsics}.npy` |
| Replica | 8 | `traj.txt` | same as Sintel |
| TUM | 6 | `groundtruth.txt` | same as Sintel |
| Syn4D | 8 | `colmap/` | same as Sintel |
| NVIDIA | 7 | `poses_bounds.npy` | same as Sintel |
| Dycheck | 7 | `cameras/*.json` | `output_tapnet_da3.npz`, `depth_anything_3/{conf_maps,intrinsics}.npy`, `depths/aligned_da3/{depth_maps,intrinsics}.npy` |
| IMC | 6 | `sparse/` (COLMAP text model) | `output_tapnet_unidepth.npz` |

Every scene also has `images/`. All scenes except Replica and IMC also have `masks/`, the dynamic-object masks.

- **`output_tapnet_da3.npz`**: the static 2D tracks (TAPNext, outside the masks), lifted to 3D with the depth below.
  This is the solver's main input.
- **`depth_anything_3/`**: Depth Anything 3 depth and confidence per frame, plus the intrinsics the depth was lifted
  with. The intrinsics are Depth Anything 3's own estimate on Sintel, Replica, TUM and NVIDIA, and the GT intrinsics
  on Syn4D and Dycheck.
- **`depths/aligned_da3/`** (Dycheck only): the Depth Anything 3 depth after a per-frame scale and shift fit to the
  iPhone LiDAR, with the GT intrinsics.
- **`output_tapnet_unidepth.npz`** (IMC only): unordered photos have no video tracks. The tracks are therefore COLMAP
  SIFT matches chained by the GLOMAP view graph, lifted with UniDepth depth and the GT intrinsics.

Sintel `mountain_1` and `shaman_2` take their `cameras/` from the MPI-Sintel depth training set (`camdata_left`).

# Preprocessed inputs for the rendering benchmark

`datasets/render/<Dataset>/<scene>/` holds the inputs of `scripts/run_render.sh` for the 22 rendering scenes (Syn4D 8,
NVIDIA 7, Dycheck 7). They are the inputs the paper's rendering numbers were produced from: everything up to and
including the merged dynamic tracks, plus the camera solve of that run. The rendering run starts from these files, not
from the outputs of `scripts/run_camera.sh`.

## Per-scene contents

| dataset | scene folder | `outputs/` |
|---|---|---|
| Syn4D | `images/`, `masks/`, `colmap/` (GT cameras), `novel/` (held-out views) | common files |
| NVIDIA | `images/`, `masks/`, `poses_bounds.npy` (GT cameras), `novel_orig/` (held-out views) | common files, without `results_all.npz` |
| Dycheck | `images/`, `masks/`, `cameras/` (GT cameras), `depth/` (iPhone LiDAR), `frame_map.txt`, `novel/` (held-out views) | common files + `depths/` |

Common files in `outputs/`:

- **`tracks_dynamic_merged.npz`**: the merged 2D tracks on the dynamic objects. The EDG stages start from these.
- **`tracks_dynamic_only.npz`**: the dynamic tracks before merging.
- **`result_XM.npz`**: the camera solve of the rendering run (XM static solver).
- **`output_tapnet_da3.npz`**: the static 2D tracks that camera solve used.
- **`depth_anything_3/`**: Depth Anything 3 depth, confidence and intrinsics per frame.
- **`results_all.npz`**: static landmarks and the lifted dynamic tracks from the stages before the EDG. The completion
  stage writes the EDG into a copy of it.
- **`depths/`** (Dycheck only): the depth variants. The rendering run lifts with `lidar_filled`, the iPhone LiDAR depth
  with holes filled.
