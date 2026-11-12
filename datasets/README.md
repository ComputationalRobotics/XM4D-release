# Data

## Download

Download link: **TODO**.

## Where to put it

Unpack it into this folder, `datasets/`, so that the layout below exists. The runners only read from here and write
everything to `results/`.

## Expected structure

Each scene holds the dataset itself (frames, masks, ground truth, held-out views) and the preprocessed depth and
tracks. Everything else is computed by `scripts/run_camera.sh` and `scripts/run_render.sh`.

```
datasets/
├── Sintel/<scene>/                    # 23 scenes
│   ├── images/
│   ├── masks/
│   ├── cameras/                       # GT
│   └── outputs/
│       ├── output_tapnet_da3.npz      # static tracks
│       └── depth_anything_3/{depth_maps,conf_maps,intrinsics}.npy
├── Replica/<scene>/                   # 8 scenes: images/, traj.txt (GT), outputs/ as Sintel; no masks/
├── TUM/<scene>/                       # 6 scenes: images/, masks/, groundtruth.txt (GT), outputs/ as Sintel
├── Syn4D/<scene>/                     # 8 scenes
│   ├── images/
│   ├── masks/
│   ├── colmap/                        # GT
│   ├── novel/                         # held-out views
│   └── outputs/
│       ├── output_tapnet_da3.npz      # static tracks
│       ├── tracks_dynamic_only.npz    # dynamic tracks
│       ├── tracks_dynamic_merged.npz  # dynamic tracks, merged
│       └── depth_anything_3/{depth_maps,conf_maps,intrinsics}.npy
├── NVIDIA/<scene>/                    # 7 scenes: as Syn4D, with poses_bounds.npy (GT) and novel_orig/ (held-out views)
├── Dycheck/<scene>/                   # 7 scenes
│   ├── images/
│   ├── masks/
│   ├── cameras/                       # GT
│   ├── novel/                         # held-out views
│   ├── frame_map.txt
│   └── outputs/
│       ├── output_tapnet_da3.npz      # static tracks
│       ├── tracks_dynamic_only.npz    # dynamic tracks
│       ├── tracks_dynamic_merged.npz  # dynamic tracks, merged
│       ├── depth_anything_3/{conf_maps,intrinsics}.npy
│       └── depths/lidar_filled/{depth_maps,intrinsics}.npy   # iPhone LiDAR depth, holes filled
└── IMC/<scene>/                       # 6 scenes
    ├── images/
    ├── sparse/                        # GT (COLMAP text model)
    └── outputs/output_tapnet_unidepth.npz   # static tracks with depth
```

The static tracks are TAPNext tracks. The NVIDIA `depth_anything_3/intrinsics.npy` holds the GT intrinsics.
