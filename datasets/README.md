# Data

## Download

[Google Drive folder](https://drive.google.com/drive/folders/1j1M6_MA6fdwu9S6mOCl-iGKQvYR1UkIU?usp=sharing): one
archive per dataset (`Sintel`, `Replica`, `TUM`, `Syn4D`, `NVIDIA`, `Dycheck`, `IMC`; 19 GB in total) and
`SHA256SUMS`. The rendering benchmark needs only `Syn4D`, `NVIDIA` and `Dycheck`.

## Where to put it

Put the archives into this folder, `datasets/`, and unpack them there, so that the layout below exists. The runners
only read from here and write everything to `results/`.

```bash
cd datasets
sha256sum -c --ignore-missing SHA256SUMS      # optional: checks the downloaded archives
for f in *.tar.gz; do tar -xzf "$f"; done
```

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
│       └── depth/{depth_maps,conf_maps,intrinsics}.npy   # depth, confidence, intrinsics
├── Replica/<scene>/                   # 8 scenes: images/, traj.txt (GT), outputs/ as Sintel; no masks/
├── TUM/<scene>/                       # 6 scenes: images/, masks/, groundtruth.txt (GT), outputs/ as Sintel
├── Syn4D/<scene>/                     # 8 scenes
│   ├── images/
│   ├── masks/
│   ├── colmap/                        # GT
│   ├── novel/                         # held-out views
│   └── outputs/
│       ├── output_tapnet_da3.npz      # static tracks
│       ├── tracks_dynamic_merged.npz  # dynamic tracks, merged
│       └── depth/{depth_maps,conf_maps,intrinsics}.npy   # depth, confidence, intrinsics
├── NVIDIA/<scene>/                    # 7 scenes: as Syn4D, with poses_bounds.npy (GT) and novel_orig/ (held-out views)
├── Dycheck/<scene>/                   # 7 scenes
│   ├── images/
│   ├── masks/
│   ├── cameras/                       # GT
│   ├── novel/                         # held-out views
│   ├── frame_map.txt
│   └── outputs/
│       ├── output_tapnet_da3.npz      # static tracks
│       ├── tracks_dynamic_merged.npz  # dynamic tracks, merged
│       └── depth/{depth_maps,conf_maps,intrinsics}.npy   # depth, confidence, intrinsics
└── IMC/<scene>/                       # 6 scenes
    ├── images/
    ├── sparse/                        # GT (COLMAP text model)
    └── outputs/output_tapnet_unidepth.npz   # static tracks with depth
```

