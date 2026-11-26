# XM: Building Rome with Convex Optimization (static branch of XM4D)

**Haoyu Han, Heng Yang**

### [Project page](https://computationalrobotics.seas.harvard.edu/XM/) · [arXiv](https://arxiv.org/abs/2502.04640)

XM is a scalable, initialization-free solver for global bundle adjustment with learned depth: every camera's
observations are lifted to 3D with monocular depth, and XM solves the resulting scaled bundle adjustment globally with
a convex relaxation on the GPU, before a Ceres refinement. This `static` branch is the static part of
[XM4D](https://github.com/ComputationalRobotics/XM4D-release): structure from motion for a **video** or an unordered
**photo collection**, with the depth and 2D-track models we tested for XM4D. It is the updated
[XM-code](https://github.com/ComputationalRobotics/XM-code): the same solver, with the Python of XM4D around it.

| Step | Video | Photo collection |
|---|---|---|
| 0. masks of moving objects (optional) | SAM 3 or GMOS | — |
| 1. depth + intrinsics | Depth Anything 3 (default) or UniDepth V2 | same, per photo |
| 2. 2D tracks | TAPNext++ (default) or AllTracker | COLMAP SIFT matching + GLOMAP view graph |
| 3. cameras + points | XM, then Ceres | same |

## Install

Requirements: Linux x86-64, an NVIDIA GPU with a driver that supports CUDA 12.8, conda, git, wget, and a C++ compiler.

```bash
git clone -b static https://github.com/ComputationalRobotics/XM4D-release.git XM-static
cd XM-static
bash install/da3.sh        # step 1, depth: Depth Anything 3
bash install/tapnet.sh     # step 2 for videos, tracking: TAPNext++ and AllTracker
bash install/XM.sh         # step 3, the XM solver + Ceres
python install/check_env.py
```

Optional, for what you use:

```bash
bash install/sam3.sh       # step 0, masks of the moving objects in a video: SAM 3 (or install/gmos.sh, no notebook)
bash install/unidepth.sh   # step 1 with UniDepth V2
bash install/glomap.sh     # step 2 for photos: builds COLMAP 3.11 + our GLOMAP fork (2-10 min, 3.6 GB)
```

Each script makes one conda environment, named after the script, clones its code into `deps/` at a fixed commit and
downloads its model weights (SAM 3, Depth Anything 3 and UniDepth on their first run). `install/glomap.sh` compiles
with conda-forge's compilers and CUDA 12.4 into `build/glomap/` (no system CUDA toolkit or sudo needed); its binaries
use the `glomap` environment's libraries, so keep that environment.

> [!WARNING]
> **The SAM 3 checkpoint is gated.** Request access at [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3),
> then log in once with `conda activate sam3 && hf auth login`.

**The XM solver** ships compiled in `XM/build` (Python 3.10, CUDA 12.4). To build it from its source in `XM/` instead
(CUDA 12 `nvcc`, about 20 s), in the `XM` environment:

```bash
conda activate XM
cmake -S XM -B XM/build -DPython3_EXECUTABLE=$(which python) -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc
cmake --build XM/build -j
```

## Usage

Put the frames of your video, or your photos, in `datasets/<scene>/images/` (.jpg or .png; video frames sorted by the
number in the file name) and set in `config.json`, which holds every parameter:

```json
{
  "scene": "my_scene",
  "input": "video",
  "DEPTH_MODEL": "da3",
  "STATIC_TRACKER": "tapnext",
  ...
}
```

`"scene"` is the folder name or an absolute path; `"input"` is `"video"` or `"photos"`; `"DEPTH_MODEL"` is `"da3"`
(Depth Anything 3) or `"unidepth"` (UniDepth V2); for a video, `"STATIC_TRACKER"` is `"tapnext"` (TAPNext++) or
`"alltracker"`.

The repo ships an example video, `datasets/sintel_alley_2` (10 frames of MPI Sintel with the mask of the running girl),
which `config.json` points to.

**Step 0 (videos with moving objects): their masks.** The static scene is tracked off the moving objects. Make their
masks with `Python/make_masks_sam3.ipynb` (SAM 3, a few clicks or a text prompt per object) or
`python make_masks_gmos.py --fps 24` (GMOS, automatic), as in step 0 of
[XM4D](https://github.com/ComputationalRobotics/XM4D-release#usage). For a video without moving objects, set
`"no_masks": true`. Photos need no masks.

**Steps 1–3** run as one command from the repo root:

```bash
bash run_pipeline.sh
```

<details>
<summary>Or one step at a time, from <code>Python/</code></summary>

```bash
cd Python
conda activate da3 && python get_depths_clean.py            # 1. depth + K -> outputs/depth_anything_3/
#  or conda activate unidepth && python get_depths_unidepth.py   #    (UniDepth V2) -> outputs/unidepth/
conda activate tapnet && python static_tracking_online.py   # 2. video: 2D tracks -> outputs/output_tapnet_da3.npz
#  or conda activate XM && python match_photos.py            #    photos: matches -> outputs/photo_tracks.npz
conda activate XM && python static_solver.py                # 3. cameras + points -> outputs/result_XM.npz, colmap/
```

</details>

**Outputs.** `outputs/colmap/` holds the cameras and the 3D points as a COLMAP text model (`cameras.txt`,
`images.txt`, `points3D.txt`; the points without colour). `outputs/result_XM.npz` holds the raw solution: `R`
(3 × 3N camera-to-world rotations), `t` (3 × N camera centres), `p` (3 × M points), `s` (per-camera depth scale),
`indices_all` (frame or photo → solved camera, -1 if dropped) and `focal_scale` (Ceres' correction of step 1's focal).

<details>
<summary>Options</summary>

- `DEPTH_MODEL`: `"da3"` (Depth Anything 3) or `"unidepth"` (UniDepth V2). For a video both give every frame one K:
  the median of the model's focals, principal point at the centre (`DA3_SINGLE_INTRINSIC` for Depth Anything 3). For
  photos each photo keeps its own K; Depth Anything 3 runs on sets of up to 120 photos of the same orientation, resized
  to one size, with square pixels (on single photos its depth is much worse). Known intrinsics, one camera for every
  frame or photo, replace the model's, with `"CERES_OPTIMIZE_INTRINSICS": false`:
  `"INTRINSICS": {"<scene>": {"fx": …, "fy": …, "cx": …, "cy": …, "width": …, "height": …}}`. UniDepth then predicts
  its depth with them; Depth Anything 3 ignores them.
- `STATIC_TRACKER` (videos): `"tapnext"` (TAPNext++, online, `ONLINE_*` keys) or `"alltracker"` (AllTracker from
  keyframes every `ALLTRACKER_KEYFRAME` frames at `STATIC_ALLTRACKER_RES` px, visible at
  `STATIC_ALLTRACKER_VIS_THRESHOLD`; ~24 GB of GPU memory at 1024 px for 1024 × 768 frames).
- Photos: every photo's camera is set to its K from step 1 (so step 2 runs after step 1); COLMAP extracts SIFT features
  and matches every pair of photos on the GPU (exhaustive, with calibrated two-view geometry); our GLOMAP fork
  (`glomap/`) estimates the relative poses, drops the pairs that disagree with rotation averaging, keeps the largest
  connected set of photos and joins the matches into tracks.
- Step 3: `LAMBDA` (scale regularization of the first XM solve), `STATIC_TRACK_DEPTH_MAX` / `_GRAD_MAX` (videos:
  observations dropped at far depth or at depth edges, in metres), `CERES_LOSS` / `CERES_LOSS_PX` (robust loss of the
  refinement), `CERES_OPTIMIZE_INTRINSICS` (refine one shared focal correction).

</details>

**Which options.** On our tests (XM + Ceres; ATE-T: camera-position error after a similarity alignment, relative to
the extent of the camera path; mean over the scenes):

| | Depth Anything 3 | UniDepth V2 |
|---|---|---|
| videos: 5 scenes of Sintel, Replica and TUM, TAPNext++ tracks | 0.0039 | 0.0038 |
| photos: 4 IMC scenes of 200 photos, no known intrinsics | 0.064 | 0.16 |

**Examples of the solver alone** (`examples/`, in the `XM` environment, from the repo root):
`python examples/xm_solve_Q.py` solves XM's relaxation for a given matrix Q; `python examples/xm_solve_observations.py`
builds the problem from 3D observations, solves it and compares with the ground truth.

## License

The code is released under the [MIT License](LICENSE). `glomap/` is a modified copy of
[GLOMAP](https://github.com/colmap/glomap) (BSD 3-Clause, see `glomap/LICENSE`). The example scene
`datasets/sintel_alley_2` comes from MPI Sintel (© Blender Foundation, CC BY 3.0; see its README). The third-party code
and models installed into `deps/` keep their own licenses.

## Citation

```bibtex
@inproceedings{han25rss-xm,
  title={Building Rome with Convex Optimization},
  author={Han, Haoyu and Yang, Heng},
  booktitle={Robotics: Science and Systems (RSS)},
  year={2025}
}

@article{han26arxiv-xm4d,
  title={XM4D: 4D Reconstruction through Certifiable Geometric Optimization},
  author={Han, Haoyu and Jedlicki, Jack Benarroch and Lei, Jiahui and Yang, Heng},
  journal={arXiv preprint arXiv:2611.07902},
  year={2026}
}
```
