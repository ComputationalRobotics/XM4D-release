# XM: Building Rome with Convex Optimization (static branch of XM4D)

**Haoyu Han, Heng Yang**

### [Project page](https://computationalrobotics.seas.harvard.edu/XM/) · [arXiv](https://arxiv.org/abs/2502.04640)

XM is a scalable, initialization-free solver for global bundle adjustment with learned depth: every camera's
observations are lifted to 3D with monocular depth, and XM solves the resulting scaled bundle adjustment globally with
a convex relaxation on the GPU, before a Ceres refinement. This `static` branch is the static part of
[XM4D](https://github.com/ComputationalRobotics/XM4D-release), structure from motion for a video of a static scene, and
the updated [XM-code](https://github.com/ComputationalRobotics/XM-code): the same solver, with the Python of XM4D around
it.

**Pipeline:** frames → depth and intrinsics (Depth Anything 3) → 2D tracks of the scene (TAPNext++) → cameras and 3D
points (XM, then Ceres). Other depth and tracking models, photo collections and masks of moving objects are optional
(the folded sections below).

## Install

Requirements: Linux x86-64, an NVIDIA GPU with a driver that supports CUDA 12.8, conda, git, wget, and a C++ compiler.

```bash
git clone -b static https://github.com/ComputationalRobotics/XM4D-release.git XM-static
cd XM-static
bash install/da3.sh        # step 1, depth: Depth Anything 3
bash install/tapnet.sh     # step 2, tracking: TAPNext++
bash install/XM.sh         # step 3, the XM solver + Ceres
python install/check_env.py
```

Each script makes one conda environment, named after the script, clones its code into `deps/` at a fixed commit and
downloads its model weights (Depth Anything 3 on its first run).

<details>
<summary>Optional installs: masks of moving objects, UniDepth V2, photo collections</summary>

```bash
bash install/sam3.sh       # masks of the moving objects in a video: SAM 3 (or install/gmos.sh, without a notebook)
bash install/unidepth.sh   # step 1 with UniDepth V2
bash install/glomap.sh     # step 2 for photo collections: builds COLMAP 3.11 + our GLOMAP fork (2-10 min, 3.6 GB)
```

`install/tapnet.sh` also installs AllTracker, the optional tracker. `install/glomap.sh` compiles with conda-forge's
compilers and CUDA 12.4 into `build/glomap/` (no system CUDA toolkit or sudo needed); its binaries use the `glomap`
environment's libraries, so keep that environment. `python install/check_env.py` checks the optional environments too
when they are installed.

> [!WARNING]
> **The SAM 3 checkpoint is gated.** Request access at [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3),
> then log in once with `conda activate sam3 && hf auth login`.

</details>

<details>
<summary>Build the XM solver from source</summary>

The XM solver ships compiled in `XM/build` (Python 3.10, CUDA 12.4). To build it from its source in `XM/` instead (CUDA
12 `nvcc`, about 20 s), in the `XM` environment:

```bash
conda activate XM
cmake -S XM -B XM/build -DPython3_EXECUTABLE=$(which python) -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc
cmake --build XM/build -j
```

</details>

## Usage

Put the frames of your video in `datasets/<scene>/images/` (.jpg or .png, sorted by the number in the file name) and
set `"scene"` in `config.json` (the folder name, or an absolute path), which holds every parameter. For a static scene,
also set `"no_masks": true`:

```json
{
  "scene": "my_scene",
  "no_masks": true,
  ...
}
```

The repo ships an example video, `datasets/sintel_alley_2` (10 frames of MPI Sintel with the mask of the running girl),
which `config.json` points to. Then run steps 1–3 from the repo root:

```bash
bash run_pipeline.sh
```

<details>
<summary>Or one step at a time, from <code>Python/</code></summary>

```bash
cd Python
conda activate da3 && python get_depths_clean.py            # 1. depth + K -> outputs/depth_anything_3/
conda activate tapnet && python static_tracking_online.py   # 2. 2D tracks -> outputs/output_tapnet_da3.npz
conda activate XM && python static_solver.py                # 3. cameras + points -> outputs/result_XM.npz, colmap/
```

</details>

**Outputs.** `outputs/colmap/` holds the cameras and the 3D points as a COLMAP text model (`cameras.txt`,
`images.txt`, `points3D.txt`; the points without colour). `outputs/result_XM.npz` holds the raw solution: `R`
(3 × 3N camera-to-world rotations), `t` (3 × N camera centres), `p` (3 × M points), `s` (per-camera depth scale),
`indices_all` (frame → solved camera, -1 if dropped) and `focal_scale` (Ceres' correction of step 1's focal).

<details>
<summary>Videos with moving objects: their masks (SAM 3 or GMOS)</summary>

The static scene is tracked off the moving objects, so a video with moving objects needs their masks, and
`"no_masks": false`. Make them with `Python/make_masks_sam3.ipynb` in the `sam3` environment (SAM 3, a few clicks or a
text prompt per object) or, without a notebook, with GMOS in the `gmos` environment:

```python
track_points("dog", frame=0, points=[(420, 310)])   # or a text prompt: track_text("people", "person")
preview_video("dog")                                 # check it
save("dog", "other")                                 # category: "rigid", "human" or "other"
merge_masks()                                        # at the end: all objects into <scene>/masks/
```

```bash
cd Python && conda activate gmos && python make_masks_gmos.py --fps 24   # --fps: the frame rate of your frames
```

More in step 0 of [XM4D](https://github.com/ComputationalRobotics/XM4D-release#usage).

</details>

<details>
<summary>Options: UniDepth V2, AllTracker, photo collections</summary>

Set in `config.json`:

- `"DEPTH_MODEL"`: `"da3"` (Depth Anything 3, default) or `"unidepth"` (UniDepth V2: run `get_depths_unidepth.py` in the
  `unidepth` environment for step 1). For a video both give every frame one K: the median of the model's focals,
  principal point at the centre (`DA3_SINGLE_INTRINSIC` for Depth Anything 3). Known intrinsics, one camera for every
  frame, replace the model's, with `"CERES_OPTIMIZE_INTRINSICS": false`:
  `"INTRINSICS": {"<scene>": {"fx": …, "fy": …, "cx": …, "cy": …, "width": …, "height": …}}`. UniDepth then predicts
  its depth with them; Depth Anything 3 ignores them.
- `"STATIC_TRACKER"`: `"tapnext"` (TAPNext++, default; `ONLINE_*` keys) or `"alltracker"` (AllTracker from keyframes
  every `ALLTRACKER_KEYFRAME` frames at `STATIC_ALLTRACKER_RES` px, visible at `STATIC_ALLTRACKER_VIS_THRESHOLD`;
  ~24 GB of GPU memory at 1024 px for 1024 × 768 frames).
- `"input"`: `"video"` (default) or `"photos"`, for an unordered photo collection in `images/` (photos of any size; no
  masks). Step 2 is then `match_photos.py` in the `XM` environment, after step 1: every photo's camera is set to its K
  from step 1, COLMAP extracts SIFT features and matches every pair of photos on the GPU, and our GLOMAP fork
  (`glomap/`) estimates the relative poses, drops the pairs that disagree with rotation averaging, keeps the largest
  connected set of photos and joins the matches into tracks. Each photo keeps its own K; Depth Anything 3 runs on sets
  of up to 120 photos of the same orientation, resized to one size, with square pixels.
- Step 3: `LAMBDA` (scale regularization of the first XM solve), `STATIC_TRACK_DEPTH_MAX` / `_GRAD_MAX` (videos:
  observations dropped at far depth or at depth edges, in metres), `CERES_LOSS` / `CERES_LOSS_PX` (robust loss of the
  refinement), `CERES_OPTIMIZE_INTRINSICS` (refine one shared focal correction).

`run_pipeline.sh` runs the steps that these settings select. On our tests (XM + Ceres; ATE-T: camera-position error
after a similarity alignment, relative to the extent of the camera path; mean over the scenes):

| | Depth Anything 3 | UniDepth V2 |
|---|---|---|
| videos: 5 scenes of Sintel, Replica and TUM, TAPNext++ tracks | 0.0039 | 0.0038 |
| photos: 4 IMC scenes of 200 photos, no known intrinsics | 0.064 | 0.16 |

</details>

<details>
<summary>Examples of the solver alone</summary>

In the `XM` environment, from the repo root: `python examples/xm_solve_Q.py` solves XM's relaxation for a given matrix
Q; `python examples/xm_solve_observations.py` builds the problem from 3D observations, solves it and compares with the
ground truth.

</details>

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
