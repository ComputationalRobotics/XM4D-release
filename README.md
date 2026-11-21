# XM4D: 4D Reconstruction through Certifiable Geometric Optimization

**Haoyu Han, Jack Benarroch Jedlicki, Jiahui Lei, Heng Yang**

### [Project page](https://computationalrobotics.seas.harvard.edu/XM4D/) · [arXiv](https://arxiv.org/abs/2611.07902)

XM4D splits 4D reconstruction into a static and a dynamic stage, and solves each one with a certifiably globally optimal
GPU solver before any photometric training.

<p align="center">
  <img src="assets/teaser.gif" width="100%" alt="XM4D on DAVIS train: 4D Gaussian render, 3D tracks, deformation graph">
  <br>
  <em>DAVIS train. Left to right: 4D Gaussian render, 3D tracks, embedded deformation graph.</em>
</p>

> **Reproducing the paper's results:** please see the
> [`reproduce` branch](https://github.com/ComputationalRobotics/XM4D-release/tree/reproduce), which ships the
> preprocessed benchmark inputs, the pinned configs and the runners. This `main` branch is the method code, for running
> XM4D on your own videos.

## Install

Requirements: Linux x86-64, an NVIDIA GPU with a driver that supports CUDA 12.8, conda, git, wget, and a C++ compiler.

```bash
git clone https://github.com/ComputationalRobotics/XM4D-release.git
cd XM4D-release
```

Each stage of the pipeline runs in its own conda environment, with its own install script. A script clones its
third-party code into `deps/` at a specific commit. Each takes the environment name as an optional argument, e.g.
`bash install/tapnet.sh mytap`; without it, the name is the one in the heading (`sam3`, `gmos`, `da3`, `tapnet`, `XM`,
`gs`).
When all environments are installed, check them with one command (see [Check the installation](#check-the-installation)).

### 1. `sam3`: masks (step 0)

> [!WARNING]
> **The SAM 3 checkpoint is gated: request access yourself.** Open [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3),
> request access to the weights and wait for the approval. Then log in with your Hugging Face token. Without access, the
> installation check fails at "SAM 3 weights".

```bash
bash install/sam3.sh
conda activate sam3 && hf auth login          # once, with your Hugging Face token
```

Installs [SAM 3](https://github.com/facebookresearch/sam3) (`deps/sam3`) with PyTorch 2.10 and Jupyter. The
checkpoint (3.5 GB) is downloaded on the first run of the mask notebook.

### 1b. `gmos`: masks without a notebook (optional)

```bash
bash install/gmos.sh
```

Installs [GMOS](https://github.com/Jyxarthur/gmos) (`deps/gmos`) with PyTorch 2.5 and its weights: Pi3 (3.8 GB),
SAM 2.1 (0.9 GB) and the GMOS proposer (0.1 GB, from Google Drive). It replaces the `sam3` notebook where no notebook
can run, e.g. on a workstation (see step 0). It is not part of the one-environment install below.

### 2. `da3`: depth (step 1)

```bash
bash install/da3.sh
```

Installs [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) (`deps/Depth-Anything-3`) with
PyTorch 2.9. The model (DA3NESTED-GIANT-LARGE-1.1, 6.8 GB) is downloaded from Hugging Face on the first run.

### 3. `tapnet`: tracking (step 2) and the deformation-graph completion (step 4)

```bash
bash install/tapnet.sh
```

Installs the two 2D trackers, with PyTorch 2.9: the [TAPNet](https://github.com/google-deepmind/tapnet) code
(`deps/tapnet`) with the TAPNext++ checkpoint (2.5 GB), which tracks the static scene, and
[AllTracker](https://github.com/aharley/alltracker) (`deps/alltracker`, checkpoint 66 MB), which tracks the moving
objects. The check runs both trackers on the GPU.

### 4. `XM`: solver, deformation graph and 4DGS data (steps 3–5)

```bash
bash install/XM.sh
```

Installs the environment for the XM solver, which ships compiled in `XM/build` (Python 3.10, CUDA 12.4). Its CUDA
libraries come from pip, so no CUDA toolkit is needed. Also installs Ceres and COLMAP bindings, and
[TEASER++](https://github.com/MIT-SPARK/TEASER-plusplus) (`deps/TEASER-plusplus`, built from source).

### 5. `gs`: 4D Gaussian training (step 6)

```bash
bash install/gs.sh
```

Installs [gsplat](https://github.com/nerfstudio-project/gsplat) (`deps/gsplat`) with PyTorch 2.10 and compiles its CUDA
kernels, which needs the CUDA toolkit (nvcc 12.x; set `CUDA_HOME` if it is not in `/usr/local/cuda`). The trainer is
gsplat's `examples/` with our two changed files from `gsplat_patch/examples/` copied over them, in
`build/gsplat_examples/`; `deps/gsplat` itself stays unmodified.

### One environment instead (optional)

The environments above are the reference setup. Alternatively, `sam3`, `da3`, `tapnet` and `XM` install into one
environment `xm4d` (Python 3.10, PyTorch 2.9 for every stage); step 6 keeps its own `gs`:

```bash
bash install/xm4d.sh
```

Then use `conda activate xm4d` wherever the steps below activate `sam3`, `da3`, `tapnet` or `XM`. On our tests it gives
the same results as the four environments within their run-to-run variation. The one consistent difference is that the
SAM 3 masks differ in fewer than 0.1% of the mask pixels (PyTorch 2.9 instead of 2.10).

### Check the installation

When the environments are installed, check them all at once, with any Python 3 that has conda on the PATH (e.g. conda's
`base`):

```bash
python install/check_env.py                  # the environments above
python install/check_env.py --env xm4d       # or: the one-environment install
```

It runs the check of every stage in that stage's environment (the imports, the GPU and the model weights) and ends with
a summary; it exits with an error if an environment is missing or fails. For step 0 it checks `sam3`, or `gmos` if
`sam3` is not installed (both if both are), and warns if neither is installed. An environment installed under another
name is given as `<stage>=<env>`, e.g. `python install/check_env.py tapnet=mytap`.

## Usage

The repo ships an example scene, `datasets/sintel_alley_2`: 10 frames of MPI Sintel `alley_2` with the mask of the
running girl. `config.json` points to it, so every step below runs on it as it is. A second one, `datasets/hoi_ip1073`
(`"scene": "hoi_ip1073"`), has 10 frames of a person picking up a ring, with a human and a rigid object mask.

For your own video, put its frames in a scene folder:

```
datasets/<scene>/images/     frames (.jpg or .png), sorted by the number in the file name
```

Then set the scene in `config.json` at the repo root. The scene can also be an absolute path to a folder anywhere.
The same file holds every parameter of the pipeline; change them there.

```json
{
  "scene": "my_scene",
  "no_masks": false,
  ...
}
```

Every step runs from `Python/` and reads `config.json`; no environment variables are needed.

### Step 0: masks of the moving objects

XM starts from a mask of everything that moves. Open `Python/make_masks_sam3.ipynb` in the `sam3` environment. For
each moving object, give SAM 3 a few clicked points or a text prompt. SAM 3 tracks the object through the whole video;
check the result and save it:

```python
track_points("dog", frame=0, points=[(420, 310)])   # points on the object (labels=[1, 0, ...] marks negatives)
track_text("people", "person")                      # or a text prompt
preview_video("dog")                                 # check it
save("dog", "other")                                 # with its category: "rigid", "human" or "other"
merge_masks()                                        # at the end: all objects into the mask the pipeline reads
```

The category tells the later steps what the object is: `rigid` (moves as one rigid body), `human` (a person, hands
included) or `other` (anything else that moves); `hand` is coming soon. Each object is saved in its own folder named
after its category, and `merge_masks()` (in `Python/check_masks.py`, callable outside the notebook too) merges them:

```
<scene>/masks/<category>_<object>/<frame>.png   one folder per object (white = object)
<scene>/masks/objects.json                      the object folders in the order they were added
<scene>/masks/<frame>.png                       all objects merged: the mask the pipeline reads
<scene>/mask_labels/<frame>.png                 which object each pixel belongs to (labels.json: label -> object)
```

Where objects overlap, the pixel goes to the rigid object first, then hand, then human, then other; within a category
to the object added first. `save_object()` and `remove_object()` in `Python/check_masks.py` save and delete objects
outside the notebook.

Without a notebook, e.g. on a workstation, [GMOS](https://github.com/Jyxarthur/gmos) finds the moving objects
automatically instead, in the `gmos` environment:

```bash
cd Python
conda activate gmos
python make_masks_gmos.py --fps 24     # the frame rate of the frames in images/
```

GMOS separates the moving objects and tracks each one through the video. Each is saved as its own object,
`masks/other_gmos<k>/`, with the category `other` for now, and merged as above. Objects saved with the notebook stay;
a new GMOS run replaces the objects of the previous one. GMOS compares frames about half a second apart, so give the
frame rate of your frames (default 24). On our test videos of up to 150 frames it used 12–19 GB of GPU memory;
`--offload-cpu` lowers that for long videos.

Step 1 merges the object folders itself when they are not merged yet, and stops with a warning while masks are missing.
For a scene without any moving object, set `"no_masks": true` in `config.json`.

### Steps 1–6

```bash
cd Python
export NUMPY_MADVISE_HUGEPAGE=0      # see below

conda activate da3
python get_depths_clean.py           # 1. monocular depth (Depth Anything 3) -> outputs/depth_anything_3/

conda activate tapnet
python static_tracking_online.py     # 2. 2D tracks of the static scene (TAPNext++) -> outputs/output_tapnet_da3.npz
python dynamic_tracking.py           #    and of the moving objects (AllTracker) -> outputs/tracks_dynamic_only.npz

conda activate XM
python static_solver.py              # 3. cameras + static points (XM solver) -> outputs/result_XM.npz
python smooth_dynamic_tracks.py      # 4. the moving points' 3D tracks, smoothed -> outputs/tracks_dynamic_only_smoothed.npz
python edg_topology.py               #    embedded deformation graph: nodes, edges, point links -> outputs/edg_topology.npz
python EDG_test_fast.py              #    its motion, by XM segment solves -> outputs/edg_test/edg_table.npz
conda activate tapnet
python complete_edg_table.py         #    completed over every frame -> outputs/edg_test/edg_completed.npz, results_all.npz
conda activate XM
python train_4dgs_edg.py             # 5. 4D Gaussian training data -> outputs/gs_edg_target_0000/

conda activate gs                    # 6. 4D Gaussians: the command step 5 prints, e.g.
python ../build/gsplat_examples/simple_trainer.py default --data-dir ../datasets/<scene>/outputs/gs_edg_target_0000 \
    --result-dir ../datasets/<scene>/outputs/gsplat --data-factor 1 \
    --edg-results ../datasets/<scene>/outputs/results_all.npz --edg-target-frame 0
```

**DyCheck, NVIDIA and Syn4D:** `configs/` holds our tuned config for each of these datasets. Copy
`configs/dycheck.json`, `configs/nvidia.json`, `configs/syn4d_bear.json` (bear1–4) or `configs/syn4d_casa.json`
(casa1–4) over `config.json` and set `"scene"` (the scene folder keeps the dataset's scene name, e.g. `apple`). They set
each scene's intrinsics from the dataset (`INTRINSICS`; the focal is then not refined), DyCheck's LiDAR depth
(`LIDAR_DEPTH`), and the 4D Gaussian settings and training length of each dataset (NVIDIA trains at half resolution);
the Syn4D configs also turn on `SMOOTH_DEPTH_REPAIR` and `EDG_RIGID_ENDS` (step 4 below). Everything else is as in
`config.json`, the config for any other video.

Steps 1–5 also run as one command from the repository root, `bash run_pipeline.sh`. It starts the steps that do not
wait for each other at the same time: with two GPUs, step 1 and then step 3 on the first, the two trackers of step 2
on the second (the moving-object tracker waits for the depth only for its 3D lift); with one GPU, step 3 next to the
moving-object tracker. Each step runs in its conda environment (`DA3_ENV`, `TAPNET_ENV`, `XM_ENV`; `xm4d` for all three
with the one-environment install), its output goes to `outputs/logs/run_<step>.log`, and `SEQUENTIAL=1` runs the steps
one after the other. It checks and merges the masks first (`check_masks.py`), and with `"no_masks": true` it stops after
step 3. It also sets `NUMPY_MADVISE_HUGEPAGE=0`: on our workstation numpy's requests for huge pages for
big arrays stalled for seconds in memory compaction (step 5 took 233 s instead of 139 s); set it too when you run the
steps one by one. On DyCheck teddy (472 frames), steps 1–4 took 3.9 min with two GPUs and 5.5 min with one; the same
steps one by one took 10.9 min with an earlier release (15.3 min without `NUMPY_MADVISE_HUGEPAGE=0`). Steps 1–5 took
4.3 and 5.8 min (12.9 before).

Set `EDG_NO_VIZ=1` to skip the Open3D windows. Step 2 writes its visualisation videos (`outputs/viz_*.mp4`) and the
unfiltered static tracks (`outputs/tracks_debug.npz`), step 3 its lifted tracks (`tracks_debug_3d_solver.npz`) and
XM matrices (`Abar.bin`, `Q.mat`), and step 5 its depth-check and density images and videos (`dyn_depth_reject*`,
`dyn_density_2d*`), only with `"DEBUG": true` in `config.json`. Step 6 reads its training settings
(learning rates, loss weights, `max_steps`, `steps_scaler`) from `config.json` too; a command-line flag overrides them.

Step 1 gives every frame the same intrinsics: the median of Depth Anything 3's per-frame focals, with the principal
point at the image centre (`DA3_SINGLE_INTRINSIC`). Frames wider than 2:1 are squeezed to 2:1 for Depth Anything 3
(`DA3_ASPECT_CAP`). Known intrinsics replace them: `"INTRINSICS": {"<scene>": {"fx": …, "fy": …, "cx": …, "cy": …,
"width": …, "height": …}}`, in px of frames of that width and height (scaled to the size of yours), with
`"CERES_OPTIMIZE_INTRINSICS": false` so that step 3 keeps them. With `"LIDAR_DEPTH": true` the depth is the scene's
sensor depth, `<scene>/depth/*.png` (one 16-bit PNG per frame, in mm, 0 where there is none), and Depth Anything 3's
depth, fitted to it by a scale and a shift per frame, fills its holes.

Step 2 tracks the static scene with TAPNext++ (`STATIC_TRACKER`) and the moving objects with AllTracker
(`DYNAMIC_TRACKER`). A static point is hidden while it is on a moving object, a moving point while it is off its object.

- Static scene: TAPNext++ starts from a 50 × 50 grid of points off the moving objects and tracks them frame by frame.
  At every frame it draws `ONLINE_RESAMPLE_GRID`² random points (30², on the frame resized to 256 × 256) and adds
  those with no visible point within `ONLINE_RESAMPLE_RADIUS` px (4), up to 20000 points. A point counts as visible
  while TAPNext's visibility × certainty is above `ONLINE_CERTAINTY_THRESHOLD` (0.8). A track whose frame-to-frame
  motion differs from that of the points around it by more than `MOVE_DELETE_FLOW_ERROR_THRESHOLD` px (10) keeps only
  its longest stable part. With 20000 points it used up to about 45 GB of GPU memory on our videos (nvidia-smi),
  independent of the video length.
- Moving objects: every `ALLTRACKER_KEYFRAME` frames (default 10), AllTracker adds points on a grid on the object
  wherever no tracked point is visible nearby and tracks them forward in one dense pass at `ALLTRACKER_RES` px (512),
  to the end of the video. A point counts as visible while visibility × confidence is at least
  `ALLTRACKER_VIS_THRESHOLD` (0.6). `DYN_ALLTRACKER_GRID` (2) is the spacing of the moving points, in px at the
  tracking resolution. 384 px takes 1.6 times less time per pass and gives fewer points (teddy: 30k instead of 50k).
  Each pass costs the same however few points it adds; `ALLTRACKER_MIN_NEW` (0, off) skips a
  keyframe that would add fewer new points than this share of the points visible there. On DyCheck, 0.05 made this
  tracking 1.4 times faster again but left up to 2% more of the object untracked.

Both trackers run in half precision (`ONLINE_FP16`, `ALLTRACKER_FP16`; `false` for full precision). On our test
videos, half precision with 30² points per frame gave the same cameras (16 videos of all six datasets) and the same
moving-point accuracy (DyCheck human keypoints, 72.9 vs 73.2% within 16 px) as full precision with 60², in 5.3 times
less time for the static scene and 1.9 times less for the moving objects (DyCheck teddy, 472 frames: 11.3 → 1.8 min
and 5.3 → 2.7 min).

Step 3 refines the XM cameras with Ceres. The reprojection residuals have a Cauchy loss of `CERES_LOSS_PX` px (2;
`CERES_LOSS`: `cauchy`, `huber`, `softl1` or `trivial`), and the shared focal is refined too
(`CERES_OPTIMIZE_INTRINSICS`). Videos of at most 100 frames with at least 60 landmarks per frame use Ceres' dense
Schur solver, all others the iterative one (`XM_CERES_LINEAR_SOLVER` forces either). On our 13 test videos of at most
100 frames this made step 3 2.4 times faster (239 → 99 s). Ceres stops when an iteration lowers the cost by less than
`CERES_FUNCTION_TOLERANCE` (1e-4) of it; Ceres' default 1e-6 took 2.7 to 3.7 times longer on our 16 test videos
(316 → 85–118 s of Ceres).

Step 4 builds and solves the embedded deformation graph (EDG) of the moving objects:

- `smooth_dynamic_tracks.py`: with the cameras fixed, each moving point's 3D track keeps its 2D track (Cauchy, 2 px)
  and its depth (10%, weaker) and is smooth in time (acceleration, `SMOOTH_SIGMA_ACC` = 30 px per frame²). With
  `SMOOTH_DEPTH_REPAIR` > 0 (0, off), an observation whose log-depth is farther than that from its track's own smoothed
  log-depth is first moved along its ray onto it. The output is not compressed (1.3 GB on teddy), which saves about
  20 s here and a second in each of the four steps that read it.
- `edg_topology.py`: graph nodes are tracks that already describe their neighbourhood: a fast-moving node removes the
  tracks within 0.4 of its motion radius, a slow one the tracks that keep a nearly constant distance to it; up to
  `EDG_MAX_NODES` (200). Two nodes are joined when they move together (score of their typical distance and their
  relative motion) at a nearly constant distance. Each moving point joins its two nearest nodes that stay close in
  every frame they share.
- `EDG_test_fast.py`: the graph's motion over many short windows, each solved by XM; `complete_edg_table.py` turns
  them into one motion over every frame (as-rigid-as-possible over gaps of 1 to 256 frames, smooth in time; 200 Adam
  iterations: 400 lowered its final reprojection error by only 0.02 px on teddy). With `EDG_RIGID_ENDS` (`false`), a
  node's frames before its first and after its last solved window follow its 6 nearest solved neighbours rigidly.

Step 5 lifts the moving-object depth pixels of every frame and binds each one to the graph at its own frame. With
`DYN_BIND` `"motion"` (default) a pixel takes the nodes that best carry the track seen within `DYN_BIND_PX` px (3)
of it, fitted over that track's frames; `"nearest"` takes the 6 nearest nodes. The 4D Gaussians start from at most
`GS_MAX_POINTS` points (500000), about `GS_DYN_MIN_POINTS` (50000) of them on the moving objects. With
`GS_DATA_FACTOR` > 1 (1), step 5 also writes the frames downscaled by it and the step-6 command it prints trains on them.

### Optional: swap the trackers

Either part of step 2 can use either tracker, set in `config.json`:

```json
"STATIC_TRACKER": "alltracker",
"DYNAMIC_TRACKER": "tapnext"
```

AllTracker tracks the static scene at `STATIC_ALLTRACKER_RES` px (1024), for at most `STATIC_ALLTRACKER_MAX_LEN`
frames (100) from its keyframe, visible while visibility × confidence is at least `STATIC_ALLTRACKER_VIS_THRESHOLD`
(0.99), with an extra keyframe at any frame where fewer than `STATIC_ALLTRACKER_MIN_VISIBLE` points (200) are visible.
At 1024 px it needs about 24 GB of GPU memory for a 1024 × 768 video. TAPNext on the moving objects keeps its own filters: it cuts a track at a large 2D move against its
neighbours and at a point lost off its object.

## License

The code is released under the [MIT License](LICENSE). The example scene `datasets/sintel_alley_2` comes from MPI
Sintel (© Blender Foundation, CC BY 3.0; see its README). The third-party code and models installed into `deps/` keep
their own licenses, e.g. SAM 3 under Meta's SAM License and GMOS under Apache 2.0.

## Citation

```bibtex
@article{han26arxiv-xm4d,
  title={XM4D: 4D Reconstruction through Certifiable Geometric Optimization},
  author={Han, Haoyu and Jedlicki, Jack Benarroch and Lei, Jiahui and Yang, Heng},
  journal={arXiv preprint arXiv:2611.07902},
  year={2026}
}
```

The static solver is XM ([project page](https://computationalrobotics.seas.harvard.edu/XM/)):

```bibtex
@inproceedings{han25rss-xm,
  title={Building Rome with Convex Optimization},
  author={Han, Haoyu and Yang, Heng},
  booktitle={Robotics: Science and Systems (RSS)},
  year={2025}
}
```
