# Pipeline details

The [README](../README.md) has the commands. This page gives the details: what each environment installs, the mask
files, the one-command runner, and what each step does with its parameters in `config.json`.

## Environments

Each stage of the pipeline runs in its own conda environment, with its own install script. A script clones its
third-party code into `deps/` at a specific commit. Each takes the environment name as an optional argument, e.g.
`bash install/tapnet.sh mytap`; without it, the name is the one of the script (`sam3`, `gmos`, `da3`, `tapnet`, `XM`,
`gs`).

- `sam3` (step 0): [SAM 3](https://github.com/facebookresearch/sam3) (`deps/sam3`) with PyTorch 2.10 and Jupyter. The
  checkpoint (3.5 GB) is gated (request access at [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3)
  and log in with `hf auth login`) and is downloaded on the first run of the mask notebook.
- `gmos` (step 0 without a notebook, optional): [GMOS](https://github.com/Jyxarthur/gmos) (`deps/gmos`) with PyTorch
  2.5 and its weights: Pi3 (3.8 GB), SAM 2.1 (0.9 GB) and the GMOS proposer (0.1 GB, from Google Drive). It is not part
  of the one-environment install below.
- `da3` (step 1): [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) (`deps/Depth-Anything-3`)
  with PyTorch 2.9. The model (DA3NESTED-GIANT-LARGE-1.1, 6.8 GB) is downloaded from Hugging Face on the first run.
- `tapnet` (step 2 and the graph completion of step 4): the two 2D trackers, with PyTorch 2.9: the
  [TAPNet](https://github.com/google-deepmind/tapnet) code (`deps/tapnet`) with the TAPNext++ checkpoint (2.5 GB),
  which tracks the static scene, and [AllTracker](https://github.com/aharley/alltracker) (`deps/alltracker`, checkpoint
  66 MB), which tracks the moving objects.
- `XM` (steps 3–5): the environment for the XM solver, which ships compiled in `XM/build` (Python 3.10, CUDA 12.4). Its
  CUDA libraries come from pip, so no CUDA toolkit is needed. Also Ceres and COLMAP bindings.
- `gs` (step 6): [gsplat](https://github.com/nerfstudio-project/gsplat) (`deps/gsplat`) with PyTorch 2.10, with its
  CUDA kernels compiled, which needs the CUDA toolkit (nvcc 12.x; set `CUDA_HOME` if it is not in `/usr/local/cuda`).
  The trainer is gsplat's `examples/` with our two changed files from `gsplat_patch/examples/` copied over them, in
  `build/gsplat_examples/`; `deps/gsplat` itself stays unmodified.

**One environment instead.** The environments above are the reference setup. Alternatively, `bash install/xm4d.sh`
installs `sam3`, `da3`, `tapnet` and `XM` into one environment `xm4d` (Python 3.10, PyTorch 2.9 for every stage); step 6
keeps its own `gs`. Then use `conda activate xm4d` wherever the steps activate `sam3`, `da3`, `tapnet` or `XM`. On our
tests it gives the same results as the four environments within their run-to-run variation. The one consistent
difference is that the SAM 3 masks differ in fewer than 0.1% of the mask pixels (PyTorch 2.9 instead of 2.10).

**Check.** `python install/check_env.py` (with any Python 3 that has conda on the PATH, e.g. conda's `base`) runs the
check of every stage in that stage's environment (the imports, the GPU and the model weights) and ends with a summary;
it exits with an error if an environment is missing or fails. For step 0 it checks `sam3`, or `gmos` if `sam3` is not
installed (both if both are), and warns if neither is installed. `--env xm4d` checks the one-environment install, and an
environment installed under another name is given as `<stage>=<env>`, e.g. `python install/check_env.py tapnet=mytap`.

## Step 0: masks

The category of an object tells the later steps what it is: `rigid` (moves as one rigid body), `human` (a person, hands
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

GMOS (`make_masks_gmos.py`) separates the moving objects and tracks each one through the video. Each is saved as its own
object, `masks/other_gmos<k>/`, with the category `other` for now, and merged as above. Objects saved with the notebook
stay; a new GMOS run replaces the objects of the previous one. GMOS compares frames about half a second apart, so give
the frame rate of your frames (`--fps`, default 24). On our test videos of up to 150 frames it used 12–19 GB of GPU
memory; `--offload-cpu` lowers that for long videos.

Step 1 merges the object folders itself when they are not merged yet, and stops with a warning while masks are missing.

## Running the steps

`bash run_pipeline.sh` starts the steps that do not wait for each other at the same time: with two GPUs, step 1 and
then step 3 on the first, the two trackers of step 2 on the second (the moving-object tracker waits for the depth only
for its 3D lift); with one GPU, step 3 next to the moving-object tracker. Each step runs in its conda environment
(`DA3_ENV`, `TAPNET_ENV`, `XM_ENV`; `xm4d` for all three with the one-environment install), its output goes to
`outputs/logs/run_<step>.log`, and `SEQUENTIAL=1` runs the steps one after the other. It checks and merges the masks
first (`check_masks.py`), and with `"no_masks": true` it stops after step 3.

It also sets `NUMPY_MADVISE_HUGEPAGE=0`: on our workstation numpy's requests for huge pages for big arrays stalled for
seconds in memory compaction (step 5 took 233 s instead of 139 s); set it too when you run the steps one by one. On
DyCheck teddy (472 frames), steps 1–4 took 3.9 min with two GPUs and 5.5 min with one; the same steps one by one took
10.9 min with an earlier release (15.3 min without `NUMPY_MADVISE_HUGEPAGE=0`). Steps 1–5 took 4.3 and 5.8 min (12.9
before).

Step 2 writes its visualisation videos (`outputs/viz_*.mp4`) and the unfiltered static tracks
(`outputs/tracks_debug.npz`), step 3 its lifted tracks (`tracks_debug_3d_solver.npz`) and XM matrices (`Abar.bin`,
`Q.mat`), and step 5 its depth-check and density images and videos (`dyn_depth_reject*`, `dyn_density_2d*`), only with
`"DEBUG": true` in `config.json`. Step 6 reads its training settings (learning rates, loss weights, `max_steps`,
`steps_scaler`) from `config.json` too; a command-line flag overrides them.

## Step 1: depth and intrinsics

Step 1 gives every frame the same intrinsics: the median of Depth Anything 3's per-frame focals, with the principal
point at the image centre (`DA3_SINGLE_INTRINSIC`). Frames wider than 2:1 are squeezed to 2:1 for Depth Anything 3
(`DA3_ASPECT_CAP`). Known intrinsics replace them: `"INTRINSICS": {"<scene>": {"fx": …, "fy": …, "cx": …, "cy": …,
"width": …, "height": …}}`, in px of frames of that width and height (scaled to the size of yours), with
`"CERES_OPTIMIZE_INTRINSICS": false` so that step 3 keeps them. With `"LIDAR_DEPTH": true` the depth is the scene's
sensor depth, `<scene>/depth/*.png` (one 16-bit PNG per frame, in mm, 0 where there is none), and Depth Anything 3's
depth, fitted to it by a scale and a shift per frame, fills its holes.

## Step 2: 2D tracks

Step 2 tracks the static scene with TAPNext++ and the moving objects with AllTracker. A static point is hidden while it
is on a moving object, a moving point while it is off its object.

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
  Each pass costs the same however few points it adds; `ALLTRACKER_MIN_NEW` (0, off) skips a keyframe that would add
  fewer new points than this share of the points visible there. On DyCheck, 0.05 made this tracking 1.4 times faster
  again but left up to 2% more of the object untracked.

Both trackers run in half precision (`ONLINE_FP16`, `ALLTRACKER_FP16`; `false` for full precision). On our test
videos, half precision with 30² points per frame gave the same cameras (16 videos of all six datasets) and the same
moving-point accuracy (DyCheck human keypoints, 72.9 vs 73.2% within 16 px) as full precision with 60², in 5.3 times
less time for the static scene and 1.9 times less for the moving objects (DyCheck teddy, 472 frames: 11.3 → 1.8 min
and 5.3 → 2.7 min).

## Step 3: cameras and static points

Step 3 refines the XM cameras with Ceres. The reprojection residuals have a Cauchy loss of `CERES_LOSS_PX` px (2;
`CERES_LOSS`: `cauchy`, `huber`, `softl1` or `trivial`), and the shared focal is refined too
(`CERES_OPTIMIZE_INTRINSICS`). Videos of at most 100 frames with at least 60 landmarks per frame use Ceres' dense
Schur solver, all others the iterative one (`XM_CERES_LINEAR_SOLVER` forces either). On our 13 test videos of at most
100 frames this made step 3 2.4 times faster (239 → 99 s). Ceres stops when an iteration lowers the cost by less than
`CERES_FUNCTION_TOLERANCE` (1e-4) of it; Ceres' default 1e-6 took 2.7 to 3.7 times longer on our 16 test videos
(316 → 85–118 s of Ceres).

## Step 4: embedded deformation graph

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

## Step 5: 4D Gaussian data

Step 5 lifts the moving-object depth pixels of every frame and binds each one to the graph at its own frame. With
`DYN_BIND` `"motion"` (default) a pixel takes the nodes that best carry the track seen within `DYN_BIND_PX` px (3)
of it, fitted over that track's frames; `"nearest"` takes the 6 nearest nodes. The 4D Gaussians start from at most
`GS_MAX_POINTS` points (500000), about `GS_DYN_MIN_POINTS` (50000) of them on the moving objects. With
`GS_DATA_FACTOR` > 1 (1), step 5 also writes the frames downscaled by it and the step-6 command it prints trains on
them.
