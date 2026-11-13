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
third-party code into `deps/` at the tested commit and creates the environment with specific versions. Each takes the
environment name as an optional argument, e.g.
`bash install/tapnet.sh mytap`; without it, the name is the one in the heading (`sam3`, `gmos`, `da3`, `tapnet`, `XM`).
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

### 3. `tapnet`: tracking (step 2)

```bash
bash install/tapnet.sh
```

Installs the 2D tracker [AllTracker](https://github.com/aharley/alltracker) (`deps/alltracker`, checkpoint 66 MB)
and, for the optional TAPNext tracking below, the [TAPNet](https://github.com/google-deepmind/tapnet) code
(`deps/tapnet`) with the TAPNext++ checkpoint (2.5 GB), with PyTorch 2.9. The check runs both trackers on the GPU.

### 4. `XM`: solver, deformation graph and 4DGS data (steps 3–5)

```bash
bash install/XM.sh
```

Installs the environment for the XM solver, which ships compiled in `XM/build` (Python 3.10, CUDA 12.4). Its CUDA
libraries come from pip, so no CUDA toolkit is needed. Also installs Ceres and COLMAP bindings, and
[TEASER++](https://github.com/MIT-SPARK/TEASER-plusplus) (`deps/TEASER-plusplus`, built from source).

### One environment instead (optional)

The four environments above are the reference setup. Alternatively, everything installs into one environment `xm4d`
(Python 3.10, PyTorch 2.9 for every stage):

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

### Steps 1–5

```bash
cd Python

conda activate da3
python get_depths_clean.py           # 1. monocular depth (Depth Anything 3) -> outputs/depth_anything_3/

conda activate tapnet
python static_tracking_online.py     # 2. 2D tracks (AllTracker) of the static scene
python dynamic_tracking.py           #    and of the moving objects -> outputs/tracks_dynamic_only.npz

conda activate XM
python static_solver.py              # 3. cameras + static points (XM solver) -> outputs/result_XM.npz
python EDG_old.py                    # 4. embedded deformation graph + dynamic 3D tracks -> outputs/results_all.npz
python train_4dgs_edg.py --dynamic-init-source depth   # 5. 4D Gaussian training data -> outputs/gs_edg_target_0000/
```

Set `EDG_NO_VIZ=1` to skip the Open3D window at the end of step 4. Step 5 lifts the moving points from the depth and
the masks (`--dynamic-init-source depth`; without it, step 5 expects a shape prior). It writes the training set and
prints the matching training command for a modified gsplat `simple_trainer.py` (`--edg-results`). That trainer is not
included in this repository yet.

Step 2 tracks with AllTracker. Every `ALLTRACKER_KEYFRAME` frames (default 10), it adds points on a grid wherever no
tracked point is visible nearby: off the moving objects for the static scene, on them for the moving objects. Each
keyframe's points are tracked forward to the end of the video in one dense pass, at `ALLTRACKER_RES` px (512) on the long
side. A point counts as visible while AllTracker's visibility × confidence is at least `ALLTRACKER_VIS_THRESHOLD`
(0.6). A static point is hidden while it is on a moving object, a moving point while it is off its object.
`DYN_ALLTRACKER_GRID` (2) is the spacing of the moving points, in px at the tracking resolution.

### Optional: track with TAPNext instead

The earlier tracker, the online TAPNext++ tracker, is still included. To use it for step 2, set it in `config.json`:

```json
"TRACKER": "tapnext"
```

It is not used by default. The TAPNext path keeps its own filters, which AllTracker does not need: it cuts a track at
a large 2D move against its neighbours and at a point lost off its object.

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
