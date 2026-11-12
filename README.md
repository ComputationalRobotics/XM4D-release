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

Requirements: Linux x86-64, an NVIDIA GPU with a driver that supports CUDA 12.8, conda, git, wget, and a C++ compiler
(to build TEASER++).

```bash
git clone https://github.com/ComputationalRobotics/XM4D-release.git
cd XM4D-release
```

Each stage of the pipeline runs in its own conda environment, with its own install script. A script clones its
third-party code into `deps/` at the tested commit, creates the environment with pinned versions, and ends with a check
that the environment can run its stage: the imports, the GPU and the model weights. A ready environment prints
`[check_env] <env>: ready`. Run the check again at any time:

```bash
conda activate <env> && python install/check_env.py <env>
```

The scripts never modify an existing environment. Set `ENV=<name>` to install under another name.

### 1. `sam3`: masks (step 0)

> [!WARNING]
> **The SAM 3 checkpoint is gated: request access yourself.** Open [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3),
> request access to the weights and wait for the approval. Then log in with your Hugging Face token. Without access, the
> check below fails at "SAM 3 weights".

```bash
bash install/sam3.sh
conda activate sam3 && hf auth login          # once, with your Hugging Face token
python install/check_env.py sam3
```

Installs [SAM 3](https://github.com/facebookresearch/sam3) (`deps/sam3`) with PyTorch 2.10 and Jupyter. The
checkpoint (3.5 GB) is downloaded on the first run of the mask notebook.

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

Installs the [TAPNet](https://github.com/google-deepmind/tapnet) code (`deps/tapnet`) and the TAPNext++ checkpoint
(2.5 GB) with PyTorch 2.9. The check runs TAPNext++ on the GPU.

### 4. `XM`: solver, deformation graph and 4DGS data (steps 3–5)

```bash
bash install/XM.sh
```

Installs the environment for the XM solver, which ships compiled in `XM/build` (Python 3.10, CUDA 12.4). Its CUDA
libraries come from pip, so no CUDA toolkit is needed. Also installs Ceres and COLMAP bindings, and
[TEASER++](https://github.com/MIT-SPARK/TEASER-plusplus) (`deps/TEASER-plusplus`, built from source).

### One environment instead (optional)

The four environments above are the reference setup. Alternatively, everything installs into one environment `xm4d`
(Python 3.10, PyTorch 2.9 for every stage). The script runs all four checks:

```bash
bash install/xm4d.sh
```

Then use `conda activate xm4d` wherever the steps below activate `sam3`, `da3`, `tapnet` or `XM`. On our tests it gives
the same results as the four environments within their run-to-run variation. The one consistent difference is that the
SAM 3 masks differ in fewer than 0.1% of the mask pixels (PyTorch 2.9 instead of 2.10).

## Usage

The repo ships an example scene, `datasets/sintel_alley_2`: 10 frames of MPI Sintel `alley_2` with the mask of the
running girl. `config.json` points to it, so every step below runs on it as it is.

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
save("dog")
```

Each object is saved in its own folder, and every save rebuilds the combined mask that the pipeline reads:

```
<scene>/masks/<object>/<frame>.png    one folder per object (white = object)
<scene>/masks/<frame>.png             all objects combined
```

Step 1 stops with a warning while these masks are missing. For a scene without any moving object, set
`"no_masks": true` in `config.json`.

### Steps 1–5

```bash
cd Python

conda activate da3
python get_depths_clean.py           # 1. monocular depth (Depth Anything 3) -> outputs/depth_anything_3/

conda activate tapnet
python static_tracking_online.py     # 2. 2D tracks of the static scene
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

## License

The code is released under the [MIT License](LICENSE). The example scene `datasets/sintel_alley_2` comes from MPI
Sintel (© Blender Foundation, CC BY 3.0; see its README). The third-party code and models installed into `deps/` keep
their own licenses, e.g. SAM 3 under Meta's SAM License.

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
