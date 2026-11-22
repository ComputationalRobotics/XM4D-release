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
bash install/sam3.sh       # step 0, masks: SAM 3
bash install/da3.sh        # step 1, depth: Depth Anything 3
bash install/tapnet.sh     # step 2, tracking: TAPNext++ and AllTracker
bash install/XM.sh         # steps 3-5: XM solver, deformation graph, 4D Gaussian data
bash install/gs.sh         # step 6, 4D Gaussians: gsplat (needs the CUDA toolkit, nvcc 12.x)
python install/check_env.py
```

Each script makes one conda environment, named after the script, clones its code into `deps/` at a fixed commit and
downloads its model weights (SAM 3 and Depth Anything 3 on their first run). `install/xm4d.sh` installs steps 0–5 into
one environment instead, and `install/gmos.sh` installs GMOS for masks without a notebook. See
[docs/pipeline.md](docs/pipeline.md#environments).

> [!WARNING]
> **The SAM 3 checkpoint is gated.** Request access at [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3),
> then log in once with `conda activate sam3 && hf auth login`.

## Usage

Put the frames of your video in `datasets/<scene>/images/` (.jpg or .png, sorted by the number in the file name) and
set `"scene"` in `config.json` (the folder name, or an absolute path), which holds every parameter of the pipeline. The
repo ships two example scenes, `datasets/sintel_alley_2` (the default) and `datasets/hoi_ip1073`.

**Step 0: masks of the moving objects.** Open `Python/make_masks_sam3.ipynb` in the `sam3` environment and prompt SAM 3
for each moving object:

```python
track_points("dog", frame=0, points=[(420, 310)])   # or a text prompt: track_text("people", "person")
preview_video("dog")                                 # check it
save("dog", "other")                                 # category: "rigid", "human" or "other"
merge_masks()                                        # at the end: all objects into <scene>/masks/
```

Without a notebook, GMOS finds the moving objects automatically (`--fps`: the frame rate of your frames):

```bash
cd Python && conda activate gmos && python make_masks_gmos.py --fps 24
```

For a scene without moving objects, set `"no_masks": true` in `config.json`.

**Steps 1–5** run as one command from the repo root, on one or two GPUs:

```bash
bash run_pipeline.sh
```

or one by one, from `Python/`:

```bash
cd Python
export NUMPY_MADVISE_HUGEPAGE=0
conda activate da3
python get_depths_clean.py           # 1. depth (Depth Anything 3) -> outputs/depth_anything_3/
conda activate tapnet
python static_tracking_online.py     # 2. 2D tracks of the static scene (TAPNext++) -> outputs/output_tapnet_da3.npz
python dynamic_tracking.py           #    and of the moving objects (AllTracker) -> outputs/tracks_dynamic_only.npz
conda activate XM
python static_solver.py              # 3. cameras + static points (XM solver) -> outputs/result_XM.npz
python smooth_dynamic_tracks.py      # 4. the moving points' 3D tracks, smoothed
python edg_topology.py               #    embedded deformation graph
python EDG_test_fast.py              #    its motion, by XM segment solves
conda activate tapnet
python complete_edg_table.py         #    completed over every frame -> outputs/results_all.npz
conda activate XM
python train_4dgs_edg.py             # 5. 4D Gaussian training data -> outputs/gs_edg_target_0000/
```

**Step 6: 4D Gaussians.** Step 5 prints the command, e.g.:

```bash
conda activate gs
python ../build/gsplat_examples/simple_trainer.py default --data-dir ../datasets/<scene>/outputs/gs_edg_target_0000 \
    --result-dir ../datasets/<scene>/outputs/gsplat --data-factor 1 \
    --edg-results ../datasets/<scene>/outputs/results_all.npz --edg-target-frame 0
```

**DyCheck, NVIDIA and Syn4D:** use our tuned configs, see [configs/README.md](configs/README.md). What each step does
and its parameters: [docs/pipeline.md](docs/pipeline.md).

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
