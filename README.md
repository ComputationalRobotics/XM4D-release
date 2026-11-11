# XM4D — reproduction branch

This branch reproduces the results of XM4D ("Ours") from preprocessed inputs:

- **camera accuracy** on the seven benchmarks of the paper: Sintel, Replica, TUM, Syn4D, NVIDIA, DyCheck and IMC;
- **novel-view rendering** on Syn4D, NVIDIA and DyCheck.

The learned-model preprocessing (depth estimation, 2D tracking) is already done and is shipped as data. This branch
contains the XM static solver that turns those inputs into camera poses, the EDG solver and 4D Gaussian training that
turn the dynamic tracks into a 4D reconstruction, and the evaluation of both.

## Contents

| path | what |
|---|---|
| `Python/static_solver.py` | XM static solver: camera poses + static points from 2D tracks and monocular depth; scores the poses against the GT (TEASER++ similarity alignment, ATE / RPE) |
| `Python/configs/<dataset>.json` | per-dataset parameters, selected automatically from the dataset name |
| `Python/utils/` | GT readers, problem construction, Ceres refinement, EDG helpers |
| `Python/optimize_dynamic_tracks.py`, `build_edg.py`, `EDG_test_fast.py` (+ `EDG_test.py`), `EDG_old.py`, `complete_edg_table.py` | rendering part, EDG: lift the dynamic tracks to 3D, build the EDG graph, solve it per segment with XM, complete it over the whole video |
| `Python/train_4dgs_edg.py` | rendering part: prepares the 4D Gaussian training data from the camera solve and the completed EDG |
| `Python/estimate_ours_render_tto.py`, `estimate_nvidia.py` (+ `estimate_ours_render.py`, `eval_depth_gaussian_ours.py`) | rendering part: novel-view evaluation (PSNR / SSIM / LPIPS) |
| `gsplat_patch/examples/` | our 4D Gaussian trainer: modified copies of two gsplat example files (`simple_trainer.py`, `datasets/colmap.py`) |
| `XM/` | XM solver core: CUDA source (`src/`, `include/`, CMake) and our compiled Python binding `XM/build/XM.cpython-310-x86_64-linux-gnu.so` |
| `deps/TEASER-plusplus` | git submodule, [TEASER++](https://github.com/MIT-SPARK/TEASER-plusplus) at `52a9c52` |
| `deps/gsplat` | git submodule, [gsplat](https://github.com/nerfstudio-project/gsplat) at `48df619` (1.5.3), unmodified |
| `scripts/run_camera.sh` | solve + evaluate any subset of the 65 scenes, then print the dataset averages |
| `scripts/run_render.sh` | EDG + 4D Gaussians + novel-view evaluation for any subset of the 22 rendering scenes, then print the dataset averages |
| `scripts/setup_gsplat.sh` | builds `build/gsplat_examples/` = gsplat's `examples/` with our two files in place |
| `datasets/` | preprocessed inputs, downloaded separately — see [`datasets/README.md`](datasets/README.md) |

## Install

Requirements: Linux x86-64, an NVIDIA GPU with a driver for CUDA 12, a CUDA 12.x toolkit, conda, and a C++ compiler
(for TEASER++, and for XM if you build it yourself).

```bash
git clone --recursive -b reproduce https://github.com/ComputationalRobotics/XM4D-release.git
cd XM4D-release
conda create -n xm4d python=3.10 eigen=3.4.0 cmake -c conda-forge -y
conda activate xm4d
pip install -r requirements.txt
pip install ./deps/TEASER-plusplus

export CUDA_HOME=/usr/local/cuda-12.4      # your CUDA 12.x toolkit
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

The XM solver core is a CUDA library with a Python binding. You can use our compiled binding
`XM/build/XM.cpython-310-x86_64-linux-gnu.so` (Python 3.10, built with CUDA 12.4 and Eigen 3.4.0) as it is, or build
it yourself from `XM/`. The build needs `nvcc` from CUDA 12 (XM does not compile with CUDA 11) and CMake ≥ 3.18,
fetches pybind11, and replaces the binding in `XM/build/`:

```bash
cmake -S XM -B XM/build -DPython3_EXECUTABLE=$(which python) -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc
cmake --build XM/build -j
```

Check the installation from `Python/`, the solver's working directory:

```bash
cd Python && python -c "import sys; sys.path.append('../XM/build'); import XM, teaserpp_python, pyceres, pycolmap; print('ok')" && cd ..
```

### Rendering part

The EDG stages run in the `xm4d` env and need PyTorch there:

```bash
conda activate xm4d
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

Keep torch 2.9.1 here. The EDG completion is an optimisation whose result depends on the PyTorch version; with 2.9.1 it
reproduces the completed EDG of our run bit for bit.

The 4D Gaussian training and evaluation run in a second env, `xm4d-gs`, because gsplat's training code needs
numpy < 2. gsplat itself is the `deps/gsplat` submodule, built from source and left unmodified:

```bash
conda create -n xm4d-gs python=3.10 -y
conda activate xm4d-gs
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-gs.txt
pip install --no-build-isolation git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5
pip install --no-build-isolation -e deps/gsplat        # compiles gsplat's CUDA kernels (CUDA_HOME as above)
bash scripts/setup_gsplat.sh
conda activate xm4d
```

Our 4D Gaussian trainer is a modified copy of two gsplat example files, kept in `gsplat_patch/examples/`.
`scripts/setup_gsplat.sh` copies gsplat's `examples/` to `build/gsplat_examples/` and puts our two files over the
originals there, so the submodule itself stays clean. The trainer sets `prune_scale3d=0.3` for gsplat's
`DefaultStrategy`, whose default is 0.1.

## Run

### Camera accuracy

```bash
bash scripts/run_camera.sh all          # 65 scenes, one GPU; prints the dataset averages
```

`bash scripts/run_camera.sh Sintel` runs one dataset and `bash scripts/run_camera.sh Sintel alley_1 cave_2` runs single
scenes. `GPU=1` selects another GPU. The runner never writes into `datasets/`: each scene is solved in
`results/camera/<Dataset>/<scene>/` (links to the inputs + the solver outputs `result_XM.npz`, logs) and its metrics
are written to `metrics.json` there. The dataset averages are also written to `results/camera/averages.txt`.

### Rendering

```bash
bash scripts/run_render.sh all          # 22 scenes, one GPU, in the xm4d env; prints the dataset averages
```

`bash scripts/run_render.sh Dycheck` and `bash scripts/run_render.sh Dycheck apple` work as for the camera runner, and
`GPU=1` selects another GPU. The runner uses the `xm4d` env for the EDG stages and runs the training and evaluation
with `conda run -n xm4d-gs` (`GS_ENV` changes the env name). Each scene starts from the merged dynamic tracks in
`datasets/render/` and goes through these stages:

1. lift the tracks to 3D;
2. build the EDG;
3. solve it per segment with XM;
4. complete it;
5. train the 4D Gaussians;
6. evaluate the held-out views with test-time camera alignment.

Everything is written to `results/render/<Dataset>/<scene>/` (log `render.log`, metrics `metrics.json`); the dataset
averages also go to `results/render/averages.txt`.

## Expected results

### Camera accuracy

Dataset averages from `bash scripts/run_camera.sh all` (one RTX 6000 Ada, about 40 min in total):

| dataset | scenes | ATE-T | RPE-T | RPE-R |
|---|---|---|---|---|
| Sintel | 23 | 0.017 | 0.026 | 0.040 |
| Replica | 8 | 0.002 | 0.002 | 0.001 |
| TUM | 6 | 0.006 | 0.008 | 0.006 |
| Syn4D | 8 | 0.002 | 0.002 | 0.003 |
| NVIDIA | 7 | 0.038 | 0.056 | 0.005 |
| DyCheck | 7 | 0.008 | 0.011 | 0.010 |
| IMC | 6 | 0.046 | 0.076 | 0.048 |

Per-scene numbers can differ slightly between runs: the Ceres refinement is multi-threaded and not bit-for-bit
repeatable, and the solves stop at wall-clock limits (`XM_MAX_TIME`, `CERES_MAX_SOLVE_TIME` in the configs).
IMC varies the most. Its row is the mean of three runs, whose averages spanned ATE-T 0.044–0.051.
`XM_MAX_TIME` limits each rank of the staircase separately. IMC uses 40 s: `gate_01` needs 34 s at rank 4, and with
the default 30 s it is cut off and fails about half the time. On a slower GPU, raise `XM_MAX_TIME`.

### Rendering

Novel-view dataset averages (one RTX 6000 Ada, about 9 h for the 22 scenes; per scene about 13 min on Syn4D, 27 min
on NVIDIA and 33 min on DyCheck):

| dataset | scenes | PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| Syn4D | 8 | 21.74 | 0.822 | 0.148 |
| NVIDIA | 7 | 26.96 | 0.867 | 0.063 |
| DyCheck | 7 | 19.51 | 0.725 | 0.162 |

On Syn4D and DyCheck, everything up to the completed EDG is deterministic and identical to our run. On NVIDIA it is
not, because `EDG_old.py` picks its first EDG node at random (farthest-point sampling with an unseeded start). The 4D
Gaussian training and the test-time alignment are not deterministic on any dataset, so per-scene PSNR moves by up to
about 0.3 dB between runs.
