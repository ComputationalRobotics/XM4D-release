# XM4D — reproduction branch

Reproduces the XM4D camera-accuracy and novel-view-rendering results from preprocessed inputs.

## Install

Requirements: Linux x86-64, an NVIDIA GPU with a driver for CUDA 12, a CUDA 12.x toolkit, conda, and a C++ compiler.

### Camera env (`xm4d`)

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

The compiled XM binding `XM/build/XM.cpython-310-x86_64-linux-gnu.so` (Python 3.10, CUDA 12.4) works as it is. To
build it yourself instead (needs CUDA 12 `nvcc` and CMake ≥ 3.18):

```bash
cmake -S XM -B XM/build -DPython3_EXECUTABLE=$(which python) -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc
cmake --build XM/build -j
```

Check:

```bash
cd Python && python -c "import sys; sys.path.append('../XM/build'); import XM, teaserpp_python, pyceres, pycolmap; print('ok')" && cd ..
```

### Rendering envs

PyTorch in `xm4d` for the EDG stages. Keep 2.9.1: the EDG completion reproduces our run only on this version.

```bash
conda activate xm4d
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

A second env, `xm4d-gs`, for the 4D Gaussian training and evaluation:

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

`scripts/setup_gsplat.sh` copies gsplat's `examples/` to `build/gsplat_examples/` and puts our two modified files from
`gsplat_patch/examples/` over it; the `deps/gsplat` submodule stays unmodified.

### Data

Download the preprocessed inputs into `datasets/` as described in [`datasets/README.md`](datasets/README.md).

## Run

Run everything from the repository root in the `xm4d` env. `GPU=1` selects another GPU.

### Camera accuracy

```bash
bash scripts/run_camera.sh all                  # all 65 scenes
bash scripts/run_camera.sh Sintel               # one dataset
bash scripts/run_camera.sh Sintel alley_1 cave_2  # single scenes
```

Per-scene outputs and `metrics.json` go to `results/camera/<Dataset>/<scene>/`, the dataset averages to
`results/camera/averages.txt`.

### Rendering

Rendering starts from the camera solve, so run the camera solve of the same scenes first:

```bash
bash scripts/run_camera.sh Syn4D && bash scripts/run_camera.sh NVIDIA && bash scripts/run_camera.sh Dycheck
bash scripts/run_render.sh all                  # all 22 scenes
bash scripts/run_render.sh Dycheck              # one dataset
bash scripts/run_render.sh Dycheck apple        # single scenes
```

The training and evaluation run with `conda run -n xm4d-gs` (`GS_ENV` changes the env name). Per-scene outputs,
`render.log` and `metrics.json` go to `results/render/<Dataset>/<scene>/`, the dataset averages to
`results/render/averages.txt`.

## Expected results

Camera accuracy, `run_camera.sh all` (one RTX 6000 Ada, about 40 min):

| dataset | scenes | ATE-T | RPE-T | RPE-R |
|---|---|---|---|---|
| Sintel | 23 | 0.017 | 0.026 | 0.040 |
| Replica | 8 | 0.002 | 0.002 | 0.001 |
| TUM | 6 | 0.006 | 0.008 | 0.006 |
| Syn4D | 8 | 0.002 | 0.002 | 0.003 |
| NVIDIA | 7 | 0.053 | 0.079 | 0.006 |
| DyCheck | 7 | 0.008 | 0.011 | 0.010 |
| IMC | 6 | 0.046 | 0.076 | 0.048 |

Novel-view rendering, `run_render.sh all` (one RTX 6000 Ada, about 9 h):

| dataset | scenes | PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| Syn4D | 8 | 21.74 | 0.824 | 0.148 |
| NVIDIA | 7 | 27.02 | 0.864 | 0.064 |
| DyCheck | 7 | 19.45 | 0.725 | 0.162 |

Per-scene numbers vary slightly between runs. The camera solves stop at wall-clock limits (`XM_MAX_TIME` per staircase
rank, `CERES_MAX_SOLVE_TIME`) and Ceres is multi-threaded; IMC varies most (ATE-T 0.044–0.051 over three runs) and on
a slower GPU needs a higher `XM_MAX_TIME` in `Python/configs/imc.json`. The 4D Gaussian training and test-time
alignment move per-scene PSNR by up to about 0.3 dB, because gsplat accumulates gradients with atomic adds.
