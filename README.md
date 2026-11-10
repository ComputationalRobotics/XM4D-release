# XM4D — reproduction branch

This branch reproduces the **camera-accuracy results of XM4D ("Ours")** on the seven benchmarks of the paper —
Sintel, Replica, TUM, Syn4D, NVIDIA, DyCheck and IMC — starting from preprocessed inputs. The learned-model
preprocessing (depth estimation, 2D tracking) is already done and is shipped as data; this branch contains the XM
static solver that turns those inputs into camera poses, and the evaluation. The dynamic part (EDG solver and 4D
Gaussian training) will be added to this branch next.

## Contents

| path | what |
|---|---|
| `Python/static_solver.py` | XM static solver: camera poses + static points from 2D tracks and monocular depth |
| `Python/eval_xm.py` | pose evaluation (TEASER++ similarity alignment, ATE / RPE) |
| `Python/configs/<dataset>.json` | per-dataset parameters, selected automatically from the dataset name |
| `Python/utils/` | GT readers, problem construction, Ceres refinement |
| `XM/` | XM solver core: CUDA source (`src/`, `include/`, CMake) and our compiled Python binding `XM/build/XM.cpython-310-x86_64-linux-gnu.so` |
| `deps/TEASER-plusplus` | git submodule, [TEASER++](https://github.com/MIT-SPARK/TEASER-plusplus) at `52a9c52` |
| `scripts/run_camera.sh` | solve + evaluate any subset of the 65 scenes, then print the dataset averages |
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

## Run

```bash
bash scripts/run_camera.sh all          # 65 scenes, one GPU; prints the dataset averages
```

`bash scripts/run_camera.sh Sintel` runs one dataset and `bash scripts/run_camera.sh Sintel alley_1 cave_2` runs single
scenes. `GPU=1` selects another GPU. The runner never writes into `datasets/`: each scene is solved in
`results/camera/<Dataset>/<scene>/` (links to the inputs + the solver outputs `result_XM.npz`, logs) and its metrics
are written to `metrics.json` there. The dataset averages are also written to `results/camera/averages.txt`.

## Expected results

Dataset averages from `bash scripts/run_camera.sh all` (one RTX 6000 Ada, about 35 min in total):

| dataset | scenes | ATE-T | RPE-T | RPE-R |
|---|---|---|---|---|
| Sintel | 23 | 0.017 | 0.025 | 0.041 |
| Replica | 8 | 0.002 | 0.002 | 0.001 |
| TUM | 6 | 0.006 | 0.008 | 0.006 |
| Syn4D | 8 | 0.002 | 0.002 | 0.003 |
| NVIDIA | 7 | 0.038 | 0.056 | 0.005 |
| DyCheck | 7 | 0.008 | 0.010 | 0.010 |
| IMC | 6 | 0.042 | 0.067 | 0.042 |

