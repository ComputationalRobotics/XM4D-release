#!/usr/bin/env bash
# Conda env for steps 3-5 (Python/static_solver.py, EDG_old.py, train_4dgs_edg.py): the XM solver.
# The solver ships compiled in XM/build (Python 3.10, CUDA 12.4); its CUDA libraries come from pip, so no system CUDA
# toolkit is needed. TEASER++ is built from source, which needs a C++ compiler.
# usage: bash install/XM.sh [env name]      (default env name: XM)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ENV="${1:-XM}"

clone https://github.com/MIT-SPARK/TEASER-plusplus.git "$ROOT/deps/TEASER-plusplus" 52a9c52
new_env "$ENV" python=3.10 eigen=3.4.0 cmake -c conda-forge
pip install numpy==2.2.6 scipy==1.15.3 opencv-python==4.13.0.92 open3d==0.19.0 networkx==3.4.2 pyceres==2.6 \
    pycolmap==3.13.0 pandas==2.3.3 matplotlib==3.10.9 tqdm==4.67.3 viser==1.0.24
pip install nvidia-cublas-cu12==12.4.5.8 nvidia-cusolver-cu12==11.6.1.9 nvidia-cusparse-cu12==12.3.1.170 \
    nvidia-nvjitlink-cu12==12.4.127

# put those CUDA libraries on LD_LIBRARY_PATH whenever the env is activated
SITE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
echo "export LD_LIBRARY_PATH=\"$SITE/nvidia/cublas/lib:$SITE/nvidia/cusolver/lib:$SITE/nvidia/cusparse/lib:$SITE/nvidia/nvjitlink/lib:\${LD_LIBRARY_PATH:-}\"" \
    > "$CONDA_PREFIX/etc/conda/activate.d/xm_cuda_libs.sh"
conda deactivate && conda activate "$ENV"

pip install "$ROOT/deps/TEASER-plusplus"

installed XM
