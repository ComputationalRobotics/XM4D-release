#!/usr/bin/env bash
# Alternative to the four per-stage envs: ONE conda env for every stage (masks, depth, tracking, XM solver).
# Python 3.10 and PyTorch 2.9 for all; the per-stage envs stay the reference. UniDepth V2 and GLOMAP keep their own
# envs (install/unidepth.sh, install/glomap.sh).
# usage: bash install/xm4d.sh [env name]      (default env name: xm4d)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ENV="${1:-xm4d}"

clone https://github.com/facebookresearch/sam3.git "$ROOT/deps/sam3" 32f5f81
clone https://github.com/ByteDance-Seed/Depth-Anything-3.git "$ROOT/deps/Depth-Anything-3" 2c21ea8
clone https://github.com/google-deepmind/tapnet.git "$ROOT/deps/tapnet" ceedd60
clone https://github.com/aharley/alltracker.git "$ROOT/deps/alltracker" e755313  # imported from deps/alltracker, not installed
AT_CKPT="$ROOT/deps/alltracker/checkpoints/alltracker.pth"
if [ ! -f "$AT_CKPT" ]; then
    mkdir -p "$(dirname "$AT_CKPT")"
    wget -O "$AT_CKPT.part" https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth
    mv "$AT_CKPT.part" "$AT_CKPT"
fi
CKPT="$ROOT/deps/tapnet/checkpoints/tapnextpp_ckpt.pt"
if [ ! -f "$CKPT" ]; then
    mkdir -p "$(dirname "$CKPT")"
    wget -O "$CKPT.part" https://storage.googleapis.com/dm-tapnet/tapnextpp/tapnextpp_ckpt.pt
    mv "$CKPT.part" "$CKPT"
fi

new_env "$ENV" python=3.10 eigen=3.4.0 cmake -c conda-forge
pip install torch==2.9.0 torchvision==0.24.0 xformers==0.0.33.post1    # xformers (Depth Anything 3) needs exactly 2.9.0
pip install -e "$ROOT/deps/sam3[notebooks]"
pip install -e "$ROOT/deps/Depth-Anything-3"
pip install numpy==1.26.4 opencv-python==4.11.0.86 open3d==0.19.0 scipy==1.15.3 pandas==2.3.3 einops==0.8.1 \
    pyceres==2.6 pycolmap==3.13.0 matplotlib==3.10.9 tqdm==4.67.3 setuptools==80.10.2

# the XM binding (CUDA 12.4 build) uses the CUDA 12 libraries that come with torch
SITE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
echo "export LD_LIBRARY_PATH=\"$SITE/nvidia/cublas/lib:$SITE/nvidia/cusolver/lib:$SITE/nvidia/cusparse/lib:$SITE/nvidia/nvjitlink/lib:\${LD_LIBRARY_PATH:-}\"" \
    > "$CONDA_PREFIX/etc/conda/activate.d/xm_cuda_libs.sh"
conda deactivate && conda activate "$ENV"

echo "[install] every stage is installed in env $ENV. Check it:  python install/check_env.py --env $ENV"
