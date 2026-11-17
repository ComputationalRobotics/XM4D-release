#!/usr/bin/env bash
# Conda env for step 6, training the 4D Gaussians (gsplat). gsplat's CUDA kernels are compiled here: this needs the CUDA
# toolkit (nvcc 12.x; set CUDA_HOME if it is not /usr/local/cuda). The trainer is gsplat's examples/ with our two changed
# files from gsplat_patch/examples/ copied over them, in build/gsplat_examples/ (deps/gsplat itself stays unmodified).
# usage: bash install/gs.sh [env name]      (default env name: gs)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ENV="${1:-gs}"

clone https://github.com/nerfstudio-project/gsplat.git "$ROOT/deps/gsplat" 48df619
new_env "$ENV" python=3.10
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip install numpy==1.26.4 scipy==1.15.3 opencv-python==4.11.0.86 "imageio[ffmpeg]==2.37.3" scikit-learn==1.7.2 \
    matplotlib==3.10.9 pandas==2.3.3 pyyaml==6.0.3 tqdm==4.67.3 "torchmetrics[image]==1.8.2" tensorboard==2.20.0 \
    tyro==1.0.13 viser==1.0.24 splines==0.3.3 piexif==1.1.3 tensorly==0.9.0 ninja \
    "nerfview @ git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787"
pip install --no-build-isolation git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5
pip install --no-build-isolation -e "$ROOT/deps/gsplat"

rm -rf "$ROOT/build/gsplat_examples"; mkdir -p "$ROOT/build"
cp -r "$ROOT/deps/gsplat/examples" "$ROOT/build/gsplat_examples"
cp -r "$ROOT/gsplat_patch/examples/." "$ROOT/build/gsplat_examples/"
echo "[install] trainer: $ROOT/build/gsplat_examples/simple_trainer.py"

installed gs
