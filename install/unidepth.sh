#!/usr/bin/env bash
# Conda env for step 1 with UniDepth V2 instead of Depth Anything 3 (Python/get_depths_unidepth.py, config.json
# "DEPTH_MODEL": "unidepth"). UniDepth is imported from deps/UniDepth; its model (UniDepth-V2 ViT-L) is downloaded
# from Hugging Face on the first run.
# usage: bash install/unidepth.sh [env name]      (default env name: unidepth)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ENV="${1:-unidepth}"

clone https://github.com/lpiccinelli-eth/UniDepth.git "$ROOT/deps/UniDepth" 8d8cfe4  # imported, not installed
new_env "$ENV" python=3.11 -c conda-forge
pip install torch==2.4.1 torchvision==0.19.1 xformers==0.0.28.post1 --index-url https://download.pytorch.org/whl/cu124
pip install numpy==2.2.6 scipy==1.15.3 opencv-python==4.13.0.92 pillow==12.2.0 einops==0.8.2 timm==1.0.27 \
    huggingface_hub==1.18.0 safetensors==0.8.0 matplotlib==3.10.9 wandb==0.27.2   # wandb: imported by UniDepth

installed unidepth
