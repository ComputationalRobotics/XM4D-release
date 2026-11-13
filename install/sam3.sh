#!/usr/bin/env bash
# Conda env for step 0, the moving-object masks (Python/make_masks_sam3.ipynb): SAM 3 + Jupyter.
# usage: bash install/sam3.sh      (ENV=<name> bash install/sam3.sh to use another env name)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ENV="${ENV:-sam3}"

clone https://github.com/facebookresearch/sam3.git "$ROOT/deps/sam3" 32f5f81
new_env "$ENV" python=3.12
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e "$ROOT/deps/sam3[notebooks]"
pip install setuptools==80.10.2   # SAM 3 imports pkg_resources, which newer setuptools no longer ships

installed sam3
