#!/usr/bin/env bash
# Conda env for step 1, monocular depth (Python/get_depths_clean.py): Depth Anything 3.
# usage: bash install/da3.sh      (ENV=<name> bash install/da3.sh to use another env name)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ENV="${ENV:-da3}"

clone https://github.com/ByteDance-Seed/Depth-Anything-3.git "$ROOT/deps/Depth-Anything-3" 2c21ea8
new_env "$ENV" python=3.10
pip install torch==2.9.0 torchvision==0.24.0 xformers==0.0.33.post1
pip install -e "$ROOT/deps/Depth-Anything-3"
pip install open3d==0.19.0 scipy==1.15.3 pandas==2.3.3

installed da3
