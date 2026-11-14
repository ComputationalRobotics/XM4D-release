#!/usr/bin/env bash
# Conda env for step 2, 2D tracking: TAPNext++ for the static scene (Python/static_tracking_online.py) and AllTracker
# for the moving objects (Python/dynamic_tracking.py); config STATIC_TRACKER / DYNAMIC_TRACKER.
# usage: bash install/tapnet.sh [env name]      (default env name: tapnet)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ENV="${1:-tapnet}"

clone https://github.com/google-deepmind/tapnet.git "$ROOT/deps/tapnet" ceedd60  # imported from deps/tapnet, not installed
CKPT="$ROOT/deps/tapnet/checkpoints/tapnextpp_ckpt.pt"
if [ ! -f "$CKPT" ]; then
    mkdir -p "$(dirname "$CKPT")"
    wget -O "$CKPT.part" https://storage.googleapis.com/dm-tapnet/tapnextpp/tapnextpp_ckpt.pt
    mv "$CKPT.part" "$CKPT"
fi
clone https://github.com/aharley/alltracker.git "$ROOT/deps/alltracker" e755313  # imported from deps/alltracker, not installed
AT_CKPT="$ROOT/deps/alltracker/checkpoints/alltracker.pth"
if [ ! -f "$AT_CKPT" ]; then
    mkdir -p "$(dirname "$AT_CKPT")"
    wget -O "$AT_CKPT.part" https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth
    mv "$AT_CKPT.part" "$AT_CKPT"
fi
new_env "$ENV" python=3.10
pip install torch==2.9.1 torchvision==0.24.1 einops==0.8.1 numpy==2.0.2 opencv-python==4.12.0.88 pandas==2.3.3 \
    pillow tqdm

installed tapnet
