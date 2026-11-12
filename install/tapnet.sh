#!/usr/bin/env bash
# Conda env for step 2, 2D tracking (Python/static_tracking_online.py, dynamic_tracking.py): TAPNext++.
# usage: bash install/tapnet.sh      (ENV=<name> bash install/tapnet.sh to use another env name)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ENV="${ENV:-tapnet}"

clone https://github.com/google-deepmind/tapnet.git "$ROOT/deps/tapnet" ceedd60  # imported from deps/tapnet, not installed
CKPT="$ROOT/deps/tapnet/checkpoints/tapnextpp_ckpt.pt"
if [ ! -f "$CKPT" ]; then
    mkdir -p "$(dirname "$CKPT")"
    wget -O "$CKPT.part" https://storage.googleapis.com/dm-tapnet/tapnextpp/tapnextpp_ckpt.pt
    mv "$CKPT.part" "$CKPT"
fi
new_env "$ENV" python=3.10
pip install torch==2.9.1 torchvision==0.24.1 einops==0.8.1 numpy==2.0.2 opencv-python==4.12.0.88 pandas==2.3.3 \
    pillow tqdm

python "$ROOT/install/check_env.py" tapnet
