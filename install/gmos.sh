#!/usr/bin/env bash
# Conda env for step 0 without a notebook (Python/make_masks_gmos.py): GMOS finds the moving objects automatically.
# usage: bash install/gmos.sh      (ENV=<name> bash install/gmos.sh to use another env name)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ENV="${ENV:-gmos}"

clone https://github.com/Jyxarthur/gmos.git "$ROOT/deps/gmos" 2411284
CK="$ROOT/deps/gmos/checkpoints"
get() {  # get <url> <file>: download into deps/gmos/checkpoints/ once
    [ -f "$CK/$2" ] && return
    mkdir -p "$CK"
    wget -O "$CK/$2.part" "$1"
    mv "$CK/$2.part" "$CK/$2"
}
get https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors pi3_model.safetensors                     # 3.8 GB
get https://huggingface.co/facebook/sam2.1-hiera-large/resolve/main/sam2.1_hiera_large.pt sam2.1_hiera_large.pt  # 0.9 GB
new_env "$ENV" python=3.11
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install numpy==1.26.4 einops==0.8.2 pillow==13.0.0 mediapy==1.2.7 opencv-python==4.11.0.86 safetensors==0.8.0 \
    hydra-core==1.3.8 omegaconf==2.3.1 iopath==0.1.10 huggingface_hub==2.4.0 scipy==1.17.1 pandas==3.0.6 tqdm==4.70.1 \
    gdown==6.5.1
if [ ! -f "$CK/gmos_proposer.pth" ]; then   # the GMOS proposer (0.1 GB), shared on Google Drive by the authors
    gdown 1OUsz6VJBD6r-VrrV-Q1sonb_XwthrbIk -O "$CK/gmos_proposer.pth.part"
    mv "$CK/gmos_proposer.pth.part" "$CK/gmos_proposer.pth"
fi

installed gmos
