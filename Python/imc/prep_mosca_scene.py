#!/usr/bin/env python3
"""Prepare an IMC scene workspace for MoSca (uses the resized IMC_msam images):
 - write all-zero dynamic masks at 1024x768 (IMC is static -> no dynamic region),
   one per image, filename = image stem + .png, into IMC_msam/<scene>/masks
 - print a representative GT intrinsic at 1024x768 for the config
   (median GT focal from the original per-image PINHOLE cameras, rescaled).
"""
import argparse
import glob
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path("/home/fenchel/4DReconstruction")
W, H = 1024, 768


def load_imc_gt_cameras(scene):
    """Return {image_name: (w0,h0,fx,fy,cx,cy)} from sparse/{cameras,images}.txt."""
    sdir = REPO / "datasets" / "IMC" / scene / "sparse"
    cams = {}
    for line in (sdir / "cameras.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        cid = int(p[0]); w0 = int(p[2]); h0 = int(p[3])
        fx, fy, cx, cy = map(float, p[4:8])
        cams[cid] = (w0, h0, fx, fy, cx, cy)
    name2cam = {}
    lines = [l for l in (sdir / "images.txt").read_text().splitlines()
             if not l.startswith("#") and l.strip()]
    # images.txt: two lines per image; first line ends with CAMERA_ID NAME
    for i in range(0, len(lines), 2):
        p = lines[i].split()
        cid = int(p[8]); name = p[9]
        name2cam[name] = cams[cid]
    return name2cam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    args = ap.parse_args()

    ws = REPO / "datasets" / "IMC_msam" / args.scene
    imgs = sorted(glob.glob(str(ws / "images" / "*.jpg"))) + \
        sorted(glob.glob(str(ws / "images" / "*.png")))
    mask_dir = ws / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    zero = Image.fromarray(np.zeros((H, W), dtype=np.uint8), mode="L")
    for f in imgs:
        zero.save(mask_dir / (Path(f).stem + ".png"))
    print(f"{args.scene}: wrote {len(imgs)} all-zero {W}x{H} masks -> {mask_dir}")

    # representative intrinsic at 1024x768: rescale each image's GT K then take median
    name2cam = load_imc_gt_cameras(args.scene)
    fxs, fys = [], []
    for f in imgs:
        name = Path(f).name
        if name not in name2cam:
            continue
        w0, h0, fx, fy, cx, cy = name2cam[name]
        fxs.append(fx * W / w0)
        fys.append(fy * H / h0)
    fx_m, fy_m = float(np.median(fxs)), float(np.median(fys))
    (ws / "mosca_intrinsic.txt").write_text(f"{fx_m:.6f} {fy_m:.6f} {W/2:.1f} {H/2:.1f}\n")
    print(f"Representative GT intrinsic @ {W}x{H}: "
          f"gt_fx={fx_m:.2f} gt_fy={fy_m:.2f} gt_cx={W/2:.1f} gt_cy={H/2:.1f}  "
          f"(from {len(fxs)} images) -> {ws / 'mosca_intrinsic.txt'}")


if __name__ == "__main__":
    main()
