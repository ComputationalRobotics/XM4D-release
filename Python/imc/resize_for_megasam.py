#!/usr/bin/env python3
"""Resize IMC scene images to a uniform resolution so MegaSAM (a video method
that assumes a single fixed frame size) can run on them.

IMC scenes are unordered internet photos with ~165 distinct resolutions and
mixed portrait/landscape aspect ratios per scene, which breaks MegaSAM's
per-frame depth stacking and DROID buffer. We resize every frame to a common
1024x768 (median AR ~1.35; MegaSAM downsamples to 512x384 internally anyway).
Filenames are preserved so evaluated poses match the original GT by name.
"""
import argparse
import glob
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=768)
    args = ap.parse_args()

    src = REPO / "datasets" / "IMC" / args.scene / "images"
    dst = REPO / "datasets" / "IMC_msam" / args.scene / "images"
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(src / "*.jpg"))) + sorted(glob.glob(str(src / "*.png")))
    for f in files:
        img = Image.open(f).convert("RGB").resize((args.width, args.height), Image.LANCZOS)
        img.save(dst / Path(f).name, quality=95)
    print(f"{args.scene}: resized {len(files)} images -> {dst}")


if __name__ == "__main__":
    main()
