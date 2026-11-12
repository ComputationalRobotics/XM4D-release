"""Step 0 of XM: a scene needs its moving-object masks before anything else runs.
<scene>/masks/ must hold one mask PNG per frame of <scene>/images/ (white = moving). make_masks_sam3.ipynb builds them with
SAM 3 (one sub-folder per object, plus the combined masks at the top level that the pipeline reads).
get_depths_clean.py, the first step, calls require_masks(); for a scene with no moving object set XM_NO_MASKS=1.
usage: python check_masks.py   (the scene is selected by XM_DATASET_CAT / XM_DATASET_NAME or XM_DATASET_ROOT, as for every step)"""
import glob
import os
import sys

from utils import definitions as D


def require_masks():
    if os.environ.get("XM_NO_MASKS") == "1":
        print("[masks] XM_NO_MASKS=1: running without moving-object masks")
        return
    n_img = len(glob.glob(str(D.video_dir / D.image_pattern)))
    n_mask = len(glob.glob(str(D.mask_dir / D.mask_pattern)))
    if n_img and n_mask >= n_img:
        print(f"[masks] {n_mask} masks in {D.mask_dir}")
        return
    if not n_img:
        problem = f"no frames matching {D.image_pattern} in {D.video_dir}"
    elif not n_mask:
        problem = f"no masks in {D.mask_dir}"
    else:
        problem = f"only {n_mask} masks for {n_img} frames in {D.mask_dir}"
    print(f"\n[masks] WARNING: {problem}.\n"
          "  XM needs the moving-object masks first: make them with Python/make_masks_sam3.ipynb (SAM 3, a few clicks or a\n"
          "  text prompt per object), then rerun. For a scene without any moving object, set XM_NO_MASKS=1.\n", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    require_masks()
