"""Step 0 of XM: a scene needs its moving-object masks before anything else runs.
<scene>/masks/ must hold one mask PNG per frame of <scene>/images/ (white = moving). make_masks_sam3.ipynb builds them with
SAM 3: one sub-folder per object, merged by merge_masks() into the masks at the top level that the pipeline reads.
get_depths_clean.py, the first step, calls require_masks(), which runs merge_masks() itself when the object folders are not
merged yet; for a scene with no moving object set "no_masks": true in config.json.
usage: python check_masks.py   (merges and checks the scene in config.json, as for every step)"""
import glob
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from utils import definitions as D

MARKER = ".combined_from_object_folders"   # in masks/: the top-level masks were written by merge_masks()


def _object_dirs():
    return sorted(p for p in D.mask_dir.iterdir() if p.is_dir()) if D.mask_dir.is_dir() else []


def _foreign_masks():   # top-level masks made some other way: merge_masks() never overwrites them
    return bool(glob.glob(str(D.mask_dir / D.mask_pattern))) and not (D.mask_dir / MARKER).exists()


def merge_masks():
    """masks/<frame>.png = union of every object folder masks/<object>/ (white = moving): the mask XM reads. Writes only
    the frames whose merged mask is missing or differs."""
    objs = _object_dirs()
    if not objs:
        raise RuntimeError(f"no object folders in {D.mask_dir}: save at least one object first")
    if _foreign_masks():
        raise RuntimeError(f"{D.mask_dir} already holds masks that merge_masks() did not make. Move them into a sub-folder "
                           "(e.g. masks/old/, then they count as one object) or delete them, and merge again.")
    frames = sorted(glob.glob(str(D.video_dir / D.image_pattern)))
    W, H = Image.open(frames[0]).size
    n_written = 0
    for f in frames:
        u = np.zeros((H, W), bool)
        for o in objs:
            p = o / f"{Path(f).stem}.png"
            if p.exists():
                u |= np.asarray(Image.open(p).convert("L")) > 0
        out = D.mask_dir / f"{Path(f).stem}.png"
        if not out.exists() or not np.array_equal(np.asarray(Image.open(out).convert("L")) > 0, u):
            Image.fromarray(u.astype(np.uint8) * 255).save(out)
            n_written += 1
    (D.mask_dir / MARKER).write_text("masks/*.png are the union of the object folders, written by make_masks_sam3.ipynb\n")
    print(f"[masks] merged {[o.name for o in objs]} into {D.mask_dir}: "
          + (f"{n_written} of {len(frames)} frames written" if n_written else "already up to date"))


def require_masks():
    if D.active_config.get("no_masks"):
        print("[masks] no_masks in config.json: running without moving-object masks")
        return
    n_img = len(glob.glob(str(D.video_dir / D.image_pattern)))
    if n_img and _object_dirs() and not _foreign_masks():
        merge_masks()   # object folders from make_masks_sam3.ipynb: merge them if that was skipped or is out of date
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
          "  text prompt per object), then rerun. For a scene without any moving object, set \"no_masks\": true in\n"
          "  config.json.\n", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    require_masks()
