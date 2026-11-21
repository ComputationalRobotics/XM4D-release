"""Step 0 of XM: a scene needs its moving-object masks before anything else runs.
<scene>/masks/ must hold one mask PNG per frame of <scene>/images/ (white = moving). make_masks_sam3.ipynb builds them with
SAM 3: save_object() writes one sub-folder per object, masks/<category>_<name>/, and merge_masks() merges them into the
masks at the top level that the pipeline reads, plus <scene>/mask_labels/ (which object each pixel belongs to).
get_depths_clean.py, the first step, calls require_masks(), which runs merge_masks() itself when the object folders are not
merged yet; for a scene with no moving object set "no_masks": true in config.json.
usage: python check_masks.py   (merges and checks the scene in config.json, as for every step)"""
import glob
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from utils import definitions as D

MARKER = ".combined_from_object_folders"   # in masks/: the top-level masks were written by merge_masks()
ORDER = "objects.json"                     # in masks/: the object folders in the order they were added
LABEL_DIR = D.rootpath / "mask_labels"     # <frame>.png: 0 = no object, k = object k of labels.json
# Object categories in merge priority: a pixel covered by several objects goes to the first category here, within a
# category to the object added first. A folder without a category prefix (masks/<name>/) counts as "other".
CATEGORIES = ("rigid", "hand", "human", "other")


def category_of(folder):
    c = folder.split("_", 1)[0]
    return c if "_" in folder and c in CATEGORIES else "other"


def _order():
    p = D.mask_dir / ORDER
    return json.loads(p.read_text()) if p.exists() else []


def _object_dirs():
    return sorted(p for p in D.mask_dir.iterdir() if p.is_dir()) if D.mask_dir.is_dir() else []


def _foreign_masks():   # top-level masks made some other way: merge_masks() never overwrites them
    return bool(glob.glob(str(D.mask_dir / D.mask_pattern))) and not (D.mask_dir / MARKER).exists()


def save_object(name, category, frames, masks):
    """Write one object to masks/<category>_<name>/: masks (T, H, W) bool, one per frame path in frames. category is
    "rigid" (moves as one rigid body), "human" (a person, hands included) or "other" (anything else that moves).
    Saving the same name again replaces its masks and keeps its place in the adding order."""
    if category == "hand":
        raise NotImplementedError("hand masks are not supported: save hands as part of 'human'")
    if category not in CATEGORIES:
        raise ValueError(f"category {category!r}: use 'rigid', 'human' or 'other'")
    other = [f"{c}_{name}" for c in CATEGORIES if c != category and (D.mask_dir / f"{c}_{name}").is_dir()]
    if other:
        raise ValueError(f"{name} is saved as {other[0]} already: remove_object({name!r}) first")
    d = D.mask_dir / f"{category}_{name}"
    d.mkdir(parents=True, exist_ok=True)
    for f, m in zip(frames, masks):
        Image.fromarray(np.asarray(m, bool).astype(np.uint8) * 255).save(d / f"{Path(f).stem}.png")
    order = _order()
    if d.name not in order:
        (D.mask_dir / ORDER).write_text(json.dumps(order + [d.name], indent=1) + "\n")
    print(f"saved {name} ({category}) to {d}")


def remove_object(name):
    """Delete a saved object (masks/<category>_<name>/) and its place in the adding order."""
    for c in CATEGORIES:
        if (D.mask_dir / f"{c}_{name}").is_dir():
            shutil.rmtree(D.mask_dir / f"{c}_{name}")
    (D.mask_dir / ORDER).write_text(json.dumps([f for f in _order() if (D.mask_dir / f).is_dir()], indent=1) + "\n")


def merge_masks():
    """masks/<frame>.png = union of every object folder (white = moving): the mask XM reads. mask_labels/<frame>.png =
    which object each pixel belongs to (0 = none, k = object k of mask_labels/labels.json): where objects overlap, the
    pixel goes by CATEGORIES (rigid, hand, human, other), then to the object added first. Writes only what changed."""
    order = _order()
    objs = sorted(_object_dirs(), key=lambda p: (CATEGORIES.index(category_of(p.name)),
                                                 order.index(p.name) if p.name in order else len(order), p.name))
    if not objs:
        raise RuntimeError(f"no object folders in {D.mask_dir}: save at least one object first")
    if _foreign_masks():
        raise RuntimeError(f"{D.mask_dir} already holds masks that merge_masks() did not make. Move them into a sub-folder "
                           "(e.g. masks/old/, then they count as one object) or delete them, and merge again.")
    LABEL_DIR.mkdir(exist_ok=True)
    labels = json.dumps([dict(label=k, folder=o.name, category=category_of(o.name)) for k, o in enumerate(objs, 1)],
                        indent=1) + "\n"
    if not (LABEL_DIR / "labels.json").exists() or (LABEL_DIR / "labels.json").read_text() != labels:
        (LABEL_DIR / "labels.json").write_text(labels)
    frames = sorted(glob.glob(str(D.video_dir / D.image_pattern)))
    W, H = Image.open(frames[0]).size
    n_written = 0
    for f in frames:
        lab = np.zeros((H, W), np.uint8)
        for k, o in enumerate(objs, 1):
            p = o / f"{Path(f).stem}.png"
            if p.exists():
                lab[(np.asarray(Image.open(p).convert("L")) > 0) & (lab == 0)] = k
        out, out_lab = D.mask_dir / f"{Path(f).stem}.png", LABEL_DIR / f"{Path(f).stem}.png"
        new = not out.exists() or not np.array_equal(np.asarray(Image.open(out).convert("L")) > 0, lab > 0)
        if new:
            Image.fromarray((lab > 0).astype(np.uint8) * 255).save(out)
        if not out_lab.exists() or not np.array_equal(np.asarray(Image.open(out_lab)), lab):
            Image.fromarray(lab).save(out_lab)
            new = True
        n_written += new
    (D.mask_dir / MARKER).write_text("masks/*.png are the union of the object folders, written by make_masks_sam3.ipynb\n")
    print(f"[masks] merged {[o.name for o in objs]} (in priority order) into {D.mask_dir} and {LABEL_DIR}: "
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
