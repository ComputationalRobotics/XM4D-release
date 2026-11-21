"""Dynamic-mask loading shared by the stages. numpy / PIL / cv2 only, so that the XM environment (no torch) loads the masks
the same way as the tracking steps (utils.track_tapnet re-exports these)."""
import glob
import os
from pathlib import Path

import numpy as np
from PIL import Image


def _numeric_key(path: str):
    stem = Path(path).stem
    try:
        return int(stem)
    except ValueError:
        return stem


def load_dynamic_masks(
    mask_dir: str,
    pattern: str,
    t_expected: int,
    mask_mode: str = "auto",
    target_size: tuple | None = None,
) -> np.ndarray:
    """Load dynamic masks as bool array [T, H, W].

    If target_size is provided, masks are resized to that (width, height) using NEAREST.
    mask_mode: "dynamic_positive" (>0 means dynamic) or "auto" (as dynamic_positive, but inverted when >0 covers more
    than half of all pixels, i.e. static stored as white).
    """
    paths = glob.glob(os.path.join(mask_dir, pattern))
    if not paths:
        raise FileNotFoundError(f"No masks found in {mask_dir} matching {pattern}")
    paths = sorted(paths, key=_numeric_key)

    if len(paths) < t_expected:
        raise ValueError(
            f"Mask count mismatch: found {len(paths)} masks, but video has {t_expected} frames"
        )

    masks = []
    for p in paths[:t_expected]:
        im = Image.open(p).convert("L")
        if target_size is not None:
            # target_size: (width, height)
            im = im.resize((int(target_size[0]), int(target_size[1])), Image.NEAREST)
        arr = np.asarray(im)
        masks.append(arr > 0)
    masks_pos = np.stack(masks, axis=0)

    if mask_mode == "dynamic_positive":
        return masks_pos
    if mask_mode != "auto":
        raise ValueError(f"Unknown --mask_mode: {mask_mode}")

    pos_ratio = float(np.mean(masks_pos))
    if pos_ratio > 0.5:
        return ~masks_pos
    return masks_pos


def dilate_dynamic_masks(dynamic_masks: np.ndarray, dilate_px: int) -> np.ndarray:
    """Dilate (expand) or erode (shrink) dynamic masks by `dilate_px` pixels.

    Positive values dilate; negative values erode; zero returns input unchanged.
    """
    if dilate_px == 0:
        return dynamic_masks

    import cv2

    k = 2 * int(abs(dilate_px)) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    out = np.empty_like(dynamic_masks, dtype=bool)
    for t in range(dynamic_masks.shape[0]):
        m = (dynamic_masks[t].astype(np.uint8) * 255)
        if dilate_px > 0:
            out[t] = cv2.dilate(m, kernel, iterations=1) > 0
        else:
            out[t] = cv2.erode(m, kernel, iterations=1) > 0
    return out
