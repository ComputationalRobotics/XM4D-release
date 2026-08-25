from pathlib import Path
import re

import cv2
import numpy as np


def frame_id_from_name(name: str) -> int:
    stem = Path(name).stem
    match = re.search(r"(?<!\d)(\d+)$", stem)
    if match is None:
        raise ValueError(f"Could not extract frame id from image name: {name}")
    return int(match.group(1))


def _fallback_intrinsics(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image for fallback intrinsics: {image_path}")
    height, width = image.shape[:2]
    focal = float(max(width, height))
    return np.array(
        [
            [focal, 0.0, (width - 1) * 0.5],
            [0.0, focal, (height - 1) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def load_custom_gt(dataset_path: str | Path):
    """
    Pseudo-GT loader for Custom datasets.

    Custom data has no true depth, intrinsics, or camera poses. This loader only
    exposes a per-frame K so existing tracking/lifting code can run after DA3.
    R/t are intentionally None to prevent true-GT pose evaluation.
    """
    root = Path(dataset_path)
    image_dir = root / "images"
    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No Custom images found in {image_dir}")

    intrinsics_path = root / "outputs" / "depth_anything_3" / "intrinsics.npy"
    intrinsics = None
    if intrinsics_path.is_file():
        intrinsics = np.load(intrinsics_path).astype(np.float64)
        if intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
            raise ValueError(f"Expected {intrinsics_path} shape (T,3,3), got {intrinsics.shape}")

    results = {}
    for idx, image_path in enumerate(image_paths):
        if intrinsics is not None:
            if idx >= intrinsics.shape[0]:
                raise ValueError(f"Only {intrinsics.shape[0]} intrinsics for {len(image_paths)} Custom images.")
            K = intrinsics[idx]
        else:
            K = _fallback_intrinsics(image_path)
        try:
            frame_id = frame_id_from_name(image_path.name)
        except ValueError:
            frame_id = idx
        results[image_path.name] = {
            "id": frame_id,
            "K": K,
            "R": None,
            "t": None,
            "camera_id": 1,
        }
    return results
