import os
from pathlib import Path

import numpy as np

from utils.readgt_colmap import frame_id_from_name, load_image_data, quat2rot


def _image_names(dataset_path):
    image_dir = Path(dataset_path) / "images"
    return sorted(
        path.name
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def _load_da3_intrinsics(dataset_path, n_images):
    intrinsics_path = Path(dataset_path) / "outputs" / "depth_anything_3" / "intrinsics.npy"
    if not intrinsics_path.is_file():
        raise FileNotFoundError(
            "Could not find Depth Anything intrinsics. Tried: "
            + os.fspath(intrinsics_path)
        )

    intrinsics = np.load(intrinsics_path)
    if intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
        raise ValueError(f"Expected intrinsics shape (T,3,3), got {intrinsics.shape}")
    if intrinsics.shape[0] < n_images:
        raise ValueError(
            f"Not enough intrinsics for images: intrinsics T={intrinsics.shape[0]}, images={n_images}"
        )
    return intrinsics


def load_lab_mobileplatform_gt(dataset_path):
    """Load Lab-MobilePlatform poses from COLMAP text and K from Depth Anything.

    Expected layout:
      dataset_path/images
      dataset_path/colmap/sparse/images.txt
      dataset_path/outputs/depth_anything_3/intrinsics.npy
    """
    dataset_root = Path(dataset_path)
    image_names = _image_names(dataset_root)
    intrinsics = _load_da3_intrinsics(dataset_root, len(image_names))
    image_name_to_index = {name: i for i, name in enumerate(image_names)}

    image_df = load_image_data(os.fspath(dataset_root / "colmap" / "sparse" / "images.txt"))
    results = {}
    for name, row in image_df.iterrows():
        if name not in image_name_to_index:
            continue

        pose = row["POSE"]
        R = quat2rot(pose["qw"], pose["qx"], pose["qy"], pose["qz"])
        t = np.array([pose["tx"], pose["ty"], pose["tz"]])
        i = image_name_to_index[name]

        results[name] = {
            "id": frame_id_from_name(name),
            "K": intrinsics[i].astype(np.float32),
            "R": R,
            "t": t,
            "camera_id": row["CAMERA_ID"],
        }

    return results


def load_lab_mobileplatform_camera(dataset_path):
    image_names = _image_names(dataset_path)
    intrinsics = _load_da3_intrinsics(dataset_path, len(image_names))
    K = intrinsics[0]

    width = 0
    height = 0
    cameras_txt = Path(dataset_path) / "colmap" / "sparse" / "cameras.txt"
    if cameras_txt.is_file():
        with cameras_txt.open("r") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                width = int(parts[2])
                height = int(parts[3])
                break

    return {
        1: {
            "model": "PINHOLE",
            "width": width,
            "height": height,
            "params": [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])],
        }
    }
