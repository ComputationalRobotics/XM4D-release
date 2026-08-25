
import glob
import numpy as np
import os

# DAVIS don't have GT, so need to use depth anyhting's intrinsic

def load_davis_gt(dataset_path):
    results = {}
    image_dir = os.path.join(dataset_path, "images")
    img_files = sorted(
        [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    )

    # DAVIS has no GT cameras; use DA3 intrinsics instead.
    intrinsics_candidates = [
        os.path.join(dataset_path, "outputs", "depth_anything_3", "intrinsics.npy"),
    ]
    intrinsics_path = next((p for p in intrinsics_candidates if os.path.isfile(p)), None)
    if intrinsics_path is None:
        raise FileNotFoundError(
            "Could not find intrinsics.npy. Tried: " + ", ".join(intrinsics_candidates)
        )

    intrinsics = np.load(intrinsics_path)
    if intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
        raise ValueError(f"Expected intrinsics shape (T,3,3), got {intrinsics.shape}")

    if intrinsics.shape[0] < len(img_files):
        raise ValueError(
            f"Not enough intrinsics for images: intrinsics T={intrinsics.shape[0]}, images={len(img_files)}"
        )

    for i, img_name in enumerate(img_files):
        K = intrinsics[i].astype(np.float32)
        results[img_name] = {
            "id": i,
            "K": K,
            "R": None,
            "t": None,
            "camera_id": i,
        }

    return results

def load_sintel_camera(dataset_path):
    results = {}
    results[1] = {
        "model": "PINHOLE",
        "width": 1200,
        "height": 680,
        "params": [600, 600, 599.5, 339.5]
    }
    return results

