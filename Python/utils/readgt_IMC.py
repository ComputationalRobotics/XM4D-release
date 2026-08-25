from pathlib import Path

import numpy as np

from utils.readgt_colmap import load_camera_data, load_image_data, quat2rot


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _image_names(dataset_path):
    image_dir = Path(dataset_path) / "images"
    return sorted(
        path.name
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def load_imc_gt(dataset_path):
    """Load IMC subset camera intrinsics and COLMAP world-to-camera poses.

    Expected layout:
      dataset_path/images
      dataset_path/sparse/cameras.txt
      dataset_path/sparse/images.txt

    Returns the same dictionary shape used by the other readgt_* loaders.
    """
    root = Path(dataset_path)
    camera_df = load_camera_data(str(root / "sparse" / "cameras.txt"))
    image_df = load_image_data(str(root / "sparse" / "images.txt"))
    image_names = _image_names(root)

    results = {}
    for idx, name in enumerate(image_names):
        if name not in image_df.index:
            continue
        row = image_df.loc[name]
        pose = row["POSE"]
        camera_id = row["CAMERA_ID"]
        K = camera_df.loc[camera_id, "K"].astype(np.float64)
        R = quat2rot(pose["qw"], pose["qx"], pose["qy"], pose["qz"])
        t = np.array([pose["tx"], pose["ty"], pose["tz"]], dtype=np.float64)
        results[name] = {
            "id": idx,
            "K": K,
            "R": R,
            "t": t,
            "camera_id": int(camera_id),
        }
    return results


def load_imc_camera(dataset_path):
    root = Path(dataset_path)
    camera_df = load_camera_data(str(root / "sparse" / "cameras.txt"))
    results = {}
    for camera_id, row in camera_df.iterrows():
        K = np.asarray(row["K"], dtype=np.float64)
        results[int(camera_id)] = {
            "model": "PINHOLE",
            "width": int(row["width"]),
            "height": int(row["height"]),
            "params": [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])],
        }
    return results
