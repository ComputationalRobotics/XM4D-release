import os
from pathlib import Path

import cv2
import numpy as np

# This is a script only for Replica dataset
# If you are loading your own dataset, please write you own script
# If you are not comparing with groud truth, R and t is not neccessary, but OTHER is needed

REPLICA_K = np.array([[600, 0, 599.5], [0, 600, 339.5], [0, 0, 1]], dtype=np.float64)
REPLICA_DEPTH_SCALE = float(os.environ.get("XM_REPLICA_DEPTH_SCALE", "6553.5"))


def _replica_image_dir(dataset_path: str | os.PathLike[str]) -> Path | None:
    dataset_root = Path(dataset_path)
    for name in ("images", "image"):
        candidate = dataset_root / name
        if candidate.is_dir():
            return candidate
    return None


def _replica_image_names(dataset_path: str | os.PathLike[str], n_poses: int) -> list[str]:
    image_dir = _replica_image_dir(dataset_path)
    if image_dir is None:
        return [f"frame{i:06d}.jpg" for i in range(n_poses)]

    image_files = sorted(
        [
            path.name
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )
    if len(image_files) < n_poses:
        raise ValueError(
            f"Replica image count mismatch in {image_dir}: "
            f"{len(image_files)} images for {n_poses} poses"
        )
    return image_files[:n_poses]


def read_replica_depth(filename: str | os.PathLike[str]) -> np.ndarray:
    depth_raw = cv2.imread(str(filename), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"Could not read Replica depth file: {filename}")
    if depth_raw.ndim != 2:
        raise ValueError(
            f"Expected single-channel Replica depth PNG, got shape {depth_raw.shape} for {filename}"
        )
    return depth_raw.astype(np.float64) / REPLICA_DEPTH_SCALE

def load_replica_gt(dataset_path):
    results = {}
    data = np.loadtxt(Path(dataset_path) / "traj.txt")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    n_poses = data.shape[0]
    image_names = _replica_image_names(dataset_path, n_poses)
    for i in range(n_poses):
        name = image_names[i]
        pose = data[i, :].reshape((4, 4))
        # transform to world-2-camera
        R = pose[:3, :3].T
        t = - pose[:3, :3].T @ pose[:3, 3]
        results[name] = {
            "id": i,
            "K": REPLICA_K.copy(),
            "R": R,
            "t": t,
            "camera_id": 1
        }
    return results

def load_replica_camera(dataset_path):
    width = 1200
    height = 680
    image_dir = _replica_image_dir(dataset_path)
    if image_dir is not None:
        first_image = next(
            (
                path
                for path in sorted(image_dir.iterdir())
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ),
            None,
        )
        if first_image is not None:
            image = cv2.imread(str(first_image), cv2.IMREAD_UNCHANGED)
            if image is not None:
                height, width = image.shape[:2]
    results = {}
    results[1] = {
        "model": "PINHOLE",
        "width": width,
        "height": height,
        "params": [600, 600, 599.5, 339.5],
    }
    return results
