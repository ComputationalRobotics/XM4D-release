import json
import os
from pathlib import Path

import numpy as np

# Ground-truth reader for the NVIDIA Dynamic Scenes dataset.
# Native GT lives in `poses_bounds.npy` (LLFF format, one 3x5 pose + 2 bounds
# per frame). This module converts those poses to the same convention used by
# the rest of the pipeline (OpenCV / COLMAP world-to-camera) and exposes the
# same interface as `readgt_sintel.load_sintel_gt`.
#
# LLFF stores a camera-to-world 3x5 matrix [R | t | hwf] with camera axes in
# the order [down, right, backwards]. We remap to OpenCV camera axes
# (x=right, y=down, z=forward) and assume a centered principal point.


def _llff_pose_to_opencv(pose: np.ndarray):
    """Convert one LLFF 3x5 pose to OpenCV camera parameters.

    Args:
        pose: (3, 5) array [R(3x3) | t(3x1) | hwf(3x1)].

    Returns:
        K     : (3, 3) intrinsics (centered principal point).
        R_wc  : (3, 3) world-to-camera rotation.
        t_wc  : (3,)   world-to-camera translation.
        R_cw  : (3, 3) camera-to-world rotation (OpenCV axes).
        center: (3,)   camera center in world coordinates.
    """
    pose = np.asarray(pose, dtype=np.float64)
    # camera-to-world rotation columns in OpenCV order:
    #   x_right = LLFF "right"    = pose[:, 1]
    #   y_down  = LLFF "down"     = pose[:, 0]
    #   z_fwd   = -LLFF "backward"= -pose[:, 2]
    R_cw = np.stack([pose[:, 1], pose[:, 0], -pose[:, 2]], axis=1)
    center = pose[:, 3].copy()

    R_wc = R_cw.T
    t_wc = -R_wc @ center

    h, w, f = pose[:, 4]
    K = np.array(
        [[f, 0.0, w / 2.0],
         [0.0, f, h / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return K, R_wc, t_wc, R_cw, center


def _poses_bounds_path(dataset_root: Path) -> Path:
    path = dataset_root / "poses_bounds.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Missing poses_bounds.npy in {dataset_root}")
    return path


def _image_names(dataset_root: Path, n_frames: int):
    """Return the frame image names, matching the Syn4D-style naming if present."""
    image_dir = dataset_root / "images"
    if image_dir.is_dir():
        names = sorted(
            f for f in os.listdir(image_dir)
            if Path(f).suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if len(names) == n_frames:
            return names
    # Fallback to canonical Syn4D-style names.
    return [f"frame{i:06d}.png" for i in range(n_frames)]


def load_nvidia_gt(dataset_path):
    """Load NVIDIA GT cameras, keyed by image name.

    Each value is {id, K (3x3), R (w2c 3x3), t (w2c 3,), camera_id}.
    """
    dataset_root = Path(dataset_path)
    poses_bounds = np.load(_poses_bounds_path(dataset_root))
    poses = poses_bounds[:, :15].reshape(-1, 3, 5)
    n_frames = poses.shape[0]
    img_names = _image_names(dataset_root, n_frames)

    results = {}
    for i in range(n_frames):
        K, R_wc, t_wc, _, _ = _llff_pose_to_opencv(poses[i])
        results[img_names[i]] = {
            "id": i,
            "K": K,
            "R": R_wc,
            "t": t_wc,
            "camera_id": i,
        }
    return results


def load_nvidia_camera(dataset_path):
    """Return PINHOLE intrinsics per camera (one entry per frame)."""
    dataset_root = Path(dataset_path)
    poses_bounds = np.load(_poses_bounds_path(dataset_root))
    poses = poses_bounds[:, :15].reshape(-1, 3, 5)

    results = {}
    for i in range(poses.shape[0]):
        h, w, f = poses[i][:, 4]
        results[i] = {
            "model": "PINHOLE",
            "width": int(round(w)),
            "height": int(round(h)),
            "params": [float(f), float(f), float(w / 2.0), float(h / 2.0)],
        }
    return results
