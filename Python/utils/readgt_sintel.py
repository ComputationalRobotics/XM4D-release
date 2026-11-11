import numpy as np
import os
from pathlib import Path

# This is a script only for Sintel dataset
# If you are loading your own dataset, please write you own script
# If you are not comparing with groud truth, R and t is not neccessary, but OTHER is needed

TAG_FLOAT = 202021.25
TAG_CHAR  = b'PIEH'


def depth_read(filename):
    """Read depth data from file, return as numpy array."""
    with open(filename, "rb") as f:
        check = np.fromfile(f, dtype=np.float32, count=1)[0]
        assert check == TAG_FLOAT, (
            " depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? ".format(
                TAG_FLOAT, check
            )
        )
        width = np.fromfile(f, dtype=np.int32, count=1)[0]
        height = np.fromfile(f, dtype=np.int32, count=1)[0]
        size = width * height
        assert width > 0 and height > 0 and size > 1 and size < 100000000, (
            " depth_read:: Wrong input size (width = {0}, height = {1}).".format(width, height)
        )
        depth = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))
        return depth

def cam_read(filename):
    """Read camera data, return (M, N).
    M: intrinsic 3x3
    N: extrinsic  3x4
    so that x = M * N * X
    """
    with open(filename, "rb") as f:
        check = np.fromfile(f, dtype=np.float32, count=1)[0]
        assert check == TAG_FLOAT, (
            f"cam_read:: Wrong tag. Expected {TAG_FLOAT}, got {check}. "
            "Big-endian machine?"
        )
        M = np.fromfile(f, dtype=np.float64, count=9).reshape((3, 3))
        N = np.fromfile(f, dtype=np.float64, count=12).reshape((3, 4))
    return M, N

def load_sintel_gt(dataset_path):
    results = {}
    dataset_root = Path(dataset_path)
    image_dir = dataset_root / "images"
    camera_dir = dataset_root / "cameras"
    fallback_training_root = Path(__file__).resolve().parents[2] / "datasets" / "MPI-Sintel-depth-training-20150305" / "training"
    fallback_camera_dir = fallback_training_root / "camdata_left" / dataset_root.name
    # Prefer the actual frame set used by the Sintel pipeline. Some local folders
    # contain extra JPG files alongside the canonical PNG frames.
    img_files_all = sorted(
        [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    )
    png_files = [f for f in img_files_all if Path(f).suffix.lower() == ".png"]
    if png_files:
        img_files = png_files
    else:
        img_files = [
            f for f in img_files_all if Path(f).suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]

    for i, img_name in enumerate(img_files):
        stem = os.path.splitext(img_name)[0]
        cam_path = None

        for candidate_dir in (camera_dir, fallback_camera_dir):
            candidate = candidate_dir / f"{stem}.cam"
            if candidate.is_file():
                cam_path = str(candidate)
                break

        if cam_path is None:
            raise FileNotFoundError(
                f"Missing Sintel camera for frame {img_name} in "
                f"{camera_dir} or fallback {fallback_camera_dir}"
            )

        K, E = cam_read(cam_path)     # K: 3x3, E: 3x4
        R = E[:, :3]                  # world->camera rotation (as stored)
        t = E[:, 3]                   # world->camera translation (as stored)

        results[img_name] = {
            "id": i,
            "K": K,
            "R": R,
            "t": t,
            "camera_id": i
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
