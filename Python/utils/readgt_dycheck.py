import json
import re
from pathlib import Path

import cv2
import numpy as np

# Loader for the DyCheck iPhone dataset (Nerfies camera format).
# Same return contract as readgt_colmap.load_colmap_gt: a dict keyed by image
# name -> {"id", "K", "R", "t", "camera_id"} with R, t in world-to-camera
# (OpenCV) convention.
#
# DyCheck cameras store:
#   orientation  = R (world -> camera rotation, OpenCV axes: x right, y down, z fwd)
#   position     = C (camera center in world)         -> t = -R @ C
#   focal_length / principal_point are given at the json's image_size (full res);
#   the exported images are downscaled, so K is rescaled by actual_w / json_w.


def frame_id_from_name(name):
    stem = Path(name).stem
    match = re.search(r"(\d+)$", stem)
    if match is None:
        raise ValueError(f"Could not extract frame id from image name: {name}")
    return int(match.group(1))


def _K_from_camera(cam, scale):
    focal = cam["focal_length"]
    par = cam.get("pixel_aspect_ratio", 1.0)
    cx, cy = cam["principal_point"]
    skew = cam.get("skew", 0.0)
    return np.array(
        [
            [focal * scale, skew * scale, cx * scale],
            [0.0, focal * par * scale, cy * scale],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _extrinsics_from_camera(cam):
    R = np.array(cam["orientation"], dtype=np.float64)  # world -> camera
    C = np.array(cam["position"], dtype=np.float64)      # camera center in world
    t = -R @ C
    return R, t


def _image_files(image_dir):
    return sorted(
        path
        for path in Path(image_dir).iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


def _scale_from_first(cam, image_path):
    """DyCheck intrinsics are at cam['image_size'] (=[W, H]); exported images may
    be downscaled uniformly, so recover the factor from the actual image width."""
    json_w = float(cam["image_size"][0])
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image for scale: {image_path}")
    return image.shape[1] / json_w


def _load_split(image_dir, camera_dir):
    image_dir = Path(image_dir)
    camera_dir = Path(camera_dir)
    images = _image_files(image_dir)
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")

    first_cam = json.loads((camera_dir / f"{images[0].stem}.json").read_text())
    scale = _scale_from_first(first_cam, images[0])

    results = {}
    for image_path in images:
        cam = json.loads((camera_dir / f"{image_path.stem}.json").read_text())
        R, t = _extrinsics_from_camera(cam)
        results[image_path.name] = {
            "id": frame_id_from_name(image_path.name),
            "K": _K_from_camera(cam, scale),
            "R": R,
            "t": t,
            "camera_id": 1,
        }
    return results


def load_dycheck_gt(dataset_path):
    root = Path(dataset_path)
    return _load_split(root / "images", root / "cameras")


def load_dycheck_novel_gt(dataset_path):
    root = Path(dataset_path)
    return _load_split(root / "novel" / "images", root / "novel" / "cameras")


def load_dycheck_camera(dataset_path):
    root = Path(dataset_path)
    images = _image_files(root / "images")
    cam = json.loads((root / "cameras" / f"{images[0].stem}.json").read_text())
    scale = _scale_from_first(cam, images[0])
    image = cv2.imread(str(images[0]), cv2.IMREAD_UNCHANGED)
    height, width = image.shape[:2]
    K = _K_from_camera(cam, scale)
    return {
        1: {
            "model": "PINHOLE",
            "width": width,
            "height": height,
            "params": [K[0, 0], K[1, 1], K[0, 2], K[1, 2]],
        }
    }
