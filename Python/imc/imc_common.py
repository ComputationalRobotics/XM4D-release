"""Shared helpers for the IMC XM-SfM pipeline.

Parses the COLMAP-format ground-truth sparse model shipped with each IMC scene
(sparse/cameras.txt + sparse/images.txt) so downstream stages can obtain the
per-image PINHOLE intrinsics keyed by image file name.

Frame ordering convention (must match Python/static_solver.py and
Python/utils/readgt_IMC.py): frames are indexed by the *sorted* image file
names in <scene>/images.
"""
from pathlib import Path

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def sorted_image_names(scene_dir):
    image_dir = Path(scene_dir) / "images"
    return sorted(
        p.name for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def name_to_frame_index(scene_dir):
    """Map image file name -> frame index (position in the sorted name list)."""
    return {name: idx for idx, name in enumerate(sorted_image_names(scene_dir))}


def _parse_cameras(cameras_txt):
    """CAMERA_ID -> dict with PINHOLE intrinsics K, width, height, params."""
    cameras = {}
    for line in Path(cameras_txt).read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cam_id = int(parts[0])
        model = parts[1]
        if model != "PINHOLE":
            raise ValueError(f"Expected PINHOLE camera model, got {model}")
        width, height = int(parts[2]), int(parts[3])
        fx, fy, cx, cy = (float(x) for x in parts[4:8])
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        cameras[cam_id] = {
            "K": K, "width": width, "height": height,
            "params": (fx, fy, cx, cy),
        }
    return cameras


def _image_name_to_camera_id(images_txt):
    lines = [
        l for l in Path(images_txt).read_text().splitlines()
        if l and not l.startswith("#")
    ]
    mapping = {}
    for i in range(0, len(lines), 2):  # header then POINTS2D line
        parts = lines[i].split()
        mapping[parts[9]] = int(parts[8])
    return mapping


def load_intrinsics_by_name(scene_dir):
    """image file name -> 3x3 intrinsics K for every registered image."""
    root = Path(scene_dir)
    cameras = _parse_cameras(root / "sparse" / "cameras.txt")
    name2cam = _image_name_to_camera_id(root / "sparse" / "images.txt")
    return {name: cameras[cid]["K"] for name, cid in name2cam.items()}


INTRINSIC_SOURCES = ("gt", "glomap", "unidepth")


def load_intrinsics(scene_dir, source):
    """Unified per-image intrinsics loader for the pipeline switch.

    Returns {image_name: {'K': 3x3 np.ndarray, 'width', 'height'}}.
      gt       -> ground truth from sparse/cameras.txt
      glomap   -> GLOMAP-estimated, recorded by stage1
      unidepth -> UniDepth-estimated, recorded by stage2
    """
    source = source.lower()
    if source == "glomap":
        return load_glomap_intrinsics(scene_dir)
    if source == "gt":
        pin = load_pinhole_by_name(scene_dir)
        out = {}
        for name, info in pin.items():
            fx, fy, cx, cy = info["params"]
            out[name] = {
                "K": np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64),
                "width": info["width"],
                "height": info["height"],
            }
        return out
    if source == "unidepth":
        import json

        path = Path(scene_dir) / "outputs" / "unidepth_intrinsics.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found; run stage2 with --intrinsics unidepth first."
            )
        raw = json.loads(path.read_text())
        return {
            name: {
                "K": np.asarray(info["K"], dtype=np.float64),
                "width": int(info["width"]),
                "height": int(info["height"]),
            }
            for name, info in raw.items()
        }
    raise ValueError(f"Unknown intrinsic source {source!r}; use one of {INTRINSIC_SOURCES}")


def load_glomap_intrinsics(scene_dir):
    """image file name -> {'K': 3x3 np.ndarray, 'width', 'height'} as ESTIMATED
    by GLOMAP and recorded by stage1 to outputs/glomap_work/glomap_intrinsics.json.
    Only images GLOMAP registered are present.
    """
    import json

    path = Path(scene_dir) / "outputs" / "glomap_work" / "glomap_intrinsics.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run stage1 first (it records GLOMAP's intrinsics)."
        )
    raw = json.loads(path.read_text())
    return {
        name: {
            "K": np.asarray(info["K"], dtype=np.float64),
            "width": int(info["width"]),
            "height": int(info["height"]),
        }
        for name, info in raw.items()
    }


def load_pinhole_by_name(scene_dir):
    """image file name -> dict(params=(fx,fy,cx,cy), width, height)."""
    root = Path(scene_dir)
    # DyCheck: Nerfies per-frame camera json in cameras/, no COLMAP sparse. Intrinsics
    # are constant per scene; readgt_dycheck rescales the full-res K to image res.
    if not (root / "sparse" / "cameras.txt").exists() and (root / "cameras").is_dir():
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from utils.readgt_dycheck import load_dycheck_gt, load_dycheck_camera
        gt = load_dycheck_gt(str(root))
        cam = load_dycheck_camera(str(root))[1]
        w, h = int(cam["width"]), int(cam["height"])
        return {
            name: {
                "params": (float(info["K"][0, 0]), float(info["K"][1, 1]),
                           float(info["K"][0, 2]), float(info["K"][1, 2])),
                "width": w,
                "height": h,
            }
            for name, info in gt.items()
        }
    # NVIDIA has no COLMAP sparse/; GT intrinsics come from poses_bounds.npy (LLFF
    # hwf), constant per scene, centered principal point (matches readgt_nvidia).
    if not (root / "sparse" / "cameras.txt").exists() and (root / "poses_bounds.npy").exists():
        hwf = np.load(root / "poses_bounds.npy")[:, :15].reshape(-1, 3, 5)[:, :, 4]
        names = sorted_image_names(root)
        return {
            name: {
                "params": (float(hwf[i][2]), float(hwf[i][2]), float(hwf[i][1] / 2.0), float(hwf[i][0] / 2.0)),
                "width": int(round(hwf[i][1])),
                "height": int(round(hwf[i][0])),
            }
            for i, name in enumerate(names)
        }
    cameras = _parse_cameras(root / "sparse" / "cameras.txt")
    name2cam = _image_name_to_camera_id(root / "sparse" / "images.txt")
    return {
        name: {
            "params": cameras[cid]["params"],
            "width": cameras[cid]["width"],
            "height": cameras[cid]["height"],
        }
        for name, cid in name2cam.items()
    }
