#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from utils import definitions as defs


SINTEL_DPT_TAG = 202021.25
DEFAULT_SCENE_DIR = Path(defs.rootpath)
DEFAULT_DA3_DEPTH_DIR = Path(defs.da3_depth_dir)


@dataclass(frozen=True)
class NovelRecord:
    novel_id: int
    base_frame: int
    image_path: Path
    depth_path: Path
    c2w: np.ndarray


@dataclass(frozen=True)
class TrainRecord:
    frame_id: int
    image_name: str
    depth_path: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate DepthAnything novel-view depth by lifting DA3 depths into the DA3 camera world, "
            "using static pixels from all frames plus dynamic pixels from each novel view's time step."
        )
    )
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE_DIR)
    parser.add_argument("--da3-depth-dir", type=Path, default=DEFAULT_DA3_DEPTH_DIR)
    parser.add_argument("--novel-dir", type=Path, default=None, help="Default: <scene-dir>/novel")
    parser.add_argument(
        "--eval-split",
        choices=["novel", "train"],
        default="novel",
        help="'novel' evaluates novel metadata views; 'train' renders training views with the same lifted-cloud technique.",
    )
    parser.add_argument("--gt-depth-dir", type=Path, default=None, help="Training GT depth dir for --eval-split train. Default: <scene-dir>/depths")
    parser.add_argument("--gt-scale", type=float, default=1000.0, help="Divide GT png depth by this scale.")
    parser.add_argument("--min-depth", type=float, default=1e-6)
    parser.add_argument("--max-depth", type=float, default=-1.0, help="Optional max depth; <=0 disables.")
    parser.add_argument(
        "--normalize-stat",
        "--scale-align",
        dest="normalize_stat",
        choices=["median", "mean"],
        default="median",
        help="Normalize GT and rendered depth independently over the whole novel set.",
    )
    parser.add_argument("--point-stride", type=int, default=1, help="Lift every Nth static source pixel.")
    parser.add_argument("--dynamic-point-stride", type=int, default=1, help="Lift every Nth dynamic source pixel; independent of --point-stride.")
    parser.add_argument("--static-frame-stride", type=int, default=1, help="Lift static pixels from every Nth training frame.")
    parser.add_argument("--static-max-points", type=int, default=0, help="Optional deterministic cap for the static world cloud; <=0 disables.")
    parser.add_argument(
        "--point-radius",
        type=int,
        default=1,
        help="Integer splat radius in target pixels. 0=1x1, 1=3x3, 2=5x5.",
    )
    parser.add_argument("--mask-dir", type=Path, default=None, help="Dynamic masks directory. Default: <scene-dir>/masks")
    parser.add_argument("--mask-threshold", type=float, default=127.0, help="Mask pixels above this value are dynamic.")
    parser.add_argument(
        "--alpha-threshold",
        type=float,
        default=0.05,
        help=(
            "Treat rendered novel pixels as invalid when projected point-density alpha is below this value. "
            "0 disables the alpha filter."
        ),
    )
    parser.add_argument(
        "--alpha-density-radius",
        type=int,
        default=10,
        help="Pixel radius used to accumulate projected point density for alpha; <0 uses --point-radius.",
    )
    parser.add_argument(
        "--alpha-density-scale",
        type=float,
        default=1.0,
        help="Density scale in alpha=1-exp(-density/scale). Larger values require denser nearby points.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="Optional deterministic cap after combining the rendered world point cloud; <=0 disables.",
    )
    parser.add_argument("--vis-frame", type=int, default=0, help="Novel id to visualize; -1 uses first valid frame.")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--vis-out", type=Path, default=None)
    parser.add_argument("--rendered-depth-dir", type=Path, default=None)
    parser.add_argument(
        "--save-depth-png",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save one rendered/GT/error depth PNG for every novel view.",
    )
    parser.add_argument(
        "--depth-png-dir",
        type=Path,
        default=None,
        help="Default: <da3-depth-dir>/novel_depth_png",
    )
    parser.add_argument("--debug-open3d", action="store_true", help="Save Open3D point/cloud camera debug geometry.")
    parser.add_argument("--debug-open3d-window", action="store_true", help="Open the Open3D visualization window for debug geometry.")
    parser.add_argument("--debug-novel-id", type=int, default=0, help="Novel id to export in Open3D debug mode.")
    parser.add_argument("--debug-open3d-dir", type=Path, default=None, help="Default: <da3-depth-dir>/open3d_debug")
    parser.add_argument("--debug-open3d-max-points", type=int, default=200000, help="Max points per Open3D debug point cloud; <=0 disables.")
    parser.add_argument(
        "--gt-normalizer",
        type=float,
        default=None,
        help="Use this GT normalizer instead of fitting one from novel views.",
    )
    parser.add_argument(
        "--est-normalizer",
        type=float,
        default=None,
        help="Use this estimated-depth normalizer instead of fitting one from novel views.",
    )
    parser.add_argument(
        "--normalizer-scope",
        type=str,
        default="novel_dataset",
        help="Label recorded in JSON for the normalizer source.",
    )
    parser.add_argument("--json", action="store_true", help="Print RESULT_JSON line for scripts.")
    return parser.parse_args()


def _frame_id_from_name(name: str) -> int:
    stem = Path(name).stem
    match = re.search(r"(?<!\d)(\d+)$", stem)
    if match is None:
        raise ValueError(f"Could not extract frame id from image name: {name}")
    return int(match.group(1))


def _quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    qw, qx, qy, qz = (q / norm).tolist()
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _read_depth(path: Path, value_scale: float = 1.0) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path).astype(np.float64)
    if suffix == ".npz":
        data = np.load(path)
        for key in ("depth", "depths", "depth_map", "depth_maps", "pred_depth", "arr_0"):
            if key in data:
                return data[key].astype(np.float64)
        raise KeyError(f"No depth key found in {path.name}")
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Could not read depth image: {path}")
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth = depth.astype(np.float64)
        if value_scale > 0:
            depth /= float(value_scale)
        return depth
    if suffix == ".dpt":
        with path.open("rb") as f:
            tag = np.fromfile(f, dtype=np.float32, count=1)[0]
            if tag != SINTEL_DPT_TAG:
                raise ValueError(f"Wrong Sintel .dpt tag in {path}: expected {SINTEL_DPT_TAG}, got {tag}")
            width = int(np.fromfile(f, dtype=np.int32, count=1)[0])
            height = int(np.fromfile(f, dtype=np.int32, count=1)[0])
            size = width * height
            depth = np.fromfile(f, dtype=np.float32, count=size)
        if width <= 0 or height <= 0 or depth.size != size:
            raise ValueError(f"Invalid or truncated Sintel .dpt file: {path}")
        return depth.reshape(height, width).astype(np.float64)
    raise ValueError(f"Unsupported depth file: {path}")


def _prefer_npy_depth_path(path: Path) -> Path:
    if path.suffix.lower() == ".npy":
        return path
    npy_path = path.with_suffix(".npy")
    if npy_path.is_file():
        return npy_path
    return path


def _depth_files(depth_dir: Path) -> list[Path]:
    exts = {".npy", ".npz", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dpt"}
    skip = {"depth_maps.npy", "conf_maps.npy", "intrinsics.npy", "extrinsics_c2w.npy"}
    rank = {".npy": 0, ".npz": 1, ".dpt": 2, ".png": 3, ".tif": 4, ".tiff": 5, ".jpg": 6, ".jpeg": 7}
    by_stem: dict[str, Path] = {}
    if not depth_dir.is_dir():
        return []
    for path in depth_dir.iterdir():
        suffix = path.suffix.lower()
        if not path.is_file() or suffix not in exts or path.name in skip:
            continue
        old = by_stem.get(path.stem)
        if old is None or rank[suffix] < rank[old.suffix.lower()]:
            by_stem[path.stem] = path
    return sorted(by_stem.values(), key=lambda p: p.stem)


def _gt_depth_path_for_frame(
    scene_dir: Path,
    gt_depth_dir: Path,
    frame_idx: int,
    image_name: str | None,
) -> Path:
    files = _depth_files(gt_depth_dir)
    if not files:
        raise FileNotFoundError(f"No GT depth files found in {gt_depth_dir}")
    if image_name is not None:
        stem = Path(image_name).stem
        for path in files:
            if path.stem == stem:
                return path
    by_frame: dict[int, Path] = {}
    for path in files:
        try:
            by_frame[_frame_id_from_name(path.name)] = path
        except ValueError:
            pass
    if int(frame_idx) in by_frame:
        return by_frame[int(frame_idx)]
    if 0 <= int(frame_idx) < len(files):
        return files[int(frame_idx)]
    raise FileNotFoundError(f"No GT depth for frame {frame_idx} in {gt_depth_dir}")


def _load_camera_data(camera_path: Path) -> dict[int, dict[str, Any]]:
    cameras: dict[int, dict[str, Any]] = {}
    with camera_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(x) for x in parts[4:]]
            K = np.eye(3, dtype=np.float64)
            if model == "SIMPLE_PINHOLE":
                K[0, 0] = params[0]
                K[1, 1] = params[0]
                K[0, 2] = params[1]
                K[1, 2] = params[2]
            elif model == "PINHOLE":
                K[0, 0] = params[0]
                K[1, 1] = params[1]
                K[0, 2] = params[2]
                K[1, 2] = params[3]
            else:
                raise ValueError(f"Unsupported COLMAP camera model {model} in {camera_path}")
            cameras[camera_id] = {"K": K, "width": width, "height": height, "model": model}
    return cameras


def _load_image_data(image_path: Path) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    with image_path.open("r") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        qw, qx, qy, qz = [float(x) for x in parts[1:5]]
        tx, ty, tz = [float(x) for x in parts[5:8]]
        camera_id = int(parts[8])
        name = " ".join(parts[9:])
        images.append(
            {
                "name": name,
                "id": _frame_id_from_name(name),
                "camera_id": camera_id,
                "R": _quat_to_rot(qw, qx, qy, qz),
                "t": np.array([tx, ty, tz], dtype=np.float64),
            }
        )
        if i < len(lines):
            i += 1
    return images


def _load_colmap_gt(scene_dir: Path) -> dict[str, dict[str, Any]]:
    sparse_dir = scene_dir / "colmap" / "sparse"
    if not (sparse_dir / "cameras.txt").is_file() and (sparse_dir / "0").is_dir():
        sparse_dir = sparse_dir / "0"
    cameras = _load_camera_data(sparse_dir / "cameras.txt")
    images = _load_image_data(sparse_dir / "images.txt")
    gt: dict[str, dict[str, Any]] = {}
    for image in images:
        camera = cameras[image["camera_id"]]
        gt[image["name"]] = {
            "id": image["id"],
            "K": camera["K"].copy(),
            "R": image["R"],
            "t": image["t"],
            "camera_id": image["camera_id"],
            "width": camera["width"],
            "height": camera["height"],
        }
    return gt


def _load_gt(scene_dir: Path) -> dict[str, dict[str, Any]]:
    dataset_cat = scene_dir.parent.name.lower()
    if dataset_cat == "tum":
        from utils.readgt_TUM import load_tum_gt

        return load_tum_gt(str(scene_dir))
    if dataset_cat == "sintel":
        from utils.readgt_sintel import load_sintel_gt

        return load_sintel_gt(str(scene_dir))
    if dataset_cat == "replica":
        from utils.readgt_replica import load_replica_gt

        return load_replica_gt(str(scene_dir))
    return _load_colmap_gt(scene_dir)


def _c2w_from_gt(info: dict[str, Any]) -> np.ndarray:
    R_w2c = np.asarray(info["R"], dtype=np.float64)
    t_w2c = np.asarray(info["t"], dtype=np.float64).reshape(3)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R_w2c.T
    c2w[:3, 3] = -R_w2c.T @ t_w2c
    return c2w


def _load_novel_metadata(novel_dir: Path) -> list[NovelRecord]:
    meta_path = novel_dir / "metadata.txt"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing novel metadata: {meta_path}")
    records: list[NovelRecord] = []
    with meta_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 11:
                raise ValueError(f"Expected 11 fields in {meta_path}, got {len(parts)}: {line}")
            novel_id = int(parts[0])
            base_frame = int(parts[1])
            tx, ty, tz = [float(x) for x in parts[4:7]]
            qw, qx, qy, qz = [float(x) for x in parts[7:11]]
            c2w = np.eye(4, dtype=np.float64)
            c2w[:3, :3] = _quat_to_rot(qw, qx, qy, qz)
            c2w[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
            records.append(
                NovelRecord(
                    novel_id=novel_id,
                    base_frame=base_frame,
                    image_path=novel_dir / parts[2],
                    depth_path=_prefer_npy_depth_path(novel_dir / parts[3]),
                    c2w=c2w,
                )
            )
    return records


def _resolve_scene_image(scene_dir: Path, name: str) -> Path | None:
    candidates = [
        scene_dir / name,
        scene_dir / "images" / name,
        scene_dir / "images" / Path(name).name,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _image_size(path: Path | None) -> tuple[int, int] | None:
    if path is None:
        return None
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    return int(img.shape[1]), int(img.shape[0])


def _scale_K(K: np.ndarray, src_size: tuple[int, int] | None, dst_size: tuple[int, int]) -> np.ndarray:
    scaled = np.asarray(K, dtype=np.float64).copy()
    if src_size is None:
        return scaled
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    if src_w <= 0 or src_h <= 0:
        return scaled
    scaled[0, :] *= float(dst_w) / float(src_w)
    scaled[1, :] *= float(dst_h) / float(src_h)
    print(f"Scaling K from {src_size} to {dst_size}:")
    return scaled


def _as_homogeneous_c2ws(c2ws: np.ndarray) -> np.ndarray:
    c2ws = np.asarray(c2ws, dtype=np.float64)
    if c2ws.ndim != 3:
        raise ValueError(f"Expected camera extrinsics with shape (N,3/4,4), got {c2ws.shape}")
    if c2ws.shape[1:] == (4, 4):
        return c2ws
    if c2ws.shape[1:] == (3, 4):
        out = np.repeat(np.eye(4, dtype=np.float64)[None], c2ws.shape[0], axis=0)
        out[:, :3, :4] = c2ws
        return out
    raise ValueError(f"Expected camera extrinsics with shape (N,3/4,4), got {c2ws.shape}")


def _load_da3_cameras(da3_depth_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    intr_path = da3_depth_dir / "intrinsics.npy"
    c2w_path = da3_depth_dir / "extrinsics_c2w.npy"
    if not intr_path.is_file():
        raise FileNotFoundError(f"Missing DA3 intrinsics: {intr_path}")
    if not c2w_path.is_file():
        raise FileNotFoundError(
            f"Missing DA3 camera poses: {c2w_path}. This evaluator needs DepthAnything camera output."
        )
    intrinsics = np.load(intr_path).astype(np.float64)
    if intrinsics.ndim == 2:
        intrinsics = intrinsics[None]
    if intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
        raise ValueError(f"Expected DA3 intrinsics shape (N,3,3) or (3,3), got {intrinsics.shape}")
    c2ws = _as_homogeneous_c2ws(np.load(c2w_path))
    return intrinsics, c2ws


def _load_da3_depths(da3_depth_dir: Path) -> np.ndarray:
    for name in ("depth_maps.npy", "depths.npy"):
        path = da3_depth_dir / name
        if path.is_file():
            depths = np.load(path).astype(np.float64)
            if depths.ndim != 3:
                raise ValueError(f"Expected (N,H,W) depth stack in {path}, got {depths.shape}")
            return depths
    raise FileNotFoundError(f"Could not find depth_maps.npy or depths.npy in {da3_depth_dir}")


def _image_order(scene_dir: Path) -> list[Path]:
    image_dir = scene_dir / "images"
    exts = {".png", ".jpg", ".jpeg"}
    if not image_dir.is_dir():
        return []
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in exts)


def _mask_files(mask_dir: Path) -> list[Path]:
    if not mask_dir.is_dir():
        return []
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy"}
    return sorted(path for path in mask_dir.iterdir() if path.is_file() and path.suffix.lower() in exts)


def _mask_path_for_frame(mask_dir: Path, frame_index: int, image_name: str | None = None) -> Path | None:
    files = _mask_files(mask_dir)
    if not files:
        return None
    if image_name is not None:
        stem = Path(image_name).stem
        for path in files:
            if path.stem == stem:
                return path
    by_frame: dict[int, Path] = {}
    for path in files:
        try:
            by_frame[_frame_id_from_name(path.name)] = path
        except ValueError:
            pass
    if int(frame_index) in by_frame:
        return by_frame[int(frame_index)]
    if 0 <= int(frame_index) < len(files):
        return files[int(frame_index)]
    return None


def _read_dynamic_mask(
    mask_dir: Path,
    frame_index: int,
    shape: tuple[int, int],
    threshold: float,
    image_name: str | None = None,
) -> np.ndarray:
    path = _mask_path_for_frame(mask_dir, frame_index, image_name=image_name)
    if path is None:
        return np.zeros(shape, dtype=bool)
    if path.suffix.lower() == ".npy":
        mask = np.load(path)
    else:
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {path}")
        if mask.ndim == 3:
            mask = mask[..., 0]
    if mask.shape != shape:
        mask = cv2.resize(mask.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.asarray(mask, dtype=np.float64) > float(threshold)


def _base_info_for_frame(
    scene_dir: Path,
    gt: dict[str, dict[str, Any]],
    depths: np.ndarray,
    base_frame: int,
) -> tuple[str, dict[str, Any], int, Path | None]:
    by_frame = {int(info["id"]): (name, info) for name, info in gt.items() if "id" in info}
    if base_frame not in by_frame:
        raise KeyError(f"Base frame {base_frame} was not found in GT poses.")
    name, info = by_frame[base_frame]

    images = _image_order(scene_dir)
    by_basename = {path.name: i for i, path in enumerate(images)}
    by_stem_frame = {}
    for i, path in enumerate(images):
        try:
            by_stem_frame[_frame_id_from_name(path.name)] = i
        except ValueError:
            pass

    image_path = _resolve_scene_image(scene_dir, name)
    depth_index = by_basename.get(Path(name).name)
    if depth_index is None:
        depth_index = by_stem_frame.get(base_frame)
    if depth_index is None and 0 <= base_frame < depths.shape[0]:
        depth_index = base_frame
    if depth_index is None or depth_index >= depths.shape[0]:
        raise IndexError(f"Could not map base frame {base_frame} to DA3 depth index.")
    return name, info, int(depth_index), image_path


def _umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"Expected matched point arrays (N,3), got {src.shape} and {dst.shape}")
    if src.shape[0] < 3:
        return 1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src0 = src - mu_src
    dst0 = dst - mu_dst
    cov = (dst0.T @ src0) / float(src.shape[0])
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3, dtype=np.float64)
    if np.linalg.det(U @ Vt) < 0:
        D[-1, -1] = -1.0
    R = U @ D @ Vt
    var_src = float(np.mean(np.sum(src0 * src0, axis=1)))
    scale = float(np.trace(np.diag(S) @ D) / max(var_src, 1e-12))
    t = mu_dst - scale * (R @ mu_src)
    return scale, R, t


def _gt_to_da_similarity(gt: dict[str, dict[str, Any]], da_c2ws: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src = []
    dst = []
    for info in gt.values():
        if "id" not in info:
            continue
        idx = int(info["id"])
        if 0 <= idx < da_c2ws.shape[0]:
            src.append(_c2w_from_gt(info)[:3, 3])
            dst.append(da_c2ws[idx, :3, 3])
    if len(src) < 3:
        print("[da-camera] Too few GT/DA camera pairs for similarity; using novel poses as-is.")
        return 1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    scale, R, t = _umeyama_similarity(np.stack(src, axis=0), np.stack(dst, axis=0))
    err = scale * (np.stack(src, axis=0) @ R.T) + t.reshape(1, 3) - np.stack(dst, axis=0)
    print(
        "[da-camera] GT->DA camera alignment "
        f"pairs={len(src)} scale={scale:.6g} center_rmse={float(np.sqrt(np.mean(np.sum(err * err, axis=1)))):.6g}"
    )
    return scale, R, t


def _transform_c2w_similarity(c2w: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R @ c2w[:3, :3]
    out[:3, 3] = scale * (R @ c2w[:3, 3]) + t
    return out


def _novel_c2w_from_da_base(
    rec: NovelRecord,
    gt: dict[str, dict[str, Any]],
    da_c2ws: np.ndarray,
    offset_scale: float,
) -> np.ndarray:
    base_info = None
    for info in gt.values():
        if int(info.get("id", -1)) == int(rec.base_frame):
            base_info = info
            break
    if base_info is None or not (0 <= int(rec.base_frame) < da_c2ws.shape[0]):
        raise KeyError(f"Could not build DA novel pose for base frame {rec.base_frame}")

    gt_base = _c2w_from_gt(base_info)
    da_base = da_c2ws[int(rec.base_frame)]
    # Novel metadata perturbs the base camera in the base camera's right/up
    # plane. Preserve that local delta and the relative rotation, but express
    # them in DepthAnything's camera world using the DA base pose.
    delta_cam = gt_base[:3, :3].T @ (rec.c2w[:3, 3] - gt_base[:3, 3])
    rel_rot = gt_base[:3, :3].T @ rec.c2w[:3, :3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = da_base[:3, :3] @ rel_rot
    out[:3, 3] = da_base[:3, 3] + float(offset_scale) * (da_base[:3, :3] @ delta_cam)
    return out


def _lift_depth_to_world(
    depth: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    min_depth: float,
    max_depth: float,
    point_stride: int,
    max_points: int,
    pixel_mask: np.ndarray | None = None,
) -> np.ndarray:
    stride = max(1, int(point_stride))
    sampled = depth[::stride, ::stride]
    sampled_mask = None
    if pixel_mask is not None:
        if pixel_mask.shape != depth.shape:
            pixel_mask = cv2.resize(
                pixel_mask.astype(np.uint8),
                (depth.shape[1], depth.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        sampled_mask = pixel_mask[::stride, ::stride]
    v, u = np.indices(sampled.shape, dtype=np.float64)
    u *= stride
    v *= stride
    z = sampled.reshape(-1)
    u = u.reshape(-1)
    v = v.reshape(-1)

    valid = np.isfinite(z) & (z > float(min_depth))
    if sampled_mask is not None:
        valid &= sampled_mask.reshape(-1).astype(bool)
    if float(max_depth) > 0:
        valid &= z <= float(max_depth)
    u = u[valid]
    v = v[valid]
    z = z[valid]
    if z.size == 0:
        return np.zeros((0, 3), dtype=np.float64)

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    points_cam = np.stack([x, y, z], axis=1)
    points_world = points_cam @ c2w[:3, :3].T + c2w[:3, 3]

    if max_points > 0 and points_world.shape[0] > max_points:
        rng = np.random.default_rng(0)
        keep = rng.choice(points_world.shape[0], size=max_points, replace=False)
        points_world = points_world[keep]
    return points_world


def _render_depth_from_points(
    points_world: np.ndarray,
    c2w: np.ndarray,
    K: np.ndarray,
    height: int,
    width: int,
    min_depth: float,
    max_depth: float,
    radius: int,
    alpha_density_radius: int,
    alpha_density_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    R_c2w = c2w[:3, :3]
    C = c2w[:3, 3]
    points_cam = (points_world - C) @ R_c2w
    z = points_cam[:, 2]
    valid = np.isfinite(z) & (z > float(min_depth))
    if float(max_depth) > 0:
        valid &= z <= float(max_depth)
    points_cam = points_cam[valid]
    z = z[valid]
    if z.size == 0:
        empty = np.zeros((height, width), dtype=np.float64)
        return empty, empty.copy()

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    u = np.rint(fx * (points_cam[:, 0] / z) + cx).astype(np.int64)
    v = np.rint(fy * (points_cam[:, 1] / z) + cy).astype(np.int64)

    depth_flat = np.full(height * width, np.inf, dtype=np.float64)
    r = max(0, int(radius))
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            uu = u + dx
            vv = v + dy
            inside = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
            if np.any(inside):
                pix = vv[inside] * width + uu[inside]
                np.minimum.at(depth_flat, pix, z[inside])

    density_flat = np.zeros(height * width, dtype=np.float64)
    ar = max(0, int(alpha_density_radius))
    for dy in range(-ar, ar + 1):
        for dx in range(-ar, ar + 1):
            uu = u + dx
            vv = v + dy
            inside = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
            if np.any(inside):
                pix = vv[inside] * width + uu[inside]
                np.add.at(density_flat, pix, np.exp(-0.5 * (dx * dx + dy * dy)))

    rendered = depth_flat.reshape(height, width)
    rendered[~np.isfinite(rendered)] = 0.0
    density = density_flat.reshape(height, width)
    alpha = 1.0 - np.exp(-density / max(float(alpha_density_scale), 1e-12))
    return rendered, alpha


def _cap_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    max_points = int(max_points)
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(seed)
    keep = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[keep]


def _build_static_world_points(
    scene_dir: Path,
    depths: np.ndarray,
    da_intrinsics: np.ndarray,
    da_c2ws: np.ndarray,
    gt: dict[str, dict[str, Any]],
    mask_dir: Path,
    args: argparse.Namespace,
) -> tuple[np.ndarray, int]:
    chunks: list[np.ndarray] = []
    image_names_by_frame = {int(info["id"]): name for name, info in gt.items() if "id" in info}
    n = min(depths.shape[0], da_c2ws.shape[0])
    frame_stride = max(1, int(args.static_frame_stride))
    for frame_idx in range(0, n, frame_stride):
        depth = depths[frame_idx]
        K = da_intrinsics[frame_idx if da_intrinsics.shape[0] > 1 else 0]
        dynamic_mask = _read_dynamic_mask(
            mask_dir,
            frame_idx,
            depth.shape,
            args.mask_threshold,
            image_name=image_names_by_frame.get(frame_idx),
        )
        static_mask = ~dynamic_mask
        pts = _lift_depth_to_world(
            depth,
            K,
            da_c2ws[frame_idx],
            args.min_depth,
            args.max_depth,
            args.point_stride,
            0,
            pixel_mask=static_mask,
        )
        if pts.size:
            chunks.append(pts)
    if not chunks:
        return np.zeros((0, 3), dtype=np.float64), 0
    points = np.concatenate(chunks, axis=0)
    points = _cap_points(points, int(args.static_max_points), seed=0)
    return points, len(range(0, n, frame_stride))


def _dynamic_world_points_for_frame(
    depths: np.ndarray,
    da_intrinsics: np.ndarray,
    da_c2ws: np.ndarray,
    gt: dict[str, dict[str, Any]],
    mask_dir: Path,
    frame_idx: int,
    args: argparse.Namespace,
) -> np.ndarray:
    if frame_idx < 0 or frame_idx >= min(depths.shape[0], da_c2ws.shape[0]):
        return np.zeros((0, 3), dtype=np.float64)
    image_name = None
    for name, info in gt.items():
        if int(info.get("id", -1)) == int(frame_idx):
            image_name = name
            break
    depth = depths[frame_idx]
    dynamic_mask = _read_dynamic_mask(mask_dir, frame_idx, depth.shape, args.mask_threshold, image_name=image_name)
    return _lift_depth_to_world(
        depth,
        da_intrinsics[frame_idx if da_intrinsics.shape[0] > 1 else 0],
        da_c2ws[frame_idx],
        args.min_depth,
        args.max_depth,
        args.dynamic_point_stride,
        0,
        pixel_mask=dynamic_mask,
    )


def _train_records(
    scene_dir: Path,
    gt: dict[str, dict[str, Any]],
    gt_depth_dir: Path,
    n_frames: int,
) -> list[TrainRecord]:
    image_names_by_frame = {int(info["id"]): name for name, info in gt.items() if "id" in info}
    records: list[TrainRecord] = []
    for frame_idx in range(n_frames):
        image_name = image_names_by_frame.get(frame_idx)
        try:
            depth_path = _gt_depth_path_for_frame(scene_dir, gt_depth_dir, frame_idx, image_name)
        except FileNotFoundError:
            continue
        records.append(TrainRecord(frame_id=frame_idx, image_name=image_name or f"frame{frame_idx:06d}", depth_path=depth_path))
    return records


def _valid_mask(gt: np.ndarray, est: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    if gt.shape != est.shape:
        raise ValueError(f"Shape mismatch: GT {gt.shape} vs estimate {est.shape}")
    valid = np.isfinite(gt) & np.isfinite(est)
    valid &= gt > float(min_depth)
    valid &= est > float(min_depth)
    if float(max_depth) > 0:
        valid &= gt <= float(max_depth)
        valid &= est <= float(max_depth)
    return valid


def _gt_valid_mask(
    gt: np.ndarray,
    est: np.ndarray,
    min_depth: float,
    max_depth: float,
    alpha: np.ndarray | None = None,
    alpha_threshold: float = 0.0,
) -> np.ndarray:
    if gt.shape != est.shape:
        raise ValueError(f"Shape mismatch: GT {gt.shape} vs estimate {est.shape}")
    valid = np.isfinite(gt) & np.isfinite(est)
    valid &= gt > float(min_depth)
    if float(max_depth) > 0:
        valid &= gt <= float(max_depth)
    if alpha is not None and float(alpha_threshold) > 0.0:
        if alpha.shape != gt.shape:
            raise ValueError(f"Shape mismatch: GT {gt.shape} vs alpha {alpha.shape}")
        valid &= np.isfinite(alpha) & (alpha >= float(alpha_threshold))
    return valid


def _normalizer(values: np.ndarray, method: str) -> float:
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return 1.0
    scale = float(np.mean(values)) if method == "mean" else float(np.median(values))
    if not np.isfinite(scale) or scale <= 1e-12:
        return 1.0
    return scale


def _colorize(depth: np.ndarray, valid: np.ndarray | None = None, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    if valid is None:
        valid = np.isfinite(depth) & (depth > 0)
    if vmin is None:
        vmin = float(np.percentile(depth[valid], 2)) if np.any(valid) else 0.0
    if vmax is None:
        vmax = float(np.percentile(depth[valid], 98)) if np.any(valid) else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = np.clip((depth - vmin) / (vmax - vmin), 0.0, 1.0)
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~valid] = 0
    return img


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _save_visualization(
    path: Path,
    gt_norm: np.ndarray,
    est_norm: np.ndarray,
    valid: np.ndarray,
    source_depth: np.ndarray,
    frame_label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vmin = float(np.percentile(gt_norm[valid], 2)) if np.any(valid) else 0.0
    vmax = float(np.percentile(gt_norm[valid], 98)) if np.any(valid) else 1.0
    err = np.zeros_like(gt_norm)
    err[valid] = np.abs(gt_norm[valid] - est_norm[valid])
    emax = float(np.percentile(err[valid], 98)) if np.any(valid) else 1.0

    gt_img = _label(_colorize(gt_norm, valid, vmin, vmax), f"GT novel {frame_label}")
    est_img = _label(_colorize(est_norm, valid, vmin, vmax), "DA3 base point cloud render")
    err_img = _label(_colorize(err, valid, 0.0, max(emax, 1e-6)), "abs error")
    src_valid = np.isfinite(source_depth) & (source_depth > 0)
    src_img = _label(_colorize(source_depth, src_valid), "source DA3 base depth")
    src_img = cv2.resize(src_img, (gt_img.shape[1], gt_img.shape[0]), interpolation=cv2.INTER_AREA)

    canvas = np.vstack([np.hstack([gt_img, est_img]), np.hstack([err_img, src_img])])
    cv2.imwrite(str(path), canvas)


def _save_depth_png(
    path: Path,
    gt_depth: np.ndarray,
    est_depth: np.ndarray,
    alpha: np.ndarray,
    valid: np.ndarray,
    gt_normalizer: float,
    est_normalizer: float,
    frame_label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gt_norm = gt_depth / max(float(gt_normalizer), 1e-12)
    est_norm = est_depth / max(float(est_normalizer), 1e-12)
    gt_valid = np.isfinite(gt_depth) & (gt_depth > 0)
    est_valid = np.isfinite(est_depth) & (est_depth > 0)
    scale_valid = valid if np.any(valid) else (gt_valid | est_valid)
    vmin = float(np.percentile(gt_norm[scale_valid], 2)) if np.any(scale_valid) else 0.0
    vmax = float(np.percentile(gt_norm[scale_valid], 98)) if np.any(scale_valid) else 1.0
    err = np.zeros_like(gt_norm)
    err[valid] = np.abs(gt_norm[valid] - est_norm[valid])
    emax = float(np.percentile(err[valid], 98)) if np.any(valid) else 1.0
    alpha_valid = np.isfinite(alpha) & (alpha > 0)

    gt_img = _label(_colorize(gt_norm, gt_valid, vmin, vmax), f"GT novel {frame_label}")
    est_img = _label(_colorize(est_norm, est_valid, vmin, vmax), "rendered DA3 depth")
    err_img = _label(_colorize(err, valid, 0.0, max(emax, 1e-6)), "abs error")
    alpha_img = _label(_colorize(alpha, alpha_valid, 0.0, 1.0), "render alpha")
    canvas = np.vstack([np.hstack([gt_img, est_img]), np.hstack([err_img, alpha_img])])
    cv2.imwrite(str(path), canvas)


def _camera_frustum_lines(
    c2w: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    corners_px = np.array(
        [
            [0.0, 0.0],
            [float(width - 1), 0.0],
            [float(width - 1), float(height - 1)],
            [0.0, float(height - 1)],
        ],
        dtype=np.float64,
    )
    z = float(depth)
    corners_cam = np.column_stack(
        [
            (corners_px[:, 0] - cx) / max(fx, 1e-12) * z,
            (corners_px[:, 1] - cy) / max(fy, 1e-12) * z,
            np.full(4, z, dtype=np.float64),
        ]
    )
    center_world = c2w[:3, 3].reshape(1, 3)
    corners_world = corners_cam @ c2w[:3, :3].T + c2w[:3, 3]
    points = np.concatenate([center_world, corners_world], axis=0)
    lines = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1],
        ],
        dtype=np.int32,
    )
    return points, lines


def _open3d_point_cloud(points: np.ndarray, color: tuple[float, float, float], max_points: int = 200000):
    import open3d as o3d

    points = np.asarray(points, dtype=np.float64)
    if max_points > 0 and points.shape[0] > max_points:
        rng = np.random.default_rng(7)
        points = points[rng.choice(points.shape[0], size=max_points, replace=False)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(color, dtype=np.float64), (points.shape[0], 1)))
    return pcd


def _open3d_lineset(
    frustums: list[tuple[np.ndarray, np.ndarray, tuple[float, float, float]]],
):
    import open3d as o3d

    all_points = []
    all_lines = []
    all_colors = []
    offset = 0
    for points, lines, color in frustums:
        all_points.append(points)
        all_lines.append(lines + offset)
        all_colors.append(np.tile(np.asarray(color, dtype=np.float64), (lines.shape[0], 1)))
        offset += points.shape[0]
    line_set = o3d.geometry.LineSet()
    if all_points:
        line_set.points = o3d.utility.Vector3dVector(np.concatenate(all_points, axis=0))
        line_set.lines = o3d.utility.Vector2iVector(np.concatenate(all_lines, axis=0))
        line_set.colors = o3d.utility.Vector3dVector(np.concatenate(all_colors, axis=0))
    return line_set


def _save_open3d_debug(
    out_dir: Path,
    static_points: np.ndarray,
    dynamic_points: np.ndarray,
    da_c2ws: np.ndarray,
    novel_c2w_da: np.ndarray,
    train_c2w_da: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    rec: NovelRecord,
    open_window: bool,
    debug_max_points: int,
) -> None:
    import open3d as o3d

    out_dir.mkdir(parents=True, exist_ok=True)
    if static_points.size:
        static_pcd = _open3d_point_cloud(static_points, (0.62, 0.62, 0.62), max_points=int(debug_max_points))
    else:
        static_pcd = _open3d_point_cloud(np.zeros((0, 3), dtype=np.float64), (0.62, 0.62, 0.62))
    if dynamic_points.size:
        dynamic_pcd = _open3d_point_cloud(dynamic_points, (1.0, 0.05, 0.02), max_points=int(debug_max_points))
    else:
        dynamic_pcd = _open3d_point_cloud(np.zeros((0, 3), dtype=np.float64), (1.0, 0.05, 0.02))

    camera_stride = max(1, da_c2ws.shape[0] // 20)
    depth = max(0.05, float(np.median(np.linalg.norm(da_c2ws[:, :3, 3] - da_c2ws[0, :3, 3], axis=1))) * 0.03)
    frustums: list[tuple[np.ndarray, np.ndarray, tuple[float, float, float]]] = []
    for c2w in da_c2ws[::camera_stride]:
        pts, lines = _camera_frustum_lines(c2w, K, width, height, depth)
        frustums.append((pts, lines, (0.1, 0.35, 1.0)))
    pts, lines = _camera_frustum_lines(train_c2w_da, K, width, height, depth * 1.8)
    frustums.append((pts, lines, (1.0, 0.55, 0.05)))
    pts, lines = _camera_frustum_lines(novel_c2w_da, K, width, height, depth * 2.2)
    frustums.append((pts, lines, (0.0, 0.9, 0.1)))
    line_set = _open3d_lineset(frustums)

    centers = np.concatenate(
        [
            da_c2ws[::camera_stride, :3, 3],
            train_c2w_da[:3, 3].reshape(1, 3),
            novel_c2w_da[:3, 3].reshape(1, 3),
        ],
        axis=0,
    )
    center_colors = np.concatenate(
        [
            np.tile(np.array([[0.1, 0.35, 1.0]]), (da_c2ws[::camera_stride].shape[0], 1)),
            np.array([[1.0, 0.55, 0.05], [0.0, 0.9, 0.1]]),
        ],
        axis=0,
    )
    centers_pcd = o3d.geometry.PointCloud()
    centers_pcd.points = o3d.utility.Vector3dVector(centers)
    centers_pcd.colors = o3d.utility.Vector3dVector(center_colors)

    o3d.io.write_point_cloud(str(out_dir / f"novel{rec.novel_id:06d}_static_points.ply"), static_pcd)
    o3d.io.write_point_cloud(str(out_dir / f"novel{rec.novel_id:06d}_dynamic_points.ply"), dynamic_pcd)
    o3d.io.write_point_cloud(str(out_dir / f"novel{rec.novel_id:06d}_camera_centers.ply"), centers_pcd)
    o3d.io.write_line_set(str(out_dir / f"novel{rec.novel_id:06d}_camera_frustums.ply"), line_set)
    print(
        f"[open3d-debug] saved static/dynamic points and camera frustums for novel={rec.novel_id} to {out_dir}"
    )
    print("[open3d-debug] colors: static=gray dynamic=red DA train cameras=blue base camera=orange novel camera=green")
    if open_window:
        o3d.visualization.draw_geometries(
            [static_pcd, dynamic_pcd, centers_pcd, line_set],
            window_name=f"DA3 novel depth debug {rec.novel_id}",
        )


def _summarize_rendered_items(
    *,
    rendered_items: list[tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    gt_normalizer: float,
    est_normalizer: float,
    args: argparse.Namespace,
    da3_depth_dir: Path,
    vis_out: Path | None,
    depth_png_dir: Path | None,
) -> tuple[dict[str, Any], Any | None]:
    if not np.isfinite(gt_normalizer) or gt_normalizer <= 1e-12:
        raise ValueError(f"Invalid GT normalizer: {gt_normalizer}")
    if not np.isfinite(est_normalizer) or est_normalizer <= 1e-12:
        raise ValueError(f"Invalid estimated-depth normalizer: {est_normalizer}")

    total_abs = 0.0
    total_valid = 0
    frame_l1s: list[float] = []
    vis_item = None
    first_valid_item = None
    for item in rendered_items:
        rec, gt_depth, est_depth, alpha, valid, source_depth = item
        rec_id = int(getattr(rec, "novel_id", getattr(rec, "frame_id", 0)))
        if depth_png_dir is not None:
            _save_depth_png(
                depth_png_dir / f"{args.eval_split}_depth_eval_{rec_id:06d}.png",
                gt_depth,
                est_depth,
                alpha,
                valid,
                gt_normalizer,
                est_normalizer,
                f"{rec_id:06d}",
            )
        if not np.any(valid):
            continue
        gt_norm = gt_depth / gt_normalizer
        est_norm = est_depth / est_normalizer
        diff = np.abs(gt_norm[valid] - est_norm[valid])
        print(np.median(diff), np.mean(diff), np.max(diff))
        total_abs += float(np.sum(diff))
        total_valid += int(diff.size)
        frame_l1s.append(float(np.mean(diff)))
        if first_valid_item is None:
            first_valid_item = item
        if args.vis_frame >= 0 and rec_id == args.vis_frame:
            vis_item = item

    if vis_item is None:
        vis_item = first_valid_item
    if vis_out is None:
        vis_out = da3_depth_dir / f"{args.eval_split}_depth_eval_vis.png"
    if vis_item is not None:
        rec, gt_depth, est_depth, alpha, valid, source_depth = vis_item
        rec_id = int(getattr(rec, "novel_id", getattr(rec, "frame_id", 0)))
        _save_visualization(
            vis_out.resolve(),
            gt_depth / gt_normalizer,
            est_depth / est_normalizer,
            valid,
            source_depth,
            f"{rec_id:06d}",
        )

    global_l1 = total_abs / max(total_valid, 1)
    avg_l1 = float(np.mean(frame_l1s)) if frame_l1s else None
    return (
        {
            "num_frames_used": len(frame_l1s),
            "num_valid_pixels": total_valid,
            "gt_normalizer": float(gt_normalizer),
            "est_normalizer": float(est_normalizer),
            "l1": float(global_l1),
            "l1_avg_frame": avg_l1,
            "vis_out": str(vis_out.resolve()),
        },
        vis_item,
    )


def _evaluate(args: argparse.Namespace) -> dict[str, Any]:
    scene_dir = args.scene_dir.resolve()
    da3_depth_dir = args.da3_depth_dir.resolve()
    novel_dir = (args.novel_dir or (scene_dir / "novel")).resolve()
    gt_depth_dir = (args.gt_depth_dir or (scene_dir / "depths")).resolve()
    rendered_depth_dir = (
        args.rendered_depth_dir.resolve()
        if args.rendered_depth_dir
        else (da3_depth_dir / f"{args.eval_split}_render_depth_arrays").resolve()
    )
    rendered_depth_dir.mkdir(parents=True, exist_ok=True)
    depth_png_dir = None
    if args.save_depth_png:
        depth_png_dir = (args.depth_png_dir or (da3_depth_dir / f"{args.eval_split}_depth_png")).resolve()
        depth_png_dir.mkdir(parents=True, exist_ok=True)

    gt = _load_gt(scene_dir)
    depths = _load_da3_depths(da3_depth_dir)
    da_intrinsics, da_c2ws = _load_da3_cameras(da3_depth_dir)
    da_n = min(depths.shape[0], da_c2ws.shape[0])
    if da_intrinsics.shape[0] not in (1, da_n, depths.shape[0]):
        raise ValueError(
            f"DA3 intrinsics count {da_intrinsics.shape[0]} does not match depth/camera count "
            f"{depths.shape[0]}/{da_c2ws.shape[0]}"
        )

    mask_dir = (args.mask_dir or (scene_dir / "masks")).resolve()
    gt_to_da_scale, _gt_to_da_R, _gt_to_da_t = _gt_to_da_similarity(gt, da_c2ws)
    static_points, static_frames = _build_static_world_points(
        scene_dir,
        depths,
        da_intrinsics,
        da_c2ws,
        gt,
        mask_dir,
        args,
    )
    print(
        f"[da3-{args.eval_split}] lifted static cloud: {static_points.shape[0]} points "
        f"from {static_frames} frame(s), mask_dir={mask_dir}"
    )

    gt_chunks: list[np.ndarray] = []
    est_chunks: list[np.ndarray] = []
    rendered_items: list[tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    dynamic_cache: dict[int, np.ndarray] = {}
    alpha_density_radius = args.point_radius if int(args.alpha_density_radius) < 0 else int(args.alpha_density_radius)

    def combined_points(frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
        if frame_idx not in dynamic_cache:
            dynamic_cache[frame_idx] = _dynamic_world_points_for_frame(
                depths,
                da_intrinsics,
                da_c2ws,
                gt,
                mask_dir,
                frame_idx,
                args,
            )
            print(
                f"[da3-{args.eval_split}] lifted dynamic cloud frame={frame_idx}: "
                f"{dynamic_cache[frame_idx].shape[0]} points"
            )
        dynamic_points = dynamic_cache[frame_idx]
        if static_points.size and dynamic_points.size:
            points_world = np.concatenate([static_points, dynamic_points], axis=0)
        elif static_points.size:
            points_world = static_points
        else:
            points_world = dynamic_points
        return _cap_points(points_world, int(args.max_points), seed=int(frame_idx)), dynamic_points

    def append_rendered(
        rec: Any,
        frame_idx: int,
        target_c2w: np.ndarray,
        gt_depth_path: Path,
        label: str,
        dynamic_points: np.ndarray | None = None,
    ) -> None:
        points_world, cached_dynamic_points = combined_points(frame_idx)
        if dynamic_points is None:
            dynamic_points = cached_dynamic_points
        source_depth = depths[frame_idx] if 0 <= frame_idx < depths.shape[0] else depths[0]
        gt_depth = _read_depth(gt_depth_path, args.gt_scale)
        target_h, target_w = gt_depth.shape
        K_base = da_intrinsics[frame_idx if da_intrinsics.shape[0] > 1 and frame_idx < da_intrinsics.shape[0] else 0]
        K_target = _scale_K(K_base, (source_depth.shape[1], source_depth.shape[0]), (target_w, target_h))

        if args.eval_split == "novel" and args.debug_open3d and int(rec.novel_id) == int(args.debug_novel_id):
            train_c2w_da = da_c2ws[frame_idx if 0 <= frame_idx < da_c2ws.shape[0] else 0]
            debug_dir = (args.debug_open3d_dir or (da3_depth_dir / "open3d_debug")).resolve()
            _save_open3d_debug(
                debug_dir,
                static_points,
                dynamic_points,
                da_c2ws,
                target_c2w,
                train_c2w_da,
                K_target,
                target_w,
                target_h,
                rec,
                bool(args.debug_open3d_window),
                int(args.debug_open3d_max_points),
            )

        est_depth, alpha = _render_depth_from_points(
            points_world,
            target_c2w,
            K_target,
            target_h,
            target_w,
            args.min_depth,
            args.max_depth,
            args.point_radius,
            alpha_density_radius,
            args.alpha_density_scale,
        )
        print(
            f"[da3-{args.eval_split}] {label} points={points_world.shape[0]} "
            f"render_px={int(np.count_nonzero(est_depth > 0))} "
            f"alpha_px={int(np.count_nonzero(alpha >= float(args.alpha_threshold)))}"
        )
        valid = _gt_valid_mask(
            gt_depth,
            est_depth,
            args.min_depth,
            args.max_depth,
            alpha=alpha,
            alpha_threshold=args.alpha_threshold,
        )
        rendered_items.append((rec, gt_depth, est_depth, alpha, valid, source_depth))
        if np.any(valid):
            gt_chunks.append(gt_depth[valid])
            est_chunks.append(est_depth[valid])
        rec_id = int(getattr(rec, "novel_id", getattr(rec, "frame_id", frame_idx)))
        np.save(rendered_depth_dir / f"{args.eval_split}{rec_id:06d}_depth.npy", est_depth.astype(np.float32))
        np.save(rendered_depth_dir / f"{args.eval_split}{rec_id:06d}_alpha.npy", alpha.astype(np.float32))
        np.save(rendered_depth_dir / f"{args.eval_split}{rec_id:06d}_gt.npy", gt_depth.astype(np.float32))
        np.save(rendered_depth_dir / f"{args.eval_split}{rec_id:06d}_valid.npy", valid.astype(bool))

    if args.eval_split == "train":
        records: list[Any] = _train_records(scene_dir, gt, gt_depth_dir, da_n)
        if not records:
            raise ValueError(f"No train GT depth records found in {gt_depth_dir}")
        for rec in records:
            frame_idx = int(rec.frame_id)
            append_rendered(
                rec,
                frame_idx,
                da_c2ws[frame_idx],
                rec.depth_path,
                f"frame={frame_idx:06d}",
            )
    else:
        records = _load_novel_metadata(novel_dir)
        if not records:
            raise ValueError(f"No novel records found in {novel_dir / 'metadata.txt'}")
        for rec in records:
            frame_idx = int(rec.base_frame)
            target_c2w = _novel_c2w_from_da_base(rec, gt, da_c2ws, gt_to_da_scale)
            append_rendered(
                rec,
                frame_idx,
                target_c2w,
                rec.depth_path,
                f"novel={rec.novel_id:06d} base={rec.base_frame:06d}",
            )

    if not gt_chunks:
        raise ValueError(f"No valid pixels found between rendered {args.eval_split} depths and GT.")

    gt_all = np.concatenate(gt_chunks)
    est_all = np.concatenate(est_chunks)
    gt_normalizer = float(args.gt_normalizer) if args.gt_normalizer is not None else _normalizer(gt_all, args.normalize_stat)
    est_normalizer = float(args.est_normalizer) if args.est_normalizer is not None else _normalizer(est_all, args.normalize_stat)
    print(f"[da3-{args.eval_split}] est_normalizer={est_normalizer}")
    summary, _vis_item = _summarize_rendered_items(
        rendered_items=rendered_items,
        gt_normalizer=gt_normalizer,
        est_normalizer=est_normalizer,
        args=args,
        da3_depth_dir=da3_depth_dir,
        vis_out=args.vis_out,
        depth_png_dir=depth_png_dir,
    )
    print(
        f"DepthAnything {args.eval_split} rendered-depth avg L1: {summary['l1_avg_frame']:.6f} "
        f"(global {summary['l1']:.6f}, frames {summary['num_frames_used']}/{len(records)})"
    )

    return {
        "scene_dir": str(scene_dir),
        "da3_depth_dir": str(da3_depth_dir),
        "eval_split": str(args.eval_split),
        "novel_dir": str(novel_dir) if args.eval_split == "novel" else None,
        "gt_depth_dir": str(gt_depth_dir) if args.eval_split == "train" else None,
        "mask_dir": str(mask_dir),
        "num_frames": len(records),
        "num_frames_used": int(summary["num_frames_used"]),
        "num_valid_pixels": int(summary["num_valid_pixels"]),
        "normalizer_scope": str(args.normalizer_scope),
        "evaluation_mask": "gt_valid_pixels_and_render_alpha",
        "depth_source": (
            "da3_static_all_frames_dynamic_base_frame_masks"
            if args.eval_split == "novel"
            else "da3_static_all_frames_dynamic_same_frame_masks"
        ),
        "camera_source": "depth_anything_3/extrinsics_c2w.npy",
        "static_points": int(static_points.shape[0]),
        "static_frames": int(static_frames),
        "dynamic_frames_cached": int(len(dynamic_cache)),
        "mask_threshold": float(args.mask_threshold),
        "normalize_stat": args.normalize_stat,
        "gt_normalizer": float(summary["gt_normalizer"]),
        "est_normalizer": float(summary["est_normalizer"]),
        "l1": float(summary["l1"]),
        "l1_avg_frame": summary["l1_avg_frame"],
        "point_stride": int(args.point_stride),
        "dynamic_point_stride": int(args.dynamic_point_stride),
        "static_frame_stride": int(args.static_frame_stride),
        "static_max_points": int(args.static_max_points),
        "point_radius": int(args.point_radius),
        "alpha_threshold": float(args.alpha_threshold),
        "alpha_density_radius": int(alpha_density_radius),
        "alpha_density_scale": float(args.alpha_density_scale),
        "max_points": int(args.max_points),
        "debug_open3d_max_points": int(args.debug_open3d_max_points),
        "vis_out": str(Path(summary["vis_out"]).resolve()),
        "rendered_depth_dir": str(rendered_depth_dir),
        "depth_png_dir": str(depth_png_dir) if depth_png_dir is not None else None,
    }


def main() -> None:
    args = _parse_args()
    result = _evaluate(args)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    if args.json:
        print("RESULT_JSON " + json.dumps(result, sort_keys=True))
    elif args.output_json is None:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
