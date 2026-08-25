#!/usr/bin/env python3
"""Minimal Viser debug view for solver cameras, Ceres points, and GT depth lifts.

Edit the UPPERCASE constants below, then run:

    python Python/debug_gt_depth_lift_viser.py

The solver gauge is left unchanged.  GT cameras and GT-camera-lifted depth
points are transformed into the solver gauge with one Sim(3):

    X_solver ~= ALIGN_SCALE * ALIGN_R @ X_gt + ALIGN_T

That alignment scale is also used when reporting the camera/point losses.
"""

from __future__ import annotations

import colorsys
import os
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from utils.depth_sources import load_np_depth_like
from utils.readgt_colmap import load_colmap_gt
from utils.readgt_sintel import depth_read, load_sintel_gt
from utils.viser_vis import block_until_ctrl_c, get_viser_server


# ------------------------------ parameters ---------------------------------

ROOTPATH = ""          # "" uses utils.definitions.rootpath
DATASET_CAT = ""       # "" uses utils.definitions.dataset_cat
XM_RESULT = ""         # "" uses ROOTPATH/outputs/result_XM.npz
COLMAP_DIR = ""        # "" auto-detects ROOTPATH/colmap for Syn4D/Lab

IMAGE_DIR = ""         # "" uses utils.definitions.video_dir
DEPTH_SOURCE = "da3"   # "da3" / "depthanything" / "gt"
DEPTH_DIR = ""         # GT depth dir; "" uses utils.definitions.gt_depth_dir
DA3_DIR = ""           # "" uses ROOTPATH/outputs/depth_anything_3

FRAME_ID_OFFSET: int | str = "auto"  # "auto", 0, or 1
FRAME_STEP = 1
MAX_FRAMES = 0         # <= 0 means all selected frames

STRIDE = 4
DEPTH_MIN = 0.0
DEPTH_MAX = 0.0        # <= 0 disables max-depth filtering
MAX_POINTS_PER_FRAME = 8000
MAX_TOTAL_POINTS = 400000

FIT_DA_PER_FRAME_SCALE = True
PER_FRAME_SCALE_MIN = 0.25
PER_FRAME_SCALE_MAX = 4.0
PER_FRAME_SCALE_LEVELS = 5
PER_FRAME_SCALE_CANDIDATES = 15
PER_FRAME_SCALE_FIT_POINTS = 3000
PER_FRAME_SOLVER_FIT_POINTS = 8000
PER_FRAME_FORWARD_KEEP_PERCENTILE = 85.0
PER_FRAME_REVERSE_KEEP_PERCENTILE = 20.0
PER_FRAME_REVERSE_WEIGHT = 0.25
PER_FRAME_LOG_SCALE_REG = 0.0
SOLVER_FIT_KEEP_PERCENTILE = 98.0

CAMERA_SIZE = 0.0      # <= 0 auto
POINT_SIZE = 0.0       # <= 0 auto
VISER_HOST = os.environ.get("VISER_HOST", "0.0.0.0")
VISER_PORT = int(os.environ.get("VISER_PORT", "8080"))

PRINT_CHAMFER_LOSS = True
LOSS_SAMPLES = 20000
SEED = 0
DRY_RUN = False


# ------------------------------- loading ------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _paths() -> tuple[Path, str, Path, Path, Path, Path | None, Path]:
    import utils.definitions as definitions

    root = Path(ROOTPATH).expanduser().resolve() if ROOTPATH else Path(definitions.rootpath).resolve()
    dataset_cat = DATASET_CAT or str(definitions.dataset_cat)
    image_dir = (
        Path(IMAGE_DIR).expanduser().resolve()
        if IMAGE_DIR
        else Path(definitions.video_dir).resolve()
    )
    depth_dir = (
        Path(DEPTH_DIR).expanduser().resolve()
        if DEPTH_DIR
        else Path(definitions.gt_depth_dir).resolve()
    )
    da3_dir = (
        Path(DA3_DIR).expanduser().resolve()
        if DA3_DIR
        else root / "outputs" / "depth_anything_3"
    )
    colmap_dir = Path(COLMAP_DIR).expanduser().resolve() if COLMAP_DIR else None
    xm_result = Path(XM_RESULT).expanduser().resolve() if XM_RESULT else root / "outputs" / "result_XM.npz"
    if xm_result.is_dir():
        xm_result = xm_result / "result_XM.npz"
    return root, dataset_cat, image_dir, depth_dir, da3_dir, colmap_dir, xm_result


def _load_gt(root: Path, dataset_cat: str, colmap_dir: Path | None) -> dict[str, Any]:
    cat = dataset_cat.lower()
    if colmap_dir is not None:
        return load_colmap_gt(str(colmap_dir))
    if cat in {"syn4d", "lab"} or (root / "colmap" / "sparse" / "images.txt").is_file():
        return load_colmap_gt(str(root / "colmap"))
    if cat in {"sintel", "sintel_debug"}:
        return load_sintel_gt(str(root))
    raise ValueError(f"Unsupported DATASET_CAT={dataset_cat!r}")


def _load_solver(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing solver result: {path}")
    data = np.load(path, allow_pickle=True)
    for key in ("R", "t", "p"):
        if key not in data.files:
            raise KeyError(f"{path} is missing key {key!r}")

    R_flat = np.asarray(data["R"], dtype=np.float64)
    t_all = np.asarray(data["t"], dtype=np.float64)
    p = np.asarray(data["p"], dtype=np.float64)

    if R_flat.ndim != 2 or R_flat.shape[0] != 3 or R_flat.shape[1] % 3 != 0:
        raise ValueError(f"Expected solver R shape (3, 3T), got {R_flat.shape}")
    if t_all.ndim != 2 or t_all.shape[0] != 3:
        raise ValueError(f"Expected solver t shape (3, T), got {t_all.shape}")

    n = min(R_flat.shape[1] // 3, t_all.shape[1])
    R_c2w = [R_flat[:, 3 * i : 3 * (i + 1)] for i in range(n)]
    centers = t_all[:, :n].T.astype(np.float64)

    if p.ndim == 2 and p.shape[0] == 3:
        p = p.T
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"Expected solver p shape (3, N) or (N, 3), got {p.shape}")
    p = p[np.isfinite(p).all(axis=1)]

    print(f"[solver] {n} cameras, {p.shape[0]} Ceres points from {path}")
    return {"R_c2w": R_c2w, "centers": centers, "p": p.astype(np.float32), "path": path}


def _trailing_frame_id(name: str) -> int | None:
    match = re.search(r"(?<!\d)(\d+)$", Path(name).stem)
    return int(match.group(1)) if match else None


def _read_depth(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".dpt":
        depth = depth_read(str(path))
    else:
        depth = load_np_depth_like(path)
    depth = np.squeeze(np.asarray(depth, dtype=np.float32))
    if depth.ndim != 2:
        raise ValueError(f"Expected depth map shape (H, W), got {depth.shape} from {path}")
    return depth


def _image_maps(image_dir: Path) -> tuple[dict[str, Path], dict[int, Path], list[Path]]:
    by_name: dict[str, Path] = {}
    by_frame: dict[int, Path] = {}
    if not image_dir.is_dir():
        return by_name, by_frame, []
    ordered = sorted(
        [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: (_trailing_frame_id(p.name) if _trailing_frame_id(p.name) is not None else 10**12, p.name),
    )
    for path in ordered:
        by_name[path.name] = path
        frame_id = _trailing_frame_id(path.name)
        if frame_id is not None and frame_id not in by_frame:
            by_frame[frame_id] = path
    return by_name, by_frame, ordered


def _image_for_frame(
    root: Path,
    image_dir: Path,
    by_name: dict[str, Path],
    by_frame: dict[int, Path],
    ordered: list[Path],
    image_name: str,
    frame_id: int,
    order_idx: int,
    view_idx: int,
) -> Path | None:
    for path in (image_dir / image_name, image_dir / Path(image_name).name, root / image_name):
        if path.is_file():
            return path
    if image_name in by_name:
        return by_name[image_name]
    basename = Path(image_name).name
    if basename in by_name:
        return by_name[basename]
    if frame_id in by_frame:
        return by_frame[frame_id]
    if frame_id + 1 in by_frame:
        return by_frame[frame_id + 1]
    if 0 <= order_idx < len(ordered):
        return ordered[order_idx]
    if 0 <= view_idx < len(ordered):
        return ordered[view_idx]
    return None


def _read_rgb(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"[image] could not read {path}; using fallback colors")
        return None
    return image[:, :, ::-1].copy()


def _load_depth_source(root: Path, depth_dir: Path, da3_dir: Path) -> dict[str, Any]:
    source = DEPTH_SOURCE.strip().lower()
    if source in {"da3", "depth_anything", "depthanything", "depth_anything_3"}:
        path = da3_dir / "depth_maps.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Missing DepthAnything maps: {path}")
        stack = np.asarray(np.load(path), dtype=np.float32)
        if stack.ndim != 3:
            raise ValueError(f"Expected DepthAnything shape (T, H, W), got {stack.shape}")
        intrinsics = None
        k_path = da3_dir / "intrinsics.npy"
        if k_path.is_file():
            intrinsics = np.asarray(np.load(k_path), dtype=np.float64)
            if intrinsics.ndim == 2:
                intrinsics = intrinsics[None, :, :]
            if intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
                raise ValueError(f"Expected intrinsics shape (T, 3, 3), got {intrinsics.shape}")
        print(f"[depth] DepthAnything stack {stack.shape} from {path}")
        return {"stack": stack, "intrinsics": intrinsics, "files": {}, "ordered": [], "desc": str(path)}

    if source != "gt":
        raise ValueError("DEPTH_SOURCE must be 'da3'/'depthanything' or 'gt'")

    for path in (
        depth_dir / "depths.npy",
        depth_dir / "depth_maps.npy",
        depth_dir / "depths.npz",
        depth_dir / "depth_maps.npz",
        root / "depths.npy",
        root / "depth_maps.npy",
    ):
        if path.is_file():
            stack = np.asarray(load_np_depth_like(path), dtype=np.float32)
            if stack.ndim == 2:
                stack = stack[None, :, :]
            if stack.ndim != 3:
                raise ValueError(f"Expected GT depth stack (T, H, W), got {stack.shape} from {path}")
            print(f"[depth] GT depth stack {stack.shape} from {path}")
            return {"stack": stack, "intrinsics": None, "files": {}, "ordered": [], "desc": str(path)}

    exts = {".dpt", ".npy", ".npz", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    rank = {".npy": 0, ".npz": 1, ".dpt": 2, ".png": 3, ".tif": 4, ".tiff": 5, ".jpg": 6, ".jpeg": 7}
    files_by_frame: dict[int, Path] = {}
    ordered = sorted(
        [p for p in depth_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts],
        key=lambda p: (_trailing_frame_id(p.name) if _trailing_frame_id(p.name) is not None else 10**12, p.name),
    )
    for path in ordered:
        frame_id = _trailing_frame_id(path.name)
        if frame_id is None:
            continue
        prev = files_by_frame.get(frame_id)
        if prev is None or rank[path.suffix.lower()] < rank[prev.suffix.lower()]:
            files_by_frame[frame_id] = path
    if not ordered:
        raise FileNotFoundError(f"No GT depth maps found in {depth_dir}")
    print(f"[depth] {len(ordered)} GT depth files from {depth_dir}")
    return {"stack": None, "intrinsics": None, "files": files_by_frame, "ordered": ordered, "desc": str(depth_dir)}


# ------------------------------- geometry -----------------------------------

def _c2w_from_gt(info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    R_w2c = np.asarray(info["R"], dtype=np.float64).reshape(3, 3)
    t_w2c = np.asarray(info["t"], dtype=np.float64).reshape(3)
    R_c2w = R_w2c.T
    center = -R_c2w @ t_w2c
    return R_c2w, center


def _index_for_frame(frame_id: int, order_idx: int, count: int, all_frame_ids: list[int]) -> int | None:
    if count <= 0:
        return None
    if FRAME_ID_OFFSET != "auto":
        idx = int(frame_id) - int(FRAME_ID_OFFSET)
    else:
        ids = np.asarray(all_frame_ids, dtype=np.int64)
        if ids.size and int(ids.min()) == 0 and int(ids.max()) < count:
            idx = int(frame_id)
        elif ids.size and int(ids.min()) == 1 and int(ids.max()) <= count:
            idx = int(frame_id) - 1
        elif 0 <= int(frame_id) < count:
            idx = int(frame_id)
        else:
            idx = int(order_idx)
    return idx if 0 <= idx < count else None


def _sim3_align(src_gt: np.ndarray, dst_solver: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Return scale, R, t for dst_solver ~= scale * R @ src_gt + t."""
    src = np.asarray(src_gt, dtype=np.float64).reshape(3, -1)
    dst = np.asarray(dst_solver, dtype=np.float64).reshape(3, -1)
    keep = np.isfinite(src).all(axis=0) & np.isfinite(dst).all(axis=0)
    src = src[:, keep]
    dst = dst[:, keep]
    if src.shape[1] < 3:
        raise RuntimeError(f"Need at least 3 matched camera centers, got {src.shape[1]}")

    src_mu = src.mean(axis=1, keepdims=True)
    dst_mu = dst.mean(axis=1, keepdims=True)
    src_c = src - src_mu
    dst_c = dst - dst_mu

    H = dst_c @ src_c.T
    U, singular_values, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        singular_values[-1] *= -1.0
        R = U @ Vt

    denom = float(np.sum(src_c * src_c))
    scale = float(np.sum(singular_values) / denom) if denom > 1e-12 else 1.0
    t = dst_mu - scale * R @ src_mu

    pred = scale * R @ src + t
    rmse = float(np.sqrt(np.mean(np.sum((pred - dst) ** 2, axis=0))))
    print(f"[align] GT -> solver: scale={scale:.9g}, camera_rmse={rmse:.6g}, pairs={src.shape[1]}")
    return scale, R.astype(np.float64), t.reshape(3, 1).astype(np.float64), rmse


def _apply_sim3(points: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return (scale * (R @ pts.T) + t).T.astype(np.float32)


def _sample_rgb(
    rgb: np.ndarray | None,
    xx: np.ndarray,
    yy: np.ndarray,
    depth_shape: tuple[int, int],
    fallback: np.ndarray,
) -> np.ndarray:
    if rgb is None:
        return np.tile(fallback.reshape(1, 3), (xx.shape[0], 1)).astype(np.uint8)

    h_depth, w_depth = depth_shape
    h_img, w_img = rgb.shape[:2]
    if (h_img, w_img) == (h_depth, w_depth):
        x_img = xx.astype(np.int64)
        y_img = yy.astype(np.int64)
    else:
        x_img = np.rint(xx * (w_img - 1) / max(w_depth - 1, 1)).astype(np.int64)
        y_img = np.rint(yy * (h_img - 1) / max(h_depth - 1, 1)).astype(np.int64)
    x_img = np.clip(x_img, 0, w_img - 1)
    y_img = np.clip(y_img, 0, h_img - 1)
    return rgb[y_img, x_img].astype(np.uint8)


def _lift_depth(
    depth: np.ndarray,
    K: np.ndarray,
    R_c2w: np.ndarray,
    center: np.ndarray,
    order_idx: int,
    rgb: np.ndarray | None,
    fallback_color: np.ndarray,
    depth_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth, dtype=np.float32) * float(depth_scale)
    h, w = depth.shape
    stride = max(1, int(STRIDE))
    yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    z = depth[yy, xx].astype(np.float64)

    valid = np.isfinite(z) & (z > float(DEPTH_MIN))
    if float(DEPTH_MAX) > 0:
        valid &= z <= float(DEPTH_MAX)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    xx = xx[valid].astype(np.float64)
    yy = yy[valid].astype(np.float64)
    z = z[valid]
    colors = _sample_rgb(rgb, xx, yy, (h, w), fallback_color)

    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    x = (xx - cx) / fx * z
    y = (yy - cy) / fy * z
    pts_cam = np.stack([x, y, z], axis=1)
    pts_world = (R_c2w @ pts_cam.T).T + center.reshape(1, 3)

    if MAX_POINTS_PER_FRAME > 0 and pts_world.shape[0] > MAX_POINTS_PER_FRAME:
        rng = np.random.default_rng(SEED + int(order_idx))
        pick = rng.choice(pts_world.shape[0], MAX_POINTS_PER_FRAME, replace=False)
        pts_world = pts_world[pick]
        colors = colors[pick]
    return pts_world.astype(np.float32), colors.astype(np.uint8)


def _frustum_lines(K: np.ndarray, hw: tuple[int, int], R_c2w: np.ndarray, center: np.ndarray, size: float) -> np.ndarray:
    h, w = hw
    corners = np.asarray([[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]])
    z = float(size)
    x = (corners[:, 0] - float(K[0, 2])) / float(K[0, 0]) * z
    y = (corners[:, 1] - float(K[1, 2])) / float(K[1, 1]) * z
    corners_cam = np.stack([x, y, np.full(4, z)], axis=1)
    corners_world = (R_c2w @ corners_cam.T).T + center.reshape(1, 3)
    return np.asarray(
        [[center, corners_world[i]] for i in range(4)]
        + [[corners_world[i], corners_world[(i + 1) % 4]] for i in range(4)],
        dtype=np.float32,
    )


def _hw_from_K(K: np.ndarray) -> tuple[int, int]:
    h = max(2, int(round(float(K[1, 2]) * 2.0 + 1.0)))
    w = max(2, int(round(float(K[0, 2]) * 2.0 + 1.0)))
    return h, w


def _trajectory(centers: np.ndarray) -> np.ndarray:
    if centers.shape[0] < 2:
        return np.zeros((0, 2, 3), dtype=np.float32)
    return np.stack([centers[:-1], centers[1:]], axis=1).astype(np.float32)


def _downsample(points: np.ndarray, colors: np.ndarray | None, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray | None]:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points, colors
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], max_points, replace=False)
    return points[idx], (None if colors is None else colors[idx])


def _is_depthanything_source() -> bool:
    return DEPTH_SOURCE.strip().lower() in {"da3", "depth_anything", "depthanything", "depth_anything_3"}


def _filter_point_outliers(points: np.ndarray, keep_percentile: float) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] == 0 or keep_percentile >= 100.0:
        return pts
    center = np.median(pts.astype(np.float64), axis=0)
    radius = np.linalg.norm(pts.astype(np.float64) - center.reshape(1, 3), axis=1)
    threshold = np.percentile(radius, np.clip(float(keep_percentile), 1.0, 100.0))
    return pts[radius <= threshold]


def _camera_depth_samples(depth: np.ndarray, K: np.ndarray, order_idx: int, max_points: int) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    h, w = depth.shape
    yy, xx = np.mgrid[0:h:max(1, int(STRIDE)), 0:w:max(1, int(STRIDE))]
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    z = depth[yy, xx].astype(np.float64)
    valid = np.isfinite(z) & (z > float(DEPTH_MIN))
    if float(DEPTH_MAX) > 0:
        valid &= z <= float(DEPTH_MAX)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)

    xx = xx[valid].astype(np.float64)
    yy = yy[valid].astype(np.float64)
    z = z[valid]
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    pts_cam = np.stack([(xx - cx) / fx * z, (yy - cy) / fy * z, z], axis=1)

    if max_points > 0 and pts_cam.shape[0] > max_points:
        rng = np.random.default_rng(SEED + 1000 + int(order_idx))
        pts_cam = pts_cam[rng.choice(pts_cam.shape[0], max_points, replace=False)]
    return pts_cam.astype(np.float32)


def _trimmed_mean_square(distances: np.ndarray, keep_percentile: float) -> float:
    d = np.asarray(distances, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("inf")
    keep_percentile = float(np.clip(keep_percentile, 1.0, 100.0))
    if keep_percentile < 100.0 and d.size > 4:
        threshold = np.percentile(d, keep_percentile)
        d = d[d <= threshold]
    if d.size == 0:
        return float("inf")
    return float(np.mean(d * d))


def _hierarchical_scale_search(loss_fn) -> tuple[float, float]:
    min_scale = max(float(PER_FRAME_SCALE_MIN), 1e-8)
    max_scale = max(float(PER_FRAME_SCALE_MAX), min_scale * (1.0 + 1e-8))
    lo0 = float(np.log(min_scale))
    hi0 = float(np.log(max_scale))
    lo = lo0
    hi = hi0
    candidates = max(3, int(PER_FRAME_SCALE_CANDIDATES))
    levels = max(1, int(PER_FRAME_SCALE_LEVELS))
    best_log_s = 0.0
    best_loss = float("inf")

    for _level in range(levels):
        grid = np.linspace(lo, hi, candidates)
        losses = np.asarray([loss_fn(float(np.exp(x))) for x in grid], dtype=np.float64)
        finite = np.isfinite(losses)
        if not np.any(finite):
            break

        local_idx = int(np.argmin(np.where(finite, losses, np.inf)))
        if float(losses[local_idx]) < best_loss:
            best_loss = float(losses[local_idx])
            best_log_s = float(grid[local_idx])

        step = float((hi - lo) / max(candidates - 1, 1))
        lo = max(lo0, best_log_s - 2.0 * step)
        hi = min(hi0, best_log_s + 2.0 * step)
        if hi <= lo:
            break
    return float(np.exp(best_log_s)), best_loss


def _estimate_frame_depth_scale(
    *,
    depth: np.ndarray,
    K: np.ndarray,
    R_gt: np.ndarray,
    C_gt: np.ndarray,
    align_scale: float,
    align_R: np.ndarray,
    align_t: np.ndarray,
    solver_fit_points: np.ndarray,
    solver_tree: Any,
    order_idx: int,
) -> tuple[float, float]:
    if not FIT_DA_PER_FRAME_SCALE or not _is_depthanything_source():
        return 1.0, float("nan")
    if solver_fit_points.shape[0] < 10 or solver_tree is None:
        return 1.0, float("nan")

    try:
        from scipy.spatial import cKDTree
    except Exception as exc:
        print(f"[scale] skipped per-frame scale; scipy unavailable: {exc}")
        return 1.0, float("nan")

    pts_cam = _camera_depth_samples(depth, K, order_idx, PER_FRAME_SCALE_FIT_POINTS)
    if pts_cam.shape[0] < 10:
        return 1.0, float("nan")

    aligned_center = (float(align_scale) * (align_R @ C_gt.reshape(3, 1)) + align_t).reshape(1, 3)
    aligned_dirs = (float(align_scale) * (align_R @ R_gt @ pts_cam.T)).T

    def loss_at(scale: float) -> float:
        src = (aligned_center + float(scale) * aligned_dirs).astype(np.float32)
        fwd, _ = solver_tree.query(src, workers=-1)
        loss = _trimmed_mean_square(fwd, PER_FRAME_FORWARD_KEEP_PERCENTILE)
        if PER_FRAME_REVERSE_WEIGHT > 0.0:
            src_tree = cKDTree(src)
            rev, _ = src_tree.query(solver_fit_points, workers=-1)
            loss += float(PER_FRAME_REVERSE_WEIGHT) * _trimmed_mean_square(
                rev, PER_FRAME_REVERSE_KEEP_PERCENTILE
            )
        if PER_FRAME_LOG_SCALE_REG > 0.0:
            loss += float(PER_FRAME_LOG_SCALE_REG) * float(np.log(max(scale, 1e-8)) ** 2)
        return float(loss)

    return _hierarchical_scale_search(loss_at)


def _frame_color(i: int) -> np.ndarray:
    rgb = colorsys.hsv_to_rgb((i * 0.61803398875) % 1.0, 0.70, 1.0)
    return np.asarray(np.round(np.asarray(rgb) * 255.0), dtype=np.uint8)


def _bbox_diag(arrays: list[np.ndarray]) -> float:
    pts = [np.asarray(a).reshape(-1, 3) for a in arrays if a is not None and np.asarray(a).size > 0]
    if not pts:
        return 1.0
    all_pts = np.concatenate(pts, axis=0)
    all_pts = all_pts[np.isfinite(all_pts).all(axis=1)]
    if all_pts.shape[0] == 0:
        return 1.0
    return float(np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0)))


def _chamfer_loss(a: np.ndarray, b: np.ndarray) -> None:
    if not PRINT_CHAMFER_LOSS or a.shape[0] == 0 or b.shape[0] == 0:
        return
    try:
        from scipy.spatial import cKDTree
    except Exception as exc:
        print(f"[loss] skipped Chamfer loss; scipy unavailable: {exc}")
        return

    a_s, _ = _downsample(a.astype(np.float32), None, LOSS_SAMPLES, SEED)
    b_s, _ = _downsample(b.astype(np.float32), None, LOSS_SAMPLES, SEED + 1)
    da, _ = cKDTree(b_s).query(a_s, workers=-1)
    db, _ = cKDTree(a_s).query(b_s, workers=-1)
    loss = float(np.mean(da**2) + np.mean(db**2))
    rmse = float(np.sqrt(0.5 * loss))
    print(f"[loss] chamfer={loss:.6g}, rmse={rmse:.6g} using GT points after scaled GT->solver alignment")


# ------------------------------- prepare ------------------------------------

def _depth_for_frame(depth_data: dict[str, Any], frame_id: int, order_idx: int, view_idx: int, frame_ids: list[int], K_gt: np.ndarray) -> tuple[np.ndarray | None, np.ndarray, str]:
    stack = depth_data["stack"]
    if stack is not None:
        depth_idx = _index_for_frame(frame_id, order_idx, stack.shape[0], frame_ids)
        if depth_idx is None:
            return None, K_gt, "missing stack index"
        K = K_gt
        intrinsics = depth_data["intrinsics"]
        if intrinsics is not None:
            K = np.asarray(intrinsics[min(depth_idx, intrinsics.shape[0] - 1)], dtype=np.float64).reshape(3, 3)
        return np.squeeze(stack[depth_idx]), K, f"{depth_data['desc']}[{depth_idx}]"

    files = depth_data["files"]
    ordered = depth_data["ordered"]
    path = files.get(frame_id) or files.get(frame_id + 1)
    if path is None and 0 <= order_idx < len(ordered):
        path = ordered[order_idx]
    if path is None and 0 <= view_idx < len(ordered):
        path = ordered[view_idx]
    if path is None:
        return None, K_gt, "missing file"
    return _read_depth(path), K_gt, str(path)


def _prepare() -> dict[str, Any]:
    root, dataset_cat, image_dir, depth_dir, da3_dir, colmap_dir, xm_result = _paths()
    gt = _load_gt(root, dataset_cat, colmap_dir)
    solver = _load_solver(xm_result)
    depth_data = _load_depth_source(root, depth_dir, da3_dir)
    image_by_name, image_by_frame, ordered_images = _image_maps(image_dir)
    print(f"[image] {len(ordered_images)} images from {image_dir}")

    all_items = sorted(gt.items(), key=lambda kv: int(kv[1]["id"]))
    all_frame_ids = [int(info["id"]) for _name, info in all_items]
    selected = list(enumerate(all_items))
    selected = selected[:: max(1, int(FRAME_STEP))]
    if MAX_FRAMES > 0:
        selected = selected[: int(MAX_FRAMES)]
    if not selected:
        raise RuntimeError("No GT frames selected")

    src_gt_centers = []
    dst_solver_centers = []
    for order_idx, (_name, info) in enumerate(all_items):
        frame_id = int(info["id"])
        solver_idx = _index_for_frame(frame_id, order_idx, solver["centers"].shape[0], all_frame_ids)
        if solver_idx is None:
            continue
        _R_gt, C_gt = _c2w_from_gt(info)
        src_gt_centers.append(C_gt)
        dst_solver_centers.append(solver["centers"][solver_idx])

    align_scale, align_R, align_t, align_rmse = _sim3_align(
        np.stack(src_gt_centers, axis=1),
        np.stack(dst_solver_centers, axis=1),
    )

    solver_fit_points = _filter_point_outliers(solver["p"], SOLVER_FIT_KEEP_PERCENTILE)
    solver_fit_points, _ = _downsample(solver_fit_points, None, PER_FRAME_SOLVER_FIT_POINTS, SEED + 17)
    solver_tree = None
    if FIT_DA_PER_FRAME_SCALE and _is_depthanything_source() and solver_fit_points.shape[0] >= 10:
        try:
            from scipy.spatial import cKDTree

            solver_tree = cKDTree(solver_fit_points)
            print(
                f"[scale] per-frame DA scale search enabled; target={solver_fit_points.shape[0]} "
                f"Ceres points, range=[{PER_FRAME_SCALE_MIN}, {PER_FRAME_SCALE_MAX}]"
            )
        except Exception as exc:
            print(f"[scale] per-frame DA scale search disabled; scipy unavailable: {exc}")

    gt_R_aligned: list[np.ndarray] = []
    gt_C_aligned: list[np.ndarray] = []
    gt_K: list[np.ndarray] = []
    gt_hw: list[tuple[int, int]] = []
    lifted_parts: list[np.ndarray] = []
    lifted_colors: list[np.ndarray] = []
    depth_descs: list[str] = []
    frames: list[dict[str, Any]] = []
    frame_scales: list[float] = []
    frame_scale_losses: list[float] = []

    for view_idx, (order_idx, (name, info)) in enumerate(selected):
        frame_id = int(info["id"])
        K_gt = np.asarray(info["K"], dtype=np.float64).reshape(3, 3)
        R_gt, C_gt = _c2w_from_gt(info)
        depth, K_depth, desc = _depth_for_frame(depth_data, frame_id, order_idx, view_idx, all_frame_ids, K_gt)
        image_path = _image_for_frame(
            root=root,
            image_dir=image_dir,
            by_name=image_by_name,
            by_frame=image_by_frame,
            ordered=ordered_images,
            image_name=name,
            frame_id=frame_id,
            order_idx=order_idx,
            view_idx=view_idx,
        )
        rgb = _read_rgb(image_path)

        gt_R_aligned.append(align_R @ R_gt)
        gt_C_aligned.append((align_scale * align_R @ C_gt.reshape(3, 1) + align_t).reshape(3))
        gt_K.append(K_depth)
        gt_hw.append(tuple(depth.shape) if depth is not None else _hw_from_K(K_depth))
        gt_idx = len(gt_R_aligned) - 1
        solver_idx = _index_for_frame(frame_id, order_idx, solver["centers"].shape[0], all_frame_ids)

        if depth is None:
            print(f"[depth] skipped frame={frame_id} {name}: {desc}")
            frames.append(
                {
                    "frame_id": frame_id,
                    "name": name,
                    "gt_idx": gt_idx,
                    "solver_idx": solver_idx,
                    "points": np.zeros((0, 3), dtype=np.float32),
                    "colors": np.zeros((0, 3), dtype=np.uint8),
                    "depth_desc": desc,
                    "image_path": image_path,
                    "depth_scale": float("nan"),
                    "depth_scale_loss": float("nan"),
                }
            )
            continue

        fallback_color = _frame_color(view_idx)
        depth_scale, depth_scale_loss = _estimate_frame_depth_scale(
            depth=depth,
            K=K_depth,
            R_gt=R_gt,
            C_gt=C_gt,
            align_scale=align_scale,
            align_R=align_R,
            align_t=align_t,
            solver_fit_points=solver_fit_points,
            solver_tree=solver_tree,
            order_idx=order_idx,
        )
        frame_scales.append(depth_scale)
        frame_scale_losses.append(depth_scale_loss)
        print(f"[scale] frame={frame_id} depth_scale={depth_scale:.6g} loss={depth_scale_loss:.6g}")

        lifted_gt, colors = _lift_depth(
            depth,
            K_depth,
            R_gt,
            C_gt,
            order_idx,
            rgb,
            fallback_color,
            depth_scale=depth_scale,
        )
        lifted_solver = _apply_sim3(lifted_gt, align_scale, align_R, align_t)
        lifted_parts.append(lifted_solver)
        lifted_colors.append(colors)
        depth_descs.append(desc)
        frames.append(
            {
                "frame_id": frame_id,
                "name": name,
                "gt_idx": gt_idx,
                "solver_idx": solver_idx,
                "points": lifted_solver,
                "colors": colors,
                "depth_desc": desc,
                "image_path": image_path,
                "depth_scale": depth_scale,
                "depth_scale_loss": depth_scale_loss,
            }
        )

    lifted_points = (
        np.concatenate(lifted_parts, axis=0).astype(np.float32)
        if lifted_parts
        else np.zeros((0, 3), dtype=np.float32)
    )
    lifted_color_arr = (
        np.concatenate(lifted_colors, axis=0).astype(np.uint8)
        if lifted_colors
        else np.zeros((0, 3), dtype=np.uint8)
    )
    lifted_points, lifted_color_arr = _downsample(lifted_points, lifted_color_arr, MAX_TOTAL_POINTS, SEED)
    solver_points, _ = _downsample(solver["p"], None, MAX_TOTAL_POINTS, SEED + 2)

    _chamfer_loss(lifted_points, solver_points)
    print(
        f"[ready] lifted={lifted_points.shape[0]} points from {len(lifted_parts)} frames, "
        f"solver_p={solver_points.shape[0]}, depth_source={DEPTH_SOURCE}"
    )
    finite_scales = np.asarray(frame_scales, dtype=np.float64)
    finite_scales = finite_scales[np.isfinite(finite_scales)]
    if finite_scales.size > 0 and FIT_DA_PER_FRAME_SCALE and _is_depthanything_source():
        print(
            f"[scale] per-frame depth scale summary: "
            f"min={finite_scales.min():.6g}, median={np.median(finite_scales):.6g}, "
            f"max={finite_scales.max():.6g}"
        )

    return {
        "solver": solver,
        "solver_points": solver_points,
        "gt_R_aligned": gt_R_aligned,
        "gt_C_aligned": np.asarray(gt_C_aligned, dtype=np.float32),
        "gt_K": gt_K,
        "gt_hw": gt_hw,
        "lifted_points": lifted_points,
        "lifted_colors": lifted_color_arr,
        "align": (align_scale, align_R, align_t, align_rmse),
        "depth_descs": depth_descs,
        "frames": frames,
        "frame_scales": np.asarray(frame_scales, dtype=np.float64),
        "frame_scale_losses": np.asarray(frame_scale_losses, dtype=np.float64),
    }


# -------------------------------- viewer -------------------------------------

def _serve(data: dict[str, Any]) -> None:
    solver = data["solver"]
    solver_centers = solver["centers"].astype(np.float32)
    solver_points = data["solver_points"]
    gt_centers = data["gt_C_aligned"]
    lifted_points = data["lifted_points"]
    frames = data["frames"]

    diag = _bbox_diag([solver_points, solver_centers, gt_centers, lifted_points])
    point_size = float(POINT_SIZE) if POINT_SIZE > 0 else float(np.clip(0.002 * diag, 0.001, 0.03))
    frustum_size = float(CAMERA_SIZE) if CAMERA_SIZE > 0 else float(np.clip(0.04 * diag, 0.05, 0.5))

    K_ref = data["gt_K"][0] if data["gt_K"] else np.asarray([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    hw_ref = data["gt_hw"][0] if data["gt_hw"] else (480, 640)

    solver_frustums = [
        _frustum_lines(K_ref, hw_ref, solver["R_c2w"][i], solver_centers[i], frustum_size)
        for i in range(len(solver["R_c2w"]))
    ]
    gt_frustums = [
        _frustum_lines(data["gt_K"][i], data["gt_hw"][i], data["gt_R_aligned"][i], gt_centers[i], frustum_size)
        for i in range(len(data["gt_R_aligned"]))
    ]

    server = get_viser_server(host=VISER_HOST, port=VISER_PORT)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    server.gui.add_markdown("### Minimal GT Depth Lift")

    frame_slider = server.gui.add_slider("Frame", min=0, max=max(0, len(frames) - 1), step=1, initial_value=0)
    btn_prev = server.gui.add_button("Prev")
    btn_next = server.gui.add_button("Next")
    show_solver_cameras = server.gui.add_checkbox("Solver cameras", initial_value=True)
    show_gt_cameras = server.gui.add_checkbox("Aligned GT cameras", initial_value=True)
    show_solver_points = server.gui.add_checkbox("Solver p", initial_value=True)
    show_lifted_points = server.gui.add_checkbox("Lifted RGB points", initial_value=True)
    show_all_lifted = server.gui.add_checkbox("All lifted frames", initial_value=False)

    handles: dict[str, Any] = {}

    def _toggle(path: str, visible: bool) -> None:
        handle = handles.get(path)
        if handle is not None:
            try:
                handle.visible = bool(visible)
                return
            except Exception:
                pass
        try:
            server.scene.remove(path)
        except Exception:
            pass

    if solver_points.shape[0] > 0:
        handles["/solver/p_from_ceres"] = server.scene.add_point_cloud(
            "/solver/p_from_ceres",
            points=solver_points,
            colors=np.tile(np.array([30, 210, 90], dtype=np.uint8), (solver_points.shape[0], 1)),
            point_size=point_size,
            point_shape="circle",
            visible=bool(show_solver_points.value),
        )
    if lifted_points.shape[0] > 0:
        handles["/gt_aligned/lifted_depth_all"] = server.scene.add_point_cloud(
            "/gt_aligned/lifted_depth_all",
            points=lifted_points,
            colors=data["lifted_colors"],
            point_size=point_size,
            point_shape="circle",
            visible=bool(show_lifted_points.value) and bool(show_all_lifted.value),
        )
    if solver_frustums:
        handles["/solver/cameras"] = server.scene.add_line_segments(
            "/solver/cameras",
            points=np.concatenate(solver_frustums, axis=0),
            colors=np.array([255, 110, 30], dtype=np.uint8),
            line_width=1.5,
            visible=bool(show_solver_cameras.value),
        )
    if gt_frustums:
        handles["/gt_aligned/cameras"] = server.scene.add_line_segments(
            "/gt_aligned/cameras",
            points=np.concatenate(gt_frustums, axis=0),
            colors=np.array([40, 140, 255], dtype=np.uint8),
            line_width=1.5,
            visible=bool(show_gt_cameras.value),
        )

    solver_traj = _trajectory(solver_centers)
    gt_traj = _trajectory(gt_centers)
    if solver_traj.shape[0] > 0:
        handles["/solver/trajectory"] = server.scene.add_line_segments(
            "/solver/trajectory",
            points=solver_traj,
            colors=np.array([210, 70, 10], dtype=np.uint8),
            line_width=3.0,
            visible=bool(show_solver_cameras.value),
        )
    if gt_traj.shape[0] > 0:
        handles["/gt_aligned/trajectory"] = server.scene.add_line_segments(
            "/gt_aligned/trajectory",
            points=gt_traj,
            colors=np.array([20, 90, 220], dtype=np.uint8),
            line_width=3.0,
            visible=bool(show_gt_cameras.value),
        )

    dynamic_paths = [
        "/solver/current_camera",
        "/gt_aligned/current_camera",
        "/gt_aligned/current_lifted_depth",
        "/gt_aligned/current_pair_line",
    ]

    def _clear_dynamic() -> None:
        for path in dynamic_paths:
            try:
                server.scene.remove(path)
            except Exception:
                pass

    state = {"idx": 0}

    def _refresh(idx: int) -> None:
        if not frames:
            return
        idx = int(idx) % len(frames)
        state["idx"] = idx
        rec = frames[idx]
        _clear_dynamic()

        gt_idx = int(rec["gt_idx"])
        solver_idx = rec["solver_idx"]

        if (
            bool(show_lifted_points.value)
            and not bool(show_all_lifted.value)
            and rec["points"].shape[0] > 0
        ):
            server.scene.add_point_cloud(
                "/gt_aligned/current_lifted_depth",
                points=rec["points"].astype(np.float32),
                colors=rec["colors"].astype(np.uint8),
                point_size=max(point_size * 1.25, 0.003),
                point_shape="circle",
                visible=True,
            )

        if bool(show_gt_cameras.value) and gt_idx < len(data["gt_R_aligned"]):
            server.scene.add_line_segments(
                "/gt_aligned/current_camera",
                points=_frustum_lines(
                    data["gt_K"][gt_idx],
                    data["gt_hw"][gt_idx],
                    data["gt_R_aligned"][gt_idx],
                    gt_centers[gt_idx],
                    frustum_size * 1.4,
                ),
                colors=np.array([80, 200, 255], dtype=np.uint8),
                line_width=3.0,
                visible=True,
            )

        if bool(show_solver_cameras.value) and solver_idx is not None and solver_idx < len(solver["R_c2w"]):
            server.scene.add_line_segments(
                "/solver/current_camera",
                points=_frustum_lines(K_ref, hw_ref, solver["R_c2w"][solver_idx], solver_centers[solver_idx], frustum_size * 1.4),
                colors=np.array([255, 55, 0], dtype=np.uint8),
                line_width=3.0,
                visible=True,
            )

        if (
            bool(show_gt_cameras.value)
            and bool(show_solver_cameras.value)
            and solver_idx is not None
            and solver_idx < solver_centers.shape[0]
            and gt_idx < gt_centers.shape[0]
        ):
            server.scene.add_line_segments(
                "/gt_aligned/current_pair_line",
                points=np.asarray([[gt_centers[gt_idx], solver_centers[solver_idx]]], dtype=np.float32),
                colors=np.array([60, 60, 60], dtype=np.uint8),
                line_width=1.5,
                visible=True,
            )

        cam_err = float("nan")
        if solver_idx is not None and solver_idx < solver_centers.shape[0] and gt_idx < gt_centers.shape[0]:
            cam_err = float(np.linalg.norm(gt_centers[gt_idx] - solver_centers[solver_idx]))
        print(
            f"[VIS] slider={idx} frame={rec['frame_id']} solver_idx={solver_idx} "
            f"points={rec['points'].shape[0]} depth_scale={rec['depth_scale']:.6g} "
            f"scale_loss={rec['depth_scale_loss']:.6g} cam_err={cam_err:.6g} "
            f"image={rec['image_path']} depth={rec['depth_desc']}"
        )

    def _set_frame(idx: int) -> None:
        if not frames:
            return
        idx = int(idx) % len(frames)
        if int(frame_slider.value) != idx:
            frame_slider.value = idx
        _refresh(idx)

    @frame_slider.on_update
    def _(_event):
        _refresh(int(frame_slider.value))

    @btn_next.on_click
    def _(_event):
        _set_frame(state["idx"] + 1)

    @btn_prev.on_click
    def _(_event):
        _set_frame(state["idx"] - 1)

    @show_solver_cameras.on_update
    def _(_event):
        _toggle("/solver/cameras", bool(show_solver_cameras.value))
        _toggle("/solver/trajectory", bool(show_solver_cameras.value))
        _refresh(state["idx"])

    @show_gt_cameras.on_update
    def _(_event):
        _toggle("/gt_aligned/cameras", bool(show_gt_cameras.value))
        _toggle("/gt_aligned/trajectory", bool(show_gt_cameras.value))
        _refresh(state["idx"])

    @show_solver_points.on_update
    def _(_event):
        _toggle("/solver/p_from_ceres", bool(show_solver_points.value))

    @show_lifted_points.on_update
    def _(_event):
        _toggle("/gt_aligned/lifted_depth_all", bool(show_lifted_points.value) and bool(show_all_lifted.value))
        _refresh(state["idx"])

    @show_all_lifted.on_update
    def _(_event):
        _toggle("/gt_aligned/lifted_depth_all", bool(show_lifted_points.value) and bool(show_all_lifted.value))
        _refresh(state["idx"])

    scale, _R, _t, rmse = data["align"]
    print(f"[viser] http://localhost:{VISER_PORT}")
    print(f"[viser] orange=solver cameras, green=solver p, blue=aligned GT cameras, RGB=GT-camera lifted {DEPTH_SOURCE}")
    print(f"[viser] GT alignment scale={scale:.9g}, camera_rmse={rmse:.6g}")
    _refresh(0)
    block_until_ctrl_c()


def main() -> None:
    data = _prepare()
    if not DRY_RUN:
        _serve(data)


if __name__ == "__main__":
    main()
