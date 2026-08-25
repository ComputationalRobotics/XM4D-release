#!/usr/bin/env python3
"""Show solver cameras and solver-aligned DepthAnything3 point clouds in Viser.

No GT is loaded here.  DA3 depth maps are lifted with the solver cameras from
result_XM.npz.  Each frame can get a small depth-scale alignment chosen by a
robust trimmed Chamfer search against the solver Ceres point cloud p.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from utils.viser_vis import block_until_ctrl_c, get_viser_server


# ------------------------------ parameters ---------------------------------

ROOTPATH = ""       # "" uses utils.definitions.rootpath
XM_RESULT = ""      # "" uses ROOTPATH/outputs/result_XM.npz
DA3_DIR = ""        # "" uses ROOTPATH/outputs/depth_anything_3
IMAGE_DIR = ""      # "" uses utils.definitions.video_dir

FRAME_ID_OFFSET: int | str = "auto"  # "auto", 0, or 1
FRAME_STEP = 1
MAX_FRAMES = 0      # <= 0 means all usable frames

STRIDE = 4
DEPTH_MIN = 0.0
DEPTH_MAX = 0.0     # <= 0 disables max-depth filtering
MAX_POINTS_PER_FRAME = 8000
MAX_TOTAL_POINTS = 500000

USE_SOLVER_DA3_SCALE = True
REQUIRE_SOLVER_DA3_SCALE = True
SOLVER_DA3_SCALE_KEY = "da3_frame_scale"
SOLVER_DA3_SCALE_LOSS_KEY = "da3_frame_scale_loss"

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

CAMERA_SIZE = 0.0   # <= 0 auto
POINT_SIZE = 0.0    # <= 0 auto
AUTO_SIZE_KEEP_PERCENTILE = 95.0
AUTO_POINT_SIZE_FRAC = 0.0006
AUTO_POINT_SIZE_MIN = 0.0002
AUTO_POINT_SIZE_MAX = 0.006
AUTO_CAMERA_SIZE_FRAC = 0.015
AUTO_CAMERA_SIZE_MIN = 0.01
AUTO_CAMERA_SIZE_MAX = 0.12
VISER_HOST = os.environ.get("VISER_HOST", "0.0.0.0")
VISER_PORT = int(os.environ.get("VISER_PORT", "8080"))
SEED = 0
DRY_RUN = False


# ------------------------------- loading ------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _paths() -> tuple[Path, Path, Path, Path]:
    import utils.definitions as definitions

    root = Path(ROOTPATH).expanduser().resolve() if ROOTPATH else Path(definitions.rootpath).resolve()
    xm_result = Path(XM_RESULT).expanduser().resolve() if XM_RESULT else root / "outputs" / "result_XM.npz"
    if xm_result.is_dir():
        xm_result = xm_result / "result_XM.npz"
    da3_dir = Path(DA3_DIR).expanduser().resolve() if DA3_DIR else root / "outputs" / "depth_anything_3"
    image_dir = Path(IMAGE_DIR).expanduser().resolve() if IMAGE_DIR else Path(definitions.video_dir).resolve()
    return root, xm_result, da3_dir, image_dir


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
    solver_da3_scale = None
    solver_da3_scale_loss = None
    if USE_SOLVER_DA3_SCALE:
        if SOLVER_DA3_SCALE_KEY not in data.files:
            message = (
                f"{path} is missing {SOLVER_DA3_SCALE_KEY!r}; rerun static_solver.py "
                "after the DA3 scale change, or set USE_SOLVER_DA3_SCALE=False."
            )
            if REQUIRE_SOLVER_DA3_SCALE:
                raise KeyError(message)
            print(f"[solver-scale] {message}")
        else:
            solver_da3_scale = np.asarray(data[SOLVER_DA3_SCALE_KEY], dtype=np.float64).reshape(-1)
            if solver_da3_scale.shape[0] < n and REQUIRE_SOLVER_DA3_SCALE:
                raise ValueError(
                    f"{SOLVER_DA3_SCALE_KEY} has {solver_da3_scale.shape[0]} entries, "
                    f"but solver has {n} frames."
                )
            if SOLVER_DA3_SCALE_LOSS_KEY in data.files:
                solver_da3_scale_loss = np.asarray(data[SOLVER_DA3_SCALE_LOSS_KEY], dtype=np.float64).reshape(-1)
            print(
                f"[solver-scale] using {SOLVER_DA3_SCALE_KEY} from result_XM.npz "
                f"({solver_da3_scale.shape[0]} entries)"
            )

    if p.ndim == 2 and p.shape[0] == 3:
        p = p.T
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"Expected solver p shape (3, N) or (N, 3), got {p.shape}")
    p = p[np.isfinite(p).all(axis=1)]

    print(f"[solver] {n} cameras, {p.shape[0]} Ceres points from {path}")
    return {
        "R_c2w": R_c2w,
        "centers": centers,
        "p": p.astype(np.float32),
        "da3_frame_scale": solver_da3_scale,
        "da3_frame_scale_loss": solver_da3_scale_loss,
        "path": path,
    }


def _load_da3(da3_dir: Path) -> dict[str, Any]:
    depth_path = da3_dir / "depth_maps.npy"
    if not depth_path.is_file():
        raise FileNotFoundError(f"Missing DA3 depth maps: {depth_path}")
    depths = np.asarray(np.load(depth_path), dtype=np.float32)
    if depths.ndim != 3:
        raise ValueError(f"Expected DA3 depth shape (T, H, W), got {depths.shape}")

    intrinsics = None
    intr_path = da3_dir / "intrinsics.npy"
    if intr_path.is_file():
        intrinsics = np.asarray(np.load(intr_path), dtype=np.float64)
        if intrinsics.ndim == 2:
            intrinsics = intrinsics[None, :, :]
        if intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
            raise ValueError(f"Expected DA3 intrinsics shape (T, 3, 3), got {intrinsics.shape}")
        print(f"[da3] intrinsics from {intr_path}")
    else:
        print("[da3] no intrinsics.npy; using center-principal fallback intrinsics")

    print(f"[da3] depth stack {depths.shape} from {depth_path}")
    return {"depths": depths, "intrinsics": intrinsics, "path": depth_path}


def _trailing_frame_id(name: str) -> int | None:
    match = re.search(r"(?<!\d)(\d+)$", Path(name).stem)
    return int(match.group(1)) if match else None


def _image_files(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        return []
    return sorted(
        [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: (_trailing_frame_id(p.name) if _trailing_frame_id(p.name) is not None else 10**12, p.name),
    )


def _read_rgb(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"[image] could not read {path}; using gray fallback")
        return None
    return image[:, :, ::-1].copy()


# ------------------------------- geometry -----------------------------------

def _index_for_frame(frame_id: int, order_idx: int, count: int) -> int | None:
    if count <= 0:
        return None
    if FRAME_ID_OFFSET != "auto":
        idx = int(frame_id) - int(FRAME_ID_OFFSET)
    elif 0 <= int(frame_id) < count:
        idx = int(frame_id)
    elif 1 <= int(frame_id) <= count:
        idx = int(frame_id) - 1
    else:
        idx = int(order_idx)
    return idx if 0 <= idx < count else None


def _intrinsics_for_frame(da3: dict[str, Any], frame_idx: int) -> np.ndarray:
    intrinsics = da3["intrinsics"]
    if intrinsics is not None:
        return np.asarray(intrinsics[min(frame_idx, intrinsics.shape[0] - 1)], dtype=np.float64).reshape(3, 3)
    depth = da3["depths"][frame_idx]
    h, w = depth.shape
    f = float(max(h, w))
    return np.asarray([[f, 0.0, (w - 1) * 0.5], [0.0, f, (h - 1) * 0.5], [0.0, 0.0, 1.0]], dtype=np.float64)


def _camera_depth_samples(depth: np.ndarray, K: np.ndarray, order_idx: int, max_points: int) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    h, w = depth.shape
    yy, xx = np.mgrid[0:h:max(1, int(STRIDE)), 0:w:max(1, int(STRIDE))]
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    z = depth[yy, xx].astype(np.float64)
    valid = np.isfinite(z) & (z > float(DEPTH_MIN))
    if DEPTH_MAX > 0:
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


def _sample_rgb(
    rgb: np.ndarray | None,
    xx: np.ndarray,
    yy: np.ndarray,
    depth_shape: tuple[int, int],
) -> np.ndarray:
    if rgb is None:
        return np.tile(np.array([180, 180, 180], dtype=np.uint8), (xx.shape[0], 1))
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


def _lift_da3_depth(
    depth: np.ndarray,
    K: np.ndarray,
    R_c2w: np.ndarray,
    center: np.ndarray,
    ray_scale: float,
    order_idx: int,
    rgb: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth, dtype=np.float32)
    h, w = depth.shape
    yy, xx = np.mgrid[0:h:max(1, int(STRIDE)), 0:w:max(1, int(STRIDE))]
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    z = depth[yy, xx].astype(np.float64)
    valid = np.isfinite(z) & (z > float(DEPTH_MIN))
    if DEPTH_MAX > 0:
        valid &= z <= float(DEPTH_MAX)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    xx = xx[valid].astype(np.float64)
    yy = yy[valid].astype(np.float64)
    z = z[valid]
    colors = _sample_rgb(rgb, xx, yy, (h, w))

    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    pts_cam = np.stack([(xx - cx) / fx * z, (yy - cy) / fy * z, z], axis=1)
    pts_world = (float(ray_scale) * (R_c2w @ pts_cam.T)).T + center.reshape(1, 3)

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


def _bbox_diag(arrays: list[np.ndarray], keep_percentile: float = 100.0) -> float:
    pts = [np.asarray(a).reshape(-1, 3) for a in arrays if a is not None and np.asarray(a).size > 0]
    if not pts:
        return 1.0
    all_pts = np.concatenate(pts, axis=0)
    all_pts = all_pts[np.isfinite(all_pts).all(axis=1)]
    if all_pts.shape[0] == 0:
        return 1.0
    keep_percentile = float(np.clip(keep_percentile, 1.0, 100.0))
    if keep_percentile < 100.0 and all_pts.shape[0] > 16:
        center = np.median(all_pts.astype(np.float64), axis=0)
        radius = np.linalg.norm(all_pts.astype(np.float64) - center.reshape(1, 3), axis=1)
        threshold = np.percentile(radius, keep_percentile)
        kept = all_pts[radius <= threshold]
        if kept.shape[0] >= 8:
            all_pts = kept
    return float(np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0)))


# ------------------------------ scale search --------------------------------

def _filter_point_outliers(points: np.ndarray, keep_percentile: float) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] == 0 or keep_percentile >= 100.0:
        return pts
    center = np.median(pts.astype(np.float64), axis=0)
    radius = np.linalg.norm(pts.astype(np.float64) - center.reshape(1, 3), axis=1)
    threshold = np.percentile(radius, np.clip(float(keep_percentile), 1.0, 100.0))
    return pts[radius <= threshold]


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


def _estimate_frame_scale(
    *,
    depth: np.ndarray,
    K: np.ndarray,
    R_c2w: np.ndarray,
    center: np.ndarray,
    solver_fit_points: np.ndarray,
    solver_tree: Any,
    order_idx: int,
) -> tuple[float, float]:
    if not FIT_DA_PER_FRAME_SCALE or solver_tree is None or solver_fit_points.shape[0] < 10:
        return 1.0, float("nan")
    try:
        from scipy.spatial import cKDTree
    except Exception as exc:
        print(f"[scale] skipped per-frame scale; scipy unavailable: {exc}")
        return 1.0, float("nan")

    pts_cam = _camera_depth_samples(depth, K, order_idx, PER_FRAME_SCALE_FIT_POINTS)
    if pts_cam.shape[0] < 10:
        return 1.0, float("nan")

    center = np.asarray(center, dtype=np.float64).reshape(1, 3)
    base_dirs = (R_c2w @ pts_cam.T).T

    def loss_at(scale: float) -> float:
        src = (center + float(scale) * base_dirs).astype(np.float32)
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


# ------------------------------- prepare ------------------------------------

def _prepare() -> dict[str, Any]:
    _root, xm_result, da3_dir, image_dir = _paths()
    solver = _load_solver(xm_result)
    da3 = _load_da3(da3_dir)
    image_paths = _image_files(image_dir)
    print(f"[image] {len(image_paths)} images from {image_dir}")

    n = min(len(solver["R_c2w"]), solver["centers"].shape[0], da3["depths"].shape[0])
    frame_indices = list(range(n))[:: max(1, int(FRAME_STEP))]
    if MAX_FRAMES > 0:
        frame_indices = frame_indices[: int(MAX_FRAMES)]
    if not frame_indices:
        raise RuntimeError("No usable solver/DA3 frames.")

    solver_scales = solver.get("da3_frame_scale")
    solver_scale_losses = solver.get("da3_frame_scale_loss")
    use_solver_scales = bool(USE_SOLVER_DA3_SCALE) and solver_scales is not None

    solver_fit_points = _filter_point_outliers(solver["p"], SOLVER_FIT_KEEP_PERCENTILE)
    solver_fit_points, _ = _downsample(solver_fit_points, None, PER_FRAME_SOLVER_FIT_POINTS, SEED + 17)
    solver_tree = None
    if use_solver_scales:
        print("[scale] using static_solver da3_frame_scale; viewer Chamfer recompute is disabled.")
    elif FIT_DA_PER_FRAME_SCALE and solver_fit_points.shape[0] >= 10:
        try:
            from scipy.spatial import cKDTree

            solver_tree = cKDTree(solver_fit_points)
            print(
                f"[scale] per-frame DA scale search enabled; target={solver_fit_points.shape[0]} "
                f"Ceres points, range=[{PER_FRAME_SCALE_MIN}, {PER_FRAME_SCALE_MAX}]"
            )
        except Exception as exc:
            print(f"[scale] disabled; scipy unavailable: {exc}")

    frames: list[dict[str, Any]] = []
    point_parts: list[np.ndarray] = []
    color_parts: list[np.ndarray] = []
    camera_K: list[np.ndarray] = []
    camera_hw: list[tuple[int, int]] = []
    frame_scales: list[float] = []
    frame_scale_losses: list[float] = []

    for view_idx, frame_idx in enumerate(frame_indices):
        depth = np.asarray(da3["depths"][frame_idx], dtype=np.float32)
        K = _intrinsics_for_frame(da3, frame_idx)
        R_c2w = solver["R_c2w"][frame_idx]
        center = solver["centers"][frame_idx]
        image_path = image_paths[frame_idx] if frame_idx < len(image_paths) else None
        rgb = _read_rgb(image_path)

        if use_solver_scales:
            frame_scale = float(solver_scales[frame_idx])
            if not np.isfinite(frame_scale) or frame_scale <= 0.0:
                raise ValueError(f"Invalid solver DA3 scale at frame {frame_idx}: {frame_scale}")
            if solver_scale_losses is not None and frame_idx < solver_scale_losses.shape[0]:
                frame_loss = float(solver_scale_losses[frame_idx])
            else:
                frame_loss = float("nan")
        else:
            frame_scale, frame_loss = _estimate_frame_scale(
                depth=depth,
                K=K,
                R_c2w=R_c2w,
                center=center,
                solver_fit_points=solver_fit_points,
                solver_tree=solver_tree,
                order_idx=frame_idx,
            )
        ray_scale = frame_scale
        pts, colors = _lift_da3_depth(depth, K, R_c2w, center, ray_scale, frame_idx, rgb)

        point_parts.append(pts)
        color_parts.append(colors)
        camera_K.append(K)
        camera_hw.append(tuple(depth.shape))
        frame_scales.append(frame_scale)
        frame_scale_losses.append(frame_loss)
        frames.append(
            {
                "view_idx": view_idx,
                "frame_idx": frame_idx,
                "points": pts,
                "colors": colors,
                "K": K,
                "hw": tuple(depth.shape),
                "frame_scale": frame_scale,
                "ray_scale": ray_scale,
                "scale_loss": frame_loss,
                "scale_source": SOLVER_DA3_SCALE_KEY if use_solver_scales else "viewer_chamfer",
                "image_path": image_path,
            }
        )
        print(
            f"[frame] {frame_idx}: points={pts.shape[0]} "
            f"da_scale={frame_scale:.6g} loss={frame_loss:.6g} "
            f"source={SOLVER_DA3_SCALE_KEY if use_solver_scales else 'viewer_chamfer'}"
        )

    points = np.concatenate(point_parts, axis=0).astype(np.float32) if point_parts else np.zeros((0, 3), dtype=np.float32)
    colors = np.concatenate(color_parts, axis=0).astype(np.uint8) if color_parts else np.zeros((0, 3), dtype=np.uint8)
    points, colors = _downsample(points, colors, MAX_TOTAL_POINTS, SEED)

    scales = np.asarray(frame_scales, dtype=np.float64)
    finite_scales = scales[np.isfinite(scales)]
    if finite_scales.size > 0:
        print(
            f"[scale] summary: min={finite_scales.min():.6g}, "
            f"median={np.median(finite_scales):.6g}, max={finite_scales.max():.6g}"
        )
    print(f"[ready] frames={len(frames)}, DA3 points={points.shape[0]}, solver_p={solver['p'].shape[0]}")

    return {
        "solver": solver,
        "frames": frames,
        "points": points,
        "colors": colors,
        "camera_K": camera_K,
        "camera_hw": camera_hw,
        "frame_scales": scales,
        "frame_scale_losses": np.asarray(frame_scale_losses, dtype=np.float64),
        "scale_source": SOLVER_DA3_SCALE_KEY if use_solver_scales else "viewer_chamfer",
    }


# -------------------------------- viewer -------------------------------------

def _serve(data: dict[str, Any]) -> None:
    solver = data["solver"]
    solver_centers = solver["centers"].astype(np.float32)
    solver_points = solver["p"].astype(np.float32)
    da3_points = data["points"]
    da3_colors = data["colors"]
    frames = data["frames"]

    diag = _bbox_diag([solver_centers, solver_points, da3_points], AUTO_SIZE_KEEP_PERCENTILE)
    point_size = (
        float(POINT_SIZE)
        if POINT_SIZE > 0
        else float(np.clip(AUTO_POINT_SIZE_FRAC * diag, AUTO_POINT_SIZE_MIN, AUTO_POINT_SIZE_MAX))
    )
    frustum_size = (
        float(CAMERA_SIZE)
        if CAMERA_SIZE > 0
        else float(np.clip(AUTO_CAMERA_SIZE_FRAC * diag, AUTO_CAMERA_SIZE_MIN, AUTO_CAMERA_SIZE_MAX))
    )
    print(
        f"[VIS] auto_size_diag={diag:.6g} point_size={point_size:.6g} "
        f"camera_size={frustum_size:.6g}"
    )

    if frames:
        K_ref = frames[0]["K"]
        hw_ref = frames[0]["hw"]
    else:
        K_ref = np.asarray([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
        hw_ref = (480, 640)

    solver_frustums = [
        _frustum_lines(K_ref, hw_ref, solver["R_c2w"][i], solver_centers[i], frustum_size)
        for i in range(len(solver["R_c2w"]))
    ]

    server = get_viser_server(host=VISER_HOST, port=VISER_PORT)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    server.gui.add_markdown("### Solver + DA3 Depth")

    frame_slider = server.gui.add_slider("Frame", min=0, max=max(0, len(frames) - 1), step=1, initial_value=0)
    btn_prev = server.gui.add_button("Prev")
    btn_next = server.gui.add_button("Next")
    show_solver_cameras = server.gui.add_checkbox("Solver cameras", initial_value=True)
    show_da3_points = server.gui.add_checkbox("DA3 RGB points", initial_value=True)
    show_all_da3 = server.gui.add_checkbox("All DA3 frames", initial_value=False)
    show_solver_p = server.gui.add_checkbox("Solver p fit target", initial_value=False)

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

    if solver_frustums:
        handles["/solver/cameras"] = server.scene.add_line_segments(
            "/solver/cameras",
            points=np.concatenate(solver_frustums, axis=0),
            colors=np.array([255, 110, 30], dtype=np.uint8),
            line_width=1.5,
            visible=True,
        )
    solver_traj = _trajectory(solver_centers)
    if solver_traj.shape[0] > 0:
        handles["/solver/trajectory"] = server.scene.add_line_segments(
            "/solver/trajectory",
            points=solver_traj,
            colors=np.array([210, 70, 10], dtype=np.uint8),
            line_width=3.0,
            visible=True,
        )
    if da3_points.shape[0] > 0:
        handles["/da3/all_points"] = server.scene.add_point_cloud(
            "/da3/all_points",
            points=da3_points,
            colors=da3_colors,
            point_size=point_size,
            point_shape="circle",
            visible=bool(show_da3_points.value) and bool(show_all_da3.value),
        )
    if solver_points.shape[0] > 0:
        handles["/solver/p_fit_target"] = server.scene.add_point_cloud(
            "/solver/p_fit_target",
            points=solver_points,
            colors=np.tile(np.array([30, 210, 90], dtype=np.uint8), (solver_points.shape[0], 1)),
            point_size=point_size,
            point_shape="circle",
            visible=bool(show_solver_p.value),
        )

    dynamic_paths = ["/solver/current_camera", "/da3/current_points"]

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
        frame_idx = int(rec["frame_idx"])
        _clear_dynamic()

        if bool(show_da3_points.value) and not bool(show_all_da3.value) and rec["points"].shape[0] > 0:
            server.scene.add_point_cloud(
                "/da3/current_points",
                points=rec["points"].astype(np.float32),
                colors=rec["colors"].astype(np.uint8),
                point_size=max(point_size * 1.25, 0.003),
                point_shape="circle",
                visible=True,
            )
        if bool(show_solver_cameras.value):
            server.scene.add_line_segments(
                "/solver/current_camera",
                points=_frustum_lines(rec["K"], rec["hw"], solver["R_c2w"][frame_idx], solver_centers[frame_idx], frustum_size * 1.4),
                colors=np.array([255, 55, 0], dtype=np.uint8),
                line_width=3.0,
                visible=True,
            )

        print(
            f"[VIS] slider={idx} frame={frame_idx} points={rec['points'].shape[0]} "
            f"da_scale={rec['frame_scale']:.6g} loss={rec['scale_loss']:.6g} "
            f"source={rec.get('scale_source', 'unknown')} "
            f"image={rec['image_path']}"
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

    @show_da3_points.on_update
    def _(_event):
        _toggle("/da3/all_points", bool(show_da3_points.value) and bool(show_all_da3.value))
        _refresh(state["idx"])

    @show_all_da3.on_update
    def _(_event):
        _toggle("/da3/all_points", bool(show_da3_points.value) and bool(show_all_da3.value))
        _refresh(state["idx"])

    @show_solver_p.on_update
    def _(_event):
        _toggle("/solver/p_fit_target", bool(show_solver_p.value))

    print(f"[viser] http://localhost:{VISER_PORT}")
    print("[viser] orange=solver cameras, RGB=solver-aligned DA3 points")
    _refresh(0)
    block_until_ctrl_c()


def main() -> None:
    data = _prepare()
    if not DRY_RUN:
        _serve(data)


if __name__ == "__main__":
    main()
