#!/usr/bin/env python3
"""Compare solver output with GT cameras and GT point cloud in Viser.

Loads:
  - GT cameras (COLMAP or Sintel)
  - GT point cloud (lifted from GT depth maps with GT cameras)
  - Solver cameras from result_XM.npz  (R, t keys)
  - Solver point cloud from result_XM.npz  (p key)

A Sim(3) alignment is computed from GT camera centers → solver camera centers,
then GT cameras and GT cloud are transformed into the solver gauge for comparison.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from utils.readgt_colmap import load_colmap_gt
from utils.readgt_sintel import depth_read, load_sintel_gt
from utils.viser_vis import block_until_ctrl_c, get_viser_server


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _quat2rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    return np.array([
        [1 - 2*(qy**2+qz**2),  2*(qx*qy-qz*qw),  2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),  1 - 2*(qx**2+qz**2),  2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),  2*(qy*qz+qx*qw),  1 - 2*(qx**2+qy**2)],
    ], dtype=np.float64)


def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def _trim_mean_np(values: np.ndarray, proportiontocut: float, axis: int) -> np.ndarray:
    values = np.sort(np.asarray(values, dtype=np.float64), axis=axis)
    n = values.shape[axis]
    cut = int(proportiontocut * n)
    if cut <= 0 or 2 * cut >= n:
        return np.mean(values, axis=axis)
    sl = [slice(None)] * values.ndim
    sl[axis] = slice(cut, n - cut)
    return np.mean(values[tuple(sl)], axis=axis)


def _rigid_align_no_scale(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SVD rigid alignment: dst ~= R @ src + t  (both 3×N)."""
    src_mean = np.mean(src, axis=1, keepdims=True)
    dst_mean = np.mean(dst, axis=1, keepdims=True)
    H = (dst - dst_mean) @ (src - src_mean).T
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    t = dst_mean - R @ src_mean
    return R.astype(np.float64), t.astype(np.float64)


def _sim3_align_centers(
    src: np.ndarray,
    dst: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimate Sim(3): dst ~= s * R @ src + t  for 3×N camera centers.

    Uses camera-spread normalisation + SVD (falls back to no-scale SVD on
    degenerate inputs).  Returns (s, R, t).
    """
    src = np.asarray(src, dtype=np.float64).reshape(3, -1)
    dst = np.asarray(dst, dtype=np.float64).reshape(3, -1)
    finite = np.isfinite(src).all(axis=0) & np.isfinite(dst).all(axis=0)
    src_f, dst_f = src[:, finite], dst[:, finite]
    if src_f.shape[1] < 3:
        raise ValueError(f"Too few finite camera center pairs: {src_f.shape[1]}")

    # Trim outliers
    src_avg = _trim_mean_np(src_f, 0.05, axis=1)
    dst_avg = _trim_mean_np(dst_f, 0.05, axis=1)
    src_dis = np.linalg.norm(src_f - src_avg.reshape(3, 1), axis=0)
    dst_dis = np.linalg.norm(dst_f - dst_avg.reshape(3, 1), axis=0)
    keep = (src_dis < np.percentile(src_dis, 90)) & (dst_dis < np.percentile(dst_dis, 90))
    if np.count_nonzero(keep) >= 3:
        src_f, dst_f = src_f[:, keep], dst_f[:, keep]

    src_mu = np.mean(src_f, axis=1)
    dst_mu = np.mean(dst_f, axis=1)
    scale_src = float(np.mean(np.linalg.norm(src_f - src_mu.reshape(3, 1), axis=0)))
    scale_dst = float(np.mean(np.linalg.norm(dst_f - dst_mu.reshape(3, 1), axis=0)))

    if not (np.isfinite(scale_src) and np.isfinite(scale_dst)
            and scale_src > 1e-12 and scale_dst > 1e-12):
        print("[align] Degenerate scales; using no-scale SVD.")
        R, t = _rigid_align_no_scale(src_f, dst_f)
        return 1.0, R, t

    src_norm = src_f / scale_src
    dst_norm = dst_f / scale_dst

    try:
        import teaserpp_python
        params = teaserpp_python.RobustRegistrationSolver.Params()
        params.cbar2 = 1
        params.noise_bound = 0.3
        params.estimate_scaling = False
        params.rotation_estimation_algorithm = (
            teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
        )
        params.rotation_gnc_factor = 1.4
        params.rotation_max_iterations = 100
        params.rotation_cost_threshold = 1e-12
        solver = teaserpp_python.RobustRegistrationSolver(params)
        solver.solve(src_norm, dst_norm)
        sol = solver.getSolution()
        R_norm = np.asarray(sol.rotation, dtype=np.float64)
        t_norm = np.asarray(sol.translation, dtype=np.float64).reshape(3, 1)
    except Exception as exc:
        print(f"[align] TEASER unavailable ({exc}); using SVD.")
        R_norm, t_norm = _rigid_align_no_scale(src_norm, dst_norm)

    s = float(scale_dst / scale_src)
    t_out = scale_dst * t_norm
    pred = s * R_norm @ src[:, finite] + t_out
    rmse = float(np.sqrt(np.mean(np.sum((pred - dst[:, finite]) ** 2, axis=0))))
    print(f"[align] GT→solver Sim(3): scale={s:.6g}, rmse={rmse:.6g}, pairs={src_f.shape[1]}")
    return s, R_norm.astype(np.float64), t_out.astype(np.float64)


# ---------------------------------------------------------------------------
# GT loading
# ---------------------------------------------------------------------------

def _load_gt(rootpath: Path, dataset_cat: str, colmap_dir: Path | None) -> dict[str, Any]:
    cat = dataset_cat.strip().lower()
    if colmap_dir is not None:
        return load_colmap_gt(str(colmap_dir))
    if cat in {"syn4d", "lab"} or (rootpath / "colmap" / "sparse" / "images.txt").is_file():
        return load_colmap_gt(str(rootpath / "colmap"))
    if cat in {"sintel", "sintel_debug"}:
        return load_sintel_gt(str(rootpath))
    raise ValueError(f"Unsupported dataset_cat='{dataset_cat}'.")


def _trailing_frame_id(name: str) -> int | None:
    m = re.search(r"(?<!\d)(\d+)$", Path(name).stem)
    return int(m.group(1)) if m else None


_DEPTH_EXTS = {".dpt", ".npy", ".npz", ".png", ".tif", ".tiff"}


def _read_depth_frame(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".dpt":
        from utils.readgt_sintel import depth_read as sintel_depth_read
        depth = sintel_depth_read(str(path))
    else:
        arr = np.load(str(path)) if path.suffix.lower() in {".npy", ".npz"} else \
              cv2.imread(str(path), cv2.IMREAD_ANYDEPTH).astype(np.float32)
        depth = np.squeeze(arr)
    return np.asarray(depth, dtype=np.float32)


def _load_gt_depth_files(depth_dir: Path) -> dict[int, Path]:
    by_frame: dict[int, Path] = {}
    rank = {".npy": 0, ".npz": 1, ".dpt": 2, ".png": 3, ".tif": 4, ".tiff": 5}
    if not depth_dir.is_dir():
        return by_frame
    for p in depth_dir.iterdir():
        if not p.is_file() or p.suffix.lower() not in _DEPTH_EXTS:
            continue
        fid = _trailing_frame_id(p.name)
        if fid is None:
            continue
        prev = by_frame.get(fid)
        if prev is None or rank.get(p.suffix.lower(), 99) < rank.get(prev.suffix.lower(), 99):
            by_frame[fid] = p
    return by_frame


def _c2w_from_w2c(R_w2c: np.ndarray, t_w2c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R_c2w = R_w2c.T
    C = -R_w2c.T @ t_w2c
    return R_c2w, C


# ---------------------------------------------------------------------------
# Depth lifting
# ---------------------------------------------------------------------------

def _lift_frame(
    depth: np.ndarray,
    K: np.ndarray,
    R_c2w: np.ndarray,
    C: np.ndarray,
    stride: int = 4,
    depth_min: float = 0.0,
    depth_max: float = 0.0,
    depth_scale: float = 1.0,
    max_pts: int = 8000,
    seed: int = 0,
) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32) * float(depth_scale)
    h, w = depth.shape
    yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    xx, yy = xx.reshape(-1), yy.reshape(-1)
    z = depth[yy, xx].astype(np.float64)
    valid = np.isfinite(z) & (z > float(depth_min))
    if depth_max > 0:
        valid &= z <= float(depth_max)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)
    xx, yy, z = xx[valid].astype(np.float64), yy[valid].astype(np.float64), z[valid]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (xx - cx) / fx * z
    y = (yy - cy) / fy * z
    pts_cam = np.stack([x, y, z], axis=1)
    pts_world = (R_c2w @ pts_cam.T).T + C.reshape(1, 3)
    if max_pts > 0 and pts_world.shape[0] > max_pts:
        rng = np.random.default_rng(seed)
        pts_world = pts_world[rng.choice(pts_world.shape[0], max_pts, replace=False)]
    return pts_world.astype(np.float32)


# ---------------------------------------------------------------------------
# Solver loading
# ---------------------------------------------------------------------------

def _resolve_xm_result(rootpath: Path, xm_result: str) -> Path:
    if xm_result:
        path = Path(xm_result).expanduser().resolve()
        if path.is_dir():
            path = path / "result_XM.npz"
    else:
        path = rootpath / "outputs" / "result_XM.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing XM result file: {path}")
    return path


def _load_solver(rootpath: Path, xm_result: str) -> dict[str, Any]:
    path = _resolve_xm_result(rootpath, xm_result)
    data = np.load(path)
    required = {"R", "t", "p"}
    missing = required - set(data.files)
    if missing:
        raise KeyError(f"{path} is missing keys: {missing}")

    R_flat = np.asarray(data["R"], dtype=np.float64)   # (3, 3T)
    t_all  = np.asarray(data["t"], dtype=np.float64)   # (3, T)
    p      = np.asarray(data["p"], dtype=np.float64)   # (3, N)

    n_frames = min(R_flat.shape[1] // 3, t_all.shape[1])
    R_all = R_flat[:, :3 * n_frames]
    t_all = t_all[:, :n_frames]
    p = p.T if (p.ndim == 2 and p.shape[0] == 3) else p  # → (N, 3)

    # Camera centers in solver gauge
    centers = t_all.T  # (T, 3)  — t in result_XM is already the camera center C

    # Per-camera R_c2w (c2w rotation)
    R_c2w_list = [R_all[:, 3*i:3*(i+1)] for i in range(n_frames)]

    print(f"Loaded solver: {n_frames} cameras, {p.shape[0]} points from {path}")
    return {
        "n_frames": n_frames,
        "R_c2w_list": R_c2w_list,
        "centers": centers,
        "p": p.astype(np.float32),
        "path": path,
    }


# ---------------------------------------------------------------------------
# Viser geometry helpers
# ---------------------------------------------------------------------------

def _frustum_lines(
    K: np.ndarray,
    hw: tuple[int, int],
    R_c2w: np.ndarray,
    C: np.ndarray,
    size: float,
) -> np.ndarray:
    h, w = hw
    corners = np.array([[0., 0.], [w-1., 0.], [w-1., h-1.], [0., h-1.]])
    z = size
    x = (corners[:, 0] - K[0, 2]) / K[0, 0] * z
    y = (corners[:, 1] - K[1, 2]) / K[1, 1] * z
    cc = (R_c2w @ np.stack([x, y, np.full(4, z)], axis=0)).T + C.reshape(1, 3)
    segs = [[C, cc[i]] for i in range(4)] + [[cc[i], cc[(i+1)%4]] for i in range(4)]
    return np.asarray(segs, dtype=np.float32)


def _bbox_diag(pts: np.ndarray) -> float:
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] == 0:
        return 1.0
    return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))


def _downsample(pts: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    if n <= 0 or pts.shape[0] <= n:
        return pts
    rng = np.random.default_rng(seed)
    return pts[rng.choice(pts.shape[0], n, replace=False)]


def _traj_segs(centers: np.ndarray) -> np.ndarray:
    if centers.shape[0] < 2:
        return np.zeros((0, 2, 3), dtype=np.float32)
    return np.stack([centers[:-1], centers[1:]], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Main data preparation
# ---------------------------------------------------------------------------

def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    rootpath  = Path(args.rootpath).resolve()
    depth_dir = Path(args.depth_dir).resolve() if args.depth_dir else rootpath / "depths"
    colmap_dir = Path(args.colmap_dir).resolve() if args.colmap_dir else None

    # --- GT cameras ---
    gt = _load_gt(rootpath, args.dataset_cat, colmap_dir)
    gt_items = sorted(gt.items(), key=lambda kv: int(kv[1]["id"]))
    if int(args.frame_step) > 1:
        gt_items = gt_items[::int(args.frame_step)]
    if int(args.max_frames) > 0:
        gt_items = gt_items[:int(args.max_frames)]
    if not gt_items:
        raise RuntimeError("No GT frames loaded.")

    depth_by_frame = _load_gt_depth_files(depth_dir)

    # GT camera geometry
    gt_K_list: list[np.ndarray] = []
    gt_R_c2w_list: list[np.ndarray] = []
    gt_centers: list[np.ndarray] = []
    gt_hw: tuple[int, int] | None = None

    for _name, info in gt_items:
        K = np.asarray(info["K"], dtype=np.float64).reshape(3, 3)
        R_c2w, C = _c2w_from_w2c(
            np.asarray(info["R"], dtype=np.float64).reshape(3, 3),
            np.asarray(info["t"], dtype=np.float64).reshape(3),
        )
        gt_K_list.append(K)
        gt_R_c2w_list.append(R_c2w)
        gt_centers.append(C)
        if gt_hw is None:
            depth_path = depth_by_frame.get(int(info["id"]))
            if depth_path is not None:
                d = _read_depth_frame(depth_path)
                gt_hw = (d.shape[0], d.shape[1])

    if gt_hw is None:
        gt_hw = (int(round(gt_K_list[0][1, 2] * 2)), int(round(gt_K_list[0][0, 2] * 2)))

    gt_centers_arr = np.stack(gt_centers, axis=0)  # (T, 3)

    # --- GT point cloud (lifted from GT depths) ---
    gt_cloud_list: list[np.ndarray] = []
    for view_idx, (_name, info) in enumerate(gt_items):
        frame_id = int(info["id"])
        depth_path = depth_by_frame.get(frame_id)
        if depth_path is None:
            continue
        depth = _read_depth_frame(depth_path)
        pts = _lift_frame(
            depth=depth,
            K=gt_K_list[view_idx],
            R_c2w=gt_R_c2w_list[view_idx],
            C=gt_centers[view_idx],
            stride=int(args.stride),
            depth_min=float(args.depth_min),
            depth_max=float(args.depth_max),
            depth_scale=float(args.depth_scale),
            max_pts=int(args.max_pts_per_frame),
            seed=view_idx,
        )
        gt_cloud_list.append(pts)

    gt_cloud = (
        np.concatenate(gt_cloud_list, axis=0) if gt_cloud_list
        else np.zeros((0, 3), dtype=np.float32)
    )
    gt_cloud = _downsample(gt_cloud, int(args.max_total_pts), seed=0)
    print(f"GT cloud: {gt_cloud.shape[0]} points from {len(gt_cloud_list)} frames")

    # --- Solver ---
    solver = _load_solver(rootpath, args.xm_result)

    # --- GT → solver Sim(3) alignment ---
    n_gt = gt_centers_arr.shape[0]
    n_xm = solver["centers"].shape[0]
    src_centers = []
    dst_centers = []
    for view_idx, (_name, info) in enumerate(gt_items):
        frame_id = int(info["id"])
        xm_idx = frame_id if frame_id < n_xm else view_idx
        if xm_idx >= n_xm:
            continue
        src_centers.append(gt_centers[view_idx])
        dst_centers.append(solver["centers"][xm_idx])

    if len(src_centers) < 3:
        raise RuntimeError(f"Need ≥3 matched camera pairs for alignment; got {len(src_centers)}.")

    s, R_align, t_align = _sim3_align_centers(
        np.stack(src_centers, axis=1),
        np.stack(dst_centers, axis=1),
    )

    # Transform GT cameras and GT cloud into solver gauge
    def _xform_pt(p3: np.ndarray) -> np.ndarray:
        return (s * (R_align @ p3.reshape(3, 1)) + t_align.reshape(3, 1)).reshape(3)

    gt_centers_xm = np.stack([_xform_pt(C) for C in gt_centers], axis=0).astype(np.float32)
    gt_R_c2w_xm   = [R_align @ R for R in gt_R_c2w_list]

    gt_cloud_xm = np.zeros((0, 3), dtype=np.float32)
    if gt_cloud.shape[0] > 0:
        gt_cloud_xm = (s * (R_align @ gt_cloud.T.astype(np.float64)) + t_align).T.astype(np.float32)

    return {
        "gt_K_list": gt_K_list,
        "gt_R_c2w_xm": gt_R_c2w_xm,
        "gt_centers_xm": gt_centers_xm,
        "gt_cloud_xm": gt_cloud_xm,
        "gt_hw": gt_hw,
        "solver": solver,
        "sim3": (s, R_align, t_align),
        "n_gt_frames": len(gt_items),
    }


# ---------------------------------------------------------------------------
# Viser visualisation
# ---------------------------------------------------------------------------

def _serve_viser(data: dict[str, Any], args: argparse.Namespace) -> None:
    server = get_viser_server(host=args.viser_host, port=args.viser_port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    server.gui.add_markdown("### Solver vs GT")

    n_frames = max(data["n_gt_frames"], data["solver"]["n_frames"])
    frame_slider = server.gui.add_slider(
        "Frame", min=0, max=n_frames - 1, step=1, initial_value=0
    )
    btn_prev = server.gui.add_button("Prev")
    btn_next = server.gui.add_button("Next")

    show_gt_cam   = server.gui.add_checkbox("GT cameras",    initial_value=True)
    show_xm_cam   = server.gui.add_checkbox("Solver cameras", initial_value=True)
    show_gt_cloud  = server.gui.add_checkbox("GT cloud",      initial_value=True)
    show_xm_cloud  = server.gui.add_checkbox("Solver cloud",  initial_value=True)

    # Auto-size from point cloud diagonal
    all_pts = np.concatenate([
        data["gt_cloud_xm"] if data["gt_cloud_xm"].shape[0] > 0 else np.zeros((0, 3)),
        data["solver"]["p"],
        data["gt_centers_xm"],
        data["solver"]["centers"].astype(np.float32),
    ], axis=0)
    diag = _bbox_diag(all_pts)
    pt_size      = float(args.point_size)   if args.point_size   > 0 else float(np.clip(0.002 * diag, 0.001, 0.03))
    frustum_size = float(args.camera_size)  if args.camera_size  > 0 else float(np.clip(0.04  * diag, 0.05,  0.5))

    COLOR = {
        "gt_frustum":     np.array([30,  140, 255], dtype=np.uint8),   # blue
        "gt_current":     np.array([80,  200, 255], dtype=np.uint8),
        "gt_traj":        np.array([20,  100, 220], dtype=np.uint8),
        "gt_cloud":       np.array([100, 180, 255], dtype=np.uint8),
        "xm_frustum":     np.array([255, 100,  20], dtype=np.uint8),   # orange
        "xm_current":     np.array([255,  50,   0], dtype=np.uint8),
        "xm_traj":        np.array([200,  60,   0], dtype=np.uint8),
        "xm_cloud":       np.array([ 30,  200,  80], dtype=np.uint8),  # green
    }

    handles: dict[str, Any] = {}

    def _toggle(path: str, visible: bool) -> None:
        h = handles.get(path)
        if h is not None:
            try:
                h.visible = bool(visible)
                return
            except Exception:
                pass
        try:
            server.scene.remove(path)
        except Exception:
            pass

    # --- Static: solver cloud ---
    solver_p = data["solver"]["p"]
    if solver_p.shape[0] > 0:
        handles["/xm/cloud"] = server.scene.add_point_cloud(
            "/xm/cloud",
            points=solver_p,
            colors=COLOR["xm_cloud"],
            point_size=pt_size,
            point_shape="circle",
            visible=True,
        )

    # --- Static: GT cloud ---
    gt_cloud_xm = data["gt_cloud_xm"]
    if gt_cloud_xm.shape[0] > 0:
        handles["/gt/cloud"] = server.scene.add_point_cloud(
            "/gt/cloud",
            points=gt_cloud_xm,
            colors=COLOR["gt_cloud"],
            point_size=pt_size,
            point_shape="circle",
            visible=True,
        )

    # --- Static: camera frustums + trajectories ---
    gt_hw   = data["gt_hw"]
    K_list  = data["gt_K_list"]
    gt_Rc2w = data["gt_R_c2w_xm"]
    gt_C    = data["gt_centers_xm"]

    solver_n  = data["solver"]["n_frames"]
    solver_Rc = data["solver"]["R_c2w_list"]
    solver_C  = data["solver"]["centers"].astype(np.float32)

    # GT frustums
    all_gt_frusta = [
        _frustum_lines(K_list[i], gt_hw, gt_Rc2w[i], gt_C[i], frustum_size)
        for i in range(len(K_list))
    ]
    if all_gt_frusta:
        handles["/gt/frustums"] = server.scene.add_line_segments(
            "/gt/frustums",
            points=np.concatenate(all_gt_frusta, axis=0),
            colors=COLOR["gt_frustum"],
            line_width=1.0,
            visible=True,
        )

    gt_traj = _traj_segs(gt_C)
    if gt_traj.shape[0] > 0:
        handles["/gt/traj"] = server.scene.add_line_segments(
            "/gt/traj",
            points=gt_traj,
            colors=COLOR["gt_traj"],
            line_width=2.0,
            visible=True,
        )

    # Solver frustums — use GT K for aspect ratio (solver has no intrinsics)
    K_ref = K_list[0]
    all_xm_frusta = [
        _frustum_lines(K_ref, gt_hw, solver_Rc[i], solver_C[i], frustum_size)
        for i in range(solver_n)
    ]
    if all_xm_frusta:
        handles["/xm/frustums"] = server.scene.add_line_segments(
            "/xm/frustums",
            points=np.concatenate(all_xm_frusta, axis=0),
            colors=COLOR["xm_frustum"],
            line_width=1.0,
            visible=True,
        )

    xm_traj = _traj_segs(solver_C)
    if xm_traj.shape[0] > 0:
        handles["/xm/traj"] = server.scene.add_line_segments(
            "/xm/traj",
            points=xm_traj,
            colors=COLOR["xm_traj"],
            line_width=2.0,
            visible=True,
        )

    # --- Dynamic: current-frame highlights ---
    state = {"idx": 0}
    DYNAMIC = ["/gt/current_frustum", "/xm/current_frustum"]

    def _refresh(idx: int) -> None:
        idx = int(idx) % n_frames
        state["idx"] = idx

        for p in DYNAMIC:
            try:
                server.scene.remove(p)
            except Exception:
                pass

        if idx < len(K_list) and bool(show_gt_cam.value):
            server.scene.add_line_segments(
                "/gt/current_frustum",
                points=_frustum_lines(K_list[idx], gt_hw, gt_Rc2w[idx], gt_C[idx], frustum_size * 1.4),
                colors=COLOR["gt_current"],
                line_width=3.0,
                visible=True,
            )

        if idx < solver_n and bool(show_xm_cam.value):
            server.scene.add_line_segments(
                "/xm/current_frustum",
                points=_frustum_lines(K_ref, gt_hw, solver_Rc[idx], solver_C[idx], frustum_size * 1.4),
                colors=COLOR["xm_current"],
                line_width=3.0,
                visible=True,
            )

        gt_pos  = gt_C[idx]   if idx < len(gt_C)    else None
        xm_pos  = solver_C[idx] if idx < solver_n   else None
        if gt_pos is not None and xm_pos is not None:
            server.scene.add_line_segments(
                "/gt/pair_line",
                points=np.asarray([[gt_pos, xm_pos]], dtype=np.float32),
                colors=np.array([60, 60, 60], dtype=np.uint8),
                line_width=1.5,
                visible=bool(show_gt_cam.value) and bool(show_xm_cam.value),
            )

        gt_err = (
            float(np.linalg.norm(gt_C[idx] - xm_pos))
            if gt_pos is not None and xm_pos is not None else float("nan")
        )
        print(
            f"[VIS] frame={idx}  "
            f"GT_C={gt_pos.tolist() if gt_pos is not None else '—'}  "
            f"XM_C={xm_pos.tolist() if xm_pos is not None else '—'}  "
            f"|err|={gt_err:.4f}"
        )

    def _set_frame(idx: int) -> None:
        idx = int(idx) % n_frames
        if int(frame_slider.value) != idx:
            frame_slider.value = idx
        _refresh(idx)

    @frame_slider.on_update
    def _(_e): _refresh(int(frame_slider.value))

    @btn_next.on_click
    def _(_e): _set_frame(state["idx"] + 1)

    @btn_prev.on_click
    def _(_e): _set_frame(state["idx"] - 1)

    @show_gt_cam.on_update
    def _(_e):
        _toggle("/gt/frustums", bool(show_gt_cam.value))
        _toggle("/gt/traj", bool(show_gt_cam.value))
        _refresh(state["idx"])

    @show_xm_cam.on_update
    def _(_e):
        _toggle("/xm/frustums", bool(show_xm_cam.value))
        _toggle("/xm/traj", bool(show_xm_cam.value))
        _refresh(state["idx"])

    @show_gt_cloud.on_update
    def _(_e): _toggle("/gt/cloud", bool(show_gt_cloud.value))

    @show_xm_cloud.on_update
    def _(_e): _toggle("/xm/cloud", bool(show_xm_cloud.value))

    _refresh(0)
    print(f"[VIS] Viser at http://localhost:{args.viser_port}")
    print(f"[VIS] SSH: ssh -L {args.viser_port}:localhost:{args.viser_port} <host>")
    block_until_ctrl_c()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visualise GT cameras + GT depth cloud alongside solver cameras "
            "+ solver point cloud (p_est), aligned into solver gauge."
        )
    )
    parser.add_argument("--rootpath",    default="", help="Dataset root")
    parser.add_argument("--dataset-cat", default="", help="Dataset category (Syn4D, Sintel, …)")
    parser.add_argument("--depth-dir",   default="", help="GT depth directory (default: rootpath/depths)")
    parser.add_argument("--colmap-dir",  default="", help="COLMAP sparse directory")
    parser.add_argument(
        "--xm-result", default="",
        help="result_XM.npz or its parent dir (default: rootpath/outputs/result_XM.npz)",
    )
    parser.add_argument("--stride",     type=int,   default=4,   help="Depth pixel sampling stride")
    parser.add_argument("--frame-step", type=int,   default=1,   help="Use every Nth GT frame")
    parser.add_argument("--max-frames", type=int,   default=0,   help="Limit GT frames; 0 = all")
    parser.add_argument("--depth-scale", type=float, default=1.0)
    parser.add_argument("--depth-min",   type=float, default=0.0)
    parser.add_argument("--depth-max",   type=float, default=0.0, help="0 = disabled")
    parser.add_argument("--max-pts-per-frame", type=int, default=8000)
    parser.add_argument("--max-total-pts",     type=int, default=400000)
    parser.add_argument("--point-size",  type=float, default=0.0, help="0 = auto")
    parser.add_argument("--camera-size", type=float, default=0.0, help="0 = auto")
    parser.add_argument("--viser-host", default=os.environ.get("VISER_HOST", "0.0.0.0"))
    parser.add_argument("--viser-port", type=int, default=int(os.environ.get("VISER_PORT", "8080")))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.rootpath or not args.dataset_cat:
        import utils.definitions
        if not args.rootpath:
            args.rootpath = str(utils.definitions.rootpath)
        if not args.dataset_cat:
            args.dataset_cat = str(utils.definitions.dataset_cat)

    data = _prepare(args)
    s, _, _ = data["sim3"]
    print(
        f"GT→solver scale={s:.6g}  "
        f"GT cloud={data['gt_cloud_xm'].shape[0]} pts  "
        f"Solver cloud={data['solver']['p'].shape[0]} pts  "
        f"GT frames={data['n_gt_frames']}  "
        f"Solver frames={data['solver']['n_frames']}"
    )

    if args.dry_run:
        return

    _serve_viser(data, args)


if __name__ == "__main__":
    main()
