#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

from visualize_edg_solution_open3d import (
    _camera_axes_lines,
    _load_dict,
    _load_dynamic_points,
    _load_geometry_points,
    _load_gt_c2w,
    _load_solved_c2w,
    _resolve_gt_geometry_paths,
    _symmetric_chamfer,
)
from utils import definitions
from utils.error import ATE_TEASER

try:
    import open3d as o3d
except Exception:  # pragma: no cover - optional dependency
    o3d = None

_OPEN3D_WINDOWS_ENABLED = (
    os.environ.get("ENABLE_OPEN3D_WINDOWS", "0").lower() in {"1", "true", "yes", "on"}
    or os.environ.get("ALLOW_OPEN3D_WINDOWS", "0").lower() in {"1", "true", "yes", "on"}
)
_OPEN3D_WINDOWS_DISABLED = (
    not _OPEN3D_WINDOWS_ENABLED
    or os.environ.get("DISABLE_OPEN3D_WINDOWS", "0").lower() in {"1", "true", "yes", "on"}
    or os.environ.get("MOSCA_DISABLE_OPEN3D", "0").lower() in {"1", "true", "yes", "on"}
)

if o3d is not None and _OPEN3D_WINDOWS_DISABLED:
    def _open3d_no_window(*_args, **_kwargs):
        print("[Open3D] window creation disabled. Set ENABLE_OPEN3D_WINDOWS=1 to allow it.")
        return False

    class _NoWindowVisualizer:
        def create_window(self, *_args, **_kwargs):
            return _open3d_no_window()

        def add_geometry(self, *_args, **_kwargs):
            return False

        def update_geometry(self, *_args, **_kwargs):
            return False

        def get_render_option(self):
            class _RenderOption:
                point_size = 1.0
            return _RenderOption()

        def register_key_callback(self, *_args, **_kwargs):
            return False

        def run(self):
            return None

        def close(self):
            return None

        def destroy_window(self):
            return None

    o3d.visualization.draw_geometries = _open3d_no_window
    o3d.visualization.VisualizerWithKeyCallback = _NoWindowVisualizer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate Chamfer distance for our EDG outputs against GT geometry."
    )
    parser.add_argument(
        "--scene-path",
        type=Path,
        default=None,
        help="Optional path inside a scene; used to auto-detect dataset root.",
    )
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--gt-mesh", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--target-points", type=int, default=1000)
    parser.add_argument("--scale-min", type=float, default=0.2)
    parser.add_argument("--scale-max", type=float, default=5.0)
    parser.add_argument("--scale-steps", type=int, default=21)
    parser.add_argument("--scale-levels", type=int, default=3)
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save/report visualization data; does not open an Open3D window unless --show-open3d is also set.",
    )
    parser.add_argument(
        "--show-open3d",
        action="store_true",
        help="Actually open the interactive Open3D Chamfer/debug viewer when --visualize is set.",
    )
    parser.add_argument("--no-vis", action="store_true")
    parser.add_argument("--camera-axis-len", type=float, default=0.08)
    parser.add_argument("--point-size", type=float, default=3.0)
    return parser.parse_args()


def _load_prior_points(data: np.lib.npyio.NpzFile) -> np.ndarray | None:
    for key in (
        "prior_source_points_world",
        "prior_edg_rest_world",
        "prior_edg_rest",
        "prior_points_world",
    ):
        if key in data:
            pts = np.asarray(data[key], dtype=np.float64).reshape(-1, 3)
            if pts.size > 0:
                return pts
    return None


def _sample_exact(points: np.ndarray, count: int, seed: int = 0) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if count <= 0 or points.shape[0] == 0:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=int(count), replace=points.shape[0] < int(count))
    return points[idx]


def _augment_with_prior(dyn_pts: np.ndarray, prior_pts: np.ndarray | None, target: int) -> Tuple[np.ndarray, int]:
    dyn_pts = np.asarray(dyn_pts, dtype=np.float64).reshape(-1, 3)
    if dyn_pts.shape[0] >= target or prior_pts is None:
        return dyn_pts, 0

    needed = int(target - dyn_pts.shape[0])
    prior_pts = np.asarray(prior_pts, dtype=np.float64).reshape(-1, 3)
    if prior_pts.shape[0] == 0:
        return dyn_pts, 0

    extra = _sample_exact(prior_pts, min(needed, prior_pts.shape[0]), seed=1)
    merged = np.concatenate([dyn_pts, extra], axis=0)
    return merged, int(extra.shape[0])


def _find_scene_root(path: Path) -> Path:
    if path.is_file():
        path = path.parent
    path = path.resolve()
    for parent in [path] + list(path.parents):
        if (parent / "outputs").is_dir() and (parent / "images").is_dir():
            return parent
        if (parent / "outputs").is_dir() and (parent / "masks").is_dir():
            return parent
    return path


def _find_latest_results_npz(output_dir: Path) -> Path:
    if not output_dir.exists():
        raise FileNotFoundError(f"Missing outputs folder: {output_dir}")
    candidates = sorted(output_dir.glob("results_all*.npz"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        candidates = sorted(
            [p for p in output_dir.glob("*.npz") if p.name not in {"result_XM.npz", "tracks_dynamic_only.npz"}],
            key=lambda p: p.stat().st_mtime,
        )
    if not candidates:
        raise FileNotFoundError(f"No results npz found in {output_dir}")
    return candidates[-1]


def _c2w_to_w2c(c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R_c2w = np.asarray(c2w[:3, :3], dtype=np.float64)
    t_c2w = np.asarray(c2w[:3, 3], dtype=np.float64)
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ t_c2w
    return R_w2c, t_w2c


def _apply_inverse_similarity(points: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return (pts - np.asarray(t, dtype=np.float64).reshape(1, 3)) @ np.asarray(R, dtype=np.float64) / float(scale)


def _best_fit_rotation(gt_c2w: np.ndarray, solver_c2w: np.ndarray) -> np.ndarray:
    R_sum = np.zeros((3, 3), dtype=np.float64)
    for solver_pose, gt_pose in zip(solver_c2w, gt_c2w):
        R_sum += np.asarray(solver_pose[:3, :3], dtype=np.float64) @ np.asarray(gt_pose[:3, :3], dtype=np.float64).T
    U, _, Vt = np.linalg.svd(R_sum)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def _load_gt_points(paths: list[Path | None], frame_id: int) -> np.ndarray:
    if frame_id < 0 or frame_id >= len(paths):
        return np.zeros((0, 3), dtype=np.float64)
    path = paths[frame_id]
    if path is None:
        return np.zeros((0, 3), dtype=np.float64)
    return _load_geometry_points(path, sample_points=0, voxel_size=0.0)


def _search_global_scale(
    solver_pts_by_frame: list[np.ndarray],
    gt_pts_by_frame: list[np.ndarray],
    scale_min: float,
    scale_max: float,
    scale_steps: int,
) -> tuple[float, float]:
    best_scale, best_score, _ = _search_global_scale_multilevel(
        solver_pts_by_frame,
        gt_pts_by_frame,
        scale_min=scale_min,
        scale_max=scale_max,
        scale_steps=scale_steps,
        scale_levels=1,
    )
    return best_scale, best_score


def _search_global_scale_multilevel(
    solver_pts_by_frame: list[np.ndarray],
    gt_pts_by_frame: list[np.ndarray],
    scale_min: float,
    scale_max: float,
    scale_steps: int,
    scale_levels: int,
) -> tuple[float, float, list[dict]]:
    lo = min(float(scale_min), float(scale_max))
    hi = max(float(scale_min), float(scale_max))
    if lo <= 0.0 or hi <= 0.0:
        raise ValueError("Scale search bounds must be positive.")

    best_scale = 1.0
    best_score = float("inf")
    history = []
    steps = max(int(scale_steps), 2)
    levels = max(int(scale_levels), 1)

    for level in range(levels):
        scales = np.geomspace(lo, hi, steps) if lo < hi else np.asarray([lo], dtype=np.float64)
        if lo <= 1.0 <= hi:
            scales = np.unique(np.concatenate([scales, np.asarray([1.0], dtype=np.float64)]))

        level_rows = []
        print(f"[scale-search] level {level + 1}/{levels} interval=[{lo:.8g}, {hi:.8g}]")
        for scale in scales.tolist():
            scores = []
            for src, tgt in zip(solver_pts_by_frame, gt_pts_by_frame):
                if src.shape[0] == 0 or tgt.shape[0] == 0:
                    continue
                score = _symmetric_chamfer(src * float(scale), tgt)
                if np.isfinite(score):
                    scores.append(score)
            mean_score = float(np.mean(scores)) if scores else float("inf")
            level_rows.append({"scale": float(scale), "mean_chamfer": mean_score})
            print(f"[scale-search]   scale={float(scale):.10f} mean_chamfer={mean_score:.10f}")
            if mean_score < best_score:
                best_score = mean_score
                best_scale = float(scale)

        finite = [row for row in level_rows if np.isfinite(row["mean_chamfer"])]
        history.append(
            {
                "level": int(level + 1),
                "interval": [float(lo), float(hi)],
                "candidates": level_rows,
                "best_scale": float(min(finite, key=lambda r: r["mean_chamfer"])["scale"]) if finite else None,
                "best_mean_chamfer": float(min(finite, key=lambda r: r["mean_chamfer"])["mean_chamfer"]) if finite else None,
            }
        )
        if not finite or level == levels - 1:
            continue

        sorted_scales = np.asarray([row["scale"] for row in level_rows], dtype=np.float64)
        sorted_scores = np.asarray([row["mean_chamfer"] for row in level_rows], dtype=np.float64)
        order = np.argsort(sorted_scales)
        sorted_scales = sorted_scales[order]
        sorted_scores = sorted_scores[order]
        best_idx = int(np.nanargmin(sorted_scores))
        left_idx = max(0, best_idx - 1)
        right_idx = min(sorted_scales.shape[0] - 1, best_idx + 1)
        if left_idx == best_idx and right_idx < sorted_scales.shape[0] - 1:
            right_idx += 1
        if right_idx == best_idx and left_idx > 0:
            left_idx -= 1
        lo = float(sorted_scales[left_idx])
        hi = float(sorted_scales[right_idx])

    if not np.isfinite(best_score):
        raise RuntimeError("Could not compute a valid chamfer scale search.")
    print(f"[scale-search] best scale={best_scale:.10f} mean_chamfer={best_score:.10f}")
    return best_scale, best_score, history


def _point_cloud_std(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] < 2:
        return float("nan")
    centered = points - np.mean(points, axis=0, keepdims=True)
    return float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))


def _build_point_cloud(points: np.ndarray, color: Tuple[float, float, float]):
    if o3d is None:
        return None
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] == 0:
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(list(color))
    return pcd


def _colored_point_cloud(points: np.ndarray, colors: np.ndarray | Tuple[float, float, float]):
    if o3d is None:
        return None
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if isinstance(colors, tuple):
        pcd.paint_uniform_color(list(colors))
    else:
        colors = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
        if colors.shape[0] != points.shape[0]:
            colors = _stable_colors(points.shape[0], seed=17)
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    return pcd


def _stable_colors(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    colors = rng.uniform(0.25, 1.0, size=(max(int(count), 0), 3))
    colors[:, 2] = np.maximum(colors[:, 2], 0.55)
    return colors


def _load_edg_points(edg_tracks: dict, frame_id: int) -> tuple[np.ndarray, list[int]]:
    pts = []
    gids = []
    for gid in sorted(int(k) for k in edg_tracks.keys()):
        tr = np.asarray(edg_tracks[gid], dtype=np.float64)
        if tr.ndim == 2 and frame_id < tr.shape[0] and np.isfinite(tr[frame_id]).all():
            pts.append(tr[frame_id])
            gids.append(gid)
    if not pts:
        return np.zeros((0, 3), dtype=np.float64), []
    return np.asarray(pts, dtype=np.float64), gids


def _load_edges(data: np.lib.npyio.NpzFile) -> list[tuple[int, int]]:
    for key in ("edg_edges_union", "prior_edges_global", "edg_edges"):
        if key not in data:
            continue
        arr = np.asarray(data[key])
        if arr.size == 0:
            continue
        flat = arr.reshape(-1)
        if flat.size < 2:
            continue
        if flat.size % 2 != 0:
            print(f"[vis] warning: {key} has odd size {flat.size}; dropping last value")
            flat = flat[: flat.size - 1]
        pairs = flat.reshape(-1, 2)
        return [(int(a), int(b)) for a, b in pairs.tolist()]
    return []


def _make_edg_edges(points: np.ndarray, gids: list[int], edges: list[tuple[int, int]]):
    if o3d is None:
        return None
    gid_to_idx = {int(gid): idx for idx, gid in enumerate(gids)}
    line_idx = [[gid_to_idx[a], gid_to_idx[b]] for a, b in edges if a in gid_to_idx and b in gid_to_idx]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64).reshape(-1, 3))
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(line_idx, dtype=np.int32).reshape(-1, 2))
    if line_idx:
        line_set.paint_uniform_color([0.0, 0.0, 0.0])
    return line_set


def _update_point_cloud(pcd, points: np.ndarray, colors: np.ndarray | Tuple[float, float, float]):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    pcd.points = o3d.utility.Vector3dVector(points)
    if isinstance(colors, tuple):
        pcd.paint_uniform_color(list(colors))
    else:
        colors = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
        if colors.shape[0] != points.shape[0]:
            colors = _stable_colors(points.shape[0], seed=17)
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))


def _show_animated_world_viewer(
    *,
    data: np.lib.npyio.NpzFile,
    edg_tracks: dict,
    edg_rot_tracks: dict,
    solver_c2w: np.ndarray,
    gt_paths: list[Path | None],
    scale_ate: float,
    R_ate: np.ndarray,
    t_ate: np.ndarray,
    prior_pts: np.ndarray | None,
    target_points: int,
    chamfer_scale: float,
    start_frame: int,
    point_size: float,
) -> None:
    if o3d is None:
        print("[vis] Open3D is not available; skipping visualization.")
        return

    n_frames = min(
        max(int(arr.shape[0]) for arr in edg_tracks.values()) if edg_tracks else 0,
        len(gt_paths),
        solver_c2w.shape[0],
    )
    if n_frames <= 0:
        print("[vis] no frames available for visualization.")
        return

    static_landmarks = data.get("static_landmarks")
    static_pts = np.zeros((0, 3), dtype=np.float64)
    if static_landmarks is not None and np.asarray(static_landmarks).size > 0:
        static_pts = np.asarray(static_landmarks, dtype=np.float64).reshape(-1, 3)
        static_pts = static_pts[np.isfinite(static_pts).all(axis=1)]
        static_pts = _sample_exact(static_pts, 30000, seed=404)

    edges = _load_edges(data)
    state = {"frame": int(np.clip(start_frame, 0, n_frames - 1))}

    def frame_clouds(frame_id: int):
        dyn_pts, *_ = _load_dynamic_points(frame_id, data, edg_tracks, edg_rot_tracks)
        dyn_pts, _ = _augment_with_prior(dyn_pts, prior_pts, target_points)
        dyn_pts = _sample_exact(dyn_pts, target_points, seed=2)
        gt_pts = _load_gt_points(gt_paths, frame_id)
        gt_pts = _sample_exact(gt_pts, target_points, seed=3)
        gt_pts = _apply_inverse_similarity(gt_pts, scale_ate, R_ate, t_ate)
        edg_pts, gids = _load_edg_points(edg_tracks, frame_id)
        R_w2c, t_w2c = _c2w_to_w2c(solver_c2w[frame_id])
        dyn_cam = ((R_w2c @ dyn_pts.T).T + t_w2c.reshape(1, 3)) * float(chamfer_scale)
        gt_cam = (R_w2c @ gt_pts.T).T + t_w2c.reshape(1, 3)
        edg_cam = (R_w2c @ edg_pts.T).T + t_w2c.reshape(1, 3) if edg_pts.shape[0] > 0 else edg_pts
        return dyn_cam, gt_cam, edg_cam, gids

    dyn0, gt0, edg0, gids0 = frame_clouds(state["frame"])
    def to_solver_camera(points: np.ndarray, frame_id: int) -> np.ndarray:
        R_w2c, t_w2c = _c2w_to_w2c(solver_c2w[frame_id])
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if pts.shape[0] == 0:
            return pts
        return (R_w2c @ pts.T).T + t_w2c.reshape(1, 3)

    static0 = to_solver_camera(static_pts, state["frame"])
    static_pcd = _colored_point_cloud(static0, (0.62, 0.62, 0.62))
    dyn_pcd = _colored_point_cloud(dyn0, (1.0, 0.05, 0.05))
    gt_pcd = _colored_point_cloud(gt0, (0.1, 0.85, 0.25))
    edg_pcd = _colored_point_cloud(edg0, (1.0, 0.55, 0.05))
    edge_geom = _make_edg_edges(edg0, gids0, edges)
    cam_geom = _camera_axes_lines(np.eye(4, dtype=np.float64), axis_len=0.08)

    def refresh(vis):
        frame = state["frame"]
        dyn_pts, gt_pts, edg_pts, gids = frame_clouds(frame)
        _update_point_cloud(static_pcd, to_solver_camera(static_pts, frame), (0.62, 0.62, 0.62))
        _update_point_cloud(dyn_pcd, dyn_pts, (1.0, 0.05, 0.05))
        _update_point_cloud(gt_pcd, gt_pts, (0.1, 0.85, 0.25))
        _update_point_cloud(edg_pcd, edg_pts, (1.0, 0.55, 0.05))
        new_edges = _make_edg_edges(edg_pts, gids, edges)
        edge_geom.points = new_edges.points
        edge_geom.lines = new_edges.lines
        edge_geom.colors = new_edges.colors
        new_cam = _camera_axes_lines(np.eye(4, dtype=np.float64), axis_len=0.08)
        cam_geom.points = new_cam.points
        cam_geom.lines = new_cam.lines
        cam_geom.colors = new_cam.colors
        update_geoms = [dyn_pcd, gt_pcd, edg_pcd, edge_geom, cam_geom]
        if static_pts.shape[0] > 0:
            update_geoms.insert(0, static_pcd)
        for geom in update_geoms:
            vis.update_geometry(geom)
        frame_chamfer = _symmetric_chamfer(dyn_pts, gt_pts)
        print(f"[Open3D] frame={frame}/{n_frames - 1} metric_chamfer={frame_chamfer:.10f}")
        return False

    def on_space(vis):
        state["frame"] = (state["frame"] + 1) % n_frames
        return refresh(vis)

    def on_b(vis):
        state["frame"] = (state["frame"] - 1) % n_frames
        return refresh(vis)

    def on_q(vis):
        vis.close()
        return False

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Ours Chamfer / EDG Viewer", width=1400, height=900)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2))
    if static_pts.shape[0] > 0:
        vis.add_geometry(static_pcd)
    for geom in (dyn_pcd, gt_pcd, edg_pcd, edge_geom, cam_geom):
        vis.add_geometry(geom)
    opt = vis.get_render_option()
    opt.point_size = float(point_size)
    print("Metric view: red=predicted cloud used in Chamfer, green=GT cloud used in Chamfer.")
    print("Helpers: static=gray, EDG=orange, edges=black. Keys: space=next frame, b=previous, q=quit.")
    print(f"[Open3D] frame={state['frame']}/{n_frames - 1} metric_chamfer={_symmetric_chamfer(dyn0, gt0):.10f}")
    vis.register_key_callback(ord(" "), on_space)
    vis.register_key_callback(ord("b"), on_b)
    vis.register_key_callback(ord("B"), on_b)
    vis.register_key_callback(ord("q"), on_q)
    vis.register_key_callback(ord("Q"), on_q)
    vis.register_key_callback(256, on_q)
    vis.run()
    vis.destroy_window()


def main() -> None:
    args = _parse_args()
    dataset_root = (
        _find_scene_root(args.scene_path) if args.scene_path is not None else definitions.rootpath
    ).resolve()
    output_dir = dataset_root / "outputs"
    results_npz = _find_latest_results_npz(output_dir)
    result_xm = output_dir / "result_XM.npz"

    if not results_npz.exists():
        raise FileNotFoundError(f"Missing results npz: {results_npz}")
    if not result_xm.exists():
        raise FileNotFoundError(f"Missing result_XM.npz: {result_xm}")

    data = np.load(results_npz, allow_pickle=True)
    edg_tracks = _load_dict(data, "edg_reconstruct")
    edg_rot_tracks = _load_dict(data, "edg_rotation")
    if edg_tracks is None or edg_rot_tracks is None:
        raise RuntimeError("Missing edg_reconstruct/edg_rotation in results npz.")
    solver_c2w = _load_solved_c2w(result_xm)
    gt_c2w = _load_gt_c2w(dataset_root)
    n_pose = min(solver_c2w.shape[0], gt_c2w.shape[0])
    if n_pose < 1:
        raise RuntimeError("Not enough camera poses for alignment.")

    solver_c2w = solver_c2w[:n_pose]
    gt_c2w = gt_c2w[:n_pose]

    R_est = np.hstack([_c2w_to_w2c(pose)[0] for pose in solver_c2w])
    t_est = np.stack([_c2w_to_w2c(pose)[1] for pose in solver_c2w], axis=1)
    R_gt = np.hstack([_c2w_to_w2c(pose)[0] for pose in gt_c2w])
    t_gt = np.stack([_c2w_to_w2c(pose)[1] for pose in gt_c2w], axis=1)

    scale_ate, R_ate, t_ate = ATE_TEASER(R_est, t_est, R_gt, t_gt)
    scale_ate = float(scale_ate)
    t_ate = np.asarray(t_ate, dtype=np.float64).reshape(3)

    # Match MoSca's evaluator for degenerate/line-like camera trajectories:
    # keep TEASER's scale, but use the camera-orientation best-fit rotation and
    # recompute translation from camera centers. This avoids a weak roll/twist
    # ambiguity making the first-frame clouds appear misaligned.
    scale_gt_to_solver = 1.0 / max(scale_ate, 1e-12)
    R_gt_to_solver = _best_fit_rotation(gt_c2w, solver_c2w)
    c_solver = solver_c2w[:, :3, 3]
    c_gt = gt_c2w[:, :3, 3]
    t_gt_to_solver = np.mean(
        c_solver - scale_gt_to_solver * (R_gt_to_solver @ c_gt.T).T,
        axis=0,
    )
    R_ate = R_gt_to_solver.T
    t_ate = -scale_ate * (R_ate @ t_gt_to_solver)

    gt_c2w_aligned = []
    for pose in gt_c2w:
        R_gt_pose = pose[:3, :3]
        c_gt_pose = pose[:3, 3]
        R_aligned = np.asarray(R_ate, dtype=np.float64).T @ R_gt_pose
        c_aligned = _apply_inverse_similarity(c_gt_pose.reshape(1, 3), scale_ate, R_ate, t_ate).reshape(3)
        aligned = np.eye(4, dtype=np.float64)
        aligned[:3, :3] = R_aligned
        aligned[:3, 3] = c_aligned
        gt_c2w_aligned.append(aligned)
    gt_c2w_aligned = np.stack(gt_c2w_aligned, axis=0)

    max_frames = max(int(arr.shape[0]) for arr in edg_tracks.values()) if edg_tracks else 0
    gt_paths = _resolve_gt_geometry_paths(args.gt_mesh, dataset_root, max_frames)
    if gt_paths is None:
        raise FileNotFoundError("No GT mesh/point-cloud files found; pass --gt-mesh.")

    frame_stride = max(1, int(args.frame_stride))
    prior_pts = _load_prior_points(data)

    solver_cam_pts = []
    gt_cam_pts = []
    rows = []

    for frame_id in range(0, min(max_frames, len(gt_paths), n_pose), frame_stride):
        dyn_pts, *_ = _load_dynamic_points(frame_id, data, edg_tracks, edg_rot_tracks)
        # Do NOT pad with prior points: the prior cloud is a single static rest-pose
        # cloud that never deforms per frame, so padding drags a dynamic-vs-GT chamfer
        # upward as the object moves. Use only the deformed dynamic points.
        prior_added = 0
        dyn_pts = _sample_exact(dyn_pts, args.target_points, seed=2)

        gt_pts = _load_gt_points(gt_paths, frame_id)
        if gt_pts.shape[0] == 0:
            continue
        gt_pts = _sample_exact(gt_pts, args.target_points, seed=3)
        gt_pts_aligned = _apply_inverse_similarity(gt_pts, scale_ate, R_ate, t_ate)

        solver_R_w2c, solver_t_w2c = _c2w_to_w2c(solver_c2w[frame_id])

        # Match MoSca's metric convention: after aligning GT points into solver
        # world coordinates, evaluate both clouds in the same solver camera frame.
        solver_cam = (solver_R_w2c @ dyn_pts.T).T + solver_t_w2c.reshape(1, 3)
        gt_cam = (solver_R_w2c @ gt_pts_aligned.T).T + solver_t_w2c.reshape(1, 3)

        solver_cam_pts.append(solver_cam)
        gt_cam_pts.append(gt_cam)
        rows.append({
            "frame": int(frame_id),
            "solver_points": int(dyn_pts.shape[0]),
            "gt_points": int(gt_pts.shape[0]),
            "prior_points_added": int(prior_added),
        })

    if not solver_cam_pts:
        raise RuntimeError("No valid frames found for chamfer evaluation.")

    chamfer_scale, chamfer_score, scale_search_history = _search_global_scale_multilevel(
        solver_cam_pts,
        gt_cam_pts,
        scale_min=float(args.scale_min),
        scale_max=float(args.scale_max),
        scale_steps=int(args.scale_steps),
        scale_levels=int(args.scale_levels),
    )
    

    for row, src, tgt in zip(rows, solver_cam_pts, gt_cam_pts):
        chamfer = float(_symmetric_chamfer(src * chamfer_scale, tgt))
        gt_std = _point_cloud_std(tgt)
        row["chamfer"] = chamfer
        row["gt_std"] = float(gt_std)
        row["chamfer_norm"] = float(chamfer / gt_std)

    result = {
        "results_npz": str(results_npz),
        "dataset_root": str(dataset_root),
        "frame_stride": int(frame_stride),
        "target_points": int(args.target_points),
        "chamfer_scale": float(chamfer_scale),
        "scale_search_score": float(chamfer_score),
        "scale_search_history": scale_search_history,
        "mean_chamfer": float(np.mean([row["chamfer"] for row in rows])),
        "mean_chamfer_norm": float(np.mean([row["chamfer_norm"] for row in rows])),
        "alignment_ate": {
            "scale": float(scale_ate),
            "R": np.asarray(R_ate).tolist(),
            "t": np.asarray(t_ate).tolist(),
        },
        "rows": rows,
    }

    output_json = args.output_json or (output_dir / "ours_chamfer.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[chamfer] saved {output_json}")

    if args.no_vis or not args.visualize or not args.show_open3d:
        if args.visualize and not args.show_open3d and not args.no_vis:
            print("[vis] --visualize requested; Open3D window is disabled by default. Use --show-open3d to open it.")
        return

    _show_animated_world_viewer(
        data=data,
        edg_tracks=edg_tracks,
        edg_rot_tracks=edg_rot_tracks,
        solver_c2w=solver_c2w,
        gt_paths=gt_paths,
        scale_ate=scale_ate,
        R_ate=R_ate,
        t_ate=t_ate,
        prior_pts=prior_pts,
        target_points=int(args.target_points),
        chamfer_scale=float(chamfer_scale),
        start_frame=int(args.frame),
        point_size=float(args.point_size),
    )


if __name__ == "__main__":
    main()
