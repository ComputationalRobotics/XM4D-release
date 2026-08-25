#!/usr/bin/env python3
"""Visualize solved EDG result (static landmarks, EDG graph, skinned dynamic points,
GT geometry, cameras) using Viser.  Navigate frames with the slider or Prev/Next buttons.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import utils.definitions
from utils.readgt_colmap import load_colmap_gt
from utils.readgt_DAVIS import load_davis_gt
from utils.readgt_sintel import load_sintel_gt
from utils.viser_vis import block_until_ctrl_c, get_viser_server

# ── Viser server config ───────────────────────────────────────────────────────
VISER_HOST = os.environ.get("VISER_HOST", "0.0.0.0")
VISER_PORT = int(os.environ.get("VISER_PORT", "8080"))

# ── Auto-size defaults ────────────────────────────────────────────────────────
AUTO_SIZE_KEEP_PERCENTILE = 95.0
AUTO_POINT_SIZE_FRAC      = 0.004
AUTO_POINT_SIZE_MIN       = 0.002
AUTO_POINT_SIZE_MAX       = 0.05
AUTO_AXIS_LEN_FRAC        = 0.04
AUTO_AXIS_LEN_MIN         = 0.01
AUTO_AXIS_LEN_MAX         = 0.5


# ── CLI ───────────────────────────────────────────────────────────────────────

def _default_results_path() -> Path:
    return Path(utils.definitions.output_dir) / "results_all.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize solved EDG result with Viser."
    )
    parser.add_argument("--npz", type=Path, default=_default_results_path())
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument(
        "--result-xm",
        type=Path,
        default=Path(utils.definitions.output_dir) / "result_XM.npz",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(utils.definitions.rootpath),
    )
    parser.add_argument("--gt-mesh", type=Path, default=None)
    parser.add_argument("--gt-mesh-voxel-size", type=float, default=0.0)
    parser.add_argument("--gt-mesh-sample-points", type=int, default=1000)
    parser.add_argument("--hide-gt-mesh", action="store_true")
    parser.add_argument(
        "--coordinate-space",
        choices=("robocasa", "solver"),
        default="robocasa",
    )
    parser.add_argument("--show-cameras", action="store_true")
    parser.add_argument("--camera-axis-len", type=float, default=0.0,
                        help="Camera axis length (0 = auto).")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--scale-search", action="store_true", default=True)
    parser.add_argument("--no-scale-search", dest="scale_search", action="store_false")
    parser.add_argument("--scale-min", type=float, default=0.2)
    parser.add_argument("--scale-max", type=float, default=5.0)
    parser.add_argument("--scale-steps", type=int, default=21)
    parser.add_argument("--scale-iters", type=int, default=3)
    parser.add_argument("--scale-search-frame", type=int, default=None)
    parser.add_argument("--chamfer-max-points", type=int, default=20000)
    parser.add_argument("--viser-port", type=int, default=VISER_PORT)
    parser.add_argument("--viser-host", type=str, default=VISER_HOST)
    return parser.parse_args()


# ── Data helpers (unchanged from open3d version) ──────────────────────────────

def _load_dict(data, key):
    value = data.get(key)
    if value is None:
        return None
    return value.item()


def _load_tracks(track_dict, frame_id: int):
    points, gids = [], []
    for key, arr in track_dict.items():
        gid = int(key)
        arr = np.asarray(arr, dtype=np.float64)
        if frame_id < 0 or frame_id >= arr.shape[0]:
            continue
        pt = arr[frame_id]
        if np.isfinite(pt).all():
            gids.append(gid)
            points.append(pt)
    if not points:
        return np.zeros((0, 3), dtype=np.float64), []
    return np.asarray(points, dtype=np.float64), gids


def _load_edges(edges_obj):
    if edges_obj is None:
        return []
    if isinstance(edges_obj, (list, tuple, set)):
        iterable = edges_obj
    else:
        iterable = np.asarray(edges_obj, dtype=object).tolist()
    edges = []
    for item in iterable:
        if item is None:
            continue
        a, b = int(item[0]), int(item[1])
        if a != b:
            edges.append((min(a, b), max(a, b)))
    return sorted(set(edges))


def _dynamic_point_at_frame(
    *, gid, frame_id, skin_nodes, skin_weights, bind_rest, bind_frame,
    edg_tracks, edg_rot_tracks,
):
    if bind_frame < 0 or not np.isfinite(bind_rest).all():
        return None
    accum = np.zeros(3, dtype=np.float64)
    w_accum = 0.0
    for node_gid, weight in zip(skin_nodes.tolist(), skin_weights.tolist()):
        node_gid, weight = int(node_gid), float(weight)
        if node_gid < 0 or not np.isfinite(weight) or weight <= 0.0:
            continue
        if node_gid not in edg_tracks or node_gid not in edg_rot_tracks:
            print(f"Warning: skinning node not in EDG tracks for gid={gid}.")
            continue
        node_track = np.asarray(edg_tracks[node_gid], dtype=np.float64)
        node_rot   = np.asarray(edg_rot_tracks[node_gid], dtype=np.float64)
        if (bind_frame >= node_track.shape[0] or frame_id >= node_track.shape[0]
                or bind_frame >= node_rot.shape[0] or frame_id >= node_rot.shape[0]):
            continue
        x_bind, x_frame = node_track[bind_frame], node_track[frame_id]
        r_bind, r_frame = node_rot[bind_frame], node_rot[frame_id]
        if not (np.isfinite(x_bind).all() and np.isfinite(x_frame).all()
                and np.isfinite(r_bind).all() and np.isfinite(r_frame).all()):
            continue
        rel    = r_bind.T @ (bind_rest - x_bind)
        mapped = x_frame + r_frame @ rel
        accum    += weight * mapped
        w_accum  += weight
    if w_accum <= 1e-12:
        return None
    return accum / w_accum


def _load_dynamic_points(frame_id: int, data, edg_tracks, edg_rot_tracks):
    dynamic_global_ids = np.asarray(data.get("dynamic_global_ids"), dtype=np.int64)
    skin_nodes   = np.asarray(data.get("dynamic_skinning_nodes_array"),    dtype=np.int64)
    skin_weights = np.asarray(data.get("dynamic_skinning_weights_array"),  dtype=np.float64)
    bind_rest    = np.asarray(data.get("dynamic_skinning_rest_array"),     dtype=np.float64)
    bind_frame   = np.asarray(data.get("dynamic_skinning_bind_frame_array"), dtype=np.int64)
    assigned     = np.asarray(data.get("dynamic_skinning_assigned_mask"),  dtype=bool)
    if dynamic_global_ids.size == 0:
        return np.zeros((0, 3), dtype=np.float64), 0, 0
    points, skinned_now, bound_total = [], 0, int(np.sum(assigned))
    for gid in dynamic_global_ids.tolist():
        gid = int(gid)
        if gid >= assigned.shape[0] or not bool(assigned[gid]):
            continue
        pt = _dynamic_point_at_frame(
            gid=gid, frame_id=frame_id,
            skin_nodes=skin_nodes[gid], skin_weights=skin_weights[gid],
            bind_rest=bind_rest[gid], bind_frame=int(bind_frame[gid]),
            edg_tracks=edg_tracks, edg_rot_tracks=edg_rot_tracks,
        )
        if pt is not None:
            points.append(pt)
            skinned_now += 1
    if not points:
        return np.zeros((0, 3), dtype=np.float64), skinned_now, bound_total
    return np.asarray(points, dtype=np.float64), skinned_now, bound_total


# ── Camera / similarity math ──────────────────────────────────────────────────

def _normalize_rotation(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64))
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


def _rot_to_wxyz(R: np.ndarray) -> np.ndarray:
    q = Rotation.from_matrix(_normalize_rotation(R)).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def _load_solved_c2w(result_xm: Path) -> np.ndarray:
    rt    = np.load(result_xm, allow_pickle=True)
    R_raw = np.asarray(rt["R"], dtype=np.float64)
    t_raw = np.asarray(rt["t"], dtype=np.float64)
    s_raw = np.asarray(rt["s"], dtype=np.float64) if "s" in rt else None
    T = int(t_raw.shape[1])
    c2w = np.tile(np.eye(4, dtype=np.float64), (T, 1, 1))
    for i in range(T):
        R = R_raw[:, 3 * i : 3 * (i + 1)]
        if s_raw is not None and i < len(s_raw) and abs(float(s_raw[i])) > 1e-12:
            R = R / float(s_raw[i])
        c2w[i, :3, :3] = _normalize_rotation(R)
        c2w[i, :3, 3]  = t_raw[:, i]
    return c2w


def _load_gt_c2w(dataset_root: Path) -> np.ndarray:
    dataset_cat = utils.definitions.dataset_cat
    if dataset_cat in {"Sintel", "Sintel_debug"}:
        gt = load_sintel_gt(str(dataset_root))
    elif dataset_cat == "DAVIS":
        gt = load_davis_gt(str(dataset_root))
    else:
        gt = load_colmap_gt(str(dataset_root / "colmap"))
    items = sorted(gt.items(), key=lambda kv: int(kv[1]["id"]))
    c2w = []
    for _, item in items:
        R_w2c = np.asarray(item.get("R"), dtype=np.float64)
        t_w2c = np.asarray(item.get("t"), dtype=np.float64)
        if R_w2c is None or t_w2c is None:
            continue
        if R_w2c.size != 9 or t_w2c.size != 3:
            continue
        R_w2c = R_w2c.reshape(3, 3)
        t_w2c = t_w2c.reshape(3)
        if not (np.isfinite(R_w2c).all() and np.isfinite(t_w2c).all()):
            continue
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = R_w2c.T
        pose[:3, 3]  = -R_w2c.T @ t_w2c
        c2w.append(pose)
    if not c2w:
        raise RuntimeError(f"No GT camera poses found under {dataset_root}")
    return np.stack(c2w, axis=0)


def _apply_similarity(points: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return float(scale) * (points @ R.T) + np.asarray(t, dtype=np.float64).reshape(1, 3)


def _apply_inverse_similarity(points: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return (points - np.asarray(t, dtype=np.float64).reshape(1, 3)) @ np.asarray(R, dtype=np.float64) / float(scale)


def _first_camera_aligned_similarity(solver_c2w, gt_c2w, scale):
    R_align = _normalize_rotation(solver_c2w[0, :3, :3] @ gt_c2w[0, :3, :3].T)
    t_align = solver_c2w[0, :3, 3] - float(scale) * (R_align @ gt_c2w[0, :3, 3])
    return float(scale), R_align, t_align


def _transform_c2w_by_similarity(c2w, scale, R_align, t_align):
    pose = np.asarray(c2w, dtype=np.float64).copy()
    pose[:3, :3] = _normalize_rotation(R_align @ pose[:3, :3])
    pose[:3, 3]  = _apply_similarity(pose[:3, 3].reshape(1, 3), scale, R_align, t_align).reshape(3)
    return pose


def _transform_c2w_by_inv_similarity(c2w, scale, R_align, t_align):
    pose = np.asarray(c2w, dtype=np.float64).copy()
    pose[:3, :3] = _normalize_rotation(R_align.T @ pose[:3, :3])
    pose[:3, 3]  = _apply_inverse_similarity(pose[:3, 3].reshape(1, 3), scale, R_align, t_align).reshape(3)
    return pose


# ── GT geometry loading ───────────────────────────────────────────────────────

def _load_geometry_points(path: Path, sample_points: int, voxel_size: float) -> np.ndarray:
    import open3d as o3d
    if not path.exists():
        return np.zeros((0, 3), dtype=np.float64)
    pts = np.zeros((0, 3), dtype=np.float64)
    if path.suffix.lower() in {".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb"}:
        pcd = o3d.io.read_point_cloud(str(path))
        pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.size == 0:
        mesh = o3d.io.read_triangle_mesh(str(path))
        if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
            mesh.compute_vertex_normals()
            if sample_points > 0:
                pcd = mesh.sample_points_uniformly(number_of_points=int(sample_points))
                pts = np.asarray(pcd.points, dtype=np.float64)
            else:
                pts = np.asarray(mesh.vertices, dtype=np.float64)
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if voxel_size > 0.0 and pts.shape[0] > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd = pcd.voxel_down_sample(float(voxel_size))
        pts = np.asarray(pcd.points, dtype=np.float64)
    return pts


def _resolve_gt_geometry_paths(gt_mesh, dataset_root, T):
    root = gt_mesh.resolve() if gt_mesh is not None else None
    if root is None:
        for name in ("dynamic_pointcloud_world", "object_pointcloud_world"):
            auto = dataset_root / name
            if auto.exists():
                root = auto
                print(f"[gt] using {name} for GT object point clouds.")
                break
    if root is None:
        return None
    if root.is_file():
        return [root for _ in range(T)]
    if not root.is_dir():
        print(f"[warning] GT mesh path does not exist: {root}")
        return None
    paths = []
    for frame in range(T):
        candidates = [
            root / f"frame{frame:06d}.ply", root / f"frame{frame:06d}.obj",
            root / f"{frame:06d}.ply",      root / f"{frame:06d}.obj",
        ]
        paths.append(next((p for p in candidates if p.exists()), None))
    if not any(p is not None for p in paths):
        print(f"[warning] no GT mesh/point-cloud files found in {root}")
        return None
    return paths


# ── Chamfer scale search ──────────────────────────────────────────────────────

def _subsample_points(points, max_points, seed=0):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    idx = np.random.default_rng(seed).choice(points.shape[0], int(max_points), replace=False)
    return points[idx]


def _symmetric_chamfer(src, dst):
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    if src.shape[0] == 0 or dst.shape[0] == 0:
        return np.inf
    d_fwd, _ = cKDTree(dst).query(src, k=1)
    d_rev, _ = cKDTree(src).query(dst, k=1)
    return float(np.mean(d_fwd ** 2) + np.mean(d_rev ** 2))


def _point_cloud_std(points):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] < 2:
        return float("nan")
    centered = points - np.mean(points, axis=0, keepdims=True)
    return float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1))))


def _line_search_gt_to_dynamic_scale(
    *, dyn_pts_solver, gt_pts, solver_c2w, gt_c2w,
    scale_min, scale_max, scale_steps, scale_iters, max_points,
):
    dyn_pts_solver = _subsample_points(dyn_pts_solver, max_points, seed=0)
    gt_pts         = _subsample_points(gt_pts, max_points, seed=1)
    if dyn_pts_solver.shape[0] == 0 or gt_pts.shape[0] == 0:
        print("[scale-search] skipped: empty dynamic or GT point cloud.")
        return None
    scale_steps = max(2, int(scale_steps))
    scale_iters = max(1, int(scale_iters))
    lo = max(min(float(scale_min), float(scale_max)), 1e-6)
    hi = max(float(scale_min), float(scale_max))
    gt_std = _point_cloud_std(gt_pts)
    best = None
    for pass_idx in range(scale_iters):
        scales = np.geomspace(lo, hi, num=scale_steps) if lo < hi else np.asarray([lo])
        if lo <= 1.0 <= hi:
            scales = np.unique(np.concatenate([scales, np.asarray([1.0])]))
        losses = []
        for scale in scales.tolist():
            s, R, t    = _first_camera_aligned_similarity(solver_c2w, gt_c2w, float(scale))
            gt_aligned = _apply_similarity(gt_pts, s, R, t)
            raw        = _symmetric_chamfer(dyn_pts_solver, gt_aligned)
            norm       = raw / gt_std if np.isfinite(raw) and np.isfinite(gt_std) and gt_std > 1e-12 else raw
            key        = norm if np.isfinite(norm) else raw
            if np.isfinite(key):
                losses.append((s, raw, norm, R, t))
                if best is None or key < best[0]:
                    best = (key, raw, norm, s, R, t)
        if not losses:
            break
        losses.sort(key=lambda x: x[0])
        bi = min(range(len(losses)), key=lambda i: losses[i][2] if np.isfinite(losses[i][2]) else losses[i][1])
        lo = losses[max(bi - 1, 0)][0]
        hi = losses[min(bi + 1, len(losses) - 1)][0]
        print(
            f"[scale-search] pass={pass_idx+1}/{scale_iters} "
            f"best_scale={best[3]:.8f} raw_chamfer={best[1]:.8f} norm_chamfer={best[2]:.8f} "
            f"next_range=[{lo:.8f}, {hi:.8f}]"
        )
    if best is None:
        return None
    _, raw, norm, s, R, t = best
    print(
        f"[scale-search] best scale={s:.6f}, raw_chamfer={raw:.8f}, "
        f"norm_chamfer={norm:.8f}, gt_std={gt_std:.8f}, "
        f"dynamic_pts={dyn_pts_solver.shape[0]}, gt_pts={gt_pts.shape[0]}"
    )
    return s, R, t, raw, norm


# ── Viser scene helpers ───────────────────────────────────────────────────────

def _bbox_diag(arrays: list[np.ndarray], keep_percentile: float = 100.0) -> float:
    pts = [np.asarray(a).reshape(-1, 3) for a in arrays if a is not None and np.asarray(a).size > 0]
    if not pts:
        return 1.0
    all_pts = np.concatenate(pts, axis=0)
    all_pts = all_pts[np.isfinite(all_pts).all(axis=1)]
    if all_pts.shape[0] == 0:
        return 1.0
    if keep_percentile < 100.0 and all_pts.shape[0] > 16:
        center = np.median(all_pts.astype(np.float64), axis=0)
        radius = np.linalg.norm(all_pts.astype(np.float64) - center, axis=1)
        thresh = np.percentile(radius, keep_percentile)
        kept   = all_pts[radius <= thresh]
        if kept.shape[0] >= 8:
            all_pts = kept
    return float(np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0)))


def _remove(server, *paths):
    for path in paths:
        try:
            server.scene.remove(path)
        except Exception:
            pass


def _edges_to_segments(edg_pts: np.ndarray, gid_to_idx: dict, edges: list) -> np.ndarray:
    """Convert EDG edge list to (N, 2, 3) line segments for Viser."""
    segs = []
    for a, b in edges:
        if a in gid_to_idx and b in gid_to_idx:
            segs.append([edg_pts[gid_to_idx[a]], edg_pts[gid_to_idx[b]]])
    if not segs:
        return np.zeros((0, 2, 3), dtype=np.float32)
    return np.asarray(segs, dtype=np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    npz_path = args.npz.resolve()
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing results file: {npz_path}")
    print(f"Loading EDG result: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    static_landmarks = data.get("static_landmarks")
    edg_tracks      = _load_dict(data, "edg_reconstruct")
    edg_rot_tracks  = _load_dict(data, "edg_rotation")
    edges_union     = data.get("edg_edges_union")

    if edg_tracks is None or edg_rot_tracks is None:
        raise RuntimeError(f"Missing EDG tracks or rotations in {npz_path}.")
    if data.get("dynamic_skinning_nodes_array") is None:
        raise RuntimeError(
            f"Missing persistent skinning arrays in {npz_path}. "
            "Run EDG_old.py / EDG_prior.py with updated code, or point --npz at a compatible file."
        )

    max_frames = max(int(arr.shape[0]) for arr in edg_tracks.values()) if edg_tracks else 0
    if max_frames == 0:
        raise RuntimeError("No EDG tracks found to visualize.")

    edges = _load_edges(edges_union)

    # ── Camera alignment ──────────────────────────────────────────────────────
    solver_c2w = gt_c2w = align = None
    try:
        solver_c2w = _load_solved_c2w(args.result_xm)
        gt_c2w     = _load_gt_c2w(args.dataset_root)
        n_pose     = min(solver_c2w.shape[0], gt_c2w.shape[0], max_frames)
        solver_c2w = solver_c2w[:n_pose]
        gt_c2w     = gt_c2w[:n_pose]
        if n_pose >= 1:
            align = _first_camera_aligned_similarity(solver_c2w, gt_c2w, scale=1.0)
            print("[align] initialized GT-to-solver transform from first camera.")
    except Exception as exc:
        print(f"[warning] could not load/align camera poses: {exc}")

    # ── GT mesh paths ─────────────────────────────────────────────────────────
    gt_mesh_paths = None
    if not args.hide_gt_mesh:
        gt_mesh_paths = _resolve_gt_geometry_paths(args.gt_mesh, args.dataset_root, max_frames)

    # ── Chamfer scale search ──────────────────────────────────────────────────
    if args.scale_search and gt_mesh_paths is not None and solver_c2w is not None and gt_c2w is not None:
        sf = args.frame if args.scale_search_frame is None else args.scale_search_frame
        sf = int(np.clip(sf, 0, min(max_frames, len(gt_mesh_paths)) - 1))
        if gt_mesh_paths[sf] is None:
            print(f"[scale-search] skipped: no GT geometry for frame={sf}.")
        else:
            dyn_pts_s, *_ = _load_dynamic_points(sf, data, edg_tracks, edg_rot_tracks)
            gt_pts_s      = _load_geometry_points(gt_mesh_paths[sf], args.gt_mesh_sample_points, args.gt_mesh_voxel_size)
            result = _line_search_gt_to_dynamic_scale(
                dyn_pts_solver=dyn_pts_s, gt_pts=gt_pts_s,
                solver_c2w=solver_c2w, gt_c2w=gt_c2w,
                scale_min=args.scale_min, scale_max=args.scale_max,
                scale_steps=args.scale_steps, scale_iters=args.scale_iters,
                max_points=args.chamfer_max_points,
            )
            if result is not None:
                scale, R, t, raw_loss, norm_loss = result
                align = (scale, R, t)
                print(f"[align] Chamfer scale={scale:.6f} raw={raw_loss:.8f} norm={norm_loss:.8f}")

    # ── Coordinate space helpers ──────────────────────────────────────────────
    coordinate_space = args.coordinate_space
    if coordinate_space == "robocasa":
        if align is None:
            print("[warning] robocasa space requested but alignment unavailable; hiding GT.")
            gt_mesh_paths = None
        else:
            scale, _, _ = align
            print(f"[coords] raw COLMAP/world coordinates (inv scale={scale:.6f}).")
    else:
        if align is None and gt_mesh_paths is not None:
            print("[warning] solver space requested but alignment unavailable; hiding GT.")
            gt_mesh_paths = None
        print("[coords] EDG/XM solver coordinates.")

    show_gt_cameras = gt_c2w is not None and align is not None

    def _solver_pts(pts):
        if coordinate_space == "robocasa" and align is not None:
            s, R, t = align
            return _apply_inverse_similarity(pts, s, R, t)
        return pts

    def _gt_pts(pts):
        if coordinate_space == "solver" and align is not None:
            s, R, t = align
            return _apply_similarity(pts, s, R, t)
        return pts

    def _solver_pose(pose):
        if coordinate_space == "robocasa" and align is not None:
            s, R, t = align
            return _transform_c2w_by_inv_similarity(pose, s, R, t)
        return pose

    def _gt_pose(pose):
        if coordinate_space == "solver" and align is not None:
            s, R, t = align
            return _transform_c2w_by_similarity(pose, s, R, t)
        return pose

    # ── Dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        frame_id = int(np.clip(args.frame, 0, max_frames - 1))
        edg_pts, edg_gids = _load_tracks(edg_tracks, frame_id)
        dyn_pts, sk, bnd  = _load_dynamic_points(frame_id, data, edg_tracks, edg_rot_tracks)
        static_count = np.asarray(static_landmarks).reshape(-1, 3).shape[0] if static_landmarks is not None else 0
        print(
            f"[dry-run] frame={frame_id} static={static_count} edg={edg_pts.shape[0]} "
            f"dyn_skinned={sk} dyn_bound_total={bnd}"
        )
        return

    # ── Scene center ──────────────────────────────────────────────────────────
    # Use solver camera positions (in display space) to center the scene.
    # Filter out near-zero positions: frame 0 is typically anchored at (0,0,0)
    # in solver space and uninitialized frames also produce (0,0,0), which would
    # corrupt the median if included.
    _cam_positions = []
    if solver_c2w is not None:
        for i in range(solver_c2w.shape[0]):
            p = _solver_pose(solver_c2w[i])[:3, 3]
            if np.isfinite(p).all() and np.linalg.norm(p) > 1e-6:
                _cam_positions.append(p)
    if _cam_positions:
        cam_pos_arr = np.stack(_cam_positions, axis=0)
        scene_center = np.median(cam_pos_arr, axis=0)
        # camera spread drives axis_len so axes are visible relative to trajectory
        cam_spread = float(np.linalg.norm(cam_pos_arr.max(axis=0) - cam_pos_arr.min(axis=0)))
    else:
        # fall back to centroid of EDG nodes at frame 0
        _sp, _ = _load_tracks(edg_tracks, 0)
        scene_center = np.median(_solver_pts(_sp), axis=0) if _sp.size > 0 else np.zeros(3)
        cam_spread = 0.0
    print(f"[center] scene_center={scene_center}  cam_spread={cam_spread:.4g}")

    # ── Auto size ─────────────────────────────────────────────────────────────
    # Sample points from frame 0 to estimate scene scale.
    sample_pts, _ = _load_tracks(edg_tracks, 0)
    if sample_pts.size > 0:
        sample_pts = _solver_pts(sample_pts) - scene_center
    diag = _bbox_diag([sample_pts], AUTO_SIZE_KEEP_PERCENTILE)
    point_size = float(np.clip(AUTO_POINT_SIZE_FRAC * diag, AUTO_POINT_SIZE_MIN, AUTO_POINT_SIZE_MAX))
    # axis_len is driven by camera trajectory spread so it's visible even when
    # the point cloud is much denser/smaller than the camera motion range.
    _axis_ref  = max(cam_spread, diag, 1e-6)
    axis_len   = (
        float(args.camera_axis_len)
        if args.camera_axis_len > 0
        else float(np.clip(AUTO_AXIS_LEN_FRAC * _axis_ref, AUTO_AXIS_LEN_MIN, AUTO_AXIS_LEN_MAX))
    )
    print(f"[size] diag={diag:.4g} cam_spread={cam_spread:.4g} point_size={point_size:.4g} axis_len={axis_len:.4g}")

    # ── Viser server ──────────────────────────────────────────────────────────
    server = get_viser_server(host=args.viser_host, port=args.viser_port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    server.gui.add_markdown("### EDG Solution Viewer")
    server.gui.add_markdown(
        "**Colors:** yellow=static, red=EDG nodes, black=edges, "
        "blue=dynamic/skinned, green=GT mesh, solver cam=orange, GT cam=cyan"
    )

    frame_slider  = server.gui.add_slider("Frame", min=0, max=max(0, max_frames - 1), step=1,
                                          initial_value=int(np.clip(args.frame, 0, max_frames - 1)))
    btn_prev      = server.gui.add_button("Prev")
    btn_next      = server.gui.add_button("Next")
    cb_static     = server.gui.add_checkbox("Static landmarks", initial_value=True)
    cb_edg        = server.gui.add_checkbox("EDG nodes", initial_value=True)
    cb_edges      = server.gui.add_checkbox("EDG edges", initial_value=True)
    cb_dynamic    = server.gui.add_checkbox("Dynamic / skinned", initial_value=True)
    cb_gt         = server.gui.add_checkbox("GT mesh", initial_value=gt_mesh_paths is not None)
    cb_cameras    = server.gui.add_checkbox("Cameras", initial_value=args.show_cameras)

    # ── Static landmarks (added once, toggled via handle) ─────────────────────
    static_handle = None
    if static_landmarks is not None:
        static_pts = np.asarray(static_landmarks, dtype=np.float64).reshape(-1, 3)
        static_pts = static_pts[np.isfinite(static_pts).all(axis=1)]
        if static_pts.shape[0] > 0:
            static_pts = (_solver_pts(static_pts) - scene_center).astype(np.float32)
            static_handle = server.scene.add_point_cloud(
                "/edg/static",
                points=static_pts,
                colors=np.tile(np.array([255, 204, 51], dtype=np.uint8), (static_pts.shape[0], 1)),
                point_size=point_size,
                point_shape="circle",
                visible=bool(cb_static.value),
            )

    DYNAMIC_PATHS = ["/edg/nodes", "/edg/edges", "/edg/dynamic", "/edg/gt_mesh",
                     "/cameras/solver", "/cameras/gt"]

    state = {"frame": int(frame_slider.value)}

    def _refresh(frame_id: int) -> None:
        frame_id = int(np.clip(frame_id, 0, max_frames - 1))
        state["frame"] = frame_id
        _remove(server, *DYNAMIC_PATHS)

        # EDG nodes
        edg_pts, edg_gids = _load_tracks(edg_tracks, frame_id)
        edg_vis = (_solver_pts(edg_pts) - scene_center).astype(np.float32) if edg_pts.size > 0 else edg_pts

        if edg_vis.shape[0] > 0 and bool(cb_edg.value):
            server.scene.add_point_cloud(
                "/edg/nodes",
                points=edg_vis,
                colors=np.tile(np.array([230, 26, 26], dtype=np.uint8), (edg_vis.shape[0], 1)),
                point_size=point_size * 1.2,
                point_shape="circle",
                visible=True,
            )

        # EDG edges
        if edg_vis.shape[0] > 0 and edges and bool(cb_edges.value):
            gid_to_idx = {int(gid): i for i, gid in enumerate(edg_gids)}
            segs = _edges_to_segments(edg_vis, gid_to_idx, edges)
            if segs.shape[0] > 0:
                server.scene.add_line_segments(
                    "/edg/edges",
                    points=segs,
                    colors=np.array([30, 30, 30], dtype=np.uint8),
                    line_width=1.0,
                    visible=True,
                )

        # Dynamic / skinned points
        dyn_pts, skinned_now, bound_total = _load_dynamic_points(frame_id, data, edg_tracks, edg_rot_tracks)
        dyn_vis = (_solver_pts(dyn_pts) - scene_center).astype(np.float32) if dyn_pts.size > 0 else dyn_pts

        if dyn_vis.shape[0] > 0 and bool(cb_dynamic.value):
            server.scene.add_point_cloud(
                "/edg/dynamic",
                points=dyn_vis,
                colors=np.tile(np.array([0, 102, 255], dtype=np.uint8), (dyn_vis.shape[0], 1)),
                point_size=point_size,
                point_shape="circle",
                visible=True,
            )

        # GT mesh
        gt_pts_count = 0
        if (gt_mesh_paths is not None and frame_id < len(gt_mesh_paths)
                and gt_mesh_paths[frame_id] is not None and bool(cb_gt.value)):
            gt_raw = _load_geometry_points(
                gt_mesh_paths[frame_id], args.gt_mesh_sample_points, args.gt_mesh_voxel_size,
            )
            if gt_raw.size > 0:
                gt_vis = (_gt_pts(gt_raw) - scene_center).astype(np.float32)
                gt_pts_count = gt_vis.shape[0]
                server.scene.add_point_cloud(
                    "/edg/gt_mesh",
                    points=gt_vis,
                    colors=np.tile(np.array([0, 204, 51], dtype=np.uint8), (gt_vis.shape[0], 1)),
                    point_size=point_size,
                    point_shape="circle",
                    visible=True,
                )

        # Cameras
        if bool(cb_cameras.value):
            if solver_c2w is not None and frame_id < solver_c2w.shape[0]:
                pose = _solver_pose(solver_c2w[frame_id])
                server.scene.add_frame(
                    "/cameras/solver",
                    wxyz=_rot_to_wxyz(pose[:3, :3]),
                    position=(pose[:3, 3] - scene_center).astype(np.float64),
                    axes_length=axis_len,
                    axes_radius=axis_len * 0.05,
                    visible=True,
                )
            if show_gt_cameras and gt_c2w is not None and frame_id < gt_c2w.shape[0]:
                pose = _gt_pose(gt_c2w[frame_id])
                server.scene.add_frame(
                    "/cameras/gt",
                    wxyz=_rot_to_wxyz(pose[:3, :3]),
                    position=(pose[:3, 3] - scene_center).astype(np.float64),
                    axes_length=axis_len,
                    axes_radius=axis_len * 0.05,
                    visible=True,
                )

        print(
            f"[frame={frame_id}] edg={edg_vis.shape[0]} dyn_skinned={skinned_now} "
            f"dyn_bound_total={bound_total} gt_pts={gt_pts_count}"
        )

    def _set_frame(idx: int) -> None:
        idx = int(idx) % max_frames
        if int(frame_slider.value) != idx:
            frame_slider.value = idx
        _refresh(idx)

    @frame_slider.on_update
    def _(_event):
        _refresh(int(frame_slider.value))

    @btn_next.on_click
    def _(_event):
        _set_frame(state["frame"] + 1)

    @btn_prev.on_click
    def _(_event):
        _set_frame(state["frame"] - 1)

    @cb_static.on_update
    def _(_event):
        if static_handle is not None:
            try:
                static_handle.visible = bool(cb_static.value)
            except Exception:
                pass

    @cb_edg.on_update
    def _(_event):
        _refresh(state["frame"])

    @cb_edges.on_update
    def _(_event):
        _refresh(state["frame"])

    @cb_dynamic.on_update
    def _(_event):
        _refresh(state["frame"])

    @cb_gt.on_update
    def _(_event):
        _refresh(state["frame"])

    @cb_cameras.on_update
    def _(_event):
        _refresh(state["frame"])

    print(f"[viser] http://localhost:{args.viser_port}")
    print("[viser] use the Prev/Next buttons or frame slider to navigate.")
    _refresh(state["frame"])
    block_until_ctrl_c()


if __name__ == "__main__":
    main()
