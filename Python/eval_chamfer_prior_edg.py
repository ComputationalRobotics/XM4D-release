#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from scipy.spatial import cKDTree

# Make the script usable when copied into Python/ or run from the repo root.
_THIS = Path(__file__).resolve().parent
for _cand in (
    _THIS,
    _THIS / "Python",
    _THIS.parent / "Python",
    Path.cwd(),
    Path.cwd() / "Python",
):
    if (_cand / "visualize_edg_solution_open3d.py").exists():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from utils import definitions  # noqa: E402
from utils.readgt_colmap import load_colmap_gt  # noqa: E402
from utils.readgt_DAVIS import load_davis_gt  # noqa: E402
from utils.readgt_sintel import load_sintel_gt  # noqa: E402
from estimate_ours_chamfer import (  # noqa: E402
    _camera_axes_lines,
    _colored_point_cloud,
    _make_edg_edges,
    _search_global_scale_multilevel,
    _stable_colors,
    _update_point_cloud,
)

try:
    import open3d as o3d
except Exception:  # pragma: no cover - optional visualization dependency
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
        print("[Open3D] window creation disabled. Set ALLOW_OPEN3D_WINDOWS=1 to allow it.")
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


def _load_dict(data: np.lib.npyio.NpzFile, key: str):
    if key not in data:
        return None
    value = data[key]
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if not isinstance(value, dict):
        return None
    return {int(k): np.asarray(v) for k, v in value.items()}


def _normalize_rotation(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64))
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


def _load_solved_c2w(result_xm: Path) -> np.ndarray:
    rt = np.load(result_xm, allow_pickle=True)
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
        c2w[i, :3, 3] = t_raw[:, i]
    return c2w


def _load_gt_c2w(dataset_root: Path) -> np.ndarray:
    dataset_cat = definitions.dataset_cat
    if dataset_cat in {"Sintel", "Sintel_debug"}:
        gt = load_sintel_gt(str(dataset_root))
    elif dataset_cat == "DAVIS":
        gt = load_davis_gt(str(dataset_root))
    else:
        gt = load_colmap_gt(str(dataset_root / "colmap"))

    items = sorted(gt.items(), key=lambda kv: int(kv[1]["id"]))
    c2w = []
    for _, item in items:
        R_raw = item.get("R")
        t_raw = item.get("t")
        if R_raw is None or t_raw is None:
            continue
        R_w2c = np.asarray(R_raw, dtype=np.float64).reshape(3, 3)
        t_w2c = np.asarray(t_raw, dtype=np.float64).reshape(3)
        if not (np.isfinite(R_w2c).all() and np.isfinite(t_w2c).all()):
            continue
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = R_w2c.T
        pose[:3, 3] = -R_w2c.T @ t_w2c
        c2w.append(pose)
    if not c2w:
        raise RuntimeError(f"No GT camera poses found under {dataset_root}")
    return np.stack(c2w, axis=0)


def _load_geometry_points(path: Path, sample_points: int = 0, voxel_size: float = 0.0) -> np.ndarray:
    if not path.exists():
        return np.zeros((0, 3), dtype=np.float64)
    try:
        import open3d as o3d  # lazy import; --help and JSON-only logic do not need it
    except Exception as exc:
        raise RuntimeError("Open3D is required to read GT mesh/point-cloud files.") from exc

    pts = np.zeros((0, 3), dtype=np.float64)
    mesh = None
    if path.suffix.lower() in {".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb"}:
        pcd = o3d.io.read_point_cloud(str(path))
        pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.size == 0:
        mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh is not None and len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        mesh.compute_vertex_normals()
        if sample_points > 0:
            pcd = mesh.sample_points_uniformly(number_of_points=int(sample_points))
            pts = np.asarray(pcd.points, dtype=np.float64)
        else:
            pts = np.asarray(mesh.vertices, dtype=np.float64)
    elif pts.size == 0:
        pcd = o3d.io.read_point_cloud(str(path))
        pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if voxel_size > 0.0 and pts.shape[0] > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd = pcd.voxel_down_sample(float(voxel_size))
        pts = np.asarray(pcd.points, dtype=np.float64)
    return pts


def _resolve_gt_geometry_paths(gt_mesh: Path | None, dataset_root: Path, T: int):
    root = gt_mesh.resolve() if gt_mesh is not None else None
    if root is None:
        for auto_dir_name in ("dynamic_pointcloud_world", "object_pointcloud_world"):
            auto_dir = dataset_root / auto_dir_name
            if auto_dir.exists():
                root = auto_dir
                print(f"[gt] using {auto_dir_name} for GT object point clouds.")
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
            root / f"frame{frame:06d}.ply",
            root / f"frame{frame:06d}.obj",
            root / f"{frame:06d}.ply",
            root / f"{frame:06d}.obj",
        ]
        paths.append(next((p for p in candidates if p.exists()), None))
    if not any(p is not None for p in paths):
        print(f"[warning] no GT mesh/point-cloud files found in {root}")
        return None
    return paths


def _symmetric_chamfer(src: np.ndarray, dst: np.ndarray) -> float:
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    if src.shape[0] == 0 or dst.shape[0] == 0:
        return np.inf
    src_tree = cKDTree(src)
    dst_tree = cKDTree(dst)
    d_src_to_dst, _ = dst_tree.query(src, k=1)
    d_dst_to_src, _ = src_tree.query(dst, k=1)
    return float(np.mean(d_src_to_dst ** 2) + np.mean(d_dst_to_src ** 2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Chamfer by deforming first-frame prior/source points through the fixed EDG. "
            "This does not use dynamics_points_reconstruct for predicted geometry."
        )
    )
    parser.add_argument(
        "--scene-path",
        type=Path,
        default=None,
        help="Optional path inside a scene; used to auto-detect dataset root.",
    )
    parser.add_argument(
        "--results-npz",
        type=Path,
        default=None,
        help="Explicit EDG results npz. If omitted, auto-detects outputs/results_all*.npz.",
    )
    parser.add_argument(
        "--result-xm",
        type=Path,
        default=None,
        help="XM camera result. Defaults to <dataset_root>/outputs/result_XM.npz.",
    )
    parser.add_argument("--gt-mesh", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--target-points",
        type=int,
        default=1000,
        help="Number of predicted prior points and GT points to sample per frame. Use <=0 for all.",
    )
    parser.add_argument("--scale-min", type=float, default=0.2)
    parser.add_argument("--scale-max", type=float, default=5.0)
    parser.add_argument("--scale-steps", type=int, default=41)
    parser.add_argument("--scale-levels", type=int, default=3)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save/report visualization data; does not open an Open3D window unless --show-open3d is also set.",
    )
    parser.add_argument(
        "--show-open3d",
        action="store_true",
        help="Actually open the interactive Open3D viewer when --visualize is set.",
    )
    parser.add_argument("--no-vis", action="store_true")
    parser.add_argument("--camera-axis-len", type=float, default=0.08)
    parser.add_argument("--point-size", type=float, default=3.0)
    parser.add_argument(
        "--bind-frame",
        type=int,
        default=0,
        help="Frame where prior/source points are attached to EDG nodes.",
    )
    parser.add_argument(
        "--knn",
        type=int,
        default=4,
        help="Number of EDG nodes used to skin each prior/source point.",
    )
    parser.add_argument(
        "--prior-key",
        type=str,
        default="auto",
        help=(
            "NPZ key for first-frame source points. Use 'auto' to try "
            "prior_source_points_world, prior_points_world, aligned_prior_points_world, "
            "prior_edg_rest_world. Camera-frame prior keys are intentionally not auto-used."
        ),
    )
    parser.add_argument(
        "--node-ids-key",
        type=str,
        default="prior_edg_global_ids",
        help=(
            "NPZ key containing the fixed EDG node gids. If absent/incomplete, sorted edg_reconstruct keys are used."
        ),
    )
    return parser.parse_args()


def _find_scene_root(path: Path) -> Path:
    if path.is_file():
        path = path.parent
    path = path.resolve()
    for parent in [path] + list(path.parents):
        if (parent / "outputs").is_dir() and ((parent / "images").is_dir() or (parent / "masks").is_dir()):
            return parent
    return path


def _npz_has_prior_edg(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=True) as data:
            keys = set(data.files)
            return (
                "edg_reconstruct" in keys
                and "edg_rotation" in keys
                and any(
                    k in keys
                    for k in (
                        "prior_source_points_world",
                        "prior_points_world",
                        "aligned_prior_points_world",
                        "prior_edg_rest_world",
                    )
                )
            )
    except Exception:
        return False


def _find_results_npz(output_dir: Path) -> Path:
    if not output_dir.exists():
        raise FileNotFoundError(f"Missing outputs folder: {output_dir}")

    # Prefer prior-aware results if several result files exist.
    candidates = []
    for pattern in ("results_all_prior*.npz", "results_all*.npz", "*.npz"):
        candidates.extend(output_dir.glob(pattern))
    candidates = sorted(set(candidates), key=lambda p: (p.stat().st_mtime, str(p)))

    valid = [p for p in candidates if _npz_has_prior_edg(p)]
    if valid:
        return valid[-1]

    detail = "\n".join(f"  {p}" for p in candidates[:20])
    raise FileNotFoundError(
        "No EDG results npz with edg_reconstruct/edg_rotation and prior source points found."
        + (f"\nChecked:\n{detail}" if detail else "")
    )


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


def _load_prior_source_points(data: np.lib.npyio.NpzFile, key: str) -> tuple[np.ndarray, str]:
    if key != "auto":
        if key not in data:
            raise KeyError(f"Requested prior key '{key}' is not in results npz.")
        if key.endswith("_camera") or "camera" in key:
            raise ValueError(
                f"Requested prior key '{key}' appears to be camera-frame data. "
                "This evaluator expects first-frame prior/source points in EDG/solver world coordinates. "
                "Use prior_source_points_world or convert the camera-frame prior before evaluation."
            )
        pts = np.asarray(data[key], dtype=np.float64).reshape(-1, 3)
        return pts[np.isfinite(pts).all(axis=1)], key

    for candidate in (
        "prior_source_points_world",
        "prior_points_world",
        "aligned_prior_points_world",
        "prior_edg_rest_world",
    ):
        if candidate in data:
            pts = np.asarray(data[candidate], dtype=np.float64).reshape(-1, 3)
            pts = pts[np.isfinite(pts).all(axis=1)]
            if pts.shape[0] > 0:
                return pts, candidate

    raise RuntimeError(
        "Could not find prior/source points. Expected one of: "
        "prior_source_points_world, prior_points_world, aligned_prior_points_world, "
        "prior_edg_rest_world."
    )


def _sample_exact(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if count <= 0 or pts.shape[0] == 0:
        return pts
    rng = np.random.default_rng(seed)
    idx = rng.choice(pts.shape[0], size=int(count), replace=pts.shape[0] < int(count))
    return pts[idx]


def _point_cloud_std(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 2:
        return float("nan")
    centered = pts - np.mean(pts, axis=0, keepdims=True)
    return float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))


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


def _camera_axes_lines(c2w: np.ndarray, axis_len: float):
    if o3d is None:
        return None
    c2w = np.asarray(c2w, dtype=np.float64)
    center = c2w[:3, 3]
    R = c2w[:3, :3]
    endpoints = np.stack(
        [
            center,
            center + R[:, 0] * float(axis_len),
            center,
            center + R[:, 1] * float(axis_len),
            center,
            center + R[:, 2] * float(axis_len),
        ],
        axis=0,
    )
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(endpoints)
    line_set.lines = o3d.utility.Vector2iVector(np.asarray([[0, 1], [2, 3], [4, 5]], dtype=np.int32))
    line_set.colors = o3d.utility.Vector3dVector(
        np.asarray([[1.0, 0.0, 0.0], [0.0, 0.8, 0.0], [0.0, 0.2, 1.0]], dtype=np.float64)
    )
    return line_set


def _choose_edg_node_ids(
    data: np.lib.npyio.NpzFile,
    edg_tracks: dict,
    edg_rot_tracks: dict,
    node_ids_key: str,
) -> np.ndarray:
    track_keys = set(int(k) for k in edg_tracks.keys())
    rot_keys = set(int(k) for k in edg_rot_tracks.keys())
    common_keys = track_keys & rot_keys

    if node_ids_key in data:
        ids = np.asarray(data[node_ids_key], dtype=np.int64).reshape(-1)
        ids = np.asarray([int(g) for g in ids.tolist() if int(g) in common_keys], dtype=np.int64)
        if ids.size > 0:
            return ids

    # Fallback: all EDG nodes that have both positions and rotations.
    return np.asarray(sorted(common_keys), dtype=np.int64)


def _stack_edg_tracks(
    edg_tracks: dict,
    edg_rot_tracks: dict,
    node_ids: Iterable[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.asarray([int(g) for g in node_ids], dtype=np.int64)
    if ids.size == 0:
        raise RuntimeError("No EDG node ids found.")

    tracks = []
    rots = []
    kept = []
    for gid in ids.tolist():
        if gid not in edg_tracks or gid not in edg_rot_tracks:
            continue
        tr = np.asarray(edg_tracks[gid], dtype=np.float64)
        rr = np.asarray(edg_rot_tracks[gid], dtype=np.float64)
        if tr.ndim != 2 or tr.shape[1] != 3:
            continue
        if rr.ndim != 3 or rr.shape[1:] != (3, 3):
            continue
        T = min(tr.shape[0], rr.shape[0])
        if T <= 0:
            continue
        tracks.append(tr[:T])
        rots.append(rr[:T])
        kept.append(gid)

    if not kept:
        raise RuntimeError("No valid EDG tracks/rotations after filtering node ids.")

    T_min = min(a.shape[0] for a in tracks)
    tracks_arr = np.stack([a[:T_min] for a in tracks], axis=0)
    rots_arr = np.stack([a[:T_min] for a in rots], axis=0)
    return np.asarray(kept, dtype=np.int64), tracks_arr, rots_arr


def _compute_skinning_from_bind_frame(
    prior_points_bind: np.ndarray,
    edg_tracks: np.ndarray,
    edg_rots: np.ndarray,
    bind_frame: int,
    knn: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Attach every first-frame prior point to fixed EDG topology using KNN at bind_frame.

    Returns local EDG node indices and normalized inverse-distance weights.
    """
    if bind_frame < 0 or bind_frame >= edg_tracks.shape[1]:
        raise ValueError(f"bind_frame {bind_frame} out of range [0, {edg_tracks.shape[1] - 1}].")
    if knn <= 0:
        raise ValueError("knn must be positive.")

    node_bind = np.asarray(edg_tracks[:, bind_frame, :], dtype=np.float64)
    rot_bind = np.asarray(edg_rots[:, bind_frame, :, :], dtype=np.float64)
    node_ok = np.isfinite(node_bind).all(axis=1) & np.isfinite(rot_bind).all(axis=(1, 2))
    valid_node_ids = np.where(node_ok)[0]
    if valid_node_ids.size == 0:
        raise RuntimeError(f"No finite EDG nodes at bind_frame={bind_frame}.")

    k_eff = min(int(knn), int(valid_node_ids.size))
    tree = cKDTree(node_bind[valid_node_ids])
    dist, nn_compact = tree.query(prior_points_bind, k=k_eff)
    dist = np.asarray(dist, dtype=np.float64)
    nn_compact = np.asarray(nn_compact, dtype=np.int64)
    if k_eff == 1:
        dist = dist[:, None]
        nn_compact = nn_compact[:, None]
    nn = valid_node_ids[nn_compact]

    weights = np.zeros_like(dist, dtype=np.float64)
    exact = dist <= 1e-8
    any_exact = np.any(exact, axis=1)
    if np.any(any_exact):
        first_exact = np.argmax(exact[any_exact], axis=1)
        weights[any_exact, first_exact] = 1.0
    if np.any(~any_exact):
        inv = 1.0 / (dist[~any_exact] + 1e-8)
        weights[~any_exact] = inv / np.sum(inv, axis=1, keepdims=True)

    return nn.astype(np.int64), weights.astype(np.float64)


def _deform_bound_points_with_edg(
    prior_points_bind: np.ndarray,
    edg_tracks: np.ndarray,
    edg_rots: np.ndarray,
    skin_idx: np.ndarray,
    skin_w: np.ndarray,
    frame: int,
    bind_frame: int,
) -> np.ndarray:
    """
    EDG warp for arbitrary bound points:

        p_f = sum_j w_j * (g_j(f) + R_j(f) R_j(bind)^T (p_bind - g_j(bind)))

    Invalid node contributions are dropped and remaining weights are renormalized per point.
    """
    if frame < 0 or frame >= edg_tracks.shape[1]:
        return np.zeros((0, 3), dtype=np.float64)

    p = np.asarray(prior_points_bind, dtype=np.float64).reshape(-1, 3)
    idx = np.asarray(skin_idx, dtype=np.int64)
    w = np.asarray(skin_w, dtype=np.float64)

    g_bind = edg_tracks[idx, bind_frame, :]      # (N,K,3)
    g_frame = edg_tracks[idx, frame, :]          # (N,K,3)
    R_bind = edg_rots[idx, bind_frame, :, :]     # (N,K,3,3)
    R_frame = edg_rots[idx, frame, :, :]         # (N,K,3,3)

    valid = (
        np.isfinite(g_bind).all(axis=2)
        & np.isfinite(g_frame).all(axis=2)
        & np.isfinite(R_bind).all(axis=(2, 3))
        & np.isfinite(R_frame).all(axis=(2, 3))
        & np.isfinite(w)
        & (w > 0.0)
    )

    delta_bind = p[:, None, :] - g_bind
    rel = np.einsum("nkji,nkj->nki", R_bind, delta_bind)
    mapped = g_frame + np.einsum("nkij,nkj->nki", R_frame, rel)
    mapped[~valid] = 0.0

    w_valid = np.where(valid, w, 0.0)
    w_sum = np.sum(w_valid, axis=1, keepdims=True)
    point_ok = w_sum[:, 0] > 1e-12
    if not np.any(point_ok):
        return np.zeros((0, 3), dtype=np.float64)

    w_norm = np.zeros_like(w_valid)
    w_norm[point_ok] = w_valid[point_ok] / w_sum[point_ok]
    out = np.sum(mapped * w_norm[:, :, None], axis=1)
    out = out[point_ok]
    return out[np.isfinite(out).all(axis=1)]


def _load_gt_points(paths: list[Path | None], frame_id: int) -> np.ndarray:
    if frame_id < 0 or frame_id >= len(paths):
        return np.zeros((0, 3), dtype=np.float64)
    path = paths[frame_id]
    if path is None:
        return np.zeros((0, 3), dtype=np.float64)
    return _load_geometry_points(path, sample_points=0, voxel_size=0.0)


def _align_gt_to_solver_with_ate(solver_c2w: np.ndarray, gt_c2w: np.ndarray):
    # Lazy import so --help/basic inspection does not require teaserpp_python.
    from utils.error import ATE_TEASER

    R_est = np.hstack([_c2w_to_w2c(pose)[0] for pose in solver_c2w])
    t_est = np.stack([_c2w_to_w2c(pose)[1] for pose in solver_c2w], axis=1)
    R_gt = np.hstack([_c2w_to_w2c(pose)[0] for pose in gt_c2w])
    t_gt = np.stack([_c2w_to_w2c(pose)[1] for pose in gt_c2w], axis=1)
    scale_ate, R_ate, t_ate = ATE_TEASER(R_est, t_est, R_gt, t_gt)
    scale_ate = float(scale_ate)

    # For nearly-linear camera paths, TEASER can leave the roll/twist weakly
    # constrained. Match MoSca's evaluator by using the best camera-orientation
    # rotation, then recomputing translation from the aligned camera centers.
    scale_gt_to_solver = 1.0 / max(scale_ate, 1e-12)
    R_gt_to_solver = _best_fit_rotation(gt_c2w, solver_c2w)
    c_solver = solver_c2w[:, :3, 3]
    c_gt = gt_c2w[:, :3, 3]
    t_gt_to_solver = np.mean(
        c_solver - scale_gt_to_solver * (R_gt_to_solver @ c_gt.T).T,
        axis=0,
    )
    R_solver_to_gt = R_gt_to_solver.T
    t_solver_to_gt = -scale_ate * (R_solver_to_gt @ t_gt_to_solver)
    return scale_ate, R_solver_to_gt, np.asarray(t_solver_to_gt, dtype=np.float64).reshape(3)


def main() -> None:
    args = parse_args()

    dataset_root = (_find_scene_root(args.scene_path) if args.scene_path is not None else definitions.rootpath).resolve()
    output_dir = dataset_root / "outputs"
    results_npz = args.results_npz.resolve() if args.results_npz is not None else _find_results_npz(output_dir)
    result_xm = args.result_xm.resolve() if args.result_xm is not None else output_dir / "result_XM.npz"

    if not results_npz.exists():
        raise FileNotFoundError(f"Missing results npz: {results_npz}")
    if not result_xm.exists():
        raise FileNotFoundError(f"Missing result_XM.npz: {result_xm}")

    data = np.load(results_npz, allow_pickle=True)
    edg_tracks_dict = _load_dict(data, "edg_reconstruct")
    edg_rot_dict = _load_dict(data, "edg_rotation")
    if edg_tracks_dict is None or edg_rot_dict is None:
        raise RuntimeError("Missing edg_reconstruct/edg_rotation in results npz.")

    prior_points, prior_key_used = _load_prior_source_points(data, args.prior_key)
    prior_points = _sample_exact(prior_points, int(args.target_points), seed=1234)
    if prior_points.shape[0] == 0:
        raise RuntimeError(f"No finite prior points from key {prior_key_used}.")

    node_ids = _choose_edg_node_ids(data, edg_tracks_dict, edg_rot_dict, args.node_ids_key)
    node_ids, edg_tracks, edg_rots = _stack_edg_tracks(edg_tracks_dict, edg_rot_dict, node_ids)
    max_frames = edg_tracks.shape[1]

    bind_frame = int(args.bind_frame)
    skin_idx, skin_w = _compute_skinning_from_bind_frame(
        prior_points_bind=prior_points,
        edg_tracks=edg_tracks,
        edg_rots=edg_rots,
        bind_frame=bind_frame,
        knn=int(args.knn),
    )

    solver_c2w = _load_solved_c2w(result_xm)
    gt_c2w = _load_gt_c2w(dataset_root)
    n_pose = min(solver_c2w.shape[0], gt_c2w.shape[0], max_frames)
    if n_pose < 1:
        raise RuntimeError("Not enough camera poses for alignment/evaluation.")
    solver_c2w = solver_c2w[:n_pose]
    gt_c2w = gt_c2w[:n_pose]

    scale_ate, R_ate, t_ate = _align_gt_to_solver_with_ate(solver_c2w, gt_c2w)

    gt_paths = _resolve_gt_geometry_paths(args.gt_mesh, dataset_root, max_frames)
    if gt_paths is None:
        raise FileNotFoundError("No GT mesh/point-cloud files found; pass --gt-mesh.")

    solver_cam_pts = []
    gt_cam_pts = []
    pred_world_by_frame = {}
    gt_world_by_frame = {}
    pred_cam_by_frame = {}
    gt_cam_by_frame = {}
    static_cam_by_frame = {}
    rows = []
    static_landmarks = data.get("static_landmarks")
    static_world = None
    if static_landmarks is not None and np.asarray(static_landmarks).size > 0:
        static_world = np.asarray(static_landmarks, dtype=np.float64).reshape(-1, 3)
        static_world = static_world[np.isfinite(static_world).all(axis=1)]
        static_world = _sample_exact(static_world, 30000, seed=404)
    frame_stride = max(1, int(args.frame_stride))
    for frame_id in range(0, min(max_frames, len(gt_paths), n_pose), frame_stride):
        pred_world = _deform_bound_points_with_edg(
            prior_points_bind=prior_points,
            edg_tracks=edg_tracks,
            edg_rots=edg_rots,
            skin_idx=skin_idx,
            skin_w=skin_w,
            frame=frame_id,
            bind_frame=bind_frame,
        )
        pred_world = _sample_exact(pred_world, int(args.target_points), seed=frame_id + 17)
        if pred_world.shape[0] == 0:
            continue

        gt_world_raw = _load_gt_points(gt_paths, frame_id)
        if gt_world_raw.shape[0] == 0:
            continue
        gt_world_raw = _sample_exact(gt_world_raw, int(args.target_points), seed=frame_id + 29)
        gt_world_aligned = _apply_inverse_similarity(gt_world_raw, scale_ate, R_ate, t_ate)

        solver_R_w2c, solver_t_w2c = _c2w_to_w2c(solver_c2w[frame_id])

        # Fair-comparison convention: evaluate both predicted prior-EDG geometry and
        # aligned GT geometry in the SAME solver camera frame. Do not use aligned-GT camera here.
        pred_cam = (solver_R_w2c @ pred_world.T).T + solver_t_w2c.reshape(1, 3)
        gt_cam = (solver_R_w2c @ gt_world_aligned.T).T + solver_t_w2c.reshape(1, 3)

        solver_cam_pts.append(pred_cam)
        gt_cam_pts.append(gt_cam)
        pred_world_by_frame[int(frame_id)] = pred_world
        gt_world_by_frame[int(frame_id)] = gt_world_aligned
        pred_cam_by_frame[int(frame_id)] = pred_cam
        gt_cam_by_frame[int(frame_id)] = gt_cam
        if static_world is not None and static_world.shape[0] > 0:
            static_cam_by_frame[int(frame_id)] = (
                solver_R_w2c @ static_world.T
            ).T + solver_t_w2c.reshape(1, 3)
        rows.append(
            {
                "frame": int(frame_id),
                "pred_points": int(pred_cam.shape[0]),
                "gt_points": int(gt_cam.shape[0]),
            }
        )

    if not solver_cam_pts:
        raise RuntimeError("No valid frames found for prior-EDG Chamfer evaluation.")

    chamfer_scale, scale_search_score, scale_search_history = _search_global_scale_multilevel(
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
        row["solver_points"] = int(row.pop("pred_points"))
        row["chamfer"] = chamfer
        row["gt_std"] = float(gt_std)
        row["chamfer_norm"] = float(chamfer / gt_std)

    chamfers = [r["chamfer"] for r in rows if np.isfinite(r["chamfer"])]
    chamfers_norm = [r["chamfer_norm"] for r in rows if np.isfinite(r["chamfer_norm"])]

    result = {
        "results_npz": str(results_npz),
        "dataset_root": str(dataset_root),
        "result_xm": str(result_xm),
        "prior_key": prior_key_used,
        "prior_bind_frame": int(bind_frame),
        "prior_points_bound": int(prior_points.shape[0]),
        "edg_nodes_used": int(node_ids.shape[0]),
        "node_ids_key": args.node_ids_key if args.node_ids_key in data else None,
        "knn": int(args.knn),
        "frame_stride": int(frame_stride),
        "target_points": int(args.target_points),
        "coordinate_space": "solver_camera_frame_for_both_pred_and_gt",
        "chamfer_scale": float(chamfer_scale),
        "scale_search_score": float(scale_search_score),
        "scale_search_history": scale_search_history,
        "mean_chamfer": float(np.mean(chamfers)) if chamfers else float("nan"),
        "max_chamfer": float(np.max(chamfers)) if chamfers else float("nan"),
        "mean_chamfer_norm": float(np.mean(chamfers_norm)) if chamfers_norm else float("nan"),
        "max_chamfer_norm": float(np.max(chamfers_norm)) if chamfers_norm else float("nan"),
        "alignment_ate": {
            "scale": float(scale_ate),
            "R": np.asarray(R_ate).tolist(),
            "t": np.asarray(t_ate).tolist(),
        },
        "rows": rows,
    }

    output_json = args.output_json or (output_dir / "prior_edg_chamfer.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"[prior-edg chamfer] saved {output_json}; "
        f"mean={result['mean_chamfer']:.8f} mean_norm={result['mean_chamfer_norm']:.8f} "
        f"scale={chamfer_scale:.8f} prior_key={prior_key_used} "
        f"points={prior_points.shape[0]} nodes={node_ids.shape[0]}"
    )

    if args.no_vis or not args.visualize or not args.show_open3d:
        if args.visualize and not args.show_open3d and not args.no_vis:
            print("[vis] --visualize requested; Open3D window is disabled by default. Use --show-open3d to open it.")
        return

    if o3d is None:
        print("[vis] Open3D is not available; skipping visualization.")
        return

    eval_frames = [row["frame"] for row in rows]
    start_frame = int(args.frame)
    if start_frame not in pred_world_by_frame:
        start_frame = min(eval_frames, key=lambda f: abs(f - start_frame))
    state = {"idx": eval_frames.index(start_frame)}

    edge_arr = np.asarray(data["prior_edges_global"], dtype=np.int64).reshape(-1, 2) if "prior_edges_global" in data else np.zeros((0, 2), dtype=np.int64)
    edge_pairs = [(int(a), int(b)) for a, b in edge_arr.tolist()]

    def _frame_payload(frame: int):
        pred = pred_cam_by_frame[frame] * float(chamfer_scale)
        gt = gt_cam_by_frame[frame]
        solver_R_w2c, solver_t_w2c = _c2w_to_w2c(solver_c2w[frame])
        edg = (solver_R_w2c @ edg_tracks[:, frame, :].T).T + solver_t_w2c.reshape(1, 3)
        valid = np.isfinite(edg).all(axis=1)
        edg = edg[valid]
        gids = [int(g) for g, ok in zip(node_ids.tolist(), valid.tolist()) if ok]
        static = static_cam_by_frame.get(frame, np.zeros((0, 3), dtype=np.float64))
        return pred, gt, edg, gids, static

    pred0, gt0, edg0, gids0, static0 = _frame_payload(start_frame)
    pred_pcd = _colored_point_cloud(pred0, (1.0, 0.05, 0.05))
    gt_pcd = _colored_point_cloud(gt0, (0.1, 0.85, 0.25))
    edg_pcd = _colored_point_cloud(edg0, (1.0, 0.55, 0.05))
    edge_geom = _make_edg_edges(edg0, gids0, edge_pairs)
    cam_geom = _camera_axes_lines(np.eye(4, dtype=np.float64), float(args.camera_axis_len))
    static_pcd = _colored_point_cloud(static0, (0.62, 0.62, 0.62))

    def refresh(vis):
        frame = eval_frames[state["idx"]]
        pred, gt, edg, gids, static = _frame_payload(frame)
        _update_point_cloud(static_pcd, static, (0.62, 0.62, 0.62))
        _update_point_cloud(pred_pcd, pred, (1.0, 0.05, 0.05))
        _update_point_cloud(gt_pcd, gt, (0.1, 0.85, 0.25))
        _update_point_cloud(edg_pcd, edg, (1.0, 0.55, 0.05))
        new_edges = _make_edg_edges(edg, gids, edge_pairs)
        edge_geom.points = new_edges.points
        edge_geom.lines = new_edges.lines
        edge_geom.colors = new_edges.colors
        new_cam = _camera_axes_lines(np.eye(4, dtype=np.float64), float(args.camera_axis_len))
        cam_geom.points = new_cam.points
        cam_geom.lines = new_cam.lines
        cam_geom.colors = new_cam.colors
        for geom in (static_pcd, pred_pcd, gt_pcd, edg_pcd, edge_geom, cam_geom):
            vis.update_geometry(geom)
        frame_chamfer = _symmetric_chamfer(pred, gt)
        print(f"[Open3D] frame={frame}/{max(eval_frames)} metric_chamfer={frame_chamfer:.10f}")
        return False

    def on_space(vis):
        state["idx"] = (state["idx"] + 1) % len(eval_frames)
        return refresh(vis)

    def on_b(vis):
        state["idx"] = (state["idx"] - 1) % len(eval_frames)
        return refresh(vis)

    def on_q(vis):
        vis.close()
        return False

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Prior EDG Chamfer Viewer", width=1400, height=900)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2))
    vis.add_geometry(static_pcd)
    for geom in (pred_pcd, gt_pcd, edg_pcd, edge_geom, cam_geom):
        vis.add_geometry(geom)
    opt = vis.get_render_option()
    opt.point_size = float(args.point_size)
    print("Metric view: red=predicted cloud used in Chamfer, green=GT cloud used in Chamfer.")
    print("Helpers: static=gray, EDG=orange, edges=black. Keys: space=next frame, b=previous, q=quit.")
    print(f"[Open3D] frame={start_frame}/{max(eval_frames)} metric_chamfer={_symmetric_chamfer(pred0, gt0):.10f}")
    vis.register_key_callback(ord(" "), on_space)
    vis.register_key_callback(ord("b"), on_b)
    vis.register_key_callback(ord("B"), on_b)
    vis.register_key_callback(ord("q"), on_q)
    vis.register_key_callback(ord("Q"), on_q)
    vis.register_key_callback(256, on_q)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
