#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

import utils.definitions
from utils.readgt_colmap import load_colmap_gt
from utils.readgt_DAVIS import load_davis_gt
from utils.readgt_sintel import load_sintel_gt


def _npz_has_persistent_skinning(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=True) as data:
            return data.get("dynamic_skinning_nodes_array") is not None
    except Exception:
        return False


def _default_results_path() -> Path:
    output_dir = Path(utils.definitions.output_dir)
    return output_dir / "results_all.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize solved EDG, static landmarks, skinned dynamic points, GT geometry, and cameras."
    )

    parser.add_argument(
        "--npz",
        type=Path,
        default=_default_results_path(),
        help="Path to results_all.npz or results_all_prior.npz produced by EDG.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index to visualize.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=4.0,
        help="Open3D point size.",
    )
    parser.add_argument(
        "--result-xm",
        type=Path,
        default=Path(utils.definitions.output_dir) / "result_XM.npz",
        help="XM pose result file containing R/t/s for alignment.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(utils.definitions.rootpath),
        help="Dataset root containing GT cameras and dynamic/object point clouds.",
    )
    parser.add_argument(
        "--gt-mesh",
        type=Path,
        default=None,
        help="Optional GT mesh/point-cloud file or directory.",
    )
    parser.add_argument(
        "--gt-mesh-voxel-size",
        type=float,
        default=0.0,
        help="Optional voxel downsample size for GT mesh/point-cloud display.",
    )
    parser.add_argument(
        "--gt-mesh-sample-points",
        type=int,
        default=1000,
        help="Number of surface points sampled when a GT triangle mesh is loaded.",
    )
    parser.add_argument(
        "--hide-gt-mesh",
        action="store_true",
        help="Hide aligned GT mesh/point-cloud geometry.",
    )
    parser.add_argument(
        "--coordinate-space",
        choices=("robocasa", "solver"),
        default="robocasa",
        help=(
            "Visualization coordinate convention. 'robocasa' shows geometry in raw "
            "COLMAP/world coordinates like visualize_moving_outputs_open3d.py; "
            "'solver' shows GT transformed into EDG/XM solver coordinates."
        ),
    )

    parser.add_argument(
        "--show-cameras",
        action="store_true",
        help="Show solver and GT camera poses as RGB axes.",
    )
    parser.add_argument(
        "--camera-axis-len",
        type=float,
        default=0.08,
        help="Camera axis length for visualization.",
    )
    parser.add_argument(
        "--edg-axis-len",
        type=float,
        default=0.02,
        help="Length (solver units) of the RGB orientation triad drawn at each EDG node from its rotation.",
    )
    parser.add_argument(
        "--hide-edg-axes",
        action="store_true",
        help="Do not draw the per-node EDG rotation axes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build one frame of geometry and print counts without opening Open3D.",
    )
    parser.add_argument(
        "--fill-edg",
        action="store_true",
        help="(EDG_test format) HOLD each EDG node across the whole interval (back/forward-fill holes with "
             "its nearest solved pose). Off by default: a node that is a HOLE at a frame is not drawn, so "
             "holey tables (EDG_test_fast.py) show empty where nothing was solved.",
    )
    parser.add_argument(
        "--raw-dyn",
        action="store_true",
        help="(EDG_test format) draw the stored dyn_reconstruct directly (each point only over its observed "
             "lifetime). Default SKINS every dynamic point from its EDG nodes at EVERY frame, so ALL dynamic "
             "points show on every frame (embedded-deformation fusion via the skinning weights).",
    )

    parser.add_argument(
        "--scale-search",
        action="store_true",
        default=True,
        help="Optimize GT-to-solver scale by Chamfer loss between dynamic points and GT geometry.",
    )
    parser.add_argument(
        "--no-scale-search",
        dest="scale_search",
        action="store_false",
        help="Disable Chamfer scale search and use scale=1.0 after first-camera alignment.",
    )
    parser.add_argument(
        "--scale-min",
        type=float,
        default=0.2,
        help="Minimum scale for Chamfer line search.",
    )
    parser.add_argument(
        "--scale-max",
        type=float,
        default=5.0,
        help="Maximum scale for Chamfer line search.",
    )
    parser.add_argument(
        "--scale-steps",
        type=int,
        default=21,
        help="Number of scale candidates for Chamfer line search.",
    )
    parser.add_argument(
        "--scale-iters",
        type=int,
        default=3,
        help="Number of adaptive scale line-search refinement passes.",
    )
    parser.add_argument(
        "--scale-search-frame",
        type=int,
        default=None,
        help="Frame used for Chamfer scale search. Default: same as --frame.",
    )
    parser.add_argument(
        "--chamfer-max-points",
        type=int,
        default=20000,
        help="Maximum points used from each cloud during Chamfer scale search.",
    )

    return parser.parse_args()


def _load_dict(data, key):
    value = data.get(key)
    if value is None:
        return None
    return value.item()


def _load_tracks(track_dict, frame_id: int):
    points = []
    gids = []

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

        a, b = item
        a = int(a)
        b = int(b)

        if a != b:
            edges.append((min(a, b), max(a, b)))

    return sorted(set(edges))


def _hold_fill_stack(arr: np.ndarray) -> np.ndarray:
    """Complete each track [T,...] over the whole interval so it is finite at every frame: hold the first
    finite value backward, the last finite value forward, and carry the last finite across interior gaps.
    (EDG_test solves a node forward from its first appearance, so this mainly back-fills the pre-appearance
    frames -> the WHOLE EDG shows across the whole time interval.)"""
    arr = np.asarray(arr, dtype=np.float64).copy()
    N, T = arr.shape[0], arr.shape[1]
    flat = arr.reshape(N, T, -1)
    for i in range(N):
        fin = np.isfinite(flat[i]).all(axis=1)
        idx = np.where(fin)[0]
        if idx.size == 0 or idx.size == T:
            continue
        arr[i, : idx[0]] = arr[i, idx[0]]
        arr[i, idx[-1] + 1:] = arr[i, idx[-1]]
        last = idx[0]
        for t in range(idx[0] + 1, idx[-1] + 1):
            if fin[t]:
                last = t
            else:
                arr[i, t] = arr[i, last]
    return arr


def _edges_from_topology(node_ids: np.ndarray):
    """Fallback EDG edges for EDG_test files that predate the saved `edg_edges`: read the global topology
    from output_dir/edg_topology.npz and keep pairs whose endpoints are both in node_ids."""
    p = Path(utils.definitions.output_dir) / "edg_topology.npz"
    if not p.exists():
        return None
    topo = np.load(p)
    if "edg_edges" not in topo.files:
        return None
    ns = {int(x) for x in np.asarray(node_ids).reshape(-1).tolist()}
    ee = np.asarray(topo["edg_edges"], dtype=np.int64).reshape(-1, 2)
    keep = [[a, b] for a, b in ee.tolist() if a in ns and b in ns]
    return np.asarray(keep, dtype=np.int64).reshape(-1, 2)


def _dynamic_point_at_frame(
    *,
    gid: int,
    frame_id: int,
    skin_nodes: np.ndarray,
    skin_weights: np.ndarray,
    bind_rest: np.ndarray,
    bind_frame: int,
    edg_tracks,
    edg_rot_tracks,
):
    if bind_frame < 0 or not np.isfinite(bind_rest).all():
        return None

    accum = np.zeros(3, dtype=np.float64)
    w_accum = 0.0

    for node_gid, weight in zip(skin_nodes.tolist(), skin_weights.tolist()):
        node_gid = int(node_gid)
        weight = float(weight)

        if node_gid < 0 or not np.isfinite(weight) or weight <= 0.0:
            continue

        if node_gid not in edg_tracks or node_gid not in edg_rot_tracks:
            print(f"Warning: skinning node not found in EDG tracks for gid={gid}.")
            continue

        node_track = np.asarray(edg_tracks[node_gid], dtype=np.float64)
        node_rot = np.asarray(edg_rot_tracks[node_gid], dtype=np.float64)

        if (
            bind_frame >= node_track.shape[0]
            or frame_id >= node_track.shape[0]
            or bind_frame >= node_rot.shape[0]
            or frame_id >= node_rot.shape[0]
        ):
            continue

        x_bind = node_track[bind_frame]
        x_frame = node_track[frame_id]
        r_bind = node_rot[bind_frame]
        r_frame = node_rot[frame_id]

        if not (
            np.isfinite(x_bind).all()
            and np.isfinite(x_frame).all()
            and np.isfinite(r_bind).all()
            and np.isfinite(r_frame).all()
        ):
            continue

        rel = r_bind.T @ (bind_rest - x_bind)
        mapped = x_frame + r_frame @ rel

        accum += weight * mapped
        w_accum += weight

    if w_accum <= 1e-12:
        return None

    return accum / w_accum


def _load_dynamic_points(frame_id: int, data, edg_tracks, edg_rot_tracks):
    dynamic_global_ids = np.asarray(data.get("dynamic_global_ids"), dtype=np.int64)
    skin_nodes = np.asarray(data.get("dynamic_skinning_nodes_array"), dtype=np.int64)
    skin_weights = np.asarray(data.get("dynamic_skinning_weights_array"), dtype=np.float64)
    bind_rest = np.asarray(data.get("dynamic_skinning_rest_array"), dtype=np.float64)
    bind_frame = np.asarray(data.get("dynamic_skinning_bind_frame_array"), dtype=np.int64)
    assigned = np.asarray(data.get("dynamic_skinning_assigned_mask"), dtype=bool)

    if dynamic_global_ids.size == 0:
        return np.zeros((0, 3), dtype=np.float64), 0, 0, 0, 0

    points = []
    skinned_now = 0
    solver_now = 0
    bind_rest_now = 0
    bound_total = int(np.sum(assigned))

    for gid in dynamic_global_ids.tolist():
        gid = int(gid)

        if gid >= assigned.shape[0] or not bool(assigned[gid]):
            continue

        pt = _dynamic_point_at_frame(
            gid=gid,
            frame_id=frame_id,
            skin_nodes=skin_nodes[gid],
            skin_weights=skin_weights[gid],
            bind_rest=bind_rest[gid],
            bind_frame=int(bind_frame[gid]),
            edg_tracks=edg_tracks,
            edg_rot_tracks=edg_rot_tracks,
        )

        if pt is not None:
            points.append(pt)
            skinned_now += 1

    if not points:
        return np.zeros((0, 3), dtype=np.float64), skinned_now, solver_now, bind_rest_now, bound_total

    return np.asarray(points, dtype=np.float64), skinned_now, solver_now, bind_rest_now, bound_total


def _build_dyn_skinning(dyn_ids, dyn_reconstruct, node_ids, edg_pos, edg_rot):
    """Precompute embedded-deformation skinning so EVERY dynamic point can be placed at EVERY frame (not
    only over its observed lifetime). For each dyn track: bind frame s_k = its first finite frame, rest
    x_src = dyn_reconstruct[k, s_k] (the warp at the bind frame is the identity, so this IS the track's
    world position). Neighbour EDG nodes come from edg_topology.npz's dyn_edg_edges (restricted to solved
    nodes that are finite at s_k); weights = normalized inverse distance to the node positions at s_k (same
    rule as complete_edg_table.build_skinning). Returns flat (point, node-local, weight, rel) pair arrays
    for a vectorised per-frame warp; rel = R_m(s_k)^T (x_src - X_m(s_k))."""
    topo = Path(utils.definitions.output_dir) / "edg_topology.npz"
    if not topo.exists():
        print("[skin-all] edg_topology.npz missing; cannot skin -> falling back to raw dyn_reconstruct.")
        return None
    td = np.load(topo, allow_pickle=True)
    if "dyn_edg_edges" not in td.files:
        print("[skin-all] dyn_edg_edges missing in edg_topology.npz -> falling back to raw dyn_reconstruct.")
        return None
    dyn_edg = np.asarray(td["dyn_edg_edges"], dtype=np.int64).reshape(-1, 2)
    node_loc = {int(g): i for i, g in enumerate(np.asarray(node_ids).reshape(-1).tolist())}
    d2n = {}
    for d, n in dyn_edg.tolist():
        if int(n) in node_loc:
            d2n.setdefault(int(d), []).append(node_loc[int(n)])

    P_p, P_nl, P_w, P_rel, pids = [], [], [], [], []
    for i, gid in enumerate(np.asarray(dyn_ids).reshape(-1).tolist()):
        nls = d2n.get(int(gid))
        if not nls:
            continue
        vf = np.where(np.isfinite(dyn_reconstruct[i]).all(axis=1))[0]
        if vf.size == 0:
            continue
        sk = int(vf[0]); xsrc = dyn_reconstruct[i, sk]
        nls = np.asarray(nls, dtype=np.int64)
        Xsk = edg_pos[nls, sk]; Rsk = edg_rot[nls, sk]
        ok = np.isfinite(Xsk).all(1) & np.isfinite(Rsk).reshape(nls.size, -1).all(1)
        nls, Xsk, Rsk = nls[ok], Xsk[ok], Rsk[ok]
        if nls.size == 0:
            continue
        dist = np.linalg.norm(Xsk - xsrc[None, :], axis=1)
        w = 1.0 / (dist + 1e-6); w = w / w.sum()
        rel = np.einsum("mji,mj->mi", Rsk, xsrc[None, :] - Xsk)       # R_m(s_k)^T (x_src - X_m(s_k))
        pidx = len(pids); pids.append(int(gid))
        for m in range(nls.size):
            P_p.append(pidx); P_nl.append(int(nls[m])); P_w.append(float(w[m])); P_rel.append(rel[m])
    if not pids:
        print("[skin-all] no skinnable dynamic points -> falling back to raw dyn_reconstruct.")
        return None
    print(f"[skin-all] skinning {len(pids)} dynamic points via {len(P_p)} (point,node) pairs at every frame.")
    return dict(P=len(pids), p=np.asarray(P_p, np.int64), nl=np.asarray(P_nl, np.int64),
                w=np.asarray(P_w, np.float64), rel=np.asarray(P_rel, np.float64),
                edg_pos=edg_pos, edg_rot=edg_rot)


def _skin_all_dyn(sp, frame):
    """Warp every skinned dynamic point to `frame`: pos_k = sum_m w_km (X_m(t) + R_m(t) rel_km) / sum w,
    using only (point,node) pairs whose node is finite at t. Returns finite warped points (N,3)."""
    nl = sp["nl"]
    Xt = sp["edg_pos"][nl, frame]                                     # (E,3)
    Rt = sp["edg_rot"][nl, frame]                                     # (E,3,3)
    ok = np.isfinite(Xt).all(1) & np.isfinite(Rt).reshape(nl.size, -1).all(1)
    Xs = np.where(ok[:, None], Xt, 0.0)
    Rs = np.where(ok[:, None, None], Rt, 0.0)
    mapped = Xs + np.einsum("eij,ej->ei", Rs, sp["rel"])             # X_m(t) + R_m(t) rel
    num = np.zeros((sp["P"], 3)); den = np.zeros(sp["P"])
    np.add.at(num, sp["p"][ok], (sp["w"][ok, None] * mapped[ok]))
    np.add.at(den, sp["p"][ok], sp["w"][ok])
    pos = np.full((sp["P"], 3), np.nan)
    good = den > 1e-9
    pos[good] = num[good] / den[good, None]
    return pos[np.isfinite(pos).all(axis=1)]


def _normalize_rotation(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64))
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


def _load_solved_c2w(result_xm: Path) -> np.ndarray:
    """
    Load solver/XM camera pose as c2w.

    Assumption:
        result_XM.npz stores R[:, 3*i:3*i+3] as c2w rotation up to optional scale,
        and t[:, i] as camera center / translation in solver world.
    """
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
        R_raw = item.get("R")
        t_raw = item.get("t")
        if R_raw is None or t_raw is None:
            continue

        R_w2c = np.asarray(R_raw, dtype=np.float64)
        t_w2c = np.asarray(t_raw, dtype=np.float64)
        if R_w2c.size != 9 or t_w2c.size != 3:
            continue

        R_w2c = R_w2c.reshape(3, 3)
        t_w2c = t_w2c.reshape(3)
        if not (np.isfinite(R_w2c).all() and np.isfinite(t_w2c).all()):
            continue

        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = R_w2c.T
        pose[:3, 3] = -R_w2c.T @ t_w2c
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


def _first_camera_aligned_similarity(
    solver_c2w: np.ndarray,
    gt_c2w: np.ndarray,
    scale: float,
):
    """
    Build similarity that maps GT/world geometry into solver/EDG world:

        x_solver = scale * R_align @ x_gt + t_align

    First-camera alignment constraints:
        R_align @ R_gt0 = R_solver0
        scale * R_align @ c_gt0 + t_align = c_solver0
    """
    R_solver = solver_c2w[0, :3, :3]
    c_solver = solver_c2w[0, :3, 3]

    R_gt = gt_c2w[0, :3, :3]
    c_gt = gt_c2w[0, :3, 3]

    R_align = _normalize_rotation(R_solver @ R_gt.T)
    t_align = c_solver - float(scale) * (R_align @ c_gt)

    return float(scale), R_align, t_align


def _load_geometry_points(path: Path, sample_points: int, voxel_size: float) -> np.ndarray:
    if not path.exists():
        return np.zeros((0, 3), dtype=np.float64)

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


def _subsample_points(points: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]

    if max_points <= 0 or points.shape[0] <= max_points:
        return points

    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=int(max_points), replace=False)
    return points[idx]


def _point_cloud_std(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] < 2:
        return float("nan")
    centered = points - np.mean(points, axis=0, keepdims=True)
    return float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))


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


def _normalized_chamfer(raw: float, gt_std: float) -> float:
    if not np.isfinite(raw) or not np.isfinite(gt_std) or gt_std <= 1e-12:
        return raw
    return float(raw / gt_std)


def _line_search_gt_to_dynamic_scale(
    *,
    dyn_pts_solver: np.ndarray,
    gt_pts: np.ndarray,
    solver_c2w: np.ndarray,
    gt_c2w: np.ndarray,
    scale_min: float,
    scale_max: float,
    scale_steps: int,
    scale_iters: int,
    max_points: int,
):
    """
    Search for scale in:

        x_solver = scale * R_align @ x_gt + t_align(scale)

    R_align is fixed by first-camera orientation.
    t_align changes with scale so the first GT camera center remains aligned
    with the first solver camera center.
    """
    dyn_pts_solver = _subsample_points(dyn_pts_solver, max_points, seed=0)
    gt_pts = _subsample_points(gt_pts, max_points, seed=1)

    if dyn_pts_solver.shape[0] == 0 or gt_pts.shape[0] == 0:
        print("[scale-search] skipped: empty dynamic or GT point cloud.")
        return None

    scale_steps = max(2, int(scale_steps))
    scale_iters = max(1, int(scale_iters))
    lo = min(float(scale_min), float(scale_max))
    hi = max(float(scale_min), float(scale_max))
    if lo <= 0.0:
        lo = 1e-6

    gt_std = _point_cloud_std(gt_pts)
    best = None

    for pass_idx in range(scale_iters):
        scales = np.geomspace(lo, hi, num=scale_steps) if lo < hi else np.asarray([lo])
        if lo <= 1.0 <= hi:
            scales = np.unique(np.concatenate([scales, np.asarray([1.0])]))

        losses = []
        for scale in scales.tolist():
            s, R, t = _first_camera_aligned_similarity(
                solver_c2w,
                gt_c2w,
                scale=float(scale),
            )

            gt_aligned = _apply_similarity(gt_pts, s, R, t)
            raw_loss = _symmetric_chamfer(dyn_pts_solver, gt_aligned)
            norm_loss = _normalized_chamfer(raw_loss, gt_std)
            loss_for_search = norm_loss if np.isfinite(norm_loss) else raw_loss

            if np.isfinite(loss_for_search):
                losses.append((s, raw_loss, norm_loss, R, t))
                if best is None or loss_for_search < best[0]:
                    best = (loss_for_search, raw_loss, norm_loss, s, R, t)

        if not losses:
            break

        losses.sort(key=lambda item: item[0])
        best_idx = min(
            range(len(losses)),
            key=lambda idx: losses[idx][2] if np.isfinite(losses[idx][2]) else losses[idx][1],
        )
        lo = losses[max(best_idx - 1, 0)][0]
        hi = losses[min(best_idx + 1, len(losses) - 1)][0]

        print(
            f"[scale-search] pass={pass_idx + 1}/{scale_iters} "
            f"best_scale={best[3]:.8f} "
            f"raw_chamfer={best[1]:.8f} "
            f"norm_chamfer={best[2]:.8f} "
            f"next_range=[{lo:.8f}, {hi:.8f}]"
        )

    if best is None:
        return None

    _, raw_loss, norm_loss, s, R, t = best

    print(
        f"[scale-search] best scale={s:.6f}, "
        f"raw_chamfer={raw_loss:.8f}, "
        f"norm_chamfer={norm_loss:.8f}, "
        f"gt_std={gt_std:.8f}, "
        f"dynamic_pts={dyn_pts_solver.shape[0]}, "
        f"gt_pts={gt_pts.shape[0]}"
    )

    return s, R, t, raw_loss, norm_loss


def _camera_axes_lines(
    c2w_pose: np.ndarray,
    axis_len: float,
):
    pose = np.asarray(c2w_pose, dtype=np.float64).reshape(4, 4)

    center = pose[:3, 3]
    R = pose[:3, :3]

    pts = np.vstack(
        [
            center,
            center + float(axis_len) * R[:, 0],
            center + float(axis_len) * R[:, 1],
            center + float(axis_len) * R[:, 2],
        ]
    )

    lines = np.asarray(
        [
            [0, 1],
            [0, 2],
            [0, 3],
        ],
        dtype=np.int32,
    )

    colors = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(pts)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def _edg_axes_lineset(centers_solver, gids, frame_id, edg_rot_tracks, axis_len, to_vis):
    """One LineSet of RGB orientation triads, one per EDG node: from the node's rotation R at this frame draw
    center -> center + axis_len * R[:,k] for k = x(red), y(green), z(blue). Built in SOLVER coords then mapped
    to vis coords together with the centers, so the axes stay consistent with the point cloud in either space."""
    if edg_rot_tracks is None:
        return None
    pts, lines, colors = [], [], []
    axis_c = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    base = 0
    for i, gid in enumerate(gids):
        R = edg_rot_tracks.get(int(gid))
        if R is None:
            continue
        R = np.asarray(R, dtype=np.float64)
        if frame_id < 0 or frame_id >= R.shape[0]:
            continue
        Rf = R[frame_id]
        if not np.isfinite(Rf).all():
            continue
        c = np.asarray(centers_solver[i], dtype=np.float64)
        pts.append(c)
        for k in range(3):
            pts.append(c + float(axis_len) * Rf[:, k])
            lines.append([base, base + 1 + k])
            colors.append(axis_c[k])
        base += 4

    if not lines:
        return None

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(to_vis(np.asarray(pts, dtype=np.float64)))
    ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return ls


def _transform_c2w_pose_by_similarity(
    c2w_pose: np.ndarray,
    scale: float,
    R_align: np.ndarray,
    t_align: np.ndarray,
):
    """
    Transform a c2w camera pose by x_new = scale * R_align @ x_old + t_align.

    Rotation is only rotated by R_align.
    Translation is scaled + rotated + translated.
    """
    pose = np.asarray(c2w_pose, dtype=np.float64).copy()
    pose[:3, :3] = _normalize_rotation(R_align @ pose[:3, :3])
    pose[:3, 3] = _apply_similarity(
        pose[:3, 3].reshape(1, 3),
        scale,
        R_align,
        t_align,
    ).reshape(3)
    return pose


def _transform_c2w_pose_by_inverse_similarity(
    c2w_pose: np.ndarray,
    scale: float,
    R_align: np.ndarray,
    t_align: np.ndarray,
):
    """
    Transform a c2w camera pose by inverse of x_new = scale * R_align @ x_old + t_align.

    This maps a solver-space camera back into raw GT/COLMAP world coordinates.
    """
    pose = np.asarray(c2w_pose, dtype=np.float64).copy()
    pose[:3, :3] = _normalize_rotation(R_align.T @ pose[:3, :3])
    pose[:3, 3] = _apply_inverse_similarity(
        pose[:3, 3].reshape(1, 3),
        scale,
        R_align,
        t_align,
    ).reshape(3)
    return pose


def main() -> None:
    args = parse_args()

    npz_path = args.npz.resolve()
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing results file: {npz_path}")

    print(f"Loading EDG result: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    static_landmarks = data.get("static_landmarks")
    # EDG_test.py saves edg_test/edg_global.npz (stacked arrays + already-deformed dynamic points) instead
    # of EDG_old's results_all.npz (dict tracks + per-point skinning arrays). Support both formats.
    edg_test_mode = ("node_ids" in data.files and "dyn_reconstruct" in data.files)
    dyn_ids = dyn_reconstruct = None
    dyn_skin = None
    if edg_test_mode:
        print("[edg_test] detected EDG_test edg_global.npz (direct EDG + dynamic tracks; no skinning re-warp).")
        node_ids = np.asarray(data["node_ids"], dtype=np.int64)
        er = np.asarray(data["edg_reconstruct"], dtype=np.float64)          # [Nn,T,3]
        ero = np.asarray(data["edg_rotation"], dtype=np.float64)            # [Nn,T,3,3]
        if args.fill_edg:                                                 # show the WHOLE EDG at every frame
            er = _hold_fill_stack(er)
            ero = _hold_fill_stack(ero)
            print(f"[edg_test] --fill-edg: held {int(node_ids.size)} EDG nodes across the whole "
                  f"{er.shape[1]}-frame interval.")
        else:                                                             # holes stay blank (the default)
            print("[edg_test] holes are blank: an EDG node is drawn only at frames it was actually solved "
                  "(pass --fill-edg to hold nodes across the whole interval).")
        edg_tracks = {int(node_ids[i]): er[i] for i in range(int(node_ids.size))}
        edg_rot_tracks = {int(node_ids[i]): ero[i] for i in range(int(node_ids.size))}
        edges_union = data["edg_edges"] if "edg_edges" in data.files else _edges_from_topology(node_ids)
        if edges_union is None or np.asarray(edges_union).size == 0:
            print("[edg_test] no EDG edges in the npz or edg_topology.npz; re-run EDG_test.py to save them.")
        dyn_ids = np.asarray(data["dyn_ids"], dtype=np.int64)
        dyn_reconstruct = np.asarray(data["dyn_reconstruct"], dtype=np.float64)  # [Nd,T,3]
        # SKIN every dynamic point at EVERY frame from its EDG nodes (default), so all dynamic points show
        # on every frame instead of only over their observed lifetime. --raw-dyn keeps the stored subset.
        dyn_skin = None if args.raw_dyn else _build_dyn_skinning(dyn_ids, dyn_reconstruct, node_ids, er, ero)
    else:
        edg_tracks = _load_dict(data, "edg_reconstruct")
        edg_rot_tracks = _load_dict(data, "edg_rotation")
        edges_union = data.get("edg_edges_union")
        if edg_tracks is None or edg_rot_tracks is None:
            raise RuntimeError(f"Missing EDG tracks or EDG rotations in {npz_path}.")
        if data.get("dynamic_skinning_nodes_array") is None:
            raise RuntimeError(
                f"Missing persistent skinning arrays in {npz_path}. "
                "Run EDG_old.py or EDG_prior.py with the updated code, "
                "or pass --npz to EDG_test's edg_test/edg_global.npz."
            )

    max_frames = max(int(arr.shape[0]) for arr in edg_tracks.values()) if edg_tracks else 0

    if max_frames == 0:
        raise RuntimeError("No EDG tracks found to visualize.")

    edges = _load_edges(edges_union)

    solver_c2w = None
    gt_c2w = None
    align = None

    try:
        solver_c2w = _load_solved_c2w(args.result_xm)
        gt_c2w = _load_gt_c2w(args.dataset_root)

        n_pose = min(solver_c2w.shape[0], gt_c2w.shape[0], max_frames)
        solver_c2w = solver_c2w[:n_pose]
        gt_c2w = gt_c2w[:n_pose]

        if n_pose >= 1:
            align = _first_camera_aligned_similarity(
                solver_c2w,
                gt_c2w,
                scale=1.0,
            )
            print("[align] initialized GT-to-solver transform from first camera.")

    except Exception as exc:
        print(f"[warning] could not load/align camera poses: {exc}")
        solver_c2w = None
        gt_c2w = None
        align = None

    gt_mesh_paths = None
    if not args.hide_gt_mesh:
        gt_mesh_paths = _resolve_gt_geometry_paths(
            args.gt_mesh,
            args.dataset_root,
            max_frames,
        )

    if (
        args.scale_search
        and gt_mesh_paths is not None
        and solver_c2w is not None
        and gt_c2w is not None
    ):
        search_frame = args.frame if args.scale_search_frame is None else args.scale_search_frame
        search_frame = int(np.clip(search_frame, 0, min(max_frames, len(gt_mesh_paths)) - 1))

        if gt_mesh_paths[search_frame] is None:
            print(f"[scale-search] skipped: no GT geometry for frame={search_frame}.")
        else:
            if edg_test_mode:
                dyn_pts_search = dyn_reconstruct[:, search_frame, :]
                dyn_pts_search = dyn_pts_search[np.isfinite(dyn_pts_search).all(axis=1)]
            else:
                dyn_pts_search, *_ = _load_dynamic_points(
                    search_frame, data, edg_tracks, edg_rot_tracks,
                )

            gt_pts_search = _load_geometry_points(
                gt_mesh_paths[search_frame],
                sample_points=args.gt_mesh_sample_points,
                voxel_size=args.gt_mesh_voxel_size,
            )

            scale_result = _line_search_gt_to_dynamic_scale(
                dyn_pts_solver=dyn_pts_search,
                gt_pts=gt_pts_search,
                solver_c2w=solver_c2w,
                gt_c2w=gt_c2w,
                scale_min=args.scale_min,
                scale_max=args.scale_max,
                scale_steps=args.scale_steps,
                scale_iters=args.scale_iters,
                max_points=args.chamfer_max_points,
            )

            if scale_result is not None:
                scale, R, t, raw_loss, norm_loss = scale_result
                align = (scale, R, t)
                print(
                    f"[align] using Chamfer scale-search alignment: "
                    f"scale={scale:.6f}, "
                    f"raw_chamfer={raw_loss:.8f}, "
                    f"norm_chamfer={norm_loss:.8f}"
                )

    coordinate_space = args.coordinate_space
    if coordinate_space == "robocasa":
        if align is None:
            print(
                "[warning] --coordinate-space robocasa requested, but camera alignment "
                "is unavailable. Solver geometries will stay in solver coordinates; "
                "GT mesh/cameras will be hidden to avoid mixing coordinate frames."
            )
            gt_mesh_paths = None
        else:
            scale, _, _ = align
            print(
                "[coords] displaying in raw COLMAP/world coordinates "
                "to match robocasa visualize_moving_outputs_open3d.py "
                f"(inverse GT-to-solver scale={scale:.6f})."
            )
    else:
        if align is None and gt_mesh_paths is not None:
            print(
                "[warning] --coordinate-space solver requested, but GT-to-solver alignment "
                "is unavailable. GT mesh/cameras will be hidden."
            )
            gt_mesh_paths = None
        print("[coords] displaying in EDG/XM solver coordinates.")

    def _solver_points_to_vis(points: np.ndarray) -> np.ndarray:
        if coordinate_space == "robocasa" and align is not None:
            scale, R, t = align
            return _apply_inverse_similarity(points, scale, R, t)
        return points

    def _gt_points_to_vis(points: np.ndarray) -> np.ndarray:
        if coordinate_space == "solver" and align is not None:
            scale, R, t = align
            return _apply_similarity(points, scale, R, t)
        return points

    def _solver_pose_to_vis(pose: np.ndarray) -> np.ndarray:
        if coordinate_space == "robocasa" and align is not None:
            scale, R, t = align
            return _transform_c2w_pose_by_inverse_similarity(pose, scale, R, t)
        return pose

    def _gt_pose_to_vis(pose: np.ndarray) -> np.ndarray:
        if coordinate_space == "solver" and align is not None:
            scale, R, t = align
            return _transform_c2w_pose_by_similarity(pose, scale, R, t)
        return pose

    show_gt_cameras = gt_c2w is not None and (coordinate_space == "robocasa" or align is not None)
    if align is None:
        show_gt_cameras = False

    state = {"frame": int(np.clip(args.frame, 0, max_frames - 1))}

    def _make_geoms(frame_id: int):
        geoms = []

        if static_landmarks is not None and np.asarray(static_landmarks).size > 0:
            static_pts = np.asarray(static_landmarks, dtype=np.float64).reshape(-1, 3)
            static_pts = static_pts[np.isfinite(static_pts).all(axis=1)]
            static_pts = _solver_points_to_vis(static_pts)

            if static_pts.shape[0] > 0:
                pcd_static = o3d.geometry.PointCloud()
                pcd_static.points = o3d.utility.Vector3dVector(static_pts)
                pcd_static.paint_uniform_color([1.0, 0.8, 0.2])
                geoms.append(("static", pcd_static))

        edg_pts, edg_gids = _load_tracks(edg_tracks, frame_id)

        if edg_pts.size > 0:
            edg_pts_solver = edg_pts                                   # keep solver coords for the rotation axes
            edg_pts = _solver_points_to_vis(edg_pts)
            gid_to_idx = {int(gid): i for i, gid in enumerate(edg_gids)}

            pcd_edg = o3d.geometry.PointCloud()
            pcd_edg.points = o3d.utility.Vector3dVector(edg_pts)
            pcd_edg.paint_uniform_color([0.9, 0.1, 0.1])
            geoms.append(("edg", pcd_edg))

            line_idx = []
            for a, b in edges:
                if a in gid_to_idx and b in gid_to_idx:
                    line_idx.append([gid_to_idx[a], gid_to_idx[b]])

            if line_idx:
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(edg_pts)
                line_set.lines = o3d.utility.Vector2iVector(np.asarray(line_idx, dtype=np.int32))
                line_set.paint_uniform_color([0.72, 0.72, 0.72])       # weak light-grey edges
                geoms.append(("edges", line_set))

            if not args.hide_edg_axes and args.edg_axis_len > 0:       # per-node orientation (RGB triads)
                axes = _edg_axes_lineset(edg_pts_solver, edg_gids, frame_id, edg_rot_tracks,
                                         args.edg_axis_len, _solver_points_to_vis)
                if axes is not None:
                    geoms.append(("edg_axes", axes))

        if edg_test_mode:
            if dyn_skin is not None and 0 <= frame_id < dyn_reconstruct.shape[1]:
                dyn_pts = _skin_all_dyn(dyn_skin, frame_id)           # ALL dyn points, skinned at this frame
            else:                                                     # --raw-dyn: stored lifetime-only subset
                dyn_pts = (dyn_reconstruct[:, frame_id, :] if 0 <= frame_id < dyn_reconstruct.shape[1]
                           else np.zeros((0, 3), dtype=np.float64))
                dyn_pts = np.asarray(dyn_pts, dtype=np.float64)[np.isfinite(dyn_pts).all(axis=1)]
            skinned_now = int(dyn_pts.shape[0]); solver_now = bind_rest_now = bound_total = 0
        else:
            dyn_pts, skinned_now, solver_now, bind_rest_now, bound_total = _load_dynamic_points(
                frame_id, data, edg_tracks, edg_rot_tracks,
            )

        if dyn_pts.size > 0:
            dyn_pts = _solver_points_to_vis(dyn_pts)
            pcd_dyn = o3d.geometry.PointCloud()
            pcd_dyn.points = o3d.utility.Vector3dVector(dyn_pts)
            pcd_dyn.paint_uniform_color([0.0, 0.4, 1.0])
            geoms.append(("dyn", pcd_dyn))

        gt_pts_count = 0
        if (
            gt_mesh_paths is not None
            and frame_id < len(gt_mesh_paths)
            and gt_mesh_paths[frame_id] is not None
        ):
            gt_pts = _load_geometry_points(
                gt_mesh_paths[frame_id],
                sample_points=args.gt_mesh_sample_points,
                voxel_size=args.gt_mesh_voxel_size,
            )

            if gt_pts.size > 0:
                gt_pts = _gt_points_to_vis(gt_pts)

                gt_pts_count = int(gt_pts.shape[0])

                pcd_gt = o3d.geometry.PointCloud()
                pcd_gt.points = o3d.utility.Vector3dVector(gt_pts)
                pcd_gt.paint_uniform_color([0.0, 0.8, 0.2])
                geoms.append(("gt_mesh", pcd_gt))

        if args.show_cameras and solver_c2w is not None:
            if frame_id < solver_c2w.shape[0]:
                solver_cam = _camera_axes_lines(
                    _solver_pose_to_vis(solver_c2w[frame_id]),
                    axis_len=float(args.camera_axis_len),
                )
                geoms.append(("solver_camera", solver_cam))

            if show_gt_cameras and frame_id < gt_c2w.shape[0]:
                gt_pose_vis = _gt_pose_to_vis(gt_c2w[frame_id])

                gt_cam = _camera_axes_lines(
                    gt_pose_vis,
                    axis_len=float(args.camera_axis_len),
                )
                geoms.append(("gt_camera", gt_cam))

        print(
            f"[Open3D] frame={frame_id} "
            f"edg={edg_pts.shape[0]} "
            f"dyn_shown={dyn_pts.shape[0]} "
            f"dyn_skinned={skinned_now} "
            f"dyn_solver_fallback={solver_now} "
            f"dyn_bind_rest_fallback={bind_rest_now} "
            f"dyn_bound_total={bound_total} "
            f"gt_pts={gt_pts_count}"
        )

        if not geoms:                                                     # all nodes are holes this frame:
            origin = o3d.geometry.PointCloud()                            # keep the viewer alive with a single
            origin.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # point at the origin -> the frame
            origin.paint_uniform_color([0.15, 0.15, 0.15])               # looks empty (no phantom geometry)
            geoms.append(("placeholder", origin))

        return geoms

    named_geoms = _make_geoms(state["frame"])

    if not named_geoms:
        raise RuntimeError("No geometry to visualize.")

    if args.dry_run:
        print("[dry-run] built geometries:", ", ".join(name for name, _ in named_geoms))
        return

    geom_map = {name: geom for name, geom in named_geoms}

    def _refresh(vis):
        new_named = _make_geoms(state["frame"])
        new_map = {name: geom for name, geom in new_named}

        # Add new geometry types that were absent at initialization.
        for name, new_geom in new_map.items():
            if name not in geom_map:
                vis.add_geometry(new_geom)
                geom_map[name] = new_geom

        # Update existing geometries; a type absent this frame (its nodes/points are holes) is CLEARED to
        # empty so it disappears from the screen instead of showing stale points from the previous frame.
        empty_pts = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
        for name, geom in list(geom_map.items()):
            new_geom = new_map.get(name)

            if isinstance(geom, o3d.geometry.PointCloud):
                if isinstance(new_geom, o3d.geometry.PointCloud):
                    geom.points = new_geom.points
                    geom.colors = new_geom.colors
                else:
                    geom.points = empty_pts
                    geom.colors = empty_pts
                vis.update_geometry(geom)

            elif isinstance(geom, o3d.geometry.LineSet):
                if isinstance(new_geom, o3d.geometry.LineSet):
                    geom.points = new_geom.points
                    geom.lines = new_geom.lines
                    geom.colors = new_geom.colors
                else:
                    geom.points = empty_pts
                    geom.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
                vis.update_geometry(geom)

        return False

    def on_space(vis):
        state["frame"] = (state["frame"] + 1) % max_frames
        return _refresh(vis)

    def on_b(vis):
        state["frame"] = (state["frame"] - 1) % max_frames
        return _refresh(vis)

    def on_q(vis):
        vis.close()
        return False

    print("Open3D controls: SPACE=next frame, B=prev frame, Q/ESC=close.")
    print(
        "Colors: static=yellow, EDG=red, dynamic/skinned=blue, GT mesh=green, "
        "EDG edges=grey, EDG orientation + cameras=RGB axes."
    )
    if edg_test_mode:
        print("[dyn] " + ("--raw-dyn: showing stored dyn_reconstruct (observed lifetime only)."
                          if dyn_skin is None else
                          "showing ALL dynamic points every frame, skinned from their EDG nodes."))

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="EDG Solution Viewer", width=1280, height=720)

    for _, geom in named_geoms:
        vis.add_geometry(geom)

    opt = vis.get_render_option()
    opt.point_size = float(args.point_size)

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
