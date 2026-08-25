import os
import sys
import signal
import time
import subprocess
from pathlib import Path
import glob

import numpy as np
import cv2
import trimesh
from scipy.spatial import cKDTree

import utils.definitions
from utils.edg import build_world_trajectories_from_tracks, farthest_point_sample, solve_qcqp_and_recover_until_part5
from utils.readgt_DAVIS import load_davis_gt
from utils.viser_vis import get_viser_server
from utils.io import load_matrix_from_bin

XM_BUILD = (Path(__file__).resolve().parent / "../XM/build").resolve()
if XM_BUILD.exists():
    sys.path.append(str(XM_BUILD))
    import XM
else:
    raise FileNotFoundError(f"XM build directory not found: {XM_BUILD}")

def _close_viser_port(port: int) -> None:
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{int(port)}"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return
    pids = [int(p) for p in result.stdout.strip().split() if p.strip().isdigit()]
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass
    if pids:
        print(f"[VIS] Closed {len(pids)} process(es) on port {port}.")


def _block_until_ctrl_c() -> None:
    print("[VIS] Press Ctrl-C to exit.")
    stop = {"flag": False}
    prev_handler = signal.getsignal(signal.SIGINT)
    def _handle_sigint(signum, frame):
        stop["flag"] = True
    signal.signal(signal.SIGINT, _handle_sigint)
    try:
        while not stop["flag"]:
            time.sleep(0.5)
    finally:
        signal.signal(signal.SIGINT, prev_handler)

BUILD_EDG_DELETE_SMOOTHNESS_PERCENTILE = 0.1
BUILD_EDG_SEGMENR_LENGTH = utils.definitions.segment_length + 1
# BUILD_EDG_SEGMENR_LENGTH = 11

# Load tracks
track_data = np.load(utils.definitions.tracks_dynamics_path, allow_pickle=True)
tracks_xy = track_data["tracks_xy"]
tracks_xyz = track_data["tracks_xyz"]
occluded = track_data["occluded"]
dynamic_obs_mask = track_data["dynamic_obs_mask"]

T = tracks_xy.shape[2]
tracks_xy = tracks_xy[:, :, :T]
tracks_xyz = tracks_xyz[:, :, :T]
occluded = occluded[:, :, :T]
dynamic_obs_mask = dynamic_obs_mask[:, :T]
print(f"Loaded {T} frames and {tracks_xy.shape[1]} tracks")

# Load GT intrinsics
GT = load_davis_gt(str(utils.definitions.rootpath))
if len(GT) == 0:
    raise RuntimeError("DAVIS GT/intrinsics dictionary is empty.")

gt_items = sorted(GT.items(), key=lambda kv: int(kv[1]["id"]))
intrinsics_all = np.stack([kv[1]["K"].astype(np.float64) for kv in gt_items], axis=0)

# Load XM extrinsics
rt = np.load(str(utils.definitions.rootpath / "outputs" / "result_XM.npz"))
R_raw = np.zeros((T, 3, 3))
t_raw = np.zeros((T, 3))
for i in range(T):
    R_raw[i] = rt["R"][:, 3 * i : 3 * (i + 1)]
    t_raw[i] = rt["t"][:, i]

s_raw = rt["s"] if "s" in rt else None
R_for_edg = R_raw.astype(np.float64).copy()
if s_raw is not None:
    R_for_edg *= s_raw[:, None, None]
    
glob_all_full = np.einsum("tij,qtj->qti", R_for_edg, tracks_xyz[0].astype(np.float64)) + t_raw[None, :, :]
valid_mask_global = (
    np.isfinite(glob_all_full).all(axis=-1)
    & (occluded[0] == 0)
    & dynamic_obs_mask.astype(bool)
)
print(f"Global trajectories: {glob_all_full.shape}, valid mask: {valid_mask_global.shape}")

def _connected_components_nodes(nodes, edges_node):
    nodes = list(map(int, nodes))
    node_set = set(nodes)
    adj = {n: set() for n in node_set}
    for a, b in edges_node:
        a = int(a)
        b = int(b)
        if a in node_set and b in node_set and a != b:
            adj[a].add(b)
            adj[b].add(a)

    comps = []
    seen = set()
    for n in nodes:
        if n in seen:
            continue
        stack = [n]
        cc = set([n])
        seen.add(n)
        while stack:
            cur = stack.pop()
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    cc.add(nxt)
                    stack.append(nxt)
        comps.append(cc)
    return comps

def _load_sam3d_foreground_indices(fg_path: Path) -> np.ndarray | None:
    if not fg_path.exists():
        print(f"[WARN] Foreground mask not found: {fg_path}")
        return None
    data = np.load(str(fg_path), allow_pickle=True)
    if "foreground_indices" in data:
        indices = np.asarray(data["foreground_indices"], dtype=np.int64)
        return indices
    if "foreground_mask" in data:
        mask = np.asarray(data["foreground_mask"], dtype=bool)
        return np.flatnonzero(mask).astype(np.int64)
    print(f"[WARN] Foreground mask file missing expected keys: {fg_path}")
    return None


def _sample_mesh_points_fg_bg(
    mesh_verts: np.ndarray,
    foreground_indices: np.ndarray | None,
    num_points: int,
    ):
    rng = np.random.default_rng(0)
    if foreground_indices is None or len(foreground_indices) == 0:
        n0 = min(num_points, mesh_verts.shape[0])
        idx = rng.choice(mesh_verts.shape[0], size=n0, replace=False)
        return (
            np.asarray(mesh_verts[idx], dtype=np.float64),
            np.zeros((n0,), dtype=bool),
            np.asarray(idx, dtype=np.int64),
        )

    fg_idx = np.asarray(foreground_indices, dtype=np.int64)
    fg_idx = fg_idx[(fg_idx >= 0) & (fg_idx < mesh_verts.shape[0])]
    if fg_idx.size == 0:
        n0 = min(num_points, mesh_verts.shape[0])
        idx = rng.choice(mesh_verts.shape[0], size=n0, replace=False)
        return (
            np.asarray(mesh_verts[idx], dtype=np.float64),
            np.zeros((n0,), dtype=bool),
            np.asarray(idx, dtype=np.int64),
        )

    fg_mask_all = np.zeros((mesh_verts.shape[0],), dtype=bool)
    fg_mask_all[fg_idx] = True
    fg_verts = mesh_verts[fg_mask_all]
    bg_verts = mesh_verts[~fg_mask_all]
    if bg_verts.shape[0] == 0:
        n0 = min(num_points, fg_verts.shape[0])
        idx = rng.choice(fg_verts.shape[0], size=n0, replace=False)
        fg_idx = fg_idx[idx]
        return (
            np.asarray(fg_verts[idx], dtype=np.float64),
            np.ones((n0,), dtype=bool),
            np.asarray(fg_idx, dtype=np.int64),
        )

    n_fg = min(num_points // 2, fg_verts.shape[0])
    n_bg = min(num_points - n_fg, bg_verts.shape[0])
    remaining = num_points - (n_fg + n_bg)
    if remaining > 0:
        extra_fg = min(remaining, fg_verts.shape[0] - n_fg)
        n_fg += extra_fg
        remaining = num_points - (n_fg + n_bg)
    if remaining > 0:
        extra_bg = min(remaining, bg_verts.shape[0] - n_bg)
        n_bg += extra_bg

    fg_idx_sel = rng.choice(fg_verts.shape[0], size=n_fg, replace=False) if n_fg > 0 else np.array([], dtype=int)
    bg_idx_sel = rng.choice(bg_verts.shape[0], size=n_bg, replace=False) if n_bg > 0 else np.array([], dtype=int)
    pts = np.concatenate([fg_verts[fg_idx_sel], bg_verts[bg_idx_sel]], axis=0)
    mask = np.concatenate([np.ones((n_fg,), dtype=bool), np.zeros((n_bg,), dtype=bool)], axis=0)
    idx = np.concatenate([fg_idx[fg_idx_sel], np.flatnonzero(~fg_mask_all)[bg_idx_sel]], axis=0)
    if pts.shape[0] > 1:
        order = rng.permutation(pts.shape[0])
        pts = pts[order]
        mask = mask[order]
        idx = idx[order]
    return np.asarray(pts, dtype=np.float64), mask, np.asarray(idx, dtype=np.int64)


def _load_sam3d_mesh_points(
    mesh_path: Path,
    num_points: int,
    foreground_indices: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not mesh_path.exists():
        raise FileNotFoundError(f"SAM3D mesh not found: {mesh_path}")
    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"Expected a Trimesh, got {type(mesh)}")
    verts = np.asarray(mesh.vertices)
    if verts.size == 0:
        raise RuntimeError("Mesh has no vertices")
    pts, mask, idx = _sample_mesh_points_fg_bg(verts, foreground_indices, num_points)
    return pts, mask, idx


def _build_edg_from_mesh_points(
    mesh_pts: np.ndarray,
    mesh_point_indices: np.ndarray,
    graph_samples=30,
    graph_knn=6,
    fg_mask: np.ndarray | None = None,
    ):
    if mesh_pts.ndim != 2 or mesh_pts.shape[1] != 3:
        raise ValueError(f"Expected mesh_pts shape [M,3], got {mesh_pts.shape}")
    if mesh_point_indices.shape[0] != mesh_pts.shape[0]:
        raise ValueError(
            f"mesh_point_indices shape {mesh_point_indices.shape} does not match mesh_pts {mesh_pts.shape}"
        )
    n0 = min(graph_samples, mesh_pts.shape[0])
    if fg_mask is not None and fg_mask.shape[0] == mesh_pts.shape[0] and np.any(fg_mask) and np.any(~fg_mask):
        fg_pts = mesh_pts[fg_mask]
        bg_pts = mesh_pts[~fg_mask]
        fg_mesh_idx = mesh_point_indices[fg_mask]
        bg_mesh_idx = mesh_point_indices[~fg_mask]
        n_fg = min(int(round(n0 * 0.5)), fg_pts.shape[0])
        n_bg = min(n0 - n_fg, bg_pts.shape[0])
        remaining = n0 - (n_fg + n_bg)
        if remaining > 0:
            extra_fg = min(remaining, fg_pts.shape[0] - n_fg)
            n_fg += extra_fg
            remaining = n0 - (n_fg + n_bg)
        if remaining > 0:
            extra_bg = min(remaining, bg_pts.shape[0] - n_bg)
            n_bg += extra_bg

        fg_idx = farthest_point_sample(fg_pts, n_fg) if n_fg > 0 else np.array([], dtype=int)
        bg_idx = farthest_point_sample(bg_pts, n_bg) if n_bg > 0 else np.array([], dtype=int)
        edg_pts = np.concatenate([fg_pts[fg_idx], bg_pts[bg_idx]], axis=0)
        edg_fg_mask = np.concatenate([np.ones((len(fg_idx),), dtype=bool), np.zeros((len(bg_idx),), dtype=bool)])
        edg_node_index = np.concatenate([fg_mesh_idx[fg_idx], bg_mesh_idx[bg_idx]], axis=0).astype(np.int64)
    else:
        fps_idx = farthest_point_sample(mesh_pts, n0)
        edg_pts = mesh_pts[fps_idx]
        if fg_mask is not None and fg_mask.shape[0] == mesh_pts.shape[0]:
            edg_fg_mask = fg_mask[fps_idx]
        else:
            edg_fg_mask = np.zeros((edg_pts.shape[0],), dtype=bool)
        edg_node_index = mesh_point_indices[fps_idx].astype(np.int64)

    tree = cKDTree(edg_pts)
    edges = set()
    k0 = min(graph_knn, edg_pts.shape[0] - 1)
    for i in range(edg_pts.shape[0]):
        _, nn = tree.query(edg_pts[i], k=k0 + 1)
        for j in nn[1:]:
            a, b = (i, int(j)) if i < int(j) else (int(j), i)
            edges.add((a, b))
    return edg_pts, sorted(edges), edg_fg_mask, edg_node_index

graph_samples = 120
graph_knn = 6

MESH_SAMPLE_POINTS = 5000
USE_MESH_EDG = True
SHOW_VISER = False
VISER_BACKGROUND = True
VISER_HOST = "0.0.0.0"
VISER_PORT = 8080

_close_viser_port(VISER_PORT)

current_frame = 0
end_frame_excl = int(min(T, current_frame + BUILD_EDG_SEGMENR_LENGTH))
frames_seg_full = np.arange(current_frame, end_frame_excl, dtype=int)
rest_frame = int(current_frame)

# Load SAM3D mesh for the rest frame and sample points (fast)
image_paths = sorted(glob.glob(str(utils.definitions.video_dir / utils.definitions.image_pattern)))
if rest_frame >= len(image_paths):
    raise RuntimeError(f"Not enough images to resolve rest_frame={rest_frame}.")
image_path = Path(image_paths[rest_frame])
sam3d_dir = Path(utils.definitions.output_dir) / "sam3d"
mesh_path = sam3d_dir / f"sam3d_{image_path.stem}_aligned.ply"
fg_mask_path = sam3d_dir / f"sam3d_{image_path.stem}_foreground_mask.npz"
fg_indices = _load_sam3d_foreground_indices(fg_mask_path)
if fg_indices is not None:
    print(f"Loaded foreground indices: {fg_indices.size}")

mesh_pts, mesh_fg_mask, mesh_pts_idx = _load_sam3d_mesh_points(
    mesh_path,
    MESH_SAMPLE_POINTS,
    foreground_indices=fg_indices,
)
print(f"Loaded SAM3D mesh points: {mesh_path} (sampled {mesh_pts.shape[0]} pts)")

# Build EDG directly from mesh points
edg_pts, edges_node, edg_fg_mask, edg_node_index = _build_edg_from_mesh_points(
    mesh_pts,
    mesh_point_indices=mesh_pts_idx,
    graph_samples=graph_samples,
    graph_knn=graph_knn,
    fg_mask=mesh_fg_mask,
 )
edg_nodes = list(range(edg_pts.shape[0]))
print(f"Initial EDG nodes={len(edg_nodes)} edges={len(edges_node)}")

fg_edg_mask = edg_fg_mask
fg_edg_indices = np.flatnonzero(fg_edg_mask).astype(int)
print(f"Foreground EDG nodes: {fg_edg_indices.size}/{edg_pts.shape[0]}")

# ---- Solve all segments with fixed EDG nodes and persistent dynamic assignments ----
# EDG rest pose g is updated each segment to the last frame pose from the previous segment.
K_NEIGHBORS = 2
edg_idx_map = {int(i): int(i) for i in range(edg_pts.shape[0])}
edges_arr = np.array(edges_node, dtype=int) if len(edges_node) > 0 else np.zeros((0, 2), dtype=int)

if fg_edg_indices.size == 0:
    raise RuntimeError("No foreground EDG nodes available for assignment")
fg_edg_pts = edg_pts[fg_edg_indices]
tree_edg = cKDTree(fg_edg_pts)
K_eff = min(K_NEIGHBORS, fg_edg_pts.shape[0])
if K_eff < 2:
    raise RuntimeError("Need at least 2 EDG nodes for solver neighbors.")

# Persistent assignments: gid -> dict(neighbors, weights)
dyn_assignments = {}

edg_tracks_full = np.full((edg_pts.shape[0], T, 3), np.nan, dtype=np.float64)
edg_rot_full = np.full((edg_pts.shape[0], T, 3, 3), np.nan, dtype=np.float64)
dyn_tracks_full = {}
dyn_input_tracks_full = {}

results_all = {
    "edg_reconstruct": {},
    "edg_rotation": {},
    "edg_edges_union": set(),
    "edg_node_index": {}
}

results_all["edg_node_index"] = {
    int(gid): int(edg_node_index[int(gid)]) for gid in range(edg_pts.shape[0])
}

def _ensure_track(store, gid, T):
    if gid not in store:
        store[gid] = np.full((T, 3), np.nan, dtype=np.float64)
    return store[gid]

def _ensure_rotation(store, gid, T):
    if gid not in store:
        store[gid] = np.tile(np.eye(3, dtype=np.float64), (T, 1, 1))
    return store[gid]


def _compose_global_rotations(R_local, R_rest_global):
    """Compose local EDG rotations with the stored global rest rotation."""
    R_local = np.asarray(R_local, dtype=np.float64)
    R_rest_global = np.asarray(R_rest_global, dtype=np.float64)
    if R_local.ndim != 3 or R_local.shape[1:] != (3, 3):
        raise RuntimeError(
            f"Unexpected R_local shape {R_local.shape}; expected [T,3,3] or [T_rev,3,3]."
        )
    if R_rest_global.shape != (3, 3):
        raise RuntimeError(
            f"Unexpected R_rest_global shape {R_rest_global.shape}; expected [3,3]."
        )
    return R_rest_global[None, :, :] @ R_local

# Start with mesh EDG as the initial rest pose.
edg_rest = edg_pts.astype(np.float64).copy()

seg_id = 0
current_frame = 0
keyframes = [0, 30, 50, 79]
pre_keyframe = None
while True:
    # Segment window [current_frame, end_frame_excl)
    if current_frame >= T:
        break
    flag_new_keyframe_segment = False
    end_frame_excl = int(min(T, current_frame + BUILD_EDG_SEGMENR_LENGTH))
    frames_anchor_first = np.arange(current_frame, end_frame_excl, dtype=int)
    rest_frame = int(current_frame)
    if frames_anchor_first.size == 0:
        break

    frame_to_trev = {int(f): (i + 1) for i, f in enumerate(frames_anchor_first.tolist())}

    # Filter good dynamics points in this segment (segment-local M).
    current_keyframe = max([kf for kf in keyframes if kf <= current_frame], default=0)
    end_keyframe = min([kf for kf in keyframes if kf >= end_frame_excl - 1], default=keyframes[-1])
    print(f"USe dynamics point from keyframe {current_keyframe} to {end_keyframe} for segment {seg_id} (frames {current_frame} to {end_frame_excl - 1})")
    if pre_keyframe == None or current_keyframe > pre_keyframe:
        flag_new_keyframe_segment = True
        pre_keyframe = current_keyframe
    seg_filter_output = build_world_trajectories_from_tracks(
        glob_all=glob_all_full,
        valid_mask_global=valid_mask_global,
        frames=np.arange(current_keyframe, end_keyframe, dtype=int),
        delete_smoothness_percentile=BUILD_EDG_DELETE_SMOOTHNESS_PERCENTILE,
    )
    dyn_ids = np.array(seg_filter_output["selected_edg_idx"], dtype=int)
    print(f"[Segment {seg_id}] keep {len(dyn_ids)}/{glob_all_full.shape[0]} dynamics points")

    dyn_kp_idx_list = []
    dyn_frame_list = []
    dyn_global_list = []
    kp_neighbors = {}
    kp_weights = {}
    kp_rest = {}

    # Use only dynamics points visible at the segment rest frame.
    # If already assigned in a previous segment, reuse neighbors/weights.
    print(f"[Segment {seg_id}] Has {len(dyn_ids)} candidate dynamic observations before visibility filtering at rest frame {rest_frame}.")
    for local_id, gid in enumerate(dyn_ids.tolist()):
        gid = int(gid)
        if not bool(valid_mask_global[gid, rest_frame]):
            continue
        p_rest = glob_all_full[gid, rest_frame, :].astype(np.float64)
        if not np.isfinite(p_rest).all():
            continue

        # If already assigned in a previous segment, reuse neighbors/weights.
        if (gid not in dyn_assignments) and flag_new_keyframe_segment:
            d, nn = tree_edg.query(p_rest, k=K_eff)
            nn = np.atleast_1d(nn).astype(int)
            d = np.atleast_1d(d).astype(np.float64)
            if np.any(d <= 1e-8):
                w = np.zeros_like(d)
                w[np.argmin(d)] = 1.0
            else:
                w = 1.0 / (d + 1e-8)
                w = w / w.sum()
            dyn_assignments[gid] = {
                "neighbors": fg_edg_indices[nn].astype(int),
                "weights": w.astype(np.float64),
            }

        assign = dyn_assignments[gid]
        kp_neighbors[int(local_id)] = assign["neighbors"].astype(int)
        kp_weights[int(local_id)] = assign["weights"].astype(np.float64)
        # Always use the current segment rest-frame position (rest pose for dynamics).
        kp_rest[int(local_id)] = p_rest.astype(np.float64)

        for f in frames_anchor_first.tolist():
            if not bool(valid_mask_global[gid, int(f)]):
                continue
            p = glob_all_full[gid, int(f), :].astype(np.float64)
            if not np.isfinite(p).all():
                continue
            dyn_kp_idx_list.append(int(local_id))
            dyn_frame_list.append(int(f))
            dyn_global_list.append(p.tolist())
            if gid not in dyn_input_tracks_full:
                dyn_input_tracks_full[gid] = np.full((T, 3), np.nan, dtype=np.float64)
            dyn_input_tracks_full[gid][int(f), :] = p
    print(f"[Segment {seg_id}] Selected {len(np.unique(dyn_frame_list))} dynamic observations for solver.")

    if len(dyn_kp_idx_list) == 0:
        print(f"[Solver] Segment {seg_id}: no dynamic observations, skipping.")
        if end_frame_excl >= T:
            break
        current_frame = int(end_frame_excl - 1)
        seg_id += 1
        continue

    dyn_kp_idx = np.array(dyn_kp_idx_list, dtype=int)
    dyn_frame = np.array(dyn_frame_list, dtype=int)
    dyn_global = np.array(dyn_global_list, dtype=np.float64)

    output_dir = Path(utils.definitions.output_dir) / "edg_debug_solver" / f"seg_{seg_id:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Solve this segment with rest pose g from the previous segment's last frame.
    solver_out = solve_qcqp_and_recover_until_part5(
        edges=edges_arr,
        edg_idx_map=edg_idx_map,
        g=edg_rest,
        edg_obs=None,
        edg_obs_mask=None,
        N_dyn_obs=len(dyn_kp_idx),
        dyn_kp_idx=dyn_kp_idx,
        dyn_frame=dyn_frame,
        dyn_global=dyn_global,
        frame_to_trev=frame_to_trev,
        kp_neighbors=kp_neighbors,
        kp_weights=kp_weights,
        kp_rest=kp_rest,
        output_dir=str(output_dir),
        solve_Ronly_fn=XM.solve_Ronly,
        load_matrix_from_bin_fn=load_matrix_from_bin,
    )

    edg_deformed = np.asarray(solver_out["edg_deformed"], dtype=np.float64)  # (M, T_rev, 3)
    edg_rot = np.asarray(solver_out["R_edg"], dtype=np.float64)  # (M, T_rev, 3, 3)
    dyn_mapped = np.asarray(solver_out["dyn_mapped"], dtype=np.float64)      # (Nd, T_rev, 3)

    for a, b in edges_arr.tolist():
        aa, bb = (int(a), int(b)) if int(a) < int(b) else (int(b), int(a))
        results_all["edg_edges_union"].add((aa, bb))

    for j in range(edg_deformed.shape[0]):
        gid = int(j)
        arr = _ensure_track(results_all["edg_reconstruct"], gid, T)
        rot_arr = _ensure_rotation(results_all["edg_rotation"], gid, T)

        if rest_frame < T and np.isfinite(rot_arr[int(rest_frame)]).all():
            rest_global = rot_arr[int(rest_frame)].copy()
        else:
            rest_global = np.eye(3, dtype=np.float64)

        R_local = edg_rot[j]
        R_global = _compose_global_rotations(R_local, rest_global)

        for t_idx, f in enumerate(frames_anchor_first.tolist()):
            arr[int(f), :] = edg_deformed[j, t_idx, :]
            rot_arr[int(f), :, :] = R_global[t_idx]
            edg_tracks_full[j, int(f), :] = edg_deformed[j, t_idx, :]
            edg_rot_full[j, int(f), :, :] = R_global[t_idx]

    # Update rest pose for next segment: last frame in this segment.
    edg_rest = edg_deformed[:, -1, :].copy()

    unique_kps = np.unique(dyn_kp_idx).astype(int)
    for i, local_id in enumerate(unique_kps.tolist()):
        gid = int(dyn_ids[int(local_id)])
        if gid not in dyn_tracks_full:
            dyn_tracks_full[gid] = np.full((T, 3), np.nan, dtype=np.float64)
        for t_idx, f in enumerate(frames_anchor_first.tolist()):
            dyn_tracks_full[gid][int(f), :] = dyn_mapped[i, t_idx, :]

    if end_frame_excl >= T:
        break
    current_frame = int(end_frame_excl - 1)
    seg_id += 1

# Forward/back-fill EDG tracks to avoid visual jumps between segments.
for i in range(edg_tracks_full.shape[0]):
    pts = edg_tracks_full[i]
    valid = np.isfinite(pts).all(axis=1)
    if not valid.any():
        continue
    last = pts[valid][0]
    for t in range(T):
        if valid[t]:
            last = pts[t]
        else:
            pts[t] = last
    # Back-fill if leading NaNs existed.
    last = pts[valid][0]
    for t in range(T):
        if np.isfinite(pts[t]).all():
            last = pts[t]
        else:
            pts[t] = last
    edg_tracks_full[i] = pts

    rots = edg_rot_full[i]
    valid_r = np.isfinite(rots).all(axis=(1, 2))
    if not valid_r.any():
        rots[:, :, :] = np.eye(3, dtype=np.float64)
    else:
        last_r = rots[valid_r][0]
        for t in range(T):
            if valid_r[t]:
                last_r = rots[t]
            else:
                rots[t] = last_r
        last_r = rots[valid_r][0]
        for t in range(T):
            if np.isfinite(rots[t]).all():
                last_r = rots[t]
            else:
                rots[t] = last_r
    edg_rot_full[i] = rots

# Save EDG + dynamics for Open3D visualization
vis_out = Path(utils.definitions.output_dir) / "edg_solution_vis.npz"
dyn_gids = sorted(dyn_tracks_full.keys())
if len(dyn_gids) > 0:
    dyn_mapped = np.full((len(dyn_gids), T, 3), np.nan, dtype=np.float64)
    for i, gid in enumerate(dyn_gids):
        dyn_mapped[i, :, :] = dyn_tracks_full[int(gid)]
else:
    dyn_mapped = np.zeros((0, T, 3), dtype=np.float64)

np.savez_compressed(
    vis_out,
    edg_deformed=edg_tracks_full.astype(np.float64),
    edg_rotation=edg_rot_full.astype(np.float64),
    edge_pairs=edges_arr.astype(np.int64),
    dyn_mapped=dyn_mapped.astype(np.float64),
)
print(f"[VIS] Wrote {vis_out}")

np.savez_compressed(Path(utils.definitions.output_dir) / "results_all.npz", **results_all)
print("[VIS] Wrote results_all.npz")

# ---- Visualize solved EDG + dynamics over full timeline ----
server = get_viser_server(host=VISER_HOST, port=VISER_PORT)
server.gui.add_markdown("### Solver View (All Points)")
solver_state = {"frame": 0}
btn_solver_next = server.gui.add_button("Solver: Next Frame")
btn_solver_prev = server.gui.add_button("Solver: Prev Frame")

dyn_gids = sorted(dyn_tracks_full.keys())
dyn_input_gids = sorted(dyn_input_tracks_full.keys())

def _frame_dyn_points(frame_id: int):
    pts = []
    for gid in dyn_gids:
        p = dyn_tracks_full[gid][frame_id, :]
        if np.isfinite(p).all():
            pts.append(p)
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(pts, dtype=np.float64)

def _frame_dyn_input_points(frame_id: int):
    pts = []
    for gid in dyn_input_gids:
        p = dyn_input_tracks_full[gid][frame_id, :]
        if np.isfinite(p).all():
            pts.append(p)
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(pts, dtype=np.float64)

def _update_solver_scene(frame_idx: int):
    frame_idx = int(frame_idx) % T
    solver_state["frame"] = frame_idx
    edg_pts_f = edg_tracks_full[:, frame_idx, :]
    dyn_pts_f = _frame_dyn_points(frame_idx)
    dyn_in_f = _frame_dyn_input_points(frame_idx)

    edg_ok = np.isfinite(edg_pts_f).all(axis=1)
    edg_colors = np.tile(np.array([255, 80, 80], dtype=np.uint8), (int(edg_ok.sum()), 1))
    server.scene.add_point_cloud(
        "/solver/edg",
        points=edg_pts_f[edg_ok].astype(np.float32),
        colors=edg_colors,
        point_size=0.012,
        point_shape="circle",
        visible=True,
    )

    if dyn_pts_f.size > 0:
        dyn_colors = np.tile(np.array([0, 120, 255], dtype=np.uint8), (dyn_pts_f.shape[0], 1))
        server.scene.add_point_cloud(
            "/solver/dyn",
            points=dyn_pts_f.astype(np.float32),
            colors=dyn_colors,
            point_size=0.01,
            point_shape="circle",
            visible=True,
        )
    else:
        server.scene.add_point_cloud(
            "/solver/dyn",
            points=np.zeros((1, 3), dtype=np.float32),
            colors=np.array([[0, 0, 0]], dtype=np.uint8),
            point_size=0.001,
            visible=False,
        )

    if dyn_in_f.size > 0:
        din_colors = np.tile(np.array([255, 165, 0], dtype=np.uint8), (dyn_in_f.shape[0], 1))
        server.scene.add_point_cloud(
            "/solver/dyn_input",
            points=dyn_in_f.astype(np.float32),
            colors=din_colors,
            point_size=0.008,
            point_shape="circle",
            visible=True,
        )
    else:
        server.scene.add_point_cloud(
            "/solver/dyn_input",
            points=np.zeros((1, 3), dtype=np.float32),
            colors=np.array([[0, 0, 0]], dtype=np.uint8),
            point_size=0.001,
            visible=False,
        )

    if len(edges_node) > 0 and edg_ok.any():
        segs = []
        for a, b in edges_node:
            a = int(a)
            b = int(b)
            if a < edg_pts_f.shape[0] and b < edg_pts_f.shape[0]:
                if edg_ok[a] and edg_ok[b]:
                    segs.append([edg_pts_f[a], edg_pts_f[b]])
        if len(segs) > 0:
            segs = np.asarray(segs, dtype=np.float32)
            edge_colors = np.tile(np.array([80, 255, 80], dtype=np.uint8)[None, None, :], (segs.shape[0], 2, 1))
            server.scene.add_line_segments(
                "/solver/edges",
                points=segs,
                colors=edge_colors,
                line_width=1.5,
                visible=True,
            )

def _solver_next(event):
    _update_solver_scene((solver_state["frame"] + 1) % T)

def _solver_prev(event):
    _update_solver_scene((solver_state["frame"] - 1) % T)

btn_solver_next.on_click(_solver_next)
btn_solver_prev.on_click(_solver_prev)

_update_solver_scene(0)
print("[VIS] Solver view ready. Use the Solver Next/Prev buttons to step frames.")

_block_until_ctrl_c()


        
