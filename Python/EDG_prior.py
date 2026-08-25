import argparse
from pathlib import Path
import os
import subprocess
import sys

import numpy as np
from scipy.spatial import cKDTree

try:
    import open3d as o3d
except Exception as exc:
    o3d = None
    print(f"[Open3D] import failed: {exc}; initial EDG preview disabled.")

import utils.definitions
from utils.script_logger import install_script_logger, pause_script_logger, resume_script_logger
from utils.edg import (
    scatter_solved_frame_vector,
    scatter_solved_poses,
    solve_qcqp_and_recover_until_part5,
)
from utils.io import load_matrix_from_bin


_ap = argparse.ArgumentParser(description="Prior-guided EDG solver (writes the EDG_old-contract results_all.npz)")
_ap.add_argument("--merged", action="store_true",
                 help="solve on the MERGED dynamic tracks (outputs/tracks_dynamic_merged.npz from "
                      "test_extend_tracks.py) instead of the raw tracks_dynamic_only.npz. Downstream "
                      "consumers must use the same track-id space (n_tracks is stamped into the result).")
_args = _ap.parse_args()

output_dir = utils.definitions.output_dir
output_dir.mkdir(parents=True, exist_ok=True)
install_script_logger("dynamic_solver", output_dir)
BUILD_EDG_SEGMENT_LENGTH = 11
# Pipeline-check / speed knobs (env-overridable). EDIT HERE:
EDG_FRAME_LIMIT = int(os.environ.get("EDG_FRAME_LIMIT", "0"))       # >0: only use the first N frames (quick check)
EDG_SOLVER_MAX_TIME = float(os.environ.get("EDG_SOLVER_MAX_TIME", "30"))  # XM solve wall-clock budget in SECONDS per segment (XMtrustregion, duration_cast<seconds>; static_solver uses 30). Lower = faster.
EDG_SOLVER_TOL = float(os.environ.get("EDG_SOLVER_TOL", "1e-4"))    # XM.solve_Ronly convergence tol; looser = stops sooner
# If 1, only use dynamic keypoints visible at the current rest/keyframe; do NOT bind/add
# keypoints that first appear later within the block. 0 = bind keypoints at first-visible.
EDG_ONLY_INITIAL_POINTS = os.environ.get("EDG_ONLY_INITIAL_POINTS", "0") == "1"
# ARAP/rest reference is a KEYFRAME that advances every EDG_KEYFRAME_INTERVAL frames: the
# rest g is re-anchored to the EDG's solved state at each keyframe (frame-0 prior for the
# first block). This bounds drift (re-anchor every N frames) while following the object,
# instead of frame-0 (starves rotating scenes) or per-segment chaining (drifts every 10).
EDG_KEYFRAME_INTERVAL = int(os.environ.get("EDG_KEYFRAME_INTERVAL", "40"))
PRIOR_FRAME_INDEX = 0
PRIOR_EDG_SAMPLES = 100
EDGE_TARGET_NEIGHBORS = 4
K_neighbors = 2
PRIOR_RESULT_NAME = "results_all.npz"
EDG_GID_OFFSET = 10_000_000
SHOW_INITIAL_EDG_OPEN3D = os.environ.get("EDG_SHOW_INITIAL", "1") == "1"  # set EDG_SHOW_INITIAL=0 for headless/batch
# After each segment is solved, pop an interactive Open3D viewer of that segment's EDG nodes+graph,
# dynamic points, and each dynamic point's k-neighbor EDG connections. SPACE/B step frames, Q/ESC
# closes and solving continues. Off by default; enable with EDG_DEBUG_VIZ=1.
DEBUG_SEGMENT_VIZ = os.environ.get("EDG_DEBUG_VIZ", "1") == "1"
DYNAMIC_TO_EDG_MAX_EDGE_RATIO = 1.0
RUN_XM_SOLVER_IN_SUBPROCESS = True
ACCEPT_XM_ABORT_AFTER_RBIN = True
XM_BUILD_DIR = Path(__file__).resolve().parent.parent / "XM" / "build"
EDG_OLD_RESULT_KEYS = {
    "edg_reconstruct",
    "edg_rotation",
    "dynamics_points_reconstruct",
    "dynamic_skinning_nodes",
    "dynamic_skinning_weights",
    "dynamic_skinning_rest",
    "dynamic_skinning_bind_frame",
    "edg_edges_union",
    "segments",
    "edg_input",
    "rest_candidates",
    "static_landmarks",
    "glob_all_full",
    "valid_mask_global",
    "dynamic_global_ids",
    "dynamic_skinning_nodes_array",
    "dynamic_skinning_weights_array",
    "dynamic_skinning_rest_array",
    "dynamic_skinning_bind_frame_array",
    "dynamic_skinning_assigned_mask",
}


_tracks_path = (output_dir / "tracks_dynamic_merged.npz") if _args.merged \
    else Path(utils.definitions.tracks_dynamics_path)
print(f"[tracks] {_tracks_path.name}" + (" (MERGED tracks)" if _args.merged else " (raw tracks)"))
data = np.load(_tracks_path, allow_pickle=True)
tracks_xy = data["tracks_xy"]
tracks_xyz = data["tracks_xyz"]
occluded = data["occluded"]
dynamic_obs_mask = data["dynamic_obs_mask"][:, :]

T = tracks_xy.shape[2]
if EDG_FRAME_LIMIT > 0:
    T = min(T, EDG_FRAME_LIMIT)  # pipeline check: only reconstruct the first N frames
    print(f"[EDG_FRAME_LIMIT] restricting to first {T} frames for a pipeline check")
tracks_xy = tracks_xy[:, :, :T]
tracks_xyz = tracks_xyz[:, :, :T]
occluded = occluded[:, :, :T]
dynamic_obs_mask = dynamic_obs_mask[:, :T]
print(f"Loaded {T} frames and {tracks_xy.shape[1]} tracks")

rt = np.load(str(utils.definitions.rootpath / "outputs" / "result_XM.npz"))
landmarks = rt["p"].T if "p" in rt else None
# The static solver may drop frames (checklandmarks / XM^2), so R/t are stored in solved
# order (N<=T). Scatter them onto the full T-frame timeline via indices_all; dropped
# frames get identity R + NaN t so valid_mask_global excludes them.
R_raw, t_raw = scatter_solved_poses(rt, T)


def _load_da3_frame_scale(rt_data, frame_count):
    if "da3_frame_scale" not in rt_data.files:
        print("[EDG scale] result_XM.npz has no da3_frame_scale; using scale=1 for DA3 lifts.")
        return np.ones((int(frame_count),), dtype=np.float64), "missing_da3_frame_scale"
    # da3_frame_scale is per-solved-frame (N<=T); scatter it onto the full timeline via
    # indices_all so it stays aligned with R_raw/t_raw when the solver dropped frames.
    scales = scatter_solved_frame_vector(rt_data, "da3_frame_scale", int(frame_count), default=1.0)
    invalid = ~np.isfinite(scales) | (scales <= 0.0)
    if np.any(invalid):
        print(f"[EDG scale] replacing {int(np.count_nonzero(invalid))} invalid DA3 frame scales with 1.")
        scales[invalid] = 1.0
    return scales, "da3_frame_scale"


def _solve_Ronly_isolated(output_dir_arg, solver_rank, solver_tol, solver_mu, solver_max_iter):
    """Run XM.solve_Ronly outside this process to isolate CUDA/CUSPARSE teardown bugs."""
    output_dir_path = Path(output_dir_arg).resolve()
    r_bin = output_dir_path / "R.bin"
    if r_bin.exists():
        r_bin.unlink()

    if not RUN_XM_SOLVER_IN_SUBPROCESS:
        sys.path.insert(0, str(XM_BUILD_DIR))
        import XM as XM_inprocess

        return XM_inprocess.solve_Ronly(
            str(output_dir_path),
            int(solver_rank),
            float(solver_tol),
            float(solver_mu),
            float(solver_max_iter),
        )

    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(XM_BUILD_DIR)!r})\n"
        "import XM\n"
        "XM.solve_Ronly("
        f"{str(output_dir_path)!r}, "
        f"{int(solver_rank)!r}, "
        f"{float(solver_tol)!r}, "
        f"{float(solver_mu)!r}, "
        f"{float(solver_max_iter)!r})\n"
    )
    print(f"[XM] running solve_Ronly in isolated subprocess: output_dir={output_dir_path}")
    def _disable_core_dump():
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        except Exception:
            pass

    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(Path(__file__).resolve().parent.parent),
        preexec_fn=_disable_core_dump if os.name == "posix" else None,
    )
    if proc.returncode != 0:
        if ACCEPT_XM_ABORT_AFTER_RBIN and r_bin.exists() and r_bin.stat().st_size > 8:
            print(
                "[XM] subprocess exited non-zero after writing R.bin "
                f"(returncode={proc.returncode}); continuing because CUDA teardown is isolated."
            )
            return None
        raise RuntimeError(f"XM.solve_Ronly subprocess failed with returncode={proc.returncode}")
    if not r_bin.exists() or r_bin.stat().st_size <= 8:
        raise RuntimeError(f"XM.solve_Ronly finished but did not produce a valid R.bin at {r_bin}")
    return None


da3_frame_scale, da3_frame_scale_source = _load_da3_frame_scale(rt, T)
R_for_depth_lift = da3_frame_scale[:, None, None] * R_raw.astype(np.float64)

glob_all_full = np.einsum("tij,qtj->qti", R_for_depth_lift, tracks_xyz[0].astype(np.float64)) + t_raw[None, :, :]
valid_mask_global = np.isfinite(glob_all_full).all(axis=-1) & (occluded[0] == 0) & dynamic_obs_mask.astype(bool)
finite_scales = da3_frame_scale[np.isfinite(da3_frame_scale)]
if finite_scales.size > 0:
    print(
        "[EDG scale] lifting DA3 dynamics with static_solver scale: "
        f"min={finite_scales.min():.6g}, median={np.median(finite_scales):.6g}, max={finite_scales.max():.6g}"
    )


def _load_prior_alignment():
    npz_path = output_dir / f"aligned_pointcloud_prior_frame{PRIOR_FRAME_INDEX:06d}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing aligned prior file: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    prior_key = (
        "aligned_full_prior_points_camera"
        if "aligned_full_prior_points_camera" in data
        else "aligned_prior_points_camera"
    )
    if prior_key not in data:
        raise RuntimeError(
            f"{npz_path} missing 'aligned_full_prior_points_camera' or 'aligned_prior_points_camera'"
        )
    pts_cam = np.asarray(data[prior_key], dtype=np.float64)
    if pts_cam.ndim != 2 or pts_cam.shape[1] != 3 or pts_cam.shape[0] == 0:
        raise RuntimeError(f"Unexpected aligned prior shape {pts_cam.shape} in {npz_path}")
    frame_scale = float(da3_frame_scale[PRIOR_FRAME_INDEX])
    pts_world = (frame_scale * (R_raw[PRIOR_FRAME_INDEX] @ pts_cam.T)).T + t_raw[PRIOR_FRAME_INDEX][None, :]
    return pts_world, npz_path, prior_key


def _farthest_point_sample(points, num_samples):
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] <= num_samples:
        return np.arange(points.shape[0], dtype=np.int64)
    idx = [0]
    min_dist = np.linalg.norm(points - points[0:1], axis=1)
    for _ in range(1, int(num_samples)):
        next_idx = int(np.argmax(min_dist))
        idx.append(next_idx)
        dist = np.linalg.norm(points - points[next_idx:next_idx + 1], axis=1)
        min_dist = np.minimum(min_dist, dist)
    return np.asarray(idx, dtype=np.int64)


def _build_fixed_edges_knn(points, target_neighbors):
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] <= 1:
        return []
    tree = cKDTree(points)
    k_eff = min(int(target_neighbors) + 1, int(points.shape[0]))
    _, nn = tree.query(points, k=k_eff)
    edges = set()
    for i in range(points.shape[0]):
        neigh = np.atleast_1d(nn[i]).tolist()
        for j in neigh:
            j = int(j)
            if j == i:
                continue
            a, b = (i, j) if i < j else (j, i)
            edges.add((a, b))
    return sorted(edges)


def _edg_node_edge_radius(points, edge_local_pairs):
    points = np.asarray(points, dtype=np.float64)
    edge_local_pairs = np.asarray(edge_local_pairs, dtype=np.int64).reshape(-1, 2)
    radius = np.full((points.shape[0],), np.nan, dtype=np.float64)
    if points.shape[0] <= 1:
        return np.full((points.shape[0],), np.inf, dtype=np.float64)
    for a, b in edge_local_pairs:
        if a < 0 or b < 0 or a >= points.shape[0] or b >= points.shape[0]:
            continue
        d = float(np.linalg.norm(points[int(a)] - points[int(b)]))
        if not np.isfinite(d) or d <= 0:
            continue
        for idx in (int(a), int(b)):
            if not np.isfinite(radius[idx]):
                radius[idx] = d
            else:
                radius[idx] = min(radius[idx], d)
    missing = ~np.isfinite(radius)
    if np.any(missing):
        tree = cKDTree(points)
        k_eff = min(2, points.shape[0])
        dist, _ = tree.query(points[missing], k=k_eff)
        radius[missing] = np.asarray(dist)[:, -1]
    return radius


def _build_candidate_views(global_index_cand):
    global_index_cand = np.asarray(global_index_cand, dtype=int)
    tracks_glob_candidate = glob_all_full[None, global_index_cand, :, :].copy()
    tracks_xy_candidate = tracks_xy[:, global_index_cand, :, :].copy()
    valid_mask_candidate_global = valid_mask_global[global_index_cand, :].copy()
    return (
        global_index_cand,
        tracks_glob_candidate,
        tracks_xy_candidate,
        valid_mask_candidate_global,
        valid_mask_candidate_global.copy(),
    )


def _save_rest_candidates(results_all, seg_id, rest_frame, frames_seg_full, global_index_cand, edg_global_ids):
    rest_points_all = glob_all_full[global_index_cand, rest_frame, :].astype(np.float64)
    rest_valid_all = valid_mask_global[global_index_cand, rest_frame].astype(bool)
    finite_rest = np.isfinite(rest_points_all).all(axis=1)
    rest_keep = rest_valid_all & finite_rest
    results_all["rest_candidates"].append(
        dict(
            segment_id=int(seg_id),
            rest_frame=int(rest_frame),
            frames_anchor_first=np.asarray(frames_seg_full, dtype=int).tolist(),
            candidate_global_ids=np.asarray(global_index_cand, dtype=int),
            candidate_visible_rest_mask=rest_keep.astype(bool),
            global_ids=np.asarray(global_index_cand, dtype=int)[rest_keep].astype(int),
            points_world=rest_points_all[rest_keep].astype(np.float64),
            edg_global_ids=np.asarray(edg_global_ids, dtype=int),
        )
    )


def _build_fixed_edg_solver_inputs_for_segment(
    tracks_glob_candidate,
    valid_mask_candidate_global,
    frames_anchor_first,
    rest_frame,
    global_index_cand,
    edg_rest_points,
    edge_local_pairs,
    edg_global_ids,
    persistent_skinning_nodes,
    persistent_skinning_weights,
    persistent_skinning_rest,
    persistent_skinning_bind_frame,
    K_neighbors=4,
    edg_recon=None,
    edg_rot=None,
):
    frames_anchor_first = np.asarray(frames_anchor_first, dtype=int)
    T_rev = len(frames_anchor_first)
    frame_to_trev = {int(f): (i + 1) for i, f in enumerate(frames_anchor_first.tolist())}
    valid_mask_candidate = valid_mask_candidate_global.astype(bool)
    global_index_cand = np.asarray(global_index_cand, dtype=int)
    edg_rest_points = np.asarray(edg_rest_points, dtype=np.float64)
    edg_global_ids = np.asarray(edg_global_ids, dtype=int)
    M = edg_rest_points.shape[0]

    edg_idx_map = {int(i): int(i) for i in range(M)}
    g = edg_rest_points.copy()
    edg_node_radius = _edg_node_edge_radius(g, edge_local_pairs)

    edg_obs = np.full((M, T_rev, 3), np.nan, dtype=np.float64)
    edg_obs_mask = np.zeros((M, T_rev), dtype=bool)
    rest_visible = np.where(valid_mask_candidate[:, rest_frame])[0].astype(int)
    rest_pts = tracks_glob_candidate[0, rest_visible, rest_frame, :].astype(np.float64) if rest_visible.size > 0 else np.zeros((0, 3), dtype=np.float64)
    anchor_local_ids = np.full((M,), -1, dtype=int)
    if rest_pts.shape[0] > 0:
        tree = cKDTree(rest_pts)
        _, nn = tree.query(g, k=1)
        anchor_local_ids = rest_visible[np.asarray(nn, dtype=int)]

    rest_t_idx = int(np.where(frames_anchor_first == int(rest_frame))[0][0])
    for j in range(M):
        lid = int(anchor_local_ids[j]) if j < anchor_local_ids.shape[0] else -1
        if lid >= 0:
            for t_idx, f in enumerate(frames_anchor_first.tolist()):
                if valid_mask_candidate[lid, int(f)]:
                    P = tracks_glob_candidate[0, lid, int(f)].astype(np.float64)
                    if np.isfinite(P).all():
                        edg_obs[j, t_idx, :] = P
                        edg_obs_mask[j, t_idx] = True
        edg_obs[j, rest_t_idx, :] = g[j]
        edg_obs_mask[j, rest_t_idx] = True

    dyn_kp_idx_list = []
    dyn_frame_list = []
    dyn_global_list = []
    Qcand = tracks_glob_candidate.shape[1]
    for local_id in range(Qcand):
        for f in frames_anchor_first.tolist():
            if not valid_mask_candidate[local_id, int(f)]:
                continue
            P = tracks_glob_candidate[0, local_id, int(f)].astype(np.float64)
            if not np.isfinite(P).all():
                continue
            dyn_kp_idx_list.append(int(local_id))
            dyn_frame_list.append(int(f))
            dyn_global_list.append(P.tolist())

    dyn_kp_idx = np.asarray(dyn_kp_idx_list, dtype=int)
    dyn_frame = np.asarray(dyn_frame_list, dtype=int)
    dyn_global = np.asarray(dyn_global_list, dtype=np.float64)
    if dyn_kp_idx.size == 0:
        raise RuntimeError("No dynamic observations available for solver in this segment.")

    # Node deformed positions/rotations at a frame, from accumulated prior-segment solves
    # (fall back to the frame-0 rest g / identity where a frame is not solved yet). Used to
    # bind keypoints that appear AFTER frame 0 and transport them to the frame-0 rest.
    edg_gid_list = [int(x) for x in edg_global_ids.tolist()]

    def _node_state_at(frame):
        pos = g.copy()
        rot = np.tile(np.eye(3, dtype=np.float64), (M, 1, 1))
        for jj, gid in enumerate(edg_gid_list):
            if edg_recon is not None:
                arr = edg_recon.get(int(gid))
                if arr is not None and 0 <= int(frame) < arr.shape[0] and np.isfinite(arr[int(frame)]).all():
                    pos[jj] = arr[int(frame)]
            if edg_rot is not None:
                rarr = edg_rot.get(int(gid))
                if rarr is not None and 0 <= int(frame) < rarr.shape[0] and np.isfinite(rarr[int(frame)]).all():
                    rot[jj] = rarr[int(frame)]
        return pos, rot

    kp_neighbors = {}
    kp_weights = {}
    kp_rest = {}
    reused_skinning_count = 0
    skipped_far_from_edg_count = 0
    edg_global_to_local = {int(gid): int(i) for i, gid in enumerate(edg_global_ids.tolist())}
    _node_state_cache = {}
    for local_id in range(Qcand):
        # Bind each keypoint at its FIRST-visible frame in the segment (not just rest_frame),
        # so keypoints that rotate/appear after frame 0 are still skinned.
        bind_f = -1
        for f in frames_anchor_first.tolist():
            if valid_mask_candidate[local_id, int(f)]:
                bind_f = int(f)
                break
        if bind_f < 0:
            continue
        # Optionally only keep keypoints present at the keyframe/rest (no later-appearing pts).
        if EDG_ONLY_INITIAL_POINTS and bind_f != int(rest_frame):
            continue
        x_bind = tracks_glob_candidate[0, local_id, bind_f].astype(np.float64)
        if not np.isfinite(x_bind).all():
            continue
        global_gid = int(global_index_cand[local_id])
        # Reuse persistent skinning if bound in a prior segment, regardless of visibility at
        # the current rest frame (its rest is already expressed relative to the frame-0 nodes).
        if global_gid in persistent_skinning_nodes and global_gid in persistent_skinning_weights:
            saved_global_ids = np.asarray(persistent_skinning_nodes[global_gid], dtype=int)
            saved_w = np.asarray(persistent_skinning_weights[global_gid], dtype=np.float64)
            can_reuse = (
                saved_global_ids.size >= 2
                and saved_global_ids.size == saved_w.size
                and all(int(gid) in edg_global_to_local for gid in saved_global_ids.tolist())
            )
            if can_reuse:
                kp_neighbors[int(local_id)] = np.asarray(
                    [edg_global_to_local[int(gid)] for gid in saved_global_ids.tolist()], dtype=int)
                kp_weights[int(local_id)] = saved_w.astype(np.float64)
                kp_rest[int(local_id)] = np.asarray(
                    persistent_skinning_rest.get(global_gid, x_bind), dtype=np.float64)
                reused_skinning_count += 1
                continue

        # Associate to the K nearest EDG nodes AT THE BIND FRAME (their deformed positions).
        if bind_f not in _node_state_cache:
            _node_state_cache[bind_f] = _node_state_at(bind_f)
        npos, nrot = _node_state_cache[bind_f]
        d = np.linalg.norm(npos - x_bind[None, :], axis=1)
        if d.size == 0:
            continue
        K_eff = min(int(K_neighbors), int(d.size))
        nn = np.argpartition(d, K_eff - 1)[:K_eff]
        nn = nn[np.argsort(d[nn])]
        nn_dist = d[nn]
        nearest_node = int(nn[0])
        nearest_dist = float(nn_dist[0])
        max_allowed_dist = float(edg_node_radius[nearest_node]) * float(DYNAMIC_TO_EDG_MAX_EDGE_RATIO)
        if np.isfinite(max_allowed_dist) and nearest_dist > max_allowed_dist:
            skipped_far_from_edg_count += 1
            continue
        if nn.size < 2:
            continue
        if np.any(nn_dist <= 1e-8):
            w = np.zeros_like(nn_dist)
            w[np.argmin(nn_dist)] = 1.0
        else:
            w = 1.0 / (nn_dist + 1e-8)
            w = w / np.sum(w)
        # Transport the keypoint's bind-frame position back to the frame-0 rest through each
        # node's bind-frame rotation, so x_rest is expressed relative to the frame-0 nodes g
        # (matches the solver's data term x = w(R(x_rest - g_j) + g_j + p)). For bind_f=0 this
        # reduces to x_rest = x_bind (identity transport).
        x_rest = np.zeros(3, dtype=np.float64)
        for wj, jj in zip(w, nn):
            local_off = nrot[jj].T @ (x_bind - npos[jj])
            x_rest += float(wj) * (g[jj] + local_off)
        kp_neighbors[int(local_id)] = nn.astype(int)
        kp_weights[int(local_id)] = w.astype(np.float64)
        kp_rest[int(local_id)] = x_rest.astype(np.float64)
        if global_gid not in persistent_skinning_nodes:
            persistent_skinning_nodes[global_gid] = edg_global_ids[nn].astype(int)
            persistent_skinning_weights[global_gid] = w.astype(np.float64)
            persistent_skinning_rest[global_gid] = x_rest.astype(np.float64)
            persistent_skinning_bind_frame[global_gid] = int(bind_f)

    valid_kps = np.asarray(list(kp_neighbors.keys()), dtype=int)
    keep = np.isin(dyn_kp_idx, valid_kps)
    dyn_kp_idx = dyn_kp_idx[keep]
    dyn_frame = dyn_frame[keep]
    dyn_global = dyn_global[keep]
    if dyn_kp_idx.size == 0:
        raise RuntimeError("All dynamic obs removed by kp_neighbors filtering.")
    if skipped_far_from_edg_count > 0:
        print(
            "[Prior EDG] skipped "
            f"{skipped_far_from_edg_count} dynamic points farther than "
            f"{DYNAMIC_TO_EDG_MAX_EDGE_RATIO:.2f}x their nearest-node EDG edge length; "
            f"reused {reused_skinning_count} existing skinned points."
        )

    return dict(
        edges=np.asarray(edge_local_pairs, dtype=int),
        edg_idx_map=edg_idx_map,
        g=g,
        edg_obs=edg_obs,
        edg_obs_mask=edg_obs_mask,
        N_dyn_obs=len(dyn_kp_idx),
        dyn_kp_idx=dyn_kp_idx,
        dyn_frame=dyn_frame,
        dyn_global=dyn_global,
        frame_to_trev=frame_to_trev,
        frames_anchor_first=frames_anchor_first,
        kp_neighbors=kp_neighbors,
        kp_weights=kp_weights,
        kp_rest=kp_rest,
        skipped_far_from_edg_count=int(skipped_far_from_edg_count),
        reused_skinning_count=int(reused_skinning_count),
    )


def _dyn_grid_from_solver_output(dyn_mapped, Nd, T_rev):
    dyn_mapped = np.asarray(dyn_mapped)
    if dyn_mapped.ndim == 3 and dyn_mapped.shape == (Nd, T_rev, 3):
        return dyn_mapped.astype(np.float64)
    if dyn_mapped.ndim == 2 and dyn_mapped.shape == (Nd * T_rev, 3):
        return dyn_mapped.reshape(Nd, T_rev, 3).astype(np.float64)
    raise RuntimeError(f"Unexpected dyn_mapped shape {dyn_mapped.shape}")


def _ensure_track(store, gid, T):
    if gid not in store:
        store[gid] = np.full((T, 3), np.nan, dtype=np.float64)
    return store[gid]


def _ensure_rotation(store, gid, T):
    if gid not in store:
        store[gid] = np.tile(np.eye(3, dtype=np.float64), (T, 1, 1))
    return store[gid]


def _compose_global_rotations(R_local, R_rest_global):
    return np.asarray(R_rest_global, dtype=np.float64)[None, :, :] @ np.asarray(R_local, dtype=np.float64)


def _init_results_store():
    return {
        "edg_reconstruct": {},
        "edg_rotation": {},
        "dynamics_points_reconstruct": {},
        "dynamic_skinning_nodes": {},
        "dynamic_skinning_weights": {},
        "dynamic_skinning_rest": {},
        "dynamic_skinning_bind_frame": {},
        "edg_edges_union": set(),
        "segments": [],
        "edg_input": {},
        "rest_candidates": [],
        "static_landmarks": landmarks.astype(np.float64) if landmarks is not None else None,
        "glob_all_full": glob_all_full,
        "valid_mask_global": valid_mask_global,
        "da3_frame_scale": da3_frame_scale.astype(np.float64),
        "da3_frame_scale_source": np.array(da3_frame_scale_source),
    }


def _finalize_dynamic_outputs(results_all, num_points, T, K_neighbors):
    for gid in range(int(num_points)):
        _ensure_track(results_all["dynamics_points_reconstruct"], int(gid), T)
    skin_nodes = np.full((int(num_points), int(K_neighbors)), -1, dtype=np.int64)
    skin_weights = np.full((int(num_points), int(K_neighbors)), np.nan, dtype=np.float64)
    bind_rest = np.full((int(num_points), 3), np.nan, dtype=np.float64)
    bind_frame = np.full((int(num_points),), -1, dtype=np.int64)
    assigned = np.zeros((int(num_points),), dtype=bool)
    for gid in range(int(num_points)):
        if gid in results_all["dynamic_skinning_nodes"]:
            nodes = np.asarray(results_all["dynamic_skinning_nodes"][gid], dtype=np.int64).reshape(-1)
            weights = np.asarray(results_all["dynamic_skinning_weights"].get(gid, []), dtype=np.float64).reshape(-1)
            k_eff = min(int(K_neighbors), int(nodes.size), int(weights.size))
            if k_eff > 0:
                skin_nodes[gid, :k_eff] = nodes[:k_eff]
                skin_weights[gid, :k_eff] = weights[:k_eff]
                assigned[gid] = True
        if gid in results_all["dynamic_skinning_rest"]:
            bind_rest[gid] = np.asarray(results_all["dynamic_skinning_rest"][gid], dtype=np.float64).reshape(3)
        if gid in results_all["dynamic_skinning_bind_frame"]:
            bind_frame[gid] = int(results_all["dynamic_skinning_bind_frame"][gid])
    results_all["dynamic_global_ids"] = np.arange(int(num_points), dtype=np.int64)
    results_all["dynamic_skinning_nodes_array"] = skin_nodes
    results_all["dynamic_skinning_weights_array"] = skin_weights
    results_all["dynamic_skinning_rest_array"] = bind_rest
    results_all["dynamic_skinning_bind_frame_array"] = bind_frame
    results_all["dynamic_skinning_assigned_mask"] = assigned


def _fill_nan_edg_tracks(edg_tracks_in, edg_rot_tracks_in, edges_union):
    edg_tracks_out = {gid: arr.copy() for gid, arr in edg_tracks_in.items()}
    edg_rot_tracks_out = {gid: arr.copy() for gid, arr in edg_rot_tracks_in.items()}
    adjacency = {gid: [] for gid in edg_tracks_out}
    for ga, gb in edges_union:
        if ga in adjacency:
            adjacency[ga].append(gb)
        if gb in adjacency:
            adjacency[gb].append(ga)
    for gid, arr in edg_tracks_out.items():
        valid_mask = np.isfinite(arr).all(axis=1)
        if valid_mask.all() or not valid_mask.any():
            continue
        first_valid = int(np.argmax(valid_mask))
        p_self_fv = arr[first_valid]
        nan_frames = np.where(~valid_mask)[0]
        neighbors = adjacency.get(gid, [])
        valid_neighbors, dists = [], []
        for nb in neighbors:
            nb_pos = edg_tracks_out[nb]
            if np.isfinite(nb_pos[first_valid]).all():
                valid_neighbors.append(nb)
                dists.append(np.linalg.norm(nb_pos[first_valid] - p_self_fv) + 1e-6)
        if not valid_neighbors:
            continue
        weights = 1.0 / np.asarray(dists, dtype=np.float64)
        weights /= np.sum(weights)
        for t in nan_frames:
            pred_sum = np.zeros(3, dtype=np.float64)
            w_sum = 0.0
            for nb, w in zip(valid_neighbors, weights):
                nb_pos = edg_tracks_out[nb]
                nb_rot = edg_rot_tracks_out.get(nb)
                if not (np.isfinite(nb_pos[first_valid]).all() and np.isfinite(nb_pos[t]).all()):
                    continue
                if nb_rot is None or not np.isfinite(nb_rot[first_valid]).all() or not np.isfinite(nb_rot[t]).all():
                    pred = nb_pos[t] + (p_self_fv - nb_pos[first_valid])
                else:
                    pred = nb_pos[t] + nb_rot[t] @ nb_rot[first_valid].T @ (p_self_fv - nb_pos[first_valid])
                pred_sum += w * pred
                w_sum += w
            if w_sum > 0:
                edg_tracks_out[gid][t] = pred_sum / w_sum
    return edg_tracks_out, edg_rot_tracks_out


def _check_old_result_schema(results_all):
    missing = sorted(EDG_OLD_RESULT_KEYS.difference(results_all.keys()))
    if missing:
        raise RuntimeError(f"Prior EDG result is missing EDG_old-compatible keys: {missing}")


def _make_open3d_point_cloud(points, color):
    pcd = o3d.geometry.PointCloud()
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.paint_uniform_color(np.asarray(color, dtype=np.float64).reshape(3).tolist())
    return pcd


def _make_open3d_line_set(points, edges, color):
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64).reshape(-1, 3))
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(edges, dtype=np.int32).reshape(-1, 2))
    line_set.paint_uniform_color(np.asarray(color, dtype=np.float64).reshape(3).tolist())
    return line_set


def _show_initial_edg(prior_points_world, edg_points_world, edges_local):
    if not SHOW_INITIAL_EDG_OPEN3D:
        return
    if o3d is None:
        print("[Open3D] initial EDG preview skipped because Open3D is unavailable.")
        return
    prior_points_world = np.asarray(prior_points_world, dtype=np.float64).reshape(-1, 3)
    edg_points_world = np.asarray(edg_points_world, dtype=np.float64).reshape(-1, 3)
    edges_local = np.asarray(edges_local, dtype=np.int32).reshape(-1, 2)
    if prior_points_world.shape[0] == 0 or edg_points_world.shape[0] == 0:
        print("[Open3D] initial EDG preview skipped because the prior cloud or EDG nodes are empty.")
        return
    geoms = [
        _make_open3d_point_cloud(prior_points_world, [0.35, 0.55, 1.0]),
        _make_open3d_point_cloud(edg_points_world, [1.0, 0.08, 0.05]),
    ]
    if edges_local.shape[0] > 0:
        geoms.append(_make_open3d_line_set(edg_points_world, edges_local, [1.0, 1.0, 1.0]))
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
    print(
        "[Open3D] Showing fixed initial prior EDG: "
        f"{prior_points_world.shape[0]} aligned-prior points, "
        f"{edg_points_world.shape[0]} nodes, {edges_local.shape[0]} edges."
    )
    pause_script_logger("Open3D initial EDG preview")
    try:
        o3d.visualization.draw_geometries(
            geoms,
            window_name="Initial fixed prior EDG topology",
            width=1440,
            height=900,
        )
    finally:
        resume_script_logger("Open3D initial EDG preview closed")


def _debug_visualize_segment(
    seg_id,
    edg_deformed,
    dyn_grid,
    unique_kps,
    kp_neighbors,
    edges_local,
    frames_anchor_first,
):
    """Interactive per-segment debug view (enable with EDG_DEBUG_VIZ=1). Shows, for the current
    frame of the just-solved segment: the deformed EDG nodes (red) and their graph edges (gray),
    this segment's dynamic points (green), and a yellow line from every dynamic point to each of
    its k-nearest EDG nodes (the skinning neighbours). SPACE = next frame, B = previous frame,
    Q/ESC = close and continue solving the next segment.
    """
    if not DEBUG_SEGMENT_VIZ or o3d is None:
        return
    edg = np.asarray(edg_deformed, dtype=np.float64)  # (M, T, 3)
    dyn = np.asarray(dyn_grid, dtype=np.float64)       # (Nd, T, 3)
    edges_local = np.asarray(edges_local, dtype=np.int32).reshape(-1, 2)
    if edg.ndim != 3 or edg.shape[0] == 0 or edg.shape[1] == 0:
        return
    M, T = int(edg.shape[0]), int(edg.shape[1])
    Nd = int(dyn.shape[0]) if dyn.ndim == 3 else 0
    unique_kps = np.asarray(unique_kps, dtype=int).reshape(-1)
    frames = np.asarray(frames_anchor_first, dtype=int).reshape(-1)
    # k-neighbour EDG node (local) indices for each dynamic point, in dyn_grid row order
    dyn_nbr = [
        np.asarray(kp_neighbors.get(int(unique_kps[i]), []), dtype=int).reshape(-1)
        if i < unique_kps.shape[0] else np.empty(0, dtype=int)
        for i in range(Nd)
    ]

    pcd_nodes = o3d.geometry.PointCloud()
    pcd_dyn = o3d.geometry.PointCloud()
    ls_graph = o3d.geometry.LineSet()
    ls_skin = o3d.geometry.LineSet()
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    state = {"t": 0}

    def _fill(t):
        E = edg[:, t, :].copy()
        mE = np.isfinite(E).all(axis=1)
        E[~mE] = 0.0
        pcd_nodes.points = o3d.utility.Vector3dVector(E)
        pcd_nodes.paint_uniform_color([1.0, 0.10, 0.05])  # red EDG nodes
        if Nd > 0:
            D = dyn[:, t, :].copy()
            mD = np.isfinite(D).all(axis=1)
            D[~mD] = 0.0
        else:
            D = np.zeros((0, 3), dtype=np.float64)
            mD = np.zeros(0, dtype=bool)
        pcd_dyn.points = o3d.utility.Vector3dVector(D)
        if Nd > 0:
            pcd_dyn.paint_uniform_color([0.15, 0.9, 0.35])  # green dynamic points
        # EDG graph edges (only where both endpoints are finite this frame)
        g_edges = [e for e in edges_local if mE[e[0]] and mE[e[1]]]
        ls_graph.points = o3d.utility.Vector3dVector(E)
        ls_graph.lines = o3d.utility.Vector2iVector(
            np.asarray(g_edges, dtype=np.int32).reshape(-1, 2) if g_edges else np.zeros((0, 2), dtype=np.int32)
        )
        ls_graph.paint_uniform_color([0.6, 0.6, 0.6])  # gray graph
        # skinning lines: dyn point i -> each of its k-neighbour nodes
        allpts = np.vstack([E, D]) if Nd > 0 else E
        s_lines = []
        for i in range(Nd):
            if not mD[i]:
                continue
            for nb in dyn_nbr[i]:
                if 0 <= nb < M and mE[nb]:
                    s_lines.append([M + i, int(nb)])  # dyn i offset by M
        ls_skin.points = o3d.utility.Vector3dVector(allpts)
        ls_skin.lines = o3d.utility.Vector2iVector(
            np.asarray(s_lines, dtype=np.int32).reshape(-1, 2) if s_lines else np.zeros((0, 2), dtype=np.int32)
        )
        ls_skin.paint_uniform_color([0.95, 0.75, 0.1])  # yellow skinning
        return int(mE.sum()), int(mD.sum()) if Nd > 0 else 0, len(s_lines)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"EDG debug - segment {seg_id}", width=1440, height=900)
    _fill(0)  # populate BEFORE add_geometry so the camera fits the content
    for g in (pcd_nodes, pcd_dyn, ls_graph, ls_skin, axes):
        vis.add_geometry(g)

    def _refresh():
        nE, nD, nS = _fill(state["t"])
        for g in (pcd_nodes, pcd_dyn, ls_graph, ls_skin):
            vis.update_geometry(g)
        wf = int(frames[state["t"]]) if state["t"] < frames.shape[0] else state["t"]
        print(f"[EDG-debug] seg {seg_id} frame {state['t'] + 1}/{T} (world frame {wf}): "
              f"nodes={nE} dyn={nD} skin-lines={nS}")

    def on_next(v):
        state["t"] = (state["t"] + 1) % T
        _refresh()
        return False

    def on_prev(v):
        state["t"] = (state["t"] - 1) % T
        _refresh()
        return False

    def on_close(v):
        v.close()
        return False

    vis.register_key_callback(32, on_next)        # SPACE
    vis.register_key_callback(ord("B"), on_prev)  # B
    vis.register_key_callback(ord("Q"), on_close)
    vis.register_key_callback(256, on_close)      # ESC

    print(f"[EDG-debug] segment {seg_id}: {M} EDG nodes, {Nd} dynamic points, {T} frames. "
          f"red=nodes gray=graph green=dyn yellow=skinning. SPACE=next frame, B=prev, Q/ESC=continue.")
    pause_script_logger("EDG debug segment viewer")
    try:
        _refresh()
        vis.run()
    finally:
        vis.destroy_window()
        resume_script_logger("EDG debug segment viewer closed")


def _assert_fixed_topology(edg_global_ids, edges_local, fixed_global_ids, fixed_edges_local):
    if not np.array_equal(np.asarray(edg_global_ids, dtype=np.int64), fixed_global_ids):
        raise RuntimeError("Prior EDG global node IDs changed after initialization.")
    if not np.array_equal(np.asarray(edges_local, dtype=np.int64), fixed_edges_local):
        raise RuntimeError("Prior EDG edge topology changed after initialization.")


prior_world_points, prior_npz_path, prior_alignment_source_key = _load_prior_alignment()
prior_sel = _farthest_point_sample(prior_world_points, PRIOR_EDG_SAMPLES)
prior_edg_world = prior_world_points[prior_sel].astype(np.float64)
prior_edges_local = _build_fixed_edges_knn(prior_edg_world, EDGE_TARGET_NEIGHBORS)
prior_edg_global_ids = (EDG_GID_OFFSET + np.arange(prior_edg_world.shape[0], dtype=np.int64)).astype(int)
prior_edges_global = [
    (int(prior_edg_global_ids[a]), int(prior_edg_global_ids[b])) for a, b in prior_edges_local
]
fixed_edg_global_ids = prior_edg_global_ids.astype(np.int64).copy()
fixed_edges_local = np.asarray(prior_edges_local, dtype=np.int64).copy()
fixed_edges_global = np.asarray(prior_edges_global, dtype=np.int64).copy()
_show_initial_edg(prior_world_points, prior_edg_world, fixed_edges_local)

results_all = _init_results_store()
results_all["n_tracks"] = int(tracks_xy.shape[1])          # track-id space stamp (merged vs raw)
results_all["tracks_source"] = np.array(_tracks_path.name)
results_all["prior_alignment_path"] = str(prior_npz_path)
results_all["prior_alignment_source_key"] = prior_alignment_source_key
results_all["prior_full_source_points_world"] = prior_world_points.astype(np.float64)
results_all["prior_topology_fixed"] = True
results_all["prior_source_points_world"] = prior_world_points.astype(np.float64)
results_all["prior_edg_rest_world"] = prior_edg_world.astype(np.float64)
results_all["prior_edg_source_indices"] = prior_sel.astype(np.int64)
results_all["prior_edg_global_ids"] = fixed_edg_global_ids.copy()
results_all["prior_edges_local"] = fixed_edges_local.copy()
results_all["prior_edges_global"] = fixed_edges_global.copy()

seg_id = 0
current_frame = 0
prev_segment = None

while True:
    if current_frame >= T:
        break
    end_frame_excl = int(min(T, current_frame + BUILD_EDG_SEGMENT_LENGTH))
    window = np.arange(current_frame, end_frame_excl, dtype=int)
    if window.size == 0:
        break
    # Rest/ARAP reference is a KEYFRAME that advances every EDG_KEYFRAME_INTERVAL frames:
    # within a block, all segments reference the block's keyframe rest (frame-0 prior for the
    # first block; the solved EDG state at the keyframe for later blocks). This bounds drift
    # (re-anchor only every N frames) while following the object. Prepend the keyframe as t=1.
    kf = int(min((current_frame // EDG_KEYFRAME_INTERVAL) * EDG_KEYFRAME_INTERVAL, T - 1))
    rest_frame = kf
    if kf in window.tolist():
        frames_seg_full = window
    else:
        frames_seg_full = np.concatenate([[kf], window]).astype(int)

    print(f"\n=== Prior Segment {seg_id}: frames [{current_frame}, {end_frame_excl - 1}] rest={rest_frame} (keyframe @ {kf}) ===")
    seg_mask_any = np.any(valid_mask_global[:, frames_seg_full], axis=1)
    global_index_cand = np.where(seg_mask_any)[0].astype(int)
    if global_index_cand.size == 0:
        print("No candidates appear in this segment; stop.")
        break

    global_index_cand, tracks_glob_candidate, _, valid_mask_candidate_global, _ = _build_candidate_views(global_index_cand)

    _assert_fixed_topology(prior_edg_global_ids, prior_edges_local, fixed_edg_global_ids, fixed_edges_local)
    # rest pose g = the keyframe's EDG state: frame-0 canonical prior for the first block;
    # for later blocks, the solved deformed node positions at the keyframe (from prior
    # segments). ARAP compares to this keyframe rest, re-anchored every EDG_KEYFRAME_INTERVAL.
    if kf == 0:
        current_edg_rest_world = prior_edg_world.copy()
    else:
        kf_rest = prior_edg_world.copy()
        for j, gid in enumerate(prior_edg_global_ids.tolist()):
            arr = results_all["edg_reconstruct"].get(int(gid))
            if arr is not None and np.isfinite(arr[kf]).all():
                kf_rest[j] = np.asarray(arr[kf], dtype=np.float64)
        current_edg_rest_world = kf_rest

    _save_rest_candidates(results_all, seg_id, rest_frame, frames_seg_full, global_index_cand, prior_edg_global_ids)

    solver_in = _build_fixed_edg_solver_inputs_for_segment(
        tracks_glob_candidate=tracks_glob_candidate,
        valid_mask_candidate_global=valid_mask_candidate_global,
        frames_anchor_first=frames_seg_full,
        rest_frame=rest_frame,
        global_index_cand=global_index_cand,
        edg_rest_points=current_edg_rest_world,
        edge_local_pairs=prior_edges_local,
        edg_global_ids=prior_edg_global_ids,
        persistent_skinning_nodes=results_all["dynamic_skinning_nodes"],
        persistent_skinning_weights=results_all["dynamic_skinning_weights"],
        persistent_skinning_rest=results_all["dynamic_skinning_rest"],
        persistent_skinning_bind_frame=results_all["dynamic_skinning_bind_frame"],
        K_neighbors=K_neighbors,
        edg_recon=results_all["edg_reconstruct"],
        edg_rot=results_all["edg_rotation"],
    )
    results_all["edg_input"][seg_id] = solver_in["g"]

    edg_solver_out = solve_qcqp_and_recover_until_part5(
        edges=solver_in["edges"],
        edg_idx_map=solver_in["edg_idx_map"],
        g=solver_in["g"],
        edg_obs=solver_in["edg_obs"],
        edg_obs_mask=solver_in["edg_obs_mask"],
        N_dyn_obs=solver_in["N_dyn_obs"],
        dyn_kp_idx=solver_in["dyn_kp_idx"],
        dyn_frame=solver_in["dyn_frame"],
        dyn_global=solver_in["dyn_global"],
        frame_to_trev=solver_in["frame_to_trev"],
        kp_neighbors=solver_in["kp_neighbors"],
        kp_weights=solver_in["kp_weights"],
        kp_rest=solver_in["kp_rest"],
        output_dir=str(output_dir),
        solve_Ronly_fn=_solve_Ronly_isolated,
        load_matrix_from_bin_fn=load_matrix_from_bin,
        solver_max_iter=EDG_SOLVER_MAX_TIME,  # NOTE: edg.py 'solver_max_iter' is passed to XM as max_time (SECONDS), not iterations
        solver_tol=EDG_SOLVER_TOL,
    )

    edg_deformed = np.asarray(edg_solver_out["edg_deformed"], dtype=np.float64)
    R_edg = np.asarray(edg_solver_out["R_edg"], dtype=np.float64)
    dyn_mapped = edg_solver_out["dyn_mapped"]
    frames_anchor_first = solver_in["frames_anchor_first"]
    T_rev = len(frames_anchor_first)

    for j, gid in enumerate(prior_edg_global_ids.tolist()):
        arr = _ensure_track(results_all["edg_reconstruct"], gid, T)
        for t_idx, f in enumerate(frames_anchor_first.tolist()):
            arr[int(f), :] = edg_deformed[j, t_idx, :]
        rot_arr = _ensure_rotation(results_all["edg_rotation"], gid, T)
        rest_global = rot_arr[int(rest_frame)].copy() if rest_frame < T and np.isfinite(rot_arr[int(rest_frame)]).all() else np.eye(3, dtype=np.float64)
        R_global = _compose_global_rotations(R_edg[j], rest_global)
        for t_idx, f in enumerate(frames_anchor_first.tolist()):
            rot_arr[int(f), :, :] = R_global[t_idx]

    for ga, gb in prior_edges_global:
        aa, bb = (int(ga), int(gb)) if int(ga) < int(gb) else (int(gb), int(ga))
        results_all["edg_edges_union"].add((aa, bb))

    unique_kps = np.unique(solver_in["dyn_kp_idx"]).astype(int)
    Nd = len(unique_kps)
    dyn_grid = _dyn_grid_from_solver_output(dyn_mapped, Nd=Nd, T_rev=T_rev)
    dyn_global_ids = global_index_cand[unique_kps].astype(int).tolist()
    for i, gid in enumerate(dyn_global_ids):
        arr = _ensure_track(results_all["dynamics_points_reconstruct"], gid, T)
        for t_idx, f in enumerate(frames_anchor_first.tolist()):
            arr[int(f), :] = dyn_grid[i, t_idx, :]

    # Debug: interactively inspect this just-solved segment (EDG nodes+graph, dynamic points, and
    # each dynamic point's k-neighbour EDG connections). Blocks until the window is closed, then the
    # solve continues to the next segment. No-op unless EDG_DEBUG_VIZ=1.
    _debug_visualize_segment(
        seg_id=seg_id,
        edg_deformed=edg_deformed,
        dyn_grid=dyn_grid,
        unique_kps=unique_kps,
        kp_neighbors=solver_in["kp_neighbors"],
        edges_local=prior_edges_local,
        frames_anchor_first=frames_anchor_first,
    )

    results_all["segments"].append(
        dict(
            segment_id=seg_id,
            frames_anchor_first=frames_anchor_first.tolist(),
            edg_global_ids=[int(g) for g in prior_edg_global_ids.tolist()],
            edges_global=[(int(a), int(b)) for a, b in prior_edges_global],
            edg_deformed=edg_deformed.copy(),
            skipped_far_from_edg_count=int(solver_in.get("skipped_far_from_edg_count", 0)),
            reused_skinning_count=int(solver_in.get("reused_skinning_count", 0)),
        )
    )
    prev_segment = results_all["segments"][-1]

    if end_frame_excl >= T:
        break
    current_frame = int(end_frame_excl - 1)
    seg_id += 1

_finalize_dynamic_outputs(results_all, num_points=glob_all_full.shape[0], T=T, K_neighbors=K_neighbors)
results_all["edg_reconstruct"], results_all["edg_rotation"] = _fill_nan_edg_tracks(
    results_all["edg_reconstruct"],
    results_all["edg_rotation"],
    results_all["edg_edges_union"],
)
_check_old_result_schema(results_all)

vis_out = output_dir / "edg_solution_vis.npz"
edg_gids = prior_edg_global_ids.astype(int).tolist()
edg_tracks_full = np.stack(
    [results_all["edg_reconstruct"][gid] for gid in edg_gids], axis=0
).astype(np.float64)
edg_rot_full = np.stack(
    [results_all["edg_rotation"][gid] for gid in edg_gids], axis=0
).astype(np.float64)
edges_arr = np.asarray(sorted(results_all["edg_edges_union"]), dtype=np.int64)
dyn_gids = sorted(results_all["dynamics_points_reconstruct"].keys())
if len(dyn_gids) > 0:
    dyn_mapped = np.full((len(dyn_gids), T, 3), np.nan, dtype=np.float64)
    for i, gid in enumerate(dyn_gids):
        dyn_mapped[i, :, :] = results_all["dynamics_points_reconstruct"][gid]
else:
    dyn_mapped = np.zeros((0, T, 3), dtype=np.float64)

np.savez_compressed(
    vis_out,
    edg_deformed=edg_tracks_full,
    edg_rotation=edg_rot_full,
    edge_pairs=edges_arr,
    dyn_mapped=dyn_mapped,
)
print(f"[VIS] Wrote {vis_out}")

save_path = output_dir / PRIOR_RESULT_NAME
np.savez_compressed(save_path, **results_all)
print(f"Saved prior-guided EDG results to {save_path}")
