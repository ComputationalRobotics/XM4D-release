import glob
from pathlib import Path

import numpy as np
import cv2

import utils.definitions
from utils.edg import (
    build_initial_edg_graph,
    scatter_solved_frame_vector,
    scatter_solved_poses,
    solve_qcqp_and_recover_until_part5,
)
from utils.io import load_matrix_from_bin
from utils.readgt_DAVIS import load_davis_gt
from utils.readgt_TUM import load_tum_gt
from utils.readgt_sintel import load_sintel_gt
from utils.readgt_colmap import load_colmap_gt
from utils.readgt_dycheck import load_dycheck_gt
from visualize_syn4d_gt import (
    _apply_similarity_points as _syn4d_apply_similarity_points,
    _load_gt_cameras as _syn4d_load_gt_cameras,
    _load_solved as _syn4d_load_solved,
    _mosca_align_with_evo as _syn4d_mosca_align_with_evo,
    _teaser_align_camera_centers as _syn4d_teaser_align_camera_centers,
)


import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../XM/build/"))
import XM

########################################################################
### Part 1 load data
########################################################################

output_dir = utils.definitions.output_dir

BUILD_EDG_SEGMENR_LENGTH = 11  # step in tracking + 1 to include last frame

data = np.load(utils.definitions.tracks_dynamics_path, allow_pickle=True)
tracks_xy = data["tracks_xy"]  # [1, Q, T, 2]
tracks_xyz = data["tracks_xyz"]  # [1, Q, T, 3]
occluded = data["occluded"]  # [1, Q, T]
dynamic_obs_mask = data["dynamic_obs_mask"][:, :]

T = tracks_xy.shape[2]
tracks_xy = tracks_xy[:, :, :T]
tracks_xyz = tracks_xyz[:, :, :T]
occluded = occluded[:, :, :T]
dynamic_obs_mask = dynamic_obs_mask[:, :T]
print(f"Loaded {T} frames and {tracks_xy.shape[1]} tracks")

dataset_cat = utils.definitions.dataset_cat
rootpath = utils.definitions.rootpath

gt = None
try:
    if dataset_cat in {"Sintel", "Sintel_debug"}:
        gt = load_sintel_gt(str(rootpath))
    elif dataset_cat == "DAVIS":
        gt = load_davis_gt(str(rootpath))
    elif dataset_cat == "Syn4D":
        gt = load_colmap_gt(str(rootpath / "colmap"))
    elif dataset_cat == "Dycheck":
        gt = load_dycheck_gt(str(rootpath))
    else:
        raise ValueError(f"Unsupported dataset_cat '{dataset_cat}'")
except Exception as exc:
    print(f"[GT] Could not load GT metadata; continuing without GT visualization/alignment: {exc}")
    gt = None


# Build intrinsics_all from gt entries ordered by frame id
if gt:
    gt_items = sorted(gt.items(), key=lambda kv: int(kv[1]["id"]))
    intrinsics_all = np.stack([kv[1]["K"].astype(np.float64) for kv in gt_items], axis=0)
else:
    gt_items = []
    intrinsics_all = None

# load gt extrinsics
rt = np.load(str(utils.definitions.rootpath / "outputs" / "result_XM.npz"))
landmarks = rt["p"].T if "p" in rt else None
# The static solver may drop frames (checklandmarks / XM^2), so R/t/s are stored in
# solved order (N<=T). Scatter them back onto the full T-frame timeline via indices_all;
# dropped frames get identity R + NaN t so valid_mask_global excludes them.
R_raw, t_raw = scatter_solved_poses(rt, T)
s_raw = scatter_solved_frame_vector(rt, "s", T, default=1.0)

R_for_edg = R_raw.astype(np.float64).copy()
if s_raw is not None:
    R_for_edg *= s_raw[:, None, None]

# Build once over the whole video in world frame

glob_all_full = np.einsum("tij,qtj->qti", R_for_edg, tracks_xyz[0].astype(np.float64)) + t_raw[None, :, :]
valid_mask_global = (
    np.isfinite(glob_all_full).all(axis=-1)
    & (occluded[0] == 0)
    & dynamic_obs_mask.astype(bool)
)



########################################################################
### Helpers
########################################################################

# =========================================================================
# MoSca-borrowed robust topology  (github.com/JiahuiLei/MoSca, arXiv:2405.17421)
# -------------------------------------------------------------------------
# Two tracks are "neighbors" only if they stay close in BOTH 3D (world) and 2D
# (pixels) across the whole video. Per-pair, per-frame distance is aggregated
# over co-visible frames by a robust high percentile (MoSca's robust curve
# distance = a robustified max, lib_mosca/mosca.py:robust_curve_dist_kernel), so
# a few bad depth/tracking frames cannot create or destroy an edge. Requiring
# BOTH 3D and 2D proximity rejects the false neighbours (good-2D/bad-depth or
# 2D-outlier) that made the old raw-max, 3D-only topology produce weird shapes.
#
# Only the TOPOLOGY is changed here. Node sampling stays as EDG_old's 2D-spatial
# farthest-point selector (build_initial_edg_graph): it spreads nodes across the
# image and picks well-observed tracks, which keeps the segment solver well
# constrained. Sampling nodes by curve distance instead over-selects erratic /
# outlier tracks that are poorly observed within a segment, making the solver's
# Schur complement singular -- so it is deliberately not used. Everything
# downstream (segment solver, skinning, results_all.npz) is unchanged.
# =========================================================================
_MOSCA_ROBUST_PCT = 85.0      # percentile of per-frame distance over co-visible frames
_MOSCA_MIN_COMMON = 3         # need >= this many co-visible frames to form/consider an edge


def _mosca_node_ref_pos(xyz, mask):
    # xyz [K,T,C], mask [K,T] -> [K,C] mean over each node's visible frames
    w = mask[..., None].astype(np.float64)
    s = (np.nan_to_num(xyz) * w).sum(axis=1)
    c = np.clip(w.sum(axis=1), 1.0, None)
    return s / c


def _mosca_curve_scales(xyz_ref, xy_ref):
    # typical node spacing in 3D and 2D -> makes the 3D and 2D distances comparable
    from scipy.spatial import cKDTree

    def _median_nn(P):
        P = np.asarray(P, dtype=np.float64)
        P = P[np.isfinite(P).all(axis=1)]
        if P.shape[0] < 2:
            return 1.0
        d, _ = cKDTree(P).query(P, k=2)
        nn = d[:, 1]
        nn = nn[np.isfinite(nn) & (nn > 0)]
        return float(np.median(nn)) if nn.size else 1.0

    return max(_median_nn(xyz_ref), 1e-9), max(_median_nn(xy_ref), 1e-9)


def _mosca_robust_dist_1_to_M(a_xyz, a_xy, a_m, B_xyz, B_xy, B_m, s3d, s2d,
                              robust_pct=_MOSCA_ROBUST_PCT, min_common=_MOSCA_MIN_COMMON):
    """Combined robust 3D+2D curve distance from curve `a` to M curves `B`.

    a_xyz [T,3] a_xy [T,2] a_m [T];  B_xyz [M,T,3] B_xy [M,T,2] B_m [M,T].
    Per co-visible frame: max(||d3d||/s3d, ||d2d||/s2d)  -> small only if BOTH are
    near. Aggregate with the `robust_pct` percentile (robustified max). Returns [M],
    inf where < min_common co-visible frames.
    """
    import warnings
    co = a_m[None, :].astype(bool) & B_m.astype(bool)             # [M,T]
    d3 = np.linalg.norm(a_xyz[None] - B_xyz, axis=-1) / float(s3d)  # [M,T]
    d2 = np.linalg.norm(a_xy[None] - B_xy, axis=-1) / float(s2d)
    dc = np.where(co, np.maximum(d3, d2), np.nan)                 # both-near -> small
    cnt = co.sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        rob = np.nanpercentile(dc, robust_pct, axis=1)
    rob = np.where((cnt >= int(min_common)) & np.isfinite(rob), rob, np.inf)
    return rob


def _mosca_build_edges(tracks_glob_candidate, tracks_xy_candidate,
                       valid_mask_candidate_global, edg_nodes,
                       target_neighbors=6, min_common_frames=_MOSCA_MIN_COMMON):
    """k-NN topology on the robust 3D+2D curve distance (MoSca-style, no gate).

    Robustness comes from the distance metric itself: a neighbor is only "near" when
    BOTH its 3D and 2D distances stay small across co-visible frames (robust
    percentile), so good nodes never pick a bad-depth / 2D-outlier track as a
    neighbor. Every node still gets its `target_neighbors` nearest so the graph is
    connected enough for the solver (an isolated node makes the Schur complement
    singular). A node with no finite-distance neighbor (too few co-visible frames
    with anyone) falls back to its spatially nearest node so it is never isolated.
    """
    nodes = list(map(int, edg_nodes))
    if len(nodes) <= 1:
        return [], {}
    xyz = tracks_glob_candidate[0][nodes]        # [K,T,3]
    xy = tracks_xy_candidate[0][nodes]           # [K,T,2]
    m = valid_mask_candidate_global[nodes]       # [K,T]
    K = len(nodes)
    ref_xyz = _mosca_node_ref_pos(xyz, m)
    ref_xy = _mosca_node_ref_pos(xy, m)
    s3d, s2d = _mosca_curve_scales(ref_xyz, ref_xy)
    D = np.full((K, K), np.inf, dtype=np.float64)
    for i in range(K):
        D[i] = _mosca_robust_dist_1_to_M(xyz[i], xy[i], m[i], xyz, xy, m, s3d, s2d,
                                         min_common=min_common_frames)
        D[i, i] = np.inf
    # Euclidean fallback distance (always finite) for nodes disconnected in D.
    E = np.linalg.norm(ref_xyz[:, None, :] - ref_xyz[None, :, :], axis=-1)
    np.fill_diagonal(E, np.inf)
    edges, scores = set(), {}
    for i in range(K):
        order = np.argsort(D[i])
        cnt = 0
        for j in order:
            if cnt >= int(target_neighbors) or not np.isfinite(D[i, int(j)]):
                break
            a, b = sorted((nodes[i], nodes[int(j)]))
            if a != b:
                edges.add((a, b))
                scores[(a, b)] = float(D[i, int(j)])
                cnt += 1
        if cnt == 0:  # no co-visible neighbor -> keep spatially nearest so it is not isolated
            j = int(np.argmin(E[i]))
            a, b = sorted((nodes[i], nodes[j]))
            if a != b:
                edges.add((a, b))
                scores[(a, b)] = float(E[i, j])
    edges = sorted(edges)
    return edges, {e: scores.get(e, np.nan) for e in edges}


def _visualize_edg_rest_2d(
    *,
    tracks_xy_candidate,
    valid_mask_candidate_global,
    edg_nodes,
    edges_node,
    edge_scores,
    rest_frame,
    output_path,
):
    """Render a 2D EDG snapshot at rest_frame with edge colors by stability score."""
    image_paths = sorted(glob.glob(str(utils.definitions.video_dir / utils.definitions.image_pattern)))
    if rest_frame >= len(image_paths):
        raise RuntimeError(f"Not enough images to render rest_frame={rest_frame}.")

    img = cv2.imread(image_paths[rest_frame], cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_paths[rest_frame]}")

    edg_nodes = list(map(int, edg_nodes))
    edg_set = set(edg_nodes)
    pts = tracks_xy_candidate[0, edg_nodes, rest_frame, :]
    valid_nodes = np.isfinite(pts).all(axis=1) & valid_mask_candidate_global[edg_nodes, rest_frame]

    # normalize scores for color mapping (lower score -> green, higher -> red)
    score_vals = [edge_scores.get(tuple(edge), np.nan) for edge in edges_node]
    score_vals = np.asarray(score_vals, dtype=np.float64)
    score_vals = score_vals[np.isfinite(score_vals)]
    if score_vals.size > 0:
        smin, smax = float(score_vals.min()), float(score_vals.max())
        if smax - smin < 1e-12:
            smax = smin + 1e-12
    else:
        smin, smax = 0.0, 1.0

    for edge in edges_node:
        a, b = int(edge[0]), int(edge[1])
        if a not in edg_set or b not in edg_set:
            continue
        ia = edg_nodes.index(a)
        ib = edg_nodes.index(b)
        if not (valid_nodes[ia] and valid_nodes[ib]):
            continue
        xa, ya = pts[ia]
        xb, yb = pts[ib]

        score = edge_scores.get(tuple((a, b)) if a < b else tuple((b, a)), np.nan)
        if np.isfinite(score):
            val = int(np.clip(255 * (score - smin) / (smax - smin), 0, 255))
            color = cv2.applyColorMap(np.uint8([[val]]), cv2.COLORMAP_TURBO)[0, 0].tolist()
            color = (int(color[0]), int(color[1]), int(color[2]))
        else:
            color = (200, 200, 200)

        cv2.line(img, (int(round(xa)), int(round(ya))), (int(round(xb)), int(round(yb))), color, 2)

    for i, ok in enumerate(valid_nodes):
        if not ok:
            continue
        x, y = pts[i]
        cv2.circle(img, (int(round(x)), int(round(y))), 3, (0, 0, 255), -1)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    print(f"Saved EDG rest-frame visualization: {output_path}")


def _add_far_nodes_greedy(
    tracks_glob_candidate,
    valid_mask_candidate_global,
    valid_mask_candidate_global_orig,
    rest_frame,
    edg_nodes,
    edges_node,
    graph_knn,
    max_new_nodes=999999,
    dist_factor_3d=1.2,
    force_min_new=0,
):
    """
    Add new EDG nodes far from currently visible EDG nodes.

    valid_mask_candidate_global:
        possibly modified mask used by the solver, including synthetic carryover validity.

    valid_mask_candidate_global_orig:
        original observation mask from occluded/dynamic_obs_mask before synthetic carryover fixes.
        This is used for visibility and farthest-point selection.
    """
    edg_nodes = list(map(int, edg_nodes))
    edges_node = list(edges_node)

    valid_rest_solver = valid_mask_candidate_global[:, rest_frame].astype(bool)
    valid_rest_orig = valid_mask_candidate_global_orig[:, rest_frame].astype(bool)

    # Keep solver-valid EDG nodes in the graph.
    edg_nodes = [n for n in edg_nodes if bool(valid_rest_solver[int(n)])]
    edg_set = set(edg_nodes)

    if len(edg_nodes) == 0:
        return edg_nodes, edges_node, []

    # For farthest-point distance, only use originally visible EDG nodes.
    edg_pos_all = tracks_glob_candidate[0, edg_nodes, rest_frame].astype(np.float64)

    visible_ref_mask = (
        valid_rest_orig[np.asarray(edg_nodes, dtype=int)]
        & np.isfinite(edg_pos_all).all(axis=1)
    )

    visible_ref_nodes = [n for n, ok in zip(edg_nodes, visible_ref_mask) if bool(ok)]

    if len(visible_ref_nodes) == 0:
        print("[EDG grow] No originally visible EDG reference nodes; skip adding.")
        return edg_nodes, sorted(set((min(a, b), max(a, b)) for a, b in edges_node if a != b)), []

    ref_pos = tracks_glob_candidate[0, visible_ref_nodes, rest_frame].astype(np.float64)
    finite_ref = np.isfinite(ref_pos).all(axis=1)
    ref_pos = ref_pos[finite_ref]
    visible_ref_nodes = [n for n, ok in zip(visible_ref_nodes, finite_ref) if bool(ok)]

    if len(visible_ref_nodes) == 0:
        print("[EDG grow] No finite originally visible EDG reference nodes; skip adding.")
        return edg_nodes, sorted(set((min(a, b), max(a, b)) for a, b in edges_node if a != b)), []

    if ref_pos.shape[0] >= 2:
        dmat = np.linalg.norm(ref_pos[:, None, :] - ref_pos[None, :, :], axis=2)
        np.fill_diagonal(dmat, np.inf)
        avg_nn_dist = float(np.mean(np.min(dmat, axis=1)))
    else:
        avg_nn_dist = 0.0

    candidate_ids = []
    for i in range(tracks_glob_candidate.shape[1]):
        # Candidate must be truly visible in the original mask.
        if not valid_rest_orig[i] or i in edg_set:
            continue

        pxyz = tracks_glob_candidate[0, i, rest_frame, :]

        if not np.isfinite(pxyz).all():
            continue

        candidate_ids.append(i)

    print(
        f"[EDG grow debug] rest_frame={rest_frame} "
        f"solver_valid_rest={int(np.sum(valid_rest_solver))} "
        f"orig_valid_rest={int(np.sum(valid_rest_orig))} "
        f"edg_nodes={len(edg_nodes)} "
        f"visible_ref_nodes={len(visible_ref_nodes)} "
        f"candidate_ids={len(candidate_ids)} "
        f"avg_nn_dist_3d={avg_nn_dist:.6f}"
    )

    new_nodes = []

    while candidate_ids and len(new_nodes) < max_new_nodes:
        cand_pts = tracks_glob_candidate[0, candidate_ids, rest_frame].astype(np.float64)
        dmat = np.linalg.norm(cand_pts[:, None, :] - ref_pos[None, :, :], axis=2)
        min_d = np.min(dmat, axis=1)

        ok3d = min_d > float(dist_factor_3d) * float(avg_nn_dist)
        ok = ok3d

        print(
            f"[EDG grow debug] candidates={len(candidate_ids)} "
            f"ok3d={int(np.sum(ok3d))} "
            f"ok_both={int(np.sum(ok))} "
            f"max_min_d_3d={float(np.max(min_d)):.6f} "
            f"thr3d={float(dist_factor_3d) * float(avg_nn_dist):.6f}"
        )

        if np.any(ok):
            score = (
                min_d / (avg_nn_dist + 1e-8)
            )
            valid_ids = np.where(ok)[0]
            best_idx = int(valid_ids[np.argmax(score[valid_ids])])
            reason = "passed_threshold"
        else:
            if len(new_nodes) < force_min_new:
                score = (
                    min_d / (avg_nn_dist + 1e-8)
                )
                best_idx = int(np.argmax(score))
                reason = "forced_farthest"
            else:
                break

        nid = int(candidate_ids[best_idx])
        p = tracks_glob_candidate[0, nid, rest_frame].astype(np.float64)

        print(
            f"[EDG grow debug] adding node local={nid}, "
            f"reason={reason}, "
            f"min_d_3d={float(min_d[best_idx]):.6f}, "
        )

        # Connect new node to kNN among originally visible reference EDG nodes.
        d = np.linalg.norm(ref_pos - p[None, :], axis=1)
        k_eff = min(graph_knn, len(d))

        if k_eff > 0:
            nn = np.argpartition(d, k_eff - 1)[:k_eff]
            nn = nn[np.argsort(d[nn])]

            for j in nn:
                other = int(visible_ref_nodes[int(j)])
                a, b = (nid, other) if nid < other else (other, nid)
                if a != b:
                    edges_node.append((a, b))

        edg_nodes.append(nid)
        edg_set.add(nid)
        new_nodes.append(nid)

        # Newly added node is truly visible, so it becomes a reference.
        ref_pos = np.vstack([ref_pos, p.reshape(1, 3)])
        visible_ref_nodes.append(nid)

        candidate_ids.pop(best_idx)

    edges_node = sorted(set((min(a, b), max(a, b)) for a, b in edges_node if a != b))
    return edg_nodes, edges_node, new_nodes


def _build_solver_inputs_for_segment(
    tracks_glob_candidate,
    valid_mask_candidate_global,
    frames_anchor_first,
    rest_frame,
    edg_ids_node,
    edges_node,
    global_index_cand,
    persistent_skinning_nodes,
    persistent_skinning_weights,
    persistent_skinning_rest,
    persistent_skinning_bind_frame,
    K_neighbors=4,
):
    frames_anchor_first = np.array(frames_anchor_first, dtype=int)
    T_rev = len(frames_anchor_first)
    frame_to_trev = {int(f): (i + 1) for i, f in enumerate(frames_anchor_first.tolist())}

    valid_mask_candidate = valid_mask_candidate_global.astype(bool)

    edg_ids = np.array(edg_ids_node, dtype=int)
    edg_idx_map = {int(nid): j for j, nid in enumerate(edg_ids.tolist())}

    global_index_cand = np.asarray(global_index_cand, dtype=int)

    edg_local_ids = edg_ids.copy()
    edg_global_ids = global_index_cand[edg_local_ids]

    edg_global_to_local_id = {
        int(gid): int(lid)
        for gid, lid in zip(edg_global_ids.tolist(), edg_local_ids.tolist())
    }

    edg_local_id_set = set(map(int, edg_local_ids.tolist()))
    
    g = tracks_glob_candidate[0, edg_ids, rest_frame].astype(np.float64)

    edg_obs = np.full((len(edg_ids), T_rev, 3), np.nan, dtype=np.float64)
    edg_obs_mask = np.zeros((len(edg_ids), T_rev), dtype=bool)
    for j, nid in enumerate(edg_ids.tolist()):
        for t_idx, f in enumerate(frames_anchor_first.tolist()):
            if valid_mask_candidate[int(nid), int(f)]:
                P = tracks_glob_candidate[0, int(nid), int(f)].astype(np.float64)
                if np.isfinite(P).all():
                    edg_obs[j, t_idx, :] = P
                    edg_obs_mask[j, t_idx] = True

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

    dyn_kp_idx = np.array(dyn_kp_idx_list, dtype=int)
    dyn_frame = np.array(dyn_frame_list, dtype=int)
    dyn_global = np.array(dyn_global_list, dtype=np.float64)

    if dyn_kp_idx.size == 0:
        raise RuntimeError("No dynamic observations available for solver in this segment.")

    kp_neighbors = {}
    kp_weights = {}
    edg_pos = g.copy()
    edg_gid = edg_ids.copy()
    edg_gid_to_local = {int(gid): i for i, gid in enumerate(edg_gid.tolist())}

    for local_id in range(Qcand):
        if not valid_mask_candidate[local_id, rest_frame]:
            continue
        pr = tracks_glob_candidate[0, local_id, rest_frame].astype(np.float64)
        if not np.isfinite(pr).all():
            continue

        global_gid = int(global_index_cand[local_id])
        if global_gid in persistent_skinning_nodes and global_gid in persistent_skinning_weights:
            saved_edg_global_ids = np.asarray(
                persistent_skinning_nodes[global_gid], dtype=int
            )
            saved_w = np.asarray(
                persistent_skinning_weights[global_gid], dtype=np.float64
            )

            can_reuse = (
                saved_edg_global_ids.size >= 2
                and saved_edg_global_ids.size == saved_w.size
                and all(int(gid) in edg_global_to_local_id for gid in saved_edg_global_ids.tolist())
            )

            if can_reuse:
                saved_edg_local_ids = np.array(
                    [edg_global_to_local_id[int(gid)] for gid in saved_edg_global_ids.tolist()],
                    dtype=int,
                )

                kp_neighbors[int(local_id)] = saved_edg_local_ids
                kp_weights[int(local_id)] = saved_w.astype(np.float64)

                if global_gid not in persistent_skinning_rest:
                    persistent_skinning_rest[global_gid] = pr.astype(np.float64)
                if global_gid not in persistent_skinning_bind_frame:
                    persistent_skinning_bind_frame[global_gid] = int(rest_frame)

                continue

        d = np.linalg.norm(edg_pos - pr[None, :], axis=1)
        if d.size == 0:
            continue

        K_eff = min(K_neighbors, len(d))
        nn = np.argpartition(d, K_eff - 1)[:K_eff]
        nn = nn[np.argsort(d[nn])]
        nn_gids = edg_gid[nn]
        nn_dist = d[nn]

        if np.any(nn_dist <= 1e-8):
            w = np.zeros_like(nn_dist)
            w[np.argmin(nn_dist)] = 1.0
        else:
            w = 1.0 / (nn_dist + 1e-8)
            w = w / w.sum()

        if nn_gids.size < 2:
            continue

        nn_local_ids = edg_local_ids[nn]
        nn_global_ids = global_index_cand[nn_local_ids]

        kp_neighbors[int(local_id)] = nn_local_ids.astype(int)
        kp_weights[int(local_id)] = w.astype(np.float64)

        if global_gid not in persistent_skinning_nodes:
            persistent_skinning_nodes[global_gid] = nn_global_ids.astype(int)
            persistent_skinning_weights[global_gid] = w.astype(np.float64)
            persistent_skinning_rest[global_gid] = pr.astype(np.float64)
            persistent_skinning_bind_frame[global_gid] = int(rest_frame)

    valid_kps = np.array(list(kp_neighbors.keys()), dtype=int)
    keep = np.isin(dyn_kp_idx, valid_kps)
    dyn_kp_idx = dyn_kp_idx[keep]
    dyn_frame = dyn_frame[keep]
    dyn_global = dyn_global[keep]
    N_dyn_obs = len(dyn_kp_idx)

    if N_dyn_obs == 0:
        raise RuntimeError("All dynamic obs removed by kp_neighbors filtering.")

    kp_rest = {}
    mask_rest = (dyn_frame == int(rest_frame))
    for k in np.flatnonzero(mask_rest):
        kp = int(dyn_kp_idx[k])
        if kp not in kp_rest:
            kp_rest[kp] = dyn_global[k, :].astype(np.float64)

    valid_edge_set = set(edg_idx_map.keys())
    edges_node = [
        (int(a), int(b))
        for a, b in edges_node
        if int(a) in valid_edge_set and int(b) in valid_edge_set and int(a) != int(b)
    ]
    edges_node = sorted(set((min(a, b), max(a, b)) for a, b in edges_node))

    return dict(
        edges=np.array(edges_node, dtype=int),
        edg_idx_map=edg_idx_map,
        g=g,
        edg_obs=edg_obs,
        edg_obs_mask=edg_obs_mask,
        N_dyn_obs=N_dyn_obs,
        dyn_kp_idx=dyn_kp_idx,
        dyn_frame=dyn_frame,
        dyn_global=dyn_global,
        frame_to_trev=frame_to_trev,
        frames_anchor_first=frames_anchor_first,
        kp_neighbors=kp_neighbors,
        kp_weights=kp_weights,
        kp_rest=kp_rest,
    )


def _dyn_grid_from_solver_output(dyn_mapped, Nd, T_rev):
    dyn_mapped = np.asarray(dyn_mapped)
    if dyn_mapped.ndim == 3 and dyn_mapped.shape[0] == Nd and dyn_mapped.shape[1] == T_rev and dyn_mapped.shape[2] == 3:
        return dyn_mapped.astype(np.float64)
    if dyn_mapped.ndim == 2 and dyn_mapped.shape[1] == 3 and dyn_mapped.shape[0] == Nd * T_rev:
        return dyn_mapped.reshape(Nd, T_rev, 3).astype(np.float64)
    raise RuntimeError(f"Unexpected dyn_mapped shape {dyn_mapped.shape}; expected [Nd,T,3] or [Nd*T,3].")


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
        raise RuntimeError(f"Unexpected R_local shape {R_local.shape}; expected [T,3,3] or [T_rev,3,3].")
    if R_rest_global.shape != (3, 3):
        raise RuntimeError(f"Unexpected R_rest_global shape {R_rest_global.shape}; expected [3,3].")
    return R_rest_global[None, :, :] @ R_local


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

def _save_rest_candidates(results_all, seg_id, rest_frame, frames_seg_full, global_index_cand, edg_nodes):
    rest_points_all = glob_all_full[global_index_cand, rest_frame, :].astype(np.float64)
    rest_valid_all = valid_mask_global[global_index_cand, rest_frame].astype(bool)
    finite_rest = np.isfinite(rest_points_all).all(axis=1)
    rest_keep = rest_valid_all & finite_rest
    rest_global_ids = np.asarray(global_index_cand, dtype=int)
    rest_edg_global_ids = np.array([int(global_index_cand[int(nid)]) for nid in edg_nodes], dtype=int)
    results_all["rest_candidates"].append(
        dict(
            segment_id=int(seg_id),
            rest_frame=int(rest_frame),
            frames_anchor_first=np.asarray(frames_seg_full, dtype=int).tolist(),
            candidate_global_ids=rest_global_ids.astype(int),
            candidate_visible_rest_mask=rest_keep.astype(bool),
            global_ids=rest_global_ids[rest_keep].astype(int),
            points_world=rest_points_all[rest_keep].astype(np.float64),
            edg_global_ids=rest_edg_global_ids.astype(int),
        )
    )


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
    }


def _finalize_dynamic_outputs(results_all, num_points, T, K_neighbors):
    for gid in range(int(num_points)):
        _ensure_track(results_all["dynamics_points_reconstruct"], int(gid), T)

    skin_nodes = np.full((int(num_points), int(K_neighbors)), -1, dtype=np.int64)
    skin_weights = np.full((int(num_points), int(K_neighbors)), np.nan, dtype=np.float64)
    bind_rest = np.full((int(num_points), 3), np.nan, dtype=np.float64)
    bind_frame = np.full((int(num_points),), -1, dtype=np.int64)
    assigned = np.zeros((int(num_points),), dtype=bool)

    node_store = results_all["dynamic_skinning_nodes"]
    weight_store = results_all["dynamic_skinning_weights"]
    rest_store = results_all["dynamic_skinning_rest"]
    frame_store = results_all["dynamic_skinning_bind_frame"]

    for gid in range(int(num_points)):
        if gid in node_store:
            nodes = np.asarray(node_store[gid], dtype=np.int64).reshape(-1)
            weights = np.asarray(weight_store.get(gid, []), dtype=np.float64).reshape(-1)
            k_eff = min(int(K_neighbors), int(nodes.size), int(weights.size))
            if k_eff > 0:
                skin_nodes[gid, :k_eff] = nodes[:k_eff]
                skin_weights[gid, :k_eff] = weights[:k_eff]
                assigned[gid] = True
        if gid in rest_store:
            bind_rest[gid] = np.asarray(rest_store[gid], dtype=np.float64).reshape(3)
        if gid in frame_store:
            bind_frame[gid] = int(frame_store[gid])

    results_all["dynamic_global_ids"] = np.arange(int(num_points), dtype=np.int64)
    results_all["dynamic_skinning_nodes_array"] = skin_nodes
    results_all["dynamic_skinning_weights_array"] = skin_weights
    results_all["dynamic_skinning_rest_array"] = bind_rest
    results_all["dynamic_skinning_bind_frame_array"] = bind_frame
    results_all["dynamic_skinning_assigned_mask"] = assigned

########################################################################
### Segment loop
########################################################################

results_all = _init_results_store()

graph_samples = 50
graph_knn = 4
edge_target_neighbors = 6
K_neighbors = 4
# Node-growth knobs (fewer nodes -> faster solver: the segment solve is ~O(M^3) in
# node count M, and nodes accumulate across the sliding segments). Capping growth
# keeps the later, node-heavy segments cheap; graph_samples=50 stays well above the
# render skinning K (=6) so the dynamic object is not under-covered.
edg_max_new_nodes = 6      # was 20: max far nodes added per grow segment
edg_dist_factor_3d = 1.8   # was 1.2: only add a far node past this * avg node spacing

seg_id = 0
current_frame = 0
prev_segment = None

while True:
    if current_frame >= T:
        break

    end_frame_excl = int(min(T, current_frame + BUILD_EDG_SEGMENR_LENGTH))
    frames_seg_full = np.arange(current_frame, end_frame_excl, dtype=int)
    rest_frame = int(current_frame)

    if frames_seg_full.size == 0:
        break

    print(f"\n=== Segment {seg_id}: frames [{current_frame}, {end_frame_excl-1}] rest={rest_frame} ===")

    # (A) candidate set for this segment:
    # include every point that appears at least once in this segment window.
    seg_mask_any = np.any(valid_mask_global[:, frames_seg_full], axis=1)
    global_index_cand = np.where(seg_mask_any)[0].astype(int)
    if global_index_cand.size == 0:
        print("No candidates appear in this segment; stop.")
        break

    print(
        f"Segment candidate pool: {global_index_cand.size}/{glob_all_full.shape[0]} points "
        f"appear in frames [{current_frame}, {end_frame_excl - 1}]"
    )

    (
        global_index_cand,
        tracks_glob_candidate,
        tracks_xy_candidate,
        valid_mask_candidate_global,
        valid_mask_candidate_global_orig,
    ) = _build_candidate_views(global_index_cand)
    if prev_segment is None:
        # Node sampling: 2D-spatial farthest-point at the rest frame (EDG_old's proven
        # selector). It spreads nodes across the image and picks well-observed tracks,
        # which keeps the segment solver well-constrained. Only the topology below is
        # replaced with the robust 3D+2D rule.
        edg_init = build_initial_edg_graph(
            tracks_xy_candidate=tracks_xy_candidate,
            valid_mask_global=valid_mask_candidate_global,
            seed_frame=rest_frame,
            graph_samples=graph_samples,
            graph_knn=graph_knn,
        )
        edg_nodes = list(map(int, edg_init["graph_indices"]))

        # ensure valid at rest
        edg_nodes = [nid for nid in edg_nodes if bool(valid_mask_candidate_global[int(nid), rest_frame])]
        if len(edg_nodes) == 0:
            raise RuntimeError("No EDG nodes valid at rest_frame in first segment.")

        edges_node, edge_scores = _mosca_build_edges(
            tracks_glob_candidate=tracks_glob_candidate,
            tracks_xy_candidate=tracks_xy_candidate,
            valid_mask_candidate_global=valid_mask_candidate_global,
            edg_nodes=edg_nodes,
            target_neighbors=edge_target_neighbors,
        )

        try:
            _visualize_edg_rest_2d(
                tracks_xy_candidate=tracks_xy_candidate,
                valid_mask_candidate_global=valid_mask_candidate_global,
                edg_nodes=edg_nodes,
                edges_node=edges_node,
                edge_scores=edge_scores,
                rest_frame=rest_frame,
                output_path=output_dir / f"edg_rest_seg_{seg_id:02d}.png",
            )
        except Exception as exc:
            print(f"Warning: failed to render EDG rest-frame visualization: {exc}")

        new_nodes = []
        print(f"EDG init: nodes={len(edg_nodes)} edges={len(edges_node)}")
    else:
        # ---- Next segment: carry over previous EDG and grow ----
        prev_edg_global = prev_segment["edg_global_ids"]
        prev_edg_deformed = prev_segment["edg_deformed"]
        prev_frames = prev_segment["frames_anchor_first"]
        prev_edges_global = prev_segment["edges_global"]
        overlap_frame = int(current_frame)
        prev_t = prev_frames.index(overlap_frame) if overlap_frame in prev_frames else len(prev_frames) - 1

        cand_set = set(global_index_cand.tolist())
        missing = [int(g) for g in prev_edg_global if int(g) not in cand_set]
        if missing:
            global_index_cand = np.concatenate([global_index_cand, np.array(missing, dtype=int)], axis=0)
            (
                global_index_cand,
                tracks_glob_candidate,
                tracks_xy_candidate,
                valid_mask_candidate_global,
                valid_mask_candidate_global_orig,
            ) = _build_candidate_views(global_index_cand)

        prev_gid_to_j = {int(gid): j for j, gid in enumerate(prev_edg_global)}
        global_to_local = {int(gid): i for i, gid in enumerate(global_index_cand.tolist())}

        carryover_local_to_fix = []
        missing_set = set(int(g) for g in missing)
        for gid in prev_edg_global:
            gid = int(gid)
            if gid not in global_to_local:
                continue
            lid = int(global_to_local[gid])
            if gid in missing_set or not bool(valid_mask_candidate_global[lid, rest_frame]):
                carryover_local_to_fix.append(lid)

        carryover_local_to_fix = np.array(sorted(set(carryover_local_to_fix)), dtype=int)

        if carryover_local_to_fix.size > 0:
            xyz_seed = []
            for lid in carryover_local_to_fix.tolist():
                gid = int(global_index_cand[lid])
                j_prev = int(prev_gid_to_j[gid])
                prev_pose = prev_edg_deformed[j_prev, prev_t, :].astype(np.float64)

                # If node appears later in current segment but not at rest,
                # set rest observation to average(prev pose, first current pose).
                first_curr_pose = None
                for f in frames_seg_full.tolist():
                    if int(f) == int(rest_frame):
                        continue
                    if bool(valid_mask_candidate_global[lid, int(f)]):
                        cur_pose = tracks_glob_candidate[0, lid, int(f), :].astype(np.float64)
                        if np.isfinite(cur_pose).all():
                            first_curr_pose = cur_pose
                            break

                if first_curr_pose is not None:
                    seed_pose = 0.5 * (prev_pose + first_curr_pose)
                else:
                    seed_pose = prev_pose

                xyz_seed.append(seed_pose)

            xyz_seed = np.asarray(xyz_seed, dtype=np.float64)
            tracks_glob_candidate[0, carryover_local_to_fix, rest_frame, :] = xyz_seed

            # Mark carryover nodes as valid at rest frame (even if they were invalid before)
            valid_mask_candidate_global[carryover_local_to_fix, rest_frame] = True

        global_to_local = {int(gid): i for i, gid in enumerate(global_index_cand.tolist())}

        edg_nodes = [int(global_to_local[int(gid)]) for gid in prev_edg_global if int(gid) in global_to_local]
        edg_nodes = list(dict.fromkeys(edg_nodes))
        edg_nodes = [nid for nid in edg_nodes if bool(valid_mask_candidate_global[int(nid), rest_frame])]
        if len(edg_nodes) == 0:
            raise RuntimeError("No carryover EDG nodes valid at rest_frame.")

        edges_node = []
        for ga, gb in prev_edges_global:
            ga = int(ga)
            gb = int(gb)
            if ga in global_to_local and gb in global_to_local:
                a = int(global_to_local[ga])
                b = int(global_to_local[gb])
                if a != b:
                    edges_node.append((min(a, b), max(a, b)))

        # add far nodes, forcing at least one if candidates exist
        edg_before = len(edg_nodes)
        edg_nodes, edges_node, new_nodes = _add_far_nodes_greedy(
            tracks_glob_candidate=tracks_glob_candidate,
            valid_mask_candidate_global=valid_mask_candidate_global,
            valid_mask_candidate_global_orig=valid_mask_candidate_global_orig,
            rest_frame=rest_frame,
            edg_nodes=edg_nodes,
            edges_node=edges_node,
            graph_knn=graph_knn,
            max_new_nodes=edg_max_new_nodes,
            dist_factor_3d=edg_dist_factor_3d,
            force_min_new=0,
        )
        
        if len(edg_nodes) == edg_before and len(new_nodes) == 0:
            print("EDG grow: no far nodes added (EDG size unchanged).")
        else:
            print(f"EDG grow: added {len(new_nodes)} nodes (now {len(edg_nodes)}).")

        # Rebuild edges after node growth using the same robust 3D+2D curve-distance rule,
        # but preserve previous edges and only add edges incident to new nodes.
        stable_edges, edge_scores = _mosca_build_edges(
            tracks_glob_candidate=tracks_glob_candidate,
            tracks_xy_candidate=tracks_xy_candidate,
            valid_mask_candidate_global=valid_mask_candidate_global,
            edg_nodes=edg_nodes,
            target_neighbors=edge_target_neighbors,
        )

        prev_edge_set = set(tuple(sorted(e)) for e in edges_node)
        new_node_set = set(int(n) for n in new_nodes)
        added_edges = [
            tuple(edge)
            for edge in stable_edges
            if edge not in prev_edge_set and (edge[0] in new_node_set or edge[1] in new_node_set)
        ]
        edges_node = sorted(prev_edge_set.union(added_edges))
        edge_scores = {edge: edge_scores.get(edge, np.nan) for edge in edges_node}

    _save_rest_candidates(results_all, seg_id, rest_frame, frames_seg_full, global_index_cand, edg_nodes)

    # (C) Build solver inputs and solve
    solver_in = _build_solver_inputs_for_segment(
        tracks_glob_candidate=tracks_glob_candidate,
        valid_mask_candidate_global=valid_mask_candidate_global,
        frames_anchor_first=frames_seg_full,
        rest_frame=rest_frame,
        edg_ids_node=edg_nodes,
        edges_node=edges_node,
        global_index_cand=global_index_cand,
        persistent_skinning_nodes=results_all["dynamic_skinning_nodes"],
        persistent_skinning_weights=results_all["dynamic_skinning_weights"],
        persistent_skinning_rest=results_all["dynamic_skinning_rest"],
        persistent_skinning_bind_frame=results_all["dynamic_skinning_bind_frame"],
        K_neighbors=K_neighbors,
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
        solve_Ronly_fn=XM.solve_Ronly,
        load_matrix_from_bin_fn=load_matrix_from_bin,
    )

    edg_deformed = np.asarray(edg_solver_out["edg_deformed"], dtype=np.float64)  # [M, T_rev, 3]
    R_edg = np.asarray(edg_solver_out["R_edg"], dtype=np.float64)  # [M, T_rev, 3, 3]
    dyn_mapped = edg_solver_out["dyn_mapped"]
    frames_anchor_first = solver_in["frames_anchor_first"]
    T_rev = len(frames_anchor_first)

    # (D) Save this segment outputs to results_all
    edg_ids = np.array(edg_nodes, dtype=int)
    edg_global_ids = global_index_cand[edg_ids].astype(int).tolist()

    edges_global = []
    for a, b in edges_node:
        ga = int(global_index_cand[int(a)])
        gb = int(global_index_cand[int(b)])
        aa, bb = (ga, gb) if ga < gb else (gb, ga)
        edges_global.append((aa, bb))
        results_all["edg_edges_union"].add((aa, bb))

    for j, gid in enumerate(edg_global_ids):
        arr = _ensure_track(results_all["edg_reconstruct"], gid, T)
        for t_idx, f in enumerate(frames_anchor_first.tolist()):
            arr[int(f), :] = edg_deformed[j, t_idx, :]

        rot_arr = _ensure_rotation(results_all["edg_rotation"], gid, T)
        if rest_frame < T and np.isfinite(rot_arr[int(rest_frame)]).all():
            rest_global = rot_arr[int(rest_frame)].copy()
        else:
            rest_global = np.eye(3, dtype=np.float64)

        R_local = R_edg[j]
        if R_local.shape[0] != T_rev:
            raise RuntimeError(
                f"Unexpected local rotation length {R_local.shape[0]} for node {gid}; expected {T_rev}."
            )
        R_global = _compose_global_rotations(R_local, rest_global)

        for t_idx, f in enumerate(frames_anchor_first.tolist()):
            rot_arr[int(f), :, :] = R_global[t_idx]

    unique_kps = np.unique(solver_in["dyn_kp_idx"]).astype(int)
    Nd = len(unique_kps)
    dyn_grid = _dyn_grid_from_solver_output(dyn_mapped, Nd=Nd, T_rev=T_rev)

    dyn_global_ids = global_index_cand[unique_kps].astype(int).tolist()
    for i, gid in enumerate(dyn_global_ids):
        arr = _ensure_track(results_all["dynamics_points_reconstruct"], gid, T)
        for t_idx, f in enumerate(frames_anchor_first.tolist()):
            arr[int(f), :] = dyn_grid[i, t_idx, :]

    results_all["segments"].append(
        dict(
            segment_id=seg_id,
            frames_anchor_first=frames_anchor_first.tolist(),
            edg_global_ids=edg_global_ids,
            edges_global=edges_global,
            edg_deformed=edg_deformed.copy(),
        )
    )

    prev_segment = results_all["segments"][-1]

    if end_frame_excl >= T:
        break

    # advance with overlap (rest frame equals previous segment end frame)
    current_frame = int(end_frame_excl - 1)
    seg_id += 1

_finalize_dynamic_outputs(
    results_all,
    num_points=glob_all_full.shape[0],
    T=T,
    K_neighbors=K_neighbors,
)

np.savez_compressed(output_dir / "results_all.npz", **results_all)

########################################################################
### Post-processing: back-fill NaN EDG positions using neighbor skinning
########################################################################

def _fill_nan_edg_tracks(edg_tracks_in, edg_rot_tracks_in, edges_union):
    """Back-fill NaN position frames in EDG tracks using neighbor relative transforms.

    For each EDG node whose position is NaN before its first_valid_frame, predict
    its position at each missing frame t from every valid graph neighbor n:
        pred = R_n(t) @ R_n(fv).T @ (p_self(fv) - p_n(fv)) + p_n(t)
    where fv = first_valid_frame. Final position is inverse-distance weighted average.
    Operates on copies; originals are not mutated.
    """
    edg_tracks_out    = {gid: arr.copy() for gid, arr in edg_tracks_in.items()}
    edg_rot_tracks_out = {gid: arr.copy() for gid, arr in edg_rot_tracks_in.items()}

    # Build adjacency from edge union
    adjacency = {gid: [] for gid in edg_tracks_out}
    for ga, gb in edges_union:
        if ga in adjacency:
            adjacency[ga].append(gb)
        if gb in adjacency:
            adjacency[gb].append(ga)

    # Find nodes with NaN frames
    nan_node_info = {}
    for gid, arr in edg_tracks_out.items():
        valid_mask = np.isfinite(arr).all(axis=1)
        if not valid_mask.all():
            first_valid = int(np.argmax(valid_mask)) if valid_mask.any() else None
            nan_node_info[gid] = first_valid

    print(f"[fill_nan_edg] Nodes with NaN frames: {list(nan_node_info.items())}")

    for gid, first_valid in nan_node_info.items():
        if first_valid is None:
            print(f"  gid={gid}: all frames NaN, skipping")
            continue

        pos_self   = edg_tracks_out[gid]
        p_self_fv  = pos_self[first_valid]
        nan_frames = np.where(~np.isfinite(pos_self).all(axis=1))[0]
        neighbors  = adjacency.get(gid, [])

        # Skinning weights: inverse distance at first_valid_frame
        valid_neighbors, dists = [], []
        for nb in neighbors:
            nb_pos = edg_tracks_out[nb]
            if not np.isfinite(nb_pos[first_valid]).all():
                continue
            d = np.linalg.norm(nb_pos[first_valid] - p_self_fv)
            valid_neighbors.append(nb)
            dists.append(d + 1e-6)

        if not valid_neighbors:
            print(f"  gid={gid}: no valid neighbors at first_valid={first_valid}, cannot fill")
            continue

        weights = 1.0 / np.array(dists)
        weights /= weights.sum()

        filled_count = 0
        for t in nan_frames:
            pred_sum, w_sum = np.zeros(3), 0.0
            for nb, w in zip(valid_neighbors, weights):
                nb_pos = edg_tracks_out[nb]
                nb_rot = edg_rot_tracks_out.get(nb)
                if not (np.isfinite(nb_pos[first_valid]).all() and np.isfinite(nb_pos[t]).all()):
                    continue
                if nb_rot is None or not np.isfinite(nb_rot[first_valid]).all() or not np.isfinite(nb_rot[t]).all():
                    pred = nb_pos[t] + (p_self_fv - nb_pos[first_valid])
                else:
                    rel_R = nb_rot[t] @ nb_rot[first_valid].T
                    pred  = nb_pos[t] + rel_R @ (p_self_fv - nb_pos[first_valid])
                pred_sum += w * pred
                w_sum    += w
            if w_sum > 0:
                edg_tracks_out[gid][t] = pred_sum / w_sum
                filled_count += 1

        print(f"  gid={gid}: first_valid={first_valid}, filled {filled_count}/{len(nan_frames)} NaN frames")

    remaining = sum(1 for arr in edg_tracks_out.values() if not np.isfinite(arr).all())
    print(f"[fill_nan_edg] Nodes still containing NaN after fill: {remaining}")
    return edg_tracks_out, edg_rot_tracks_out


results_all["edg_reconstruct"], results_all["edg_rotation"] = _fill_nan_edg_tracks(
    results_all["edg_reconstruct"],
    results_all["edg_rotation"],
    results_all["edg_edges_union"],
)

# Re-save with filled tracks
np.savez_compressed(output_dir / "results_all.npz", **results_all)
print("Re-saved results_all.npz with NaN-filled EDG tracks.")

########################################################################
### Visualization using Open3D
########################################################################

import open3d as o3d
from scipy.stats import trim_mean

try:
    import teaserpp_python
except ImportError:
    teaserpp_python = None

edg_tracks = results_all["edg_reconstruct"]
edg_rot_tracks = results_all["edg_rotation"]
dyn_tracks = results_all["dynamics_points_reconstruct"]
edges_global = sorted(results_all["edg_edges_union"])


def _normalize_rotation(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64))
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _rotation_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    n = np.linalg.norm(axis)
    if n <= 1e-12 or abs(angle) <= 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = axis / n
    K = _skew(axis)
    return np.eye(3, dtype=np.float64) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _rotation_align_vector_to_vector(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = np.asarray(src, dtype=np.float64).reshape(3)
    dst = np.asarray(dst, dtype=np.float64).reshape(3)
    src_n = np.linalg.norm(src)
    dst_n = np.linalg.norm(dst)
    if src_n <= 1e-12 or dst_n <= 1e-12:
        return np.eye(3, dtype=np.float64)
    src = src / src_n
    dst = dst / dst_n
    dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    if dot > 1.0 - 1e-10:
        return np.eye(3, dtype=np.float64)
    if dot < -1.0 + 1e-10:
        basis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(src[0]) > 0.9:
            basis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = np.cross(src, basis)
        axis /= np.linalg.norm(axis)
        return _rotation_from_axis_angle(axis, np.pi)
    axis = np.cross(src, dst)
    angle = np.arccos(dot)
    return _rotation_from_axis_angle(axis, angle)


def _principal_line_direction(points: np.ndarray):
    pts = np.asarray(points, dtype=np.float64)
    center = np.mean(pts, axis=0)
    pts_c = pts - center.reshape(1, 3)
    _, _, vh = np.linalg.svd(pts_c, full_matrices=False)
    direction = vh[0]
    return center, direction / np.linalg.norm(direction)


def _line_like(points: np.ndarray, ratio_thresh: float = 1e-3) -> bool:
    pts = np.asarray(points, dtype=np.float64)
    pts_c = pts - np.mean(pts, axis=0, keepdims=True)
    _, svals, _ = np.linalg.svd(pts_c, full_matrices=False)
    if svals.size < 2 or svals[0] <= 1e-12:
        return True
    return float(svals[1] / svals[0]) < ratio_thresh


def _project_to_plane(v: np.ndarray, axis: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return v - axis * np.dot(axis, v)


def _rigid_align_cols(src_pts: np.ndarray, dst_pts: np.ndarray):
    src_mean = np.mean(src_pts, axis=1, keepdims=True)
    dst_mean = np.mean(dst_pts, axis=1, keepdims=True)
    src_centered = src_pts - src_mean
    dst_centered = dst_pts - dst_mean
    H = dst_centered @ src_centered.T
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    t = dst_mean - R @ src_mean
    return R, t


def _apply_similarity_points(pts: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray):
    pts = np.asarray(pts, dtype=np.float64)
    if pts.size == 0:
        return pts.reshape(-1, 3)
    return scale * (pts @ R.T) + t.reshape(1, 3)


def _apply_similarity_track_dict(track_dict, scale: float, R: np.ndarray, t: np.ndarray):
    out = {}
    for gid, arr in track_dict.items():
        arr = np.asarray(arr, dtype=np.float64)
        transformed = arr.copy()
        valid = np.isfinite(arr).all(axis=1)
        if np.any(valid):
            transformed[valid] = _apply_similarity_points(arr[valid], scale, R, t)
        out[gid] = transformed
    return out


def _apply_rotation_track_dict(rot_dict, R_align: np.ndarray):
    out = {}
    for gid, arr in rot_dict.items():
        arr = np.asarray(arr, dtype=np.float64)
        transformed = arr.copy()
        valid = np.isfinite(arr).all(axis=(1, 2))
        if np.any(valid):
            transformed[valid] = R_align[None, :, :] @ arr[valid]
        out[gid] = transformed
    return out


def _compute_gt_camera_poses(gt_dict, T_expected: int):
    if not gt_dict:
        return None
    gt_items = sorted(gt_dict.items(), key=lambda kv: int(kv[1]["id"]))
    centers = np.full((T_expected, 3), np.nan, dtype=np.float64)
    rotations = np.tile(np.eye(3, dtype=np.float64), (T_expected, 1, 1))
    for _, item in gt_items:
        frame_id = int(item["id"])
        if frame_id < 0 or frame_id >= T_expected:
            continue
        R_raw_item = item.get("R")
        t_raw_item = item.get("t")
        if R_raw_item is None or t_raw_item is None:
            continue
        R_wc = np.asarray(R_raw_item, dtype=np.float64)
        t_wc = np.asarray(t_raw_item, dtype=np.float64)
        if R_wc.size != 9 or t_wc.size != 3:
            continue
        R_wc = R_wc.reshape(3, 3)
        t_wc = t_wc.reshape(3)
        if not (np.isfinite(R_wc).all() and np.isfinite(t_wc).all()):
            continue
        centers[frame_id] = (-R_wc.T @ t_wc).astype(np.float64)
        rotations[frame_id] = R_wc.T
    if int(np.sum(np.isfinite(centers).all(axis=1))) == 0:
        return None
    return centers, rotations


def _teaser_align_camera_centers(src_pts: np.ndarray, dst_pts: np.ndarray):
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    src = src.T.copy()
    dst = dst.T.copy()
    finite_mask = np.isfinite(src).all(axis=0) & np.isfinite(dst).all(axis=0)
    src = src[:, finite_mask]
    dst = dst[:, finite_mask]
    if src.shape[1] < 3:
        raise RuntimeError("Too few finite correspondences for TEASER alignment.")

    dst_avg = trim_mean(dst, proportiontocut=0.05, axis=1)
    src_avg = trim_mean(src, proportiontocut=0.05, axis=1)
    dst_dis = np.linalg.norm(dst - dst_avg.reshape(3, 1), axis=0)
    src_dis = np.linalg.norm(src - src_avg.reshape(3, 1), axis=0)
    keep = (src_dis < np.percentile(src_dis, 90)) & (dst_dis < np.percentile(dst_dis, 90))
    if np.count_nonzero(keep) >= 3:
        src = src[:, keep]
        dst = dst[:, keep]

    dst_avg = np.mean(dst, axis=1)
    src_avg = np.mean(src, axis=1)
    scale_dst = np.mean(np.linalg.norm(dst - dst_avg.reshape(3, 1), axis=0))
    scale_src = np.mean(np.linalg.norm(src - src_avg.reshape(3, 1), axis=0))
    scale = float(scale_dst / scale_src)
    print(
        f"[Align] TEASER prep: pairs={src.shape[1]} scale_dst={scale_dst:.6f} "
        f"scale_src={scale_src:.6f} ratio={scale:.6f}"
    )

    if (
        teaserpp_python is None
        or not np.isfinite(scale_dst)
        or not np.isfinite(scale_src)
        or scale_dst <= 1e-12
        or scale_src <= 1e-12
    ):
        scaled_src = src * scale
        R_fallback, t_fallback = _rigid_align_cols(scaled_src, dst)
        return scale, R_fallback, t_fallback.reshape(3)

    src_norm = src / scale_src
    dst_norm = dst / scale_dst
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1
    solver_params.noise_bound = 0.3
    solver_params.estimate_scaling = False
    solver_params.rotation_estimation_algorithm = (
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    )
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12
    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    solver.solve(src_norm, dst_norm)
    solution = solver.getSolution()
    translation = scale_dst * solution.translation.reshape(3)
    return scale, solution.rotation, translation


def _line_pose_align(solved_cam_centers, solved_cam_rot, gt_cam_centers, gt_cam_rot):
    valid = np.isfinite(solved_cam_centers).all(axis=1) & np.isfinite(gt_cam_centers).all(axis=1)
    if int(np.sum(valid)) < 3:
        return None
    src_pts = solved_cam_centers[valid]
    dst_pts = gt_cam_centers[valid]
    src_center, src_dir = _principal_line_direction(src_pts)
    dst_center, dst_dir = _principal_line_direction(dst_pts)
    if np.dot(src_dir, dst_dir) < 0.0:
        dst_dir = -dst_dir
    src_scale = np.mean(np.linalg.norm(src_pts - src_center.reshape(1, 3), axis=1))
    dst_scale = np.mean(np.linalg.norm(dst_pts - dst_center.reshape(1, 3), axis=1))
    if src_scale <= 1e-12 or dst_scale <= 1e-12:
        return None
    scale = float(dst_scale / src_scale)
    R_line = _rotation_align_vector_to_vector(src_dir, dst_dir)
    axis_world = dst_dir / np.linalg.norm(dst_dir)
    sum_cross = 0.0
    sum_dot = 0.0
    for Re, Rg in zip(solved_cam_rot[valid], gt_cam_rot[valid]):
        Re0 = R_line @ Re
        for k in (0, 1, 2):
            ve = _project_to_plane(Re0[:, k], axis_world)
            vg = _project_to_plane(Rg[:, k], axis_world)
            ne = np.linalg.norm(ve)
            ng = np.linalg.norm(vg)
            if ne <= 1e-8 or ng <= 1e-8:
                continue
            ve /= ne
            vg /= ng
            sum_dot += float(np.dot(ve, vg))
            sum_cross += float(np.dot(axis_world, np.cross(ve, vg)))
    twist = np.arctan2(sum_cross, sum_dot) if abs(sum_cross) > 0.0 or abs(sum_dot) > 0.0 else 0.0
    R_align = _normalize_rotation(_rotation_from_axis_angle(axis_world, twist) @ R_line)
    t_align = dst_center - scale * (R_align @ src_center)
    solved_cam_centers_aligned = _apply_similarity_points(solved_cam_centers, scale, R_align, t_align)
    solved_cam_rot_aligned = np.einsum("ij,tjk->tik", R_align, solved_cam_rot)
    err = np.linalg.norm(solved_cam_centers_aligned[valid] - gt_cam_centers[valid], axis=1)
    print(
        f"[Align] line fallback: frames={int(np.sum(valid))} scale={scale:.6f} "
        f"twist_deg={np.degrees(twist):.3f} mean_t_err={np.mean(err):.6f}"
    )
    return scale, R_align, t_align, solved_cam_centers_aligned, solved_cam_rot_aligned


def _mosca_align_with_evo(solved_cam_centers, solved_cam_rot, gt_cam_centers, gt_cam_rot):
    try:
        import importlib
        from scipy.spatial.transform import Rotation
        PoseTrajectory3D = importlib.import_module("evo.core.trajectory").PoseTrajectory3D
    except Exception as e:
        print(f"[Align] evo import failed: {e}")
        return None

    T_local = min(solved_cam_centers.shape[0], gt_cam_centers.shape[0], solved_cam_rot.shape[0], gt_cam_rot.shape[0])
    valid = np.isfinite(solved_cam_centers[:T_local]).all(axis=1) & np.isfinite(gt_cam_centers[:T_local]).all(axis=1)
    if int(np.sum(valid)) < 3:
        return None
    est_xyz = solved_cam_centers[:T_local][valid]
    ref_xyz = gt_cam_centers[:T_local][valid]
    if _line_like(est_xyz) or _line_like(ref_xyz):
        print("[Align] Camera trajectory is line-like; using line-aware pose fallback.")
        return _line_pose_align(solved_cam_centers[:T_local], solved_cam_rot[:T_local], gt_cam_centers[:T_local], gt_cam_rot[:T_local])

    est_quat = []
    ref_quat = []
    for Re, Rg in zip(solved_cam_rot[:T_local][valid], gt_cam_rot[:T_local][valid]):
        est_xyzw = Rotation.from_matrix(Re).as_quat()
        ref_xyzw = Rotation.from_matrix(Rg).as_quat()
        est_quat.append(np.array([est_xyzw[3], est_xyzw[0], est_xyzw[1], est_xyzw[2]], dtype=np.float64))
        ref_quat.append(np.array([ref_xyzw[3], ref_xyzw[0], ref_xyzw[1], ref_xyzw[2]], dtype=np.float64))

    timestamps = np.arange(est_xyz.shape[0], dtype=np.float64)
    traj_ref = PoseTrajectory3D(positions_xyz=ref_xyz, orientations_quat_wxyz=np.asarray(ref_quat), timestamps=timestamps)
    traj_est = PoseTrajectory3D(positions_xyz=est_xyz, orientations_quat_wxyz=np.asarray(est_quat), timestamps=timestamps)
    try:
        align_result = traj_est.align(traj_ref, correct_scale=True)
    except Exception as e:
        msg = str(e).lower()
        print(f"[Align] evo align failed: {e}")
        if "singular" in msg or "covariance" in msg:
            print("[Align] Falling back to line-aware pose alignment.")
            return _line_pose_align(solved_cam_centers[:T_local], solved_cam_rot[:T_local], gt_cam_centers[:T_local], gt_cam_rot[:T_local])
        return None
    if align_result is None or len(align_result) != 3:
        return None
    R_align, t_align, scale = align_result
    R_align = np.asarray(R_align, dtype=np.float64).reshape(3, 3)
    t_align = np.asarray(t_align, dtype=np.float64).reshape(3)
    scale = float(scale)
    solved_cam_centers_aligned = _apply_similarity_points(solved_cam_centers, scale, R_align, t_align)
    solved_cam_rot_aligned = np.einsum("ij,tjk->tik", R_align, solved_cam_rot)
    err = np.linalg.norm(solved_cam_centers_aligned[:T_local][valid] - gt_cam_centers[:T_local][valid], axis=1)
    print(
        f"[Align] evo pose alignment: frames={int(np.sum(valid))} scale={scale:.6f} "
        f"mean_t_err={np.mean(err):.6f}"
    )
    return scale, R_align, t_align, solved_cam_centers_aligned, solved_cam_rot_aligned


def _load_gt_object_cloud_paths(root: Path, T_expected: int):
    obj_dir = None
    for dirname in ("dynamic_pointcloud_world", "object_pointcloud_world"):
        cand = root / dirname
        if cand.exists():
            obj_dir = cand
            break
    if obj_dir is None:
        return None
    paths = []
    for i in range(T_expected):
        p = obj_dir / f"frame{i:06d}.ply"
        paths.append(p if p.exists() else None)
    return paths if any(p is not None for p in paths) else None


def _identity_alignment(pred_cam_centers, pred_cam_rot):
    align_scale = 1.0
    align_R = np.eye(3, dtype=np.float64)
    align_t = np.zeros(3, dtype=np.float64)
    cam_centers_vis = np.asarray(pred_cam_centers, dtype=np.float64).copy()
    cam_rot_vis = np.asarray(pred_cam_rot, dtype=np.float64).copy()
    return align_scale, align_R, align_t, cam_centers_vis, cam_rot_vis


if dataset_cat == "Syn4D":
    pred_cam_centers, pred_cam_rot, _ = _syn4d_load_solved()
    gt_cam_centers = None
    gt_cam_rot = None
    try:
        _, gt_cam_centers, gt_cam_rot = _syn4d_load_gt_cameras(rootpath / "colmap")
    except Exception as exc:
        print(f"[GT] Syn4D GT camera poses unavailable; skipping GT alignment: {exc}")

    if gt_cam_centers is None or gt_cam_rot is None:
        T_align = min(T, pred_cam_centers.shape[0], pred_cam_rot.shape[0])
        pred_cam_centers = pred_cam_centers[:T_align]
        pred_cam_rot = pred_cam_rot[:T_align]
        align_scale, align_R, align_t, cam_centers_vis, cam_rot_vis = _identity_alignment(
            pred_cam_centers, pred_cam_rot
        )
        gt_cam_centers = np.full((T_align, 3), np.nan, dtype=np.float64)
        gt_cam_rot = np.tile(np.eye(3, dtype=np.float64), (T_align, 1, 1))
        print("[Align] No valid GT cameras; using solver coordinates unchanged.")
    else:
        T_align = min(T, gt_cam_centers.shape[0], gt_cam_rot.shape[0], pred_cam_centers.shape[0], pred_cam_rot.shape[0])
        gt_cam_centers = gt_cam_centers[:T_align]
        gt_cam_rot = gt_cam_rot[:T_align]
        pred_cam_centers = pred_cam_centers[:T_align]
        pred_cam_rot = pred_cam_rot[:T_align]

        evo_align = _syn4d_mosca_align_with_evo(pred_cam_centers, pred_cam_rot, gt_cam_centers, gt_cam_rot)
        if evo_align is not None:
            align_scale, align_R, align_t, cam_centers_vis, cam_rot_vis = evo_align
        else:
            cam_valid = np.isfinite(pred_cam_centers).all(axis=1) & np.isfinite(gt_cam_centers).all(axis=1)
            if int(np.sum(cam_valid)) >= 3:
                align_scale, align_R, align_t = _syn4d_teaser_align_camera_centers(
                    pred_cam_centers[cam_valid], gt_cam_centers[cam_valid]
                )
                print(f"[Align] fallback TEASER alignment: frames={int(np.sum(cam_valid))} scale={align_scale:.6f}")
            else:
                align_scale, align_R, align_t, cam_centers_vis, cam_rot_vis = _identity_alignment(
                    pred_cam_centers, pred_cam_rot
                )
                print("[Align] Not enough valid GT camera centers; using solver coordinates unchanged.")
            cam_centers_vis = _syn4d_apply_similarity_points(pred_cam_centers, align_scale, align_R, align_t)
            cam_rot_vis = np.einsum("ij,tjk->tik", align_R, pred_cam_rot)

    if T_align != T:
        print(f"[Align] Truncating visualization to {T_align} frames to match solved/GT pose lengths.")
        T = T_align
else:
    pred_cam_centers = np.asarray(t_raw, dtype=np.float64)
    pred_cam_rot = np.asarray([_normalize_rotation(R_raw[i]) for i in range(T)], dtype=np.float64)
    gt_pose_result = _compute_gt_camera_poses(gt, T)
    if gt_pose_result is None:
        gt_cam_centers = np.full((T, 3), np.nan, dtype=np.float64)
        gt_cam_rot = np.tile(np.eye(3, dtype=np.float64), (T, 1, 1))
        align_scale, align_R, align_t, cam_centers_vis, cam_rot_vis = _identity_alignment(
            pred_cam_centers, pred_cam_rot
        )
        print("[Align] No valid GT camera poses; using solver coordinates unchanged.")
    else:
        gt_cam_centers, gt_cam_rot = gt_pose_result
        evo_align = _mosca_align_with_evo(pred_cam_centers, pred_cam_rot, gt_cam_centers, gt_cam_rot)
        if evo_align is not None:
            align_scale, align_R, align_t, cam_centers_vis, cam_rot_vis = evo_align
        else:
            cam_valid = np.isfinite(pred_cam_centers).all(axis=1) & np.isfinite(gt_cam_centers).all(axis=1)
            if int(np.sum(cam_valid)) >= 3:
                align_scale, align_R, align_t = _teaser_align_camera_centers(
                    pred_cam_centers[cam_valid], gt_cam_centers[cam_valid]
                )
                print(f"[Align] fallback TEASER alignment: frames={int(np.sum(cam_valid))} scale={align_scale:.6f}")
            else:
                align_scale, align_R, align_t, cam_centers_vis, cam_rot_vis = _identity_alignment(
                    pred_cam_centers, pred_cam_rot
                )
                print("[Align] Not enough valid GT camera centers; using solver coordinates unchanged.")
            cam_centers_vis = _apply_similarity_points(pred_cam_centers, align_scale, align_R, align_t)
            cam_rot_vis = np.einsum("ij,tjk->tik", align_R, pred_cam_rot)

edg_tracks = _apply_similarity_track_dict(edg_tracks, align_scale, align_R, align_t)
dyn_tracks = _apply_similarity_track_dict(dyn_tracks, align_scale, align_R, align_t)
edg_rot_tracks = _apply_rotation_track_dict(edg_rot_tracks, align_R)
if landmarks is not None:
    landmarks_vis = _syn4d_apply_similarity_points(landmarks.astype(np.float64), align_scale, align_R, align_t)
else:
    landmarks_vis = np.zeros((0, 3), dtype=np.float64)
gt_object_cloud_paths = _load_gt_object_cloud_paths(rootpath, T)
has_valid_gt_poses = (
    gt_cam_centers is not None
    and np.asarray(gt_cam_centers).ndim == 2
    and int(np.sum(np.isfinite(gt_cam_centers).all(axis=1))) > 0
)
has_gt_object_clouds = gt_object_cloud_paths is not None and any(
    p is not None for p in gt_object_cloud_paths
)
if not has_valid_gt_poses:
    print("[GT] No valid GT camera poses; GT camera axes will not be shown.")
if not has_gt_object_clouds:
    print("[GT] No GT object point cloud directory/files; GT object mesh will not be shown.")

frame_list = list(range(T))
state = {"idx": 0}

# Precompute global ID list for EDG and dynamics
edg_gids = sorted(edg_tracks.keys())
dyn_gids = sorted(dyn_tracks.keys())

def _frame_points_from_tracks(track_dict, gids, frame_id):
    pts = []
    idx_map = {}
    for gid in gids:
        arr = track_dict.get(gid)
        if arr is None:
            continue
        p = arr[int(frame_id), :]
        if np.isfinite(p).all():
            idx_map[int(gid)] = len(pts)
            pts.append(p)
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float64), idx_map
    return np.asarray(pts, dtype=np.float64), idx_map


def _frame_rotations_from_tracks(rot_dict, gids, frame_id, idx_map):
    rots = [np.eye(3, dtype=np.float64) for _ in range(len(idx_map))]
    for gid, idx in idx_map.items():
        arr = rot_dict.get(int(gid))
        if arr is None:
            continue
        R = arr[int(frame_id), :, :]
        if np.isfinite(R).all():
            rots[int(idx)] = R
    if len(rots) == 0:
        return np.zeros((0, 3, 3), dtype=np.float64)
    return np.asarray(rots, dtype=np.float64)


def _camera_axes_lines(cam_center, cam_R, axis_len):
    cam_center = np.asarray(cam_center, dtype=np.float64).reshape(3,)
    cam_R = np.asarray(cam_R, dtype=np.float64).reshape(3, 3)
    axes = cam_R @ (axis_len * np.eye(3))
    pts = np.vstack([cam_center, cam_center + axes.T])
    lines = np.array([[0, 1], [0, 2], [0, 3]], dtype=np.int32)
    colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return pts, lines, colors


def get_frame_geoms(frame_id: int):
    edg_pts, edg_idx = _frame_points_from_tracks(edg_tracks, edg_gids, frame_id)
    dyn_pts, _ = _frame_points_from_tracks(dyn_tracks, dyn_gids, frame_id)
    edg_rots = _frame_rotations_from_tracks(edg_rot_tracks, edg_gids, frame_id, edg_idx)

    line_idx = []
    for ga, gb in edges_global:
        if ga in edg_idx and gb in edg_idx:
            line_idx.append([edg_idx[ga], edg_idx[gb]])
    line_idx = np.asarray(line_idx, dtype=np.int32)

    return edg_pts, dyn_pts, line_idx, edg_rots


edg_pts0, dyn_pts0, line_idx0, edg_rots0 = get_frame_geoms(frame_list[state["idx"]])

pcd_edg = o3d.geometry.PointCloud()
pcd_edg.points = o3d.utility.Vector3dVector(edg_pts0)
pcd_edg.paint_uniform_color([1.0, 0.0, 0.0])

pcd_dyn = o3d.geometry.PointCloud()
pcd_dyn.points = o3d.utility.Vector3dVector(dyn_pts0)
pcd_dyn.paint_uniform_color([0.0, 0.4, 1.0])

# pcd_gt_dyn = o3d.geometry.PointCloud()
# pcd_gt_dyn.paint_uniform_color([0.0, 0.8, 0.0])
# if gt_object_cloud_paths is not None and gt_object_cloud_paths[frame_list[state["idx"]]] is not None:
#     gt_pcd0 = o3d.io.read_point_cloud(str(gt_object_cloud_paths[frame_list[state["idx"]]]))
#     pcd_gt_dyn.points = o3d.utility.Vector3dVector(np.asarray(gt_pcd0.points, dtype=np.float64))


pcd_landmarks = o3d.geometry.PointCloud()
pcd_landmarks.points = o3d.utility.Vector3dVector(landmarks_vis.astype(np.float64))
pcd_landmarks.paint_uniform_color([1.0, 0.8, 0.2])

lines = o3d.geometry.LineSet()
lines.points = o3d.utility.Vector3dVector(edg_pts0)
lines.lines = o3d.utility.Vector2iVector(line_idx0)
lines.paint_uniform_color([0.0, 0.0, 0.0])

bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(edg_pts0))
diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
pt_size = 3.0 if diag <= 0 else float(np.clip(800.0 / (diag * 1000.0 + 1e-6), 2.0, 6.0))
cam_axis_len = 0.05 * diag if diag > 0 else 0.05

cam_pts0, cam_lines0, cam_colors0 = _camera_axes_lines(
    cam_centers_vis[frame_list[state["idx"]]],
    cam_rot_vis[frame_list[state["idx"]]],
    cam_axis_len,
)
cam_axes = o3d.geometry.LineSet()
cam_axes.points = o3d.utility.Vector3dVector(cam_pts0)
cam_axes.lines = o3d.utility.Vector2iVector(cam_lines0)
cam_axes.colors = o3d.utility.Vector3dVector(cam_colors0)

gt_cam_axes = o3d.geometry.LineSet()
if has_valid_gt_poses and np.isfinite(gt_cam_centers[frame_list[state["idx"]]]).all():
    gt_cam_pts0, gt_cam_lines0, gt_cam_colors0 = _camera_axes_lines(
        gt_cam_centers[frame_list[state["idx"]]],
        gt_cam_rot[frame_list[state["idx"]]],
        cam_axis_len,
    )
    gt_cam_axes.points = o3d.utility.Vector3dVector(gt_cam_pts0)
    gt_cam_axes.lines = o3d.utility.Vector2iVector(gt_cam_lines0)
    gt_cam_axes.colors = o3d.utility.Vector3dVector(gt_cam_colors0)

rot_axis_len = 0.04 * diag if diag > 0 else 0.04

def _rotation_axes_lines(pts, rots, axis_len):
    if pts.size == 0 or rots.size == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 2), dtype=np.int32), np.zeros((0, 3), dtype=np.float64)

    pts = np.asarray(pts, dtype=np.float64)
    rots = np.asarray(rots, dtype=np.float64)
    all_pts = []
    lines_out = []
    colors = []
    for i, p in enumerate(pts):
        R = rots[i]
        if not np.isfinite(R).all():
            continue
        start_idx = len(all_pts)
        axes = R @ (axis_len * np.eye(3))
        all_pts.append(p)
        all_pts.append(p + axes[:, 0])
        all_pts.append(p + axes[:, 1])
        all_pts.append(p + axes[:, 2])
        lines_out.extend([[start_idx, start_idx + 1], [start_idx, start_idx + 2], [start_idx, start_idx + 3]])
        colors.extend([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    if len(all_pts) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 2), dtype=np.int32), np.zeros((0, 3), dtype=np.float64)

    return np.asarray(all_pts, dtype=np.float64), np.asarray(lines_out, dtype=np.int32), np.asarray(colors, dtype=np.float64)


rot_pts0, rot_lines0, rot_colors0 = _rotation_axes_lines(edg_pts0, edg_rots0, rot_axis_len)
rot_axes = o3d.geometry.LineSet()
rot_axes.points = o3d.utility.Vector3dVector(rot_pts0)
rot_axes.lines = o3d.utility.Vector2iVector(rot_lines0)
rot_axes.colors = o3d.utility.Vector3dVector(rot_colors0)


def refresh(vis, frame_id: int):
    edg_pts, dyn_pts, line_idx, edg_rots = get_frame_geoms(frame_id)
    pcd_edg.points = o3d.utility.Vector3dVector(edg_pts)
    pcd_dyn.points = o3d.utility.Vector3dVector(dyn_pts)
    lines.points = o3d.utility.Vector3dVector(edg_pts)
    lines.lines = o3d.utility.Vector2iVector(line_idx)

    gt_pts = np.zeros((0, 3), dtype=np.float64)
    if has_gt_object_clouds and gt_object_cloud_paths[frame_id] is not None:
        gt_pcd = o3d.io.read_point_cloud(str(gt_object_cloud_paths[frame_id]))
        gt_pts = np.asarray(gt_pcd.points, dtype=np.float64)
    # pcd_gt_dyn.points = o3d.utility.Vector3dVector(gt_pts)

    rot_pts, rot_lines, rot_colors = _rotation_axes_lines(edg_pts, edg_rots, rot_axis_len)
    rot_axes.points = o3d.utility.Vector3dVector(rot_pts)
    rot_axes.lines = o3d.utility.Vector2iVector(rot_lines)
    rot_axes.colors = o3d.utility.Vector3dVector(rot_colors)

    cam_pts, cam_lines, cam_colors = _camera_axes_lines(
        cam_centers_vis[frame_id], cam_rot_vis[frame_id], cam_axis_len
    )
    cam_axes.points = o3d.utility.Vector3dVector(cam_pts)
    cam_axes.lines = o3d.utility.Vector2iVector(cam_lines)
    cam_axes.colors = o3d.utility.Vector3dVector(cam_colors)

    if has_valid_gt_poses and frame_id < gt_cam_centers.shape[0] and np.isfinite(gt_cam_centers[frame_id]).all():
        gt_cam_pts, gt_cam_lines, gt_cam_colors = _camera_axes_lines(
            gt_cam_centers[frame_id], gt_cam_rot[frame_id], cam_axis_len
        )
        gt_cam_axes.points = o3d.utility.Vector3dVector(gt_cam_pts)
        gt_cam_axes.lines = o3d.utility.Vector2iVector(gt_cam_lines)
        gt_cam_axes.colors = o3d.utility.Vector3dVector(gt_cam_colors)
        vis.update_geometry(gt_cam_axes)

    vis.update_geometry(pcd_edg)
    vis.update_geometry(pcd_dyn)
    # vis.update_geometry(pcd_gt_dyn)
    vis.update_geometry(lines)
    vis.update_geometry(cam_axes)
    vis.update_geometry(rot_axes)
    print(f"[Open3D] frame={frame_id} edg={len(edg_pts)} dyn={len(dyn_pts)} gt_dyn={len(gt_pts)}")


def on_space(vis):
    state["idx"] = (state["idx"] + 1) % len(frame_list)
    frame_id = frame_list[state["idx"]]
    refresh(vis, frame_id)
    return False


def on_b(vis):
    state["idx"] = (state["idx"] - 1) % len(frame_list)
    frame_id = frame_list[state["idx"]]
    refresh(vis, frame_id)
    return False


print(f"Open3D controls: SPACE=next frame, B=prev frame. Start at frame={frame_list[state['idx']]}.")
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(window_name="EDG deformation (full)", width=1280, height=720)
vis.add_geometry(pcd_edg)
vis.add_geometry(lines)
vis.add_geometry(pcd_dyn)
# vis.add_geometry(pcd_gt_dyn)
vis.add_geometry(pcd_landmarks)
vis.add_geometry(cam_axes)
if has_valid_gt_poses:
    vis.add_geometry(gt_cam_axes)
vis.add_geometry(rot_axes)
opt = vis.get_render_option()
opt.point_size = pt_size
vis.register_key_callback(ord(" "), on_space)
vis.register_key_callback(ord("b"), on_b)
vis.run()
vis.destroy_window()
