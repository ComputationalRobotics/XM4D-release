"""Per-segment EDG solve, used by EDG_test_fast.py (step 4c): load_all, build_segment_inputs, solve_segment.

Segments:
  * rest frame = FIRST frame of the segment window; the WHOLE window is solved (so the last/overlap frame is
    always present -- that is what chains segments).
  * world lift with the XM solver scale "s":  X_world = s_t * R_t @ x_cam + t_t.
  * QCQP solve via utils.edg.solve_qcqp_and_recover_until_part5 (w_obs=1, w_arap and w_smooth from the caller),
    rest anchored (trev=1, R=I, p=0).

From edg_topology.py (edg_topology.npz):
  * EDG nodes + edges = the FIXED global topology, restricted per segment to nodes valid at the rest frame
    (plus the nodes carried from the previous segment).
  * dynamic skinning = each dynamic point is skinned to the EDG nodes assigned to it in dyn_edg_edges,
    inverse-distance weighted at the rest frame.
"""
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import utils.definitions
from utils.edg import scatter_solved_poses, scatter_solved_frame_vector, solve_qcqp_and_recover_until_part5
from utils.io import load_matrix_from_bin

sys.path.append(os.path.join(os.path.dirname(__file__), "../XM/build/"))
import XM


# ------------------------------------------------------------------------------ load tracks + topology
def load_all():
    """Load the step-4a dynamic tracks, world-lift with scale 's', and load the fixed EDG topology +
    dynamic->EDG skinning assignment produced by edg_topology.py from the same tracks (same track id space)."""
    root = Path(utils.definitions.rootpath)
    data = np.load(utils.definitions.dynamic_tracks_for_edg(), allow_pickle=True)
    tracks_xyz = data["tracks_xyz"]                  # (1,Q,T,3)
    occluded = data["occluded"]                      # (1,Q,T)
    dyn_mask = data["dynamic_obs_mask"]              # (Q,T)
    T = tracks_xyz.shape[2]
    Q = tracks_xyz.shape[1]

    rt = np.load(str(root / "outputs" / "result_XM.npz"))
    R_raw, t_raw = scatter_solved_poses(rt, T)
    s_raw = scatter_solved_frame_vector(rt, "s", T, default=1.0)          # XM solver per-frame scale
    R_for_edg = R_raw.astype(np.float64).copy()
    R_for_edg *= s_raw[:, None, None]
    glob = np.einsum("tij,qtj->qti", R_for_edg, tracks_xyz[0].astype(np.float64))  # (Q,T,3)
    glob += t_raw[None, :, :]                                                      # in place (speed only)
    valid = (np.isfinite(glob).all(axis=-1) & (occluded[0] == 0) & dyn_mask.astype(bool))              # (Q,T)

    topo = np.load(Path(utils.definitions.output_dir) / "edg_topology.npz")
    edg_nodes = np.asarray(topo["edg_nodes"], dtype=int)                  # (M,) global track ids
    edg_edges = np.asarray(topo["edg_edges"], dtype=int).reshape(-1, 2)   # (E,2) global id pairs
    dyn_edg = np.asarray(topo["dyn_edg_edges"], dtype=int).reshape(-1, 2) # (N,2) (dyn_id, node_id) global
    mx = max(int(edg_nodes.max(initial=0)), int(dyn_edg.max(initial=0)))
    topo_q = int(topo["n_tracks"])                                          # id-space stamp
    if topo_q != Q or mx >= Q:
        raise ValueError(
            f"TRACK-ID SPACE MISMATCH: edg_topology.npz was built for {topo_q} tracks but the loaded tracks have "
            f"Q={Q}. Rerun edg_topology.py on these tracks.")
    dyn_to_nodes = defaultdict(list)
    for d, n in dyn_edg.tolist():
        dyn_to_nodes[int(d)].append(int(n))
    dyn_to_nodes = {d: np.asarray(v, dtype=int) for d, v in dyn_to_nodes.items()}
    dyn_assign = np.full((Q, max([v.size for v in dyn_to_nodes.values()], default=0)), -1, dtype=int)
    for d, v in dyn_to_nodes.items():                                      # the same lists, -1 padded (speed only)
        dyn_assign[d, :v.size] = v

    print(f"[load] Q={Q} T={T} | EDG: {edg_nodes.size} nodes, {edg_edges.shape[0]} edges | "
          f"skinning: {len(dyn_to_nodes)} dynamic points assigned")
    return dict(glob=glob, valid=valid, T=T, Q=Q,
                edg_nodes=edg_nodes, edg_edges=edg_edges, dyn_to_nodes=dyn_to_nodes, dyn_assign=dyn_assign)


# ------------------------------------------------------------------------------- per-segment solver glue
def build_segment_inputs(D, seg_frames, edg_pos):
    """Assemble solver inputs for one segment (rest = first frame; the WHOLE window is
    solved). None if unsolvable.

    EDG nodes present in this segment = (nodes OBSERVED at the rest frame) UNION (nodes CARRIED from a
    previous segment -- i.e. already holding a finite pose at this rest/overlap frame in `edg_pos`). A
    carried node that is NOT observed at the rest frame keeps its carried REST POSE as g_i, so it stays in
    the graph and keeps being solved. Because every previous-
    segment node writes its pose at the shared overlap frame, the node set grows monotonically.

    Skinning neighbours = each dynamic point's assigned nodes (dyn_edg_edges) that are present, inverse-
    distance weighted at the rest frame using the node REST POSES g (works for carried nodes too)."""
    glob, valid = D["glob"], D["valid"]
    edg_nodes, edg_edges = D["edg_nodes"], D["edg_edges"]
    rest = int(seg_frames[0])

    valid_at_rest = edg_nodes[valid[edg_nodes, rest]]                          # observed at rest
    carried = [n for n in edg_pos.keys() if np.isfinite(edg_pos[n][rest]).all()]  # held from prev segment
    present = np.array(sorted(set(valid_at_rest.tolist()) | set(carried)), dtype=int)
    if present.size < 2:
        return None
    present_set = set(present.tolist())
    edg_idx_map = {int(n): j for j, n in enumerate(present.tolist())}
    M = present.size

    # rest positions g: observed nodes -> obs at rest; carried (unobserved) nodes -> carried rest pose
    g = np.empty((M, 3), dtype=np.float64)
    n_carried = 0
    for j, n in enumerate(present.tolist()):
        if valid[n, rest]:
            g[j] = glob[n, rest]
        else:
            g[j] = edg_pos[n][rest]                                            # carried rest pose
            n_carried += 1

    # solve the ENTIRE window (rest is frames[0]); keeping the last frame lets the next segment chain.
    frames = np.asarray(seg_frames, dtype=int)
    T_rev = frames.size
    if T_rev < 2:
        return None
    frame_to_trev = {int(f): i + 1 for i, f in enumerate(frames.tolist())}

    edges = np.array([[a, b] for a, b in edg_edges.tolist()
                      if a in present_set and b in present_set], dtype=int).reshape(-1, 2)

    edg_obs = np.full((M, T_rev, 3), np.nan, dtype=np.float64)     # ACTIVE: anchors nodes at observed frames
    # with utils.edg's default w_edg_obs=10.0 (vs w_obs=1 for the dyn data term) -- the solve call below does
    # not override it.
    edg_obs_mask = np.zeros((M, T_rev), dtype=bool)
    for j, n in enumerate(present.tolist()):
        for ti, f in enumerate(frames.tolist()):
            if valid[n, f]:
                edg_obs[j, ti, :] = glob[n, f]
                edg_obs_mask[j, ti] = True

    # dynamic keypoints = tracks valid at rest that have an assignment, skinned to their PRESENT nodes
    kp_neighbors, kp_weights, kp_rest = {}, {}, {}
    cand_q = np.where(valid[:, rest])[0]
    # batched over the candidates with the same number of present assigned nodes K (speed only; the same arrays and
    # the same per-row norm / sum as one keypoint at a time, in candidate order)
    loc = np.full(D["Q"], -1, dtype=int); loc[present] = np.arange(M)    # global id -> row of g (-1 = not present)
    A = D["dyn_assign"][cand_q]                                            # assigned nodes, -1 padded
    keep = (A >= 0) & (loc[np.maximum(A, 0)] >= 0)                         # ... that are present (assignment order)
    cnt = keep.sum(1)
    ok = cnt >= 2                                                          # solver needs >= 2 neighbours
    nb_rows, w_rows, pr_rows = [None] * cand_q.size, [None] * cand_q.size, [None] * cand_q.size
    for K in np.unique(cnt[ok]).tolist():
        r = np.nonzero(ok & (cnt == K))[0]
        nb = A[r][keep[r]].reshape(-1, K)
        pr = glob[cand_q[r], rest].astype(np.float64)
        nb_pos = g[loc[nb]]                                               # node REST POSES (carried or observed)
        d = np.linalg.norm(nb_pos - pr[:, None, :], axis=2)
        near = (d <= 1e-8).any(1)
        w = np.zeros_like(d); w[near, np.argmin(d[near], axis=1)] = 1.0
        wf = 1.0 / (d[~near] + 1e-8); w[~near] = wf / wf.sum(1, keepdims=True)
        for i, ri in enumerate(r.tolist()):
            nb_rows[ri], w_rows[ri], pr_rows[ri] = nb[i], w[i], pr[i]
    qs = cand_q[ok]
    for i, q in zip(np.nonzero(ok)[0].tolist(), qs.tolist()):
        kp_neighbors[q] = nb_rows[i]
        kp_weights[q] = w_rows[i]
        kp_rest[q] = pr_rows[i]
    oi, ti = np.nonzero(valid[qs][:, frames])                              # (keypoint, frame) in candidate order
    if not oi.size:
        return None
    dyn_kp_idx = qs[oi].astype(int)
    dyn_frame = frames[ti].astype(int)
    dyn_global = glob[qs[oi], frames[ti]].astype(np.float64)

    return dict(edges=edges, edg_idx_map=edg_idx_map, g=g, edg_obs=edg_obs, edg_obs_mask=edg_obs_mask,
                N_dyn_obs=int(dyn_kp_idx.size), dyn_kp_idx=dyn_kp_idx, dyn_frame=dyn_frame,
                dyn_global=dyn_global, frame_to_trev=frame_to_trev, kp_neighbors=kp_neighbors,
                kp_weights=kp_weights, kp_rest=kp_rest,
                present=present, frames=frames, rest=rest, M=M, T_rev=T_rev, n_carried=n_carried)


def solve_segment(inp, output_dir, w_arap, w_smooth):
    out = solve_qcqp_and_recover_until_part5(
        edges=inp["edges"], edg_idx_map=inp["edg_idx_map"], g=inp["g"],
        edg_obs=inp["edg_obs"], edg_obs_mask=inp["edg_obs_mask"],
        N_dyn_obs=inp["N_dyn_obs"], dyn_kp_idx=inp["dyn_kp_idx"], dyn_frame=inp["dyn_frame"],
        dyn_global=inp["dyn_global"], frame_to_trev=inp["frame_to_trev"],
        kp_neighbors=inp["kp_neighbors"], kp_weights=inp["kp_weights"], kp_rest=inp["kp_rest"],
        output_dir=str(output_dir), solve_Ronly_fn=XM.solve_Ronly, load_matrix_from_bin_fn=load_matrix_from_bin,
        w_obs=1.0, w_arap=w_arap, w_smooth=w_smooth,
    )
    return (np.asarray(out["edg_deformed"], np.float64), np.asarray(out["R_edg"], np.float64),
            np.asarray(out["dyn_mapped"], np.float64), np.unique(inp["dyn_kp_idx"]))
