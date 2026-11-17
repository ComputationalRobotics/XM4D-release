#!/usr/bin/env python3
"""EDG_test.py -- per-segment EDG solve with the SAME solver + carryover logic as EDG_old.py, but with
the EDG topology AND dynamic skinning taken from build_edg.py (outputs/edg_topology.npz). Visualizes
after each segment.

Kept identical to EDG_old.py:
  * 11-frame segment windows with a 1-frame OVERLAP; rest frame = FIRST frame of the window; the WHOLE
    window is solved (so the last/overlap frame is always present -- that is what chains segments).
  * world lift with the XM solver scale "s":  X_world = s_t * R_t @ x_cam + t_t.
  * QCQP solve via utils.edg.solve_qcqp_and_recover_until_part5 (w_obs=1, w_arap=0.2, w_smooth=0.2),
    rest anchored (trev=1, R=I, p=0).
  * CROSS-SEGMENT PRESERVATION: a persistent per-node global-rotation store (init identity) spans the
    whole video. Each segment reads rest_global = store[node, rest_frame] -- written by the PREVIOUS
    segment at the shared overlap frame -- and composes R_global = rest_global @ R_local before storing
    (EDG_old's _compose_global_rotations). So a node's rotation ACCUMULATES across segments instead of
    resetting; positions are re-anchored to observations at each rest (exactly as EDG_old).

Swapped in from build_edg.py (edg_topology.npz):
  * EDG nodes + edges = the FIXED global topology, restricted per segment to nodes valid at the rest frame.
  * dynamic skinning = each dynamic point is skinned to the EDG nodes assigned to it in dyn_edg_edges,
    inverse-distance weighted at the rest frame.

Run (from Python/):  XM_DATASET_NAME=<scene> conda run -n XM python EDG_test.py [--no-viz] [--seg-len 11]
"""
import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import utils.definitions
from utils.edg import scatter_solved_poses, scatter_solved_frame_vector, solve_qcqp_and_recover_until_part5
from utils.io import load_matrix_from_bin
import visualize_track_edges as VTE          # for _edg_sphere_mesh (module main-guarded, safe to import)

sys.path.append(os.path.join(os.path.dirname(__file__), "../XM/build/"))
import XM


# ------------------------------------------------------------------------------ load tracks + topology
def load_all(tracks_path=None):
    """Load tracks (same npz as EDG_old, or the merged tracks if tracks_path is given), world-lift with
    scale 's' (EDG_old convention), and load the fixed EDG topology + dynamic->EDG skinning assignment
    produced by build_edg.py. tracks_path MUST match what build_edg used (same track id space)."""
    root = Path(utils.definitions.rootpath)
    data = np.load(tracks_path or utils.definitions.dynamic_tracks_for_edg(), allow_pickle=True)
    tracks_xyz = data["tracks_xyz"]                  # (1,Q,T,3)
    occluded = data["occluded"]                      # (1,Q,T)
    dyn_mask = data["dynamic_obs_mask"]              # (Q,T)
    T = tracks_xyz.shape[2]
    Q = tracks_xyz.shape[1]

    rt = np.load(str(root / "outputs" / "result_XM.npz"))
    R_raw, t_raw = scatter_solved_poses(rt, T)
    s_raw = scatter_solved_frame_vector(rt, "s", T, default=1.0)          # XM solver per-frame scale
    R_for_edg = R_raw.astype(np.float64).copy()
    if s_raw is not None:
        R_for_edg *= s_raw[:, None, None]
    glob = np.einsum("tij,qtj->qti", R_for_edg, tracks_xyz[0].astype(np.float64)) + t_raw[None, :, :]  # (Q,T,3)
    valid = (np.isfinite(glob).all(axis=-1) & (occluded[0] == 0) & dyn_mask.astype(bool))              # (Q,T)

    topo = np.load(Path(utils.definitions.output_dir) / "edg_topology.npz")
    edg_nodes = np.asarray(topo["edg_nodes"], dtype=int)                  # (M,) global track ids
    edg_edges = np.asarray(topo["edg_edges"], dtype=int).reshape(-1, 2)   # (E,2) global id pairs
    dyn_edg = np.asarray(topo["dyn_edg_edges"], dtype=int).reshape(-1, 2) # (N,2) (dyn_id, node_id) global
    mx = max(int(edg_nodes.max(initial=0)), int(dyn_edg.max(initial=0)))
    topo_q = int(topo["n_tracks"]) if "n_tracks" in topo.files else None    # id-space stamp (newer topos)
    if (topo_q is not None and topo_q != Q) or mx >= Q:
        raise ValueError(
            f"TRACK-ID SPACE MISMATCH: edg_topology.npz was built for "
            f"{topo_q if topo_q is not None else f'>={mx + 1}'} tracks but the loaded tracks have "
            f"Q={Q}. build_edg.py and this solver MUST use the same --merged setting (merged: Q~7634, "
            f"raw: Q~14877 on paper-windmill). Rebuild the topology with the matching flag, e.g. "
            f"`python build_edg.py --merged`, or drop/add --merged on this run.")
    dyn_to_nodes = defaultdict(list)
    for d, n in dyn_edg.tolist():
        dyn_to_nodes[int(d)].append(int(n))
    dyn_to_nodes = {d: np.asarray(v, dtype=int) for d, v in dyn_to_nodes.items()}

    print(f"[load] Q={Q} T={T} | EDG: {edg_nodes.size} nodes, {edg_edges.shape[0]} edges | "
          f"skinning: {len(dyn_to_nodes)} dynamic points assigned")
    return dict(glob=glob, valid=valid, T=T, Q=Q,
                edg_nodes=edg_nodes, edg_edges=edg_edges, dyn_to_nodes=dyn_to_nodes)


# ------------------------------------------------------------------------------- per-segment solver glue
def build_segment_inputs(D, seg_frames, edg_pos, max_dyn=None, rng=None):
    """Assemble solver inputs for one segment (EDG_old convention: rest = first frame; the WHOLE window is
    solved). None if unsolvable.

    EDG nodes present in this segment = (nodes OBSERVED at the rest frame) UNION (nodes CARRIED from a
    previous segment -- i.e. already holding a finite pose at this rest/overlap frame in `edg_pos`). A
    carried node that is NOT observed at the rest frame keeps its carried REST POSE as g_i, so it stays in
    the graph and keeps being solved (EDG_old's seed-missing-nodes behaviour). Because every previous-
    segment node writes its pose at the shared overlap frame, the node set grows monotonically.

    Skinning neighbours = each dynamic point's assigned nodes (dyn_edg_edges) that are present, inverse-
    distance weighted at the rest frame using the node REST POSES g (works for carried nodes too)."""
    glob, valid = D["glob"], D["valid"]
    edg_nodes, edg_edges, dyn_to_nodes = D["edg_nodes"], D["edg_edges"], D["dyn_to_nodes"]
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
    dyn_kp_idx, dyn_frame, dyn_global = [], [], []
    cand_q = np.where(valid[:, rest])[0]
    if max_dyn is not None and cand_q.size > int(max_dyn):     # subsample dyn keypoints (data) for speed
        cand_q = (rng if rng is not None else np.random.default_rng()).choice(
            cand_q, size=int(max_dyn), replace=False)
    for q in cand_q.tolist():
        assigned = dyn_to_nodes.get(q)
        if assigned is None:
            continue
        nb = np.array([n for n in assigned.tolist() if n in present_set], dtype=int)
        if nb.size < 2:                                           # solver needs >= 2 neighbours
            continue
        pr = glob[q, rest].astype(np.float64)
        nb_pos = g[[edg_idx_map[int(n)] for n in nb.tolist()]]   # node REST POSES (carried or observed)
        d = np.linalg.norm(nb_pos - pr[None, :], axis=1)
        if np.any(d <= 1e-8):
            w = np.zeros_like(d); w[int(np.argmin(d))] = 1.0
        else:
            w = 1.0 / (d + 1e-8); w = w / w.sum()
        kp_neighbors[q] = nb
        kp_weights[q] = w.astype(np.float64)
        kp_rest[q] = pr
        for f in frames.tolist():
            if valid[q, f]:
                dyn_kp_idx.append(q); dyn_frame.append(int(f)); dyn_global.append(glob[q, f])
    if not dyn_kp_idx:
        return None
    dyn_kp_idx = np.array(dyn_kp_idx, dtype=int)
    dyn_frame = np.array(dyn_frame, dtype=int)
    dyn_global = np.array(dyn_global, dtype=np.float64)

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


# ----------------------------------------------------------------------------------------- per-segment viz
def visualize_segment(edg_deformed, R_global_seg, dyn_mapped, inp, node_r, seg_id):
    """Open3D window for ONE solved segment: red = solved EDG (nodes+edges), blue = deformed dynamic
    points, and per-node RGB axes = the CHAINED global rotation (R_global). SPACE/B step frames, Q/ESC -> next."""
    import open3d as o3d
    present, frames, idx_map = inp["present"], inp["frames"], inp["edg_idx_map"]
    edges_local = np.array([[idx_map[int(a)], idx_map[int(b)]] for a, b in inp["edges"].tolist()],
                           dtype=np.int32).reshape(-1, 2)
    M = present.size
    axlen = node_r * 8.0
    ax_lines = np.concatenate([np.stack([np.arange(M), np.arange(M) + M], 1),
                               np.stack([np.arange(M), np.arange(M) + 2 * M], 1),
                               np.stack([np.arange(M), np.arange(M) + 3 * M], 1)], 0).astype(np.int32)
    ax_cols = np.concatenate([np.tile([1., 0, 0], (M, 1)), np.tile([0, 1., 0], (M, 1)),
                              np.tile([0, 0, 1.], (M, 1))], 0)

    mesh = o3d.geometry.TriangleMesh()
    line = o3d.geometry.LineSet()
    rot_ls = o3d.geometry.LineSet()          # per-node rotation axes (chained global R)
    dyn_pcd = o3d.geometry.PointCloud()
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=node_r * 20)
    st = {"i": 0}

    def fill(i):
        P = edg_deformed[:, i, :]
        sm = VTE._edg_sphere_mesh(P, node_r, [1.0, 0.05, 0.05])
        mesh.vertices, mesh.triangles = sm.vertices, sm.triangles
        mesh.vertex_colors, mesh.vertex_normals = sm.vertex_colors, sm.vertex_normals
        if edges_local.size:
            line.points = o3d.utility.Vector3dVector(P)
            line.lines = o3d.utility.Vector2iVector(edges_local)
            line.paint_uniform_color([1.0, 0.05, 0.05])
        Rg = R_global_seg[:, i, :, :]                                          # (M,3,3) composed global R
        axpts = np.concatenate([P, P + axlen * Rg[:, :, 0], P + axlen * Rg[:, :, 1], P + axlen * Rg[:, :, 2]], 0)
        rot_ls.points = o3d.utility.Vector3dVector(axpts)
        rot_ls.lines = o3d.utility.Vector2iVector(ax_lines)
        rot_ls.colors = o3d.utility.Vector3dVector(ax_cols)
        dyn_pcd.points = o3d.utility.Vector3dVector(dyn_mapped[:, i, :])
        dyn_pcd.paint_uniform_color([0.2, 0.4, 1.0])
        tag = " (REST/overlap)" if int(frames[i]) == inp["rest"] else ""
        print(f"[seg {seg_id}] frame {int(frames[i])}{tag}  trev {i+1}/{len(frames)}  "
              f"nodes={M} dyn={dyn_mapped.shape[0]}")

    def refresh(vis):
        fill(st["i"])
        for gm in (mesh, line, rot_ls, dyn_pcd):
            vis.update_geometry(gm)
        return False

    def nxt(vis): st["i"] = (st["i"] + 1) % len(frames); return refresh(vis)
    def prv(vis): st["i"] = (st["i"] - 1) % len(frames); return refresh(vis)
    def clo(vis): vis.close(); return False

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"EDG_test segment {seg_id}", width=1280, height=720)
    fill(0)
    for gm in (mesh, line, rot_ls, dyn_pcd, axes):
        vis.add_geometry(gm)
    vis.get_render_option().point_size = 5.0
    vis.register_key_callback(ord(" "), nxt)
    vis.register_key_callback(ord("b"), prv); vis.register_key_callback(ord("B"), prv)
    vis.register_key_callback(ord("q"), clo); vis.register_key_callback(ord("Q"), clo); vis.register_key_callback(256, clo)
    print(f"[seg {seg_id}] SPACE=next frame, B=prev, Q/ESC=close (-> next segment). RGB axes = chained global rotation.")
    vis.run(); vis.destroy_window()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg-len", type=int, default=11, help="segment window length (EDG_old: 11, 1-frame overlap)")
    ap.add_argument("--w-arap", type=float, default=0.2)
    ap.add_argument("--w-smooth", type=float, default=0.2)
    ap.add_argument("--max-segments", type=int, default=0, help="stop after N segments (0 = all; for testing)")
    ap.add_argument("--no-viz", action="store_true", help="skip the per-segment Open3D viewer")
    ap.add_argument("--merged", action="store_true",
                    help="solve on the MERGED tracks (outputs/tracks_dynamic_merged.npz); MUST match how "
                         "build_edg.py built edg_topology.npz (i.e. build_edg.py --merged)")
    args = ap.parse_args()

    merged_path = str(Path(utils.definitions.output_dir) / "tracks_dynamic_merged.npz")
    if args.merged:
        print(f"[merged] solving on consolidated tracks: {merged_path}")
    D = load_all(tracks_path=merged_path if args.merged else None)
    T = D["T"]
    out_dir = Path(utils.definitions.output_dir) / "edg_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    en = D["edg_nodes"]
    gp = D["glob"][en][D["valid"][en]]
    diag = float(np.linalg.norm(gp.max(0) - gp.min(0))) if gp.size else 1.0
    node_r = max(1e-4, 0.006 * diag)

    # persistent global stores over the whole video (chained across segments, like EDG_old)
    edg_rot = {}   # gid -> [T,3,3] global rotation (init identity; carries across the overlap frame)
    edg_pos = {}   # gid -> [T,3] deformed EDG-node position
    dyn_pos = {}   # gid -> [T,3] deformed dynamic-point position

    def ens_rot(gid):
        if gid not in edg_rot:
            edg_rot[gid] = np.tile(np.eye(3, dtype=np.float64), (T, 1, 1))
        return edg_rot[gid]

    def ens_pos(store, gid):
        if gid not in store:
            store[gid] = np.full((T, 3), np.nan, dtype=np.float64)
        return store[gid]

    current, seg_id, solved = 0, 0, []
    while True:
        end = min(T, current + args.seg_len)
        seg_frames = np.arange(current, end, dtype=int)
        inp = build_segment_inputs(D, seg_frames, edg_pos)
        if inp is None:
            print(f"=== Segment {seg_id} frames [{current},{end-1}]: not solvable (no valid EDG nodes); skipped ===")
        else:
            print(f"\n=== Segment {seg_id} frames [{current},{end-1}]  rest={inp['rest']}  nodes={inp['M']} "
                  f"({inp['n_carried']} carried)  frames={inp['T_rev']}  edges={inp['edges'].shape[0]}  "
                  f"dyn-obs={inp['N_dyn_obs']} ===")
            edg_deformed, R_edg, dyn_mapped, unique_kps = solve_segment(inp, out_dir, args.w_arap, args.w_smooth)
            present, frames, rest = inp["present"], inp["frames"], inp["rest"]

            # --- CHAIN (EDG_old): compose this segment's local rotations onto the global rotation carried
            #     at the rest/overlap frame from the previous segment; scatter global R + positions. ---
            R_global_seg = np.empty_like(R_edg)                    # [M,T_rev,3,3] composed, for viz/save
            for j, gid in enumerate(present.tolist()):
                rot_arr = ens_rot(gid)
                rest_global = rot_arr[rest] if np.isfinite(rot_arr[rest]).all() else np.eye(3)
                Rg = rest_global[None, :, :] @ R_edg[j]            # _compose_global_rotations
                R_global_seg[j] = Rg
                pos_arr = ens_pos(edg_pos, gid)
                for ti, f in enumerate(frames.tolist()):
                    rot_arr[f] = Rg[ti]
                    pos_arr[f] = edg_deformed[j, ti]
            for i, gid in enumerate(unique_kps.tolist()):
                arr = ens_pos(dyn_pos, gid)
                for ti, f in enumerate(frames.tolist()):
                    arr[f] = dyn_mapped[i, ti]

            np.savez_compressed(out_dir / f"segment_{seg_id:03d}.npz",
                                edg_deformed=edg_deformed, R_edg=R_edg, R_global=R_global_seg,
                                dyn_mapped=dyn_mapped, present=present, frames=frames,
                                rest_frame=rest, edges=inp["edges"], dyn_kp_ids=unique_kps)
            solved.append(seg_id)
            if not args.no_viz:
                visualize_segment(edg_deformed, R_global_seg, dyn_mapped, inp, node_r, seg_id)
        if args.max_segments and len(solved) >= args.max_segments:
            break
        if end >= T:
            break
        current = end - 1                      # EDG_old: 1-frame overlap (next rest = this window's last frame)
        seg_id += 1

    # save the chained global trajectories (rotation accumulates across segments; positions per frame)
    node_ids = np.array(sorted(edg_pos.keys()), dtype=int)
    dyn_ids = np.array(sorted(dyn_pos.keys()), dtype=int)
    edg_reconstruct = np.stack([edg_pos[g] for g in node_ids.tolist()]) if node_ids.size else np.zeros((0, T, 3))
    edg_rotation = np.stack([edg_rot[g] for g in node_ids.tolist()]) if node_ids.size else np.zeros((0, T, 3, 3))
    dyn_reconstruct = np.stack([dyn_pos[g] for g in dyn_ids.tolist()]) if dyn_ids.size else np.zeros((0, T, 3))
    node_set = set(node_ids.tolist())                                   # global EDG edges among solved nodes
    edg_edges_global = np.array([[a, b] for a, b in D["edg_edges"].tolist()
                                 if a in node_set and b in node_set], dtype=np.int64).reshape(-1, 2)
    np.savez_compressed(out_dir / "edg_global.npz",
                        node_ids=node_ids, edg_reconstruct=edg_reconstruct, edg_rotation=edg_rotation,
                        edg_edges=edg_edges_global,
                        dyn_ids=dyn_ids, dyn_reconstruct=dyn_reconstruct)
    print(f"\n[done] solved {len(solved)} segments; chained global -> {out_dir}/edg_global.npz "
          f"({node_ids.size} nodes, {dyn_ids.size} dyn points over T={T})")


if __name__ == "__main__":
    main()
    # All segments are solved and saved by here. The XM C++ extension can double-free at interpreter
    # teardown ("free(): invalid pointer" -> SIGABRT/exit 134) AFTER the work is done; skip that teardown
    # with a hard exit so the script returns 0 cleanly.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
