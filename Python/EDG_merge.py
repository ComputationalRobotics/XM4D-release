#!/usr/bin/env python3
"""Divide-and-conquer EDG solve on a FIXED global topology (Notes/main_1.pdf).

Pipeline
--------
1. Build the GLOBAL EDG ONCE (topology never changes during optimization):
     * track graph  = visualize_track_edges.build_edg_edges   (persistent-proximity)
     * EDG nodes/edges = visualize_track_edges.build_edg_graph (length-priority NMS + hop edges)
   `edg_nodes`/`edg_edges` are global track ids and are treated as a fixed global variable.

2. Split the video into INDEPENDENT non-overlapping segments [0,L),[L,2L),... (L=10 by default) and
   solve each one on its own. For a segment we solve the local EDG deformation {(R_it, p_it)} that
   fills the node positions across the segment (a node's position is g_i + p_it; ARAP couples
   neighbours; temporal smoothness couples consecutive frames -- eqs (8)-(10) of main_1.pdf).

   * REST frame = the FIRST frame of the segment (with >=2 valid EDG nodes) -- the temporal anchor
     (R_i=I, p_i=0, solver trev=1), as in EDG_old. Keeping rest first puts the whole segment in true
     time order, so the smoothness chain (consecutive trev) matches eq(10) exactly.
   * A segment frame where NO EDG node is valid is skipped (not solved).
   * The solve reuses utils.edg.solve_qcqp_and_recover_until_part5 (+ XM.solve_Ronly) -- the SAME
     QCQP/Schur/rotation solver as EDG_old.py. Data term: each EDG node is a dynamic keypoint skinned
     to its K nearest EDG nodes; since a node is its own nearest neighbour (dist 0 -> weight 1) this
     reduces to g_i + p_it ~= observation, i.e. the paper's node data term.

3. After each segment is solved, visualize it (solved EDG nodes+edges vs. observations), SPACE to step
   through the segment frames.

Run (from Python/):
    XM_DATASET_NAME=<scene> conda run -n XM python solve_edg_segments.py [--no-viz] [--seg-len 10]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import utils.definitions
from utils.edg import solve_qcqp_and_recover_until_part5
from utils.io import load_matrix_from_bin
import visualize_track_edges as VTE

sys.path.append(os.path.join(os.path.dirname(__file__), "../XM/build/"))
import XM


# --------------------------------------------------------------------------- global EDG (fixed) build
def build_global_edg(args):
    """Build the fixed global EDG (nodes + edges, global track ids) and the world-lifted tracks."""
    D = VTE._load_all()
    D["dyn_mask"] = VTE._dilate_dyn_mask(D, args.mask_dilate_px)
    s = VTE._frame_scale(D["rt"], D["T"], no_scale=False)
    cfg = VTE.EdgeConfig(prune_redundant=False)          # keep every track; NMS happens in node selection
    edges, _ = VTE.build_edg_edges(D, s, cfg)
    aux = D["_edg_aux"]
    Q = D["tracks_xyz"].shape[1]
    edg_nodes, edg_edges = VTE.build_edg_graph(
        edges, Q, aux["Ps"], aux["Vs"], aux["Ocount"], aux["thr"],
        near_factor=args.edg_near_factor, overlap_frac=args.edg_overlap_frac,
        edge_hops=args.edg_edge_hops, max_nodes=args.edg_max_nodes)

    # world lift of ALL tracks over ALL frames, IDENTICAL to VTE._valid_at (da3-frame-scale convention)
    T = D["T"]
    Rf = s[:, None, None] * D["R_raw"].astype(np.float64)                     # (T,3,3)
    glob = np.einsum("tij,qtj->qti", Rf, D["tracks_xyz"][0].astype(np.float64)) + D["t_raw"][None, :, :]
    valid = (np.isfinite(glob).all(axis=-1)
             & (D["occluded"][0] == 0)
             & D["dyn_mask"].astype(bool))               # (Q,T)
    print(f"[global-edg] {edg_nodes.size} EDG nodes, {edg_edges.shape[0]} EDG edges, T={T}")
    return D, s, glob, valid, np.asarray(edg_nodes, dtype=int), np.asarray(edg_edges, dtype=int)


# ----------------------------------------------------------------------------- per-segment solver glue
def build_segment_inputs(seg_frames, edg_nodes, edg_edges, glob, valid, K_neighbors):
    """Assemble solver inputs for one segment, or None if the segment cannot be solved.

    Returns a dict with everything solve_qcqp_and_recover_until_part5 needs plus book-keeping
    (rest_frame, frames = frames_anchor_first, present global node ids, local edges)."""
    # rest = the FIRST frame of the segment with >=2 valid EDG nodes (temporal anchor, as in EDG_old).
    # Making rest the earliest frame keeps the smoothness chain (eq.10, consecutive trev) in true time
    # order with no fold.
    node_valid = valid[edg_nodes][:, seg_frames]                 # (Nnode, Lseg)
    per_frame = node_valid.sum(axis=0)
    usable = np.where(per_frame >= 2)[0]
    if usable.size == 0:
        return None                                              # nothing to solve here
    rest_frame = int(seg_frames[int(usable[0])])

    present = edg_nodes[valid[edg_nodes, rest_frame]]            # nodes with a rest position g_i
    if present.size < 2:
        return None
    present = np.asarray(sorted(present.tolist()), dtype=int)
    M = present.size
    edg_idx_map = {int(n): j for j, n in enumerate(present.tolist())}

    # frames_anchor_first: rest (trev=1) then the later segment frames in TEMPORAL order that still see
    # >=1 present node; frames with no present node are dropped (not solved).
    later = [int(f) for f in seg_frames
             if int(f) > rest_frame and bool(valid[present, int(f)].any())]
    frames = np.array([rest_frame] + later, dtype=int)
    T_rev = frames.size
    if T_rev < 2:                                   # no later frame sees a present node -> nothing to solve
        return None
    frame_to_trev = {int(f): i + 1 for i, f in enumerate(frames.tolist())}

    g = glob[present, rest_frame].astype(np.float64)             # (M,3) rest positions

    # skinning: each EDG node is a dynamic keypoint skinned to its K nearest EDG nodes (self incl.).
    tree = cKDTree(g)
    K_eff = int(min(K_neighbors, M))
    kp_neighbors, kp_weights, kp_rest = {}, {}, {}
    for j, n in enumerate(present.tolist()):
        d, nn = tree.query(g[j], k=K_eff)
        d = np.atleast_1d(d); nn = np.atleast_1d(nn)
        if np.any(d <= 1e-8):                                    # self is nearest (dist 0) -> weight 1
            w = np.zeros_like(d); w[int(np.argmin(d))] = 1.0
        else:
            w = 1.0 / (d + 1e-8); w = w / w.sum()
        kp_neighbors[int(n)] = present[nn].astype(int)
        kp_weights[int(n)] = w.astype(np.float64)
        kp_rest[int(n)] = g[j].astype(np.float64)

    # dynamic observations = every (present node, frame) where the node is valid
    dyn_kp_idx, dyn_frame, dyn_global = [], [], []
    for n in present.tolist():
        for f in frames.tolist():
            if valid[n, f]:
                dyn_kp_idx.append(int(n)); dyn_frame.append(int(f)); dyn_global.append(glob[n, f])
    dyn_kp_idx = np.array(dyn_kp_idx, dtype=int)
    dyn_frame = np.array(dyn_frame, dtype=int)
    dyn_global = np.array(dyn_global, dtype=np.float64)

    # edges restricted to present nodes (global ids; solver maps via edg_idx_map)
    edges = np.array([[int(a), int(b)] for a, b in edg_edges
                      if int(a) in edg_idx_map and int(b) in edg_idx_map], dtype=int).reshape(-1, 2)

    return dict(edges=edges, edg_idx_map=edg_idx_map, g=g, edg_obs=None, edg_obs_mask=None,
                N_dyn_obs=int(dyn_kp_idx.size), dyn_kp_idx=dyn_kp_idx, dyn_frame=dyn_frame,
                dyn_global=dyn_global, frame_to_trev=frame_to_trev, kp_neighbors=kp_neighbors,
                kp_weights=kp_weights, kp_rest=kp_rest,
                present=present, frames=frames, rest_frame=rest_frame, M=M, T_rev=T_rev)


def solve_segment(inp, output_dir, w_arap, w_smooth):
    """Run the QCQP solve for one segment. Returns edg_deformed (M,T_rev,3) and R_edg (M,T_rev,3,3)."""
    out = solve_qcqp_and_recover_until_part5(
        edges=inp["edges"], edg_idx_map=inp["edg_idx_map"], g=inp["g"],
        edg_obs=inp["edg_obs"], edg_obs_mask=inp["edg_obs_mask"],
        N_dyn_obs=inp["N_dyn_obs"], dyn_kp_idx=inp["dyn_kp_idx"], dyn_frame=inp["dyn_frame"],
        dyn_global=inp["dyn_global"], frame_to_trev=inp["frame_to_trev"],
        kp_neighbors=inp["kp_neighbors"], kp_weights=inp["kp_weights"], kp_rest=inp["kp_rest"],
        output_dir=str(output_dir), solve_Ronly_fn=XM.solve_Ronly, load_matrix_from_bin_fn=load_matrix_from_bin,
        w_obs=1.0, w_arap=w_arap, w_smooth=w_smooth,
    )
    return np.asarray(out["edg_deformed"], np.float64), np.asarray(out["R_edg"], np.float64)


# ----------------------------------------------------------------------------------------- per-segment viz
def visualize_segment(edg_deformed, inp, glob, valid, node_r, seg_id):
    """Open3D window for ONE solved segment: red = solved EDG (nodes+edges), blue = observations.
    SPACE = next frame, B = prev, Q/ESC = close (continue to next segment)."""
    import open3d as o3d
    present, frames = inp["present"], inp["frames"]
    idx_map = inp["edg_idx_map"]
    edges_local = np.array([[idx_map[int(a)], idx_map[int(b)]] for a, b in inp["edges"]], dtype=np.int32).reshape(-1, 2)

    mesh = o3d.geometry.TriangleMesh()          # solved nodes (red spheres)
    line = o3d.geometry.LineSet()               # EDG edges (red)
    obs_pcd = o3d.geometry.PointCloud()         # observations (blue)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=node_r * 20)
    st = {"i": 0}

    def fill(i):
        P = edg_deformed[:, i, :]                                  # (M,3) solved node positions
        sm = VTE._edg_sphere_mesh(P, node_r, [1.0, 0.05, 0.05])
        mesh.vertices, mesh.triangles = sm.vertices, sm.triangles
        mesh.vertex_colors, mesh.vertex_normals = sm.vertex_colors, sm.vertex_normals
        if edges_local.size:
            line.points = o3d.utility.Vector3dVector(P)
            line.lines = o3d.utility.Vector2iVector(edges_local)
            line.paint_uniform_color([1.0, 0.05, 0.05])
        f = int(frames[i])
        obs_ids = present[valid[present, f]]
        obs_pcd.points = o3d.utility.Vector3dVector(glob[obs_ids, f] if obs_ids.size else np.zeros((0, 3)))
        if obs_ids.size:
            obs_pcd.paint_uniform_color([0.2, 0.4, 1.0])
        tag = " (REST)" if f == inp["rest_frame"] else ""
        print(f"[seg {seg_id}] frame {f}{tag}  trev {i+1}/{len(frames)}  nodes={inp['M']} obs={obs_ids.size}")

    def refresh(vis):
        fill(st["i"])
        for gm in (mesh, line, obs_pcd):
            vis.update_geometry(gm)
        return False

    def nxt(vis): st["i"] = (st["i"] + 1) % len(frames); return refresh(vis)
    def prv(vis): st["i"] = (st["i"] - 1) % len(frames); return refresh(vis)
    def clo(vis): vis.close(); return False

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"EDG segment {seg_id}", width=1280, height=720)
    fill(0)
    for gm in (mesh, line, obs_pcd, axes):
        vis.add_geometry(gm)
    vis.get_render_option().point_size = 6.0
    vis.register_key_callback(ord(" "), nxt)
    vis.register_key_callback(ord("b"), prv); vis.register_key_callback(ord("B"), prv)
    vis.register_key_callback(ord("q"), clo); vis.register_key_callback(ord("Q"), clo); vis.register_key_callback(256, clo)
    print(f"[seg {seg_id}] SPACE=next frame, B=prev, Q/ESC=close (-> next segment)")
    vis.run(); vis.destroy_window()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg-len", type=int, default=10, help="segment length in frames (independent, non-overlapping)")
    ap.add_argument("--k-neighbors", type=int, default=4, help="skinning K (EDG node -> nearest EDG nodes)")
    ap.add_argument("--w-arap", type=float, default=0.2)
    ap.add_argument("--w-smooth", type=float, default=0.2)
    ap.add_argument("--mask-dilate-px", type=int, default=5)
    ap.add_argument("--edg-near-factor", type=float, default=2.0)
    ap.add_argument("--edg-overlap-frac", type=float, default=0.5)
    ap.add_argument("--edg-edge-hops", type=int, default=6)
    ap.add_argument("--edg-max-nodes", type=int, default=1500)
    ap.add_argument("--max-segments", type=int, default=0, help="stop after N segments (0 = all; for testing)")
    ap.add_argument("--no-viz", action="store_true", help="skip the per-segment Open3D viewer")
    args = ap.parse_args()

    D, s, glob, valid, edg_nodes, edg_edges = build_global_edg(args)
    T = D["T"]
    out_dir = Path(utils.definitions.output_dir) / "edg_segments"
    out_dir.mkdir(parents=True, exist_ok=True)

    # node marker radius from the scene scale
    fin = valid[edg_nodes].any(1)
    gp = glob[edg_nodes[fin], :, :][np.isfinite(glob[edg_nodes[fin]]).all(-1)]
    diag = float(np.linalg.norm(gp.max(0) - gp.min(0))) if gp.size else 1.0
    node_r = max(1e-4, 0.006 * diag)

    seg_id, solved = 0, []
    for start in range(0, T, args.seg_len):
        seg_frames = np.arange(start, min(T, start + args.seg_len), dtype=int)
        inp = build_segment_inputs(seg_frames, edg_nodes, edg_edges, glob, valid, args.k_neighbors)
        if inp is None:
            print(f"=== Segment {seg_id} frames [{start},{seg_frames[-1]}]: no valid EDG nodes; skipped ===")
            seg_id += 1
            continue
        print(f"\n=== Segment {seg_id} frames [{start},{seg_frames[-1]}]  rest={inp['rest_frame']}  "
              f"nodes={inp['M']}  solved-frames={inp['T_rev']}  edges={inp['edges'].shape[0]} ===")
        edg_deformed, R_edg = solve_segment(inp, out_dir, args.w_arap, args.w_smooth)
        np.savez_compressed(out_dir / f"segment_{seg_id:03d}.npz",
                            edg_deformed=edg_deformed, R_edg=R_edg,
                            present=inp["present"], frames=inp["frames"], rest_frame=inp["rest_frame"],
                            edges=inp["edges"])
        solved.append(seg_id)
        if not args.no_viz:
            visualize_segment(edg_deformed, inp, glob, valid, node_r, seg_id)
        seg_id += 1
        if args.max_segments and len(solved) >= args.max_segments:
            break

    print(f"\n[done] solved {len(solved)} segments -> {out_dir}/segment_*.npz")


if __name__ == "__main__":
    main()
