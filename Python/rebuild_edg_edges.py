#!/usr/bin/env python3
"""Rebuild the EDG edges from the LOCAL, PARTIALLY-OBSERVED table (edg_test/edg_table.npz) -- instead of
pruning long edges after the solve.

Pairwise metric (Notes/edg_table_completion(1).pdf sec 7.1, the partial-observation analogue of MoSca's
trajectory-distance topology):
    C_mn = frames where BOTH nodes are filled in the partial table
    D_mn = the --pct (default 90) percentile of ||c_m,t - c_n,t|| over t in C_mn   ("top-90% max distance")
so two nodes are close only if they stay close over their whole co-observed history -- nodes that are only
TEMPORARILY close never get an edge. Pairs with < --min-covis co-filled frames get D = inf (never connected).
Edges = union of each node's --knn nearest neighbours under D; optionally also capped at
--max-factor x median edge D.

Saves outputs/edg_edges_rebuilt.npz (edg_nodes, edg_edges [E,2] global ids, edge_dist [E]).
Use the rebuilt edges in the global completion:
    conda run -n tapnet python complete_edg_table.py --merged --edges-npz <outputs>/edg_edges_rebuilt.npz

Visualize (Open3D; SPACE/B step frames, Q quit): red = EDG nodes, grey = rebuilt edges, purple = observed
dynamic points (context). Default draws on the PARTIAL table (a node/edge shows only at frames it is
filled); --use-completed draws on the dense completed table instead.

Run (numpy/scipy/open3d only):  XM_DATASET_NAME=<scene> conda run -n XM python rebuild_edg_edges.py --merged
"""
import argparse
from pathlib import Path

import numpy as np

import utils.definitions
from utils.edg import scatter_solved_poses, scatter_solved_frame_vector


def build_distance(table, filled, pct, min_covis):
    """D[m,n] = pct-percentile of the per-frame distance over co-filled frames; inf if < min_covis such
    frames (or none). Diagonal = inf."""
    M = table.shape[0]
    D = np.full((M, M), np.inf)
    for m in range(M):
        both = filled[m][None, :] & filled                            # [M,T]
        if not both.any():
            continue
        d = np.linalg.norm(table[m][None, :, :] - table, axis=-1)     # [M,T] (NaN at holes)
        d = np.where(both, d, np.nan)
        ok = both.sum(1) >= min_covis
        ok[m] = False
        if ok.any():
            D[m, ok] = np.nanpercentile(d[ok], pct, axis=1)
    return D


def build_edges(D, knn, max_factor):
    """Union of each node's knn nearest neighbours under D (finite only); optional length cap."""
    M = D.shape[0]
    edges = {}
    for m in range(M):
        nb = np.argsort(D[m])[:knn]
        for n in nb.tolist():
            if np.isfinite(D[m, n]):
                key = (min(m, n), max(m, n))
                edges[key] = D[m, n]
    e = np.array(sorted(edges.keys()), dtype=int).reshape(-1, 2)
    dist = np.array([edges[tuple(k)] for k in e.tolist()])
    if max_factor > 0 and e.size:
        thr = max_factor * np.median(dist)
        keep = dist <= thr
        print(f"[cap] dropping {int((~keep).sum())}/{e.shape[0]} edges with D > {max_factor:.1f}x median ({thr:.4f})")
        e, dist = e[keep], dist[keep]
    return e, dist


def load_dyn_context(merged, T):
    """Observed dynamic world points per frame (purple context in the viewer)."""
    out_dir = Path(utils.definitions.output_dir)
    tracks_path = (out_dir / "tracks_dynamic_merged.npz") if merged else Path(utils.definitions.tracks_dynamics_path)
    data = np.load(str(tracks_path), allow_pickle=True)
    rt = np.load(str(out_dir / "result_XM.npz"))
    R_raw, t_raw = scatter_solved_poses(rt, T)
    s_raw = scatter_solved_frame_vector(rt, "s", T, default=1.0)
    R = R_raw.astype(np.float64) * (s_raw[:, None, None] if s_raw is not None else 1.0)
    glob = np.einsum("tij,qtj->qti", R, data["tracks_xyz"][0].astype(np.float64)) + t_raw[None, :, :]
    valid = np.isfinite(glob).all(-1) & (data["occluded"][0] == 0) & data["dynamic_obs_mask"].astype(bool)
    return glob, valid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pct", type=float, default=90.0, help="percentile of co-observed distance (top-90%% max)")
    ap.add_argument("--min-covis", type=int, default=10, help="min co-filled frames for a pair to be connectable")
    ap.add_argument("--knn", type=int, default=8, help="connect each node to its k nearest under D")
    ap.add_argument("--max-factor", type=float, default=0.0,
                    help="also drop edges with D > factor x median edge D (0 = off; kNN already avoids far pairs)")
    ap.add_argument("--merged", action="store_true", help="dynamic-context tracks = the merged set")
    ap.add_argument("--use-completed", action="store_true",
                    help="draw on the DENSE completed table (edg_completed.npz) instead of the partial table")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--point-size", type=float, default=5.0)
    ap.add_argument("--no-viz", action="store_true", help="compute + save only")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir = Path(utils.definitions.output_dir)
    tbl = np.load(out_dir / "edg_test" / "edg_table.npz")
    node_ids = np.asarray(tbl["node_ids"], dtype=int)
    table = np.asarray(tbl["table"], dtype=np.float64)                 # [M,T,3] NaN=hole
    filled = np.asarray(tbl["filled"], dtype=bool)
    M, T = filled.shape
    print(f"[load] {M} nodes, T={T}, partial table {100*filled.mean():.1f}% filled")

    D = build_distance(table, filled, args.pct, args.min_covis)
    n_conn = np.isfinite(D).sum(1)
    print(f"[dist] connectable neighbours per node: median={int(np.median(n_conn))} "
          f"min={int(n_conn.min())} (pairs need >= {args.min_covis} co-filled frames)")
    edges, dist = build_edges(D, args.knn, args.max_factor)
    deg = np.bincount(edges.reshape(-1), minlength=M)
    print(f"[edges] {edges.shape[0]} edges (knn={args.knn}) | D: median={np.median(dist):.4f} "
          f"p90={np.percentile(dist, 90):.4f} max={dist.max():.4f} | degree: median={int(np.median(deg))} "
          f"min={int(deg.min())} max={int(deg.max())} | isolated nodes: {int((deg == 0).sum())}")

    out = out_dir / "edg_edges_rebuilt.npz"
    np.savez_compressed(out, edg_nodes=node_ids, edg_edges=node_ids[edges], edge_dist=dist,
                        pct=args.pct, min_covis=args.min_covis, knn=args.knn)
    print(f"[done] {out}  (global-id pairs; feed to complete_edg_table.py --edges-npz {out})")

    if args.no_viz:
        return

    # ---------------------------------------------------------------- Open3D: nodes red, edges grey, dyn purple
    pos = table
    if args.use_completed:
        comp = out_dir / "edg_test" / "edg_completed.npz"
        z = np.load(comp)
        keep = {int(g): i for i, g in enumerate(np.asarray(z["node_ids"], dtype=int).tolist())}
        pos = np.full((M, z["edg_reconstruct"].shape[1], 3), np.nan)
        for i, g in enumerate(node_ids.tolist()):                     # completed may have trimmed some nodes
            if int(g) in keep:
                pos[i] = z["edg_reconstruct"][keep[int(g)]]
        print(f"[viz] drawing on the completed table ({comp.name})")
    glob, valid = load_dyn_context(args.merged, T)

    import open3d as o3d
    state = {"frame": int(np.clip(args.frame, 0, T - 1))}

    def make_geoms(f):
        geoms = []
        p = pos[:, f]
        fin = np.isfinite(p).all(1)
        idx = np.full(M, -1, dtype=int)
        idx[fin] = np.arange(int(fin.sum()))
        if fin.any():
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(p[fin])
            pcd.paint_uniform_color([0.9, 0.1, 0.1])
            geoms.append(("nodes", pcd))
            lines = [[idx[a], idx[b]] for a, b in edges.tolist() if fin[a] and fin[b]]
            if lines:
                ls = o3d.geometry.LineSet()
                ls.points = o3d.utility.Vector3dVector(p[fin])
                ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
                ls.paint_uniform_color([0.55, 0.55, 0.55])
                geoms.append(("edges", ls))
        dyn = glob[valid[:, f], f]
        if dyn.size:
            pd = o3d.geometry.PointCloud()
            pd.points = o3d.utility.Vector3dVector(dyn)
            pd.paint_uniform_color([0.6, 0.3, 0.8])
            geoms.append(("dyn", pd))
        print(f"[viz] frame={f} nodes={int(fin.sum())}/{M} edges_drawn="
              f"{len(geoms) > 1 and np.asarray(geoms[1][1].lines).shape[0] or 0} dyn={dyn.shape[0]}")
        return geoms

    named = make_geoms(state["frame"])
    if args.dry_run:
        print("[dry-run] built:", ", ".join(n for n, _ in named))
        return
    gm = {n: g for n, g in named}
    empty = o3d.utility.Vector3dVector(np.zeros((0, 3)))

    def refresh(vis):
        new = {n: g for n, g in make_geoms(state["frame"])}
        for n, g in new.items():
            if n not in gm:
                vis.add_geometry(g); gm[n] = g
        for n, g in list(gm.items()):
            ng = new.get(n)
            if isinstance(g, o3d.geometry.PointCloud):
                g.points = ng.points if ng is not None else empty
                if ng is not None:
                    g.colors = ng.colors
            elif isinstance(g, o3d.geometry.LineSet):
                if ng is not None:
                    g.points, g.lines = ng.points, ng.lines
                else:
                    g.points = empty
                    g.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
            vis.update_geometry(g)
        return False

    def nxt(vis):
        state["frame"] = (state["frame"] + 1) % T
        return refresh(vis)

    def prv(vis):
        state["frame"] = (state["frame"] - 1) % T
        return refresh(vis)

    print("SPACE=next frame, B=prev, Q/ESC=quit. red=EDG nodes, grey=rebuilt edges, purple=observed dyn points")
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Rebuilt EDG edges", width=1280, height=720)
    for _, g in named:
        vis.add_geometry(g)
    vis.get_render_option().point_size = float(args.point_size)
    vis.register_key_callback(ord(" "), nxt)
    for k in "bB":
        vis.register_key_callback(ord(k), prv)
    for k in (ord("q"), ord("Q"), 256):
        vis.register_key_callback(k, lambda v: (v.close(), False)[1])
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
