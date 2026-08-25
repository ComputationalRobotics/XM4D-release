#!/usr/bin/env python3
"""PRE-SOLVE EDG topology viewer: show the EDG nodes + edges (and optional dynamic->EDG orange links)
straight from build_edg's edg_topology.npz, on the WORLD-LIFTED track positions -- BEFORE EDG_test_fast
solving. Uses build_edg's own loaders (_load_all/_frame_scale/_valid_at) so the coordinate space matches
build_edg exactly. This is the view to judge the build_edg node-node edges / dyn->EDG links themselves.

Run (needs a display):
  XM_DATASET_CAT=Dycheck XM_DATASET_NAME=wheel conda run --no-capture-output -n XM \
    python visualize_edg_topology_open3d.py --merged --edge-radius 0.005 --node-radius 0.006
  add --frame F to lift at one frame (default: per-node median over frames), --show-dyn for orange links,
  --dry-run to build geometry + print counts without opening a window (headless check).
"""
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

import utils.definitions
import build_edg as BE
from visualize_edg_filtered_open3d import _edges_cylinders, _spheres_mesh


def _node_positions(D, s, frame):
    """(Q,3) world-lifted positions + (Q,) valid. frame=None -> per-track MEDIAN over sampled valid
    frames (a stable canonical position, no per-frame NaN gaps)."""
    if frame is not None:
        return BE._valid_at(D, s, int(frame))
    T = int(D["T"]); Q = D["tracks_xyz"].shape[1]
    frames = np.unique(np.linspace(0, T - 1, min(T, 40)).astype(int))
    acc = np.full((frames.size, Q, 3), np.nan)
    for i, f in enumerate(frames):
        P, valid = BE._valid_at(D, s, int(f))
        acc[i][valid] = P[valid]
    import warnings
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # all-NaN tracks -> excluded by the valid mask
        Pmed = np.nanmedian(acc, axis=0)
    return Pmed, np.isfinite(Pmed).all(1)


def _jet(v):
    """v in [0,1] -> (r,g,b) 0..1 jet colormap: blue=low, cyan/green/yellow=mid, red=high."""
    v = float(np.clip(v, 0.0, 1.0))
    return [float(np.clip(1.5 - abs(4 * v - 3), 0, 1)),
            float(np.clip(1.5 - abs(4 * v - 2), 0, 1)),
            float(np.clip(1.5 - abs(4 * v - 1), 0, 1))]


def _node_trajectories(D, s, nodes):
    """(T, Nn, 3) world positions + (T, Nn) valid for the node subset over ALL frames."""
    T = int(D["T"])
    traj = np.full((T, nodes.size, 3), np.nan); tval = np.zeros((T, nodes.size), bool)
    for f in range(T):
        P, v = BE._valid_at(D, s, f)
        traj[f] = P[nodes]; tval[f] = v[nodes]
    return traj, tval


def _edge_metric(traj, tval, line_idx, kind, min_ov):
    """Per-edge scalar over co-visible frames. kind: maxdist|meandist|range|cov. NaN if < min_ov frames."""
    vals = np.full(len(line_idx), np.nan)
    for e, (a, b) in enumerate(line_idx):
        co = tval[:, a] & tval[:, b]
        if int(co.sum()) < min_ov:
            continue
        d = np.linalg.norm(traj[co, a] - traj[co, b], axis=1)
        vals[e] = (d.max() if kind == "maxdist" else d.min() if kind == "mindist"
                   else d.mean() if kind == "meandist" else d.max() - d.min() if kind == "range"
                   else d.max() / (d.min() + 1e-9) if kind == "maxmin"
                   else d.std() / (d.mean() + 1e-9))
    return vals


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topology", type=str,
                    default=str(Path(utils.definitions.output_dir) / "edg_topology.npz"))
    ap.add_argument("--merged", action="store_true", help="load tracks_dynamic_merged.npz (build_edg --merged)")
    ap.add_argument("--frame", type=int, default=None, help="lift nodes at this frame (default: median over frames)")
    ap.add_argument("--no-scale", action="store_true", help="ignore da3_frame_scale in the lift")
    ap.add_argument("--edge-radius", type=float, default=0.0, help="EDG edges as cylinders of this radius (else thin lines)")
    ap.add_argument("--color-by", choices=["none", "maxdist", "mindist", "meandist", "range", "maxmin", "cov"],
                    default="none",
                    help="colour EDG edges by a per-edge metric over co-visible frames (jet: blue=low, red=high). "
                         "maxdist=MAX 3D dist; mindist=MIN; meandist=MEAN; range=max-min; maxmin=max/min RATIO "
                         "(high=distance varies a lot=likely cross-body); cov=std/mean (rigidity variation).")
    ap.add_argument("--color-min-ov", type=int, default=5, help="min co-visible frames for an edge to get a colour (else grey)")
    ap.add_argument("--color-hi-pct", type=float, default=95.0, help="percentile of the metric mapped to RED (top of the scale)")
    ap.add_argument("--color-lo-pct", type=float, default=2.0, help="percentile of the metric mapped to BLUE (bottom of the scale)")
    ap.add_argument("--color-log", action="store_true", help="LOG-scale the metric before colouring (spreads heavy-tailed metrics like maxmin/range so the gradient is visible)")
    ap.add_argument("--node-radius", type=float, default=0.0, help="nodes as spheres of this radius (else points)")
    ap.add_argument("--point-size", type=float, default=8.0)
    ap.add_argument("--show-dyn", action="store_true", help="also show dynamic points (grey) + dyn->EDG links (orange)")
    ap.add_argument("--dyn-frac", type=float, default=0.15, help="fraction of dynamic points to show (--show-dyn)")
    ap.add_argument("--dry-run", action="store_true", help="build geometry + print counts, do not open a window")
    args = ap.parse_args()

    topo = np.load(args.topology, allow_pickle=True)
    nodes = np.asarray(topo["edg_nodes"], np.int64).reshape(-1)
    edg_edges = np.asarray(topo["edg_edges"], np.int64).reshape(-1, 2)
    dyn_edges = (np.asarray(topo["dyn_edg_edges"], np.int64).reshape(-1, 2)
                 if "dyn_edg_edges" in topo.files else np.zeros((0, 2), np.int64))

    tracks_path = str(Path(utils.definitions.output_dir) / "tracks_dynamic_merged.npz") if args.merged else None
    D = BE._load_all(tracks_path)
    s = BE._frame_scale(D["rt"], D["T"], args.no_scale)
    P, valid = _node_positions(D, s, args.frame)
    P_safe = np.nan_to_num(P)                                   # finite points array for LineSet/PointCloud

    loc = {int(g): i for i, g in enumerate(nodes)}
    node_xyz = P[nodes]
    node_ok = valid[nodes]
    line_idx = [[loc[a], loc[b]] for a, b in edg_edges.tolist()
                if a in loc and b in loc and node_ok[loc[a]] and node_ok[loc[b]]]

    print(f"[edg-topology] frame={'median' if args.frame is None else args.frame}: "
          f"{int(node_ok.sum())}/{nodes.size} nodes valid, {len(line_idx)}/{edg_edges.shape[0]} edges drawn")

    geoms = []
    if args.node_radius > 0:
        geoms.append(_spheres_mesh(node_xyz[node_ok], args.node_radius, [0.9, 0.1, 0.1]))
    else:
        pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(np.nan_to_num(node_xyz[node_ok]))
        pc.paint_uniform_color([0.9, 0.1, 0.1]); geoms.append(pc)

    node_xyz_safe = np.nan_to_num(node_xyz)
    edge_colors = None
    if args.color_by != "none" and line_idx:
        traj, tval = _node_trajectories(D, s, nodes)
        mvals = _edge_metric(traj, tval, line_idx, args.color_by, args.color_min_ov)
        fin = np.isfinite(mvals)
        if fin.any():
            vv = np.where(fin, np.log(np.maximum(mvals, 1e-9)), np.nan) if args.color_log else mvals
            lo, hi = float(np.percentile(vv[fin], args.color_lo_pct)), float(np.percentile(vv[fin], args.color_hi_pct))
            rng = max(hi - lo, 1e-9)
            edge_colors = [_jet((vv[e] - lo) / rng) if np.isfinite(vv[e]) else [0.55, 0.55, 0.55]
                           for e in range(len(line_idx))]
            sc = "log-scale" if args.color_log else "linear"
            blue_v = float(np.exp(lo)) if args.color_log else lo
            red_v = float(np.exp(hi)) if args.color_log else hi
            print(f"[color-by {args.color_by}] ({sc}) {int(fin.sum())} coloured edges: min {mvals[fin].min():.4f} "
                  f"median {np.median(mvals[fin]):.4f} max {mvals[fin].max():.4f}  "
                  f"(scale: {blue_v:.4f}=blue .. {red_v:.4f}=red; {int((~fin).sum())} grey edges <{args.color_min_ov} co-vis frames)")
    if line_idx:
        if args.edge_radius > 0:
            cyl = _edges_cylinders(node_xyz_safe, line_idx, args.edge_radius, [0.72, 0.72, 0.72], colors=edge_colors)
            if cyl is not None:
                geoms.append(cyl)
        else:
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(node_xyz_safe)
            ls.lines = o3d.utility.Vector2iVector(np.asarray(line_idx, np.int32))
            if edge_colors is not None:
                ls.colors = o3d.utility.Vector3dVector(np.asarray(edge_colors, np.float64))
            else:
                ls.paint_uniform_color([0.72, 0.72, 0.72])
            geoms.append(ls)

    if args.show_dyn and dyn_edges.size:
        dyn_ids = np.unique(dyn_edges[:, 0])
        show = set(dyn_ids[np.random.default_rng(0).random(dyn_ids.size) < float(args.dyn_frac)].tolist())
        opts, oidx, dpts = [], [], []
        for a, b in dyn_edges.tolist():
            if a not in show or b not in loc or not valid[a] or not node_ok[loc[b]]:
                continue
            opts.append(P_safe[a]); opts.append(P[nodes[loc[b]]])
            oidx.append([len(opts) - 2, len(opts) - 1]); dpts.append(P_safe[a])
        if dpts:
            dp = o3d.geometry.PointCloud(); dp.points = o3d.utility.Vector3dVector(np.asarray(dpts))
            dp.paint_uniform_color([0.5, 0.5, 0.5]); geoms.append(dp)
            ol = o3d.geometry.LineSet(); ol.points = o3d.utility.Vector3dVector(np.asarray(opts))
            ol.lines = o3d.utility.Vector2iVector(np.asarray(oidx, np.int32))
            ol.paint_uniform_color([1.0, 0.5, 0.0]); geoms.append(ol)
        print(f"[edg-topology] dyn->EDG: {len(oidx)} orange links from {len(dpts)} shown dynamic points")

    if args.dry_run:
        print(f"[dry-run] built {len(geoms)} geometries; not opening a window.")
        return
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="EDG topology (pre-solve)", width=1280, height=720)
    for g in geoms:
        vis.add_geometry(g)
    vis.get_render_option().point_size = float(args.point_size)
    vis.run(); vis.destroy_window()


if __name__ == "__main__":
    main()
