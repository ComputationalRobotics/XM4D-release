"""Helpers of step 4b (edg_topology.py): load the moving points' tracks and the cameras, lift the tracks to world 3D,
and link each moving point to the graph nodes that stay close to it (build_dyn_to_edg_edges)."""
from concurrent.futures import ThreadPoolExecutor
import glob
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import utils.definitions
from utils.edg import scatter_solved_poses, scatter_solved_frame_vector


def _load_all(tracks_path):
    root = Path(utils.definitions.rootpath)
    data = np.load(tracks_path, allow_pickle=True)
    tracks_xyz = data["tracks_xyz"]          # (1, Q, T, 3)
    tracks_xy = data["tracks_xy"]            # (1, Q, T, 2)  image px
    occluded = data["occluded"]              # (1, Q, T)
    dyn_mask = data["dynamic_obs_mask"]      # (Q, T)
    T = tracks_xyz.shape[2]
    rt = np.load(str(root / "outputs" / "result_XM.npz"))
    R_raw, t_raw = scatter_solved_poses(rt, T)
    return dict(tracks_xyz=tracks_xyz, tracks_xy=tracks_xy, occluded=occluded,
                dyn_mask=dyn_mask, T=T, rt=rt, R_raw=R_raw, t_raw=t_raw,
                image_paths=sorted(glob.glob(str(Path(utils.definitions.video_dir) /
                                                 utils.definitions.image_pattern))))


def _dilate_dyn_mask(D, dilate_px):
    """Grow the per-track dynamic reliability mask by dilating the 2D SAM masks `dilate_px` px
    (same load_dynamic_masks + dilate_dynamic_masks utilities as static_tracking_online.py), then
    re-testing each track's image position against the enlarged region. Returns a (Q,T) bool mask
    that is a superset of the original dynamic_obs_mask, so tracks within `dilate_px` of the moving
    object's boundary also count as reliable.
    (Temporal filtering such as move-delete lives in `occluded`, not here, so it is preserved.)"""
    import cv2
    from utils.masks import load_dynamic_masks, dilate_dynamic_masks
    mask_dir = utils.definitions.mask_dir
    H, W = cv2.imread(D["image_paths"][0]).shape[:2]
    masks = load_dynamic_masks(str(mask_dir), utils.definitions.mask_pattern,
                               t_expected=D["T"], mask_mode="dynamic_positive", target_size=(W, H))  # (T,H,W) bool
    D["dyn_masks"] = masks                    # loaded once: _dyn_region_near_px reuses them
    masks = dilate_dynamic_masks(masks, int(dilate_px))
    xy = D["tracks_xy"][0]                    # (Q, T, 2) native px
    dyn = np.asarray(D["dyn_mask"]).astype(bool).copy()  # start from original -> only grows

    def one(t):                               # one frame (its own column of dyn): frames in threads (speed)
        m = masks[t]
        finite = np.isfinite(xy[:, t, 0]) & np.isfinite(xy[:, t, 1])
        x = np.round(np.where(finite, xy[:, t, 0], -1.0)).astype(np.int64)
        y = np.round(np.where(finite, xy[:, t, 1], -1.0)).astype(np.int64)
        ok = finite & (x >= 0) & (x < W) & (y >= 0) & (y < H)
        idx = np.where(ok)[0]
        dyn[idx, t] |= m[y[idx], x[idx]].astype(bool)
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(one, range(D["T"])))
    print(f"[dyn-mask] dilated dynamic masks by {dilate_px}px: dynamic obs "
          f"{int(np.asarray(D['dyn_mask']).sum())} -> {int(dyn.sum())}")
    return dyn


def _dyn_region_near_px(D, frac, fallback=15.0):
    """ADAPTIVE EDG node-selection radius: frac * sqrt(median dynamic-mask area px^2), median over
    every 10th frame with any dynamic pixels. frac 0.16 reproduces the tuned 15px on paper-windmill
    (dyn-region scale ~94px), so larger dynamic regions get proportionally larger deletion radii
    (constant relative node density). Falls back to `fallback` px when the masks are empty. Uses the masks
    _dilate_dyn_mask loaded."""
    import cv2
    H, W = cv2.imread(D["image_paths"][0]).shape[:2]
    masks = D["dyn_masks"]
    areas = masks[::10].reshape(-1, H * W).sum(1)
    areas = areas[areas > 0]
    if areas.size == 0:
        print(f"[edg-near] empty dynamic masks; using fixed {fallback}px")
        return float(fallback)
    scale = float(np.sqrt(np.median(areas)))
    px = max(3.0, float(frac) * scale)
    print(f"[edg-near] ADAPTIVE node-selection radius: dyn-region scale {scale:.1f}px "
          f"(sqrt median mask area) x frac {frac:g} -> near_px {px:.1f}")
    return px


def _frame_scale(rt, T):
    s = scatter_solved_frame_vector(rt, "da3_frame_scale", T, default=1.0)
    return np.where(np.isfinite(s) & (s > 0), s, 1.0)


def _valid_at(D, s, frame):
    """world-lift (Q,3) + reliability mask (Q,) for ALL tracks at `frame`."""
    Rf = float(s[frame]) * D["R_raw"][frame].astype(np.float64)
    P = (Rf @ D["tracks_xyz"][0, :, frame, :].astype(np.float64).T).T + D["t_raw"][frame][None, :]
    valid = (
        np.isfinite(P).all(axis=1)
        & (D["occluded"][0, :, frame] == 0)
        & D["dyn_mask"][:, frame].astype(bool)
    )
    return P, valid


def _stack_sampled(D, s, sampled):
    """Return Ps (S,Q,3) lifted positions and Vs (S,Q) reliability over the sampled frames."""
    Ps = np.empty((sampled.size, D["tracks_xyz"].shape[1], 3), dtype=np.float64)
    Vs = np.empty((sampled.size, D["tracks_xyz"].shape[1]), dtype=bool)

    def one(k):
        Ps[k], Vs[k] = _valid_at(D, s, int(sampled[k]))
    with ThreadPoolExecutor(8) as ex:                         # frames in threads (speed)
        list(ex.map(one, range(sampled.size)))
    return Ps, Vs


def _auto_edge_threshold(Ps, Vs):
    """Robust default: ~2.5x the median nearest-neighbour distance across a few sampled frames."""
    nn = []
    for k in range(min(5, Ps.shape[0])):
        idx = np.where(Vs[k])[0]
        if idx.size < 5:
            continue
        dd, _ = cKDTree(Ps[k, idx]).query(Ps[k, idx], k=2)
        nn.append(dd[:, 1])
    if not nn:
        return 0.1
    return float(5.0 * np.median(np.concatenate(nn)))


def _frame_trees_2d(xy, Vs, T):
    """Per-frame KD-tree over the reliable tracks' 2D image positions (for candidate gathering)."""
    idx_per = [np.where(Vs[t])[0] for t in range(T)]
    with ThreadPoolExecutor(8) as ex:                         # the builds release the GIL (speed)
        tree_per = list(ex.map(lambda t: cKDTree(xy[idx_per[t], t]) if idx_per[t].size else None, range(T)))
    return idx_per, tree_per


def _nanquantile_fast(a, q, axis=0):
    """np.nanquantile(a, q, axis) with the default linear method, vectorised (speed only: numpy loops over
    columns in Python when NaNs are present). Same interpolation (numpy's _lerp); NaN where a slice has no finite value."""
    a = np.moveaxis(np.asarray(a, np.float64), axis, 0)
    fin = ~np.isnan(a); n = fin.sum(0)
    s = np.sort(np.where(fin, a, np.inf), axis=0)
    v = q * (np.maximum(n, 1) - 1); lo = np.floor(v).astype(np.int64); hi = np.minimum(lo + 1, np.maximum(n - 1, 0)); g = v - lo
    A = np.take_along_axis(s, lo[None], 0)[0]; B = np.take_along_axis(s, hi[None], 0)[0]; d = B - A
    with np.errstate(invalid="ignore"):
        out = np.where(g >= 0.5, B - d * (1 - g), A + d * g)
    return np.where(n > 0, out, np.nan)


def build_dyn_to_edg_edges(Q, edg_nodes, Ps, Vs, Ocount, thr, min_conn=2, max_factor=3.0, quantile=90,
                           max_dist_factor=3.0, min_max_ratio=0.5, cand_near=0):
    """Attach each DYNAMIC point (track) to its `min_conn` EDG nodes, chosen so the
    connection stays SPATIALLY CLOSE ACROSS ALL co-visible frames. The skinning weight is set once and
    used for every frame (global), so a node that is near at one frame but far at others would corrupt
    the warp -- such nodes are rejected. Selection:
      * candidate nodes = the cand_near nodes nearest to the point at up to 5 of its frames (every node when there
        are no more than cand_near);
      * among candidates, rank by ROBUST per-frame 3D distance = p<quantile> of the point<->node distance
        over their co-visible frames, and keep the min_conn smallest whose robust distance is <=
        max_factor * node spacing.
      * TWO HARD stability filters reject a link outright (no matter how small its p90):
          (1) max_dist_factor: its MAXIMUM co-visible distance must be <= max_dist_factor * node spacing
              (rejects a link that is far in even a few frames, not just on average);
          (2) min_max_ratio: the distance must not VARY a lot -- min co-visible distance >= min_max_ratio *
              max (applied only when the point ever gets far, dmax > spacing), rejecting a point that
              slides toward/away from the node (non-rigid).
        A point with candidates passing (1)&(2) but none under the p90 cap keeps its single closest such
        node (still skinnable); a point whose every candidate fails (1) or (2) gets NO link (dropped).
        Tracks that ARE EDG nodes are excluded."""
    from collections import defaultdict
    if edg_nodes.size == 0:
        return np.zeros((0, 2), np.int64)
    node_mask = np.zeros(Q, bool); node_mask[edg_nodes] = True
    dyn_ids = np.where((Ocount > 0) & ~node_mask)[0]         # dynamic points (not EDG nodes)
    if dyn_ids.size == 0:
        return np.zeros((0, 2), np.int64)
    K = edg_nodes.size

    # --- candidate pool ---
    cand = defaultdict(set)                                   # dyn track id -> set of node LOCAL indices
    if cand_near and cand_near < K:                           # the cand_near
        nodes_P = Ps[:, edg_nodes]                           # nodes nearest at up to 5 of the point's frames (min distance)
        nodes_V = Vs[:, edg_nodes]

        def _near(c0):                                        # one block of points (numpy only: blocks in threads)
            qs = dyn_ids[c0:c0 + 2048]; dmin = np.full((qs.size, K), np.inf)
            vq = Vs[:, qs].T; n = vq.sum(1); m = np.minimum(5, n)          # frame j of point i: its seen frame number
            fr = np.nonzero(vq)[1]; s0 = np.cumsum(n) - n                    # np.linspace(0, n - 1, m).astype(int)[j], same float ops
            step = (n - 1).astype(np.float64) / np.maximum(m - 1, 1)
            for j in range(5):
                ii = np.where(m > j)[0]
                if not ii.size:
                    break
                tj = fr[s0[ii] + np.where(j == m[ii] - 1, n[ii] - 1, (j * step[ii]).astype(int))]
                v = nodes_P[tj] - Ps[tj, qs[ii]][:, None, :]
                d = np.sqrt((v[..., 0] * v[..., 0] + v[..., 1] * v[..., 1]) + v[..., 2] * v[..., 2])   # (n, K) = np.linalg.norm
                d = np.where(nodes_V[tj], d, np.inf); dmin[ii] = np.minimum(dmin[ii], d)
            near = np.argsort(dmin, axis=1)[:, :int(cand_near)]
            return qs, near, np.isfinite(np.take_along_axis(dmin, near, 1))
        with ThreadPoolExecutor(8) as ex:
            for qs, near, fin in ex.map(_near, range(0, dyn_ids.size, 2048)):
                for q, nr, f in zip(qs.tolist(), near.tolist(), fin.tolist()):
                    cand[q] = {nl for nl, ok in zip(nr, f) if ok}
    else:                                                    # every node is a candidate
        for q in dyn_ids.tolist():
            cand[q] = set(range(K))

    # --- robust p<quantile> + MIN + MAX per-frame distance for each candidate pair, batched by node ---
    node_pts = defaultdict(list)
    for q, nls in cand.items():
        for nl in nls:
            node_pts[nl].append(q)

    def _node_stats(item):                                    # one node; distances only at the co-visible frames (speed)
        nl, pts = item
        pts = np.asarray(pts, np.int64)
        n_gid = int(edg_nodes[nl])
        tn = np.flatnonzero(Vs[:, n_gid])
        if not tn.size:                                       # a node never seen: no distance, as the NaN rows before
            nan = np.full(pts.size, np.nan)
            return pts, np.full(pts.size, n_gid), nan, nan, nan
        ti, pi = np.nonzero(Vs[tn][:, pts])                  # co-visible (frame, point)
        v = Ps[tn[ti], pts[pi]] - Ps[tn[ti], n_gid]
        d = np.full((pts.size, tn.size), np.nan)             # (n_pts, T_node), NaN where not co-visible
        d[pi, ti] = np.sqrt((v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1]) + v[:, 2] * v[:, 2])   # = np.linalg.norm
        with np.errstate(all="ignore"):
            rob = _nanquantile_fast(d, quantile / 100.0, axis=1)   # (n_pts,) robust over co-visible frames (= np.nanpercentile)
            dmn = np.nanmin(d, axis=1)
            dmx = np.nanmax(d, axis=1)
        return pts, np.full(pts.size, n_gid), rob, dmn, dmx

    with ThreadPoolExecutor(8) as ex:                         # nodes in threads, results in node order
        res = list(ex.map(_node_stats, node_pts.items()))
    # every candidate link with a co-visible frame as flat arrays (point, node gid, robust, dmin, dmax), grouped by point and
    # ranked nearest first; ties keep the node order, like the stable sort of per-point dicts this replaces (speed)
    lq, lg, rob, dmn, dmx = (np.concatenate(x) for x in zip(*res)) if res else [np.zeros(0)] * 5
    o = np.isfinite(rob); lq, lg, rob, dmn, dmx = lq[o], lg[o], rob[o], dmn[o], dmx[o]
    if not lq.size:
        return np.zeros((0, 2), np.int64)
    o = np.lexsort((rob, lq)); lq, lg, rob, dmn, dmx = lq[o], lg[o], rob[o], dmn[o], dmx[o]
    first = np.r_[True, lq[1:] != lq[:-1]]                   # each point's nearest link
    grp = np.cumsum(first) - 1

    # --- adaptive cap + HARD stability filters. "far" is relative to node spacing (scale = median
    # nearest-node robust distance). A link needs (a) p90 <= max_factor*scale (soft cap, has a skinnable
    # fallback), AND (b) MAX co-visible distance <= max_dist_factor*scale, AND (c) it must not vary a lot
    # (min >= min_max_ratio*max, applied only when the point ever gets far, dmax > scale). (b),(c) are HARD:
    # a point whose every candidate fails them gets NO link. ---
    nearest = rob[first]
    scale = float(np.median(nearest)) if nearest.size else float(thr)
    cap = float(max_factor) * scale
    max_cap = float(max_dist_factor) * scale

    far = (dmx > max_cap) if max_dist_factor > 0 else np.zeros(rob.size, bool)
    vary = (~far & (dmx > scale) & (dmn < float(min_max_ratio) * dmx)) if min_max_ratio > 0 else np.zeros(rob.size, bool)
    stable = ~far & ~vary
    under = stable & (rob <= cap)

    def _rank(m):                                             # rank of each True entry among its point's True entries
        c = np.cumsum(m)
        return c - 1 - (c - m)[first][grp]
    has_under = np.bincount(grp, under)[grp] > 0
    keep = np.where(has_under, under & (_rank(under) < int(min_conn)),   # the min_conn nearest under the p90 cap, else
                    stable & (_rank(stable) == 0))                       # skinnable fallback: closest that passes (b)&(c)

    if not keep.any():
        return np.zeros((0, 2), np.int64)
    und = np.unique(np.stack([lq[keep], lg[keep]], axis=1), axis=0)   # (E,2) = (dynamic track, edg node)
    return und.astype(np.int64)

