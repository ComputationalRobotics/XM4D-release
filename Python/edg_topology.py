"""Step 4b: the embedded deformation graph (EDG) of the moving objects, from their tracks alone: which tracks become graph
nodes, which nodes are joined, and which nodes carry each moving point. Reads the smoothed tracks (smooth_dynamic_tracks.py;
the raw ones otherwise) and the cameras; writes outputs/edg_topology.npz for EDG_test_fast.py and complete_edg_table.py.

Nodes (longest tracks first). A candidate is a node only if it removes the tracks it already describes:
  * fast (3D motion radius R > 2 thr, after removing the motion all points share): tracks within p80 distance 0.4 R of it;
  * slow: tracks at a median distance <= 2 thr that keep it (p90 |d - median| <= max(0.2 median, 1.2 thr));
  * it must remove >= 10 tracks and >= 10% of the tracks it reaches, and must not stay within near_px in 2D (p90) of a kept
    node; at most 200 nodes (length-biased farthest-point sampling). thr = the scene's track spacing (build_edg).
Edges: every pair of nodes sharing > 30% of the shorter one's frames with score c (0.5 + v) <= max(0.3 x object extent,
  1.17 x the larger node reach), c = the most frequent 3D distance of the pair, v = how differently they move (relative
  step); a pair touching a fast node also needs p90 distance < 0.8 R unless v <= 0.25; and p90 |d - median| <= 1.5 median,
  v <= 0.28 (the distance must stay roughly constant).
Point links: each moving point joins its 2 nearest nodes (p90 distance, among its 16 nearest) that stay close in every
  frame they share (build_edg.build_dyn_to_edg_edges: max <= 3 x spacing, not varying, p90 <= 2.5 x spacing).
usage: python edg_topology.py
"""
import time

import numpy as np

import build_edg as B
import utils.dataset_config as dataset_config
import utils.definitions

try:
    _nq = B._nanquantile_fast
except AttributeError:
    def _nq(a, q, axis):
        return np.nanquantile(a, q, axis=axis)


# ------------------------------------------------------------------ node selection (motion-radius rule)
def motion_radius(P, Vs):
    """RMS distance of each track's positions over its seen frames from their mean. P (Q, T, 3), Vs (T, Q)."""
    with np.errstate(all="ignore"):
        mu = np.nanmean(np.where(Vs.T[..., None], P, np.nan), axis=1)
        R = np.sqrt(np.nanmean(np.where(Vs.T, ((P - mu[:, None]) ** 2).sum(-1), np.nan), axis=1))
    return np.nan_to_num(R)


def shared_offset(P, Vs, iters=8):
    """Motion shared by all moving points as a per-frame offset c(t), by alternating medians (median polish). (T, 3)."""
    T = Vs.shape[0]; c = np.zeros((T, 3)); n = Vs.sum(1)
    for ax in range(3):
        X = np.where(Vs.T, P[..., ax], np.nan).astype(np.float32)
        ca = np.zeros(T, np.float32)
        with np.errstate(all="ignore"):
            for _ in range(iters):
                mu = np.nanmedian(X - ca[None], axis=1)
                ca = np.nan_to_num(np.nanmedian(X - mu[:, None], axis=0))
        ca[n < 10] = 0
        c[:, ax] = ca - np.median(ca[n >= 10]) if (n >= 10).any() else ca
    return c


def compensated_radius(P, Vs, thr, gate=2.0):
    """Motion radius after removing the shared motion when it is large (its RMS radius over frames >= gate x thr).
    Returns R, R before, the offset c (T, 3), its radius, and whether it was removed."""
    R0 = motion_radius(P, Vs); c = shared_offset(P, Vs); seen = Vs.sum(1) >= 10
    rc = float(np.sqrt(np.mean(np.sum((c[seen] - c[seen].mean(0)) ** 2, axis=1)))) if seen.any() else 0.0
    on = rc >= gate * thr
    return (motion_radius(P - c[None], Vs) if on else R0), R0, c, rc, bool(on)


def select_nodes(P, Vs, Ocount, R, thr, prm, trees, close2d, overlap_frac=0.5, min_node_frames=20, quantile=0.80, nsamp=30):
    """Picked nodes in pick order (see the module docstring). trees = build_edg._frame_trees_2d(P, Vs, T)."""
    idx_per, tree_per = trees
    has = np.where(Ocount > int(min_node_frames))[0]
    order = has[np.argsort(-Ocount[has], kind="stable")]
    alive = np.zeros(P.shape[0], bool); alive[has] = True
    nodes = []
    for g in order:
        if not alive[g]:
            continue
        if nodes:                                                      # 2D closeness to a kept node
            kn = np.asarray(nodes, np.int64)
            co = Vs[:, kn] & Vs[:, [g]]
            ok = co.sum(0) > overlap_frac * np.minimum(Ocount[kn], Ocount[g])
            if ok.any():
                kn, co = kn[ok], co[:, ok]
                d = np.where(co.T, np.linalg.norm(close2d[0][kn] - close2d[0][g][None], axis=2), np.nan)
                if (_nq(d, close2d[2], axis=1) <= close2d[1]).any():
                    alive[g] = False; continue
        is_fast = R[g] > prm["r0"] * thr
        r = prm["a"] * R[g] if is_fast else prm["cap"] * thr
        gframes = np.where(Vs[:, g])[0]
        if gframes.size > nsamp:
            gframes = gframes[np.linspace(0, gframes.size - 1, nsamp).astype(int)]
        cand = set()
        for t in gframes:
            if tree_per[t] is None:
                continue
            for h in tree_per[t].query_ball_point(P[g, t], r):
                b = int(idx_per[t][h])
                if b != g and alive[b]:
                    cand.add(b)
        kill = np.zeros(0, np.int64); n_o = 0
        if cand:
            cand = np.fromiter(cand, np.int64)
            co = Vs[:, g][:, None] & Vs[:, cand]
            ok = co.sum(0) > overlap_frac * np.minimum(Ocount[g], Ocount[cand])
            cand, co = cand[ok], co[:, ok]; n_o = int(cand.size)
            if cand.size:
                d = np.where(co.T, np.linalg.norm(P[cand] - P[g][None], axis=2), np.nan)
                if is_fast:
                    hit = _nq(d, quantile, axis=1) <= prm["a"] * R[g]
                else:
                    med = np.nanmedian(d, axis=1)
                    dev = _nq(np.abs(d - med[:, None]), 0.90, axis=1)
                    hit = (med <= prm["cap"] * thr) & (dev <= np.maximum(prm["b"] * med, prm["e"] * thr))
                kill = cand[hit]
        alive[g] = False
        if kill.size < prm["k"] or kill.size < prm["rho"] * n_o:       # support: it must describe tracks
            continue
        alive[kill] = False
        nodes.append(int(g))
    return np.array(nodes, np.int64)


def fps_budget(P, Vs, Ocount, nodes, max_nodes, quantile=0.80):
    """build_edg's node budget: length-biased farthest-point sampling on the persistent distance, keeping max_nodes."""
    if nodes.size <= max_nodes:
        return nodes
    N = nodes.size
    xyn = P[nodes]; Vn = Vs[:, nodes]; length = Ocount[nodes].astype(np.float64)
    d_min = np.full(N, np.inf); keep_local = [int(np.argmax(length))]
    for _ in range(int(max_nodes) - 1):
        k = keep_local[-1]
        co = Vn[:, k][:, None] & Vn
        d = np.linalg.norm(xyn - xyn[k][None], axis=2).T
        dk = np.where(co, d, np.nan)
        has = co.sum(0) >= 1
        pd = np.full(N, np.inf)
        if np.any(has):
            pd[has] = np.nanquantile(dk[:, has], quantile, axis=0)
        d_min = np.minimum(d_min, pd)
        d_min[keep_local] = -1.0
        m = float(d_min.max())
        if m <= 0:
            break
        band = d_min >= (0.8 * m if np.isfinite(m) else np.inf)
        cand = np.where(band)[0]
        keep_local.append(int(cand[np.argmax(length[cand])]))
    return nodes[np.asarray(keep_local, np.int64)]


# ------------------------------------------------------------------ edges
def robust_mode(v):
    """Most frequent value of a sample: the median of its shortest half."""
    v = np.sort(v[np.isfinite(v)]); n = v.size
    if n < 3:
        return float(np.median(v)) if n else np.nan
    h = (n + 1) // 2
    i = int(np.argmin(v[h - 1:] - v[:n - h + 1]))
    return float(np.median(v[i:i + h]))


def object_extent(P, Vs):
    """Size of the moving object: per frame the p95 distance of the seen tracks from their centroid, median over frames."""
    return float(np.median([np.percentile(np.linalg.norm(P[Vs[t], t] - P[Vs[t], t].mean(0), axis=1), 95)
                            for t in range(Vs.shape[0]) if Vs[t].sum() > 10]))


def select_edges(nodes, P, Vs, Ocount, R, thr, ext, c_sh, on_sh, prm):
    """Node pairs (local indices) joined by the moving-together score (see the module docstring)."""
    M = len(nodes)
    X = P[nodes] - (c_sh[None] if on_sh else 0); V = Vs[:, nodes].T
    st = V[:, 1:] & V[:, :-1]; d = np.where(st[..., None], np.diff(X, axis=1), np.nan); nd = np.linalg.norm(d, axis=-1)
    with np.errstate(all="ignore"):
        noise = float(np.nanpercentile(np.nanmedian(nd, axis=1), 10))
    eps = max(2.0 * noise, 0.08 * thr)                                 # a floor where the noise step is tiny
    reach = np.where(R[nodes] > prm["r0"] * thr, 0.4 * R[nodes], 2.0 * thr)  # how far node selection removes tracks
    score = np.full((M, M), np.inf); vel = np.full((M, M), np.nan); q90 = np.full((M, M), np.nan)
    med = np.full((M, M), np.nan); dev = np.full((M, M), np.nan)
    for i in range(M):
        for j in range(i + 1, M):
            co = V[i] & V[j]
            if co.sum() <= prm["overlap"] * min(Ocount[nodes[i]], Ocount[nodes[j]]) or co.sum() < 3:
                continue
            c = robust_mode(np.linalg.norm(X[i, co] - X[j, co], axis=1))
            with np.errstate(all="ignore"):
                v = np.nanmedian(np.linalg.norm(d[i] - d[j], axis=-1) / (nd[i] + nd[j] + eps))
            if np.isfinite(v):
                score[i, j] = score[j, i] = c * (0.5 + v); vel[i, j] = vel[j, i] = v
                d3 = np.linalg.norm(X[i, co] - X[j, co], axis=1); q90[i, j] = q90[j, i] = np.quantile(d3, 0.90)
                med[i, j] = med[j, i] = np.median(d3); dev[i, j] = dev[j, i] = np.quantile(np.abs(d3 - med[i, j]), 0.90)
    lim = np.where(R[nodes] > prm["r0"] * thr, prm["ball"] * R[nodes], np.inf)
    with np.errstate(invalid="ignore"):
        in_ball = q90 < np.minimum(lim[:, None], lim[None])
        ok = score <= np.maximum(prm["alpha"] * ext, 0.78 * 1.5 * np.maximum(reach[:, None], reach[None]))
        ok &= (dev <= prm["const_f"] * med) | (med <= 0)
        ok &= vel <= prm["vel_max"]
        ok &= in_ball | (vel <= prm["strong"])
    ii, jj = np.nonzero(np.triu(ok, 1))
    return np.stack([ii, jj], 1).astype(np.int64)


def main():
    t0 = time.time()
    tracks = utils.definitions.dynamic_tracks_for_edg()
    D = B._load_all(tracks_path=str(tracks))
    D["dyn_mask"] = B._dilate_dyn_mask(D, 5)
    T = D["T"]; Q = D["tracks_xyz"].shape[1]
    s = B._frame_scale(D["rt"], T, False)
    Ps, Vs = B._stack_sampled(D, s, np.arange(T, dtype=int))
    Ocount = Vs.sum(axis=0); thr = B._auto_edge_threshold(Ps, Vs)
    P = np.transpose(Ps, (1, 0, 2))
    R, R_raw, c_sh, rc_sh, on_sh = compensated_radius(P, Vs, thr)
    prm = dict(a=dataset_config.get_float("EDG_FAST_REACH", 0.4), b=0.2, r0=dataset_config.get_float("EDG_FAST_RADIUS", 2.0),
               cap=2.0, e=1.2, k=10, rho=0.1, overlap=0.3, alpha=0.3, ball=0.8, strong=0.25, const_f=1.5, vel_max=0.28)
    min_frames = dataset_config.get_int("EDG_NODE_MIN_FRAMES", min(20, max(2, int(round(0.4 * T)))))
    near_px = B._dyn_region_near_px(D, 0.16)
    picked = select_nodes(P, Vs, Ocount, R, thr, prm, B._frame_trees_2d(P, Vs, T), (D["tracks_xy"][0].astype(np.float64), near_px, 0.90),
                          min_node_frames=min_frames)
    nodes = fps_budget(P, Vs, Ocount, picked, dataset_config.get_int("EDG_MAX_NODES", 200))
    ext = object_extent(P, Vs)
    e_loc = select_edges(nodes, P, Vs, Ocount, R, thr, ext, c_sh, on_sh, prm)
    edges = nodes[e_loc] if len(e_loc) else np.zeros((0, 2), np.int64)
    dyn = B.build_dyn_to_edg_edges(np.zeros((0, 2), np.int64), Q, nodes, Ps, Vs, Ocount, thr, hop_lt=3, min_conn=2, max_factor=2.5,
                                   max_dist_factor=3.0, min_max_ratio=0.5, cand_near=16, verbose=False)
    out = utils.definitions.output_dir / "edg_topology.npz"
    np.savez_compressed(out, edg_nodes=nodes.astype(np.int64), edg_edges=edges.astype(np.int64), dyn_edg_edges=dyn.astype(np.int64),
                        n_tracks=np.array(Q), thr=np.array(thr), node_fast=R[nodes] > prm["r0"] * thr, tracks_file=np.array(str(tracks)))
    deg = np.bincount(e_loc.ravel(), minlength=len(nodes)) if len(e_loc) else np.zeros(len(nodes), int)
    print(f"[edg] {tracks.name}: {Q} tracks, {T} frames, thr {thr:.4f}, shared motion {'removed' if on_sh else 'kept'}; "
          f"{len(picked)} picked -> {len(nodes)} nodes ({int((R[nodes] > prm['r0'] * thr).sum())} fast), {len(edges)} edges "
          f"(nodes without an edge: {int((deg == 0).sum())}), {len(dyn)} point links on {len(np.unique(dyn[:, 0])) if len(dyn) else 0} "
          f"points; {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
