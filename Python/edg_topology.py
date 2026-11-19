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
from concurrent.futures import ThreadPoolExecutor

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
def motion_radius(P, Vs, off=None):
    """RMS distance of each track's positions (minus a per-frame offset off (T, 3)) over its seen frames from their mean.
    P (Q, T, 3), Vs (T, Q)."""
    R = np.empty(P.shape[0])

    def one(r):                                                        # np.nanmean over the seen frames (P is finite there)
        V = Vs.T[r]; n = V.sum(1); Pr = P[r] if off is None else P[r] - off[None]   # as sum / count, without its NaN
        with np.errstate(all="ignore"):                                # copies, on blocks of tracks in threads (speed)
            mu = np.where(V[..., None], Pr, 0.0).sum(1) / n[:, None]
            d = Pr - mu[:, None]
            R[r] = np.sqrt(np.where(V, (d[..., 0] * d[..., 0] + d[..., 1] * d[..., 1]) + d[..., 2] * d[..., 2], 0.0).sum(1) / n)
    step = max(1, -(-P.shape[0] // 32))
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(one, [slice(i, i + step) for i in range(0, P.shape[0], step)]))
    return np.nan_to_num(R)


def nanmedian_rows(a, off=None):
    """np.nanmedian(a - off[None], axis=1) for a 2D a: one sort per row, large arrays in row blocks in threads (speed only:
    numpy takes a masked-array path for rows shorter than 600 and a Python loop otherwise). The two middle finite values
    averaged in the array's dtype, as np.median does; NaN for an all-NaN row."""
    out = np.empty(a.shape[0], a.dtype if off is None else np.result_type(a, off))
    if a.shape[1] == 0:                                                # no values: NaN, as np.nanmedian
        out[:] = np.nan
        return out

    def one(r):
        s = np.sort(a[r] if off is None else a[r] - off[None], axis=1)  # NaN last
        n = (~np.isnan(s)).sum(1)
        lo = np.take_along_axis(s, ((np.maximum(n, 1) - 1) // 2)[:, None], 1)[:, 0]
        hi = np.take_along_axis(s, (n // 2)[:, None], 1)[:, 0]
        out[r] = (lo + hi) / 2
    if a.size < 1 << 16:
        one(slice(None))
        return out
    step = max(1, -(-a.shape[0] // 32))
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(one, [slice(i, i + step) for i in range(0, a.shape[0], step)]))
    return out


def shared_offset(P, Vs, iters=8):
    """Motion shared by all moving points as a per-frame offset c(t), by alternating medians (median polish). (T, 3)."""
    T = Vs.shape[0]; c = np.zeros((T, 3)); n = Vs.sum(1)
    for ax in range(3):
        X = np.where(Vs.T, P[..., ax], np.nan).astype(np.float32)
        XQ, XT = np.ascontiguousarray(X), np.ascontiguousarray(X.T)    # rows per track, rows per frame (speed)
        ca = np.zeros(T, np.float32)
        with np.errstate(all="ignore"):
            for _ in range(iters):
                mu = nanmedian_rows(XQ, ca)                            # = np.nanmedian(X - ca[None], axis=1)
                ca = np.nan_to_num(nanmedian_rows(XT, mu))             # = np.nanmedian(X - mu[:, None], axis=0)
        ca[n < 10] = 0
        c[:, ax] = ca - np.median(ca[n >= 10]) if (n >= 10).any() else ca
    return c


def compensated_radius(P, Vs, thr, gate=2.0):
    """Motion radius after removing the shared motion when it is large (its RMS radius over frames >= gate x thr).
    Returns R, R before, the offset c (T, 3), its radius, and whether it was removed."""
    R0 = motion_radius(P, Vs); c = shared_offset(P, Vs); seen = Vs.sum(1) >= 10
    rc = float(np.sqrt(np.mean(np.sum((c[seen] - c[seen].mean(0)) ** 2, axis=1)))) if seen.any() else 0.0
    on = rc >= gate * thr
    return (motion_radius(P, Vs, c) if on else R0), R0, c, rc, bool(on)


def select_nodes(P, Vs, Ocount, R, thr, prm, trees, close2d, overlap_frac=0.5, min_node_frames=20, quantile=0.80, nsamp=30):
    """Picked nodes in pick order (see the module docstring). trees = build_edg._frame_trees_2d(P, Vs, T)."""
    idx_per, tree_per = trees
    VQ = np.ascontiguousarray(Vs.T)                                    # (Q, T): one row per track (speed)

    def close(kn, g, gseen):                                           # is g close in 2D to one of the nodes kn?
        co = VQ[kn][:, gseen]                                          # (co-visible frames are among g's seen frames gseen;
        ok = co.sum(1) > overlap_frac * np.minimum(Ocount[kn], Ocount[g])   # the distances skip only NaN entries: speed)
        if not ok.any():
            return False
        kn, co = kn[ok], co[ok]
        v = close2d[0][kn[:, None], gseen] - close2d[0][g, gseen][None]
        d = np.where(co, np.sqrt(v[..., 0] * v[..., 0] + v[..., 1] * v[..., 1]), np.nan)   # = np.linalg.norm
        near = np.nanmin(d, axis=1) <= close2d[1]                      # a quantile is >= the minimum: only these can be close
        return near.any() and (_nq(d[near], close2d[2], axis=1) <= close2d[1]).any()

    has = np.where(Ocount > int(min_node_frames))[0]
    order = has[np.argsort(-Ocount[has], kind="stable")]
    alive = np.zeros(P.shape[0], bool); alive[has] = True
    nodes = []
    ex = ThreadPoolExecutor(8)
    for g in order:
        if not alive[g]:
            continue
        gseen = np.flatnonzero(VQ[g])
        if nodes:                                                      # 2D closeness to a kept node; the kept node nearest
            kn = np.asarray(nodes, np.int64)                           # at g's middle seen frame is tested first (speed)
            f = gseen[gseen.size // 2]; w = close2d[0][kn, f] - close2d[0][g, f]
            c = np.argmin(np.where(VQ[kn, f], w[:, 0] * w[:, 0] + w[:, 1] * w[:, 1], np.inf))
            if close(kn[c:c + 1], g, gseen) or close(np.delete(kn, c), g, gseen):
                alive[g] = False; continue
        is_fast = R[g] > prm["r0"] * thr
        r = prm["a"] * R[g] if is_fast else prm["cap"] * thr
        gframes = gseen
        if gframes.size > nsamp:
            gframes = gframes[np.linspace(0, gframes.size - 1, nsamp).astype(int)]
        cand = []
        for t in gframes:
            if tree_per[t] is None:
                continue
            cand.append(idx_per[t][tree_per[t].query_ball_point(P[g, t], r)])
        cand = np.unique(np.concatenate(cand)) if cand else np.zeros(0, np.int64)   # the set of tracks in the balls (order-free)
        cand = cand[(cand != g) & alive[cand]]
        kill = np.zeros(0, np.int64); n_o = 0
        if cand.size:
            co = VQ[cand][:, gseen]
            ok = co.sum(1) > overlap_frac * np.minimum(Ocount[g], Ocount[cand])
            cand, co = cand[ok], co[ok]; n_o = int(cand.size)

            def hit(r):                                                # the test of the candidates cand[r], row by row:
                v = P[cand[r, None], gseen] - P[g, gseen][None]        # row blocks in threads (speed)
                d = np.where(co[r], np.sqrt((v[..., 0] * v[..., 0] + v[..., 1] * v[..., 1]) + v[..., 2] * v[..., 2]), np.nan)
                if is_fast:
                    return _nq(d, quantile, axis=1) <= prm["a"] * R[g]
                med = np.nanmedian(d, axis=1)
                dev = _nq(np.abs(d - med[:, None]), 0.90, axis=1)
                return (med <= prm["cap"] * thr) & (dev <= np.maximum(prm["b"] * med, prm["e"] * thr))
            if cand.size:
                kill = cand[np.concatenate(list(ex.map(hit, [slice(i, i + 256) for i in range(0, cand.size, 256)])))]
        alive[g] = False
        if kill.size < prm["k"] or kill.size < prm["rho"] * n_o:       # support: it must describe tracks
            continue
        alive[kill] = False
        nodes.append(int(g))
    ex.shutdown()
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
def robust_mode_rows(a):
    """Most frequent value of each row's sample (NaN = missing): the median of its shortest half (the plain median below 3)."""
    if a.shape[1] == 0:
        return np.full(a.shape[0], np.nan)
    s = np.sort(a, axis=1); n = (~np.isnan(s)).sum(1); h = (n + 1) // 2      # NaN last
    k = np.arange(s.shape[1])[None]
    w = np.take_along_axis(s, np.minimum(k + h[:, None] - 1, s.shape[1] - 1), 1) - s   # spread of the half starting at k
    i = np.argmin(np.where(k <= (n - h)[:, None], w, np.inf), axis=1)
    lo = np.take_along_axis(s, (i + (h - 1) // 2)[:, None], 1)[:, 0]; hi = np.take_along_axis(s, (i + h // 2)[:, None], 1)[:, 0]
    return np.where(n >= 3, (lo + hi) / 2, nanmedian_rows(a))


def object_extent(P, Vs):
    """Size of the moving object: per frame the p95 distance of the seen tracks from their centroid, median over frames."""
    with ThreadPoolExecutor(8) as ex:                                  # frames in threads (speed)
        return float(np.median(list(ex.map(lambda t: np.percentile(np.linalg.norm(P[Vs[t], t] - P[Vs[t], t].mean(0), axis=1), 95),
                                           [t for t in range(Vs.shape[0]) if Vs[t].sum() > 10]))))


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
    cnt = V.astype(np.int64) @ V.T.astype(np.int64)                    # co-visible frames of each pair
    norm3 = lambda u: np.sqrt((u[..., 0] * u[..., 0] + u[..., 1] * u[..., 1]) + u[..., 2] * u[..., 2])   # = np.linalg.norm
    for i in range(M):                                                 # all partners j > i of node i at once (speed)
        j = np.arange(i + 1, M)
        j = j[(cnt[i, j] > prm["overlap"] * np.minimum(Ocount[nodes[i]], Ocount[nodes[j]])) & (cnt[i, j] >= 3)]
        if not j.size:
            continue
        d3 = np.where(V[i] & V[j], norm3(X[i][None] - X[j]), np.nan)  # pair distance in each co-visible frame
        with np.errstate(all="ignore"):
            v = nanmedian_rows(norm3(d[i][None] - d[j]) / (nd[i] + nd[j] + eps))
        ok = np.isfinite(v); j, d3, v = j[ok], d3[ok], v[ok]
        m3 = nanmedian_rows(d3)
        score[i, j] = score[j, i] = robust_mode_rows(d3) * (0.5 + v); vel[i, j] = vel[j, i] = v
        q90[i, j] = q90[j, i] = _nq(d3, 0.90, axis=1)                  # = np.quantile over the co-visible frames
        med[i, j] = med[j, i] = m3; dev[i, j] = dev[j, i] = _nq(np.abs(d3 - m3[:, None]), 0.90, axis=1)
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
