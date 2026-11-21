"""Step 4a: smooth the moving points' 3D tracks with the cameras fixed. Per track, the world positions X_t from its first to its
last observation minimise
  sum_t  rho(|pi_t(X_t) - u_t|^2; c_px)                       reprojection of the 2D track, px, Cauchy
+ sum_t  rho((log(z_t / (s_t d_t)) / sigma_d)^2; c_d)         depth along the ray vs the step-2 lift, relative, weaker
+ sum_t  |f (X_{t+1} - 2 X_t + X_{t-1}) / z_t|^2 / sigma_a^2  acceleration, px-equivalent per frame^2
(frames where the point is hidden are tied in by the acceleration term only). Cameras: X = s R x_cam + t (result_XM.npz),
K from outputs/depth_anything_3/intrinsics.npy, u_t / d_t = the 2D track and its depth (tracks_dynamic_only.npz). Gauss-Newton
with IRLS weights; each track's normal equations are banded (xyz interleaved, bandwidth 8) and solved exactly.
Without it the lifted points jitter frame to frame with the per-frame depth and the deformation graph inherits the zig-zag.

Writes outputs/tracks_dynamic_only_smoothed.npz: the input keys with tracks_xyz replaced by the smoothed points in each frame's
camera (x_cam = R^T (X - t) / s, NaN where the input is invalid), plus tracks_world_smoothed (Q, T, 3).
config.json: SMOOTH_SIGMA_DEPTH (0.1), SMOOTH_SIGMA_ACC (30 px/frame^2), SMOOTH_DEPTH_REPAIR (0 = off): before the solve, an
observation whose log-depth is more than this from its track's own smoothed log-depth (depth spike) is moved along its ray
onto that smoothed depth. Cauchy scales: 2 px (reprojection), 2 sigma-depth (depth).
usage: python smooth_dynamic_tracks.py
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np
from scipy.linalg import LinAlgError, solveh_banded
from types import SimpleNamespace

import utils.dataset_config as dataset_config
import utils.definitions
from utils.edg import scatter_solved_frame_vector, scatter_solved_poses

args = SimpleNamespace(
    sigma_depth=dataset_config.get_float("SMOOTH_SIGMA_DEPTH", 0.1),   # relative depth noise (log); larger = weaker depth term
    cauchy_depth=2.0,                                                  # Cauchy scale of the depth term, in sigma-depth units
    sigma_acc=dataset_config.get_float("SMOOTH_SIGMA_ACC", 30.0),      # acceleration, px-equivalent per frame^2
    cauchy_px=2.0,                                                     # Cauchy scale of the reprojection term, px
    depth_repair=dataset_config.get_float("SMOOTH_DEPTH_REPAIR", 0.0), # log-depth distance from the track's smoothed depth
                                                                       # above which an observation is repaired (0 = off)
    sigma_logz=0.02,                                      # repair reference: log-depth 2nd difference per frame^2
    iters=10, min_obs=3, workers=min(32, os.cpu_count() or 1), out="tracks_dynamic_only_smoothed.npz")

O = str(utils.definitions.output_dir)
src = f"{O}/tracks_dynamic_only.npz"
keys = np.load(src).files
with ThreadPoolExecutor(4) as ex:                         # every key once (all are written back), decompressed in parallel
    d = dict(zip(keys, ex.map(lambda k: np.load(src, allow_pickle=True)[k], keys)))   # (a handle per key: thread safe)
UV = np.asarray(d["tracks_xy"][0], np.float64)           # read only: no copy when it is float64 already
XC = d["tracks_xyz"][0].astype(np.float64)
Q, T = UV.shape[:2]
rt = np.load(f"{O}/result_XM.npz", allow_pickle=True)
R, C = scatter_solved_poses(rt, T)                       # camera-to-world rotation, camera centre (NaN if dropped)
s = scatter_solved_frame_vector(rt, "s", T, default=1.0)
K = np.load(f"{O}/depth_anything_3/intrinsics.npy").astype(np.float64)[0]
fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
f_mean = 0.5 * (fx + fy)
camok = np.isfinite(C).all(1)


def rotate(Rt, X, blk=2048):
    """np.einsum("tij,qtj->qti", Rt, X), the same einsum on blocks of tracks in threads (speed only)."""
    out = np.empty(X.shape, np.result_type(Rt, X))
    def one(a):
        out[a:a + blk] = np.einsum("tij,qtj->qti", Rt, X[a:a + blk])
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(one, range(0, len(X), blk)))
    return out


X0 = rotate(R * s[:, None, None], XC) + C[None]
V = np.isfinite(X0).all(-1) & ~d["occluded"][0].astype(bool) & d["dynamic_obs_mask"].astype(bool) & camok[None]
# frames without a solved camera: borrow the nearest solved camera for the acceleration weight only
near = np.array([np.flatnonzero(camok)[np.argmin(np.abs(np.flatnonzero(camok) - t))] for t in range(T)])
Rn, Cn = R[near], C[near]
RT = np.transpose(Rn, (0, 2, 1))                         # world-to-camera rotation
BW = 8                                                    # band half-width: xyz interleaved, acceleration spans 3 frames


def cost_terms(X, t0, v, u, dd):
    """Per-term costs and pieces for one track span. X (L,3) world; frames t0..t0+L-1."""
    L = len(X)
    tt = slice(t0, t0 + L)                                # views, not copies (speed)
    xc = np.einsum("tij,tj->ti", RT[tt], X - Cn[tt])
    if (xc[:, 2] < 0.05 * (s[tt] * dd).min()).any():     # in front of every camera: the robust losses alone would let a
        return np.inf, None, None, None, None, None      # point escape behind the camera, so such a step is rejected
    z = xc[:, 2]
    r = np.stack([fx * xc[:, 0] / z + cx, fy * xc[:, 1] / z + cy], 1) - u
    r2 = (r ** 2).sum(1)
    e = np.log(z / (s[tt] * dd)) / args.sigma_depth
    e[~v] = 0
    r2[~v] = 0
    acc = (X[2:] - 2 * X[1:-1] + X[:-2]) * (f_mean / z[1:-1])[:, None] / args.sigma_acc
    c, cd = args.cauchy_px, args.cauchy_depth
    cost = (c * c * np.log1p(r2 / (c * c))).sum() + (cd * cd * np.log1p(e * e / (cd * cd))).sum() + (acc ** 2).sum()
    return cost, xc, z, r, r2, e


def solve_track(q):
    v_all = V[q]
    obs = np.flatnonzero(v_all)
    if len(obs) < args.min_obs:
        return q, 0, None
    t0, t1 = obs[0], obs[-1] + 1
    L = t1 - t0
    v = v_all[t0:t1]
    u = UV[q, t0:t1]
    dd = np.where(v, XC[q, t0:t1, 2], np.median(XC[q, t0:t1, 2][v]))
    X = np.empty((L, 3))
    for k in range(3):                                    # hidden frames start on the line between their neighbours
        X[:, k] = np.interp(np.arange(L), np.flatnonzero(v), X0[q, t0:t1][v, k])
    tt = slice(t0, t1)
    cost, xc, z, r, r2, e = cost_terms(X, t0, v, u, dd)
    if not np.isfinite(cost):                             # the raw lift itself is behind a camera: leave it as is
        return q, t0, X
    lam = 1e-6
    for _ in range(args.iters):
        # data terms: 3x3 Gauss-Newton block per frame (IRLS weights for the Cauchy losses)
        wr = np.where(v, 1.0 / (1.0 + r2 / args.cauchy_px ** 2), 0.0)
        wd = np.where(v, 1.0 / (1.0 + e * e / args.cauchy_depth ** 2), 0.0)
        Jc = np.zeros((L, 2, 3))
        Jc[:, 0, 0] = fx / z
        Jc[:, 0, 2] = -fx * xc[:, 0] / z ** 2
        Jc[:, 1, 1] = fy / z
        Jc[:, 1, 2] = -fy * xc[:, 1] / z ** 2
        J = Jc @ RT[tt]                                   # d(pixel)/dX
        Jd = Rn[tt][:, :, 2] / (z * args.sigma_depth)[:, None]   # d(e)/dX, e = log z / sigma
        H = wr[:, None, None] * np.einsum("tai,taj->tij", J, J) + wd[:, None, None] * Jd[:, :, None] * Jd[:, None, :]
        g = wr[:, None] * np.einsum("tai,ta->ti", J, r) + wd[:, None] * Jd * e[:, None]
        ab = np.zeros((BW + 1, 3 * L))
        for k in range(3):
            for l in range(k, 3):
                ab[BW + k - l, l::3] += H[:, k, l]
        # acceleration: weight beta per interior frame, coefficients (1, -2, 1) on frames (t-1, t, t+1), each axis
        beta = (f_mean / (z[1:-1] * args.sigma_acc)) ** 2
        acc = X[2:] - 2 * X[1:-1] + X[:-2]
        cf = (1.0, -2.0, 1.0)
        for a in range(3):
            g[a:L - 2 + a] += cf[a] * beta[:, None] * acc
            for b in range(a, 3):                         # (all three axes at once: the same sums per entry, speed)
                ab[BW - 3 * (b - a), 3 * b:3 * (L - 2 + b)] += np.repeat(beta * cf[a] * cf[b], 3)
        diag = ab[BW].copy()
        for _try in range(8):
            ab[BW] = diag * (1 + lam) + 1e-12
            try:
                dx = solveh_banded(ab, -g.ravel(), lower=False, check_finite=False).reshape(L, 3)
            except LinAlgError:
                lam *= 10
                continue
            new = cost_terms(X + dx, t0, v, u, dd)
            if new[0] <= cost:
                X = X + dx
                lam = max(lam / 10, 1e-9)
                break
            lam *= 10
        else:
            break
        rel = (cost - new[0]) / max(cost, 1e-12)
        cost, xc, z, r, r2, e = new
        if rel < 1e-7:
            break
    return q, t0, X


def smooth_logz(v, lo):
    """Robust temporal smoothing of one track's log-depth lo (valid where v): Cauchy data term + 2nd difference / sigma_logz."""
    L = len(v)
    l = np.interp(np.arange(L), np.flatnonzero(v), lo[v])
    D = np.zeros((3, L))                                  # (D2^T D2) / sigma_logz^2 in upper banded form (bandwidth 2)
    if L >= 3:
        c = np.zeros(L); c[:L - 2] += 1; c[1:L - 1] += 4; c[2:] += 1
        D[2] = c
        o1 = np.zeros(L - 1); o1[:L - 2] -= 2; o1[1:] -= 2
        D[1, 1:] = o1
        D[0, 2:] = 1
        D /= args.sigma_logz ** 2
    for _ in range(args.iters):
        e = (l - lo) / args.sigma_depth
        w = np.where(v, 1.0 / (1.0 + e * e / args.cauchy_depth ** 2), 0.0) / args.sigma_depth ** 2
        ab = D.copy(); ab[2] += w + 1e-12
        l_new = solveh_banded(ab, w * lo, lower=False, check_finite=False)
        done = np.abs(l_new - l).max() < 1e-6
        l = l_new
        if done:
            break
    return l


def depth_spikes(q):
    """Observations of track q whose log-depth is more than depth_repair from its own smoothed log-depth, and the depth
    ratio that moves each onto it."""
    obs = np.flatnonzero(V[q])
    if len(obs) < args.min_obs:
        return q, obs[:0], np.zeros(0)
    t0, t1 = obs[0], obs[-1] + 1
    v = V[q, t0:t1]
    lo = np.where(v, np.log(np.where(v, s[t0:t1] * XC[q, t0:t1, 2], 1.0)), 0.0)
    l = smooth_logz(v, lo)
    bad = v & (np.abs(lo - l) > args.depth_repair)
    return q, t0 + np.flatnonzero(bad), np.exp(l - lo)[bad]


if __name__ == "__main__":
    np.seterr(invalid="ignore", divide="ignore")
    ids = list(range(Q))
    t_start = time.time()
    if args.depth_repair > 0:                             # before the solve's workers fork, so they see the repaired lift
        n_bad = 0
        with Pool(args.workers) as pool:
            for q, bad, ratio in pool.imap_unordered(depth_spikes, ids, chunksize=64):
                XC[q, bad] *= ratio[:, None]
                X0[q, bad] = np.einsum("tij,tj->ti", R[bad] * s[bad, None, None], XC[q, bad]) + C[bad]
                n_bad += len(bad)
        print(f"depth spikes (> {args.depth_repair} in log-depth from the track's smoothed depth): {n_bad} of {int(V.sum())} "
              f"observations moved onto it along their ray")
    with Pool(args.workers) as pool:
        XS = np.full((Q, T, 3), np.nan)                   # after the fork (smaller workers start faster)
        for q, t0, X in pool.imap_unordered(solve_track, ids, chunksize=64):
            if X is not None:
                XS[q, t0:t0 + len(X)] = X
    t_solve = time.time() - t_start
    solved = np.isfinite(XS).all(-1).any(1)
    print(f"{utils.definitions.dataset_name}: solved {solved.sum()} of {len(ids)} tracks in {t_solve:.1f} s on {args.workers} workers "
          f"(sigma_depth {args.sigma_depth}, sigma_acc {args.sigma_acc} px/frame^2, Cauchy {args.cauchy_px} px)")
    # written in the background while the statistics run (speed)
    xcam = rotate(np.transpose(R, (0, 2, 1)) / s[:, None, None], XS - C[None])
    xcam[~V] = np.nan
    out = dict(d)
    out["tracks_xyz"] = xcam[None]
    out["tracks_world_smoothed"] = XS.astype(np.float32)
    writer = ThreadPoolExecutor(1)                        # uncompressed: 1.3 GB (0.4 GB compressed) on teddy, but ~20 s
                                                          # faster to write and ~1 s faster to read for each later step
    written = writer.submit(np.savez, f"{O}/{args.out}", **out)

    # ---- statistics over the observations of the solved tracks (evaluated at those observations only) ----
    m = V & solved[:, None]
    fm = np.flatnonzero(m)                                # observations as flat (track, frame) indices
    xcs = rotate(RT, XS - Cn[None]).reshape(-1, 3)
    xm = xcs[fm]; zm = xm[:, 2]
    du = np.stack([fx * xm[:, 0] / zm + cx, fy * xm[:, 1] / zm + cy], -1) - UV.reshape(-1, 2)[fm]
    rep = np.sqrt(du[:, 0] * du[:, 0] + du[:, 1] * du[:, 1])  # = np.linalg.norm(du, axis=-1), bit for bit
    z0 = XC.reshape(-1, 3)[fm, 2] * s[fm % T]             # raw camera depth in world units = s_t * d_t
    dz = np.abs(np.log(zm / z0))
    aq, at = np.nonzero(m[:, 2:] & m[:, 1:-1] & m[:, :-2])
    fa = aq * T + at + 1                                  # middle frame of a run of 3 visible frames

    def accel(X):                                         # px-equivalent acceleration on runs of 3 visible frames
        X = X.reshape(-1, 3)
        a = X[fa + 1] - 2 * X[fa] + X[fa - 1]
        return np.sqrt((a[:, 0] * a[:, 0] + a[:, 1] * a[:, 1]) + a[:, 2] * a[:, 2]) * f_mean / xcs[fa, 2]
    with ThreadPoolExecutor(2) as ex:
        a0, a1 = ex.map(accel, (X0, XS))
    pct = lambda x: " / ".join(f"{v:.2f}" for v in np.percentile(x, (50, 90)))
    print(f"  reprojection of smoothed points  median / p90 {pct(rep)} px, > 5 px {100 * (rep > 5).mean():.1f}%")
    print(f"  depth change |log(z/z_DA3)|      median / p90 {pct(dz)}; behind a camera: {int((zm <= 0).sum())}")
    print(f"  acceleration (px-eq./frame^2)    raw {pct(a0)}  ->  smoothed {pct(a1)}")
    stats = dict(sigma_depth=args.sigma_depth, sigma_acc=args.sigma_acc, cauchy_px=args.cauchy_px,
                 reproj_med=float(np.median(rep)), reproj_p90=float(np.percentile(rep, 90)), reproj_gt5=float((rep > 5).mean()),
                 dz_med=float(np.median(dz)), acc_raw=float(np.median(a0)), acc_smooth=float(np.median(a1)),
                 solve_seconds=t_solve, workers=args.workers, tracks=int(solved.sum()), obs=int(m.sum()))
    written.result(); writer.shutdown()
    json.dump(stats, open(f"{O}/{os.path.splitext(args.out)[0]}.json", "w"), indent=1)
    print(f"  wrote {O}/{args.out}")
