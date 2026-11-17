"""Step 4a: smooth the moving points' 3D tracks with the cameras fixed. Per track, the world positions X_t from its first to its
last observation minimise
  sum_t  rho(|pi_t(X_t) - u_t|^2; c_px)                       reprojection of the 2D track, px, Cauchy
+ sum_t  rho((log(z_t / (s_t d_t)) / sigma_d)^2; c_d)         depth along the ray vs the step-2 lift, relative, weaker
+ sum_t  |f (X_{t+1} - 2 X_t + X_{t-1}) / z_t|^2 / sigma_a^2  acceleration, px-equivalent per frame^2
(frames where the point is hidden are tied in by the acceleration term only). Cameras: X = s R x_cam + t (result_XM.npz),
K from outputs/depth_anything_3/intrinsics.npy, u_t / d_t = the 2D track and its depth (tracks_dynamic_only.npz). Gauss-Newton
with IRLS weights; each track's normal equations are banded (xyz interleaved, bandwidth 8) and solved exactly.
Without it the lifted points jitter frame to frame with the per-frame depth, the deformation graph inherits the zig-zag and
the segment solves lose their optimality certificate more often.

Writes outputs/tracks_dynamic_only_smoothed.npz: the input keys with tracks_xyz replaced by the smoothed points in each frame's
camera (x_cam = R^T (X - t) / s, NaN where the input is invalid), plus tracks_world_smoothed (Q, T, 3).
config.json: SMOOTH_SIGMA_DEPTH (0.1), SMOOTH_CAUCHY_DEPTH (2), SMOOTH_SIGMA_ACC (3 px/frame^2), SMOOTH_CAUCHY_PX (2 px).
usage: python smooth_dynamic_tracks.py
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.linalg import LinAlgError, solveh_banded
from types import SimpleNamespace

import utils.dataset_config as dataset_config
import utils.definitions
from utils.edg import scatter_solved_frame_vector, scatter_solved_poses

args = SimpleNamespace(
    sigma_depth=dataset_config.get_float("SMOOTH_SIGMA_DEPTH", 0.1),   # relative depth noise (log); larger = weaker depth term
    cauchy_depth=dataset_config.get_float("SMOOTH_CAUCHY_DEPTH", 2.0), # Cauchy scale of the depth term, in sigma-depth units
    sigma_acc=dataset_config.get_float("SMOOTH_SIGMA_ACC", 3.0),       # acceleration, px-equivalent per frame^2
    cauchy_px=dataset_config.get_float("SMOOTH_CAUCHY_PX", 2.0),       # Cauchy scale of the reprojection term, px
    iters=10, min_obs=3, workers=min(32, os.cpu_count() or 1), out="tracks_dynamic_only_smoothed.npz")

O = str(utils.definitions.output_dir)
d = np.load(f"{O}/tracks_dynamic_only.npz", allow_pickle=True)
UV = d["tracks_xy"][0].astype(np.float64)
XC = d["tracks_xyz"][0].astype(np.float64)
Q, T = UV.shape[:2]
rt = np.load(f"{O}/result_XM.npz", allow_pickle=True)
R, C = scatter_solved_poses(rt, T)                       # camera-to-world rotation, camera centre (NaN if dropped)
s = scatter_solved_frame_vector(rt, "s", T, default=1.0)
K = np.load(f"{O}/depth_anything_3/intrinsics.npy").astype(np.float64)[0]
fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
f_mean = 0.5 * (fx + fy)
camok = np.isfinite(C).all(1)
X0 = np.einsum("tij,qtj->qti", R * s[:, None, None], XC) + C[None]
V = np.isfinite(X0).all(-1) & ~d["occluded"][0].astype(bool) & d["dynamic_obs_mask"].astype(bool) & camok[None]
# frames without a solved camera: borrow the nearest solved camera for the acceleration weight only
near = np.array([np.flatnonzero(camok)[np.argmin(np.abs(np.flatnonzero(camok) - t))] for t in range(T)])
Rn, Cn = R[near], C[near]
RT = np.transpose(Rn, (0, 2, 1))                         # world-to-camera rotation
BW = 8                                                    # band half-width: xyz interleaved, acceleration spans 3 frames


def cost_terms(X, t0, v, u, dd):
    """Per-term costs and pieces for one track span. X (L,3) world; frames t0..t0+L-1."""
    L = len(X)
    tt = np.arange(t0, t0 + L)
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
    tt = np.arange(t0, t1)
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
        idx = 3 * np.arange(L)
        for k in range(3):
            for l in range(k, 3):
                ab[BW + k - l, idx + l] += H[:, k, l]
        # acceleration: weight beta per interior frame, coefficients (1, -2, 1) on frames (t-1, t, t+1), each axis
        beta = (f_mean / (z[1:-1] * args.sigma_acc)) ** 2
        acc = X[2:] - 2 * X[1:-1] + X[:-2]
        cf = (1.0, -2.0, 1.0)
        for a in range(3):
            g[a:L - 2 + a] += cf[a] * beta[:, None] * acc
            for b in range(a, 3):
                for k in range(3):
                    ab[BW - 3 * (b - a), 3 * (np.arange(L - 2) + b) + k] += beta * cf[a] * cf[b]
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


if __name__ == "__main__":
    np.seterr(invalid="ignore", divide="ignore")
    ids = list(range(Q))
    t_start = time.time()
    XS = np.full((Q, T, 3), np.nan)
    with Pool(args.workers) as pool:
        for q, t0, X in pool.imap_unordered(solve_track, ids, chunksize=64):
            if X is not None:
                XS[q, t0:t0 + len(X)] = X
    t_solve = time.time() - t_start
    solved = np.isfinite(XS).all(-1).any(1)
    print(f"{utils.definitions.dataset_name}: solved {solved.sum()} of {len(ids)} tracks in {t_solve:.1f} s on {args.workers} workers "
          f"(sigma_depth {args.sigma_depth}, sigma_acc {args.sigma_acc} px/frame^2, Cauchy {args.cauchy_px} px)")

    # ---- statistics over the observations of the solved tracks ----
    m = V & solved[:, None]
    xcs = np.einsum("tij,qtj->qti", RT, XS - Cn[None])
    zs = xcs[..., 2]
    us = np.stack([fx * xcs[..., 0] / zs + cx, fy * xcs[..., 1] / zs + cy], -1)
    rep = np.linalg.norm(us - UV, axis=-1)[m]
    z0 = XC[..., 2] * s[None]                             # raw camera depth in world units = s_t * d_t
    dz = np.abs(np.log(zs / z0))[m]

    def accel(X):                                         # px-equivalent acceleration on runs of 3 visible frames
        a = np.linalg.norm(X[:, 2:] - 2 * X[:, 1:-1] + X[:, :-2], axis=-1) * f_mean / zs[:, 1:-1]
        ok = m[:, 2:] & m[:, 1:-1] & m[:, :-2]
        return a[ok]
    a0, a1 = accel(X0), accel(XS)
    pct = lambda x: " / ".join(f"{np.percentile(x, p):.2f}" for p in (50, 90))
    print(f"  reprojection of smoothed points  median / p90 {pct(rep)} px, > 5 px {100 * (rep > 5).mean():.1f}%")
    print(f"  depth change |log(z/z_DA3)|      median / p90 {pct(dz)}; behind a camera: {int((zs[m] <= 0).sum())}")
    print(f"  acceleration (px-eq./frame^2)    raw {pct(a0)}  ->  smoothed {pct(a1)}")
    stats = dict(sigma_depth=args.sigma_depth, sigma_acc=args.sigma_acc, cauchy_px=args.cauchy_px,
                 reproj_med=float(np.median(rep)), reproj_p90=float(np.percentile(rep, 90)), reproj_gt5=float((rep > 5).mean()),
                 dz_med=float(np.median(dz)), acc_raw=float(np.median(a0)), acc_smooth=float(np.median(a1)),
                 solve_seconds=t_solve, workers=args.workers, tracks=int(solved.sum()), obs=int(m.sum()))
    if args.out:
        xcam = np.einsum("tij,qtj->qti", np.transpose(R, (0, 2, 1)) / s[:, None, None], XS - C[None])
        xcam[~V] = np.nan
        out = {k: d[k] for k in d.files}
        out["tracks_xyz"] = xcam[None]
        out["tracks_world_smoothed"] = XS.astype(np.float32)
        np.savez_compressed(f"{O}/{args.out}", **out)
        json.dump(stats, open(f"{O}/{os.path.splitext(args.out)[0]}.json", "w"), indent=1)
        print(f"  wrote {O}/{args.out}")
