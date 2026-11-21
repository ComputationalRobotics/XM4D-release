#!/usr/bin/env python3
"""Global completion of the partially-observed EDG table produced by EDG_test_fast.py.

One SINGLE global optimisation over the whole M x T table of
EDG poses Q_{m,t} = (R_{m,t}, c_{m,t}) in SE(3), combining

  L_obs   -- dynamic-point observations via a rest-frame-free source->target warp
             Phi_k(t) = sum_{m in N_k} w_km [ R_{m,t} R_{m,s_k}^T (x_{k,s_k} - c_{m,s_k}) + c_{m,t} ]
             attached ONCE at each track's first visible frame s_k (N_k restricted to nodes observed at s_k
             i.e. (m,s_k) in Omega, w_km fixed there); a Huber robust kernel caps outlier tracks;
  L_arap  -- rest-frame-free ARAP: for edge (m,n), gaps delta in D, keep node n's coordinates in node m's
             local frame (R_{m,t}^T d_{mn,t}) and the edge length ||d_{mn,t}|| slowly varying in time;
  L_vel   -- 1st-difference temporal smoothness on c and on the rotation Lie increment omega=Log(R_{t+1}R_t^T);
  L_acc   -- 2nd-difference temporal smoothness (carries most of the burden for fast motion);
  L_init  -- confidence-weighted prior pulling the ORIGINALLY-OBSERVED cells (Omega) toward the local
             EDG_test_fast estimates (translation + geodesic rotation). Interpolated holes get NO init prior.
  L_p2d   -- 2D REPROJECTION: the warped dynamic points Phi_k(t) AND the EDG nodes (at their observed
             frames), projected back into each camera (inverse XM lift + per-frame pinhole intrinsics
             recovered from the stored tracks), must match their ORIGINAL 2D pixel tracks. Depth-free, so it
             pins the image-plane drift that the depth-noisy 3D lift cannot see.

min_Q  lambda_obs L_obs + lambda_arap L_arap + lambda_vel L_vel + lambda_acc L_acc + lambda_init L_init
each residual whitened by a robust scale sigma (est. from reliable cells) and normalised by its scalar count
N_j; L_obs and L_arap additionally pass through a Huber robust kernel rho so a single bad track/edge cannot
dominate.
Rotations live on SO(3) via the 6D continuous representation (Zhou et al.); holes are initialised by linear
(c) / geodesic (R) interpolation between the nearest observed cells, then everything is optimised jointly
with Adam + cosine LR decay.

Input : outputs/edg_test/edg_table.npz + edg_global.npz (EDG_test_fast.py) + edg_topology.npz + tracks + XM.
Output: outputs/edg_test/edg_completed.npz -- SAME schema as edg_global.npz, DENSE (holes filled).

Run (GPU torch lives in the `tapnet` env; no XM needed):
    python complete_edg_table.py
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

import utils.dataset_config as dataset_config
import utils.definitions
from utils.edg import scatter_solved_poses, scatter_solved_frame_vector
from utils.io import savez_compressed_fast

ITERS = 200               # Adam iterations (cosine LR decay from LR)
LR = 1e-2
ETA_COORD = 1.0           # weight of the ARAP coordinate residuals
ETA_LEN = 0.1             # weight of the ARAP edge-length residuals
SIG_ROT = 0.1             # rotation residual scale floor (radians)
HUBER_DELTA = 2.0         # Huber transition for L_obs/L_arap/L_p2d, in whitened residual std-devs (PDF: 1.5-2.5)
# after the solve, a node's cells before its first / after its last table cell follow its observed neighbours rigidly
# (Kabsch on their positions); config.json EDG_RIGID_ENDS
EDG_RIGID_ENDS = dataset_config.get_bool("EDG_RIGID_ENDS", False)


# --------------------------------------------------------------------------------------- data loading
def load_inputs():
    """Rebuild glob/valid (world-lifted observations, no XM needed) + topology + the fast-solver table."""
    out_dir = Path(utils.definitions.output_dir)
    tracks_path = Path(utils.definitions.dynamic_tracks_for_edg())
    data = np.load(str(tracks_path), allow_pickle=True)
    tracks_xyz = data["tracks_xyz"]                                   # (1,Q,T,3)
    occluded = data["occluded"]                                      # (1,Q,T)
    dyn_mask = data["dynamic_obs_mask"]                              # (Q,T)
    T, Q = tracks_xyz.shape[2], tracks_xyz.shape[1]

    rt = np.load(str(out_dir / "result_XM.npz"))
    R_raw, t_raw = scatter_solved_poses(rt, T)
    s_raw = scatter_solved_frame_vector(rt, "s", T, default=1.0)
    R_for_edg = R_raw.astype(np.float64).copy()
    if s_raw is not None:
        R_for_edg *= s_raw[:, None, None]
    glob = np.einsum("tij,qtj->qti", R_for_edg, tracks_xyz[0].astype(np.float64)) + t_raw[None, :, :]  # (Q,T,3)
    valid = (np.isfinite(glob).all(-1) & (occluded[0] == 0) & dyn_mask.astype(bool))                    # (Q,T)

    topo = np.load(out_dir / "edg_topology.npz")
    edg_nodes = np.asarray(topo["edg_nodes"], dtype=int)
    edg_edges = np.asarray(topo["edg_edges"], dtype=int).reshape(-1, 2)
    dyn_edg = np.asarray(topo["dyn_edg_edges"], dtype=int).reshape(-1, 2)

    tbl = np.load(out_dir / "edg_test" / "edg_table.npz")              # rows = edg_nodes (EDG_test_fast.py)
    table = np.asarray(tbl["table"], dtype=np.float64)               # [M,T,3] NaN=hole
    rotation = np.asarray(tbl["rotation"], dtype=np.float64)         # [M,T,3,3] NaN=hole
    weight = np.asarray(tbl["weight"], dtype=np.float64)             # [M,T] accumulated reliability
    filled = np.asarray(tbl["filled"], dtype=bool)                  # [M,T] position known
    rot_filled = np.asarray(tbl["rot_filled"], dtype=bool)         # [M,T] rotation known

    node_set = set(edg_nodes.tolist())
    node_loc = {int(n): i for i, n in enumerate(edg_nodes.tolist())}
    dyn_to_nodes = {}                                                 # skinning N_k as LOCAL node indices
    for d, n in dyn_edg.tolist():
        if int(n) in node_set:
            dyn_to_nodes.setdefault(int(d), []).append(node_loc[int(n)])
    # for the 2D reprojection loss: pixel tracks + per-frame pinhole intrinsics recovered from the stored
    # (camera-space, pixel) pairs -- exact by construction, no dependence on the intrinsics source
    xy = np.asarray(data["tracks_xy"], dtype=np.float64)[0]           # (Q,T,2) pixels
    K = fit_frame_intrinsics(tracks_xyz[0].astype(np.float64), xy, occluded[0].astype(bool), T)

    print(f"[load] Q={Q} T={T} | M={edg_nodes.size} nodes, {edg_edges.shape[0]} edges | "
          f"{len(dyn_to_nodes)} tracks skinned | table {100*filled.mean():.1f}% filled, "
          f"rot {100*rot_filled.mean():.1f}%")
    return dict(glob=glob, valid=valid, T=T, Q=Q, edg_nodes=edg_nodes, edg_edges=edg_edges,
                node_loc=node_loc, dyn_to_nodes=dyn_to_nodes, table=table, rotation=rotation,
                weight=weight, filled=filled, rot_filled=rot_filled,
                xy=xy, K=K, R_cam=R_raw.astype(np.float64), t_cam=t_raw.astype(np.float64),
                s_cam=(s_raw.astype(np.float64) if s_raw is not None else np.ones(T)))


def fit_frame_intrinsics(xyz_cam, xy, occ, T):
    """Per-frame pinhole [fx,fy,cx,cy] by least squares from camera-space tracks vs their pixel tracks
    (the tracking lifted with u=fx*X/Z+cx, v=fy*Y/Z+cy, so the fit is exact)."""
    K = np.full((T, 4), np.nan)
    for t in range(T):
        Z = xyz_cam[:, t, 2]
        sel = (~occ[:, t]) & np.isfinite(xyz_cam[:, t]).all(1) & np.isfinite(xy[:, t]).all(1) & (Z > 1e-9)
        if sel.sum() < 8:
            continue
        A = np.stack([xyz_cam[sel, t, 0] / Z[sel], np.ones(int(sel.sum()))], 1)
        fx, cx = np.linalg.lstsq(A, xy[sel, t, 0], rcond=None)[0]
        A[:, 0] = xyz_cam[sel, t, 1] / Z[sel]
        fy, cy = np.linalg.lstsq(A, xy[sel, t, 1], rcond=None)[0]
        K[t] = (fx, fy, cx, cy)
    good = np.where(np.isfinite(K).all(1))[0]
    if good.size == 0:
        raise RuntimeError("could not fit intrinsics from any frame")
    for t in np.where(~np.isfinite(K).all(1))[0]:                     # fill unfittable frames with nearest
        K[t] = K[good[np.argmin(np.abs(good - t))]]
    return K


# --------------------------------------------------------------------------- numpy SO(3) + interpolation init
def _so3_log_np(R):
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = np.arccos(cos)
    if th < 1e-6:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return th / (2.0 * np.sin(th)) * v


def _so3_exp_np(w):
    th = float(np.linalg.norm(w))
    if th < 1e-8:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def _slerp(Ra, Rb, u):
    return Ra @ _so3_exp_np(u * _so3_log_np(Ra.T @ Rb))


def _follow_neighbours(c, R, filled, edges):
    """Before a node's first / after its last observed cell, move it rigidly with its EDG neighbours frame by frame
    (from the known frame p = t -+ 1: c_m(t) = mean_n [c_n(t) + Q_n (c_m(p) - c_n(p))], Q_n = R_n(t) R_n(p)^T,
    R_m(t) = mean Q_n projected to SO(3) times R_m(p)); a held-constant end leaves a node behind on a moving object
    and the temporal terms make it too stiff to catch up. Sweeps repeat until no node
    gains a frame; a node with no known neighbour keeps its held-constant end."""
    M, T, _ = c.shape
    nbr = [[] for _ in range(M)]
    for a, b in edges.tolist():
        nbr[a].append(b); nbr[b].append(a)
    f = [np.where(filled[m])[0] for m in range(M)]
    known = filled.copy()
    for step, ts in ((-1, range(T - 2, -1, -1)), (1, range(1, T))):     # leading holes, then trailing holes
        changed = True
        while changed:
            changed = False
            for t in ts:
                p = t - step
                for m in range(M):
                    if known[m, t] or not known[m, p] or f[m].size == 0 or f[m][0] < t < f[m][-1]:
                        continue
                    ns = [n for n in nbr[m] if known[n, t] and known[n, p]]
                    if not ns:
                        continue
                    Q = [R[n, t] @ R[n, p].T for n in ns]
                    c[m, t] = np.mean([c[n, t] + q @ (c[m, p] - c[n, p]) for n, q in zip(ns, Q)], axis=0)
                    U, _, Vt = np.linalg.svd(np.sum(Q, axis=0))
                    U[:, -1] *= np.sign(np.linalg.det(U @ Vt))
                    R[m, t] = U @ Vt @ R[m, p]; known[m, t] = changed = True


def init_dense(table, rotation, filled, rot_filled, edges):
    """Dense starting point: linear interp for c, geodesic interp for R, between nearest observed cells;
    endpoints moved with the node's EDG neighbours `edges` (_follow_neighbours).
    Interpolated cells are NOT anchored (their init prior is off elsewhere)."""
    M, T, _ = table.shape
    c = np.zeros((M, T, 3), dtype=np.float64)
    for m in range(M):
        f = np.where(filled[m])[0]
        if f.size == 0:
            continue
        for d in range(3):
            c[m, :, d] = np.interp(np.arange(T), f, table[m, f, d])
    R = np.tile(np.eye(3), (M, T, 1, 1))
    for m in range(M):
        f = np.where(rot_filled[m])[0]
        if f.size == 0:
            continue
        for t in range(T):
            if rot_filled[m, t]:
                R[m, t] = rotation[m, t]
            elif t < f[0]:
                R[m, t] = rotation[m, f[0]]
            elif t > f[-1]:
                R[m, t] = rotation[m, f[-1]]
            else:
                a = f[f < t][-1]; b = f[f > t][0]
                R[m, t] = _slerp(rotation[m, a], rotation[m, b], (t - a) / (b - a))
    _follow_neighbours(c, R, filled, edges)
    return c, R


# ------------------------------------------------------------------------------------- torch SO(3) helpers
def rot6d_to_matrix(r6):
    """[...,6] (first two rotation columns) -> [...,3,3] via Gram-Schmidt (always in SO(3))."""
    a1, a2 = r6[..., 0:3], r6[..., 3:6]
    b1 = a1 / (a1.norm(dim=-1, keepdim=True) + 1e-9)
    a2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = a2 / (a2.norm(dim=-1, keepdim=True) + 1e-9)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)                          # columns b1,b2,b3


def matrix_to_rot6d(R):
    """[...,3,3] -> [...,6] : the first two columns."""
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def so3_log(R):
    """[...,3,3] -> [...,3] rotation vector (small-angle safe; theta~pi is rare with a close init)."""
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = ((tr - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    th = torch.arccos(cos)
    v = torch.stack([R[..., 2, 1] - R[..., 1, 2],
                     R[..., 0, 2] - R[..., 2, 0],
                     R[..., 1, 0] - R[..., 0, 1]], dim=-1)            # = 2 sin(th) * axis
    sin = torch.sin(th)
    factor = torch.where(sin.abs() < 1e-5, torch.full_like(th, 0.5), th / (2.0 * sin))
    return factor[..., None] * v


def huber(s, delta):
    """Robust penalty rho applied to a WHITENED SQUARED residual s = ||r||^2 / sigma^2: quadratic
    (=s) while ||r|| <= delta*sigma, linear beyond -> a single outlier contributes O(||r||) not O(||r||^2)."""
    r = torch.sqrt(s.clamp_min(1e-12))
    return torch.where(r <= delta, s, 2.0 * delta * r - delta * delta)


def robustscale(x, floor):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return floor
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(max(1.4826 * mad, floor))


def estimate_scales(c_init, R_init, filled, rot_filled, e, gaps, OB, phi_init, scene, sig_rot):
    """Robust residual scales from RELIABLE (originally-observed) init cells only. Estimating
    them from the interpolated holes would whiten the model error into the noise and unbalance the terms."""
    T = c_init.shape[1]
    fl = 1e-3 * scene
    em, en = e[:, 0], e[:, 1]
    d = c_init[en] - c_init[em]                                       # [E,T,3]
    fm, fn, rfm = filled[em], filled[en], rot_filled[em]
    coord, length = [], []
    for g in gaps:
        dt, dtd = d[:, :T - g], d[:, g:]
        both = fm[:, :T - g] & fn[:, :T - g] & fm[:, g:] & fn[:, g:]  # both endpoints observed at t and t+g
        length.append((np.linalg.norm(dt, axis=2) - np.linalg.norm(dtd, axis=2))[both])
        Rt, Rtd = R_init[em][:, :T - g], R_init[em][:, g:]
        cr = np.einsum("etji,etj->eti", Rt, dt) - np.einsum("etji,etj->eti", Rtd, dtd)
        coord.append(np.linalg.norm(cr[both & rfm[:, :T - g] & rfm[:, g:]], axis=1))
    # dynamic obs scale: only observations whose FULL neighbourhood was observed at both s_k and t
    m, s, t = OB["p_m"].cpu().numpy(), OB["p_s"].cpu().numpy(), OB["p_t"].cpu().numpy()
    po = OB["p_o"].cpu().numpy()
    pair_ok = filled[m, s] & filled[m, t] & rot_filled[m, s] & rot_filled[m, t]
    obs_ok = np.ones(OB["O"], dtype=bool)
    np.logical_and.at(obs_ok, po, pair_ok)                            # obs reliable iff every pair reliable
    res_x = np.linalg.norm(phi_init - OB["obs_x"].cpu().numpy(), axis=1)[obs_ok]
    # temporal position scales from reliable consecutive cells
    dc = np.linalg.norm(np.diff(c_init, axis=1), axis=2)[filled[:, 1:] & filled[:, :-1]]
    ac = np.linalg.norm(np.diff(c_init, n=2, axis=1), axis=2)[filled[:, 2:] & filled[:, :-2]]
    return dict(
        x=robustscale(res_x, fl), coord=robustscale(np.concatenate(coord), fl),
        length=robustscale(np.concatenate(length), fl), vc=robustscale(dc, fl), ac=robustscale(ac, fl),
        vR=sig_rot, aR=sig_rot, R0=sig_rot, c0=max(robustscale(res_x, fl), fl),
    )


# ----------------------------------------------------------------------------------- skinning / obs pairs
def build_skinning(D, c_init):
    """Fix N_k and w_km at each track's first visible frame s_k (inverse-distance, treated as constants).
    Returns per-track: s_k, x_{k,s_k}, and flat (node, weight) lists. Excludes EDG-node tracks (those are
    constrained by L_init). N_k is restricted to nodes whose SOURCE pose is observed at s_k
    ((m,s_k) in Omega) so the warp is anchored to a real cell, not an interpolated/free hole; a track with no
    such neighbour is dropped."""
    glob, valid, filled = D["glob"], D["valid"], D["filled"]
    node_set = set(D["edg_nodes"].tolist())
    tracks, dropped = [], 0
    for k, nodes in D["dyn_to_nodes"].items():
        if k in node_set:
            continue                                                 # node's own motion handled by L_init
        vk = np.where(valid[k])[0]
        if vk.size < 2:
            continue
        sk = int(vk[0])
        nb = np.asarray(nodes, dtype=int)
        nb = nb[filled[nb, sk]]                                       # (m,s_k) in Omega: source pose observed
        targets = vk[vk != sk]
        if nb.size == 0 or targets.size == 0:
            dropped += 1
            continue
        xsrc = glob[k, sk]
        dist = np.linalg.norm(c_init[nb, sk] - xsrc[None, :], axis=1)  # source pos at s_k (a real cell now)
        w = 1.0 / (dist + 1e-6)
        w = w / w.sum()
        tracks.append(dict(k=k, sk=sk, xsrc=xsrc, nb=nb, w=w, targets=targets))
    if dropped:
        print(f"[skin] dropped {dropped} tracks with no neighbour observed at their first frame s_k")
    return tracks


def pair_arrays(tracks, frames):
    """Observations = (track i, frame t) for t in frames[i], track by track; pairs = each observation with every
    neighbour of its track, in nb order. Returns per observation (track index, frame) and per pair (obs, node,
    weight, frame, source frame, source point)."""
    n_f = np.array([f.size for f in frames], dtype=np.int64)
    n_nb = np.array([tr["nb"].size for tr in tracks], dtype=np.int64)
    o_i = np.repeat(np.arange(len(tracks)), n_f)                      # track of each observation
    o_t = np.concatenate(frames).astype(np.int64)
    n_p = n_nb[o_i]                                                   # pairs of each observation
    p_o = np.repeat(np.arange(o_i.size), n_p)
    p_first = np.cumsum(n_p) - n_p                                    # first pair of each observation
    nb_first = np.cumsum(n_nb) - n_nb                                 # first neighbour of each track (flat)
    j = (nb_first[o_i] - p_first)[p_o] + np.arange(p_o.size)          # flat neighbour index of each pair
    p_i = o_i[p_o]
    nb = np.concatenate([tr["nb"] for tr in tracks]).astype(np.int64)
    w = np.concatenate([tr["w"] for tr in tracks])
    sk = np.array([tr["sk"] for tr in tracks], dtype=np.int64)
    xsrc = np.stack([tr["xsrc"] for tr in tracks])
    return o_i, o_t, p_o, nb[j], w[j], o_t[p_o], sk[p_i], xsrc[p_i]


def obs_pairs(tracks, glob, xy, device):
    """Flatten to per-(observation, neighbour) tensors so L_obs is one scatter-add. One observation = a
    (track, target-frame) pair; Phi is the neighbour-blend of the source->target warp. Also records each
    observation's frame + 2D pixel target (for the 2D reprojection loss on the warped point)."""
    o_i, o_t, p_o, p_m, p_w, p_t, p_s, p_xsrc = pair_arrays(tracks, [tr["targets"] for tr in tracks])
    o_k = np.array([tr["k"] for tr in tracks], dtype=np.int64)[o_i]
    T_ = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)
    return dict(O=int(o_i.size),
                obs_x=T_(glob[o_k, o_t], torch.float32), obs_xy=T_(xy[o_k, o_t], torch.float32),
                obs_t=T_(o_t, torch.long),
                p_o=T_(p_o, torch.long), p_m=T_(p_m, torch.long),
                p_w=T_(p_w, torch.float32), p_t=T_(p_t, torch.long),
                p_s=T_(p_s, torch.long), p_xsrc=T_(p_xsrc, torch.float32))


def make_projector(D, device):
    """Differentiable world -> pixel projection: x_cam = R_t^T (x - t_t) / s_t (inverse of the XM lift),
    then per-frame pinhole. Returns project(xw [N,3], frames [N]) -> [N,2] pixels."""
    Rc = torch.as_tensor(D["R_cam"], dtype=torch.float32, device=device)      # [T,3,3]
    tc = torch.as_tensor(D["t_cam"], dtype=torch.float32, device=device)      # [T,3]
    sc = torch.as_tensor(D["s_cam"], dtype=torch.float32, device=device)      # [T]
    K = torch.as_tensor(D["K"], dtype=torch.float32, device=device)           # [T,4] fx,fy,cx,cy

    def project(xw, tf):
        xc = torch.einsum("nji,nj->ni", Rc[tf], xw - tc[tf]) / sc[tf][:, None]
        Z = xc[..., 2].clamp_min(1e-6)                                        # in front of the camera
        u = K[tf, 0] * xc[..., 0] / Z + K[tf, 2]
        v = K[tf, 1] * xc[..., 1] / Z + K[tf, 3]
        return torch.stack([u, v], -1)

    return project


_SIDE_STREAMS = {}


def _side_gather(x, i, j, k):
    """x[i, j], on CUDA run on side stream k: its backward (run by autograd on the same stream) is then free to
    overlap with the rest of the backward. Same kernels, same summation order -> same result."""
    if not x.is_cuda:
        return x[i, j]
    cur = torch.cuda.current_stream(x.device)
    side = _SIDE_STREAMS.setdefault((x.device, k), torch.cuda.Stream(x.device))
    side.wait_stream(cur)
    with torch.cuda.stream(side):
        y = x[i, j]
    cur.wait_stream(side)
    y.record_stream(cur)                                              # allocated on side, used on cur
    return y


def warp_phi(c, R, p_o, p_m, p_w, p_t, p_s, p_xsrc, O):
    """Phi_k(t) = sum_m w_km [ R_{m,t} R_{m,s}^T (xsrc - c_{m,s}) + c_{m,t} ], as a scatter over pairs.
    The source-frame gathers' backward is a serial sum over every pair sharing a (node, s_k) cell (~190k pairs
    in one cell on teddy: most tracks start at frame 0), so they run on side streams (_side_gather)."""
    Rms = _side_gather(R, p_m, p_s, 0)                                # [P,3,3]
    Rmt = R[p_m, p_t]
    cms = _side_gather(c, p_m, p_s, 1)                                # [P,3]
    cmt = c[p_m, p_t]
    rel = torch.einsum("pji,pj->pi", Rms, p_xsrc - cms)              # R_{m,s}^T (xsrc - c_{m,s})
    warped = torch.einsum("pij,pj->pi", Rmt, rel) + cmt              # R_{m,t} rel + c_{m,t}
    contrib = p_w[:, None] * warped
    phi = torch.zeros((O, 3), dtype=c.dtype, device=c.device)
    phi.index_add_(0, p_o, contrib)
    return phi


# --------------------------------------------------------------------------------------------- optimisation
def complete(D, device):
    T = D["T"]; M = D["edg_nodes"].size
    table, rotation = D["table"], D["rotation"]
    filled, rot_filled, weight = D["filled"], D["rot_filled"], D["weight"]

    # ---- ARAP edge index arrays (local node indices)
    nl = D["node_loc"]
    e = np.array([[nl[a], nl[b]] for a, b in D["edg_edges"].tolist() if a in nl and b in nl], dtype=np.int64)

    c_init, R_init = init_dense(table, rotation, filled, rot_filled, e)
    scene = robustscale(np.linalg.norm(c_init[filled] - np.median(c_init[filled], 0), axis=1), 1e-6)

    # ---- fixed skinning + observation pairs
    tracks = build_skinning(D, c_init)
    OB = obs_pairs(tracks, D["glob"], D["xy"], device)
    print(f"[obs] {len(tracks)} tracks -> {OB['O']} observations, {OB['p_o'].numel()} (obs,neighbour) pairs")

    # ---- 2D reprojection targets: each EDG node at its directly-observed frames vs its own pixel track
    # (the dynamic side reuses the warped Phi per observation + OB["obs_xy"])
    project = make_projector(D, device)
    en_glob = D["edg_nodes"]
    n2d_ok = D["valid"][en_glob] & np.isfinite(D["xy"][en_glob]).all(-1)      # [M,T]
    n2d_m, n2d_t = np.where(n2d_ok)
    n2d = dict(m=torch.as_tensor(n2d_m, device=device), t=torch.as_tensor(n2d_t, device=device),
               xy=torch.as_tensor(D["xy"][en_glob][n2d_ok], dtype=torch.float32, device=device))
    print(f"[2d] reprojection targets: {n2d_m.size} node cells + {OB['O']} dyn observations")

    em = torch.as_tensor(e[:, 0], device=device); en = torch.as_tensor(e[:, 1], device=device)

    # ---- variables
    c = torch.tensor(c_init, dtype=torch.float32, device=device, requires_grad=True)
    R0t = torch.as_tensor(R_init, dtype=torch.float32, device=device)
    r6 = matrix_to_rot6d(R0t).clone().requires_grad_(True)

    # ---- init-prior targets + confidence (only on ORIGINALLY observed cells Omega)
    c0 = torch.as_tensor(np.where(filled[..., None], table, 0.0), dtype=torch.float32, device=device)
    mc = torch.as_tensor(filled, dtype=torch.float32, device=device)               # [M,T] translation Omega
    mR = torch.as_tensor(rot_filled, dtype=torch.float32, device=device)           # [M,T] rotation Omega
    R0omega = torch.as_tensor(np.where(rot_filled[..., None, None], rotation, np.eye(3)),
                              dtype=torch.float32, device=device)
    a = weight / max(np.median(weight[weight > 0]), 1e-9)                           # confidence ~1
    a = torch.as_tensor(np.clip(a, 0.0, 4.0), dtype=torch.float32, device=device)

    # ---- robust residual scales from the RELIABLE (originally observed) init cells
    # ARAP temporal gaps D (those shorter than the video): the long gaps keep a node's motion consistent over the
    # whole video, not only between neighbouring frames
    gaps = [g for g in (1, 4, 16, 64, 128, 256) if g < T]
    with torch.no_grad():
        phi_init = warp_phi(torch.as_tensor(c_init, dtype=torch.float32, device=device), R0t,
                            OB["p_o"], OB["p_m"], OB["p_w"], OB["p_t"], OB["p_s"], OB["p_xsrc"],
                            OB["O"]).cpu().numpy()
    sig = estimate_scales(c_init, R_init, filled, rot_filled, e, gaps, OB, phi_init, scene, SIG_ROT)
    with torch.no_grad():                                             # robust 2D pixel scale from the init
        ct = torch.as_tensor(c_init, dtype=torch.float32, device=device)
        r_node = (project(ct[n2d["m"], n2d["t"]], n2d["t"]) - n2d["xy"]).norm(dim=-1).cpu().numpy()
        r_dyn = (project(torch.as_tensor(phi_init, dtype=torch.float32, device=device), OB["obs_t"])
                 - OB["obs_xy"]).norm(dim=-1).cpu().numpy()
        sig["p2d"] = robustscale(np.concatenate([r_node, r_dyn]), 0.5)
        print(f"[2d] init reprojection error: node median={np.median(r_node):.2f}px "
              f"dyn median={np.median(r_dyn):.2f}px  -> sigma_2d={sig['p2d']:.2f}px")
    print("[scales] " + "  ".join(f"{k}={v:.4g}" for k, v in sig.items()) + f"  (scene~{scene:.4g})")

    lam = dict(obs=1.0, arap=1.0, vel=0.0, acc=1.0, init=3.0, p2d=1.0)
    opt = torch.optim.Adam([c, r6], lr=LR)

    def losses():
        R = rot6d_to_matrix(r6)                                       # [M,T,3,3]
        L = {}
        # L_init: confidence-weighted prior on the ORIGINALLY-observed cells only (Omega)
        rot_res = so3_log(R0omega.transpose(-1, -2) @ R).pow(2).sum(-1)    # ||Log(R0^T R)||^2  [M,T]
        L["init"] = ((a * mc * (c - c0).pow(2).sum(-1)).sum() / (sig["c0"] ** 2)
                     + (a * mR * rot_res).sum() / (sig["R0"] ** 2)) / max(float(mc.sum() * 3 + mR.sum() * 3), 1.0)
        # L_obs (robust Huber on the whitened residual)
        phi = warp_phi(c, R, OB["p_o"], OB["p_m"], OB["p_w"], OB["p_t"], OB["p_s"], OB["p_xsrc"], OB["O"])
        s_obs = (phi - OB["obs_x"]).pow(2).sum(-1) / (sig["x"] ** 2)  # [O] whitened squared residual per obs
        L["obs"] = huber(s_obs, HUBER_DELTA).sum() / max(OB["O"] * 3, 1)
        # L_p2d: 2D reprojection of the warped dynamic points AND the EDG nodes onto their pixel tracks
        # (depth-free -- pins the image-plane drift that the depth-noisy 3D lift cannot see)
        s_d2 = (project(phi, OB["obs_t"]) - OB["obs_xy"]).pow(2).sum(-1) / (sig["p2d"] ** 2)
        s_n2 = (project(c[n2d["m"], n2d["t"]], n2d["t"]) - n2d["xy"]).pow(2).sum(-1) / (sig["p2d"] ** 2)
        L["p2d"] = (huber(s_d2, HUBER_DELTA).sum() + huber(s_n2, HUBER_DELTA).sum()) \
            / max(2 * (OB["O"] + int(n2d["m"].numel())), 1)
        # L_arap over gaps (robust Huber on each coord/length residual)
        d = c[en] - c[em]                                            # [E,T,3]
        Rm = R[em]                                                   # [E,T,3,3]
        la = c.new_zeros(()); ncoord = nlen = 0
        for g in gaps:
            dt, dtd = d[:, :T - g], d[:, g:]
            Rt, Rtd = Rm[:, :T - g], Rm[:, g:]
            rc = torch.einsum("etji,etj->eti", Rt, dt) - torch.einsum("etji,etj->eti", Rtd, dtd)
            rl = dt.norm(dim=-1) - dtd.norm(dim=-1)
            s_c = rc.pow(2).sum(-1) / (sig["coord"] ** 2)            # [E,T-g] whitened coord residual
            s_l = rl.pow(2) / (sig["length"] ** 2)                   # [E,T-g] whitened length residual
            la = la + ETA_COORD * huber(s_c, HUBER_DELTA).sum() \
                    + ETA_LEN * huber(s_l, HUBER_DELTA).sum()
            ncoord += rc.numel(); nlen += rl.numel()
        L["arap"] = la / max(ncoord + nlen, 1)
        # temporal
        dc = c[:, 1:] - c[:, :-1]
        om = so3_log(R[:, 1:] @ R[:, :-1].transpose(-1, -2))          # [M,T-1,3]
        L["vel"] = (dc.pow(2).sum() / (sig["vc"] ** 2) + om.pow(2).sum() / (sig["vR"] ** 2)) \
            / max((M * (T - 1)) * 6, 1)
        acc = c[:, 2:] - 2 * c[:, 1:-1] + c[:, :-2]
        oma = om[:, 1:] - om[:, :-1]
        L["acc"] = (acc.pow(2).sum() / (sig["ac"] ** 2) + oma.pow(2).sum() / (sig["aR"] ** 2)) \
            / max((M * (T - 2)) * 6, 1)
        total = sum(lam[k] * L[k] for k in L)
        return total, L

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ITERS)
    for it in range(ITERS):
        opt.zero_grad()
        total, L = losses()
        total.backward()
        opt.step(); sched.step()
        if it % max(1, ITERS // 20) == 0 or it == ITERS - 1:
            print(f"[opt {it:4d}/{ITERS}] total={total.item():.4f}  " +
                  "  ".join(f"{k}={L[k].item():.4f}" for k in L), flush=True)

    with torch.no_grad():
        R_best = rot6d_to_matrix(r6)
        phi_f = warp_phi(c, R_best, OB["p_o"], OB["p_m"], OB["p_w"], OB["p_t"], OB["p_s"],
                         OB["p_xsrc"], OB["O"])
        r_node = (project(c[n2d["m"], n2d["t"]], n2d["t"]) - n2d["xy"]).norm(dim=-1).median().item()
        r_dyn = (project(phi_f, OB["obs_t"]) - OB["obs_xy"]).norm(dim=-1).median().item()
        print(f"[2d] final reprojection median: node={r_node:.2f}px dyn={r_dyn:.2f}px")
        c_fin = c.detach().cpu().numpy()
        R_fin = rot6d_to_matrix(r6).detach().cpu().numpy()
    return c_fin, R_fin, tracks


def rigid_ends(c, R, filled, k=6):
    """Before a node's first / after its last table cell nothing observes it: replace those cells by a rigid follow of its
    observed neighbourhood: the k nodes nearest to it at the boundary frame b that are table-observed at both b and t,
    Kabsch on their solved POSITIONS b -> t (the rotation table is mostly interpolated), applied to the node's position and
    rotation at b."""
    M, T, _ = c.shape; n = 0
    for m in range(M):
        f = np.where(filled[m])[0]
        if f.size == 0:
            continue
        for ts, b in ((range(f[0] - 1, -1, -1), f[0]), (range(f[-1] + 1, T), f[-1])):
            cand = np.array([j for j in range(M) if j != m and filled[j, b]], dtype=np.int64)
            if cand.size < 3:
                continue
            cand = cand[np.argsort(np.linalg.norm(c[cand, b] - c[m, b], axis=1))]
            for t in ts:
                nb = cand[filled[cand, t]][:k]
                if nb.size < 3:
                    continue
                P, Q = c[nb, b].astype(np.float64), c[nb, t].astype(np.float64); mp, mq = P.mean(0), Q.mean(0)
                U, _, Vt = np.linalg.svd((Q - mq).T @ (P - mp)); Dg = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
                Rk = U @ Dg @ Vt
                c[m, t] = Rk @ (c[m, b] - mp) + mq; R[m, t] = Rk @ R[m, b]; n += 1
    return n


# --------------------------------------------------------------------------- warp dynamic points for output
def warp_dyn_output(tracks, c_fin, R_fin, D, device):
    """Deform every skinned track over its observed lifetime [min V_k, max V_k] with the COMPLETED table,
    filling occlusion holes. Returns dyn_ids [Nd], dyn_reconstruct [Nd,T,3] (NaN outside the lifetime)."""
    valid = D["valid"]; T = D["T"]
    c = torch.as_tensor(c_fin, dtype=torch.float32, device=device)
    R = torch.as_tensor(R_fin, dtype=torch.float32, device=device)
    dyn_ids = np.array([tr["k"] for tr in tracks], dtype=np.int64)
    vt = valid[dyn_ids]                                               # every skinned track has >= 2 valid frames
    lo, hi = vt.argmax(1), T - 1 - vt[:, ::-1].argmax(1)
    o_i, o_t, p_o, p_m, p_w, p_t, p_s, p_xsrc = pair_arrays(tracks, [np.arange(a, b + 1) for a, b in zip(lo, hi)])
    Tn = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)
    phi = warp_phi(c, R, Tn(p_o, torch.long), Tn(p_m, torch.long), Tn(p_w, torch.float32),
                   Tn(p_t, torch.long), Tn(p_s, torch.long), Tn(p_xsrc, torch.float32), int(o_i.size)).cpu().numpy()
    dyn = np.full((len(tracks), T, 3), np.nan, dtype=np.float64)
    dyn[o_i, o_t] = phi
    return dyn_ids, dyn


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")
    D = load_inputs()
    c_fin, R_fin, tracks = complete(D, torch.device(device))
    if EDG_RIGID_ENDS:
        print(f"[rigid-ends] {rigid_ends(c_fin, R_fin, D['filled'])} node-frames before/after a node's table cells follow its neighbours rigidly")
    dyn_ids, dyn_reconstruct = warp_dyn_output(tracks, c_fin, R_fin, D, torch.device(device))

    node_ids = D["edg_nodes"]
    ns = set(node_ids.tolist())
    edges = np.array([[a, b] for a, b in D["edg_edges"].tolist() if a in ns and b in ns],
                     dtype=np.int64).reshape(-1, 2)

    out = Path(utils.definitions.output_dir) / "edg_test" / "edg_completed.npz"
    completed = dict(node_ids=node_ids, edg_reconstruct=c_fin, edg_rotation=R_fin,
                     edg_edges=edges, dyn_ids=dyn_ids, dyn_reconstruct=dyn_reconstruct,
                     n_tracks=int(D["Q"]))                             # track-id space stamp (train_4dgs_edg.py checks it)
    pool = ThreadPoolExecutor(1)                                      # zlib releases the GIL: written while the
    written = pool.submit(savez_compressed_fast, out, **completed)    # results_all update below runs

    # Also push this completed EDG into results_all.npz (the EDG the 4DGS trainer reads).
    import shutil
    from completed_to_results import convert
    ra = Path(utils.definitions.output_dir) / "results_all.npz"
    if not ra.exists():                                              # no base results_all.npz yet: start from the camera solve's
        rt = np.load(Path(utils.definitions.output_dir) / "result_XM.npz")   # static points
        np.savez_compressed(ra, static_landmarks=rt["p"].T.astype(np.float64) if "p" in rt else None)
        print(f"[results_all] created {ra.name} with the {rt['p'].shape[1] if 'p' in rt else 0} static points of result_XM.npz")
    bak = ra.with_name("results_all_preEDGupdate.npz")
    if not bak.exists():
        shutil.copy2(ra, bak)
        print(f"[results_all] backed up existing base -> {bak.name}")
    convert(completed, ra, ra)                                       # overwrite results_all.npz EDG in place
    print(f"[results_all] updated {ra.name} with this completed EDG (the EDG the 4DGS trainer reads). "
          f"NOTE: re-run train_4dgs_edg.py to rebuild the dynamic "
          f"binding (dynamic_edg.npz) against this EDG before training.")
    written.result(); pool.shutdown()
    print(f"\n[done] {out}  ({node_ids.size} nodes, DENSE, T={D['T']}, {dyn_ids.size} dyn tracks)")


if __name__ == "__main__":
    main()
