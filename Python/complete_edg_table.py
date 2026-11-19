#!/usr/bin/env python3
"""Global completion of the partially-observed EDG table produced by EDG_test_fast.py.

Implements Notes/edg_table_completion(1).pdf: one SINGLE global optimisation over the whole M x T table of
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

BEFORE solving, the input table is CLEANED cell-by-cell (same rules as visualize_edg_filtered_open3d.py):
filter 1 deletes cells that jump from their track's smoothed trajectory (--smooth-win/--smooth-sigma);
filter 2 deletes cells whose 2D projection is farther than --reproj-del-px from the node's initial 2D
observation or falls outside the (dilated) dynamic mask. Deleted cells become holes -- no init prior, no
skinning anchor -- and the global solve re-fills them.

min_Q  lambda_obs L_obs + lambda_arap L_arap + lambda_vel L_vel + lambda_acc L_acc + lambda_init L_init
each residual whitened by a robust scale sigma (est. from reliable cells) and normalised by its scalar count
N_j; L_obs and L_arap additionally pass through a Huber robust kernel rho so a single bad track/edge cannot
dominate. AFTER solving, a track whose solved trajectory fails the same 2D criterion in more than
--post-del-frac (default 80%) of its evaluable frames is deleted WHOLE from the output; and the BAD ("red")
EDG nodes -- whose skinning-weighted fraction of render-colour-OUTLIER dynamic points is >= --bad-node-frac
(the visualize_edg_filtered_open3d --edg-quality classification) -- are DELETED, their dynamic points
reassigned to the surviving nodes (weights renormalized; fully-orphaned points re-skinned to the nearest
surviving nodes) and re-warped.
Rotations live on SO(3) via the 6D continuous representation (Zhou et al.); holes are initialised by linear
(c) / geodesic (R) interpolation between the nearest observed cells, then everything is optimised jointly
with Adam + cosine LR decay.

Input : outputs/edg_test/edg_table.npz + edg_global.npz (EDG_test_fast.py) + edg_topology.npz + tracks + XM.
Output: outputs/edg_test/edg_completed.npz -- SAME schema as edg_global.npz, DENSE (holes filled), so
        `visualize_edg_solution_open3d.py --npz .../edg_completed.npz --no-scale-search` shows it directly.

Run (GPU torch lives in the `tapnet` env; no XM needed):
    XM_DATASET_NAME=<scene> conda run -n tapnet python complete_edg_table.py --merged [--iters 1000]
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

import utils.definitions
from utils.edg import scatter_solved_poses, scatter_solved_frame_vector
from utils.io import savez_compressed_fast


# --------------------------------------------------------------------------------------- data loading
def load_inputs(merged, smoothed=True, edges_npz=None, table_path=None):
    """Rebuild glob/valid (world-lifted observations, no XM needed) + topology + the fast-solver table."""
    root = Path(utils.definitions.rootpath)
    out_dir = Path(utils.definitions.output_dir)
    if merged:
        name = "tracks_dynamic_merged_smoothed.npz" if smoothed else "tracks_dynamic_merged.npz"
        tracks_path = out_dir / name                                  # smoothed holds the 3D (merge is 2D-only)
        if smoothed and not tracks_path.exists():
            tracks_path = out_dir / "tracks_dynamic_merged.npz"
            print("[complete] WARNING: smoothed file missing -- raw merged is 2D-only (tracks_xyz all-NaN); "
                  "run optimize_dynamic_tracks.py first.")
    else:
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
    if edges_npz:                                                      # rebuilt topology (rebuild_edg_edges.py)
        edg_edges = np.asarray(np.load(edges_npz)["edg_edges"], dtype=int).reshape(-1, 2)
        print(f"[edges] using rebuilt edges from {edges_npz}: {edg_edges.shape[0]} edges")
    dyn_edg = np.asarray(topo["dyn_edg_edges"], dtype=int).reshape(-1, 2)

    tbl = np.load(table_path if table_path else out_dir / "edg_test" / "edg_table.npz")
    tbl_ids = np.asarray(tbl["node_ids"], dtype=int)
    if tbl_ids.shape == edg_nodes.shape and (tbl_ids == edg_nodes).all():
        table = np.asarray(tbl["table"], dtype=np.float64)           # [M,T,3] NaN=hole
        rotation = np.asarray(tbl["rotation"], dtype=np.float64)     # [M,T,3,3] NaN=hole
        weight = np.asarray(tbl["weight"], dtype=np.float64)         # [M,T] accumulated reliability
        filled = np.asarray(tbl["filled"], dtype=bool)              # [M,T] position known
        rot_filled = np.asarray(tbl["rot_filled"], dtype=bool)     # [M,T] rotation known
    else:
        # COARSE (e.g. XM-300) table over a DIFFERENT node set -> scatter onto the DENSE topology by global
        # track id: matched nodes inherit the XM solution (L_init anchor); unmatched dense nodes are
        # obs-seeded from glob (their own world-lifted observations) so every node has a valid init + anchor
        # (mimics EDG_test_fast --seed-obs). The global solve then completes/refines all nodes.
        M = edg_nodes.size
        table = np.full((M, T, 3), np.nan); rotation = np.full((M, T, 3, 3), np.nan)
        weight = np.zeros((M, T)); filled = np.zeros((M, T), bool); rot_filled = np.zeros((M, T), bool)
        src = {int(g): i for i, g in enumerate(tbl_ids.tolist())}
        ct = np.asarray(tbl["table"], np.float64); cr = np.asarray(tbl["rotation"], np.float64)
        cw = np.asarray(tbl["weight"], np.float64); cf = np.asarray(tbl["filled"], bool); crf = np.asarray(tbl["rot_filled"], bool)
        hit = 0
        for j, g in enumerate(edg_nodes.tolist()):
            i = src.get(int(g))
            if i is not None:                                        # matched -> XM-solved pos + rotation
                table[j], rotation[j], weight[j], filled[j], rot_filled[j] = ct[i], cr[i], cw[i], cf[i], crf[i]; hit += 1
            else:                                                    # unmatched -> obs-seed from world obs
                v = valid[g]
                table[j, v] = glob[g, v]; filled[j] = v; weight[j, v] = 1.0
        print(f"[scatter] coarse table {tbl_ids.size} nodes -> dense {M}: {hit} XM-seeded, {M-hit} obs-seeded")

    node_set = set(edg_nodes.tolist())
    node_loc = {int(n): i for i, n in enumerate(edg_nodes.tolist())}
    dyn_to_nodes = {}                                                 # skinning N_k as LOCAL node indices
    for d, n in dyn_edg.tolist():
        if int(n) in node_set:
            dyn_to_nodes.setdefault(int(d), []).append(node_loc[int(n)])
    # for the 2D reprojection loss: pixel tracks + per-frame pinhole intrinsics recovered from the stored
    # (camera-space, pixel) pairs -- exact by construction, no dependence on the DA3/GT intrinsics source
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


# ------------------------------------------------------------------- pre-solve table cleaning (per cell)
def _smooth_traj_nan(arr, win):
    """NaN-aware centred moving average of a [T,3] trajectory. At the clip ends the window shrinks symmetrically
    (half-width min(half, t, T-1-t); user 2026-11-16): a cut-off one-sided window lags a moving track, so every fast
    node used to look like an outlier in the first / last frames and its table cells were deleted there."""
    half = max(1, int(win)) // 2
    T = arr.shape[0]
    fin = np.isfinite(arr).all(axis=1)
    vals = np.where(fin[:, None], arr, 0.0)
    cs = np.vstack([np.zeros((1, 3)), np.cumsum(vals, axis=0)])
    cc = np.concatenate([[0.0], np.cumsum(fin.astype(np.float64))])
    t = np.arange(T)
    h = np.minimum(half, np.minimum(t, T - 1 - t))
    lo, hi = t - h, t + h + 1
    num = cs[hi] - cs[lo]
    cnt = cc[hi] - cc[lo]
    out = np.full_like(arr, np.nan)
    ok = fin & (cnt > 0)
    out[ok] = num[ok] / cnt[ok, None]
    return out


def _load_dyn_mask_images(T, dilate_px):
    """(T,H,W) bool dynamic-mask images (dilated), or None when unavailable."""
    import glob as _glob
    mask_dir = getattr(utils.definitions, "mask_dir", None)
    if mask_dir is None or not Path(str(mask_dir)).exists():
        print(f"[clean] no mask dir ({mask_dir}); skipping the outside-mask deletion")
        return None
    try:
        import cv2
        from utils.masks import load_dynamic_masks, dilate_dynamic_masks   # torch-free (utils.track_tapnet imports tapnet)
        image_paths = sorted(_glob.glob(str(Path(utils.definitions.video_dir) / utils.definitions.image_pattern)))
        H, W = cv2.imread(image_paths[0]).shape[:2]
        masks = load_dynamic_masks(str(mask_dir), utils.definitions.mask_pattern, t_expected=T,
                                   mask_mode="dynamic_positive", target_size=(W, H))
        if dilate_px > 0:
            masks = dilate_dynamic_masks(masks, int(dilate_px))
        masks = np.asarray(masks).astype(bool)
        print(f"[clean] loaded {masks.shape[0]} dynamic masks ({masks.shape[2]}x{masks.shape[1]}, "
              f"dilated {dilate_px}px)")
        return masks
    except Exception as e:
        print(f"[clean] cannot load dynamic masks ({e}); skipping the outside-mask deletion")
        return None


def _bad_2d_cells(P, D, masks, del_px):
    """The shared 2D criterion, per cell, for ANY [M,T,3] node-position array P (input table or solved
    result): a cell is BAD when its projected pixel is farther than `del_px` from the node's initial 2D
    observation at that frame, or falls OUTSIDE the (dilated) dynamic mask. Returns (bad [M,T],
    evaluable [M,T], n_far, n_out) -- `evaluable` marks the cells where at least one rule could be tested."""
    en = D["edg_nodes"]
    M, T = P.shape[0], P.shape[1]
    bad = np.zeros((M, T), bool)
    ev = np.zeros((M, T), bool)
    n_far = n_out = 0
    for f in range(T):
        xc = ((P[:, f] - D["t_cam"][f]) @ D["R_cam"][f]) / D["s_cam"][f]   # R^T (x - t) / s
        Z = xc[:, 2]
        solved = np.isfinite(xc).all(1) & (Z > 1e-9)
        if not solved.any():
            continue
        fx, fy, cx, cy = D["K"][f]
        u = fx * xc[:, 0] / np.where(Z > 0, Z, 1.0) + cx
        v = fy * xc[:, 1] / np.where(Z > 0, Z, 1.0) + cy
        if del_px > 0:
            obs = D["xy"][en, f]
            ok_o = D["valid"][en, f] & np.isfinite(obs).all(1) & solved
            far = ok_o & (np.hypot(u - obs[:, 0], v - obs[:, 1]) > del_px)
            n_far += int(far.sum())
            bad[:, f] |= far
            ev[:, f] |= ok_o
        if masks is not None:
            Hm, Wm = masks.shape[1], masks.shape[2]
            ui = np.round(np.nan_to_num(u, nan=-1.0)).astype(int)
            vi = np.round(np.nan_to_num(v, nan=-1.0)).astype(int)
            inb = solved & (ui >= 0) & (ui < Wm) & (vi >= 0) & (vi < Hm)
            mv = masks[f][np.clip(vi, 0, Hm - 1), np.clip(ui, 0, Wm - 1)]
            out = solved & ~(inb & mv)                                # out of bounds OR outside the mask
            n_out += int(out.sum())
            bad[:, f] |= out
            ev[:, f] |= solved
    return bad, ev, n_far, n_out


def post_delete_tracks(c_fin, D, args, masks):
    """AFTER solving: the same 2D criterion, but per TRACK -- a node whose SOLVED trajectory is bad (too far
    from its observation / outside the dynamic mask) in more than --post-del-frac of its evaluable frames is
    deleted WHOLE from the output. Returns a keep mask [M]."""
    bad, ev, n_far, n_out = _bad_2d_cells(c_fin, D, masks, args.reproj_del_px)
    denom = np.maximum(ev.sum(1), 1)
    frac = np.where(ev.any(1), bad.sum(1) / denom, 0.0)
    kill = frac > args.post_del_frac
    en = D["edg_nodes"]
    print(f"[post] track-level 2D check on the SOLVED table ({n_far} far cells, {n_out} outside-mask cells): "
          f"deleting {int(kill.sum())}/{en.size} whole tracks bad in > {100 * args.post_del_frac:.0f}% "
          f"of evaluable frames")
    for i in np.argsort(frac)[::-1][:8]:
        if kill[i]:
            print(f"       gid={int(en[i])}: bad {100 * frac[i]:.0f}% of {int(ev[i].sum())} evaluable frames")
    return ~kill


def clean_table(D, args, masks):
    """DELETE bad cells from the input table BEFORE solving (same per-cell rules as the filtered viewer):
      filter 1 (--smooth-win/--smooth-sigma): cell's residual to the track's NaN-aware moving average is a
        pooled robust outlier (> median + sigma * 1.4826*MAD over all cells);
      filter 2 (--reproj-del-px / --mask-dilate-px / --no-mask-del): the cell's PROJECTED pixel is farther
        than --reproj-del-px from the node's initial 2D observation at that frame, or falls OUTSIDE the
        (dilated) dynamic mask.
    Deleted cells become HOLES (table/rotation NaN, filled/rot_filled False): they get no init prior, no
    skinning source anchor, and are excluded from the scale estimation -- the global solve re-fills them
    from ARAP/temporal/observation terms instead of being pulled toward the bad values."""
    table, filled = D["table"], D["filled"]
    M, T = filled.shape
    bad = np.zeros((M, T), bool)

    if args.smooth_win > 0:                                            # ---- filter 1: smooth-error
        res = np.full((M, T), np.nan)
        for i in range(M):
            fin = np.isfinite(table[i]).all(1)
            if fin.sum() < max(3, args.smooth_win):
                continue
            res[i] = np.linalg.norm(table[i] - _smooth_traj_nan(table[i], args.smooth_win), axis=1)
        pooled = res[np.isfinite(res)]
        if pooled.size:
            med = float(np.median(pooled))
            thr = med + float(args.smooth_sigma) * 1.4826 * float(np.median(np.abs(pooled - med)))
            m1 = np.isfinite(res) & (res > thr)
            bad |= m1
            print(f"[clean] filter1 smooth-error (win={args.smooth_win} sigma={args.smooth_sigma}): "
                  f"{int(m1.sum())} cells above {thr:.5f}")

    if args.reproj_del_px > 0 or masks is not None:                    # ---- filter 2: 2D projection
        m2, _, n_far, n_out = _bad_2d_cells(table, D, masks, args.reproj_del_px)
        bad |= m2
        print(f"[clean] filter2 reproj-2d: {int(m2.sum())} cells ({n_far} farther than "
              f"{args.reproj_del_px:.0f}px from the observation, {n_out} outside the dynamic mask)")

    bad &= filled
    frac0 = filled.mean()
    table[bad] = np.nan
    filled[bad] = False
    D["rotation"][bad] = np.nan
    D["rot_filled"][bad] = False
    empty = int((~filled.any(1)).sum())
    print(f"[clean] deleted {int(bad.sum())} table cells: filled {100 * frac0:.1f}% -> "
          f"{100 * filled.mean():.1f}%"
          + (f"  (WARNING: {empty} nodes now have NO filled cells)" if empty else ""))


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
    and the temporal terms make it too stiff to catch up (car-roundabout, user 2026-11-16). Sweeps repeat until no node
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


def init_dense(table, rotation, filled, rot_filled, edges=None):
    """Dense starting point: linear interp for c, geodesic interp for R, between nearest observed cells;
    endpoints held constant, or with `edges` moved with the node's EDG neighbours (_follow_neighbours).
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
    if edges is not None:
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
    """The PDF's robust penalty rho applied to a WHITENED SQUARED residual s = ||r||^2 / sigma^2: quadratic
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
    """Robust residual scales from RELIABLE (originally-observed) init cells only (PDF section 6). Estimating
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
    constrained by L_init). Per the PDF, N_k is restricted to nodes whose SOURCE pose is observed at s_k
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
    weight, frame, source frame, source point) -- the order the old nested append loops produced."""
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
def complete(D, args, device):
    T = D["T"]; M = D["edg_nodes"].size
    table, rotation = D["table"], D["rotation"]
    filled, rot_filled, weight = D["filled"], D["rot_filled"], D["weight"]

    # ---- ARAP edge index arrays (local node indices)
    nl = D["node_loc"]
    e = np.array([[nl[a], nl[b]] for a, b in D["edg_edges"].tolist() if a in nl and b in nl], dtype=np.int64)

    c_init, R_init = init_dense(table, rotation, filled, rot_filled, None if args.hold_ends else e)
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
    E = e.shape[0]

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

    # ---- robust residual scales from the RELIABLE (originally observed) init cells (PDF section 6)
    gaps = args.gaps if args.gaps else [g for g in (1, 4, 16, 64, 128, 256) if g < T]
    with torch.no_grad():
        phi_init = warp_phi(torch.as_tensor(c_init, dtype=torch.float32, device=device), R0t,
                            OB["p_o"], OB["p_m"], OB["p_w"], OB["p_t"], OB["p_s"], OB["p_xsrc"],
                            OB["O"]).cpu().numpy()
    sig = estimate_scales(c_init, R_init, filled, rot_filled, e, gaps, OB, phi_init, scene, args.sig_rot)
    with torch.no_grad():                                             # robust 2D pixel scale from the init
        ct = torch.as_tensor(c_init, dtype=torch.float32, device=device)
        r_node = (project(ct[n2d["m"], n2d["t"]], n2d["t"]) - n2d["xy"]).norm(dim=-1).cpu().numpy()
        r_dyn = (project(torch.as_tensor(phi_init, dtype=torch.float32, device=device), OB["obs_t"])
                 - OB["obs_xy"]).norm(dim=-1).cpu().numpy()
        sig["p2d"] = robustscale(np.concatenate([r_node, r_dyn]), 0.5)
        print(f"[2d] init reprojection error: node median={np.median(r_node):.2f}px "
              f"dyn median={np.median(r_dyn):.2f}px  -> sigma_2d={sig['p2d']:.2f}px")
    print("[scales] " + "  ".join(f"{k}={v:.4g}" for k, v in sig.items()) + f"  (scene~{scene:.4g})")

    lam = dict(obs=args.lambda_obs, arap=args.lambda_arap, vel=args.lambda_vel,
               acc=args.lambda_acc, init=args.lambda_init, p2d=args.lambda_2d)
    opt = torch.optim.Adam([c, r6], lr=args.lr)

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
        L["obs"] = huber(s_obs, args.huber_delta).sum() / max(OB["O"] * 3, 1)
        # L_p2d: 2D reprojection of the warped dynamic points AND the EDG nodes onto their pixel tracks
        # (depth-free -- pins the image-plane drift that the depth-noisy 3D lift cannot see)
        s_d2 = (project(phi, OB["obs_t"]) - OB["obs_xy"]).pow(2).sum(-1) / (sig["p2d"] ** 2)
        s_n2 = (project(c[n2d["m"], n2d["t"]], n2d["t"]) - n2d["xy"]).pow(2).sum(-1) / (sig["p2d"] ** 2)
        L["p2d"] = (huber(s_d2, args.huber_delta).sum() + huber(s_n2, args.huber_delta).sum()) \
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
            la = la + args.eta_coord * huber(s_c, args.huber_delta).sum() \
                    + args.eta_len * huber(s_l, args.huber_delta).sum()
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

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters)
    for it in range(args.iters):
        opt.zero_grad()
        total, L = losses()
        total.backward()
        opt.step(); sched.step()
        if it % max(1, args.iters // 20) == 0 or it == args.iters - 1:
            print(f"[opt {it:4d}/{args.iters}] total={total.item():.4f}  " +
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


# ------------------------------------------------- bad ("red") EDG-node deletion + dynamic reassignment
def node_bad_fraction(node_ids, c_fin, R_fin, dyn_ids, dyn_rec, D):
    """Per-node skinning-weighted fraction of attached dynamic points that are render-colour OUTLIERS --
    the SAME classification as visualize_edg_filtered_open3d.py --edg-quality (its RED nodes). Uses the
    shared edg_viz_common scorer on the just-completed arrays. NaN = no dynamic points attached."""
    import glob as _glob
    from types import SimpleNamespace
    from edg_viz_common import _build_dyn_skin, detect_render_color_outliers
    sp, pids = _build_dyn_skin(dyn_ids, dyn_rec, node_ids, c_fin, R_fin)
    if sp is None:
        print("[bad-nodes] no skinnable dynamic points; skipping the bad-node deletion")
        return None
    image_paths = sorted(_glob.glob(str(Path(utils.definitions.video_dir) / utils.definitions.image_pattern)))
    ctx = dict(image_paths=image_paths, valid=D["valid"], R=D["R_cam"], t=D["t_cam"],
               s=D["s_cam"], K=D["K"])
    scorer = SimpleNamespace(render_color_space="hue", render_color_thresh=30.0, render_color_quantile=0.5,
                             render_min_span=5, render_min_obs=3, render_patch=2, render_sat_min=40)
    flag, _score = detect_render_color_outliers(sp, pids, dyn_ids, dyn_rec, ctx, scorer)
    bad_w = np.zeros(node_ids.size); tot_w = np.zeros(node_ids.size)
    np.add.at(tot_w, sp["nl"], sp["w"])
    np.add.at(bad_w, sp["nl"], sp["w"] * flag[sp["p"]].astype(np.float64))
    return np.where(tot_w > 1e-9, bad_w / np.maximum(tot_w, 1e-9), np.nan)


def reassign_tracks(tracks, keep, c_fin_full, k_new=4):
    """Repoint every skinned track at the SURVIVING nodes after a node deletion: drop deleted neighbours
    and renormalize the weights; a track whose WHOLE neighbourhood was deleted is REASSIGNED to its k_new
    nearest surviving nodes at its source frame (inverse-distance, the build_skinning rule; the completed
    table is dense so every surviving node has a pose there). nb is remapped to the subset numbering."""
    new_idx = np.cumsum(keep) - 1                                      # old local -> new local
    surv_old = np.where(keep)[0]
    n_renorm = n_reassign = 0
    for tr in tracks:
        nb_old = tr["nb"]; m = keep[nb_old]
        if m.any():
            if not m.all():
                n_renorm += 1
            w = tr["w"][m]
            tr["w"] = w / w.sum()
            tr["nb"] = new_idx[nb_old[m]]
        else:                                                          # whole neighbourhood deleted
            d = np.linalg.norm(c_fin_full[surv_old, tr["sk"]] - tr["xsrc"][None, :], axis=1)
            sel = np.argsort(d)[: int(min(k_new, surv_old.size))]
            w = 1.0 / (d[sel] + 1e-6)
            tr["nb"] = new_idx[surv_old[sel]]
            tr["w"] = w / w.sum()
            n_reassign += 1
    print(f"[bad-nodes] reassigned dynamic points: {n_renorm} tracks renormalized (lost some neighbours), "
          f"{n_reassign} fully reassigned to their {int(k_new)} nearest surviving nodes")
    return tracks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged", action="store_true", help="use the merged tracks (match EDG_test_fast --merged)")
    ap.add_argument("--smoothed", action=argparse.BooleanOptionalAction, default=True,
                    help="use optimize_dynamic_tracks' output (holds the 3D; raw merged is 2D-only now). Default on.")
    ap.add_argument("--edges-npz", default=None,
                    help="optionally override the EDG edges with a rebuilt set "
                         "(outputs/edg_edges_rebuilt.npz from rebuild_edg_edges.py); default = edg_topology.npz")
    ap.add_argument("--iters", type=int, default=200)    # 400 lowered the final reprojection median by only ~0.02 px
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--gaps", type=int, nargs="+", default=None,
                    help="ARAP temporal gaps D (default 1 4 16 64 128 256, those shorter than the video): the long gaps keep a node's "
                         "motion consistent over the whole video, not only between neighbouring frames")
    ap.add_argument("--lambda-obs", type=float, default=1.0)
    ap.add_argument("--lambda-arap", type=float, default=1.0)
    ap.add_argument("--lambda-vel", type=float, default=0.0)
    ap.add_argument("--lambda-acc", type=float, default=1.0,
                    help="second-difference smoothness of every node trajectory (the table from the segment solves zig-zags)")
    ap.add_argument("--lambda-init", type=float, default=3.0)
    ap.add_argument("--hold-ends", action="store_true",
                    help="hold a node constant before its first / after its last observed cell (old init) instead of "
                         "moving it with its EDG neighbours")
    ap.add_argument("--lambda-2d", type=float, default=1.0,
                    help="2D reprojection loss (dyn warped points + EDG nodes vs their pixel tracks); 0 = off")
    ap.add_argument("--eta-coord", type=float, default=1.0)
    ap.add_argument("--eta-len", type=float, default=0.1)
    ap.add_argument("--sig-rot", type=float, default=0.1, help="rotation residual scale floor (radians)")
    ap.add_argument("--huber-delta", type=float, default=2.0,
                    help="Huber transition for L_obs/L_arap, in whitened residual std-devs (PDF: 1.5-2.5)")
    ap.add_argument("--smooth-win", type=int, default=0,
                    help="pre-solve clean, filter 1: smoothing window for the per-cell smooth-error deletion "
                         "(0 disables)")
    ap.add_argument("--smooth-sigma", type=float, default=5.0,
                    help="pre-solve clean, filter 1: delete a cell when its residual to the smoothed track "
                         "exceeds median + sigma * 1.4826*MAD (pooled over all cells)")
    ap.add_argument("--reproj-del-px", type=float, default=0.0,
                    help="pre-solve clean, filter 2: delete a cell whose projected 2D position is farther "
                         "than this many px from the node's initial 2D observation (0 disables)")
    ap.add_argument("--mask-del", action=argparse.BooleanOptionalAction, default=False,
                    help="pre-solve clean, filter 2: delete cells projecting outside the dynamic mask (default off)")
    ap.add_argument("--mask-dilate-px", type=int, default=5,
                    help="pre-solve clean, filter 2: dilate the dynamic masks before the outside-mask test")
    ap.add_argument("--post-del-frac", type=float, default=0.0,
                    help="AFTER solving, delete a WHOLE track whose solved trajectory fails the same 2D "
                         "criterion (far from observation / outside mask) in more than this fraction of its "
                         "evaluable frames (0 disables)")
    ap.add_argument("--bad-node-frac", type=float, default=0.0,
                    help="AFTER solving, DELETE the BAD ('red') EDG nodes -- nodes whose skinning-weighted "
                         "fraction of render-colour-OUTLIER dynamic points is >= this (the same "
                         "classification as visualize_edg_filtered_open3d --edg-quality) -- and REASSIGN "
                         "their dynamic points to the surviving nodes. 0 disables")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out", default=None, help="output npz (default outputs/edg_test/edg_completed.npz)")
    ap.add_argument("--table", default=None, help="input table npz (default outputs/edg_test/edg_table.npz)")
    ap.add_argument("--no-results-all", action="store_true",
                    help="do NOT also update results_all.npz. By default this completion is ALSO written into "
                         "results_all.npz (the EDG the 4DGS trainer + visualize_training_input_dynamic.py "
                         "read), so that viewer and visualize_best_color_2d.py show the SAME EDG.")
    ap.add_argument("--results-all", default=None,
                    help="results_all.npz to update with this completed EDG (default outputs/results_all.npz). "
                         "Its static/prior/da3 keys are the base and are carried over unchanged; a one-time "
                         "backup is written to results_all_preEDGupdate.npz.")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"[device] {device}")
    D = load_inputs(args.merged, args.smoothed, args.edges_npz, args.table)
    masks = _load_dyn_mask_images(D["T"], args.mask_dilate_px) if args.mask_del else None
    clean_table(D, args, masks)                                       # delete bad cells, THEN solve
    c_fin, R_fin, tracks = complete(D, args, torch.device(device))
    dyn_ids, dyn_reconstruct = warp_dyn_output(tracks, c_fin, R_fin, D, torch.device(device))

    node_ids = D["edg_nodes"]
    keep = np.ones(node_ids.size, bool)
    # post-solve: delete WHOLE tracks that fail the same 2D criterion in > --post-del-frac of frames
    if args.post_del_frac > 0:
        keep &= post_delete_tracks(c_fin, D, args, masks)
    # DELETE the BAD ("red") EDG nodes -- the --edg-quality classification -- and reassign their points.
    if args.bad_node_frac > 0:
        frac = node_bad_fraction(node_ids, c_fin, R_fin, dyn_ids, dyn_reconstruct, D)
        if frac is not None:
            bad = np.isfinite(frac) & (frac >= args.bad_node_frac)
            print(f"[bad-nodes] {int(bad.sum())}/{node_ids.size} RED nodes (bad-dyn fraction >= "
                  f"{args.bad_node_frac:g}): "
                  + (", ".join(f"gid={int(node_ids[i])}({100 * frac[i]:.0f}%)" for i in np.where(bad)[0])
                     if bad.any() else "none"))
            keep &= ~bad
    if not keep.all():
        # reassign the dynamic points to the surviving nodes, then RE-WARP them with the pruned table
        tracks = reassign_tracks(tracks, keep, c_fin)
        node_ids, c_fin, R_fin = node_ids[keep], c_fin[keep], R_fin[keep]
        dyn_ids, dyn_reconstruct = warp_dyn_output(tracks, c_fin, R_fin, D, torch.device(device))
        print(f"[bad-nodes] re-warped {dyn_ids.size} dynamic tracks against {node_ids.size} surviving nodes")
    ns = set(node_ids.tolist())
    edges = np.array([[a, b] for a, b in D["edg_edges"].tolist() if a in ns and b in ns],
                     dtype=np.int64).reshape(-1, 2)

    out = Path(args.out) if args.out else Path(utils.definitions.output_dir) / "edg_test" / "edg_completed.npz"
    completed = dict(node_ids=node_ids, edg_reconstruct=c_fin, edg_rotation=R_fin,
                     edg_edges=edges, dyn_ids=dyn_ids, dyn_reconstruct=dyn_reconstruct,
                     n_tracks=int(D["Q"]))                             # track-id space stamp (merged vs raw)
    pool = ThreadPoolExecutor(1)                                      # zlib releases the GIL: written while the
    written = pool.submit(savez_compressed_fast, out, **completed)    # results_all update below runs

    # Also push this completed EDG into results_all.npz (the EDG the 4DGS trainer AND
    # visualize_training_input_dynamic.py read) so that viewer shows the SAME EDG as
    # visualize_best_color_2d.py (which reads edg_completed.npz directly).
    if not args.no_results_all:
        import shutil
        from completed_to_results import convert
        ra = Path(args.results_all) if args.results_all else Path(utils.definitions.output_dir) / "results_all.npz"
        if not ra.exists():                                          # no EDG_old.py run: the base is the camera solve's
            rt = np.load(Path(utils.definitions.output_dir) / "result_XM.npz")   # static points (as EDG_old.py writes them)
            np.savez_compressed(ra, static_landmarks=rt["p"].T.astype(np.float64) if "p" in rt else None)
            print(f"[results_all] created {ra.name} with the {rt['p'].shape[1] if 'p' in rt else 0} static points of result_XM.npz")
        if ra.exists():
            bak = ra.with_name("results_all_preEDGupdate.npz")
            if not bak.exists():
                shutil.copy2(ra, bak)
                print(f"[results_all] backed up existing base -> {bak.name}")
            convert(out, ra, ra, comp=completed)                     # overwrite results_all.npz EDG in place
            print(f"[results_all] updated {ra.name} with this completed EDG. "
                  f"visualize_training_input_dynamic.py + the 4DGS trainer now use the SAME EDG as "
                  f"visualize_best_color_2d.py. NOTE: re-run train_4dgs_edg.py to rebuild the dynamic "
                  f"binding (dynamic_edg.npz) against this EDG before training/viewing training input.")
        else:
            print(f"[results_all] {ra} not found -- skipping the results_all update (need an existing "
                  f"results_all.npz to supply static/prior/da3 keys). Pass --results-all or run the "
                  f"static/prior pipeline first; or pass --no-results-all to silence this.")
    written.result(); pool.shutdown()
    print(f"\n[done] {out}  ({node_ids.size} nodes, DENSE, T={D['T']}, {dyn_ids.size} dyn tracks)")
    print(f"       visualize: conda run -n XM python visualize_edg_solution_open3d.py "
          f"--npz {out} --no-scale-search")


if __name__ == "__main__":
    main()
