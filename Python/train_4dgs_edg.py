#!/usr/bin/env python3
"""4DGS training prep from EDG outputs.

Build a canonical (target-frame) dataset for Gaussian Splatting by:
1) Loading EDG + dynamic tracks from results_all.npz.
2) Mapping dynamic points from each frame to a target frame using EDG skinning.
3) Writing the COLMAP text dataset read by gsplat simple_trainer.py.
4) Printing the matching gsplat simple_trainer.py command.
"""

from __future__ import annotations

import glob
import os
import shlex
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

import utils.definitions
import utils.dataset_config as dataset_config
from utils.edg import scatter_solved_frame_vector, scatter_solved_poses
from utils.depth_sources import frame_sort_key
from utils.convertCOLMAP import (
    _rotation_matrix_to_qvec,
    write_cameras_text,
    write_images_text,
    write_points3D_text,
)

### S0: CONFIG
DEFAULT_TARGET_FRAME = 0
MAX_CANONICAL_POINTS = dataset_config.get_int("GS_MAX_POINTS", 500_000)   # initial Gaussians (static + moving), at most
GS_DYN_MIN_POINTS = dataset_config.get_int("GS_DYN_MIN_POINTS", 50_000)   # about this many of them on the moving objects
GS_DATA_FACTOR = dataset_config.get_int("GS_DATA_FACTOR", 1)   # step 6 trains on the frames downscaled by this (images_<f>/)
K_NEIGHBORS = 6
POINT_SAMPLE_SEED = 0
STATIC_SAMPLE_STRIDE = 4
STATIC_MAX_POINTS_PER_FRAME = 8000
# which EDG nodes carry a lifted pixel (config.json DYN_BIND): "motion" = those that best carry the track
# seen within DYN_BIND_PX px of it at its frame (fitted over the track's frames); "nearest" = the K nearest nodes in 3D.
DYN_BIND = dataset_config.get_str("DYN_BIND", "motion").lower()
DYN_BIND_PX = dataset_config.get_float("DYN_BIND_PX", 3.0)
# a lifted moving point whose warped depth violates the per-frame depth map (relative error > DYN_DEPTH_REJECT_TOL;
# baseline noise ~0.14, floaters >0.3) in MORE than DYN_DEPTH_REJECT_FRAC of its checkable frames floats off the surface
DYN_DEPTH_REJECT_FRAC = 0.3
DYN_DEPTH_REJECT_TOL = 0.3
# a lifted moving point whose mean k-NN distance exceeds median + DYN_FLYAWAY_STD * MAD (per frame) in more than
# DYN_FLYAWAY_FRAC of its frames is isolated from the cloud ("circles in the sky")
DYN_FLYAWAY_STD = 3.0
DYN_FLYAWAY_FRAC = 0.3
# Intrinsics come from one source: outputs/depth_anything_3/intrinsics.npy (written by
# get_depths_clean.py: config.json INTRINSICS or Depth Anything 3's). Used for cameras.txt
# and the dense static lift, so the whole pipeline stays on one intrinsic.
DA3_OUTLIER_K = 10        # neighbours for mean-distance computation
DA3_OUTLIER_STD_RATIO = 2.0  # remove if mean_k_dist > global_mean + ratio * global_std
DEBUG = dataset_config.get_bool("DEBUG", False)  # also write the depth-reject/density png, npz, videos
# Threads of the KD-tree queries and per-frame loops (the results do not depend on them). workers=-1 spawns a thread per
# core (huge overhead for the many small per-frame queries on a big machine).
N_WORKERS = 16


def _as_track_array(track_dict: Dict[int, np.ndarray], T: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert dict {gid: (T,3)} into (N,T,3) and gid list."""
    gids = sorted(track_dict.keys())
    if not gids:
        return np.zeros((0, T, 3), dtype=np.float64), np.zeros((0,), dtype=np.int64)
    tracks = np.stack([track_dict[g] for g in gids], axis=0).astype(np.float64)
    return tracks, np.array(gids, dtype=np.int64)


def _as_rotation_array(track_dict: Dict[int, np.ndarray], T: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert dict {gid: (T,3,3)} into (N,T,3,3) and gid list."""
    gids = sorted(track_dict.keys())
    if not gids:
        return (
            np.zeros((0, T, 3, 3), dtype=np.float64),
            np.zeros((0,), dtype=np.int64),
        )
    tracks = np.stack([track_dict[g] for g in gids], axis=0).astype(np.float64)
    return tracks, np.array(gids, dtype=np.int64)


def _world_to_target_transform(R_c2w: np.ndarray, t_c2w: np.ndarray) -> np.ndarray:
    """Return world->target transform using target camera pose (c2w)."""
    R = np.asarray(R_c2w, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t_c2w, dtype=np.float64).reshape(3)
    w2t = np.eye(4, dtype=np.float64)
    w2t[:3, :3] = R.T
    w2t[:3, 3] = -R.T @ t
    return w2t


def _apply_transform(points: np.ndarray, mat: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    hom = np.concatenate([pts, ones], axis=1)
    out = (mat @ hom.T).T
    return out[:, :3]


def _density_adaptive_downsample(pts, target_n, seed=0):
    """Density-ADAPTIVE downsample (vs a uniform voxel grid): thin only DENSE regions, keep SPARSE
    regions (near-)fully, so under-covered areas retain their points instead of being decimated at the
    same rate as well-covered ones. For each point, local density = #neighbours within r (r=3x median NN
    dist); keep-prob = min(1, K/density) with K binary-searched to hit ~target_n. Returns kept indices."""
    from scipy.spatial import cKDTree
    pts = np.asarray(pts, np.float64); n = pts.shape[0]
    if target_n <= 0 or n <= int(target_n):
        return np.arange(n)
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=min(2, n), workers=-1)   # one big query: every core (same result)
    nn = d[:, 1] if d.ndim == 2 and d.shape[1] > 1 else np.ones(n)
    r = max(float(np.median(nn[np.isfinite(nn)])) * 3.0, 1e-9)
    dens = np.asarray(tree.query_ball_point(pts, r, return_length=True, workers=-1), np.float64)
    dens = np.maximum(dens, 1.0)                       # local density (incl self); sparse -> small
    rng = np.random.default_rng(seed); u = rng.random(n)
    lo, hi, keep = 0.0, float(dens.max()), np.arange(n)
    for _ in range(40):
        K = 0.5 * (lo + hi)
        sel = u < np.minimum(1.0, K / dens)           # sparse points (dens<=K) kept w.p. 1
        c = int(sel.sum())
        keep = np.where(sel)[0]
        if c > int(target_n):
            hi = K
        else:
            lo = K
        if abs(c - int(target_n)) <= max(1, int(0.01 * target_n)):
            break
    return keep


def _save_dyn_density_2d(output_dir, knn, w, rel, er, ero, R_raw, t_raw, K_all, depth_maps,
                         image_paths, fps=15, bin_px=4):
    """Project the FINAL dynamic cloud into EVERY frame and render a 2D point-DENSITY heatmap (log count
    per bin_px cell, JET) blended over the RGB, so under-covered regions are obvious. Writes
    dyn_density_2d.mp4 (all frames) + dyn_density_2d.png (6-frame montage) to output_dir."""
    import cv2
    try:
        T = min(int(er.shape[1]), int(R_raw.shape[0]), int(K_all.shape[0]), len(image_paths), int(depth_maps.shape[0]))
        img0 = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR); H, W = img0.shape[:2]
        Hd, Wd = depth_maps[0].shape; sx, sy = W / float(Wd), H / float(Hd)
        gh, gw = max(1, H // bin_px), max(1, W // bin_px)
        def dens(t):
            Xm = er[knn, t]; Rm = ero[knn, t]
            P = np.einsum("nk,nkd->nd", w, Xm + np.einsum("nkij,nkj->nki", Rm, rel))
            R = np.asarray(R_raw[t], np.float64)
            u, v, z = _proj_uv_z(P, R[:3, :3], t_raw[t], np.asarray(K_all[t], np.float64), sx, sy)
            ok = np.isfinite(u) & np.isfinite(v) & (z > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            hist = np.zeros((gh, gw), np.float64)
            np.add.at(hist, (np.clip((v[ok] / bin_px).astype(int), 0, gh - 1),
                             np.clip((u[ok] / bin_px).astype(int), 0, gw - 1)), 1.0)
            return hist
        pos = [h[h > 0] for h in (dens(t) for t in range(0, T, max(1, T // 12)))]
        pos = np.concatenate(pos) if any(len(p) for p in pos) else np.array([1.0])
        vmax = float(np.percentile(pos, 98))
        def render(t):
            rgb = cv2.imread(str(image_paths[t]), cv2.IMREAD_COLOR)
            if rgb is None or rgb.shape[:2] != (H, W):
                rgb = np.full((H, W, 3), 60, np.uint8) if rgb is None else cv2.resize(rgb, (W, H))
            hist = dens(t); ln = np.log1p(hist) / np.log1p(max(vmax, 1.0))
            hm = cv2.applyColorMap(np.clip(ln * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
            hm = cv2.resize(hm, (W, H), interpolation=cv2.INTER_NEAREST)
            m = cv2.resize((hist > 0).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
            out = rgb.copy(); out[m] = (0.35 * rgb[m] + 0.65 * hm[m]).astype(np.uint8)
            cv2.putText(out, f"f{t} dyn-pts 2D density", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            return out
        vw = cv2.VideoWriter(str(Path(output_dir) / "dyn_density_2d.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        mont, idxs = [], set(np.linspace(0, T - 1, 6).astype(int).tolist())
        for t in range(T):
            out = render(t); vw.write(out)
            if t in idxs:
                mont.append(out)
        vw.release()
        if mont:
            cv2.imwrite(str(Path(output_dir) / "dyn_density_2d.png"), np.vstack(mont))
        print(f"[dyn-density] -> {output_dir}/dyn_density_2d.mp4 + dyn_density_2d.png (log point-density heatmap, {T} frames)")
    except Exception as exc:
        print(f"[dyn-density] viz failed (continuing): {exc}")


def _track_motion_nodes(er, ero, R_raw, t_raw, frame_scales, K, n_cand=12, min_frames=3, n_tracks=None):
    """For DYN_BIND = "motion": per moving-point track (the tracks the EDG was built from), the K EDG nodes whose rigid
    motion carries it best: among the n_cand nodes nearest to the track at its first seen frame, the K with the smallest
    spread of the local coordinate R_m(f)^T (x(f) - X_m(f)) over the track's frames; weights 1 / spread.
    Returns (xy (Q,T,2) image px, seen (Q,T), nodes (Q,K) node index or -1, weights (Q,K)).
    n_tracks: the track count the EDG was built from (its stamp); a different count means stale step-4 outputs."""
    tracks_path = utils.definitions.dynamic_tracks_for_edg()
    z = np.load(tracks_path, allow_pickle=True)
    xy = z["tracks_xy"][0].astype(np.float64); xyz = z["tracks_xyz"][0].astype(np.float64); Q = xy.shape[0]
    if n_tracks is not None and Q != int(n_tracks):
        raise ValueError(f"{tracks_path.name} has {Q} tracks but the EDG was built from {n_tracks}: rerun steps 4a-4d")
    T = min(xy.shape[1], er.shape[1], R_raw.shape[0])
    sc = np.ones(T) if frame_scales is None else np.where(np.isfinite(frame_scales[:T]), frame_scales[:T], 1.0)
    Xw = np.einsum("tij,qtj->qti", np.asarray(R_raw[:T], np.float64) * sc[:, None, None], xyz[:, :T]) + np.asarray(t_raw[:T], np.float64)[None]
    vis = ~z["occluded"][0][:, :T].astype(bool) & np.isfinite(xy[:, :T]).all(-1) & np.isfinite(Xw).all(-1)
    if "dynamic_obs_mask" in z.files:
        vis &= z["dynamic_obs_mask"][:, :T].astype(bool)
    nodes = np.full((Q, K), -1, np.int64); wts = np.zeros((Q, K))
    ok = np.where(vis.sum(1) >= min_frames)[0]

    def chunk(c0):   # tracks ok[c0:c0 + 512] -> their rows of nodes / wts
        q = ok[c0:c0 + 512]; vq = vis[q]; f0 = vq.argmax(1)
        d = np.linalg.norm(er[:, f0].transpose(1, 0, 2) - Xw[q, f0][:, None], axis=-1)        # (B, M) at the first frame
        d = np.where(np.isfinite(d), d, np.inf); cand = np.argsort(d, axis=1)[:, :n_cand]       # (B, C)
        bi, fi = np.nonzero(vq)                                    # the seen frames only, grouped by track (row-major)
        n = vq.sum(1); s0 = np.r_[0, np.cumsum(n)[:-1]]
        L = np.einsum("ncji,ncj->nci", ero[cand[bi], fi[:, None]], Xw[q[bi], fi][:, None] - er[cand[bi], fi[:, None]])
        r = np.add.reduceat(L, s0, axis=0) / n[:, None, None]                                   # local coordinates' mean
        res = np.add.reduceat(((L - r[bi]) ** 2).sum(-1), s0, axis=0) / n[:, None]              # (B, C) their spread
        res = np.where(np.isfinite(res), res, np.inf); o = np.argsort(res, axis=1)[:, :K]
        nodes[q] = np.take_along_axis(cand, o, 1)
        w = 1.0 / (np.take_along_axis(res, o, 1) + 1e-12); wts[q] = w / w.sum(1, keepdims=True)
    with ThreadPoolExecutor(N_WORKERS) as ex:   # independent chunks (numpy releases the GIL), disjoint rows
        list(ex.map(chunk, range(0, len(ok), 512)))
    return xy[:, :T], vis, nodes, wts


def _img_wh(path, _c={}):
    if path not in _c:
        im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED); _c[path] = None if im is None else (im.shape[1], im.shape[0])
    return _c[path]


def _lift_edg_bound_dynamic_from_depth(*, depth_maps, dynamic_masks, R_raw, t_raw, K_all,
                                       image_paths, er, ero, node_ids, w2t, target_frame,
                                       k_neighbors, target_points, stride=2, sigma_frac=1.5,
                                       frame_scales=None, n_tracks=None):
    """Densely lift dynamic-region depth pixels from EVERY frame to 3D, EDG-bind each at ITS OWN frame
    (rest frame = the frame it was lifted from: rel = R_m(t)^T (X - X_m(t))), colour from the image, then
    ADAPTIVE-DENSITY voxel-downsample (uniform 3D density) to ~target_points. Returns per-point arrays in
    node_ids order: (xyz_solver_target, rgb, knn_local, knn_w, rel, sk). Downstream depth-rejection then
    removes floaters."""
    from scipy.spatial import cKDTree
    import cv2
    M, T = er.shape[0], er.shape[1]
    K = int(k_neighbors)
    Tn = min(int(depth_maps.shape[0]), int(R_raw.shape[0]), int(K_all.shape[0]), len(image_paths), T)
    Xw_all, rgb_all, knn_all, w_all, rel_all, sk_all, pos_all = [], [], [], [], [], [], []
    # DYN_BIND "motion": a lifted pixel takes the nodes that best carry the track seen within DYN_BIND_PX px of it at its
    # frame (the nearest nodes in 3D often belong to another part and average its motion away); "nearest": K nearest nodes.
    # (with fewer than K nodes no point can take K motion-fitted nodes: nearest nodes only)
    mot = (_track_motion_nodes(er, ero, R_raw, t_raw, frame_scales, K, n_tracks=n_tracks)
           if DYN_BIND == "motion" and M >= K else None)
    n_mot = 0

    def frame(t):   # lift + bind frame t: (Xw, rgb, nl, w, rel, target-frame position, n motion-bound), None if empty
        present = np.where(np.isfinite(er[:, t]).all(1))[0]      # EDG nodes solved at frame t
        if present.size < min(K, M):                             # (a graph with fewer than K nodes binds to all of them)
            return None
        dm = depth_maps[t].astype(np.float64); Hd, Wd = dm.shape
        msk = np.asarray(dynamic_masks[t], bool)
        if msk.shape != (Hd, Wd):
            msk = cv2.resize(msk.astype(np.uint8), (Wd, Hd), interpolation=cv2.INTER_NEAREST) > 0
        ys = np.arange(0, Hd, int(stride)); xs = np.arange(0, Wd, int(stride))
        yy, xx = np.meshgrid(ys, xs, indexing="ij"); yy = yy.ravel(); xx = xx.ravel()
        z = dm[yy, xx]
        ok = msk[yy, xx] & np.isfinite(z) & (z > 1e-6)
        if not ok.any():
            return None
        yy, xx, z = yy[ok], xx[ok], z[ok]
        if frame_scales is not None and t < len(frame_scales) and np.isfinite(frame_scales[t]):
            z = z * float(frame_scales[t])         # per-frame depth scale: world = R@(s*Xc)+t == (s*R)@Xc+t, as the
                                                   # EDG / track lift (s*R); without it the points are off by s_t
        K_t = np.asarray(K_all[t], np.float64)
        fx, fy, cx, cy = K_t[0, 0], K_t[1, 1], K_t[0, 2], K_t[1, 2]
        Xc = np.stack([(xx - cx) / fx * z, (yy - cy) / fy * z, z], 1)      # camera
        R = np.asarray(R_raw[t], np.float64); c = np.asarray(t_raw[t], np.float64)
        Xw = (R @ Xc.T).T + c[None]                                       # solver world
        # bind to K nearest present nodes at frame t
        nodep = er[present, t]                                            # (Pn,3)
        tree = cKDTree(nodep)
        dist, nn = tree.query(Xw, k=min(K, present.size))
        dist = np.atleast_2d(dist); nn = np.atleast_2d(nn)
        sigma = max(float(np.median(dist[:, 0])) * float(sigma_frac), 1e-6)
        w = np.exp(-(dist / sigma) ** 2); w /= np.clip(w.sum(1, keepdims=True), 1e-12, None)
        nl = present[nn]                                                  # node-ids-order local index (M-space)
        n_hit = 0
        if mot is not None and t < mot[0].shape[1] and nl.shape[1] == K:   # the motion-fitted nodes of the nearest track
            tq = np.where(mot[1][:, t] & (mot[2][:, 0] >= 0))[0]
            wh = _img_wh(image_paths[t])
            if tq.size and wh is not None:
                sxd, syd = wh[0] / float(Wd), wh[1] / float(Hd)                # image px per depth px
                dd, ii = cKDTree(mot[0][tq, t] / np.array([sxd, syd])).query(np.stack([xx, yy], 1).astype(np.float64))
                hit = np.where(dd <= DYN_BIND_PX / sxd)[0]
                hit = hit[np.isfinite(er[mot[2][tq[ii[hit]]], t]).all(-1).all(-1)]
                nl[hit] = mot[2][tq[ii[hit]]]; w[hit] = mot[3][tq[ii[hit]]]; n_hit = hit.size
        Xm = er[nl, t]; Rm = ero[nl, t]                                   # (N,K,3),(N,K,3,3)
        rel = np.einsum("nkji,nkj->nki", Rm, (Xw[:, None, :] - Xm))       # R_m(t)^T (X - X_m(t))
        img = cv2.imread(str(image_paths[t]), cv2.IMREAD_COLOR)
        if img is not None:
            # Sample colour whenever the image loads. The scale is a no-op when the image and
            # depth-map sizes match (the common case) and resamples when they differ.
            sxi, syi = img.shape[1] / float(Wd), img.shape[0] / float(Hd)
            ui = np.clip((xx * sxi).astype(np.int64), 0, img.shape[1] - 1)
            vi = np.clip((yy * syi).astype(np.int64), 0, img.shape[0] - 1)
            rgb = img[vi, ui][:, ::-1]
        else:
            rgb = np.full((Xw.shape[0], 3), 128, np.uint8)
        # pad K if fewer neighbours
        if nl.shape[1] < K:
            pad = K - nl.shape[1]
            nl = np.concatenate([nl, np.repeat(nl[:, -1:], pad, 1)], 1)
            w = np.concatenate([w, np.zeros((w.shape[0], pad))], 1)
            rel = np.concatenate([rel, np.zeros((rel.shape[0], pad, 3))], 1)
        # init position = the point warped to the target frame (bind-frame-agnostic), then w2t -> target coords
        warp_t = er[nl, target_frame] + np.einsum("nkij,nkj->nki", ero[nl, target_frame], rel)
        pos = np.einsum("nk,nkd->nd", w, warp_t)                         # (N,3) solver, target frame
        return Xw, rgb.astype(np.uint8), nl, w, rel, pos, n_hit
    with ThreadPoolExecutor(N_WORKERS) as ex:   # independent frames (numpy/cv2/KD-tree release the GIL), in order
        for t, res in enumerate(ex.map(frame, range(Tn))):
            if res is None:
                continue
            Xw, rgb, nl, w, rel, pos, n_hit = res
            Xw_all.append(Xw); rgb_all.append(rgb)
            knn_all.append(nl); w_all.append(w); rel_all.append(rel); pos_all.append(pos)
            sk_all.append(np.full(Xw.shape[0], t, np.int64)); n_mot += n_hit
    if not Xw_all:
        return (np.zeros((0, 3)), np.zeros((0, 3), np.uint8),
                np.zeros((0, K), np.int64), np.zeros((0, K)), np.zeros((0, K, 3)), np.zeros((0,), np.int64))
    Xw_all = np.concatenate(Xw_all); rgb_all = np.concatenate(rgb_all)
    knn_all = np.concatenate(knn_all); w_all = np.concatenate(w_all)
    rel_all = np.concatenate(rel_all); sk_all = np.concatenate(sk_all)
    pos_target_solver = np.concatenate(pos_all)                          # (N,3) solver, target frame (rows independent)
    finite = np.isfinite(pos_target_solver).all(1)
    keep = _density_adaptive_downsample(pos_target_solver[finite], int(target_points))
    gi = np.where(finite)[0][keep]
    if mot is not None:
        print(f"[dyn-bind] motion binding: {n_mot} of {Xw_all.shape[0]} lifted pixels took the fitted nodes of a track within "
              f"{DYN_BIND_PX:g} px ({int((mot[2][:, 0] >= 0).sum())} tracks fitted, K={K})")
    print(f"[dyn-lift] per-frame depth lift: {Xw_all.shape[0]} raw -> {gi.size} after density-adaptive "
          f"downsample (target {target_points}, keeps sparse regions); each bound at its own frame.")
    xyz_target = _apply_transform(pos_target_solver[gi], w2t)
    return (xyz_target, rgb_all[gi], knn_all[gi], w_all[gi], rel_all[gi], sk_all[gi])


def _flyaway_reject(skin_fn, struct, T, k=8, std_ratio=3.0, frac=0.3, n_sample=24, min_check=3):
    """Prune dynamic points that fly far from the rest of the cloud ("circles in the sky"). Over sampled
    frames, warp the cloud, and for each point measure its mean distance to its k nearest dynamic
    neighbours; a point is an OUTLIER that frame if that distance exceeds median + std_ratio * robust-MAD
    of all points. A point flagged in more than `frac` of its sampled frames is pruned. Robust MAD +
    fractional voting keep the dense object body intact (no new holes) while removing isolated flyers.
    Returns a boolean reject mask, length struct['P']."""
    from scipy.spatial import cKDTree
    P = int(struct["P"])
    n_out = np.zeros(P); n_chk = np.zeros(P)
    frames = np.unique(np.linspace(0, T - 1, min(int(n_sample), T)).astype(int))
    for t in frames:
        X = skin_fn(struct, int(t))
        fin = np.isfinite(X).all(1)
        if int(fin.sum()) < k + 2:
            continue
        idx = np.where(fin)[0]
        tree = cKDTree(X[idx])
        d, _ = tree.query(X[idx], k=k + 1, workers=N_WORKERS)    # includes self at col 0
        mean_knn = d[:, 1:].mean(1)
        med = float(np.median(mean_knn))
        mad = float(np.median(np.abs(mean_knn - med))) * 1.4826 + 1e-9
        thr = med + float(std_ratio) * mad
        n_chk[idx] += 1
        n_out[idx[mean_knn > thr]] += 1
    chk = n_chk >= int(min_check)
    frac_out = np.zeros(P)
    frac_out[chk] = n_out[chk] / np.clip(n_chk[chk], 1, None)
    return chk & (frac_out > float(frac))


def _depth_consistency_reject(sp, skin_fn, depth_maps, R_raw, t_raw, K_all,
                              good_mask, tol=0.1, frac=0.3, min_check=5, frame_scales=None):
    """Reject dynamic points that float off the observed surface: warp each point to every frame,
    project into that frame's solver camera, sample the depth map at the projected pixel, and compare
    the point's camera-space depth to the observed depth (relative error > `tol`). A point whose depth
    disagrees in more than `frac` of its checkable frames is rejected (it is far from the main surface
    AND violates depth persistently). Returns (reject_mask, viol_frac, n_check), all length sp["P"]."""
    P = int(sp["P"])
    n_check = np.zeros(P, np.int64)
    n_viol = np.zeros(P, np.int64)
    T = min(int(depth_maps.shape[0]), int(R_raw.shape[0]), int(K_all.shape[0]))
    gm = np.asarray(good_mask, bool)

    def frame(t):   # (checked, violating) point masks of frame t, None if it checks nothing
        dm = depth_maps[t]
        Hd, Wd = dm.shape
        Xw = skin_fn(sp, t)                                   # (P,3) solver world
        R = np.asarray(R_raw[t], np.float64); c = np.asarray(t_raw[t], np.float64)
        K = np.asarray(K_all[t], np.float64)
        Xc = (Xw - c[None]) @ R                               # R.T @ (Xw-c) per row
        z = Xc[:, 2]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.round(fx * Xc[:, 0] / z + cx)
            v = np.round(fy * Xc[:, 1] / z + cy)
        inb = gm & np.isfinite(z) & (z > 1e-6) & np.isfinite(u) & np.isfinite(v)
        ui = np.where(inb, u, 0).astype(np.int64); vi = np.where(inb, v, 0).astype(np.int64)
        inb &= (ui >= 0) & (ui < Wd) & (vi >= 0) & (vi < Hd)
        if not inb.any():
            return None
        dv = np.full(P, np.nan)
        dv[inb] = dm[vi[inb], ui[inb]]
        if frame_scales is not None and t < len(frame_scales) and np.isfinite(frame_scales[t]):
            dv = dv * float(frame_scales[t])       # scale the observed depth into the (s*R) world gauge
        valid = inb & np.isfinite(dv) & (dv > 1e-6)
        if not valid.any():
            return None
        # Per-frame scale between the point cloud's camera-z and the depth map (the solver world is
        # DA3-scaled, the depth map may be lidar/metric or a different scale). Normalize by the robust
        # median ratio so the test measures RELATIVE off-surface deviation, not a global scale offset.
        s_t = float(np.median(z[valid] / dv[valid]))
        if not np.isfinite(s_t) or s_t <= 1e-8:
            return None
        rel = np.abs(z - s_t * dv) / np.clip(s_t * dv, 1e-6, None)
        return valid, valid & (rel > float(tol))
    with ThreadPoolExecutor(N_WORKERS) as ex:   # frames are independent (numpy releases the GIL), counts order-free
        for res in ex.map(frame, range(T)):
            if res is not None:
                n_check[res[0]] += 1
                n_viol[res[1]] += 1
    checked = n_check >= int(min_check)
    viol_frac = np.zeros(P)
    viol_frac[checked] = n_viol[checked] / np.clip(n_check[checked], 1, None)
    reject = checked & (viol_frac > float(frac))
    return reject, viol_frac, n_check


def _proj_uv_z(pts_world, R_c2w, c_c2w, K, sx=1.0, sy=1.0):
    """Project solver-world points into a camera. Returns (u, v, z) in image pixels (u,v scaled by sx,sy)."""
    Xc = (np.asarray(pts_world, np.float64) - np.asarray(c_c2w, np.float64)[None]) @ np.asarray(R_c2w, np.float64)
    z = Xc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (K[0, 0] * Xc[:, 0] / z + K[0, 2]) * sx
        v = (K[1, 1] * Xc[:, 1] / z + K[1, 2]) * sy
    return u, v, z


def _point_depth_panel(u, v, z, H, W, vmin, vmax):
    """Nearest-z depth buffer of projected points, colorized (JET), background black. Returns HxWx3 BGR."""
    import cv2
    buf = np.full(H * W, np.inf, np.float64)
    ok = np.isfinite(u) & np.isfinite(v) & np.isfinite(z) & (z > 1e-6)
    ui = np.clip(np.round(u), 0, W - 1).astype(np.int64)
    vi = np.clip(np.round(v), 0, H - 1).astype(np.int64)
    ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
    flat = vi[ok] * W + ui[ok]
    np.minimum.at(buf, flat, z[ok])
    buf = buf.reshape(H, W)
    filled = np.isfinite(buf)
    norm = np.clip((buf - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    color[~filled] = (0, 0, 0)
    return color


def _save_depth_reject_video(output_dir, good_before, reject, sp, skin_fn,
                             R_raw, t_raw, K_all, depth_maps, image_paths, fps=15):
    """Save ALL-FRAME visualizations of the (global) depth rejection:
      * dyn_depth_reject_overlay.mp4  -- each frame's image with kept=green, rejected=red projected points
      * dyn_depth_reject_depth.mp4    -- side-by-side colorized point-depth: BEFORE (all good) | AFTER (kept)
    The reject decision is global (computed over all frames); the video shows it applied per frame."""
    import cv2
    try:
        T = min(int(depth_maps.shape[0]), int(R_raw.shape[0]), int(K_all.shape[0]), len(image_paths))
        img0 = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
        H, W = img0.shape[:2]
        Hd, Wd = depth_maps[0].shape
        sx, sy = W / float(Wd), H / float(Hd)
        good_before = np.asarray(good_before, bool); reject = np.asarray(reject, bool)
        kept = good_before & (~reject); rej = good_before & reject
        # global depth colour range from kept points over a few frames
        zs = []
        for t in range(0, T, max(1, T // 20)):
            _, _, z = _proj_uv_z(skin_fn(sp, t), R_raw[t][:3, :3],
                                 t_raw[t], np.asarray(K_all[t], np.float64), sx, sy)
            zs.append(z[good_before & np.isfinite(z) & (z > 1e-6)])
        zc = np.concatenate(zs) if zs else np.array([1.0])
        vmin, vmax = float(np.percentile(zc, 2)), float(np.percentile(zc, 98))
        ov = cv2.VideoWriter(str(Path(output_dir) / "dyn_depth_reject_overlay.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        dv = cv2.VideoWriter(str(Path(output_dir) / "dyn_depth_reject_depth.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * W, H))
        for t in range(T):
            img = cv2.imread(str(image_paths[t]), cv2.IMREAD_COLOR)
            if img is None or img.shape[:2] != (H, W):
                img = np.full((H, W, 3), 60, np.uint8) if img is None else cv2.resize(img, (W, H))
            R = np.asarray(R_raw[t], np.float64); K = np.asarray(K_all[t], np.float64)
            u, v, z = _proj_uv_z(skin_fn(sp, t), R, t_raw[t], K, sx, sy)
            okp = np.isfinite(u) & np.isfinite(v) & (z > 1e-6)
            canvas = img.copy()
            ku = u[okp & kept]; kv = v[okp & kept]
            ru = u[okp & rej]; rv = v[okp & rej]
            for uu, vv in zip(ku.astype(int), kv.astype(int)):
                cv2.circle(canvas, (uu, vv), 1, (0, 255, 0), -1)
            for uu, vv in zip(ru.astype(int), rv.astype(int)):
                cv2.circle(canvas, (uu, vv), 2, (0, 0, 255), -1)
            cv2.putText(canvas, f"f{t}  kept(green)={int(kept.sum())} rejected(red)={int(rej.sum())}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            ov.write(canvas)
            before = _point_depth_panel(u[good_before], v[good_before], z[good_before], H, W, vmin, vmax)
            after = _point_depth_panel(u[kept], v[kept], z[kept], H, W, vmin, vmax)
            cv2.putText(before, "BEFORE (all good)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(after, "AFTER (depth-reject)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            dv.write(np.concatenate([before, after], axis=1))
        ov.release(); dv.release()
        print(f"[depth-reject] videos -> {output_dir}/dyn_depth_reject_overlay.mp4 (green/red, {T} frames) "
              f"and dyn_depth_reject_depth.mp4 (BEFORE|AFTER depth)")
    except Exception as exc:
        print(f"[depth-reject] video failed (continuing): {exc}")


def _save_depth_reject_overlay(output_dir, target_frame, pos_world_target, good_before,
                               reject, R_raw, t_raw, K_all, depth_maps, image_paths,
                               sp, skin_fn):
    """Save a target-frame overlay PNG (kept GREEN, rejected RED) + masks npz, and the ALL-FRAME overlay +
    BEFORE/AFTER depth videos of the points `sp` warped by `skin_fn`."""
    import cv2
    try:
        t = int(target_frame)
        R = np.asarray(R_raw[t], np.float64); c = np.asarray(t_raw[t], np.float64)
        K = np.asarray(K_all[t], np.float64)
        Hd, Wd = depth_maps[t].shape
        img = cv2.imread(str(image_paths[t]), cv2.IMREAD_COLOR) if t < len(image_paths) else None
        if img is None:
            img = np.full((Hd, Wd, 3), 60, np.uint8)
        H, W = img.shape[:2]
        sx, sy = W / float(Wd), H / float(Hd)                 # depth-pixel -> image-pixel
        u, v, z = _proj_uv_z(pos_world_target, R, c, K, sx, sy)
        canvas = img.copy()
        okp = np.isfinite(u) & np.isfinite(v) & (z > 1e-6)
        for i in np.where(okp & good_before & ~reject)[0]:
            cv2.circle(canvas, (int(round(u[i])), int(round(v[i]))), 1, (0, 255, 0), -1)   # kept
        for i in np.where(okp & good_before & reject)[0]:
            cv2.circle(canvas, (int(round(u[i])), int(round(v[i]))), 2, (0, 0, 255), -1)   # rejected
        nk = int((good_before & ~reject).sum()); nr = int((good_before & reject).sum())
        cv2.putText(canvas, f"depth-reject: kept={nk} (green)  rejected={nr} (red)",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out = Path(output_dir) / f"dyn_depth_reject_overlay_frame{t:06d}.png"
        cv2.imwrite(str(out), canvas)
        np.savez_compressed(Path(output_dir) / "dyn_depth_reject.npz",
                            reject=reject, good_before=good_before,
                            pos_world_target=pos_world_target)
        print(f"[depth-reject] overlay -> {out} (kept {nk} green, rejected {nr} red)")
    except Exception as exc:
        print(f"[depth-reject] overlay failed (continuing): {exc}")
    _save_depth_reject_video(output_dir, good_before, reject, sp, skin_fn,
                             R_raw, t_raw, K_all, depth_maps, image_paths)


def _build_edg_good_dynamic_points(
    *,
    output_dir: Path,
    target_frame: int,
    w2t: np.ndarray,
    edg_gids: np.ndarray,
    k_neighbors: int,
    dyn_min_points: int,
    depth_maps: np.ndarray,
    R_raw: np.ndarray,
    t_raw: np.ndarray,
    K_all: np.ndarray,
    image_paths: list,
    dynamic_masks: np.ndarray,
    frame_scales=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dynamic cloud from the completed EDG (edg_completed.npz): the dynamic-mask depth pixels of EVERY frame,
    each EDG-bound at ITS OWN frame (_lift_edg_bound_dynamic_from_depth), then the floaters depth-rejected and the
    isolated points fly-away pruned. The trainer's bind-frame-agnostic warp X_m(t) + R_m(t) rel fuses every point
    to whatever frame is rendered; `target_frame` is only the COLMAP dataset's coordinate gauge (init positions).
    Node indices are remapped to `edg_gids` (the results_all order the trainer uses). Returns
    (xyz_target, rgb, knn_idx, knn_w, rel) (empty arrays if none)."""
    npz_path = Path(output_dir) / "edg_test" / "edg_completed.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"the moving points need {npz_path} (run the EDG completion / complete_edg_table.py first)."
        )
    data = np.load(npz_path, allow_pickle=True)
    if not ("node_ids" in data.files and "dyn_reconstruct" in data.files):
        raise RuntimeError(f"{npz_path} is not a completed EDG_test npz (needs node_ids + dyn_reconstruct).")
    node_ids = np.asarray(data["node_ids"], dtype=np.int64)
    er = np.asarray(data["edg_reconstruct"], dtype=np.float64)
    ero = np.asarray(data["edg_rotation"], dtype=np.float64)

    # ---- PER-FRAME DENSE DEPTH LIFT: fill coverage by lifting dynamic-mask depth pixels from
    # EVERY frame, EDG-binding each at its own frame, adaptive voxel density, THEN depth-reject floaters. ----
    K = int(k_neighbors)
    xyzL, rgbL, knnL_nl, wL, relL, _skL = _lift_edg_bound_dynamic_from_depth(
        depth_maps=depth_maps, dynamic_masks=dynamic_masks, R_raw=R_raw, t_raw=t_raw, K_all=K_all,
        image_paths=image_paths, er=er, ero=ero, node_ids=node_ids, w2t=w2t,
        target_frame=int(target_frame), k_neighbors=K,
        target_points=int(dyn_min_points) if int(dyn_min_points) > 0 else 50000,
        frame_scales=frame_scales, n_tracks=int(data["n_tracks"]) if "n_tracks" in data.files else None)
    if xyzL.shape[0] == 0:
        return (np.zeros((0, 3)), np.zeros((0, 3), np.uint8),
                np.zeros((0, K), np.int64), np.zeros((0, K)), np.zeros((0, K, 3)))
    # remap knn node-ids-order (M-index) -> edg_gids column order; keep points with all-valid neighbours
    gid_to_col = {int(g): i for i, g in enumerate(np.asarray(edg_gids).reshape(-1).tolist())}
    col_of_nl = np.array([gid_to_col.get(int(g), -1) for g in node_ids.tolist()], np.int64)
    knnL = col_of_nl[knnL_nl]                                         # (N,K) edg_gids columns, -1 if absent
    valid = (knnL >= 0).all(1)
    xyzL, rgbL, knnL, wL, relL, knnL_nl2 = (xyzL[valid], rgbL[valid], knnL[valid],
                                            wL[valid], relL[valid], knnL_nl[valid])
    # depth-reject: warp lifted points via their binding (node_ids order) and check depth consistency
    lifted = {"knn": knnL_nl2, "w": wL, "rel": relL, "P": xyzL.shape[0]}
    def _lifted_skin(ls, t, _er=er, _ero=ero):
        Xm = _er[ls["knn"], t]; Rm = _ero[ls["knn"], t]
        return np.einsum("nk,nkd->nd", ls["w"], Xm + np.einsum("nkij,nkj->nki", Rm, ls["rel"]))
    reject, _vf, _nc = _depth_consistency_reject(
        lifted, _lifted_skin, depth_maps, R_raw, t_raw, K_all,
        np.ones(xyzL.shape[0], bool), tol=DYN_DEPTH_REJECT_TOL, frac=DYN_DEPTH_REJECT_FRAC,
        frame_scales=frame_scales)
    n_rej = int(reject.sum())
    print(f"[dyn-lift] depth-reject: {n_rej}/{xyzL.shape[0]} lifted points floated off surface "
          f"(>{100*DYN_DEPTH_REJECT_FRAC:.0f}% frames) -> dropped; kept {xyzL.shape[0]-n_rej}.")
    if DEBUG:
        _save_depth_reject_overlay(output_dir, int(target_frame),
                                   _apply_transform(xyzL, np.linalg.inv(w2t)),  # back to solver for proj
                                   np.ones(xyzL.shape[0], bool), reject, R_raw, t_raw, K_all,
                                   depth_maps, image_paths, sp=lifted, skin_fn=_lifted_skin)
    keep = ~reject
    xyzL, rgbL, knnL, wL, relL, knnL_nl2 = (xyzL[keep], rgbL[keep], knnL[keep], wL[keep],
                                            relL[keep], knnL_nl2[keep])
    # spatial fly-away prune: drop points isolated from the cloud in >frac of frames ("sky circles")
    if xyzL.shape[0] > 0:
        lifted2 = {"knn": knnL_nl2, "w": wL, "rel": relL, "P": xyzL.shape[0]}
        def _lifted_skin2(ls, t, _er=er, _ero=ero):
            Xm = _er[ls["knn"], t]; Rm = _ero[ls["knn"], t]
            return np.einsum("nk,nkd->nd", ls["w"], Xm + np.einsum("nkij,nkj->nki", Rm, ls["rel"]))
        fly = _flyaway_reject(_lifted_skin2, lifted2, er.shape[1],
                              std_ratio=DYN_FLYAWAY_STD, frac=DYN_FLYAWAY_FRAC)
        n_fly = int(fly.sum())
        print(f"[dyn-lift] fly-away prune: {n_fly}/{xyzL.shape[0]} isolated points removed "
              f"(std>{DYN_FLYAWAY_STD:g}, >{100*DYN_FLYAWAY_FRAC:.0f}% frames).")
        k2 = ~fly
        xyzL, rgbL, knnL, wL, relL, knnL_nl2 = (xyzL[k2], rgbL[k2], knnL[k2], wL[k2], relL[k2], knnL_nl2[k2])
    print(f"[dyn-lift] final dense dynamic cloud: {xyzL.shape[0]} points (per-frame lift + depth-reject "
          f"+ fly-away prune).")
    if DEBUG:
        _save_dyn_density_2d(output_dir, knnL_nl2, wL, relL, er, ero, R_raw, t_raw, K_all,
                             depth_maps, image_paths)
    return xyzL, rgbL.astype(np.uint8), knnL.astype(np.int64), wL, relL


def _load_da3_intrinsics_for_depth_lifting(rootpath: Path, T: int) -> np.ndarray:
    intrinsics_path = Path(rootpath) / "outputs" / "depth_anything_3" / "intrinsics.npy"
    if not intrinsics_path.exists():
        raise FileNotFoundError(f"Missing DA3 intrinsics for solver-camera lifting: {intrinsics_path}")
    K_all = np.load(intrinsics_path).astype(np.float64)
    if K_all.ndim == 2:
        K_all = K_all[None, :, :]
    if K_all.ndim != 3 or K_all.shape[1:] != (3, 3):
        raise ValueError(f"Expected {intrinsics_path} shape (T,3,3), got {K_all.shape}")
    if K_all.shape[0] >= T:
        return K_all[:T]
    if K_all.shape[0] == 1:
        return np.repeat(K_all, T, axis=0)
    raise ValueError(f"Only {K_all.shape[0]} DA3 intrinsics for T={T} frames.")


def _sample_image_rgb_for_depth_pixels(
    *,
    img_bgr: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    depth_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    H_d, W_d = depth_shape
    H, W = img_bgr.shape[:2]
    if (H, W) == (H_d, W_d):
        u = xx.astype(np.int64)
        v = yy.astype(np.int64)
    else:
        u = np.rint(xx * (W - 1) / max(W_d - 1, 1)).astype(np.int64)
        v = np.rint(yy * (H - 1) / max(H_d - 1, 1)).astype(np.int64)
    u = np.clip(u, 0, W - 1)
    v = np.clip(v, 0, H - 1)
    return img_bgr[v, u, ::-1].astype(np.uint8), u, v


def _statistical_outlier_mask(points: np.ndarray, k: int, std_ratio: float) -> np.ndarray:
    """Return bool mask (True = outlier) using statistical outlier removal.

    For each point, compute the mean distance to its k nearest neighbours.
    Points whose mean distance exceeds  global_mean + std_ratio * global_std
    are flagged as isolated / structureless noise floating in space.
    """
    from scipy.spatial import cKDTree
    n = points.shape[0]
    k = min(int(k), n - 1)
    if k < 1:
        return np.zeros(n, dtype=bool)
    dists, _ = cKDTree(points).query(points, k=k + 1, workers=N_WORKERS)
    mean_dists = dists[:, 1:].mean(axis=1)   # skip self (col 0, dist=0)
    threshold = mean_dists.mean() + float(std_ratio) * mean_dists.std()
    return mean_dists > threshold


def _load_intrinsics_for_training(T: int) -> np.ndarray:
    # Single intrinsic source: intrinsics.npy (get_depths_clean.py already applied the
    # config INTRINSICS-or-DA3 choice). Used for both cameras.txt and the dense static lift so the
    # 4DGS dataset is internally consistent with the solver.
    root = Path(utils.definitions.rootpath)
    return _load_da3_intrinsics_for_depth_lifting(root, T)


def _image_hw(path) -> tuple[int, int]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img.shape[:2]


def _export_colmap_text_dataset(
    *,
    sparse_dir: Path,
    image_paths: list[Path],
    c2w_all: np.ndarray,
    K_all: np.ndarray,
    points: np.ndarray,
    rgb: np.ndarray,
) -> None:
    """Write the COLMAP text files consumed by deps/gsplat/example/simple_trainer.py."""
    sparse_dir.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    rgb = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    if points.shape[0] != rgb.shape[0]:
        raise ValueError(f"Point/color count mismatch: {points.shape[0]} vs {rgb.shape[0]}")

    cameras = {}
    images = {}
    n_frames = min(len(image_paths), c2w_all.shape[0], K_all.shape[0])
    with ThreadPoolExecutor(N_WORKERS) as ex:   # decoded on threads (cv2 releases the GIL), only for their size
        hw = list(ex.map(_image_hw, image_paths[:n_frames]))
    for i in range(n_frames):
        h, w = hw[i]
        K = np.asarray(K_all[i], dtype=np.float64).reshape(3, 3)
        camera_id = i + 1
        cameras[camera_id] = {
            "model": "PINHOLE",
            "width": int(w),
            "height": int(h),
            "params": [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])],
        }

        R_c2w = np.asarray(c2w_all[i, :3, :3], dtype=np.float64)
        t_c2w = np.asarray(c2w_all[i, :3, 3], dtype=np.float64)
        # Match the local text Parser's COLMAP convention so Parser.camtoworlds
        # recovers this target-frame c2w pose.
        images[i + 1] = {
            "id": i + 1,
            "qvec": _rotation_matrix_to_qvec(R_c2w.T),
            "tvec": (-R_c2w.T @ t_c2w).astype(np.float64),
            "camera_id": camera_id,
            "name": image_paths[i].name,
        }

    xyz_list, rgb_list = points.tolist(), rgb.tolist()   # Python floats/ints: str() prints the same text, 3x faster
    points3D = {
        i + 1: {
            "id": i + 1,
            "xyz": xyz_list[i],
            "rgb": rgb_list[i],
            "error": 0.0,
        }
        for i in range(points.shape[0])
    }

    write_cameras_text(cameras, sparse_dir / "cameras.txt")
    write_images_text(images, sparse_dir / "images.txt")
    write_points3D_text(points3D, sparse_dir / "points3D.txt")


def _load_dynamic_masks_for_depth(
    *,
    mask_dir: Path,
    pattern: str,
    t_expected: int,
    target_hw: tuple[int, int],
) -> np.ndarray:
    """Load dynamic masks as [T,H,W] bool where True means dynamic pixel."""
    paths = sorted(glob.glob(str(mask_dir / pattern)))
    if len(paths) == 0:
        return np.zeros((t_expected, target_hw[0], target_hw[1]), dtype=bool)
    if len(paths) < t_expected:
        t_expected = len(paths)

    masks_pos = []
    for p in paths[:t_expected]:
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise RuntimeError(f"Failed to read mask: {p}")
        m = cv2.resize(m, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
        masks_pos.append(m > 0)
    masks_pos = np.stack(masks_pos, axis=0)

    # Same auto convention as tracking: if positives dominate, invert.
    pos_ratio = float(np.mean(masks_pos))
    dynamic_masks = ~masks_pos if pos_ratio > 0.5 else masks_pos
    return dynamic_masks.astype(bool)


def _lift_da3_static_points_to_target(
    *,
    depth_maps: np.ndarray,
    dynamic_masks: np.ndarray,
    K_da3_all: np.ndarray,
    R_raw: np.ndarray,
    t_raw: np.ndarray,
    w2t: np.ndarray,
    image_paths: list[Path],
    sample_stride: int,
    max_points_per_frame: int,
    seed: int,
    frame_scales: np.ndarray | None = None,
    frame_losses: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lift DA3 static (non-dynamic) pixels with solver cameras, then transform to target frame."""
    T, H_d, W_d = depth_maps.shape
    pts_all = []
    rgb_all = []
    if frame_scales is None:
        frame_scales = np.ones((T,), dtype=np.float64)
    else:
        frame_scales = np.asarray(frame_scales, dtype=np.float64)
    if frame_losses is None:
        frame_losses = np.full((T,), np.nan, dtype=np.float64)
    else:
        frame_losses = np.asarray(frame_losses, dtype=np.float64)

    def frame(t):   # (p_target, rgb, log line) of frame t, None if it keeps no point
        depth = depth_maps[t].astype(np.float32)
        K = K_da3_all[t].astype(np.float64)
        R = R_raw[t].astype(np.float64)
        center = t_raw[t].astype(np.float64)
        img = cv2.imread(str(image_paths[t]), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {image_paths[t]}")
        H, W = img.shape[:2]

        frame_scale = float(frame_scales[t]) if t < frame_scales.shape[0] else 1.0
        frame_loss = float(frame_losses[t]) if t < frame_losses.shape[0] else float("nan")

        dyn = dynamic_masks[t].astype(bool)

        ys = np.arange(0, H_d, int(max(1, sample_stride)), dtype=np.int64)
        xs = np.arange(0, W_d, int(max(1, sample_stride)), dtype=np.int64)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        yy = yy.reshape(-1)
        xx = xx.reshape(-1)

        z = depth[yy, xx].astype(np.float64)
        ok = np.isfinite(z) & (z > 0.0) & ~dyn[yy, xx]
        if not np.any(ok):
            return None

        yy = yy[ok]
        xx = xx[ok]
        z = z[ok]
        rgb, _u_img, _v_img = _sample_image_rgb_for_depth_pixels(
            img_bgr=img,
            xx=xx,
            yy=yy,
            depth_shape=(H_d, W_d),
        )

        # Empty/undefined image regions (undistortion or aspect-crop borders) are EXACTLY black (0,0,0);
        # they still carry a depth value, so lifting them makes floating black points. Skip them.
        nonblack = rgb.max(axis=1) > 0
        black_removed = int((~nonblack).sum())
        if black_removed:
            yy, xx, z, rgb = yy[nonblack], xx[nonblack], z[nonblack], rgb[nonblack]
        if z.shape[0] == 0:
            return None

        if z.shape[0] > int(max_points_per_frame):
            frame_rng = np.random.default_rng(seed + int(t))
            pick = frame_rng.choice(z.shape[0], size=int(max_points_per_frame), replace=False)
            yy = yy[pick]
            xx = xx[pick]
            z = z[pick]
            rgb = rgb[pick]

        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        p_cam = np.stack(
            [
                (xx.astype(np.float64) - cx) / fx * z,
                (yy.astype(np.float64) - cy) / fy * z,
                z,
            ],
            axis=1,
        )
        p_world = (float(frame_scale) * (R @ p_cam.T)).T + center[None, :]

        # Remove isolated / structureless 3D points (floating noise).
        outlier_removed = 0
        if p_world.shape[0] > int(DA3_OUTLIER_K) + 1:
            outlier = _statistical_outlier_mask(
                p_world.astype(np.float32), int(DA3_OUTLIER_K), float(DA3_OUTLIER_STD_RATIO)
            )
            outlier_removed = int(outlier.sum())
            p_world = p_world[~outlier]
            rgb = rgb[~outlier]

        if p_world.shape[0] == 0:
            return None

        p_target = _apply_transform(p_world, w2t)
        return p_target, rgb, (
            f"[da3-lift] frame={t}: kept={p_target.shape[0]} "
            f"scale={frame_scale:.6g} loss={frame_loss:.6g} "
            f"dyn_removed={int(np.count_nonzero(dyn))} black_removed={black_removed} outlier_removed={outlier_removed}"
        )
    with ThreadPoolExecutor(N_WORKERS) as ex:   # independent frames (own seeded RNG; numpy/cv2/KD-tree release the GIL)
        for res in ex.map(frame, range(T)):     # in frame order
            if res is not None:
                pts_all.append(res[0])
                rgb_all.append(res[1])
                print(res[2])

    if len(pts_all) == 0:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.uint8),
            frame_scales,
        )

    return (
        np.concatenate(pts_all, axis=0),
        np.concatenate(rgb_all, axis=0).astype(np.uint8),
        frame_scales,
    )


def _sanity_check_cameras(
    *,
    c2w_all: np.ndarray,
    K_all: np.ndarray,
    image_paths: list[Path],
) -> None:
    """Basic checks for pose/intrinsic consistency for 3DGS inputs."""
    if c2w_all.ndim != 3 or c2w_all.shape[1:] != (4, 4):
        raise ValueError(f"Expected c2w_all (T,4,4), got {c2w_all.shape}")
    if K_all.ndim != 3 or K_all.shape[1:] != (3, 3):
        raise ValueError(f"Expected K_all (T,3,3), got {K_all.shape}")
    if c2w_all.shape[0] != K_all.shape[0]:
        raise ValueError(f"Pose/intrinsic frame mismatch: {c2w_all.shape[0]} vs {K_all.shape[0]}")
    if len(image_paths) < c2w_all.shape[0]:
        raise ValueError(f"Not enough images: {len(image_paths)} for T={c2w_all.shape[0]}")

    R = c2w_all[:, :3, :3]
    det = np.linalg.det(R)
    if np.any(~np.isfinite(det)):
        raise ValueError("Non-finite camera rotations detected.")
    det_mean = float(np.mean(det))
    if abs(det_mean - 1.0) > 1e-2:
        print(f"[WARN] mean det(R_c2w)={det_mean:.4f}, expected near 1.0")

    fx = K_all[:, 0, 0]
    fy = K_all[:, 1, 1]
    if np.any(fx <= 0) or np.any(fy <= 0):
        raise ValueError("Non-positive focal lengths in intrinsics.")
    print(f"Camera sanity: T={c2w_all.shape[0]}, fx~{float(np.median(fx)):.2f}, fy~{float(np.median(fy)):.2f}")


def _load_depth_maps_for_static_lifting(
    *,
    rootpath: Path,
    t_expected: int,
) -> np.ndarray:
    """Load the depth maps of step 1 (outputs/depth_anything_3/depth_maps.npy): (T,H,W) float64."""
    depth_path = rootpath / "outputs" / "depth_anything_3" / "depth_maps.npy"
    depth_maps = np.load(str(depth_path)).astype(np.float64)
    if depth_maps.ndim != 3:
        raise ValueError(f"Expected depth_maps (T,H,W), got {depth_maps.shape}")
    return depth_maps[: min(t_expected, depth_maps.shape[0])]


def prepare_4dgs_dataset(
    *,
    target_frame: int,
    output_dir: Path,
    max_points: int = MAX_CANONICAL_POINTS,
    k_neighbors: int = K_NEIGHBORS,
    seed: int = POINT_SAMPLE_SEED,
    dyn_min_points: int = GS_DYN_MIN_POINTS,
) -> Path:
    """Prepare the COLMAP text dataset consumed by gsplat simple_trainer.py."""
    output_dir = Path(output_dir)

    gs_dir = output_dir / f"gs_edg_target_{target_frame:04d}"
    gs_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in ("points3d.ply", "transforms_train.json", "transforms_test.json", "dynamic_edg.npz"):
        stale_path = gs_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    results_path = output_dir / "results_all.npz"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results_all.npz at {results_path}")

    results = np.load(results_path, allow_pickle=True)
    edg_tracks_dict = results["edg_reconstruct"].item()
    edg_rot_dict = results.get("edg_rotation", None)
    dyn_tracks_dict = results["dynamics_points_reconstruct"].item()

    edg_tracks, edg_gids = _as_track_array(edg_tracks_dict, len(next(iter(edg_tracks_dict.values()))))
    dyn_tracks, dyn_gids = _as_track_array(dyn_tracks_dict, len(next(iter(dyn_tracks_dict.values()))))
    if edg_rot_dict is None:
        raise RuntimeError("Missing edg_rotation in results_all.npz; regenerate EDG rotations.")
    edg_rots, edg_rot_gids = _as_rotation_array(edg_rot_dict.item(), len(next(iter(edg_rot_dict.item().values()))))
    if not np.array_equal(edg_gids, edg_rot_gids):
        raise RuntimeError("Mismatch between edg_reconstruct and edg_rotation node ordering.")
    T = edg_tracks.shape[1] if edg_tracks.shape[0] > 0 else dyn_tracks.shape[1]

    # Load camera poses from XM outputs
    rt = np.load(str(utils.definitions.rootpath / "outputs" / "result_XM.npz"))
    # The static solver may drop frames (checklandmarks / XM^2), so R/t are stored in
    # solved order (N<=T). Scatter them onto the full timeline via indices_all, then fill
    # any dropped frame from the nearest solved frame so every frame has a valid camera
    # (4DGS renders all frames, so NaN/identity poses are not acceptable here).
    R_raw, t_raw = scatter_solved_poses(rt, T)
    _finite_pose = np.isfinite(t_raw).all(axis=1)
    if not _finite_pose.all():
        _kept = np.where(_finite_pose)[0]
        _dropped = np.where(~_finite_pose)[0]
        if _kept.size == 0:
            raise RuntimeError("result_XM.npz has no valid camera poses.")
        for _i in _dropped:
            _j = int(_kept[np.argmin(np.abs(_kept - _i))])
            R_raw[_i] = R_raw[_j]
            t_raw[_i] = t_raw[_j]
        print(f"[cameras] filled {_dropped.size} solver-dropped frame(s) {_dropped.tolist()} "
              f"from nearest solved frame for 4DGS.")

    # Always reuse da3_frame_scale computed by static_solver.py (single source of truth, shared with the
    # simple_trainer depth loss).
    _solver_da3_scale = None
    _solver_da3_scale_loss = None
    if "da3_frame_scale" in rt.files:
        # Scatter the per-solved-frame scale onto the full timeline (dropped frames -> 1.0).
        _s = scatter_solved_frame_vector(rt, "da3_frame_scale", T, default=1.0)
        if _s is not None and np.all(np.isfinite(_s)) and np.all(_s > 0):
            _solver_da3_scale = _s
            if "da3_frame_scale_loss" in rt.files:
                _solver_da3_scale_loss = scatter_solved_frame_vector(rt, "da3_frame_scale_loss", T, default=np.nan)
            print(
                f"[da3-scale] loaded from result_XM.npz: "
                f"min={_solver_da3_scale.min():.6g} "
                f"median={float(np.median(_solver_da3_scale)):.6g} "
                f"max={_solver_da3_scale.max():.6g}"
            )
        else:
            print(
                f"[da3-scale] result_XM.npz has da3_frame_scale but it is invalid "
                f"(shape={_s.shape}, T={T}); not used."
            )

    # Intrinsics (outputs/depth_anything_3/intrinsics.npy).
    K_all = _load_intrinsics_for_training(T)

    # Transform to target frame
    w2t = _world_to_target_transform(R_raw[target_frame], t_raw[target_frame])
    c2w_all = []                                           # target-frame cameras (w2t applied)
    for i in range(T):
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = R_raw[i]
        c2w[:3, 3] = t_raw[i]
        c2w_all.append(w2t @ c2w)
    c2w_all = np.stack(c2w_all, axis=0)

    # Static initialization is always lifted from depth.
    canonical_pts = []
    canonical_rgb = []

    # Camera/intrinsic sanity for 3DGS
    image_paths = sorted(
        Path(utils.definitions.video_dir).glob(utils.definitions.image_pattern),
        key=frame_sort_key,
    )
    _sanity_check_cameras(c2w_all=c2w_all, K_all=K_all, image_paths=image_paths)

    # Add static points lifted from depth, then
    # transform the resulting solver-world points into the target frame.
    depth_maps = _load_depth_maps_for_static_lifting(
        rootpath=Path(utils.definitions.rootpath),
        t_expected=T,
    )
    T_depth_for_scale = min(T, depth_maps.shape[0])
    # ALWAYS use the static-solver per-frame scale (result_XM da3_frame_scale) -- the SAME scale
    # simple_trainer's depth loss uses -- so the static lift and the depth loss never diverge. If the
    # solver scale is absent, lift with scale=1.0 (matching the loss's fallback); NEVER re-fit here.
    if _solver_da3_scale is not None:
        da3_frame_scales = _solver_da3_scale[:T_depth_for_scale]
        da3_frame_losses = (
            _solver_da3_scale_loss[:T_depth_for_scale]
            if _solver_da3_scale_loss is not None
            else np.full((T_depth_for_scale,), np.nan, dtype=np.float64)
        )
    else:
        print("[da3-scale] result_XM has no da3_frame_scale -> lifting with scale=1.0 to match the depth "
              "loss. Run static_solver to fit the per-frame scale.")
        da3_frame_scales = None
        da3_frame_losses = None

    T_depth = min(T, depth_maps.shape[0])
    depth_maps = depth_maps[:T_depth]

    dynamic_masks = _load_dynamic_masks_for_depth(
        mask_dir=Path(utils.definitions.rootpath) / "masks",
        pattern=utils.definitions.mask_pattern,
        t_expected=T_depth,
        target_hw=(depth_maps.shape[1], depth_maps.shape[2]),
    )
    dynamic_masks = dynamic_masks[:T_depth]
    static_pts_target, static_rgb, _da3_scales = _lift_da3_static_points_to_target(
        depth_maps=depth_maps,
        dynamic_masks=dynamic_masks,
        # Use the single-source intrinsic (K_all, from intrinsics.npy) so the
        # dense static lift matches cameras.txt exactly.
        K_da3_all=K_all[:T_depth],
        R_raw=R_raw[:T_depth],
        t_raw=t_raw[:T_depth],
        w2t=w2t,
        image_paths=image_paths[:T_depth],
        sample_stride=STATIC_SAMPLE_STRIDE,
        max_points_per_frame=STATIC_MAX_POINTS_PER_FRAME,
        seed=seed,
        frame_scales=da3_frame_scales[:T_depth] if da3_frame_scales is not None else None,
        frame_losses=da3_frame_losses[:T_depth] if da3_frame_losses is not None else None,
    )
    if static_pts_target.shape[0] > 0:
        canonical_pts.append(static_pts_target)
        canonical_rgb.append(static_rgb)
        print(f"Added static lifted points (DA3 lift_da3_points.py match): {static_pts_target.shape[0]}")

    # Build dynamic points: the moving-object depth pixels of every frame, each bound to the EDG at its own frame.
    (dyn_xyz_target, dyn_rgb, dyn_knn_idx, dyn_knn_w, dyn_rel) = _build_edg_good_dynamic_points(
        output_dir=output_dir,
        target_frame=target_frame,
        w2t=w2t,
        edg_gids=edg_gids,
        k_neighbors=int(k_neighbors),
        dyn_min_points=int(dyn_min_points),
        depth_maps=depth_maps,
        R_raw=R_raw,
        t_raw=t_raw,
        K_all=K_all,
        image_paths=image_paths,
        dynamic_masks=dynamic_masks,   # the static lift's masks (same files, frames and size)
        frame_scales=da3_frame_scales,
    )
    if dyn_xyz_target.shape[0] > 0:
        canonical_pts.append(dyn_xyz_target)
        canonical_rgb.append(dyn_rgb.astype(np.uint8))
        print(f"Added EDG-good dynamic points (edg_completed.npz): {dyn_xyz_target.shape[0]}")
    else:
        print("[WARN] No EDG-good dynamic points survived skinning/EDG binding.")

    if len(canonical_pts) == 0:
        raise RuntimeError("No canonical points were generated for 4DGS initialization.")

    canonical_pts = np.concatenate(canonical_pts, axis=0)
    canonical_rgb = np.concatenate(canonical_rgb, axis=0)

    dyn_count = dyn_xyz_target.shape[0]

    if canonical_pts.shape[0] > int(max_points):
        rng = np.random.default_rng(seed)
        if dyn_count > 0:
            static_count = canonical_pts.shape[0] - dyn_count
            max_static = max(0, int(max_points) - dyn_count)
            if static_count > max_static:
                static_idx = rng.choice(static_count, size=max_static, replace=False)
                dyn_slice = slice(static_count, static_count + dyn_count)
                canonical_pts = np.concatenate([canonical_pts[static_idx], canonical_pts[dyn_slice]], axis=0)
                canonical_rgb = np.concatenate([canonical_rgb[static_idx], canonical_rgb[dyn_slice]], axis=0)
        else:
            idx = rng.choice(canonical_pts.shape[0], size=int(max_points), replace=False)
            canonical_pts = canonical_pts[idx]
            canonical_rgb = canonical_rgb[idx]

    sparse_dir = gs_dir / "sparse" / "0"
    _export_colmap_text_dataset(
        sparse_dir=sparse_dir,
        image_paths=image_paths,
        c2w_all=c2w_all,
        K_all=K_all,
        points=canonical_pts,
        rgb=canonical_rgb,
    )
    print(f"Wrote gsplat COLMAP text dataset at: {sparse_dir}")

    # Save dynamic EDG metadata (if any)
    if dyn_xyz_target.size > 0:
        dyn_offset = int(canonical_pts.shape[0] - dyn_xyz_target.shape[0])
        dyn_indices = np.arange(dyn_offset, dyn_offset + dyn_xyz_target.shape[0], dtype=np.int64)
        np.savez_compressed(
            gs_dir / "dynamic_edg.npz",
            dyn_indices=dyn_indices,
            dyn_knn_idx=dyn_knn_idx,
            dyn_knn_w=dyn_knn_w,
            dyn_rel=dyn_rel,
            dyn_rgb=dyn_rgb.astype(np.uint8),
            edg_gids=edg_gids,
            target_frame=int(target_frame),
        )

    # Link/copy images
    images_dir = gs_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for p in image_paths:
        dst = images_dir / p.name
        if dst.exists():
            continue
        try:
            os.symlink(p, dst)
        except OSError:
            # fallback to copy
            dst.write_bytes(p.read_bytes())

    return gs_dir


def main() -> None:
    gs_dir = prepare_4dgs_dataset(target_frame=DEFAULT_TARGET_FRAME, output_dir=Path(utils.definitions.output_dir))

    print(f"Prepared 4DGS dataset at: {gs_dir}")
    if GS_DATA_FACTOR > 1:                                # read by step 6's --data-factor
        small = gs_dir / f"images_{GS_DATA_FACTOR}"
        small.mkdir(exist_ok=True)
        for p in sorted((gs_dir / "images").iterdir()):
            im = cv2.imread(str(p))
            cv2.imwrite(str(small / p.name), cv2.resize(im, (im.shape[1] // GS_DATA_FACTOR, im.shape[0] // GS_DATA_FACTOR),
                                                        interpolation=cv2.INTER_AREA))
        print(f"Frames downscaled by {GS_DATA_FACTOR} for training: {small}")

    gsplat_result_dir = gs_dir.parent / "gsplat"
    simple_trainer = Path(__file__).resolve().parents[1] / "build" / "gsplat_examples" / "simple_trainer.py"   # install/gs.sh
    results_npz = Path(utils.definitions.output_dir) / "results_all.npz"
    train_cmd = [
        "CUDA_VISIBLE_DEVICES=0",
        "python",
        str(simple_trainer),
        "default",
        "--data-dir",
        str(gs_dir),
        "--result-dir",
        str(gsplat_result_dir),
        "--data-factor",
        str(GS_DATA_FACTOR),
        "--edg-results",
        str(results_npz),
        "--edg-target-frame",
        str(DEFAULT_TARGET_FRAME),
    ]
    print("Recommended gsplat 4DGS training command:")
    print(shlex.join(train_cmd))


if __name__ == "__main__":
    main()
