#!/usr/bin/env python3
"""Shared numpy/cv2 helpers for the EDG 2D visualization scripts (single source of truth).

Pure numpy/cv2 + utils (NO open3d, NO torch). Provides the world -> camera -> pixel projection, the 2D
reprojection context (initial 2D tracks + XM lift poses + per-frame intrinsics + frame images), the
embedded-deformation skinning of the dynamic tracks, and the photometric render-colour scorer. Imported by
visualize_longest_track_2d.py, visualize_best_color_2d.py and visualize_edg_filtered_open3d.py so the exact
same projection / skinning / scoring is used everywhere. This module has NO __main__.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import utils.definitions


def project_to_frame(pts_world, ctx, f):
    """World -> camera -> pixel projection of `pts_world` (N,3 solver-world) into frame `f`: camera space via
    the inverse XM lift x_cam = R_f^T (x - t_f) / s_f, then pinhole with ctx["K"][f]. Returns (u, v, ok): the
    raw pixel coordinates u(N), v(N) and a validity mask ok(N) that is True where the point lifts finitely in
    front of the camera (Z > 1e-9). u/v are NOT masked -- callers gate on `ok` exactly as the inlined code did."""
    xc = ((pts_world - ctx["t"][f]) @ ctx["R"][f]) / ctx["s"][f]     # row form of R^T (x - t) / s
    Z = xc[:, 2]
    ok = np.isfinite(xc).all(1) & (Z > 1e-9)
    fx, fy, cx, cy = ctx["K"][f]
    u = fx * xc[:, 0] / np.where(Z > 0, Z, 1.0) + cx
    v = fy * xc[:, 1] / np.where(Z > 0, Z, 1.0) + cy
    return u, v, ok


# ------------------------------------------------- Filter 3: photometric render-colour outlier detection
def _build_dyn_skin(dyn_ids, dr, node_ids, edg_pos, edg_rot):
    """Embedded-deformation skinning for the dynamic tracks (same rule as visualize_edg_solution_open3d's
    _build_dyn_skinning) but ALSO returning `pids` = the dyn global id of every skinned row, so the render-
    colour outlier flag can be attributed per track. Bind frame = each track's first finite frame; neighbour
    EDG nodes come from edg_topology.npz's dyn_edg_edges; weights = normalized inverse distance at bind.
    Returns (sp, pids) or (None, None)."""
    topo = Path(utils.definitions.output_dir) / "edg_topology.npz"
    if not topo.exists():
        print("[render-outlier] edg_topology.npz missing; cannot skin dynamic points.")
        return None, None
    td = np.load(topo, allow_pickle=True)
    if "dyn_edg_edges" not in td.files:
        print("[render-outlier] dyn_edg_edges missing in edg_topology.npz; cannot skin dynamic points.")
        return None, None
    dyn_edg = np.asarray(td["dyn_edg_edges"], dtype=np.int64).reshape(-1, 2)
    node_loc = {int(g): i for i, g in enumerate(np.asarray(node_ids).reshape(-1).tolist())}
    d2n = {}
    for d, n in dyn_edg.tolist():
        if int(n) in node_loc:
            d2n.setdefault(int(d), []).append(node_loc[int(n)])

    P_p, P_nl, P_w, P_rel, pids = [], [], [], [], []
    for i, gid in enumerate(np.asarray(dyn_ids).reshape(-1).tolist()):
        nls = d2n.get(int(gid))
        if not nls:
            continue
        vf = np.where(np.isfinite(dr[i]).all(axis=1))[0]
        if vf.size == 0:
            continue
        sk = int(vf[0]); xsrc = dr[i, sk]
        nls = np.asarray(nls, dtype=np.int64)
        Xsk = edg_pos[nls, sk]; Rsk = edg_rot[nls, sk]
        ok = np.isfinite(Xsk).all(1) & np.isfinite(Rsk).reshape(nls.size, -1).all(1)
        nls, Xsk, Rsk = nls[ok], Xsk[ok], Rsk[ok]
        if nls.size == 0:
            continue
        dist = np.linalg.norm(Xsk - xsrc[None, :], axis=1)
        w = 1.0 / (dist + 1e-6); w = w / w.sum()
        rel = np.einsum("mji,mj->mi", Rsk, xsrc[None, :] - Xsk)       # R_m(s_k)^T (x_src - X_m(s_k))
        pidx = len(pids); pids.append(int(gid))
        for m in range(nls.size):
            P_p.append(pidx); P_nl.append(int(nls[m])); P_w.append(float(w[m])); P_rel.append(rel[m])
    if not pids:
        print("[render-outlier] no skinnable dynamic points.")
        return None, None
    sp = dict(P=len(pids), p=np.asarray(P_p, np.int64), nl=np.asarray(P_nl, np.int64),
              w=np.asarray(P_w, np.float64), rel=np.asarray(P_rel, np.float64),
              edg_pos=edg_pos, edg_rot=edg_rot)
    return sp, np.asarray(pids, np.int64)


def _skin_dyn_frame(sp, frame):
    """Warp every skinned dynamic point to `frame` (embedded deformation); returns (P,3) with NaN rows where
    the point cannot be placed (all its neighbour nodes are holes at that frame)."""
    nl = sp["nl"]
    Xt = sp["edg_pos"][nl, frame]
    Rt = sp["edg_rot"][nl, frame]
    ok = np.isfinite(Xt).all(1) & np.isfinite(Rt).reshape(nl.size, -1).all(1)
    Xs = np.where(ok[:, None], Xt, 0.0)
    Rs = np.where(ok[:, None, None], Rt, 0.0)
    mapped = Xs + np.einsum("eij,ej->ei", Rs, sp["rel"])
    num = np.zeros((sp["P"], 3)); den = np.zeros(sp["P"])
    np.add.at(num, sp["p"][ok], sp["w"][ok, None] * mapped[ok])
    np.add.at(den, sp["p"][ok], sp["w"][ok])
    pos = np.full((sp["P"], 3), np.nan)
    good = den > 1e-9
    pos[good] = num[good] / den[good, None]
    return pos


def _sample_frame_color(img, positions, Rf, tf, sf, Kf, patch):
    """Project `positions` (N,3 solver-world) into one image (inverse XM lift + pinhole) and sample a BGR
    patch-median colour, returned as HSV hue/sat and the BGR median. Returns hue(N), sat(N), bgr(N,3), inb(N)
    -- exactly the projection used by the --viz-2d path so the sampled pixel matches what is drawn there."""
    import cv2
    N = positions.shape[0]
    H, W = img.shape[:2]
    xc = ((positions - tf) @ Rf) / sf                                # row form of R_f^T (x - t_f) / s_f
    Z = xc[:, 2]
    ok = np.isfinite(xc).all(1) & (Z > 1e-9)
    fx, fy, cx, cy = Kf
    u = np.where(ok, fx * xc[:, 0] / np.where(Z > 0, Z, 1.0) + cx, np.nan)
    v = np.where(ok, fy * xc[:, 1] / np.where(Z > 0, Z, 1.0) + cy, np.nan)
    ui = np.rint(u); vi = np.rint(v)
    good = ok & np.isfinite(ui) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    hue = np.full(N, np.nan, np.float32); sat = np.full(N, np.nan, np.float32)
    bgr = np.full((N, 3), np.nan, np.float32)
    gi = np.where(good)[0]
    if gi.size:
        uii = ui[gi].astype(int); vii = vi[gi].astype(int)
        p = max(int(patch), 0)
        offs = [(dx, dy) for dy in range(-p, p + 1) for dx in range(-p, p + 1)]
        samples = np.empty((gi.size, len(offs), 3), np.float32)
        for j, (dx, dy) in enumerate(offs):
            xx = np.clip(uii + dx, 0, W - 1); yy = np.clip(vii + dy, 0, H - 1)
            samples[:, j, :] = img[yy, xx]
        med = np.median(samples, axis=1).astype(np.uint8)
        hsv = cv2.cvtColor(med[None], cv2.COLOR_BGR2HSV)[0]
        hue[gi] = hsv[:, 0]; sat[gi] = hsv[:, 1]; bgr[gi] = med
    return hue, sat, bgr, good


def _circ_hue_median(h):
    """Robust circular location of hue values (deg 0..180): argmin over candidates of summed circular dist."""
    h = h[np.isfinite(h)]
    if h.size == 0:
        return np.nan
    d = np.abs(h[:, None] - h[None, :]); d = np.minimum(d, 180.0 - d)
    return float(h[np.argmin(d.sum(1))])


def detect_render_color_outliers(sp, pids, dyn_ids, dr, ctx, args, return_color=False):
    """PHOTOMETRIC render-colour outlier test, per DYNAMIC track. For every track:
      * reference colour = the colour it lands on over its OBSERVED frames, sampled at its RAW reconstructed
        position (where real 2D tracking anchors it to the correct surface). Observed-frame anchoring is
        essential -- a mostly-misplaced point would otherwise self-reference to its wrong colour and escape.
      * per-frame drift  = colour distance between the SKINNED (embedded-deformation) position projected into
        each frame and the reference. Frames outside the observed set are the render-harm frames.
      * score            = the --render-color-quantile quantile of the drift over the extrapolated frames.
    A track with score > --render-color-thresh (and enough observed/extrapolated frames) is an OUTLIER: when
    the deformation graph places it at frames it was not observed, it lands on the wrong surface colour.
    Returns (flag[P] bool, score[P] float)."""
    import cv2  # noqa: F401  (kept local; the module is otherwise torch/cv2-free at import time)
    P = int(sp["P"]); T = int(sp["edg_pos"].shape[1])
    Tframes = min(T, len(ctx["image_paths"]))
    space = args.render_color_space
    sat_min = int(args.render_sat_min) if space == "hue" else 0
    gid2di = {int(g): i for i, g in enumerate(np.asarray(dyn_ids).reshape(-1).tolist())}
    di_of_row = np.asarray([gid2di[int(g)] for g in pids.tolist()], np.int64)
    valid = ctx["valid"]                                             # (Q,T) solver observation mask
    R, t, s, K = ctx["R"], ctx["t"], ctx["s"], ctx["K"]

    hue_s = np.full((P, Tframes), np.nan, np.float32); sat_s = np.full((P, Tframes), np.nan, np.float32)
    bgr_s = np.full((P, Tframes, 3), np.nan, np.float32); inb_s = np.zeros((P, Tframes), bool)
    hue_r = np.full((P, Tframes), np.nan, np.float32); sat_r = np.full((P, Tframes), np.nan, np.float32)
    bgr_r = np.full((P, Tframes, 3), np.nan, np.float32)
    obs = np.zeros((P, Tframes), bool)
    dr_rows = dr[di_of_row]                                          # (P,T,3) raw reconstructed positions
    print(f"[render-outlier] sampling reprojection colours over {Tframes} frames "
          f"(space={space}, patch={args.render_patch}, sat_min={sat_min}) ...")
    for f in range(Tframes):
        img = cv2.imread(ctx["image_paths"][f])
        if img is None:
            continue
        pos_s = _skin_dyn_frame(sp, f)                              # skinned (all frames)
        h, sa, bg, g = _sample_frame_color(img, pos_s, R[f], t[f], s[f], K[f], args.render_patch)
        hue_s[:, f] = h; sat_s[:, f] = sa; bgr_s[:, f] = bg; inb_s[:, f] = g
        obs_f = valid[pids, f] & np.isfinite(dr_rows[:, f]).all(1)  # observed at f (real 2D anchor)
        if obs_f.any():
            h2, sa2, bg2, g2 = _sample_frame_color(img, np.where(obs_f[:, None], dr_rows[:, f], np.nan),
                                                   R[f], t[f], s[f], K[f], args.render_patch)
            hue_r[:, f] = h2; sat_r[:, f] = sa2; bgr_r[:, f] = bg2
            obs[:, f] = obs_f & g2

    score = np.full(P, np.nan, np.float32)
    ref_hue = np.full(P, np.nan, np.float32); ref_bgr = np.full((P, 3), np.nan, np.float32)
    fused = np.full((P, 3), np.nan, np.float32)          # per-track fused colour = median observed-frame BGR
    n_scored = 0
    for r in range(P):
        om = obs[r] & (np.nan_to_num(sat_r[r], nan=-1.0) >= sat_min if space == "hue" else obs[r])
        if int(om.sum()) < int(args.render_min_obs):
            continue
        fused[r] = np.median(bgr_r[r, om], axis=0)       # colour of the surface the track observes (both spaces)
        if space == "hue":
            ref_hue[r] = _circ_hue_median(hue_r[r, om])
            ext = inb_s[r] & (~obs[r]) & (np.nan_to_num(sat_s[r], nan=-1.0) >= sat_min)
            if int(ext.sum()) < int(args.render_min_span):
                continue
            d = np.abs(hue_s[r, ext] - ref_hue[r]); d = np.minimum(d, 180.0 - d)
        else:
            ref_bgr[r] = np.median(bgr_r[r, om], axis=0)
            ext = inb_s[r] & (~obs[r])
            if int(ext.sum()) < int(args.render_min_span):
                continue
            d = np.linalg.norm(bgr_s[r, ext] - ref_bgr[r][None, :], axis=1)
        score[r] = float(np.quantile(d, float(args.render_color_quantile)))
        n_scored += 1
    flag = np.isfinite(score) & (score > float(args.render_color_thresh))

    sc = score[np.isfinite(score)]
    n_dyn = int(np.asarray(dyn_ids).size)
    unit = "deg" if space == "hue" else "BGR"
    print(f"[render-outlier] dynamic tracks: {n_dyn} total | {P} skinnable | {n_scored} scored | "
          f"{int(flag.sum())} OUTLIERS (score>{args.render_color_thresh:g} {unit}, "
          f"{100.0 * flag.sum() / max(n_dyn, 1):.1f}% of all)")
    if sc.size:
        print(f"[render-outlier] score distribution ({unit}): "
              f"p50={np.percentile(sc, 50):.1f} p75={np.percentile(sc, 75):.1f} "
              f"p90={np.percentile(sc, 90):.1f} p95={np.percentile(sc, 95):.1f} max={sc.max():.1f}")
        worst = np.argsort(-np.where(np.isfinite(score), score, -1))[:6]
        print("[render-outlier] worst tracks: " +
              ", ".join(f"gid={int(pids[r])}(score={score[r]:.0f})" for r in worst if flag[r]))
    return (flag, score, fused) if return_color else (flag, score)


# ------------------------------------------------- 2D reprojection context: intrinsics + XM lift + images
def _fit_frame_intrinsics(xyz_cam, xy, occ, T):
    """Per-frame pinhole [fx,fy,cx,cy] by least squares from camera-space tracks vs their pixel tracks
    (the tracking lifted with u=fx*X/Z+cx, so the fit is exact by construction)."""
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
    for t in np.where(~np.isfinite(K).all(1))[0]:
        K[t] = K[good[np.argmin(np.abs(good - t))]]
    return K


def _load_2d_context(merged: bool, n_tracks: int | None = None):
    """Initial 2D tracks + XM lift poses + per-frame intrinsics + frame images, for the 2D reprojection view.
    If the solved npz carries an `n_tracks` stamp, the matching tracks file (merged vs raw) is selected
    AUTOMATICALLY -- a wrong --merged flag would otherwise make the node ids point at the wrong tracks
    silently (observations look missing/garbled)."""
    import glob
    from utils.edg import scatter_solved_poses, scatter_solved_frame_vector
    out_dir = Path(utils.definitions.output_dir)
    # merge is 2D-only now -> the 3D lives in the smoothed OR lift file (optimize_dynamic_tracks output,
    # the latter with --iters 0). Candidates in preference order; the raw merged file comes LAST because
    # its tracks_xyz is all-NaN. A stale smoothed/lift file from an older track pool is skipped by the
    # n_tracks stamp.
    raw_p = Path(utils.definitions.tracks_dynamics_path)
    merged_cands = [out_dir / "tracks_dynamic_merged_smoothed.npz",
                    out_dir / "tracks_dynamic_merged_lift.npz",
                    out_dir / "tracks_dynamic_merged.npz"]
    cands = ([p_ for p_ in merged_cands if p_.exists()] + [raw_p]) if merged \
        else [raw_p] + [p_ for p_ in merged_cands if p_.exists()]
    tracks_path, data = cands[0], None
    if n_tracks is None:
        data = np.load(str(tracks_path), allow_pickle=True)
    else:
        sizes = {}
        for p_ in cands:
            d_ = np.load(str(p_), allow_pickle=True)
            sizes[p_.name] = int(d_["tracks_xy"].shape[1])
            if sizes[p_.name] == int(n_tracks):
                if p_ != cands[0]:
                    print(f"[2d] npz stamp n_tracks={n_tracks} does not match {cands[0].name} -> "
                          f"auto-switching to {p_.name}")
                tracks_path, data = p_, d_
                break
        if data is None:
            raise ValueError(f"npz stamp n_tracks={n_tracks} matches none of {sizes}; "
                             "re-run the tracking/solve pipeline.")
    xy = np.asarray(data["tracks_xy"], np.float64)[0]                 # (Q,T,2) pixels
    xyz = np.asarray(data["tracks_xyz"], np.float64)[0]               # (Q,T,3) camera-space
    occ = np.asarray(data["occluded"])[0].astype(bool)                # (Q,T)
    dyn_mask = np.asarray(data["dynamic_obs_mask"]).astype(bool)      # (Q,T)
    Q, T = xy.shape[0], xy.shape[1]
    rt = np.load(str(out_dir / "result_XM.npz"))
    R_raw, t_raw = scatter_solved_poses(rt, T)
    s_raw = scatter_solved_frame_vector(rt, "s", T, default=1.0)
    # observation mask EXACTLY as EDG_test.load_all builds `valid` for the solver: the observation must
    # world-lift finitely AND be unoccluded AND pass the dynamic mask.
    R_for = R_raw.astype(np.float64) * (s_raw[:, None, None] if s_raw is not None else 1.0)
    glb = np.einsum("tij,qtj->qti", R_for, xyz) + t_raw.astype(np.float64)[None, :, :]
    valid = np.isfinite(glb).all(-1) & (~occ) & dyn_mask
    K = _fit_frame_intrinsics(xyz, xy, occ, T)
    image_paths = sorted(glob.glob(str(Path(utils.definitions.video_dir) / utils.definitions.image_pattern)))
    print(f"[2d] context: {tracks_path.name} Q={Q} T={T} | {len(image_paths)} frame images | "
          f"K[0]: fx={K[0, 0]:.1f} fy={K[0, 1]:.1f} cx={K[0, 2]:.1f} cy={K[0, 3]:.1f}")
    return dict(xy=xy, valid=valid, Q=Q, T=T, R=R_raw.astype(np.float64), t=t_raw.astype(np.float64),
                s=(s_raw.astype(np.float64) if s_raw is not None else np.ones(T)), K=K,
                image_paths=image_paths)
