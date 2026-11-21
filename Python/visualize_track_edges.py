#!/usr/bin/env python3
"""Build the fixed EDG connectivity over ALL tracks and visualize it.

Edges are decided by *persistent* world-space proximity, not single-frame closeness:
  1. lift every reliable track observation to world 3D
     (X_world = da3_frame_scale[f] * R_raw[f] @ tracks_xyz[:, f] + t_raw[f]);
  2. sample <= max_sampled_frames frames, KD-tree per frame -> candidate neighbour pairs (union);
  3. for each candidate, over reliable co-visible sampled frames, require enough overlap and a robust
     (quantile) separation <= edge_distance_threshold;
  4. keep each node's max_neighbors closest, symmetrized by union.

Then it shows the graph: the tracks lifted at the
current frame (points) with the fixed edges drawn as lines that follow the points; the edge SET is
global (fixed in time), only the 3D endpoints move per frame. SPACE = next frame, B = previous,
S = save current frame (points+edges) as PLY/OBJ, Q/ESC = close.

Run (from Python/):
    XM_DATASET_NAME=<scene> conda run -n XM python visualize_track_edges.py
Flags: --start N, --no-scale, --color {image,gray}, --edge-thresh X (0=auto), plus the edge-builder params.
"""
import argparse
import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

import utils.definitions
from utils.edg import scatter_solved_poses, scatter_solved_frame_vector


# ----------------------------------------------------------------------------- data loading
def _load_all():
    root = Path(utils.definitions.rootpath)
    data = np.load(utils.definitions.tracks_dynamics_path, allow_pickle=True)
    tracks_xyz = data["tracks_xyz"]          # (1, Q, T, 3)
    tracks_xy = data["tracks_xy"]            # (1, Q, T, 2)  image px
    occluded = data["occluded"]              # (1, Q, T)
    dyn_mask = data["dynamic_obs_mask"]      # (Q, T)
    query_t = data["query_t"] if "query_t" in data.files else None  # (Q,) frame each track was queried
    # separate deletion masks (so the depth + large-2D-move filters can be undone without re-tracking)
    dg = data["depth_grad_deleted_obs_mask"] if "depth_grad_deleted_obs_mask" in data.files else None  # (Q,T)
    dm = data["depth_max_deleted_obs_mask"] if "depth_max_deleted_obs_mask" in data.files else None     # (Q,T)
    mv = data["move_delete_query_mask"] if "move_delete_query_mask" in data.files else None             # (Q,)
    T = tracks_xyz.shape[2]
    rt = np.load(str(root / "outputs" / "result_XM.npz"))
    R_raw, t_raw = scatter_solved_poses(rt, T)
    return dict(tracks_xyz=tracks_xyz, tracks_xy=tracks_xy, occluded=occluded,
                dyn_mask=dyn_mask, query_t=query_t, T=T, rt=rt, R_raw=R_raw, t_raw=t_raw,
                depth_grad_deleted=dg, depth_max_deleted=dm, move_delete=mv,
                image_paths=sorted(glob.glob(str(Path(utils.definitions.video_dir) /
                                                 utils.definitions.image_pattern))))


def _dilate_dyn_mask(D, dilate_px):
    """Grow the per-track dynamic reliability mask by dilating the 2D SAM masks `dilate_px` px
    (same load_dynamic_masks + dilate_dynamic_masks utilities as static_tracking_online.py), then
    re-testing each track's image position against the enlarged region. Returns a (Q,T) bool mask
    that is a superset of the original dynamic_obs_mask, so tracks within `dilate_px` of the moving
    object's boundary also count as reliable. Falls back to the original mask if masks are missing.
    (Temporal filtering such as move-delete lives in `occluded`, not here, so it is preserved.)"""
    if dilate_px <= 0:
        return D["dyn_mask"]
    mask_dir = getattr(utils.definitions, "mask_dir", None)
    if mask_dir is None or not Path(str(mask_dir)).exists():
        print(f"[dyn-mask] no mask dir ({mask_dir}); using original dynamic_obs_mask (no dilation).")
        return D["dyn_mask"]
    try:
        import cv2
        from utils.track_tapnet import load_dynamic_masks, dilate_dynamic_masks
    except Exception as e:
        print(f"[dyn-mask] cannot dilate ({e}); using original dynamic_obs_mask.")
        return D["dyn_mask"]
    H, W = cv2.imread(D["image_paths"][0]).shape[:2]
    masks = load_dynamic_masks(str(mask_dir), utils.definitions.mask_pattern,
                               t_expected=D["T"], mask_mode="auto", target_size=(W, H))  # (T,H,W) bool
    masks = dilate_dynamic_masks(masks, int(dilate_px))
    xy = D["tracks_xy"][0]                    # (Q, T, 2) native px
    dyn = np.asarray(D["dyn_mask"]).astype(bool).copy()  # start from original -> only grows
    for t in range(D["T"]):
        m = masks[t]
        finite = np.isfinite(xy[:, t, 0]) & np.isfinite(xy[:, t, 1])
        x = np.round(np.where(finite, xy[:, t, 0], -1.0)).astype(np.int64)
        y = np.round(np.where(finite, xy[:, t, 1], -1.0)).astype(np.int64)
        ok = finite & (x >= 0) & (x < W) & (y >= 0) & (y < H)
        idx = np.where(ok)[0]
        dyn[idx, t] |= m[y[idx], x[idx]].astype(bool)
    print(f"[dyn-mask] dilated dynamic masks by {dilate_px}px: dynamic obs "
          f"{int(np.asarray(D['dyn_mask']).sum())} -> {int(dyn.sum())}")
    return dyn


def _frame_scale(rt, T, no_scale):
    if ("da3_frame_scale" in rt.files) and not no_scale:
        s = scatter_solved_frame_vector(rt, "da3_frame_scale", T, default=1.0)
        return np.where(np.isfinite(s) & (s > 0), s, 1.0)
    return np.ones((T,), dtype=np.float64)


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


def _colors_from_image(D, frame, xy, mode):
    if mode != "image":
        return None
    try:
        import cv2
    except Exception:
        return None
    if frame >= len(D["image_paths"]):
        return None
    img = cv2.cvtColor(cv2.imread(D["image_paths"][frame]), cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    xi = np.clip(np.round(xy[:, 0]).astype(int), 0, W - 1)
    yi = np.clip(np.round(xy[:, 1]).astype(int), 0, H - 1)
    return np.clip(img[yi, xi].astype(np.float64) / 255.0, 0.0, 1.0)


# ------------------------------------------------------------------------- show-all-tracks support:
# tracks_xyz is NaN for every observation dynamic_tracking.py deemed unreliable (occluded, off-mask,
# or bad depth), so _valid_at only ever shows the reliable subset. To draw ALL existing tracks and
# colour them by *why* they were excluded, re-lift the excluded ones straight from the DA3 depth map
# (identical lift to dynamic_tracking.py), which gives them a 3D position even though tracks_xyz is NaN.
def _load_depth(D):
    """Load DA3 depth_maps + intrinsics + native image size into D so excluded tracks can be
    re-lifted for the show-all view. Returns True on success; False (and prints why) if unavailable."""
    try:
        import cv2
        dd = Path(str(utils.definitions.da3_depth_dir))
        dm, ik = dd / "depth_maps.npy", dd / "intrinsics.npy"
        if not dm.exists() or not ik.exists():
            print(f"[show-all] DA3 depth/intrinsics missing under {dd}; showing reliable tracks only.")
            return False
        D["depth_maps"] = np.load(str(dm), mmap_mode="r")     # (T, Hd, Wd)
        D["intrinsics"] = np.load(str(ik)).astype(np.float64)  # (T, 3, 3) at depth resolution
        H, W = cv2.imread(D["image_paths"][0]).shape[:2]
        D["W_native"], D["H_native"] = float(W), float(H)
        print(f"[show-all] loaded depth {D['depth_maps'].shape} + intrinsics; excluded tracks will be re-lifted.")
        return True
    except Exception as e:
        print(f"[show-all] cannot load depth ({e}); showing reliable tracks only.")
        return False


def _lift_cam_from_depth(D, frame):
    """Camera-frame 3D (Q,3) for ALL tracks at `frame`, sampled from the DA3 depth map exactly as
    dynamic_tracking.py lifts (x_depth = x*Wd/W; X = (x_depth-cx)/fx*z). NaN where depth is invalid."""
    depth = np.asarray(D["depth_maps"][frame], dtype=np.float64)   # (Hd, Wd)
    K = D["intrinsics"][min(frame, D["intrinsics"].shape[0] - 1)]
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    Hd, Wd = depth.shape
    xy = D["tracks_xy"][0, :, frame, :].astype(np.float64)
    xd = xy[:, 0] * (Wd / D["W_native"])
    yd = xy[:, 1] * (Hd / D["H_native"])
    finite = np.isfinite(xd) & np.isfinite(yd)
    xi = np.clip(np.rint(np.where(finite, xd, 0.0)).astype(np.int64), 0, Wd - 1)
    yi = np.clip(np.rint(np.where(finite, yd, 0.0)).astype(np.int64), 0, Hd - 1)
    z = depth[yi, xi]
    ok = finite & np.isfinite(z) & (z > 0)
    Xc = np.full((xy.shape[0], 3), np.nan, dtype=np.float64)
    Xc[ok, 0] = (xd[ok] - cx) / fx * z[ok]
    Xc[ok, 1] = (yd[ok] - cy) / fy * z[ok]
    Xc[ok, 2] = z[ok]
    return Xc


def _lift_all_at(D, s, frame):
    """World-lift (Q,3) + per-track status for every track that EXISTS at `frame` (query_t<=frame).
    status: 0=reliable, 1=occluded (in-mask but tracker-occluded / depth-deleted), 2=outside dynamic
    mask, -1=don't draw (not yet queried, or no 3D even after depth re-lift). Reliable points reuse the
    precomputed tracks_xyz; excluded ones are re-lifted from depth so they still get a position."""
    Q = D["tracks_xyz"].shape[1]
    Xc = D["tracks_xyz"][0, :, frame, :].astype(np.float64).copy()   # camera-frame; NaN for excluded
    need = ~np.isfinite(Xc).all(axis=1)
    if D.get("depth_maps") is not None and need.any():
        Xc[need] = _lift_cam_from_depth(D, frame)[need]
    Rf = float(s[frame]) * D["R_raw"][frame].astype(np.float64)
    P = (Rf @ Xc.T).T + D["t_raw"][frame][None, :]

    exists = (D["query_t"] <= frame) if D.get("query_t") is not None else np.ones(Q, bool)
    show = exists & np.isfinite(P).all(axis=1)
    occ = D["occluded"][0, :, frame].astype(bool)
    dyn = D["dyn_mask"][:, frame].astype(bool)
    status = np.full(Q, -1, np.int64)
    status[show & ~occ & dyn] = 0    # reliable (matches _valid_at)
    status[show & occ & dyn] = 1     # in-mask but occluded / depth-deleted
    status[show & ~dyn] = 2          # outside the dynamic mask
    return P, status


def _disable_depth_move_filter(D):
    """TEMPORARILY undo the depth-deletion (gradient/max) and large-2D-move filters: un-occlude the
    observations they removed and re-lift their 3D from the DA3 depth map, so those points count as
    reliable again for BOTH the viewer and the edge/EDG builder. Requires depth (call _load_depth
    first). Mutates D['occluded'] and D['tracks_xyz'] in place.
    Caveat: only the COMBINED `occluded` flag is stored, so a point that was ALSO tracker-occluded is
    un-occluded here too -- this errs toward showing more points, which is the intent of the toggle."""
    occ = np.asarray(D["occluded"]).copy()                      # (1,Q,T)
    undo = np.zeros(occ.shape[1:], bool)                        # (Q,T)
    dg, dm, mv = D.get("depth_grad_deleted"), D.get("depth_max_deleted"), D.get("move_delete")
    if dg is not None and np.shape(dg) == undo.shape: undo |= np.asarray(dg, bool)
    if dm is not None and np.shape(dm) == undo.shape: undo |= np.asarray(dm, bool)
    if mv is not None and np.asarray(mv).size == undo.shape[0]: undo |= np.asarray(mv, bool)[:, None]
    occ[0] &= ~undo
    D["occluded"] = occ

    xyz = np.asarray(D["tracks_xyz"]).copy()                    # (1,Q,T,3) writable
    T = D["T"]
    filled = 0
    if D.get("depth_maps") is not None:
        dyn = D["dyn_mask"].astype(bool)
        for t in range(T):
            need = (~np.isfinite(xyz[0, :, t, :]).all(1)) & (~occ[0, :, t]) & dyn[:, t]
            if not need.any():
                continue
            Xc = _lift_cam_from_depth(D, t)                     # (Q,3) camera-frame from depth
            m = need & np.isfinite(Xc).all(1)
            xyz[0, m, t, :] = Xc[m]
            filled += int(m.sum())
        D["tracks_xyz"] = xyz
    else:
        print("[keep-depth-move] depth not loaded; occlusion undone but 3D not re-lifted (points stay NaN).")
    print(f"[keep-depth-move] un-occluded {int(undo.sum())} obs (depth+large-move), re-lifted {filled} 3D positions")


# ----------------------------------------------------------------------------- edge builder
@dataclass
class EdgeConfig:
    max_sampled_frames: int = 300
    candidate_k: int = 32
    candidate_radius: float = 0.0        # 0 -> 1.5 * edge_distance_threshold
    min_overlap_frames: int = 3
    min_overlap_ratio: float = 0.20
    distance_quantile: float = 0.90
    edge_distance_threshold: float = 0.0  # 0 -> auto from nearest-neighbour scale
    # Redundant-track PRE-FILTER (runs before candidates/edges/EDG): drop a visible track if it lies
    # within prune_radius_factor * edge_distance (canonical-3D, the edge-distance scale) of an already-
    # kept track that is at least as long (>= reliable-frame count). Length-priority NMS -> keep the
    # best-observed track per neighbourhood, thin out redundant near-duplicates. 0 factor / False = off.
    prune_redundant: bool = True
    prune_radius_factor: float = 0.25    # ~1.0 = full edge distance (~98% removed); 0.25 = moderate (~2/3)
    # 2D rescue: also accept a pair if it stays close in the IMAGE (tracks_xy px), even when the 3D
    # distance is large. Handles bad depth (wrong 3D lift) where the 2D track is still correct.
    use_2d: bool = True
    edge_thresh_2d: float = 0.0          # image-px persistent-distance threshold (0 -> auto)
    candidate_radius_2d: float = 0.0     # 2D candidate search radius px (0 -> 1.5 * edge_thresh_2d)
    max_neighbors: int = 12
    symmetrize: str = "union"            # "union" or "mutual"
    min_degree: int = 3                  # guarantee: every visible track connects to >= this many
                                         # nearest canonical neighbours (kills isolated points). 0 = off.
    # Rigid-linking of disconnected PARTS (connected components): join two parts iff a single per-frame
    # rigid transform explains the combined set A∪B (they move as one body), so cross-part config is
    # preserved. Parts with different motion stay separate. 0/False = off.
    rigid_link: bool = True
    rigid_min_part: int = 4              # only test components with >= this many visible tracks
                                         # (>=4 gives a stable rigid fit; small object fragments qualify)
    rigid_thresh: float = 0.03           # max median Procrustes residual as a fraction of scene diag
    rigid_min_frames: int = 5            # need at least this many frames where BOTH parts are co-visible
    rigid_link_edges: int = 3            # edges added between two linked parts (nearest cross pairs)


def _stack_sampled(D, s, sampled):
    """Return Ps (S,Q,3) lifted positions and Vs (S,Q) reliability over the sampled frames."""
    Ps = np.empty((sampled.size, D["tracks_xyz"].shape[1], 3), dtype=np.float64)
    Vs = np.empty((sampled.size, D["tracks_xyz"].shape[1]), dtype=bool)
    for k, f in enumerate(sampled):
        P, v = _valid_at(D, s, int(f))
        Ps[k] = P
        Vs[k] = v
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


def _auto_edge_threshold_2d(Ps2d, Vs):
    """2D (image-px) analogue of _auto_edge_threshold: ~5x the median nearest-neighbour pixel gap."""
    nn = []
    for k in range(min(5, Ps2d.shape[0])):
        idx = np.where(Vs[k])[0]
        if idx.size < 5:
            continue
        dd, _ = cKDTree(Ps2d[k, idx]).query(Ps2d[k, idx], k=2)
        nn.append(dd[:, 1])
    if not nn:
        return 12.0
    return float(5.0 * np.median(np.concatenate(nn)))


def _prune_redundant_tracks(Ps, Vs, Ocount, thr, radius_factor, verbose=True):
    """Length-priority non-max suppression to drop REDUNDANT tracks before edges/EDG. A visible track
    is removed if it lies within `radius_factor * thr` (canonical mean-position 3D distance -- the same
    scale as the edge distance) of an already-kept track that is at least as long (>= reliable-frame
    count Ocount). Processing longest-first, each surviving track is the best-observed in its
    neighbourhood; nearer, shorter tracks it 'covers' are pruned. Returns keep mask (Q,) (never-visible
    tracks -- Ocount==0 -- are kept untouched; they carry no edges anyway)."""
    Q = Ps.shape[1]
    keep = np.ones(Q, bool)
    has = np.where(Ocount > 0)[0]
    if has.size == 0 or radius_factor <= 0:
        return keep
    canon = np.where(Vs[..., None], Ps, 0.0).sum(0)[has] / Ocount[has, None]   # (H,3) mean reliable pos
    r = float(radius_factor) * float(thr)
    tree = cKDTree(canon)
    order = np.argsort(-Ocount[has])            # longest (most-observed) first
    alive = np.ones(has.size, bool)             # not-yet-suppressed
    survive = np.zeros(has.size, bool)
    for l in tqdm(order, desc="[prune] length-NMS", disable=not verbose):
        if not alive[l]:
            continue
        survive[l] = True                       # longest un-suppressed in this ball -> keep
        alive[tree.query_ball_point(canon[l], r)] = False   # suppress it + all (shorter) neighbours
    keep[has] = survive
    if verbose:
        print(f"[prune] radius={r:.4f} ({radius_factor}x edge-dist): kept {int(survive.sum())}/{has.size} "
              f"visible tracks, removed {int((~survive).sum())} redundant ({100*(~survive).mean():.0f}%)")
    return keep


def build_edg_edges(D, s, cfg: EdgeConfig, verbose=True):
    """Edge rule of the module docstring. Returns edges (E,2) int track-index pairs and scores (E,) distances."""
    T = D["T"]
    sampled = np.unique(np.linspace(0, T - 1, min(cfg.max_sampled_frames, T)).astype(int))
    Ps, Vs = _stack_sampled(D, s, sampled)
    Ocount = Vs.sum(axis=0)  # per-track reliable sampled-frame count

    thr = cfg.edge_distance_threshold or _auto_edge_threshold(Ps, Vs)

    # --- redundant-track pre-filter (length-priority NMS at the edge-distance scale) ---
    # pruned tracks are made never-reliable (Vs zeroed) so they are excluded from candidates, edges,
    # min-degree, rigid-linking AND the EDG; D["pruned"] flags them so the viewer can colour them.
    D["pruned"] = np.zeros(D["tracks_xyz"].shape[1], bool)
    if getattr(cfg, "prune_redundant", False) and getattr(cfg, "prune_radius_factor", 0.0) > 0:
        keep = _prune_redundant_tracks(Ps, Vs, Ocount, thr, cfg.prune_radius_factor, verbose)
        D["pruned"] = (Ocount > 0) & (~keep)
        Vs = Vs & keep[None, :]
        Ocount = Vs.sum(axis=0)

    # expose the sampled positions/reliability so the EDG node selection can rank by length + overlap
    D["_edg_aux"] = dict(Ps=Ps, Vs=Vs, Ocount=Ocount, thr=thr)

    radius = cfg.candidate_radius or (1.5 * thr)
    Ps2d = thr2d = radius2d = None
    if cfg.use_2d:
        xy = D["tracks_xy"][0]                                          # (Q, T, 2) image px
        Ps2d = np.where(Vs[..., None], np.transpose(xy[:, sampled, :], (1, 0, 2)).astype(np.float64), np.nan)
        thr2d = cfg.edge_thresh_2d or _auto_edge_threshold_2d(Ps2d, Vs)
        radius2d = cfg.candidate_radius_2d or (1.5 * thr2d)
    if verbose:
        print(f"[edges] {sampled.size} sampled frames, edge_thresh={thr:.4f}, cand_radius={radius:.4f}"
              + (f" | 2D rescue: edge_thresh={thr2d:.1f}px" if cfg.use_2d else ""))

    # --- step 2: candidate pairs (union of per-frame KD-tree neighbours) ------------------------
    cand = set()
    for k in tqdm(range(sampled.size), desc="[edges] 3D candidates", disable=not verbose):
        idx = np.where(Vs[k])[0]
        if idx.size < 2:
            continue
        tree = cKDTree(Ps[k, idx])
        kq = min(cfg.candidate_k + 1, idx.size)
        dists, nbrs = tree.query(Ps[k, idx], k=kq, distance_upper_bound=radius)
        dists = np.atleast_2d(dists)
        nbrs = np.atleast_2d(nbrs)
        for a in range(idx.size):
            ia = int(idx[a])
            for b, d in zip(nbrs[a], dists[a]):
                if b >= idx.size or not np.isfinite(d):  # cKDTree pads misses with n / inf
                    continue
                ib = int(idx[b])
                if ia != ib:
                    cand.add((ia, ib) if ia < ib else (ib, ia))
    # 2D candidate pass: per-frame KD-tree in IMAGE space, so image-adjacent pairs become candidates
    # even if they are far apart in 3D (bad depth). Acceptance below verifies persistent 2D closeness.
    if cfg.use_2d:
        for k in tqdm(range(sampled.size), desc="[edges] 2D candidates", disable=not verbose):
            idx = np.where(Vs[k])[0]
            if idx.size < 2:
                continue
            tree = cKDTree(Ps2d[k, idx])
            kq = min(cfg.candidate_k + 1, idx.size)
            dists, nbrs = tree.query(Ps2d[k, idx], k=kq, distance_upper_bound=radius2d)
            dists = np.atleast_2d(dists); nbrs = np.atleast_2d(nbrs)
            for a in range(idx.size):
                ia = int(idx[a])
                for b, d in zip(nbrs[a], dists[a]):
                    if b >= idx.size or not np.isfinite(d):
                        continue
                    ib = int(idx[b])
                    if ia != ib:
                        cand.add((ia, ib) if ia < ib else (ib, ia))
    if verbose:
        print(f"[edges] {len(cand)} candidate pairs")
    if not cand:
        return np.zeros((0, 2), np.int64), np.zeros((0,), np.float64)

    cand_arr = np.fromiter((x for pair in cand for x in pair), dtype=np.int64).reshape(-1, 2)
    ci, cj = cand_arr[:, 0], cand_arr[:, 1]

    # --- step 3: persistent proximity (vectorised + chunked over candidates) -------------------
    #     accept a pair if it stays close in 3D (s3 <= thr) OR in 2D image space (s2 <= thr2d).
    acc_i, acc_j, acc_s = [], [], []
    n_2d_rescue = 0
    CH = 20000
    for st in tqdm(range(0, ci.size, CH), desc="[edges] persistent-proximity", disable=not verbose):
        a = ci[st:st + CH]
        b = cj[st:st + CH]
        co = Vs[:, a] & Vs[:, b]                                   # (S, C)
        n_ov = co.sum(axis=0)                                      # (C,)
        min_ov = np.minimum(Ocount[a], Ocount[b]).clip(min=1)
        keep = (n_ov >= cfg.min_overlap_frames) & (n_ov / min_ov >= cfg.min_overlap_ratio)
        if not np.any(keep):
            continue
        a, b, co = a[keep], b[keep], co[:, keep]
        d3 = np.where(co, np.linalg.norm(Ps[:, a, :] - Ps[:, b, :], axis=2), np.nan)
        s3 = np.nanquantile(d3, cfg.distance_quantile, axis=0)      # (C',) 3D persistent distance
        ok = s3 <= thr
        if cfg.use_2d:
            d2 = np.where(co, np.linalg.norm(Ps2d[:, a, :] - Ps2d[:, b, :], axis=2), np.nan)
            s2 = np.nanquantile(d2, cfg.distance_quantile, axis=0)  # (C',) 2D persistent distance (px)
            ok2 = s2 <= thr2d
            n_2d_rescue += int((ok2 & ~ok).sum())                  # accepted by 2D but NOT by 3D
            ok = ok | ok2
        # score = 3D persistent distance (used by the stretch filter; large for bad-depth 2D-rescues)
        acc_i.append(a[ok]); acc_j.append(b[ok]); acc_s.append(s3[ok])
    if not acc_i or sum(x.size for x in acc_i) == 0:
        return np.zeros((0, 2), np.int64), np.zeros((0,), np.float64)
    ai = np.concatenate(acc_i); aj = np.concatenate(acc_j); asc = np.concatenate(acc_s)
    if verbose:
        print(f"[edges] {ai.size} accepted candidate pairs (3D s<= {thr:.4f}"
              + (f" or 2D s<= {thr2d:.1f}px; {n_2d_rescue} rescued by 2D only)" if cfg.use_2d else ")"))

    # --- step 4: per-node max_neighbors, symmetrise -------------------------------------------
    from collections import defaultdict
    nbr = defaultdict(list)
    for i, j, sij in tqdm(zip(ai, aj, asc), total=ai.size, desc="[edges] neighbour lists", disable=not verbose):
        nbr[int(i)].append((float(sij), int(j)))
        nbr[int(j)].append((float(sij), int(i)))
    selected = {}  # (i,j)->score, directed selections recorded per endpoint
    picked = defaultdict(set)
    for node, lst in tqdm(nbr.items(), desc="[edges] max-neighbours", disable=not verbose):
        lst.sort()
        for sij, other in lst[:cfg.max_neighbors]:
            picked[node].add(other)
            key = (node, other) if node < other else (other, node)
            selected[key] = min(sij, selected.get(key, sij))
    if cfg.symmetrize == "mutual":
        keys = [k for k in selected if (k[1] in picked[k[0]]) and (k[0] in picked[k[1]])]
    else:  # union: either endpoint selected the other (already true for every key in `selected`)
        keys = list(selected.keys())
    edges = np.array(sorted(keys), dtype=np.int64).reshape(-1, 2)
    scores = np.array([selected[tuple(e)] for e in edges], dtype=np.float64) if edges.size else np.zeros((0,), np.float64)

    # --- step 5: MIN-DEGREE guarantee -- every visible track connects to >= cfg.min_degree nearest
    #     canonical neighbours, so NO point is left isolated (persistent-proximity may leave short /
    #     rarely-co-visible tracks unconnected; here we add spatial-KNN fallback edges for them). ---
    if cfg.min_degree > 0:
        Q = D["tracks_xyz"].shape[1]
        canon = np.full((Q, 3), np.nan)                      # per-track canonical pos = mean reliable
        cnt = Vs.sum(0)
        has_idx = np.where(cnt > 0)[0]
        canon[has_idx] = np.where(Vs[..., None], Ps, 0.0).sum(0)[has_idx] / cnt[has_idx, None]
        from collections import defaultdict as _dd
        adj = _dd(set)
        for a, b in edges:
            adj[int(a)].add(int(b)); adj[int(b)].add(int(a))
        under = [int(u) for u in has_idx if len(adj[u]) < cfg.min_degree]
        add = {}
        if under and has_idx.size > cfg.min_degree:
            tree = cKDTree(canon[has_idx])
            kq = int(min(cfg.min_degree + 6, has_idx.size))
            dd, nn = tree.query(canon[under], k=kq)
            dd = np.atleast_2d(dd); nn = np.atleast_2d(nn)
            for r, u in enumerate(under):
                need = cfg.min_degree - len(adj[u])
                for lj, d in zip(nn[r], dd[r]):
                    if need <= 0:
                        break
                    gj = int(has_idx[lj])
                    if gj == u or gj in adj[u] or not np.isfinite(d):
                        continue
                    key = (u, gj) if u < gj else (gj, u)
                    if key in add:
                        continue
                    add[key] = float(d)
                    adj[u].add(gj); adj[gj].add(u); need -= 1
        if add:
            extra = np.array(sorted(add.keys()), dtype=np.int64).reshape(-1, 2)
            extra_s = np.array([add[tuple(e)] for e in extra], dtype=np.float64)
            edges = np.concatenate([edges, extra], 0) if edges.size else extra
            scores = np.concatenate([scores, extra_s], 0) if scores.size else extra_s
            if verbose:
                print(f"[edges] min-degree={cfg.min_degree}: +{extra.shape[0]} fallback edges "
                      f"for {len(under)} under-connected tracks")

    # --- step 6: RIGID-LINK disconnected parts that move as one body ------------------------------
    if cfg.rigid_link:
        edges, scores = _link_rigid_components(D, cfg, edges, scores, Ps, Vs, verbose=verbose)

    if verbose:
        Q = D["tracks_xyz"].shape[1]
        deg = np.bincount(edges.reshape(-1), minlength=Q) if edges.size else np.zeros(Q, int)
        n_visible = int((Vs.sum(0) > 0).sum())          # tracks reliable in >=1 sampled frame
        n_conn = int((deg > 0).sum())
        print(f"[edges] final: {edges.shape[0]} edges, mean degree {2*edges.shape[0]/max(1,n_conn):.1f}, "
              f"connected {n_conn}/{n_visible} visible tracks, isolated(visible) {n_visible - n_conn}")
    return edges, scores


def _kabsch(X0, X1):
    """Best rigid transform R,t with X1 ~= X0 @ R.T + t (both (n,3)); return R,t and RMS residual."""
    c0, c1 = X0.mean(0), X1.mean(0)
    H = (X0 - c0).T @ (X1 - c1)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = c1 - R @ c0
    res = float(np.sqrt((((X0 @ R.T + t) - X1) ** 2).sum(1).mean()))
    return R, t, res


def _rigid_link_test(A, B, Ps, Vs, canon, scale, cfg, rng):
    """Are parts A,B one rigid body? Fit a SINGLE per-frame rigid transform to the combined A∪B (from
    the best-covered reference frame) and require a small median residual, using frames where BOTH
    parts are co-visible with >=2 points each (so the fit actually couples them). Returns
    (linked, [(a,b,dist), ...]) with the nearest cross pairs to add as edges."""
    def sub(idx, n):
        return idx if idx.size <= n else idx[rng.choice(idx.size, n, replace=False)]
    A2, B2 = sub(A, 60), sub(B, 60)
    idx = np.concatenate([A2, B2]); nA = A2.size
    P, V = Ps[:, idx, :], Vs[:, idx]                     # (S,n,3), (S,n)
    vA, vB = V[:, :nA].sum(1), V[:, nA:].sum(1)
    both = np.minimum(vA, vB)
    if both.max() < 2:
        return False, None
    r = int(both.argmax())
    resids = []
    for t in range(P.shape[0]):
        if t == r:
            continue
        m = V[r] & V[t]
        if m[:nA].sum() < 2 or m[nA:].sum() < 2:         # need both parts represented
            continue
        _, _, res = _kabsch(P[r, m], P[t, m])
        resids.append(res)
    if len(resids) < cfg.rigid_min_frames or np.median(resids) > cfg.rigid_thresh * scale:
        return False, None
    # link: nearest cross pairs by canonical distance
    treeB = cKDTree(canon[B])
    dd, nn = treeB.query(canon[A], k=1)
    order = np.argsort(dd)[:cfg.rigid_link_edges]
    links = [(int(A[o]), int(B[int(nn[o])]), float(dd[o])) for o in order if np.isfinite(dd[o])]
    return (len(links) > 0), links


def _link_rigid_components(D, cfg, edges, scores, Ps, Vs, verbose=True):
    """Join separate connected components that move as ONE rigid body (see _rigid_link_test). Parts
    with different motion are left separate. Adds a few edges between each linked pair; union-find
    avoids redundant links so a chain of rigid parts collapses into one component."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    Q = D["tracks_xyz"].shape[1]
    if edges.shape[0] == 0:
        return edges, scores
    n_comp, labels = connected_components(
        coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(Q, Q)), directed=False)
    cnt = Vs.sum(0); has = cnt > 0
    canon = np.full((Q, 3), np.nan)
    canon[has] = np.where(Vs[..., None], Ps, 0.0).sum(0)[has] / cnt[has, None]
    hp = canon[has]
    scale = float(np.linalg.norm(hp.max(0) - hp.min(0))) + 1e-9
    members = {}
    for i in np.where(has)[0]:
        members.setdefault(int(labels[i]), []).append(int(i))
    parts = [np.array(v, dtype=np.int64) for v in members.values() if len(v) >= cfg.rigid_min_part]
    if len(parts) < 2:
        if verbose:
            print(f"[rigid] {len(parts)} part(s) >= {cfg.rigid_min_part} tracks; nothing to link")
        return edges, scores
    if verbose:
        print(f"[rigid] testing {len(parts)} parts (of {n_comp} components) for rigid linkage")
    parent = list(range(len(parts)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    rng = np.random.default_rng(0)
    add, n_links = {}, 0
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            if find(i) == find(j):
                continue
            linked, links = _rigid_link_test(parts[i], parts[j], Ps, Vs, canon, scale, cfg, rng)
            if linked:
                parent[find(i)] = find(j); n_links += 1
                for a, b, dst in links:
                    add[(a, b) if a < b else (b, a)] = dst
                if verbose:
                    print(f"[rigid]  link parts |{parts[i].size}| <-> |{parts[j].size}|  (+{len(links)} edges)")
    if add:
        extra = np.array(sorted(add.keys()), dtype=np.int64).reshape(-1, 2)
        extra_s = np.array([add[tuple(e)] for e in extra], dtype=np.float64)
        edges = np.concatenate([edges, extra], 0)
        scores = np.concatenate([scores, extra_s], 0)
        if verbose:
            print(f"[rigid] linked {n_links} part-pairs, +{extra.shape[0]} inter-part edges")
    return edges, scores


# --------------------------------------------------------------------- EDG (deformation graph) build
# EDG NODES are chosen by a length-priority non-max suppression on the tracks themselves: rank every
# track by its reliable LENGTH (longest first), select the longest survivor as a node, then delete every
# track that is (a) NEAR it (canonical mean-position within near_factor*edge_distance) AND (b) temporally
# OVERLAPS it by more than overlap_frac of the shorter track's length (co-visible reliable frames).
# Repeat on the survivors. A near track that is temporally DISJOINT is kept -- it covers a different time
# span of the same region, so it is a useful separate node. EDG EDGES then connect node pairs that are
# within edge_hops on the persistent-proximity track graph (independent objects stay separate).
def _select_edg_nodes_by_length(Ps, Vs, Ocount, thr, near_factor=1.0, overlap_frac=0.5,
                                max_nodes=1500, min_node_frames=20, verbose=True):
    """Length-priority NMS EDG node selection (see block comment). Returns selected node ids (global
    track indices). Only tracks observed in MORE than `min_node_frames` reliable frames are eligible
    (shorter tracks -- and never-visible ones -- are ignored)."""
    Q = Ps.shape[1]
    has = np.where(Ocount > int(min_node_frames))[0]
    if has.size == 0:
        return np.zeros(0, np.int64)
    canon = np.where(Vs[..., None], Ps, 0.0).sum(0)[has] / Ocount[has, None]     # (H,3) mean reliable pos
    r = float(near_factor) * float(thr)
    tree = cKDTree(canon)
    order = np.argsort(-Ocount[has])            # local indices, longest (most reliable frames) first
    alive = np.ones(has.size, bool)
    selected = []
    for l in tqdm(order, desc="[EDG] length-NMS nodes", disable=not verbose):
        if not alive[l]:
            continue
        selected.append(int(has[l]))            # longest survivor -> EDG node
        near = np.asarray(tree.query_ball_point(canon[l], r), dtype=np.int64)
        near = near[(near != l) & alive[near]]
        if near.size:
            g = int(has[l]); gj = has[near]
            overlap = (Vs[:, g][:, None] & Vs[:, gj]).sum(axis=0)                # co-visible reliable frames
            thresh = overlap_frac * np.minimum(Ocount[g], Ocount[gj])            # >50% of the SHORTER length
            alive[near[overlap > thresh]] = False   # delete near AND heavily-overlapping tracks
        alive[l] = False
        if len(selected) >= max_nodes:
            break
    nodes = np.array(selected, dtype=np.int64)
    if verbose:
        print(f"[EDG] length-NMS: {nodes.size} nodes from {has.size} tracks with >{int(min_node_frames)} "
              f"reliable frames (near<= {r:.4f}={near_factor}x edge-dist, overlap> {overlap_frac:.0%} of "
              f"shorter length)" + ("  [hit max_nodes]" if nodes.size >= max_nodes else ""))
    return nodes


def build_edg_graph(edges, Q, Ps, Vs, Ocount, thr, near_factor=1.0, overlap_frac=0.5,
                    edge_hops=6, max_nodes=1500, min_node_frames=20, verbose=True):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    # --- NODE SELECTION: length-priority NMS (near + >overlap_frac temporal overlap) ---
    nodes = _select_edg_nodes_by_length(Ps, Vs, Ocount, thr, near_factor, overlap_frac, max_nodes,
                                        min_node_frames, verbose)
    if nodes.size == 0 or edges.shape[0] == 0:
        return nodes, np.zeros((0, 2), np.int64)

    # --- EDG EDGES: node pairs within edge_hops on the track graph (chunked capped Dijkstra) ---
    edge_hops = int(edge_hops) if edge_hops > 0 else 6
    ii = np.concatenate([edges[:, 0], edges[:, 1]])
    jj = np.concatenate([edges[:, 1], edges[:, 0]])
    A = csr_matrix((np.ones(ii.size, np.float32), (ii, jj)), shape=(Q, Q))
    K = nodes.size
    pair_lo, pair_hi = [], []
    CH = 128
    for st in tqdm(range(0, K, CH), desc="EDG edges", disable=not verbose):
        src = nodes[st:st + CH]
        Dn = dijkstra(A, directed=False, indices=src, unweighted=True, limit=float(edge_hops))  # (c, Q)
        Hc = Dn[:, nodes]                                # (c, K) hop distances to every EDG node
        ri, bi = np.where((Hc <= edge_hops) & (Hc > 0) & np.isfinite(Hc))
        a = src[ri]; b = nodes[bi]
        pair_lo.append(np.minimum(a, b)); pair_hi.append(np.maximum(a, b))
    if pair_lo:
        und = np.unique(np.stack([np.concatenate(pair_lo), np.concatenate(pair_hi)], 1), axis=0)
        edg_edges = und.astype(np.int64)
    else:
        edg_edges = np.zeros((0, 2), np.int64)
    if verbose:
        print(f"[EDG] {nodes.size} nodes, {edg_edges.shape[0]} edges (hop distance <= {edge_hops})")
    return nodes, edg_edges


def _edg_observation_matrix(D, s, edg_nodes):
    """(M, T) bool: is each EDG node reliably OBSERVED at each frame? Uses the SAME validity as
    _valid_at (finite world-lift & ~occluded & dynamic mask) -- i.e. exactly what the EDG solver
    counts as an observation -- so this is the true per-node active horizon."""
    T = D["T"]
    xyz = D["tracks_xyz"][0, edg_nodes].astype(np.float64)                 # (M,T,3)
    Rf = s[:, None, None] * D["R_raw"].astype(np.float64)                  # (T,3,3)
    P = np.einsum("tij,mtj->mti", Rf, xyz) + D["t_raw"][None, :, :]        # (M,T,3)
    obs = (np.isfinite(P).all(axis=-1)
           & (D["occluded"][0, edg_nodes] == 0)
           & D["dyn_mask"][edg_nodes].astype(bool))                        # (M,T)
    return obs


def visualize_edg_active_horizon(D, s, edg_nodes, out_path, sort=True):
    """Save ONE big (num_EDG x T) image: a row per EDG node, a column per frame, green where the node
    is observed (its active horizon) and gray where it is not. Rows sorted by first-observed frame so
    the horizons read as a Gantt chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    edg_nodes = np.asarray(edg_nodes, dtype=np.int64)
    obs = _edg_observation_matrix(D, s, edg_nodes)                         # (M,T) bool
    M, T = obs.shape
    order = np.arange(M)
    if sort and M:
        any_obs = obs.any(axis=1)
        first = np.where(any_obs, obs.argmax(axis=1), T)
        last = np.where(any_obs, T - 1 - obs[:, ::-1].argmax(axis=1), -1)
        order = np.lexsort((last, first))
    obs_s = obs[order]
    node_ids = edg_nodes[order]

    fig_w = max(8.0, T * 0.03 + 2.0)
    fig_h = max(6.0, M * 0.06 + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    ax.imshow(obs_s, aspect="auto", interpolation="nearest", vmin=0, vmax=1,
              cmap=ListedColormap(["#eeeeee", "#1a9850"]))                 # gray=not observed, green=observed
    ax.set_xlabel("frame")
    ax.set_ylabel("EDG node (sorted by first-observed frame)")
    ax.set_title(f"EDG node active horizon  |  {M} nodes x {T} frames  |  green = observed")
    xstep = max(1, T // 30)
    ax.set_xticks(np.arange(0, T, xstep))
    ystep = max(1, M // 50)
    ax.set_yticks(np.arange(0, M, ystep))
    ax.set_yticklabels(node_ids[::ystep], fontsize=5)
    ax.tick_params(axis="x", labelsize=6, rotation=90)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    active = obs.sum(axis=1)
    print(f"[horizon] saved {M}x{T} EDG active-horizon image -> {out_path}  "
          f"(active frames/node: min={int(active.min()) if M else 0} "
          f"median={int(np.median(active)) if M else 0} max={int(active.max()) if M else 0})")


def _edg_sphere_mesh(positions, radius, color):
    """One merged TriangleMesh with a small sphere at each EDG-node position (so nodes read as 3D
    markers regardless of point size). Rebuilt each frame from the current node positions."""
    import open3d as o3d
    m = o3d.geometry.TriangleMesh()
    if len(positions) == 0:
        return m
    tmpl = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius), resolution=6)
    tv = np.asarray(tmpl.vertices); tt = np.asarray(tmpl.triangles); nv = tv.shape[0]
    V = (positions[:, None, :] + tv[None, :, :]).reshape(-1, 3)
    Tr = (tt[None, :, :] + (np.arange(len(positions)) * nv)[:, None, None]).reshape(-1, 3)
    m.vertices = o3d.utility.Vector3dVector(V)
    m.triangles = o3d.utility.Vector3iVector(Tr)
    m.paint_uniform_color(color)
    m.compute_vertex_normals()
    return m


# ----------------------------------------------------------------------------- visualization
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--no-scale", action="store_true", help="do not apply da3 per-frame scale")
    ap.add_argument("--color", choices=["image", "gray"], default="image")
    ap.add_argument("--edge-thresh", type=float, default=0.0, help="edge_distance_threshold (0=auto)")
    ap.add_argument("--no-2d", action="store_true",
                    help="disable the 2D rescue (only 3D distance decides edges)")
    ap.add_argument("--edge-thresh-2d", type=float, default=0.0,
                    help="2D image-px persistent-distance threshold for the 2D rescue (0=auto)")
    ap.add_argument("--max-draw-stretch", type=float, default=3.0,
                    help="Per-frame filter: hide an edge on any frame where its current endpoint "
                         "distance exceeds this multiple of its build score (its persistent length). "
                         "Only that frame is affected; the edge reappears when the points close up. "
                         "0 disables the filter.")
    ap.add_argument("--max-sampled-frames", type=int, default=EdgeConfig.max_sampled_frames,
                    help="How many frames to sample for topology (higher = more co-visibility "
                         "evidence per track -> far fewer isolated tracks; slower). Capped at T.")
    ap.add_argument("--candidate-k", type=int, default=32)
    ap.add_argument("--min-overlap-frames", type=int, default=3)
    ap.add_argument("--min-overlap-ratio", type=float, default=0.20)
    ap.add_argument("--distance-quantile", type=float, default=0.90)
    ap.add_argument("--max-neighbors", type=int, default=12)
    ap.add_argument("--symmetrize", choices=["union", "mutual"], default="union")
    ap.add_argument("--min-degree", type=int, default=3,
                    help="Guarantee every visible track connects to at least this many nearest "
                         "neighbours (adds spatial-KNN fallback edges -> no isolated points). 0=off.")
    ap.add_argument("--no-prune-tracks", action="store_true",
                    help="disable the redundant-track pre-filter (keep every track before edges/EDG)")
    ap.add_argument("--prune-radius-factor", type=float, default=0.0,
                    help="Redundant-track filter: drop a visible track if a >=-as-long track is within "
                         "this * edge-distance of it (length-priority NMS). 0=off; ~1.0=full edge distance "
                         "(aggressive); 0.3-0.5=moderate.")
    ap.add_argument("--no-rigid-link", action="store_true",
                    help="disable joining separate components that move as one rigid body")
    ap.add_argument("--rigid-thresh", type=float, default=0.03,
                    help="rigid-link tolerance: max median Procrustes residual as a fraction of scene diagonal")
    ap.add_argument("--rigid-min-part", type=int, default=4,
                    help="only rigid-link components with at least this many visible tracks")
    ap.add_argument("--mask-dilate-px", type=int, default=5,
                    help="Dilate the dynamic SAM mask by this many native pixels before deciding "
                         "which track observations count as reliable/dynamic (0 disables).")
    ap.add_argument("--disable-depth-move-filter", action="store_true",
                    help="TEMPORARILY undo the depth-deletion (gradient/max) and large-2D-move filters: "
                         "re-include those points (re-lifted from depth) as reliable in the viewer AND "
                         "the edge/EDG builder. No re-tracking needed (uses the stored deletion masks).")
    ap.add_argument("--valid-only", action="store_true",
                    help="Show only reliable tracks (old behavior). Default shows ALL existing tracks: "
                         "reliable in image colour, occluded/depth-deleted in gray, off-dynamic-mask in orange.")
    ap.add_argument("--no-edg", action="store_true",
                    help="do not build/draw the coarse EDG deformation graph (red nodes+edges)")
    ap.add_argument("--edg-near-factor", type=float, default=2.0,
                    help="EDG node selection: a track is 'near' a chosen node if within this * edge-distance "
                         "(canonical 3D). Larger = fewer, more-spread nodes.")
    ap.add_argument("--edg-overlap-frac", type=float, default=0.5,
                    help="EDG node selection: delete a near track only if it temporally overlaps the chosen "
                         "node by MORE than this fraction of the shorter track's length (0.5 = >50%%).")
    ap.add_argument("--edg-edge-hops", type=int, default=6,
                    help="connect two EDG nodes when their hop distance on the track graph is <= this")
    ap.add_argument("--edg-max-nodes", type=int, default=1500, help="safety cap on EDG node count")
    ap.add_argument("--edg-min-frames", type=int, default=20,
                    help="EDG node must be reliably observed in MORE than this many frames (length floor).")
    ap.add_argument("--horizon-out", type=str, default="",
                    help="save the EDG active-horizon image (num_EDG x frames, green=observed) to this "
                         "path (default: output_dir/edg_active_horizon.png)")
    ap.add_argument("--horizon-only", action="store_true",
                    help="only build the EDG + save the active-horizon image, then exit (no 3D window)")
    ap.add_argument("--save-npz", type=str, default="", help="also save edges to this .npz path")
    args = ap.parse_args()

    import open3d as o3d

    D = _load_all()
    D["dyn_mask"] = _dilate_dyn_mask(D, args.mask_dilate_px)  # grow dynamic reliability region by N px
    if args.disable_depth_move_filter:                       # undo depth + large-move filters (needs depth)
        _load_depth(D)
        _disable_depth_move_filter(D)
    s = _frame_scale(D["rt"], D["T"], args.no_scale)
    T = D["T"]
    cfg = EdgeConfig(
        max_sampled_frames=args.max_sampled_frames,
        candidate_k=args.candidate_k, min_overlap_frames=args.min_overlap_frames,
        min_overlap_ratio=args.min_overlap_ratio, distance_quantile=args.distance_quantile,
        edge_distance_threshold=args.edge_thresh, max_neighbors=args.max_neighbors,
        symmetrize=args.symmetrize, min_degree=args.min_degree,
        rigid_link=not args.no_rigid_link, rigid_thresh=args.rigid_thresh, rigid_min_part=args.rigid_min_part,
        use_2d=not args.no_2d, edge_thresh_2d=args.edge_thresh_2d,
        prune_redundant=not args.no_prune_tracks, prune_radius_factor=args.prune_radius_factor,
    )
    edges, scores = build_edg_edges(D, s, cfg)
    if args.save_npz and edges.size:
        np.savez(args.save_npz, edges=edges, scores=scores)
        print(f"[edges] saved {edges.shape[0]} edges to {args.save_npz}")

    # coarse EDG deformation graph: length-priority NMS nodes (near + temporal-overlap), drawn in red
    Q = D["tracks_xyz"].shape[1]
    if args.no_edg:
        edg_nodes, edg_edges = np.zeros(0, np.int64), np.zeros((0, 2), np.int64)
    else:
        aux = D["_edg_aux"]
        edg_nodes, edg_edges = build_edg_graph(
            edges, Q, aux["Ps"], aux["Vs"], aux["Ocount"], aux["thr"],
            near_factor=args.edg_near_factor, overlap_frac=args.edg_overlap_frac,
            edge_hops=args.edg_edge_hops, max_nodes=args.edg_max_nodes,
            min_node_frames=args.edg_min_frames)

    # EDG active-horizon image: one row per EDG node, one column per frame, green where observed
    if edg_nodes.size:
        horizon_out = args.horizon_out or str(Path(utils.definitions.output_dir) / "edg_active_horizon.png")
        visualize_edg_active_horizon(D, s, edg_nodes, horizon_out)
    elif args.horizon_only:
        print("[horizon] no EDG nodes to plot (did you pass --no-edg?).")
    if args.horizon_only:
        return

    show_all = not args.valid_only
    if show_all and not _load_depth(D):     # need depth to place excluded tracks; else fall back
        show_all = False
    C_OCCLUDED = np.array([0.4, 0.4, 0.4])   # gray   = tracker-occluded / depth-deleted (in mask)
    C_OFFMASK = np.array([0.95, 0.55, 0.1])  # orange = outside the dynamic mask (red reserved for EDG)
    C_PRUNED = np.array([0.55, 0.1, 0.85])   # purple = redundant track removed by the pre-filter
    C_EDG = [1.0, 0.05, 0.05]                # red    = EDG nodes + edges

    # EDG node marker radius from the scene scale at the start frame
    P0 = _valid_at(D, s, int(np.clip(args.start, 0, T - 1)))[0]
    fin0 = np.isfinite(P0).all(axis=1)
    diag = float(np.linalg.norm(P0[fin0].max(0) - P0[fin0].min(0))) if fin0.any() else 1.0
    node_r = max(1e-4, 0.006 * diag)

    state = {"idx": int(np.clip(args.start, 0, T - 1))}
    pcd = o3d.geometry.PointCloud()
    lines = o3d.geometry.LineSet()
    edg_line = o3d.geometry.LineSet()                    # red EDG edges
    edg_mesh = o3d.geometry.TriangleMesh()               # red EDG node spheres
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    def _fill(frame):
        pruned = D.get("pruned")
        if show_all:
            P, status = _lift_all_at(D, s, frame)
            if pruned is not None:
                status[(status == 0) & pruned] = 3   # redundant (pre-filtered) -> own colour, no edges
            v = status == 0                          # reliable+kept mask over ALL tracks (for edges)
            draw = status >= 0
            idx = np.where(draw)[0]
            cols = np.tile([0.3, 0.55, 1.0], (idx.size, 1)).astype(np.float64)
            img = _colors_from_image(D, frame, D["tracks_xy"][0, idx, frame, :], args.color)
            st = status[idx]
            if img is not None and len(img) == idx.size:
                cols[st == 0] = img[st == 0]         # reliable -> image colour
            cols[st == 1] = C_OCCLUDED               # occluded -> gray
            cols[st == 2] = C_OFFMASK                # off dynamic-mask -> orange
            cols[st == 3] = C_PRUNED                 # pruned redundant -> purple
            pcd.points = o3d.utility.Vector3dVector(P[idx])
            pcd.colors = o3d.utility.Vector3dVector(cols)
            n_valid = int((st == 0).sum()); n_occ = int((st == 1).sum())
            n_off = int((st == 2).sum()); n_pruned = int((st == 3).sum())
        else:
            P, v = _valid_at(D, s, frame)
            if pruned is not None:
                v = v & ~pruned                      # pruned tracks are hidden in --valid-only
            pts = P[v]
            pcd.points = o3d.utility.Vector3dVector(pts)
            cols = _colors_from_image(D, frame, D["tracks_xy"][0, v, frame, :], args.color)
            if cols is not None and len(cols) == len(pts):
                pcd.colors = o3d.utility.Vector3dVector(cols)
            elif len(pts) > 0:
                pcd.paint_uniform_color([0.3, 0.55, 1.0])
            n_valid = int(v.sum()); n_occ = n_off = n_pruned = 0

        P_safe = np.where(np.isfinite(P), P, 0.0)
        # edges = fixed graph, but only drawn where BOTH endpoints are reliable this frame ...
        n_e = n_hidden = 0
        if edges.size:
            em = v[edges[:, 0]] & v[edges[:, 1]]
            if args.max_draw_stretch > 0:
                cur_len = np.linalg.norm(P_safe[edges[:, 0]] - P_safe[edges[:, 1]], axis=1)
                not_stretched = cur_len <= args.max_draw_stretch * np.maximum(scores, 1e-9)
                n_hidden = int((em & ~not_stretched).sum())
                em = em & not_stretched
            lines.points = o3d.utility.Vector3dVector(P_safe)  # full track array; only valid ends referenced
            lines.lines = o3d.utility.Vector2iVector(edges[em].astype(np.int32) if em.any() else np.zeros((0, 2), np.int32))
            lines.paint_uniform_color([0.1, 0.9, 0.9])  # cyan edges
            n_e = int(em.sum())

        # --- coarse EDG deformation graph (RED): node spheres + edges, drawn where endpoints reliable ---
        n_en = n_ee = 0
        if edg_nodes.size:
            nm = v[edg_nodes]
            sm = _edg_sphere_mesh(P_safe[edg_nodes[nm]], node_r, C_EDG)
            edg_mesh.vertices = sm.vertices
            edg_mesh.triangles = sm.triangles
            edg_mesh.vertex_colors = sm.vertex_colors
            edg_mesh.vertex_normals = sm.vertex_normals
            n_en = int(nm.sum())
        if edg_edges.size:
            eem = v[edg_edges[:, 0]] & v[edg_edges[:, 1]]
            edg_line.points = o3d.utility.Vector3dVector(P_safe)
            edg_line.lines = o3d.utility.Vector2iVector(
                edg_edges[eem].astype(np.int32) if eem.any() else np.zeros((0, 2), np.int32))
            edg_line.paint_uniform_color(C_EDG)
            n_ee = int(eem.sum())
        return n_valid, n_occ, n_off, n_pruned, n_e, n_hidden, n_en, n_ee

    def refresh(vis, frame):
        n_valid, n_occ, n_off, n_pruned, n_e, n_hidden, n_en, n_ee = _fill(frame)
        for g in (pcd, lines, edg_line, edg_mesh):
            vis.update_geometry(g)
        hid = f"  hidden(over-stretched)={n_hidden}" if args.max_draw_stretch > 0 else ""
        extra = (f"  occluded(gray)={n_occ} off-mask(orange)={n_off} pruned(purple)={n_pruned}"
                 if show_all else "")
        edg = f"  EDG(red): {n_en} nodes {n_ee} edges" if edg_nodes.size else ""
        print(f"[Open3D] frame={frame}  reliable={n_valid}{extra}  edges={n_e}{hid}{edg}")
        return False

    def on_space(vis):
        state["idx"] = (state["idx"] + 1) % T
        return refresh(vis, state["idx"])

    def on_b(vis):
        state["idx"] = (state["idx"] - 1) % T
        return refresh(vis, state["idx"])

    def on_save(vis):
        import trimesh
        frame = state["idx"]
        P, v = _valid_at(D, s, frame)
        out_pts = Path(utils.definitions.output_dir) / f"track_edges_frame{frame:06d}.ply"
        trimesh.PointCloud(P[v].astype(np.float64)).export(str(out_pts))
        print(f"[Open3D] saved {out_pts}")
        return False

    def on_q(vis):
        vis.close()
        return False

    legend = ("Points: image-colour=reliable, GRAY=occluded/depth-deleted, ORANGE=outside dynamic mask, "
              "PURPLE=pruned redundant. " if show_all else "Points: reliable tracks only (--valid-only). ")
    edgleg = "" if args.no_edg else "RED spheres/lines = coarse EDG (hop-distance deformation graph). "
    print("Open3D controls: SPACE=next frame, B=prev, S=save PLY, Q/ESC=close. "
          f"Start frame={state['idx']} (T={T}). {legend}Cyan lines = track connectivity (Notes/EDG.md). {edgleg}")
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Track connectivity + EDG", width=1280, height=720)
    _fill(state["idx"])  # fill BEFORE add_geometry so the camera fits
    vis.add_geometry(pcd)
    vis.add_geometry(lines)
    vis.add_geometry(edg_line)
    vis.add_geometry(edg_mesh)
    vis.add_geometry(axes)
    vis.get_render_option().point_size = 4.0
    for key in (" ",):
        vis.register_key_callback(ord(key), on_space)
    vis.register_key_callback(ord("b"), on_b); vis.register_key_callback(ord("B"), on_b)
    vis.register_key_callback(ord("s"), on_save); vis.register_key_callback(ord("S"), on_save)
    vis.register_key_callback(ord("q"), on_q); vis.register_key_callback(ord("Q"), on_q)
    vis.register_key_callback(256, on_q)  # ESC
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
