#!/usr/bin/env python3
"""Build the fixed EDG connectivity over ALL tracks and visualize it.

Same builder as visualize_track_edges.py, with two differences: the topology is built from the FULL
video (every frame, not a subsample), and the viewer draws ALL dynamic points in a single PURPLE colour
(the red EDG nodes/edges and cyan track edges are unchanged).

Edges are decided by *persistent* world-space proximity, not single-frame closeness:
  1. lift every reliable track observation to world 3D:
     X_world = da3_frame_scale[f] * R_raw[f] @ tracks_xyz[:, f] + t_raw[f];
  2. over EVERY frame of the video, KD-tree per frame -> candidate neighbour pairs (union);
  3. for each candidate, over reliable co-visible frames, require enough overlap and a robust
     (quantile) separation <= edge_distance_threshold;
  4. keep each node's max_neighbors closest, symmetrized by union.

Then it shows the graph: the tracks lifted at the current frame (PURPLE points) with the fixed edges
drawn as lines that follow the points; the edge SET is global (fixed in time), only the 3D endpoints
move per frame. SPACE = next frame, B = previous, S = save current frame (points) as PLY, Q/ESC = close.

Run (from Python/):
    XM_DATASET_NAME=<scene> conda run -n XM python build_edg.py
Flags: --start N, --no-scale, --edge-thresh X (0=auto), --max-sampled-frames 0 (0=full video),
plus the EdgeConfig params.
"""
import argparse
import glob
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

import utils.definitions
from utils.edg import scatter_solved_poses, scatter_solved_frame_vector


# ----------------------------------------------------------------------------- data loading
def _load_all(tracks_path=None):
    root = Path(utils.definitions.rootpath)
    data = np.load(tracks_path or utils.definitions.tracks_dynamics_path, allow_pickle=True)
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
        from utils.masks import load_dynamic_masks, dilate_dynamic_masks
    except Exception as e:
        print(f"[dyn-mask] cannot dilate ({e}); using original dynamic_obs_mask.")
        return D["dyn_mask"]
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
    (constant relative node density). Falls back to `fallback` px when masks are missing."""
    mask_dir = getattr(utils.definitions, "mask_dir", None)
    if mask_dir is None or not Path(str(mask_dir)).exists():
        print(f"[edg-near] no mask dir ({mask_dir}); using fixed {fallback}px")
        return float(fallback)
    try:
        import cv2
        from utils.masks import load_dynamic_masks
    except Exception as e:
        print(f"[edg-near] cannot load masks ({e}); using fixed {fallback}px")
        return float(fallback)
    H, W = cv2.imread(D["image_paths"][0]).shape[:2]
    masks = D["dyn_masks"] if "dyn_masks" in D else load_dynamic_masks(
        str(mask_dir), utils.definitions.mask_pattern, t_expected=D["T"], mask_mode="dynamic_positive", target_size=(W, H))
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
    max_sampled_frames: int = 0          # 0 = full video (every frame); >0 subsamples that many
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
    prune_radius_factor: float = 0.0    # ~1.0 = full edge distance (~98% removed); 0.25 = moderate (~2/3)
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
    """Returns edges (E,2) int track-index pairs and scores (E,) distances."""
    T = D["T"]
    if cfg.max_sampled_frames and cfg.max_sampled_frames > 0:
        sampled = np.unique(np.linspace(0, T - 1, min(cfg.max_sampled_frames, T)).astype(int))
    else:
        sampled = np.arange(T, dtype=int)                  # full video: every frame
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
    D["_edg_aux"] = dict(Ps=Ps, Vs=Vs, Ocount=Ocount, thr=thr, sampled=sampled)

    radius = cfg.candidate_radius or (1.5 * thr)
    Ps2d = thr2d = radius2d = None
    if cfg.use_2d:
        xy = D["tracks_xy"][0]                                          # (Q, T, 2) image px
        Ps2d = np.where(Vs[..., None], np.transpose(xy[:, sampled, :], (1, 0, 2)).astype(np.float64), np.nan)
        thr2d = cfg.edge_thresh_2d or _auto_edge_threshold_2d(Ps2d, Vs)
        radius2d = cfg.candidate_radius_2d or (1.5 * thr2d)
    if verbose:
        print(f"[edges] {sampled.size} frames{' (full video)' if sampled.size == T else ' (sampled)'}, "
              f"edge_thresh={thr:.4f}, cand_radius={radius:.4f}"
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
# track that is (a) NEAR it in the IMAGE -- persistent 2D distance (the p<quantile> of per-frame pixel
# distance over co-visible frames, the SAME metric as the edge builder) within near_px -- AND (b)
# temporally OVERLAPS it by more than overlap_frac of the shorter track's length (co-visible reliable
# frames). Repeat on the survivors. A near track that is temporally DISJOINT is kept -- it covers a
# different time span of the same region, so it is a useful separate node. EDG EDGES then connect node
# pairs within edge_hops on the persistent-proximity track graph (independent objects stay separate).
def _frame_trees_2d(xy, Vs, T):
    """Per-frame KD-tree over the reliable tracks' 2D image positions (for candidate gathering)."""
    idx_per = [np.where(Vs[t])[0] for t in range(T)]
    with ThreadPoolExecutor(8) as ex:                         # the builds release the GIL (speed)
        tree_per = list(ex.map(lambda t: cKDTree(xy[idx_per[t], t]) if idx_per[t].size else None, range(T)))
    return idx_per, tree_per


def select_edg_nodes_2d(xy, Vs, Ocount, near_px, overlap_frac=0.5, max_nodes=100, min_node_frames=20,
                        quantile=0.90, nsamp=30, verbose=True, return_suppressed=False):
    """Length-priority NMS EDG-node selection using the 2D IMAGE persistent-distance metric (see block
    comment). `xy` is (Q, nS, 2) image px aligned to Vs' (nS, Q) time axis; ranks tracks by reliable
    length; selects the longest survivor; DELETES every still-alive track whose persistent 2D distance to
    the node -- the `quantile` (p90) of per-frame px distance over co-visible reliable frames -- is
    <= near_px AND that temporally overlaps the node by > overlap_frac of the shorter track's length.
    Only tracks reliable in > min_node_frames frames are eligible. Returns node global ids (and, if
    return_suppressed, the per-node list of deleted global ids). `near_px` is the range knob."""
    Q = xy.shape[0]; T = Vs.shape[0]
    has = np.where(Ocount > int(min_node_frames))[0]
    if has.size == 0:
        return (np.zeros(0, np.int64), []) if return_suppressed else np.zeros(0, np.int64)
    idx_per, tree_per = _frame_trees_2d(xy, Vs, T)
    order = has[np.argsort(-Ocount[has])]                    # global ids, longest first
    alive = np.zeros(Q, bool); alive[has] = True
    nodes, suppressed = [], []
    for g in tqdm(order, desc="[EDG] 2D length-NMS nodes", disable=not verbose):
        if not alive[g]:
            continue
        gframes = np.where(Vs[:, g])[0]                      # sample the node's reliable frames for candidates
        if gframes.size > nsamp:
            gframes = gframes[np.linspace(0, gframes.size - 1, nsamp).astype(int)]
        cand = set()
        for t in gframes:                                    # gather tracks within near_px of the node in 2D
            if tree_per[t] is None:
                continue
            for h in tree_per[t].query_ball_point(xy[g, t], near_px):
                b = int(idx_per[t][h])
                if b != g and alive[b]:
                    cand.add(b)
        supp = np.zeros(0, np.int64)
        if cand:
            cand = np.fromiter(cand, np.int64)
            co = Vs[:, g][:, None] & Vs[:, cand]             # (T, C) co-visible reliable frames
            n_ov = co.sum(0)
            d = np.linalg.norm(xy[cand] - xy[g][None], axis=2)   # (C, T) per-frame 2D distance
            d_co = np.where(co.T, d, np.nan)                 # only co-visible frames
            pdist = np.full(cand.size, np.inf)               # persistent 2D distance = p<quantile>
            has_ov = n_ov >= 1
            pdist[has_ov] = np.nanquantile(d_co[has_ov], quantile, axis=1)   # SAME metric as the edge builder
            min_len = np.minimum(Ocount[g], Ocount[cand])
            kill = cand[(n_ov > overlap_frac * min_len) & (pdist <= near_px)]
            alive[kill] = False
            supp = kill.astype(np.int64)
        alive[g] = False
        nodes.append(int(g)); suppressed.append(supp)
        if len(nodes) >= max_nodes:
            break
    nodes = np.array(nodes, np.int64)
    if verbose:
        print(f"[EDG] 2D length-NMS: {nodes.size} nodes from {has.size} tracks with >{int(min_node_frames)} "
              f"reliable frames (near<= {near_px:g}px p{int(quantile*100)}, overlap> {overlap_frac:.0%} of "
              f"shorter length)" + ("  [hit max_nodes]" if nodes.size >= max_nodes else ""))
    return (nodes, suppressed) if return_suppressed else nodes


def select_edg_nodes_covering(xy, Vs, Ocount, near_px0, overlap_frac=0.5, max_nodes=200,
                              min_node_frames=20, quantile=0.90, nsamp=30, verbose=True):
    """FULL-COVERAGE node selection within a node budget, two stages:
    (1) run the NMS WITHOUT an upper bound at the base radius -> the natural full-coverage node set
        (every eligible track is a node or within near_px0 of one);
    (2) if over budget, LENGTH-BIASED FARTHEST-POINT downsample: seed with the longest node, then
        repeatedly add the node farthest from the kept set (persistent 2D p<quantile> distance over
        co-visible frames; inf when temporally disjoint, so temporal coverage is kept too), breaking
        near-ties (>= 0.8 * max distance) toward the LONGEST track.
    Spread (coverage) first, track quality second. Any dropped node -- hence any eligible track --
    stays within ~(near_px0 + final spread radius) of a kept node. Returns (nodes, near_px0)."""
    nodes = select_edg_nodes_2d(xy, Vs, Ocount, near_px0, overlap_frac, 10**9,
                                min_node_frames, quantile, nsamp, verbose=False)
    if verbose:
        print(f"[EDG-cover] uncapped NMS at near_px={near_px0:.1f}: {nodes.size} nodes (budget {max_nodes})")
    if nodes.size <= max_nodes:
        return nodes, float(near_px0)
    N = nodes.size
    xyn = xy[nodes]                                  # (N, T, 2)
    Vn = Vs[:, nodes]                                # (T, N)
    length = Ocount[nodes].astype(np.float64)
    d_min = np.full(N, np.inf)
    keep_local = [int(np.argmax(length))]
    for _ in range(int(max_nodes) - 1):
        k = keep_local[-1]
        co = Vn[:, k][:, None] & Vn                  # (T, N) co-visible reliable frames
        d = np.linalg.norm(xyn - xyn[k][None], axis=2).T   # (T, N) per-frame 2D distance
        dk = np.where(co, d, np.nan)
        has = co.sum(0) >= 1
        pd = np.full(N, np.inf)
        if np.any(has):
            pd[has] = np.nanquantile(dk[:, has], quantile, axis=0)
        d_min = np.minimum(d_min, pd)
        d_min[keep_local] = -1.0                     # taken
        m = float(d_min.max())
        if m <= 0:
            break
        band = d_min >= (0.8 * m if np.isfinite(m) else np.inf)
        cand = np.where(band)[0]
        keep_local.append(int(cand[np.argmax(length[cand])]))
    kept = nodes[np.asarray(keep_local, np.int64)]
    if verbose:
        spread = float(max(d_min.max(), 0.0))
        print(f"[EDG-cover] FINAL: downsampled {N} -> {kept.size} nodes (length-biased farthest-point); "
              f"max node-to-kept spread ~{spread:.1f}px -> coverage radius ~{near_px0 + spread:.1f}px")
    return kept, float(near_px0)


def build_edg_graph(edges, Q, xy, Ps, Vs, Ocount, thr, near_px=15.0, overlap_frac=0.5,
                    edge_hops=6, max_nodes=100, min_node_frames=20, max_edge_factor=0.0,
                    quantile=0.90, min_degree=4, verbose=True, nodes_3d=False, near_frac=0.16,
                    edge_max_dist_factor=3.0, edge_min_max_ratio=0.5,
                    sphere_factor=1.3, sphere_pct=95.0):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    # --- NODE SELECTION: full-coverage NMS. EDG_NODES_3D=1 -> 3D world (Ps) distance for GT/LiDAR depth;
    #     default -> 2D image (near_px) for DA3 (3D untrustworthy). select fn is dimension-agnostic. ---
    if nodes_3d:
        pts_sel = np.transpose(Ps, (1, 0, 2))    # (T,Q,3) -> (Q,T,3) to match xy layout
        near_sel = thr * (near_frac / 0.16)      # 3D radius scaled by --edg-near-frac (smaller -> more nodes;
        #                                          then --edg-max-nodes + FPS trims to the budget)
        if verbose:
            print(f"[EDG] node selection: 3D world distance (radius={near_sel:.4f}, near_frac={near_frac})")
    else:
        pts_sel, near_sel = xy, near_px
    nodes, _ = select_edg_nodes_covering(pts_sel, Vs, Ocount, near_sel, overlap_frac, max_nodes,
                                         min_node_frames, quantile, verbose=verbose)
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

    # --- METRIC CAP + STABILITY FILTERS: hops are topological, so a chain of short hops can connect
    # FAR-APART nodes. Drop a hop-connected pair if (a) its robust p90 co-visible 3D distance exceeds
    # max_edge_factor*thr, OR (b) its MAXIMUM co-visible distance exceeds edge_max_dist_factor*thr (far in
    # even a few frames), OR (c) that distance VARIES a lot (min < edge_min_max_ratio*max, only when
    # max>thr). Never-co-visible pairs are dropped too (unverifiable). The min-degree augmentation below
    # then restores connectivity (>= min_degree per node), preferring edges that pass these filters. ---
    _do_cap = bool(max_edge_factor and max_edge_factor > 0)
    _do_far = bool(edge_max_dist_factor and edge_max_dist_factor > 0)
    _do_var = bool(edge_min_max_ratio and edge_min_max_ratio > 0)
    if (_do_cap or _do_far or _do_var) and edg_edges.shape[0]:
        cap = float(max_edge_factor) * float(thr) if _do_cap else np.inf   # p90 cap stays on thr (existing)
        p90 = np.full(edg_edges.shape[0], np.inf)
        dmn = np.full(edg_edges.shape[0], np.inf); dmx = np.full(edg_edges.shape[0], np.inf)
        for i, (a, b) in enumerate(edg_edges.tolist()):
            both = Vs[:, a] & Vs[:, b]
            if both.any():
                dd = np.linalg.norm(Ps[both, a] - Ps[both, b], axis=1)
                p90[i] = np.percentile(dd, 90); dmn[i] = float(dd.min()); dmx[i] = float(dd.max())
        # the NEW filters are relative to NODE SPACING (median finite edge p90), NOT thr -- thr can be much
        # smaller than the actual node spacing, which would over-reject and leave min-degree to re-add nearest.
        _fin = np.isfinite(p90)
        scale_node = float(np.median(p90[_fin])) if _fin.any() else float(thr)
        max_cap = float(edge_max_dist_factor) * scale_node if _do_far else np.inf
        keep_cap = p90 <= cap
        keep_far = dmx <= max_cap
        keep_var = ~((dmx > scale_node) & (dmn < float(edge_min_max_ratio) * dmx)) if _do_var else np.ones_like(keep_cap)
        keep = keep_cap & keep_far & keep_var & _fin
        if verbose:
            print(f"[EDG] edge cap/filter (node-spacing {scale_node:.4f}; p90<= {cap:.4f}, max<= {max_cap:.4f}, "
                  f"min>= {edge_min_max_ratio:g}max): kept {int(keep.sum())}/{edg_edges.shape[0]} "
                  f"(dropped {int((~keep_cap).sum())} p90, {int((keep_cap & ~keep_far).sum())} max-dist, "
                  f"{int((keep_cap & keep_far & ~keep_var).sum())} varying, {int((~_fin).sum())} never-co-visible)")
        edg_edges = edg_edges[keep]

    # --- MIN-DEGREE kNN FALLBACK: the hop graph can fragment on large / occlusion-heavy dynamic regions,
    # leaving the deformation graph disconnected so ARAP has nothing
    # to regularize. Guarantee every EDG node connects to >= min_degree nearest co-visible nodes (robust
    # p90 3D distance over co-visible frames -- same metric as the cap), adding edges as needed. ---
    if min_degree and min_degree > 0 and nodes.size > 1:
        edg_edges = _augment_edg_min_degree(edg_edges, nodes, Ps, Vs, int(min_degree), verbose,
                                            thr=float(thr), max_dist_factor=edge_max_dist_factor,
                                            min_max_ratio=edge_min_max_ratio)

    # --- FINAL SPHERE FILTER (the LAST step, after min-degree): the far node of a rigid edge must stay
    # within a sphere of radius sphere_factor * mean(distance) around the near node (the sphere that
    # "covers the EDG position"). Break an edge whose p<sphere_pct> co-visible distance escapes that sphere
    # (i.e. the node leaves it in >(100-sphere_pct)% of frames). Runs AFTER min-degree, so it CAN reduce a
    # node's degree -- that is intended (isolate cross-body links). ---
    if sphere_factor and sphere_factor > 0 and edg_edges.shape[0]:
        keep = np.ones(edg_edges.shape[0], bool)
        for i, (a, b) in enumerate(edg_edges.tolist()):
            both = Vs[:, a] & Vs[:, b]
            if int(both.sum()) < 3:
                continue
            d = np.linalg.norm(Ps[both, a] - Ps[both, b], axis=1)
            mu = float(d.mean())
            if mu > 1e-9 and float(np.percentile(d, sphere_pct)) > float(sphere_factor) * mu:
                keep[i] = False
        if verbose:
            print(f"[EDG] final sphere filter (p{sphere_pct:g} distance <= {sphere_factor:g}x mean): broke "
                  f"{int((~keep).sum())}/{edg_edges.shape[0]} edges whose far node escaped the covering sphere")
        edg_edges = edg_edges[keep]

    if verbose:
        print(f"[EDG] {nodes.size} nodes, {edg_edges.shape[0]} edges (hop distance <= {edge_hops}"
              + (f", min-degree {min_degree}" if min_degree and min_degree > 0 else "")
              + (f", sphere<= {sphere_factor:g}x mean" if sphere_factor and sphere_factor > 0 else "") + ")")
    return nodes, edg_edges


def _augment_edg_min_degree(edg_edges, nodes, Ps, Vs, min_degree, verbose=True,
                            thr=None, max_dist_factor=0.0, min_max_ratio=0.0):
    """Add kNN edges so every EDG node has >= min_degree neighbours. Distance = p90 of co-visible-frame
    3D distance (inf if never co-visible). Neighbours PREFER edges that pass the same stability filters as
    the edge cap (max co-visible dist <= max_dist_factor*thr, and not varying: min >= min_max_ratio*max),
    falling back to filter-failing neighbours only when needed so the min_degree guarantee is never broken.
    Nodes with fewer than min_degree co-visible partners connect to all available."""
    from collections import defaultdict
    K = nodes.size
    # persistent co-visible 3D distance stats (symmetric, inf on the diagonal / never-co-visible pairs)
    D = np.full((K, K), np.inf); Dmn = np.full((K, K), np.inf); Dmx = np.full((K, K), np.inf)
    Vn = [Vs[:, nodes[i]] for i in range(K)]
    for i in range(K):
        for j in range(i + 1, K):
            both = Vn[i] & Vn[j]
            if both.any():
                dd = np.linalg.norm(Ps[both, nodes[i]] - Ps[both, nodes[j]], axis=1)
                D[i, j] = D[j, i] = np.percentile(dd, 90)
                Dmn[i, j] = Dmn[j, i] = float(dd.min()); Dmx[i, j] = Dmx[j, i] = float(dd.max())
    bad = np.zeros((K, K), bool)                              # fails the same filters as the edge cap
    # scale = node spacing (median finite p90), consistent with the edge cap (NOT thr, which can be smaller)
    _finD = np.isfinite(D) & (D > 0)
    scale_node = float(np.median(D[_finD])) if _finD.any() else (float(thr) if thr else 1.0)
    if max_dist_factor and max_dist_factor > 0:
        bad |= Dmx > float(max_dist_factor) * scale_node
    if min_max_ratio and min_max_ratio > 0:
        bad |= (Dmx > scale_node) & (Dmn < float(min_max_ratio) * Dmx)
    adj = defaultdict(set)
    loc = {int(g): i for i, g in enumerate(nodes)}
    for a, b in edg_edges.tolist():
        adj[loc[int(a)]].add(loc[int(b)]); adj[loc[int(b)]].add(loc[int(a)])
    new_pairs = set()
    for i in range(K):
        need = min_degree - len(adj[i])
        if need <= 0:
            continue
        # prefer filter-passing neighbours (bad=False) nearest-first, then fall back to bad ones by distance
        order = sorted((j for j in range(K) if j != i and np.isfinite(D[i, j]) and j not in adj[i]),
                       key=lambda j: (bool(bad[i, j]), float(D[i, j])))
        for j in order:
            adj[i].add(j); adj[j].add(i)
            lo, hi = (i, j) if nodes[i] < nodes[j] else (j, i)
            new_pairs.add((int(nodes[lo]), int(nodes[hi])))
            need -= 1
            if need <= 0:
                break
    if new_pairs:
        add = np.array(sorted(new_pairs), dtype=np.int64)
        edg_edges = np.unique(np.concatenate([edg_edges, add], 0), axis=0) if edg_edges.size else add
        if verbose:
            print(f"[EDG] min-degree {min_degree}: +{len(new_pairs)} kNN edges for under-connected nodes")
    return edg_edges


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


def build_dyn_to_edg_edges(edges, Q, edg_nodes, Ps, Vs, Ocount, thr, hop_lt=2, min_conn=2,
                           max_factor=3.0, cand_k=8, quantile=90, verbose=True, hop_limit=20,
                           max_dist_factor=3.0, min_max_ratio=0.5, cand_near=0):
    """Attach each DYNAMIC point (track) to its `min_conn` EDG nodes (ORANGE edges), chosen so the
    connection stays SPATIALLY CLOSE ACROSS ALL co-visible frames. The skinning weight is set once and
    used for every frame (global), so a node that is near at one frame but far at others would corrupt
    the warp -- such nodes are rejected. Selection:
      * candidate nodes = the graph-reachable ones on the track graph (topology filter: cand_k geodesic-
        nearest + everything within < hop_lt hops), so we never even consider a node on a disconnected part;
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
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    if edg_nodes.size == 0:
        return np.zeros((0, 2), np.int64)
    node_mask = np.zeros(Q, bool); node_mask[edg_nodes] = True
    dyn_ids = np.where((Ocount > 0) & ~node_mask)[0]         # dynamic points (not EDG nodes)
    if dyn_ids.size == 0:
        return np.zeros((0, 2), np.int64)
    K = edg_nodes.size
    max_hop = int(hop_lt) - 1
    cap = float(max_factor) * float(thr)

    # --- candidate pool: graph-reachable nodes on the track graph (topology) ---
    cand = defaultdict(set)                                   # dyn track id -> set of node LOCAL indices
    Dn = None
    if edges.shape[0]:
        ii = np.concatenate([edges[:, 0], edges[:, 1]]); jj = np.concatenate([edges[:, 1], edges[:, 0]])
        A = csr_matrix((np.ones(ii.size, np.float32), (ii, jj)), shape=(Q, Q))
        Dn = dijkstra(A, directed=False, indices=edg_nodes, unweighted=True, limit=float(hop_limit))
    if Dn is not None:
        big = np.where(np.isfinite(Dn), Dn, np.inf)
        sub = big[:, dyn_ids]                                # (K, n_dyn)
        kk = int(min(max(int(cand_k), int(min_conn)), K))
        part = np.argpartition(sub, min(kk, K - 1), axis=0)[:kk]          # (kk, n_dyn) geodesic-nearest
        for c, q in enumerate(dyn_ids.tolist()):
            for nl in part[:, c].tolist():
                if np.isfinite(sub[nl, c]):
                    cand[q].add(int(nl))
        if max_hop >= 1:                                     # also everything within < hop_lt hops
            ni, qi = np.where((Dn >= 1) & (Dn <= max_hop) & np.isfinite(Dn))
            keep = ~node_mask[qi]
            for nl, q in zip(ni[keep].tolist(), qi[keep].tolist()):
                cand[q].add(int(nl))
    elif cand_near and cand_near < K:                         # no track graph (speed): the cand_near
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
    else:                                                    # no track graph: every node is a candidate
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
        res = list(tqdm(ex.map(_node_stats, node_pts.items()), total=len(node_pts), desc="[EDG] dyn->EDG dist", disable=not verbose))
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
    n_far, n_vary, n_capped = int(far.sum()), int(vary.sum()), int((stable & ~under).sum())
    n_dropped = int((np.bincount(grp, stable) == 0).sum())   # every candidate far / varying -> no link

    if not keep.any():
        return np.zeros((0, 2), np.int64)
    und = np.unique(np.stack([lq[keep], lg[keep]], axis=1), axis=0)   # (E,2) = (dynamic track, edg node)
    if verbose:
        deg = np.unique(lq[keep], return_counts=True)[1]
        print(f"[EDG] dyn->EDG: {und.shape[0]} orange edges over {deg.size} dynamic points "
              f"({min_conn} nearest by robust p{quantile} co-visible dist; scale {scale:.4f}, p90-cap "
              f"{cap:.4f} (x{max_factor:g}), max-cap {max_cap:.4f} (x{max_dist_factor:g})). Dropped links: "
              f"{n_capped} over p90-cap, {n_far} max-dist too large, {n_vary} varying(min<{min_max_ratio:g}max); "
              f"{n_dropped} points left unlinked; nodes/pt min={int(deg.min())} med={int(np.median(deg))} "
              f"max={int(deg.max())})")
    return und.astype(np.int64)


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
    ap.add_argument("--max-sampled-frames", type=int, default=0,
                    help="Frames used to build the topology; 0 = FULL video (every frame, default). "
                         ">0 subsamples that many frames (faster, less co-visibility evidence). Capped at T.")
    ap.add_argument("--candidate-k", type=int, default=32)
    ap.add_argument("--min-overlap-frames", type=int, default=3)
    ap.add_argument("--min-overlap-ratio", type=float, default=0.20)
    ap.add_argument("--distance-quantile", type=float, default=0.95)
    ap.add_argument("--max-neighbors", type=int, default=100)
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
    ap.add_argument("--edg-near-px", type=float, default=0.0,
                    help="EDG node selection RANGE: delete a track whose persistent 2D IMAGE distance to a "
                         "chosen node (the p<edg-dist-quantile> of per-frame px distance over co-visible "
                         "frames -- same metric as the edge builder) is <= this many pixels. Lower = fewer "
                         "deletions / more nodes. (2D image distance, NOT 3D.) 0 = ADAPTIVE: "
                         "edg-near-frac * sqrt(median dynamic-mask area).")
    ap.add_argument("--edg-near-frac", type=float, default=0.16,
                    help="ADAPTIVE radius fraction for --edg-near-px 0: near_px = frac * sqrt(median "
                         "dynamic-mask area px^2). 0.16 reproduces the tuned 15px on paper-windmill.")
    ap.add_argument("--edg-dist-quantile", type=float, default=0.90,
                    help="quantile for the EDG node-selection persistent 2D distance (p90 = robust near-max; "
                         "matches build_edg's edge-builder distance_quantile).")
    ap.add_argument("--edg-overlap-frac", type=float, default=0.5,
                    help="EDG node selection: delete a near track only if it temporally overlaps the chosen "
                         "node by MORE than this fraction of the shorter track's length (0.5 = >50%%).")
    ap.add_argument("--edg-edge-hops", type=int, default=4,
                    help="connect two EDG nodes when their hop distance on the track graph is <= this")
    ap.add_argument("--edg-max-edge-factor", type=float, default=5.0,
                    help="drop EDG edges whose robust 3D distance (p90 over co-visible frames) exceeds this "
                         "factor x the edge-distance threshold -- prevents hop-chains from connecting "
                         "far-apart nodes (0 = off; ~2.0 recommended)")
    ap.add_argument("--edg-max-dist-factor", type=float, default=3.0,
                    help="HARD: drop an EDG node-node edge whose MAXIMUM co-visible 3D distance exceeds this "
                         "factor x edge-threshold (far in even a few frames). The --edg-min-degree guarantee "
                         "still applies afterwards (prefers passing edges). 0=off.")
    ap.add_argument("--edg-min-max-ratio", type=float, default=0.5,
                    help="HARD: drop an EDG node-node edge whose distance VARIES a lot (min co-visible dist < "
                         "this fraction of max, only when max>threshold). --edg-min-degree still guaranteed. 0=off.")
    ap.add_argument("--edg-sphere-factor", type=float, default=1.3,
                    help="FINAL filter (LAST, after min-degree): break an EDG edge whose p<edg-sphere-pct> "
                         "co-visible distance exceeds this factor x its MEAN distance -- i.e. the far node "
                         "escapes a sphere of radius factor*mean covering its average position. 0=off. Runs "
                         "last, so it CAN drop a node below min-degree (isolating cross-body links).")
    ap.add_argument("--edg-sphere-pct", type=float, default=95.0,
                    help="percentile used as the 'max' distance in --edg-sphere-factor (95 = allow 5% outlier frames).")
    ap.add_argument("--edg-max-nodes", type=int, default=200,
                    help="EDG node BUDGET: the selection radius is auto-tuned so FULL coverage of all "
                         "eligible tracks fits within this many nodes (select_edg_nodes_covering)")
    ap.add_argument("--edg-min-frames", type=int, default=20,
                    help="EDG node must be reliably observed in MORE than this many frames (length floor).")
    ap.add_argument("--edg-min-degree", type=int, default=4,
                    help="Guarantee every EDG NODE connects to >= this many nearest co-visible nodes "
                         "(kNN fallback after the hop graph). Fixes disconnected deformation graphs on "
                         "large/occluded dynamic regions (0 disables; keeps hop-only edges).")
    ap.add_argument("--horizon-out", type=str, default="",
                    help="save the EDG active-horizon image (num_EDG x frames, green=observed) to this "
                         "path (default: output_dir/edg_active_horizon.png)")
    ap.add_argument("--horizon-only", action="store_true",
                    help="only build the EDG + save the active-horizon image, then exit (no 3D window)")
    ap.add_argument("--dyn-edg-hop-lt", type=int, default=3,
                    help="ORANGE dynamic->EDG edge when a dynamic point is within < this many hops of an "
                         "EDG node on the track graph (e.g. 2 = 1 hop / directly adjacent).")
    ap.add_argument("--dyn-edg-min", type=int, default=2,
                    help="attach every dynamic point to this-many EDG nodes, chosen among graph-reachable "
                         "candidates by ROBUST per-frame distance over co-visible frames (stays close in "
                         "ALL frames, since skinning is global). 0=off.")
    ap.add_argument("--dyn-edg-max-factor", type=float, default=2.5,
                    help="drop a dynamic->EDG link whose robust (p90) co-visible-frame distance exceeds "
                         "this factor x the typical nearest-node distance (node spacing) -- prevents "
                         "connections that are far in some frames. Lower = stricter.")
    ap.add_argument("--dyn-edg-max-dist-factor", type=float, default=3.0,
                    help="HARD: drop a dynamic->EDG link whose MAXIMUM co-visible-frame distance exceeds "
                         "this factor x node spacing -- rejects links far in even a few frames (not just on "
                         "average). 0=off.")
    ap.add_argument("--dyn-edg-min-max-ratio", type=float, default=0.5,
                    help="HARD: drop a dynamic->EDG link whose distance VARIES a lot: min co-visible distance "
                         "< this fraction of the max (only when the point ever gets far, dmax>spacing) -- "
                         "catches points sliding toward/away from a node (non-rigid). 0=off.")
    ap.add_argument("--topology-out", type=str, default="",
                    help="save the EDG + dynamic-edge topology npz here (default output_dir/edg_topology.npz)")
    ap.add_argument("--save-npz", type=str, default="", help="also save edges to this .npz path")
    ap.add_argument("--merged", action=argparse.BooleanOptionalAction, default=True,
                    help="use MERGED tracks (default on; --no-merged uses raw tracks_dynamic_only)")
    ap.add_argument("--smoothed", action=argparse.BooleanOptionalAction, default=True,
                    help="use the SMOOTHED tracks tracks_dynamic_<base>_smoothed.npz (default on; --no-smoothed)")
    ap.add_argument("--gt-depth", action="store_true",
                    help="EDG node-selection distance in 3D world (use with GT/LiDAR/completed depth); default 2D image (DA3)")
    args = ap.parse_args()

    import open3d as o3d

    # merged (which base tracks) and smoothed (whether optimized) are ORTHOGONAL; both default ON.
    base = "merged" if args.merged else "only"
    name = f"tracks_dynamic_{base}_smoothed.npz" if args.smoothed else f"tracks_dynamic_{base}.npz"
    _tp = str(Path(utils.definitions.output_dir) / name)
    if not Path(_tp).exists():
        raise FileNotFoundError(
            f"{name} not found (tracks_dynamic_only_smoothed.npz comes from smooth_dynamic_tracks.py;\n"
            f"  build_edg.py --no-smoothed uses the un-smoothed tracks_dynamic_{base}.npz)")
    print(f"[tracks] EDG on {name}  |  node-selection distance: {'3D world (--gt-depth)' if args.gt_depth else '2D image'}")
    D = _load_all(tracks_path=_tp)
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
        xy_s = D["tracks_xy"][0][:, aux["sampled"]].astype(np.float64)   # 2D px aligned to Vs' time axis
        near_px_eff = (float(args.edg_near_px) if float(args.edg_near_px) > 0
                       else _dyn_region_near_px(D, args.edg_near_frac))
        edg_nodes, edg_edges = build_edg_graph(
            edges, Q, xy_s, aux["Ps"], aux["Vs"], aux["Ocount"], aux["thr"],
            near_px=near_px_eff, overlap_frac=args.edg_overlap_frac, nodes_3d=args.gt_depth, near_frac=args.edg_near_frac,
            edge_hops=args.edg_edge_hops, max_nodes=args.edg_max_nodes,
            min_node_frames=args.edg_min_frames, max_edge_factor=args.edg_max_edge_factor,
            quantile=args.edg_dist_quantile, min_degree=args.edg_min_degree,
            edge_max_dist_factor=args.edg_max_dist_factor, edge_min_max_ratio=args.edg_min_max_ratio,
            sphere_factor=args.edg_sphere_factor, sphere_pct=args.edg_sphere_pct)

    # dynamic-point -> EDG node edges (ORANGE): nodes within < dyn_edg_hop_lt hops, unioned with each
    # point's dyn_edg_min nearest nodes AT ITS FIRST APPEARING FRAME (every appearing point gets >= that many).
    _aux = D["_edg_aux"]
    dyn_edg_edges = (build_dyn_to_edg_edges(
                        edges, Q, edg_nodes, _aux["Ps"], _aux["Vs"], _aux["Ocount"], _aux["thr"],
                        hop_lt=args.dyn_edg_hop_lt, min_conn=args.dyn_edg_min,
                        max_factor=args.dyn_edg_max_factor,
                        max_dist_factor=args.dyn_edg_max_dist_factor,
                        min_max_ratio=args.dyn_edg_min_max_ratio)
                     if edg_nodes.size else np.zeros((0, 2), np.int64))

    # save the topology result: EDG nodes/edges + dynamic-point (track) edges + dynamic->EDG edges
    topo_out = args.topology_out or str(Path(utils.definitions.output_dir) / "edg_topology.npz")
    np.savez_compressed(topo_out, edg_nodes=edg_nodes, edg_edges=edg_edges,
                        track_edges=edges, track_scores=scores, dyn_edg_edges=dyn_edg_edges,
                        dyn_edg_hop_lt=int(args.dyn_edg_hop_lt), dyn_edg_min_conn=int(args.dyn_edg_min),
                        n_tracks=int(Q))                              # track-id space stamp (merged vs raw)
    print(f"[topology] saved -> {topo_out}  ({edg_nodes.size} EDG nodes, {edg_edges.shape[0]} EDG edges, "
          f"{edges.shape[0]} track edges, {dyn_edg_edges.shape[0]} dyn->EDG edges)")

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
    C_DYN = np.array([0.6, 0.2, 0.85])       # purple = ALL dynamic points (coloured uniformly)
    C_EDG = [1.0, 0.05, 0.05]                # red    = EDG nodes + edges
    C_DYN2EDG = [0.95, 0.55, 0.1]            # orange = dynamic point -> EDG node (hop < dyn_edg_hop_lt)

    # EDG node marker radius from the scene scale at the start frame
    P0 = _valid_at(D, s, int(np.clip(args.start, 0, T - 1)))[0]
    fin0 = np.isfinite(P0).all(axis=1)
    diag = float(np.linalg.norm(P0[fin0].max(0) - P0[fin0].min(0))) if fin0.any() else 1.0
    node_r = max(1e-4, 0.006 * diag)

    state = {"idx": int(np.clip(args.start, 0, T - 1))}
    pcd = o3d.geometry.PointCloud()
    lines = o3d.geometry.LineSet()
    edg_line = o3d.geometry.LineSet()                    # red EDG edges
    dyn_edg_line = o3d.geometry.LineSet()                # orange dynamic-point -> EDG node edges
    edg_mesh = o3d.geometry.TriangleMesh()               # red EDG node spheres
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    def _fill(frame):
        pruned = D.get("pruned")
        if show_all:
            P, status = _lift_all_at(D, s, frame)
            if pruned is not None:
                status[(status == 0) & pruned] = 3   # redundant (pre-filtered) -> own colour, no edges
            v = status == 0                          # reliable (non-occluded, in-mask) dynamic points
            draw = v                                 # only these are drawn (occluded / off-mask hidden)
            idx = np.where(draw)[0]
            st = status[idx]
            pcd.points = o3d.utility.Vector3dVector(P[idx])
            pcd.paint_uniform_color(C_DYN)           # ALL dynamic points -> purple
            n_valid = int((st == 0).sum()); n_occ = int((st == 1).sum())
            n_off = int((st == 2).sum()); n_pruned = int((st == 3).sum())
        else:
            P, v = _valid_at(D, s, frame)
            if pruned is not None:
                v = v & ~pruned                      # pruned tracks are hidden in --valid-only
            pts = P[v]
            pcd.points = o3d.utility.Vector3dVector(pts)
            if len(pts) > 0:
                pcd.paint_uniform_color(C_DYN)       # ALL dynamic points -> purple
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

        # --- dynamic-point -> EDG node edges (ORANGE), drawn where both endpoints reliable this frame ---
        n_de = 0
        if dyn_edg_edges.size:
            dem = v[dyn_edg_edges[:, 0]] & v[dyn_edg_edges[:, 1]]
            dyn_edg_line.points = o3d.utility.Vector3dVector(P_safe)
            dyn_edg_line.lines = o3d.utility.Vector2iVector(
                dyn_edg_edges[dem].astype(np.int32) if dem.any() else np.zeros((0, 2), np.int32))
            dyn_edg_line.paint_uniform_color(C_DYN2EDG)
            n_de = int(dem.sum())
        return n_valid, n_occ, n_off, n_pruned, n_e, n_hidden, n_en, n_ee, n_de

    def refresh(vis, frame):
        n_valid, n_occ, n_off, n_pruned, n_e, n_hidden, n_en, n_ee, n_de = _fill(frame)
        for g in (pcd, lines, edg_line, dyn_edg_line, edg_mesh):
            vis.update_geometry(g)
        hid = f"  hidden(over-stretched)={n_hidden}" if args.max_draw_stretch > 0 else ""
        extra = (f"  hidden: occluded={n_occ} off-mask={n_off}" if show_all else "")
        edg = f"  EDG(red): {n_en} nodes {n_ee} edges" if edg_nodes.size else ""
        orn = f"  dyn->EDG(orange): {n_de}" if dyn_edg_edges.size else ""
        print(f"[Open3D] frame={frame}  reliable={n_valid}{extra}  edges={n_e}{hid}{edg}{orn}")
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

    import sys
    if not sys.stdout.isatty():
        print("[viewer] non-interactive session: skipping Open3D viewer (topology already saved)")
        return
    legend = ("Points: ALL dynamic points in PURPLE. " if show_all
              else "Points: reliable dynamic tracks only (--valid-only), PURPLE. ")
    edgleg = "" if args.no_edg else "RED spheres/lines = coarse EDG (hop-distance deformation graph). "
    ornleg = "" if not dyn_edg_edges.size else f"ORANGE lines = dynamic->EDG (hop < {args.dyn_edg_hop_lt}). "
    print("Open3D controls: SPACE=next frame, B=prev, S=save PLY, Q/ESC=close. "
          f"Start frame={state['idx']} (T={T}). {legend}Cyan lines = track connectivity (Notes/EDG.md). {edgleg}{ornleg}")
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="build_edg: dynamic points (purple) + EDG (red)", width=1280, height=720)
    _fill(state["idx"])  # fill BEFORE add_geometry so the camera fits
    vis.add_geometry(pcd)
    vis.add_geometry(lines)
    vis.add_geometry(edg_line)
    vis.add_geometry(dyn_edg_line)
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
