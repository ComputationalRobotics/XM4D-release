#!/usr/bin/env python3
"""Build the fixed EDG connectivity over ALL tracks (per Notes/EDG.md) and visualize it.

Same builder as visualize_track_edges.py, with two differences: the topology is built from the FULL
video (every frame, not a subsample), and the viewer draws ALL dynamic points in a single PURPLE colour
(the red EDG nodes/edges and cyan track edges are unchanged).

Edges are decided by *persistent* world-space proximity, not single-frame closeness (Notes/EDG.md):
  1. lift every reliable track observation to world 3D (same lift as visualize_lifted_dynamic_frame0.py:
     X_world = da3_frame_scale[f] * R_raw[f] @ tracks_xyz[:, f] + t_raw[f]);
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
plus the EDG.md params.
"""
import argparse
import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

import utils.definitions
from utils import dataset_config
from utils.edg import scatter_solved_poses, scatter_solved_frame_vector


# ----------------------------------------------------------------------------- data loading (matches
# visualize_lifted_dynamic_frame0.py so the lift/validity are identical)
def _load_all(tracks_path=None):
    root = Path(utils.definitions.rootpath)
    data = np.load(tracks_path or utils.definitions.tracks_dynamics_path, allow_pickle=True)
    dv = data["depth_variant"].item() if "depth_variant" in getattr(data, "files", []) else None
    if dv is not None and dv != utils.definitions.depth_variant:
        print(f"[depth] *** WARNING: {Path(tracks_path or utils.definitions.tracks_dynamics_path).name} was lifted with XM_DEPTH_VARIANT={dv} but this run "
              f"uses {utils.definitions.depth_variant} -- 3D will NOT match the file. Re-run with "
              f"XM_DEPTH_VARIANT={dv} or re-lift the tracks. ***")
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
                               t_expected=D["T"], mask_mode="dynamic_positive", target_size=(W, H))  # (T,H,W) bool
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


def _compute_person_ids(D, min_area_frac=0.005, match_radius_frac=0.18, min_frames_labeled=1):
    """Assign each dynamic track to a PHYSICAL BODY (person_id, int64 (Q,); -1 = unresolved), so the
    edge/node builders below never bridge two independently-moving people just because they share one
    dynamic mask (e.g. multiple people who visually overlap/merge in 2D at some frames). Independent of
    the 3D lift (which can itself be wrong) -- uses only the 2D dynamic mask's CONNECTED COMPONENTS.

    Method:
      1. Per frame with >=2 "big" components (area > min_area_frac of the frame), greedily match each
         component to a GLOBAL person id by nearest centroid to the previous labeled frame's centroids
         (within match_radius_frac of the image diagonal); unmatched components start a new global id.
         Frames with <2 big components (e.g. people have visually merged) are skipped for this matching
         step -- they can't disambiguate identity, but by then most tracks already got labeled earlier.
      2. Each track takes the global id of the FIRST (earliest) frame where it is reliable AND its pixel
         falls inside a labeled component.
      3. Any track never observed during a resolvable frame (e.g. only visible after full merge) is
         NN-propagated in 3D (mean lifted position) from the nearest already-labeled track.
      4. If nothing was ever resolvable (e.g. a genuinely single-person scene), every track gets id 0 --
         identical to the current (no-op) behaviour.
    """
    import cv2
    Q, T = D["tracks_xyz"].shape[1], D["T"]
    person_id = np.full(Q, -1, dtype=np.int64)
    mask_dir = getattr(utils.definitions, "mask_dir", None)
    if mask_dir is None or not Path(str(mask_dir)).exists():
        print("[person] no mask dir; treating scene as single-body (no-op).")
        return np.zeros(Q, dtype=np.int64)
    try:
        from utils.track_tapnet import load_dynamic_masks
    except Exception as e:
        print(f"[person] cannot load masks ({e}); treating scene as single-body (no-op).")
        return np.zeros(Q, dtype=np.int64)
    H, W = cv2.imread(D["image_paths"][0]).shape[:2]
    masks = load_dynamic_masks(str(mask_dir), utils.definitions.mask_pattern,
                               t_expected=T, mask_mode="dynamic_positive", target_size=(W, H))
    min_area = float(min_area_frac) * H * W
    match_radius = float(match_radius_frac) * float(np.hypot(H, W))

    xy = D["tracks_xy"][0]                                    # (Q, T, 2) native px
    occ = D["occluded"][0]; dyn = np.asarray(D["dyn_mask"]).astype(bool)
    reliable = (~occ) & dyn & np.isfinite(xy).all(-1)         # (Q, T)

    active_centroids = {}                                     # global_id -> last-seen centroid (px)
    next_id = 0
    frame_labels = []                                         # per frame: (label_map, {local_label: global_id})
    n_multibody_frames = 0
    for t in range(T):
        n, lbl, stats, cent = cv2.connectedComponentsWithStats(masks[t].astype(np.uint8), connectivity=8)
        big = [i for i in range(1, n) if stats[i, 4] > min_area]
        if len(big) < 2:
            frame_labels.append((None, {}))
            continue
        n_multibody_frames += 1
        assign = {}
        used_globals = set()
        # greedy nearest-first matching to previously active global ids
        pairs = []
        for i in big:
            for gid, gc in active_centroids.items():
                d = float(np.hypot(cent[i][0] - gc[0], cent[i][1] - gc[1]))
                if d <= match_radius:
                    pairs.append((d, i, gid))
        pairs.sort()
        for d, i, gid in pairs:
            if i in assign or gid in used_globals:
                continue
            assign[i] = gid; used_globals.add(gid)
        for i in big:
            if i not in assign:
                assign[i] = next_id; next_id += 1
        for i in big:
            active_centroids[assign[i]] = cent[i]
        frame_labels.append((lbl, assign))

    n_persons_found = next_id
    if n_persons_found == 0:
        print("[person] no frame had >=2 separable bodies; treating scene as single-body (no-op).")
        return np.zeros(Q, dtype=np.int64)

    for g in range(Q):
        for t in range(T):
            if not reliable[g, t]:
                continue
            lbl, assign = frame_labels[t]
            if lbl is None:
                continue
            x, y = xy[g, t]
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= xi < W and 0 <= yi < H):
                continue
            loc = int(lbl[yi, xi])
            if loc in assign:
                person_id[g] = assign[loc]
                break

    n_labeled_direct = int((person_id >= 0).sum())
    # NN-propagate stragglers from labeled tracks' mean 3D lifted position
    xyz = D["tracks_xyz"][0]
    canon = np.full((Q, 3), np.nan)
    finite_cnt = np.isfinite(xyz).all(-1).sum(1)
    has_any = finite_cnt > 0
    canon[has_any] = np.nanmean(np.where(np.isfinite(xyz[has_any]), xyz[has_any], np.nan), axis=1)
    labeled = np.where((person_id >= 0) & has_any)[0]
    unlabeled = np.where((person_id < 0) & has_any)[0]
    if labeled.size > 0 and unlabeled.size > 0:
        tree = cKDTree(canon[labeled])
        _, nn = tree.query(canon[unlabeled], k=1)
        person_id[unlabeled] = person_id[labeled[nn]]

    counts = {int(p): int((person_id == p).sum()) for p in sorted(set(person_id[person_id >= 0].tolist()))}
    print(f"[person] {n_multibody_frames}/{T} frames had >=2 separable bodies; found {n_persons_found} "
          f"physical bodies; {n_labeled_direct} tracks labeled directly, "
          f"{int((person_id >= 0).sum()) - n_labeled_direct} propagated by 3D nearest-neighbour, "
          f"{int((person_id < 0).sum())} unresolved (treated as wildcard). counts={counts}")
    return person_id


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
        from utils.track_tapnet import load_dynamic_masks
    except Exception as e:
        print(f"[edg-near] cannot load masks ({e}); using fixed {fallback}px")
        return float(fallback)
    H, W = cv2.imread(D["image_paths"][0]).shape[:2]
    masks = load_dynamic_masks(str(mask_dir), utils.definitions.mask_pattern,
                               t_expected=D["T"], mask_mode="dynamic_positive", target_size=(W, H))
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


# ----------------------------------------------------------------------------- edge builder (EDG.md)
@dataclass
class EdgeConfig:
    max_sampled_frames: int = 0          # 0 = full video (every frame); >0 subsamples that many
    candidate_k: int = 32
    candidate_radius: float = 0.0        # 0 -> 1.5 * edge_distance_threshold
    min_overlap_frames: int = 3
    min_overlap_ratio: float = 0.20
    distance_quantile: float = 0.90      # edge accepted if this quantile (x%) of per-frame track
                                         # distances <= threshold (EDG_EDGE_DIST_QUANTILE)
    edge_distance_threshold: float = 0.0  # 0 -> auto = edge_thresh_mult * median NN distance
    edge_thresh_mult: float = 5.0        # x in "threshold = x * median NN dist" (EDG_EDGE_THRESH_MULT)
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
    for k, f in enumerate(sampled):
        P, v = _valid_at(D, s, int(f))
        Ps[k] = P
        Vs[k] = v
    return Ps, Vs


def _auto_edge_threshold(Ps, Vs, mult=5.0):
    """Edge distance threshold = mult * median nearest-neighbour distance (a few sampled frames).
    `mult` is the config-controlled x in "x * median NN dist" (EDG_EDGE_THRESH_MULT)."""
    nn = []
    for k in range(min(5, Ps.shape[0])):
        idx = np.where(Vs[k])[0]
        if idx.size < 5:
            continue
        dd, _ = cKDTree(Ps[k, idx]).query(Ps[k, idx], k=2)
        nn.append(dd[:, 1])
    if not nn:
        return 0.1
    return float(mult * np.median(np.concatenate(nn)))


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
    """Implements Notes/EDG.md. Returns edges (E,2) int track-index pairs and scores (E,) distances."""
    T = D["T"]
    if cfg.max_sampled_frames and cfg.max_sampled_frames > 0:
        sampled = np.unique(np.linspace(0, T - 1, min(cfg.max_sampled_frames, T)).astype(int))
    else:
        sampled = np.arange(T, dtype=int)                  # full video: every frame
    Ps, Vs = _stack_sampled(D, s, sampled)
    Ocount = Vs.sum(axis=0)  # per-track reliable sampled-frame count

    thr = cfg.edge_distance_threshold or _auto_edge_threshold(Ps, Vs, cfg.edge_thresh_mult)

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

    radius = cfg.candidate_radius or (2.0 * thr)
    Ps2d = thr2d = radius2d = None
    if cfg.use_2d:
        xy = D["tracks_xy"][0]                                          # (Q, T, 2) image px
        Ps2d = np.where(Vs[..., None], np.transpose(xy[:, sampled, :], (1, 0, 2)).astype(np.float64), np.nan)
        thr2d = cfg.edge_thresh_2d or _auto_edge_threshold_2d(Ps2d, Vs)
        radius2d = cfg.candidate_radius_2d or (2.0 * thr2d)
    if verbose:
        print(f"[edges] {sampled.size} frames{' (full video)' if sampled.size == T else ' (sampled)'}, "
              f"edge_thresh={thr:.4f}, cand_radius={radius:.4f}"
              + (f" | 2D rescue: edge_thresh={thr2d:.1f}px" if cfg.use_2d else ""))

    # --- step 2: candidate pairs (union of per-frame KD-tree neighbours) ------------------------
    # Vectorised + parallel (workers=-1): the old per-entry Python set-insert loop dominated this pass.
    # Pairs are encoded min*Q + max into int64 and unique'd once -- the same SET as before, and the final
    # edge array is sorted downstream, so the output is identical.
    Qtot = Vs.shape[1]
    enc = []

    def _collect(P_k, idx, rad):
        tree = cKDTree(P_k)
        kq = min(cfg.candidate_k + 1, idx.size)
        dists, nbrs = tree.query(P_k, k=kq, distance_upper_bound=rad, workers=-1)
        dists = np.atleast_2d(dists); nbrs = np.atleast_2d(nbrs)
        A = np.repeat(np.arange(idx.size), nbrs.shape[1])
        B, Dv = nbrs.ravel(), dists.ravel()
        m = (B < idx.size) & np.isfinite(Dv)
        ia, ib = idx[A[m]], idx[B[m]]
        m = ia != ib
        ia, ib = ia[m], ib[m]
        lo, hi = np.minimum(ia, ib), np.maximum(ia, ib)
        enc.append(lo.astype(np.int64) * Qtot + hi)

    for k in tqdm(range(sampled.size), desc="[edges] 3D candidates", disable=not verbose):
        idx = np.where(Vs[k])[0]
        if idx.size >= 2:
            _collect(Ps[k, idx], idx, radius)
    # 2D candidate pass: per-frame KD-tree in IMAGE space, so image-adjacent pairs become candidates
    # even if they are far apart in 3D (bad depth). Acceptance below verifies persistent 2D closeness.
    if cfg.use_2d:
        for k in tqdm(range(sampled.size), desc="[edges] 2D candidates", disable=not verbose):
            idx = np.where(Vs[k])[0]
            if idx.size >= 2:
                _collect(Ps2d[k, idx], idx, radius2d)
    uniq = np.unique(np.concatenate(enc)) if enc else np.zeros(0, np.int64)
    if verbose:
        print(f"[edges] {uniq.size} candidate pairs")
    if uniq.size == 0:
        return np.zeros((0, 2), np.int64), np.zeros((0,), np.float64)

    cand_arr = np.stack([uniq // Qtot, uniq % Qtot], 1)
    ci, cj = cand_arr[:, 0], cand_arr[:, 1]

    # --- MULTI-BODY SEPARATION: never propose an edge between two DIFFERENT physical bodies (e.g.
    # several people who visually overlap in 2D at some frames). person_id[-1] = unresolved -> treated
    # as a wildcard (kept) so ambiguous points aren't needlessly dropped. ---
    pid = D.get("person_id")
    if pid is not None and int(pid.max(initial=-1)) > 0:
        same_body = (pid[ci] == pid[cj]) | (pid[ci] < 0) | (pid[cj] < 0)
        n_cross = int((~same_body).sum())
        if n_cross:
            cand_arr = cand_arr[same_body]; ci, cj = cand_arr[:, 0], cand_arr[:, 1]
            if verbose:
                print(f"[edges] multi-body filter: dropped {n_cross} cross-body candidate pairs")
        if cand_arr.shape[0] == 0:
            return np.zeros((0, 2), np.int64), np.zeros((0,), np.float64)

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
        s3 = _nanq_cols(d3, cfg.distance_quantile)                  # (C',) 3D persistent distance
        ok = s3 <= thr
        if cfg.use_2d:
            nd = ~ok                                               # 2D can only RESCUE pairs 3D rejected,
            if nd.any():                                           # so compute it just for those (same ok)
                d2 = np.where(co[:, nd], np.linalg.norm(Ps2d[:, a[nd], :] - Ps2d[:, b[nd], :], axis=2), np.nan)
                s2 = _nanq_cols(d2, cfg.distance_quantile)         # 2D persistent distance (px)
                ok2 = s2 <= thr2d
                n_2d_rescue += int(ok2.sum())                      # accepted by 2D but NOT by 3D
                ok[nd] = ok2
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
        pid = D.get("person_id")
        multi_body = pid is not None and int(pid.max(initial=-1)) > 0
        # MULTI-BODY SEPARATION: this fallback used to search ALL has_idx globally for whoever is
        # nearest -- exactly the mechanism that bridges two different people when one is under-observed.
        # Build one candidate pool PER BODY (own-body only) and search unresolved tracks (person_id -1)
        # against the full global pool, since we don't know which body they belong to.
        groups = []  # list of (under_connected_subset, candidate_pool)
        if multi_body:
            for p in sorted(set(pid[has_idx].tolist())):
                if p < 0:
                    continue
                sel = has_idx[pid[has_idx] == p]
                u_group = [u for u in under if pid[u] == p]
                groups.append((u_group, sel))
            unresolved_u = [u for u in under if pid[u] < 0]
            if unresolved_u:
                groups.append((unresolved_u, has_idx))
        else:
            groups.append((under, has_idx))
        for u_group, sel in groups:
            if not u_group or sel.size <= 1:
                continue
            tree = cKDTree(canon[sel])
            kq = int(min(cfg.min_degree + 6, sel.size))
            dd, nn = tree.query(canon[u_group], k=kq)
            dd = np.atleast_2d(dd); nn = np.atleast_2d(nn)
            for r, u in enumerate(u_group):
                need = cfg.min_degree - len(adj[u])
                for lj, d in zip(nn[r], dd[r]):
                    if need <= 0:
                        break
                    gj = int(sel[lj])
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
    # MULTI-BODY SEPARATION: never rigid-link two parts that belong to different PHYSICAL bodies (e.g.
    # two people), even if a sparse/short co-visible window makes their motion look spuriously rigid.
    pid = D.get("person_id")
    part_body = None
    if pid is not None and int(pid.max(initial=-1)) > 0:
        part_body = []
        for pr in parts:
            labs = pid[pr]; labs = labs[labs >= 0]
            part_body.append(int(np.bincount(labs).argmax()) if labs.size else -1)
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
            if part_body is not None and part_body[i] >= 0 and part_body[j] >= 0 and part_body[i] != part_body[j]:
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
# (This is the exact algorithm that visualize_edg_suppressed.py uses/visualizes.)
def _frame_trees_2d(xy, Vs, T):
    """Per-frame KD-tree over the reliable tracks' 2D image positions (for candidate gathering)."""
    idx_per, tree_per = [], []
    for t in range(T):
        idx = np.where(Vs[t])[0]
        idx_per.append(idx)
        tree_per.append(cKDTree(xy[idx, t]) if idx.size else None)
    return idx_per, tree_per


def _nanq_cols(x, q):
    """np.nanquantile(x, q, axis=0) via ONE vectorised sort (NaNs sort to the end) -- same 'linear'
    interpolation (incl. numpy's two-sided lerp), orders of magnitude faster than nanquantile, which
    dominated the persistent-proximity pass."""
    xs = np.sort(x, axis=0)
    n = (~np.isnan(x)).sum(0)
    out = np.full(x.shape[1], np.nan)
    ok = n > 0
    if not ok.any():
        return out
    cols = np.where(ok)[0]
    pos = q * (n[ok] - 1.0)
    lo = np.floor(pos).astype(np.int64)
    hi = np.ceil(pos).astype(np.int64)
    a, b, t = xs[lo, cols], xs[hi, cols], pos - lo
    out[ok] = np.where(t < 0.5, a + t * (b - a), b - (1.0 - t) * (b - a))
    return out


def select_edg_nodes_2d(xy, Vs, Ocount, near_px, overlap_frac=0.5, max_nodes=100, min_node_frames=20,
                        quantile=0.90, nsamp=30, verbose=True, return_suppressed=False,
                        xy2=None, near2=None, min_overlap=0):
    """Length-priority NMS EDG-node selection using the 2D IMAGE persistent-distance metric (see block
    comment). `xy` is (Q, nS, 2) image px aligned to Vs' (nS, Q) time axis; ranks tracks by reliable
    length; selects the longest survivor; DELETES every still-alive track whose persistent 2D distance to
    the node -- the `quantile` (p90) of per-frame px distance over co-visible reliable frames -- is
    <= near_px AND that temporally overlaps the node by > overlap_frac of the shorter track's length.
    Only tracks reliable in > min_node_frames frames are eligible. Returns node global ids (and, if
    return_suppressed, the per-node list of deleted global ids). `near_px` is the range knob.
    Paper rule (eq:track-accept): with `xy2`/`near2` a track is ALSO suppressed when its persistent
    distance in that second space (3D world <-> 2D image) is <= near2 -- "close in 3D OR in the image";
    `min_overlap` adds the n_min floor on the number of co-visible frames."""
    Q = xy.shape[0]; T = Vs.shape[0]
    has = np.where(Ocount > int(min_node_frames))[0]
    if has.size == 0:
        return (np.zeros(0, np.int64), []) if return_suppressed else np.zeros(0, np.int64)
    idx_per, tree_per = _frame_trees_2d(xy, Vs, T)
    use2 = xy2 is not None and near2 is not None and near2 > 0
    if use2:
        idx_per2, tree_per2 = _frame_trees_2d(xy2, Vs, T)
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
            if use2 and tree_per2[t] is not None:            # candidates close in the second space too
                for h in tree_per2[t].query_ball_point(xy2[g, t], near2):
                    b = int(idx_per2[t][h])
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
            close = pdist <= near_px
            if use2:                                         # ... OR close in the second space
                d2 = np.linalg.norm(xy2[cand] - xy2[g][None], axis=2)
                d2_co = np.where(co.T, d2, np.nan)
                pdist2 = np.full(cand.size, np.inf)
                pdist2[has_ov] = np.nanquantile(d2_co[has_ov], quantile, axis=1)
                close = close | (pdist2 <= near2)
            min_len = np.minimum(Ocount[g], Ocount[cand])
            kill = cand[(n_ov >= int(min_overlap)) & (n_ov > overlap_frac * min_len) & close]
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
                              min_node_frames=20, quantile=0.90, nsamp=30, verbose=True,
                              xy2=None, near2=None, min_overlap=0):
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
                                min_node_frames, quantile, nsamp, verbose=False,
                                xy2=xy2, near2=near2, min_overlap=min_overlap)
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


def build_edg_graph(edges, xy, Ps, Vs, Ocount, thr, near_px=15.0, overlap_frac=0.5,
                    edge_hops=6, max_nodes=100, min_node_frames=20, max_edge_factor=0.0,
                    quantile=0.90, min_degree=4, verbose=True, nodes_3d=False, near_frac=0.16,
                    edge_min_max_ratio=0.5, person_id=None, min_overlap=0, legacy=False):
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
    # paper rule: suppress if close in 3D OR in the image (eq:track-accept) + the n_min overlap floor.
    # legacy=True reproduces the pre-2026-10-08 single-metric rule (certified release numbers).
    if legacy:
        sel2 = dict(xy2=None, near2=None, min_overlap=0)
    else:
        P3 = np.transpose(Ps, (1, 0, 2))
        sel2 = (dict(xy2=xy, near2=float(near_px), min_overlap=int(min_overlap)) if nodes_3d else
                dict(xy2=P3, near2=float(thr * (near_frac / 0.16)), min_overlap=int(min_overlap)))
        if verbose:
            print(f"[EDG] node rule: {'3D' if nodes_3d else '2D'} p90<= {near_sel:.4g} OR "
                  f"{'2D' if nodes_3d else '3D'} p90<= {sel2['near2']:.4g}; co-visible >= {int(min_overlap)} "
                  f"frames and > {overlap_frac:.0%} of the shorter track")
    # MULTI-BODY SEPARATION: with >1 physical body, run the SAME coverage-NMS selection once PER BODY
    # (each gets its own full max_nodes budget, so a body with fewer/shorter tracks is never starved by
    # a bigger one) instead of once globally. Tracks with unresolved person_id (-1) are excluded from
    # candidacy as NODES here (they still bind as ordinary dynamic points via the track graph).
    if person_id is not None and int(np.max(person_id, initial=-1)) > 0:
        bodies = sorted(int(p) for p in set(person_id.tolist()) if p >= 0)
        node_list = []
        for b in bodies:
            Ocount_b = np.where(person_id == b, Ocount, 0)
            nb, _ = select_edg_nodes_covering(pts_sel, Vs, Ocount_b, near_sel, overlap_frac, max_nodes,
                                              min_node_frames, quantile, verbose=verbose, **sel2)
            if verbose:
                print(f"[EDG] body {b}: {nb.size} nodes")
            node_list.append(nb)
        nodes = np.concatenate(node_list) if node_list else np.zeros(0, np.int64)
    else:
        nodes, _ = select_edg_nodes_covering(pts_sel, Vs, Ocount, near_sel, overlap_frac, max_nodes,
                                             min_node_frames, quantile, verbose=verbose, **sel2)
    if nodes.size == 0 or edges.shape[0] == 0:
        return nodes, np.zeros((0, 2), np.int64), np.zeros((0, 2), np.int64), np.zeros((0, 2), np.int64)

    # --- STEP 1: FILTER FIRST -- build the BLUE (all-pairs, non-hop) graph over EDG nodes BEFORE any hop
    # restriction: a pair passes iff (a) it is co-visible at least once, (b) its robust p90 co-visible 3D
    # distance does NOT exceed max_edge_factor*scale_node, and (c) it does NOT vary a lot (min co-visible
    # distance >= edge_min_max_ratio * max). scale_node (typical node spacing) is now the median finite
    # p90 over ALL co-visible node pairs (there is no separate hop-candidate set to derive it from
    # anymore). Cross-body pairs (multi-body) are excluded outright (never co-visible for this purpose),
    # so blue -- and everything built from it below -- can never bridge two different people. ---
    edge_hops = int(edge_hops) if edge_hops > 0 else 6
    K = nodes.size
    node_person = person_id[nodes] if person_id is not None else None
    cross_body = None
    if node_person is not None and int(np.max(node_person, initial=-1)) > 0:
        cross_body = ((node_person[:, None] != node_person[None, :])
                      & (node_person[:, None] >= 0) & (node_person[None, :] >= 0))
    P90 = np.full((K, K), np.inf); Dmn = np.full((K, K), np.inf); Dmx = np.full((K, K), np.inf)
    Vn = [Vs[:, nodes[i]] for i in range(K)]
    for i in tqdm(range(K), desc="blue pairs", disable=not verbose):
        for j in range(i + 1, K):
            if cross_body is not None and cross_body[i, j]:
                continue
            both = Vn[i] & Vn[j]
            if both.any():
                dd = np.linalg.norm(Ps[both, nodes[i]] - Ps[both, nodes[j]], axis=1)
                P90[i, j] = P90[j, i] = np.percentile(dd, 90)
                Dmn[i, j] = Dmn[j, i] = float(dd.min()); Dmx[i, j] = Dmx[j, i] = float(dd.max())
    _fin = np.isfinite(P90)
    scale_node = float(np.median(P90[_fin])) if _fin.any() else float(thr)
    _do_cap = bool(max_edge_factor and max_edge_factor > 0)
    _do_var = bool(edge_min_max_ratio and edge_min_max_ratio > 0)
    cap = float(max_edge_factor) * scale_node if _do_cap else np.inf
    too_far = P90 > cap
    too_variable = (Dmn < float(edge_min_max_ratio) * Dmx) if _do_var else np.zeros_like(too_far)
    blue_adj = _fin & ~too_far & ~too_variable
    np.fill_diagonal(blue_adj, False)
    cap_filtered_edges = np.zeros((0, 2), np.int64)   # kept only for return-arity: filtering now happens
                                                        # BEFORE hop selection, so nothing is "dropped from
                                                        # a hop-candidate set" anymore -- always empty.
    if verbose:
        n_possible = K * (K - 1) // 2
        print(f"[EDG] blue filter (node-spacing {scale_node:.4f}; p90<= {cap:.4f}, "
              f"min>= {edge_min_max_ratio:g}max): {int(blue_adj.sum()) // 2}/{max(n_possible, 1)} "
              f"all-pairs connections pass (BEFORE hop restriction)")

    # --- TRACK-GRAPH HOP DISTANCE (computed ONCE, on the RAW dynamic-point graph `edges` -- NOT on the
    # blue graph): this is a genuine topological-proximity signal, independent of the metric filter, used
    # two ways below. BUG FIXED HERE: an earlier version computed hop-distance ON the blue graph itself,
    # which reintroduces exactly the "chain of good hops connects far-apart nodes" failure the metric
    # filter exists to prevent -- verified on Jumping that 896/1609 (56%!) of the resulting red edges were
    # NOT directly blue at all, just reachable via a multi-hop chain through OTHER blue pairs (e.g. gid
    # 6962<->157: P90=1.47 > cap=1.35, correctly blue=False, but still became a "red" edge). Hop distance
    # on the FINE raw track graph does not have this problem: it is independent of blue_adj, so
    # RED = blue_adj AND hop<=edge_hops requires the DIRECT pair to pass the metric test too. ---
    Qn = int(Vs.shape[1])
    Hn = np.where(np.eye(K, dtype=bool), 0.0, np.inf)   # (K,K) hop distance on the raw track graph
    if edges.shape[0]:
        ii = np.concatenate([edges[:, 0], edges[:, 1]])
        jj = np.concatenate([edges[:, 1], edges[:, 0]])
        A = csr_matrix((np.ones(ii.size, np.float32), (ii, jj)), shape=(Qn, Qn))
        Hn = dijkstra(A, directed=False, indices=nodes, unweighted=True)[:, nodes]   # (K,K), inf if disconnected
    reachable = np.isfinite(Hn)          # ANY finite hop count -- used by the fallback (STEP 3) below
    hop_close = reachable & (Hn <= float(edge_hops)) & (Hn > 0)   # <= edge_hops -- used by STEP 2 below
    if verbose:
        n_pairs = K * (K - 1) // 2
        print(f"[EDG] track-graph hops: {int(np.triu(reachable, 1).sum())}/{max(n_pairs, 1)} pairs "
              f"finite-hop-connected, {int(np.triu(hop_close, 1).sum())} within {edge_hops} hop(s)")

    # --- STEP 2: RED edges = blue pairs (direct 3D distance + variability) that are ALSO within
    # edge_hops on the RAW track graph. Both conditions on the SAME pair -- no chaining through blue. ---
    final_adj = blue_adj & hop_close
    iu, ju = np.triu_indices(K, k=1)
    sel = final_adj[iu, ju]
    a, b = nodes[iu[sel]], nodes[ju[sel]]
    edg_edges = np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1).astype(np.int64)
    if verbose:
        print(f"[EDG] {edg_edges.shape[0]} red edges (blue AND within {edge_hops} hop(s) on the track graph)")

    # --- STEP 3: MIN-DEGREE FALLBACK, drawing ONLY from the BLUE graph AND requiring finite track-graph
    # reachability (matches "fallback after every filter": it can never add a filter-failing OR
    # disconnected pair -- a node with too few such neighbours simply stays under-connected). ---
    if min_degree and min_degree > 0 and K > 1:
        edg_edges = _augment_edg_min_degree(edg_edges, nodes, P90, blue_adj & reachable, int(min_degree), verbose)

    # --- DEBUG (BLUE viewer edges): blue-approved pairs NOT included in the final graph above (beyond
    # --edg-edge-hops on the blue graph, or not needed by the fallback) -- i.e. quality-passing
    # connections the hop restriction is still leaving out. NOT used to build edg_edges. ---
    nonhop_debug_edges = np.zeros((0, 2), np.int64)
    if blue_adj.any():
        iu, ju = np.triu_indices(K, k=1)
        sel = blue_adj[iu, ju]
        a, b = nodes[iu[sel]], nodes[ju[sel]]
        blue_pairs = np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1).astype(np.int64)
        if blue_pairs.size and edg_edges.size:
            final_set = set(map(tuple, edg_edges.tolist()))
            keep = np.array([tuple(p) not in final_set for p in blue_pairs.tolist()], dtype=bool)
            nonhop_debug_edges = blue_pairs[keep]
        else:
            nonhop_debug_edges = blue_pairs
        if verbose:
            print(f"[EDG] debug non-hop: {nonhop_debug_edges.shape[0]}/{blue_pairs.shape[0] if blue_pairs.size else 0} "
                  f"blue-approved pairs excluded from the final graph (beyond --edg-edge-hops / not needed "
                  f"by fallback) -- viewer-only, not part of the graph")

    # --- FINAL WITHIN-BODY OUTLIER DELETION: a node whose canonical position is still an extreme
    # outlier relative to its OWN body's other nodes (>4x that body's typical inter-node spacing) is a
    # genuine placement defect (e.g. an occlusion-induced track mis-association), not a multi-body
    # artifact (already structurally prevented above) -- DELETE it as an EDG node entirely (and any
    # edges referencing it) instead of leaving it in the graph. ---
    if person_id is not None and nodes.size > 2:
        cnt_n = Vs[:, nodes].sum(0)
        canon_n = np.full((nodes.size, Ps.shape[-1]), np.nan)
        has_n = cnt_n > 0
        canon_n[has_n] = np.where(Vs[:, nodes][..., None], Ps[:, nodes], 0.0).sum(0)[has_n] / cnt_n[has_n, None]
        node_body = person_id[nodes]
        keep_node = np.ones(nodes.size, bool)
        for b in sorted(int(x) for x in set(node_body.tolist()) if x >= 0):
            sel = np.where(node_body == b)[0]
            fin = sel[np.isfinite(canon_n[sel]).all(1)]
            if fin.size < 3:
                continue
            med = np.median(canon_n[fin], axis=0)
            d = np.linalg.norm(canon_n[fin] - med[None], axis=1)
            typical = float(np.median(d))
            if typical <= 1e-9:
                continue
            bad_n = fin[d > 4.0 * typical]
            keep_node[bad_n] = False
        n_deleted = int((~keep_node).sum())
        if n_deleted:
            deleted_gids = nodes[~keep_node]
            if verbose:
                print(f"[EDG] within-body outlier deletion: removed {n_deleted}/{nodes.size} nodes "
                      f"whose own-body canonical position is >4x that body's typical node spacing "
                      f"(gids={sorted(int(g) for g in deleted_gids)})")
            nodes = nodes[keep_node]
            if edg_edges.shape[0]:
                keep_e = ~(np.isin(edg_edges[:, 0], deleted_gids) | np.isin(edg_edges[:, 1], deleted_gids))
                edg_edges = edg_edges[keep_e]

    if verbose:
        print(f"[EDG] {nodes.size} nodes, {edg_edges.shape[0]} edges (hop distance <= {edge_hops}"
              + (f", min-degree {min_degree}" if min_degree and min_degree > 0 else "") + ")")
    return nodes, edg_edges, cap_filtered_edges, nonhop_debug_edges


def _augment_edg_min_degree(edg_edges, nodes, P90, blue_adj, min_degree, verbose=True):
    """Add edges so every EDG node has >= min_degree neighbours, drawing ONLY from blue_adj (node pairs
    that already passed the too-far/variability filter AND the track-graph reachability safeguard in
    build_edg_graph -- this fallback NEVER adds a filter-failing, disconnected, or cross-body pair; a
    node with fewer than min_degree available candidates simply stays under-connected). Neighbours are
    added nearest-first by P90 (robust co-visible-frame 3D distance). P90/blue_adj are the (K,K) matrices
    already computed over `nodes` by the caller (the caller passes blue_adj & reachable, and cross-body
    pairs are already excluded from blue_adj too)."""
    from collections import defaultdict
    K = nodes.size
    adj = defaultdict(set)
    loc = {int(g): i for i, g in enumerate(nodes)}
    for a, b in edg_edges.tolist():
        adj[loc[int(a)]].add(loc[int(b)]); adj[loc[int(b)]].add(loc[int(a)])
    new_pairs = set()
    for i in range(K):
        need = min_degree - len(adj[i])
        if need <= 0:
            continue
        order = sorted((j for j in range(K) if j != i and blue_adj[i, j] and j not in adj[i]),
                       key=lambda j: float(P90[i, j]))
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
            print(f"[EDG] min-degree {min_degree}: +{len(new_pairs)} blue-only kNN edges for under-connected nodes")
    return edg_edges


def build_dyn_to_edg_edges(edges, Q, edg_nodes, Ps, Vs, Ocount, thr, hop_lt=2, min_conn=2,
                           max_factor=3.0, cand_k=8, quantile=90, verbose=True, hop_limit=20,
                           max_dist_factor=3.0, min_max_ratio=0.5, person_id=None, legacy=False):
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
        if legacy:                                           # legacy: also the cand_k hop-nearest nodes
            kk = int(min(max(int(cand_k), int(min_conn)), K))
            part = np.argpartition(sub, min(kk, K - 1), axis=0)[:kk]      # (kk, n_dyn) geodesic-nearest
            for c, q in enumerate(dyn_ids.tolist()):
                for nl in part[:, c].tolist():
                    if np.isfinite(sub[nl, c]):
                        cand[q].add(int(nl))
        if max_hop >= 1:                                     # also everything within < hop_lt hops
            ni, qi = np.where((Dn >= 1) & (Dn <= max_hop) & np.isfinite(Dn))
            keep = ~node_mask[qi]
            for nl, q in zip(ni[keep].tolist(), qi[keep].tolist()):
                cand[q].add(int(nl))
    else:                                                    # no track graph: every node is a candidate
        for q in dyn_ids.tolist():
            cand[q] = set(range(K))

    # MULTI-BODY SEPARATION (defense in depth): the graph-reachable candidate pool above should already
    # respect body boundaries (the underlying track graph never has cross-body edges), but the no-graph
    # fallback just above does not -- so re-filter explicitly by body here too. Unresolved person_id (-1,
    # on either the point or the node) is a wildcard (kept).
    if person_id is not None and int(np.max(person_id, initial=-1)) > 0:
        node_body = person_id[edg_nodes]                     # (K,)
        for q in list(cand.keys()):
            pq = int(person_id[q])
            if pq < 0:
                continue
            cand[q] = {nl for nl in cand[q] if node_body[nl] < 0 or node_body[nl] == pq}

    # --- robust p<quantile> + MIN + MAX per-frame distance for each candidate pair, batched by node ---
    node_pts = defaultdict(list)
    for q, nls in cand.items():
        for nl in nls:
            node_pts[nl].append(q)
    stats = defaultdict(dict)                                 # q -> {node_gid: (robust, dmin, dmax)}
    for nl, pts in tqdm(node_pts.items(), desc="[EDG] dyn->EDG dist", disable=not verbose):
        pts = np.asarray(pts, np.int64)
        n_gid = int(edg_nodes[nl])
        covis = Vs[:, pts] & Vs[:, n_gid][:, None]           # (T, n_pts)
        d = np.linalg.norm(Ps[:, pts, :] - Ps[:, n_gid][:, None, :], axis=2)   # (T, n_pts)
        d = np.where(covis, d, np.nan)
        with np.errstate(all="ignore"):
            rob = np.nanpercentile(d, quantile, axis=0)      # (n_pts,) robust over co-visible frames
            dmn = np.nanmin(d, axis=0)
            dmx = np.nanmax(d, axis=0)
        for pi, q in enumerate(pts.tolist()):
            if np.isfinite(rob[pi]):
                stats[q][n_gid] = (float(rob[pi]), float(dmn[pi]), float(dmx[pi]))

    # --- adaptive cap + HARD stability filters. "far" is relative to node spacing (scale = median
    # nearest-node robust distance). A link needs (a) p90 <= max_factor*scale (soft cap, has a skinnable
    # fallback), AND (b) MAX co-visible distance <= max_dist_factor*scale, AND (c) it must not vary a lot
    # (min >= min_max_ratio*max, applied only when the point ever gets far, dmax > scale). (b),(c) are HARD:
    # a point whose every candidate fails them gets NO link. ---
    nearest = np.array([min(v[0] for v in nd.values()) for nd in stats.values() if nd], dtype=np.float64)
    scale = float(np.median(nearest)) if nearest.size else float(thr)
    cap = float(max_factor) * scale
    max_cap = float(max_dist_factor) * scale

    def _unstable(s):                                        # None if OK, else reason it is rejected
        _r, dmn, dmx = s
        if max_dist_factor > 0 and dmx > max_cap:
            return "far"
        if min_max_ratio > 0 and dmx > scale and dmn < float(min_max_ratio) * dmx:
            return "vary"
        return None

    conn = defaultdict(set)
    n_capped = n_far = n_vary = n_dropped = 0
    for q, nd in stats.items():
        ranked = sorted(nd.items(), key=lambda kv: kv[1][0])  # (node_gid, (rob,dmin,dmax)) nearest first
        stable = []
        for g, s in ranked:
            why = _unstable(s)
            if why == "far":
                n_far += 1
            elif why == "vary":
                n_vary += 1
            else:
                stable.append((g, s))
        under = [(g, s[0]) for g, s in stable if s[0] <= cap]
        n_capped += max(0, len(stable) - len(under))
        if under:
            conn[q].update(g for g, _ in under[:int(min_conn)])
        elif stable:
            conn[q].add(stable[0][0])                         # skinnable fallback: closest that passes (b)&(c)
        else:
            n_dropped += 1                                    # every candidate far / varying -> no link

    n_fallback = 0
    if not legacy:
        # paper rule: a dynamic point with NO node within the hop limit (or none surviving the filters)
        # falls back to its single nearest node by p<quantile> co-visible 3D distance over ALL nodes.
        orphans = np.array([q for q in dyn_ids.tolist() if not conn.get(q)], np.int64)
        if orphans.size:
            best_d = np.full(orphans.size, np.inf); best_n = np.full(orphans.size, -1, np.int64)
            for nl in range(K):
                n_gid = int(edg_nodes[nl])
                covis = Vs[:, orphans] & Vs[:, n_gid][:, None]
                d = np.where(covis, np.linalg.norm(Ps[:, orphans, :] - Ps[:, n_gid][:, None, :], axis=2), np.nan)
                with np.errstate(all="ignore"):
                    rob = np.nanpercentile(d, quantile, axis=0)
                better = np.isfinite(rob) & (rob < best_d)
                best_d[better] = rob[better]; best_n[better] = n_gid
            for q, n_gid in zip(orphans.tolist(), best_n.tolist()):
                if n_gid >= 0:
                    conn[q].add(int(n_gid)); n_fallback += 1
            n_dropped = int((best_n < 0).sum())             # only never-co-visible points stay unlinked
    if not conn:
        return np.zeros((0, 2), np.int64)
    trk = np.fromiter((t for t, nds in conn.items() for _ in nds), dtype=np.int64)
    nod = np.fromiter((nd for nds in conn.values() for nd in nds), dtype=np.int64)
    und = np.unique(np.stack([trk, nod], axis=1), axis=0)     # (E,2) = (dynamic track, edg node)
    if verbose:
        deg = np.array([len(v) for v in conn.values()])
        print(f"[EDG] dyn->EDG: {und.shape[0]} orange edges over {len(conn)} dynamic points "
              f"({min_conn} nearest by robust p{quantile} co-visible dist; scale {scale:.4f}, p90-cap "
              f"{cap:.4f} (x{max_factor:g}), max-cap {max_cap:.4f} (x{max_dist_factor:g})). Dropped links: "
              f"{n_capped} over p90-cap, {n_far} max-dist too large, {n_vary} varying(min<{min_max_ratio:g}max); "
              f"{n_fallback} points fell back to their nearest node; {n_dropped} points left unlinked; nodes/pt min={int(deg.min())} med={int(np.median(deg))} "
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
    markers regardless of point size). Rebuilt each frame from the current node positions.
    color: either a single RGB triple (uniform) or an (N,3) array, one colour per position (e.g. one
    per physical body)."""
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
    color = np.asarray(color, dtype=np.float64)
    if color.ndim == 1:
        m.paint_uniform_color(color)
    else:
        m.vertex_colors = o3d.utility.Vector3dVector(np.repeat(color, nv, axis=0))
    m.compute_vertex_normals()
    return m


# ----------------------------------------------------------------------------- visualization
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--no-scale", action="store_true", help="do not apply da3 per-frame scale")
    ap.add_argument("--edge-thresh", type=float, default=dataset_config.get_float("EDG_EDGE_THRESH", 0.0),
                    help="absolute edge_distance_threshold; 0 = auto = edge-thresh-mult * median NN dist")
    ap.add_argument("--edge-thresh-mult", type=float,
                    default=dataset_config.get_float("EDG_EDGE_THRESH_MULT", 5.0),
                    help="x in 'edge threshold = x * median NN dist' (config EDG_EDGE_THRESH_MULT)")
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
    ap.add_argument("--min-overlap-frames", type=int,
                    default=dataset_config.get_int("EDG_MIN_OVERLAP_FRAMES", 3),
                    help="an edge needs >= this many co-visible frames (config EDG_MIN_OVERLAP_FRAMES)")
    ap.add_argument("--min-overlap-ratio", type=float, default=0.20)
    ap.add_argument("--distance-quantile", type=float,
                    default=dataset_config.get_float("EDG_EDGE_DIST_QUANTILE", 0.95),
                    help="edge accepted if this quantile (x%%) of the two tracks' per-frame distances "
                         "<= threshold (config EDG_EDGE_DIST_QUANTILE)")
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
    ap.add_argument("--edg-near-frac", type=float, default=dataset_config.get_float("EDG_NEAR_FRAC", 0.16),
                    help="ADAPTIVE radius fraction for --edg-near-px 0: near_px = frac * sqrt(median "
                         "dynamic-mask area px^2). 0.16 reproduces the tuned 15px on paper-windmill.")
    ap.add_argument("--edg-dist-quantile", type=float, default=0.90,
                    help="quantile for the EDG node-selection persistent 2D distance (p90 = robust near-max; "
                         "matches build_edg's edge-builder distance_quantile).")
    ap.add_argument("--edg-overlap-frac", type=float, default=0.5,
                    help="EDG node selection: delete a near track only if it temporally overlaps the chosen "
                         "node by MORE than this fraction of the shorter track's length (0.5 = >50%%).")
    ap.add_argument("--edg-edge-hops", type=int,
                    default=dataset_config.get_int("EDG_EDGE_HOPS", 4),
                    help="connect two EDG nodes when their hop distance on the track graph is <= this "
                         "(config EDG_EDGE_HOPS)")
    ap.add_argument("--edg-max-edge-factor", type=float,
                    default=dataset_config.get_float("EDG_MAX_EDGE_FACTOR", 2.0),
                    help="drop EDG edges whose robust 3D distance (p90 over co-visible frames) exceeds this "
                         "factor x the edge-distance threshold -- prevents hop-chains from connecting "
                         "far-apart nodes (0 = off; ~2.0 recommended) (config EDG_MAX_EDGE_FACTOR)")
    ap.add_argument("--edg-min-max-ratio", type=float,
                    default=dataset_config.get_float("EDG_MIN_MAX_RATIO", 0.3),
                    help="HARD: drop an EDG node-node edge whose distance VARIES a lot (min co-visible dist < "
                         "this fraction of max, only when max>threshold). --edg-min-degree still guaranteed. "
                         "0=off. (config EDG_MIN_MAX_RATIO)")
    ap.add_argument("--edg-max-nodes", type=int,
                    default=dataset_config.get_int("EDG_MAX_NODES", 100),
                    help="EDG node BUDGET: the selection radius is auto-tuned so FULL coverage of all "
                         "eligible tracks fits within this many nodes (select_edg_nodes_covering) "
                         "(config EDG_MAX_NODES)")
    ap.add_argument("--edg-min-frames", type=int,
                    default=dataset_config.get_int("EDG_MIN_FRAMES", 20),
                    help="EDG node must be reliably observed in MORE than this many frames (length floor). "
                         "MUST be < number of frames T, else 0 nodes. (config EDG_MIN_FRAMES)")
    ap.add_argument("--legacy-topology", action=argparse.BooleanOptionalAction,
                    default=dataset_config.get_bool("EDG_LEGACY_TOPOLOGY", False),
                    help="pre-2026-10-08 node/skinning rules (single-metric node suppression without the "
                         "n_min floor; skinning candidates incl. the 8 hop-nearest nodes, no nearest-node "
                         "fallback). Reproduces the certified release topology. (config EDG_LEGACY_TOPOLOGY)")
    ap.add_argument("--edg-min-degree", type=int, default=3,
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
    ap.add_argument("--dyn-edg-max-dist-factor", type=float,
                    default=dataset_config.get_float("DYN_EDG_MAX_DIST_FACTOR", 3.0),
                    help="HARD: drop a dynamic->EDG link whose MAXIMUM co-visible-frame distance exceeds "
                         "this factor x node spacing -- rejects links far in even a few frames (not just on "
                         "average). 0=off. (config DYN_EDG_MAX_DIST_FACTOR)")
    ap.add_argument("--dyn-edg-min-max-ratio", type=float, default=0.5,
                    help="HARD: drop a dynamic->EDG link whose distance VARIES a lot: min co-visible distance "
                         "< this fraction of the max (only when the point ever gets far, dmax>spacing) -- "
                         "catches points sliding toward/away from a node (non-rigid). 0=off.")
    ap.add_argument("--topology-out", type=str, default="",
                    help="save the EDG + dynamic-edge topology npz here (default output_dir/edg_topology.npz)")
    ap.add_argument("--save-npz", type=str, default="", help="also save edges to this .npz path")
    ap.add_argument("--tracks", default="", help="explicit tracks npz (name under outputs/ or a full path); "
                                                 "overrides --merged/--smoothed. Use the lifted file "
                                                 "(tracks_dynamic_merged_lift.npz) when the merge skipped "
                                                 "its smoothing stage (--no-optimize).")
    ap.add_argument("--merged", action=argparse.BooleanOptionalAction, default=True,
                    help="use MERGED tracks (default on; --no-merged uses raw tracks_dynamic_only)")
    ap.add_argument("--smoothed", action=argparse.BooleanOptionalAction, default=True,
                    help="use SMOOTHED tracks from optimize_dynamic_tracks.py (default on; --no-smoothed)")
    ap.add_argument("--gt-depth", action="store_true",
                    help="EDG node-selection distance in 3D world (use with GT/LiDAR/completed depth); default 2D image (DA3)")
    args = ap.parse_args()

    import open3d as o3d

    # merged (which base tracks) and smoothed (whether optimized) are ORTHOGONAL; both default ON.
    base = "merged" if args.merged else "only"
    name = f"tracks_dynamic_{base}_smoothed.npz" if args.smoothed else f"tracks_dynamic_{base}.npz"
    if args.tracks:                                                   # explicit file wins
        name = args.tracks
        _tp = args.tracks if "/" in args.tracks else str(Path(utils.definitions.output_dir) / args.tracks)
    else:
        _tp = str(Path(utils.definitions.output_dir) / name)
    if not Path(_tp).exists():
        raise FileNotFoundError(
            f"{name} not found. Generate the smoothed tracks first:\n"
            f"    python optimize_dynamic_tracks.py --input tracks_dynamic_{base}.npz\n"
            f"  (or run  build_edg.py --no-smoothed  to use un-smoothed tracks_dynamic_{base}.npz)")
    print(f"[tracks] EDG on {name}  |  node-selection distance: {'3D world (--gt-depth)' if args.gt_depth else '2D image'}")
    D = _load_all(tracks_path=_tp)
    D["dyn_mask"] = _dilate_dyn_mask(D, args.mask_dilate_px)  # grow dynamic reliability region by N px
    # multi-body (person_id) separation DISABLED -- the 2D-mask-based labeling proved unreliable under
    # fast/occlusion-heavy motion (Jumping), and the blue filter + track-graph hop restriction already
    # reject bad cross-part connections on their own merits (distance/variability/topology), without
    # needing an explicit, error-prone body label. Every consumer below already guards on
    # `person_id is not None`, so simply never setting it cleanly disables the whole mechanism -- the EDG
    # graph is now a single uniform (red) graph regardless of how many people are in the scene.
    if args.disable_depth_move_filter:                       # undo depth + large-move filters (needs depth)
        _load_depth(D)
        _disable_depth_move_filter(D)
    s = _frame_scale(D["rt"], D["T"], args.no_scale)
    T = D["T"]
    cfg = EdgeConfig(
        max_sampled_frames=args.max_sampled_frames,
        candidate_k=args.candidate_k, min_overlap_frames=args.min_overlap_frames,
        min_overlap_ratio=args.min_overlap_ratio, distance_quantile=args.distance_quantile,
        edge_distance_threshold=args.edge_thresh, edge_thresh_mult=args.edge_thresh_mult,
        max_neighbors=args.max_neighbors,
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
        edg_edges_capfiltered = np.zeros((0, 2), np.int64)
        edg_edges_nonhop_debug = np.zeros((0, 2), np.int64)
    else:
        aux = D["_edg_aux"]
        xy_s = D["tracks_xy"][0][:, aux["sampled"]].astype(np.float64)   # 2D px aligned to Vs' time axis
        near_px_eff = (float(args.edg_near_px) if float(args.edg_near_px) > 0
                       else _dyn_region_near_px(D, args.edg_near_frac))
        edg_nodes, edg_edges, edg_edges_capfiltered, edg_edges_nonhop_debug = build_edg_graph(
            edges, xy_s, aux["Ps"], aux["Vs"], aux["Ocount"], aux["thr"],
            near_px=near_px_eff, overlap_frac=args.edg_overlap_frac, nodes_3d=args.gt_depth, near_frac=args.edg_near_frac,
            edge_hops=args.edg_edge_hops, max_nodes=args.edg_max_nodes,
            min_node_frames=args.edg_min_frames, max_edge_factor=args.edg_max_edge_factor,
            quantile=args.edg_dist_quantile, min_degree=args.edg_min_degree,
            edge_min_max_ratio=args.edg_min_max_ratio,
            person_id=D.get("person_id"), min_overlap=args.min_overlap_frames,
            legacy=args.legacy_topology)

    # dynamic-point -> EDG node edges (ORANGE): nodes within < dyn_edg_hop_lt hops, unioned with each
    # point's dyn_edg_min nearest nodes AT ITS FIRST APPEARING FRAME (every appearing point gets >= that many).
    _aux = D["_edg_aux"]
    dyn_edg_edges = (build_dyn_to_edg_edges(
                        edges, Q, edg_nodes, _aux["Ps"], _aux["Vs"], _aux["Ocount"], _aux["thr"],
                        hop_lt=args.dyn_edg_hop_lt, min_conn=args.dyn_edg_min,
                        max_factor=args.dyn_edg_max_factor,
                        max_dist_factor=args.dyn_edg_max_dist_factor,
                        min_max_ratio=args.dyn_edg_min_max_ratio, person_id=D.get("person_id"),
                        legacy=args.legacy_topology)
                     if edg_nodes.size else np.zeros((0, 2), np.int64))

    # save the topology result: EDG nodes/edges + dynamic-point (track) edges + dynamic->EDG edges
    topo_out = args.topology_out or str(Path(utils.definitions.output_dir) / "edg_topology.npz")
    np.savez_compressed(topo_out, edg_nodes=edg_nodes, edg_edges=edg_edges,
                        track_edges=edges, track_scores=scores, dyn_edg_edges=dyn_edg_edges,
                        dyn_edg_hop_lt=int(args.dyn_edg_hop_lt), dyn_edg_min_conn=int(args.dyn_edg_min),
                        n_tracks=int(Q),                              # track-id space stamp (merged vs raw)
                        person_id=D.get("person_id", np.zeros(Q, np.int64)))  # per-track physical-body id
                                                                       # (same track-index space as Q above;
                                                                       # -1 = unresolved). Lets downstream
                                                                       # consumers (train_4dgs_edg.py) apply
                                                                       # a final within-body outlier check on
                                                                       # the ACTUAL solved positions.
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
    C_CAPFILT = [0.55, 0.27, 0.07]           # brown  = EDG edges DROPPED by the cap/stability filter
    C_NONHOP = [0.15, 0.35, 0.95]            # blue   = debug: all-pairs (non-hop) links failing the
                                              #          too-far + variability test (shown with capfilt_line
                                              #          when dynamic points are hidden, D-toggle)
    C_BODY = [                               # MULTI-BODY: one distinct colour per physical body, used for
        [0.90, 0.10, 0.10],                  # the EDG (red) nodes/edges INSTEAD of uniform C_EDG whenever
        [0.10, 0.65, 0.10],                  # person_id has > 1 body -- makes the "should be N separate
        [0.10, 0.35, 0.95],                  # EDG graphs" claim directly checkable by eye (single-body
        [0.95, 0.60, 0.05],                  # scenes are completely unaffected, still plain red).
        [0.60, 0.10, 0.75],
        [0.95, 0.90, 0.10],
        [0.10, 0.80, 0.80],
        [0.85, 0.30, 0.55],
    ]
    person_id_all = D.get("person_id")
    multi_body = person_id_all is not None and int(np.max(person_id_all, initial=-1)) > 0

    # EDG node marker radius from the scene scale at the start frame
    P0 = _valid_at(D, s, int(np.clip(args.start, 0, T - 1)))[0]
    fin0 = np.isfinite(P0).all(axis=1)
    diag = float(np.linalg.norm(P0[fin0].max(0) - P0[fin0].min(0))) if fin0.any() else 1.0
    node_r = max(1e-4, 0.006 * diag)

    state = {"idx": int(np.clip(args.start, 0, T - 1)), "show_dyn": True, "show_nonhop": False}
    pcd = o3d.geometry.PointCloud()
    lines = o3d.geometry.LineSet()
    edg_line = o3d.geometry.LineSet()                    # red EDG edges
    dyn_edg_line = o3d.geometry.LineSet()                # orange dynamic-point -> EDG node edges
    capfilt_line = o3d.geometry.LineSet()                # brown edges dropped by the cap/stability filter
    nonhop_line = o3d.geometry.LineSet()                 # blue debug: all-pairs (non-hop) bad links
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
            if state["show_dyn"]:
                pcd.points = o3d.utility.Vector3dVector(P[idx])
                pcd.paint_uniform_color(C_DYN)       # ALL dynamic points -> purple
            else:
                pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            n_valid = int((st == 0).sum()); n_occ = int((st == 1).sum())
            n_off = int((st == 2).sum()); n_pruned = int((st == 3).sum())
        else:
            P, v = _valid_at(D, s, frame)
            if pruned is not None:
                v = v & ~pruned                      # pruned tracks are hidden in --valid-only
            pts = P[v]
            if state["show_dyn"]:
                pcd.points = o3d.utility.Vector3dVector(pts)
                if len(pts) > 0:
                    pcd.paint_uniform_color(C_DYN)   # ALL dynamic points -> purple
            else:
                pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
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
            if state["show_dyn"]:
                lines.lines = o3d.utility.Vector2iVector(edges[em].astype(np.int32) if em.any() else np.zeros((0, 2), np.int32))
                lines.paint_uniform_color([0.1, 0.9, 0.9])  # cyan edges
            else:
                lines.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), np.int32))
            n_e = int(em.sum())

        # --- coarse EDG deformation graph: node spheres + edges, drawn where endpoints reliable.
        # RED (uniform) for single-body scenes; one colour PER PHYSICAL BODY (C_BODY) when multi_body,
        # so "should be N separate EDG graphs" is directly checkable by eye. ---
        n_en = n_ee = 0
        if edg_nodes.size:
            nm = v[edg_nodes]
            if multi_body:
                nb = person_id_all[edg_nodes[nm]]
                node_cols = np.array([C_BODY[int(b) % len(C_BODY)] if b >= 0 else C_EDG for b in nb.tolist()])
                sm = _edg_sphere_mesh(P_safe[edg_nodes[nm]], node_r, node_cols)
            else:
                sm = _edg_sphere_mesh(P_safe[edg_nodes[nm]], node_r, C_EDG)
            edg_mesh.vertices = sm.vertices
            edg_mesh.triangles = sm.triangles
            edg_mesh.vertex_colors = sm.vertex_colors
            edg_mesh.vertex_normals = sm.vertex_normals
            n_en = int(nm.sum())
        if edg_edges.size:
            eem = v[edg_edges[:, 0]] & v[edg_edges[:, 1]]
            kept = edg_edges[eem] if eem.any() else np.zeros((0, 2), np.int64)
            edg_line.points = o3d.utility.Vector3dVector(P_safe)
            edg_line.lines = o3d.utility.Vector2iVector(kept.astype(np.int32))
            if multi_body and kept.shape[0]:
                eb = person_id_all[kept[:, 0]]                   # both endpoints share a body (0 cross-body)
                edge_cols = np.array([C_BODY[int(b) % len(C_BODY)] if b >= 0 else C_EDG for b in eb.tolist()])
                edg_line.colors = o3d.utility.Vector3dVector(edge_cols)
            else:
                edg_line.paint_uniform_color(C_EDG)
            n_ee = int(eem.sum())

        # --- dynamic-point -> EDG node edges (ORANGE), drawn where both endpoints reliable this frame ---
        n_de = 0
        if dyn_edg_edges.size:
            dem = v[dyn_edg_edges[:, 0]] & v[dyn_edg_edges[:, 1]]
            dyn_edg_line.points = o3d.utility.Vector3dVector(P_safe)
            if state["show_dyn"]:
                dyn_edg_line.lines = o3d.utility.Vector2iVector(
                    dyn_edg_edges[dem].astype(np.int32) if dem.any() else np.zeros((0, 2), np.int32))
                dyn_edg_line.paint_uniform_color(C_DYN2EDG)
            else:
                dyn_edg_line.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), np.int32))
            n_de = int(dem.sum())

        # --- EDG edges DROPPED by the cap/stability filter (BROWN): drawn so you can visually check
        # whether the filter is removing edges that were actually necessary (e.g. genuine long-but-rigid
        # connections vs spurious hop-chain shortcuts). ---
        n_cf = 0
        if edg_edges_capfiltered.size:
            cfm = v[edg_edges_capfiltered[:, 0]] & v[edg_edges_capfiltered[:, 1]]
            capfilt_line.points = o3d.utility.Vector3dVector(P_safe)
            if state["show_dyn"]:
                capfilt_line.lines = o3d.utility.Vector2iVector(
                    edg_edges_capfiltered[cfm].astype(np.int32) if cfm.any() else np.zeros((0, 2), np.int32))
                capfilt_line.paint_uniform_color(C_CAPFILT)
            else:
                capfilt_line.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), np.int32))
            n_cf = int(cfm.sum())

        # --- DEBUG all-pairs (non-hop) surviving links (BLUE): OFF by default, independent of the D
        # (dynamic-points) toggle -- press N to show/hide. When on: every rigid, plausible EDG-node pair
        # the hop restriction left out of the real graph (red). ---
        n_nh = 0
        if edg_edges_nonhop_debug.size:
            nhm = v[edg_edges_nonhop_debug[:, 0]] & v[edg_edges_nonhop_debug[:, 1]]
            nonhop_line.points = o3d.utility.Vector3dVector(P_safe)
            if state["show_nonhop"]:
                nonhop_line.lines = o3d.utility.Vector2iVector(
                    edg_edges_nonhop_debug[nhm].astype(np.int32) if nhm.any() else np.zeros((0, 2), np.int32))
                nonhop_line.paint_uniform_color(C_NONHOP)
            else:
                nonhop_line.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), np.int32))
            n_nh = int(nhm.sum())
        return n_valid, n_occ, n_off, n_pruned, n_e, n_hidden, n_en, n_ee, n_de, n_cf, n_nh

    def refresh(vis, frame):
        n_valid, n_occ, n_off, n_pruned, n_e, n_hidden, n_en, n_ee, n_de, n_cf, n_nh = _fill(frame)
        for g in (pcd, lines, edg_line, dyn_edg_line, capfilt_line, nonhop_line, edg_mesh):
            vis.update_geometry(g)
        hid = f"  hidden(over-stretched)={n_hidden}" if args.max_draw_stretch > 0 else ""
        extra = (f"  hidden: occluded={n_occ} off-mask={n_off}" if show_all else "")
        edg = f"  EDG(red): {n_en} nodes {n_ee} edges" if edg_nodes.size else ""
        orn = f"  dyn->EDG(orange): {n_de}" if dyn_edg_edges.size else ""
        cpf = f"  cap-filtered(brown): {n_cf}" if edg_edges_capfiltered.size else ""
        dynflag = "" if state["show_dyn"] else "  [dynamic points HIDDEN, D to show]"
        nhf = f"  non-hop-ok(blue): {n_nh}" if (edg_edges_nonhop_debug.size and state["show_nonhop"]) else ""
        print(f"[Open3D] frame={frame}  reliable={n_valid}{extra}  edges={n_e}{hid}{edg}{orn}{cpf}{dynflag}{nhf}")
        return False

    def on_toggle_dyn(vis):
        state["show_dyn"] = not state["show_dyn"]
        return refresh(vis, state["idx"])

    def on_toggle_nonhop(vis):
        state["show_nonhop"] = not state["show_nonhop"]
        return refresh(vis, state["idx"])

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
    cpfleg = "" if not edg_edges_capfiltered.size else "BROWN lines = EDG edges dropped by the cap/stability filter. "
    nhleg = ("" if not edg_edges_nonhop_debug.size else
             "BLUE lines (OFF by default, N to toggle) = every non-hop node pair that PASSES the "
             "too-far/variability test (rigid candidates the hop graph is missing). ")
    print("Open3D controls: SPACE=next frame, B=prev, D=toggle dynamic points, N=toggle blue debug edges, "
          f"S=save PLY, Q/ESC=close. Start frame={state['idx']} (T={T}). {legend}"
          f"Cyan lines = track connectivity (Notes/EDG.md). {edgleg}{ornleg}{cpfleg}"
          f"Press D to hide dynamic points/track edges and show ONLY the EDG nodes+edges (red). {nhleg}")
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="build_edg: dynamic points (purple) + EDG (red)", width=1280, height=720)
    _fill(state["idx"])  # fill BEFORE add_geometry so the camera fits
    vis.add_geometry(pcd)
    vis.add_geometry(lines)
    vis.add_geometry(edg_line)
    vis.add_geometry(dyn_edg_line)
    vis.add_geometry(capfilt_line)
    vis.add_geometry(nonhop_line)
    vis.add_geometry(edg_mesh)
    vis.add_geometry(axes)
    vis.get_render_option().point_size = 4.0
    for key in (" ",):
        vis.register_key_callback(ord(key), on_space)
    vis.register_key_callback(ord("b"), on_b); vis.register_key_callback(ord("B"), on_b)
    vis.register_key_callback(ord("d"), on_toggle_dyn); vis.register_key_callback(ord("D"), on_toggle_dyn)
    vis.register_key_callback(ord("n"), on_toggle_nonhop); vis.register_key_callback(ord("N"), on_toggle_nonhop)
    vis.register_key_callback(ord("s"), on_save); vis.register_key_callback(ord("S"), on_save)
    vis.register_key_callback(ord("q"), on_q); vis.register_key_callback(ord("Q"), on_q)
    vis.register_key_callback(256, on_q)  # ESC
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
