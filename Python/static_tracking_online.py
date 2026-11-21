#!/usr/bin/env python3
"""Run 2D tracking of the static scene for XM datasets. Tracker (config STATIC_TRACKER): "tapnext" (default; the
online TAPNext++ tracker) or "alltracker" (AllTracker from keyframes, utils/track_alltracker.py).

Paths come from utils.definitions, the parameters from config.json.
"""

from pathlib import Path
from collections import deque
import sys
import numpy as np
import torch
from scipy.spatial import cKDTree

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deps" / "tapnet"))

from tapnet.tapnext.tapnext_torch import TAPNext
from tapnet.tapnext.tapnext_torch_utils import tracker_certainty

from utils.track_tapnet import (
    apply_dynamic_mask_filter,
    dilate_dynamic_masks,
    load_dynamic_masks,
    load_frames_256,
    make_grid_query_points,
    write_viz_mp4,
)

import utils.definitions
import utils.dataset_config as dataset_config
from utils.script_logger import install_script_logger

### S0: CONFIG
IMAGE_DIR = utils.definitions.video_dir
MASK_DIR = utils.definitions.mask_dir  # a non-existent path if no masks
OUT_DIR = utils.definitions.output_dir

IMAGE_PATTERN = utils.definitions.image_pattern
MASK_PATTERN = utils.definitions.mask_pattern

GRID = 50

# 2D tracker (STATIC_TRACKER). TAPNext (default): a GRID x GRID grid at frame 0, then at every frame new points
# wherever no visible point is within ONLINE_RESAMPLE_RADIUS (px @256), tracked online; the flow-consistency trim
# (move-delete) below runs for TAPNext only.
# AllTracker: points seeded off the (dilated) moving-object masks, a GRID x GRID grid at frame 0 and then
# every ALLTRACKER_KEYFRAME frames (and at any frame where fewer than STATIC_ALLTRACKER_MIN_VISIBLE points are visible)
# an ONLINE_RESAMPLE_GRID grid wherever no tracked point is visible within ONLINE_RESAMPLE_RADIUS (px @256); each
# keyframe's points are tracked at most STATIC_ALLTRACKER_MAX_LEN frames; visible where AllTracker's visibility x
# confidence >= STATIC_ALLTRACKER_VIS_THRESHOLD and the point is off the moving objects. At STATIC_ALLTRACKER_RES = 1024
# (AllTracker's own default) a 1024 x 768 video needs ~24 GB of GPU memory; 512 makes the static tracks less accurate.
TRACKER = dataset_config.get_str("STATIC_TRACKER", "tapnext").lower()
if TRACKER not in {"alltracker", "tapnext"}:
    raise ValueError(f"STATIC_TRACKER must be 'alltracker' or 'tapnext', got '{TRACKER}'")
AT_KEYFRAME = dataset_config.get_int("ALLTRACKER_KEYFRAME", 10)
AT_RES = dataset_config.get_int("STATIC_ALLTRACKER_RES", 1024)
AT_VIS_THRESHOLD = dataset_config.get_float("STATIC_ALLTRACKER_VIS_THRESHOLD", 0.99)
AT_MAX_LEN = dataset_config.get_int("STATIC_ALLTRACKER_MAX_LEN", 100)
AT_MIN_VISIBLE = dataset_config.get_int("STATIC_ALLTRACKER_MIN_VISIBLE", 200)

MOVE_DELETE_ENABLED = TRACKER == "tapnext"
MOVE_WINDOW_SIZE = 10.0
MOVE_DELETE_FLOW_ERROR_THRESHOLD = dataset_config.get_float(
    "MOVE_DELETE_FLOW_ERROR_THRESHOLD", 10.0
)  # Pixel threshold for local flow-vector deviation.
# Dilation (native-resolution px) applied to dynamic masks for BOTH the query
# exclusion (before track_online_tapnext) and the final track filter, so a
# conservative margin around moving objects is used consistently.
MASK_DILATE_PX = 10

CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "deps"
    / "tapnet"
    / "checkpoints"
    / "tapnextpp_ckpt.pt"
)

# Online tracking hyperparameters (new points are added at every frame)
ONLINE_RESAMPLE_GRID = dataset_config.get_int(
    "ONLINE_RESAMPLE_GRID", 60
)  # random candidate density; candidates ~= grid^2
ONLINE_RESAMPLE_MARGIN = 1
ONLINE_RESAMPLE_RADIUS = dataset_config.get_float(
    "ONLINE_RESAMPLE_RADIUS", 4.0
)
ONLINE_CERTAINTY_RADIUS = 8
ONLINE_CERTAINTY_THRESHOLD = dataset_config.get_float(
    "ONLINE_CERTAINTY_THRESHOLD", 0.8
)
ONLINE_MAX_POINTS = 20000
ONLINE_FP16 = dataset_config.get_bool("ONLINE_FP16", True)  # TAPNext under fp16 autocast
DEBUG = dataset_config.get_bool("DEBUG", False)  # also write tracks_debug.npz and the viz videos

OUT_DIR.mkdir(parents=True, exist_ok=True)
install_script_logger("static_tracking", OUT_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _exclude_queries_in_dynamic_masks(query_np, masks_256, swap_xy_for_query=False):
    """Remove query points that fall inside dynamic masks.

    query_np: [1, Q, 3] with (t, x, y) by default.
    If swap_xy_for_query=True, interpret as (t, y, x).
    masks_256: [T, 256, 256] where >0 means dynamic.
    """
    if query_np is None or query_np.size == 0 or masks_256 is None:
        return query_np

    q = query_np[0]
    if q.shape[0] == 0:
        return query_np

    t = np.rint(q[:, 0]).astype(np.int64)
    if swap_xy_for_query:
        y = np.rint(q[:, 1]).astype(np.int64)
        x = np.rint(q[:, 2]).astype(np.int64)
    else:
        x = np.rint(q[:, 1]).astype(np.int64)
        y = np.rint(q[:, 2]).astype(np.int64)

    Tm, Hm, Wm = masks_256.shape
    in_bounds = (t >= 0) & (t < Tm) & (x >= 0) & (x < Wm) & (y >= 0) & (y < Hm)
    keep = np.ones(q.shape[0], dtype=bool)
    if np.any(in_bounds):
        inside_dynamic = masks_256[t[in_bounds], y[in_bounds], x[in_bounds]] > 0
        keep[np.where(in_bounds)[0]] = ~inside_dynamic

    q_kept = q[keep]
    if q_kept.shape[0] == 0:
        return query_np
    return q_kept[None, ...]

def _make_uncovered_query_points_at_frame(
    tracks_xy_256,
    occluded,
    frame_idx,
    grid,
    margin=8,
    radius=10.0,
    prev_added_xy: np.ndarray | None = None,
):
    """Create random query points in regions not covered at a single frame.

    `grid` is used as a density knob: number of random candidates is grid^2.
        Returns [1, Q, 3] with (t, y, x), TAPNext's query order.

        A candidate is selected if it is farther than `radius` from:
            1) visible tracked points at the current frame, and
            2) `prev_added_xy` points (if provided), e.g. recent added points.
    """
    t = int(frame_idx)
    pts = tracks_xy_256[0, :, t, :]
    valid = np.isfinite(pts).all(axis=1) & (~occluded[0, :, t])

    n_candidates = int(max(1, grid * grid))
    x0, x1 = float(margin), float(255 - margin)
    y0, y1 = float(margin), float(255 - margin)
    if x1 <= x0 or y1 <= y0:
        return None

    cand_x = np.random.uniform(x0, x1, size=n_candidates).astype(np.float64)
    cand_y = np.random.uniform(y0, y1, size=n_candidates).astype(np.float64)
    cand = np.stack([cand_x, cand_y], axis=1)  # [N,2] as (x,y)

    radius2 = float(radius) * float(radius)
    ref_sets = []
    if np.any(valid):
        ref_sets.append(pts[valid].astype(np.float64))
    if prev_added_xy is not None and prev_added_xy.shape[0] > 0:
        ref_sets.append(prev_added_xy.astype(np.float64))

    if ref_sets:
        refs = np.concatenate(ref_sets, axis=0)
        min_d, _ = cKDTree(refs).query(cand, k=1)
        mask_uncovered = min_d * min_d > radius2
    else:
        mask_uncovered = np.ones(cand.shape[0], dtype=bool)

    if not np.any(mask_uncovered):
        return None

    t_arr = np.full(np.sum(mask_uncovered), float(t), dtype=np.float64)
    x = cand[mask_uncovered, 0]
    y = cand[mask_uncovered, 1]
    return np.stack([t_arr, y, x], axis=1)[None, ...]


def _compute_flow_error_by_transition(
    tracks_xy: np.ndarray,
    occluded_obs: np.ndarray,
    window: float,
) -> np.ndarray:
    """Compute per-transition local flow-vector error for each query.

    Returns [Q, T] where entry [:, t] measures the difference between a
    query's flow and the median flow of nearby points from t-1 to t. Values
    are NaN when that transition cannot be evaluated.
    """
    # Neighbours = every other valid point within half a window in x AND y (Chebyshev distance <= half, KD-tree
    # pairs); neighbour flow = per-component median (np.median semantics: mean of the two middle values for an even
    # count). Same result as a per-point loop with np.median, ~20x faster.
    Q = tracks_xy.shape[1]
    T = tracks_xy.shape[2]
    flow_error = np.full((Q, T), np.nan, dtype=np.float64)
    window = float(max(1e-6, window))
    half = window / 2.0

    for t in range(1, T):
        pts = tracks_xy[0, :, t, :]
        prev = tracks_xy[0, :, t - 1, :]

        valid = (
            np.isfinite(pts).all(axis=1)
            & np.isfinite(prev).all(axis=1)
            & (~occluded_obs[0, :, t])
            & (~occluded_obs[0, :, t - 1])
        )
        idx_valid = np.where(valid)[0]
        if idx_valid.size < 2:
            continue

        P = pts[idx_valid].astype(np.float64)
        flow = P - prev[idx_valid].astype(np.float64)
        n = len(P)

        pairs = cKDTree(P).query_pairs(half, p=np.inf, output_type="ndarray")
        if pairs.size == 0:
            continue
        a = np.r_[pairs[:, 0], pairs[:, 1]]
        b = np.r_[pairs[:, 1], pairs[:, 0]]
        cnt = np.bincount(a, minlength=n)
        has = cnt > 0
        start = np.r_[0, np.cumsum(cnt)[:-1]]
        lo = (start + (cnt - 1) // 2)[has]
        hi = (start + cnt // 2)[has]
        neighbor_flow = np.empty((int(has.sum()), 2))
        for d in range(2):
            v = flow[b, d][np.lexsort((flow[b, d], a))]      # grouped by point, sorted within each group
            neighbor_flow[:, d] = (v[lo] + v[hi]) / 2.0
        flow_error[idx_valid[has], t] = np.linalg.norm(flow[has] - neighbor_flow, axis=1)

    return flow_error


def _select_longest_stable_segment(
    valid_obs: np.ndarray,
    bad_transition: np.ndarray,
    transition_score: np.ndarray | None = None,
) -> np.ndarray:
    """Keep the longest contiguous valid segment split by bad transitions."""
    T = valid_obs.shape[0]
    keep = np.zeros(T, dtype=bool)
    if T == 0 or not np.any(valid_obs):
        return keep

    best_start = -1
    best_end = -1
    best_len = -1
    best_stability = np.inf

    t = 0
    while t < T:
        if not valid_obs[t]:
            t += 1
            continue

        start = t
        end = t
        stability = 0.0

        while end + 1 < T and valid_obs[end + 1] and not bad_transition[end + 1]:
            end += 1
            if transition_score is not None and np.isfinite(transition_score[end]):
                stability += float(transition_score[end])

        seg_len = end - start + 1
        if (
            seg_len > best_len
            or (seg_len == best_len and stability < best_stability)
            or (
                seg_len == best_len
                and np.isclose(stability, best_stability)
                and (best_start < 0 or start < best_start)
            )
        ):
            best_start = start
            best_end = end
            best_len = seg_len
            best_stability = stability

        t = end + 1

    if best_start >= 0:
        keep[best_start : best_end + 1] = True
    return keep

def _expand_state_for_new_queries(state, n_add: int):
    """Expand TAPNext recurrent state for newly appended query tokens.

    TAPNext cache is per-token along dim-0 for each block cache.
    When query count grows by `n_add` (batch=1), append `n_add` zero rows
    to each cache tensor on that token dimension.
    """
    if state is None or n_add <= 0 or state.hidden_state is None:
        return state

    expanded = []
    for cache in state.hidden_state:
        if cache is None:
            expanded.append(cache)
            continue

        rg = cache.rg_lru_state
        cv = cache.conv1d_state

        rg_pad = torch.zeros((n_add,) + tuple(rg.shape[1:]), dtype=rg.dtype, device=rg.device)
        cv_pad = torch.zeros((n_add,) + tuple(cv.shape[1:]), dtype=cv.dtype, device=cv.device)

        rg_new = torch.cat([rg, rg_pad], dim=0)
        cv_new = torch.cat([cv, cv_pad], dim=0)

        if hasattr(cache, "_replace"):
            cache_new = cache._replace(rg_lru_state=rg_new, conv1d_state=cv_new)
        else:
            cache_new = type(cache)(rg_lru_state=rg_new, conv1d_state=cv_new)
        expanded.append(cache_new)

    state.hidden_state = expanded
    return state


@torch.no_grad()
def track_online_tapnext(
    video_256: torch.Tensor,
    query_points_np: np.ndarray,
    device: torch.device,
    checkpoint: str,
    *,
    resample_grid: int = 20,
    resample_margin: int = 8,
    resample_radius: float = 10.0,
    certainty_radius: int = 8,
    certainty_threshold: float = 0.5,
    exclude_masks_256: np.ndarray | None = None,
    max_points: int = 6000,
    fp16: bool = False,
):
    """Online tracking with per-frame query augmentation (new queries in TAPNext's (t, y, x) order)."""
    model = TAPNext(image_size=(256, 256))
    ckpt = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict({k.replace('tapnext.', ''): v for k, v in ckpt['state_dict'].items()})
    model.cuda()

    T = int(video_256.shape[1])
    all_query_points_abs = query_points_np.astype(np.float64, copy=True)
    init_q = int(all_query_points_abs.shape[1])
    if init_q > max_points:
        raise ValueError(f"init_q={init_q} exceeds max_points={max_points}")

    active_query_indices = np.arange(init_q, dtype=np.int64)
    query_points_t = torch.from_numpy(all_query_points_abs[:, active_query_indices, :]).float().to(device)

    state = None
    tracks_per_frame = []
    occ_per_frame = []
    active_indices_per_frame = []
    prev_added_xy_history = deque(maxlen=5)  # keep added-point XY from last 5 add-frames
    last_tracks_xy = None
    last_occ_np = None
    last_active_query_indices = None
    add_points = True   # until max_points is reached

    for t in tqdm(range(T), desc="Online tracking"):
        query_grew = False
        n_add_effective = 0

        if add_points and (t + 1 < T):
            prev_added_xy = None
            if len(prev_added_xy_history) > 0:
                prev_added_xy = np.concatenate(list(prev_added_xy_history), axis=0)

            q_curr = int(active_query_indices.shape[0])
            tmp_tracks = np.zeros((1, q_curr, 1, 2), dtype=np.float32)
            tmp_occ = np.ones((1, q_curr, 1), dtype=bool)
            if last_tracks_xy is not None and last_active_query_indices is not None:
                _, curr_pos, prev_pos = np.intersect1d(
                    active_query_indices,
                    last_active_query_indices,
                    assume_unique=True,
                    return_indices=True,
                )
                if curr_pos.size > 0:
                    tmp_tracks[0, curr_pos, 0, :] = last_tracks_xy[prev_pos]
                    tmp_occ[0, curr_pos, 0] = last_occ_np[prev_pos]

            new_q = _make_uncovered_query_points_at_frame(
                tmp_tracks,
                tmp_occ,
                frame_idx=0,
                grid=resample_grid,
                margin=resample_margin,
                radius=resample_radius,
                prev_added_xy=prev_added_xy,
            )
            if new_q is not None:
                new_q[:, :, 0] = float(t + 1)
                if exclude_masks_256 is not None:
                    new_q = _exclude_queries_in_dynamic_masks(
                        new_q,
                        exclude_masks_256,
                        swap_xy_for_query=True,
                    )
                if new_q is not None and new_q.shape[1] > 0:
                    q_total = int(all_query_points_abs.shape[1])
                    q_free = int(max_points - q_curr)
                    if q_free <= 0:
                        print("Reached max_points limit; skipping new query additions.")
                        add_points = False
                    else:
                        n_add = int(min(q_free, new_q.shape[1]))
                        if n_add < new_q.shape[1]:
                            print(f"Reached max_points limit; truncated new queries: keeping {n_add}/{new_q.shape[1]}")
                            add_points = False
                        add_np = new_q[:, :n_add, :].astype(np.float64, copy=False)
                        new_ids = np.arange(q_total, q_total + n_add, dtype=np.int64)
                        all_query_points_abs = np.concatenate([all_query_points_abs, add_np], axis=1)
                        active_query_indices = np.concatenate([active_query_indices, new_ids])
                        add_t = torch.from_numpy(add_np).float().to(device)
                        query_points_t = torch.cat([query_points_t, add_t], dim=1)

                        # Update rolling history with actually added points in XY convention.
                        add_xy = add_np[0, :, 1:3]
                        add_xy = add_xy[:, [1, 0]]
                        prev_added_xy_history.append(add_xy.astype(np.float64, copy=False))

                        query_grew = n_add > 0
                        n_add_effective = n_add

        if query_grew and state is not None:
            # Keep recurrent state alive: update query points and pad block caches
            # with zero-init rows for each newly appended query token.
            state.query_points = query_points_t
            state = _expand_state_for_new_queries(state, n_add_effective)


        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
            if state is None:
                pred_tracks, track_logits, visible_logits, state = model(
                    video=video_256[:, t : t + 1], query_points=query_points_t
                )
            else:
                state.query_points = query_points_t
                pred_tracks, track_logits, visible_logits, state = model(
                    video=video_256[:, t : t + 1], state=state
                )
        pred_tracks, track_logits, visible_logits = pred_tracks.float(), track_logits.float(), visible_logits.float()

        tracks_yx = pred_tracks.detach().cpu()
        tracks_xy = torch.flip(tracks_yx, dims=[-1]).numpy()[0, 0]  # [Q,2]

        certainty = tracker_certainty(tracks_yx, track_logits.detach().cpu(), certainty_radius)
        pred_visible = (torch.sigmoid(visible_logits.detach().cpu()) * certainty) > certainty_threshold
        occluded = ~(pred_visible.squeeze(0).squeeze(0))

        occluded_np = occluded.detach().cpu().numpy().astype(bool).reshape(-1)

        tracks_per_frame.append(tracks_xy)
        occ_per_frame.append(occluded_np)
        active_indices_per_frame.append(active_query_indices.copy())
        last_tracks_xy = tracks_xy.copy()
        last_occ_np = occluded_np.copy()
        last_active_query_indices = active_query_indices.copy()

    Q_final = int(all_query_points_abs.shape[1])
    tracks_xy_full = np.full((1, Q_final, T, 2), np.nan, dtype=np.float32)
    occluded_full = np.ones((1, Q_final, T), dtype=bool)

    for t, (tracks_xy, occ, active_idx) in enumerate(zip(tracks_per_frame, occ_per_frame, active_indices_per_frame)):
        tracks_xy_full[0, active_idx, t, :] = tracks_xy
        occluded_full[0, active_idx, t] = np.asarray(occ, dtype=bool).reshape(-1)

    query_t = np.floor(all_query_points_abs[0, :, 0]).astype(np.int64)
    time_idx = np.arange(T, dtype=np.int64)[None, :]
    pre_query_mask = time_idx < query_t[:, None]
    occluded_full[0] |= pre_query_mask

    return tracks_xy_full, occluded_full


video_np_256, video_np = load_frames_256(str(IMAGE_DIR), IMAGE_PATTERN)
H, W = video_np.shape[2], video_np.shape[3]

if TRACKER == "tapnext":
    query_exclude_masks_256 = None
    if Path(MASK_DIR).exists():
        try:
            _masks_256 = load_dynamic_masks(
                str(MASK_DIR),
                MASK_PATTERN,
                t_expected=video_np.shape[1],
                mask_mode="dynamic_positive",
                target_size=(256, 256),
            )
            _masks_256 = np.asarray(_masks_256)
            query_exclude_masks_256 = _masks_256
            # Dilate the query-exclusion masks (in 256-grid space) to match the
            # margin used by the final track filter (MASK_DILATE_PX is defined at
            # native resolution, so scale it to the 256 grid).
            dilate_px_256 = max(1, int(round(MASK_DILATE_PX * 256.0 / float(max(H, W)))))
            query_exclude_masks_256 = dilate_dynamic_masks(query_exclude_masks_256, dilate_px_256)
            print(
                f"Dilated query-exclusion masks by {dilate_px_256}px @256 "
                f"(= {MASK_DILATE_PX}px @native {W}x{H})"
            )
        except Exception as e:
            print(f"Failed to load 256 dynamic masks for query exclusion: {e}")

    query_np = make_grid_query_points(GRID)
    if query_exclude_masks_256 is not None:
        query_np = _exclude_queries_in_dynamic_masks(
            query_np,
            query_exclude_masks_256,
            swap_xy_for_query=False,
        )

    video_256 = torch.from_numpy(video_np_256).to(device)

    tracks_xy_256, occluded = track_online_tapnext(
        video_256,
        query_np,
        device=device,
        checkpoint=str(CHECKPOINT),
        resample_grid=ONLINE_RESAMPLE_GRID,
        resample_margin=ONLINE_RESAMPLE_MARGIN,
        resample_radius=ONLINE_RESAMPLE_RADIUS,
        certainty_radius=ONLINE_CERTAINTY_RADIUS,
        certainty_threshold=ONLINE_CERTAINTY_THRESHOLD,
        exclude_masks_256=query_exclude_masks_256,
        max_points=ONLINE_MAX_POINTS,
        fp16=ONLINE_FP16,
    )


    # Map tracks from 256-grid space back to original image space
    tracks_xy = tracks_xy_256
    tracks_xy[..., 0] = tracks_xy[..., 0] / 256 * W
    tracks_xy[..., 1] = tracks_xy[..., 1] / 256 * H
else:
    from utils.track_alltracker import grid_seeds, track_keyframes

    exclude = np.zeros((video_np.shape[1], H, W), bool)                 # moving objects, dilated as below
    if Path(MASK_DIR).exists():
        exclude = np.asarray(dilate_dynamic_masks(load_dynamic_masks(
            str(MASK_DIR), MASK_PATTERN, t_expected=video_np.shape[1], mask_mode="dynamic_positive", target_size=(W, H),
        ), MASK_DILATE_PX))
    side = float(max(H, W))
    radius = ONLINE_RESAMPLE_RADIUS * side / 256.0
    print(f"AllTracker: {GRID}x{GRID} seeds at frame 0, re-seeding every {AT_KEYFRAME} frames (and below {AT_MIN_VISIBLE} "
          f"visible points) where no point is visible within {radius:.1f}px, tracks <= {AT_MAX_LEN} frames, {AT_RES}px, "
          f"visible at vis x conf >= {AT_VIS_THRESHOLD}")
    trk, vis, _ = track_keyframes(
        video_np[0],
        lambda t, covered: grid_seeds(~(exclude[t] > 0), side / (GRID if t == 0 else ONLINE_RESAMPLE_GRID), covered, radius),
        keyframe=AT_KEYFRAME, res=AT_RES, vis_threshold=AT_VIS_THRESHOLD, max_points=ONLINE_MAX_POINTS,
        max_len=AT_MAX_LEN, min_visible=AT_MIN_VISIBLE, device=str(device), label="AllTracker (static)",
    )
    tracks_xy = trk[None]
    occluded = ~vis[None]
    print(f"AllTracker static tracks: {trk.shape[0]}")

# Optional dynamic masks
dynamic_masks = None
if Path(MASK_DIR).exists():
    try:
        # Load masks at the original image size so we can apply them directly on tracks_xy
        dynamic_masks = load_dynamic_masks(
            str(MASK_DIR),
            "*.png",
            t_expected=video_np.shape[1],
            mask_mode="dynamic_positive",
            target_size=(W, H),
        )
        dynamic_masks = dilate_dynamic_masks(dynamic_masks, MASK_DILATE_PX)
        occluded_before = occluded.copy()

        # tracks_xy is in original image coordinates (W,H).
        occluded = apply_dynamic_mask_filter(tracks_xy, occluded, dynamic_masks)
        print(
            f"Applied masks from {MASK_DIR}. Newly hidden: "
            f"{int(np.sum(occluded) - np.sum(occluded_before))}"
        )
    except Exception as e:
        print(f"Failed to load/apply masks from {MASK_DIR}: {e}")

# Save 2D outputs
if DEBUG:
    np.savez_compressed(
        OUT_DIR / "tracks_debug.npz",
        tracks_xy=tracks_xy,
        occluded=occluded,
        video_dir=str(IMAGE_DIR),
        grid=int(GRID),
        mask_dir=str(MASK_DIR),
    )

move_delete_deleted_obs_mask = None

# Delete points with abnormally large 2D motion vs nearby points. Applied after the
# tracks_debug save (which stays unfiltered) and before the solver output below.
if MOVE_DELETE_ENABLED:
    T_md = tracks_xy.shape[2]
    Q_md = tracks_xy.shape[1]
    move_delete_deleted_obs_mask = np.zeros((1, Q_md, T_md), dtype=bool)
    window = float(max(1e-6, MOVE_WINDOW_SIZE))
    flow_error_threshold = float(max(0.0, MOVE_DELETE_FLOW_ERROR_THRESHOLD))
    n_trimmed_obs = 0
    flow_error_by_transition = _compute_flow_error_by_transition(
        tracks_xy[:, :, :T_md, :],
        occluded[:, :, :T_md],
        window=window,
    )
    move_delete_query_mask = np.any(
        np.isfinite(flow_error_by_transition) & (flow_error_by_transition > flow_error_threshold),
        axis=1,
    )
    n_move_delete = int(np.sum(move_delete_query_mask))
    if n_move_delete > 0:
        for q_idx in np.where(move_delete_query_mask)[0]:
            valid_obs = (
                (~occluded[0, q_idx, :T_md])
                & np.isfinite(tracks_xy[0, q_idx, :T_md, :]).all(axis=1)
            )
            if not np.any(valid_obs):
                occluded[0, q_idx, :T_md] = True
                continue

            bad_transition = np.zeros(T_md, dtype=bool)
            bad_transition[1:] = (
                np.isfinite(flow_error_by_transition[q_idx, 1:T_md])
                & (flow_error_by_transition[q_idx, 1:T_md] > flow_error_threshold)
            )
            keep_obs = _select_longest_stable_segment(
                valid_obs,
                bad_transition,
                transition_score=flow_error_by_transition[q_idx, :T_md],
            )
            drop_obs = valid_obs & (~keep_obs)
            if np.any(drop_obs):
                n_trimmed_obs += int(np.sum(drop_obs))
                occluded[0, q_idx, drop_obs] = True
                move_delete_deleted_obs_mask[0, q_idx, drop_obs] = True
    print(
        f"[move-delete] flagged {n_move_delete} queries; "
        f"trimmed {n_trimmed_obs} observations "
        f"(flow error > {flow_error_threshold:.3f}px, window={window:.1f}px)"
    )

occluded_viz = occluded.copy()

output_tapnet_src_path = OUT_DIR / "output_tapnet_da3.npz"
np.savez_compressed(
    output_tapnet_src_path,
    tracks_xy=tracks_xy.astype(np.float32),
    occluded=occluded.astype(bool),
    video_dir=str(IMAGE_DIR),
    mask_dir=str(MASK_DIR),
    depth_source=np.array("da3"),
    grid=np.array(int(GRID), dtype=np.int64),
)
print(f"Wrote 2D tracking output for solver: {output_tapnet_src_path}")

if DEBUG:
    write_viz_mp4(
        video_np,
        tracks_xy,
        occluded_viz,
        str(OUT_DIR / "viz_debug.mp4"),
        fps=10,
        dynamic_masks=dynamic_masks,
        highlight_radius=5,
        highlight_obs_mask_3=move_delete_deleted_obs_mask,
        highlight_obs_color_3=(255, 0, 255),
    )
    write_viz_mp4(video_np, None, None, str(OUT_DIR / "viz_origin.mp4"))
