#!/usr/bin/env python3
"""Dynamic-only tracking aligned with static_tracking_online.py. Tracker (config DYNAMIC_TRACKER): "alltracker" (default;
AllTracker from keyframes, utils/track_alltracker.py) or "tapnext" (the online TAPNext++ tracker)."""

from pathlib import Path
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deps" / "tapnet"))

from utils.readgt_DAVIS import load_davis_gt
from utils.readgt_TUM import load_tum_gt
from utils.readgt_colmap import load_colmap_gt
from utils.readgt_custom import load_custom_gt
from utils.readgt_replica import load_replica_gt, read_replica_depth
from utils.readgt_sintel import load_sintel_gt, depth_read
from utils.readgt_dycheck import load_dycheck_gt
from utils.masks import _numeric_key, dilate_dynamic_masks, load_dynamic_masks  # utils.track_tapnet imports tapnet (~3 s)

import utils.definitions
import utils.dataset_config as dataset_config
from utils.io import savez_compressed_fast
from utils.script_logger import install_script_logger, pause_script_logger, resume_script_logger


### S0: CONFIG
dataset_cat = utils.definitions.dataset_cat
rootpath = utils.definitions.rootpath
DEPTH_SOURCE = utils.definitions.depth_source

IMAGE_DIR = utils.definitions.video_dir
MASK_DIR = utils.definitions.mask_dir
OUT_DIR = utils.definitions.output_dir
DA3_OUT_DIR = utils.definitions.da3_depth_dir
GT_DEPTH_DIR = utils.definitions.gt_depth_dir

IMAGE_PATTERN = utils.definitions.image_pattern
MASK_PATTERN = utils.definitions.mask_pattern

# Tunable per-dataset params come from the dataset config (utils.dataset_config),
# same pattern as static_tracking_online.py. Dynamic tracking uses DYN_-prefixed
# keys so its values don't collide with the static tracker's shared param names.
# Env vars override the config; the config overrides these defaults.
# Depth filters (both deletions -> occluded). A value <= 0 DISABLES the respective filter, keeping
# every point with valid positive depth regardless of its magnitude / gradient.
DEPTH_MAX = dataset_config.get_float("DYN_DEPTH_MAX", 0.0, env="DYN_DEPTH_MAX")
DEPTH_GRAD_THRESHOLD = dataset_config.get_float("DYN_DEPTH_GRAD_THRESHOLD", 0.0, env="DYN_DEPTH_GRAD_THRESHOLD")
DEPTH_JUMP_THRESHOLD = None

# 2D tracker. AllTracker: points seeded on the object mask every ALLTRACKER_KEYFRAME frames where no tracked point
# is visible nearby (grid spacing DYN_ALLTRACKER_GRID px at the tracking resolution ALLTRACKER_RES), visible where
# AllTracker's visibility x confidence >= ALLTRACKER_VIS_THRESHOLD and the point is on the object mask. The
# TAPNext-only filters below (large-2D-move truncation, lost-off-object) are not applied to AllTracker tracks.
TRACKER = dataset_config.get_str("DYNAMIC_TRACKER", "alltracker", env="DYNAMIC_TRACKER").lower()
if TRACKER not in {"alltracker", "tapnext"}:
    raise ValueError(f"DYNAMIC_TRACKER must be 'alltracker' or 'tapnext', got '{TRACKER}'")
AT_KEYFRAME = dataset_config.get_int("ALLTRACKER_KEYFRAME", 10, env="ALLTRACKER_KEYFRAME")
AT_RES = dataset_config.get_int("ALLTRACKER_RES", 512, env="ALLTRACKER_RES")
AT_VIS_THRESHOLD = dataset_config.get_float("ALLTRACKER_VIS_THRESHOLD", 0.6, env="ALLTRACKER_VIS_THRESHOLD")
DYN_AT_GRID = dataset_config.get_float("DYN_ALLTRACKER_GRID", 2.0, env="DYN_ALLTRACKER_GRID")
AT_MIN_NEW = dataset_config.get_float("ALLTRACKER_MIN_NEW", 0.0, env="ALLTRACKER_MIN_NEW")
AT_FP16 = dataset_config.get_bool("ALLTRACKER_FP16", True, env="ALLTRACKER_FP16")
DEBUG = dataset_config.get_bool("DEBUG", False, env="DEBUG")  # also write the per-reason visualisation video

MOVE_DELETE_ENABLED = TRACKER == "tapnext"
MOVE_WINDOW_SIZE = 10.0
MOVE_DELETE_TOP_PERCENT = dataset_config.get_float("DYN_MOVE_DELETE_TOP_PERCENT", 100.0, env="DYN_MOVE_DELETE_TOP_PERCENT")
# Only scan every Nth frame for the large-2D-move red flag (5 -> ~5x cheaper move-delete). The
# truncation point is then coarser by up to (stride-1) frames, which is fine for this filter.
MOVE_DELETE_STRIDE = dataset_config.get_int("DYN_MOVE_DELETE_STRIDE", 5, env="DYN_MOVE_DELETE_STRIDE")
# Lost-off-object deletion: if a point is invisible (tracker-occluded) AND out of the dynamic mask for
# MORE THAN this many consecutive frames, it has left the object -> truncate the track there. 0 disables.
LOST_OFFMASK_MAX_FRAMES = dataset_config.get_int("DYN_LOST_OFFMASK_MAX_FRAMES", 2, env="DYN_LOST_OFFMASK_MAX_FRAMES")
if TRACKER != "tapnext":
    LOST_OFFMASK_MAX_FRAMES = 0
# --- Smart max-points eviction (when the active set is full): evict only points that are clearly BAD,
# instead of dropping every currently-invisible one. Per frame a present point is BAD if it is off the
# dynamic mask by > DYN_MAXPT_OFFMASK_PX px OR has a large 2D move vs its neighbours. A point is evicted
# if it was BAD in > DYN_MAXPT_BAD_MIN of the past DYN_MAXPT_PAST frames, OR it was invisible for the last
# DYN_MAXPT_INVIS_CONSEC consecutive frames -- so temporarily-hidden points on the object survive. ---
MAXPT_OFFMASK_PX = dataset_config.get_float("DYN_MAXPT_OFFMASK_PX", 10.0, env="DYN_MAXPT_OFFMASK_PX")
MAXPT_PAST = dataset_config.get_int("DYN_MAXPT_PAST", 10, env="DYN_MAXPT_PAST")
MAXPT_BAD_MIN = dataset_config.get_int("DYN_MAXPT_BAD_MIN", 7, env="DYN_MAXPT_BAD_MIN")
MAXPT_INVIS_CONSEC = dataset_config.get_int("DYN_MAXPT_INVIS_CONSEC", 20, env="DYN_MAXPT_INVIS_CONSEC")
SMOOTH_3D_ENABLED = False
SMOOTH_3D_WINDOW = 10

MASK_MODE = "dynamic_positive"
# Dilation (>0) / erosion (<0) in native px for the dynamic masks. Query masks
# select points ON the moving object, so erosion keeps queries safely inside
# (conservative = fewer, safer dynamic points).
DILATE_PX_QUERY = dataset_config.get_int("DYN_DILATE_PX_QUERY", -2, env="DYN_DILATE_PX_QUERY")
DILATE_PX_VIS = dataset_config.get_int("DYN_DILATE_PX_VIS", -2, env="DYN_DILATE_PX_VIS")
# Single density knob (config key DYN_MASK_KEEP_FRAC in (0,1]): fraction of dynamic-mask points to
# track. 1.0 = 100% = densest (~one query per mask pixel); 0.8 = 80%, etc. It scales BOTH the initial
# mask sampling and the online resample spacing so the whole track density follows it.
DENSITY_FRAC = float(min(1.0, max(0.02, dataset_config.get_float("DYN_MASK_KEEP_FRAC", 1.0, env="DYN_MASK_KEEP_FRAC"))))
EDGE_ERODE_PX = 1

RESAMPLE_SWAP_XY = True
ONLINE_ADD_POINTS = True
ONLINE_ADD_EVERY = 1
ONLINE_ADD_START = 1
ONLINE_RESAMPLE_MARGIN = 8
# Coverage radius (256-px): a mask pixel within this distance of a live track counts as "already
# sampled" and is not re-seeded. Derived from DENSITY_FRAC (~the spacing at density=frac) so the one
# knob also defines when a region is considered fully covered. Larger -> less density creep over time.
ONLINE_RESAMPLE_RADIUS = float(1.0 / (DENSITY_FRAC ** 0.5))
ONLINE_USE_CERTAINTY = True
ONLINE_CERTAINTY_RADIUS = 8
ONLINE_CERTAINTY_THRESHOLD = dataset_config.get_float("DYN_ONLINE_CERTAINTY_THRESHOLD", 0.8, env="DYN_ONLINE_CERTAINTY_THRESHOLD")
ONLINE_QUERY_DOWNSAMPLE_THRESHOLD = None
ONLINE_MAX_POINTS = 50000
ONLINE_QUERY_DOWNSAMPLE_TARGET_RATIO = 0.5

CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "deps"
    / "tapnet"
    / "checkpoints"
    / "tapnextpp_ckpt.pt"
)

DEPTH_MAPS_PATH = DA3_OUT_DIR / "depth_maps.npy"

OUT_DIR.mkdir(parents=True, exist_ok=True)
install_script_logger("dynamic_tracking", OUT_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- direction mode: forward (default) or --reverse (track from the LAST frame to the first). Reverse is
# implemented by flipping every time-indexed input up front, running the whole pipeline unchanged in reversed
# time (so all forward-semantics code is exactly correct), then flipping the outputs back before saving. ---
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--reverse", action="store_true", help="track backward: seed at the last frame, track to the first")
_ap.add_argument("--out-name", type=str, default="", help="output npz name in OUT_DIR (default depends on direction)")
_args, _ = _ap.parse_known_args()
REVERSE = bool(_args.reverse)
OUT_NAME = _args.out_name or ("tracks_dynamic_backward.npz" if REVERSE else "tracks_dynamic_only.npz")
if REVERSE:
    print("[reverse] BACKWARD tracking: flipping time-indexed inputs; queries seed at the original last frame")


# per-frame K of the dataset, loaded right before the 3D lift (a Custom scene's K comes from step 1, which
# run_pipeline.sh may still be running while this step tracks)
GT_LOADERS = {
    "Sintel": lambda: load_sintel_gt(str(rootpath)),
    "Sintel_debug": lambda: load_sintel_gt(str(rootpath)),
    "TUM": lambda: load_tum_gt(str(rootpath)),
    "DAVIS": lambda: load_davis_gt(str(rootpath)),
    "Replica": lambda: load_replica_gt(str(rootpath)),
    "Syn4D": lambda: load_colmap_gt(str(rootpath / "colmap")),
    "Dycheck": lambda: load_dycheck_gt(str(rootpath)),
    "Custom": lambda: load_custom_gt(rootpath),
}
if dataset_cat not in GT_LOADERS:
    raise ValueError(f"Unsupported dataset_cat '{dataset_cat}'")


def _make_mask_query_points(mask_256: np.ndarray, keep_frac: float = 1.0, query_t: int = 0) -> np.ndarray | None:
    if mask_256 is None:
        return None
    if mask_256.ndim == 3:
        mask_256 = mask_256[..., 0]
    xs, ys = np.where(mask_256 > 0)
    if ys.size == 0:
        return None
    # keep `keep_frac` of the mask pixels as initial queries (1.0 = 100% = every pixel)
    if keep_frac < 1.0:
        n_keep = max(1, int(round(float(keep_frac) * xs.size)))
        sel = np.random.default_rng(0).choice(xs.size, size=n_keep, replace=False)
        xs, ys = xs[sel], ys[sel]
    t_arr = np.full_like(xs, float(query_t), dtype=np.float64)
    pts = np.stack([t_arr, xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    return pts[None, ...]


def _exclude_queries_outside_dynamic_masks(query_np, masks_256, swap_xy_for_query=False):
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
    keep = np.zeros(q.shape[0], dtype=bool)
    if np.any(in_bounds):
        keep[np.where(in_bounds)[0]] = masks_256[t[in_bounds], y[in_bounds], x[in_bounds]] > 0

    q_kept = q[keep]
    if q_kept.shape[0] == 0:
        print("All online-added queries fell outside dynamic masks; keeping previous query set.")
        return None
    return q_kept[None, ...]


def _make_uncovered_query_points_at_frame(
    tracks_xy_256,
    occluded,
    frame_idx,
    mask_256,
    keep_frac=1.0,
    margin=8,
    radius=10.0,
    prev_added_xy: np.ndarray | None = None,
    swap_xy_for_query=False,
    rng=None,
):
    """Seed new dynamic queries the SAME way frame 0 is sampled: look at the dynamic-mask pixels that
    are NOT yet sampled (farther than `radius` from every visible track / recent addition), then keep
    `keep_frac` of them (1.0 = every uncovered mask pixel). So the single knob DYN_MASK_KEEP_FRAC sets
    the fraction of mask points used at frame 0 and at every later frame identically. `radius` only
    marks a pixel as already-sampled, so a region already held near this density is not re-seeded."""
    t = int(frame_idx)
    pts = tracks_xy_256[0, :, t, :]
    valid = np.isfinite(pts).all(axis=1) & (~occluded[0, :, t])

    if mask_256 is None:
        return None
    m = mask_256[..., 0] if mask_256.ndim == 3 else mask_256
    rows, cols = np.where(m > 0)                       # mask pixels: row = y, col = x
    lo, hi = float(margin), float(255 - margin)
    inb = (cols >= lo) & (cols <= hi) & (rows >= lo) & (rows <= hi)
    if not np.any(inb):
        return None
    cand = np.stack([cols[inb].astype(np.float64), rows[inb].astype(np.float64)], axis=1)  # (M,2) as (x,y)

    # Subsample mask candidates by keep_frac BEFORE the distance test, so
    # DYN_MASK_KEEP_FRAC also bounds the per-frame pairwise cost. This makes the resample as light as the
    # static tracker's fixed small candidate set instead of scanning every mask pixel each frame.
    if keep_frac < 1.0 and cand.shape[0] > 1:
        n_keep = max(1, int(round(float(keep_frac) * cand.shape[0])))
        rng = rng if rng is not None else np.random.default_rng()
        cand = cand[rng.choice(cand.shape[0], size=n_keep, replace=False)]

    # "not sampled" = farther than `radius` from every existing visible track / recent addition
    radius2 = float(radius) * float(radius)
    ref_sets = []
    if np.any(valid):
        ref_sets.append(pts[valid].astype(np.float64))
    if prev_added_xy is not None and prev_added_xy.shape[0] > 0:
        ref_sets.append(prev_added_xy.astype(np.float64))
    if ref_sets:
        refs = np.concatenate(ref_sets, axis=0)
        uncovered = np.ones(cand.shape[0], dtype=bool)
        for st in range(0, cand.shape[0], 4096):       # chunk the pairwise distance to bound memory
            c = cand[st:st + 4096]
            min_d2 = np.min(np.sum((c[:, None, :] - refs[None, :, :]) ** 2, axis=2), axis=1)
            uncovered[st:st + 4096] = min_d2 > radius2
    else:
        uncovered = np.ones(cand.shape[0], dtype=bool)

    idx = np.where(uncovered)[0]
    if idx.size == 0:
        return None
    x = cand[idx, 0]
    y = cand[idx, 1]
    t_arr = np.full(idx.size, float(t), dtype=np.float64)
    if swap_xy_for_query:
        return np.stack([t_arr, y, x], axis=1)[None, ...]
    return np.stack([t_arr, x, y], axis=1)[None, ...]


def _compute_move_delete_first_frame(
    tracks_xy: np.ndarray,
    occluded_obs: np.ndarray,
    window: float,
    top_percent: float,
    stride: int = 1,
) -> np.ndarray:
    """For each track, the FIRST frame whose 2D displacement deviates from the median of its spatial
    neighbours (within a `window`-side square) by more than eps = top_percent/100 -- the first
    'large-2D-move' red flag. Returns first_red (Q,) int; == T means the track never triggered.
    The caller TRUNCATES each track at first_red (keeps the good prefix) rather than deleting it whole.
    `stride` scans only every Nth frame (5 -> ~5x cheaper); the detected red flag is then coarser by up
    to (stride-1) frames."""
    Q = tracks_xy.shape[1]
    T = tracks_xy.shape[2]
    eps = float(np.clip(top_percent, 0.0, 100.0)) / 100.0
    first_red = np.full(Q, T, dtype=np.int64)          # T -> never flagged (whole track kept)
    window = float(max(1e-6, window))
    half = window / 2.0

    for t in range(1, T, max(1, int(stride))):
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
        disp = np.linalg.norm(P - prev[idx_valid].astype(np.float64), axis=1)

        bins = {}
        for i, p in enumerate(P):
            bx = int(np.floor(float(p[0]) / window))
            by = int(np.floor(float(p[1]) / window))
            bins.setdefault((bx, by), []).append(i)

        for i, p in enumerate(P):
            bx = int(np.floor(float(p[0]) / window))
            by = int(np.floor(float(p[1]) / window))
            cand = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    cand.extend(bins.get((bx + dx, by + dy), []))
            if not cand:
                continue

            cand = np.asarray(cand, dtype=np.int64)
            dx_local = np.abs(P[cand, 0] - p[0])
            dy_local = np.abs(P[cand, 1] - p[1])
            in_win = (dx_local <= half) & (dy_local <= half)
            cand = cand[in_win]
            cand = cand[cand != i]
            if cand.size == 0:
                continue

            med_disp = float(np.median(disp[cand]))
            rel_err = float(disp[i] / (med_disp + 1e-6))
            dev = abs(rel_err - 1.0)
            if dev > eps:
                q_idx = idx_valid[i]
                if t < first_red[q_idx]:      # frames scanned in order -> keep the earliest red flag
                    first_red[q_idx] = t

    return first_red

def _draw_cross(img: np.ndarray, center_xy: tuple[int, int], color: tuple[int, int, int], radius: int = 4, thickness: int = 1):
    x, y = int(center_xy[0]), int(center_xy[1])
    cv2.line(img, (x - radius, y - radius), (x + radius, y + radius), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(img, (x - radius, y + radius), (x + radius, y - radius), color, thickness, lineType=cv2.LINE_AA)


def _write_dynamic_reason_viz_mp4(
    video_np: np.ndarray,
    tracks_xy: np.ndarray,
    occluded_final: np.ndarray,
    out_path: str,
    *,
    raw_occluded: np.ndarray,
    dynamic_obs_mask: np.ndarray,
    depth_grad_deleted_obs_mask: np.ndarray,
    depth_max_deleted_obs_mask: np.ndarray,
    move_delete_query_mask: np.ndarray,
    query_t: np.ndarray,
    move_delete_first_frame: np.ndarray | None = None,
    fps: int = 10,
):
    assert video_np.ndim == 5 and video_np.shape[0] == 1, "Expected video_np [1,T,H,W,3]"
    _, T_local, H_local, W_local, _ = video_np.shape
    _, Q_local, T_tracks, _ = tracks_xy.shape
    assert T_local == T_tracks, "video and tracks must share the same number of frames"

    out_path = str(out_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, float(fps), (int(W_local), int(H_local)))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_path}")

    move_delete_query_mask = np.asarray(move_delete_query_mask, dtype=bool).reshape(-1)
    if move_delete_first_frame is not None:
        move_delete_first_frame = np.asarray(move_delete_first_frame, dtype=np.int64).reshape(-1)
    depth_deleted_obs_mask = np.asarray(depth_grad_deleted_obs_mask | depth_max_deleted_obs_mask, dtype=bool)
    query_t = np.asarray(query_t, dtype=np.int64).reshape(-1)

    color_visible = (60, 220, 60)
    color_invisible = (255, 255, 255)
    color_depth = (0, 165, 255)
    color_move = (40, 40, 255)

    for t in range(T_local):
        frame_rgb = video_np[0, t]
        if frame_rgb.dtype != np.uint8:
            if frame_rgb.max() <= 1.5:
                frame_rgb = np.clip(frame_rgb * 255.0, 0, 255).astype(np.uint8)
            else:
                frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        cv2.putText(frame_bgr, "filled green: visible", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color_visible, 1, cv2.LINE_AA)
        cv2.putText(frame_bgr, "hollow white: temporarily invisible", (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color_invisible, 1, cv2.LINE_AA)
        cv2.putText(frame_bgr, "orange X: depth deleted", (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color_depth, 1, cv2.LINE_AA)
        cv2.putText(frame_bgr, "red X: large-2D-move (truncated from red flag)", (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color_move, 1, cv2.LINE_AA)

        for q in range(Q_local):
            if t < int(query_t[q]):
                continue
            xy = tracks_xy[0, q, t]
            if not np.isfinite(xy).all():
                continue
            x = int(round(float(xy[0])))
            y = int(round(float(xy[1])))
            if x < 0 or x >= W_local or y < 0 or y >= H_local:
                continue

            if bool(move_delete_query_mask[q]) and (
                move_delete_first_frame is None or t >= int(move_delete_first_frame[q])
            ):
                _draw_cross(frame_bgr, (x, y), color_move, radius=5, thickness=2)
                continue

            if bool(depth_deleted_obs_mask[q, t]):
                _draw_cross(frame_bgr, (x, y), color_depth, radius=4, thickness=2)
                continue

            if bool(raw_occluded[0, q, t]):
                cv2.circle(frame_bgr, (x, y), 4, color_invisible, 1, lineType=cv2.LINE_AA)
                continue

            if not bool(dynamic_obs_mask[q, t]):
                cv2.circle(frame_bgr, (x, y), 3, (180, 180, 180), 1, lineType=cv2.LINE_AA)
                continue

            if not bool(occluded_final[0, q, t]):
                cv2.circle(frame_bgr, (x, y), 3, color_visible, -1, lineType=cv2.LINE_AA)
            else:
                cv2.circle(frame_bgr, (x, y), 4, color_invisible, 1, lineType=cv2.LINE_AA)

        writer.write(frame_bgr)

    writer.release()


def _expand_state_for_new_queries(state, n_add: int):
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


def _subset_state_for_queries(state, keep_idx: np.ndarray, prefix_tokens: int = 0):
    if state is None:
        return state

    keep_idx = np.asarray(keep_idx, dtype=np.int64).reshape(-1)
    keep_idx_t = torch.as_tensor(keep_idx, dtype=torch.long)
    prefix_tokens = int(max(0, prefix_tokens))

    if getattr(state, "query_points", None) is not None:
        state.query_points = state.query_points[:, keep_idx_t.to(state.query_points.device), :]

    if getattr(state, "hidden_state", None) is None:
        return state

    reduced = []
    for cache in state.hidden_state:
        if cache is None:
            reduced.append(cache)
            continue

        rg_device_idx = keep_idx_t.to(cache.rg_lru_state.device)
        cv_device_idx = keep_idx_t.to(cache.conv1d_state.device)

        if prefix_tokens > 0:
            rg_prefix = cache.rg_lru_state[:prefix_tokens]
            cv_prefix = cache.conv1d_state[:prefix_tokens]
            rg_queries = cache.rg_lru_state[prefix_tokens:].index_select(0, rg_device_idx)
            cv_queries = cache.conv1d_state[prefix_tokens:].index_select(0, cv_device_idx)
            rg = torch.cat([rg_prefix, rg_queries], dim=0)
            cv = torch.cat([cv_prefix, cv_queries], dim=0)
        else:
            rg = cache.rg_lru_state.index_select(0, rg_device_idx)
            cv = cache.conv1d_state.index_select(0, cv_device_idx)

        if hasattr(cache, "_replace"):
            cache_new = cache._replace(rg_lru_state=rg, conv1d_state=cv)
        else:
            cache_new = type(cache)(rg_lru_state=rg, conv1d_state=cv)
        reduced.append(cache_new)

    state.hidden_state = reduced
    return state


def _large_move_flag_single(cur, prev, valid, window, eps):
    """(N,) bool: a point's 2D displacement (cur-prev) deviates from the median of its spatial neighbours
    (within a `window`-side square) by more than `eps`. False where invalid. Single-frame version of the
    move-delete red flag, used online for max-points eviction."""
    N = cur.shape[0]
    flag = np.zeros(N, bool)
    idx = np.where(valid & np.isfinite(cur).all(1) & np.isfinite(prev).all(1))[0]
    if idx.size < 2:
        return flag
    P = cur[idx].astype(np.float64)
    disp = np.linalg.norm(P - prev[idx].astype(np.float64), axis=1)
    half = float(window) / 2.0
    bins = {}
    for i, p in enumerate(P):
        bins.setdefault((int(p[0] // window), int(p[1] // window)), []).append(i)
    for i, p in enumerate(P):
        bx, by = int(p[0] // window), int(p[1] // window)
        cand = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cand.extend(bins.get((bx + dx, by + dy), []))
        if not cand:
            continue
        cand = np.asarray(cand, np.int64)
        cand = cand[(np.abs(P[cand, 0] - p[0]) <= half) & (np.abs(P[cand, 1] - p[1]) <= half)]
        cand = cand[cand != i]
        if cand.size == 0:
            continue
        med = float(np.median(disp[cand]))
        if abs(disp[i] / (med + 1e-6) - 1.0) > eps:
            flag[idx[i]] = True
    return flag


def _maxpoints_bad_eviction(active_gids, tpf, opf, aipf, masks256, offmask_px, past, bad_min,
                            invis_consec, move_window, move_eps):
    """KEEP boolean (per active point) for max-points eviction. Evict a point if it was BAD in > bad_min
    of the past `past` frames (BAD = off the dilated dynamic mask OR large-2D-move vs neighbours), OR it
    was invisible for the last `invis_consec` consecutive frames. History is read from the per-frame
    buffers (global-id indexed via aipf), so the online hot path is untouched (this runs only at max)."""
    Qa = int(active_gids.shape[0])
    Tprev = len(opf)
    if Tprev == 0:
        return np.ones(Qa, bool)
    frames = list(range(max(0, Tprev - int(max(past, invis_consec))), Tprev))
    nf = len(frames)
    ker = None
    if masks256 is not None and offmask_px > 0:
        r = max(1, int(round(offmask_px)))
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    bad = np.zeros((Qa, nf), bool)
    invis = np.ones((Qa, nf), bool)
    for fi, tt in enumerate(frames):
        aidx = np.asarray(aipf[tt]).reshape(-1)
        if aidx.size == 0:
            continue
        pos = np.clip(np.searchsorted(aidx, active_gids), 0, aidx.size - 1)
        present = aidx[pos] == active_gids
        occ = np.asarray(opf[tt]).reshape(-1)
        cur = np.full((Qa, 2), np.nan)
        cur[present] = np.asarray(tpf[tt])[pos[present]]
        occ_tt = np.ones(Qa, bool)
        occ_tt[present] = occ[pos[present]]
        invis[:, fi] = occ_tt | (~present)
        fin = present & np.isfinite(cur).all(1)
        offmask = np.zeros(Qa, bool)
        if ker is not None and tt < masks256.shape[0]:
            m = masks256[tt]; m = m[..., 0] if m.ndim == 3 else m
            md = cv2.dilate((m > 0).astype(np.uint8), ker) > 0
            Hm, Wm = md.shape
            onm = np.zeros(Qa, bool)
            xi = np.rint(cur[:, 0]); yi = np.rint(cur[:, 1])
            inb = fin & (xi >= 0) & (xi < Wm) & (yi >= 0) & (yi < Hm)
            if np.any(inb):
                onm[inb] = md[yi[inb].astype(np.int64), xi[inb].astype(np.int64)]
            offmask = fin & (~onm)
        largemove = np.zeros(Qa, bool)
        if tt - 1 >= 0:
            aidxp = np.asarray(aipf[tt - 1]).reshape(-1)
            if aidxp.size:
                posp = np.clip(np.searchsorted(aidxp, active_gids), 0, aidxp.size - 1)
                presp = aidxp[posp] == active_gids
                prev = np.full((Qa, 2), np.nan)
                prev[presp] = np.asarray(tpf[tt - 1])[posp[presp]]
                largemove = _large_move_flag_single(cur, prev, fin & presp, move_window, move_eps)
        bad[:, fi] = offmask | largemove
    evict = (bad[:, -past:] if nf >= past else bad).sum(1) > int(bad_min)
    if nf >= invis_consec:
        evict |= invis[:, -invis_consec:].all(1)
    return ~evict


@torch.no_grad()
def track_online_tapnext(
    video_256: torch.Tensor,
    query_points_np: np.ndarray,
    device: torch.device,
    checkpoint: str,
    *,
    add_points: bool = True,
    add_every: int = 1,
    add_start: int = 1,
    keep_frac: float = 1.0,
    resample_margin: int = 8,
    resample_radius: float = 10.0,
    swap_xy_for_query: bool = False,
    use_certainty: bool = True,
    certainty_radius: int = 8,
    certainty_threshold: float = 0.5,
    dynamic_masks_256: np.ndarray | None = None,
    max_points: int = 6000,
    downsample_threshold: int | None = None,
    downsample_target_ratio: float = 0.5,
    maxpt_offmask_px256: float = 10.0,
    maxpt_past: int = 10,
    maxpt_bad_min: int = 7,
    maxpt_invis_consec: int = 20,
    maxpt_move_window: float = 6.0,
    maxpt_move_eps: float = 1.0,
):
    from tapnet.tapnext.tapnext_torch import TAPNext   # here: tapnet's package import pulls in tensorflow (~3 s)
    from tapnet.tapnext.tapnext_torch_utils import restore_model_from_jax_checkpoint, tracker_certainty

    if checkpoint.endswith(".npz"):
        model = TAPNext(image_size=(256, 256)).to(device).eval()
        model = restore_model_from_jax_checkpoint(model, checkpoint)
    else:
        model = TAPNext(image_size=(256, 256))
        ckpt = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict({k.replace("tapnext.", ""): v for k, v in ckpt["state_dict"].items()})
        model.cuda().eval()

    num_video_tokens = (
        (model.image_size[0] // model.patch_size[0])
        * (model.image_size[1] // model.patch_size[1])
        * int(video_256.shape[0])
    )

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
    prev_added_xy_history = deque(maxlen=5)
    last_tracks_xy = None
    last_occ_np = None
    last_active_query_indices = None

    if downsample_threshold is not None and downsample_threshold <= 0:
        downsample_threshold = None

    for t in tqdm(range(T), desc="Online tracking dynamic"):
        query_grew = False
        n_add_effective = 0

        if downsample_threshold is not None:
            q_active = int(active_query_indices.shape[0])
            if q_active > downsample_threshold:
                q_keep = max(1, int(np.floor(downsample_target_ratio * downsample_threshold)))
                q_keep = min(q_keep, q_active)
                keep_local = np.sort(np.random.choice(q_active, size=q_keep, replace=False))
                keep_local_t = torch.as_tensor(keep_local, dtype=torch.long, device=query_points_t.device)
                active_query_indices = active_query_indices[keep_local]
                query_points_t = query_points_t.index_select(1, keep_local_t)
                state = _subset_state_for_queries(state, keep_local, prefix_tokens=num_video_tokens)
                if last_tracks_xy is not None:
                    last_tracks_xy = last_tracks_xy[keep_local]
                if last_occ_np is not None:
                    last_occ_np = last_occ_np[keep_local]
                if last_active_query_indices is not None:
                    last_active_query_indices = active_query_indices.copy()
                prev_added_xy_history.clear()
                print(f"[online] timestep {t}: downsampled active queries from {q_active} to {q_keep}")

        # Once the active set reaches max_points, evict only the points that are clearly BAD (off the
        # dynamic mask by >maxpt_offmask_px256 OR large-2D-move vs neighbours in >maxpt_bad_min of the past
        # maxpt_past frames, OR invisible for the last maxpt_invis_consec consecutive frames), instead of
        # dropping every currently-invisible point -- frees capacity while keeping temporarily-hidden points.
        if last_occ_np is not None and int(active_query_indices.shape[0]) >= int(max_points):
            n_active = int(active_query_indices.shape[0])
            keep_bool = _maxpoints_bad_eviction(
                active_query_indices, tracks_per_frame, occ_per_frame, active_indices_per_frame,
                dynamic_masks_256, maxpt_offmask_px256, maxpt_past, maxpt_bad_min, maxpt_invis_consec,
                maxpt_move_window, maxpt_move_eps)
            keep_local = np.where(keep_bool)[0]
            if 0 < keep_local.size < n_active:
                keep_local = np.sort(keep_local)
                keep_local_t = torch.as_tensor(keep_local, dtype=torch.long, device=query_points_t.device)
                active_query_indices = active_query_indices[keep_local]
                query_points_t = query_points_t.index_select(1, keep_local_t)
                state = _subset_state_for_queries(state, keep_local, prefix_tokens=num_video_tokens)
                if last_tracks_xy is not None:
                    last_tracks_xy = last_tracks_xy[keep_local]
                last_occ_np = last_occ_np[keep_local]
                last_active_query_indices = active_query_indices.copy()
                prev_added_xy_history.clear()
                print(f"[online] timestep {t}: max_points ({n_active}); evicted "
                      f"{n_active - keep_local.size} bad, kept {keep_local.size}")

        if (
            add_points
            and (t >= add_start)
            and (t + 1 < T)
            and (add_every > 0)
            and (t % add_every == 0)
        ):
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
                mask_256=(dynamic_masks_256[t + 1] if dynamic_masks_256 is not None else None),
                keep_frac=keep_frac,
                margin=resample_margin,
                radius=resample_radius,
                prev_added_xy=prev_added_xy,
                swap_xy_for_query=swap_xy_for_query,
                rng=np.random.default_rng(1000 + t),
            )
            if new_q is not None:
                new_q[:, :, 0] = float(t + 1)
                if dynamic_masks_256 is not None:
                    new_q = _exclude_queries_outside_dynamic_masks(
                        new_q,
                        dynamic_masks_256,
                        swap_xy_for_query=swap_xy_for_query,
                    )
                if new_q is not None and new_q.shape[1] > 0:
                    q_total = int(all_query_points_abs.shape[1])
                    q_free = int(max_points - q_curr)
                    if q_free <= 0:
                        # Full and no invisible points to evict this frame; skip adding
                        # now but keep additions enabled so we retry once eviction frees room.
                        pass
                    else:
                        n_add = int(min(q_free, new_q.shape[1]))
                        if n_add < new_q.shape[1]:
                            print(f"max_points limit: adding {n_add}/{new_q.shape[1]} new queries this frame")
                        add_np = new_q[:, :n_add, :].astype(np.float64, copy=False)
                        new_ids = np.arange(q_total, q_total + n_add, dtype=np.int64)
                        all_query_points_abs = np.concatenate([all_query_points_abs, add_np], axis=1)
                        active_query_indices = np.concatenate([active_query_indices, new_ids])
                        add_t = torch.from_numpy(add_np).float().to(device)
                        query_points_t = torch.cat([query_points_t, add_t], dim=1)

                        add_xy = add_np[0, :, 1:3]
                        if swap_xy_for_query:
                            add_xy = add_xy[:, [1, 0]]
                        prev_added_xy_history.append(add_xy.astype(np.float64, copy=False))

                        query_grew = n_add > 0
                        n_add_effective = n_add

        if query_grew and state is not None:
            state.query_points = query_points_t
            state = _expand_state_for_new_queries(state, n_add_effective)

        if state is None:
            pred_tracks, track_logits, visible_logits, state = model(
                video=video_256[:, t : t + 1], query_points=query_points_t
            )
        else:
            state.query_points = query_points_t
            pred_tracks, track_logits, visible_logits, state = model(
                video=video_256[:, t : t + 1], state=state
            )

        tracks_yx = pred_tracks.detach().cpu()
        tracks_xy = torch.flip(tracks_yx, dims=[-1]).numpy()[0, 0]

        if use_certainty:
            certainty = tracker_certainty(tracks_yx, track_logits.detach().cpu(), certainty_radius)
            pred_visible = (torch.sigmoid(visible_logits.detach().cpu()) * certainty) > certainty_threshold
            occluded = ~(pred_visible.squeeze(0).squeeze(0))
        else:
            pred_visible = visible_logits.detach().cpu() > 0
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

    return tracks_xy_full, occluded_full, query_t


if TRACKER == "tapnext":
    from utils.track_tapnet import load_frames_256
    video_np_256, video_np = load_frames_256(str(IMAGE_DIR), IMAGE_PATTERN)
else:                                                     # AllTracker resizes the full-size frames itself
    _paths = sorted(glob.glob(os.path.join(str(IMAGE_DIR), IMAGE_PATTERN)), key=_numeric_key)
    if not _paths:
        raise FileNotFoundError(f"No frames found in {IMAGE_DIR} matching {IMAGE_PATTERN}")
    with ThreadPoolExecutor(8) as _ex:                    # as load_frames_256 decodes them, in threads
        video_np = np.stack(list(_ex.map(lambda p: np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8), _paths)))[None]
video_np_full = video_np                                  # kept in ORIGINAL order (viz + output flip-back)
H, W = video_np.shape[2], video_np.shape[3]
T = video_np.shape[1]
if REVERSE and TRACKER == "tapnext":
    video_np_256 = video_np_256[:, ::-1].copy()          # track on the time-flipped video (masks flipped below)

if MASK_DIR is None or not Path(MASK_DIR).exists():
    raise FileNotFoundError(f"Mask directory not found: {MASK_DIR}")

dynamic_masks_base = load_dynamic_masks(
    str(MASK_DIR),
    MASK_PATTERN,
    t_expected=T,
    mask_mode=MASK_MODE,
    target_size=(W, H),
)
dynamic_masks_query_256 = load_dynamic_masks(
    str(MASK_DIR),
    MASK_PATTERN,
    t_expected=T,
    mask_mode=MASK_MODE,
    target_size=(256, 256),
)
dynamic_masks_query_256 = (
    dilate_dynamic_masks(dynamic_masks_query_256, DILATE_PX_QUERY)
    if DILATE_PX_QUERY != 0
    else dynamic_masks_query_256
)
dynamic_masks_vis = (
    dilate_dynamic_masks(dynamic_masks_base, DILATE_PX_VIS)
    if DILATE_PX_VIS != 0
    else dynamic_masks_base
)
if REVERSE:                                               # flip masks in time so the reversed body is consistent
    dynamic_masks_base = dynamic_masks_base[::-1].copy()
    dynamic_masks_query_256 = dynamic_masks_query_256[::-1].copy()
    dynamic_masks_vis = dynamic_masks_vis[::-1].copy()

if TRACKER == "tapnext":
    query_np = _make_mask_query_points(dynamic_masks_query_256[0], keep_frac=DENSITY_FRAC, query_t=0)
    print(f"Density: DYN_MASK_KEEP_FRAC={DENSITY_FRAC:.2f} (kept at frame 0 AND every online resample); "
          f"coverage radius={ONLINE_RESAMPLE_RADIUS:.2f}px(@256)")
    if query_np is None or query_np.shape[1] == 0:
        raise RuntimeError("No initial dynamic query points generated from the first mask.")
    print(f"Initial dynamic queries: {query_np.shape[1]}")

    video_256 = torch.from_numpy(video_np_256).to(device)
    tracks_xy_256, occluded, query_t = track_online_tapnext(
        video_256,
        query_np,
        device=device,
        checkpoint=str(CHECKPOINT),
        add_points=ONLINE_ADD_POINTS,
        add_every=ONLINE_ADD_EVERY,
        add_start=ONLINE_ADD_START,
        keep_frac=DENSITY_FRAC,
        resample_margin=ONLINE_RESAMPLE_MARGIN,
        resample_radius=ONLINE_RESAMPLE_RADIUS,
        swap_xy_for_query=RESAMPLE_SWAP_XY,
        use_certainty=ONLINE_USE_CERTAINTY,
        certainty_radius=ONLINE_CERTAINTY_RADIUS,
        certainty_threshold=ONLINE_CERTAINTY_THRESHOLD,
        dynamic_masks_256=dynamic_masks_query_256,
        max_points=ONLINE_MAX_POINTS,
        downsample_threshold=ONLINE_QUERY_DOWNSAMPLE_THRESHOLD,
        downsample_target_ratio=ONLINE_QUERY_DOWNSAMPLE_TARGET_RATIO,
        maxpt_offmask_px256=MAXPT_OFFMASK_PX * 256.0 / float(max(H, W)),   # 10px native -> 256-grid
        maxpt_past=MAXPT_PAST,
        maxpt_bad_min=MAXPT_BAD_MIN,
        maxpt_invis_consec=MAXPT_INVIS_CONSEC,
        maxpt_move_window=MOVE_WINDOW_SIZE * 256.0 / float(max(H, W)),
        maxpt_move_eps=float(MOVE_DELETE_TOP_PERCENT) / 100.0,
    )

    tracks_xy = tracks_xy_256.astype(np.float64, copy=True)
    tracks_xy[..., 0] = tracks_xy[..., 0] / 256.0 * W
    tracks_xy[..., 1] = tracks_xy[..., 1] / 256.0 * H
else:
    from utils.track_alltracker import grid_seeds, track_keyframes

    dyn_masks_seed = dilate_dynamic_masks(dynamic_masks_base, DILATE_PX_QUERY) if DILATE_PX_QUERY != 0 else dynamic_masks_base
    dyn_masks_seed = np.asarray(dyn_masks_seed)
    if dyn_masks_seed.ndim == 4:
        dyn_masks_seed = dyn_masks_seed[..., 0]
    spacing = DYN_AT_GRID * max(H, W) / float(AT_RES)                  # grid spacing in native px
    frames_track = video_np[0][::-1].copy() if REVERSE else video_np[0]
    print(f"AllTracker: seeds every {spacing:.1f}px on the object mask, keyframe every {AT_KEYFRAME} frames, "
          f"{AT_RES}px, visible at vis x conf >= {AT_VIS_THRESHOLD}")
    trk, vis, query_t = track_keyframes(
        frames_track,
        lambda t, covered: grid_seeds(dyn_masks_seed[t] > 0, spacing, covered, spacing),
        keyframe=AT_KEYFRAME, res=AT_RES, vis_threshold=AT_VIS_THRESHOLD, max_points=ONLINE_MAX_POINTS,
        min_new=AT_MIN_NEW, fp16=AT_FP16, device=str(device), label="AllTracker (dynamic)",
    )
    if trk.shape[0] == 0:
        raise RuntimeError("No dynamic points: the object mask is empty at every keyframe.")
    tracks_xy = trk[None].astype(np.float64)
    occluded = ~vis[None]
    print(f"AllTracker dynamic tracks: {trk.shape[0]} from {len(np.unique(query_t))} keyframes")

Q = tracks_xy.shape[1]
dynamic_obs_mask = np.zeros((Q, T), dtype=bool)
for t in range(T):
    mask_t = dynamic_masks_vis[t]
    if mask_t.ndim == 3:
        mask_t = mask_t[..., 0]
    x_t = tracks_xy[0, :, t, 0]
    y_t = tracks_xy[0, :, t, 1]
    valid_t = np.isfinite(x_t) & np.isfinite(y_t)
    if not np.any(valid_t):
        continue
    xi = np.rint(x_t[valid_t]).astype(np.int64)
    yi = np.rint(y_t[valid_t]).astype(np.int64)
    in_bounds = (xi >= 0) & (xi < mask_t.shape[1]) & (yi >= 0) & (yi < mask_t.shape[0])
    if not np.any(in_bounds):
        continue
    idx_valid = np.where(valid_t)[0][in_bounds]
    dynamic_obs_mask[idx_valid, t] = mask_t[yi[in_bounds], xi[in_bounds]] > 0

occluded_dynamic = occluded.copy()
occluded_dynamic[0] |= ~dynamic_obs_mask

# if EDGE_ERODE_PX > 0:
#     edge_obs_mask = np.zeros((Q, T), dtype=bool)
#     k = 2 * int(EDGE_ERODE_PX) + 1
#     kernel = np.ones((k, k), np.uint8)
#     for t in range(T):
#         mask_t = dynamic_masks_vis[t]
#         if mask_t.ndim == 3:
#             mask_t = mask_t[..., 0]
#         m = (mask_t > 0).astype(np.uint8)
#         if m.size == 0:
#             continue
#         eroded = cv2.erode(m, kernel, iterations=1)
#         edge = (m > 0) & (eroded == 0)

#         x_t = tracks_xy[0, :, t, 0]
#         y_t = tracks_xy[0, :, t, 1]
#         valid_t = np.isfinite(x_t) & np.isfinite(y_t)
#         xi = np.rint(x_t[valid_t]).astype(np.int64)
#         yi = np.rint(y_t[valid_t]).astype(np.int64)
#         in_bounds = (xi >= 0) & (xi < edge.shape[1]) & (yi >= 0) & (yi < edge.shape[0])
#         if not np.any(in_bounds):
#             continue
#         idx_valid = np.where(valid_t)[0][in_bounds]
#         edge_obs_mask[idx_valid, t] = edge[yi[in_bounds], xi[in_bounds]]

#     occluded_dynamic[0] |= edge_obs_mask
#     print(f"Filtered edge observations: {int(np.sum(edge_obs_mask))}")


# run_pipeline.sh starts this step next to step 1: wait for the depth (its marker) before the 3D lift
_depth_done = os.environ.get("XM_WAIT_FOR_DEPTH")
if _depth_done and not os.path.exists(_depth_done):
    pause_script_logger("waiting for step 1's depth")              # (not this step's runtime)
    while not os.path.exists(_depth_done):
        if os.path.exists(_depth_done + ".failed"):
            raise RuntimeError("step 1 (depth) failed, so there is no depth to lift the tracks with")
        time.sleep(1)
    resume_script_logger()
gt = GT_LOADERS[dataset_cat]()

depth_paths = sorted(glob.glob(os.path.join(str(GT_DEPTH_DIR), "*.dpt")))
if REVERSE:
    depth_paths = depth_paths[::-1]
depth_source = DEPTH_SOURCE.strip().lower()
depth_grad_deleted_obs_mask = np.zeros((Q, T), dtype=bool)
depth_max_deleted_obs_mask = np.zeros((Q, T), dtype=bool)
tracks_xyz = None

if depth_source == "gt":
    if dataset_cat == "Replica":
        T_lift = T
    elif not depth_paths:
        print(f"No GT depth files found in {GT_DEPTH_DIR}; skipping 3D lift")
        T_lift = 0
    else:
        T_lift = min(T, len(depth_paths))
elif depth_source == "da3":
    if not DEPTH_MAPS_PATH.exists():
        print(f"No DA3 depth file found at {DEPTH_MAPS_PATH}; skipping 3D lift")
        T_lift = 0
    else:
        depth_maps = np.load(DEPTH_MAPS_PATH)
        if REVERSE:
            depth_maps = depth_maps[::-1].copy()
        if depth_maps.ndim != 3:
            raise ValueError(f"Expected depth_maps.npy shape (T,H,W), got {depth_maps.shape}")
        T_lift = min(T, depth_maps.shape[0])
        print(f"Loaded DA3 depth maps: {depth_maps.shape} from {DEPTH_MAPS_PATH}")
else:
    raise ValueError(f"Unsupported DEPTH_SOURCE='{DEPTH_SOURCE}'. Use 'gt' or 'da3'.")

if T_lift > 0:
    image_paths = sorted(glob.glob(os.path.join(str(IMAGE_DIR), IMAGE_PATTERN)))
    if REVERSE:
        image_paths = image_paths[::-1]
    if len(image_paths) < T_lift:
        print("Not enough image frames to match video; skipping 3D lift")
    else:
        tracks_xyz = np.full((1, Q, T_lift, 3), np.nan, dtype=np.float64)
        prev_depths = np.full(Q, np.nan, dtype=np.float64)

        # Single intrinsic source: da3 depth is lifted with intrinsics.npy (identical
        # to the static solver); only the "gt" depth source keeps GT intrinsics.
        K_all_da3 = None
        if depth_source != "gt":
            _intr_path = DA3_OUT_DIR / "intrinsics.npy"
            if _intr_path.exists():
                K_all_da3 = np.load(_intr_path).astype(np.float64)
                if REVERSE:
                    K_all_da3 = K_all_da3[::-1].copy()

        for t in range(T_lift):
            img_name = os.path.basename(image_paths[t])
            if img_name not in gt:
                continue

            if K_all_da3 is not None and t < K_all_da3.shape[0]:
                K = K_all_da3[t]
            else:
                K = gt[img_name]["K"]
            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])

            if depth_source == "gt":
                if dataset_cat == "Replica":
                    depth = read_replica_depth(str(GT_DEPTH_DIR / img_name.replace(".jpg", ".png"))).astype(np.float64)
                else:
                    depth = depth_read(depth_paths[t]).astype(np.float64)
            else:
                depth = depth_maps[t].astype(np.float64)

            H_s, W_s = depth.shape
            x_scale = W_s / float(W)
            y_scale = H_s / float(H)

            grad_x = cv2.Sobel(depth, cv2.CV_64F, 2, 0, ksize=3)
            grad_y = cv2.Sobel(depth, cv2.CV_64F, 0, 2, ksize=3)
            grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)

            x = tracks_xy[0, :, t, 0]
            y = tracks_xy[0, :, t, 1]
            x_depth = x * x_scale
            y_depth = y * y_scale
            valid = (~occluded_dynamic[0, :, t]) & np.isfinite(x) & np.isfinite(y)

            xi = np.rint(x_depth).astype(np.int64)
            yi = np.rint(y_depth).astype(np.int64)
            in_bounds = (xi >= 0) & (xi < W_s) & (yi >= 0) & (yi < H_s)
            valid &= in_bounds
            if not np.any(valid):
                continue

            idx_valid = np.where(valid)[0]
            z = depth[yi[valid], xi[valid]]
            grad = grad_mag[yi[valid], xi[valid]]

            finite_positive = np.isfinite(z) & (z > 0)
            max_ok = (z <= float(DEPTH_MAX)) if DEPTH_MAX > 0 else np.ones_like(z, dtype=bool)
            basic_depth_ok = finite_positive & max_ok
            grad_ok = (grad <= float(DEPTH_GRAD_THRESHOLD)) if DEPTH_GRAD_THRESHOLD > 0 else np.ones_like(grad, dtype=bool)

            depth_max_deleted_local = finite_positive & (~max_ok)
            if np.any(depth_max_deleted_local):
                depth_max_deleted_obs_mask[idx_valid[depth_max_deleted_local], t] = True

            grad_deleted_local = basic_depth_ok & (~grad_ok)
            if np.any(grad_deleted_local):
                depth_grad_deleted_obs_mask[idx_valid[grad_deleted_local], t] = True

            valid_depth = basic_depth_ok & grad_ok
            if not np.any(valid_depth):
                continue

            idx = idx_valid[valid_depth]
            zv = z[valid_depth].astype(np.float64)
            if DEPTH_JUMP_THRESHOLD is not None and float(DEPTH_JUMP_THRESHOLD) > 0:
                prev = prev_depths[idx]
                jump_ok = np.isnan(prev) | (np.abs(zv - prev) <= float(DEPTH_JUMP_THRESHOLD))
                if not np.any(jump_ok):
                    continue
                idx = idx[jump_ok]
                zv = zv[jump_ok]

            xv = x_depth[idx]
            yv = y_depth[idx]
            X = (xv - cx) / fx * zv
            Y = (yv - cy) / fy * zv
            Z = zv
            tracks_xyz[0, idx, t, :] = np.stack([X, Y, Z], axis=1).astype(np.float64)
            prev_depths[idx] = zv

        if np.any(depth_grad_deleted_obs_mask):
            occluded_dynamic[0] |= depth_grad_deleted_obs_mask
        if np.any(depth_max_deleted_obs_mask):
            occluded_dynamic[0] |= depth_max_deleted_obs_mask

        print(f"Filtered depth-gradient observations: {int(np.sum(depth_grad_deleted_obs_mask))}")
        print(f"Filtered depth-max observations: {int(np.sum(depth_max_deleted_obs_mask))}")

if MOVE_DELETE_ENABLED:
    move_delete_first_frame = _compute_move_delete_first_frame(
        tracks_xy=tracks_xy,
        occluded_obs=occluded_dynamic,
        window=MOVE_WINDOW_SIZE,
        top_percent=MOVE_DELETE_TOP_PERCENT,
        stride=MOVE_DELETE_STRIDE,
    )
    move_delete_query_mask = move_delete_first_frame < T          # track had >= 1 red flag
    n_move_delete = int(np.sum(move_delete_query_mask))
    # TRUNCATE at the first red flag: keep the good prefix [0, first_red), occlude+NaN from first_red on
    # (instead of deleting the whole track). A large 2D jump usually means the tracker lost its target,
    # so the prefix is trustworthy but the suffix is not.
    n_obs_removed = 0
    for q in np.where(move_delete_query_mask)[0]:
        fr = int(move_delete_first_frame[q])
        occluded_dynamic[0, q, fr:] = True
        if tracks_xyz is not None:
            tracks_xyz[0, q, fr:, :] = np.nan
        n_obs_removed += (T - fr)
    print(
        f"Large-2D-move: truncated {n_move_delete}/{Q} tracks at their first red flag (kept the prefix); "
        f"removed {n_obs_removed} obs vs {n_move_delete * T} if whole-track "
        f"(saved {n_move_delete * T - n_obs_removed}). window={MOVE_WINDOW_SIZE:.1f}px, "
        f"top_percent={MOVE_DELETE_TOP_PERCENT:.1f}%"
    )
else:
    move_delete_query_mask = np.zeros(Q, dtype=bool)
    move_delete_first_frame = np.full(Q, T, dtype=np.int64)

# --- lost-off-object deletion: invisible (tracker-occluded) AND out of the dynamic mask for MORE THAN
#     LOST_OFFMASK_MAX_FRAMES consecutive frames -> the point left the object, so TRUNCATE the track from
#     where that run began (keep the good prefix, drop the rest incl. any spurious reappearance). ---
lost_offmask_first_frame = np.full(Q, T, dtype=np.int64)
if LOST_OFFMASK_MAX_FRAMES > 0 and T > LOST_OFFMASK_MAX_FRAMES:
    _exist = np.arange(T, dtype=np.int64)[None, :] >= query_t[:, None]
    _lost = _exist & occluded[0] & (~dynamic_obs_mask)           # invisible AND off-mask (existing frames)
    L = int(LOST_OFFMASK_MAX_FRAMES)
    run = np.ones((Q, T - L), dtype=bool)                        # run[q,s] = frames s..s+L all lost (L+1 = >L)
    for k in range(L + 1):
        run &= _lost[:, k:T - L + k]
    has = run.any(axis=1)
    lost_offmask_first_frame[has] = run.argmax(axis=1)[has]      # start of first run longer than L frames
    n_lost = int(has.sum())
    n_lost_obs = 0
    for q in np.where(has)[0]:
        fr = int(lost_offmask_first_frame[q])
        occluded_dynamic[0, q, fr:] = True
        if tracks_xyz is not None:
            tracks_xyz[0, q, fr:, :] = np.nan
        n_lost_obs += (T - fr)
    print(f"Lost-off-object: truncated {n_lost}/{Q} tracks invisible+off-mask > {L} consecutive frames "
          f"(removed {n_lost_obs} obs from where they were lost).")

# --- per-source occlusion breakdown: how many observations each source flips to occluded=True ---
# Sources OVERLAP (a point can be off-mask AND depth-deleted), so per-source counts sum to more than
# the final ANY count. Counts are over "existing" observations only (frames at/after each track's birth).
_ti = np.arange(T, dtype=np.int64)[None, :]
_pre = _ti < query_t[:, None]                                    # (Q,T) pre-birth frames
_ex = ~_pre
_nex = int(_ex.sum())
_mv = _ti >= move_delete_first_frame[:, None]   # occluded from the first red flag onward (truncation)
_lo = _ti >= lost_offmask_first_frame[:, None]  # occluded from where the point was lost off the object
_sources = [
    ("visibility(certainty)", occluded[0] & _ex),               # tracker sigmoid*certainty <= threshold
    ("off-dynamic-mask", (~dynamic_obs_mask) & _ex),
    ("depth-gradient", depth_grad_deleted_obs_mask & _ex),
    ("depth-max", depth_max_deleted_obs_mask & _ex),
    ("large-2D-move", _mv & _ex),
    ("lost-off-object", _lo & _ex),
]
print(f"Occlusion breakdown over {_nex} existing observations (sources overlap; they sum to >= final):")
for _nm, _m in _sources:
    _c = int(_m.sum())
    print(f"  {_nm:22s}: {_c:9d}  ({100.0 * _c / max(1, _nex):5.1f}%)")
_final = int((occluded_dynamic[0] & _ex).sum())
print(f"  {'=> final occluded (ANY)':22s}: {_final:9d}  ({100.0 * _final / max(1, _nex):5.1f}%)")
print(f"  (pre-query frames, always occluded: {int(_pre.sum())})")

if REVERSE:
    # flip every output back to ORIGINAL frame order (the body ran entirely in reversed time)
    tracks_xy = tracks_xy[:, :, ::-1, :].copy()
    if tracks_xyz is not None:
        tracks_xyz = tracks_xyz[:, :, ::-1, :].copy()
    occluded = occluded[:, :, ::-1].copy()
    occluded_dynamic = occluded_dynamic[:, :, ::-1].copy()
    dynamic_obs_mask = dynamic_obs_mask[:, ::-1].copy()
    depth_grad_deleted_obs_mask = depth_grad_deleted_obs_mask[:, ::-1].copy()
    depth_max_deleted_obs_mask = depth_max_deleted_obs_mask[:, ::-1].copy()
    _fin = np.isfinite(tracks_xy[0]).all(-1)                      # query_t = first EXISTING original frame
    query_t = np.where(_fin.any(1), _fin.argmax(1), 0).astype(np.int64)
    # truncation frame-indices were in reversed time; map to original (viz metadata only, data already flipped)
    move_delete_first_frame = np.where(move_delete_first_frame < T,
                                       (T - 1) - move_delete_first_frame, T).astype(np.int64)
    lost_offmask_first_frame = np.where(lost_offmask_first_frame < T,
                                        (T - 1) - lost_offmask_first_frame, T).astype(np.int64)
    print(f"[reverse] flipped outputs back to original frame order")

savez_compressed_fast(                                   # zlib level 1: same arrays, ~3x faster than numpy's level 6
    OUT_DIR / OUT_NAME,
    tracks_xy=tracks_xy,
    tracks_xyz=tracks_xyz,
    occluded=occluded_dynamic,
    dynamic_obs_mask=dynamic_obs_mask,
    depth_grad_deleted_obs_mask=depth_grad_deleted_obs_mask,
    depth_max_deleted_obs_mask=depth_max_deleted_obs_mask,
    move_delete_query_mask=move_delete_query_mask,
    move_delete_first_frame=move_delete_first_frame.astype(np.int32),
    lost_offmask_first_frame=lost_offmask_first_frame.astype(np.int32),
    query_t=query_t.astype(np.int32),
    video_dir=str(IMAGE_DIR),
    mask_dir=str(MASK_DIR),
)
print(f"Wrote: {OUT_DIR / OUT_NAME}")
if DEBUG:
    _viz_name = "viz_" + (OUT_NAME[:-4] if OUT_NAME.endswith('.npz') else OUT_NAME) + ".mp4"

    _write_dynamic_reason_viz_mp4(
        video_np_full,
        tracks_xy,
        occluded_dynamic,
        str(OUT_DIR / _viz_name),
        raw_occluded=occluded,
        dynamic_obs_mask=dynamic_obs_mask,
        depth_grad_deleted_obs_mask=depth_grad_deleted_obs_mask,
        depth_max_deleted_obs_mask=depth_max_deleted_obs_mask,
        move_delete_query_mask=move_delete_query_mask,
        move_delete_first_frame=move_delete_first_frame,
        query_t=query_t,
        fps=10,
    )
    print(f"Wrote: {OUT_DIR / _viz_name}")
