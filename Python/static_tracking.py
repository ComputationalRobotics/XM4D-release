#!/usr/bin/env python3
"""Run static TAPNext tracking and 2D-to-3D lifting for XM datasets.

Paths and dataset settings come from utils.definitions / XM_* environment
variables.
"""

from pathlib import Path
import glob
import os
import numpy as np
import torch

import cv2

from utils.readgt_sintel import load_sintel_gt, depth_read
from utils.readgt_DAVIS import load_davis_gt
from utils.readgt_TUM import load_tum_gt
from utils.readgt_custom import load_custom_gt
from utils.track_tapnet import (
    apply_dynamic_mask_filter,
    dilate_dynamic_masks,
    load_dynamic_masks,
    load_frames_256,
    make_grid_query_points,
    track_full_video_tapnext,
    write_viz_mp4,
)

import utils.definitions

### S0: CONFIG
dataset_cat = utils.definitions.dataset_cat
rootpath = utils.definitions.rootpath
DEPTH_SOURCE = utils.definitions.depth_source  # "gt" or "da3"
DEPTH_MAX = 20.0
DEPTH_GRAD_THRESHOLD = 2.0  # depth gradient threshold (in depth units per pixel)
DEPTH_JUMP_THRESHOLD = None  # Optional max allowed |z_t - z_{t-1}| per track.

IMAGE_DIR = utils.definitions.video_dir
MASK_DIR = utils.definitions.mask_dir  # set to None or non-existent path if no masks
OUT_DIR = utils.definitions.output_dir
DA3_OUT_DIR = utils.definitions.da3_depth_dir
GT_DEPTH_DIR = utils.definitions.gt_depth_dir

IMAGE_PATTERN = utils.definitions.image_pattern
MASK_PATTERN = utils.definitions.mask_pattern

GRID = 50
MAX_QUERY_POINTS = 5000
SEGMENT_LEN = 200
RESAMPLE_BLANK = True
RESAMPLE_GRID = 40
RESAMPLE_MARGIN = 8
RESAMPLE_STEP = 10 if dataset_cat == "TUM" else 1  # larger step for faster motion; try 10–30
RESAMPLE_COVERAGE_RADIUS = 15.0  # pixels in 256-space; add points farther than this from tracked points

# If blank-region resampling appears mirrored (e.g., top-right becomes bottom-left),
# enable this to emit resample queries as (t, y, x) instead of (t, x, y).
RESAMPLE_SWAP_XY = True
MOVE_DELETE_ENABLED = True
MOVE_WINDOW_SIZE = 10.0
MOVE_DELETE_FLOW_ERROR_THRESHOLD = 50.0  # Pixel threshold for local flow-vector deviation.
SMOOTH_3D_ENABLED = False
SMOOTH_3D_WINDOW = 10  # moving-average window (odd preferred)
CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "deps"
    / "tapnet"
    / "checkpoints"
    / "tapnextpp_ckpt.pt"
)

# bootstapnext_ckpt.npz, tapnextpp_ckpt.pt

OUT_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if dataset_cat == "Sintel":
    gt = load_sintel_gt(str(rootpath))
elif dataset_cat == "TUM":
    gt = load_tum_gt(str(rootpath))
elif dataset_cat == "DAVIS":
    gt = load_davis_gt(str(rootpath))
elif dataset_cat == "Custom":
    gt = load_custom_gt(rootpath)
else:
    raise ValueError(f"Unsupported dataset_cat '{dataset_cat}'")

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
    n_removed = int(np.sum(~keep))
    if n_removed > 0:
        print(f"Excluded {n_removed}/{q.shape[0]} query points inside dynamic masks")
    if q_kept.shape[0] == 0:
        print("All query points were inside dynamic masks; keeping original queries.")
        return query_np
    return q_kept[None, ...]


def _query_keep_mask(query_np, masks_256, swap_xy_for_query=False):
    if query_np is None or query_np.size == 0 or masks_256 is None:
        return None
    q = query_np[0]
    if q.shape[0] == 0:
        return None
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
    return keep


def _downsample_queries(query_np, max_points, carry_idx=None, seed=0):
    if query_np is None or query_np.size == 0:
        return query_np, carry_idx
    q = query_np.shape[1]
    if q <= max_points:
        return query_np, carry_idx

    rng = np.random.default_rng(seed)

    if carry_idx is None or carry_idx.size == 0:
        sel = rng.choice(q, size=max_points, replace=False)
        sel.sort()
        return query_np[:, sel, :], carry_idx

    # Carry points correspond to the current query ordering.
    if carry_idx.size >= max_points:
        sel = rng.choice(carry_idx.size, size=max_points, replace=False)
        sel.sort()
        return query_np[:, sel, :], carry_idx[sel]

    # Keep all carry points; sample remaining from extras.
    keep_carry = np.arange(carry_idx.size)
    extra = np.arange(carry_idx.size, q)
    extra_needed = max_points - carry_idx.size
    sel_extra = rng.choice(extra, size=extra_needed, replace=False)
    sel = np.concatenate([keep_carry, sel_extra])
    sel.sort()
    return query_np[:, sel, :], carry_idx


def _make_uncovered_query_points(
    tracks_xy_256,
    occluded,
    grid,
    margin=8,
    step=1,
    start_t=1,
    radius=10.0,
    swap_xy_for_query=False,
    seed: int | None = None,
):
    """Create query points in regions not covered by first-pass tracked points.

    Uses random candidate sampling per frame (candidate count ~= grid^2).
    A candidate is selected if it is farther than `radius` from:
      1) visible tracked points at frame t, and
      2) points newly selected at the previous sampled frame.
    """
    T = tracks_xy_256.shape[2]
    n_candidates = int(max(1, int(grid) * int(grid)))
    rng = np.random.default_rng(seed)

    radius2 = float(radius) * float(radius)
    all_pts = []
    prev_added_xy = None
    for t in range(int(start_t), T, max(1, int(step))):
        x0, x1 = float(margin), float(255 - margin)
        y0, y1 = float(margin), float(255 - margin)
        if x1 <= x0 or y1 <= y0:
            break

        cand_x = rng.uniform(x0, x1, size=n_candidates).astype(np.float64)
        cand_y = rng.uniform(y0, y1, size=n_candidates).astype(np.float64)
        cand = np.stack([cand_x, cand_y], axis=1)  # [N,2] as (x,y)

        pts = tracks_xy_256[0, :, t, :]
        valid = np.isfinite(pts).all(axis=1) & (~occluded[0, :, t])

        ref_sets = []
        if np.any(valid):
            ref_sets.append(pts[valid].astype(np.float64))
        if prev_added_xy is not None and prev_added_xy.shape[0] > 0:
            ref_sets.append(prev_added_xy.astype(np.float64))

        if ref_sets:
            refs = np.concatenate(ref_sets, axis=0)
            d2 = np.sum((cand[:, None, :] - refs[None, :, :]) ** 2, axis=2)
            min_d2 = np.min(d2, axis=1)
            mask_uncovered = min_d2 > radius2
        else:
            mask_uncovered = np.ones(cand.shape[0], dtype=bool)

        t_arr = np.full(np.sum(mask_uncovered), float(t), dtype=np.float64)
        x = cand[mask_uncovered, 0]
        y = cand[mask_uncovered, 1]
        if swap_xy_for_query:
            all_pts.append(np.stack([t_arr, y, x], axis=1))
        else:
            all_pts.append(np.stack([t_arr, x, y], axis=1))

    if not all_pts:
        return None
    return np.concatenate(all_pts, axis=0)[None, ...]


def _smooth_tracks_xyz_visible_segments(
    tracks_xyz: np.ndarray,
    occluded: np.ndarray,
    window: int = 5,
):
    """Smooth 3D trajectories on contiguous visible segments.

    Only interior points in each segment are smoothed; segment endpoints are preserved.
    """
    if tracks_xyz is None:
        return tracks_xyz

    w = int(max(1, window))
    if w % 2 == 0:
        w += 1

    smoothed = tracks_xyz.copy()
    _, Q, T, _ = tracks_xyz.shape

    for q in range(Q):
        vis = (~occluded[0, q, :]) & np.isfinite(tracks_xyz[0, q, :, :]).all(axis=1)
        if not np.any(vis):
            continue

        idx = np.where(vis)[0]
        if idx.size == 0:
            continue

        split_pos = np.where(np.diff(idx) > 1)[0]
        starts = np.concatenate(([0], split_pos + 1))
        ends = np.concatenate((split_pos, [idx.size - 1]))

        for s, e in zip(starts, ends):
            seg_t = idx[s : e + 1]
            n = seg_t.size
            if n <= 2:
                continue

            seg_xyz = tracks_xyz[0, q, seg_t, :]  # [n,3]
            half = w // 2

            # keep endpoints unchanged; smooth interior only
            for i in range(1, n - 1):
                l = max(0, i - half)
                r = min(n, i + half + 1)
                smoothed[0, q, seg_t[i], :] = np.mean(seg_xyz[l:r, :], axis=0)

            smoothed[0, q, seg_t[0], :] = seg_xyz[0]
            smoothed[0, q, seg_t[-1], :] = seg_xyz[-1]

    return smoothed


video_np_256, video_np = load_frames_256(str(IMAGE_DIR), IMAGE_PATTERN)
H, W = video_np.shape[2], video_np.shape[3]

query_exclude_masks_256 = None
if MASK_DIR is not None and Path(MASK_DIR).exists():
    try:
        _masks_256 = load_dynamic_masks(
            str(MASK_DIR),
            MASK_PATTERN,
            t_expected=video_np.shape[1],
            mask_mode="auto",
            target_size=(256, 256),
        )
        _masks_256 = np.asarray(_masks_256)
        if _masks_256.ndim == 4 and _masks_256.shape[-1] == 1:
            _masks_256 = _masks_256[..., 0]
        if _masks_256.ndim == 4 and _masks_256.shape[0] == 1:
            _masks_256 = _masks_256[0]
        if _masks_256.ndim == 3:
            query_exclude_masks_256 = _masks_256
    except Exception as e:
        print(f"Failed to load 256 dynamic masks for query exclusion: {e}")

video_256 = torch.from_numpy(video_np_256).to(device)

T_total = video_np.shape[1]
segment_starts = list(range(0, T_total, SEGMENT_LEN))

tracks_xy_256 = None
occluded = None
global_q = 0

for seg_idx, start_t in enumerate(segment_starts):
    end_t = min(start_t + SEGMENT_LEN, T_total)
    seg_len = end_t - start_t
    print(f"Tracking segment {seg_idx + 1}/{len(segment_starts)}: frames {start_t}:{end_t}")

    seg_video = video_256[:, start_t:end_t]
    seg_masks_256 = None
    if query_exclude_masks_256 is not None:
        seg_masks_256 = query_exclude_masks_256[start_t:end_t]

    carry_idx = None
    if seg_idx == 0:
        query_np = make_grid_query_points(GRID, step=10)
    else:
        last_tracks = tracks_xy_256[0, :, start_t - 1, :]
        last_occ = occluded[0, :, start_t - 1]
        valid = np.isfinite(last_tracks).all(axis=1) & (~last_occ)
        carry_idx = np.where(valid)[0]
        carry = last_tracks[valid]
        if carry.shape[0] == 0:
            query_np = make_grid_query_points(GRID, step=10)
        else:
            t0 = np.zeros((carry.shape[0], 1), dtype=np.float64)
            carry_xy = carry.astype(np.float64)
            if RESAMPLE_SWAP_XY:
                carry_xy = carry_xy[:, [1, 0]]
            query_np = np.concatenate([t0, carry_xy], axis=1)[None, ...]

    if seg_masks_256 is not None:
        keep = _query_keep_mask(query_np, seg_masks_256, swap_xy_for_query=False)
        if keep is not None:
            query_np = query_np[:, keep, :]
            if carry_idx is not None:
                carry_idx = carry_idx[keep]
    if query_np.shape[1] == 0:
        query_np = make_grid_query_points(GRID, step=10)
        carry_idx = None

    query_np, carry_idx = _downsample_queries(
        query_np,
        MAX_QUERY_POINTS,
        carry_idx=carry_idx,
        seed=seg_idx,
    )

    query = torch.from_numpy(query_np).float().to(device)
    seg_tracks_xy_256, seg_occluded = track_full_video_tapnext(
        seg_video, query, device=device, checkpoint=str(CHECKPOINT)
    )

    if RESAMPLE_BLANK:
        blank_query_np = _make_uncovered_query_points(
            seg_tracks_xy_256,
            seg_occluded,
            RESAMPLE_GRID,
            margin=RESAMPLE_MARGIN,
            step=RESAMPLE_STEP,
            start_t=1,
            radius=RESAMPLE_COVERAGE_RADIUS,
            swap_xy_for_query=RESAMPLE_SWAP_XY,
        )
        if seg_masks_256 is not None and blank_query_np is not None:
            blank_query_np = _exclude_queries_in_dynamic_masks(
                blank_query_np,
                seg_masks_256,
                swap_xy_for_query=RESAMPLE_SWAP_XY,
            )

        base_query_np = query_np
        query_parts = [base_query_np]
        if blank_query_np is not None and blank_query_np.shape[1] > 0:
            query_parts.append(blank_query_np)

        if len(query_parts) > 1:
            query_np_2 = np.concatenate(query_parts, axis=1)
            query_2 = torch.from_numpy(query_np_2).float().to(device)
            seg_tracks_xy_256, seg_occluded = track_full_video_tapnext(
                seg_video, query_2, device=device, checkpoint=str(CHECKPOINT)
            )
            print(
                "Resampled queries: "
                f"blank={0 if blank_query_np is None else blank_query_np.shape[1]}"
            )
        else:
            print("No blank queries to resample")

    seg_q = seg_tracks_xy_256.shape[1]
    if tracks_xy_256 is None:
        tracks_xy_256 = np.full((1, seg_q, T_total, 2), np.nan, dtype=np.float32)
        occluded = np.ones((1, seg_q, T_total), dtype=bool)
        tracks_xy_256[0, :, start_t:end_t, :] = seg_tracks_xy_256[0]
        occluded[0, :, start_t:end_t] = seg_occluded[0]
        global_q = seg_q
    else:
        # Carry-over points occupy the indices from last segment's visible set.
        if carry_idx is None:
            carry_idx = np.arange(0)
        q_carry = carry_idx.shape[0]
        if q_carry > 0:
            tracks_xy_256[0, carry_idx, start_t:end_t, :] = seg_tracks_xy_256[0, :q_carry]
            occluded[0, carry_idx, start_t:end_t] = seg_occluded[0, :q_carry]

        new_q = seg_q - q_carry
        if new_q > 0:
            tracks_xy_256 = np.concatenate(
                [
                    tracks_xy_256,
                    np.full((1, new_q, T_total, 2), np.nan, dtype=np.float32),
                ],
                axis=1,
            )
            occluded = np.concatenate(
                [occluded, np.ones((1, new_q, T_total), dtype=bool)], axis=1
            )
            tracks_xy_256[0, global_q:global_q + new_q, start_t:end_t, :] = seg_tracks_xy_256[0, q_carry:]
            occluded[0, global_q:global_q + new_q, start_t:end_t] = seg_occluded[0, q_carry:]
            global_q += new_q

# Map tracks from 256-grid space back to original image space
tracks_xy = tracks_xy_256
tracks_xy[..., 0] = tracks_xy[..., 0] / 256 * W
tracks_xy[..., 1] = tracks_xy[..., 1] / 256 * H

# Optional dynamic masks
dynamic_masks = None
if MASK_DIR is not None and Path(MASK_DIR).exists():
    try:
        # Load masks at the original image size so we can apply them directly on tracks_xy
        dynamic_masks = load_dynamic_masks(
            str(MASK_DIR),
            "*.png",
            t_expected=video_np.shape[1],
            mask_mode="auto",
            target_size=(W, H),
        )
        dynamic_masks = dilate_dynamic_masks(dynamic_masks, 10)
        occluded_before = occluded.copy()

        # tracks_xy is in original image coordinates (W,H).
        occluded = apply_dynamic_mask_filter(tracks_xy, occluded, dynamic_masks, static_only=True)
        print(
            f"Applied masks from {MASK_DIR}. Newly hidden: "
            f"{int(np.sum(occluded) - np.sum(occluded_before))}"
        )
    except Exception as e:
        print(f"Failed to load/apply masks from {MASK_DIR}: {e}")

# Save 2D outputs
np.savez_compressed(
    OUT_DIR / "tracks_debug.npz",
    tracks_xy=tracks_xy,
    occluded=occluded,
    video_dir=str(IMAGE_DIR),
    grid=int(GRID),
    mask_dir=str(MASK_DIR) if MASK_DIR is not None else "",
)

# Lift 2D tracks to 3D using Depth-Anything v3 outputs
# Expected files are produced by deps/Depth-Anything-3/get_depths*.py
DEPTH_MAPS_PATH = DA3_OUT_DIR / "depth_maps.npy"
CONF_MAPS_PATH = DA3_OUT_DIR / "conf_maps.npy"
depth_paths = sorted(glob.glob(os.path.join(str(GT_DEPTH_DIR), "*.dpt")))
conf_maps = None

tracks_xyz = None
move_delete_query_mask = None
occluded_viz = occluded.copy()
depth_grad_deleted_obs_mask = None
depth_max_deleted_obs_mask = None
depth_source = DEPTH_SOURCE.strip().lower()
if depth_source == "gt":
    if not depth_paths:
        print(f"No GT depth files found in {GT_DEPTH_DIR}; skipping 3D lift")
    T = min(video_np.shape[1], len(depth_paths))
elif depth_source == "da3":
    if not DEPTH_MAPS_PATH.exists():
        print(f"No DA3 depth file found at {DEPTH_MAPS_PATH}; skipping 3D lift")
        T = 0
    else:
        depth_maps = np.load(DEPTH_MAPS_PATH)
        conf_maps = np.load(CONF_MAPS_PATH) if CONF_MAPS_PATH.exists() else None

        if depth_maps.ndim != 3:
            raise ValueError(f"Expected depth_maps.npy shape (T,H,W), got {depth_maps.shape}")

        T_depth, H_depth, W_depth = depth_maps.shape
        T = min(video_np.shape[1], T_depth)
        print(f"Loaded DA3 depth maps: {depth_maps.shape} from {DEPTH_MAPS_PATH}")
        if conf_maps is not None:
            print(f"Loaded DA3 confidence maps: {conf_maps.shape} from {CONF_MAPS_PATH}")
else:
    raise ValueError(f"Unsupported DEPTH_SOURCE='{DEPTH_SOURCE}'. Use 'gt' or 'da3'.")

if T > 0:
    Q = tracks_xy.shape[1]
    tracks_xyz = np.full((1, Q, T, 3), np.nan, dtype=np.float64)
    conf_xyz = np.full((Q, T), np.nan, dtype=np.float64)
    depth_grad_deleted_obs_mask = np.zeros((1, Q, T), dtype=bool)
    depth_max_deleted_obs_mask = np.zeros((1, Q, T), dtype=bool)
    prev_depths = np.full(Q, np.nan, dtype=np.float64)

    image_paths = sorted(glob.glob(os.path.join(str(IMAGE_DIR), IMAGE_PATTERN)))
    if len(image_paths) < T:
        print("Not enough image frames to match video; skipping 3D lift")
    else:
        for t in range(T):
            img_name = os.path.basename(image_paths[t])
            if img_name not in gt:
                continue

            K = gt[img_name]["K"]
            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])

            if depth_source == "gt":
                depth = depth_read(depth_paths[t])
                H_s, W_s = depth.shape
                x_scale = W_s / float(W)
                y_scale = H_s / float(H)
            else:
                depth = depth_maps[t]
                H_s, W_s = depth.shape
                # Tracks are in original-image coords; map to DA3 depth-map coords.
                x_scale = W_s / float(W)
                y_scale = H_s / float(H)

            # Depth gradient magnitude for boundary filtering
            depth_f = depth.astype(np.float64)
            grad_x = cv2.Sobel(depth_f, cv2.CV_64F, 2, 0, ksize=3)
            grad_y = cv2.Sobel(depth_f, cv2.CV_64F, 0, 2, ksize=3)
            grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)

            x = tracks_xy[0, :, t, 0]
            y = tracks_xy[0, :, t, 1]
            x_depth = x * x_scale
            y_depth = y * y_scale
            valid = (~occluded[0, :, t]) & np.isfinite(x) & np.isfinite(y)

            xi = np.rint(x_depth).astype(np.int64)
            yi = np.rint(y_depth).astype(np.int64)
            in_bounds = (xi >= 0) & (xi < W_s) & (yi >= 0) & (yi < H_s)
            valid &= in_bounds
            if not np.any(valid):
                continue

            z = depth[yi[valid], xi[valid]]
            grad = grad_mag[yi[valid], xi[valid]]
            conf_vals = None
            if depth_source == "da3" and conf_maps is not None:
                conf_vals = conf_maps[t][yi[valid], xi[valid]].astype(np.float64)
            
            finite_positive = np.isfinite(z) & (z > 0)
            max_ok = z <= float(DEPTH_MAX)
            basic_depth_ok = finite_positive & max_ok
            grad_ok = grad <= float(DEPTH_GRAD_THRESHOLD)

            # mark observations deleted specifically by DEPTH_MAX filtering
            depth_max_deleted_local = finite_positive & (~max_ok)
            if np.any(depth_max_deleted_local):
                idx_valid = np.where(valid)[0]
                idx_depth_max_deleted = idx_valid[depth_max_deleted_local]
                depth_max_deleted_obs_mask[0, idx_depth_max_deleted, t] = True

            # mark observations deleted specifically by depth-gradient filtering
            grad_deleted_local = basic_depth_ok & (~grad_ok)
            if np.any(grad_deleted_local):
                idx_valid = np.where(valid)[0]
                idx_grad_deleted = idx_valid[grad_deleted_local]
                depth_grad_deleted_obs_mask[0, idx_grad_deleted, t] = True

            valid_depth = basic_depth_ok & grad_ok
            if not np.any(valid_depth):
                continue

            idx = np.where(valid)[0][valid_depth]
            zv = z[valid_depth].astype(np.float64)
            cv = None
            if conf_vals is not None:
                cv = conf_vals[valid_depth].astype(np.float64)

            # Depth jump filtering per track
            if DEPTH_JUMP_THRESHOLD is not None and float(DEPTH_JUMP_THRESHOLD) > 0:
                prev = prev_depths[idx]
                jump_ok = np.isnan(prev) | (np.abs(zv - prev) <= float(DEPTH_JUMP_THRESHOLD))
                if not np.any(jump_ok):
                    continue
                idx = idx[jump_ok]
                zv = zv[jump_ok]
                if cv is not None:
                    cv = cv[jump_ok]

            xv = x_depth[idx]
            yv = y_depth[idx]
            # print(xv,yv,zv)

            X = (xv - cx) / fx * zv
            Y = (yv - cy) / fy * zv
            Z = zv

            # Camera-frame points
            Pc = np.stack([X, Y, Z], axis=1)  # [N, 3]

            tracks_xyz[0, idx, t, :] = Pc.astype(np.float64)
            if cv is not None:
                conf_xyz[idx, t] = cv
            prev_depths[idx] = zv
            
    print("Deleted observations by depth-gradient filter:", int(np.sum(depth_grad_deleted_obs_mask)))
    print("Deleted observations by depth-max filter:", int(np.sum(depth_max_deleted_obs_mask)))

    if tracks_xyz is not None:
        # Delete points whose 2D flow disagrees with nearby points.
        if MOVE_DELETE_ENABLED:
            move_delete_query_mask = np.zeros(Q, dtype=bool)
            window = float(max(1e-6, MOVE_WINDOW_SIZE))
            half = window / 2.0
            flow_error_threshold = float(max(0.0, MOVE_DELETE_FLOW_ERROR_THRESHOLD))
            move_flow_error_score = np.full(Q, np.nan, dtype=np.float64)

            for t in range(1, T):
                pts = tracks_xy[0, :, t, :]
                prev = tracks_xy[0, :, t - 1, :]

                valid = (
                    np.isfinite(pts).all(axis=1)
                    & np.isfinite(prev).all(axis=1)
                    & (~occluded[0, :, t])
                    & (~occluded[0, :, t - 1])
                )
                idx_valid = np.where(valid)[0]
                if idx_valid.size < 2:
                    continue

                P = pts[idx_valid].astype(np.float64)
                flow = P - prev[idx_valid].astype(np.float64)

                # Spatial bins for faster local-neighborhood lookup
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

                    neighbor_flow = np.median(flow[cand], axis=0)
                    dev = float(np.linalg.norm(flow[i] - neighbor_flow))
                    q_idx = idx_valid[i]
                    if not np.isfinite(move_flow_error_score[q_idx]) or dev > float(move_flow_error_score[q_idx]):
                        move_flow_error_score[q_idx] = np.float64(dev)

            finite_move = np.isfinite(move_flow_error_score)
            if np.any(finite_move):
                move_delete_query_mask = finite_move & (move_flow_error_score > flow_error_threshold)

            n_move_delete = int(np.sum(move_delete_query_mask))
            if n_move_delete > 0:
                occluded[0, move_delete_query_mask, :] = True
                tracks_xyz[0, move_delete_query_mask, :, :] = np.nan

            print(
                f"Deleted large-2D-motion points: {n_move_delete}/{Q} "
                f"(flow error > {flow_error_threshold:.3f}px, window={window:.1f}px)"
            )

        if SMOOTH_3D_ENABLED:
            tracks_xyz = _smooth_tracks_xyz_visible_segments(
                tracks_xyz,
                occluded,
                window=SMOOTH_3D_WINDOW,
            )
            print(f"Applied 3D trajectory smoothing on visible segments (window={int(SMOOTH_3D_WINDOW)})")

        np.savez_compressed(
            OUT_DIR / "tracks_debug_3d.npz",
            tracks_xyz=tracks_xyz,
            video_dir=str(IMAGE_DIR),
            depth_dir=str(GT_DEPTH_DIR) if depth_source == "gt" else str(DA3_OUT_DIR),
            grid=int(GRID),
        )
        print(f"Wrote: {OUT_DIR / 'tracks_debug_3d.npz'}")

        # Build edges/weights/landmarks for XM input
        edges_list = []
        weights_list = []
        landmarks_list = []

        for t in range(T):
            pts_cam = tracks_xyz[0, :, t, :]

            valid_cam = np.isfinite(pts_cam).all(axis=1) & (~occluded[0, :, t])
            if not np.any(valid_cam):
                continue

            idx = np.where(valid_cam)[0]
            for i in idx:
                edges_list.append([int(i), int(t)])  # [landmark_index, frame_index]
                landmarks_list.append(pts_cam[i].tolist())
                if depth_source == "da3" and conf_maps is not None:
                    c = conf_xyz[i, t]
                    w = float(c) if np.isfinite(c) else 1.0
                else:
                    w = 1.0
                weights_list.append([w])

        edges = np.asarray(edges_list, dtype=np.int64)
        weights = np.asarray(weights_list, dtype=np.float64)
        landmarks = np.asarray(landmarks_list, dtype=np.float64)

        output_tapnet_src_path = OUT_DIR / f"output_tapnet_{depth_source}.npz"
        np.savez_compressed(
            output_tapnet_src_path,
            edges=edges,
            weights=weights,
            landmarks=landmarks,
        )

        if edges.size > 0:
            lm_idx = edges[:, 0].astype(np.int64)   # landmark index
            fr_idx = edges[:, 1].astype(np.int64)   # frame index

            # optional safety: keep only valid indices
            valid = (fr_idx >= 0) & (fr_idx < T) & (lm_idx >= 0) & (lm_idx < Q)
            fr_idx = fr_idx[valid]
            lm_idx = lm_idx[valid]

            # these (frame, landmark) pairs are NOT occluded
            occluded_viz[0, :, :] = True  # set invalid pairs to occluded
            occluded_viz[0, lm_idx, fr_idx] = False
            
        else:
            print("No valid 3D edges were built; keeping 2D visibility for visualization.")
    
write_viz_mp4(
    video_np,
    tracks_xy,
    occluded_viz,
    str(OUT_DIR / "viz_debug.mp4"),
    fps=10,
    tail_length=12,
    tail_thickness=1,
    dynamic_masks=dynamic_masks,
    highlight_query_mask_always=move_delete_query_mask,
    highlight_query_color_always=(255, 0, 255),
    highlight_radius=5,
    highlight_obs_mask=depth_grad_deleted_obs_mask,
    highlight_obs_color=(0, 0, 255),
    highlight_obs_mask_2=depth_max_deleted_obs_mask,
    highlight_obs_color_2=(0, 165, 255),
)
