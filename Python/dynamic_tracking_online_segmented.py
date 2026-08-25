#!/usr/bin/env python3
"""Segmented online TAPNext tracking for dynamic points only."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import glob
import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deps" / "tapnet"))

from tapnet.tapnext.tapnext_torch import TAPNext
from tapnet.tapnext.tapnext_torch_utils import restore_model_from_jax_checkpoint, tracker_certainty

from utils.readgt_DAVIS import load_davis_gt
from utils.readgt_TUM import load_tum_gt
from utils.readgt_colmap import load_colmap_gt
from utils.readgt_custom import load_custom_gt
from utils.readgt_replica import load_replica_gt, read_replica_depth
from utils.readgt_sintel import load_sintel_gt, depth_read
from utils.readgt_dycheck import load_dycheck_gt
from utils.track_tapnet import (
    dilate_dynamic_masks,
    load_dynamic_masks,
    load_frames_256,
    write_viz_mp4,
)

import utils.definitions
import utils.dataset_config as dataset_config


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

DEPTH_MAX = 20.0
DEPTH_GRAD_THRESHOLD = 2.0
DEPTH_JUMP_THRESHOLD = None

MOVE_DELETE_ENABLED = True
MOVE_WINDOW_SIZE = 10.0
MOVE_DELETE_TOP_PERCENT = 100.0
SMOOTH_3D_ENABLED = False
SMOOTH_3D_WINDOW = 10

MASK_MODE = "auto"
DILATE_PX_QUERY = 0
DILATE_PX_VIS = -5
MASK_SAMPLE_STEP = 2

RESAMPLE_SWAP_XY = True
ONLINE_ADD_POINTS = True
ONLINE_ADD_EVERY = 1
ONLINE_ADD_START = 1
ONLINE_RESAMPLE_GRID = 40
ONLINE_RESAMPLE_MARGIN = 8
ONLINE_RESAMPLE_RADIUS = 1.0
ONLINE_USE_CERTAINTY = True
ONLINE_CERTAINTY_RADIUS = 8
ONLINE_CERTAINTY_THRESHOLD = 0.2
ONLINE_MAX_POINTS = dataset_config.get_int("DYN_ONLINE_MAX_POINTS", 5000, env="DYN_ONLINE_MAX_POINTS")
ONLINE_RESTART_VISIBLE_THRESHOLD = 1000
ONLINE_SEGMENT_DEBUG_DIR = OUT_DIR / "segments_dynamic_online"

CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "deps"
    / "tapnet"
    / "checkpoints"
    / "tapnextpp_ckpt.pt"
)

DEPTH_MAPS_PATH = DA3_OUT_DIR / "depth_maps.npy"

OUT_DIR.mkdir(parents=True, exist_ok=True)
ONLINE_SEGMENT_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if dataset_cat in {"Sintel", "Sintel_debug"}:
    gt = load_sintel_gt(str(rootpath))
elif dataset_cat == "TUM":
    gt = load_tum_gt(str(rootpath))
elif dataset_cat == "DAVIS":
    gt = load_davis_gt(str(rootpath))
elif dataset_cat == "Replica":
    gt = load_replica_gt(str(rootpath))
elif dataset_cat == "Syn4D":
    gt = load_colmap_gt(str(rootpath / "colmap"))
elif dataset_cat == "Dycheck":
    gt = load_dycheck_gt(str(rootpath))
elif dataset_cat == "Custom":
    gt = load_custom_gt(rootpath)
else:
    raise ValueError(f"Unsupported dataset_cat '{dataset_cat}'")


def _make_mask_query_points(mask_256: np.ndarray, step: int = 1, query_t: int = 0) -> np.ndarray | None:
    if mask_256 is None:
        return None
    if mask_256.ndim == 3:
        mask_256 = mask_256[..., 0]
    xs, ys = np.where(mask_256 > 0)
    if step > 1:
        keep = (ys % step == 0) & (xs % step == 0)
        ys = ys[keep]
        xs = xs[keep]
    if ys.size == 0:
        return None
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
        return None
    return q_kept[None, ...]


def _make_uncovered_query_points_at_frame(
    tracks_xy_256,
    occluded,
    frame_idx,
    grid,
    margin=8,
    radius=10.0,
    prev_added_xy: np.ndarray | None = None,
    swap_xy_for_query=False,
):
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
    cand = np.stack([cand_x, cand_y], axis=1)

    radius2 = float(radius) * float(radius)
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

    if not np.any(mask_uncovered):
        return None

    t_arr = np.full(np.sum(mask_uncovered), float(t), dtype=np.float64)
    x = cand[mask_uncovered, 0]
    y = cand[mask_uncovered, 1]
    if swap_xy_for_query:
        return np.stack([t_arr, y, x], axis=1)[None, ...]
    return np.stack([t_arr, x, y], axis=1)[None, ...]


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


def _load_tapnext_model(device: torch.device, checkpoint: str):
    if checkpoint.endswith(".npz"):
        model = TAPNext(image_size=(256, 256)).to(device).eval()
        model = restore_model_from_jax_checkpoint(model, checkpoint)
    else:
        model = TAPNext(image_size=(256, 256))
        ckpt = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict({k.replace("tapnext.", ""): v for k, v in ckpt["state_dict"].items()})
        model = model.to(device).eval()
    return model


@torch.no_grad()
def track_online_tapnext_segment(
    model,
    video_256: torch.Tensor,
    query_points_np: np.ndarray,
    *,
    add_points: bool = True,
    add_every: int = 1,
    add_start: int = 1,
    resample_grid: int = 20,
    resample_margin: int = 8,
    resample_radius: float = 10.0,
    swap_xy_for_query: bool = False,
    use_certainty: bool = True,
    certainty_radius: int = 8,
    certainty_threshold: float = 0.5,
    dynamic_masks_256: np.ndarray | None = None,
    max_active_points: int = 6000,
    restart_visible_threshold: int = 2000,
):
    T = int(video_256.shape[1])
    all_query_points_abs = query_points_np.astype(np.float64, copy=True)
    active_query_indices = np.arange(all_query_points_abs.shape[1], dtype=np.int64)
    query_points_t = torch.from_numpy(all_query_points_abs).float().to(video_256.device)

    state = None
    tracks_per_frame = []
    occ_per_frame = []
    active_indices_per_frame = []
    prev_added_xy_history = deque(maxlen=5)
    last_tracks_xy = None
    last_occ_np = None
    last_active_query_indices = None
    visible_counts = []
    saturation_frame_local = None
    restart_frame_local = None
    additions_enabled = bool(add_points)

    if int(active_query_indices.shape[0]) >= int(max_active_points):
        additions_enabled = False
        saturation_frame_local = 0

    for t in tqdm(range(T), desc="Online segmented dynamic tracking"):
        query_grew = False
        n_add_effective = 0
        q_curr = int(active_query_indices.shape[0])

        if (
            additions_enabled
            and (t >= add_start)
            and (t + 1 < T)
            and (add_every > 0)
            and (t % add_every == 0)
        ):
            prev_added_xy = None
            if len(prev_added_xy_history) > 0:
                prev_added_xy = np.concatenate(list(prev_added_xy_history), axis=0)

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
                swap_xy_for_query=swap_xy_for_query,
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
                    q_free = int(max_active_points - q_curr)
                    if q_free <= 0:
                        additions_enabled = False
                        if saturation_frame_local is None:
                            saturation_frame_local = t
                    else:
                        n_add = int(min(q_free, new_q.shape[1]))
                        add_np = new_q[:, :n_add, :].astype(np.float64, copy=False)
                        if n_add < new_q.shape[1]:
                            additions_enabled = False
                            if saturation_frame_local is None:
                                saturation_frame_local = t

                        new_ids = np.arange(
                            all_query_points_abs.shape[1],
                            all_query_points_abs.shape[1] + n_add,
                            dtype=np.int64,
                        )
                        all_query_points_abs = np.concatenate([all_query_points_abs, add_np], axis=1)
                        active_query_indices = np.concatenate([active_query_indices, new_ids])
                        add_t = torch.from_numpy(add_np).float().to(video_256.device)
                        query_points_t = torch.cat([query_points_t, add_t], dim=1)

                        add_xy = add_np[0, :, 1:3]
                        if swap_xy_for_query:
                            add_xy = add_xy[:, [1, 0]]
                        prev_added_xy_history.append(add_xy.astype(np.float64, copy=False))

                        query_grew = n_add > 0
                        n_add_effective = n_add
                        if int(active_query_indices.shape[0]) >= int(max_active_points):
                            additions_enabled = False
                            if saturation_frame_local is None:
                                saturation_frame_local = t

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
        visible_count = int(np.sum(~occluded_np))
        visible_counts.append(visible_count)

        if (
            saturation_frame_local is not None
            and restart_frame_local is None
            and t > saturation_frame_local
            and visible_count < int(restart_visible_threshold)
        ):
            restart_frame_local = t

        tracks_per_frame.append(tracks_xy)
        occ_per_frame.append(occluded_np)
        active_indices_per_frame.append(active_query_indices.copy())
        last_tracks_xy = tracks_xy.copy()
        last_occ_np = occluded_np.copy()
        last_active_query_indices = active_query_indices.copy()

    q_total = int(all_query_points_abs.shape[1])
    tracks_xy_full = np.full((1, q_total, T, 2), np.nan, dtype=np.float32)
    occluded_full = np.ones((1, q_total, T), dtype=bool)
    for t, (tracks_xy, occ, active_idx) in enumerate(zip(tracks_per_frame, occ_per_frame, active_indices_per_frame)):
        tracks_xy_full[0, active_idx, t, :] = tracks_xy
        occluded_full[0, active_idx, t] = np.asarray(occ, dtype=bool).reshape(-1)

    query_t = np.floor(all_query_points_abs[0, :, 0]).astype(np.int64)
    time_idx = np.arange(T, dtype=np.int64)[None, :]
    occluded_full[0] |= time_idx < query_t[:, None]

    meta = {
        "visible_counts": np.asarray(visible_counts, dtype=np.int64),
        "restart_frame_local": restart_frame_local,
        "saturation_frame_local": saturation_frame_local,
        "num_queries_total": q_total,
        "query_t": query_t,
    }
    return tracks_xy_full, occluded_full, meta


def run_segmented_online_dynamic_tracking(
    model,
    video_256: torch.Tensor,
    video_np: np.ndarray,
    *,
    dynamic_masks_256: np.ndarray,
    dynamic_masks_orig: np.ndarray | None,
    max_active_points: int,
    restart_visible_threshold: int,
):
    T_total = int(video_256.shape[1])
    H = int(video_np.shape[2])
    W = int(video_np.shape[3])

    tracks_xy_global = None
    occluded_global = None
    query_t_global = None
    global_q = 0
    segment_infos = []
    start_t = 0
    seg_idx = 0

    while start_t < T_total:
        seg_video_256 = video_256[:, start_t:]
        seg_video_np = video_np[:, start_t:]
        seg_masks_256 = dynamic_masks_256[start_t:]
        seg_masks_orig = None if dynamic_masks_orig is None else dynamic_masks_orig[start_t:]

        init_query_np = _make_mask_query_points(seg_masks_256[0], step=MASK_SAMPLE_STEP, query_t=0)
        if init_query_np is None or init_query_np.shape[1] == 0:
            print(f"Segment {seg_idx}: no dynamic mask queries at frame {start_t}; stop.")
            break
        if init_query_np.shape[1] > max_active_points:
            rng = np.random.default_rng(seg_idx)
            keep = np.sort(rng.choice(init_query_np.shape[1], size=max_active_points, replace=False))
            init_query_np = init_query_np[:, keep, :]

        print(f"Segment {seg_idx}: start={start_t}, init_dynamic_queries={init_query_np.shape[1]}")

        seg_tracks_256, seg_occ, seg_meta = track_online_tapnext_segment(
            model,
            seg_video_256,
            init_query_np,
            add_points=ONLINE_ADD_POINTS,
            add_every=ONLINE_ADD_EVERY,
            add_start=ONLINE_ADD_START,
            resample_grid=ONLINE_RESAMPLE_GRID,
            resample_margin=ONLINE_RESAMPLE_MARGIN,
            resample_radius=ONLINE_RESAMPLE_RADIUS,
            swap_xy_for_query=RESAMPLE_SWAP_XY,
            use_certainty=ONLINE_USE_CERTAINTY,
            certainty_radius=ONLINE_CERTAINTY_RADIUS,
            certainty_threshold=ONLINE_CERTAINTY_THRESHOLD,
            dynamic_masks_256=seg_masks_256,
            max_active_points=max_active_points,
            restart_visible_threshold=restart_visible_threshold,
        )

        q_seg_total = int(seg_tracks_256.shape[1])
        if tracks_xy_global is None:
            tracks_xy_global = np.full((1, q_seg_total, T_total, 2), np.nan, dtype=np.float32)
            occluded_global = np.ones((1, q_seg_total, T_total), dtype=bool)
            query_t_global = np.full(q_seg_total, -1, dtype=np.int64)
            global_ids = np.arange(q_seg_total, dtype=np.int64)
            global_q = q_seg_total
        else:
            tracks_xy_global = np.concatenate(
                [tracks_xy_global, np.full((1, q_seg_total, T_total, 2), np.nan, dtype=np.float32)],
                axis=1,
            )
            occluded_global = np.concatenate(
                [occluded_global, np.ones((1, q_seg_total, T_total), dtype=bool)],
                axis=1,
            )
            query_t_global = np.concatenate([query_t_global, np.full(q_seg_total, -1, dtype=np.int64)], axis=0)
            global_ids = np.arange(global_q, global_q + q_seg_total, dtype=np.int64)
            global_q += q_seg_total

        tracks_xy_global[0, global_ids, start_t:, :] = seg_tracks_256[0]
        occluded_global[0, global_ids, start_t:] = seg_occ[0]
        query_t_global[global_ids] = start_t + np.asarray(seg_meta["query_t"], dtype=np.int64)

        seg_tracks_out = seg_tracks_256.copy()
        seg_tracks_out[..., 0] = seg_tracks_out[..., 0] / 256.0 * W
        seg_tracks_out[..., 1] = seg_tracks_out[..., 1] / 256.0 * H
        seg_viz_path = ONLINE_SEGMENT_DEBUG_DIR / f"segment_{seg_idx:03d}_start_{start_t:04d}.mp4"
        write_viz_mp4(
            seg_video_np,
            seg_tracks_out,
            seg_occ,
            str(seg_viz_path),
            fps=10,
            tail_length=12,
            tail_thickness=1,
            dynamic_masks=seg_masks_orig,
        )

        segment_infos.append(
            {
                "segment_index": seg_idx,
                "start_t": start_t,
                "query_global_ids": global_ids.copy(),
                "restart_frame_local": seg_meta["restart_frame_local"],
                "saturation_frame_local": seg_meta["saturation_frame_local"],
                "visible_counts": seg_meta["visible_counts"],
                "viz_path": str(seg_viz_path),
            }
        )

        restart_local = seg_meta["restart_frame_local"]
        if restart_local is None:
            break

        next_start_t = start_t + int(restart_local)
        if next_start_t <= start_t or next_start_t >= T_total:
            break
        start_t = next_start_t
        seg_idx += 1

    if tracks_xy_global is None or occluded_global is None or query_t_global is None:
        raise RuntimeError("Segmented dynamic tracking produced no tracks.")

    return tracks_xy_global, occluded_global, query_t_global, segment_infos


def _compute_move_delete_query_mask(tracks_xy: np.ndarray, occluded_obs: np.ndarray, window: float, top_percent: float) -> np.ndarray:
    Q = tracks_xy.shape[1]
    T = tracks_xy.shape[2]
    move_rel_dev_score = np.full(Q, np.nan, dtype=np.float64)
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
            q_idx = idx_valid[i]
            if not np.isfinite(move_rel_dev_score[q_idx]) or dev > float(move_rel_dev_score[q_idx]):
                move_rel_dev_score[q_idx] = np.float64(dev)

    eps = float(np.clip(top_percent, 0.0, 100.0)) / 100.0
    finite_move = np.isfinite(move_rel_dev_score)
    return finite_move & (move_rel_dev_score > eps)


def _smooth_tracks_xyz_visible_segments(tracks_xyz: np.ndarray, occluded: np.ndarray, window: int = 5):
    if tracks_xyz is None:
        return tracks_xyz

    w = int(max(1, window))
    if w % 2 == 0:
        w += 1

    smoothed = tracks_xyz.copy()
    _, Q, _, _ = tracks_xyz.shape
    for q in range(Q):
        vis = (~occluded[0, q, :]) & np.isfinite(tracks_xyz[0, q, :, :]).all(axis=1)
        if not np.any(vis):
            continue
        idx = np.where(vis)[0]
        split_pos = np.where(np.diff(idx) > 1)[0]
        starts = np.concatenate(([0], split_pos + 1))
        ends = np.concatenate((split_pos, [idx.size - 1]))
        for s, e in zip(starts, ends):
            seg_t = idx[s : e + 1]
            n = seg_t.size
            if n <= 2:
                continue
            seg_xyz = tracks_xyz[0, q, seg_t, :]
            half = w // 2
            for i in range(1, n - 1):
                l = max(0, i - half)
                r = min(n, i + half + 1)
                smoothed[0, q, seg_t[i], :] = np.mean(seg_xyz[l:r, :], axis=0)
            smoothed[0, q, seg_t[0], :] = seg_xyz[0]
            smoothed[0, q, seg_t[-1], :] = seg_xyz[-1]
    return smoothed


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
    fps: int = 10,
):
    """Per-point filter-reason overlay (same scheme as dynamic_tracking.py):
      filled green = visible/kept, hollow white = temporarily invisible (tracker),
      gray = outside dynamic mask, orange X = depth deleted, red X = large-2D-move deleted.
    """
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
        cv2.putText(frame_bgr, "gray: outside dynamic mask", (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame_bgr, "orange X: depth deleted", (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color_depth, 1, cv2.LINE_AA)
        cv2.putText(frame_bgr, "red X: large-2D-move deleted", (12, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color_move, 1, cv2.LINE_AA)

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

            if bool(move_delete_query_mask[q]):
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


video_np_256, video_np = load_frames_256(str(IMAGE_DIR), IMAGE_PATTERN)
video_np_full = video_np
H, W = video_np.shape[2], video_np.shape[3]
T = video_np.shape[1]

if MASK_DIR is None or not Path(MASK_DIR).exists():
    raise FileNotFoundError(f"Mask directory not found: {MASK_DIR}")

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
dynamic_masks_vis = load_dynamic_masks(
    str(MASK_DIR),
    MASK_PATTERN,
    t_expected=T,
    mask_mode=MASK_MODE,
    target_size=(W, H),
)
dynamic_masks_vis = (
    dilate_dynamic_masks(dynamic_masks_vis, DILATE_PX_VIS)
    if DILATE_PX_VIS != 0
    else dynamic_masks_vis
)

video_256 = torch.from_numpy(video_np_256).to(device)
model = _load_tapnext_model(device, str(CHECKPOINT))
tracks_xy_256, occluded, query_t, segment_infos = run_segmented_online_dynamic_tracking(
    model,
    video_256,
    video_np,
    dynamic_masks_256=dynamic_masks_query_256,
    dynamic_masks_orig=dynamic_masks_vis,
    max_active_points=ONLINE_MAX_POINTS,
    restart_visible_threshold=ONLINE_RESTART_VISIBLE_THRESHOLD,
)

np.savez_compressed(
    OUT_DIR / "segment_dynamic_debug.npz",
    segment_start=np.asarray([info["start_t"] for info in segment_infos], dtype=np.int64),
    segment_restart_local=np.asarray(
        [-1 if info["restart_frame_local"] is None else info["restart_frame_local"] for info in segment_infos],
        dtype=np.int64,
    ),
    segment_saturation_local=np.asarray(
        [-1 if info["saturation_frame_local"] is None else info["saturation_frame_local"] for info in segment_infos],
        dtype=np.int64,
    ),
    segment_num_queries=np.asarray([len(info["query_global_ids"]) for info in segment_infos], dtype=np.int64),
)

tracks_xy = tracks_xy_256.astype(np.float64, copy=True)
tracks_xy[..., 0] = tracks_xy[..., 0] / 256.0 * W
tracks_xy[..., 1] = tracks_xy[..., 1] / 256.0 * H

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

depth_paths = sorted(glob.glob(os.path.join(str(GT_DEPTH_DIR), "*.dpt")))
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
        if depth_maps.ndim != 3:
            raise ValueError(f"Expected depth_maps.npy shape (T,H,W), got {depth_maps.shape}")
        T_lift = min(T, depth_maps.shape[0])
else:
    raise ValueError(f"Unsupported DEPTH_SOURCE='{DEPTH_SOURCE}'. Use 'gt' or 'da3'.")

if T_lift > 0:
    image_paths = sorted(glob.glob(os.path.join(str(IMAGE_DIR), IMAGE_PATTERN)))
    if len(image_paths) >= T_lift:
        tracks_xyz = np.full((1, Q, T_lift, 3), np.nan, dtype=np.float64)
        prev_depths = np.full(Q, np.nan, dtype=np.float64)

        for t in range(T_lift):
            img_name = os.path.basename(image_paths[t])
            if img_name not in gt:
                continue

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
            max_ok = z <= float(DEPTH_MAX)
            basic_depth_ok = finite_positive & max_ok
            grad_ok = grad <= float(DEPTH_GRAD_THRESHOLD)

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

if MOVE_DELETE_ENABLED:
    move_delete_query_mask = _compute_move_delete_query_mask(
        tracks_xy=tracks_xy,
        occluded_obs=occluded_dynamic,
        window=MOVE_WINDOW_SIZE,
        top_percent=MOVE_DELETE_TOP_PERCENT,
    )
    n_move_delete = int(np.sum(move_delete_query_mask))
    if n_move_delete > 0:
        occluded_dynamic[0, move_delete_query_mask, :] = True
        if tracks_xyz is not None:
            tracks_xyz[0, move_delete_query_mask, :, :] = np.nan
else:
    move_delete_query_mask = np.zeros(Q, dtype=bool)

if SMOOTH_3D_ENABLED and tracks_xyz is not None:
    tracks_xyz = _smooth_tracks_xyz_visible_segments(tracks_xyz, occluded_dynamic, window=SMOOTH_3D_WINDOW)

# Filter breakdown (why points get dropped), mirroring dynamic_tracking.py.
_kept_per_frame = int(np.sum(~occluded_dynamic[0])) / max(1, T)
print(
    f"[filter] queries Q={Q} | outside-dynamic-mask obs={int(np.sum(~dynamic_obs_mask))} | "
    f"depth-grad-deleted obs={int(np.sum(depth_grad_deleted_obs_mask))} | "
    f"depth-max-deleted obs={int(np.sum(depth_max_deleted_obs_mask))} | "
    f"large-2D-move-deleted queries={int(np.sum(move_delete_query_mask))}/{Q} | "
    f"avg kept points/frame={_kept_per_frame:.1f}"
)

np.savez_compressed(
    OUT_DIR / "tracks_dynamic_only.npz",
    tracks_xy=tracks_xy,
    tracks_xyz=tracks_xyz,
    occluded=occluded_dynamic,
    dynamic_obs_mask=dynamic_obs_mask,
    depth_grad_deleted_obs_mask=depth_grad_deleted_obs_mask,
    depth_max_deleted_obs_mask=depth_max_deleted_obs_mask,
    move_delete_query_mask=move_delete_query_mask,
    query_t=query_t.astype(np.int32),
    video_dir=str(IMAGE_DIR),
    mask_dir=str(MASK_DIR),
)

_write_dynamic_reason_viz_mp4(
    video_np_full,
    tracks_xy,
    occluded_dynamic,
    str(OUT_DIR / "viz_dynamic_only.mp4"),
    raw_occluded=occluded,
    dynamic_obs_mask=dynamic_obs_mask,
    depth_grad_deleted_obs_mask=depth_grad_deleted_obs_mask,
    depth_max_deleted_obs_mask=depth_max_deleted_obs_mask,
    move_delete_query_mask=move_delete_query_mask,
    query_t=query_t,
    fps=10,
)
print(f"Wrote: {OUT_DIR / 'viz_dynamic_only.mp4'}")
