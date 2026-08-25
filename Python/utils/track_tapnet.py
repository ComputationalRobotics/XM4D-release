#!/usr/bin/env python3
"""Minimal TAPNet (TAPNext) tracking script.

Tracks a grid of points from the first frame across the full video.

Example:
  python3 deps/tapnet/simple_track_video.py \
    --video_dir deps/tapnet/inputs/earth \
    --out_dir deps/tapnet/outputs/simple_test

Outputs:
  - tracks.npz (tracks + occluded)
  - viz.mp4 (tracks overlaid on 256x256 resized frames)
"""

import argparse
import glob
import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F

_TAPNET_DEPS_DIR = Path(__file__).resolve().parents[2] / "deps" / "tapnet"
if _TAPNET_DEPS_DIR.exists() and str(_TAPNET_DEPS_DIR) not in sys.path:
    sys.path.append(str(_TAPNET_DEPS_DIR))

TAPNext = None
restore_model_from_jax_checkpoint = None
tracker_certainty = None
_TAPNEXT_IMPORT_ERROR = None

try:
    from tapnet.tapnext.tapnext_torch import TAPNext
    from tapnet.tapnext.tapnext_torch_utils import restore_model_from_jax_checkpoint, tracker_certainty
except ImportError as e:
    _TAPNEXT_IMPORT_ERROR = e


def _numeric_key(path: str):
    stem = Path(path).stem
    try:
        return int(stem)
    except ValueError:
        return stem


def load_frames_256(video_dir: str, pattern: str) -> np.ndarray:
    paths = glob.glob(os.path.join(video_dir, pattern))
    if not paths:
        raise FileNotFoundError(f"No frames found in {video_dir} matching {pattern}")
    paths = sorted(paths, key=_numeric_key)

    frames_256 = []
    frames = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((256, 256), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32)
        arr = arr / 255.0 * 2.0 - 1.0  # [-1, 1]
        frames_256.append(arr)
        frames.append(np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8))  # original size for viz

    video_256 = np.stack(frames_256, axis=0)  # [T, 256, 256, 3]
    video = np.stack(frames, axis=0)  # [T, H, W, 3]
    return video_256[None, ...], video[None, ...]  # [1, T, 256, 256, 3]


def refine_tracks_with_sift(
    video_np: np.ndarray,
    tracks_xy: np.ndarray,
    occluded: np.ndarray,
    *,
    search_radius: float = 12.0,
    ratio: float = 0.75,
    blend: float = 1.0,
    passes: int = 1,
):
    """
    Local refinement: snap TAPNext tracks to SIFT keypoints using descriptor matching.

    Args:
      video_np:  [1, T, H, W, 3] uint8/float in RGB (as used by your pipeline)
      tracks_xy: [1, Q, T, 2] in pixel coords (x,y) for the SAME H,W as video_np
      occluded:  [1, Q, T] bool
      search_radius: neighborhood radius (pixels) to look for keypoints near predicted point
      ratio: Lowe ratio threshold
      blend: 1.0 = fully replace by matched keypoint, 0.5 = halfway toward match
      passes: do multiple forward passes (cheap). You can also do forward+back if desired.

    Returns:
      tracks_xy_refined, occluded (unchanged)
    """
    import cv2

    assert video_np.ndim == 5 and video_np.shape[0] == 1, "Expected video_np [1,T,H,W,3]"
    _, T, H, W, C = video_np.shape
    _, Q, T2, _ = tracks_xy.shape
    assert T == T2, "tracks and video must have same T"
    assert C == 3, "Expected RGB video"

    # Convert frames to grayscale uint8
    def to_gray_u8(frame_rgb):
        if frame_rgb.dtype != np.uint8:
            # assume 0..1 or 0..255 float
            fr = frame_rgb
            if fr.max() <= 1.5:
                fr = (fr * 255.0)
            frame_rgb = np.clip(fr, 0, 255).astype(np.uint8)
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # Precompute SIFT per frame
    kp_xy = []
    des_list = []
    for t in range(T):
        gray = to_gray_u8(video_np[0, t])
        kps, des = sift.detectAndCompute(gray, None)
        if des is None or len(kps) == 0:
            kp_xy.append(np.zeros((0, 2), dtype=np.float32))
            des_list.append(np.zeros((0, 128), dtype=np.float32))
        else:
            kp_xy.append(np.array([k.pt for k in kps], dtype=np.float32))  # (x,y)
            des_list.append(des.astype(np.float32))

    # Helper: neighbor indices within radius (brute-force, ok for moderate #keypoints)
    r2 = float(search_radius) ** 2

    def neighbors(frame_kp_xy: np.ndarray, x: float, y: float):
        if frame_kp_xy.shape[0] == 0:
            return np.empty((0,), dtype=np.int64)
        d2 = (frame_kp_xy[:, 0] - x) ** 2 + (frame_kp_xy[:, 1] - y) ** 2
        return np.where(d2 <= r2)[0].astype(np.int64)

    refined = tracks_xy.copy()

    for _ in range(int(passes)):
        for t in range(T - 1):
            des1_all = des_list[t]
            des2_all = des_list[t + 1]
            if des1_all.shape[0] == 0 or des2_all.shape[0] == 0:
                continue

            kp1 = kp_xy[t]
            kp2 = kp_xy[t + 1]

            for i in range(Q):
                if occluded[0, i, t] or occluded[0, i, t + 1]:
                    continue

                x1, y1 = refined[0, i, t]
                x2p, y2p = refined[0, i, t + 1]  # predicted next
                if not (np.isfinite(x1) and np.isfinite(y1) and np.isfinite(x2p) and np.isfinite(y2p)):
                    continue

                idx1 = neighbors(kp1, float(x1), float(y1))
                idx2 = neighbors(kp2, float(x2p), float(y2p))
                if idx1.size == 0 or idx2.size == 0:
                    continue

                # Slice descriptors to local neighborhoods
                d1 = des1_all[idx1]
                d2 = des2_all[idx2]

                # Need at least 2 candidates for ratio test; if not, we can do simple best match
                best_match = None
                best_dist = float("inf")

                if d2.shape[0] >= 2:
                    matches = bf.knnMatch(d1, d2, k=2)
                    for m2 in matches:
                        if len(m2) < 2:
                            continue
                        m, n = m2[0], m2[1]
                        if m.distance >= ratio * n.distance:
                            continue
                        if m.distance < best_dist:
                            best_dist = m.distance
                            best_match = m
                else:
                    matches = bf.match(d1, d2)
                    for m in matches:
                        if m.distance < best_dist:
                            best_dist = m.distance
                            best_match = m

                if best_match is None:
                    continue

                # Map local match indices back to global keypoint index in frame t+1
                kp2_global = idx2[best_match.trainIdx]
                newx, newy = kp2[kp2_global]

                # Update next-frame track (blend)
                refined[0, i, t + 1, 0] = (1.0 - blend) * x2p + blend * float(newx)
                refined[0, i, t + 1, 1] = (1.0 - blend) * y2p + blend * float(newy)

    return refined, occluded


def make_grid_query_points(
    grid: int,
    margin: int = 8,
    step: int = 5,
    t_max: int = 0,
    seed: int = 0,
) -> np.ndarray:
    """Random query points for TAPNext.

    Returns shape [1, Q, 3] with (t, x, y). If `t_max` is provided, samples are
    stacked every `step` frames from t=0 to t_max (inclusive). If `t_max` is None,
    it defaults to 0 (single set at t=0).
    """
    q = int(grid) * int(grid)
    rng = np.random.default_rng(seed)
    x = rng.uniform(margin, 255 - margin, size=(q,)).astype(np.float32)
    y = rng.uniform(margin, 255 - margin, size=(q,)).astype(np.float32)

    if t_max is None:
        t_list = [0]
    else:
        t_list = list(range(0, int(t_max) + 1, int(step)))

    # Stack random points for each time in t_list
    all_points = []
    for t in t_list:
        t_arr = np.full(q, float(t), dtype=np.float32)
        all_points.append(np.stack([t_arr, x, y], axis=1))

    return np.concatenate(all_points, axis=0)[None, ...]


def infer_mask_dir(video_dir: str) -> str | None:
    """Find sibling dynamic-mask directory for a video directory.

    Tries both "*_mask" and "*_masks".
    """
    candidates = [f"{video_dir}_mask", f"{video_dir}_masks"]
    for cand in candidates:
        if os.path.isdir(cand):
            return cand
    return None


def load_dynamic_masks_256(
    mask_dir: str,
    pattern: str,
    t_expected: int,
    mask_mode: str = "auto",
) -> np.ndarray:
    """Load dynamic masks as bool array [T, 256, 256].

        `mask_mode`:
            - "auto": if foreground (>0) occupies >50%, treat it as static and invert
            - "dynamic_positive": >0 means dynamic
            - "dynamic_zero": ==0 means dynamic
    """
    paths = glob.glob(os.path.join(mask_dir, pattern))
    if not paths:
        raise FileNotFoundError(f"No masks found in {mask_dir} matching {pattern}")
    paths = sorted(paths, key=_numeric_key)

    if len(paths) < t_expected:
        raise ValueError(
            f"Mask count mismatch: found {len(paths)} masks, but video has {t_expected} frames"
        )

    masks = []
    for p in paths[:t_expected]:
        im = Image.open(p).convert("L").resize((256, 256), Image.NEAREST)
        arr = np.asarray(im)
        masks.append(arr > 0)
    masks_pos = np.stack(masks, axis=0)

    if mask_mode == "dynamic_positive":
        return masks_pos
    if mask_mode == "dynamic_zero":
        return ~masks_pos
    if mask_mode != "auto":
        raise ValueError(f"Unknown --mask_mode: {mask_mode}")

    # Auto mode: many datasets store static as white and dynamic as black.
    pos_ratio = float(np.mean(masks_pos))
    if pos_ratio > 0.5:
        return ~masks_pos
    return masks_pos


def load_dynamic_masks(
    mask_dir: str,
    pattern: str,
    t_expected: int,
    mask_mode: str = "auto",
    target_size: tuple | None = None,
) -> np.ndarray:
    """Load dynamic masks as bool array [T, H, W].

    If target_size is provided, masks are resized to that (width, height) using NEAREST.
    mask_mode has the same semantics as load_dynamic_masks_256.
    """
    paths = glob.glob(os.path.join(mask_dir, pattern))
    if not paths:
        raise FileNotFoundError(f"No masks found in {mask_dir} matching {pattern}")
    paths = sorted(paths, key=_numeric_key)

    if len(paths) < t_expected:
        raise ValueError(
            f"Mask count mismatch: found {len(paths)} masks, but video has {t_expected} frames"
        )

    masks = []
    for p in paths[:t_expected]:
        im = Image.open(p).convert("L")
        if target_size is not None:
            # target_size: (width, height)
            im = im.resize((int(target_size[0]), int(target_size[1])), Image.NEAREST)
        arr = np.asarray(im)
        masks.append(arr > 0)
    masks_pos = np.stack(masks, axis=0)

    if mask_mode == "dynamic_positive":
        return masks_pos
    if mask_mode == "dynamic_zero":
        return ~masks_pos
    if mask_mode != "auto":
        raise ValueError(f"Unknown --mask_mode: {mask_mode}")

    pos_ratio = float(np.mean(masks_pos))
    if pos_ratio > 0.5:
        return ~masks_pos
    return masks_pos


def dilate_dynamic_masks(dynamic_masks: np.ndarray, dilate_px: int) -> np.ndarray:
    """Dilate (expand) or erode (shrink) dynamic masks by `dilate_px` pixels.

    Positive values dilate; negative values erode; zero returns input unchanged.
    """
    if dilate_px == 0:
        return dynamic_masks

    import cv2

    k = 2 * int(abs(dilate_px)) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    out = np.empty_like(dynamic_masks, dtype=bool)
    for t in range(dynamic_masks.shape[0]):
        m = (dynamic_masks[t].astype(np.uint8) * 255)
        if dilate_px > 0:
            out[t] = cv2.dilate(m, kernel, iterations=1) > 0
        else:
            out[t] = cv2.erode(m, kernel, iterations=1) > 0
    return out


def apply_dynamic_mask_filter(
    tracks_xy: np.ndarray,
    occluded: np.ndarray,
    dynamic_masks: np.ndarray,
    static_only: bool = True,
    dynamic_ratio_thresh: float = 0.2,
) -> np.ndarray:
    
    """
    Hide tracks on dynamic regions, using an occupancy ratio threshold.

    A track is considered "dynamic" if, over its VALID & VISIBLE timesteps,
    it lands on dynamic mask pixels for > dynamic_ratio_thresh fraction of time.

    Policy (threshold-based):
      (i) ratio > dynamic_ratio_thresh  -> occlude the whole track
      (ii) ratio <= dynamic_ratio_thresh -> occlude only masked frames

    Args:
      tracks_xy: [1, Q, T, 2] (x, y) in mask pixel coords
      occluded:  [1, Q, T]   True = already occluded
      dynamic_masks: [T, H, W], True = dynamic pixel
      static_only: deprecated (kept for backward compatibility; ignored)
      dynamic_ratio_thresh: threshold on dynamic-frame ratio per track

    Returns:
      out: [1, Q, T] boolean occlusion mask
    """

    out = occluded.copy()
    b, q, t_total, _ = tracks_xy.shape
    assert b == 1, "Expected batch size 1"
    assert dynamic_masks.shape[0] >= t_total, "Not enough mask frames"

    # Track per-(query, time) whether we were able to sample the mask and whether it was dynamic
    valid_sample = np.zeros((q, t_total), dtype=bool)
    dyn_hit = np.zeros((q, t_total), dtype=bool)

    for t in range(t_total):
        mask_t = dynamic_masks[t]
        H, W = mask_t.shape[:2]

        # only consider frames that are not already occluded
        vis = ~out[0, :, t]  # [Q]

        x = tracks_xy[0, :, t, 0]
        y = tracks_xy[0, :, t, 1]

        finite = np.isfinite(x) & np.isfinite(y)
        xi = np.rint(x).astype(np.int64)
        yi = np.rint(y).astype(np.int64)

        in_bounds = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)

        ok = vis & finite & in_bounds
        valid_sample[:, t] = ok

        if np.any(ok):
            dyn_vals = mask_t[yi[ok], xi[ok]]
            dyn_hit[ok, t] = dyn_vals.astype(bool)

    # Compute dynamic occupancy ratio per query over valid samples
    denom = valid_sample.sum(axis=1)  # [Q]
    num = (dyn_hit & valid_sample).sum(axis=1)  # [Q]
    ratio = np.zeros((q,), dtype=np.float64)
    nonzero = denom > 0
    ratio[nonzero] = num[nonzero] / denom[nonzero]

    is_dynamic_track = ratio > dynamic_ratio_thresh  # strictly greater than threshold

    # (i) whole-track occlusion for tracks above threshold
    out[0, is_dynamic_track, :] = True

    # (ii) frame-wise occlusion for tracks at/below threshold
    is_not_dynamic_track = ~is_dynamic_track
    out[0, is_not_dynamic_track, :] |= dyn_hit[is_not_dynamic_track, :]

    return out


@torch.no_grad()
def track_full_video_tapnext(
    video: torch.Tensor,
    query_points: torch.Tensor,
    device: torch.device,
    checkpoint: str,
    use_certainty: bool = True,
    radius: int = 8,
    threshold: float = 0.5,
):
    """
    Tracks points across the whole video.

    Args:
      video: [1, T, 256, 256, 3] float, in [-1, 1]
      query_points: [1, Q, 3] float, (t, x, y)

    Returns:
      tracks_xy: [1, Q, T, 2] float32 (x, y)
      occluded:  [1, Q, T] bool
    """

    if TAPNext is None or restore_model_from_jax_checkpoint is None or tracker_certainty is None:
        raise ImportError(
            "Failed to import TAPNext dependencies. "
            f"Looked for local deps at '{_TAPNET_DEPS_DIR}'. "
            "Ensure deps/tapnet is available and/or set PYTHONPATH to include it. "
            f"Original error: {_TAPNEXT_IMPORT_ERROR}"
        )

    query_points = query_points.to(dtype=torch.float32, device=device)

    if checkpoint.endswith(".npz"):
        model = TAPNext(image_size=(256, 256)).to(device).eval()
        model = restore_model_from_jax_checkpoint(model, checkpoint)
    else:
        model = TAPNext(image_size=(256, 256))
        ckpt = torch.load(checkpoint, map_location='cpu')
        model.load_state_dict({k.replace('tapnext.', ''): v for k, v in ckpt['state_dict'].items()})
        model.cuda()

    # init on first frame
    pred_tracks, track_logits, visible_logits, state = model(
        video=video[:, :1], query_points=query_points
    )
    pred_visible = visible_logits > 0

    tracks_list = [pred_tracks.detach().cpu()]
    visible_list = [pred_visible.detach().cpu()]
    track_logits_list = [track_logits.detach().cpu()]
    visible_logits_list = [visible_logits.detach().cpu()]

    T = int(video.shape[1])
    for t in tqdm(range(1, T), desc="TAPNext tracking"):
        curr_tracks, curr_track_logits, curr_visible_logits, state = model(
            video=video[:, t : t + 1], state=state
        )
        tracks_list.append(curr_tracks.detach().cpu())
        track_logits_list.append(curr_track_logits.detach().cpu())
        visible_logits_list.append(curr_visible_logits.detach().cpu())
        visible_list.append((curr_visible_logits > 0).detach().cpu())

    # concat along time, then transpose to [B, Q, T, ...]
    tracks_yx = torch.cat(tracks_list, dim=1).transpose(1, 2)  # [1, Q, T, 2] (y, x)
    pred_visible = torch.cat(visible_list, dim=1).transpose(1, 2).squeeze(-1)  # [1, Q, T]

    if use_certainty:
        track_logits = torch.cat(track_logits_list, dim=1).transpose(1, 2)
        visible_logits = torch.cat(visible_logits_list, dim=1).transpose(1, 2)
        certainty = tracker_certainty(tracks_yx, track_logits, radius)
        pred_visible_and_certain = (F.sigmoid(visible_logits) * certainty) > threshold
        occluded = ~(pred_visible_and_certain.squeeze(-1))
    else:
        occluded = ~pred_visible

    # Queries may start at frame t_q (>0). Mark all frames before t_q as occluded.
    # query_points is [1, Q, 3] with (t, x, y).
    query_t = torch.floor(query_points[..., 0]).to(device=occluded.device, dtype=torch.int64)  # [1, Q]
    time_idx = torch.arange(T, device=occluded.device, dtype=torch.int64).view(1, 1, T)
    pre_query_mask = time_idx < query_t.unsqueeze(-1)  # [1, Q, T]
    occluded = occluded | pre_query_mask

    tracks_xy = torch.flip(tracks_yx, dims=[-1])  # (y, x) -> (x, y)
    return tracks_xy.numpy().astype(np.float32), occluded.numpy().astype(bool)

# Write viz on original image size (video_np contains original-size frames)
def write_viz_mp4(
    video_orig_np,
    tracks_xy,
    occluded,
    out_path: str,
    fps: int = 10,
    tail_length: int = 12,
    tail_thickness: int = 1,
    dynamic_masks: np.ndarray | None = None,
    mask_alpha: float = 0.35,
    highlight_query_mask: np.ndarray | None = None,
    highlight_color: tuple[int, int, int] = (0, 0, 255),
    highlight_query_mask_always: np.ndarray | None = None,
    highlight_query_color_always: tuple[int, int, int] = (255, 0, 255),
    highlight_radius: int = 5,
    highlight_obs_mask: np.ndarray | None = None,
    highlight_obs_color: tuple[int, int, int] = (0, 0, 255),
    highlight_obs_mask_2: np.ndarray | None = None,
    highlight_obs_color_2: tuple[int, int, int] = (0, 165, 255),
    highlight_obs_mask_3: np.ndarray | None = None,
    highlight_obs_color_3: tuple[int, int, int] = (255, 0, 255),
    std_norm: np.ndarray | None = None,
    std_max: float | None = None,
):
    import cv2

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # video_orig_np: [1, T, H, W, 3], uint8
    vid = video_orig_np[0]
    T, H, W, _ = vid.shape

    draw_tracks = tracks_xy is not None
    if draw_tracks:
        Q = tracks_xy.shape[1]
        if occluded is None:
            occluded = np.zeros((1, Q, T), dtype=bool)
        rng = np.random.default_rng(0)
        colors = rng.integers(0, 255, size=(Q, 3), dtype=np.uint8)
        if highlight_query_mask is not None:
            highlight_query_mask = np.asarray(highlight_query_mask).astype(bool).reshape(-1)
            if highlight_query_mask.shape[0] != Q:
                raise ValueError(
                    f"highlight_query_mask must have length {Q}, got {highlight_query_mask.shape[0]}"
                )
        if highlight_query_mask_always is not None:
            highlight_query_mask_always = np.asarray(highlight_query_mask_always).astype(bool).reshape(-1)
            if highlight_query_mask_always.shape[0] != Q:
                raise ValueError(
                    f"highlight_query_mask_always must have length {Q}, got {highlight_query_mask_always.shape[0]}"
                )
        if highlight_obs_mask is not None:
            highlight_obs_mask = np.asarray(highlight_obs_mask).astype(bool)
            if highlight_obs_mask.ndim == 2:
                if highlight_obs_mask.shape != (Q, T):
                    raise ValueError(
                        f"highlight_obs_mask with ndim=2 must have shape (Q,T)=({Q},{T}), got {highlight_obs_mask.shape}"
                    )
                highlight_obs_mask = highlight_obs_mask[None, ...]
            if highlight_obs_mask.shape != (1, Q, T):
                raise ValueError(
                    f"highlight_obs_mask must have shape (1,Q,T)=(1,{Q},{T}), got {highlight_obs_mask.shape}"
                )
        if highlight_obs_mask_2 is not None:
            highlight_obs_mask_2 = np.asarray(highlight_obs_mask_2).astype(bool)
            if highlight_obs_mask_2.ndim == 2:
                if highlight_obs_mask_2.shape != (Q, T):
                    raise ValueError(
                        f"highlight_obs_mask_2 with ndim=2 must have shape (Q,T)=({Q},{T}), got {highlight_obs_mask_2.shape}"
                    )
                highlight_obs_mask_2 = highlight_obs_mask_2[None, ...]
            if highlight_obs_mask_2.shape != (1, Q, T):
                raise ValueError(
                    f"highlight_obs_mask_2 must have shape (1,Q,T)=(1,{Q},{T}), got {highlight_obs_mask_2.shape}"
                )
        if highlight_obs_mask_3 is not None:
            highlight_obs_mask_3 = np.asarray(highlight_obs_mask_3).astype(bool)
            if highlight_obs_mask_3.ndim == 2:
                if highlight_obs_mask_3.shape != (Q, T):
                    raise ValueError(
                        f"highlight_obs_mask_3 with ndim=2 must have shape (Q,T)=({Q},{T}), got {highlight_obs_mask_3.shape}"
                    )
                highlight_obs_mask_3 = highlight_obs_mask_3[None, ...]
            if highlight_obs_mask_3.shape != (1, Q, T):
                raise ValueError(
                    f"highlight_obs_mask_3 must have shape (1,Q,T)=(1,{Q},{T}), got {highlight_obs_mask_3.shape}"
                )
        if std_norm is not None:
            std_norm = np.asarray(std_norm).reshape(-1)
            if std_norm.shape[0] != Q:
                raise ValueError(f"std_norm must have length {Q}, got {std_norm.shape[0]}")
            finite_std = np.isfinite(std_norm)
            if std_max is None:
                std_max = float(np.nanmax(std_norm[finite_std])) if np.any(finite_std) else 1.0
            std_max = max(float(std_max), 1e-6)

    frames_bgr = []

    # If dynamic_masks provided at original size, overlay directly.
    use_masks = dynamic_masks is not None

    for t in range(T):
        frame = vid[t].copy()
        # frame is RGB; convert to BGR for cv2 drawing
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # overlay dynamic mask (resize if needed)
        if use_masks and t < dynamic_masks.shape[0]:
            m = dynamic_masks[t]
            if m.dtype != np.bool_:
                m = m > 0
            if m.any():
                if m.shape[:2] != (H, W):
                    m = cv2.resize(m.astype(np.uint8) * 255, (W, H), interpolation=cv2.INTER_NEAREST) > 0
                overlay = frame.copy()
                overlay[m] = np.array([0, 0, 255], dtype=np.uint8)
                frame = cv2.addWeighted(overlay, float(mask_alpha), frame, 1.0 - float(mask_alpha), 0.0)

        # draw tracks and tails (if provided)
        if draw_tracks:
            for i in range(Q):
                c = tuple(int(v) for v in colors[i])

                # # tail
                # if tail_length > 1:
                #     t0 = max(0, t - tail_length + 1)
                #     prev_pt = None
                #     for tt in range(t0, t + 1):
                #         if occluded[0, i, tt]:
                #             prev_pt = None
                #             continue
                #         xt, yt = tracks_xy[0, i, tt]
                #         if not (np.isfinite(xt) and np.isfinite(yt)):
                #             prev_pt = None
                #             continue
                #         xi_t, yi_t = int(round(float(xt))), int(round(float(yt)))
                #         if not (0 <= xi_t < W and 0 <= yi_t < H):
                #             prev_pt = None
                #             continue

                #         if prev_pt is not None:
                #             cv2.line(frame, prev_pt, (xi_t, yi_t), c, tail_thickness, lineType=cv2.LINE_AA)
                #         prev_pt = (xi_t, yi_t)

                x, y = tracks_xy[0, i, t]
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                xi, yi = int(round(float(x))), int(round(float(y)))
                if 0 <= xi < W and 0 <= yi < H:
                    if highlight_obs_mask is not None and highlight_obs_mask[0, i, t]:
                        cv2.circle(
                            frame,
                            (xi, yi),
                            int(max(1, highlight_radius)),
                            tuple(int(v) for v in highlight_obs_color),
                            -1,
                            lineType=cv2.LINE_AA,
                        )
                        continue

                    if highlight_obs_mask_2 is not None and highlight_obs_mask_2[0, i, t]:
                        cv2.circle(
                            frame,
                            (xi, yi),
                            int(max(1, highlight_radius)),
                            tuple(int(v) for v in highlight_obs_color_2),
                            -1,
                            lineType=cv2.LINE_AA,
                        )
                        continue

                    if highlight_obs_mask_3 is not None and highlight_obs_mask_3[0, i, t]:
                        cv2.circle(
                            frame,
                            (xi, yi),
                            int(max(1, highlight_radius)),
                            tuple(int(v) for v in highlight_obs_color_3),
                            -1,
                            lineType=cv2.LINE_AA,
                        )
                        continue

                    if highlight_query_mask_always is not None and highlight_query_mask_always[i]:
                        cv2.circle(
                            frame,
                            (xi, yi),
                            int(max(1, highlight_radius)),
                            tuple(int(v) for v in highlight_query_color_always),
                            -1,
                            lineType=cv2.LINE_AA,
                        )
                        continue

                    if occluded[0, i, t]:
                        continue

                    if highlight_query_mask is not None and highlight_query_mask[i]:
                        cv2.circle(
                            frame,
                            (xi, yi),
                            int(max(1, highlight_radius)),
                            tuple(int(v) for v in highlight_color),
                            -1,
                            lineType=cv2.LINE_AA,
                        )
                    elif std_norm is not None and np.isfinite(std_norm[i]):
                        # color by std error: green (low) -> red (high)
                        r = min(1.0, max(0.0, float(std_norm[i]) / float(std_max)))
                        g = 1.0 - r
                        col = (int(255 * r), int(255 * g), 0)
                        cv2.circle(frame, (xi, yi), 3, col, -1, lineType=cv2.LINE_AA)
                    else:
                        cv2.circle(frame, (xi, yi), 3, c, -1, lineType=cv2.LINE_AA)

        frames_bgr.append(frame)

    # Prefer ffmpeg/libx264 (same compatibility style as alltracker demo).
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is not None:
        with tempfile.TemporaryDirectory(prefix="tapnet_viz_") as temp_dir:
            for t, frame in enumerate(frames_bgr):
                cv2.imwrite(
                    os.path.join(temp_dir, f"{t:06d}.jpg"),
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95],
                )
            cmd = [
                ffmpeg_bin,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
                str(int(fps)),
                "-i",
                os.path.join(temp_dir, "%06d.jpg"),
                "-c:v",
                "libx264",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                out_path,
            ]
            subprocess.run(cmd, check=True)
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, float(fps), (W, H))
        for frame in frames_bgr:
            writer.write(frame)
        writer.release()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video_dir", required=True, help="Directory with frames like 000.jpg / 00000.png")
    p.add_argument("--pattern", default="*.jpg", help="Glob pattern for frames (e.g. '*.jpg' or '*.png')")
    p.add_argument(
        "--checkpoint",
        default=str(Path(__file__).resolve().parent / "checkpoints/bootstapnext_ckpt.npz"),
        help="TAPNext JAX checkpoint (.npz) to load",
    )
    p.add_argument("--grid", type=int, default=20, help="Grid resolution per axis; total points = grid^2")
    p.add_argument("--out_dir", default="outputs/simple_test", help="Output directory")
    p.add_argument("--fps", type=int, default=10, help="FPS for viz.mp4")
    p.add_argument("--tail_length", type=int, default=12, help="Tail length in frames for each track")
    p.add_argument("--tail_thickness", type=int, default=1, help="Line thickness for trajectory tails")
    p.add_argument("--mask_dir", default=None, help="Dynamic-mask directory. If omitted, tries <video_dir>_mask and <video_dir>_masks")
    p.add_argument("--mask_pattern", default="*.png", help="Glob pattern for dynamic masks")
    p.add_argument(
        "--mask_mode",
        choices=["auto", "dynamic_positive", "dynamic_zero"],
        default="auto",
        help="How to interpret mask pixels",
    )
    p.add_argument("--mask_dilate", type=int, default=0, help="Expand dynamic masks by this many pixels")
    p.add_argument(
        "--static_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, remove any point that ever touches dynamic mask (keep only static points)",
    )
    p.add_argument("--use_certainty", action="store_true", help="More conservative visibility using certainty")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    video_np = load_frames_256(args.video_dir, args.pattern)
    query_np = make_grid_query_points(args.grid)

    model = TAPNext(image_size=(256, 256)).to(device).eval()
    model = restore_model_from_jax_checkpoint(model, args.checkpoint)

    video = torch.from_numpy(video_np).to(device)
    query = torch.from_numpy(query_np).to(device)

    tracks_xy, occluded = track_full_video_tapnext(
        video, query, model, use_certainty=args.use_certainty
    )

    # Optional dynamic masks: hide dynamic regions and keep static points only.
    mask_dir = args.mask_dir if args.mask_dir is not None else infer_mask_dir(args.video_dir)
    dynamic_masks = None
    if mask_dir is not None:
        dynamic_masks = load_dynamic_masks_256(
            mask_dir,
            args.mask_pattern,
            t_expected=video_np.shape[1],
            mask_mode=str(args.mask_mode),
        )
        dynamic_masks = dilate_dynamic_masks(dynamic_masks, int(args.mask_dilate))
        occluded_before = occluded.copy()
        occluded = apply_dynamic_mask_filter(tracks_xy, occluded, dynamic_masks, static_only=bool(args.static_only))
        n_newly_hidden = int(np.sum(occluded) - np.sum(occluded_before))
        print(f"Applied dynamic masks from: {mask_dir}")
        print(f"Mask mode: {args.mask_mode}")
        print(f"Mask dilation (pixels): {int(args.mask_dilate)}")
        print(f"Newly hidden samples by mask filtering: {n_newly_hidden}")
    else:
        print("No sibling mask dir found (<video_dir>_mask / <video_dir>_masks); skipping mask filtering")

    np.savez_compressed(
        out_dir / "tracks.npz",
        tracks_xy=tracks_xy,
        occluded=occluded,
        video_dir=str(args.video_dir),
        pattern=str(args.pattern),
        grid=int(args.grid),
        mask_dir=str(mask_dir) if mask_dir is not None else "",
        static_only=bool(args.static_only),
    )

    write_viz_mp4(
        video_np,
        tracks_xy,
        occluded,
        str(out_dir / "viz.mp4"),
        fps=args.fps,
        tail_length=max(1, int(args.tail_length)),
        tail_thickness=max(1, int(args.tail_thickness)),
        dynamic_masks=dynamic_masks,
    )
    print(f"Wrote: {out_dir / 'tracks.npz'}")
    print(f"Wrote: {out_dir / 'viz.mp4'}")
