#!/usr/bin/env python3
"""Dynamic-only tracking aligned with static_tracking_online.py. Moving objects are tracked with AllTracker from
keyframes (utils/track_alltracker.py)."""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import glob
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image

from utils.masks import _numeric_key, dilate_dynamic_masks, load_dynamic_masks

import utils.definitions
import utils.dataset_config as dataset_config
from utils.io import savez_compressed_fast
from utils.script_logger import install_script_logger, pause_script_logger, resume_script_logger


### S0: CONFIG
IMAGE_DIR = utils.definitions.video_dir
MASK_DIR = utils.definitions.mask_dir
OUT_DIR = utils.definitions.output_dir
DA3_OUT_DIR = utils.definitions.da3_depth_dir

IMAGE_PATTERN = utils.definitions.image_pattern
MASK_PATTERN = utils.definitions.mask_pattern

# Parameters from config.json (utils.dataset_config), as in static_tracking_online.py.
# 2D tracker: AllTracker. Points seeded on the object mask every ALLTRACKER_KEYFRAME frames where no tracked point
# is visible nearby (grid spacing DYN_ALLTRACKER_GRID px at the tracking resolution ALLTRACKER_RES), visible where
# AllTracker's visibility x confidence >= ALLTRACKER_VIS_THRESHOLD and the point is on the object mask.
AT_KEYFRAME = dataset_config.get_int("ALLTRACKER_KEYFRAME", 10)
AT_RES = dataset_config.get_int("ALLTRACKER_RES", 512)
AT_VIS_THRESHOLD = dataset_config.get_float("ALLTRACKER_VIS_THRESHOLD", 0.6)
DYN_AT_GRID = dataset_config.get_float("DYN_ALLTRACKER_GRID", 2.0)
AT_MIN_NEW = dataset_config.get_float("ALLTRACKER_MIN_NEW", 0.0)
AT_FP16 = dataset_config.get_bool("ALLTRACKER_FP16", True)
DEBUG = dataset_config.get_bool("DEBUG", False)  # also write the per-reason visualisation video

MASK_MODE = "dynamic_positive"
# Erosion (<0) in native px for the dynamic masks. Query masks
# select points ON the moving object, so erosion keeps queries safely inside
# (conservative = fewer, safer dynamic points).
DILATE_PX_QUERY = -2
DILATE_PX_VIS = -2

ONLINE_MAX_POINTS = 50000

DEPTH_MAPS_PATH = DA3_OUT_DIR / "depth_maps.npy"

OUT_DIR.mkdir(parents=True, exist_ok=True)
install_script_logger("dynamic_tracking", OUT_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_NAME = "tracks_dynamic_only.npz"


def _write_dynamic_reason_viz_mp4(
    video_np: np.ndarray,
    tracks_xy: np.ndarray,
    occluded_final: np.ndarray,
    out_path: str,
    *,
    raw_occluded: np.ndarray,
    dynamic_obs_mask: np.ndarray,
    query_t: np.ndarray,
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

    query_t = np.asarray(query_t, dtype=np.int64).reshape(-1)

    color_visible = (60, 220, 60)
    color_invisible = (255, 255, 255)

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


# Full-size frames (AllTracker resizes them itself).
_paths = sorted(glob.glob(os.path.join(str(IMAGE_DIR), IMAGE_PATTERN)), key=_numeric_key)
if not _paths:
    raise FileNotFoundError(f"No frames found in {IMAGE_DIR} matching {IMAGE_PATTERN}")
with ThreadPoolExecutor(8) as _ex:                    # as load_frames_256 decodes them, in threads
    video_np = np.stack(list(_ex.map(lambda p: np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8), _paths)))[None]
H, W = video_np.shape[2], video_np.shape[3]
T = video_np.shape[1]

if MASK_DIR is None or not Path(MASK_DIR).exists():
    raise FileNotFoundError(f"Mask directory not found: {MASK_DIR}")

dynamic_masks_base = load_dynamic_masks(
    str(MASK_DIR),
    MASK_PATTERN,
    t_expected=T,
    mask_mode=MASK_MODE,
    target_size=(W, H),
)
dynamic_masks_vis = dilate_dynamic_masks(dynamic_masks_base, DILATE_PX_VIS)

from utils.track_alltracker import grid_seeds, track_keyframes

dyn_masks_seed = dilate_dynamic_masks(dynamic_masks_base, DILATE_PX_QUERY)
dyn_masks_seed = np.asarray(dyn_masks_seed)
if dyn_masks_seed.ndim == 4:
    dyn_masks_seed = dyn_masks_seed[..., 0]
spacing = DYN_AT_GRID * max(H, W) / float(AT_RES)                  # grid spacing in native px
print(f"AllTracker: seeds every {spacing:.1f}px on the object mask, keyframe every {AT_KEYFRAME} frames, "
      f"{AT_RES}px, visible at vis x conf >= {AT_VIS_THRESHOLD}")
trk, vis, query_t = track_keyframes(
    video_np[0],
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

# run_pipeline.sh starts this step next to step 1: wait for the depth (its marker) before the 3D lift
_depth_done = os.environ.get("XM_WAIT_FOR_DEPTH")
if _depth_done and not os.path.exists(_depth_done):
    pause_script_logger("waiting for step 1's depth")              # (not this step's runtime)
    while not os.path.exists(_depth_done):
        if os.path.exists(_depth_done + ".failed"):
            raise RuntimeError("step 1 (depth) failed, so there is no depth to lift the tracks with")
        time.sleep(1)
    resume_script_logger()

depth_grad_deleted_obs_mask = np.zeros((Q, T), dtype=bool)     # no depth filter: written all False
depth_max_deleted_obs_mask = np.zeros((Q, T), dtype=bool)

depth_maps = np.load(DEPTH_MAPS_PATH)
if depth_maps.ndim != 3:
    raise ValueError(f"Expected depth_maps.npy shape (T,H,W), got {depth_maps.shape}")
T_lift = min(T, depth_maps.shape[0])
print(f"Loaded DA3 depth maps: {depth_maps.shape} from {DEPTH_MAPS_PATH}")

tracks_xyz = np.full((1, Q, T_lift, 3), np.nan, dtype=np.float64)

# Single intrinsic source: the depth is lifted with intrinsics.npy (identical to the static solver).
K_all_da3 = np.load(DA3_OUT_DIR / "intrinsics.npy").astype(np.float64)

for t in range(T_lift):
    K = K_all_da3[t]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    depth = depth_maps[t].astype(np.float64)

    H_s, W_s = depth.shape
    x_scale = W_s / float(W)
    y_scale = H_s / float(H)

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

    valid_depth = np.isfinite(z) & (z > 0)
    if not np.any(valid_depth):
        continue

    idx = idx_valid[valid_depth]
    zv = z[valid_depth].astype(np.float64)

    xv = x_depth[idx]
    yv = y_depth[idx]
    X = (xv - cx) / fx * zv
    Y = (yv - cy) / fy * zv
    Z = zv
    tracks_xyz[0, idx, t, :] = np.stack([X, Y, Z], axis=1).astype(np.float64)

# No track is truncated; these output keys say so (no track flagged, first flagged frame = T).
move_delete_query_mask = np.zeros(Q, dtype=bool)
move_delete_first_frame = np.full(Q, T, dtype=np.int64)
lost_offmask_first_frame = np.full(Q, T, dtype=np.int64)

# --- per-source occlusion breakdown: how many observations each source flips to occluded=True ---
# Sources OVERLAP (a point can be invisible AND off the mask), so per-source counts sum to more than
# the final ANY count. Counts are over "existing" observations only (frames at/after each track's birth).
_ti = np.arange(T, dtype=np.int64)[None, :]
_pre = _ti < query_t[:, None]                                    # (Q,T) pre-birth frames
_ex = ~_pre
_nex = int(_ex.sum())
_sources = [
    ("visibility(certainty)", occluded[0] & _ex),               # AllTracker visibility x confidence < threshold
    ("off-dynamic-mask", (~dynamic_obs_mask) & _ex),
]
print(f"Occlusion breakdown over {_nex} existing observations (sources overlap; they sum to >= final):")
for _nm, _m in _sources:
    _c = int(_m.sum())
    print(f"  {_nm:22s}: {_c:9d}  ({100.0 * _c / max(1, _nex):5.1f}%)")
_final = int((occluded_dynamic[0] & _ex).sum())
print(f"  {'=> final occluded (ANY)':22s}: {_final:9d}  ({100.0 * _final / max(1, _nex):5.1f}%)")
print(f"  (pre-query frames, always occluded: {int(_pre.sum())})")

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
    _viz_name = "viz_tracks_dynamic_only.mp4"

    _write_dynamic_reason_viz_mp4(
        video_np,
        tracks_xy,
        occluded_dynamic,
        str(OUT_DIR / _viz_name),
        raw_occluded=occluded,
        dynamic_obs_mask=dynamic_obs_mask,
        query_t=query_t,
        fps=10,
    )
    print(f"Wrote: {OUT_DIR / _viz_name}")
