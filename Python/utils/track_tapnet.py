"""TAPNext tracking helpers of the tracking steps: the frames at 256 x 256, the initial query grid, the moving-object
mask filter and the track visualisation video."""

import glob
import os
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image

from utils.masks import _numeric_key, dilate_dynamic_masks, load_dynamic_masks  # noqa: F401  (torch-free, re-exported)


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


def make_grid_query_points(
    grid: int,
    margin: int = 8,
    seed: int = 0,
) -> np.ndarray:
    """Random query points for TAPNext.

    Returns shape [1, Q, 3] with (t, x, y), all at t=0.
    """
    q = int(grid) * int(grid)
    rng = np.random.default_rng(seed)
    x = rng.uniform(margin, 255 - margin, size=(q,)).astype(np.float32)
    y = rng.uniform(margin, 255 - margin, size=(q,)).astype(np.float32)

    t_arr = np.full(q, 0.0, dtype=np.float32)
    return np.stack([t_arr, x, y], axis=1)[None, ...]


def apply_dynamic_mask_filter(
    tracks_xy: np.ndarray,
    occluded: np.ndarray,
    dynamic_masks: np.ndarray,
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

# Write viz on original image size (video_np contains original-size frames)
def write_viz_mp4(
    video_orig_np,
    tracks_xy,
    occluded,
    out_path: str,
    fps: int = 10,
    dynamic_masks: np.ndarray | None = None,
    mask_alpha: float = 0.35,
    highlight_radius: int = 5,
    highlight_obs_mask_3: np.ndarray | None = None,
    highlight_obs_color_3: tuple[int, int, int] = (255, 0, 255),
):
    import cv2

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # video_orig_np: [1, T, H, W, 3], uint8
    vid = video_orig_np[0]
    T, H, W, _ = vid.shape

    draw_tracks = tracks_xy is not None
    if draw_tracks:
        Q = tracks_xy.shape[1]
        rng = np.random.default_rng(0)
        colors = rng.integers(0, 255, size=(Q, 3), dtype=np.uint8)
        if highlight_obs_mask_3 is not None:
            highlight_obs_mask_3 = np.asarray(highlight_obs_mask_3).astype(bool)
            if highlight_obs_mask_3.shape != (1, Q, T):
                raise ValueError(
                    f"highlight_obs_mask_3 must have shape (1,Q,T)=(1,{Q},{T}), got {highlight_obs_mask_3.shape}"
                )

    frames_bgr = []

    # If dynamic_masks provided at original size, overlay directly.
    use_masks = dynamic_masks is not None

    for t in range(T):
        frame = vid[t].copy()
        # frame is RGB; convert to BGR for cv2 drawing
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # overlay dynamic mask (bool, at the frame size)
        if use_masks and t < dynamic_masks.shape[0]:
            m = dynamic_masks[t]
            if m.any():
                overlay = frame.copy()
                overlay[m] = np.array([0, 0, 255], dtype=np.uint8)
                frame = cv2.addWeighted(overlay, float(mask_alpha), frame, 1.0 - float(mask_alpha), 0.0)

        # draw tracks (if provided)
        if draw_tracks:
            for i in range(Q):
                c = tuple(int(v) for v in colors[i])

                x, y = tracks_xy[0, i, t]
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                xi, yi = int(round(float(x))), int(round(float(y)))
                if 0 <= xi < W and 0 <= yi < H:
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

                    if occluded[0, i, t]:
                        continue

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
