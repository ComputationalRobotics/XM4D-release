"""AllTracker (Harley et al., ICCV 2025, github.com/aharley/alltracker) as the 2D tracker of step 2.

Dense tracking from keyframes: at every `keyframe`-th frame, `seed_fn` places new points where no tracked point is
visible nearby, and that keyframe's points are tracked forward to the end of the video (or `max_len` frames) with ONE
dense AllTracker pass from the keyframe (the flow of every pixel, sampled at the points). A point is visible where
AllTracker's visibility x confidence >= `vis_threshold`. With `min_visible` > 0, any frame where fewer points are
visible also becomes a keyframe. With `min_new` > 0, a keyframe is skipped while it would add fewer new points than
`min_new` x the points visible there (each pass costs the same however few points it adds). Frames are resized so that
the long side is `res` px. `fp16` runs AllTracker under fp16 autocast.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

AT_DIR = Path(__file__).resolve().parents[2] / "deps" / "alltracker"
CHECKPOINT = AT_DIR / "checkpoints" / "alltracker.pth"
CHECKPOINT_URL = "https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth"


def load_model(device):
    # AllTracker's code imports its own top-level `utils` package, like this repo: set ours aside while importing.
    ours = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "utils" or k.startswith("utils.")}
    sys.path.insert(0, str(AT_DIR))
    try:
        from nets.alltracker import Net
    finally:
        sys.path.remove(str(AT_DIR))
        for k in [k for k in sys.modules if k == "utils" or k.startswith("utils.")]:
            del sys.modules[k]
        sys.modules.update(ours)
    if CHECKPOINT.exists():
        state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    else:
        state = torch.hub.load_state_dict_from_url(CHECKPOINT_URL, map_location="cpu")
    model = Net(16)
    model.load_state_dict(state["model"])
    return model.to(device).eval()


def _sample(maps, pts):
    """maps (Tf, C, h, w), pts (n, 2) xy in map px -> (n, Tf, C), bilinear."""
    Tf, C, h, w = maps.shape
    g = torch.as_tensor(np.stack([pts[:, 0] / (w - 1) * 2 - 1, pts[:, 1] / (h - 1) * 2 - 1], -1),
                        dtype=maps.dtype, device=maps.device)
    out = F.grid_sample(maps, g[None, None].expand(Tf, 1, -1, 2), mode="bilinear", padding_mode="border", align_corners=True)
    return out[:, :, 0].permute(2, 0, 1)


@torch.no_grad()
def track_keyframes(frames, seed_fn, *, keyframe=10, res=512, vis_threshold=0.6, max_points=None, max_len=None,
                    min_visible=0, min_new=0.0, fp16=False, device="cuda", label="AllTracker"):
    """frames (T, H, W, 3) uint8 RGB. seed_fn(t, visible_xy (n, 2)) -> new points (m, 2) at frame t, native px.
    Returns tracks (N, T, 2) float32 native px (NaN before each point's keyframe), visible (N, T) bool, query_t (N,)."""
    T, H, W = frames.shape[:3]
    s = res / float(max(H, W))
    h, w = max(8, int(round(H * s / 8)) * 8), max(8, int(round(W * s / 8)) * 8)
    sx, sy = w / float(W), h / float(H)
    video = torch.from_numpy(np.stack([np.asarray(Image.fromarray(f).resize((w, h), Image.BILINEAR)) for f in frames]))
    video = video.permute(0, 3, 1, 2).float()[None]                        # [1, T, 3, h, w], 0..255, on CPU
    model = load_model(device)
    tracks, vis, qts, n_total = [], [], [], 0
    keyframe = max(1, int(keyframe))
    for t0 in tqdm(range(T), desc=label):
        if t0 % keyframe and (min_visible <= 0 or sum(int(v[:, t0].sum()) for v in vis) >= min_visible):
            continue
        cur = [tr[v[:, t0], t0] for tr, v in zip(tracks, vis)]
        cur = np.concatenate(cur) if cur else np.zeros((0, 2))
        new = np.asarray(seed_fn(t0, cur), np.float64).reshape(-1, 2)
        if max_points is not None:
            new = new[:max(0, int(max_points) - n_total)]
        if not len(new) or len(new) < min_new * len(cur):
            continue
        tr = np.full((len(new), T, 2), np.nan, np.float32)
        v = np.zeros((len(new), T), bool)
        tr[:, t0], v[:, t0] = new, True
        t1 = T if not max_len else min(T, t0 + int(max_len) + 1)
        if t0 < t1 - 1:
            with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
                flows, visconfs, _, _ = model.forward_sliding(video[:, t0:t1].to(device), iters=4, sw=None, is_training=False)
            flows, visconfs = flows.float(), visconfs.float()
            if flows.dim() == 4:                                            # a single next frame comes back unstacked
                flows, visconfs = flows[:, None], visconfs[:, None]
            else:                                                           # drop the keyframe itself
                flows, visconfs = flows[:, 1:], visconfs[:, 1:]
            q = np.stack([new[:, 0] * sx, new[:, 1] * sy], 1)
            for c in range(0, flows.shape[1], 64):                         # frame chunks keep GPU memory flat
                f_ = _sample(flows[0, c:c + 64].to(device), q).cpu().numpy()
                vc = _sample(visconfs[0, c:c + 64].to(device), q).cpu().numpy()
                p = q[:, None] + f_
                tr[:, t0 + 1 + c:t0 + 1 + c + f_.shape[1]] = np.stack([p[..., 0] / sx, p[..., 1] / sy], -1)
                v[:, t0 + 1 + c:t0 + 1 + c + f_.shape[1]] = vc[..., 0] * vc[..., 1] >= vis_threshold
            del flows, visconfs
            if device != "cpu" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        tracks.append(tr); vis.append(v); qts.append(np.full(len(new), t0, np.int64)); n_total += len(new)
    if not tracks:
        return np.zeros((0, T, 2), np.float32), np.zeros((0, T), bool), np.zeros(0, np.int64)
    return np.concatenate(tracks), np.concatenate(vis), np.concatenate(qts)


def grid_seeds(mask, spacing, covered_xy, radius):
    """Points on a regular grid (spacing px, native) inside `mask` (H, W bool), minus those within `radius` px of a
    point in covered_xy (n, 2)."""
    H, W = mask.shape
    off = spacing / 2.0
    ys, xs = np.mgrid[off:H:spacing, off:W:spacing]
    xs, ys = xs.ravel(), ys.ravel()
    keep = mask[np.clip(np.round(ys).astype(int), 0, H - 1), np.clip(np.round(xs).astype(int), 0, W - 1)]
    pts = np.stack([xs[keep], ys[keep]], 1)
    if len(pts) and len(covered_xy):
        from scipy.spatial import cKDTree
        d, _ = cKDTree(covered_xy).query(pts, k=1)
        pts = pts[d > radius]
    return pts
