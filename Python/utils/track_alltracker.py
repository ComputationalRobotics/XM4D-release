"""AllTracker (Harley et al., ICCV 2025, github.com/aharley/alltracker) as the 2D tracker of step 2.

Dense tracking from keyframes: at every `keyframe`-th frame, `seed_fn` places new points where no tracked point is
visible nearby, and that keyframe's points are tracked forward to the end of the video with ONE dense AllTracker
pass from the keyframe (the flow of every pixel, sampled at the points). A point is visible where AllTracker's
visibility x confidence >= `vis_threshold`. With `min_new` > 0, a keyframe is skipped while it would add fewer new
points than `min_new` x the points visible there (each pass costs the same however few points it adds). Frames are
resized so that the long side is `res` px. `fp16` runs AllTracker under fp16 autocast.
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


def _sample(maps, g):
    """maps (Tf, C, h, w), g (n, 2) points in grid_sample coordinates -> (n, Tf, C), bilinear."""
    out = F.grid_sample(maps, g[None, None].expand(maps.shape[0], 1, -1, 2), mode="bilinear", padding_mode="border",
                        align_corners=True)
    return out[:, :, 0].permute(2, 0, 1)


def _encode(model, video, t, padder, device):
    """AllTracker encoder features of frames t of video (1, T, 3, h, w), encoded in ONE batch, normalised and padded as
    in Net.forward_sliding -> (len(t), C, H8, W8)."""
    images = video[:, t].to(device)
    mean = torch.as_tensor([0.485, 0.456, 0.406], device=device).reshape(1, 1, 3, 1, 1).to(images.dtype)
    std = torch.as_tensor([0.229, 0.224, 0.225], device=device).reshape(1, 1, 3, 1, 1).to(images.dtype)
    images = (images / 255.0 - mean) / std
    return model.get_fmaps(padder.pad(images.reshape(-1, *images.shape[2:]))[0], 1, len(t), None, False)


def _window(model, corr_block, fmap_anchor, fmaps2, flows8, visconfs8, iters):
    """Net.forward_window at inference (the defaults Net(16) is built with) without its full-resolution upsampling,
    which forward_sliding uses only for the last iteration and the frames it keeps -> flows8, visconfs8 (S, 2, H8, W8)
    and the last iteration's upsampling weights, for Net.upsample_data of those frames."""
    B, S, C, H8, W8 = fmaps2.shape
    dtype = fmaps2.dtype
    fmap1 = fmap_anchor.unsqueeze(1).repeat(1, S, 1, 1, 1).reshape(B * S, C, H8, W8).contiguous()
    fmap2 = fmaps2.reshape(B * S, C, H8, W8).contiguous()
    visconfs8 = visconfs8.reshape(B * S, 2, H8, W8).contiguous()
    corr_fn = corr_block(fmap1, fmap2, model.corr_levels, model.corr_radius)
    coords1 = model.coords_grid(B * S, H8, W8, device=fmap1.device, dtype=dtype)
    flowfeat, ctxfeat = torch.split(fmap1, [model.dim, model.dim], dim=1)
    time_emb = model.fetch_time_embed(S, ctxfeat.dtype, False).reshape(1, S, model.dim, 1, 1).repeat(B, 1, 1, 1, 1)
    ctxfeat = ctxfeat + time_emb.reshape(B * S, model.dim, 1, 1)
    for _ in range(iters):
        corr = corr_fn(coords1 + flows8).to(dtype)
        flowfeat = model.update_block(flowfeat, ctxfeat, visconfs8, corr, flows8, S)
        flows8 = flows8 + model.flow_head(flowfeat)
        visconfs8 = visconfs8 + model.visconf_head(flowfeat)
    return flows8, visconfs8, .25 * model.upsample_weight(flowfeat)


class _Features:
    """Encoder features of every frame, encoded in batches of S/2 consecutive frames from frame 0 (the last batch
    padded with the last frame), as forward_sliding encodes the new half of a window: the encoder output depends on the
    batch size, not on which frames share the batch, so these are the features forward_sliding computes. A batch is
    encoded when first needed and dropped once the passes start after it (they only move forward): with max_len the
    cache holds about max_len + S frames instead of the whole video."""

    def __init__(self, encode, T, half):
        self.encode, self.T, self.half, self.batches = encode, T, half, {}

    def __getitem__(self, idx):                    # (len(idx), C, H8, W8)
        for b in np.unique(np.asarray(idx) // self.half).tolist():
            if b not in self.batches:
                t = b * self.half
                self.batches[b] = self.encode(np.minimum(np.arange(t, t + self.half), self.T - 1))[:self.T - t]
        return torch.stack([self.batches[i // self.half][i % self.half] for i in np.asarray(idx).tolist()])

    def drop_before(self, t):
        for b in [b for b in self.batches if (b + 1) * self.half <= t]:
            del self.batches[b]


def _pass(model, video, t0, t1, g, iters, feats, padder, device):
    """Net.forward_sliding(video[:, t0:t1]) at inference -- its time padding (the last frame repeated), windows,
    carry-over between windows and fill rule -- but returning only the final flow and visconf of frames t0+1..t1-1
    sampled at the points g (n, 2) (grid_sample coordinates), (n, t1-t0-1, 2) each on the GPU, instead of the dense maps
    of every frame copied to the CPU. The new half of each later window takes its encoder features from feats (every
    frame of the video encoded in a batch of S/2, as forward_sliding encodes it there) instead of re-encoding them."""
    corr_block = sys.modules[type(model).__module__].CorrBlock      # nets.alltracker's, as Net uses it
    Ts = t1 - t0
    enc = lambda t: _encode(model, video, t0 + np.minimum(t, Ts - 1), padder, device)
    if Ts == 2:                                                         # forward_sliding's two-frame branch
        fmaps = enc(np.arange(2))
        z = torch.zeros((1, 2) + fmaps.shape[-2:], dtype=video.dtype, device=device)
        flows8, visconfs8, weight = _window(model, corr_block, fmaps[:1], fmaps[None, 1:], z, z, iters)
        return (_sample(padder.unpad(model.upsample_data(flows8, weight)), g),
                _sample(padder.unpad(torch.sigmoid(model.upsample_data(visconfs8, weight))), g))
    S = model.seqlen
    half = S // 2                                                       # the window stride
    indices, start = [], 0                                              # as Net.get_T_padded_images
    while start + S < Ts:
        indices.append(start)
        start += half
    indices.append(start)
    fmap_anchor = enc(np.arange(1))
    flows, visconfs = (torch.empty((len(g), Ts - 1, 2), device=device) for _ in range(2))
    filled = 0
    for ii, ind in enumerate(indices):
        if ii == 0:
            flows8 = visconfs8 = torch.zeros((S, 2) + fmap_anchor.shape[-2:], dtype=video.dtype, device=device)
            fmaps2 = enc(np.arange(S))[None]
        else:                                       # the previous window's second half, then its last frame repeated
            flows8 = torch.cat([flows8[half:], flows8[-1:].repeat(half, 1, 1, 1)])
            visconfs8 = torch.cat([visconfs8[half:], visconfs8[-1:].repeat(half, 1, 1, 1)])
            new = t0 + np.minimum(np.arange(ind + half, ind + S), Ts - 1)
            fmaps2 = torch.cat([fmaps2[:, half:], feats[new][None]], dim=1)
        flows8, visconfs8, weight = _window(model, corr_block, fmap_anchor, fmaps2, flows8, visconfs8, iters)
        fill = np.arange(filled, ind + S)                               # the window's frames no earlier window filled
        filled = ind + S
        keep = np.flatnonzero((fill >= 1) & (fill < Ts))                # minus the keyframe and the time padding
        k = S - len(fill) + keep                                        # their positions in the window
        flows[:, fill[keep] - 1] = _sample(padder.unpad(model.upsample_data(flows8[k], weight[k])), g)
        visconf = torch.sigmoid(model.upsample_data(visconfs8[k], weight[k]))
        visconfs[:, fill[keep] - 1] = _sample(padder.unpad(visconf), g)
    return flows, visconfs


@torch.no_grad()
def track_keyframes(frames, seed_fn, *, keyframe=10, res=512, vis_threshold=0.6, max_points=None,
                    min_new=0.0, fp16=False, device="cuda", label="AllTracker"):
    """frames (T, H, W, 3) uint8 RGB. seed_fn(t, visible_xy (n, 2)) -> new points (m, 2) at frame t, native px.
    Returns tracks (N, T, 2) float32 native px (NaN before each point's keyframe), visible (N, T) bool, query_t (N,)."""
    T, H, W = frames.shape[:3]
    s = res / float(max(H, W))
    h, w = max(8, int(round(H * s / 8)) * 8), max(8, int(round(W * s / 8)) * 8)
    sx, sy = w / float(W), h / float(H)
    video = torch.from_numpy(np.stack([np.asarray(Image.fromarray(f).resize((w, h), Image.BILINEAR)) for f in frames]))
    video = video.permute(0, 3, 1, 2).float()[None]                        # [1, T, 3, h, w], 0..255, on CPU
    model = load_model(device)
    padder = sys.modules[type(model).__module__].InputPadder(video.shape)
    feats = _Features(lambda t: _encode(model, video, t, padder, device), T, model.seqlen // 2)   # see _pass
    tracks, vis, qts, n_total = [], [], [], 0
    keyframe = max(1, int(keyframe))
    for t0 in tqdm(range(T), desc=label):
        if t0 % keyframe:
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
        t1 = T
        if t0 < t1 - 1:
            q = np.stack([new[:, 0] * sx, new[:, 1] * sy], 1)
            g = torch.as_tensor(np.stack([q[:, 0] / (w - 1) * 2 - 1, q[:, 1] / (h - 1) * 2 - 1], -1),
                                dtype=torch.float32, device=device)                         # grid_sample coordinates
            with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
                feats.drop_before(t0)
                flows, visconfs = _pass(model, video, t0, t1, g, 4, feats, padder, device)
            for c in range(0, t1 - t0 - 1, 64):                            # frame chunks keep CPU memory flat
                f_ = flows[:, c:c + 64].cpu().numpy()
                vc = visconfs[:, c:c + 64].cpu().numpy()
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
