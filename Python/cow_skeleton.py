#!/usr/bin/env python3
"""CoWTracker skeleton for cow_attach_tracks.py (env SpaTrack2, ~37 GB GPU with the 100-frame window).

Runs CoWTracker (dense, frame-anchored) restarted every --every frames: run k uses reference frame s = k*every, with
fresh queries sampled on a --stride grid inside the (eroded) mask of that frame, and tracks BOTH directions: forward on
frames [s, T) and backward on the reversed frames [s, s-1, ..., 0] (same reference frame, so the two halves join at s).
All runs are stacked into ONE skeleton: tracks [N,T,2] spanning the whole video, raw visibility and confidence [N,T],
start_frame [N] (= reference frame), queries [N,2].

Usage: python cow_skeleton.py --images-dir <dir> --masks-dir <dir> --out <npz> [--every 50 --stride 8]  (bidirectional)
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "deps/cowtracker"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", required=True); p.add_argument("--image-pattern", default="*.png")
    p.add_argument("--masks-dir", required=True); p.add_argument("--mask-pattern", default="*.png")
    p.add_argument("--out", required=True)
    p.add_argument("--every", type=int, default=50, help="restart period: reference frames 0, every, 2*every, ...")
    p.add_argument("--stride", type=int, default=8, help="query grid stride (px) inside the reference-frame mask")
    p.add_argument("--erode-px", type=int, default=2)
    p.add_argument("--window", type=int, default=100, help="CoWTrackerWindowed window_len (= stride)")
    p.add_argument("--cow-size", type=int, nargs=2, default=(448, 336), metavar=("H", "W"), help="model input (multiples of 14)")
    a = p.parse_args()
    from cowtracker import CoWTrackerWindowed

    ip = sorted(glob.glob(str(Path(a.images_dir) / a.image_pattern)))
    mp = sorted(glob.glob(str(Path(a.masks_dir) / a.mask_pattern)))[: len(ip)]
    assert len(ip) and len(mp) == len(ip), (len(ip), len(mp))
    frames = np.stack([np.asarray(Image.open(f).convert("RGB")) for f in ip]); T, H0, W0, _ = frames.shape
    masks = np.stack([np.asarray(Image.open(f).convert("L")) > 0 for f in mp])
    Hc, Wc = a.cow_size; sc = np.array([Wc / W0, Hc / H0], np.float32)
    video = torch.from_numpy(np.stack([np.asarray(Image.fromarray(f).resize((Wc, Hc), Image.BILINEAR)) for f in frames])).permute(0, 3, 1, 2).float().cuda()
    model = CoWTrackerWindowed.from_checkpoint(None, window_len=a.window, stride=a.window, device="cuda", dtype=torch.float16)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.erode_px + 1,) * 2) if a.erode_px > 0 else None

    TR, VIS, CONF, ST, Q = [], [], [], [], []
    for s in range(0, T - 1, a.every):
        m = masks[s].astype(np.uint8) * 255
        if k is not None:
            m = cv2.erode(m, k)
        ys, xs = np.where(m > 0); keep = (xs % a.stride == 0) & (ys % a.stride == 0)
        q = np.stack([xs[keep], ys[keep]], 1).astype(np.float32)
        if len(q) == 0:
            print(f"[cow_skeleton] start {s}: empty mask, skipped"); continue
        qi = np.clip(np.rint(q * sc), 0, [Wc - 1, Hc - 1]).astype(int)
        tr = np.full((len(q), T, 2), np.nan, np.float32); v = np.zeros((len(q), T), np.float32); c = np.zeros((len(q), T), np.float32)

        def run(clip):  # dense CoWTracker on a clip whose frame 0 is the reference -> per-query [N,len,2], [N,len], [N,len]
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                pred = model.forward(video=clip, queries=None)
            out = ((pred["track"][0].float().cpu().numpy()[:, qi[:, 1], qi[:, 0]] / sc).transpose(1, 0, 2),
                   pred["vis"][0].float().cpu().numpy()[:, qi[:, 1], qi[:, 0]].T, pred["conf"][0].float().cpu().numpy()[:, qi[:, 1], qi[:, 0]].T)
            del pred; torch.cuda.empty_cache(); return out
        tf, vf, cf = run(video[s:])                                   # forward: frames s..T-1
        tr[:, s:], v[:, s:], c[:, s:] = tf, vf, cf
        if s > 0:                                                     # backward: reversed clip s, s-1, ..., 0 (index k -> frame s-k)
            tb, vb, cb = run(video[: s + 1].flip(0).contiguous())
            tr[:, :s], v[:, :s], c[:, :s] = tb[:, 1:][:, ::-1], vb[:, 1:][:, ::-1], cb[:, 1:][:, ::-1]
        TR.append(tr); VIS.append(v); CONF.append(c); ST.append(np.full(len(q), s, np.int32)); Q.append(q)
        print(f"[cow_skeleton] ref {s}: {len(q)} queries, forward [{s},{T - 1}]" + (f" + backward [{s},0]" if s > 0 else ""), flush=True)
    np.savez_compressed(a.out, tracks=np.concatenate(TR), vis=np.concatenate(VIS), conf=np.concatenate(CONF),
                        start_frame=np.concatenate(ST), queries=np.concatenate(Q), every=a.every, stride=a.stride)
    print(f"[cow_skeleton] {sum(len(q) for q in Q)} skeleton tracks from {len(Q)} runs -> {a.out}")


if __name__ == "__main__":
    main()
