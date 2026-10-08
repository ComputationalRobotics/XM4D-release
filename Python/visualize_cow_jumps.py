#!/usr/bin/env python3
"""Detect large 2D jumps in the CoWTracker skeleton tracks (outputs/cow_skeleton.npz) and CHANGE THE COLOUR of a
track from the frame its first jump is detected, so drift/identity swaps are immediately visible.

Two triggers (both on the GATED track: vis > --cow-vis & conf > --cow-conf):
  (a) STEP   two consecutive visible frames with ‖x(t) - x(t-1)‖ > --jump-px.
  (b) GAP    a reappearance after an invisible gap: the expected position at the reappearance frame is predicted
             from the OTHER points visible on both sides of the gap with a 2D similarity fit (Umeyama, so camera
             motion and rigid rotation are accounted for); a deviation > --gap-tol-px is a jump. With fewer than
             --min-ref reference points, fall back to ‖x(b) - x(a)‖ > --gap-tol-px.
Colours: rainbow (hue by angle around the frame-0 query centroid) BEFORE the first trigger; after it the track is
drawn in --after-color (default red). The trigger frame itself is drawn as a larger yellow ring. With
--mode segment each further trigger advances to the next palette colour instead of staying red.

Usage: XM_DATASET_NAME=<scene> python visualize_cow_jumps.py [--jump-px 20] [--gap-tol-px 60] [--out <mp4>]
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
def _similarity_predict(src, dst, x):
    """Fit a 2D similarity (scale, rotation, translation; Umeyama) from src->dst points [M,2] and map x [2].
    (Inlined from compare_trackers_masked.py so the pipeline does not depend on the comparison harness.)"""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    cov = D.T @ S / len(src)
    U, sig, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, d]) @ Vt
    scale = (sig[0] + d * sig[1]) / max((S ** 2).sum() / len(src), 1e-9)
    return scale * (R @ (x - mu_s)) + mu_d

PALETTE = [(0, 0, 255), (0, 140, 255), (0, 255, 255), (255, 0, 255), (255, 255, 255)]


def detect_jumps(tracks, vis, jump_px, gap_tol_px, min_ref):
    """-> trigger [N,T] bool (jump DETECTED at this frame), cause [N,T] in {0 none, 1 step, 2 gap}."""
    N, T = vis.shape
    trig = np.zeros((N, T), bool); cause = np.zeros((N, T), np.uint8)
    for i in range(N):
        f = np.where(vis[i])[0]
        if f.size < 2:
            continue
        for a, b in zip(f[:-1], f[1:]):
            dist = float(np.linalg.norm(tracks[i, b] - tracks[i, a]))
            if b - a == 1:                                     # (a) consecutive-frame step
                if dist > jump_px:
                    trig[i, b] = True; cause[i, b] = 1
            else:                                              # (b) reappearance after a gap
                ref = np.where(vis[:, a] & vis[:, b])[0]; ref = ref[ref != i]
                if ref.size >= min_ref:
                    pred = _similarity_predict(tracks[ref, a], tracks[ref, b], tracks[i, a])
                    bad = float(np.linalg.norm(tracks[i, b] - pred)) > gap_tol_px
                else:
                    bad = dist > gap_tol_px
                if bad:
                    trig[i, b] = True; cause[i, b] = 2
    return trig, cause


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-dir", type=Path, default=None); ap.add_argument("--skeleton", type=Path, default=None)
    ap.add_argument("--cow-vis", type=float, default=None); ap.add_argument("--cow-conf", type=float, default=None)
    ap.add_argument("--jump-px", type=float, default=20.0, help="consecutive-frame step that counts as a jump (windmill: p99 of normal motion is 16 px)")
    ap.add_argument("--gap-tol-px", type=float, default=60.0, help="deviation from the similarity-predicted position at a reappearance")
    ap.add_argument("--min-ref", type=int, default=4)
    ap.add_argument("--mode", choices=["first", "segment"], default="first")
    ap.add_argument("--after-color", type=int, nargs=3, default=(0, 0, 255), metavar=("B", "G", "R"))
    ap.add_argument("--out", type=Path, default=None); ap.add_argument("--fps", type=int, default=10); ap.add_argument("--radius", type=int, default=3)
    a = ap.parse_args()
    import utils.definitions as D
    from utils import dataset_config
    S = (a.scene_dir or Path(D.rootpath)).resolve(); out = S / "outputs"
    gv = a.cow_vis if a.cow_vis is not None else dataset_config.get_float("COW_GATE_VIS", 0.5)
    gc = a.cow_conf if a.cow_conf is not None else dataset_config.get_float("COW_GATE_CONF", 0.5)
    d = np.load(a.skeleton or (out / "cow_skeleton.npz"))
    tr = d["tracks"].astype(np.float64); vis = (d["vis"] > gv) & (d["conf"] > gc) & np.isfinite(tr).all(-1)
    N, T = vis.shape
    trig, cause = detect_jumps(tr, vis, a.jump_px, a.gap_tol_px, a.min_ref)
    ever = trig.any(1); first = np.where(ever, trig.argmax(1), T)
    n_step, n_gap = int((cause == 1).sum()), int((cause == 2).sum())
    print(f"[jumps] gate vis>{gv} conf>{gc} | {N} skeleton tracks, {int(vis.sum())} visible point-frames")
    print(f"[jumps] {int(ever.sum())}/{N} tracks ({100*ever.mean():.1f}%) jump at least once; "
          f"{n_step + n_gap} triggers total ({n_step} step > {a.jump_px:g}px, {n_gap} gap > {a.gap_tol_px:g}px)")
    print(f"[jumps] visible point-frames after a track's first jump: {int((vis & (np.arange(T)[None] >= first[:, None])).sum())} "
          f"({100 * (vis & (np.arange(T)[None] >= first[:, None])).sum() / max(vis.sum(), 1):.1f}% of all visible)")
    ip = sorted(glob.glob(str(S / "images" / "*.png"))); mp = sorted(glob.glob(str(S / "masks" / "*.png")))
    frames = [cv2.imread(p) for p in ip]; H, W = frames[0].shape[:2]
    q = d["queries"]; cx, cy = q[:, 0].mean(), q[:, 1].mean()
    hue = ((np.arctan2(q[:, 1] - cy, q[:, 0] - cx) + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    base = cv2.cvtColor(np.stack([hue, np.full(N, 255, np.uint8), np.full(N, 255, np.uint8)], 1)[None], cv2.COLOR_HSV2BGR)[0].astype(int)
    n_trig_before = np.cumsum(trig, axis=1)                       # triggers strictly up to and including t
    head, foot = 26, 20; PH = head + H + foot
    outp = (a.out or (out / "cow_skeleton_jumps.mp4")); tmp = outp.with_suffix(".tmp.avi")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"MJPG"), a.fps, (W, PH))
    for t in range(T):
        img = frames[t].copy()
        if t < len(mp):
            mk = cv2.imread(mp[t], 0); cnts, _ = cv2.findContours((mk > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, (255, 255, 255), 1)
        n_ok = n_bad = 0
        for i in np.where(vis[:, t])[0]:
            pt = (int(round(tr[i, t, 0])), int(round(tr[i, t, 1])))
            k = int(n_trig_before[i, t])
            if k == 0:
                c = tuple(int(x) for x in base[i]); n_ok += 1
            else:
                c = tuple(int(x) for x in a.after_color) if a.mode == "first" else PALETTE[(k - 1) % len(PALETTE)]
                n_bad += 1
            cv2.circle(img, pt, a.radius, c, -1, cv2.LINE_AA)
            if trig[i, t]:
                cv2.circle(img, pt, a.radius + 3, (0, 255, 255), 1, cv2.LINE_AA)   # jump detected THIS frame
        pnl = np.zeros((PH, W, 3), np.uint8); pnl[head:head + H] = img
        cv2.putText(pnl, f"CoWTracker skeleton, gate {gv}/{gc}: colour changes after a jump (step>{a.jump_px:g}px or gap>{a.gap_tol_px:g}px)",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(pnl, f"clean {n_ok}  jumped {n_bad}  (this frame {int(trig[:, t].sum())} yellow rings)   t={t:03d}/{T-1}",
                    (6, head + H + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        vw.write(pnl)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(outp)], check=True); tmp.unlink()
    np.savez(out / "cow_skeleton_jumps.npz", trigger=trig, cause=cause, first_jump=first, gate=[gv, gc], jump_px=a.jump_px, gap_tol_px=a.gap_tol_px)
    (out / "cow_skeleton_jumps.json").write_text(json.dumps({"tracks": int(N), "tracks_with_jump": int(ever.sum()),
        "triggers_step": n_step, "triggers_gap": n_gap, "gate": [gv, gc], "jump_px": a.jump_px, "gap_tol_px": a.gap_tol_px}, indent=2))
    print("video ->", outp)


if __name__ == "__main__":
    main()
