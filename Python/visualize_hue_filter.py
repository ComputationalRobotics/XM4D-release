#!/usr/bin/env python3
"""Before/after video for filter (b) of cow_attach_tracks.py -- COLOUR vs the track's first frame.

Reproduces the pipeline's own filter chain on the CoWTracker skeleton (gate -> filter (a) jumps ->
filter (b) colour) and renders a side-by-side comparison of the state around filter (b):

  LEFT  "before": every track visible under the gate + filter (a).  Points that filter (b) will cut are
                  drawn RED from their cut frame on, so you see exactly what is about to be removed, and
                  ORANGE before it (the part that survives).
  RIGHT "after" : what the pipeline keeps.

Filter (b) is TERMINAL (cow_attach_tracks._truncate_from): from the first observation whose hue deviates
more than --cow-hue-deg from the track's reference hue (circular mean of its first 3 visible SATURATED
frames), the rest of the track is dropped -- it is a truncation, not a whole-track deletion, so a track
usually contributes observations to both panels.

Run: XM_DATASET_CAT=Dycheck XM_DATASET_NAME=teddy conda run -n XM python visualize_hue_filter.py
"""
import argparse
import glob
from pathlib import Path

import cv2
import numpy as np

import utils.definitions as D
from utils import dataset_config
from cow_attach_tracks import filter_color_vs_median, filter_jumpy_tracks


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skeleton", default="cow_skeleton.npz")
    ap.add_argument("--out", default="hue_filter_compare.mp4")
    ap.add_argument("--cow-vis", type=float, default=dataset_config.get_float("COW_GATE_VIS", 0.5))
    ap.add_argument("--cow-conf", type=float, default=dataset_config.get_float("COW_GATE_CONF", 0.5))
    ap.add_argument("--cow-hue-deg", type=float, default=dataset_config.get_float("COW_HUE_DEG", 30.0))
    ap.add_argument("--cow-hue-outlier-frac", type=float,
                    default=dataset_config.get_float("COW_HUE_OUTLIER_FRAC", 0.3))
    ap.add_argument("--cow-hue-win", type=int, default=dataset_config.get_int("COW_HUE_WIN", 4))
    ap.add_argument("--cow-hue-strict-deg", type=float,
                    default=dataset_config.get_float("COW_HUE_STRICT_DEG", 15.0))
    ap.add_argument("--cow-jump-px", type=float, default=dataset_config.get_float("COW_JUMP_PX", 20.0))
    ap.add_argument("--cow-gap-tol-px", type=float, default=dataset_config.get_float("COW_GAP_TOL_PX", 60.0))
    ap.add_argument("--cow-max-jumps", type=int, default=dataset_config.get_int("COW_MAX_JUMPS", 5))
    ap.add_argument("--no-jump-filter", action="store_true", help="isolate filter (b): skip filter (a)")
    ap.add_argument("--video", action="store_true",
                    help="write the side-by-side mp4 instead of opening the interactive inspector")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--scale", type=float, default=1.0)
    return ap.parse_args()


def sample_bgr(tracks, vis, frames, patch=1):
    """Patch-median BGR at every visible track position (the exact pixels filter (b) judges)."""
    N, T = vis.shape
    bgr = np.full((N, T, 3), np.nan)
    for t in range(T):
        idx = np.where(vis[:, t])[0]
        if idx.size == 0 or frames[t] is None:
            continue
        H, W = frames[t].shape[:2]
        for q in idx:
            x = int(np.clip(round(tracks[q, t, 0]), 0, W - 1)); y = int(np.clip(round(tracks[q, t, 1]), 0, H - 1))
            win = frames[t][max(y - patch, 0):y + patch + 1, max(x - patch, 0):x + patch + 1].reshape(-1, 3)
            bgr[q, t] = np.median(win, axis=0)
    return bgr


def inspect(a, cow, before, after, cut, frames, T):
    """Interactive per-track inspector: for each track filter (b) truncated, show the REFERENCE colour
    (first visible saturated frames) against the colour AT THE CURRENT FRAME, their hue deviation, and
    where the cut falls on the track's timeline."""
    bgr = sample_bgr(cow, before, frames)
    hsv = np.full_like(bgr, np.nan)
    ok = np.isfinite(bgr).all(-1)
    if ok.any():
        hsv[ok] = cv2.cvtColor(bgr[ok].astype(np.uint8)[None], cv2.COLOR_BGR2HSV)[0]
    hue, sat = hsv[..., 0], hsv[..., 1]
    ang = hue / 180.0 * 2 * np.pi
    judged = np.isfinite(ang) & (sat > 70)                            # same saturation gate as the filter
    hue_win = None
    if a.cow_hue_win > 0:                                             # the filter also searches a window and
        from edg_viz_common import _circ_hue_median                   # takes min(centre, window) -- so must we,
        from cow_attach_tracks import sample_hue_best_in_window       # or a kept frame can read as dev > thresh
        N = hue.shape[0]
        ref0 = np.array([_circ_hue_median(hue[i, judged[i]]) if judged[i].any() else np.nan
                         for i in range(N)])
        hue_win, _ = sample_hue_best_in_window(cow, before, frames, ref0, int(a.cow_hue_win), 70)

    ids = np.where(cut.any(1))[0]
    ids = ids[np.argsort(-cut[ids].sum(1))]                           # most-removed first
    if ids.size == 0:
        print("[hue-filter] filter (b) cut nothing; nothing to inspect"); return
    print(f"[hue-filter] {ids.size} truncated tracks. N/P = next/prev track, SPACE/B = frame, Q = quit")

    k = 0
    while True:
        q = int(ids[k])
        from edg_viz_common import _circ_hue_median
        f_j = np.where(judged[q])[0]
        ref_hue = _circ_hue_median(hue[q, f_j])                       # reference = median over ALL judged obs
        ref_ang = ref_hue / 180.0 * 2 * np.pi
        dall = np.minimum(np.abs(hue[q, f_j] - ref_hue), 180.0 - np.abs(hue[q, f_j] - ref_hue))
        ref_bgr = np.median(bgr[q, f_j[dall <= a.cow_hue_deg]], axis=0).astype(np.uint8) \
            if (dall <= a.cow_hue_deg).any() else np.median(bgr[q, f_j], axis=0).astype(np.uint8)
        cut_f = int(np.where(cut[q])[0][0])
        t = cut_f
        while True:
            img = frames[t].copy() if frames[t] is not None else np.zeros((480, 360, 3), np.uint8)
            H, W = img.shape[:2]
            f0 = max(0, t - 20)                                       # trail
            prev = None
            for tt in range(f0, t + 1):
                if not before[q, tt]:
                    prev = None; continue
                pt = (int(round(cow[q, tt, 0])), int(round(cow[q, tt, 1])))
                if prev is not None:
                    cv2.line(img, prev, pt, (0, 0, 255) if cut[q, tt] else (60, 220, 60), 1, cv2.LINE_AA)
                prev = pt
            dev = np.nan
            if before[q, t] and np.isfinite(cow[q, t]).all():
                pt = (int(round(cow[q, t, 0])), int(round(cow[q, t, 1])))
                cv2.circle(img, pt, 6, (0, 0, 255) if cut[q, t] else (60, 220, 60), 2, cv2.LINE_AA)
                if judged[q, t]:
                    dev = abs(np.angle(np.exp(1j * (ang[q, t] - ref_ang)))) / np.pi * 180.0
                    if hue_win is not None and np.isfinite(hue_win[q, t]):
                        dw = abs(hue_win[q, t] - ref_hue); dw = min(dw, 180.0 - dw)
                        dev = min(dev, dw)                            # the window can only help
            img = cv2.resize(img, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_NEAREST)
            pan = np.zeros((img.shape[0], 260, 3), np.uint8)
            cv2.rectangle(pan, (12, 40), (112, 100), tuple(int(c) for c in ref_bgr), -1)
            cv2.putText(pan, "REF (median)", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            cv2.putText(pan, f"hue {ref_hue:.0f}", (12, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            now = bgr[q, t]
            if np.isfinite(now).all():
                cv2.rectangle(pan, (140, 40), (240, 100), tuple(int(c) for c in now.astype(np.uint8)), -1)
                cv2.putText(pan, f"hue {hue[q, t]:.0f}", (140, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            (255, 255, 255), 1)
            cv2.putText(pan, f"NOW f{t}", (140, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            dcol = (0, 0, 255) if np.isfinite(dev) and dev > a.cow_hue_deg else (60, 220, 60)
            cv2.putText(pan, (f"dev {dev:.0f} deg" if np.isfinite(dev) else "dev -- (unsaturated)"),
                        (12, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, dcol, 1)
            cv2.putText(pan, f"thresh {a.cow_hue_deg:.0f} deg", (12, 174), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (200, 200, 200), 1)
            cv2.putText(pan, f"track {k+1}/{ids.size}  gid {q}", (12, 204), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1)
            n_keep = int(after[q].sum()); n_lost = int(cut[q].sum())
            whole = n_keep == 0
            cv2.putText(pan, (f"EMPTY after filter" if whole else f"PARTIAL: {n_keep} frames KEPT"),
                        (12, 226), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 0, 255) if whole else (60, 220, 60), 1)
            cv2.putText(pan, f"lost {n_lost}  first cut f{cut_f}", (12, 246),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 160, 255), 1)
            y0 = 268                                                  # timeline: green kept, red cut
            for tt in range(T):
                x = 12 + int(236 * tt / max(T - 1, 1))
                c = ((0, 0, 255) if cut[q, tt] else (60, 220, 60)) if before[q, tt] else (70, 70, 70)
                cv2.line(pan, (x, y0), (x, y0 + 16), c, 1)
            cv2.line(pan, (12 + int(236 * t / max(T - 1, 1)), y0 - 4),
                     (12 + int(236 * t / max(T - 1, 1)), y0 + 20), (255, 255, 255), 1)
            cv2.imshow("filter (b) colour vs first frame", np.hstack([img, pan]))
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("q"), 27):
                cv2.destroyAllWindows(); return
            if key == ord(" "):
                t = min(T - 1, t + 1)
            elif key == ord("b"):
                t = max(0, t - 1)
            elif key == ord("n"):
                k = (k + 1) % ids.size; break
            elif key == ord("p"):
                k = (k - 1) % ids.size; break


def main():
    a = parse_args()
    out_dir = Path(D.output_dir)
    sk = np.load(str(out_dir / a.skeleton), allow_pickle=True)
    cow = np.asarray(sk["tracks"], np.float64)                        # (N,T,2)
    gate = (np.asarray(sk["vis"]) > a.cow_vis) & (np.asarray(sk["conf"]) > a.cow_conf) & np.isfinite(cow).all(-1)
    images = sorted(glob.glob(str(Path(D.video_dir) / D.image_pattern)))[: cow.shape[1]]
    frames = [cv2.imread(f) for f in images]
    T = len(frames)
    cow, gate = cow[:, :T], gate[:, :T]
    print(f"[hue-filter] {cow.shape[0]} skeleton tracks, T={T}; gated visible {gate.mean():.3f}")

    vis = gate.copy()
    if not a.no_jump_filter:                                          # filter (a) runs first in the pipeline
        vis, n_rm, n_del, _ = filter_jumpy_tracks(cow, vis, a.cow_jump_px, a.cow_gap_tol_px, a.cow_max_jumps)
        print(f"[hue-filter] filter (a) jumps: {n_del} tracks deleted ({n_rm} observations)")

    before = vis.copy()
    after, n_rm, n_hit = filter_color_vs_median(cow, vis, frames, a.cow_hue_deg,
                                               max_outlier_frac=a.cow_hue_outlier_frac, win=a.cow_hue_win,
                                               strict_deg=a.cow_hue_strict_deg)
    cut = before & ~after                                             # the observations filter (b) removes
    print(f"[hue-filter] filter (b) colour vs first frame (> {a.cow_hue_deg:g} deg): "
          f"{n_hit} tracks truncated, {n_rm} observations removed "
          f"({100 * n_rm / max(before.sum(), 1):.1f}% of the visible ones)")
    fully = (before.any(1) & ~after.any(1)).sum()
    print(f"[hue-filter] of those, {fully} tracks lose EVERY observation; "
          f"{n_hit - fully} keep a prefix and are only truncated")

    if not a.video:
        inspect(a, cow, before, after, cut, frames, T)
        return

    H, W = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(out_dir / a.out), cv2.VideoWriter_fourcc(*"mp4v"), a.fps,
                         (int(2 * W * a.scale), int(H * a.scale)))
    for t in range(T):
        L, R = frames[t].copy(), frames[t].copy()
        for i in np.where(before[:, t])[0]:                           # LEFT: state before filter (b)
            p = (int(round(cow[i, t, 0])), int(round(cow[i, t, 1])))
            kept = after[i, t]
            cv2.circle(L, p, 2, (60, 220, 60) if kept else (0, 0, 255), -1, cv2.LINE_AA)
            if not kept:
                cv2.circle(L, p, 5, (0, 0, 255), 1, cv2.LINE_AA)      # ring the doomed observations
        for i in np.where(after[:, t])[0]:                            # RIGHT: what the pipeline keeps
            p = (int(round(cow[i, t, 0])), int(round(cow[i, t, 1])))
            cv2.circle(R, p, 2, (60, 220, 60), -1, cv2.LINE_AA)
        nb, na = int(before[:, t].sum()), int(after[:, t].sum())
        whole_ids = before.any(1) & ~after.any(1)
        cv2.putText(L, f"before  frame {t}/{T-1}  visible={nb}  cut={nb-na}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(R, f"after filter (b) hue>{a.cow_hue_deg:g}deg  visible={na}  "
                       f"(green = preserved)", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(L, f"red = removed  ({int(whole_ids.sum())} tracks dropped whole)", (8, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
        img = np.hstack([L, R])
        if a.scale != 1.0:
            img = cv2.resize(img, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
        vw.write(img)
    vw.release()
    print(f"[hue-filter] wrote {out_dir / a.out}  ({T} frames, {2*W}x{H})")


if __name__ == "__main__":
    main()
