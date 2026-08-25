#!/usr/bin/env python3
"""Debug the large-2D-move deletion (dynamic_tracking.py `_compute_move_delete_query_mask`).

No re-tracking: it reads the existing tracks_dynamic_only.npz, recomputes the SAME per-frame move
deviation (each point's frame-to-frame 2D displacement vs the median of its spatial neighbours, in a
window-sized square), and lets you inspect ONE move-deleted track frame by frame:

  * GREEN  point  = "preserved part"  : this frame's deviation <= eps (normal 2D motion)
  * RED    point  = "deleted part"    : this frame's deviation >  eps (large 2D move -> triggers delete)
  * GRAY   point  = not scored this frame (occluded / off-mask / no neighbours)

KEY POINT this exposes: the real filter deletes the WHOLE track if its MAX per-frame deviation exceeds
eps at *any* frame -- it does NOT keep the largest good segment. So a track that is green on almost
every frame still gets fully removed because of a few red frames.

Controls (OpenCV window): SPACE = next frame, B = prev, N = next deleted track, Q/ESC = quit.
Run:  XM_DATASET_NAME=<scene> conda run -n XM python debug_move_delete.py [--track ID] [--headless]
"""
import argparse
import glob
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

import utils.definitions
import utils.dataset_config as dataset_config

WINDOW = 10.0  # MOVE_WINDOW_SIZE in dynamic_tracking.py (native px)


def _reconstruct_pre_move_occluded(z, T):
    """The npz `occluded` is POST-move-delete (deleted tracks are all-True), but the filter used the
    PRE-move occlusion. Rebuild it: non-deleted tracks are unchanged; deleted tracks are rebuilt from
    the stored component masks (off-mask | depth-grad | depth-max | pre-query). Only the tracker
    certainty-occlusion of deleted tracks is unrecoverable -- negligible when certainty threshold is low."""
    occ = z["occluded"][0].astype(bool).copy()                       # (Q,T)
    dyn = z["dynamic_obs_mask"].astype(bool)                          # (Q,T)
    qt = z["query_t"].astype(np.int64)                               # (Q,)
    pre_query = np.arange(T, dtype=np.int64)[None, :] < qt[:, None]   # (Q,T)
    dg = z["depth_grad_deleted_obs_mask"].astype(bool) if "depth_grad_deleted_obs_mask" in z.files else np.zeros_like(occ)
    dm = z["depth_max_deleted_obs_mask"].astype(bool) if "depth_max_deleted_obs_mask" in z.files else np.zeros_like(occ)
    mv = z["move_delete_query_mask"].astype(bool) if "move_delete_query_mask" in z.files else np.zeros(occ.shape[0], bool)
    recon = (~dyn) | dg | dm | pre_query
    occ[mv] = recon[mv]
    return occ, pre_query, mv


def _recompute_dev(tracks_xy, occ_pre, window):
    """Per-frame move deviation dev_full (Q,T), NaN where not scored. Faithful to the tracker:
    dev = |disp_i / median(neighbour disp) - 1| over neighbours within a `window`-side square."""
    Q, T = tracks_xy.shape[1], tracks_xy.shape[2]
    dev_full = np.full((Q, T), np.nan, dtype=np.float64)
    half = float(window) / 2.0
    for t in tqdm(range(1, T), desc="[move] recompute dev"):
        pts = tracks_xy[0, :, t, :]
        prev = tracks_xy[0, :, t - 1, :]
        valid = (np.isfinite(pts).all(1) & np.isfinite(prev).all(1)
                 & (~occ_pre[:, t]) & (~occ_pre[:, t - 1]))
        idx = np.where(valid)[0]
        if idx.size < 2:
            continue
        P = pts[idx].astype(np.float64)
        disp = np.linalg.norm(P - prev[idx].astype(np.float64), axis=1)
        nb = cKDTree(P).query_ball_point(P, r=half, p=np.inf)   # square window (Chebyshev), == tracker
        for i in range(idx.size):
            cand = [j for j in nb[i] if j != i]
            if not cand:
                continue
            med = float(np.median(disp[cand]))
            dev_full[idx[i], t] = abs(disp[i] / (med + 1e-6) - 1.0)
    return dev_full


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", type=int, default=-1, help="track id to inspect (default: auto-pick)")
    ap.add_argument("--eps", type=float, default=-1.0,
                    help="deviation threshold (default: DYN_MOVE_DELETE_TOP_PERCENT/100 from config)")
    ap.add_argument("--window", type=float, default=WINDOW)
    ap.add_argument("--headless", action="store_true", help="no window; just print the analysis")
    args = ap.parse_args()

    z = np.load(utils.definitions.tracks_dynamics_path, allow_pickle=True)
    tracks_xy = z["tracks_xy"]              # (1,Q,T,2) native px
    Q, T = tracks_xy.shape[1], tracks_xy.shape[2]
    top_percent = dataset_config.get_float("DYN_MOVE_DELETE_TOP_PERCENT", 100.0, env="DYN_MOVE_DELETE_TOP_PERCENT")
    eps = args.eps if args.eps >= 0 else float(np.clip(top_percent, 0.0, 100.0)) / 100.0

    occ_pre, pre_query, mv_stored = _reconstruct_pre_move_occluded(z, T)
    dev_full = _recompute_dev(tracks_xy, occ_pre, args.window)

    scored = np.isfinite(dev_full)
    maxdev = np.full(Q, np.nan)
    any_scored = scored.any(1)
    maxdev[any_scored] = np.nanmax(dev_full[any_scored], axis=1)
    recomputed_delete = any_scored & (maxdev > eps)

    # faithfulness check vs the stored decision
    agree = int((recomputed_delete == mv_stored).sum())
    print(f"[move] eps={eps:.3f} (DYN_MOVE_DELETE_TOP_PERCENT={top_percent}), window={args.window}px")
    print(f"[move] stored deleted={int(mv_stored.sum())}/{Q}  recomputed deleted={int(recomputed_delete.sum())}  "
          f"agreement={100*agree/Q:.1f}% (mismatch mostly = unrecoverable certainty-occ of deleted tracks)")

    # per deleted track: how many frames are 'preserved' (green) vs 'deleted-part' (red)
    preserved = (scored & (dev_full <= eps))
    deleted_part = (scored & (dev_full > eps))
    n_pres = preserved.sum(1)
    n_del = deleted_part.sum(1)
    del_ids = np.where(mv_stored)[0]
    print(f"[move] of {del_ids.size} deleted tracks: avg preserved frames={n_pres[del_ids].mean():.1f}, "
          f"avg deleted-part frames={n_del[del_ids].mean():.1f}  "
          f"-> {100*n_pres[del_ids].sum()/max(1,(n_pres[del_ids]+n_del[del_ids]).sum()):.0f}% of their scored "
          f"frames are actually NORMAL motion but the whole track is removed.")

    if args.track >= 0:
        tid = args.track
    else:
        # auto-pick the deleted track with the MOST preserved frames (best illustrates the waste)
        cand = del_ids[(n_del[del_ids] >= 1)]
        cand = cand if cand.size else del_ids
        tid = int(cand[np.argmax(n_pres[cand])]) if cand.size else -1
    if tid < 0:
        print("No move-deleted track found.")
        return
    print(f"\n[track {tid}] move-deleted={bool(mv_stored[tid])}  maxdev={maxdev[tid]:.2f} (eps={eps:.2f})  "
          f"preserved(green)={int(n_pres[tid])} frames  deleted-part(red)={int(n_del[tid])} frames")
    red_frames = np.where(deleted_part[tid])[0]
    print(f"[track {tid}] red frames (dev>eps): {red_frames.tolist()[:40]}{' ...' if red_frames.size>40 else ''}")

    if args.headless:
        print("\nPer-frame (t: dev  status):")
        for t in range(T):
            d = dev_full[tid, t]
            st = "gray(not-scored)" if not np.isfinite(d) else ("RED  del-part" if d > eps else "green preserved")
            xy = tracks_xy[0, tid, t]
            pos = f"({xy[0]:.0f},{xy[1]:.0f})" if np.isfinite(xy).all() else "(nan)"
            if np.isfinite(d) or np.isfinite(xy).all():
                print(f"  t={t:4d} {pos:>12}  dev={d if np.isfinite(d) else float('nan'):7.2f}  {st}")
        return

    # ---- OpenCV frame stepper ----
    import cv2
    img_paths = sorted(glob.glob(str(Path(utils.definitions.video_dir) / utils.definitions.image_pattern)))
    trail = []  # (x,y,color)
    st = {"t": int(red_frames[0]) if red_frames.size else 0, "tid": tid, "cand": del_ids, "ci": 0}

    def draw(t):
        img = cv2.imread(img_paths[t]) if t < len(img_paths) else np.zeros((480, 640, 3), np.uint8)
        img = img.copy()
        d = dev_full[st["tid"], t]
        xy = tracks_xy[0, st["tid"], t]
        color = (150, 150, 150)
        status = "not scored (occluded/off-mask/no-neighbour)"
        if np.isfinite(d):
            if d > eps:
                color = (0, 0, 255); status = f"DELETED-part  dev={d:.2f} > eps={eps:.2f}"
            else:
                color = (0, 220, 0); status = f"preserved     dev={d:.2f} <= eps={eps:.2f}"
        if np.isfinite(xy).all():
            p = (int(round(xy[0])), int(round(xy[1])))
            trail.append((p, color))
            for pp, cc in trail[-60:]:
                cv2.circle(img, pp, 2, cc, -1, cv2.LINE_AA)
            cv2.circle(img, p, 7, color, 2, cv2.LINE_AA)
            cv2.drawMarker(img, p, color, cv2.MARKER_CROSS, 16, 1, cv2.LINE_AA)
        lines = [
            f"track {st['tid']}  frame {t}/{T-1}   (SPACE/B step, N next deleted track, Q quit)",
            status,
            f"track maxdev={maxdev[st['tid']]:.2f}  -> WHOLE track deleted={bool(mv_stored[st['tid']])} "
            f"(green frames={int(n_pres[st['tid']])}, red frames={int(n_del[st['tid']])})",
            "current filter removes the ENTIRE track (all frames) -- it does NOT keep the green part.",
        ]
        y = 22
        for i, ln in enumerate(lines):
            col = (0, 255, 255) if i == 0 else (255, 255, 255)
            cv2.putText(img, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
            y += 20
        cv2.imshow("move-2D-delete debug", img)

    print("\nOpenCV window: SPACE=next frame, B=prev, N=next deleted track, Q/ESC=quit")
    draw(st["t"])
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == 32:      # SPACE
            st["t"] = (st["t"] + 1) % T; draw(st["t"])
        elif k in (ord("b"), ord("B")):
            st["t"] = (st["t"] - 1) % T; draw(st["t"])
        elif k in (ord("n"), ord("N")):
            st["ci"] = (st["ci"] + 1) % st["cand"].size
            st["tid"] = int(st["cand"][st["ci"]]); trail.clear()
            rf = np.where(deleted_part[st["tid"]])[0]
            st["t"] = int(rf[0]) if rf.size else 0
            print(f"[track {st['tid']}] maxdev={maxdev[st['tid']]:.2f} preserved={int(n_pres[st['tid']])} "
                  f"deleted-part={int(n_del[st['tid']])}")
            draw(st["t"])
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
