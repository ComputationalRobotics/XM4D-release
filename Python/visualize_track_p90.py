#!/usr/bin/env python3
"""EDG-topology debugger: for ONE reference track, colour every other track by its PERSISTENT distance --
the exact quantity build_edg's edge test uses (quantile of the per-frame distance over co-visible frames).

Pops a window with TWO panels on the video frame: left = 3D persistent distance (world units, the edge
criterion `s3 <= thr`), right = 2D persistent distance (px, the rescue criterion `s2 <= thr2d`). Colormap is
anchored at the AUTO thresholds build_edg would use: blue/green = would pass, orange/red = would not; tracks
with too few co-visible frames are grey; the reference track is the red star. Dots are drawn at each track's
2D position in the shown frame (only tracks visible there).

Interactive: LEFT-CLICK a dot = make that track the reference | N/P = next/prev track id | SPACE/B =
next/prev frame | Q = quit. The view is ZOOMED to the dynamic region (bbox of the visible tracks + margin;
--no-zoom for the full frame).
Same spaces/thresholds as build_edg (same tracks file, same per-frame scale, same quantile).

Run:  conda run -n XM python visualize_track_p90.py --track 123
      conda run -n XM python visualize_track_p90.py --track 123 --frame 100 --save out.png   (headless)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

import utils.definitions as D_
from utils import dataset_config
from build_edg import (_auto_edge_threshold, _auto_edge_threshold_2d, _frame_scale, _load_all, _nanq_cols,
                       _stack_sampled)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", type=int, default=-1, help="reference track id (-1 = most-observed track)")
    ap.add_argument("--frame", type=int, default=-1, help="display frame (-1 = reference's first visible)")
    ap.add_argument("--tracks", default="", help="tracks npz (default: lift if present, else smoothed)")
    ap.add_argument("--quantile", type=float,
                    default=dataset_config.get_float("EDG_EDGE_DIST_QUANTILE", 0.95),
                    help="persistent-distance quantile (build_edg --distance-quantile; 0.9 = p90)")
    ap.add_argument("--min-covis", type=int, default=dataset_config.get_int("EDG_MIN_OVERLAP_FRAMES", 3),
                    help="pairs with fewer co-visible frames are UNDEFINED (grey)")
    ap.add_argument("--no-scale", action="store_true", help="match build_edg --no-scale")
    ap.add_argument("--vmax-mult", type=float, default=3.0, help="colour range = [0, vmax_mult * threshold]")
    ap.add_argument("--dot", type=float, default=14.0)
    ap.add_argument("--no-zoom", action="store_true", help="show the full frame instead of zooming to "
                                                           "the dynamic region")
    ap.add_argument("--zoom-pad", type=float, default=0.18, help="zoom margin as a fraction of the bbox size")
    ap.add_argument("--save", default=None, help="write ONE PNG headlessly instead of opening a window")
    a = ap.parse_args()
    if a.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    out_dir = Path(D_.output_dir)
    tp = a.tracks if a.tracks else next(
        (str(out_dir / n) for n in ("tracks_dynamic_merged_lift.npz", "tracks_dynamic_merged_smoothed.npz")
         if (out_dir / n).exists()), None)
    if tp is None:
        raise SystemExit("no lifted tracks file found; pass --tracks")
    if "/" not in tp:
        tp = str(out_dir / tp)
    D = _load_all(tracks_path=tp)
    T = D["T"]
    s = _frame_scale(D["rt"], T, a.no_scale)
    sampled = np.arange(T, dtype=int)
    Ps, Vs = _stack_sampled(D, s, sampled)                             # (T,Q,3) world / (T,Q) reliable
    xy_all = D["tracks_xy"][0]                                         # (Q,T,2) px
    Ps2d = np.where(Vs[..., None], np.transpose(xy_all, (1, 0, 2)).astype(np.float64), np.nan)
    thr = dataset_config.get_float("EDG_EDGE_THRESH", 0.0) or \
        _auto_edge_threshold(Ps, Vs, dataset_config.get_float("EDG_EDGE_THRESH_MULT", 5.0))
    thr2d = _auto_edge_threshold_2d(Ps2d, Vs)                          # same auto as build_edg (no config key)
    Q = Ps.shape[1]
    import glob as _g
    images = sorted(_g.glob(str(Path(D_.video_dir) / D_.image_pattern)))
    print(f"[p90] {Path(tp).name}: Q={Q} T={T} | quantile={a.quantile:g} "
          f"| 3D thr={thr:.4f} (world) 2D thr={thr2d:.1f}px | min-covis={a.min_covis}")

    st = {"r": a.track if a.track >= 0 else int(np.argmax(Vs.sum(0))), "t": a.frame}
    alive = np.where(Vs.sum(0) > 0)[0]                                 # N/P cycle through these

    def pdist(r):
        """Persistent 3D/2D distance of every track to r + co-visible counts (exactly build_edg's s3/s2)."""
        co = Vs & Vs[:, r:r + 1]                                        # (T,Q)
        n_ov = co.sum(0)
        d3 = np.where(co, np.linalg.norm(Ps - Ps[:, r:r + 1, :], axis=2), np.nan)
        d2 = np.where(co, np.linalg.norm(Ps2d - Ps2d[:, r:r + 1, :], axis=2), np.nan)
        s3 = _nanq_cols(d3, a.quantile)
        s2 = _nanq_cols(d2, a.quantile)
        bad = n_ov < a.min_covis
        s3[bad] = np.nan; s2[bad] = np.nan
        return s3, s2, n_ov

    fig, axs = plt.subplots(1, 2, figsize=(13, 6.5))
    cbs = []

    def draw():
        r = st["r"]
        if st["t"] < 0:
            st["t"] = int(np.argmax(Vs[:, r]))
        t = st["t"] = int(np.clip(st["t"], 0, min(T, len(images)) - 1))
        s3, s2, n_ov = pdist(r)
        img = plt.imread(images[t])
        vis = Vs[t] & np.isfinite(xy_all[:, t]).all(-1)
        vis[r] = False
        zoom = None
        if not a.no_zoom and vis.any():                                # dynamic-region bbox + margin
            px, py = xy_all[vis, t, 0], xy_all[vis, t, 1]
            x0, x1, y0, y1 = px.min(), px.max(), py.min(), py.max()
            mx = a.zoom_pad * max(x1 - x0, 20); my = a.zoom_pad * max(y1 - y0, 20)
            H, W = img.shape[:2]
            zoom = (max(0, x0 - mx), min(W, x1 + mx), min(H, y1 + my), max(0, y0 - my))
        for ax, val, th, name, unit in ((axs[0], s3, thr, "3D persistent distance", "world"),
                                        (axs[1], s2, thr2d, "2D persistent distance", "px")):
            ax.clear()
            ax.imshow(img)
            und = vis & ~np.isfinite(val)
            ax.scatter(xy_all[und, t, 0], xy_all[und, t, 1], s=a.dot, c="0.55", alpha=0.6,
                       label=f"covis < {a.min_covis}")
            ok = vis & np.isfinite(val)
            sc = ax.scatter(xy_all[ok, t, 0], xy_all[ok, t, 1], s=a.dot, c=val[ok], cmap="turbo",
                            norm=Normalize(0.0, a.vmax_mult * th), edgecolors="none")
            if Vs[t, r] and np.isfinite(xy_all[r, t]).all():
                ax.scatter([xy_all[r, t, 0]], [xy_all[r, t, 1]], s=180, marker="*", c="red",
                           edgecolors="white", linewidths=1.0, zorder=5)
            npass = int((val[ok] <= th).sum())
            ax.set_title(f"{name} to track {r}  (q={a.quantile:g})\n"
                         f"thr={th:.4g} {unit}: {npass}/{int(ok.sum())} pass | frame {t}"
                         + ("" if Vs[t, r] else "  [ref NOT visible here]"), fontsize=10)
            if zoom is not None:
                ax.set_xlim(zoom[0], zoom[1]); ax.set_ylim(zoom[2], zoom[3])
            ax.set_axis_off()
            if len(cbs) < 2:
                cbs.append(fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.01))
        fig.suptitle("LEFT-CLICK = new reference | N/P = next/prev track | SPACE/B = frame +/- | Q = quit",
                     fontsize=9)
        fig.canvas.draw_idle()

    def on_click(ev):
        if ev.inaxes not in list(axs) or ev.button != 1 or ev.xdata is None:
            return
        t = st["t"]
        vis = np.where(Vs[t] & np.isfinite(xy_all[:, t]).all(-1))[0]
        if vis.size == 0:
            return
        d = np.hypot(xy_all[vis, t, 0] - ev.xdata, xy_all[vis, t, 1] - ev.ydata)
        st["r"] = int(vis[np.argmin(d)])
        print(f"[p90] reference -> track {st['r']}")
        draw()

    def on_key(ev):
        if ev.key == " ":
            st["t"] += 1; draw()
        elif ev.key == "b":
            st["t"] -= 1; draw()
        elif ev.key in ("n", "p"):
            k = int(np.searchsorted(alive, st["r"]))
            k = (k + 1) % alive.size if ev.key == "n" else (k - 1) % alive.size
            st["r"] = int(alive[k])
            print(f"[p90] reference -> track {st['r']}")
            draw()
        elif ev.key == "q":
            plt.close(fig)

    draw()
    if a.save:
        fig.savefig(a.save, dpi=130, bbox_inches="tight")
        print(f"[p90] wrote {a.save}")
        return
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


if __name__ == "__main__":
    main()
