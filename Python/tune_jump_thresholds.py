#!/usr/bin/env python3
"""Pick --jump-px / --gap-tol-px for the 2D-jump detector (visualize_tracks_2d.py, visualize_cow_jumps.py) from the
scene's OWN motion statistics, instead of guessing.

Prints, for the chosen track source (same names as visualize_tracks_2d.py --source):
  1. the STEP distribution between consecutive VISIBLE frames -> --jump-px must sit ABOVE genuine motion
     (recommended: the p99.5 of that distribution, ~4-5x the median);
  2. the GAP residual distribution = deviation from the position predicted by a 2D similarity fit on the OTHER
     points visible on both sides of the gap (this is what the detector tests, NOT the raw displacement) ->
     --gap-tol-px recommended at its p95; residuals are ~independent of gap length beyond ~3 frames;
  3. a SWEEP of the flagged-track fraction for both thresholds. Read it as an elbow curve: a threshold that
     flags more than ~half the tracks has dropped inside normal motion.
With --hue-check it also reports agreement with the per-track hue-consistency verdict (an independent drift
label). On paper-windmill that agreement is at chance level (precision ~= the 13% base rate), i.e. jumps and
colour drift are DIFFERENT failure modes -- do not tune the jump threshold against the colour label.

Usage: XM_DATASET_NAME=<scene> conda run -n XM python tune_jump_thresholds.py [--source cow] [--hue-check]
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

import utils.definitions as D
from utils import dataset_config
from compare_trackers_masked import _similarity_predict
from visualize_cow_jumps import detect_jumps

FILES = {"raw": "tracks_dynamic_only.npz", "cow": "cow_skeleton.npz",
         "merged": "tracks_dynamic_merged.npz", "smoothed": "tracks_dynamic_merged_smoothed.npz"}


def load(source, out, gv, gc):
    d = np.load(out / FILES[source])
    if source == "cow":
        tr = d["tracks"].astype(float)
        vis = (d["vis"] > gv) & (d["conf"] > gc) & np.isfinite(tr).all(-1)
    else:
        tr = np.asarray(d["tracks_xy"], float)[0]
        vis = (~np.asarray(d["occluded"])[0].astype(bool)) & np.isfinite(tr).all(-1)
    return tr, vis


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=list(FILES), default="cow")
    ap.add_argument("--cow-vis", type=float, default=None); ap.add_argument("--cow-conf", type=float, default=None)
    ap.add_argument("--min-ref", type=int, default=4)
    ap.add_argument("--steps", type=float, nargs="+", default=[12, 16, 20, 25, 30, 40])
    ap.add_argument("--gaps", type=float, nargs="+", default=[30, 40, 50, 60, 80, 120])
    ap.add_argument("--hue-check", action="store_true", help="also score agreement with the per-track hue-consistency verdict")
    a = ap.parse_args()
    out = Path(D.output_dir)
    gv = a.cow_vis if a.cow_vis is not None else dataset_config.get_float("COW_GATE_VIS", 0.5)
    gc = a.cow_conf if a.cow_conf is not None else dataset_config.get_float("COW_GATE_CONF", 0.5)
    tr, vis = load(a.source, out, gv, gc); N, T = vis.shape
    print(f"[tune] {a.source}: {N} tracks, T={T}, {int(vis.sum())} visible point-frames"
          + (f" (gate {gv}/{gc})" if a.source == "cow" else ""))

    s = np.linalg.norm(np.diff(tr, axis=1), axis=-1)[vis[:, 1:] & vis[:, :-1]]
    print("\n1. STEP between consecutive visible frames (px)")
    print("   " + "  ".join(f"p{q}={np.percentile(s, q):.1f}" for q in (50, 90, 99, 99.5, 99.9)) + f"  max={s.max():.1f}")
    rec_step = float(np.percentile(s, 99.5))
    print(f"   -> recommended --jump-px {rec_step:.0f}  (p99.5; {rec_step/np.median(s):.1f}x the median step)")

    res, gl = [], []
    for i in range(N):
        f = np.where(vis[i])[0]
        for x, y in zip(f[:-1], f[1:]):
            if y - x <= 1:
                continue
            ref = np.where(vis[:, x] & vis[:, y])[0]; ref = ref[ref != i]
            if ref.size < a.min_ref:
                continue
            res.append(float(np.linalg.norm(tr[i, y] - _similarity_predict(tr[ref, x], tr[ref, y], tr[i, x])))); gl.append(int(y - x))
    if res:
        r = np.array(res); g = np.array(gl)
        print(f"\n2. GAP residual vs the similarity prediction (px; {r.size} reappearances, gap length median {np.median(g):.0f})")
        print("   " + "  ".join(f"p{q}={np.percentile(r, q):.1f}" for q in (50, 90, 95, 99)))
        print("   by gap length: " + "  ".join(f"{lo}-{hi}f median {np.median(r[(g >= lo) & (g < hi)]):.1f}"
              for lo, hi in [(2, 3), (3, 6), (6, 15), (15, 10**6)] if ((g >= lo) & (g < hi)).any()))
        rec_gap = float(np.percentile(r, 95))
        print(f"   -> recommended --gap-tol-px {rec_gap:.0f}  (p95; half of all reappearances already deviate "
              f"{np.median(r):.0f} px from a rigid prediction, so a tolerance near that value flags normal motion)")
    else:
        print("\n2. GAP residual: no reappearances with enough reference points (nothing to tune)")

    lab = None
    if a.hue_check:
        from visualize_tracks_2d import _sample_track_hue, _score_hue
        ip = sorted(glob.glob(str(Path(D.video_dir) / D.image_pattern)))
        hue, sat = _sample_track_hue(tr, ~vis, ip, T); good, scored = _score_hue(hue, sat, 30.0)
        lab = scored & ~good
        print(f"\n   hue label: {int((scored & good).sum())} consistent, {int(lab.sum())} inconsistent "
              f"(base rate {100 * lab.mean():.1f}%)")

    def line(flag):
        t = f"flagged {100 * flag.mean():5.1f}% of tracks"
        if lab is not None:
            tp = int((flag & lab).sum()); fp = int((flag & ~lab).sum())
            t += f" | vs hue label: precision {tp / max(tp + fp, 1):.2f} recall {tp / max(int(lab.sum()), 1):.2f}"
        return t
    print("\n3. SWEEP  (elbow rule: a threshold that flags > ~50% of tracks is inside normal motion)")
    print("   STEP only:")
    for j in a.steps:
        t, _ = detect_jumps(tr, vis, j, 1e9, a.min_ref); print(f"     --jump-px {j:6.0f}: {line(t.any(1))}")
    print("   GAP only:")
    for gg in a.gaps:
        t, _ = detect_jumps(tr, vis, 1e9, gg, a.min_ref); print(f"     --gap-tol-px {gg:6.0f}: {line(t.any(1))}")
    t, _ = detect_jumps(tr, vis, rec_step, (rec_gap if res else 1e9), a.min_ref)
    print(f"   recommended pair ({rec_step:.0f} / {rec_gap if res else float('inf'):.0f}): {line(t.any(1))}")


if __name__ == "__main__":
    main()
