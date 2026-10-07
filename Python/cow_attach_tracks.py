#!/usr/bin/env python3
"""Pipeline step B1 (replacement for test_extend_tracks.py): long dynamic tracks from a CoWTracker skeleton + TAPNet.

  1. SKELETON (GPU, env --cow-env): CoWTracker (dense, reference-frame anchored) restarted every --every frames on the
     masked object, tracking forward AND backward from each reference frame (cow_skeleton.py) -> outputs/cow_skeleton.npz.
     Gated at vis > --cow-vis & conf > --cow-conf, then FILTERS in this order:
       (a) JUMPY TRACKS -- FIRST, and all-or-nothing: count each track's large 2D jumps and DELETE the whole track when it
           has MORE THAN --cow-max-jumps of them; every other track is left COMPLETELY UNTOUCHED (no truncation, no
           per-observation edits). A jump is a step between two consecutive visible frames above --cow-jump-px, or a
           reappearance after an invisible gap deviating more than --cow-gap-tol-px from the position predicted by a 2D
           similarity fit on the OTHER points visible on both sides (so camera motion / rigid rotation are not jumps).
           Thresholds from tune_jump_thresholds.py; the jump-count histogram is printed so --cow-max-jumps can be chosen.
       (b) COLOUR vs the track's MEDIAN (--cow-hue-deg): the reference is the circular MEDIAN hue over all the track's
           visible saturated observations; an observation deviating more than --cow-hue-deg from it is an outlier. Fewer
           than --cow-hue-outlier-frac outliers -> only those observations are dropped; at or above it the whole track is
           dropped. Saturation-gated: colourless samples cannot testify and are kept.
       (c) COVERAGE (--cow-track-frac / --cow-frame-frac, per-OBSERVATION, not terminal): keep an observation only if
           EITHER its track is visible in >= --cow-track-frac of the whole video (a long-lived, trustworthy track)
           OR its frame has more than --cow-frame-frac of the started points visible (a frame the tracker handles
           well -- the per-frame count swings 3-6x on a rotating object, and the sparse frames are the unreliable
           ones; see visualize_point_count.py).
  2. ATTACH (CPU): every forward TAPNet track (dynamic_tracking.py, --forward-npz) is attached to EVERY skeleton point it
     stays close to: over their co-visible frames the --pct percentile of the 2D distance must be <= --attach-px (and
     >= --min-covis co-visible frames); the observations above that percentile are deleted for that pair. Candidates are
     added LONGEST FIRST. Optional peer check (--attach-peer-px >= 0, OFF by default: the 2026-09-09 ablation showed it does
     not improve the merged tracks): a new track must also stay close to the running average of the already-attached tracks.
  3. AVERAGE: per skeleton point, the attached tracks are offset-corrected (mean offset to the skeleton point over
     co-visible frames) and averaged per frame -> one long track per skeleton point, visible wherever >= 1 attached
     track is visible (optionally --fill-skeleton: gated CoW position where none is). TAPNet tracks that matched
     NO skeleton point are appended unchanged (--keep-tapnet unattached; skeleton_index = -1) so the parts
     of the object the skeleton missed survive into build_edg / EDG_test_fast (--keep-tapnet unattached,
     the DEFAULT); --keep-tapnet all appends EVERY raw TAPNet track for the densest dynamic cloud. Written as
     outputs/tracks_dynamic_merged.npz in the tracks_dynamic_only.npz schema (2D only; 3D is lifted downstream).
  4. optimize_dynamic_tracks.py --input tracks_dynamic_merged.npz (same env) -> tracks_dynamic_merged_smoothed.npz,
     which build_edg.py reads by default.

Scene via XM_DATASET_CAT / XM_DATASET_NAME (utils.definitions).
Run: XM_DATASET_CAT=Dycheck XM_DATASET_NAME=paper-windmill conda run -n XM python cow_attach_tracks.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

import utils.definitions as D
from utils import dataset_config

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _truncate_from(vis, bad):
    """Both skeleton filters are terminal: from the first offending frame on, the track is dropped."""
    N, T = vis.shape
    first = np.where(bad.any(1), bad.argmax(1), T)
    return vis & (np.arange(T)[None] < first[:, None]), int((vis & (np.arange(T)[None] >= first[:, None])).sum()), int(bad.any(1).sum())


def filter_jumpy_tracks(tracks, vis, jump_px, gap_tol_px, max_jumps, min_ref=4):
    """FILTER (a), first in the chain and all-or-nothing: count each track's large 2D jumps (consecutive-frame step >
    jump_px, or a gap reappearance deviating > gap_tol_px from the similarity-predicted position) and DELETE every track
    with more than max_jumps of them. Tracks that survive are returned UNCHANGED -- no truncation, no per-observation
    edits. Returns (vis, n_removed, n_tracks_deleted, jump_counts)."""
    from visualize_cow_jumps import detect_jumps
    trig, _ = detect_jumps(tracks, vis, jump_px, gap_tol_px, min_ref)
    k = trig.sum(1); drop = k > int(max_jumps)
    new = vis.copy(); new[drop] = False
    return new, int(vis[drop].sum()), int(drop.sum()), k


def filter_coverage(vis, track_frac, frame_frac, started=None):
    """FILTER (c): keep observation (i,t) iff track i is visible in >= track_frac of the WHOLE video, or frame t has
    more than frame_frac of the started points visible. Per-observation (a track can lose its sparse frames and keep
    its dense ones). `started` = number of tracks whose reference frame is <= t (denominator for the frame test; the
    multi-pass skeleton adds runs over time, so using N would make early frames unreachable). Returns
    (vis, n_removed, n_long_tracks, n_dense_frames)."""
    N, T = vis.shape
    long_track = vis.mean(1) >= float(track_frac)                        # visible in >= track_frac of the video
    denom = np.maximum(np.asarray(started, float) if started is not None else np.full(T, N, float), 1.0)
    dense_frame = vis.sum(0) > float(frame_frac) * denom                 # > frame_frac of the started points visible
    keep = long_track[:, None] | dense_frame[None, :]
    return vis & keep, int((vis & ~keep).sum()), int(long_track.sum()), int(dense_frame.sum())


def sample_hue_sat(tracks, vis, frames_bgr, patch=1):
    """Patch-median hue/saturation at every visible track position (NaN elsewhere)."""
    N, T = vis.shape; hue = np.full((N, T), np.nan); sat = np.full((N, T), np.nan)
    for t in range(T):
        idx = np.where(vis[:, t])[0]
        if idx.size == 0:
            continue
        img = frames_bgr[t]; H, W = img.shape[:2]; hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        xi = np.clip(np.rint(tracks[idx, t, 0]), 0, W - 1).astype(int); yi = np.clip(np.rint(tracks[idx, t, 1]), 0, H - 1).astype(int)
        for r, (x, y) in enumerate(zip(xi, yi)):
            win = hsv[max(y - patch, 0):y + patch + 1, max(x - patch, 0):x + patch + 1].reshape(-1, 3)
            hue[idx[r], t] = np.median(win[:, 0]); sat[idx[r], t] = np.median(win[:, 1])
    return hue, sat


def sample_hue_best_in_window(tracks, vis, frames_bgr, ref, win, sat_min=70):
    """Second-pass sampling that tolerates a few px of LOCAL DRIFT: inside a (2*win+1)^2 window around each
    visible track position, return the hue of the pixel CLOSEST to that track's reference hue (and its
    saturation). A track that slid a little off its surface but whose surface is still inside the window is
    then judged on the surface, not on whatever it landed on. Falls back to the centre pixel when a track
    has no reference (colourless) or the window is empty."""
    N, T = vis.shape
    hue = np.full((N, T), np.nan); sat = np.full((N, T), np.nan)
    dy, dx = np.mgrid[-win:win + 1, -win:win + 1]
    dy, dx = dy.ravel(), dx.ravel()
    for t in range(T):
        idx = np.where(vis[:, t])[0]
        if idx.size == 0 or frames_bgr[t] is None:
            continue
        hsv = cv2.cvtColor(frames_bgr[t], cv2.COLOR_BGR2HSV)
        H, W = hsv.shape[:2]
        x = np.clip(np.rint(tracks[idx, t, 0]).astype(int), 0, W - 1)
        y = np.clip(np.rint(tracks[idx, t, 1]).astype(int), 0, H - 1)
        ys = np.clip(y[:, None] + dy[None, :], 0, H - 1)               # (n, win^2)
        xs = np.clip(x[:, None] + dx[None, :], 0, W - 1)
        hw = hsv[ys, xs, 0].astype(np.float64)                         # hue in the window
        sw = hsv[ys, xs, 1].astype(np.float64)
        r = ref[idx]
        ok = np.isfinite(r)
        d = np.abs(hw - r[:, None]); d = np.minimum(d, 180.0 - d)      # circular distance to the reference
        d[sw <= sat_min] = np.inf                                      # unsaturated pixels cannot testify
        best = np.argmin(np.where(np.isfinite(d), d, np.inf), axis=1)
        rows = np.arange(idx.size)
        pick_h, pick_s = hw[rows, best], sw[rows, best]
        centre = (win * (2 * win + 1) + win)                           # fall back to the centre pixel
        bad = ~ok | ~np.isfinite(d[rows, best])
        pick_h[bad] = hw[rows[bad], centre]; pick_s[bad] = sw[rows[bad], centre]
        hue[idx, t] = pick_h; sat[idx, t] = pick_s
    return hue, sat


def filter_color_vs_median(tracks, vis, frames_bgr, max_deg, sat_min=70, max_outlier_frac=0.3, win=0,
                           strict_deg=15.0):
    """FILTER (b): reference hue = circular MEDIAN over ALL of the track's visible saturated observations, so a
    track that drifts late is still judged against the colour it spent most of its life on (the old reference --
    the first few frames -- made an unreliable early sample condemn the whole track, and truncating from the
    first offence deleted the good tail as well). An observation deviating more than max_deg is an OUTLIER:
      * outliers < max_outlier_frac of the judged observations -> drop ONLY those observations;
      * otherwise the track is unreliable, but is NOT deleted: only the observations within strict_deg of
        the reference are kept.
    Tracks with no saturated sample are left untouched (hue cannot testify).
    Returns (vis, n_removed, n_tracks_hit)."""
    from edg_viz_common import _circ_hue_median
    hue, sat = sample_hue_sat(tracks, vis, frames_bgr)           # centre patch: decides WHICH obs are judged
    N, T = vis.shape
    judged = np.isfinite(hue) & (sat > sat_min)
    hue_win = None
    if win > 0:
        # Two-pass. The window may only REDUCE an observation's deviation (local drift: the track's own
        # surface is still nearby), never make an unjudged observation judged -- otherwise forcing a
        # saturated pixel into the window would bring new observations under test and delete MORE, which is
        # the opposite of tolerating drift.
        ref0 = np.array([_circ_hue_median(hue[i, judged[i]]) if judged[i].any() else np.nan
                         for i in range(N)])
        hue_win, _ = sample_hue_best_in_window(tracks, vis, frames_bgr, ref0, int(win), sat_min)
    out = vis.copy(); n_rm = 0; n_hit = 0; n_whole = 0
    for i in range(N):
        f = np.where(judged[i])[0]
        if f.size == 0:
            continue
        ref = _circ_hue_median(hue[i, f])
        d = np.abs(hue[i] - ref); d = np.minimum(d, 180.0 - d)   # circular distance, deg (0..90)
        if hue_win is not None:                                  # best match within the window, if closer
            dw = np.abs(hue_win[i] - ref); dw = np.minimum(dw, 180.0 - dw)
            d = np.fmin(d, np.where(np.isfinite(dw), dw, np.inf))
        bad = judged[i] & (d > max_deg)
        if not bad.any():
            continue
        n_hit += 1
        if bad.sum() >= max_outlier_frac * f.size:
            # Mostly off its own reference colour. Rather than deleting the track outright, keep only the
            # observations that agree CLOSELY with the reference (strict_deg, well inside max_deg): the
            # track is unreliable, but its best-matching frames are still evidence.
            keep_strict = judged[i] & (d < strict_deg)
            drop = vis[i] & ~keep_strict
            n_rm += int(drop.sum()); out[i] &= keep_strict; n_whole += 1
        else:                                                    # a few bad observations -> drop just them
            n_rm += int((vis[i] & bad).sum()); out[i] &= ~bad
    print(f"[filter-b] reference = circular median of all saturated observations; "
          f"{n_whole} tracks had >= {max_outlier_frac:.0%} outliers -> kept only frames within "
          f"{strict_deg:g} deg, {n_hit - n_whole} tracks had only outlier observations removed; "
          f"{int((vis.any(1) & ~out.any(1)).sum())} tracks ended up empty")
    return out, n_rm, n_hit


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forward-npz", default="tracks_dynamic_only.npz", help="forward dynamic_tracking.py npz (in outputs/)")
    ap.add_argument("--output", default="tracks_dynamic_merged.npz", help="output npz (in outputs/); build_edg reads <stem>_smoothed.npz")
    ap.add_argument("--offmask-max-frac", type=float,
                    default=dataset_config.get_float("DYN_OFFMASK_MAX_FRAC", 0.2),
                    help="drop a track whose visible observations lie outside the dynamic mask more than "
                         "this fraction of the time; its off-mask observations are dropped either way. "
                         "Negative disables the filter entirely.")
    ap.add_argument("--skeleton", default="cow_skeleton.npz", help="skeleton npz (in outputs/)")
    ap.add_argument("--reuse-skeleton", action="store_true", help="skip the GPU stage if the skeleton npz exists")
    ap.add_argument("--cow-env", default="SpaTrack2", help="conda env with CoWTracker deps")
    ap.add_argument("--every", type=int, default=50, help="CoWTracker restart period (frames)")
    ap.add_argument("--stride", type=int, default=8, help="skeleton query grid stride (px)")
    ap.add_argument("--cow-window", type=int, default=100, help="CoWTracker window length (100 needs ~37 GB GPU)")
    ap.add_argument("--cow-vis", type=float, default=dataset_config.get_float("COW_GATE_VIS", 0.5), help="skeleton visibility gate (config COW_GATE_VIS)")
    ap.add_argument("--cow-conf", type=float, default=dataset_config.get_float("COW_GATE_CONF", 0.5), help="skeleton confidence gate (config COW_GATE_CONF)")
    ap.add_argument("--cow-jump-px", type=float, default=dataset_config.get_float("COW_JUMP_PX", 20.0),
                    help="filter (a): consecutive-frame step that counts as a jump, px (config COW_JUMP_PX; 0 = filter off). "
                         "Pick it with tune_jump_thresholds.py (~p99.5 of the step distribution)")
    ap.add_argument("--cow-gap-tol-px", type=float, default=dataset_config.get_float("COW_GAP_TOL_PX", 60.0),
                    help="filter (a): deviation from the similarity-predicted position at a reappearance, px (config COW_GAP_TOL_PX)")
    ap.add_argument("--cow-max-jumps", type=int, default=dataset_config.get_int("COW_MAX_JUMPS", 5),
                    help="filter (a): DELETE a track with MORE than this many jumps; other tracks are left untouched (config COW_MAX_JUMPS)")
    ap.add_argument("--cow-hue-strict-deg", type=float,
                    default=dataset_config.get_float("COW_HUE_STRICT_DEG", 15.0),
                    help="filter (b): for a track whose outliers reach --cow-hue-outlier-frac, keep only the "
                         "observations within this many degrees of its reference (instead of deleting it)")
    ap.add_argument("--cow-hue-win", type=int, default=dataset_config.get_int("COW_HUE_WIN", 4),
                    help="filter (b): search radius in px; the observation's hue is taken from the pixel in "
                         "this window closest to the track's reference hue, so small local drift does not "
                         "condemn the observation. 0 = the old centre-patch median")
    ap.add_argument("--cow-hue-outlier-frac", type=float,
                    default=dataset_config.get_float("COW_HUE_OUTLIER_FRAC", 0.3),
                    help="filter (b): drop the WHOLE track when this fraction or more of its saturated "
                         "observations deviate past --cow-hue-deg; below it, drop only those observations")
    ap.add_argument("--cow-hue-deg", type=float, default=dataset_config.get_float("COW_HUE_DEG", 30.0),
                    help="filter (b): max hue deviation from the track's FIRST-frame colour, deg (config COW_HUE_DEG; 0 = off)")
    ap.add_argument("--cow-track-frac", type=float, default=dataset_config.get_float("COW_MIN_TRACK_FRAC", 0.8),
                    help="filter (c): a track visible in >= this fraction of the video is kept everywhere (config COW_MIN_TRACK_FRAC; 0 = filter off)")
    ap.add_argument("--cow-frame-frac", type=float, default=dataset_config.get_float("COW_MIN_FRAME_FRAC", 0.5),
                    help="filter (c): a frame with more than this fraction of the started points visible is kept for every track (config COW_MIN_FRAME_FRAC)")
    ap.add_argument("--attach-px", type=float, default=12.0, help="max --pct-percentile distance to attach a track")
    ap.add_argument("--pct", type=float, default=90.0); ap.add_argument("--min-covis", type=int, default=5)
    ap.add_argument("--attach-peer-px", type=float, default=-1.0, help="peer-consistency check (OFF by default = negative): max --pct-percentile distance of a new track to the running average of already-attached tracks; 0 = 1.5 x attach-px")
    ap.add_argument("--min-obs", type=int, default=5, help="drop output tracks with fewer visible frames")
    ap.add_argument("--fill-skeleton", action="store_true", help="where no TAPNet track is attached, use the gated CoW position")
    ap.add_argument("--keep-tapnet", choices=["none", "unattached", "all"], default="unattached",
                    help="also keep the raw TAPNet tracks as their own (parentless, skeleton_index = -1) "
                         "tracks, beside the skeleton averages: 'unattached' (default) only those that "
                         "matched no skeleton point, 'all' every one of them (~5x the tracks, much slower "
                         "lift/build_edg for ~the same PSNR), 'none' for the old behaviour.")
    ap.add_argument("--no-optimize", action="store_true", help="skip optimize_dynamic_tracks.py")
    ap.add_argument("--optimize-args", default="", help="extra args for optimize_dynamic_tracks.py, e.g. \"--w-smooth 5\"")
    ap.add_argument("--no-video", action="store_true", help="skip outputs/cow_attach_compare.mp4")
    ap.add_argument("--fps", type=int, default=10)
    return ap.parse_args()


def main():
    a = parse_args()
    out = Path(D.output_dir); out.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(glob.glob(str(Path(D.video_dir) / D.image_pattern)))
    fwd = np.load(out / a.forward_npz, allow_pickle=True)
    Dxy = fwd["tracks_xy"][0].astype(np.float64); T = Dxy.shape[1]; M = len(Dxy)
    V = (~fwd["occluded"][0]) & fwd["dynamic_obs_mask"] & np.isfinite(Dxy).all(-1)   # TAPNet visible observations
    print(f"[load] {a.forward_npz}: {M} forward tracks, T={T}")

    # ---- 1. skeleton (GPU, other env)
    sk_path = out / a.skeleton
    if not (a.reuse_skeleton and sk_path.exists()):
        cmd = ["conda", "run", "--no-capture-output", "-n", a.cow_env, "python", str(HERE / "cow_skeleton.py"),
               "--images-dir", str(D.video_dir), "--image-pattern", D.image_pattern, "--masks-dir", str(D.mask_dir), "--mask-pattern", D.mask_pattern,
               "--out", str(sk_path), "--every", str(a.every), "--stride", str(a.stride), "--window", str(a.cow_window)]
        print("[skeleton]", " ".join(cmd)); subprocess.run(cmd, check=True)
    sk = np.load(sk_path)
    cow = sk["tracks"].astype(np.float64); start = sk["start_frame"]; N = len(cow)
    assert cow.shape[1] == T, (cow.shape, T)
    skel_vis = (sk["vis"] > a.cow_vis) & (sk["conf"] > a.cow_conf) & np.isfinite(cow).all(-1)
    print(f"[skeleton] {N} points from {len(np.unique(start))} runs (every {int(sk['every'])}, stride {int(sk['stride'])}); gated visible {skel_vis.mean():.2f}")
    frames = [cv2.imread(p) for p in image_paths[:T]]
    if a.cow_jump_px > 0:
        skel_vis, n_rm, n_del, kj = filter_jumpy_tracks(cow, skel_vis, a.cow_jump_px, a.cow_gap_tol_px, a.cow_max_jumps)
        hist = {int(n): int((kj == n).sum()) for n in range(0, min(int(kj.max()) + 1, 8))}
        print(f"[skeleton] filter (a) jumpy tracks (step > {a.cow_jump_px:g} px or gap > {a.cow_gap_tol_px:g} px; "
              f"delete when jumps > {a.cow_max_jumps}): jumps-per-track {hist}" + (f" max {int(kj.max())}" if kj.size else ""))
        print(f"[skeleton]   -> {n_del}/{N} tracks deleted, {n_rm} point-frames removed, survivors UNCHANGED "
              f"-> visible {skel_vis.mean():.2f}")
    if a.cow_hue_deg > 0:
        skel_vis, n_rm, n_hit = filter_color_vs_median(cow, skel_vis, frames, a.cow_hue_deg,
                                                       max_outlier_frac=a.cow_hue_outlier_frac,
                                                       win=a.cow_hue_win, strict_deg=a.cow_hue_strict_deg)
        print(f"[skeleton] filter (b) colour vs median (> {a.cow_hue_deg:g} deg): "
              f"{n_hit} tracks hit, {n_rm} point-frames removed -> visible {skel_vis.mean():.2f}")
    if a.cow_track_frac > 0:
        started = np.array([(start <= t).sum() for t in range(T)])
        skel_vis, n_rm, n_long, n_dense = filter_coverage(skel_vis, a.cow_track_frac, a.cow_frame_frac, started)
        print(f"[skeleton] filter (c) coverage (track >= {100*a.cow_track_frac:.0f}% of the video OR frame > "
              f"{100*a.cow_frame_frac:.0f}% of started points): {n_long} long-lived tracks, {n_dense}/{T} dense frames, "
              f"{n_rm} point-frames removed -> visible {skel_vis.mean():.2f}")
        if not skel_vis.any():
            raise RuntimeError("filter (c) removed every skeleton observation; lower --cow-track-frac / --cow-frame-frac")

    # ---- 2+3. attach (multi-assignment, percentile rule, LONGEST FIRST, peer-consistency) + offset-corrected average
    peer_px = np.inf if a.attach_peer_px < 0 else (a.attach_peer_px if a.attach_peer_px > 0 else 1.5 * a.attach_px)
    length = V.sum(1)

    def _row_pct(x, pct):
        """np.nanpercentile(x, pct, axis=1) via one vectorised sort (NaNs sort last) -- exact, much faster."""
        xs = np.sort(x, axis=1)
        n = (~np.isnan(x)).sum(1)
        out = np.full(x.shape[0], np.nan)
        okr = n > 0
        if okr.any():
            rows = np.where(okr)[0]
            pos = (pct / 100.0) * (n[okr] - 1.0)
            lo = np.floor(pos).astype(np.int64); hi = np.ceil(pos).astype(np.int64)
            va, vb, t = xs[rows, lo], xs[rows, hi], pos - lo
            out[okr] = np.where(t < 0.5, va + t * (vb - va), vb - (1.0 - t) * (vb - va))
        return out

    # per-track 2D bounding box over its VALID frames: a sound prefilter for the attach distance. Any
    # co-visible per-frame distance is >= the gap between the track's bbox and the skeleton point's bbox,
    # and the percentile is >= the minimum, so bbox-gap > attach_px proves the pair can never attach.
    _Dx = np.where(V[..., None], Dxy, np.nan)
    with np.errstate(all="ignore"):
        tb = np.stack([np.nanmin(_Dx[..., 0], 1), np.nanmax(_Dx[..., 0], 1),
                       np.nanmin(_Dx[..., 1], 1), np.nanmax(_Dx[..., 1], 1)], 1)   # (M,4) x0 x1 y0 y1
    del _Dx
    A = np.zeros((N, M), bool); DQ = np.full((N, M), np.inf)
    avg = np.full((N, T, 2), np.nan); n_at = np.zeros((N, T), int); n_del = 0; n_peer_rej = 0
    for i in range(N):
        # The naive version builds a DENSE (M,T) distance matrix + nanpercentile per skeleton point --
        # O(N*M*T) and ~all of the merge runtime. After filter (c) a skeleton point is visible in only a
        # few frames, so restrict to ITS frames and to tracks with enough co-visibility first: identical
        # result (the dropped columns were all-NaN), orders of magnitude less work.
        fr = np.where(skel_vis[i])[0]
        dq = np.full(M, np.inf)
        if fr.size == 0:
            continue
        covf = V[:, fr]                                                # (M,k) co-visibility on those frames
        nc = covf.sum(1)
        sx, sy = cow[i, fr, 0], cow[i, fr, 1]                          # skeleton bbox over ITS frames
        gx = np.maximum(0.0, np.maximum(tb[:, 0] - sx.max(), sx.min() - tb[:, 1]))
        gy = np.maximum(0.0, np.maximum(tb[:, 2] - sy.max(), sy.min() - tb[:, 3]))
        near = np.hypot(gx, gy) <= a.attach_px                         # bbox gap: provably attachable only
        sel = np.where((nc >= a.min_covis) & near)[0]
        if sel.size == 0:
            continue
        df = np.where(covf[sel], np.linalg.norm(Dxy[np.ix_(sel, fr)] - cow[i, fr][None], axis=-1), np.nan)
        dq[sel] = _row_pct(df, a.pct)
        cov = V & skel_vis[i][None]                                    # full-T, for the per-candidate loop
        cand = sel[dq[sel] <= a.attach_px]
        if cand.size == 0:
            continue
        cand = cand[np.argsort(-length[cand], kind="stable")]                      # longest tracks first
        ssum = np.zeros((T, 2)); scnt = np.zeros(T, int)                           # running offset-corrected average
        for m in cand:
            P = Dxy[m]; d = np.linalg.norm(P - cow[i], axis=-1)
            bad = cov[m] & (d > dq[m]); val = V[m] & ~bad; covm = cov[m] & ~bad    # delete the worst (100-pct)% obs of this pair
            if not covm.any():
                continue
            Pc = P - np.nanmean(P[covm] - cow[i][covm], axis=0)
            have = (scnt > 0) & val                                                # frames where the current average exists
            if have.sum() >= a.min_covis:
                dpeer = np.linalg.norm(Pc[have] - ssum[have] / scnt[have, None], axis=-1)
                if np.percentile(dpeer, a.pct) > peer_px:                          # inconsistent with the tracks already attached
                    n_peer_rej += 1; continue
            A[i, m] = True; DQ[i, m] = dq[m]; n_del += int(bad.sum())
            ssum[val] += Pc[val]; scnt[val] += 1
        n_at[i] = scnt
        with np.errstate(all="ignore"):
            avg[i] = np.where(scnt[:, None] > 0, ssum / np.maximum(scnt, 1)[:, None], np.nan)
    attached = A.any(0)
    print(f"[attach] {attached.sum()}/{M} TAPNet tracks attached, {A.sum()} point-track pairs, skeleton points with tracks {(A.any(1)).sum()}/{N}; "
          f"longest-first, peer check <= {peer_px:g} px rejected {n_peer_rej} candidate pairs")
    avg_vis = n_at >= 1
    print(f"[average] deleted {n_del} outlier observations; averaged visible {avg_vis.mean():.2f} (mean {n_at[avg_vis].mean() if avg_vis.any() else 0:.1f} tracks/pt)")
    xy = np.where(avg_vis[..., None], avg, np.nan); vis = avg_vis.copy()
    if a.fill_skeleton:
        fill = skel_vis & ~avg_vis; xy = np.where(fill[..., None], cow, xy); vis |= fill
        print(f"[fill] {fill.sum()} frames filled with gated CoW positions")
    # The merged output is one track per SKELETON point. A TAPNet track that matched no skeleton point would
    # otherwise be thrown away, taking its part of the object with it -- append those unchanged (they are
    # still ordinary dynamic tracks, just short) so build_edg / EDG_test_fast can use them. skeleton_index
    # = -1 marks them as parentless.
    skel_idx = np.arange(len(xy), dtype=np.int64)
    if a.keep_tapnet != "none":
        ua = np.arange(M) if a.keep_tapnet == "all" else np.where(~attached)[0]
        if ua.size:
            xy = np.concatenate([xy, np.where(V[ua][..., None], Dxy[ua], np.nan)], 0)
            vis = np.concatenate([vis, V[ua]], 0)
            start = np.concatenate([start, np.where(V[ua].any(1), V[ua].argmax(1), 0)])
            skel_idx = np.concatenate([skel_idx, np.full(ua.size, -1, np.int64)])
        print(f"[tapnet] --keep-tapnet {a.keep_tapnet}: kept {ua.size} raw TAPNet tracks alongside the "
              f"{len(avg)} skeleton averages (median "
              f"{int(np.median(V[ua].sum(1))) if ua.size else 0} observed frames)")
    # Mask filter (same rule as dynamic_tracking.py, so it applies whichever B1 path produced these tracks):
    #   (i) drop a track whose visible observations are outside the dynamic mask more than --offmask-max-frac
    #       of the time; (ii) otherwise drop just those observations. The CoWTracker skeleton is only
    #       mask-seeded at its reference frame, so without this a track that drifts off the object survives.
    if a.offmask_max_frac >= 0:
        import cv2 as _cv2
        mpx = sorted(glob.glob(str(Path(D.mask_dir) / D.mask_pattern)))[:T]
        inside = np.zeros_like(vis)
        for t, mf in enumerate(mpx):
            mk = _cv2.imread(mf, _cv2.IMREAD_GRAYSCALE)
            if mk is None:
                inside[:, t] = vis[:, t]; continue
            mk = mk > 127; H_, W_ = mk.shape
            xi = np.rint(xy[:, t, 0]).astype(np.int64); yi = np.rint(xy[:, t, 1]).astype(np.int64)
            ok = vis[:, t] & np.isfinite(xy[:, t]).all(-1) & (xi >= 0) & (xi < W_) & (yi >= 0) & (yi < H_)
            inside[ok, t] = mk[yi[ok], xi[ok]]
        off = vis & ~inside
        den = vis.sum(1)
        ratio = np.divide(off.sum(1), den, out=np.zeros(len(den), np.float64), where=den > 0)
        drop = ratio > a.offmask_max_frac
        vis = vis & inside                                       # (ii) per-observation
        vis[drop] = False                                        # (i)  whole track
        print(f"[mask] off-mask filter (> {a.offmask_max_frac:.0%} outside): {int(drop.sum())}/{len(den)} "
              f"tracks dropped entirely, {int((off & ~drop[:, None]).sum())} further observations dropped")

    keep = vis.sum(1) >= a.min_obs
    xy, vis, start_k, skel_k = xy[keep], vis[keep], start[keep], skel_idx[keep]; n = len(xy)
    occ = ~vis; qt = np.where(vis.any(1), vis.argmax(1), 0)
    np.savez_compressed(out / a.output, tracks_xy=xy[None].astype(np.float32), tracks_xyz=np.full((1, n, T, 3), np.nan),
                        occluded=occ[None], dynamic_obs_mask=vis, query_t=qt.astype(np.int32),
                        member_src=np.full((n, T), -1, np.int32), n_forward=np.int64(n), skeleton_start_frame=start_k.astype(np.int32), skeleton_index=skel_k.astype(np.int32),
                        video_dir=str(D.video_dir), mask_dir=str(D.mask_dir))
    obs = vis.sum(1)
    print(f"[save] {out / a.output}: {n} tracks ({int((skel_k < 0).sum())} unattached TAPNet; "
          f"dropped {int((~keep).sum())} with < {a.min_obs} obs; observed frames/track median {int(np.median(obs))} max {int(obs.max())})")
    (out / "cow_attach_summary.json").write_text(json.dumps({
        "forward_tracks": int(M), "skeleton_points": int(N), "attached_tracks": int(attached.sum()), "pairs": int(A.sum()), "deleted_obs": int(n_del),
        "output_tracks": int(n), "median_obs": int(np.median(obs)), "visible_frac": float(vis.mean()),
        "params": {k: v for k, v in vars(a).items()}}, indent=2))

    # ---- video: TAPNet forward tracks | CoW skeleton (gated) | output tracks
    if not a.no_video and image_paths:
        H, W = frames[0].shape[:2]
        mpaths = sorted(glob.glob(str(Path(D.mask_dir) / D.mask_pattern)))[:T]
        col_of = np.where(attached, A.argmax(0), 0)
        hue = lambda q: cv2.cvtColor(np.stack([((np.arctan2(q[:, 1] - q[:, 1].mean(), q[:, 0] - q[:, 0].mean()) + np.pi) / (2 * np.pi) * 179).astype(np.uint8), np.full(len(q), 255, np.uint8), np.full(len(q), 255, np.uint8)], 1)[None], cv2.COLOR_HSV2BGR)[0].astype(int)
        colors = np.zeros((N, 3), int)
        for s in np.unique(start):
            sel = start == s; colors[sel] = hue(sk["queries"][sel])
        head, foot = 28, 22; PH = head + H + foot; tmp = out / "tmp_cow_attach.avi"
        vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"MJPG"), a.fps, (3 * W, PH))
        keep_idx = np.where(keep)[0]
        for t in range(T):
            row = []
            mk = cv2.imread(mpaths[t], 0) if t < len(mpaths) else None
            for lab, mode in (("TAPNet forward tracks (grey = not attached)", "dyn"), (f"CoWTracker skeleton, fwd+bwd every {a.every}, gate {a.cow_vis}, move+hue filtered", "skel"), (f"output: average of attached TAPNet tracks{' + skeleton fill' if a.fill_skeleton else ''}", "out")):
                img = frames[t].copy()
                if mk is not None:
                    cnts, _ = cv2.findContours((mk > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(img, cnts, -1, (255, 255, 255), 1)
                if mode == "dyn":
                    alive = np.where(V[:, t])[0]
                    for m in alive:
                        c = tuple(int(x) for x in colors[col_of[m]]) if attached[m] else (170, 170, 170)
                        cv2.circle(img, (int(round(Dxy[m, t, 0])), int(round(Dxy[m, t, 1]))), 1, c, -1, cv2.LINE_AA)
                    txt = f"{len(alive)} tracks ({int(attached[alive].sum())} attached)"
                elif mode == "skel":
                    for i in np.where(skel_vis[:, t])[0]:
                        cv2.circle(img, (int(round(cow[i, t, 0])), int(round(cow[i, t, 1]))), 3, tuple(int(x) for x in colors[i]), -1, cv2.LINE_AA)
                    txt = f"visible {100 * skel_vis[:, t].mean():.0f}% of {N}"
                else:
                    for j in np.where(vis[:, t])[0]:
                        cv2.circle(img, (int(round(xy[j, t, 0])), int(round(xy[j, t, 1]))), 3, tuple(int(x) for x in colors[keep_idx[j]]), -1, cv2.LINE_AA)
                    txt = f"visible {100 * vis[:, t].mean():.0f}% of {n}"
                pnl = np.zeros((PH, W, 3), np.uint8); pnl[head:head + H] = img
                cv2.putText(pnl, lab, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(pnl, txt, (6, head + H + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA); row.append(pnl)
            canvas = np.concatenate(row, 1); cv2.putText(canvas, f"t={t:03d}/{T - 1}", (3 * W - 100, head + H + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA); vw.write(canvas)
        vw.release(); vid = out / "cow_attach_compare.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(vid)], check=True); tmp.unlink()
        print(f"[video] {vid}")

    # ---- 4. optimize / smooth 3D (same env, same scene env vars)
    if not a.no_optimize:
        cmd = [sys.executable, str(HERE / "optimize_dynamic_tracks.py"), "--input", a.output] + a.optimize_args.split()
        print("[optimize]", " ".join(cmd)); subprocess.run(cmd, check=True, cwd=str(HERE), env=os.environ.copy())
        print(f"[done] build_edg.py will read {Path(a.output).stem}_smoothed.npz")


if __name__ == "__main__":
    main()
