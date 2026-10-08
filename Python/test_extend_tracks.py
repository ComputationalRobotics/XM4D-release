#!/usr/bin/env python3
"""Extend 2D tracks across occlusion, then LINK fragmented tracks that are actually the same point into
single longer tracks (1-to-1 chains, not groups).

Idea (tracks are too short -- a point gets re-tracked under a new id after an occlusion). When A becomes
invisible the tracker KEEPS predicting A's 2D position (finite but occluded), so:
  * For each track A, take its predicted positions AFTER its last visible frame, in the window
    [last_vis_A + 1, last_vis_A + WINDOW], and find a continuation B that (a) is VISIBLE and extends A
    forward (last_vis_B > last_vis_A), (b) stays within MERGE_PX of A's predicted position at every such
    frame (>= MIN_OVERLAP of them), and (c) has a similar mean colour (<= --color-thresh, BGR euclidean).
  * MATCH 1-to-1: a track can merge to only ONE other at a time, so we greedily accept the best (closest)
    candidate links such that each track has at most one successor and one predecessor -> linear CHAINS.
    Each chain is one consolidated track (tracks, not groups).
  * TWO STAGES: merge forward and backward tracks individually, then merge across the two directions.
  * POST-MERGE CLEANUP (clean_tracks): drop per-OBSERVATION outliers on the final tracks (never whole
    tracks) -- (1) 2D-jump spikes (position deviates from the line between its visible neighbours) and
    (2) colour drift (a per-track robust colour outlier vs the track's dominant/median colour).
  * FINAL COLOUR FILTER (--good-thresh, same test as visualize_tracks_2d): each final track is judged by
    its OWN 2D hue consistency (sat-gated p90 circular spread; colourless surfaces accepted); a NOT-GOOD
    track is BROKEN back into its member_src fragments and those are re-filtered -- fragments that still
    fail are DROPPED, the rest are kept as separate tracks.

Outputs:
  * outputs/extend_tracks_intervals.png -- one row per merged track, its appear interval (GREEN=observed,
    ORANGE=gap bridged by the merge), sorted by first appearance (same style as the EDG active-horizon).
  * outputs/extend_tracks_merges.npz -- accepted links + per-track final chain id.

Viz (OpenCV; SPACE=next frame, B=prev, Q/ESC=quit):
  GREEN=visible, BLUE=invisible(occluded), ORANGE=extended prediction (+line to the track it merged with).

Run: XM_DATASET_NAME=<scene> conda run -n XM python test_extend_tracks.py [--headless] [--merge-px 10]
"""
import argparse
import glob
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

import utils.definitions
import visualize_track_edges as VTE          # reuse _load_all (module main-guarded, safe to import)


def load_2d():
    D = VTE._load_all()
    xy = D["tracks_xy"][0].astype(np.float64)                 # (Q,T,2) native px
    occ = np.asarray(D["occluded"][0], dtype=bool)            # (Q,T) True = occluded/invisible
    finite = np.isfinite(xy).all(axis=-1)                     # tracker has a position (>= query_t)
    vis = finite & (~occ)                                     # visible = has position AND not occluded
    T = D["T"]
    image_paths = sorted(glob.glob(str(Path(utils.definitions.video_dir) / utils.definitions.image_pattern)))
    return xy, vis, finite, T, image_paths, D


def _frame_trees(xy, vis, T):
    idx_per, tree_per = [], []
    for t in range(T):
        idx = np.where(vis[:, t])[0]
        idx_per.append(idx)
        tree_per.append(cKDTree(xy[idx, t]) if idx.size else None)
    return idx_per, tree_per


def track_mean_colors(xy, vis, image_paths, T, verbose=True):
    """Per-track MEAN BGR colour sampled at the track's visible positions across the video (NaN if never
    sampled). Appearance gate for merging: the same physical point keeps a similar average colour, while a
    neighbouring point on a differently-coloured blade does not."""
    import cv2
    Q = xy.shape[0]
    acc = np.zeros((Q, 3), np.float64)
    cnt = np.zeros(Q, np.int64)
    for t in tqdm(range(min(T, len(image_paths))), desc="[color] sampling", disable=not verbose):
        img = cv2.imread(image_paths[t])
        if img is None:
            continue
        H, W = img.shape[:2]
        idx = np.where(vis[:, t])[0]
        if idx.size == 0:
            continue
        xi = np.clip(np.rint(xy[idx, t, 0]).astype(np.int64), 0, W - 1)
        yi = np.clip(np.rint(xy[idx, t, 1]).astype(np.int64), 0, H - 1)
        acc[idx] += img[yi, xi].astype(np.float64)
        cnt[idx] += 1
    col = np.full((Q, 3), np.nan)
    ok = cnt > 0
    col[ok] = acc[ok] / cnt[ok, None]
    return col


def _sample_obs_colors(xy, vis, image_paths, T, verbose=True):
    """Per-OBSERVATION BGR colour [Q,T,3] at every visible cell (NaN elsewhere)."""
    import cv2
    Q = xy.shape[0]
    col = np.full((Q, T, 3), np.nan)
    for t in tqdm(range(min(T, len(image_paths))), desc="[clean] colour", disable=not verbose):
        img = cv2.imread(image_paths[t])
        if img is None:
            continue
        H, W = img.shape[:2]
        idx = np.where(vis[:, t])[0]
        if idx.size == 0:
            continue
        xi = np.clip(np.rint(xy[idx, t, 0]).astype(np.int64), 0, W - 1)
        yi = np.clip(np.rint(xy[idx, t, 1]).astype(np.int64), 0, H - 1)
        col[idx, t] = img[yi, xi].astype(np.float64)
    return col


def track_hue_spread(xy, vis, image_paths, T, pct=90, verbose=True):
    """Per-track circular HUE spread: the p<pct> angular distance (0..90 scale, 90 = half the hue circle)
    of each visible observation's hue to the track's circular-mean hue. Hue (HSV) is far more robust to
    lighting/specular than BGR, so this SEPARATES single-colour fragments (spread ~<10) from fragments that
    sweep across colours (the drift/specular tail). NaN for tracks with < 4 sampled observations."""
    import cv2
    Q = xy.shape[0]
    hue = np.full((Q, T), np.nan, np.float32)
    for t in tqdm(range(min(T, len(image_paths))), desc="[spread] hue", disable=not verbose):
        im = cv2.imread(image_paths[t])
        if im is None:
            continue
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        H, W = im.shape[:2]
        idx = np.where(vis[:, t])[0]
        if idx.size == 0:
            continue
        xi = np.clip(np.rint(xy[idx, t, 0]).astype(np.int64), 0, W - 1)
        yi = np.clip(np.rint(xy[idx, t, 1]).astype(np.int64), 0, H - 1)
        hue[idx, t] = hsv[yi, xi, 0]
    ang = hue / 180.0 * (2 * np.pi)                                  # OpenCV hue 0..180 -> radians
    sp = np.full(Q, np.nan)
    for q in range(Q):
        m = np.isfinite(ang[q])
        if m.sum() < 4:
            continue
        a = ang[q, m]
        mean = np.arctan2(np.sin(a).mean(), np.cos(a).mean())        # circular mean hue
        dd = np.abs(np.angle(np.exp(1j * (a - mean)))) / np.pi * 90.0
        sp[q] = np.percentile(dd, pct)
    return sp


def clean_tracks(xy, xyz, occ, image_paths, T, move_sigma, move_px, color_sigma, color_min, fill_gap,
                 verbose=True):
    """Post-merge per-OBSERVATION cleanup on the consolidated tracks. Deletes single (track,frame) cells
    (NaN xy/xyz + set occluded), NEVER whole tracks. Two independent filters:
      (1) 2D-jump: a visible cell whose position deviates from the straight line between its nearest visible
          temporal neighbours by a robust outlier amount (pooled median + move_sigma*1.4826*MAD, and/or an
          absolute residual > move_px) -- catches a frame that drifted to another point (and back).
      (2) colour drift: for each track, distance of each observation's sampled BGR colour to the track's
          DOMINANT (per-channel median) colour; delete cells whose distance is a PER-TRACK robust outlier
          (> track median + color_sigma*1.4826*track MAD, and >= color_min px) -- catches a track sliding
          onto a differently-coloured region. Per-track (not fixed) because intra-track colour varies a lot
          (shading), so a fixed distance would gut normal variation.
    Then REPAIR: any interior discontinuity spanning < fill_gap frames (bounded by surviving observations
    on both sides) is linearly interpolated (xy + xyz, marked visible) -- whatever opened it, a deletion
    above OR an original tracker-occlusion gap whose carried predictions are unreliable; 0 disables.
    Returns (xy, xyz, occ, nosrc_mask) where nosrc = cells removed OR interpolated (no source track)."""
    vis = (~occ) & np.isfinite(xy).all(-1)
    Q = xy.shape[0]
    del_move = np.zeros((Q, T), bool)
    del_color = np.zeros((Q, T), bool)

    # ---- filter 1: 2D interpolation-residual spike ----
    if move_sigma > 0 or move_px > 0:
        resid = np.full((Q, T), np.nan)
        for q in range(Q):
            vf = np.where(vis[q])[0]
            if vf.size < 3:
                continue
            p = xy[q, vf]                                            # (k,2) visible positions
            a, t, b = vf[:-2], vf[1:-1], vf[2:]                      # interior triples
            alpha = (t - a) / (b - a)
            pred = p[:-2] + alpha[:, None] * (p[2:] - p[:-2])        # linear interp at t from neighbours
            resid[q, t] = np.linalg.norm(p[1:-1] - pred, axis=1)
        rvals = resid[np.isfinite(resid)]
        thr = np.inf
        if rvals.size and move_sigma > 0:
            med = np.median(rvals); mad = np.median(np.abs(rvals - med))
            thr = med + move_sigma * 1.4826 * mad
        del_move = np.isfinite(resid) & (resid > thr)
        if move_px > 0:
            del_move |= np.isfinite(resid) & (resid > move_px)
        if verbose:
            lim = f"thr={thr:.1f}px" if move_sigma > 0 else f"px>{move_px:g}"
            print(f"[clean] 2D-jump: deleted {int(del_move.sum())} observations ({lim})")

    # ---- filter 2: colour drift, per-track robust outlier from the track's dominant colour ----
    if color_sigma > 0:
        col = _sample_obs_colors(xy, vis, image_paths, T, verbose)   # (Q,T,3), NaN where not visible
        dom = np.nanmedian(np.where(np.isfinite(col), col, np.nan), axis=1)  # (Q,3) dominant colour
        with np.errstate(invalid="ignore"):
            cdist = np.linalg.norm(col - dom[:, None, :], axis=-1)   # (Q,T)
        for q in range(Q):
            fin = np.isfinite(cdist[q])
            if fin.sum() < 4:                                        # too few to estimate the spread
                continue
            d = cdist[q, fin]
            med = np.median(d); mad = np.median(np.abs(d - med))
            thr = max(med + color_sigma * 1.4826 * mad, color_min)
            del_color[q] = fin & (cdist[q] > thr)
        if verbose:
            print(f"[clean] colour: deleted {int(del_color.sum())} observations "
                  f"(per-track > median + {color_sigma:g}*MAD, floor {color_min:g})")

    dele = del_move | del_color
    xy = xy.copy(); occ = occ.copy()
    xy[dele] = np.nan
    occ[dele] = True
    if xyz is not None:
        xyz = xyz.copy(); xyz[dele] = np.nan
    if verbose:
        print(f"[clean] total {int(dele.sum())} observations deleted "
              f"({int(del_move.sum())} move + {int(del_color.sum())} colour, "
              f"{int((del_move & del_color).sum())} both)")

    # ---- repair: interpolate any SHORT interior discontinuity (< fill_gap frames), whatever its cause
    # (a deletion above OR an original tracker-occlusion gap whose predictions are unreliable) ----
    filled = np.zeros((Q, T), bool)
    if fill_gap > 0:
        visN = (~occ) & np.isfinite(xy).all(-1)                     # visibility AFTER deletion
        for q in range(Q):
            vf = np.where(visN[q])[0]
            if vf.size < 2:
                continue
            for l, r in zip(vf[:-1], vf[1:]):                       # gap = frames strictly between l and r
                if r - l - 1 <= 0 or r - l - 1 >= fill_gap:
                    continue
                seg = np.arange(l + 1, r)
                w = ((seg - l) / (r - l))[:, None]
                xy[q, seg] = xy[q, l] + w * (xy[q, r] - xy[q, l])   # linear interp across the gap
                if xyz is not None and np.isfinite(xyz[q, l]).all() and np.isfinite(xyz[q, r]).all():
                    xyz[q, seg] = xyz[q, l] + w * (xyz[q, r] - xyz[q, l])
                occ[q, seg] = False
                filled[q, seg] = True
        if verbose:
            print(f"[clean] fill: interpolated {int(filled.sum())} observations in gaps < {fill_gap} frames")

    # cells with no valid source track = removed OR interpolated
    return xy, xyz, occ, (dele | filled)


def find_candidates(xy, vis, finite, T, window, merge_px, min_overlap, topk, verbose=True,
                    track_colors=None, color_thresh=0.0, strategy="after"):
    """Continuation search. Two strategies for choosing the reference frames where A is compared to B:
      strategy="after"  -- use A's OWN 2D track AFTER it becomes invisible. When A goes occluded the tracker
        keeps predicting a position (finite but not visible); we take the window [last_vis_A+1, +window] and
        compare A's PREDICTED position there to a VISIBLE B. Best for RAW tracks that still carry the
        tracker's post-occlusion predictions.
      strategy="before" -- overlap-based. Use the window of A's last visible frames [last_vis_A-window+1,
        last_vis_A] and compare A's real position to a VISIBLE B at the CO-VISIBLE frames. Used when merging
        already-consolidated chains (F+B), which no longer carry a single tracker's post-occlusion prediction.
    In both cases B must extend A forward (last_vis_B > last_vis_A), stay within merge_px at >= min_overlap
    reference frames, and match colour (<= color_thresh). Returns (candidates, last_vis); cand=(A,B,ref,score)."""
    Q = xy.shape[0]
    idx_per, tree_per = _frame_trees(xy, vis, T)                                    # KD-trees over VISIBLE tracks
    last_vis = np.where(vis.any(1), (T - 1) - np.argmax(vis[:, ::-1], axis=1), -1)  # last visible frame
    candidates = []
    n_color_rej = 0
    for q in tqdm(range(Q), desc="[merge] tracks", disable=not verbose):
        la = int(last_vis[q])
        if la < 0 or la >= T - 1:
            continue
        if strategy == "after":
            # A's predicted positions AFTER it becomes invisible (finite but occluded), within the window
            ref = [t for t in range(la + 1, min(la + 1 + int(window), T)) if finite[q, t]]
        else:
            # A's last visible-window frames BEFORE (and including) it becomes invisible
            ref = [t for t in range(max(0, la - int(window) + 1), la + 1) if vis[q, t]]
        if not ref:
            continue
        cand = set()                                                                # visible B near A
        for t in ref:
            if tree_per[t] is None:
                continue
            for h in tree_per[t].query_ball_point(xy[q, t], merge_px):
                b = int(idx_per[t][h])
                if b != q and last_vis[b] > la:                                     # B must extend A forward
                    cand.add(b)
        quals = []
        for b in cand:
            co = [t for t in ref if vis[b, t]]                                      # A at ref AND B visible
            if len(co) < min_overlap:
                continue
            dists = [float(np.linalg.norm(xy[q, t] - xy[b, t])) for t in co]
            if max(dists) >= merge_px:                                             # A stays near B
                continue
            if (track_colors is not None and color_thresh > 0
                    and np.isfinite(track_colors[q]).all() and np.isfinite(track_colors[b]).all()
                    and float(np.linalg.norm(track_colors[q] - track_colors[b])) > color_thresh):
                n_color_rej += 1                                                    # appearance mismatch
                continue
            quals.append((float(np.mean(dists)), b))
        quals.sort()
        for score, b in quals[:topk]:
            candidates.append((int(q), int(b), int(la), float(score)))
    if track_colors is not None and color_thresh > 0:
        print(f"[color] rejected {n_color_rej} candidate links with mean-colour distance > {color_thresh:g}")
    return candidates, last_vis


def assign_chains(candidates, Q):
    """Greedy best-first 1-to-1 matching: each track gets at most one successor and one predecessor, so
    accepted links form linear CHAINS. Returns (final chain-id per track, accepted links)."""
    succ, pred, accepted = {}, {}, []
    for A, B, t_occ, score in sorted(candidates, key=lambda x: x[3]):               # best (closest) first
        if A == B or A in succ or B in pred:
            continue
        cur, seen, cyc = B, set(), False                                            # guard against cycles
        while cur in succ and cur not in seen:
            seen.add(cur); cur = succ[cur][0]
            if cur == A:
                cyc = True; break
        if cyc:
            continue
        succ[A] = (B, t_occ); pred[B] = A
        accepted.append((A, B, int(t_occ), float(score)))
    final = np.arange(Q, dtype=np.int64)
    for start in range(Q):
        if start in pred:                                                           # only walk from heads
            continue
        cur = start
        while cur in succ:
            cur = succ[cur][0]
            final[cur] = start
    return final, accepted


def draw_intervals(final, vis, T, out_path, only_merged=True, max_rows=4000):
    """One row per (merged) track: its appear interval. GREEN=observed, ORANGE=gap bridged by the merge,
    gray=outside its span. Sorted by first appearance (same style as visualize_edg_active_horizon)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    members = defaultdict(list)
    for q in range(vis.shape[0]):
        members[int(final[q])].append(q)
    chains = [m for m in members.values() if (len(m) > 1 or not only_merged)]

    rows, first_key, spans, frag_lens = [], [], [], []
    for mem in chains:
        anyvis = vis[mem].any(axis=0)                                              # (T,)
        if not anyvis.any():
            continue
        first = int(np.argmax(anyvis)); last = int(T - 1 - np.argmax(anyvis[::-1]))
        stat = np.zeros(T, np.int8)
        stat[first:last + 1] = 2                                                   # within span -> orange
        stat[anyvis] = 1                                                           # observed -> green
        rows.append(stat); first_key.append(first); spans.append(last - first + 1)
        frag_lens.extend([int(vis[q].sum()) for q in mem])
    if not rows:
        print("[intervals] no merged tracks to draw.")
        return
    order = np.argsort(first_key)
    M = np.stack(rows)[order]
    truncated = M.shape[0] > max_rows
    if truncated:
        M = M[:max_rows]
    N = M.shape[0]

    fig_w = max(8.0, T * 0.03 + 2.0); fig_h = max(6.0, N * 0.03 + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    ax.imshow(M, aspect="auto", interpolation="nearest", vmin=0, vmax=2,
              cmap=ListedColormap(["#eeeeee", "#1a9850", "#f59f0a"]))              # gray, green, orange
    ax.set_xlabel("frame")
    ax.set_ylabel("merged track (sorted by first appearance)")
    ax.set_title(f"Merged-track appear intervals  |  {N}{'+ (truncated)' if truncated else ''} tracks x {T} "
                 f"frames  |  GREEN=observed  ORANGE=bridged gap")
    ax.set_xticks(np.arange(0, T, max(1, T // 30))); ax.tick_params(axis="x", labelsize=6, rotation=90)
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    print(f"[intervals] saved {N} merged-track intervals -> {out_path}  "
          f"(median fragment len={int(np.median(frag_lens))} -> median merged span={int(np.median(spans))} frames)")


def build_merged_tracks(final, xy, vis, T):
    """For each final (chain) track, its per-frame position over the WHOLE sequence: pos[T,2], status[T]
    (0=none, 1=observed member). With overlap merges there are no extrapolated bridges -- the chain is the
    union of its members' real observations (at an overlap frame, a visible member's position is used)."""
    Q = len(final)
    members = defaultdict(list)
    for q in range(Q):
        members[int(final[q])].append(q)
    tracks = {}
    for head, mem in members.items():
        pos = np.full((T, 2), np.nan, dtype=np.float64)
        status = np.zeros(T, dtype=np.int8)
        for q in mem:                                        # observed member positions
            vq = vis[q]
            pos[vq] = xy[q, vq]
            status[vq] = 1
        tracks[head] = (pos, status)
    return tracks


def _track_span(status):
    w = np.where(status > 0)[0]
    return int(w[-1] - w[0] + 1) if w.size else 0


def _longest(tracks, n):
    return sorted(tracks.items(), key=lambda kv: _track_span(kv[1][1]), reverse=True)[:n]


def draw_longest_trajectories(tracks, T, image_paths, out_path, n=30):
    """PNG: the n longest merged tracks' full 2D trajectories over the whole sequence (each a coloured
    polyline over the first frame; dots = observed, thin line spans bridged gaps too)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cv2
    items = _longest(tracks, n)
    img0 = cv2.cvtColor(cv2.imread(image_paths[0]), cv2.COLOR_BGR2RGB) if image_paths else None
    fig, ax = plt.subplots(figsize=(14, 10), dpi=130)
    if img0 is not None:
        ax.imshow(img0, alpha=0.5)
    else:
        ax.invert_yaxis()
    colors = plt.cm.hsv(np.linspace(0, 1, len(items), endpoint=False))
    for k, (head, (pos, st)) in enumerate(items):
        obs = np.where(st == 1)[0]                           # only the REAL (observed) points, no path line
        ax.scatter(pos[obs, 0], pos[obs, 1], s=6, color=colors[k])
    ax.set_title(f"{len(items)} longest merged tracks -- real (observed) points over the whole sequence "
                 f"(T={T} frames)")
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    spans = [_track_span(st) for _, (_, st) in items]
    print(f"[longest] saved {len(items)} longest-track trajectories -> {out_path}  "
          f"(spans {min(spans) if spans else 0}..{max(spans) if spans else 0} frames)")


def visualize_longest(tracks, T, image_paths, n=30, start=0):
    """OpenCV stepper of the n longest merged tracks: each a distinct colour, full path drawn + current
    position (filled=observed, ring=bridged). SPACE=next, B=prev, Q/ESC=quit."""
    import sys
    if not sys.stdout.isatty():
        print("[viewer] non-interactive session: skipping OpenCV viewer (longest tracks)")
        return
    import cv2
    import matplotlib.cm as cm
    items = _longest(tracks, n)
    cols = [tuple(int(c) for c in (np.array(cm.hsv(i / max(1, len(items)))[:3]) * 255)[::-1]) for i in range(len(items))]
    stt = {"t": int(np.clip(start, 0, T - 1))}

    def draw(t):
        img = cv2.imread(image_paths[t]).copy() if t < len(image_paths) else np.zeros((480, 640, 3), np.uint8)
        for k, (head, (pos, status)) in enumerate(items):
            c = cols[k]
            if status[t] > 0 and np.isfinite(pos[t]).all():  # only the real point at this frame, no path line
                p = (int(pos[t, 0]), int(pos[t, 1]))
                if status[t] == 1:
                    cv2.circle(img, p, 5, c, -1, cv2.LINE_AA)     # observed (real) -> filled
                else:
                    cv2.circle(img, p, 5, c, 2, cv2.LINE_AA)      # bridged/predicted -> ring
        ln = f"{len(items)} longest merged tracks   frame {t}/{T-1}   SPACE=next B=prev Q=quit   (ring=bridged)"
        cv2.putText(img, ln, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, ln, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("longest merged tracks", img)

    print("OpenCV window (longest tracks): SPACE=next frame, B=prev, Q/ESC=quit")
    draw(stt["t"])
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == 32:
            stt["t"] = (stt["t"] + 1) % T; draw(stt["t"])
        elif k in (ord("b"), ord("B")):
            stt["t"] = (stt["t"] - 1) % T; draw(stt["t"])
    cv2.destroyAllWindows()


def visualize(xy, vis, finite, T, image_paths, accepted, last_vis, start=0):
    import sys
    if not sys.stdout.isatty():
        print("[viewer] non-interactive session: skipping OpenCV viewer")
        return
    import cv2
    links = [(int(a), int(b)) for a, b, _, _ in accepted]            # overlap-merged pairs (A extended by B)

    C_VIS = (60, 220, 60); C_INV = (255, 60, 60); C_LINK = (0, 165, 255)
    st = {"t": int(np.clip(start, 0, T - 1))}

    def draw(t):
        img = cv2.imread(image_paths[t]) if t < len(image_paths) else None
        img = (np.zeros((480, 640, 3), np.uint8) if img is None else img).copy()
        ex = finite[:, t]; vv = vis[:, t] & ex; ii = (~vis[:, t]) & ex
        for q in np.where(vv)[0]:
            p = xy[q, t]; cv2.circle(img, (int(round(p[0])), int(round(p[1]))), 2, C_VIS, -1, cv2.LINE_AA)
        for q in np.where(ii)[0]:
            p = xy[q, t]; cv2.circle(img, (int(round(p[0])), int(round(p[1]))), 2, C_INV, -1, cv2.LINE_AA)
        n_link = 0
        for a, b in links:                                          # draw the merge where A and B overlap
            if vis[a, t] and vis[b, t]:
                pa = (int(round(xy[a, t, 0])), int(round(xy[a, t, 1])))
                pb = (int(round(xy[b, t, 0])), int(round(xy[b, t, 1])))
                cv2.line(img, pa, pb, C_LINK, 1, cv2.LINE_AA)
                cv2.circle(img, pb, 4, C_LINK, 1, cv2.LINE_AA)
                n_link += 1
        for k, ln in enumerate([
            f"frame {t}/{T-1}   SPACE=next  B=prev  Q=quit",
            f"GREEN visible={int(vv.sum())}   BLUE invisible={int(ii.sum())}   "
            f"ORANGE overlap-merge links this frame={n_link}",
            f"accepted 1-to-1 merges (whole video)={len(accepted)}",
        ]):
            y = 22 + 20 * k; col = (0, 255, 255) if k == 0 else (255, 255, 255)
            cv2.putText(img, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        cv2.imshow("extend + merge 2D tracks", img)

    print("OpenCV window: SPACE=next frame, B=prev, Q/ESC=quit")
    draw(st["t"])
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == 32:
            st["t"] = (st["t"] + 1) % T; draw(st["t"])
        elif k in (ord("b"), ord("B")):
            st["t"] = (st["t"] - 1) % T; draw(st["t"])
    cv2.destroyAllWindows()


def write_merged_tracks(final, D, vis, T, out_path):
    """Collapse each 1-to-1 chain into ONE consolidated track and save a drop-in tracks npz (SAME schema
    as tracks_dynamic_only.npz) with longer tracks. Each frame takes a visible member's observation (2D +
    3D); gaps stay occluded (real observations only, no synthetic fill). Downstream build_edg.py /
    EDG_test.py read this with --merged."""
    tracks_xy = D["tracks_xy"]; tracks_xyz = D["tracks_xyz"]
    query_t = D.get("query_t")
    Q = tracks_xy.shape[1]
    heads = np.unique(final)
    n = heads.size
    xy_m = np.full((1, n, T, 2), np.nan, np.float32)
    xyz_m = np.full((1, n, T, 3), np.nan, np.float64)
    occ_m = np.ones((1, n, T), dtype=bool)
    dyn_m = np.zeros((n, T), dtype=bool)
    qt_m = np.zeros(n, dtype=np.int64)
    member_src = np.full((n, T), -1, dtype=np.int32)         # WHICH original track supplied each frame
    members = defaultdict(list)
    for q in range(Q):
        members[int(final[q])].append(q)
    for i, h in enumerate(heads.tolist()):
        mem = np.asarray(members[h], dtype=int)
        vm = vis[mem]                                        # (m,T)
        fa = np.where(vm.any(0))[0]                          # frames where some member is visible
        if fa.size:
            sel = mem[vm.argmax(0)[fa]]                      # first visible member id per active frame
            xy_m[0, i, fa] = tracks_xy[0, sel, fa]
            xyz_m[0, i, fa] = tracks_xyz[0, sel, fa]
            occ_m[0, i, fa] = False
            dyn_m[i, fa] = True
            member_src[i, fa] = sel.astype(np.int32)
        qt_m[i] = (int(min(int(query_t[q]) for q in mem.tolist())) if query_t is not None
                   else (int(fa[0]) if fa.size else 0))
    np.savez_compressed(out_path,
                        tracks_xy=xy_m, tracks_xyz=xyz_m, occluded=occ_m,
                        dynamic_obs_mask=dyn_m, query_t=qt_m.astype(np.int32),
                        member_src=member_src,               # per-frame ORIGINAL track id (-1 = occluded)
                        merged_heads=np.asarray(heads, dtype=np.int64),   # chain head per row
                        video_dir=str(utils.definitions.video_dir), mask_dir=str(utils.definitions.mask_dir))
    obs_len = (~occ_m[0]).sum(1)
    print(f"[merged-tracks] {n} consolidated tracks (from {Q}) -> {out_path}  "
          f"(observed frames/track: median={int(np.median(obs_len))} max={int(obs_len.max())})")


def _load_npz_tracks(path):
    """Load a dynamic-tracking npz -> (xy[Q,T,2], vis[Q,T], finite[Q,T], xyz=None). `finite` =
    the tracker has a 2D position (visible OR occluded-but-predicted); `vis` = finite AND not occluded.

    NOTE: merging is a 2D-only operation (linking by pixel proximity + colour). It deliberately does NOT
    read or carry 3D -- xyz is always None here, so the merged output's tracks_xyz stays all-NaN and the
    3D is (re)lifted downstream from the specified depth (utils.definitions.active_depth_maps) in
    optimize_dynamic_tracks.py. This keeps the merge depth-agnostic."""
    d = np.load(path, allow_pickle=True)
    xy = np.asarray(d["tracks_xy"], np.float64)[0]
    occ = np.asarray(d["occluded"], bool)[0]
    finite = np.isfinite(xy).all(-1)
    return xy, finite & (~occ), finite, None


def consolidate(xy, vis, finite, xyz, T, image_paths, args, tag, strategy="after"):
    """ONE merge pass: colour-gate + find_candidates(strategy) + 1-to-1 chains, then collapse each chain
    into a consolidated track. Fills a visible member's obs where available (occ=False), else an
    occluded-but-PREDICTED member position (occ=True) -- so the consolidated tracks KEEP their post-occlusion
    2D predictions. strategy="after" (raw within-direction) uses A's predictions after invisibility;
    "before" (cross-direction chains) uses A's last visible-window overlap. Returns (xy_c, xyz_c, occ_c,
    qt_c, src_c); src_c[i,t] = the INPUT index that supplied frame t (-1 = none)."""
    # SPREAD GUARD: drop fragments that sweep across colours (unreliable/drifting) BEFORE merging, so they
    # neither merge nor survive. Uses HUE (robust to the pinwheel's specular/lighting; BGR is not
    # discriminative here -- most fragments have huge BGR spread from specular alone).
    if args.merge_hue_spread > 0:
        hs = track_hue_spread(xy, vis, image_paths, T)
        keep = ~(hs > args.merge_hue_spread)                            # NaN (< 4 obs) -> kept
        print(f"[{tag}] spread guard: dropped {int((~keep).sum())}/{xy.shape[0]} fragments "
              f"(hue p90-spread > {args.merge_hue_spread:g})")
        xy, vis, finite = xy[keep], vis[keep], finite[keep]
        xyz = xyz[keep] if xyz is not None else None
    Q = xy.shape[0]
    colors = track_mean_colors(xy, vis, image_paths, T) if args.color_thresh > 0 else None
    candidates, _ = find_candidates(xy, vis, finite, T, args.window, args.merge_px, args.min_overlap,
                                    args.topk, track_colors=colors, color_thresh=args.color_thresh,
                                    strategy=strategy)
    final, accepted = assign_chains(candidates, Q)
    heads = np.unique(final)
    n = heads.size
    xy_c = np.full((n, T, 2), np.nan); xyz_c = np.full((n, T, 3), np.nan)
    occ_c = np.ones((n, T), bool); qt_c = np.zeros(n, np.int64); src_c = np.full((n, T), -1, np.int64)
    members = defaultdict(list)
    for q in range(Q):
        members[int(final[q])].append(q)
    for i, h in enumerate(heads.tolist()):
        mem = np.asarray(members[h], int)
        vm = vis[mem]; fm = finite[mem]
        fa = np.where(vm.any(0))[0]                                     # VISIBLE (real obs) frames
        if fa.size:
            sel = mem[vm.argmax(0)[fa]]
            xy_c[i, fa] = xy[sel, fa]; occ_c[i, fa] = False; src_c[i, fa] = sel
            if xyz is not None:
                xyz_c[i, fa] = xyz[sel, fa]
            qt_c[i] = int(fa[0])
        fp = np.where(fm.any(0) & ~vm.any(0))[0]                        # occluded-but-PREDICTED-only frames
        if fp.size:
            selp = mem[fm[:, fp].argmax(0)]
            xy_c[i, fp] = xy[selp, fp]; src_c[i, fp] = selp             # occ_c stays True (no obs)
    obs = (~occ_c).sum(1)
    print(f"[{tag}] {Q} -> {n} tracks ({len(accepted)} 1-to-1 merges; "
          f"median obs {int(np.median(obs))}, max {int(obs.max())})")
    return xy_c, xyz_c, occ_c, qt_c, src_c


def save_merged(out_path, xy, xyz, occ, qt, member_src, n_forward=-1):
    """Drop-in tracks npz (same schema as tracks_dynamic_only.npz) from consolidated arrays."""
    np.savez_compressed(out_path,
                        tracks_xy=xy[None].astype(np.float32), tracks_xyz=xyz[None].astype(np.float64),
                        occluded=occ[None], dynamic_obs_mask=(~occ),
                        query_t=qt.astype(np.int32), member_src=member_src.astype(np.int32),
                        n_forward=np.int64(n_forward),
                        video_dir=str(utils.definitions.video_dir), mask_dir=str(utils.definitions.mask_dir))
    obs = (~occ).sum(1)
    print(f"[save] {out_path}: {xy.shape[0]} tracks (median obs {int(np.median(obs))} max {int(obs.max())})")




# --- legacy hue-quality helpers (moved here from visualize_tracks_2d.py, which replaced them with
# the median-reference filter; this merge path keeps the ORIGINAL behaviour bit-for-bit) ---
def _sample_track_hue(xy, occ, image_paths, T, patch=2):
    """Sample each track's hue/saturation at ITS OWN visible 2D positions (patch-median BGR -> HSV).
    Input = the 2D track + the frame images only. Returns (hue[Q,T], sat[Q,T]) with NaN where unsampled."""
    Q = xy.shape[0]
    vis = (~occ) & np.isfinite(xy).all(-1)
    hue = np.full((Q, T), np.nan, np.float32)
    sat = np.full((Q, T), np.nan, np.float32)
    p = int(patch)
    for t in range(min(T, len(image_paths))):
        idx = np.where(vis[:, t])[0]
        if idx.size == 0:
            continue
        img = cv2.imread(image_paths[t])
        if img is None:
            continue
        H, W = img.shape[:2]
        bgr = np.zeros((idx.size, 3), np.uint8)
        for r, q in enumerate(idx.tolist()):                          # patch-median BGR at the track's pixel
            x = int(np.clip(round(xy[q, t, 0]), 0, W - 1)); y = int(np.clip(round(xy[q, t, 1]), 0, H - 1))
            win = img[max(y - p, 0):y + p + 1, max(x - p, 0):x + p + 1].reshape(-1, 3)
            bgr[r] = np.median(win, axis=0)
        hsv = cv2.cvtColor(bgr[None], cv2.COLOR_BGR2HSV)[0]
        hue[idx, t] = hsv[:, 0]; sat[idx, t] = hsv[:, 1]
    return hue, sat


def _score_hue(hue, sat, thresh, pct=90, sat_min=40, min_obs=5):
    """GOOD/not-good from the sampled colours: saturation-gate (specular/white pixels have unstable hue),
    reference = the track's circular-median hue, score = p<pct> circular deviation over its observations.
    GOOD = score <= thresh (deg, 0..90). A track with enough samples but too few SATURATED ones lives on a
    COLOURLESS surface -- that is consistent appearance (no evidence of colour drift), so it is ACCEPTED as
    good. Only tracks with fewer than min_obs samples overall stay UNSCORED. -> (good, scored)."""
    from edg_viz_common import _circ_hue_median
    Q = hue.shape[0]
    good = np.zeros(Q, bool); scored = np.zeros(Q, bool)
    for q in range(Q):
        fin = np.isfinite(hue[q])
        m = fin & (np.nan_to_num(sat[q], nan=-1.0) >= sat_min)
        if int(m.sum()) >= int(min_obs):                              # enough saturated samples: judge by hue
            ref = _circ_hue_median(hue[q, m])
            d = np.abs(hue[q, m] - ref); d = np.minimum(d, 180.0 - d)  # circular dist, deg (0..90)
            scored[q] = True
            good[q] = float(np.percentile(d, pct)) <= float(thresh)
        elif int(fin.sum()) >= int(min_obs):                          # colourless surface: ACCEPT (no drift
            scored[q] = True                                          # evidence; hue can't testify either way)
            good[q] = True
    return good, scored


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", type=int, default=10, help="search A's last `window` visible frames for a continuation")
    ap.add_argument("--merge-px", type=float, default=10.0, help="2D distance (px) to accept a continuation")
    ap.add_argument("--min-overlap", type=int, default=3, help="min co-visible (overlap) frames in the window to link")
    ap.add_argument("--topk", type=int, default=3, help="candidate continuations kept per onset (for fallback)")
    ap.add_argument("--color-thresh", type=float, default=30.0,
                    help="appearance criterion: reject a merge candidate whose MEAN track colour differs "
                         "from the track's by more than this (BGR euclidean, 0-255); 0 disables")
    ap.add_argument("--merge-hue-spread", type=float, default=40.0,
                    help="spread guard: DROP (before merging) any fragment whose own circular HUE spread "
                         "(p90 angular dist to its mean hue, 0..90 scale) exceeds this -- removes fragments "
                         "that sweep across colours (drift/specular). Tradeoff: ~30%% of fragments on the "
                         "paper-windmill drop at 40; raise to keep more, 0 disables")
    ap.add_argument("--clean-move-sigma", type=float, default=2.0,
                    help="post-merge cleanup: delete an observation whose 2D interpolation residual is a "
                         "robust outlier (median + this*1.4826*MAD, pooled); 0 disables the 2D-jump filter")
    ap.add_argument("--clean-move-px", type=float, default=0.0,
                    help="post-merge cleanup: also delete any observation whose 2D interpolation residual "
                         "exceeds this many px (absolute cap; 0 = off)")
    ap.add_argument("--clean-color-sigma", type=float, default=2.0,
                    help="post-merge cleanup: delete an observation whose BGR colour distance to the track's "
                         "dominant (median) colour is a PER-TRACK robust outlier (> track median + this*"
                         "1.4826*track MAD); 0 disables the colour filter")
    ap.add_argument("--clean-color-min", type=float, default=0.0,
                    help="post-merge cleanup: absolute floor (BGR px) below which a colour outlier is kept "
                         "(guards ultra-uniform tracks with a tiny MAD); 0 = no floor")
    ap.add_argument("--clean-fill-gap", type=int, default=10,
                    help="post-merge cleanup: linearly interpolate ANY interior discontinuity (deletion- or "
                         "occlusion-opened) spanning fewer than this many frames (bounded by surviving obs "
                         "on both sides); 0 disables")
    ap.add_argument("--good-thresh", type=float, default=10.0,
                    help="FINAL colour filter (same as visualize_tracks_2d): a track is GOOD when its own "
                         "sat-gated p90 circular hue spread <= this many deg (colourless surfaces accepted); "
                         "NOT-GOOD tracks are broken into their member_src fragments and re-filtered, still-"
                         "failing fragments are dropped. 0 disables")
    ap.add_argument("--forward-npz", default="tracks_dynamic_only.npz", help="forward tracking npz (in OUT_DIR)")
    ap.add_argument("--backward-npz", default="tracks_dynamic_backward.npz",
                    help="backward tracking npz (dynamic_tracking.py --reverse); if present, TWO-STAGE merge")
    ap.add_argument("--n-longest", type=int, default=120, help="how many longest merged tracks to show along the sequence")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--headless", action="store_true", help="compute + save only, no window")
    args = ap.parse_args()

    out = Path(utils.definitions.output_dir)
    image_paths = sorted(glob.glob(str(Path(utils.definitions.video_dir) / utils.definitions.image_pattern)))

    # STAGE 1: merge each direction INDIVIDUALLY.
    xy_f, vis_f, fin_f, xyz_f = _load_npz_tracks(out / args.forward_npz)
    T = xy_f.shape[1]
    print(f"[load] forward {xy_f.shape[0]} tracks, T={T}")
    xyF, xyzF, occF, qtF, srcF = consolidate(xy_f, vis_f, fin_f, xyz_f, T, image_paths, args, "fwd")

    bpath = out / args.backward_npz
    if bpath.exists():
        xy_b, vis_b, fin_b, xyz_b = _load_npz_tracks(bpath)
        print(f"[load] backward {xy_b.shape[0]} tracks")
        # Backward tracks were tracked last->first then flipped to forward frame order, so a track's tracker
        # predictions live BEFORE its first-visible frame. Run the SAME after-invisible merge in REVERSED
        # time (flip time axis + image order) so the logic is identical, then flip the result back.
        rev = slice(None, None, -1)
        xyB_r, xyzB_r, occB_r, _, _ = consolidate(
            xy_b[:, rev], vis_b[:, rev], fin_b[:, rev], (xyz_b[:, rev] if xyz_b is not None else None),
            T, image_paths[::-1], args, "bwd")
        xyB = xyB_r[:, rev].copy(); xyzB = xyzB_r[:, rev].copy(); occB = occB_r[:, rev].copy()
        # STAGE 2: merge the two DIRECTIONS -- stack both consolidated sets (same original frame indexing)
        # and merge across them. Consolidated chains no longer carry a single tracker's post-occlusion
        # prediction, so use the BEFORE-invisible overlap window instead of the after-invisible prediction.
        nF = xyF.shape[0]
        xy_all = np.concatenate([xyF, xyB], 0); xyz_all = np.concatenate([xyzF, xyzB], 0)
        occ_all = np.concatenate([occF, occB], 0)
        fin_all = np.isfinite(xy_all).all(-1)
        vis_all = (~occ_all) & fin_all
        xyC, xyzC, occC, qtC, srcC = consolidate(xy_all, vis_all, fin_all, xyz_all, T, image_paths, args,
                                                 "cross F+B", strategy="before")
        # member_src (final) = which combined-intermediate chain fed each frame: [0,nF)=a FORWARD chain,
        # [nF, nF+nB)=a BACKWARD chain -> the coloured viewer shows the forward<->backward handovers.
        xy_final, xyz_final, occ_final, qt_final, member_src = xyC, xyzC, occC, qtC, srcC
        n_fwd_chains = nF
    else:
        print(f"[merge] no backward file ({bpath.name}); forward-only (run dynamic_tracking.py --reverse first)")
        xy_final, xyz_final, occ_final, qt_final, member_src = xyF, xyzF, occF, qtF, srcF
        n_fwd_chains = xyF.shape[0]

    # POST-MERGE CLEANUP: drop per-observation outliers (2D jumps + colour drift) on the final tracks.
    if args.clean_move_sigma > 0 or args.clean_move_px > 0 or args.clean_color_sigma > 0:
        xy_final, xyz_final, occ_final, dele = clean_tracks(
            xy_final, xyz_final, occ_final, image_paths, T,
            args.clean_move_sigma, args.clean_move_px, args.clean_color_sigma, args.clean_color_min,
            args.clean_fill_gap)
        member_src[dele] = -1
        vfin = (~occ_final) & np.isfinite(xy_final).all(-1)          # first surviving obs = new query_t
        qt_final = np.where(vfin.any(1), vfin.argmax(1), 0).astype(np.int64)

    # FINAL COLOUR FILTER (same test as visualize_tracks_2d): judge each final track by its own 2D hue
    # consistency; BREAK a not-good track into its member_src fragments, re-filter those, DROP fragments
    # that still fail (colour-drifting tracking errors), keep the rest as separate tracks.
    if args.good_thresh > 0:
        hue, sat = _sample_track_hue(xy_final, occ_final, image_paths, T)
        good, scored = _score_hue(hue, sat, args.good_thresh)
        notgood = scored & ~good
        print(f"[good] final colour filter: {int(good.sum())} GOOD | {int(notgood.sum())} not-good (break) "
              f"| {xy_final.shape[0] - int(scored.sum())} unscored (kept)  (hue p90 <= {args.good_thresh:g} deg)")
        if notgood.any():
            frag = []                                                # (row, member_id, frame_mask)
            for q in np.where(notgood)[0].tolist():
                for m in np.unique(member_src[q][member_src[q] >= 0]).tolist():
                    fm = member_src[q] == m
                    if fm.any():
                        frag.append((q, int(m), fm))
            keep = ~notgood
            nk, nf = int(keep.sum()), len(frag)
            xy2 = np.concatenate([xy_final[keep], np.full((nf, T, 2), np.nan)])
            xyz2 = np.concatenate([xyz_final[keep], np.full((nf, T, 3), np.nan)])
            occ2 = np.concatenate([occ_final[keep], np.ones((nf, T), bool)])
            ms2 = np.concatenate([member_src[keep], np.full((nf, T), -1, member_src.dtype)])
            hue_f = np.full((nf, T), np.nan, np.float32); sat_f = np.full((nf, T), np.nan, np.float32)
            for i, (q, m, fm) in enumerate(frag):                    # fragment = the parent's frames from m
                r = nk + i
                xy2[r, fm] = xy_final[q, fm]; xyz2[r, fm] = xyz_final[q, fm]
                occ2[r, fm] = occ_final[q, fm]; ms2[r, fm] = m
                hue_f[i, fm] = hue[q, fm]; sat_f[i, fm] = sat[q, fm]  # reuse the sampled colours
            gf, sf = _score_hue(hue_f, sat_f, args.good_thresh)      # filter the fragments AGAIN
            keep_f = ~(sf & ~gf)                                     # drop only still-failing fragments
            sel = np.concatenate([np.arange(nk), nk + np.where(keep_f)[0]])
            xy_final, xyz_final, occ_final, member_src = xy2[sel], xyz2[sel], occ2[sel], ms2[sel]
            vfin = (~occ_final) & np.isfinite(xy_final).all(-1)
            qt_final = np.where(vfin.any(1), vfin.argmax(1), 0).astype(np.int64)
            print(f"[good] broke {int(notgood.sum())} not-good tracks into {nf} fragments: "
                  f"{int(gf.sum())} good + {int((~sf).sum())} unscored kept, "
                  f"{int((sf & ~gf).sum())} still-failing dropped -> {xy_final.shape[0]} tracks")

    save_merged(str(out / "tracks_dynamic_merged.npz"), xy_final, xyz_final, occ_final, qt_final,
                member_src, n_forward=n_fwd_chains)

    # visualizations on the FINAL consolidated tracks (each row is already one track)
    vis_final = (~occ_final) & np.isfinite(xy_final).all(-1)
    n = xy_final.shape[0]
    draw_intervals(np.arange(n), vis_final, T, str(out / "extend_tracks_intervals.png"), only_merged=False)
    tracks = {i: (xy_final[i], vis_final[i].astype(np.int8)) for i in range(n)}
    draw_longest_trajectories(tracks, T, image_paths,
                              str(out / f"extend_tracks_longest{args.n_longest}.png"), n=args.n_longest)
    if not args.headless:
        visualize_longest(tracks, T, image_paths, n=args.n_longest, start=args.start)


if __name__ == "__main__":
    main()
