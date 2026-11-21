#!/usr/bin/env python3
"""2D inspector for the RAW dynamic tracking output (outputs/tracks_dynamic_only.npz from
dynamic_tracking.py, or tracks_dynamic_merged.npz with --merged):

  SPACE / B : next / previous frame
  N / P     : next / previous RANK (tracks ranked by visible-frame count; 1 = longest)
  A         : show ALL tracks
  Q / ESC   : quit

Single-track mode draws the track's position (filled = visible, white hollow = tracker-occluded, grey =
off-dynamic-mask) plus a TRAIL of its last --trail frames, so tracking jumps/drift are directly visible.
With --merged, each ORIGINAL track that was merged into the chain gets its OWN colour (point + trail
segments), so you see exactly where one fragment hands over to the next (per-frame source ids come from
member_src saved by test_extend_tracks.py). All-tracks mode draws every visible track as a green dot.

With --merged the tracks are ALSO judged good/not-good from the 2D TRACK ALONE (input = the merged 2D
track + the frame images; nothing solved, no reprojection): each track's hue is sampled at its own visible
positions (patch-median, saturation-gated) and the track is GOOD when its p90 circular hue spread stays
<= --good-thresh; a track living on a COLOURLESS (low-saturation) surface is ACCEPTED as good (consistent
appearance = no drift evidence). The marker SHAPE encodes the verdict -- SQUARE = good, TRIANGLE =
not-good, circle = unscored (track too short to judge). --no-good skips the scoring.

Run:  XM_DATASET_NAME=<scene> conda run -n XM python visualize_tracks_2d.py [--merged] [--track-rank 1]
"""
import argparse
import glob
from pathlib import Path

import cv2
import numpy as np

import utils.definitions


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


def _marker(img, pt, shape, r, color, filled):
    """Draw a track marker: square = edg-good, triangle = not-good, circle = unscored."""
    if shape == "square":
        cv2.rectangle(img, (pt[0] - r, pt[1] - r), (pt[0] + r, pt[1] + r), color,
                      -1 if filled else 1, cv2.LINE_AA)
    elif shape == "triangle":
        tri = np.array([[pt[0], pt[1] - r - 1], [pt[0] - r - 1, pt[1] + r], [pt[0] + r + 1, pt[1] + r]],
                       np.int32)
        if filled:
            cv2.fillPoly(img, [tri], color, cv2.LINE_AA)
        else:
            cv2.polylines(img, [tri], True, color, 1, cv2.LINE_AA)
    else:
        cv2.circle(img, pt, r, color, -1 if filled else 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged", action="store_true", help="inspect tracks_dynamic_merged.npz instead")
    ap.add_argument("--track-rank", type=int, default=1,
                    help="starting rank (1 = longest by visible-frame count, 0 = all tracks); "
                         "change interactively with N/P/A")
    ap.add_argument("--trail", type=int, default=15, help="trail length (frames) in single-track mode")
    ap.add_argument("--frame", type=int, default=None, help="starting frame (default: the track's birth)")
    ap.add_argument("--scale", type=float, default=2.0, help="display upscale")
    ap.add_argument("--dry-run", action="store_true", help="write one PNG instead of opening a window")
    ap.add_argument("--no-good", action="store_true",
                    help="(--merged) skip the 2D-colour good/not-good scoring (all markers stay circles)")
    ap.add_argument("--good-thresh", type=float, default=30.0,
                    help="(--merged) GOOD when the track's own p90 circular hue spread (sampled at its 2D "
                         "positions, sat-gated) is <= this many deg (0..90 scale)")
    ap.add_argument("--select-edg", type=int, default=0,
                    help="after all filtering, select the top-N EDG nodes from the GOOD tracks with the "
                         "SAME method as build_edg.py (select_edg_nodes_2d length-NMS) and show only them "
                         "(e.g. 20). 0 = off")
    args = ap.parse_args()

    out_dir = Path(utils.definitions.output_dir)
    npz_path = out_dir / ("tracks_dynamic_merged.npz" if args.merged else "tracks_dynamic_only.npz")
    d = np.load(str(npz_path), allow_pickle=True)
    xy = np.asarray(d["tracks_xy"], np.float64)[0]                    # (Q,T,2) full-res px
    occ = np.asarray(d["occluded"])[0].astype(bool)                   # (Q,T) final occlusion (all filters)
    dyn_mask = np.asarray(d["dynamic_obs_mask"]).astype(bool)         # (Q,T)
    query_t = np.asarray(d["query_t"], np.int64).reshape(-1)
    # merged file: per-frame ORIGINAL track id that supplied the observation -> colour fragments differently
    msrc = np.asarray(d["member_src"], np.int32) if "member_src" in d.files else None
    if args.merged and msrc is None:
        print("[tracks2d] note: no member_src in the merged npz (older file); re-run test_extend_tracks.py "
              "to enable per-fragment colours")
    PALETTE = [(60, 220, 60), (60, 130, 255), (200, 80, 255), (60, 220, 220), (255, 200, 60),
               (0, 165, 255), (255, 80, 160), (120, 255, 120), (255, 120, 60), (160, 220, 0)]
    Q, T = xy.shape[0], xy.shape[1]
    image_paths = sorted(glob.glob(str(Path(utils.definitions.video_dir) / utils.definitions.image_pattern)))
    print(f"[tracks2d] {npz_path.name}: Q={Q} T={T} | {len(image_paths)} images")
    label = [str(q) for q in range(Q)]                                # display id (fragments get parent/frag)

    # 2D-colour good/not-good scoring (merged only; input = the 2D track + images, nothing solved):
    # SQUARE = good, TRIANGLE = not-good, circle = unscored
    good = scored = None
    if args.merged and not args.no_good:
        hue, sat = _sample_track_hue(xy, occ, image_paths, T)
        good, scored = _score_hue(hue, sat, args.good_thresh)
        print(f"[good] pass 1 (merged): {int(good.sum())} GOOD | {int((scored & ~good).sum())} not-good "
              f"| {Q - int(scored.sum())} unscored  (2D hue p90-spread <= {args.good_thresh:g} deg)")

        # BREAK each NOT-GOOD merged track into its ORIGINAL (un-merged) fragments via member_src, and
        # filter those again. Fragment positions == the parent's at its frames -> reuse the sampled hue.
        if msrc is not None and (scored & ~good).any():
            frag = []                                                 # (parent_q, member_id, frame_mask)
            for q in np.where(scored & ~good)[0].tolist():
                for m in np.unique(msrc[q][msrc[q] >= 0]).tolist():
                    fm = msrc[q] == m
                    if fm.any():
                        frag.append((q, int(m), fm))
            keep = ~(scored & ~good)                                  # good + unscored stay as-is
            nk, nf = int(keep.sum()), len(frag)
            xy2 = np.concatenate([xy[keep], np.full((nf, T, 2), np.nan)])
            occ2 = np.concatenate([occ[keep], np.ones((nf, T), bool)])
            dyn2 = np.concatenate([dyn_mask[keep], np.zeros((nf, T), bool)])
            ms2 = np.concatenate([msrc[keep], np.full((nf, T), -1, msrc.dtype)])
            hue2 = np.concatenate([hue[keep], np.full((nf, T), np.nan, np.float32)])
            sat2 = np.concatenate([sat[keep], np.full((nf, T), np.nan, np.float32)])
            label = [label[q] for q in np.where(keep)[0].tolist()]
            for i, (q, m, fm) in enumerate(frag):
                r = nk + i
                xy2[r, fm] = xy[q, fm]; occ2[r, fm] = occ[q, fm]; dyn2[r, fm] = dyn_mask[q, fm]
                ms2[r, fm] = m; hue2[r, fm] = hue[q, fm]; sat2[r, fm] = sat[q, fm]
                label.append(f"{q}/{m}")
            gf, sf = _score_hue(hue2[nk:], sat2[nk:], args.good_thresh)   # filter the fragments AGAIN
            good = np.concatenate([good[keep], gf]); scored = np.concatenate([scored[keep], sf])
            xy, occ, dyn_mask, msrc = xy2, occ2, dyn2, ms2
            query_t = np.where(((~occ) & np.isfinite(xy).all(-1)).any(1),
                               ((~occ) & np.isfinite(xy).all(-1)).argmax(1), 0).astype(np.int64)
            Q = xy.shape[0]
            print(f"[good] broke {int((~keep).sum())} not-good merged tracks into {nf} fragments -> "
                  f"pass 2: {int(gf.sum())} GOOD | {int((sf & ~gf).sum())} not-good | "
                  f"{nf - int(sf.sum())} unscored  (total Q={Q})")

    vis_count = (~occ).sum(1)                                         # visible frames per track
    order = np.argsort(-vis_count)                                    # rank 1 = most visible frames
    show_ids = None                                                   # all-tracks mode: which tracks to draw

    # --select-edg N: after all filtering, pick the top-N EDG nodes from the GOOD tracks with the SAME
    # method as build_edg.py (select_edg_nodes_2d: 2D p90-persistent-distance length-priority NMS) and
    # restrict the visualization to them.
    if args.select_edg > 0:
        if good is None:
            print("[select-edg] needs the good/not-good scoring (use --merged without --no-good); skipping")
        else:
            from build_edg import select_edg_nodes_2d, _dyn_region_near_px
            vis_now = (~occ) & np.isfinite(xy).all(-1)
            Vs = np.ascontiguousarray(vis_now.T)                      # (T,Q), build_edg's layout
            Vs[:, ~good] = False                                      # only GOOD tracks are eligible
            near_px = _dyn_region_near_px({"image_paths": image_paths, "T": T}, 0.16)
            nodes = select_edg_nodes_2d(xy, Vs, Vs.sum(0), near_px=near_px, overlap_frac=0.5,
                                        max_nodes=int(args.select_edg), min_node_frames=20)
            show_ids = np.asarray(nodes, np.int64)
            order = show_ids[np.argsort(-vis_count[show_ids])]        # rank over the selected nodes only
            print(f"[select-edg] {show_ids.size} EDG nodes selected from the GOOD tracks "
                  f"(build_edg 2D length-NMS): " + ", ".join(label[g] for g in order.tolist()))

    print(f"[tracks2d] Q={Q} | visible-frames rank1={int(vis_count[order[0]])} "
          f"median={int(np.median(vis_count))}")

    def shape_of(q):
        if good is None:
            return "circle"
        return "square" if good[q] else ("triangle" if scored[q] else "circle")

    def render(f, rank):
        img = cv2.imread(image_paths[f]) if f < len(image_paths) else None
        if img is None:
            img = np.zeros((480, 360, 3), np.uint8)
        H, W = img.shape[:2]

        def draw_point(q, radius=3, vis_color=(60, 220, 60)):
            p = xy[q, f]
            if not np.isfinite(p).all():
                return "no-pred"
            pt = (int(round(p[0])), int(round(p[1])))
            shp = shape_of(q)                                                     # square/triangle/circle
            if not occ[q, f]:
                _marker(img, pt, shp, radius, vis_color, filled=True)             # visible: filled
                return "visible"
            elif dyn_mask[q, f]:
                _marker(img, pt, shp, radius + 1, (255, 255, 255), filled=False)  # occluded: white hollow
                return "occluded"
            else:
                _marker(img, pt, shp, radius, (160, 160, 160), filled=False)      # off-mask: grey hollow
                return "off-mask"

        if rank <= 0:
            n_vis = 0
            iter_ids = show_ids.tolist() if show_ids is not None else range(Q)
            for q in iter_ids:
                if f >= query_t[q] and np.isfinite(xy[q, f]).all() and not occ[q, f]:
                    p = xy[q, f]
                    _marker(img, (int(round(p[0])), int(round(p[1]))), shape_of(q), 2, (60, 220, 60), True)
                    n_vis += 1
            txt = (f"selected EDG nodes ({show_ids.size})  visible={n_vis}" if show_ids is not None
                   else f"all tracks  visible={n_vis}")
        else:
            q = int(order[rank - 1])
            col_of = {}                                               # original-track id -> colour (merged
            if msrc is not None:                                      # file: each FRAGMENT gets its own)
                for m in msrc[q][msrc[q] >= 0].tolist():
                    if m not in col_of:
                        col_of[m] = PALETTE[len(col_of) % len(PALETTE)]
            f0 = max(int(query_t[q]), f - int(args.trail))            # trail of the last N frames
            prev = None
            for tt in range(f0, f + 1):
                p = xy[q, tt]
                if not np.isfinite(p).all():
                    prev = None
                    continue
                cur_pt = (int(round(p[0])), int(round(p[1])))
                if prev is not None:
                    c = (0, 220, 220)                                 # default yellow trail
                    if msrc is not None and msrc[q, tt] >= 0:
                        c = col_of.get(int(msrc[q, tt]), c)           # fragment-coloured trail segment
                    cv2.line(img, prev, cur_pt, c, 1, cv2.LINE_AA)
                prev = cur_pt
            vcol, src_txt = (60, 220, 60), ""
            if msrc is not None:
                cur = int(msrc[q, f])
                if cur >= 0:
                    vcol = col_of.get(cur, vcol)
                src_txt = f" frags={len(col_of)} src={cur if cur >= 0 else '-'}"
            state = draw_point(q, radius=4, vis_color=vcol) if f >= query_t[q] else "pre-birth"
            gtxt = "" if good is None else \
                {"square": " GOOD", "triangle": " NOT-GOOD", "circle": " unscored"}[shape_of(q)]
            txt = (f"rank {rank}/{order.size}: gid={label[q]} vis={int(vis_count[q])}f "
                   f"birth={int(query_t[q])} [{state}]{gtxt}{src_txt}")

        s = float(args.scale)
        if s != 1.0:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        cv2.putText(img, f"frame {f}/{T - 1}  {txt}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        print(f"[tracks2d] frame={f} {txt}")
        return img

    rank = int(np.clip(args.track_rank, 0, order.size))
    if args.frame is not None:
        f = int(np.clip(args.frame, 0, T - 1))
    else:
        f = int(query_t[order[rank - 1]]) if rank > 0 else 0          # start at the track's birth
    if args.dry_run:
        out = out_dir / f"tracks2d_frame{f:04d}.png"
        cv2.imwrite(str(out), render(f, rank))
        print(f"[tracks2d][dry-run] wrote {out}")
        return
    print("[tracks2d] SPACE=next frame, B=prev | N=next rank, P=prev rank, A=all | Q/ESC=quit. "
          "green=visible, white hollow=tracker-occluded, grey=off-mask, yellow=trail"
          + ("" if good is None else " | SQUARE=good(2D colour), TRIANGLE=not-good, circle=unscored"))
    while True:
        cv2.imshow("dynamic tracks 2D", render(f, rank))
        k = cv2.waitKey(0) & 0xFF
        if k in (ord("q"), ord("Q"), 27):
            break
        if k in (ord("b"), ord("B")):
            f = (f - 1) % T
        elif k in (ord("n"), ord("N")):
            rank = min(rank + 1, order.size)
            f = int(query_t[order[rank - 1]])                          # ALWAYS jump to the track's birth
        elif k in (ord("p"), ord("P")):
            rank = max(rank - 1, 0)
            if rank > 0:
                f = int(query_t[order[rank - 1]])                      # ALWAYS jump to the track's birth
        elif k in (ord("a"), ord("A")):
            rank = 0
        else:
            f = (f + 1) % T
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
