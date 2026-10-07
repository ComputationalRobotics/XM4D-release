#!/usr/bin/env python3
"""2D inspector for the dynamic tracks at any stage of the pipeline (--source):
  raw       outputs/tracks_dynamic_only.npz            raw online TAPNext tracks (dynamic_tracking.py)
  cow       outputs/cow_skeleton.npz                    CoWTracker skeleton tracks (cow_skeleton.py), gated vis>--cow-vis & conf>--cow-conf
  merged    outputs/tracks_dynamic_merged.npz           averaged TAPNet tracks anchored to the skeleton (cow_attach_tracks.py), 2D only
  smoothed  outputs/tracks_dynamic_merged_smoothed.npz  = merged + depth lift/smoothing (optimize_dynamic_tracks.py): the FINAL
                                                        tracks build_edg.py / EDG_test_fast.py read (occlusion also carries the
                                                        depth-boundary and depth-discontinuity drops)   <- default
(--merged is kept as an alias for --source merged.) Same interaction pattern as visualize_edg_filtered_open3d.py --viz-2d:

  SPACE / B : next / previous frame
  N / P     : next / previous RANK (tracks ranked by visible-frame count; 1 = longest)
  A         : show ALL tracks (the default view with --select-edg: all selected EDG nodes at once)
  Q / ESC   : quit

Single-track mode draws the track's position (filled = visible, white hollow = tracker-occluded, grey =
off-dynamic-mask) plus a TRAIL of its last --trail frames, so tracking jumps/drift are directly visible.

LARGE-2D-JUMP DETECTION (on by default, --no-jumps to disable): the track's COLOUR CHANGES AT EVERY JUMP --
the segment before the first jump keeps the normal colour, then each further jump advances to the next colour
of the jump palette (red -> orange -> yellow -> magenta -> cyan -> white -> ...), point and trail together,
with a yellow ring on the trigger frame itself. --jump-mode first keeps one colour (red) after the first jump. Two triggers: (a) STEP, ‖x(t)-x(t-1)‖ > --jump-px between consecutive VISIBLE frames;
(b) GAP, a reappearance after invisible frames whose position deviates more than --gap-tol-px from the
position predicted by a 2D similarity fit on the OTHER points visible on both sides of the gap (so camera
motion and rigid rotation do not count as jumps). Windmill reference: normal skeleton motion is ~4 px/frame
(p99 = 16), gap reappearances jump a median of 29 px.
With --merged, each ORIGINAL track that was merged into the chain gets its OWN colour (point + trail
segments), so you see exactly where one fragment hands over to the next (per-frame source ids come from
member_src saved by test_extend_tracks.py). All-tracks mode draws every visible track as a green dot.


--with-cow (merged/smoothed only) ALSO draws each track's PARENT CoWTracker skeleton point in orange -- trail,
point (filled = gated-in, hollow = gated-out) and a line joining it to the merged point -- so the offset the
attach/average introduced is visible per frame, and the header reports it as cow-offset=<px>. The pairing is
exact: cow_attach_tracks.py stores each kept track's skeleton_index in the merged npz.

Run:  XM_DATASET_NAME=<scene> conda run -n XM python visualize_tracks_2d.py [--source raw|cow|merged|smoothed]
                                                                  [--track-rank 1] [--jump-px 20] [--no-jumps]
"""
import argparse
import glob
from pathlib import Path

import cv2
import numpy as np

import utils.definitions


def _marker(img, pt, shape, r, color, filled):
    """Draw a track marker (shape kept for callers; all tracks now draw as circles)."""
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
    ap.add_argument("--source", choices=["raw", "cow", "cotracker", "merged", "smoothed"], default=None,
                    help="which tracks to show (default: smoothed = the final tracks sent to build_edg; see module doc)")
    ap.add_argument("--merged", action="store_true", help="alias for --source merged")
    ap.add_argument("--npz", type=Path, default=None,
                    help="explicit tracks npz, any stage (e.g. tracks_dynamic_merged_jfilt.npz or cow_skeleton_single_s3.npz); "
                         "the format (skeleton vs merged-like) is detected from its keys")
    ap.add_argument("--cow-filters", action="store_true",
                    help="skeleton only: also apply cow_attach_tracks.py's three filters (jumpy-track deletion, colour vs "
                         "first frame, coverage) so you see exactly what the attachment consumes")
    from utils import dataset_config
    ap.add_argument("--cow-vis", type=float, default=dataset_config.get_float("COW_GATE_VIS", 0.5)); ap.add_argument("--cow-conf", type=float, default=dataset_config.get_float("COW_GATE_CONF", 0.5))
    ap.add_argument("--track-rank", type=int, default=None,
                    help="starting rank (1 = longest by visible-frame count, 0 = all tracks); change interactively with "
                         "N/P/A. Default: 0 (ALL at once) when --select-edg is used, else 1")
    ap.add_argument("--trail", type=int, default=15, help="trail length (frames) in single-track mode")
    ap.add_argument("--frame", type=int, default=None, help="starting frame (default: the track's birth)")
    ap.add_argument("--scale", type=float, default=2.0, help="display upscale")
    ap.add_argument("--dry-run", action="store_true", help="write one PNG instead of opening a window")
    ap.add_argument("--jumps", action=argparse.BooleanOptionalAction, default=True,
                    help="detect large 2D jumps and change the track's colour at every jump (default on)")
    ap.add_argument("--jump-mode", choices=["segment", "first"], default="segment",
                    help="segment = a new colour after EVERY jump (default); first = red after the first jump")
    ap.add_argument("--jump-px", type=float, default=20.0,
                    help="jump trigger (a): step between two consecutive VISIBLE frames, px")
    ap.add_argument("--gap-tol-px", type=float, default=60.0,
                    help="jump trigger (b): deviation from the similarity-predicted position at a reappearance, px")
    ap.add_argument("--jump-color", type=int, nargs=3, default=(0, 0, 255), metavar=("B", "G", "R"),
                    help="colour used by --jump-mode first (and the first palette entry in segment mode)")
    ap.add_argument("--color-by", choices=["jump", "id"], default=None,
                    help="all-tracks view: 'id' gives every track/node its OWN colour (so individual nodes can be told "
                         "apart; jumps still show as the yellow ring), 'jump' colours by jump state. "
                         "Default: id with --select-edg, else jump")
    ap.add_argument("--with-cotracker", nargs="?", const="", default=None, metavar="NPZ",
                    help="with --source merged/smoothed: ALSO draw the CoTracker tracks (cyan) as context. "
                         "Unlike --with-cow there is NO per-track pairing -- CoTracker tracks are independent "
                         "of the skeleton -- so every visible CoTracker point is drawn. Optional path, else "
                         "outputs/tracks_cotracker_offline.npz.")
    ap.add_argument("--with-cow", action="store_true",
                    help="with --source merged/smoothed: ALSO draw each track's parent CoWTracker skeleton "
                         "point (orange), joined to the merged point by a line, so the attach/average offset "
                         "is visible. Needs skeleton_index in the merged npz (written by cow_attach_tracks.py).")
    ap.add_argument("--edg-near-frac", type=float, default=None,
                    help="ADAPTIVE node radius fraction for --select-edg: near_px = frac * sqrt(median "
                         "dynamic-mask area). Default: EDG_NEAR_FRAC from the dataset config (what the "
                         "pipeline actually builds); pass e.g. 0.16 for a sparser, easier-to-read set.")
    ap.add_argument("--edg-near-px", type=float, default=0.0,
                    help="fixed node radius in px, overriding --edg-near-frac when > 0")
    ap.add_argument("--edg-edges", action=argparse.BooleanOptionalAction, default=True,
                    help="with --select-edg: for every pair of selected nodes that is CLOSE AT THE CURRENT "
                         "FRAME, draw a line labelled 'now|p90' -- the distance you see vs the persistent "
                         "p90 distance the NMS actually tests against near_px. That pair survived because "
                         "p90 > near_px (or the overlap test failed), which the two numbers make explicit. "
                         "Red line = p90 <= near_px (should not happen), white = p90 > near_px.")
    ap.add_argument("--edg-pair-px", type=float, default=0.0,
                    help="only label pairs closer than this at the current frame (0 = use near_px)")
    ap.add_argument("--select-edg", type=int, default=0,
                    help="after all filtering, select the top-N EDG nodes from the GOOD tracks with the "
                         "SAME method as build_edg.py (select_edg_nodes_2d length-NMS) and show only them "
                         "(e.g. 20). 0 = off")
    args = ap.parse_args()
    if args.track_rank is None:                                       # --select-edg: show the selected EDG nodes ALL AT ONCE
        args.track_rank = 0 if args.select_edg > 0 else 1
    if args.color_by is None:                                         # ... and in their own colours, so they can be told apart
        args.color_by = "id" if args.select_edg > 0 else "jump"

    out_dir = Path(utils.definitions.output_dir)
    source = args.source or ("merged" if args.merged else "smoothed")
    args.merged = source in ("merged", "smoothed")                    # merged-like files (fragment colours)
    fname = {"raw": "tracks_dynamic_only.npz", "cow": "cow_skeleton.npz", "merged": "tracks_dynamic_merged.npz",
             "cotracker": "tracks_cotracker_offline.npz",
             "smoothed": "tracks_dynamic_merged_smoothed.npz"}[source]
    npz_path = Path(args.npz) if args.npz else (out_dir / fname)
    if args.npz is None and source == "smoothed" and not npz_path.exists():
        print("[tracks2d] smoothed file missing -> falling back to tracks_dynamic_merged.npz (same 2D positions)")
        npz_path = out_dir / "tracks_dynamic_merged.npz"
    n_img = len(sorted(glob.glob(str(Path(utils.definitions.video_dir) / utils.definitions.image_pattern))))
    d = np.load(str(npz_path), allow_pickle=True)
    is_cotracker = source == "cotracker" or ("tracks" in d.files and "visibility" in d.files)
    is_skeleton = (not is_cotracker) and ("tracks" in d.files and "vis" in d.files)
    if args.npz is not None and not is_cotracker:
        source = "cow" if is_skeleton else source
        args.merged = not is_skeleton
    if is_cotracker:                                                  # CoTracker: tracks + visibility, time-major
        source = "cotracker"; args.merged = False
        xy, occ = _load_cotracker(npz_path, n_img)
        dyn_mask = np.ones(occ.shape, bool)
        fin = np.isfinite(xy).all(-1)
        query_t = np.where(fin.any(1), fin.argmax(1), 0).astype(np.int64)
        msrc = None
    elif is_skeleton:                                                 # CoWTracker skeleton: raw vis/conf -> gated occlusion
        xy = np.asarray(d["tracks"], np.float64)                      # (N,T,2) full-res px (NaN where the run did not cover)
        gate = (np.asarray(d["vis"]) > args.cow_vis) & (np.asarray(d["conf"]) > args.cow_conf) & np.isfinite(xy).all(-1)
        occ = ~gate
        dyn_mask = np.ones(occ.shape, bool)
        query_t = np.where(np.isfinite(xy).all(-1).any(1), np.isfinite(xy).all(-1).argmax(1), 0).astype(np.int64)
        msrc = None
    else:
        xy = np.asarray(d["tracks_xy"], np.float64)[0]                # (Q,T,2) full-res px
        occ = np.asarray(d["occluded"])[0].astype(bool)               # (Q,T) final occlusion (all filters)
        dyn_mask = np.asarray(d["dynamic_obs_mask"]).astype(bool)     # (Q,T)
        query_t = np.asarray(d["query_t"], np.int64).reshape(-1)
        # per-frame ORIGINAL track id that supplied the observation (test_extend_tracks.py only; the cow_attach
        # average has no single source -> member_src is -1 and fragment colours are off)  # noqa: E501
        msrc = np.asarray(d["member_src"], np.int32) if "member_src" in d.files else None
        if msrc is not None and not (msrc >= 0).any():
            msrc = None
    print(f"[tracks2d] source={source} file={npz_path.name}")

    # --with-cow: the merged npz records, per kept track, the index of the skeleton point it was built from,
    # so the two files line up 1:1 and the pair can be drawn together.
    ct_xy = ct_occ = None
    if args.with_cotracker is not None:
        ct_path = Path(args.with_cotracker) if args.with_cotracker else (out_dir / "tracks_cotracker_offline.npz")
        if is_cotracker or is_skeleton:
            print("[tracks2d] --with-cotracker ignored: the source is already a single-tracker file")
        elif not ct_path.exists():
            print(f"[tracks2d] --with-cotracker ignored: {ct_path} not found "
                  "(generate one with compare_trackers_masked.py --stage cotracker3)")
        else:
            ct_xy, ct_occ = _load_cotracker(ct_path, n_img)
            print(f"[tracks2d] --with-cotracker: overlaying {ct_xy.shape[0]} CoTracker tracks "
                  f"(T={ct_xy.shape[1]}) from {ct_path.name}")

    edg_pairs = []                            # (gid_a, gid_b, p90_px, overlap_frames, overlap_ok)
    near_px_sel = 0.0                         # the NMS radius these nodes were selected with
    cow_xy = cow_occ = None
    if args.with_cow:
        sk_path = out_dir / "cow_skeleton.npz"
        if is_skeleton:
            print("[tracks2d] --with-cow ignored: the source IS the skeleton")
        elif "skeleton_index" not in d.files:
            print(f"[tracks2d] --with-cow ignored: {npz_path.name} has no skeleton_index "
                  "(only cow_attach_tracks.py writes it)")
        elif not sk_path.exists():
            print(f"[tracks2d] --with-cow ignored: {sk_path.name} not found")
        else:
            sk = np.load(str(sk_path), allow_pickle=True)
            idx = np.asarray(d["skeleton_index"], np.int64).reshape(-1)
            sxy = np.asarray(sk["tracks"], np.float64)
            sgate = ((np.asarray(sk["vis"]) > args.cow_vis) & (np.asarray(sk["conf"]) > args.cow_conf)
                     & np.isfinite(sxy).all(-1))
            par = idx >= 0                                            # -1 = unattached TAPNet track, no parent
            cow_xy, cow_occ = sxy[np.maximum(idx, 0)], ~sgate[np.maximum(idx, 0)]   # merged-track order
            cow_occ[~par] = True
            print(f"[tracks2d] --with-cow: overlaying {int(par.sum())} parent skeleton tracks from "
                  f"{sk_path.name} ({int((~par).sum())} tracks are parentless)")
    CT_COL = (255, 200, 0)                                            # CoTracker overlay (--with-cotracker)
    COW_COL = (0, 170, 255)                                           # parent CoWTracker skeleton (--with-cow)
    PALETTE = [(60, 220, 60), (60, 130, 255), (200, 80, 255), (60, 220, 220), (255, 200, 60),
               (0, 165, 255), (255, 80, 160), (120, 255, 120), (255, 120, 60), (160, 220, 0)]
    # colour after the k-th jump (k >= 1): one entry per jump, cycled
    JUMP_PALETTE = [tuple(int(c) for c in args.jump_color), (0, 140, 255), (0, 255, 255), (255, 0, 255),
                    (255, 255, 0), (255, 255, 255), (128, 0, 255), (0, 80, 160)]

    def id_colors(n):
        """One distinct colour per track: hue walks the circle by the golden ratio, so CONSECUTIVE indices (and
        spatial neighbours, which the EDG node selection visits in length order) land far apart in hue."""
        h = (np.arange(n) * 0.6180339887 % 1.0 * 179).astype(np.uint8)
        v = np.where(np.arange(n) % 2 == 0, 255, 205).astype(np.uint8)          # alternate value for extra contrast
        hsv = np.stack([h, np.full(n, 255, np.uint8), v], 1)
        return cv2.cvtColor(hsv[None], cv2.COLOR_HSV2BGR)[0].astype(int)
    Q, T = xy.shape[0], xy.shape[1]
    image_paths = sorted(glob.glob(str(Path(utils.definitions.video_dir) / utils.definitions.image_pattern)))
    print(f"[tracks2d] {npz_path.name}: Q={Q} T={T} | {len(image_paths)} images")
    label = [str(q) for q in range(Q)]                                # display id (fragments get parent/frag)
    if args.cow_filters and is_skeleton:                              # exactly the filters cow_attach_tracks.py applies
        import cow_attach_tracks as CAT
        from utils import dataset_config as _dc
        vis0 = ~occ
        jp = _dc.get_float("COW_JUMP_PX", 20.0); gt = _dc.get_float("COW_GAP_TOL_PX", 60.0)
        mj = _dc.get_int("COW_MAX_JUMPS", 5); hd = _dc.get_float("COW_HUE_DEG", 30.0)
        tf = _dc.get_float("COW_MIN_TRACK_FRAC", 0.8); ff = _dc.get_float("COW_MIN_FRAME_FRAC", 0.5)
        v = vis0
        if jp > 0:
            v, n_rm, n_del, kj = CAT.filter_jumpy_tracks(xy, v, jp, gt, mj)
            print(f"[filters] (a) jumpy tracks: {n_del} tracks deleted (jumps > {mj}), {n_rm} point-frames -> visible {v.mean():.2f}")
        if hd > 0:
            _frames = [cv2.imread(q) for q in image_paths[:T]]
            v, n_rm, n_hit = CAT.filter_color_vs_median(xy, v, _frames, hd)
            print(f"[filters] (b) colour vs first frame: {n_hit} tracks truncated, {n_rm} point-frames -> visible {v.mean():.2f}")
        if tf > 0:
            st = np.asarray(d["start_frame"]) if "start_frame" in d.files else np.zeros(len(xy), int)
            started = np.array([(st <= t).sum() for t in range(T)])
            v, n_rm, n_long, n_dense = CAT.filter_coverage(v, tf, ff, started)
            print(f"[filters] (c) coverage: {n_long} long-lived tracks, {n_dense}/{T} dense frames, {n_rm} point-frames -> visible {v.mean():.2f}")
        occ = ~v

    # large-2D-jump detection (same implementation as visualize_cow_jumps.py): n_jumped[q,t] = how many jumps
    # track q has accumulated up to and including frame t; > 0 -> the track is drawn in --jump-color from there on
    trig = n_jumped = None; first_jump = None
    if args.jumps:
        from visualize_cow_jumps import detect_jumps
        vis_now = (~occ) & np.isfinite(xy).all(-1)
        trig, cause = detect_jumps(xy, vis_now, args.jump_px, args.gap_tol_px, 4)
        n_jumped = np.cumsum(trig, axis=1)
        first_jump = np.where(trig.any(1), trig.argmax(1), T)
        ever = trig.any(1)
        after = vis_now & (np.arange(T)[None] >= first_jump[:, None])
        print(f"[jumps] {int(ever.sum())}/{Q} tracks ({100 * ever.mean():.1f}%) jump at least once; "
              f"{int((cause == 1).sum())} step > {args.jump_px:g}px + {int((cause == 2).sum())} gap > {args.gap_tol_px:g}px; "
              f"{int(after.sum())} visible point-frames ({100 * after.sum() / max(vis_now.sum(), 1):.1f}%) are after a first jump")

    ID_COLORS = id_colors(Q)
    vis_count = (~occ).sum(1)                                         # visible frames per track
    order = np.argsort(-vis_count)                                    # rank 1 = most visible frames
    show_ids = None                                                   # all-tracks mode: which tracks to draw

    # --select-edg N: after all filtering, pick the top-N EDG nodes from the GOOD tracks with the SAME
    # method as build_edg.py (select_edg_nodes_2d: 2D p90-persistent-distance length-priority NMS) and
    # restrict the visualization to them.
    if args.select_edg > 0:
        if True:
            from build_edg import select_edg_nodes_2d, _dyn_region_near_px
            vis_now = (~occ) & np.isfinite(xy).all(-1)
            Vs = np.ascontiguousarray(vis_now.T)                      # (T,Q), build_edg's layout
            frac = (args.edg_near_frac if args.edg_near_frac is not None
                    else dataset_config.get_float("EDG_NEAR_FRAC", 0.16))
            near_px = (float(args.edg_near_px) if float(args.edg_near_px) > 0
                       else _dyn_region_near_px({"image_paths": image_paths, "T": T}, frac))
            near_px_sel = float(near_px)
            print(f"[select-edg] near_frac={frac:g} -> near_px={near_px:.1f}"
                  + (" (fixed via --edg-near-px)" if float(args.edg_near_px) > 0 else ""))
            nodes = select_edg_nodes_2d(xy, Vs, Vs.sum(0), near_px=near_px, overlap_frac=0.5,
                                        max_nodes=int(args.select_edg), min_node_frames=20)
            show_ids = np.asarray(nodes, np.int64)
            if args.edg_edges and show_ids.size > 1:
                # Why did NMS keep BOTH of a visually-close pair? The kill test is
                #   p90(distance over co-visible frames) <= near_px  AND  overlap > 0.5 * shorter length.
                # Record both numbers per pair so the drawing can show them against the current distance.
                K = show_ids.size
                Oc_sel = Vs.sum(0)
                for i in range(K):
                    for j in range(i + 1, K):
                        a, b = int(show_ids[i]), int(show_ids[j])
                        both = Vs[:, a] & Vs[:, b]
                        if not both.any():
                            edg_pairs.append((a, b, np.inf, 0, False)); continue
                        dd = np.linalg.norm(xy[a][both] - xy[b][both], axis=1)
                        ov = int(both.sum())
                        edg_pairs.append((a, b, float(np.percentile(dd, 90)), ov,
                                          ov > 0.5 * min(Oc_sel[a], Oc_sel[b])))
                near_sel = [e for e in edg_pairs if e[2] <= near_px]
                why = (f"{len(near_sel)} pair(s) have p90 <= near_px -> those survived only because the "
                       f"overlap test failed" if near_sel else
                       "no pair has p90 <= near_px -> every pair is genuinely non-redundant, "
                       "however close it looks at any one frame")
                print(f"[select-edg] pair stats over {len(edg_pairs)} pairs: "
                      f"p90 min={min(e[2] for e in edg_pairs):.0f}px "
                      f"median={float(np.median([e[2] for e in edg_pairs])):.0f}px | "
                      f"near_px={near_px:.0f}px | {why}")
                for a, b, d90, ov, ovok in sorted(edg_pairs, key=lambda e: e[2])[:5]:
                    print(f"    {a:5d}-{b:5d}  p90={d90:7.1f}px  overlap={ov:4d}f  "
                          f"overlap-test={'pass' if ovok else 'FAIL'}  "
                          f"{'<= near_px -> only overlap saved it' if d90 <= near_px else '> near_px -> kept, not redundant'}")
            order = show_ids[np.argsort(-vis_count[show_ids])]        # rank over the selected nodes only
            print(f"[select-edg] {show_ids.size} EDG nodes selected "
                  f"(build_edg 2D length-NMS): " + ", ".join(label[g] for g in order.tolist()))

    print(f"[tracks2d] Q={Q} | visible-frames rank1={int(vis_count[order[0]])} "
          f"median={int(np.median(vis_count))}")

    def jump_color(q, t, base):
        """Colour of track q at frame t: base before its first jump, then one palette entry per jump."""
        if n_jumped is None:
            return base
        k = int(n_jumped[q, t])
        if k == 0:
            return base
        return JUMP_PALETTE[0] if args.jump_mode == "first" else JUMP_PALETTE[(k - 1) % len(JUMP_PALETTE)]

    def shape_of(q):
        return "circle"

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
            vis_color = jump_color(q, f, vis_color)                               # new colour after each jump
            if trig is not None and trig[q, f]:
                cv2.circle(img, pt, radius + 3, (0, 255, 255), 1, cv2.LINE_AA)    # jump detected THIS frame
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
            n_vis = n_jump = 0
            iter_ids = show_ids.tolist() if show_ids is not None else range(Q)
            for q in iter_ids:
                if f >= query_t[q] and np.isfinite(xy[q, f]).all() and not occ[q, f]:
                    p = xy[q, f]; pt = (int(round(p[0])), int(round(p[1])))
                    jumped = n_jumped is not None and n_jumped[q, f] > 0
                    col = (tuple(int(c) for c in ID_COLORS[q]) if args.color_by == "id"
                           else jump_color(q, f, (60, 220, 60)))
                    _marker(img, pt, shape_of(q), 3 if args.color_by == "id" else 2, col, True)
                    if trig is not None and trig[q, f]:
                        cv2.circle(img, pt, 5, (0, 255, 255), 1, cv2.LINE_AA)
                    n_vis += 1; n_jump += int(jumped)
            n_lab = 0
            if edg_pairs:                                         # label pairs that LOOK close at this frame
                lim = float(args.edg_pair_px) if float(args.edg_pair_px) > 0 else near_px_sel
                for a, b, d90, ov, ovok in edg_pairs:
                    pa, pb = xy[a, f], xy[b, f]
                    if not (np.isfinite(pa).all() and np.isfinite(pb).all()) or occ[a, f] or occ[b, f]:
                        continue
                    d_now = float(np.linalg.norm(pa - pb))
                    if d_now > lim:                               # not visually close here -> not the puzzle
                        continue
                    pA = (int(round(pa[0])), int(round(pa[1]))); pB = (int(round(pb[0])), int(round(pb[1])))
                    col = (0, 0, 255) if d90 <= near_px_sel else (255, 255, 255)
                    cv2.line(img, pA, pB, col, 1, cv2.LINE_AA)
                    mid = ((pA[0] + pB[0]) // 2, (pA[1] + pB[1]) // 2)
                    cv2.putText(img, f"{d_now:.0f}|{d90:.0f}", mid, cv2.FONT_HERSHEY_SIMPLEX,
                                0.35, col, 1, cv2.LINE_AA)
                    n_lab += 1
            if ct_xy is not None and f < ct_xy.shape[1]:
                for j in range(ct_xy.shape[0]):
                    pj = ct_xy[j, f]
                    if np.isfinite(pj).all() and not ct_occ[j, f]:
                        cv2.circle(img, (int(round(pj[0])), int(round(pj[1]))), 2, CT_COL, -1, cv2.LINE_AA)
            if cow_xy is not None:
                for q in iter_ids:
                    pc = cow_xy[q, f]
                    if np.isfinite(pc).all() and not cow_occ[q, f]:
                        cv2.circle(img, (int(round(pc[0])), int(round(pc[1]))), 2, COW_COL, 1, cv2.LINE_AA)
            txt = (f"selected EDG nodes ({show_ids.size})  visible={n_vis}" if show_ids is not None
                   else f"all tracks  visible={n_vis}")
            if args.color_by == "id":
                txt += "  [one colour per node]"
            if edg_pairs:
                txt += f"  near-pairs={n_lab} labelled now|p90  near_px={near_px_sel:.0f}"
            if n_jumped is not None:
                txt += f"  clean={n_vis - n_jump} jumped={n_jump} (ring = jump this frame)"
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
                    c = jump_color(q, tt, c)                          # trail recoloured at every jump
                    cv2.line(img, prev, cur_pt, c, 1, cv2.LINE_AA)
                prev = cur_pt
            vcol, src_txt = (60, 220, 60), ""
            if msrc is not None:
                cur = int(msrc[q, f])
                if cur >= 0:
                    vcol = col_of.get(cur, vcol)
                src_txt = f" frags={len(col_of)} src={cur if cur >= 0 else '-'}"
            if ct_xy is not None and f < ct_xy.shape[1]:          # context: all CoTracker points this frame
                for j in range(ct_xy.shape[0]):
                    pj = ct_xy[j, f]
                    if np.isfinite(pj).all() and not ct_occ[j, f]:
                        cv2.circle(img, (int(round(pj[0])), int(round(pj[1]))), 2, CT_COL, -1, cv2.LINE_AA)
            cow_txt = ""
            if cow_xy is not None:                                    # parent skeleton: trail, point, offset
                prevc = None
                for tt in range(f0, f + 1):
                    pc = cow_xy[q, tt]
                    if not np.isfinite(pc).all():
                        prevc = None
                        continue
                    cpt = (int(round(pc[0])), int(round(pc[1])))
                    if prevc is not None:
                        cv2.line(img, prevc, cpt, COW_COL, 1, cv2.LINE_AA)
                    prevc = cpt
                pc, pm = cow_xy[q, f], xy[q, f]
                if np.isfinite(pc).all():
                    cpt = (int(round(pc[0])), int(round(pc[1])))
                    cv2.circle(img, cpt, 4, COW_COL, -1 if not cow_occ[q, f] else 1, cv2.LINE_AA)
                    if np.isfinite(pm).all():
                        mpt = (int(round(pm[0])), int(round(pm[1])))
                        cv2.line(img, cpt, mpt, COW_COL, 1, cv2.LINE_AA)
                        cow_txt = f" cow-offset={float(np.linalg.norm(pm - pc)):.1f}px"
                    if cow_occ[q, f]:
                        cow_txt += " (cow gated-out)"
                else:
                    cow_txt = " cow=no-pred"
            state = draw_point(q, radius=4, vis_color=vcol) if f >= query_t[q] else "pre-birth"
            gtxt = ""
            if n_jumped is not None:
                gtxt += (f" jumps={int(trig[q].sum())} first={int(first_jump[q])} seg={int(n_jumped[q, f]) + 1}"
                         if trig[q].any() else " no-jump")
            txt = (f"rank {rank}/{order.size}: gid={label[q]} vis={int(vis_count[q])}f "
                   f"birth={int(query_t[q])} [{state}]{gtxt}{src_txt}{cow_txt}")

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

          + ("" if n_jumped is None else " | colour CHANGES AT EVERY 2D jump (red/orange/yellow/magenta/...), "
             "yellow ring=jump this frame")
          + (" | all-tracks view: ONE COLOUR PER NODE (--color-by jump to colour by jump state instead)"
             if args.color_by == "id" else ""))
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
