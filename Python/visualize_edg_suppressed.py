#!/usr/bin/env python3
"""Show, per EDG node, the tracks that node DELETED (suppressed) during EDG-node selection.

build_edg.py picks EDG nodes by length-priority NMS (build_edg.select_edg_nodes_2d -- THE SAME function
this tool calls): rank tracks by reliable length (longest first), select the longest survivor as a node,
then DELETE every track that is (a) NEAR it in the image AND (b) temporally OVERLAPS it by more than
overlap_frac of the shorter track's length. This tool reproduces that selection (it literally imports the
build_edg function) and RECORDS which tracks each node suppressed, and draws them.

RANGE (the 'near' test): a 2D IMAGE-pixel distance with the SAME persistent-distance metric as build_edg's
edge builder -- the p<dist-quantile> (default p90) of the per-frame image distance over co-visible frames
(np.nanquantile, a robust near-max). Delete a track whose persistent 2D distance to the node is <=
--near-px (LOWER --near-px to delete fewer/closer tracks). Set --near-px 0 to use the old 3D canonical
radius (viewer-only fallback; build_edg no longer uses 3D). Match --near-px / --dist-quantile /
--edg-max-nodes to build_edg's --edg-near-px / --edg-dist-quantile / --edg-max-nodes to reproduce the
saved edg_topology.npz exactly.

For the current rank (1 = longest / first-selected node): the EDG node is drawn in GREEN with a white
ring + "NODE" label; every track it suppressed/deleted is a distinct palette colour, over the real frame
image, with short trails. Only tracks that are RELIABLE (observed) at the current frame are drawn -- a
track that is occluded / off the dynamic mask this frame is hidden. The node is always shown (as the
subject); if it is occluded this frame it is drawn as a hollow green ring instead of a filled disc.

Keys:  N = next rank, B = previous rank (jumps to the node's birth frame);  SPACE = next frame,
       , = previous frame;  Q/ESC = quit.

Run (from Python/):
    XM_DATASET_NAME=<scene> conda run -n XM python visualize_edg_suppressed.py [--merged] [--rank 1]
    (headless single image: add --frame F --save-frame out.png)
"""
import argparse
import glob
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import utils.definitions
import build_edg as BE

# distinct BGR palette for suppressed/deleted tracks (NO green -- green is reserved for the EDG node)
PALETTE = [(0, 165, 255), (255, 60, 60), (255, 0, 255), (0, 255, 255), (200, 120, 0),
           (120, 0, 200), (0, 120, 255), (255, 180, 0), (180, 0, 120), (128, 128, 128)]
NODE_COLOR = (60, 220, 60)                                   # bright green = the EDG node


def select_with_suppression(Ps, Vs, Ocount, thr, near_factor, overlap_frac, max_nodes, min_node_frames):
    """3D-canonical-distance fallback (used only with --near-px 0): the pre-2D length-priority NMS on the
    mean 3D position within radius near_factor*thr, recording per-node suppressed global ids. Returns
    (nodes[K], suppressed[list of K int arrays])."""
    has = np.where(Ocount > int(min_node_frames))[0]
    if has.size == 0:
        return np.zeros(0, np.int64), []
    canon = np.where(Vs[..., None], Ps, 0.0).sum(0)[has] / Ocount[has, None]     # (H,3) mean reliable pos
    r = float(near_factor) * float(thr)
    tree = cKDTree(canon)
    order = np.argsort(-Ocount[has])            # local indices, longest first
    alive = np.ones(has.size, bool)
    nodes, suppressed = [], []
    for l in order:
        if not alive[l]:
            continue
        g = int(has[l])
        near = np.asarray(tree.query_ball_point(canon[l], r), dtype=np.int64)
        near = near[(near != l) & alive[near]]
        supp = np.zeros(0, np.int64)
        if near.size:
            gj = has[near]
            overlap = (Vs[:, g][:, None] & Vs[:, gj]).sum(axis=0)                # co-visible reliable frames
            thresh = overlap_frac * np.minimum(Ocount[g], Ocount[gj])            # > overlap_frac of shorter
            kill = near[overlap > thresh]
            alive[kill] = False
            supp = has[kill]                                                     # global ids suppressed here
        alive[l] = False
        nodes.append(g)
        suppressed.append(supp.astype(np.int64))
        if len(nodes) >= max_nodes:
            break
    return np.array(nodes, np.int64), suppressed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged", action="store_true", help="use the merged tracks (as build_edg --merged)")
    ap.add_argument("--rank", type=int, default=1, help="starting rank (1 = longest / first-selected node)")
    ap.add_argument("--frame", type=int, default=-1, help="starting frame (-1 = the node's birth frame)")
    ap.add_argument("--trail", type=int, default=12, help="draw each track's last N frames as a trail")
    ap.add_argument("--scale", type=float, default=2.0, help="upscale the display image by this factor")
    ap.add_argument("--save-frame", type=str, default="", help="headless: render one frame to this PNG and exit")
    # match build_edg main() defaults so the selection == the saved edg_topology.npz
    ap.add_argument("--no-scale", action="store_true")
    ap.add_argument("--edge-thresh", type=float, default=0.0)
    ap.add_argument("--mask-dilate-px", type=int, default=5)
    ap.add_argument("--near-px", type=float, default=15.0,
                    help="RANGE KNOB: delete tracks whose persistent 2D distance to the node (the p<dist-"
                         "quantile> of per-frame image distance over co-visible frames -- same metric as "
                         "build_edg edges) is within this many pixels. LOWER to delete fewer/closer tracks. "
                         "Set 0 to fall back to the old 3D canonical radius (--edg-near-factor).")
    ap.add_argument("--dist-quantile", type=float, default=0.90,
                    help="quantile for the persistent 2D distance (build_edg EdgeConfig.distance_quantile "
                         "= 0.90; main CLI uses 0.95). p90 = robust near-max ignoring the closest 10%% frames")
    ap.add_argument("--edg-near-factor", type=float, default=1.0,
                    help="3D fallback range (used only when --near-px 0): near <= this * edge-distance.")
    ap.add_argument("--edg-overlap-frac", type=float, default=0.5)
    ap.add_argument("--edg-min-frames", type=int, default=20)
    ap.add_argument("--edg-max-nodes", type=int, default=100)   # matches build_edg's default cap
    args = ap.parse_args()

    import cv2

    merged_path = str(Path(utils.definitions.output_dir) / "tracks_dynamic_merged.npz")
    D = BE._load_all(tracks_path=merged_path if args.merged else None)
    D["dyn_mask"] = BE._dilate_dyn_mask(D, args.mask_dilate_px)
    s = BE._frame_scale(D["rt"], D["T"], args.no_scale)
    T = D["T"]

    # SAME prep as build_edg (full video), but only what node selection needs (Ps, Vs, Ocount, thr)
    sampled = np.arange(T, dtype=int)
    Ps, Vs = BE._stack_sampled(D, s, sampled)
    Ocount = Vs.sum(axis=0)
    thr = args.edge_thresh or BE._auto_edge_threshold(Ps, Vs)
    xy = D["tracks_xy"][0].astype(np.float64)                 # (Q,T,2) native px
    if args.near_px > 0:                                      # 2D image-distance deletion (default)
        nodes, suppressed = BE.select_edg_nodes_2d(          # THE canonical build_edg selector (identical)
            xy, Vs, Ocount, args.near_px, args.edg_overlap_frac, args.edg_max_nodes, args.edg_min_frames,
            quantile=args.dist_quantile, verbose=False, return_suppressed=True)
        rng = f"2D near<= {args.near_px:g}px (p{int(args.dist_quantile*100)} persistent)"
    else:                                                     # 3D canonical-distance deletion (fallback)
        nodes, suppressed = select_with_suppression(
            Ps, Vs, Ocount, thr, args.edg_near_factor, args.edg_overlap_frac, args.edg_max_nodes, args.edg_min_frames)
        rng = f"3D near<= {args.edg_near_factor}x edge-dist ({args.edg_near_factor*thr:.4f})"
    K = nodes.size
    if K == 0:
        print("[suppressed] no EDG nodes selected (check --edg-min-frames / data).")
        return
    n_supp = np.array([sp.size for sp in suppressed])
    print(f"[suppressed] {K} EDG nodes ({rng}, overlap> {args.edg_overlap_frac:.0%}); "
          f"suppressed/node: total {int(n_supp.sum())} min={int(n_supp.min())} "
          f"median={int(np.median(n_supp))} max={int(n_supp.max())}")

    # cross-check against the saved topology (build_edg now uses this same selector, so 2D matches when
    # --near-px/--dist-quantile/--edg-max-nodes match build_edg's --edg-near-px/-dist-quantile/-max-nodes)
    topo = Path(utils.definitions.output_dir) / "edg_topology.npz"
    if topo.exists():
        td = np.load(topo, allow_pickle=True)
        if "n_tracks" in td.files and int(td["n_tracks"]) == D["tracks_xyz"].shape[1]:
            saved = set(np.asarray(td["edg_nodes"]).tolist())
            print(f"[suppressed] edg_topology.npz nodes: {len(saved)} | matches reproduced set: "
                  f"{saved == set(nodes.tolist())} (matches build_edg when the params above align)")
        else:
            print("[suppressed] edg_topology.npz is a different track-id space (add/drop --merged to compare).")

    birth = np.where(Vs.any(0), Vs.argmax(0), 0)              # first reliable frame per track
    image_paths = D["image_paths"]

    def render(rank, f):
        node = int(nodes[rank - 1]); supp = suppressed[rank - 1]
        img = cv2.imread(image_paths[f]) if f < len(image_paths) else None
        if img is None:
            img = np.zeros((480, 640, 3), np.uint8)
        img = img.copy()

        def draw_track(q, color, radius, is_node=False):
            # only draw tracks that are RELIABLE (observed) this frame; occluded/off-mask are hidden.
            # the node is always shown (it is the subject) even when occluded.
            if not (np.isfinite(xy[q, f]).all() and (Vs[f, q] or is_node)):
                return "hidden"
            f0 = max(int(birth[q]), f - int(args.trail))
            prev = None
            for tt in range(f0, f + 1):                        # trail (only through reliable frames)
                if not (np.isfinite(xy[q, tt]).all() and Vs[tt, q]):
                    prev = None
                    continue
                cur = (int(round(xy[q, tt, 0])), int(round(xy[q, tt, 1])))
                if prev is not None:
                    cv2.line(img, prev, cur, color, 2 if is_node else 1, cv2.LINE_AA)
                prev = cur
            pt = (int(round(xy[q, f, 0])), int(round(xy[q, f, 1])))
            if Vs[f, q]:
                cv2.circle(img, pt, radius, color, -1, cv2.LINE_AA)       # reliable this frame: FILLED
                state = "vis"
            else:                                                         # node only, currently occluded
                cv2.circle(img, pt, radius + 1, color, 1, cv2.LINE_AA)
                state = "node-occluded"
            if is_node:                                                   # make the EDG node unmistakable
                cv2.circle(img, pt, radius + 6, (255, 255, 255), 2, cv2.LINE_AA)   # white attention ring
                cv2.putText(img, "NODE", (pt[0] + radius + 9, pt[1] - radius - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, "NODE", (pt[0] + radius + 9, pt[1] - radius - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, NODE_COLOR, 1, cv2.LINE_AA)
            return state

        n_shown = 0
        for i, q in enumerate(supp.tolist()):                            # suppressed/deleted tracks (palette)
            if draw_track(q, PALETTE[i % len(PALETTE)], 3) != "hidden":
                n_shown += 1
        st = draw_track(node, NODE_COLOR, 6, is_node=True)               # EDG node (green + ring, on top)

        txt = (f"rank {rank}/{K}  GREEN=NODE gid={node} len={int(Ocount[node])}f [{st}]  "
               f"deleted shown {n_shown}/{supp.size} (reliable only)  frame {f}/{T-1}")
        if args.scale != 1.0:
            img = cv2.resize(img, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_NEAREST)
        cv2.putText(img, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        print(f"[suppressed] {txt}")
        return img

    rank = int(np.clip(args.rank, 1, K))
    f = int(birth[nodes[rank - 1]]) if args.frame < 0 else int(np.clip(args.frame, 0, T - 1))

    if args.save_frame:
        cv2.imwrite(args.save_frame, render(rank, f))
        print(f"[suppressed] wrote {args.save_frame}")
        return

    print("[suppressed] N=next rank, B=prev rank | SPACE=next frame, ,=prev frame | Q/ESC=quit. "
          "GREEN(+white ring,'NODE')=EDG node; other colours=tracks it deleted (near+overlapping). "
          "Only tracks RELIABLE this frame are drawn (occluded/off-mask hidden); node always shown.")
    while True:
        cv2.imshow("EDG node + suppressed tracks", render(rank, f))
        k = cv2.waitKey(0) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == 32:                                    # SPACE -> next frame
            f = (f + 1) % T
        elif k == ord(","):                              # , -> prev frame
            f = (f - 1) % T
        elif k in (ord("n"), ord("N")):                  # N -> next rank
            rank = min(rank + 1, K); f = int(birth[nodes[rank - 1]])
        elif k in (ord("b"), ord("B")):                  # B -> prev rank
            rank = max(rank - 1, 1); f = int(birth[nodes[rank - 1]])
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
