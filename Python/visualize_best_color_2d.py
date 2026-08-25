#!/usr/bin/env python3
"""2D image-plane view of the BEST colour-aligned dynamic points.

Every dynamic track is skinned by the embedded-deformation graph and scored by the photometric render-colour
test (detect_render_color_outliers): the N tracks whose reprojection colour aligns BEST (the LOWEST score) are
drawn as FILLED GREEN over each frame image -- a clean look at the dynamic cloud that the deformation graph
places most faithfully. Requires the COMPLETED EDG_test npz (needs the dynamic points). SPACE/B step frame,
Q/ESC quit; --dry-run writes a PNG. With --best-only only the green best points are shown; otherwise the
render-colour outliers (red) and the other dynamic points (blue) are drawn too.

Run:  XM_DATASET_NAME=<scene> conda run -n XM python visualize_best_color_2d.py \
          --npz <outputs>/edg_test/edg_completed.npz --best-dyn 300
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import utils.definitions
from edg_viz_common import (_build_dyn_skin, _skin_dyn_frame, detect_render_color_outliers,
                            _load_2d_context, project_to_frame)


def _default_results_path() -> Path:
    output_dir = Path(utils.definitions.output_dir)
    return output_dir / "results_all.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="2D view of the best colour-aligned dynamic points over the frame image."
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default=_default_results_path(),
        help="Path to the COMPLETED EDG_test npz (edg_completed.npz) carrying node_ids + dyn_reconstruct.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index to start on.",
    )
    parser.add_argument(
        "--merged",
        action="store_true",
        help="The npz was solved on the MERGED tracks; load those for the 2D context.",
    )
    parser.add_argument(
        "--viz-2d-scale",
        type=float,
        default=2.0,
        help="Display upscale factor for the image window.",
    )
    parser.add_argument(
        "--best-dyn",
        type=int,
        default=300,
        help="Draw the N dynamic tracks whose reprojection colour aligns BEST (the LOWEST render-colour "
             "score) as filled GREEN. Uses the same scoring as the outlier test.",
    )
    parser.add_argument(
        "--best-only",
        action="store_true",
        help="Show ONLY the best-aligned dynamic points (green) -- hide the other (blue/red) dynamic points, "
             "for a clean view of the best-aligned dynamic cloud.",
    )
    parser.add_argument(
        "--use-color",
        action="store_true",
        help="Instead of the green/blue/red category dots, draw every GOOD (non-outlier) dynamic point at "
             "its reprojection filled with ITS OWN fused colour (median observed-frame BGR) -- a colour "
             "render of the good dynamic cloud. Outliers are hidden.",
    )
    # ---- render-colour scoring flags (shared scorer detect_render_color_outliers) ----
    parser.add_argument(
        "--render-color-space",
        choices=("hue", "bgr"),
        default="hue",
        help="Colour space of the reprojection-colour test. 'hue' (HSV, circular distance in degrees 0..90) "
             "is specular-robust on this scene; 'bgr' uses L2 BGR distance (0..441).",
    )
    parser.add_argument(
        "--render-color-thresh",
        type=float,
        default=30.0,
        help="A dynamic track is an OUTLIER (red, when not --best-only) when its render-colour score exceeds "
             "this. Units match --render-color-space (hue: degrees 0..90; bgr: L2 BGR 0..441).",
    )
    parser.add_argument(
        "--render-color-quantile",
        type=float,
        default=0.5,
        help="Quantile of the per-frame colour drift used as the per-track score (0.5 = robust median).",
    )
    parser.add_argument(
        "--render-min-span",
        type=int,
        default=5,
        help="A track needs at least this many EXTRAPOLATED (non-observed) saturated frames to be scored.",
    )
    parser.add_argument(
        "--render-min-obs",
        type=int,
        default=3,
        help="A track needs at least this many OBSERVED saturated frames to build a reference colour.",
    )
    parser.add_argument(
        "--render-patch",
        type=int,
        default=2,
        help="Half-width of the (2p+1)x(2p+1) BGR patch median used to denoise the sampled colour "
             "(p=2 -> 5x5). 0 samples a single pixel.",
    )
    parser.add_argument(
        "--render-sat-min",
        type=int,
        default=40,
        help="(hue only) ignore frames whose sampled saturation is below this (specular/white pixels have "
             "unstable hue).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render the single --frame to a PNG in output_dir and exit (no window).",
    )
    parser.add_argument(
        "--fill-edg",
        action="store_true",
        help="(EDG_test format) HOLD each EDG node across the whole interval before skinning (back/forward-"
             "fill holes with its nearest solved pose).",
    )
    return parser.parse_args()


def _hold_fill_stack(arr: np.ndarray) -> np.ndarray:
    """Complete each track [T,...] over the whole interval so it is finite at every frame: hold the first
    finite value backward, the last finite value forward, and carry the last finite across interior gaps.
    (EDG_test solves a node forward from its first appearance, so this mainly back-fills the pre-appearance
    frames -> the WHOLE EDG shows across the whole time interval.)"""
    arr = np.asarray(arr, dtype=np.float64).copy()
    N, T = arr.shape[0], arr.shape[1]
    flat = arr.reshape(N, T, -1)
    for i in range(N):
        fin = np.isfinite(flat[i]).all(axis=1)
        idx = np.where(fin)[0]
        if idx.size == 0 or idx.size == T:
            continue
        arr[i, : idx[0]] = arr[i, idx[0]]
        arr[i, idx[-1] + 1:] = arr[i, idx[-1]]
        last = idx[0]
        for t in range(idx[0] + 1, idx[-1] + 1):
            if fin[t]:
                last = t
            else:
                arr[i, t] = arr[i, last]
    return arr


def run_best_color_2d(dyn_skin, dyn_best, dyn_flag, ctx, args, start_frame: int,
                      dyn_good=None, fused_col=None):
    """OpenCV stepper: per frame, skin every dynamic point, project it into the image (world -> camera via the
    inverse XM lift, then pinhole) and draw. Default: filled GREEN = the N best colour-aligned tracks; when
    not --best-only also filled RED = render-colour OUTLIER, small blue = other dynamic points. With
    --use-color: every GOOD (non-outlier) point is drawn filled with ITS OWN fused colour (a colour render)."""
    import cv2
    T = min(ctx["T"], dyn_skin["edg_pos"].shape[1])

    def render(f):
        img = cv2.imread(ctx["image_paths"][f]) if f < len(ctx["image_paths"]) else None
        if img is None:
            img = np.zeros((480, 360, 3), np.uint8)
        n_dyn = n_dyn_out = n_dyn_best = n_good = 0
        canvas = np.full_like(img, 255) if args.use_color else None   # --use-color: white right-hand panel
        if 0 <= f < dyn_skin["edg_pos"].shape[1]:
            dpos = _skin_dyn_frame(dyn_skin, f)
            dfin = np.isfinite(dpos).all(1)
            if dfin.any():
                du, dv, dok = project_to_frame(dpos[dfin], ctx, f)
                if args.use_color:                                    # colour render of the good dynamic cloud
                    dgood = dyn_good[dfin]; dcol = fused_col[dfin]
                    for i in np.where(dok)[0]:
                        if not bool(dgood[i]):
                            continue                                  # outlier / no reliable colour: hidden
                        col = tuple(int(v) for v in dcol[i])          # the track's own fused BGR colour
                        cv2.circle(canvas, (int(round(du[i])), int(round(dv[i]))), 2, col, -1)  # onto WHITE
                        n_good += 1
                else:
                    dflags = (dyn_flag[dfin] if dyn_flag is not None
                              else np.zeros(int(dfin.sum()), bool))
                    dbest = dyn_best[dfin]
                    for i in np.where(dok)[0]:
                        c = (int(round(du[i])), int(round(dv[i])))
                        if bool(dbest[i]):
                            cv2.circle(img, c, 3, (0, 255, 0), -1)    # BEST-aligned dynamic point: filled green
                            n_dyn_best += 1
                        elif args.best_only:
                            continue                                  # best-only: skip non-best dynamic points
                        elif bool(dflags[i]):
                            cv2.circle(img, c, 3, (0, 0, 255), -1)    # OUTLIER dynamic point: filled red
                            n_dyn_out += 1
                        else:
                            cv2.circle(img, c, 1, (255, 128, 0), -1)  # normal dynamic point: small blue
                        n_dyn += 1
        if args.use_color:                                            # LEFT = original 2D | RIGHT = white + points
            cv2.putText(img, "origin", (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(canvas, "good points", (8, canvas.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            img = np.hstack([img, np.full((img.shape[0], 2, 3), 128, np.uint8), canvas])
        s = float(args.viz_2d_scale)
        if s != 1.0:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        dyn_txt = (f" good={n_good}" if args.use_color
                   else f" best={n_dyn_best}" + ("" if args.best_only else f" dyn={n_dyn} dyn_out={n_dyn_out}"))
        cv2.putText(img, f"frame {f}/{T - 1}  [{'color-render' if args.use_color else 'best-color'}]{dyn_txt}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        print(f"[best-2d] frame={f}{dyn_txt}")
        return img

    f = int(np.clip(start_frame, 0, T - 1))
    if args.dry_run:
        out = Path(utils.definitions.output_dir) / f"best_color2d_frame{f:04d}.png"
        cv2.imwrite(str(out), render(f))
        print(f"[best-2d][dry-run] wrote {out}")
        return
    print("[best-2d] SPACE=next frame, B=prev frame | Q/ESC=quit. green=best colour-aligned dynamic points"
          + ("" if args.best_only else ", red=render-colour outliers, blue=normal dynamic points") + ".")
    while True:
        cv2.imshow("EDG best-colour dynamic 2D", render(f))
        k = cv2.waitKey(0) & 0xFF
        if k in (ord("q"), ord("Q"), 27):
            break
        if k in (ord("b"), ord("B")):
            f = (f - 1) % T
        else:
            f = (f + 1) % T
    cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()

    npz_path = args.npz.resolve()
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing results file: {npz_path}")

    print(f"Loading EDG result: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    if not ("node_ids" in data.files and "dyn_reconstruct" in data.files):
        raise RuntimeError(f"{npz_path} is not a completed EDG_test npz (needs node_ids + dyn_reconstruct); "
                           "pass the edg_completed.npz.")
    node_ids = np.asarray(data["node_ids"], dtype=np.int64)
    er = np.asarray(data["edg_reconstruct"], dtype=np.float64)              # [Nn,T,3]
    ero = np.asarray(data["edg_rotation"], dtype=np.float64)                # [Nn,T,3,3]
    if args.fill_edg:
        er = _hold_fill_stack(er)
        ero = _hold_fill_stack(ero)
        print(f"[edg_test] --fill-edg: held {int(node_ids.size)} EDG nodes across the whole "
              f"{er.shape[1]}-frame interval.")
    dyn_ids = np.asarray(data["dyn_ids"], dtype=np.int64)
    dyn_reconstruct = np.asarray(data["dyn_reconstruct"], dtype=np.float64)  # [Nd,T,3]

    # 2D context: pick the right tracks table by the npz's n_tracks stamp (auto-switch is in _load_2d_context)
    stamp = int(data["n_tracks"]) if "n_tracks" in data.files else None
    if stamp is None:
        topo_p = Path(utils.definitions.output_dir) / "edg_topology.npz"
        if topo_p.exists():
            topo = np.load(topo_p)
            if "n_tracks" in topo.files:
                tn = set(np.asarray(topo["edg_nodes"], int).tolist())
                if set(np.asarray(node_ids, int).tolist()) <= tn:
                    stamp = int(topo["n_tracks"])
                    print(f"[2d] npz is unstamped but its node_ids match edg_topology.npz -> using the "
                          f"topology's stamp n_tracks={stamp}")
    if stamp is None:
        print("[2d] npz has no n_tracks stamp (older solve): trusting --merged; if observations look "
              "missing/misconnected, the flag is probably wrong for this npz")

    ctx = _load_2d_context(args.merged, stamp)

    dyn_skin, dyn_pids = _build_dyn_skin(dyn_ids, dyn_reconstruct, node_ids, er, ero)
    if dyn_skin is None:
        raise RuntimeError("no skinnable dynamic points; cannot build the best-colour view.")
    flag, score, fused = detect_render_color_outliers(
        dyn_skin, dyn_pids, dyn_ids, dyn_reconstruct, ctx, args, return_color=True)

    dyn_flag = None if args.best_only else flag                        # outliers (red) only when not best-only
    fin = np.where(np.isfinite(score))[0]
    order = fin[np.argsort(score[fin])]                                # lowest render-colour score first
    dyn_best = np.zeros(dyn_skin["P"], bool)
    dyn_best[order[: int(args.best_dyn)]] = True
    if order.size:
        print(f"[best-dyn] {int(dyn_best.sum())} best-aligned dynamic tracks (GREEN); "
              f"score range {score[order[0]]:.1f}..{score[order[min(args.best_dyn, order.size) - 1]]:.1f} "
              f"of {order.size} scored")

    # colour-render mode: every GOOD (non-outlier) track with a reliable observed colour, drawn in that colour
    dyn_good = fused_col = None
    if args.use_color:
        dyn_good = np.isfinite(fused).all(1) & (~flag)
        fused_col = fused
        print(f"[use-color] rendering {int(dyn_good.sum())} good (non-outlier) dynamic points in their own colour")

    max_frames = int(dyn_skin["edg_pos"].shape[1])
    run_best_color_2d(dyn_skin, dyn_best, dyn_flag, ctx, args,
                      start_frame=int(np.clip(args.frame, 0, max_frames - 1)),
                      dyn_good=dyn_good, fused_col=fused_col)


if __name__ == "__main__":
    main()
