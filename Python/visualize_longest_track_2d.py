#!/usr/bin/env python3
"""2D image-plane view of the EDG tracks (the "show longest 2D track" view).

Each loaded EDG track is PROJECTED into every frame (inverse XM lift + per-frame intrinsics) and compared with
its INITIAL 2D observation (tracks_xy): GREEN dot = observation, RED dot = projected EDG, and a RED line
ALWAYS joins each pair (--reproj-thresh only feeds the on-screen 'far' counter). Tracks are ranked by observed-
frame count so you can step through them longest-first: SPACE/B step frame, N = next-longest, P = prev (longer),
A = all tracks, Q/ESC quit. No dynamic points, no 3D -- see visualize_best_color_2d.py /
visualize_edg_filtered_open3d.py for those.

Run:  XM_DATASET_NAME=<scene> conda run -n XM python visualize_longest_track_2d.py \
          --npz <outputs>/edg_test/edg_completed.npz --track-rank 1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import utils.definitions
from edg_viz_common import _load_2d_context, project_to_frame


def _default_results_path() -> Path:
    output_dir = Path(utils.definitions.output_dir)
    return output_dir / "results_all.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="2D reprojection view of the (longest) EDG tracks vs their initial 2D observations."
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default=_default_results_path(),
        help="Path to the solved EDG npz (EDG_test edg_global.npz / edg_completed.npz, or EDG_old's "
             "results_all.npz).",
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
        help="The npz was solved on the MERGED tracks; load those for the 2D observations.",
    )
    parser.add_argument(
        "--reproj-thresh",
        type=float,
        default=5.0,
        help="Count a pair as 'far' when |projected - observed| exceeds this many pixels (display only; the "
             "connecting line is always drawn).",
    )
    parser.add_argument(
        "--viz-2d-scale",
        type=float,
        default=2.0,
        help="Display upscale factor for the image window.",
    )
    parser.add_argument(
        "--track-rank",
        type=int,
        default=0,
        help="STARTING rank: show only the x-th LONGEST track (1 = longest, ranked by observed-frame count); "
             "0 = all tracks. Change interactively with N (next), P (prev), A (all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render the single --frame to a PNG in output_dir and exit (no window).",
    )
    parser.add_argument(
        "--fill-edg",
        action="store_true",
        help="(EDG_test format) HOLD each EDG node across the whole interval (back/forward-fill holes with "
             "its nearest solved pose). Off by default: a node that is a HOLE at a frame is not drawn.",
    )
    return parser.parse_args()


def _load_dict(data, key):
    value = data.get(key)
    if value is None:
        return None
    return value.item()


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


def run_viz_2d(edg_tracks: dict, ctx: dict, args, start_frame: int):
    """OpenCV stepper: per frame, project every (loaded) EDG track into the image (world -> camera via the
    inverse XM lift x_cam = R_f^T (x - t_f)/s_f, then pinhole) and draw: RED dot = projected EDG, GREEN dot =
    the node's initial 2D observation, and a red line ALWAYS joining each pair (whether the distance is large
    or not; --reproj-thresh only feeds the on-screen 'far' counter)."""
    import cv2
    gids = sorted(int(g) for g in edg_tracks.keys())
    mx = max(gids) if gids else 0
    if mx >= ctx["Q"]:
        raise ValueError(f"TRACK-ID SPACE MISMATCH: npz node ids reach {mx} but the 2D tracks have "
                         f"Q={ctx['Q']}; add/drop --merged to match how the npz was solved.")
    counts = {g: int(ctx["valid"][g].sum()) for g in gids}             # appearance length = observed frames
    rank_order = sorted(range(len(gids)), key=lambda i: -counts[gids[i]])  # local idx by length, desc
    n_rank = len(rank_order)
    rank0 = int(np.clip(args.track_rank, 0, n_rank))                   # 0 = all tracks; else x-th longest
    P = np.stack([np.asarray(edg_tracks[g], np.float64) for g in gids])   # [N,T',3]
    T = min(ctx["T"], P.shape[1])
    gid_arr = np.asarray(gids, dtype=int)

    def render(f, rank=0):
        img = cv2.imread(ctx["image_paths"][f]) if f < len(ctx["image_paths"]) else None
        if img is None:
            img = np.zeros((480, 360, 3), np.uint8)
        n_pair = n_far = n_obs = 0
        u, v, ok_p = project_to_frame(P[:, f], ctx, f)                 # row form of R^T (x - t) / s + pinhole
        obs = ctx["xy"][gid_arr, f]                                    # (N,2) initial observations
        ok_o = ctx["valid"][gid_arr, f] & np.isfinite(obs).all(1)      # same `valid` the solver used
        active = range(gid_arr.size) if rank <= 0 else [rank_order[rank - 1]]
        for i in active:
            has_p = bool(ok_p[i]) and np.isfinite(u[i])
            if has_p:
                pp = (int(round(u[i])), int(round(v[i])))
                cv2.circle(img, pp, 2, (0, 0, 255), -1)               # projected EDG: red (BGR)
            if ok_o[i]:                                               # observation drawn INDEPENDENTLY of
                po = (int(round(obs[i, 0])), int(round(obs[i, 1])))   # whether the node is solved this frame
                cv2.circle(img, po, 2, (0, 255, 0), -1)               # initial observation: green
                n_obs += 1
                if has_p:
                    n_pair += 1
                    cv2.line(img, pp, po, (0, 0, 255), 1)             # join projected <-> observation
                    if np.hypot(u[i] - obs[i, 0], v[i] - obs[i, 1]) > args.reproj_thresh:
                        n_far += 1                                    # threshold only feeds the counter
        s = float(args.viz_2d_scale)
        if s != 1.0:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        if rank > 0:
            gi = rank_order[rank - 1]
            sel_txt = f"rank {rank}/{n_rank}: gid={int(gid_arr[gi])} ({counts[int(gid_arr[gi])]}f)"
        else:
            sel_txt = "all tracks"
        cv2.putText(img, f"frame {f}/{T - 1}  [{sel_txt}]  obs={n_obs} pairs={n_pair} "
                    f"far(>{args.reproj_thresh:.0f}px)={n_far}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        print(f"[2d] frame={f} [{sel_txt}] obs={n_obs} pairs={n_pair} far={n_far}")
        return img

    f = int(np.clip(start_frame, 0, T - 1))
    rank = rank0
    if args.dry_run:
        out = Path(utils.definitions.output_dir) / f"reproj2d_frame{f:04d}.png"
        cv2.imwrite(str(out), render(f, rank))
        print(f"[2d][dry-run] wrote {out}")
        return
    print("[2d] SPACE=next frame, B=prev frame | N=next-longest track, P=prev (longer), A=all tracks | "
          "Q/ESC=quit. red dot=projected EDG, green dot=initial 2D observation, red line joins the pair.")
    while True:
        cv2.imshow("EDG 2D reprojection", render(f, rank))
        k = cv2.waitKey(0) & 0xFF
        if k in (ord("q"), ord("Q"), 27):
            break
        if k in (ord("b"), ord("B")):
            f = (f - 1) % T
        elif k in (ord("n"), ord("N")):
            rank = min(rank + 1, n_rank)                              # 0=all -> 1=longest -> ... shortest
            if rank > 0:                                               # jump to the track's first observed
                gi = int(gid_arr[rank_order[rank - 1]])                # frame so it is actually on screen
                fo = np.where(ctx["valid"][gi])[0]
                f = int(fo[0]) if fo.size else f
        elif k in (ord("p"), ord("P")):
            rank = max(rank - 1, 0)
            if rank > 0:
                gi = int(gid_arr[rank_order[rank - 1]])
                fo = np.where(ctx["valid"][gi])[0]
                f = int(fo[0]) if fo.size else f
        elif k in (ord("a"), ord("A")):
            rank = 0
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

    edg_test_mode = ("node_ids" in data.files and "dyn_reconstruct" in data.files)
    node_ids = None
    if edg_test_mode:
        node_ids = np.asarray(data["node_ids"], dtype=np.int64)
        er = np.asarray(data["edg_reconstruct"], dtype=np.float64)          # [Nn,T,3]
        if args.fill_edg:
            er = _hold_fill_stack(er)
            print(f"[edg_test] --fill-edg: held {int(node_ids.size)} EDG nodes across the whole "
                  f"{er.shape[1]}-frame interval.")
        edg_tracks = {int(node_ids[i]): er[i] for i in range(int(node_ids.size))}
    else:
        edg_tracks = _load_dict(data, "edg_reconstruct")
        if edg_tracks is None:
            raise RuntimeError(f"Missing EDG tracks (edg_reconstruct) in {npz_path}.")

    # 2D context: pick the right tracks table by the npz's n_tracks stamp (auto-switch is in _load_2d_context)
    stamp = int(data["n_tracks"]) if "n_tracks" in data.files else None
    if stamp is None and node_ids is not None:                         # unstamped npz: fall back to the
        topo_p = Path(utils.definitions.output_dir) / "edg_topology.npz"    # topology's stamp when the
        if topo_p.exists():                                            # node ids come from that topology
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

    max_frames = max(int(arr.shape[0]) for arr in edg_tracks.values()) if edg_tracks else 0
    if max_frames == 0:
        raise RuntimeError("No EDG tracks found to visualize.")

    run_viz_2d(edg_tracks, ctx, args, start_frame=int(np.clip(args.frame, 0, max_frames - 1)))


if __name__ == "__main__":
    main()
