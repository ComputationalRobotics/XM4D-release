#!/usr/bin/env python3
"""Project SOLVED EDG results into the image and lay several of them side by side for comparison.

Unlike edg_rigid_parts.py --viz (which draws each node at its 2D TRACK position), this projects the solved
3D node positions through the solver cameras, so what you see IS the solve -- a bad node lands off the
object. Nodes are dots, EDG edges are lines; each panel is labelled with its file and node/edge counts.

Run:  conda run -n XM python visualize_edg_proj2d.py --frame 100 \
          --npz v0=<...>/bk/edg_completed.npz V0=<...>/ablation/completed_V0.npz ...
      (bare paths work too; the label then defaults to the file stem)
      --video renders ALL frames to an mp4 instead of one PNG.
      --tracks <file ...> also draws each node's 2D observation (green dot), a yellow line from it to the
      projected node (the reprojection residual the completion's L_p2d term fits), and the per-frame
      median/max residual in the header. One entry broadcasts; otherwise one per panel, matching the file
      each solve was fitting (node ids index into it).
NOTE: run under the same XM_DEPTH_VARIANT the solves used (intrinsics live under depths/<variant>/).
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np

import utils.definitions as D


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", nargs="+", required=True, help="label=path (or just path) for each panel")
    ap.add_argument("--tracks", nargs="+", default=None,
                    help="tracks npz per panel (one entry broadcasts): draws each node's 2D OBSERVATION "
                         "(green), a yellow residual line to the projected node, and per-frame median/max "
                         "residual px in the header. Use the SAME file the solve was fitting.")
    ap.add_argument("--frame", type=int, default=100, help="frame for the single-PNG mode")
    ap.add_argument("--video", action="store_true", help="render every frame into an mp4")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cols", type=int, default=0, help="panels per row (0 = one row)")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--dot", type=int, default=5)
    ap.add_argument("--pairs-only", action="store_true",
                    help="with --tracks: draw ONLY the observation<->projected-node pairs (green dot, "
                         "yellow residual line, red dot) -- no EDG edges, no unobserved nodes")
    a = ap.parse_args()

    from build_edg import _load_all, _frame_scale
    from train_4dgs_edg import _proj_uv_z
    DD = _load_all()
    K = np.asarray(np.load(str(D.active_intrinsics)), np.float64)
    T = DD["T"]
    K = K if K.ndim == 3 else np.repeat(K[None], T, 0)
    images = sorted(glob.glob(str(Path(D.video_dir) / D.image_pattern)))
    T = min(T, len(images))

    tr_specs = None
    if a.tracks:
        tr_specs = a.tracks if len(a.tracks) > 1 else a.tracks * len(a.npz)
        if len(tr_specs) != len(a.npz):
            raise SystemExit(f"--tracks: give 1 entry or {len(a.npz)} (one per --npz)")
        tr_cache = {}

    sets = []
    for k, spec in enumerate(a.npz):
        lab, _, p = spec.partition("=")
        if not p:
            p, lab = lab, Path(lab).stem
        d = np.load(p, allow_pickle=True)
        ids = np.asarray(d["node_ids"]).reshape(-1)
        pos = {int(n): i for i, n in enumerate(ids.tolist())}
        e = [(pos[int(x)], pos[int(y)]) for x, y in np.asarray(d["edg_edges"], np.int64).reshape(-1, 2)
             if int(x) in pos and int(y) in pos]
        obs = None
        if tr_specs:
            tp = tr_specs[k]
            if tp not in tr_cache:
                td = np.load(tp if "/" in tp else str(Path(D.output_dir) / tp), allow_pickle=True)
                xy = np.asarray(td["tracks_xy"], np.float64)[0]
                ok = ((~np.asarray(td["occluded"])[0]) & np.asarray(td["dynamic_obs_mask"]).astype(bool)
                      & np.isfinite(xy).all(-1)
                      & np.isfinite(np.asarray(td["tracks_xyz"], np.float64)[0]).all(-1))
                tr_cache[tp] = (xy, ok)
            xy, ok = tr_cache[tp]
            obs = (np.where(ok[ids][..., None], xy[ids], np.nan))      # [M,T,2] NaN where not observed
        sets.append(dict(lab=lab, c=np.asarray(d["edg_reconstruct"], np.float64), e=e, n=len(ids), obs=obs))
        print(f"[proj2d] {lab}: {len(ids)} nodes, {len(e)} edges  ({p})"
              + (f"  obs-cells={int(np.isfinite(obs).all(-1).sum())}" if obs is not None else ""))

    def panel(S, t, img):
        img = img.copy()
        c = S["c"][:, t, :]
        ok = np.isfinite(c).all(1)
        uv = np.full((len(c), 2), np.nan)
        if ok.any():
            u, v, z = _proj_uv_z(c[ok], np.asarray(DD["R_raw"][t], np.float64)[:3, :3], DD["t_raw"][t],
                                 K[min(t, len(K) - 1)])
            w = np.stack([u, v], 1); w[~(z > 1e-6)] = np.nan
            uv[ok] = w
        if not a.pairs_only:
            for i1, i2 in S["e"]:
                if np.isfinite(uv[[i1, i2]]).all():
                    cv2.line(img, tuple(np.round(uv[i1]).astype(int)), tuple(np.round(uv[i2]).astype(int)),
                             (60, 60, 60), 1, cv2.LINE_AA)
        res = []
        paired = np.zeros(len(c), bool)
        if S["obs"] is not None:                                       # residual lines FIRST, under the dots
            for i in range(len(c)):
                o = S["obs"][i, t]
                if np.isfinite(o).all() and np.isfinite(uv[i]).all():
                    paired[i] = True
                    res.append(float(np.hypot(*(uv[i] - o))))
                    cv2.line(img, tuple(np.round(uv[i]).astype(int)), tuple(np.round(o).astype(int)),
                             (0, 230, 255), 2, cv2.LINE_AA)
                    cv2.circle(img, tuple(np.round(o).astype(int)), max(2, a.dot - 2), (60, 220, 60), -1,
                               cv2.LINE_AA)
        drawn = 0
        for i in range(len(c)):
            if a.pairs_only and not paired[i]:
                continue
            if np.isfinite(uv[i]).all():
                q = tuple(np.round(uv[i]).astype(int))
                cv2.circle(img, q, a.dot, (40, 40, 240), -1, cv2.LINE_AA)
                cv2.circle(img, q, a.dot, (255, 255, 255), 1, cv2.LINE_AA)
                drawn += 1
        head = 46
        pnl = np.zeros((img.shape[0] + head, img.shape[1], 3), np.uint8)
        pnl[head:] = img
        cv2.putText(pnl, S["lab"], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        if S["obs"] is not None:                                       # compact: the residual is the point
            sub = (f"t={t}  d: med {np.median(res):.1f} max {max(res):.0f}px ({len(res)}/{S['n']} obs)"
                   if res else f"t={t}  no obs this frame")
        else:
            sub = f"{S['n']} nodes ({drawn} proj)  {len(S['e'])} edges  t={t}"
        cv2.putText(pnl, sub, (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1, cv2.LINE_AA)
        return pnl

    def canvas(t):
        img = cv2.imread(images[t])
        panels = [panel(S, t, img) for S in sets]
        cols = a.cols or len(panels)
        rows = [np.concatenate(panels[i:i + cols], 1) for i in range(0, len(panels), cols)]
        w = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
        out = np.concatenate(rows, 0)
        if a.scale != 1.0:
            out = cv2.resize(out, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
        return out

    if a.video:
        out = Path(a.out) if a.out else Path(D.output_dir) / "edg_proj2d.mp4"
        first = canvas(0)
        vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), a.fps,
                             (first.shape[1], first.shape[0]))
        vw.write(first)
        for t in range(1, T):
            vw.write(canvas(t))
        vw.release()
        print(f"[proj2d] wrote {out}  ({T} frames, {first.shape[1]}x{first.shape[0]})")
    else:
        t = int(np.clip(a.frame, 0, T - 1))
        out = Path(a.out) if a.out else Path(D.output_dir) / f"edg_proj2d_frame{t:04d}.png"
        cv2.imwrite(str(out), canvas(t))
        print(f"[proj2d] wrote {out}")


if __name__ == "__main__":
    main()
