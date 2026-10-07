#!/usr/bin/env python3
"""MoSca-style showcase GRID VIDEO for a trained 4DGS scene, ready for a website.

Layout: rows = fixed observation poses (default 3 training cameras held fixed while scene time
advances), columns = modalities (default 5): RGB render | depth | 3D tracks | EDG graph | dynamic-
only render. The video plays the full time horizon (looped) so the dynamic object moves while each
row's viewpoint stays fixed.

Reuses the exact render stack of estimate_ours_render_tto (same gaussians, EDG warp, appearance
module), so what you see is what gets evaluated.

Example:
  conda run -n gsplat python visualize_showcase_grid.py \
      --scene-dir ../datasets/NVIDIA/Truck \
      --gaussians ../datasets/NVIDIA/Truck/outputs/gsplat_tn_appreg3/ckpts/final_rank0.pt \
      --cams 0,4,8 --out ../datasets/NVIDIA/Truck/outputs/showcase_grid.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from eval_depth_gaussian_ours import (
    _build_app_module,
    _load_camera_and_gt_intrinsics,
    _load_gaussians,
    _resolve_gaussian_paths,
)
from estimate_ours_render import _scale_K_for_size
from estimate_ours_render_tto import _prep_frozen_gaussians, _render_pose
from gsplat.rendering import rasterization


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-dir", type=Path, default=None,
                    help="scene root (default: utils.definitions rootpath, i.e. XM_DATASET_CAT/NAME)")
    ap.add_argument("--gaussians", type=Path, default=None,
                    help="trained ckpt (default: outputs/gsplat_edg_target_0000 -- the pipeline's "
                         "default result dir -- else newest outputs/gsplat*/ckpts/final_rank0.pt)")
    ap.add_argument("--nvidia", action="store_true",
                    help="NVIDIA fixed-camera rendering: split appearance -- static Gaussians use "
                         "the FIRST-frame embedding (matches a fixed physical camera), dynamic "
                         "Gaussians keep the frame-t embedding (same as eval --novel-app split0)")
    ap.add_argument("--out", type=Path, default=None, help="output mp4 (default outputs/showcase_grid.mp4)")
    ap.add_argument("--cams", default="0,4,8", help="comma list of training-frame indices used as FIXED poses (rows)")
    ap.add_argument("--cols", default="render,depth,tracks,edg,dyn",
                    help="comma list of columns from: render,depth,tracks,edg,dyn")
    ap.add_argument("--cell-width", type=int, default=480, help="per-cell width px (height keeps aspect)")
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--loops", type=int, default=1, help="repeat the forward pass this many times")
    ap.add_argument("--n-tracks", type=int, default=60, help="tracks subsampled for the track column")
    ap.add_argument("--trail", type=int, default=10, help="track window length (frames) in the track column")
    ap.add_argument("--sh-degree", type=int, default=1)
    ap.add_argument("--near-plane", type=float, default=0.01)
    ap.add_argument("--far-plane", type=float, default=1e10)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def project(pts_w: np.ndarray, w2c: np.ndarray, K: np.ndarray):
    """(N,3) world -> (N,2) px + in-front mask."""
    pc = pts_w @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    ok = z > 1e-6
    uv = np.full((pts_w.shape[0], 2), np.nan, np.float32)
    uv[ok, 0] = K[0, 0] * pc[ok, 0] / z[ok] + K[0, 2]
    uv[ok, 1] = K[1, 1] * pc[ok, 1] / z[ok] + K[1, 2]
    return uv, ok


def track_colors(n):
    hsv = np.zeros((n, 1, 3), np.uint8)
    hsv[:, 0, 0] = (np.arange(n) * 137) % 180
    hsv[:, 0, 1] = 255
    hsv[:, 0, 2] = 255
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[:, 0, :]


def label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    args = parse_args()
    if args.scene_dir is None:
        from utils import definitions
        args.scene_dir = Path(definitions.rootpath)
    scene_dir = args.scene_dir.resolve()
    if args.gaussians is None:
        # default ckpt: freshest pipeline result dir first (gsplat_newedg), then the default
        # result dir, else newest final_rank0.pt. The ckpt must match the CURRENT results_all.npz
        # (EDG row count) or the EDG/track columns degrade.
        cands = [p for d in ("gsplat_newedg", "gsplat_edg_target_0000")
                 for p in (scene_dir / "outputs" / d / "ckpts" / "final_rank0.pt",
                           scene_dir / "outputs" / d / "final_gaussians.pt") if p.is_file()]
        if not cands:
            cands = sorted(scene_dir.glob("outputs/gsplat*/ckpts/final_rank0.pt"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
        if cands:
            args.gaussians = cands[0]
            print(f"[showcase] default gaussians: {args.gaussians}")
    out_path = (args.out or scene_dir / "outputs" / "showcase_grid.mp4").resolve()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cams = [int(c) for c in args.cams.split(",")]
    cols = [c.strip() for c in args.cols.split(",") if c.strip()]

    paths = _resolve_gaussian_paths(scene_dir, args.gaussians)
    splats, edg, app_state, _ = _load_gaussians(paths, device)
    app_module = _build_app_module(app_state, splats, device)
    rargs = SimpleNamespace(no_edg_warp=False, edg_warp_quats=False, edg_target_frame=0,
                            novel_app="split0" if args.nvidia else "base", sh_degree=args.sh_degree,
                            near_plane=args.near_plane, far_plane=args.far_plane, antialiased=False)
    if args.nvidia:
        print("[showcase] --nvidia: split appearance (static -> frame-0 embedding, dynamic -> frame-t)")
    T = int(edg["edg_tracks"].shape[1]) if edg is not None and "edg_tracks" in edg else 12
    print(f"[showcase] T={T} horizon, {len(cams)} poses x {len(cols)} columns")

    # EDG graph = the TRAINED node tables from the checkpoint (edg_tracks: what the model actually
    # uses at render time), with edge topology from results_all.npz (trainer keeps sorted-gid order).
    ra = np.load(scene_dir / "outputs" / "results_all.npz", allow_pickle=True)
    node_pos = edge_rows = None
    if edg is not None and "edg_tracks" in edg:
        node_pos = edg["edg_tracks"].detach().cpu().numpy().astype(np.float64)          # (M,T,3) trained
        if "edg_reconstruct" in ra.files and "edg_edges_union" in ra.files:
            er = ra["edg_reconstruct"].item()
            gids = sorted(er.keys())
            if len(gids) != node_pos.shape[0]:
                print(f"[showcase] WARN: EDG edges SKIPPED -- checkpoint has {node_pos.shape[0]} EDG nodes "
                      f"but results_all.npz has {len(gids)}: the ckpt was trained against a DIFFERENT "
                      f"results_all (stale/newer EDG). Pass --gaussians for the ckpt matching this "
                      f"results_all (e.g. outputs/gsplat_newedg/ckpts/final_rank0.pt).")
            if len(gids) == node_pos.shape[0]:
                # sanity: trained nodes should sit near the solved init (drift is small)
                init0 = np.stack([np.asarray(er[g], np.float64)[0] for g in gids], 0)
                fin = np.isfinite(init0).all(1) & np.isfinite(node_pos[:, 0]).all(1)
                if fin.any():
                    d = np.linalg.norm(init0[fin] - node_pos[fin, 0], axis=1)
                    if np.median(d) > 0.5:
                        print(f"[showcase] WARN: ckpt EDG order may mismatch results_all "
                              f"(median t0 offset {np.median(d):.3f})")
                loc = {int(g): i for i, g in enumerate(gids)}
                edge_rows = np.array([[loc[a], loc[b]] for a, b in ra["edg_edges_union"].tolist()
                                      if a in loc and b in loc], np.int64)

    # 3D tracks column = motion of the TRAINED dynamic Gaussians (subsample, warped per frame
    # through the model's own EDG skinning -- exactly what moves in the render).
    from eval_depth_gaussian_ours import _warp_means_edg
    tr_pts = None
    if edg is not None and edg.get("dyn_indices") is not None:
        di = edg["dyn_indices"].long()
        di = di[(di >= 0) & (di < splats["means"].shape[0])]
        if di.numel() > 0:
            sel_i = di[torch.linspace(0, di.numel() - 1, min(args.n_tracks, di.numel()),
                                      device=di.device).long()]
            per_t = []
            with torch.no_grad():
                for t in range(int(edg["edg_tracks"].shape[1])):
                    warped = _warp_means_edg(splats["means"].float(), edg, t)
                    per_t.append(warped[sel_i].detach().cpu().numpy())
            tr_pts = np.stack(per_t, axis=1).astype(np.float64)                          # (n,T,3)
            tr_col = track_colors(tr_pts.shape[0])
            print(f"[showcase] tracks column: {tr_pts.shape[0]} trained dynamic Gaussians")

    # dynamic-gaussian mask for the dyn-only column
    dyn_mask_t = None
    if edg is not None and "dyn_indices" in edg:
        di = edg["dyn_indices"].long()
        di = di[(di >= 0) & (di < splats["means"].shape[0])]
        dyn_mask_t = torch.zeros(splats["means"].shape[0], dtype=torch.bool, device=device)
        dyn_mask_t[di] = True
        print(f"[showcase] dynamic gaussians: {int(dyn_mask_t.sum())}/{splats['means'].shape[0]}")

    # fixed cameras (training colmap poses + K, scaled to cell size)
    cam_data = []
    for c in cams:
        c2w, K, _, _, _ = _load_camera_and_gt_intrinsics(scene_dir, c, "colmap")
        img0 = sorted((scene_dir / "outputs" / "gs_edg_target_0000" / "images").glob("*"))[0]
        H0, W0 = cv2.imread(str(img0)).shape[:2]
        W = int(args.cell_width); H = int(round(W * H0 / W0 / 2) * 2)
        Ks = _scale_K_for_size(np.asarray(K, np.float64), (H0, W0), (H, W))
        cam_data.append((np.asarray(c2w, np.float64), Ks, H, W))

    # depth range for colorization (probe pass at t=0, cam 0)
    def render_cell(gs, w2c_t, K_t, H, W, mode):
        if mode == "depth":
            renders, _, _ = rasterization(
                means=gs["means"], quats=gs["quats"], scales=gs["scales"],
                opacities=gs["opacities"], colors=gs["colors"], viewmats=w2c_t[None], Ks=K_t[None],
                width=W, height=H, sh_degree=gs["sh_degree"], near_plane=args.near_plane,
                far_plane=args.far_plane, render_mode="RGB+ED")
            return renders[0, ..., -1].detach().cpu().numpy()
        rgb, _ = _render_pose(gs, w2c_t, K_t, H, W, rargs)
        return np.ascontiguousarray(
            (rgb.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)[..., ::-1])  # BGR

    dmin = dmax = None
    frames = []
    for t in range(T):
        row_imgs = []
        for (c2w, K, H, W) in cam_data:
            w2c = np.linalg.inv(np.vstack([np.hstack([c2w[:3, :3], c2w[:3, 3:4]]), [[0, 0, 0, 1]]]))
            w2c_t = torch.from_numpy(w2c.astype(np.float32)).to(device)
            K_t = torch.from_numpy(K.astype(np.float32)).to(device)
            gs = _prep_frozen_gaussians(splats, edg, app_module, t, c2w, rargs)
            base_rgb = render_cell(gs, w2c_t, K_t, H, W, "render")
            cells = []
            for mode in cols:
                if mode == "render":
                    cells.append(label(base_rgb.copy(), f"render  t={t}"))
                elif mode == "depth":
                    d = render_cell(gs, w2c_t, K_t, H, W, "depth")
                    if dmin is None:
                        dmin, dmax = np.percentile(d, 2), np.percentile(d, 98)
                    dn = np.clip((d - dmin) / max(dmax - dmin, 1e-6), 0, 1)
                    cells.append(label(cv2.applyColorMap((dn * 255).astype(np.uint8),
                                                         cv2.COLORMAP_TURBO), "depth"))
                elif mode == "tracks" and tr_pts is not None:
                    # windowed tracks (last --trail frames) as flat solid polylines
                    im = (base_rgb * 0.55).astype(np.uint8)
                    k0 = max(0, t - args.trail)
                    for i in range(tr_pts.shape[0]):
                        seg = tr_pts[i, k0:t + 1]                       # (w,3)
                        uv, ok = project(seg, w2c, K)
                        col = tuple(int(v) for v in tr_col[i])
                        prev = None
                        for j in range(seg.shape[0]):
                            if ok[j] and np.isfinite(uv[j]).all():
                                pt = (int(uv[j, 0]), int(uv[j, 1]))
                                if prev is not None:
                                    cv2.line(im, prev, pt, col, 2, cv2.LINE_AA)
                                prev = pt
                            else:
                                prev = None                             # break line at gaps
                        if prev is not None:
                            cv2.circle(im, prev, 3, col, -1)            # current position head
                    cells.append(label(im, "3D tracks"))
                elif mode == "edg" and node_pos is not None:
                    im = (base_rgb * 0.35).astype(np.uint8)
                    uv, ok = project(node_pos[:, t], w2c, K)
                    if edge_rows is not None:
                        for a_, b_ in edge_rows:
                            if ok[a_] and ok[b_]:
                                p1 = (int(uv[a_, 0]), int(uv[a_, 1])); p2 = (int(uv[b_, 0]), int(uv[b_, 1]))
                                cv2.line(im, p1, p2, (80, 200, 80), 1, cv2.LINE_AA)
                    for i in np.where(ok)[0]:
                        x, y = int(uv[i, 0]), int(uv[i, 1])
                        if 0 <= x < W and 0 <= y < H:
                            cv2.circle(im, (x, y), 3, (60, 60, 230), -1)
                    cells.append(label(im, "EDG graph"))
                elif mode == "dyn" and dyn_mask_t is not None:
                    gd = dict(gs)
                    gd["opacities"] = gs["opacities"] * dyn_mask_t.float()
                    cells.append(label(render_cell(gd, w2c_t, K_t, H, W, "render"), "dynamic only"))
                else:
                    cells.append(label(np.zeros((H, W, 3), np.uint8), f"{mode} (n/a)"))
            row_imgs.append(np.concatenate(cells, axis=1))
        frames.append(np.concatenate(row_imgs, axis=0))
        print(f"[showcase] frame t={t} done", flush=True)

    seq = frames * max(1, args.loops)                   # single forward pass per loop (no pingpong)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = seq[0].shape[:2]
    vw = cv2.VideoWriter(str(out_path.with_suffix(".avi")), cv2.VideoWriter_fourcc(*"MJPG"), args.fps, (w, h))
    for f in seq:
        vw.write(f)
    vw.release()
    # website-ready H.264 yuv420p via ffmpeg
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out_path.with_suffix(".avi")),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
                    str(out_path)], check=True)
    out_path.with_suffix(".avi").unlink(missing_ok=True)
    print(f"[showcase] saved -> {out_path}  ({w}x{h}, {len(seq)} frames @ {args.fps}fps)")


if __name__ == "__main__":
    main()
