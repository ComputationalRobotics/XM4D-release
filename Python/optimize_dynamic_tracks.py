#!/usr/bin/env python3
"""Smooth the 3D dynamic tracks after merging, before EDG.

Reads a merged dynamic-tracks npz for its 2D observations only (tracks_xy, occluded, dynamic_obs_mask) -- the
merge is 2D-only, so the 3D is (re)LIFTED HERE from the depth selected in utils.definitions
(definitions.depth_variant -> active_depth_maps, default 'aligned_da3'; override with XM_DEPTH_VARIANT). It lifts
each 2D obs to camera frame from that depth, then to WORLD via the solved camera poses (EDG_old.py:104:
X_world = s*R @ x_cam + t), then optimizes each track's per-frame WORLD position with three losses (weights
DECREASE, reproj most important):
  1. reprojection: project(X_t) close to the 2D observation  -> keeps X on the camera ray   (--w-reproj)
  2. smoothness  : ||X_t - X_{t-1}||^2 small                  -> temporal smoothness          (--w-smooth)
  3. prior       : ||X_t - X_init_t||^2 small                 -> stay near the lifted init     (--w-prior)
Finally converts the smoothed WORLD tracks back to per-frame CAMERA frame (inverse of EDG_old.py:104:
x_cam = (1/s) R^T (X_world - t)) and writes tracks_dynamic_smoothed.npz (same schema; tracks_xy/occluded/
dynamic_obs_mask copied from source). EDG then consumes it via  EDG_TRACKS_PATH=.../tracks_dynamic_smoothed.npz.

Scene is selected by XM_DATASET_CAT / XM_DATASET_NAME (via utils.definitions).

Example:
  XM_DATASET_CAT=Dycheck XM_DATASET_NAME=wheel conda run -n XM python optimize_dynamic_tracks.py \
    --input tracks_dynamic_merged.npz --w-reproj 1.0 --w-smooth 5.0 --w-prior 0.1
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import utils.definitions as D
from utils.edg import scatter_solved_poses, scatter_solved_frame_vector


def load_eroded_masks(mask_dir, pattern, T, hw, erode_px):
    """(T,H,W) bool: True where a pixel is >= erode_px INSIDE the dynamic (white>127) mask. Missing masks or
    erode_px<=0 -> all True (no boundary drop). Used to drop observations near the mask boundary, where a
    tracked dynamic point can sample background depth (often wrong / very large)."""
    import cv2
    import glob as _glob
    H, W = hw
    out = np.ones((T, H, W), bool)
    files = sorted(_glob.glob(str(Path(mask_dir) / pattern)))
    if not files or erode_px <= 0:
        return out
    ker = np.ones((2 * int(erode_px) + 1, 2 * int(erode_px) + 1), np.uint8)
    for t in range(min(T, len(files))):
        m = cv2.imread(files[t], cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        out[t] = cv2.erode((m > 127).astype(np.uint8), ker) > 0
    return out


def lift_camera_xyz(txy, occ, depth, K, masks_ok=None, depth_grad_thresh=0.0):
    """(Q,T,3) camera-frame lift of the 2D tracks from `depth` + per-frame K -- the 3D is fully (re)built HERE
    from the specified depth (definitions.active_depth_maps), matching dynamic_tracking.py:1089 / relift_tracks.py.
    NaN where occluded or depth invalid (<=1e-3). Also DROPS an observation (leaves it NaN) when:
      (a) masks_ok is given and the pixel is NOT in the eroded dynamic mask (i.e. within erode_px of the boundary),
      (b) depth_grad_thresh>0 and the valid 3x3 depth range exceeds depth_grad_thresh * center depth (discontinuity).
    Returns (xyz, n_drop_boundary, n_drop_grad)."""
    Q, T = txy.shape[:2]; Td, H, W = depth.shape
    xyz = np.full((Q, T, 3), np.nan, np.float64)
    n_bnd = n_grad = 0
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
    for t in range(min(T, Td)):
        vis = np.where(~occ[:, t])[0]
        if vis.size == 0:
            continue
        u = txy[vis, t, 0]; v = txy[vis, t, 1]
        ui = np.clip(np.round(u).astype(int), 0, W - 1); vi = np.clip(np.round(v).astype(int), 0, H - 1)
        z = depth[t, vi, ui]; ok = z > 1e-3
        if masks_ok is not None:                                     # (a) drop near-mask-boundary obs
            bnd = masks_ok[t, vi, ui]; n_bnd += int((ok & ~bnd).sum()); ok = ok & bnd
        if depth_grad_thresh and depth_grad_thresh > 0:              # (b) drop 3x3 depth-discontinuity obs
            patch = np.stack([depth[t, np.clip(vi + dy, 0, H - 1), np.clip(ui + dx, 0, W - 1)] for dy, dx in offs], 1)
            patch = np.where(patch > 1e-3, patch, np.nan)
            with np.errstate(invalid="ignore"):
                rng = np.nanmax(patch, 1) - np.nanmin(patch, 1)
            grad_ok = np.nan_to_num(rng, nan=np.inf) <= depth_grad_thresh * np.where(z > 1e-3, z, 0.0)
            n_grad += int((ok & ~grad_ok).sum()); ok = ok & grad_ok
        vis, u, v, z = vis[ok], u[ok], v[ok], z[ok]; k = K[min(t, K.shape[0] - 1)]
        xyz[vis, t, 0] = (u - k[0, 2]) / k[0, 0] * z
        xyz[vis, t, 1] = (v - k[1, 2]) / k[1, 1] * z
        xyz[vis, t, 2] = z
    return xyz, n_bnd, n_grad


def optimize(X_init, vis, obs, R_c2w, t_c2w, K, w_re, w_sm, w_pr, iters, lr, device):
    """World-frame batched optimize. X_init (Q,T,3) world; vis (Q,T); obs (Q,T,2); R_c2w (T,3,3); t_c2w (T,3); K (T,3,3)."""
    dev = torch.device(device)
    Xi = torch.tensor(np.nan_to_num(X_init.astype(np.float32)), device=dev)
    X = Xi.clone().requires_grad_(True)
    ob = torch.nan_to_num(torch.tensor(obs.astype(np.float32), device=dev))
    m = torch.tensor(vis.astype(np.float32), device=dev)
    Rw2c = torch.tensor(np.nan_to_num(np.transpose(R_c2w, (0, 2, 1))).astype(np.float32), device=dev)  # world->cam
    C = torch.tensor(np.nan_to_num(t_c2w).astype(np.float32), device=dev)                 # dropped frames -> NaN t; zero it (masked anyway)
    fx = torch.tensor(K[:, 0, 0].astype(np.float32), device=dev); fy = torch.tensor(K[:, 1, 1].astype(np.float32), device=dev)
    cx = torch.tensor(K[:, 0, 2].astype(np.float32), device=dev); cy = torch.tensor(K[:, 1, 2].astype(np.float32), device=dev)
    sm = m[:, 1:] * m[:, :-1]; denom = m.sum().clamp(min=1); sdenom = sm.sum().clamp(min=1)
    opt = torch.optim.Adam([X], lr=lr)
    rmse = float("nan")
    Q_, nobs = X.shape[0], int(m.sum().item())                       # tracks solved / visible 3D obs solved
    pbar = tqdm(range(iters), desc=f"solving {Q_} tracks / {nobs} obs", unit="it")
    for it in pbar:
        opt.zero_grad()
        Xc = torch.einsum("tij,qtj->qti", Rw2c, X - C[None])
        z = Xc[..., 2].clamp(min=0.05)
        u = fx[None] * Xc[..., 0] / z + cx[None]; v = fy[None] * Xc[..., 1] / z + cy[None]
        err = torch.nan_to_num(((u - ob[..., 0]) ** 2 + (v - ob[..., 1]) ** 2).clamp(max=1e4))
        re = (err * m).sum() / denom
        smL = (((X[:, 1:] - X[:, :-1]) ** 2).sum(-1) * sm).sum() / sdenom
        pr = (((X - Xi) ** 2).sum(-1) * m).sum() / denom
        (w_re * re + w_sm * smL + w_pr * pr).backward()
        torch.nn.utils.clip_grad_norm_([X], 1.0)
        opt.step()
        if it % 20 == 0 or it == iters - 1:                          # cheap postfix (avoid a GPU sync every step)
            rmse = float(re.detach().sqrt()); pbar.set_postfix(reproj=f"{rmse:.2f}px")
    pbar.close()
    return X.detach().cpu().numpy(), rmse


def visualize(tracks, vis, R_for, t_raw, K, args):
    """Open3D: static scene (image-colored, ALL frames at once) + `--viz-tracks` sampled dynamic tracks over a
    trailing `--viz-window` frame window. `tracks` is an ordered dict {name: (Q,T,3) WORLD positions}; the T key
    cycles between them (e.g. before vs after optimize). R_for=s*R and t_raw are the world-lift transform used
    for the dynamic tracks (so static aligns). Controls: SPACE=next, B=prev, T=toggle track set."""
    import open3d as o3d
    import matplotlib.cm as cm
    from PIL import Image
    O = Path(D.output_dir); Q, T = vis.shape
    depth = np.load(Path(D.active_depth_maps)); H, W = depth.shape[1:]   # static backdrop uses the SAME depth
    names = list(tracks.keys())

    live = np.where(vis.any(1))[0]                                   # only tracks with >=1 visible obs
    n = min(args.viz_tracks, live.size)
    sel = live if live.size <= n else np.random.RandomState(0).choice(live, n, replace=False)
    print(f"  [viz] {sel.size} dynamic tracks (window={args.viz_window}); sets: {names}")

    # static backdrop: lift each static track at its middle visible frame, colour from the image (all at once)
    static_pcd = o3d.geometry.PointCloud()
    sp = O / "output_tapnet_da3.npz"
    if sp.exists():
        stat = np.load(sp, allow_pickle=True); sxy = stat["tracks_xy"][0]; socc = stat["occluded"][0]; _imgc = {}
        def img(t):
            if t not in _imgc:
                _imgc[t] = np.asarray(Image.open(D.video_dir / f"frame{t:06d}.png").convert("RGB")) / 255.0
            return _imgc[t]
        spts, scol = [], []
        for i in np.random.RandomState(0).permutation(sxy.shape[0]):
            if len(spts) >= args.viz_static:
                break
            vf = np.where(~socc[i])[0]
            if vf.size == 0:
                continue
            t = int(vf[vf.size // 2]); u, v = sxy[i, t]
            ui, vi = int(np.clip(round(u), 0, W - 1)), int(np.clip(round(v), 0, H - 1))
            d = depth[t, vi, ui]
            if d <= 1e-3:
                continue
            k = K[t]; Xc = np.array([(u - k[0, 2]) / k[0, 0] * d, (v - k[1, 2]) / k[1, 1] * d, d])
            spts.append(R_for[t] @ Xc + t_raw[t]); scol.append(img(t)[vi, ui])
        static_pcd.points = o3d.utility.Vector3dVector(np.array(spts))
        static_pcd.colors = o3d.utility.Vector3dVector(np.array(scol))
        print(f"  [viz] static cloud: {len(spts)} pts (image-colored)")
    else:
        print(f"  [viz] no static tracks ({sp.name}); showing dynamic only")

    # dynamic: sampled tracks over a trailing window, coloured by recency (old=blue -> new=red)
    Xsets = {k: tracks[k][sel] for k in names}; Vs = vis[sel]
    dyn_pcd = o3d.geometry.PointCloud()
    tvis = np.where(Vs.any(0))[0]
    st = {"f": int(min(tvis[0] + args.viz_window, tvis[-1])) if tvis.size else 0, "k": 0}

    def set_frame(f):
        f = f % T; st["f"] = f
        Xs = Xsets[names[st["k"]]]
        win = list(range(max(0, f - args.viz_window + 1), f + 1))
        pts, cols = [], []
        for t in win:
            idx = np.where(Vs[:, t])[0]
            if idx.size == 0:
                continue
            pts.append(Xs[idx, t]); cols.append(np.tile(cm.turbo((t - win[0]) / max(len(win) - 1, 1))[:3], (idx.size, 1)))
        P = np.concatenate(pts) if pts else np.zeros((0, 3)); Cc = np.concatenate(cols) if cols else np.zeros((0, 3))
        dyn_pcd.points = o3d.utility.Vector3dVector(P); dyn_pcd.colors = o3d.utility.Vector3dVector(Cc)

    set_frame(st["f"])
    vv = o3d.visualization.VisualizerWithKeyCallback(); vv.create_window(f"tracks {D.dataset_name}", 1400, 1000)
    vv.add_geometry(static_pcd); vv.add_geometry(dyn_pcd)
    vv.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15))
    ro = vv.get_render_option(); ro.point_size = 3.5; ro.background_color = np.array([0.05, 0.05, 0.08])

    def refresh():
        vv.update_geometry(dyn_pcd)
        print(f"\r [{names[st['k']]}] frame {st['f']:3d}/{T-1}  window=[{max(0, st['f']-args.viz_window+1)}..{st['f']}]  "
              f"dyn pts={len(np.asarray(dyn_pcd.points))}   ", end="", flush=True)

    def step(d):
        set_frame(st["f"] + d); refresh(); return False

    def toggle(v):
        if len(names) > 1:
            st["k"] = (st["k"] + 1) % len(names); set_frame(st["f"]); refresh()
        return False
    vv.register_key_callback(ord(" "), lambda v: step(+1))
    vv.register_key_callback(ord("B"), lambda v: step(-1))
    vv.register_key_callback(ord("T"), toggle)
    tog = "  T=toggle before/after" if len(names) > 1 else ""
    print(f"  [viz] SPACE=next frame, B=prev frame{tog}  (dyn colored by recency old=blue->new=red)")
    refresh(); vv.run(); vv.destroy_window(); print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="tracks_dynamic_merged.npz", help="dynamic tracks npz under outputs/ (merged)")
    ap.add_argument("--output", default="", help="output npz under outputs/ (default: <input_stem>_smoothed.npz)")
    ap.add_argument("--w-reproj", type=float, default=1.0)
    ap.add_argument("--w-smooth", type=float, default=5.0)
    ap.add_argument("--w-prior", type=float, default=0.1)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--max-tracks", type=int, default=0, help="0 = all tracks")
    ap.add_argument("--mask-erode", type=int, default=2,
                    help="drop observations within this many px of the dynamic-mask boundary (0 = off)")
    ap.add_argument("--depth-grad-thresh", type=float, default=0.05,
                    help="drop observations whose 3x3 depth range exceeds this fraction of the center depth (0 = off)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--visualize", action="store_true",
                    help="after optimizing, open Open3D: static scene (image-colored, all frames) + sampled dynamic "
                         "tracks over a trailing window (SPACE/B to step)")
    ap.add_argument("--viz-window", type=int, default=10, help="dynamic trailing-window length (frames)")
    ap.add_argument("--viz-tracks", type=int, default=10000, help="number of dynamic tracks to display")
    ap.add_argument("--viz-static", type=int, default=12000, help="number of static backdrop points")
    ap.add_argument("--visualize-only", action="store_true",
                    help="skip the solve+write; load the existing (already-smoothed) track file and only show the viewer")
    args = ap.parse_args()
    out_name = args.output or (Path(args.input).stem + "_smoothed.npz")   # e.g. tracks_dynamic_merged -> _merged_smoothed

    O = Path(D.output_dir)
    # merge is 2D-only now -> read only 2D tracks + occlusion/dyn mask; the 3D is lifted HERE from the specified depth
    src = np.load(O / args.input, allow_pickle=True)
    txy = src["tracks_xy"][0].astype(np.float64)          # (Q,T,2) 2D obs (depth-independent)
    occ = src["occluded"][0]; dynm = src["dynamic_obs_mask"]
    Q, T = txy.shape[:2]; Q_orig = Q
    if args.max_tracks and Q > args.max_tracks:
        sel = np.random.RandomState(0).choice(Q, args.max_tracks, replace=False)
        txy, occ, dynm = txy[sel], occ[sel], dynm[sel]; Q = args.max_tracks
    subsampled = bool(args.max_tracks) and Q_orig > args.max_tracks

    rt = np.load(O / "result_XM.npz", allow_pickle=True)
    R_raw, t_raw = scatter_solved_poses(rt, T)                       # c2w (T,3,3),(T,3)
    s_raw = scatter_solved_frame_vector(rt, "s", T, default=1.0)     # (T,)
    if s_raw is None:
        s_raw = np.ones(T)
    depth = np.load(Path(D.active_depth_maps))                       # THE specified depth (definitions.depth_variant)
    K = np.load(Path(D.active_intrinsics)).astype(np.float64)        # K matching that depth
    print(f"[{D.dataset_name}] input={args.input} tracks={Q} T={T} depth={D.depth_variant} device={args.device}")

    # LIFT 3D fully here from the specified depth (camera frame), dropping near-mask-boundary + depth-discontinuity
    # observations (a tracked point that drifts to the mask edge samples background depth -> wrong / very large).
    masks_ok = load_eroded_masks(D.mask_dir, D.mask_pattern, T, depth.shape[1:], args.mask_erode)
    xyz_cam, n_bnd, n_grad = lift_camera_xyz(txy, occ, depth, K, masks_ok, args.depth_grad_thresh)
    dropped = (~occ) & dynm.astype(bool) & ~np.isfinite(xyz_cam).all(-1)         # valid dynamic obs left with no 3D
    occ = occ | dropped                                                          # mark them occluded (dropped)
    print(f"  dropped obs: {n_bnd} near-boundary(<= {args.mask_erode}px) + {n_grad} depth-discont(3x3> {args.depth_grad_thresh}) "
          f"-> {int(dropped.sum())} total newly-occluded")

    R_for = (s_raw[:, None, None] * R_raw)
    X_world = np.einsum("tij,qtj->qti", R_for, xyz_cam) + t_raw[None, :, :]     # (Q,T,3)
    vis = np.isfinite(X_world).all(-1) & (~occ) & dynm
    X_world[~vis] = 0.0                                                          # zero invalid (avoid inf->NaN in losses)
    print(f"  visible obs: {int(vis.sum())} / {Q*T}")

    if args.visualize_only:                                                      # no solve, no write -- just show
        tracks = {"lifted (from depth)": X_world}                               # 3D lifted from the specified depth
        sm = O / out_name
        if sm.exists() and not subsampled:                                       # overlay the already-smoothed result
            sxyz = np.load(sm, allow_pickle=True)["tracks_xyz"][0].astype(np.float64)
            if sxyz.shape[:2] == X_world.shape[:2] and np.isfinite(sxyz).any():
                Xa = np.einsum("tij,qtj->qti", R_for, sxyz) + t_raw[None, :, :]; Xa[~vis] = 0.0
                tracks = {"before (lifted)": X_world, "after (smoothed)": Xa}
        print(f"  --visualize-only: showing {list(tracks.keys())} (no solve)")
        visualize(tracks, vis, R_for, t_raw, K, args)
        return

    X_opt, rmse = optimize(X_world, vis, txy, R_raw, t_raw, K,
                           args.w_reproj, args.w_smooth, args.w_prior, args.iters, args.lr, args.device)
    print(f"  smoothed {Q} tracks: reproj RMSE={rmse:.3f}px (w {args.w_reproj}/{args.w_smooth}/{args.w_prior})")

    # WORLD -> per-frame CAMERA frame (inverse of EDG_old.py:104): x_cam = (1/s) R^T (X_world - t)
    invR_over_s = np.transpose(R_raw, (0, 2, 1)) / s_raw[:, None, None]
    xyz_cam_new = np.einsum("tij,qtj->qti", invR_over_s, X_opt - t_raw[None, :, :])
    xyz_cam_new[~vis] = xyz_cam[~vis]                                # keep invalid entries as-is

    if Q < Q_orig:   # subsampled -> write a minimal, self-consistent schema (all EDG needs)
        out = {"tracks_xy": txy[None].astype(np.float32), "tracks_xyz": xyz_cam_new[None].astype(np.float64),
               "occluded": occ[None], "dynamic_obs_mask": dynm}
    else:            # full set -> preserve every source key, swap in smoothed tracks_xyz + dropped-obs occlusion
        out = {k: src[k] for k in src.files}
        out["tracks_xyz"] = xyz_cam_new[None].astype(np.float64)
        out["occluded"] = occ[None]                                 # includes the boundary/depth-discont drops
    out["depth_variant"] = np.str_(D.depth_variant)                 # provenance: which depth lifted the 3D
    np.savez(O / out_name, **out)
    print(f"  wrote {O / out_name}\n  -> build EDG (GT/LiDAR depth):  python build_edg.py --gt-depth   "
          f"(uses merged+smoothed tracks by default)")

    if args.visualize:                                 # X_world=before-optimize lift, X_opt=after (both WORLD)
        visualize({"before-optimize": X_world, "after-optimize": X_opt}, vis, R_for, t_raw, K, args)


if __name__ == "__main__":
    main()
