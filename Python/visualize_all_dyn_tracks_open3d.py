#!/usr/bin/env python3
"""Open3D viz: optimize ALL dynamic tracks (batched) under OUR camera poses (GT), depth as init only, then
show every dynamic track over a trailing frame window (SPACE/B to step).

Each dynamic track's per-frame 3D position is optimized with three losses (weights should DECREASE, reproj
most important):
  1. reprojection: project(X_t) close to the 2D observation  -> keeps X on the camera ray  (--w-reproj)
  2. smoothness  : ||X_t - X_{t-1}||^2 small                  -> temporal smoothness         (--w-smooth)
  3. prior       : ||X_t - X_init_t||^2 small (depth-lift)    -> stay near the init          (--w-prior)
Optimization is batched over ALL tracks on CPU (won't fight GPU training).

Display: full static scene (image-colored, all cameras, fixed) + all dynamic tracks' OPTIMIZED positions over
the last --window frames, colored by recency (old=blue -> new=red); window camera triads.

Controls: SPACE = next frame   B = prev frame

Example:
  conda run -n gsplat python visualize_all_dyn_tracks_open3d.py --scene wheel --dyn-file merged \
    --w-reproj 1.0 --w-smooth 5.0 --w-prior 0.1 --window 20
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
import matplotlib.cm as cm
from PIL import Image


def per_frame_c2w(rt, T):
    R = np.asarray(rt["R"], float); t = np.asarray(rt["t"], float); N = R.shape[1] // 3
    Rr = np.tile(np.eye(3), (T, 1, 1)); C = np.full((T, 3), np.nan)
    idx = np.asarray(rt["indices_all"], int) if "indices_all" in rt.files else np.arange(T)
    for oi in range(min(T, len(idx))):
        j = int(idx[oi])
        if 0 <= j < N:
            Rr[oi] = R[:, 3 * j:3 * j + 3]; C[oi] = t[:, j]
    valid = np.isfinite(C).all(1); kept = np.where(valid)[0]
    for i in np.where(~valid)[0]:
        jj = kept[np.argmin(np.abs(kept - i))]; Rr[i] = Rr[jj]; C[i] = C[jj]
    return Rr, C


def axes_lineset(Cs, Rs, length):
    m = len(Cs); pts = np.empty((4 * m, 3)); lines = np.empty((3 * m, 2), int); cols = np.empty((3 * m, 3))
    base = np.eye(3) * length
    for i in range(m):
        ax = (Rs[i] @ base.T).T
        pts[4 * i] = Cs[i]; pts[4 * i + 1:4 * i + 4] = Cs[i] + ax
        lines[3 * i:3 * i + 3] = [[4 * i, 4 * i + 1], [4 * i, 4 * i + 2], [4 * i, 4 * i + 3]]
        cols[3 * i:3 * i + 3] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts); ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(cols); return ls


def optimize_all(X_init, vis, obs, Rc2w, C, K, w_re, w_sm, w_pr, iters, lr=3e-3):
    """Batched. X_init (N,T,3), vis (N,T) bool, obs (N,T,2), Rc2w (T,3,3), C (T,3), K (T,3,3)."""
    import torch
    torch.set_num_threads(max(1, (__import__("os").cpu_count() or 4) // 2))
    _Xi = np.nan_to_num(np.asarray(X_init, dtype=np.float32))
    X = torch.tensor(_Xi, requires_grad=True)
    Xi = torch.tensor(_Xi)
    ob = torch.nan_to_num(torch.tensor(np.asarray(obs, dtype=np.float32)))   # occluded obs can be NaN -> 0 (masked anyway)
    m = torch.tensor(vis.astype(np.float32))
    Rw2c = torch.tensor(np.transpose(Rc2w, (0, 2, 1)).astype(np.float32))
    Cc = torch.tensor(C.astype(np.float32))
    fx = torch.tensor(K[:, 0, 0].astype(np.float32)); fy = torch.tensor(K[:, 1, 1].astype(np.float32))
    cx = torch.tensor(K[:, 0, 2].astype(np.float32)); cy = torch.tensor(K[:, 1, 2].astype(np.float32))
    sm = m[:, 1:] * m[:, :-1]
    denom = m.sum().clamp(min=1); sdenom = sm.sum().clamp(min=1)
    opt = torch.optim.Adam([X], lr=lr)
    rmse = float("nan")
    for it in range(iters):
        opt.zero_grad()
        Xc = torch.einsum("tij,ntj->nti", Rw2c, X - Cc[None])
        z = Xc[..., 2].clamp(min=0.05)                      # keep in front of camera -> bounded gradient
        u = fx[None] * Xc[..., 0] / z + cx[None]; v = fy[None] * Xc[..., 1] / z + cy[None]
        err = ((u - ob[..., 0]) ** 2 + (v - ob[..., 1]) ** 2).clamp(max=1e4)   # bound outliers (stability)
        re = (err * m).sum() / denom
        smL = (((X[:, 1:] - X[:, :-1]) ** 2).sum(-1) * sm).sum() / sdenom
        pr = (((X - Xi) ** 2).sum(-1) * m).sum() / denom
        (w_re * re + w_sm * smL + w_pr * pr).backward()
        torch.nn.utils.clip_grad_norm_([X], 1.0)           # prevent divergence / NaN
        opt.step()
        if it == iters - 1:
            rmse = float(re.detach().sqrt())
    return X.detach().numpy(), rmse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="wheel")
    ap.add_argument("--root", type=Path, default=Path("/home/fenchel/4DReconstruction/datasets/Dycheck"))
    ap.add_argument("--dyn-file", choices=["merged", "only", "backward"], default="merged")
    ap.add_argument("--w-reproj", type=float, default=1.0)
    ap.add_argument("--w-smooth", type=float, default=5.0)
    ap.add_argument("--w-prior", type=float, default=0.1)
    ap.add_argument("--opt-iters", type=int, default=400)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--max-tracks", type=int, default=500, help="cap #tracks optimized/shown (0 = all)")
    ap.add_argument("--show-static", action="store_true", help="also overlay the static image-colored scene")
    ap.add_argument("--recompute", action="store_true", help="force re-solve (ignore cached solution)")
    ap.add_argument("--n-static", type=int, default=12000)
    ap.add_argument("--depth", type=Path, default=None)
    ap.add_argument("--result-xm", type=Path, default=None)
    ap.add_argument("--static-npz", type=Path, default=None)
    ap.add_argument("--dyn-npz", type=Path, default=None)
    ap.add_argument("--headless-preview", type=Path, default=None)
    args = ap.parse_args()

    O = args.root / args.scene / "outputs"
    _cmp = O / "depth_compare" / "completed_lidar_depth_da3.npy"          # completed-LiDAR (preferred)
    depth_path = args.depth or (_cmp if _cmp.exists() else O / "depth_anything_3" / "depth_maps_da3_orig.npy")
    depth = np.load(depth_path)
    depth_tag = depth_path.stem.replace("completed_lidar_depth_da3", "cmpLiDAR").replace("depth_maps_da3_orig", "da3")[:16]
    print(f"  depth source: {depth_path.name}  [{depth_tag}]")
    K = np.load(O / "depth_anything_3" / "intrinsics.npy").astype(np.float64)
    rt = np.load(args.result_xm or (O / "result_XM.npz"), allow_pickle=True)
    T, H, W = depth.shape
    Rr, C = per_frame_c2w(rt, T)
    dynp = args.dyn_npz or (O / f"tracks_dynamic_{args.dyn_file}.npz")
    dyn = np.load(dynp, allow_pickle=True)
    txy_full = dyn["tracks_xy"][0].astype(np.float64); occ_full = dyn["occluded"][0]
    Nf = txy_full.shape[0]
    maxt = args.max_tracks if args.max_tracks > 0 else Nf
    sel = np.arange(Nf) if maxt >= Nf else np.random.RandomState(0).choice(Nf, maxt, replace=False)
    N = len(sel)
    print(f"[{args.scene}] T={T} dyn={dynp.name} tracks={N}/{Nf}")

    # cached solved tracks (keyed by dyn-file, #tracks, weights, iters) -> skip re-solving
    cache = O / "depth_compare" / f"solved_dyn_{args.dyn_file}_{depth_tag}_n{N}_re{args.w_reproj}_sm{args.w_smooth}_pr{args.w_prior}_it{args.opt_iters}.npz"
    if cache.exists() and not args.recompute:
        z = np.load(cache); Xo = z["Xo"]; vis = z["vis"]
        print(f"  loaded cached solution: {cache.name}")
    else:
        txy = txy_full[sel]; occ = occ_full[sel]
        X_init = np.zeros((N, T, 3), np.float32); vis = np.zeros((N, T), bool)
        for t in range(T):
            idx = np.where(~occ[:, t])[0]
            if len(idx) == 0:
                continue
            u = txy[idx, t, 0]; v = txy[idx, t, 1]
            ui = np.clip(np.round(u).astype(int), 0, W - 1); vi = np.clip(np.round(v).astype(int), 0, H - 1)
            d = depth[t, vi, ui]; ok = d > 1e-3
            idx, u, v, d = idx[ok], u[ok], v[ok], d[ok]; k = K[t]
            Xc = np.stack([(u - k[0, 2]) / k[0, 0] * d, (v - k[1, 2]) / k[1, 1] * d, d], 1)
            X_init[idx, t] = ((Rr[t] @ Xc.T).T + C[t]).astype(np.float32); vis[idx, t] = True
        print(f"  lifted {vis.sum()} obs over {N} tracks; optimizing...")
        Xo, rmse = optimize_all(X_init, vis, txy, Rr, C, K, args.w_reproj, args.w_smooth, args.w_prior, args.opt_iters)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, Xo=Xo, vis=vis, sel=sel)
        print(f"  optimized {N} tracks: reproj RMSE={rmse:.2f}px (w {args.w_reproj}/{args.w_smooth}/{args.w_prior}) -> saved {cache.name}")

    # optional static backdrop (image-colored). Default OFF -> "just the dynamic masks".
    static_pcd = o3d.geometry.PointCloud()
    if args.show_static:
        stat = np.load(args.static_npz or (O / "output_tapnet_da3.npz"), allow_pickle=True)
        sxy = stat["tracks_xy"][0]; socc = stat["occluded"][0]; _imgc = {}
        def img(t):
            if t not in _imgc:
                _imgc[t] = np.asarray(Image.open(args.root / args.scene / "images" / f"frame{t:06d}.png").convert("RGB")) / 255.0
            return _imgc[t]
        spts, scol = [], []
        for i in np.random.RandomState(0).permutation(sxy.shape[0]):
            if len(spts) >= args.n_static:
                break
            vf = np.where(~socc[i])[0]
            if len(vf) == 0:
                continue
            t = int(vf[len(vf) // 2]); u, v = sxy[i, t]
            ui, vi = int(np.clip(round(u), 0, W - 1)), int(np.clip(round(v), 0, H - 1))
            d = depth[t, vi, ui]
            if d <= 1e-3:
                continue
            k = K[t]; Xc = np.array([(u - k[0, 2]) / k[0, 0] * d, (v - k[1, 2]) / k[1, 1] * d, d])
            spts.append(Rr[t] @ Xc + C[t]); scol.append(img(t)[vi, ui])
        static_pcd.points = o3d.utility.Vector3dVector(np.array(spts)); static_pcd.colors = o3d.utility.Vector3dVector(np.array(scol))
        print(f"  static cloud: {len(spts)} pts (image-colored)")
    _dv = Xo[vis]
    scene_scale = float(np.linalg.norm(_dv.max(0) - _dv.min(0))) if len(_dv) else 1.0

    dyn_pcd = o3d.geometry.PointCloud()
    _tvis = np.where(vis.any(0))[0]
    st = {"f": (int(args.frame) if (args.frame != 0 or len(_tvis) == 0) else int(min(_tvis[0] + args.window, _tvis[-1]))) % T}

    def set_frame(f):
        f = f % T; st["f"] = f
        win = list(range(max(0, f - args.window + 1), f + 1))
        pts, cols = [], []
        for t in win:
            idx = np.where(vis[:, t])[0]
            if len(idx) == 0:
                continue
            pts.append(Xo[idx, t]); cols.append(np.tile(cm.turbo((t - win[0]) / max(len(win) - 1, 1))[:3], (len(idx), 1)))
        P = np.concatenate(pts) if pts else np.zeros((0, 3)); Cc = np.concatenate(cols) if cols else np.zeros((0, 3))
        dyn_pcd.points = o3d.utility.Vector3dVector(P); dyn_pcd.colors = o3d.utility.Vector3dVector(Cc)

    set_frame(st["f"])

    if args.headless_preview:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        sp = np.asarray(static_pcd.points); sc = np.asarray(static_pcd.colors)
        dp = np.asarray(dyn_pcd.points); dc = np.asarray(dyn_pcd.colors)
        fig = plt.figure(figsize=(15, 7))
        for j, (el, az) in enumerate([(18, -60), (70, -90)]):
            axp = fig.add_subplot(1, 2, j + 1, projection="3d")
            if len(sp): axp.scatter(sp[:, 0], sp[:, 1], sp[:, 2], s=1, c=sc, alpha=0.25)
            if len(dp): axp.scatter(dp[:, 0], dp[:, 1], dp[:, 2], s=3, c=dc)
            axp.view_init(elev=el, azim=az); axp.set_title(f"view {j+1}")
        fig.suptitle(f"{args.scene}: ALL dyn tracks optimized, window={args.window} @frame {st['f']} ({args.dyn_file})", fontsize=12)
        fig.tight_layout(); fig.savefig(args.headless_preview, dpi=95, bbox_inches="tight")
        print(f"wrote {args.headless_preview}"); return

    vis3 = o3d.visualization.VisualizerWithKeyCallback(); vis3.create_window(f"all-dyn {args.scene}", 1400, 1000)
    if args.show_static:
        vis3.add_geometry(static_pcd)
    vis3.add_geometry(dyn_pcd)
    vis3.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15))
    ro = vis3.get_render_option(); ro.point_size = 3.5; ro.background_color = np.array([0.05, 0.05, 0.08])

    def refresh():
        vis3.update_geometry(dyn_pcd)
        print(f"\r frame {st['f']:3d}/{T-1}  window=[{max(0,st['f']-args.window+1)}..{st['f']}]  dyn pts={len(np.asarray(dyn_pcd.points))}", end="", flush=True)

    def step(d):
        set_frame(st["f"] + d); refresh(); return False
    vis3.register_key_callback(ord(" "), lambda v: step(+1))
    vis3.register_key_callback(ord("B"), lambda v: step(-1))
    print("Controls: SPACE=next frame, B=prev frame  (dyn colored by recency old=blue->new=red)")
    refresh()
    vis3.run(); vis3.destroy_window(); print()


if __name__ == "__main__":
    main()
