#!/usr/bin/env python3
"""Open3D frame-stepped viz of 2D-track lifting under OUR camera poses (GT), depth treated as UNKNOWN.

Given GT camera poses (result_XM.npz) + GT 2D tracks (output_tapnet_da3.npz static / tracks_dynamic_only.npz
dynamic), shows the full STATIC scene at all times, and windows only the moving track:
  1. STATIC point cloud: full scene lifted from ALL cameras (one lift per static track), colored by the IMAGE
     pixel color. Built once, NOT windowed -- always fully shown.
  2. RED track: the selected track lifted to 3D over the trailing window of the last --window frames (SPACE/B to
     step), as red points + red temporal edges. ONLY the track follows the window rule.
  3. BLACK rays: the ray from each window camera through that track's observed pixel (true point lies ON the ray;
     the red lifted point is where DEPTH places it along the ray).
Window cameras are drawn as RGB coordinate triads; the current frame is the newest.

Controls:  SPACE = next frame   B = prev frame   (window = last --window frames, default 20)

Lift: pixel (u,v) at frame t, depth d -> X_cam=((u-cx)/fx*d,(v-cy)/fy*d,d); X_world = R_t @ X_cam + C_t.

Example:
  conda run -n gsplat python visualize_track_lift_open3d.py --scene wheel --track-source dynamic
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
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
    """RGB coordinate triads for a set of poses (Cs (M,3) centers, Rs (M,3,3) c2w rots)."""
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


def optimize_track(X_init, obs, Rc2w, C, K, w_reproj, w_smooth, w_prior, iters=1000, lr=1e-2):
    """Refine per-frame 3D track positions. Losses (weights should DECREASE, reproj most important):
      1. reprojection: project(X_t) close to the 2D observation  -> keeps X on the camera ray
      2. smoothness  : ||X_t - X_{t-1}||^2 small                  -> temporal smoothness
      3. prior       : ||X_t - X_init_t||^2 small                 -> stay near the depth-lifted init
    CPU torch (small problem; avoids GPU contention). X_init,obs,Rc2w,C,K are (M,3)/(M,2)/(M,3,3)/(M,3)/(M,3,3)."""
    import torch
    X = torch.tensor(np.asarray(X_init), dtype=torch.float32, requires_grad=True)
    Xi = torch.tensor(np.asarray(X_init), dtype=torch.float32)
    ob = torch.tensor(np.asarray(obs), dtype=torch.float32)
    Rw2c = torch.tensor(np.transpose(np.asarray(Rc2w), (0, 2, 1)), dtype=torch.float32)  # world->cam
    Cc = torch.tensor(np.asarray(C), dtype=torch.float32)
    Kt = torch.tensor(np.asarray(K), dtype=torch.float32)
    opt = torch.optim.Adam([X], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        Xc = torch.einsum("mij,mj->mi", Rw2c, X - Cc)          # camera-frame
        z = Xc[:, 2].clamp(min=1e-4)
        proj = torch.stack([Kt[:, 0, 0] * Xc[:, 0] / z + Kt[:, 0, 2],
                            Kt[:, 1, 1] * Xc[:, 1] / z + Kt[:, 1, 2]], 1)
        L_re = ((proj - ob) ** 2).sum(1).mean()
        L_sm = ((X[1:] - X[:-1]) ** 2).sum(1).mean() if len(X) > 1 else torch.zeros(())
        L_pr = ((X - Xi) ** 2).sum(1).mean()
        (w_reproj * L_re + w_smooth * L_sm + w_prior * L_pr).backward()
        opt.step()
    with torch.no_grad():
        Xc = torch.einsum("mij,mj->mi", Rw2c, X - Cc); z = Xc[:, 2].clamp(min=1e-4)
        proj = torch.stack([Kt[:, 0, 0] * Xc[:, 0] / z + Kt[:, 0, 2], Kt[:, 1, 1] * Xc[:, 1] / z + Kt[:, 1, 2]], 1)
        rmse = float(((proj - ob) ** 2).sum(1).mean().sqrt())
    return X.detach().numpy(), rmse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="wheel")
    ap.add_argument("--root", type=Path, default=Path("/home/fenchel/4DReconstruction/datasets/Dycheck"))
    ap.add_argument("--track-source", choices=["static", "dynamic"], default="dynamic")
    ap.add_argument("--track-id", type=int, default=-1, help="track for red/rays; -1 = most-observed")
    ap.add_argument("--optimize", action="store_true", help="optimize the track (reproj + smooth + prior); show in BLUE")
    ap.add_argument("--w-reproj", type=float, default=1.0, help="weight of 2D reprojection loss (most important)")
    ap.add_argument("--w-smooth", type=float, default=0.1, help="weight of temporal-smoothness loss")
    ap.add_argument("--w-prior", type=float, default=0.01, help="weight of stay-near-init (depth-lift) loss")
    ap.add_argument("--opt-iters", type=int, default=1000)
    ap.add_argument("--window", type=int, default=20, help="trailing frames to keep")
    ap.add_argument("--n-static", type=int, default=12000, help="static lifted points (full scene, all cameras)")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--depth", type=Path, default=None, help="depth .npy (default: DA3 backup)")
    ap.add_argument("--result-xm", type=Path, default=None)
    ap.add_argument("--static-npz", type=Path, default=None)
    ap.add_argument("--dyn-npz", type=Path, default=None)
    ap.add_argument("--headless-preview", type=Path, default=None, help="render current --frame window to PNG (no GUI)")
    args = ap.parse_args()

    O = args.root / args.scene / "outputs"
    _cmp = O / "depth_compare" / "completed_lidar_depth_da3.npy"          # completed-LiDAR (preferred)
    depth_path = args.depth or (_cmp if _cmp.exists() else O / "depth_anything_3" / "depth_maps_da3_orig.npy")
    depth = np.load(depth_path)
    print(f"[depth] using {depth_path.name}")
    K = np.load(O / "depth_anything_3" / "intrinsics.npy")
    rt = np.load(args.result_xm or (O / "result_XM.npz"), allow_pickle=True)
    T, H, W = depth.shape
    Rr, C = per_frame_c2w(rt, T)
    stat = np.load(args.static_npz or (O / "output_tapnet_da3.npz"), allow_pickle=True)
    dyn = np.load(args.dyn_npz or (O / "tracks_dynamic_only.npz"), allow_pickle=True)
    sxy = stat["tracks_xy"][0]; socc = stat["occluded"][0]
    txy = (sxy if args.track_source == "static" else dyn["tracks_xy"][0])
    tocc = (socc if args.track_source == "static" else dyn["occluded"][0])
    tid = args.track_id if args.track_id >= 0 else int(np.argmax((~tocc).sum(1)))
    scene_scale = 0.6
    print(f"[{args.scene}] T={T} track {tid} ({args.track_source}) obs in {int((~tocc[tid]).sum())}/{T} frames")

    def lift(u, v, t):
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < W and 0 <= vi < H):
            return None
        d = float(depth[t, vi, ui])
        if d <= 1e-3:
            return None
        k = K[t]; Xc = np.array([(u - k[0, 2]) / k[0, 0] * d, (v - k[1, 2]) / k[1, 1] * d, d])
        return Rr[t] @ Xc + C[t]

    _imgc = {}
    def img(t):
        if t not in _imgc:
            _imgc[t] = np.asarray(Image.open(args.root / args.scene / "images" / f"frame{t:06d}.png").convert("RGB")) / 255.0
        return _imgc[t]

    def build_static():
        """Full static cloud from ALL cameras (one lift per static track at a mid visible frame), image-colored."""
        pts, cols = [], []
        for i in np.random.RandomState(0).permutation(sxy.shape[0]):
            if len(pts) >= args.n_static:
                break
            vf = np.where(~socc[i])[0]
            if len(vf) == 0:
                continue
            t = int(vf[len(vf) // 2]); u, v = sxy[i, t]
            P = lift(u, v, t)
            if P is None:
                continue
            ui = int(np.clip(round(u), 0, W - 1)); vi = int(np.clip(round(v), 0, H - 1))
            pts.append(P); cols.append(img(t)[vi, ui])
        return np.array(pts), np.array(cols)

    def ray(u, v, t, L):
        k = K[t]; dc = np.array([(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], 1.0]); dc /= np.linalg.norm(dc)
        return C[t], C[t] + L * (Rr[t] @ dc)

    # ---- optimize the selected track ONCE over all its visible frames (reproj + smooth + prior) ----
    opt_by_frame = {}
    if args.optimize:
        Xi, ob, fr = [], [], []
        for t in [tt for tt in range(T) if not tocc[tid, tt]]:
            P = lift(txy[tid, t, 0], txy[tid, t, 1], t)
            if P is not None:
                Xi.append(P); ob.append(txy[tid, t]); fr.append(t)
        if len(fr) > 2:
            Xo, rmse = optimize_track(Xi, ob, Rr[fr], C[fr], K[fr],
                                      args.w_reproj, args.w_smooth, args.w_prior, args.opt_iters)
            opt_by_frame = {t: Xo[i] for i, t in enumerate(fr)}
            print(f"  optimized track: {len(fr)} frames, final reproj RMSE={rmse:.2f}px "
                  f"(w reproj/smooth/prior = {args.w_reproj}/{args.w_smooth}/{args.w_prior})")

    static_pcd, red_pcd = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    red_edges, rays_ls, cam_axes = o3d.geometry.LineSet(), o3d.geometry.LineSet(), o3d.geometry.LineSet()
    opt_pcd, opt_edges = o3d.geometry.PointCloud(), o3d.geometry.LineSet()
    _tvis = np.where(~tocc[tid])[0]
    # if --frame not given (0), open on a frame where the trailing window is full of track
    _start = int(args.frame) if (args.frame != 0 or len(_tvis) == 0) else int(min(_tvis[0] + args.window, _tvis[-1]))
    st = {"f": _start % T}
    sp0, sc0 = build_static()                                  # full scene, ALL cameras, image-colored, built ONCE
    static_pcd.points = o3d.utility.Vector3dVector(sp0)
    static_pcd.colors = o3d.utility.Vector3dVector(sc0)
    print(f"  static cloud: {len(sp0)} pts (all cameras, image-colored, not windowed)")

    def set_frame(f):
        f = f % T; st["f"] = f
        win = list(range(max(0, f - args.window + 1), f + 1))
        # red track over window
        rp = [lift(txy[tid, t, 0], txy[tid, t, 1], t) for t in win if not tocc[tid, t]]
        rp = np.array([p for p in rp if p is not None]) if any(p is not None for p in rp) else np.zeros((0, 3))
        red_pcd.points = o3d.utility.Vector3dVector(rp)
        red_pcd.colors = o3d.utility.Vector3dVector(np.tile([1, 0, 0], (len(rp), 1)))
        red_edges.points = o3d.utility.Vector3dVector(rp)
        red_edges.lines = o3d.utility.Vector2iVector(np.array([[i, i + 1] for i in range(len(rp) - 1)]) if len(rp) > 1 else np.zeros((0, 2), int))
        red_edges.colors = o3d.utility.Vector3dVector(np.tile([1, 0, 0], (max(len(rp) - 1, 0), 1)))
        # optimized track over window (BLUE)
        _op = [opt_by_frame[t] for t in win if t in opt_by_frame]
        op = np.array(_op) if _op else np.zeros((0, 3))     # guard empty -> (0,3), not (0,)
        opt_pcd.points = o3d.utility.Vector3dVector(op)
        opt_pcd.colors = o3d.utility.Vector3dVector(np.tile([0.1, 0.3, 1.0], (len(op), 1)))
        opt_edges.points = o3d.utility.Vector3dVector(op)
        opt_edges.lines = o3d.utility.Vector2iVector(np.array([[i, i + 1] for i in range(len(op) - 1)]) if len(op) > 1 else np.zeros((0, 2), int))
        opt_edges.colors = o3d.utility.Vector3dVector(np.tile([0.1, 0.3, 1.0], (max(len(op) - 1, 0), 1)))
        # black rays over window
        rpts, rlines = [], []
        for t in win:
            if tocc[tid, t]:
                continue
            P = lift(txy[tid, t, 0], txy[tid, t, 1], t)
            L = np.linalg.norm(P - C[t]) * 1.15 if P is not None else scene_scale
            o, e = ray(txy[tid, t, 0], txy[tid, t, 1], t, L)
            rlines.append([len(rpts), len(rpts) + 1]); rpts += [o, e]
        rays_ls.points = o3d.utility.Vector3dVector(np.array(rpts) if rpts else np.zeros((0, 3)))
        rays_ls.lines = o3d.utility.Vector2iVector(np.array(rlines) if rlines else np.zeros((0, 2), int))
        rays_ls.colors = o3d.utility.Vector3dVector(np.tile([0, 0, 0], (len(rlines), 1)))
        # window camera triads
        ax = axes_lineset(C[win], Rr[win], 0.03 * scene_scale)
        cam_axes.points, cam_axes.lines, cam_axes.colors = ax.points, ax.lines, ax.colors

    set_frame(st["f"])

    if args.headless_preview:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        sp = np.asarray(static_pcd.points); scol = np.asarray(static_pcd.colors)
        rp = np.asarray(red_pcd.points); rl = np.asarray(rays_ls.points); op = np.asarray(opt_pcd.points)
        fig = plt.figure(figsize=(15, 7))
        for j, (el, az) in enumerate([(18, -60), (70, -90)]):
            axp = fig.add_subplot(1, 2, j + 1, projection="3d")
            if len(sp): axp.scatter(sp[:, 0], sp[:, 1], sp[:, 2], s=1.5, c=scol, alpha=0.5)
            if len(rp): axp.plot(rp[:, 0], rp[:, 1], rp[:, 2], "-o", c="red", ms=3, lw=1.5, label="lifted (raw)")
            if len(op): axp.plot(op[:, 0], op[:, 1], op[:, 2], "-o", c="blue", ms=3, lw=1.5, label="optimized")
            for i in range(0, len(rl), 2):
                axp.plot(rl[i:i+2, 0], rl[i:i+2, 1], rl[i:i+2, 2], "-", c="black", lw=0.5, alpha=0.7)
            cc = C[max(0, st["f"] - args.window + 1):st["f"] + 1]
            axp.scatter(cc[:, 0], cc[:, 1], cc[:, 2], s=12, c="green", marker="^")
            axp.view_init(elev=el, azim=az); axp.set_title(f"view {j+1}")
        fig.suptitle(f"{args.scene}: frame {st['f']} window={args.window}  track {tid} ({args.track_source})", fontsize=12)
        fig.tight_layout(); fig.savefig(args.headless_preview, dpi=95, bbox_inches="tight")
        print(f"wrote {args.headless_preview}"); return

    vis = o3d.visualization.VisualizerWithKeyCallback(); vis.create_window(f"track-lift {args.scene}", 1400, 1000)
    for g in (static_pcd, red_pcd, red_edges, opt_pcd, opt_edges, rays_ls, cam_axes):
        vis.add_geometry(g)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15))
    ro = vis.get_render_option(); ro.point_size = 3.5; ro.background_color = np.array([0.95, 0.95, 0.97])

    def refresh():
        for g in (red_pcd, red_edges, opt_pcd, opt_edges, rays_ls, cam_axes):   # static_pcd is fixed (built once)
            vis.update_geometry(g)
        print(f"\r frame {st['f']:3d}/{T-1}  window=[{max(0,st['f']-args.window+1)}..{st['f']}]", end="", flush=True)

    def step(d):
        set_frame(st["f"] + d); refresh(); return False
    vis.register_key_callback(ord(" "), lambda v: step(+1))
    vis.register_key_callback(ord("B"), lambda v: step(-1))
    print("Controls: SPACE=next frame, B=prev frame")
    refresh()
    vis.run(); vis.destroy_window(); print()


if __name__ == "__main__":
    main()
