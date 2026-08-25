#!/usr/bin/env python3
"""Open3D: lift a DENSE depth map to WORLD frame with the solved cameras (result_XM.npz), image-colored, to
sanity-check depth quality. Default depth = the ORIGINAL raw LiDAR (depth/*.png). World lift matches the
pipeline (EDG_old.py:104 / optimize_dynamic_tracks.py):  X_world = s * R_c2w @ x_cam + t.

If depth (and poses) are correct, the static scene lifted from different frames OVERLAPS in world; a bad depth
shows up as warped geometry or frames that don't register. Press A to overlay ALL frames -- the clearest test.

Controls: SPACE = next frame,  B = prev frame,  A = toggle accumulate (all frames overlaid)

Example:
  conda run -n gsplat python visualize_lidar_lift_open3d.py --scene wheel --depth lidar-png --stride 3
  conda run -n gsplat python visualize_lidar_lift_open3d.py --scene block --depth lidar-png --stride 4 --frame 0
"""
from __future__ import annotations
import argparse
import glob
from pathlib import Path
import numpy as np
import cv2
import open3d as o3d
from PIL import Image

DEPTH_SOURCES = ["lidar-png", "lidar-npy", "completed", "da3", "current"]


def per_frame_c2w(rt, T):
    """(T,3,3) c2w R, (T,3) camera centre C, (T,) per-frame depth scale, (T,) valid mask, scale-key name.
    Scattered onto the timeline via indices_all -- IDENTICAL to utils.edg.scatter_solved_poses /
    scatter_solved_frame_vector. The per-frame scale is static_solver's fitted value
    (estimate_da3_frame_scales_static_chamfer, stored as 'da3_frame_scale'; == 's'). Dropped frames
    (indices_all < 0) keep NaN centre + valid=False and are SKIPPED -- NOT filled from a neighbour (that
    neighbour-fill was the earlier per-frame error, wrong exactly on dropped frames)."""
    R = np.asarray(rt["R"], float); t = np.asarray(rt["t"], float); N = R.shape[1] // 3
    key = "da3_frame_scale" if "da3_frame_scale" in rt.files else ("s" if "s" in rt.files else None)
    sv = np.asarray(rt[key], float).ravel() if key else np.ones(N)
    idx = np.asarray(rt["indices_all"], int) if "indices_all" in rt.files else np.arange(T)
    Rr = np.tile(np.eye(3), (T, 1, 1)); C = np.full((T, 3), np.nan); s = np.ones(T)
    for oi in range(min(T, len(idx))):
        j = int(idx[oi])
        if 0 <= j < N:
            Rr[oi] = R[:, 3 * j:3 * j + 3]; C[oi] = t[:, j]
            if j < len(sv):
                s[oi] = sv[j]
    valid = np.isfinite(C).all(1)                          # dropped frames stay invalid -> skipped downstream
    return Rr, C, s, valid, (key or "none")


def load_depth(scene_dir: Path, source: str, hw) -> np.ndarray:
    """(T,H,W) metric depth for the chosen source, resized (NEAREST) to hw=(H,W)."""
    O = scene_dir / "outputs"; dd = O / "depth_anything_3"
    if source == "lidar-png":                                        # the ORIGINAL raw sensor depth (mm -> m)
        pngs = sorted(glob.glob(str(scene_dir / "depth" / "*.png")))
        if not pngs:
            raise FileNotFoundError(f"no raw LiDAR PNGs in {scene_dir/'depth'}")
        out = []
        for p in pngs:
            d = cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
            if d.shape != tuple(hw):
                d = cv2.resize(d, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
            out.append(d)
        return np.stack(out)
    path = {"lidar-npy": dd / "depth_maps_lidar.npy",
            "completed": O / "depth_compare" / "completed_lidar_depth_da3.npy",
            "da3": dd / "depth_maps_da3orig.npy",
            "current": dd / "depth_maps.npy"}[source]
    return np.load(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="wheel")
    ap.add_argument("--root", type=Path, default=Path("/home/fenchel/4DReconstruction/datasets/Dycheck"))
    ap.add_argument("--depth", choices=DEPTH_SOURCES, default="lidar-png", help="which depth to lift (default: original raw LiDAR)")
    ap.add_argument("--variant", default="", help="lift a build_depth_variants folder: outputs/depths/<variant>/depth_maps.npy (raw_da3|aligned_da3|lidar|lidar_filled)")
    ap.add_argument("--depth-npy", type=Path, default=None, help="lift an explicit depth_maps.npy path (overrides --depth/--variant)")
    ap.add_argument("--stride", type=int, default=3, help="pixel subsample stride (higher = fewer points)")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--max-depth", type=float, default=0.0, help="drop depths beyond this many metres (0 = off)")
    ap.add_argument("--no-scale", action="store_true", help="lift with R only (ignore per-frame solver scale s)")
    ap.add_argument("--accumulate", action="store_true", help="start in overlay-all-frames mode")
    args = ap.parse_args()

    scene_dir = args.root / args.scene; O = scene_dir / "outputs"
    K = np.load(O / "depth_anything_3" / "intrinsics.npy").astype(np.float64)
    ref = np.load(O / "depth_anything_3" / "depth_maps.npy", mmap_mode="r")      # reference shape at intrinsics res
    T, H, W = ref.shape
    if args.depth_npy is not None:
        depth = np.load(args.depth_npy); args.depth = args.depth_npy.parent.name or args.depth_npy.stem
    elif args.variant:
        depth = np.load(O / "depths" / args.variant / "depth_maps.npy"); args.depth = args.variant
    else:
        depth = load_depth(scene_dir, args.depth, (H, W))
    T = min(T, depth.shape[0])
    rt = np.load(O / "result_XM.npz", allow_pickle=True)
    Rr, C, s, valid, skey = per_frame_c2w(rt, T)
    if args.no_scale:
        s = np.ones_like(s)
    ndrop = int((~valid).sum())
    cov = float(np.mean(depth[:T] > 1e-3) * 100)
    print(f"[{args.scene}] depth={args.depth} {depth.shape} coverage={cov:.1f}%  scale_key={skey} "
          f"s(median over solved)={np.median(s[valid]) if valid.any() else float('nan'):.3f}  "
          f"scale={'off' if args.no_scale else 'on'}  stride={args.stride}  "
          f"dropped_frames={ndrop}{' '+str(list(np.where(~valid)[0])) if ndrop else ''}")

    _imgc = {}
    def img(t):                                                                  # image resized to depth res, cached
        if t not in _imgc:
            im = np.asarray(Image.open(scene_dir / "images" / f"frame{t:06d}.png").convert("RGB")) / 255.0
            if im.shape[:2] != (H, W):
                im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
            _imgc[t] = im
        return _imgc[t]

    _cache = {}
    def lift(t):                                                                 # (M,3) world pts + (M,3) colours
        if t in _cache:
            return _cache[t]
        if not valid[t]:                                                         # dropped frame: no pose -> skip
            _cache[t] = (np.zeros((0, 3)), np.zeros((0, 3))); return _cache[t]
        d = depth[t]
        ys, xs = np.mgrid[0:H:args.stride, 0:W:args.stride]
        z = d[ys, xs]; m = z > 1e-3
        if args.max_depth > 0:
            m &= z < args.max_depth
        yv, xv, z = ys[m], xs[m], z[m].astype(np.float64)
        k = K[min(t, K.shape[0] - 1)]
        Xc = np.stack([(xv - k[0, 2]) / k[0, 0] * z, (yv - k[1, 2]) / k[1, 1] * z, z], 1)
        Xw = ((s[t] * Rr[t]) @ Xc.T).T + C[t]
        _cache[t] = (Xw, img(t)[yv, xv]); return _cache[t]

    pcd = o3d.geometry.PointCloud()
    f0 = int(np.clip(args.frame, 0, T - 1))
    if not valid[f0] and valid.any():                                            # start on a solved frame
        vv_idx = np.where(valid)[0]; f0 = int(vv_idx[np.argmin(np.abs(vv_idx - f0))])
    st = {"f": f0, "acc": args.accumulate}

    def rebuild():
        if st["acc"]:
            pr = [lift(t) for t in range(T) if valid[t]]                         # overlay only solved frames
            P = np.concatenate([a for a, _ in pr]) if pr else np.zeros((0, 3))
            Cc = np.concatenate([b for _, b in pr]) if pr else np.zeros((0, 3))
        else:
            P, Cc = lift(st["f"])
        pcd.points = o3d.utility.Vector3dVector(P); pcd.colors = o3d.utility.Vector3dVector(Cc)
        return len(P)

    n = rebuild()
    vis = o3d.visualization.VisualizerWithKeyCallback(); vis.create_window(f"lidar-lift {args.scene} [{args.depth}]", 1500, 1000)
    vis.add_geometry(pcd)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15))
    ro = vis.get_render_option(); ro.point_size = 2.0; ro.background_color = np.array([0.05, 0.05, 0.08])

    def refresh(reset=False):
        n = rebuild(); vis.update_geometry(pcd)
        if reset:
            vis.reset_view_point(True)
        mode = "ALL frames" if st["acc"] else f"frame {st['f']:3d}/{T-1}"
        print(f"\r {mode}   pts={n:>8d}   ", end="", flush=True)

    def step(d):
        if not st["acc"]:
            f = st["f"]
            for _ in range(T):                                                   # skip dropped frames
                f = (f + d) % T
                if valid[f]:
                    break
            st["f"] = f; refresh()
        return False

    def toggle_acc(v):
        st["acc"] = not st["acc"]; refresh(reset=True); return False

    vis.register_key_callback(ord(" "), lambda v: step(+1))
    vis.register_key_callback(ord("B"), lambda v: step(-1))
    vis.register_key_callback(ord("A"), toggle_acc)
    print("Controls: SPACE=next frame, B=prev frame, A=toggle accumulate (overlay ALL frames)")
    refresh()
    vis.run(); vis.destroy_window(); print()


if __name__ == "__main__":
    main()
