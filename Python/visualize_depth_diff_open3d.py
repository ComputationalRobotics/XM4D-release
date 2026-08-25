#!/usr/bin/env python3
"""Interactive Open3D viewer: OUR DepthAnything-3 (DA3) depth vs MoSca's LiDAR sensor depth, in 3D.

Backprojects both per-frame depth maps into the camera coordinate frame using the DA3 intrinsics
(both maps live on the same 360x480 image plane) and overlays the two point clouds so you can SEE
where the surfaces disagree. DA3 is scale/shift-ambiguous, so a toggle aligns DA3 to the LiDAR depth
(least-squares scale+shift on co-valid pixels) -- with alignment ON you see *shape* error; OFF you see
the raw metric-scale error too.

Colors:
  source mode (default) : DA3 = orange, MoSca/LiDAR = cyan   (best for spotting divergence)
  rgb mode              : both colored by the image

Controls:
  SPACE  play / pause          . / ,  step frame +/-1        D / A  step +/-10
  S      toggle DA3 scale+shift alignment to LiDAR
  1 / 2  toggle DA3 cloud / LiDAR cloud visibility
  C      cycle color mode (source <-> rgb)
  ] / [  faster / slower     R  reset to frame 0     Q/ESC  quit

Example:
  conda run -n gsplat python visualize_depth_diff_open3d.py --scene wheel
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image


def backproject(depth: np.ndarray, K: np.ndarray, mask: np.ndarray):
    """(H,W) depth -> (M,3) camera-frame points + (M,) pixel index of valid pts."""
    H, W = depth.shape
    vv, uu = np.mgrid[0:H, 0:W]
    valid = (depth > 1e-3) & mask
    z = depth[valid]
    x = (uu[valid] - K[0, 2]) / K[0, 0] * z
    y = (vv[valid] - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], -1), valid


def scale_shift(pred: np.ndarray, gt: np.ndarray, m: np.ndarray):
    p = pred[m]; g = gt[m]
    A = np.stack([p, np.ones_like(p)], 1)
    ab, *_ = np.linalg.lstsq(A, g, rcond=None)
    return float(ab[0]), float(ab[1])


def load_all(scene_dir: Path):
    da3 = np.load(scene_dir / "outputs" / "depth_anything_3" / "depth_maps.npy")   # (n,H,W)
    K = np.load(scene_dir / "outputs" / "depth_anything_3" / "intrinsics.npy")     # (n,3,3)
    n = da3.shape[0]
    sd = scene_dir / "outputs_mosca" / "sensor_depth"
    lidar = np.stack([np.load(sd / f"frame{i:06d}.npz")["dep"].astype(np.float32) for i in range(n)])
    return da3, lidar, K, n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="wheel")
    ap.add_argument("--root", type=Path, default=Path("/home/fenchel/4DReconstruction/datasets/Dycheck"))
    ap.add_argument("--point-size", type=float, default=2.5)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--no-align", action="store_true", help="start with scale-align OFF (show raw scale gap)")
    args = ap.parse_args()

    scene_dir = args.root / args.scene
    DA3, LIDAR, K, n = load_all(scene_dir)
    print(f"[{args.scene}] frames={n} HxW={DA3.shape[1:]}  DA3 range={DA3.min():.2f}..{DA3.max():.2f}m  "
          f"LiDAR range={LIDAR.min():.2f}..{LIDAR.max():.2f}m")

    ORANGE = np.array([1.0, 0.55, 0.1]); CYAN = np.array([0.1, 0.75, 0.95])
    st = {"t": int(args.frame) % n, "play": False, "speed": 1, "last": 0.0,
          "align": not args.no_align, "cmode": 0, "show_da3": True, "show_lidar": True}
    cmodes = ["source", "rgb"]

    da3_pcd = o3d.geometry.PointCloud(); lidar_pcd = o3d.geometry.PointCloud()

    def rgb_of(i):
        return np.asarray(Image.open(scene_dir / "images" / f"frame{i:06d}.png").convert("RGB"))

    def build(i):
        k = K[i]; lid = LIDAR[i]; da = DA3[i].copy()
        m = lid > 1e-3
        if st["align"] and m.sum() > 100:
            a, b = scale_shift(da, lid, m); da = a * da + b
        img = rgb_of(i) if cmodes[st["cmode"]] == "rgb" else None
        # DA3 cloud (all valid DA3 pixels)
        p_da, v_da = backproject(da, k, np.ones_like(m))
        da3_pcd.points = o3d.utility.Vector3dVector(p_da if st["show_da3"] else np.zeros((0, 3)))
        # LiDAR cloud (valid sensor pixels)
        p_li, v_li = backproject(lid, k, m)
        lidar_pcd.points = o3d.utility.Vector3dVector(p_li if st["show_lidar"] else np.zeros((0, 3)))
        if cmodes[st["cmode"]] == "rgb":
            da3_pcd.colors = o3d.utility.Vector3dVector((img[v_da] / 255.0) if st["show_da3"] else np.zeros((0, 3)))
            lidar_pcd.colors = o3d.utility.Vector3dVector((img[v_li] / 255.0) if st["show_lidar"] else np.zeros((0, 3)))
        else:
            da3_pcd.colors = o3d.utility.Vector3dVector(np.tile(ORANGE, (len(p_da) if st["show_da3"] else 0, 1)))
            lidar_pcd.colors = o3d.utility.Vector3dVector(np.tile(CYAN, (len(p_li) if st["show_lidar"] else 0, 1)))

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(f"Depth diff — {args.scene}: DA3(orange) vs MoSca LiDAR(cyan)", 1400, 1000)
    build(st["t"]); vis.add_geometry(da3_pcd); vis.add_geometry(lidar_pcd)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3))
    ro = vis.get_render_option(); ro.point_size = args.point_size; ro.background_color = np.array([0.05, 0.05, 0.08])

    def refresh():
        vis.update_geometry(da3_pcd); vis.update_geometry(lidar_pcd)

    def _title():
        print(f"\r frame {st['t']:3d}/{n-1}  x{st['speed']}  align={'ON ' if st['align'] else 'OFF'}  "
              f"color={cmodes[st['cmode']]}  DA3={'on' if st['show_da3'] else 'off'} "
              f"LiDAR={'on' if st['show_lidar'] else 'off'}  {'PLAY ' if st['play'] else 'PAUSE'}", end="", flush=True)

    def go(t, pause=True):
        if pause: st["play"] = False
        st["t"] = t % n; build(st["t"]); refresh(); _title(); return True

    def toggle(key):
        st[key] = not st[key]; build(st["t"]); refresh(); _title(); return False

    def anim(_v):
        if not st["play"]: return False
        now = time.time()
        if now - st["last"] < 1.0 / max(args.fps, 1e-3): return False
        st["last"] = now; st["t"] = (st["t"] + st["speed"]) % n; build(st["t"]); refresh(); _title(); return True

    vis.register_key_callback(32, lambda v: (st.update(play=not st["play"]), _title(), False)[-1])
    vis.register_key_callback(ord("."), lambda v: go(st["t"] + 1))
    vis.register_key_callback(ord(","), lambda v: go(st["t"] - 1))
    vis.register_key_callback(ord("D"), lambda v: go(st["t"] + 10))
    vis.register_key_callback(ord("A"), lambda v: go(st["t"] - 10))
    vis.register_key_callback(ord("S"), lambda v: toggle("align"))
    vis.register_key_callback(ord("1"), lambda v: toggle("show_da3"))
    vis.register_key_callback(ord("2"), lambda v: toggle("show_lidar"))
    vis.register_key_callback(ord("C"), lambda v: (st.update(cmode=(st["cmode"] + 1) % 2), build(st["t"]), refresh(), _title(), False)[-1])
    vis.register_key_callback(ord("]"), lambda v: (st.update(speed=min(st["speed"] + 1, 20)), _title(), False)[-1])
    vis.register_key_callback(ord("["), lambda v: (st.update(speed=max(st["speed"] - 1, 1)), _title(), False)[-1])
    vis.register_key_callback(ord("R"), lambda v: go(0))
    vis.register_animation_callback(anim)

    print(__doc__.split("Example:")[0].split("Controls:")[1])
    print(" DA3 = orange, MoSca LiDAR = cyan. Toggle S to compare shape-only vs raw scale.")
    _title()
    vis.run(); vis.destroy_window(); print()


if __name__ == "__main__":
    main()
