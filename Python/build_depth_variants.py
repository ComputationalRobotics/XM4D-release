#!/usr/bin/env python3
"""Organize the different depth products into  outputs/depths/<variant>/  (depth_maps.npy + intrinsics.npy),
so each kind is unambiguous and any one can be fed to the solver by copying it into the da3 slot.

Variants (all at the DA3 depth resolution, metric metres):
  raw_da3       raw DepthAnything-3 (get_depths_clean.py output)          <- depth_anything_3/depth_maps_da3orig.npy (or depth_maps.npy)
  aligned_da3   raw DA3 per-frame scale+shift aligned to LiDAR (metric)   <- computed here (LiDAR-valid pixels)
  lidar         raw sensor LiDAR (depth/*.png, mm->m, holes = 0)          <- built here
  lidar_filled  LiDAR exact + aligned-DA3 in the holes (dense)            <- built here (self-contained; no MoSca sensor_depth)

The da3 slot (depth_anything_3/depth_maps.npy) stays the ACTIVE depth the pipeline reads. To test a variant:
  python build_depth_variants.py --scene block --activate lidar     # copies lidar -> da3 slot (backs up DA3 first)
  XM_DATASET_CAT=Dycheck XM_DATASET_NAME=block conda run -n XM python static_solver.py   # chamfer scale-fit runs

Build only (non-destructive; does NOT touch the da3 slot):
  conda run -n gsplat python build_depth_variants.py --scene block
"""
import argparse
import glob
import shutil
from pathlib import Path
import numpy as np
import cv2

VARIANTS = ["raw_da3", "aligned_da3", "lidar", "lidar_filled"]


def scale_shift_align(pred, gt, mask):
    """Least-squares a,b so a*pred+b ~= gt on masked pixels (per-frame SSI / MiDaS protocol)."""
    p = pred[mask]; g = gt[mask]
    A = np.stack([p, np.ones_like(p)], 1)
    ab, *_ = np.linalg.lstsq(A, g, rcond=None)
    return float(ab[0]), float(ab[1])


def build_lidar_stack(scene_dir: Path, hw) -> np.ndarray:
    """(T,H,W) raw sensor LiDAR from depth/*.png (16-bit mm -> m), resized NEAREST to hw, holes kept as 0."""
    pngs = sorted(glob.glob(str(scene_dir / "depth" / "*.png")))
    if not pngs:
        raise FileNotFoundError(f"no LiDAR PNGs in {scene_dir/'depth'}")
    out = np.zeros((len(pngs), hw[0], hw[1]), np.float32)
    for i, p in enumerate(pngs):
        d = cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        if d.shape != tuple(hw):
            d = cv2.resize(d, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
        out[i] = d
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="block")
    ap.add_argument("--root", type=Path, default=Path("/home/fenchel/4DReconstruction/datasets/Dycheck"))
    ap.add_argument("--activate", choices=VARIANTS, default="", help="also copy this variant into the da3 slot (backs up DA3)")
    args = ap.parse_args()

    O = args.root / args.scene / "outputs"; dd = O / "depth_anything_3"; depths = O / "depths"
    K_path = dd / "intrinsics.npy"
    if not K_path.exists():
        raise FileNotFoundError(f"{K_path} missing -- run get_depths_clean.py first")

    # raw DA3 = the get_depths_clean.py output. Prefer the da3orig backup (immune to any injected LiDAR).
    raw_src = dd / "depth_maps_da3orig.npy" if (dd / "depth_maps_da3orig.npy").exists() else dd / "depth_maps.npy"
    da3 = np.load(raw_src).astype(np.float32); T, H, W = da3.shape
    print(f"[{args.scene}] raw DA3 from {raw_src.name}  {da3.shape}")

    lidar = build_lidar_stack(args.root / args.scene, (H, W))[:T]

    # aligned DA3: per-frame scale+shift of raw DA3 onto LiDAR (metric, dense, DA3 geometry)
    aligned = da3.copy()
    n_ok = 0
    for t in range(T):
        m = (lidar[t] > 1e-3) & np.isfinite(da3[t])
        if int(m.sum()) > 50:
            a, b = scale_shift_align(da3[t], lidar[t], m); aligned[t] = a * da3[t] + b; n_ok += 1
    aligned = np.clip(aligned, 0.0, None)
    print(f"[{args.scene}] aligned_da3: per-frame SSI on {n_ok}/{T} frames")

    # lidar_filled: keep LiDAR exact where valid, fill holes with the LiDAR-aligned DA3 (dense, metric)
    filled = lidar.copy()
    holes = ~(lidar > 1e-3)
    filled[holes] = np.clip(aligned[holes], 0.0, None)
    out = {"raw_da3": da3, "aligned_da3": aligned, "lidar": lidar, "lidar_filled": filled}

    for name, arr in out.items():
        d = depths / name; d.mkdir(parents=True, exist_ok=True)
        np.save(d / "depth_maps.npy", arr.astype(np.float32))
        shutil.copy(K_path, d / "intrinsics.npy")
        cov = float(np.mean(arr > 1e-3) * 100)
        print(f"  {name:12s} {arr.shape}  coverage={cov:5.1f}%  range={arr[arr>0].min():.2f}..{arr.max():.2f}m  -> {d}")
    if (dd / "conf_maps.npy").exists():
        shutil.copy(dd / "conf_maps.npy", depths / "raw_da3" / "conf_maps.npy")

    if args.activate:
        src = depths / args.activate / "depth_maps.npy"
        if not src.exists():
            raise FileNotFoundError(f"{src} not built (variant unavailable)")
        if not (dd / "depth_maps_da3orig.npy").exists():
            shutil.copy(dd / "depth_maps.npy", dd / "depth_maps_da3orig.npy")
        shutil.copy(src, dd / "depth_maps.npy")
        print(f"[activate] {args.activate} -> depth_anything_3/depth_maps.npy  (DA3 backed up as depth_maps_da3orig.npy)\n"
              f"           now run: XM_DATASET_NAME={args.scene} conda run -n XM python static_solver.py")


if __name__ == "__main__":
    main()
