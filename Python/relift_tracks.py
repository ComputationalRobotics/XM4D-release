#!/usr/bin/env python3
"""Re-lift dynamic tracks' 3D (tracks_xyz) from the CURRENT depth, REUSING the existing 2D tracks (no re-track).

When the depth changes (e.g. DA3 -> GT LiDAR), the 2D TAPNet tracking is depth-independent and can be reused,
but the per-frame camera-space lift tracks_xyz = ((u-cx)/fx*z, (v-cy)/fy*z, z) must be recomputed from the new
depth. This does exactly that for tracks_dynamic_only.npz (+ _backward.npz if present) using the injected depth
(outputs/depth_anything_3/depth_maps.npy) and intrinsics.npy -- matching dynamic_tracking.py's lift exactly.

Scene via XM_DATASET_CAT / XM_DATASET_NAME. Run AFTER injecting the new depth, BEFORE test_extend_tracks."""
import argparse
from pathlib import Path
import numpy as np
import utils.definitions as D


def relift(npz_path, depth, K):
    data = dict(np.load(npz_path, allow_pickle=True))
    txy = data["tracks_xy"][0]                       # (Q,T,2) pixels (depth-independent, reused)
    occ = data["occluded"][0]                        # (Q,T)
    Q, T = txy.shape[:2]; H, W = depth.shape[1:]
    xyz = np.full((Q, T, 3), np.nan, np.float64)
    n = 0
    for t in range(T):
        vis = np.where(~occ[:, t])[0]
        if vis.size == 0:
            continue
        u = txy[vis, t, 0]; v = txy[vis, t, 1]
        ui = np.clip(np.round(u).astype(int), 0, W - 1); vi = np.clip(np.round(v).astype(int), 0, H - 1)
        z = depth[t, vi, ui]; ok = z > 1e-3
        vis, u, v, z = vis[ok], u[ok], v[ok], z[ok]; k = K[t]
        xyz[vis, t, 0] = (u - k[0, 2]) / k[0, 0] * z          # matches dynamic_tracking.py:1089
        xyz[vis, t, 1] = (v - k[1, 2]) / k[1, 1] * z
        xyz[vis, t, 2] = z
        n += vis.size
    data["tracks_xyz"] = xyz[None]                   # keep tracks_xy / occluded / masks; swap only 3D
    np.savez(npz_path, **data)
    print(f"  re-lifted {Path(npz_path).name}: {n} valid 3D obs from current depth (2D reused, no re-track)")


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    O = Path(D.output_dir)
    depth = np.load(Path(D.active_depth_maps))                           # the selected depth variant (definitions.depth_variant)
    K = np.load(Path(D.active_intrinsics)).astype(np.float64)
    print(f"[relift] {D.dataset_name}: depth {depth.shape} range {depth[depth>0].min():.2f}..{depth.max():.2f}m")
    for name in ("tracks_dynamic_only.npz", "tracks_dynamic_backward.npz"):
        p = O / name
        if p.exists():
            relift(str(p), depth, K)


if __name__ == "__main__":
    main()
