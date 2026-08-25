#!/usr/bin/env python3
"""Simple 2D visualization of a depth map sequence (completed-LiDAR / raw-LiDAR / DA3).

Renders a clean color-mapped grid PNG (a few frames) and, with --video, an MP4 across all frames so you
can see the full dense depth over time. Uses a single global color scale by default so depth is comparable
across frames (toggle with --per-frame).

Example:
  conda run -n gsplat python visualize_depth_2d.py --scene wheel --source completed            # grid PNG
  conda run -n gsplat python visualize_depth_2d.py --scene wheel --source completed --video     # + MP4
  conda run -n gsplat python visualize_depth_2d.py --scene wheel --source lidar                 # raw (holes black)
"""
from __future__ import annotations
import argparse, os, importlib.util
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def _load_completer():
    p = Path(__file__).with_name("complete_lidar_depth.py")
    spec = importlib.util.spec_from_file_location("cld", p); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="wheel")
    ap.add_argument("--root", type=Path, default=Path("/home/fenchel/4DReconstruction/datasets/Dycheck"))
    ap.add_argument("--source", choices=["completed", "lidar", "da3"], default="completed")
    ap.add_argument("--grid", type=int, default=12, help="frames in the overview grid PNG")
    ap.add_argument("--video", action="store_true", help="also render an MP4 over all frames")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--cmap", default="turbo")
    ap.add_argument("--per-frame", action="store_true", help="normalize each frame independently (default: global scale)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    sdir = args.root / args.scene
    out = args.out or (sdir / "outputs" / "depth_compare")
    os.makedirs(out, exist_ok=True)
    sd = sdir / "outputs_mosca" / "sensor_depth"
    n = len(list(sd.glob("frame*.npz")))

    def lidar(i): return np.load(sd / f"frame{i:06d}.npz")["dep"].astype(np.float32)
    def rgb(i): return np.asarray(Image.open(sdir / "images" / f"frame{i:06d}.png").convert("RGB"))
    da3_all = np.load(sdir / "outputs" / "depth_anything_3" / "depth_maps.npy") if args.source == "da3" else None
    completer = _load_completer() if args.source == "completed" else None

    npy = out / "completed_lidar_depth.npy"
    cache = np.load(npy) if (args.source == "completed" and npy.exists()) else None
    if cache is not None: print(f"[using cached {npy.name}]")

    def depth(i):
        if args.source == "lidar":
            g = lidar(i); return np.where(g > 1e-6, g, np.nan)
        if args.source == "da3":
            return da3_all[i]
        if cache is not None: return cache[i]
        return completer.fill_depth_colorization(rgb(i) / 255.0, lidar(i), 100.0)

    # global color scale from a sample of frames
    if not args.per_frame:
        vals = []
        for i in np.linspace(0, n - 1, 8).astype(int):
            d = np.asarray(depth(i)); vals.append(d[np.isfinite(d)].ravel())
        vmin, vmax = np.percentile(np.concatenate(vals), [2, 98]); print(f"global depth scale: {vmin:.2f}..{vmax:.2f} m")
    else:
        vmin = vmax = None

    # --- grid PNG ---
    gf = np.linspace(0, n - 1, args.grid).astype(int)
    ncol = 4; nrow = int(np.ceil(len(gf) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.6 * nrow))
    axs = np.atleast_1d(axs).ravel()
    for a in axs: a.axis("off")
    for k, i in enumerate(gf):
        d = np.asarray(depth(i))
        lo, hi = (np.nanpercentile(d, [2, 98]) if args.per_frame else (vmin, vmax))
        im = axs[k].imshow(d, cmap=args.cmap, vmin=lo, vmax=hi)
        axs[k].set_title(f"frame {i}", fontsize=9)
    fig.colorbar(im, ax=axs.tolist(), shrink=0.5, label="depth (m)")
    fig.suptitle(f"{args.scene}: {args.source} depth (2D)", fontsize=13)
    grid_png = out / f"depth2d_{args.source}_grid.png"
    fig.savefig(grid_png, dpi=95, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {grid_png}")

    # --- video ---
    if args.video:
        H, W = np.asarray(depth(0)).shape
        vpath = out / f"depth2d_{args.source}.mp4"
        vw = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
        cmap = cm.get_cmap(args.cmap)
        for i in range(n):
            d = np.asarray(depth(i))
            lo, hi = (np.nanpercentile(d, [2, 98]) if args.per_frame else (vmin, vmax))
            t = np.clip((d - lo) / (hi - lo + 1e-9), 0, 1)
            col = (cmap(np.nan_to_num(t))[:, :, :3] * 255).astype(np.uint8)
            col[~np.isfinite(d)] = 0
            vw.write(cv2.cvtColor(col, cv2.COLOR_RGB2BGR))
            if i % 100 == 0: print(f"  video frame {i}/{n}")
        vw.release(); print(f"wrote {vpath}")


if __name__ == "__main__":
    main()
