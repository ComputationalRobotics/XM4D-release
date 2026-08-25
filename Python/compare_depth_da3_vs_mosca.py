#!/usr/bin/env python3
"""Compare OUR DepthAnything-3 (DA3) monocular depth vs MoSca's depth for a DyCheck scene.

MoSca's iphone profile uses the iPhone LiDAR *sensor* depth (config depth_dirname: sensor_depth),
i.e. metric sensor depth -- NOT a monocular network. Our pipeline uses DA3 monocular depth
(scale/shift ambiguous). This script loads both, aligns DA3 to the sensor depth with a per-frame
least-squares scale+shift (the standard monocular-depth eval protocol), and reports where/how much
DA3 disagrees, plus side-by-side panels.

Outputs (to --out): depth_panels.png (RGB | DA3-aligned | MoSca/LiDAR | AbsRel error, for a few frames),
depth_metrics.png (AbsRel and delta<1.25 over all frames), and a printed summary.

Example:
  conda run -n gsplat python compare_depth_da3_vs_mosca.py --scene wheel
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_sensor_depth(scene_dir: Path, n: int, hw=None) -> np.ndarray:
    """LiDAR sensor depth (metric meters). Prefer MoSca outputs_mosca/sensor_depth/frame%06d.npz['dep'];
    fall back to the raw DyCheck LiDAR PNGs in <scene>/depth/*.png (16-bit millimetres) when there is no
    MoSca run (e.g. block). hw=(H,W) resizes the PNGs to match the DA3 stack."""
    sd = scene_dir / "outputs_mosca" / "sensor_depth"
    if sd.exists() and len(list(sd.glob("frame*.npz"))) >= n:
        return np.stack([np.load(sd / f"frame{i:06d}.npz")["dep"].astype(np.float32) for i in range(n)])
    import cv2, glob as _glob
    pngs = sorted(_glob.glob(str(scene_dir / "depth" / "*.png")))
    if len(pngs) < n:
        raise FileNotFoundError(f"no MoSca sensor_depth and only {len(pngs)} PNGs in {scene_dir/'depth'}")
    print(f"[lidar] no MoSca sensor_depth -> using raw DyCheck LiDAR PNGs: {scene_dir/'depth'} (mm->m)")
    out = []
    for i in range(n):
        d = cv2.imread(pngs[i], cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0   # mm -> m
        if hw is not None and d.shape != tuple(hw):
            d = cv2.resize(d, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
        out.append(d)
    return np.stack(out)  # (n,H,W)


def scale_shift_align(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Least-squares a,b so a*pred+b ~= gt on masked pixels (MiDaS/Eigen protocol)."""
    p = pred[mask]; g = gt[mask]
    A = np.stack([p, np.ones_like(p)], 1)
    ab, *_ = np.linalg.lstsq(A, g, rcond=None)
    return float(ab[0]), float(ab[1])


def depth_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict:
    p = pred[mask]; g = gt[mask]
    absrel = float(np.mean(np.abs(p - g) / g))
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    ratio = np.maximum(p / g, g / p)
    d1 = float(np.mean(ratio < 1.25))
    return dict(absrel=absrel, rmse=rmse, delta1=d1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="wheel")
    ap.add_argument("--root", type=Path, default=Path("/home/fenchel/4DReconstruction/datasets/Dycheck"))
    ap.add_argument("--panels", type=int, default=5, help="number of frames to show side-by-side")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    scene_dir = args.root / args.scene
    out = args.out or (scene_dir / "outputs" / "depth_compare")
    os.makedirs(out, exist_ok=True)

    da3 = np.load(scene_dir / "outputs" / "depth_anything_3" / "depth_maps.npy")  # (n,H,W) monocular
    n = da3.shape[0]
    sensor = load_sensor_depth(scene_dir, n, hw=da3.shape[1:])                     # (n,H,W) LiDAR (MoSca or raw PNG)
    assert da3.shape == sensor.shape, f"shape mismatch {da3.shape} vs {sensor.shape}"
    print(f"[{args.scene}] frames={n} HxW={da3.shape[1:]}  DA3 range={da3.min():.2f}..{da3.max():.2f}m  "
          f"sensor range={sensor.min():.2f}..{sensor.max():.2f}m")

    # per-frame metrics: DA3 raw vs sensor, and DA3 scale-shift-aligned vs sensor
    raw, aligned, cov = [], [], []
    a_all = np.zeros(n); b_all = np.zeros(n)
    for i in range(n):
        m = sensor[i] > 1e-3
        cov.append(m.mean())
        if m.sum() < 100:
            raw.append(dict(absrel=np.nan, delta1=np.nan)); aligned.append(dict(absrel=np.nan, delta1=np.nan)); continue
        raw.append(depth_metrics(da3[i], sensor[i], m))
        a, b = scale_shift_align(da3[i], sensor[i], m); a_all[i], b_all[i] = a, b
        aligned.append(depth_metrics(a * da3[i] + b, sensor[i], m))

    def mean_key(lst, k): return float(np.nanmean([d[k] for d in lst]))
    print("\n=== DA3 (ours) vs MoSca sensor/LiDAR depth, over all frames ===")
    print(f"  sensor coverage: {np.mean(cov)*100:.1f}% of pixels")
    print(f"  RAW DA3     : AbsRel={mean_key(raw,'absrel')*100:.1f}%  delta<1.25={mean_key(raw,'delta1')*100:.1f}%")
    print(f"  SSI-aligned : AbsRel={mean_key(aligned,'absrel')*100:.1f}%  delta<1.25={mean_key(aligned,'delta1')*100:.1f}%")
    print(f"  (SSI removes global scale+shift -> residual = DA3 *shape* error vs LiDAR)")
    print(f"  per-frame fitted scale a: median={np.median(a_all):.3f}  shift b: median={np.median(b_all):.3f}")

    # --- panels ---
    frames = np.linspace(0, n - 1, args.panels).astype(int)
    fig, axs = plt.subplots(len(frames), 4, figsize=(15, 3.4 * len(frames)))
    if len(frames) == 1: axs = axs[None, :]
    from PIL import Image
    for r, i in enumerate(frames):
        m = sensor[i] > 1e-3
        a, b = scale_shift_align(da3[i], sensor[i], m)
        da3a = a * da3[i] + b
        vmin, vmax = np.percentile(sensor[i][m], [2, 98])
        img = np.asarray(Image.open(scene_dir / "images" / f"frame{i:06d}.png").convert("RGB"))
        absrel = np.where(m, np.abs(da3a - sensor[i]) / np.maximum(sensor[i], 1e-3), np.nan)
        panes = [(img, None, "RGB", None),
                 (da3a, "turbo", f"DA3 (ours), SSI-aligned", (vmin, vmax)),
                 (np.where(m, sensor[i], np.nan), "turbo", "MoSca depth (LiDAR sensor)", (vmin, vmax)),
                 (absrel, "inferno", "AbsRel error", (0, 0.5))]
        for c, (data, cmap, title, vlim) in enumerate(panes):
            ax = axs[r, c]
            if cmap is None:
                ax.imshow(data)
            else:
                im = ax.imshow(data, cmap=cmap, vmin=vlim[0], vmax=vlim[1])
                fig.colorbar(im, ax=ax, shrink=0.7)
            ax.set_title(f"frame {i} — {title}" + ("" if c == 0 else ""), fontsize=9); ax.axis("off")
    fig.suptitle(f"{args.scene}: DepthAnything-3 (ours) vs MoSca LiDAR depth", fontsize=13)
    fig.tight_layout(); fig.savefig(out / "depth_panels.png", dpi=95, bbox_inches="tight"); plt.close(fig)

    # --- metrics over time ---
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    ax[0].plot([d["absrel"] for d in raw], label="raw DA3", alpha=0.7)
    ax[0].plot([d["absrel"] for d in aligned], label="SSI-aligned DA3", alpha=0.9)
    ax[0].set_title("AbsRel vs LiDAR over frames"); ax[0].set_xlabel("frame"); ax[0].set_ylabel("AbsRel"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot([d["delta1"] for d in raw], label="raw DA3", alpha=0.7)
    ax[1].plot([d["delta1"] for d in aligned], label="SSI-aligned DA3", alpha=0.9)
    ax[1].set_title("delta<1.25 accuracy over frames"); ax[1].set_xlabel("frame"); ax[1].set_ylabel("delta1"); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "depth_metrics.png", dpi=95, bbox_inches="tight"); plt.close(fig)
    print(f"\nwrote {out}/depth_panels.png and depth_metrics.png")


if __name__ == "__main__":
    main()
