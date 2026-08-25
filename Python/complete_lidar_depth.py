#!/usr/bin/env python3
"""Complete the LiDAR depth into a DENSE depth map: keep every LiDAR-covered pixel EXACT, and fill the
holes (object-edge gaps + far background) guided by the RGB image -- i.e. RGB-guided colorization
(Levin et al.), the standard depth-completion approach. This is NOT DA3-to-LiDAR alignment: the covered
region is byte-for-byte the LiDAR, and the filled region is interpolated from the given LiDAR values while
respecting image edges.

Outputs (to --out): completed_lidar_depth.npy (n,H,W) [with --all], a before/after panel PNG, and prints
how much was filled. Caveat: the far background (>3 m, no LiDAR return) has no nearby depth to base on, so
it is filled by propagating the surrounding (window-frame) depth -- geometrically arbitrary there, but that
region is typically outside the covisible eval mask.

Example:
  conda run -n gsplat python complete_lidar_depth.py --scene wheel                 # preview a few frames
  conda run -n gsplat python complete_lidar_depth.py --scene wheel --all           # batch all + save npy
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.ndimage import uniform_filter
from PIL import Image

NBRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def fill_depth_colorization(rgb: np.ndarray, depth: np.ndarray, alpha: float = 100.0) -> np.ndarray:
    """rgb in [0,1] (H,W,3), depth (H,W) with <=0 = missing. Returns dense depth; valid pixels preserved."""
    H, W = depth.shape
    N = H * W
    gray = rgb @ np.array([0.299, 0.587, 0.114])
    idx = np.arange(N).reshape(H, W)
    known = depth > 1e-6

    # per-pixel local variance of the guidance -> affinity sigma (Levin)
    mu = uniform_filter(gray, 3, mode="reflect")
    var = uniform_filter(gray * gray, 3, mode="reflect") - mu * mu
    csig = np.maximum(var * 0.6, 2e-6)

    rows, cols, vals = [], [], []
    wsum = np.zeros((H, W))
    wstore = []
    for (di, dj) in NBRS:
        sl_i = (slice(max(0, di), H + min(0, di)), slice(max(0, dj), W + min(0, dj)))
        sl_j = (slice(max(0, -di), H + min(0, -di)), slice(max(0, -dj), W + min(0, -dj)))
        d2 = (gray[sl_i] - gray[sl_j]) ** 2
        w = np.exp(-d2 / csig[sl_i])
        wsum[sl_i] += w
        wstore.append((sl_i, sl_j, w))
    for (sl_i, sl_j, w) in wstore:
        wn = w / np.maximum(wsum[sl_i], 1e-12)
        rows.append(idx[sl_i].ravel()); cols.append(idx[sl_j].ravel()); vals.append(-wn.ravel())
    # diagonal (self) = 1
    rows.append(idx.ravel()); cols.append(idx.ravel()); vals.append(np.ones(N))
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), (N, N))

    # soft data term on known pixels: (A^T A + alpha*diag(known)) d = alpha*known*depth
    kflat = known.ravel().astype(np.float64)
    G = sp.diags(kflat * alpha)
    b = kflat * depth.ravel() * alpha
    out = spsolve(A.T @ A + G, b).reshape(H, W)
    out[known] = depth[known]            # keep covered region EXACT
    return out.astype(np.float32)


def fill_depth_da3(da3: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Fill holes using DA3's RELATIVE depth anchored to LiDAR via a Poisson (harmonic-offset) blend.
    Valid LiDAR pixels kept EXACT; holes take DA3's shape (scale+shift-fit to LiDAR) with a smooth offset
    so they match LiDAR seamlessly at the hole boundary. da3,(H,W) rel depth; depth,(H,W) LiDAR (<=0=hole)."""
    H, W = depth.shape; N = H * W
    valid = depth > 1e-6
    # scale+shift align DA3 to LiDAR on covered pixels -> DA3 in LiDAR's metric
    p = da3[valid]; A = np.stack([p, np.ones_like(p)], 1)
    ab, *_ = np.linalg.lstsq(A, depth[valid], rcond=None)
    das = ab[0] * da3 + ab[1]
    # solve harmonic offset r: r = (lidar - das) on valid (Dirichlet), laplace(r)=0 in holes
    idx = np.arange(N).reshape(H, W)
    rows = [idx[valid]]; cols = [idx[valid]]; vals = [np.ones(valid.sum())]
    b = np.zeros(N); b[idx[valid]] = (depth[valid] - das[valid])
    deg = np.zeros((H, W))
    nbr = []
    for (di, dj) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        sl_i = (slice(max(0, di), H + min(0, di)), slice(max(0, dj), W + min(0, dj)))
        sl_j = (slice(max(0, -di), H + min(0, -di)), slice(max(0, -dj), W + min(0, -dj)))
        deg[sl_i] += 1; nbr.append((sl_i, sl_j))
    hole = ~valid
    rows.append(idx[hole]); cols.append(idx[hole]); vals.append(deg[hole])   # hole diagonal = #in-bounds nbrs
    for (sl_i, sl_j) in nbr:
        hm = hole[sl_i]
        rows.append(idx[sl_i][hm]); cols.append(idx[sl_j][hm]); vals.append(-np.ones(hm.sum()))
    Amat = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), (N, N))
    r = spsolve(Amat, b).reshape(H, W)
    out = das + r
    out[valid] = depth[valid]            # keep covered region EXACT
    return out.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="wheel")
    ap.add_argument("--root", type=Path, default=Path("/home/fenchel/4DReconstruction/datasets/Dycheck"))
    ap.add_argument("--method", choices=["da3", "colorization"], default="da3",
                    help="da3: fill holes with DA3 relative depth anchored to LiDAR (Poisson); "
                         "colorization: RGB-guided propagation of LiDAR values")
    ap.add_argument("--all", action="store_true", help="process all frames and save completed_lidar_depth_<method>.npy")
    ap.add_argument("--panels", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    sdir = args.root / args.scene
    out = args.out or (sdir / "outputs" / "depth_compare")
    os.makedirs(out, exist_ok=True)
    sd = sdir / "outputs_mosca" / "sensor_depth"
    n = len(list(sd.glob("frame*.npz")))

    def lidar(i): return np.load(sd / f"frame{i:06d}.npz")["dep"].astype(np.float32)
    def rgb(i): return np.asarray(Image.open(sdir / "images" / f"frame{i:06d}.png").convert("RGB")) / 255.0
    da3_all = np.load(sdir / "outputs" / "depth_anything_3" / "depth_maps.npy") if args.method == "da3" else None
    def fill(i, g):
        return fill_depth_da3(da3_all[i], g) if args.method == "da3" else fill_depth_colorization(rgb(i), g, args.alpha)

    frames = list(range(n)) if args.all else list(np.linspace(0, n - 1, args.panels).astype(int))
    completed = {}
    for i in frames:
        g = lidar(i)
        c = fill(i, g)
        completed[i] = c
        filled = (g <= 1e-6)
        assert np.allclose(c[~filled], g[~filled]), "covered region changed!"
        if args.all and i % 50 == 0:
            print(f"  frame {i}/{n} filled {filled.mean()*100:.1f}%")

    # panel viz
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    pf = list(np.linspace(0, n - 1, args.panels).astype(int))
    fig, axs = plt.subplots(len(pf), 4, figsize=(15, 3.6 * len(pf)))
    if len(pf) == 1: axs = axs[None, :]
    for r, i in enumerate(pf):
        g = lidar(i); c = completed[i] if i in completed else fill(i, g)
        vmin, vmax = np.percentile(g[g > 0], [2, 98])
        holes = np.where(g > 0, np.nan, 1.0)
        panes = [((rgb(i)), None, "RGB", None),
                 (np.where(g > 0, g, np.nan), "turbo", f"LiDAR (holes={(g<=1e-6).mean()*100:.0f}%)", (vmin, vmax)),
                 (c, "turbo", f"completed ({args.method})", (vmin, vmax)),
                 (holes, "Reds", "filled region", (0, 1))]
        for cc, (data, cmap, title, vlim) in enumerate(panes):
            ax = axs[r, cc]
            if cmap is None: ax.imshow(data)
            elif cc == 3:
                ax.imshow(rgb(i)); ax.imshow(data, cmap="autumn", alpha=0.6)
            else:
                im = ax.imshow(data, cmap=cmap, vmin=vlim[0], vmax=vlim[1]); fig.colorbar(im, ax=ax, shrink=0.7)
            ax.set_title(f"frame {i} — {title}", fontsize=9); ax.axis("off")
    fig.suptitle(f"{args.scene}: LiDAR depth completion [{args.method}] (covered region kept exact)", fontsize=13)
    fig.tight_layout(); fig.savefig(out / f"lidar_completion_{args.method}.png", dpi=95, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}/lidar_completion_{args.method}.png")

    if args.all:
        arr = np.stack([completed[i] for i in range(n)])
        np.save(out / f"completed_lidar_depth_{args.method}.npy", arr)
        print(f"saved {out}/completed_lidar_depth_{args.method}.npy  shape={arr.shape}")


if __name__ == "__main__":
    main()
