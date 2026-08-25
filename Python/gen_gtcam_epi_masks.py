"""Approach 2: find dynamic objects our SAM mask MISSED, using accurate camera poses.

Uses the DyCheck GT training cameras (readgt_dycheck; a perfect static solve) to build the
exact fundamental matrix between neighbouring frames, then the per-pixel Sampson epipolar error
of the RAFT flow correspondences (outputs_mosca/flow_raft). Static pixels satisfy the epipolar
constraint (~0.1px); truly-moving pixels do not. High error OUTSIDE the current SAM mask =
a moving object the mask missed -> add it.

Contrast with gen_mosca_epi_masks.py (approach 1), which estimates F with LMEDS from the flow
itself (no cameras) and is much noisier.

Viz mode writes a montage: [RGB | approach-1 MoSca | approach-2 GTcam-augmented | epi heatmap].
Write mode saves the augmented dynamic-positive masks (SAM union missed-dynamic).
"""
import argparse, glob, os, json, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils"))
from readgt_dycheck import _K_from_camera, _extrinsics_from_camera  # noqa: E402
import gen_mosca_epi_masks as a1  # noqa: E402


def load_cams(sdir, W):
    cams = []
    for f in sorted(glob.glob(os.path.join(sdir, "cameras", "*.json"))):
        c = json.load(open(f))
        scale = W / float(c["image_size"][0])
        K = _K_from_camera(c, scale)
        R, t = _extrinsics_from_camera(c)
        cams.append((K, R, t))
    return cams


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def F_between(c1, c2):
    K1, R1, t1 = c1
    K2, R2, t2 = c2
    Rrel = R2 @ R1.T
    trel = t2 - Rrel @ t1
    E = skew(trel) @ Rrel
    return np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)


def sampson(x1, x2, F):
    h1 = np.concatenate([x1, np.ones((len(x1), 1))], 1)
    h2 = np.concatenate([x2, np.ones((len(x2), 1))], 1)
    d1 = h1 @ F.T
    d2 = h2 @ F
    z = (h2 * d1).sum(1)
    return z ** 2 / (d1[:, 0] ** 2 + d1[:, 1] ** 2 + d2[:, 0] ** 2 + d2[:, 1] ** 2 + 1e-12)


def gtcam_epi_err(idx, imgs, names, flowdir, cams, pix, H, W):
    errs = []
    for nb in ([idx - 1] if idx - 1 >= 0 else []) + ([idx + 1] if idx + 1 < len(imgs) else []):
        fp = os.path.join(flowdir, f"{names[idx]}_to_{names[nb]}.npz")
        if not os.path.exists(fp):
            continue
        fl = np.load(fp)["flow"].reshape(-1, 2).astype(np.float64)
        F = F_between(cams[idx], cams[nb])
        e = np.sqrt(sampson(pix, pix + fl, F)).reshape(H, W)  # pixel units
        errs.append(e)
    if not errs:
        return np.zeros((H, W), np.float32)
    return np.amax(np.stack(errs, 0), 0)


K1_ = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
K2_ = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def missed_dynamic(err, sam, th, min_area_frac):
    """High epi error outside the (dilated) SAM mask, cleaned to real blobs."""
    H, W = err.shape
    sam_d = cv2.dilate(sam.astype(np.uint8), K2_) > 0
    hot = ((err > th) & ~sam_d).astype(np.uint8)
    hot = cv2.morphologyEx(hot, cv2.MORPH_OPEN, K1_)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(hot, connectivity=8)
    keep = np.zeros_like(hot)
    min_area = max(30, int(min_area_frac * H * W))
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = 1
    keep = cv2.dilate(keep, K2_)
    return keep > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--root", default="../datasets/Dycheck")
    ap.add_argument("--th", type=float, default=2.0, help="Sampson error threshold in pixels")
    ap.add_argument("--min-area-frac", type=float, default=0.002)
    ap.add_argument("--a1-error-factor", type=float, default=350.0)
    ap.add_argument("--out-name", default="masks_gtcam")
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--viz-path", default=None)
    ap.add_argument("--n-viz", type=int, default=6)
    args = ap.parse_args()

    sdir = os.path.join(args.root, args.scene)
    flowdir = os.path.join(sdir, "outputs_mosca", "flow_raft")
    imgs = sorted(glob.glob(os.path.join(sdir, "images", "*.png")))
    names = [os.path.basename(f) for f in imgs]
    sams = sorted(glob.glob(os.path.join(sdir, "masks", "*.png")))
    H, W = cv2.imread(imgs[0]).shape[:2]
    cams = load_cams(sdir, W)
    assert len(cams) == len(imgs), f"cams {len(cams)} != imgs {len(imgs)}"
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pix = np.stack([xx.ravel(), yy.ravel()], 1).astype(np.float64)
    # approach-1 setup
    uv = a1.uv_grid(H, W).reshape(-1, 2).astype(np.float32)
    fac = (H + W) / 2.0

    def sam_of(i):
        s = cv2.imread(sams[i], 0)
        return (cv2.resize(s, (W, H), interpolation=cv2.INTER_NEAREST) > 127)

    if args.viz:
        idxs = np.linspace(3, len(imgs) - 4, args.n_viz).astype(int)
        rows = []
        for i in idxs:
            rgb = cv2.imread(imgs[i])
            sam = sam_of(i)
            # approach 1
            m1 = a1.epi_mask(i, imgs, names, flowdir, uv, H, W, fac, args.a1_error_factor)
            ov1 = rgb.copy(); ov1[m1] = (0.4 * ov1[m1] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)
            # approach 2
            err = gtcam_epi_err(i, imgs, names, flowdir, cams, pix, H, W)
            miss = missed_dynamic(err, sam, args.th, args.min_area_frac)
            ov2 = rgb.copy()
            ov2[sam] = (0.5 * ov2[sam] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)   # SAM red
            ov2[miss] = (0.3 * ov2[miss] + 0.7 * np.array([0, 255, 0])).astype(np.uint8)  # missed green
            # heatmap
            hm = np.clip(err / 6.0, 0, 1)
            hm = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
            rows.append(np.hstack([rgb, ov1, ov2, hm]))
        out = args.viz_path or f"/tmp/claude-1000/-home-fenchel-4DReconstruction/63b940a8-90bb-4ad3-9b46-b21903e0da69/scratchpad/bothmasks_{args.scene}.png"
        cv2.imwrite(out, np.vstack(rows))
        print("wrote", out, "cols=[RGB | MoSca(a1,green) | GTcam(a2: SAM red + missed green) | epi heatmap]")
        return

    odir = os.path.join(sdir, args.out_name)
    os.makedirs(odir, exist_ok=True)
    add = []
    for i, nm in enumerate(names):
        sam = sam_of(i)
        err = gtcam_epi_err(i, imgs, names, flowdir, cams, pix, H, W)
        miss = missed_dynamic(err, sam, args.th, args.min_area_frac)
        aug = sam | miss
        add.append(miss.mean())
        cv2.imwrite(os.path.join(odir, nm), (aug.astype(np.uint8) * 255))
    print(f"[{args.scene}] wrote {len(names)} augmented masks to {odir}; "
          f"added(missed-dyn) mean {np.mean(add):.4f} max {np.max(add):.4f}")


if __name__ == "__main__":
    main()
