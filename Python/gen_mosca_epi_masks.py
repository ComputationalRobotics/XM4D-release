"""Generate dynamic masks with MoSca's epipolar-motion-mask decider (lib_prior/epi_error/
epipolar_motion_mask_helper.py::thread) from the RAFT flows MoSca already precomputed
(outputs_mosca/flow_raft). Replicates MoSca's recipe exactly:
  err = sampson(x1, x1+flow, F_LMEDS) * ((H+W)/2)^2   (max over the fwd/bwd neighbour)
  err[err <= quantile(err, 0.8)] = 0
  mask = binary_opening(err > H*W/error_factor^2, disk1) ; dilation(disk2)
MoSca saves the INVERSE (static=255); we save dynamic_positive (dynamic=255) to match our pipeline.

Usage:
  python gen_mosca_epi_masks.py --scene paper-windmill --error-factor 90 --out-name masks_epi
  python gen_mosca_epi_masks.py --scene paper-windmill --error-factor 90 --viz   # montage only, no write
"""
import argparse, glob, os
import numpy as np
import cv2


def uv_grid(H, W):
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")
    xx = 2 * (xx + 0.5) / W - 1
    yy = 2 * (yy + 0.5) / H - 1
    return np.stack([xx, yy], -1)


def sampson(x1, x2, F):
    h1 = np.concatenate([x1, np.ones((len(x1), 1), np.float32)], 1)
    h2 = np.concatenate([x2, np.ones((len(x2), 1), np.float32)], 1)
    d1 = h1 @ F.T
    d2 = h2 @ F
    z = (h2 * d1).sum(1)
    return z ** 2 / (d1[:, 0] ** 2 + d1[:, 1] ** 2 + d2[:, 0] ** 2 + d2[:, 1] ** 2 + 1e-12)


K1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
K2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def epi_mask(idx, imgs, names, flowdir, uv, H, W, fac, error_factor):
    errs = []
    nbs = ([idx - 1] if idx - 1 >= 0 else []) + ([idx + 1] if idx + 1 < len(imgs) else [])
    for nb in nbs:
        fp = os.path.join(flowdir, f"{names[idx]}_to_{names[nb]}.npz")
        if not os.path.exists(fp):
            continue
        fl = np.load(fp)["flow"].astype(np.float32)
        fln = np.stack([2 * fl[..., 0] / (W - 1), 2 * fl[..., 1] / (H - 1)], -1).reshape(-1, 2)
        x2 = uv + fln
        F, _ = cv2.findFundamentalMat(uv, x2, cv2.FM_LMEDS)
        if F is None or F.shape != (3, 3):
            continue
        e = sampson(uv, x2, F.astype(np.float32)).reshape(H, W) * fac ** 2
        errs.append(e)
    if not errs:
        return np.zeros((H, W), bool)
    err = np.amax(np.stack(errs, 0), 0)
    th = np.quantile(err, 0.8)
    err = np.where(err <= th, 0, err)
    m = (err > ((H * W) / (error_factor ** 2))).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, K1)
    m = cv2.dilate(m, K2)
    return m.astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--root", default="datasets/Dycheck")
    ap.add_argument("--error-factor", type=float, default=90.0)
    ap.add_argument("--out-name", default="masks_epi")
    ap.add_argument("--viz", action="store_true", help="only write a comparison montage, do not save masks")
    ap.add_argument("--viz-path", default=None)
    args = ap.parse_args()

    sdir = os.path.join(args.root, args.scene)
    flowdir = os.path.join(sdir, "outputs_mosca", "flow_raft")
    imgs = sorted(glob.glob(os.path.join(sdir, "images", "*.png")))
    names = [os.path.basename(f) for f in imgs]
    assert imgs, f"no images in {sdir}/images"
    assert os.path.isdir(flowdir), f"no flow_raft in {flowdir}"
    H, W = cv2.imread(imgs[0]).shape[:2]
    uv = uv_grid(H, W).reshape(-1, 2).astype(np.float32)
    fac = (H + W) / 2.0

    if args.viz:
        sam = sorted(glob.glob(os.path.join(sdir, "masks", "*.png")))
        idxs = np.linspace(0, len(imgs) - 1, 6).astype(int)
        rows = []
        for i in idxs:
            rgb = cv2.imread(imgs[i])
            m = epi_mask(i, imgs, names, flowdir, uv, H, W, fac, args.error_factor)
            s = cv2.imread(sam[i], 0)
            s = cv2.resize(s, (W, H), interpolation=cv2.INTER_NEAREST) > 127
            ov = rgb.copy()
            ov[s] = (0.5 * ov[s] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)   # SAM red
            ov[m] = (0.4 * ov[m] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)   # epi green
            rows.append(np.hstack([rgb, ov]))
        out = args.viz_path or f"/tmp/claude-1000/-home-fenchel-4DReconstruction/63b940a8-90bb-4ad3-9b46-b21903e0da69/scratchpad/moscaepi_{args.scene}_ef{int(args.error_factor)}.png"
        cv2.imwrite(out, np.vstack(rows))
        print("wrote viz", out)
        return

    odir = os.path.join(sdir, args.out_name)
    os.makedirs(odir, exist_ok=True)
    covs = []
    for i, nm in enumerate(names):
        m = epi_mask(i, imgs, names, flowdir, uv, H, W, fac, args.error_factor)
        covs.append(m.mean())
        cv2.imwrite(os.path.join(odir, nm), (m.astype(np.uint8) * 255))
    print(f"[{args.scene}] wrote {len(names)} masks to {odir} (ef={args.error_factor}); "
          f"coverage mean {np.mean(covs):.3f} min {np.min(covs):.3f} max {np.max(covs):.3f}")


if __name__ == "__main__":
    main()
