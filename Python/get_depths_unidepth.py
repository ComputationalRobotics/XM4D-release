"""Step 1 with UniDepth V2 (config.json "DEPTH_MODEL": "unidepth"): metric depth, its confidence and the intrinsics,
in the `unidepth` environment. Same outputs as get_depths_clean.py (Depth Anything 3), in outputs/unidepth/.

- video: every frame gets one shared K: config.json INTRINSICS if known, else the median of UniDepth's own focal over
  up to 50 evenly spaced frames, with the principal point at the image centre. The depth of every frame is then
  predicted with that K, so the depth and the K the solver lifts with agree. Writes depth_maps.npy, conf_maps.npy
  (T, H, W, at the frame size) and intrinsics.npy (T, 3, 3).
- photos: each photo with its own K (INTRINSICS if known, else UniDepth's), in photos/<photo name>.npz.

The confidence is clipped to its 1st-99th percentile over the scene (it is heavy-tailed; the solver weights its
observations with it).
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from check_masks import require_masks
import utils.definitions
from utils.script_logger import install_script_logger

require_masks()  # step 0: the scene's moving-object masks must exist (make_masks_sam3.ipynb)

sys.path.append(str(utils.definitions.REPO_ROOT / "deps" / "UniDepth"))
from unidepth.models import UniDepthV2  # noqa: E402
from unidepth.utils.camera import Pinhole  # noqa: E402

MODEL = "lpiccinelli/unidepth-v2-vitl14"
K_FRAMES = 50   # video: frames whose predicted focal gives the shared K


def read_rgb(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def known_K(width, height):
    """config.json INTRINSICS[<scene>] scaled to a frame of this size, or None."""
    known = (utils.definitions.active_config.get("INTRINSICS") or {}).get(utils.definitions.dataset_name)
    if not known:
        return None
    kx, ky = width / known["width"], height / known["height"]
    return np.array([[known["fx"] * kx, 0.0, known["cx"] * kx],
                     [0.0, known["fy"] * ky, known["cy"] * ky],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


@torch.no_grad()
def infer(model, rgb, K=None):
    """depth (H, W), confidence (H, W) at the image size and the K UniDepth used (given or predicted)."""
    H, W = rgb.shape[:2]
    x = torch.from_numpy(rgb).permute(2, 0, 1)
    cam = None if K is None else Pinhole(K=torch.from_numpy(np.asarray(K, np.float32))[None])
    pred = model.infer(x, cam) if cam is not None else model.infer(x)
    depth = pred["depth"].squeeze().float().cpu().numpy()
    conf = pred["confidence"].squeeze().float().cpu().numpy()
    if depth.shape != (H, W):
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
    if conf.shape != (H, W):
        conf = cv2.resize(conf, (W, H), interpolation=cv2.INTER_LINEAR)
    K_used = np.asarray(K if K is not None else pred["intrinsics"].squeeze().cpu().numpy(), dtype=np.float64)
    return depth.astype(np.float32), conf.astype(np.float32), K_used


def conf_range(samples):
    """The 1st and 99th percentile of the confidence samples of the whole scene."""
    lo, hi = np.percentile(np.concatenate(samples), [1.0, 99.0])
    return max(float(lo), 1e-6), float(hi)


def run_video(model, out_dir):
    paths = sorted(utils.definitions.video_dir.glob(utils.definitions.image_pattern))
    if not paths:
        raise RuntimeError(f"No frames matching {utils.definitions.image_pattern} in {utils.definitions.video_dir}")
    H, W = read_rgb(paths[0]).shape[:2]
    K = known_K(W, H)
    if K is None:
        idx = np.unique(np.linspace(0, len(paths) - 1, min(K_FRAMES, len(paths))).round().astype(int))
        focals = []
        for i in idx:
            _, _, K_i = infer(model, read_rgb(paths[i]))
            focals.append(0.5 * (K_i[0, 0] + K_i[1, 1]))
        f = float(np.median(focals))
        K = np.array([[f, 0.0, W / 2.0], [0.0, f, H / 2.0], [0.0, 0.0, 1.0]])
        print(f"[unidepth] shared K: f = median focal of {len(idx)} frames = {f:.2f} "
              f"(min {min(focals):.2f}, max {max(focals):.2f}), principal point at the centre")
    else:
        print(f"[unidepth] config.json INTRINSICS: fx={K[0, 0]:.3f} fy={K[1, 1]:.3f} cx={K[0, 2]:.3f} cy={K[1, 2]:.3f}")
    depths, confs = [], []
    for t, path in enumerate(paths):
        rgb = read_rgb(path)
        if rgb.shape[:2] != (H, W):
            raise RuntimeError(f"{path.name} is {rgb.shape[1]}x{rgb.shape[0]}, the first frame {W}x{H}: "
                               'the frames of a video share one size (use "input": "photos" otherwise)')
        d, c, _ = infer(model, rgb, K)
        depths.append(d)
        confs.append(c)
        if (t + 1) % 50 == 0 or t + 1 == len(paths):
            print(f"[unidepth] {t + 1}/{len(paths)} frames", flush=True)
    np.save(out_dir / "depth_maps.npy", np.stack(depths))
    lo, hi = conf_range([c.ravel()[::97] for c in confs])
    np.save(out_dir / "conf_maps.npy", np.clip(np.stack(confs), lo, hi))
    np.save(out_dir / "intrinsics.npy", np.repeat(K.astype(np.float32)[None], len(paths), axis=0))
    print(f"Saved depth maps, confidence maps and intrinsics of {len(paths)} frames to {out_dir}")


def run_photos(model):
    paths = utils.definitions.photo_paths()
    if not paths:
        raise RuntimeError(f"No photos (.jpg, .jpeg, .png) in {utils.definitions.video_dir}")
    out_dir = utils.definitions.photo_depth_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for i, path in enumerate(paths):
        rgb = read_rgb(path)
        depth, conf, K = infer(model, rgb, known_K(rgb.shape[1], rgb.shape[0]))
        np.savez(out_dir / f"{path.name}.npz", depth=depth, conf=conf, K=K)
        samples.append(conf.ravel()[::97])
        if (i + 1) % 25 == 0 or i + 1 == len(paths):
            print(f"[unidepth] {i + 1}/{len(paths)} photos", flush=True)
    lo, hi = conf_range(samples)
    for path in paths:   # clip the confidence to the scene's range
        f = out_dir / f"{path.name}.npz"
        z = dict(np.load(f))
        z["conf"] = np.clip(z["conf"], lo, hi)
        np.savez(f, **z)
    print(f"Saved depth, confidence and K of {len(paths)} photos to {out_dir}")


def main():
    if utils.definitions.depth_model != "unidepth":
        sys.exit(f'[depth] DEPTH_MODEL is "{utils.definitions.depth_model}" in config.json: run get_depths_clean.py '
                 '(Depth Anything 3) instead, or set "DEPTH_MODEL": "unidepth"')
    if utils.definitions.active_config.get("LIDAR_DEPTH"):
        print("[unidepth] LIDAR_DEPTH is only used with Depth Anything 3; ignored here")
    out_dir = utils.definitions.depth_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    install_script_logger("depth_estimate", utils.definitions.output_dir)
    model = UniDepthV2.from_pretrained(MODEL)
    model.interpolation_mode = "bilinear"
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    if utils.definitions.photos:
        run_photos(model)
    else:
        run_video(model, out_dir)


if __name__ == "__main__":
    main()
