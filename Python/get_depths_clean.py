import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from scipy.stats import trim_mean

import open3d as o3d
from utils import dataset_config

# DA3 predicts poorly on extreme aspect ratios (e.g. Sintel's 2.35:1). Optionally cap the
# input aspect before inference so it stays near DA3's training distribution; the depth is
# resized back afterwards, and because the resize is per-axis linear the intrinsic rescaling
# (sx=W_orig/W_pred, sy=H_orig/H_pred) still maps DA3's predicted K back correctly.
#
# These depth-stage knobs are per-dataset config values (Python/configs/*.json), so only the
# datasets that set them are affected. Defaults are DA3's originals -- no aspect cap,
# process_res 504, per-frame intrinsics -- so other datasets reproduce unchanged and don't
# OOM on wide images. Env vars still override. Enabled for Sintel in configs/sintel.json.
#   DA3_ASPECT_CAP        <= 1.0 disables; e.g. 2.0 caps the long side to 2:1
#   DA3_PROCESS_RES       DA3 internal inference resolution (default 504)
#   DA3_PROCESS_RES_METHOD "upper_bound_resize" (default) or "upper_bound_crop"
#   DA3_SINGLE_INTRINSIC  collapse per-frame DA3 K to one shared pinhole (median focal)
ASPECT_CAP = dataset_config.get_float("DA3_ASPECT_CAP", 0.0, env="DA3_ASPECT_CAP")
PROCESS_RES = dataset_config.get_int("DA3_PROCESS_RES", 504, env="DA3_PROCESS_RES")
PROCESS_RES_METHOD = str(
    dataset_config._resolve("DA3_PROCESS_RES_METHOD", "upper_bound_resize", "DA3_PROCESS_RES_METHOD")
)
SINGLE_INTRINSIC = dataset_config.get_bool("DA3_SINGLE_INTRINSIC", False, env="DA3_SINGLE_INTRINSIC")


def _resize_to_aspect_cap(img_path, cap=ASPECT_CAP, intrinsic=None):
    """Load an RGB image and, if its aspect (long/short side) exceeds `cap`,
    anisotropically resize it to exactly `cap` by shrinking the long side.
    Returns (PIL.Image, intrinsic scaled to the new size or None).
    cap <= 1.0 disables capping (returns the original image).
    """
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    ratio = W / float(H)
    if cap <= 1.0:
        return img, (None if intrinsic is None else np.asarray(intrinsic, dtype=np.float32))
    if ratio > cap:
        new_W, new_H = int(round(cap * H)), H
    elif ratio < 1.0 / cap:
        new_W, new_H = W, int(round(cap * W))
    else:
        return img, (None if intrinsic is None else np.asarray(intrinsic, dtype=np.float32))
    img_r = img.resize((new_W, new_H), Image.BILINEAR)
    if intrinsic is None:
        return img_r, None
    K = np.asarray(intrinsic, dtype=np.float32).copy()
    ax, ay = new_W / float(W), new_H / float(H)
    K[0, 0] *= ax
    K[0, 2] *= ax
    K[1, 1] *= ay
    K[1, 2] *= ay
    return img_r, K

from depth_anything_3.api import DepthAnything3

import utils.definitions
import utils.dataset_config as dataset_config
from utils.script_logger import install_script_logger

# Ensure Depth-Anything-3 module is importable when running from Python/
REPO_ROOT = utils.definitions.rootpath.parents[1]
sys.path.append(str(REPO_ROOT / "deps" / "Depth-Anything-3" / "src"))

from utils.readgt_sintel import load_sintel_gt
from utils.readgt_TUM import load_tum_gt
from utils.readgt_replica import load_replica_gt
from utils.readgt_colmap import load_colmap_gt
from utils.readgt_dycheck import load_dycheck_gt

try:
    import teaserpp_python
except ImportError:
    teaserpp_python = None


def _as_homogeneous(extrinsic: np.ndarray) -> np.ndarray:
    extrinsic = np.asarray(extrinsic, dtype=np.float64)
    if extrinsic.shape == (4, 4):
        return extrinsic
    if extrinsic.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :] = extrinsic
        return out
    raise ValueError(f"Expected extrinsic shape (3, 4) or (4, 4), got {extrinsic.shape}")


def _rotmat_to_colmap_qvec(rot: np.ndarray) -> np.ndarray:
    q_xyzw = Rotation.from_matrix(rot).as_quat()
    qvec = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)
    if qvec[0] < 0:
        qvec *= -1.0
    return qvec


def _write_points3d_txt(path: Path) -> None:
    path.write_text(
        "# 3D point list with one line of data per point:\n"
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        "# Number of points: 0, mean track length: 0\n"
    )


def _write_depthanything_colmap_sparse(
    sparse_dir: Path,
    image_paths: list[str],
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    width: int,
    height: int,
) -> None:
    """Write DA3 predicted cameras as COLMAP text.

    With use_ray_pose=True, DA3 returns camera-to-world extrinsics. COLMAP
    images.txt stores world-to-camera, so each pose is inverted here.
    """
    sparse_dir.mkdir(parents=True, exist_ok=True)

    intr0 = np.asarray(intrinsics[0], dtype=np.float64)
    fx, fy = intr0[0, 0], intr0[1, 1]
    cx, cy = intr0[0, 2], intr0[1, 2]
    (sparse_dir / "cameras.txt").write_text(
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        "# Number of cameras: 1\n"
        f"1 PINHOLE {width} {height} {fx:.17g} {fy:.17g} {cx:.17g} {cy:.17g}\n"
    )

    lines = [
        "# Image list with two lines of data per image:\n",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)\n",
        f"# Number of images: {len(image_paths)}, mean observations per image: 0\n",
    ]
    for i, image_path in enumerate(image_paths):
        c2w = _as_homogeneous(extrinsics[i])
        w2c = np.linalg.inv(c2w)
        rot = w2c[:3, :3]
        trans = w2c[:3, 3]
        qvec = _rotmat_to_colmap_qvec(rot)
        values = [
            str(i + 1),
            *(f"{v:.17g}" for v in qvec),
            *(f"{v:.17g}" for v in trans),
            "1",
            Path(image_path).name,
        ]
        lines.append(" ".join(values) + "\n")
        lines.append("\n")
    (sparse_dir / "images.txt").write_text("".join(lines))
    _write_points3d_txt(sparse_dir / "points3D.txt")


def _compute_overlap_depth_scale(
    prev_depth: np.ndarray,
    curr_depth: np.ndarray,
    prev_conf: np.ndarray | None = None,
    curr_conf: np.ndarray | None = None,
) -> float:
    prev_depth = np.asarray(prev_depth, dtype=np.float64)
    curr_depth = np.asarray(curr_depth, dtype=np.float64)
    valid = (
        np.isfinite(prev_depth)
        & np.isfinite(curr_depth)
        & (prev_depth > 1e-8)
        & (curr_depth > 1e-8)
    )
    if prev_conf is not None:
        valid &= np.isfinite(prev_conf) & (np.asarray(prev_conf, dtype=np.float64) > 0)
    if curr_conf is not None:
        valid &= np.isfinite(curr_conf) & (np.asarray(curr_conf, dtype=np.float64) > 0)

    if not np.any(valid):
        return 1.0

    ratio = prev_depth[valid] / curr_depth[valid]
    ratio = ratio[np.isfinite(ratio) & (ratio > 1e-8)]
    if ratio.size == 0:
        return 1.0

    if ratio.size >= 16:
        lo = np.percentile(ratio, 5.0)
        hi = np.percentile(ratio, 95.0)
        ratio = ratio[(ratio >= lo) & (ratio <= hi)]
        if ratio.size == 0:
            return 1.0

    scale = float(np.median(ratio))
    if not np.isfinite(scale) or scale <= 1e-8:
        return 1.0
    return scale

def _build_overlap_depth_histogram_matcher(
    prev_depth_list,
    curr_depth_list,
    prev_conf_list=None,
    curr_conf_list=None,
    min_valid: int = 1024,
    num_quantiles: int = 256,
    trim_percentile: float = 2.0,
):
    """
    Build a monotonic mapping f such that:

        matched_curr_depth = f(curr_depth)

    using overlap frames from the previous stitched chunk and current chunk.

    This is quantile matching, not a single scalar scale.
    """
    prev_values = []
    curr_values = []

    if prev_conf_list is None:
        prev_conf_list = [None] * len(prev_depth_list)
    if curr_conf_list is None:
        curr_conf_list = [None] * len(curr_depth_list)

    for prev_depth, curr_depth, prev_conf, curr_conf in zip(
        prev_depth_list,
        curr_depth_list,
        prev_conf_list,
        curr_conf_list,
    ):
        prev_depth = np.asarray(prev_depth, dtype=np.float64)
        curr_depth = np.asarray(curr_depth, dtype=np.float64)

        if prev_depth.shape != curr_depth.shape:
            raise ValueError(
                f"Depth shapes differ: prev={prev_depth.shape}, curr={curr_depth.shape}. "
                "Histogram alignment expects same-resolution overlap predictions."
            )

        valid = (
            np.isfinite(prev_depth)
            & np.isfinite(curr_depth)
            & (prev_depth > 1e-8)
            & (curr_depth > 1e-8)
        )

        if prev_conf is not None:
            prev_conf = np.asarray(prev_conf, dtype=np.float64)
            valid &= np.isfinite(prev_conf) & (prev_conf > 0)

        if curr_conf is not None:
            curr_conf = np.asarray(curr_conf, dtype=np.float64)
            valid &= np.isfinite(curr_conf) & (curr_conf > 0)

        if np.any(valid):
            prev_values.append(prev_depth[valid])
            curr_values.append(curr_depth[valid])

    if len(prev_values) == 0:
        print("[DepthAlign] No valid overlap pixels for histogram matching. Using identity.")
        return None

    prev_values = np.concatenate(prev_values, axis=0)
    curr_values = np.concatenate(curr_values, axis=0)

    valid = (
        np.isfinite(prev_values)
        & np.isfinite(curr_values)
        & (prev_values > 1e-8)
        & (curr_values > 1e-8)
    )
    prev_values = prev_values[valid]
    curr_values = curr_values[valid]

    if prev_values.size < min_valid:
        print(
            f"[DepthAlign] Too few valid overlap pixels for histogram matching: "
            f"{prev_values.size}. Using identity."
        )
        return None

    # Remove extreme outliers jointly using the depth-ratio distribution.
    ratio = prev_values / curr_values
    ratio_valid = np.isfinite(ratio) & (ratio > 1e-8)
    prev_values = prev_values[ratio_valid]
    curr_values = curr_values[ratio_valid]
    ratio = ratio[ratio_valid]

    if ratio.size < min_valid:
        print(
            f"[DepthAlign] Too few valid ratios for histogram matching: "
            f"{ratio.size}. Using identity."
        )
        return None

    lo = np.percentile(ratio, trim_percentile)
    hi = np.percentile(ratio, 100.0 - trim_percentile)
    keep = (ratio >= lo) & (ratio <= hi)

    prev_values = prev_values[keep]
    curr_values = curr_values[keep]

    if prev_values.size < min_valid:
        print(
            f"[DepthAlign] Too few pixels after trimming for histogram matching: "
            f"{prev_values.size}. Using identity."
        )
        return None

    qs = np.linspace(0.0, 1.0, int(num_quantiles), dtype=np.float64)

    curr_knots = np.quantile(curr_values, qs)
    prev_knots = np.quantile(prev_values, qs)

    # Remove duplicate current-depth knots because np.interp requires increasing x.
    curr_unique, unique_idx = np.unique(curr_knots, return_index=True)
    prev_unique = prev_knots[unique_idx]

    if curr_unique.size < 2:
        print("[DepthAlign] Degenerate histogram matcher. Using identity.")
        return None

    # Make the target mapping nondecreasing.
    prev_unique = np.maximum.accumulate(prev_unique)

    median_scale = float(np.median(prev_values / curr_values))

    print(
        f"[DepthAlign] Histogram matcher built: "
        f"valid={prev_values.size}, "
        f"curr_range=({curr_unique[0]:.6f}, {curr_unique[-1]:.6f}), "
        f"prev_range=({prev_unique[0]:.6f}, {prev_unique[-1]:.6f}), "
        f"median_scale={median_scale:.6f}"
    )

    return {
        "curr_knots": curr_unique.astype(np.float32),
        "prev_knots": prev_unique.astype(np.float32),
        "median_scale": float(median_scale),
    }


def _apply_depth_histogram_match(depth: np.ndarray, matcher: dict) -> np.ndarray:
    """
    Apply monotonic histogram-matching depth map to arbitrary depth array.

    Interior values use interpolation.
    Tails use endpoint scale extrapolation to avoid hard clamping.
    """
    if matcher is None:
        return depth

    depth = np.asarray(depth, dtype=np.float32)
    out = depth.copy()

    x = np.asarray(matcher["curr_knots"], dtype=np.float64)
    y = np.asarray(matcher["prev_knots"], dtype=np.float64)

    valid = np.isfinite(out) & (out > 1e-8)
    if not np.any(valid):
        return out

    d = out[valid].astype(np.float64)

    matched = np.interp(d, x, y)

    # Avoid flat clamping outside the learned quantile range.
    low = d < x[0]
    high = d > x[-1]

    low_scale = y[0] / max(x[0], 1e-8)
    high_scale = y[-1] / max(x[-1], 1e-8)

    matched[low] = d[low] * low_scale
    matched[high] = d[high] * high_scale

    matched = np.maximum(matched, 1e-8)

    out[valid] = matched.astype(np.float32)
    return out


def _apply_depth_histogram_match_to_prediction(
    pred,
    matcher: dict | None,
    scale_extrinsics_with_median: bool = False,
):
    """
    Apply nonlinear histogram depth alignment to pred.depth.

    If scale_extrinsics_with_median=True, camera translation is scaled by the
    matcher's median scale. This is only an approximation, because nonlinear
    depth remapping has no exact corresponding SE(3) scale.
    """
    if matcher is None:
        return pred

    pred.depth = _apply_depth_histogram_match(pred.depth, matcher)

    if scale_extrinsics_with_median:
        median_scale = float(matcher.get("median_scale", 1.0))

        if hasattr(pred, "extrinsics") and pred.extrinsics is not None:
            extr = np.asarray(pred.extrinsics)

            if extr.ndim == 3 and extr.shape[1:] == (4, 4):
                pred.extrinsics = extr.astype(np.float32, copy=True)
                pred.extrinsics[:, :3, 3] *= np.float32(median_scale)

            elif extr.ndim == 3 and extr.shape[1:] == (3, 4):
                pred.extrinsics = extr.astype(np.float32, copy=True)
                pred.extrinsics[:, :3, 3] *= np.float32(median_scale)

    return pred

def _apply_depth_scale_to_prediction(pred, scale: float):
    if abs(float(scale) - 1.0) <= 1e-8:
        return pred

    pred.depth = np.asarray(pred.depth, dtype=np.float32) * np.float32(scale)
    if hasattr(pred, "extrinsics") and pred.extrinsics is not None:
        extr = np.asarray(pred.extrinsics)
        if extr.ndim == 3 and extr.shape[1:] == (4, 4):
            pred.extrinsics = extr.astype(np.float32, copy=True)
            pred.extrinsics[:, :3, 3] *= np.float32(scale)
        elif extr.ndim == 3 and extr.shape[1:] == (3, 4):
            pred.extrinsics = extr.astype(np.float32, copy=True)
            pred.extrinsics[:, :3, 3] *= np.float32(scale)
    return pred


def main() -> None:
    # --------------------------------------------------------------
    # CONFIG (no argparse) — follow utils.definitions
    # --------------------------------------------------------------
    dataset_cat = utils.definitions.dataset_cat
    dataset_name = utils.definitions.dataset_name
    input_dir = utils.definitions.video_dir
    output_dir = utils.definitions.da3_depth_dir
    image_pattern = utils.definitions.image_pattern

    # Single intrinsics switch, shared across the whole pipeline (per-dataset config).
    #   USE_DA3_INTRINSICS=False -> use GT intrinsics: load GT, pass to DA3, and
    #                               SAVE GT as the output intrinsic (depth stays GT-based).
    #   USE_DA3_INTRINSICS=True  -> use DA3's own predicted intrinsics: don't pass GT,
    #                               and save DA3's predicted intrinsic.
    USE_DA3_INTRINSICS = dataset_config.get_bool(
        "USE_DA3_INTRINSICS", False, env="XM_USE_DA3_INTRINSICS"
    )
    USE_INTRINSICS_GT = not USE_DA3_INTRINSICS
    print(f"[intrinsics] USE_DA3_INTRINSICS={USE_DA3_INTRINSICS} -> "
          f"{'DA3 predicted' if USE_DA3_INTRINSICS else 'GT (loaded, passed, and saved)'}")

    output_dir.mkdir(parents=True, exist_ok=True)
    install_script_logger("depth_estimate", utils.definitions.output_dir)

    images = sorted(glob.glob(str(Path(input_dir) / image_pattern)))
    if len(images) == 0:
        raise RuntimeError(f"No input frames found in {input_dir}")

    print(f"Found {len(images)} frames in {input_dir}")

    # ------------------------------------------------------------------
    # Load intrinsics from Sintel GT cameras (optional)
    # ------------------------------------------------------------------
    dataset_root = Path(input_dir).parent
    if dataset_cat == "Sintel":
        gt = load_sintel_gt(str(dataset_root)) if USE_INTRINSICS_GT else None
    elif dataset_cat == "TUM":
        gt = load_tum_gt(str(dataset_root)) if USE_INTRINSICS_GT else None
    elif dataset_cat == "Replica":
        gt = load_replica_gt(str(dataset_root)) if USE_INTRINSICS_GT else None
    elif dataset_cat == "Syn4D":
        gt = load_colmap_gt(str(dataset_root / "colmap")) if USE_INTRINSICS_GT else None
    elif dataset_cat == "Dycheck":
        gt = load_dycheck_gt(str(dataset_root)) if USE_INTRINSICS_GT else None
    else:
        gt = None
        USE_INTRINSICS_GT = False

    if gt is not None:
        def _can_parse_float(path: str) -> bool:
            try:
                float(Path(path).stem)
                return True
            except ValueError:
                return False

        if not all(_can_parse_float(p) for p in images):
            ordered_names = [name for name in gt.keys() if (Path(input_dir) / name).exists()]
            if ordered_names:
                images = [str(Path(input_dir) / name) for name in ordered_names]
                print("Using GT frame order from groundtruth keys (non-numeric filenames).")

        intrinsics_list = []
        for img_path in images:
            img_name = os.path.basename(img_path)
            if img_name not in gt:
                raise RuntimeError(
                    f"Missing GT intrinsics for {img_name} in {dataset_root}/cameras"
                )
            intrinsics_list.append(gt[img_name]["K"].astype(np.float32))

        intrinsics_array = np.stack(intrinsics_list, axis=0)
    else:
        intrinsics_array = None

    # ------------------------------------------------------------------
    # Load DepthAnything3 model
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    model = model.to(device=device)

    # ------------------------------------------------------------------
    # Run inference on all frames (chunked to avoid OOM)
    # ------------------------------------------------------------------
    CHUNK_SIZE = 120
    CHUNK_OVERLAP = 2

    def _run_chunk(img_chunk, intr_chunk):
        # Cap the input aspect at 3:2 before inference (see ASPECT_CAP note above).
        imgs = []
        ks = [] if intr_chunk is not None else None
        for idx, p in enumerate(img_chunk):
            k_in = intr_chunk[idx] if intr_chunk is not None else None
            img_r, k_r = _resize_to_aspect_cap(p, ASPECT_CAP, k_in)
            imgs.append(img_r)
            if ks is not None:
                ks.append(k_r)
        if intr_chunk is not None:
            return model.inference(
                imgs,
                use_ray_pose=True,
                ref_view_strategy="middle",
                process_res=PROCESS_RES, # if GPU memory allows
                process_res_method=PROCESS_RES_METHOD,
                intrinsics=np.stack(ks, axis=0),
            )
        return model.inference(imgs, use_ray_pose=True, process_res=PROCESS_RES, process_res_method=PROCESS_RES_METHOD,ref_view_strategy="middle")

    stitched_depth = []
    stitched_conf = []
    stitched_extrinsics = []
    stitched_intrinsics = []
    stitched_processed_images = []

    start = 0
    chunk_id = 0
    while start < len(images):
        chunk_start = max(0, start - CHUNK_OVERLAP) if start > 0 else start
        end = min(start + CHUNK_SIZE, len(images))
        img_chunk = images[chunk_start:end]
        intr_chunk = intrinsics_array[chunk_start:end] if intrinsics_array is not None else None
        print(f"Running DepthAnything3 on chunk {chunk_id}: frames {chunk_start}:{end}...")
        pred_chunk = _run_chunk(img_chunk, intr_chunk)

        overlap = int(start - chunk_start)
        if chunk_id > 0 and overlap > 0 and stitched_depth:
            prev_overlap_depths = []
            curr_overlap_depths = []
            prev_overlap_confs = []
            curr_overlap_confs = []

            for j in range(overlap):
                prev_overlap_depth = stitched_depth[-1][-overlap + j]
                curr_overlap_depth = np.asarray(pred_chunk.depth[j], dtype=np.float32)

                prev_overlap_conf = stitched_conf[-1][-overlap + j] if stitched_conf else None
                curr_overlap_conf = (
                    np.asarray(pred_chunk.conf[j], dtype=np.float32)
                    if hasattr(pred_chunk, "conf") and pred_chunk.conf is not None
                    else None
                )

                prev_overlap_depths.append(prev_overlap_depth)
                curr_overlap_depths.append(curr_overlap_depth)
                prev_overlap_confs.append(prev_overlap_conf)
                curr_overlap_confs.append(curr_overlap_conf)

            matcher = _build_overlap_depth_histogram_matcher(
                prev_overlap_depths,
                curr_overlap_depths,
                prev_conf_list=prev_overlap_confs,
                curr_conf_list=curr_overlap_confs,
                min_valid=1024,
                num_quantiles=256,
                trim_percentile=2.0,
            )

            print(
                f"Aligning chunk {chunk_id} depth histogram from overlap frames "
                f"{start - overlap}:{start - 1}"
            )

            pred_chunk = _apply_depth_histogram_match_to_prediction(
                pred_chunk,
                matcher,
                scale_extrinsics_with_median=False,
            )

        keep_from = overlap
        stitched_depth.append(np.asarray(pred_chunk.depth[keep_from:], dtype=np.float32))
        stitched_conf.append(np.asarray(pred_chunk.conf[keep_from:], dtype=np.float32))
        stitched_extrinsics.append(np.asarray(pred_chunk.extrinsics[keep_from:], dtype=np.float32))
        stitched_intrinsics.append(np.asarray(pred_chunk.intrinsics[keep_from:], dtype=np.float32))
        stitched_processed_images.append(np.asarray(pred_chunk.processed_images[keep_from:]))
        print(pred_chunk.intrinsics[keep_from:])

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        start = end
        chunk_id += 1

    class _Pred:
        pass

    prediction = _Pred()
    prediction.depth = np.concatenate(stitched_depth, axis=0)
    prediction.conf = np.concatenate(stitched_conf, axis=0)
    prediction.extrinsics = np.concatenate(stitched_extrinsics, axis=0)
    prediction.intrinsics = np.concatenate(stitched_intrinsics, axis=0)
    prediction.processed_images = np.concatenate(stitched_processed_images, axis=0)

    print("processed_images:", prediction.processed_images.shape)
    print("depth:", prediction.depth.shape)
    print("conf:", prediction.conf.shape)
    print("extrinsics:", prediction.extrinsics.shape)
    print("intrinsics:", prediction.intrinsics)


    # ------------------------------------------------------------------
    # Resize to original image size and scale intrinsics accordingly
    # ------------------------------------------------------------------
    orig_hw = []
    for p in images:
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:
            raise RuntimeError(f"Failed to read image: {p}")
        orig_hw.append((int(im.shape[0]), int(im.shape[1])))

    if len(set(orig_hw)) != 1:
        raise RuntimeError(
            "Input images do not share a single resolution. "
            "Current export expects a fixed (H, W) for all frames."
        )

    H_orig, W_orig = orig_hw[0]
    T_pred, H_pred, W_pred = prediction.depth.shape

    depth_maps = prediction.depth.astype(np.float32)
    conf_maps = prediction.conf.astype(np.float32)
    intrinsics = prediction.intrinsics.astype(np.float32).copy()

    sx = float(W_orig) / float(W_pred)
    sy = float(H_orig) / float(H_pred)

    for t in range(min(T_pred, intrinsics.shape[0])):
        intrinsics[t, 0, 0] *= sx
        intrinsics[t, 1, 1] *= sy
        intrinsics[t, 0, 2] *= sx
        intrinsics[t, 1, 2] *= sy

    # Save the raw per-frame DA3 intrinsics (rescaled to original size, before any
    # single-K collapse) so focal-rule experiments can rewrite intrinsics.npy without
    # regenerating depth.
    np.save(output_dir / "intrinsics_raw.npy", intrinsics.astype(np.float32).copy())

    # GT mode: save the GT intrinsics (already at original resolution) instead of
    # DA3's predicted intrinsics, so the whole pipeline stays on one intrinsic.
    if USE_INTRINSICS_GT and intrinsics_array is not None:
        intrinsics = intrinsics_array[: intrinsics.shape[0]].astype(np.float32).copy()
        print(f"[intrinsics] GT mode: saving GT intrinsics (fx0={intrinsics[0, 0, 0]:.3f}), "
              f"not DA3 predicted.")
    elif SINGLE_INTRINSIC:
        # DA3 mode: collapse the per-frame predictions into a single shared pinhole.
        # Principal point fixed at the image center. For the focal, pick the axis
        # aligned with the LONGER image dimension: the anisotropic 3:2 resize distorts
        # the shorter-dimension focal more, so use fx when cx>cy (landscape) and fy
        # otherwise (portrait), taking the median of that axis across frames.
        # NOTE: `intrinsics` here is ALREADY rescaled to the original image size by the
        # sx/sy loop above, so the medians below are taken at original scale.
        fx_all = intrinsics[:, 0, 0].astype(np.float64)
        fy_all = intrinsics[:, 1, 1].astype(np.float64)
        print("[intrinsics] DA3 mode: rescaled per-frame K (original scale), pre-median:")
        for t in range(intrinsics.shape[0]):
            print(f"  frame {t:3d}: fx={intrinsics[t, 0, 0]:8.2f} fy={intrinsics[t, 1, 1]:8.2f} "
                  f"cx={intrinsics[t, 0, 2]:7.2f} cy={intrinsics[t, 1, 2]:7.2f}")
        cx_final = W_orig / 2.0
        cy_final = H_orig / 2.0
        # Use the focal of the axis NOT squashed by the 3:2 resize = the SHORTER image
        # dimension: for landscape (cx>cy) the width was shrunk, so fy (height) is the
        # reliable focal; for portrait, fx. Empirically beats the squashed-axis focal
        # ~3x on average (subset ATE-T 0.072 vs 0.21).
        if cx_final > cy_final:
            f_single = float(np.median(fy_all)); axis = "fy"
        else:
            f_single = float(np.median(fx_all)); axis = "fx"
        print(f"[intrinsics] rescaled fx: median={np.median(fx_all):.2f} "
              f"[min {fx_all.min():.2f}, max {fx_all.max():.2f}]  |  "
              f"fy: median={np.median(fy_all):.2f} [min {fy_all.min():.2f}, max {fy_all.max():.2f}]  |  "
              f"cx={cx_final:.1f} {'>' if cx_final > cy_final else '<='} cy={cy_final:.1f} "
              f"-> chosen f = median({axis}) = {f_single:.2f}")
        K_single = np.array(
            [[f_single, 0.0, W_orig / 2.0],
             [0.0, f_single, H_orig / 2.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        intrinsics = np.repeat(K_single[None], intrinsics.shape[0], axis=0)
        print(f"[intrinsics] DA3 mode: single shared K for all frames -> "
              f"f={f_single:.3f} (median {axis}), cx={W_orig / 2.0:.1f}, cy={H_orig / 2.0:.1f}")

    if (H_pred, W_pred) != (H_orig, W_orig):
        depth_maps_resized = np.empty((T_pred, H_orig, W_orig), dtype=np.float32)
        conf_maps_resized = np.empty((T_pred, H_orig, W_orig), dtype=np.float32)
        for t in range(T_pred):
            depth_maps_resized[t] = cv2.resize(
                depth_maps[t], (W_orig, H_orig), interpolation=cv2.INTER_LINEAR
            )
            conf_maps_resized[t] = cv2.resize(
                conf_maps[t], (W_orig, H_orig), interpolation=cv2.INTER_LINEAR
            )
        depth_maps = depth_maps_resized
        conf_maps = conf_maps_resized

    print(f"Original image size: {(H_orig, W_orig)} | Pred size: {(H_pred, W_pred)}")

    # ------------------------------------------------------------------
    # Save intrinsics, depth, and confidence for all frames
    # ------------------------------------------------------------------
    intrinsics_path = output_dir / "intrinsics.npy"
    np.save(intrinsics_path, intrinsics)
    print("Saved intrinsics to", intrinsics_path)

    depth_maps_path = output_dir / "depth_maps.npy"
    np.save(depth_maps_path, depth_maps)
    print("Saved depth maps to", depth_maps_path)

    conf_maps_path = output_dir / "conf_maps.npy"
    np.save(conf_maps_path, conf_maps)
    print("Saved confidence maps to", conf_maps_path)

    extrinsics_path = output_dir / "extrinsics_c2w.npy"
    np.save(extrinsics_path, prediction.extrinsics.astype(np.float32))
    print("Saved camera-to-world extrinsics to", extrinsics_path)

    sparse_dir = output_dir / "sparse"
    _write_depthanything_colmap_sparse(
        sparse_dir=sparse_dir,
        image_paths=images,
        intrinsics=intrinsics,
        extrinsics=prediction.extrinsics,
        width=W_orig,
        height=H_orig,
    )
    print("Saved COLMAP-format predicted poses to", sparse_dir)


if __name__ == "__main__":
    main()
