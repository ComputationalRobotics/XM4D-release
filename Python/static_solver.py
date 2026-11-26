import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import cv2
import numpy as np


sys.path.append(os.path.join(os.path.dirname(__file__), '../XM/build/'))

sphere_ratio = 0.9

import XM

from utils.creatematrix import create_matrix
from utils.io import load_matrix_from_bin
from utils.recoversolution import recover_XM
from utils.checkconnection import checklandmarks
from utils.readgt_custom import load_custom_gt
from utils.depth_sources import load_depth_source
from utils.script_logger import finish_script_logger, install_script_logger
import utils.definitions
import utils.dataset_config as dataset_config

total_start = time.perf_counter()


# Optionally let Ceres refine a single shared camera focal during bundle adjustment
# (self-calibration). The 2D landmarks are normalized by the initial intrinsic, so
# this refines a global focal-correction factor around 1.0, bounded to avoid the
# focal/depth gauge from running away. Off by default; helps scenes where the
# predicted (DA3) focal is poor.
CERES_OPTIMIZE_INTRINSICS = dataset_config.get_bool(
    "CERES_OPTIMIZE_INTRINSICS", False
)
CERES_INTRINSIC_SCALE_MIN = dataset_config.get_float(
    "CERES_INTRINSIC_SCALE_MIN", 0.25
)
CERES_INTRINSIC_SCALE_MAX = dataset_config.get_float(
    "CERES_INTRINSIC_SCALE_MAX", 4.0
)
# Robust loss on the Ceres reprojection residuals: "trivial" (plain least squares), "huber", "cauchy" or "softl1",
# with its scale in pixels (converted to the normalized landmark units with the median focal of intrinsics.npy).
CERES_LOSS = dataset_config.get_str("CERES_LOSS", "cauchy").lower()
CERES_LOSS_PX = dataset_config.get_float("CERES_LOSS_PX", 2.0)
DEBUG = dataset_config.get_bool("DEBUG", False)  # also write tracks_debug_3d_solver.npz, Abar.bin and Q.mat

STATIC_TRACK_DEPTH_MAX = dataset_config.get_float(
    "STATIC_TRACK_DEPTH_MAX", 50.0
)
STATIC_TRACK_DEPTH_GRAD_MAX = dataset_config.get_float(
    "STATIC_TRACK_DEPTH_GRAD_MAX", 2.0
)

# Intrinsics come from a single source, step 1: <depth_dir>/intrinsics.npy for a video, each photo's
# <depth_dir>/photos/<name>.npz for photos (the model's K, or config.json INTRINSICS), so the solver
# never re-decides them here -- it just loads and uses that K everywhere.

DA3_SCALE_MIN = 0.25
DA3_SCALE_MAX = 4.0
DA3_SCALE_LEVELS = 5
DA3_SCALE_CANDIDATES = 15
DA3_SCALE_STRIDE = 4
DA3_SCALE_DEPTH_MIN = 0.0
DA3_SCALE_DEPTH_MAX = 0.0
DA3_SCALE_FIT_POINTS = 1000
DA3_SCALE_REFERENCE_POINTS = 3000
DA3_SCALE_FORWARD_KEEP_PERCENTILE = 85.0
DA3_SCALE_REVERSE_KEEP_PERCENTILE = 20.0
DA3_SCALE_REVERSE_WEIGHT = 0.25
DA3_SCALE_REFERENCE_KEEP_PERCENTILE = 98.0
# Frames after the first use a narrow local search around the running-median log-scale
# (all frames share a similar scale).
# One level of 9 over +-0.1 keeps the old final grid step (~0.025); the ICP refinement below is not bounded by the window.
DA3_SCALE_LOCAL_RADIUS = 0.1  # log-scale half-width
DA3_SCALE_LOCAL_LEVELS = 1
DA3_SCALE_LOCAL_CANDIDATES = 9
# The grid only locates the basin (a flat Chamfer loss quantizes to grid nodes);
# a closed-form ICP-style refinement then gives a continuous, sub-grid per-frame
# scale: s = sum(dir . (nn - center)) / sum(dir . dir) over trimmed correspondences.
DA3_SCALE_REFINE_ITERS = 4
# KD-tree query threads (the result does not depend on them). workers=-1 spawns a thread per core (huge overhead
# for the many tiny per-frame queries on a big machine); 4 is fastest for these sizes.
DA3_SCALE_QUERY_WORKERS = 4


def run_ceres_refine(
    edges,
    landmarks_2d,
    R_real,
    t_est,
    p_est,
):
    try:
        from utils.ceresforXM import XM_Ceres_interface
    except ModuleNotFoundError as exc:
        if exc.name in {"pyceres", "pycolmap"}:
            print(f"[Ceres] skipped: optional dependency '{exc.name}' is not installed.")
            return R_real, t_est, p_est, 1.0
        raise

    return XM_Ceres_interface(
        edges,
        landmarks_2d,
        R_real,
        t_est,
        p_est,
        optimize_intrinsics=CERES_OPTIMIZE_INTRINSICS,
        intrinsic_scale_min=CERES_INTRINSIC_SCALE_MIN,
        intrinsic_scale_max=CERES_INTRINSIC_SCALE_MAX,
        loss=CERES_LOSS,
        loss_scale=_ceres_loss_scale(),
    )


def _ceres_loss_scale():
    """CERES_LOSS_PX in normalized landmark units (px / median focal), or None without intrinsics.npy."""
    if utils.definitions.photos:
        if CERES_LOSS == "trivial":
            return None
        return CERES_LOSS_PX / float(np.median([K[0, 0] for K in _photo_intrinsics()[0].values()]))
    K_all = _load_intrinsics_npy(1)
    if K_all is None or CERES_LOSS == "trivial":
        return None
    return CERES_LOSS_PX / float(np.median(K_all[:, 0, 0]))


_PHOTO_K = None


def _photo_intrinsics():
    """input "photos": step 1's K and size of every photo that has its depth file: {photo index: K}, {index: (H, W)}."""
    global _PHOTO_K
    if _PHOTO_K is None:
        _PHOTO_K = ({}, {})
        for i, path in enumerate(utils.definitions.photo_paths()):
            f = utils.definitions.photo_depth_dir / f"{path.name}.npz"
            if f.exists():
                with np.load(f) as z:
                    _PHOTO_K[0][i] = np.asarray(z["K"], dtype=np.float64)
                    _PHOTO_K[1][i] = z["depth"].shape
    return _PHOTO_K


def build_sR_real(R_real, s_real, N):
    sR_real = np.zeros((3, 3 * N), dtype=np.float64)
    for i in range(N):
        sR_real[:, 3 * i:3 * i + 3] = s_real[i] * R_real[:, 3 * i:3 * i + 3]
    return sR_real


def _load_dynamic_masks(mask_dir, pattern, t_expected, target_hw):
    mask_dir = Path(mask_dir)
    paths = sorted(mask_dir.glob(pattern))
    if not paths:
        print(f"[track-lift] no masks found at {mask_dir / pattern}; treating all pixels as static.")
        return np.zeros((t_expected, target_hw[0], target_hw[1]), dtype=bool)
    def read(path):
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {path}")
        if mask.shape[:2] != target_hw:
            mask = cv2.resize(mask, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
        return mask > 0

    with ThreadPoolExecutor(8) as executor:     # cv2 releases the GIL
        masks = list(executor.map(read, paths[:t_expected]))
    if len(masks) < t_expected:
        pad = [np.zeros(target_hw, dtype=bool) for _ in range(t_expected - len(masks))]
        masks.extend(pad)
    masks_pos = np.stack(masks, axis=0)
    # Same convention as the rest of the repo: mostly-white masks usually mean valid/static.
    return (~masks_pos if float(np.mean(masks_pos)) > 0.5 else masks_pos).astype(bool)


def _gt_intrinsics_for_depth_maps(depth_maps):
    image_paths = sorted(Path(utils.definitions.video_dir).glob(utils.definitions.image_pattern))
    k_list = []
    H_depth, W_depth = depth_maps.shape[1:]
    for path in image_paths[: depth_maps.shape[0]]:
        if path.name not in gt:
            continue
        K = np.asarray(gt[path.name]["K"], dtype=np.float64).copy()
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is not None:
            H_img, W_img = image.shape[:2]
            K[0, :] *= W_depth / float(W_img)
            K[1, :] *= H_depth / float(H_img)
        k_list.append(K)
    if not k_list:
        print("[track-lift] no GT intrinsics available; skipped.")
        return None
    if len(k_list) < depth_maps.shape[0]:
        k_list.extend([k_list[-1].copy() for _ in range(depth_maps.shape[0] - len(k_list))])
    return np.stack(k_list, axis=0)


def _unproject(x, y, z, K):
    """Pixel (x, y) + depth z -> camera-frame point (X, Y, Z).

    K must be at the same resolution as the depth map that z was sampled from.
    Works for scalar or array inputs; the point axis is the last one.
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    return np.stack([(x - cx) / fx * z, (y - cy) / fy * z, z], axis=-1)


def _load_intrinsics_npy(t_expected):
    """The single solver intrinsic source: outputs/depth_anything_3/intrinsics.npy.

    get_depths_clean.py writes this file (Depth Anything 3's intrinsics, or config.json
    INTRINSICS), so the solver just reads it. Returns K_all (T, 3, 3), or None if absent.
    """
    intrinsics_path = Path(utils.definitions.depth_dir) / "intrinsics.npy"
    if not intrinsics_path.exists():
        return None
    K_all = np.load(str(intrinsics_path)).astype(np.float64)
    if K_all.ndim == 2 and K_all.shape == (3, 3):
        K_all = np.repeat(K_all[None], int(t_expected), axis=0)
    if K_all.ndim != 3 or K_all.shape[1:] != (3, 3):
        raise ValueError(f"Expected intrinsics shape (T,3,3), got {K_all.shape}")
    return K_all


def _load_selected_depth_maps_and_intrinsics(t_expected):
    try:
        depth_maps, _, depth_path = load_depth_source("da3", utils.definitions)
    except FileNotFoundError as exc:
        print(f"[track-lift] {exc}; skipped.")
        return None, None, None
    depth_maps = depth_maps.astype(np.float64)
    if depth_maps.ndim != 3:
        raise ValueError(f"Expected depth maps shape (T,H,W), got {depth_maps.shape}")

    K_all = _load_intrinsics_npy(depth_maps.shape[0])
    if K_all is None:
        K_all = _gt_intrinsics_for_depth_maps(depth_maps)
        if K_all is None:
            return None, None, None
    t_use = min(int(t_expected), depth_maps.shape[0], K_all.shape[0])
    return depth_maps[:t_use], K_all[:t_use], depth_path


def _filter_point_outliers(points, keep_percentile):
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 2 and pts.shape[0] == 3:
        pts = pts.T
    pts = pts.reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] == 0:
        return pts
    center = np.median(pts, axis=0)
    radius = np.linalg.norm(pts - center.reshape(1, 3), axis=1)
    threshold = np.percentile(radius, np.clip(float(keep_percentile), 1.0, 100.0))
    return pts[radius <= threshold]


def _downsample_points(points, max_points, seed):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] <= int(max_points):
        return pts
    rng = np.random.default_rng(int(seed))
    return pts[rng.choice(pts.shape[0], int(max_points), replace=False)]


def _trimmed_mean_square(distances, keep_percentile):
    d = np.asarray(distances, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("inf")
    keep_percentile = float(np.clip(keep_percentile, 1.0, 100.0))
    if d.size > 4:
        threshold = np.percentile(d, keep_percentile)
        d = d[d <= threshold]
    if d.size == 0:
        return float("inf")
    return float(np.mean(d * d))


def _trimmed_ms_rows(distances, keep_percentile):
    """Vectorized per-row trimmed mean-of-squares. distances: (K, N) -> (K,)."""
    d = np.asarray(distances, dtype=np.float64)
    keep = float(np.clip(keep_percentile, 1.0, 100.0))
    if d.shape[1] > 4:
        thr = np.percentile(d, keep, axis=1, keepdims=True)
        d = np.where(d <= thr, d, np.nan)
    with np.errstate(invalid="ignore"):
        out = np.nanmean(d * d, axis=1)
    return np.where(np.isfinite(out), out, np.inf)


def _camera_depth_samples_for_da3_scale(depth, K, dynamic_mask, frame_idx, max_points, min_depth):
    depth = np.asarray(depth, dtype=np.float64)
    H, W = depth.shape
    stride = max(1, int(DA3_SCALE_STRIDE))
    yy, xx = np.meshgrid(
        np.arange(0, H, stride, dtype=np.int64),
        np.arange(0, W, stride, dtype=np.int64),
        indexing="ij",
    )
    yy = yy.reshape(-1)
    xx = xx.reshape(-1)
    z = depth[yy, xx]
    valid = np.isfinite(z) & (z > float(min_depth))
    valid &= ~np.asarray(dynamic_mask, dtype=bool)[yy, xx]
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float64)

    yy = yy[valid].astype(np.float64)
    xx = xx[valid].astype(np.float64)
    z = z[valid]
    pts_cam = _unproject(xx, yy, z, K)
    if pts_cam.shape[0] > int(max_points):
        rng = np.random.default_rng(1000 + int(frame_idx))
        pts_cam = pts_cam[rng.choice(pts_cam.shape[0], int(max_points), replace=False)]
    return pts_cam


def _da3_scale_search(center, base_dirs, target_tree, static_ref_f, log_lo, log_hi, levels, candidates):
    """Grid then closed-form refine over log-scale.

    The grid does a batched forward-Chamfer NN query per level (one call for all
    candidate scales) purely to locate the basin; the reverse term is
    NOT used for selection — it is a flat regularizer that would only add a KD-tree
    rebuild per level. A closed-form ICP refinement then pins the sub-grid scale,
    and the reverse term is evaluated once at the end for the reported loss.
    `center` must be shaped (1, 1, 3).
    """
    from scipy.spatial import cKDTree

    fwd_keep = float(DA3_SCALE_FORWARD_KEEP_PERCENTILE)
    rev_w = float(DA3_SCALE_REVERSE_WEIGHT)
    rev_keep = float(DA3_SCALE_REVERSE_KEEP_PERCENTILE)
    workers = int(DA3_SCALE_QUERY_WORKERS)

    lo, hi = float(log_lo), float(log_hi)
    best_log, best_loss = 0.0, float("inf")
    for _level in range(max(1, int(levels))):
        grid = np.linspace(lo, hi, max(3, int(candidates)))
        scales = np.exp(grid)
        # batched forward NN: src (K, N, 3) queried in one call
        src = center + scales[:, None, None] * base_dirs[None, :, :]
        Kk, Nn = src.shape[0], src.shape[1]
        dist, _ = target_tree.query(src.reshape(-1, 3).astype(np.float32), workers=workers)
        totals = _trimmed_ms_rows(dist.reshape(Kk, Nn), fwd_keep)
        j = int(np.argmin(totals))
        if float(totals[j]) < best_loss:
            best_loss, best_log = float(totals[j]), float(grid[j])
        step = (hi - lo) / max(len(grid) - 1, 1)
        lo = max(float(log_lo), best_log - 2.0 * step)
        hi = min(float(log_hi), best_log + 2.0 * step)
        if hi <= lo:
            break

    if not np.isfinite(best_loss):
        return float(np.exp(best_log)), best_loss

    # --- closed-form ICP-style sub-grid refinement on the trimmed forward term ---
    c0 = center.reshape(3).astype(np.float64)
    dirs64 = base_dirs.astype(np.float64)
    s = float(np.exp(best_log))
    for _ in range(int(DA3_SCALE_REFINE_ITERS)):
        src = (c0 + s * dirs64).astype(np.float32)
        d, idx = target_tree.query(src, workers=workers)
        keep = np.ones(d.shape[0], dtype=bool)
        if d.size > 4:
            keep = d <= np.percentile(d, fwd_keep)
        if int(np.count_nonzero(keep)) < 3:
            break
        nn = static_ref_f[idx[keep]].astype(np.float64)
        dk = dirs64[keep]
        den = float(np.sum(dk * dk))
        if den <= 1e-12:
            break
        s_new = float(np.sum(dk * (nn - c0)) / den)
        s_new = float(np.clip(s_new, float(DA3_SCALE_MIN), float(DA3_SCALE_MAX)))
        if not np.isfinite(s_new) or s_new <= 0.0:
            break
        converged = abs(s_new - s) <= 1e-4 * s
        s = s_new
        if converged:
            break

    # report the objective at the refined scale (reverse term computed once here)
    src = (c0 + s * dirs64).astype(np.float32)
    d, _ = target_tree.query(src, workers=workers)
    final_loss = _trimmed_mean_square(d, fwd_keep)
    rev, _ = cKDTree(src).query(static_ref_f, workers=workers)
    final_loss += rev_w * _trimmed_mean_square(rev, rev_keep)
    return float(s), float(final_loss)


def estimate_da3_frame_scales_static_chamfer(R_real, t_est, static_points, frame_index):
    """frame_index[i] = original frame of solver frame i (the solver may drop frames)."""
    T_expected = t_est.shape[1]
    frame_index = np.asarray(frame_index, dtype=np.int64)
    scales = np.ones((T_expected,), dtype=np.float64)
    losses = np.full((T_expected,), np.nan, dtype=np.float64)
    stats = {
        "enabled": False,
        "source": "",
        "reference_points": 0,
        "dynamic_mask_excluded": True,
        "scale_min": float(DA3_SCALE_MIN),
        "scale_max": float(DA3_SCALE_MAX),
        "stride": int(DA3_SCALE_STRIDE),
        "depth_min": float(DA3_SCALE_DEPTH_MIN),
        "depth_max": float(DA3_SCALE_DEPTH_MAX),
        "fit_points": int(DA3_SCALE_FIT_POINTS),
        "reference_points_max": int(DA3_SCALE_REFERENCE_POINTS),
    }

    depth_maps, K_all, depth_path = _load_selected_depth_maps_and_intrinsics(int(frame_index.max()) + 1)
    if depth_maps is None or K_all is None:
        stats["reason"] = "depth maps or intrinsics unavailable"
        return scales, losses, stats

    static_ref = _filter_point_outliers(static_points, DA3_SCALE_REFERENCE_KEEP_PERCENTILE)
    static_ref = _downsample_points(static_ref, DA3_SCALE_REFERENCE_POINTS, 17)
    stats.update(
        {
            "enabled": True,
            "source": str(depth_path),
            "reference_points": int(static_ref.shape[0]),
            "frames_used": int(depth_maps.shape[0]),
        }
    )
    if static_ref.shape[0] < 10:
        stats["reason"] = "not enough static solver points"
        print("[da3-scale] skipped: not enough static solver points for Chamfer target.")
        return scales, losses, stats

    from scipy.spatial import cKDTree

    static_ref_f = static_ref.astype(np.float32)
    target_tree = cKDTree(static_ref_f)
    T, H, W = depth_maps.shape
    dynamic_masks = _load_dynamic_masks(
        utils.definitions.mask_dir,
        utils.definitions.mask_pattern,
        T,
        (H, W),
    )
    log_min = float(np.log(max(float(DA3_SCALE_MIN), 1e-8)))
    log_max = float(np.log(max(float(DA3_SCALE_MAX), np.exp(log_min) * (1.0 + 1e-8))))
    local_radius = float(DA3_SCALE_LOCAL_RADIUS)
    print(
        "[da3-scale] static-only Chamfer scale search: "
        f"frames={T}, target_static_points={static_ref.shape[0]}, "
        f"range=[{DA3_SCALE_MIN}, {DA3_SCALE_MAX}], dynamic_mask_excluded=True, "
        f"anchor+local(radius={local_radius})"
    )
    found_logs = []  # log-scales of frames solved so far (for the running-median prior)
    fitted = np.zeros((T_expected,), dtype=bool)
    for i in range(T_expected):
        f = int(frame_index[i])        # the depth map, intrinsics and mask of this camera's original frame
        if f < 0 or f >= T:
            continue
        pts_cam = _camera_depth_samples_for_da3_scale(
            depth_maps[f],
            K_all[f],
            dynamic_masks[f],
            f,
            DA3_SCALE_FIT_POINTS,
            DA3_SCALE_DEPTH_MIN,
        )
        if pts_cam.shape[0] < 10:
            print(f"[da3-scale] frame={i}: skipped, static depth samples={pts_cam.shape[0]}")
            continue
        R_i = R_real[:, 3 * i:3 * i + 3]
        center = t_est[:, i].reshape(1, 1, 3)
        base_dirs = (R_i @ pts_cam.T).T

        if not found_logs:
            # first valid frame: full-range search establishes the scale prior
            lo, hi, lv, cand = log_min, log_max, DA3_SCALE_LEVELS, DA3_SCALE_CANDIDATES
        else:
            # subsequent frames: narrow search around the running-median log-scale
            c = float(np.median(found_logs))
            lo = max(log_min, c - local_radius)
            hi = min(log_max, c + local_radius)
            lv, cand = DA3_SCALE_LOCAL_LEVELS, DA3_SCALE_LOCAL_CANDIDATES

        scales[i], losses[i] = _da3_scale_search(
            center, base_dirs, target_tree, static_ref_f, lo, hi, lv, cand
        )
        fitted[i] = True
        found_logs.append(float(np.log(max(scales[i], 1e-8))))
        print(
            f"[da3-scale] frame={i}: static_samples={pts_cam.shape[0]} "
            f"scale={scales[i]:.6g} loss={losses[i]:.6g}",
            flush=True,
        )

    if found_logs and not fitted.all():
        # frames without enough static depth take the median fitted scale, not 1.0 (the solver's global scale is arbitrary)
        scales[~fitted] = float(np.exp(np.median(found_logs)))
        print(f"[da3-scale] {int((~fitted).sum())} frame(s) without a fit take the median scale {scales[~fitted][0]:.6g}")

    return scales, losses, stats


def lift_2d_tracks_to_solver_input(
    data,
    *,
    gt,
    image_dir,
    image_pattern,
    output_path,
    depth_max,
    depth_grad_max,
):
    tracks_xy = np.asarray(data["tracks_xy"], dtype=np.float64)
    occluded = np.asarray(data["occluded"], dtype=bool)
    if tracks_xy.ndim != 4 or tracks_xy.shape[0] != 1 or tracks_xy.shape[-1] != 2:
        raise ValueError(f"Expected tracks_xy shape (1,Q,T,2), got {tracks_xy.shape}")
    if occluded.shape[:3] != tracks_xy.shape[:3]:
        raise ValueError(f"occluded shape {occluded.shape} does not match tracks_xy {tracks_xy.shape}")

    depth_maps, conf_maps, depth_path = load_depth_source("da3", utils.definitions)
    depth_maps = np.asarray(depth_maps, dtype=np.float64)
    if depth_maps.ndim != 3:
        raise ValueError(f"Expected depth maps shape (T,H,W), got {depth_maps.shape}")
    T_depth = depth_maps.shape[0]

    # Single intrinsic source: lift with intrinsics.npy
    # (get_depths_clean.py already baked the GT-vs-DA3 choice into it).
    da3_K_all = _load_intrinsics_npy(T_depth)
    if da3_K_all is not None:
        print(f"[track-lift] using intrinsics.npy, shape {da3_K_all.shape}")
    else:
        print("[track-lift] intrinsics.npy missing; falling back to GT intrinsics.")

    image_paths = sorted(Path(image_dir).glob(image_pattern))
    T = min(tracks_xy.shape[2], len(image_paths), T_depth)
    Q = tracks_xy.shape[1]
    if T <= 0:
        raise RuntimeError("No frames available for 2D-to-3D lifting.")

    tracks_xyz = np.full((1, Q, T, 3), np.nan, dtype=np.float64)
    conf_xyz = np.full((Q, T), np.nan, dtype=np.float64)
    depth_grad_deleted_obs_mask = np.zeros((1, Q, T), dtype=bool)
    depth_max_deleted_obs_mask = np.zeros((1, Q, T), dtype=bool)

    def image_hw(path):         # only the frame size is needed; read in threads (cv2 releases the GIL)
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return None if img is None else img.shape[:2]

    with ThreadPoolExecutor(8) as executor:
        image_hws = list(executor.map(image_hw, image_paths[:T]))

    edges_list = []
    weights_list = []
    landmarks_list = []
    for t in range(T):
        image_path = image_paths[t]
        image_name = image_path.name
        if image_name not in gt:
            print(f"[track-lift] frame {t}: missing GT intrinsics for {image_name}; skipped.")
            continue

        if image_hws[t] is None:
            print(f"[track-lift] frame {t}: could not read {image_path}; skipped.")
            continue
        H_img, W_img = image_hws[t]

        depth = depth_maps[t]
        conf_vals_map = conf_maps[t] if conf_maps is not None and t < conf_maps.shape[0] else None
        depth = np.asarray(depth, dtype=np.float64)
        H_d, W_d = depth.shape
        x_scale = W_d / float(max(W_img, 1))
        y_scale = H_d / float(max(H_img, 1))

        if da3_K_all is not None and t < da3_K_all.shape[0]:
            # intrinsics.npy is already at the depth-map resolution.
            K_depth = np.asarray(da3_K_all[t], dtype=np.float64).copy()
        else:
            K = np.asarray(gt[image_name]["K"], dtype=np.float64).copy()
            K_depth = K.copy()
            K_depth[0, :] *= x_scale
            K_depth[1, :] *= y_scale

        depth_f = depth.astype(np.float64)
        grad_x = cv2.Sobel(depth_f, cv2.CV_64F, 2, 0, ksize=3)
        grad_y = cv2.Sobel(depth_f, cv2.CV_64F, 0, 2, ksize=3)
        grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)

        x = tracks_xy[0, :, t, 0]
        y = tracks_xy[0, :, t, 1]
        x_depth = x * x_scale
        y_depth = y * y_scale
        valid = (~occluded[0, :, t]) & np.isfinite(x_depth) & np.isfinite(y_depth)

        xi = np.zeros_like(x_depth, dtype=np.int64)
        yi = np.zeros_like(y_depth, dtype=np.int64)
        xi[valid] = np.rint(x_depth[valid]).astype(np.int64)
        yi[valid] = np.rint(y_depth[valid]).astype(np.int64)
        valid &= (xi >= 0) & (xi < W_d) & (yi >= 0) & (yi < H_d)
        if not np.any(valid):
            continue

        idx_valid = np.where(valid)[0]
        z = depth[yi[valid], xi[valid]]
        grad = grad_mag[yi[valid], xi[valid]]
        finite_positive = np.isfinite(z) & (z > 0)
        max_ok = z <= float(depth_max)
        basic_depth_ok = finite_positive & max_ok
        grad_ok = grad <= float(depth_grad_max)

        depth_max_deleted = finite_positive & (~max_ok)
        if np.any(depth_max_deleted):
            depth_max_deleted_obs_mask[0, idx_valid[depth_max_deleted], t] = True
        depth_grad_deleted = basic_depth_ok & (~grad_ok)
        if np.any(depth_grad_deleted):
            depth_grad_deleted_obs_mask[0, idx_valid[depth_grad_deleted], t] = True

        valid_depth = basic_depth_ok & grad_ok
        if not np.any(valid_depth):
            continue

        idx = idx_valid[valid_depth]
        zv = z[valid_depth].astype(np.float64)

        xv = x_depth[idx]
        yv = y_depth[idx]
        Pc = _unproject(xv, yv, zv, K_depth)
        tracks_xyz[0, idx, t, :] = Pc

        conf_vals = None
        if conf_vals_map is not None:
            conf_vals = np.asarray(conf_vals_map[yi[idx], xi[idx]], dtype=np.float64)
            conf_xyz[idx, t] = conf_vals

        # [landmark_index, frame_index] (converted below); weight = the finite confidences, else 1
        edges_list.append(np.stack([idx, np.full(idx.shape[0], t)], axis=1).astype(np.int64))
        landmarks_list.append(Pc)
        w = np.ones(idx.shape[0], dtype=np.float64)
        if conf_vals is not None:
            finite = np.isfinite(conf_vals)
            w[finite] = conf_vals[finite]
        weights_list.append(w[:, None])

    if not edges_list:
        raise RuntimeError("[track-lift] no valid 3D observations were created from 2D tracks.")
    edges = np.concatenate(edges_list, axis=0)
    weights = np.concatenate(weights_list, axis=0)
    landmarks = np.concatenate(landmarks_list, axis=0).astype(np.float64)

    if DEBUG:
        np.savez_compressed(
            Path(output_path) / "tracks_debug_3d_solver.npz",
            tracks_xyz=tracks_xyz,
            conf_xyz=conf_xyz,
            depth_grad_deleted_obs_mask=depth_grad_deleted_obs_mask,
            depth_max_deleted_obs_mask=depth_max_deleted_obs_mask,
            video_dir=str(image_dir),
            depth_dir=str(depth_path),
            depth_source=np.array("da3"),
        )
    print(
        "[track-lift] lifted 2D tracks in static_solver: "
        f"obs={edges.shape[0]}, depth_grad_deleted={int(np.sum(depth_grad_deleted_obs_mask))}, "
        f"depth_max_deleted={int(np.sum(depth_max_deleted_obs_mask))}, source={depth_path}"
    )
    return edges, weights, landmarks


def lift_photo_tracks():
    """input "photos": lift each photo's track observations (match_photos.py) to 3D with that photo's step-1 depth and
    K (finite positive depth only: the video lift's far-depth and depth-edge checks, STATIC_TRACK_DEPTH_MAX and
    _GRAD_MAX, are in metres and tuned for videos; the depth models put landmarks from tens to hundreds of metres).
    Returns edges [track, photo], weights (the depth confidence) and the camera-frame points, as
    lift_2d_tracks_to_solver_input."""
    z = np.load(utils.definitions.photo_tracks_path)
    names = [str(n) for n in z["names"]]
    if names != [p.name for p in utils.definitions.photo_paths()]:
        raise RuntimeError(f"{utils.definitions.photo_tracks_path} is for other photos: rerun match_photos.py")
    frame, track, xy = z["frame"].astype(np.int64), z["track"].astype(np.int64), z["xy"].astype(np.float64)
    edges_list, weights_list, landmarks_list = [], [], []
    for t, name in enumerate(names):
        sel = np.flatnonzero(frame == t)
        f = utils.definitions.photo_depth_dir / f"{name}.npz"
        if sel.size == 0:
            continue
        if not f.exists():
            print(f"[track-lift] {name}: no depth ({f}); skipped.")
            continue
        with np.load(f) as d:
            depth, conf, K = d["depth"].astype(np.float64), d["conf"].astype(np.float64), d["K"].astype(np.float64)
        H, W = depth.shape
        x, y = xy[sel, 0], xy[sel, 1]
        xi, yi = np.rint(x).astype(np.int64), np.rint(y).astype(np.int64)
        inside = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        sel, x, y, xi, yi = sel[inside], x[inside], y[inside], xi[inside], yi[inside]
        zv = depth[yi, xi]
        ok = np.isfinite(zv) & (zv > 0)
        if not ok.any():
            continue
        edges_list.append(np.stack([track[sel[ok]], np.full(int(ok.sum()), t)], axis=1))
        landmarks_list.append(_unproject(x[ok], y[ok], zv[ok], K))
        w = conf[yi[ok], xi[ok]]
        weights_list.append(np.where(np.isfinite(w), w, 1.0)[:, None])
    if not edges_list:
        raise RuntimeError("[track-lift] no valid 3D observations were created from the photo tracks.")
    edges = np.concatenate(edges_list, axis=0)
    print(f"[track-lift] lifted photo tracks: obs={edges.shape[0]} of {len(frame)}, "
          f"source={utils.definitions.photo_depth_dir}")
    return edges, np.concatenate(weights_list, axis=0), np.concatenate(landmarks_list, axis=0)


def export_colmap(R, t, p, indices_all, focal_scale):
    """The cameras and points as a COLMAP text model in outputs/colmap/ (cameras.txt, images.txt, points3D.txt):
    one PINHOLE camera per solved frame or photo, with step 1's K times the focal refined by Ceres. The points
    have no colour and no track."""
    from utils.convertCOLMAP import _rotation_matrix_to_qvec, write_cameras_text, write_images_text, write_points3D_text
    if utils.definitions.photos:
        paths = utils.definitions.photo_paths()
        K_by_frame, hw_by_frame = _photo_intrinsics()
    else:
        paths = sorted(Path(image_dir).glob(image_pattern))
        K_all = _load_intrinsics_npy(len(paths))
        H, W = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR).shape[:2]
        K_by_frame = {i: K_all[i] for i in range(min(len(paths), K_all.shape[0]))}
        hw_by_frame = {i: (H, W) for i in K_by_frame}
    cameras, images = {}, {}
    for orig, j in enumerate(np.asarray(indices_all, dtype=np.int64)):
        if j < 0 or orig not in K_by_frame:
            continue
        K = K_by_frame[orig]
        (h, w), cid = hw_by_frame[orig], len(cameras) + 1
        cameras[cid] = {"model": "PINHOLE", "width": int(w), "height": int(h),
                        "params": [float(K[0, 0] * focal_scale), float(K[1, 1] * focal_scale),
                                   float(K[0, 2]), float(K[1, 2])]}
        R_c2w, t_c2w = R[:, 3 * j:3 * j + 3], t[:, j]
        images[cid] = {"id": cid, "qvec": _rotation_matrix_to_qvec(R_c2w.T), "tvec": -R_c2w.T @ t_c2w,
                       "camera_id": cid, "name": paths[orig].name}
    points = {i + 1: {"id": i + 1, "xyz": p[:, i].tolist(), "rgb": [128, 128, 128], "error": 0.0}
              for i in range(p.shape[1])}
    out = Path(output_path) / "colmap"
    out.mkdir(parents=True, exist_ok=True)
    write_cameras_text(cameras, out / "cameras.txt")
    write_images_text(images, out / "images.txt")
    write_points3D_text(points, out / "points3D.txt")
    print(f"[colmap] {len(images)} cameras and {len(points)} points -> {out}")


### S0: CONFIG
dataset_path = str(utils.definitions.rootpath)
output_path = str(utils.definitions.output_dir)
Path(output_path).mkdir(parents=True, exist_ok=True)
install_script_logger("static_solver", output_path)

image_dir = utils.definitions.video_dir
image_pattern = utils.definitions.image_pattern

if utils.definitions.photos:
    edges, weights_full, landmarks = lift_photo_tracks()
else:
    gt = load_custom_gt(dataset_path)

    data = np.load(os.path.join(output_path, 'output_tapnet_da3.npz'))
    print("[track-lift] lifting 2D tracks to 3D observations for solver input.")
    edges, weights_full, landmarks = lift_2d_tracks_to_solver_input(
        data,
        gt=gt,
        image_dir=image_dir,
        image_pattern=image_pattern,
        output_path=output_path,
        depth_max=STATIC_TRACK_DEPTH_MAX,
        depth_grad_max=STATIC_TRACK_DEPTH_GRAD_MAX,
    )
weights = weights_full[:, 0].astype(np.float64, copy=False)

frame_ids = edges[:, 0].astype(np.int64)
weight_per_frame = np.bincount(
    frame_ids,
    weights=weights,
    minlength=frame_ids.max() + 1,
)

denom = np.maximum(weight_per_frame[frame_ids], 1.0)
weights = weights / denom

# dalete the row that contains the same edges in view-graph
_, unique_indices = np.unique(edges, axis=0, return_index=True)
edges = edges[unique_indices,:]
weights = weights[unique_indices]
landmarks = landmarks[unique_indices,:]

edges = edges[:, [1,0]] + 1

N = int(np.max(edges[:, 0]))
M = int(np.max(edges[:, 1]))

# Drop frames whose observation count is <= FRAME_MIN_POINTS in checklandmarks.
FRAME_MIN_POINTS = 10

"""
    input data:
        edges (np.ndarray): A 2D array of shape (num_ob, 2) where each row represents an edge in the view graph.
                            Each edge is defined as [frame_index, landmark_index].
                            For example:
                                edges = [[1, 1],
                                         [1, 2],
                                         [1, 3],
                                         ...]
        weights (np.ndarray): A 2D array of shape (num_ob, 1) where each row is the weight corresponding to the
                              respective edge in 'edges'.
        landmarks (np.ndarray): A 2D array of shape (num_ob, 3) where each row contains the 3D position of a landmark.
                                Note that the landmark coordinates have already been normalized using the camera intrinsics.

    Example:
        >>> edges = np.array([[1, 1],
                                [1, 2],
                                [1, 3]])
        >>> weights = np.array([[0.8],
                                [0.9],
                                [0.85]])
        >>> landmarks = np.array([[0.5, 0.1, 1.2],
                                  [0.6, 0.2, 1.1],
                                  [0.55, 0.15, 1.3]])
"""

edges, landmarks, weights, indices_all = checklandmarks(edges, landmarks, weights, N, M, frame_min_points=FRAME_MIN_POINTS)


N = int(np.max(edges[:, 0]))
M = int(np.max(edges[:, 1]))

# interface to create the matrix given the edges, weights and landmarks
Abar = create_matrix(weights, edges, landmarks, output_path, debug=DEBUG)
LAMBDA = dataset_config.get_float("LAMBDA", 10.0)
# RANK: max rank searched by the mixed solver.
RANK = 5
solver_time = 0.0
solver_start = time.perf_counter()
XM.solve_mixed(output_path,RANK,1e-6,LAMBDA,30,sphere_ratio)
solver_time += time.perf_counter() - solver_start

# visualize the camera poses
R,_ = load_matrix_from_bin(output_path + '/R.bin')
s,_ = load_matrix_from_bin(output_path + '/s.bin')
Q,_ = load_matrix_from_bin(output_path + '/Q.bin')    

print("S1 is", s[0])

# recover p and t
# details refer to our paper
R_real, s_real, p_est, t_est, optimal_rank = recover_XM(Q, R, s, Abar, LAMBDA)
R_real = np.asarray(R_real, dtype=np.float64)
s_real = np.asarray(s_real, dtype=np.float64)
p_est = np.asarray(p_est, dtype=np.float64)
t_est = np.asarray(t_est, dtype=np.float64)

sR_real = build_sR_real(R_real, s_real, N)

# XM^2
# this is basically a robust outlier removal, you delete landmarks that has the largest error

src_idx = edges[:, 0] - 1  # Convert from 1-based to 0-based indexing
dst_idx = edges[:, 1] - 1

R_real = R_real.reshape(3, N, 3).transpose(1, 0, 2)  # Shape: (N, 3, 3)

# Extract the correct 3x3 rotation matrices
R_matrices = R_real[src_idx]  # Shape: (N, 3, 3)

# Compute transformed landmarks
landmarks_transformed = (s_real[src_idx, None] * np.einsum('nij,nj->ni', R_matrices, landmarks)) + t_est[:, src_idx].T

diff = (p_est[:, dst_idx].T - landmarks_transformed)  # Ensure shapes match
squared_distances = np.sum(diff**2, axis=1)

error = weights * squared_distances

print("sum of error: ", np.sum(error))

# plug in sR^T Q sR TO SEE they are the same
X = sR_real.T @ sR_real
print("objective value: ", np.sum(Q * X))

# XM^2 multi-view depth-consistency filter: drop observations whose 3D residual
# ||s R x + t - p|| is above the XM2_KEEP_PERCENTILE-th percentile, then re-solve.
XM2_KEEP_PERCENTILE = 95.0
threshold = np.percentile(error, XM2_KEEP_PERCENTILE)

indices_to_remove = np.where(error > threshold)[0]

edges = np.delete(edges, indices_to_remove, axis=0)
weights = np.delete(weights, indices_to_remove)
landmarks = np.delete(landmarks, indices_to_remove, axis=0)

# second run
edges, landmarks, weights, indices = checklandmarks(edges, landmarks, weights, N, M, frame_min_points=FRAME_MIN_POINTS)
N_old = np.where(indices_all > -1)[0].shape[0]
indices_all_copy = indices_all.copy()
for i in range(N_old):
    indices_all[np.where(indices_all_copy == i)[0]] = indices[i]

N = int(np.max(edges[:, 0]))  
M = int(np.max(edges[:, 1]))  

Abar = create_matrix(weights, edges, landmarks, output_path, debug=DEBUG)

solver_start = time.perf_counter()
XM.solve_mixed(output_path, RANK,1e-4, 0.0, 30, sphere_ratio)
solver_time += time.perf_counter() - solver_start

R,_ = load_matrix_from_bin(output_path + '/R.bin')
s,_ = load_matrix_from_bin(output_path + '/s.bin')
Q,_ = load_matrix_from_bin(output_path + '/Q.bin')        

# recover p and t
R_real, s_real, p_est, t_est, optimal_rank = recover_XM(Q, R, s, Abar, LAMBDA)
R_real = np.asarray(R_real, dtype=np.float64)
s_real = np.asarray(s_real, dtype=np.float64)
p_est = np.asarray(p_est, dtype=np.float64)
t_est = np.asarray(t_est, dtype=np.float64)

# Ceres refinement with the 2D observations.

landmarks_2D = landmarks[:, :2] / landmarks[:, 2, None]
nan_mask = np.isnan(landmarks_2D).any(axis=1)

R_real, t_est, p_est, focal_scale = run_ceres_refine(
    edges[~nan_mask],
    landmarks_2D[~nan_mask],
    R_real,
    t_est,
    p_est,
)
R_real = np.asarray(R_real, dtype=np.float64)
t_est = np.asarray(t_est, dtype=np.float64)
p_est = np.asarray(p_est, dtype=np.float64)

t1 = t_est[:, 0:1].copy()
t_est = t_est - t1
p_est = p_est - t1
finite_p = np.isfinite(p_est).all(axis=0)
p_est = p_est[:, finite_p]

solver_frame_orig = np.full(N, -1, dtype=np.int64)          # original frame of each solver frame
for orig_idx, new_idx in enumerate(np.asarray(indices_all, dtype=np.int64)):
    if 0 <= new_idx < N:
        solver_frame_orig[new_idx] = orig_idx
if utils.definitions.photos:   # the per-frame depth scale is for videos (dense depth of one camera); photos keep 1
    da3_frame_scale, da3_frame_scale_loss = np.ones(N), np.full(N, np.nan)
    da3_frame_scale_stats = {"enabled": False, "reason": "photos"}
else:
    da3_frame_scale, da3_frame_scale_loss, da3_frame_scale_stats = estimate_da3_frame_scales_static_chamfer(
        R_real,
        t_est,
        p_est,
        frame_index=solver_frame_orig,
    )
s_real = da3_frame_scale

total_time = time.perf_counter() - total_start

# save the results together with timing / rank metadata
npz_path = os.path.join(output_path, 'result_XM.npz')
np.savez(
    npz_path,
    R=R_real,
    s=s_real,
    p=p_est,
    t=t_est,
    # original frame (or photo) -> solved camera index (-1 for frames dropped by checklandmarks/XM2): R, s and t are
    # stored per solved camera; this scatters them back onto every frame.
    indices_all=np.asarray(indices_all, dtype=np.int64),
    total_time=np.array(total_time, dtype=np.float64),
    solve_time=np.array(solver_time, dtype=np.float64),
    optimal_rank=np.array(optimal_rank, dtype=np.int64),
    ceres_point_scale_frame_rmse=np.full((N,), np.nan, dtype=np.float64),
    ceres_point_scale_frame_obs_count=np.zeros((N,), dtype=np.int64),
    ceres_point_scale_fit_rmse=np.array(np.nan, dtype=np.float64),
    ceres_point_scale_fit_max=np.array(np.nan, dtype=np.float64),
    ceres_point_keep_mask=np.zeros((M,), dtype=bool),
    da3_frame_scale=np.asarray(da3_frame_scale, dtype=np.float64),
    da3_frame_scale_loss=np.asarray(da3_frame_scale_loss, dtype=np.float64),
    da3_frame_scale_stats=np.array(da3_frame_scale_stats, dtype=object),
    # Ceres' refined focal: the K of step 1 times this (1 without CERES_OPTIMIZE_INTRINSICS)
    focal_scale=np.array(focal_scale, dtype=np.float64),
    ceres_depth_summary=np.array({}, dtype=object),
    depth_bend_summary=np.array({}, dtype=object),
    depth_bend_maps=np.zeros((0, 0, 0), dtype=np.float32),
)

export_colmap(R_real, t_est, p_est, indices_all, focal_scale)

print('Total time: ', total_time)
print('Solve time: ', solver_time)
print('Optimal rank: ', optimal_rank)

finish_script_logger()
