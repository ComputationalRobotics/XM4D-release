import argparse
import colorsys
import json
import os
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import cv2
import numpy as np
import networkx as nx
import random
try:
    import teaserpp_python
except ModuleNotFoundError:
    teaserpp_python = None
from scipy.stats import trim_mean
from scipy.ndimage import gaussian_filter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the static XM solver with configurable heuristic settings."
    )
    parser.add_argument("--mode", type=int, choices=[1, 2], default=1)
    parser.add_argument("--weight-mode", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--sphere-ratio", type=float, default=0.9)
    parser.add_argument(
        "--use-ceres",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the Ceres refinement stage after XM recovery.",
    )
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Serve a Viser visualization at the end.",
    )
    parser.add_argument("--viser-host", default=os.environ.get("VISER_HOST", "0.0.0.0"))
    parser.add_argument("--viser-port", type=int, default=int(os.environ.get("VISER_PORT", "8080")))
    parser.add_argument("--viser-max-points", type=int, default=int(os.environ.get("VISER_MAX_POINTS", "200000")))
    parser.add_argument("--viser-point-size", type=float, default=float(os.environ.get("VISER_POINT_SIZE", "0.0")))
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a RESULT_JSON line for machine parsing.",
    )
    parser.add_argument(
        "--bend-depth-with-ceres",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After Ceres, bend the monocular depth maps using kept Ceres landmark depths and evaluate against GT depth.",
    )
    parser.add_argument(
        "--no-da3-scale",
        dest="fit_da3_scale",
        action="store_false",
        default=True,
        help="Skip DA3 per-frame scale fitting via Chamfer loss; store scale=1.0 for all frames.",
    )
    return parser.parse_args()


args = parse_args()

MODE = args.mode  # 1 for new and 2 for old
if MODE == 1:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../XM/build/'))
elif MODE == 2:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../XM_old/build/'))

weight_mode = args.weight_mode
sphere_ratio = args.sphere_ratio

import XM

from utils.creatematrix import create_matrix
from utils.io import save_matrix_to_bin, load_matrix_from_bin
from utils.recoversolution import recover_XM
from utils.checkconnection import checklandmarks
from utils.readgt_BAL import load_BAL_gt
from utils.readgt_TUM import load_tum_gt
from utils.readgt_sintel import load_sintel_gt, load_sintel_camera, depth_read
from utils.readgt_DAVIS import load_davis_gt
from utils.readgt_replica import load_replica_gt
from utils.readgt_colmap import load_colmap_gt
from utils.readgt_custom import load_custom_gt
from utils.readgt_IMC import load_imc_gt
from utils.readgt_nvidia import load_nvidia_gt
from utils.readgt_dycheck import load_dycheck_gt
from utils.depth_sources import load_depth_source
from utils.script_logger import finish_script_logger, install_script_logger
import utils.definitions
import utils.dataset_config as dataset_config

total_start = time.perf_counter()
random.seed(0)


def env_float(name, default):
    value = os.environ.get(name)
    return float(value) if value is not None else float(default)


def env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value is not None else int(default)

USE_CERES_REGULARIZATION = dataset_config.get_bool(
    "USE_CERES_REGULARIZATION", False, env="USE_CERES_REGULARIZATION"
)
CERES_REGULARIZATION_THERESHOLD = dataset_config.get_float(
    "CERES_REGULARIZATION_THRESHOLD", 0.1, env="CERES_REGULARIZATION_THRESHOLD"
)
# Optionally let Ceres refine a single shared camera focal during bundle adjustment
# (self-calibration). The 2D landmarks are normalized by the initial intrinsic, so
# this refines a global focal-correction factor around 1.0, bounded to avoid the
# focal/depth gauge from running away. Off by default; helps scenes where the
# predicted (DA3) focal is poor.
CERES_OPTIMIZE_INTRINSICS = dataset_config.get_bool(
    "CERES_OPTIMIZE_INTRINSICS", False, env="CERES_OPTIMIZE_INTRINSICS"
)
CERES_INTRINSIC_SCALE_MIN = dataset_config.get_float(
    "CERES_INTRINSIC_SCALE_MIN", 0.25, env="CERES_INTRINSIC_SCALE_MIN"
)
CERES_INTRINSIC_SCALE_MAX = dataset_config.get_float(
    "CERES_INTRINSIC_SCALE_MAX", 4.0, env="CERES_INTRINSIC_SCALE_MAX"
)
# Robust loss on the Ceres reprojection residuals: "trivial" (plain least squares), "huber", "cauchy" or "softl1",
# with its scale in pixels (converted to the normalized landmark units with the median focal of intrinsics.npy).
CERES_LOSS = dataset_config.get_str("CERES_LOSS", "cauchy", env="CERES_LOSS").lower()
CERES_LOSS_PX = dataset_config.get_float("CERES_LOSS_PX", 2.0, env="CERES_LOSS_PX")
DEBUG = dataset_config.get_bool("DEBUG", False, env="DEBUG")  # also write tracks_debug_3d_solver.npz, Abar.bin and Q.mat

STATIC_TRACK_DEPTH_MAX = dataset_config.get_float(
    "STATIC_TRACK_DEPTH_MAX", 50.0, env="STATIC_TRACK_DEPTH_MAX"
)
STATIC_TRACK_DEPTH_GRAD_MAX = dataset_config.get_float(
    "STATIC_TRACK_DEPTH_GRAD_MAX", 2.0, env="STATIC_TRACK_DEPTH_GRAD_MAX"
)
STATIC_TRACK_DEPTH_JUMP_THRESHOLD = (
    None
    if os.environ.get("STATIC_TRACK_DEPTH_JUMP_THRESHOLD") in {None, "", "none", "None"}
    else env_float("STATIC_TRACK_DEPTH_JUMP_THRESHOLD", 0.0)
)

# Intrinsics come from a single source: outputs/depth_anything_3/intrinsics.npy,
# written by get_depths_clean.py. That file already encodes the GT-vs-DA3 choice, so
# the solver never re-decides it here -- it just loads and uses that K everywhere.

DA3_SCALE_MIN = env_float("DA3_SCALE_MIN", 0.25)
DA3_SCALE_MAX = env_float("DA3_SCALE_MAX", 4.0)
DA3_SCALE_LEVELS = env_int("DA3_SCALE_LEVELS", 5)
DA3_SCALE_CANDIDATES = env_int("DA3_SCALE_CANDIDATES", 15)
DA3_SCALE_STRIDE = env_int("DA3_SCALE_STRIDE", 4)
DA3_SCALE_DEPTH_MIN = env_float("DA3_SCALE_DEPTH_MIN", 0.0)
DA3_SCALE_DEPTH_MAX = env_float("DA3_SCALE_DEPTH_MAX", 0.0)
DA3_SCALE_FIT_POINTS = env_int("DA3_SCALE_FIT_POINTS", 1000)
DA3_SCALE_REFERENCE_POINTS = env_int("DA3_SCALE_REFERENCE_POINTS", 3000)
DA3_SCALE_FORWARD_KEEP_PERCENTILE = env_float("DA3_SCALE_FORWARD_KEEP_PERCENTILE", 85.0)
DA3_SCALE_REVERSE_KEEP_PERCENTILE = env_float("DA3_SCALE_REVERSE_KEEP_PERCENTILE", 20.0)
DA3_SCALE_REVERSE_WEIGHT = env_float("DA3_SCALE_REVERSE_WEIGHT", 0.25)
DA3_SCALE_LOG_SCALE_REG = env_float("DA3_SCALE_LOG_SCALE_REG", 0.0)
DA3_SCALE_REFERENCE_KEEP_PERCENTILE = env_float("DA3_SCALE_REFERENCE_KEEP_PERCENTILE", 98.0)
# Speed: only the top-K best-forward candidates per level pay the (expensive)
# reverse-Chamfer KD-tree rebuild. Frames after the first use a narrow local
# search around the running-median log-scale (all frames share a similar scale).
DA3_SCALE_REVERSE_TOPK = env_int("DA3_SCALE_REVERSE_TOPK", 3)
DA3_SCALE_LOCAL_RADIUS = env_float("DA3_SCALE_LOCAL_RADIUS", 0.35)  # log-scale half-width
DA3_SCALE_LOCAL_LEVELS = env_int("DA3_SCALE_LOCAL_LEVELS", 3)
DA3_SCALE_LOCAL_CANDIDATES = env_int("DA3_SCALE_LOCAL_CANDIDATES", 9)
# The grid only locates the basin (a flat Chamfer loss quantizes to grid nodes);
# a closed-form ICP-style refinement then gives a continuous, sub-grid per-frame
# scale: s = sum(dir . (nn - center)) / sum(dir . dir) over trimmed correspondences.
DA3_SCALE_REFINE_ITERS = env_int("DA3_SCALE_REFINE_ITERS", 4)
# KD-tree query threads (the result does not depend on them). workers=-1 spawns a thread per core (huge overhead
# for the many tiny per-frame queries on a big machine); 4 is fastest for these sizes (1.4x vs 1, 16 is slower).
DA3_SCALE_QUERY_WORKERS = env_int("DA3_SCALE_QUERY_WORKERS", 4)
RESULT_POINT_MAX_NORM = env_float("XM_RESULT_POINT_MAX_NORM", 2.0)


def _hsv_u8(index, total):
    color = colorsys.hsv_to_rgb(float(index) / max(1, int(total)), 0.85, 1.0)
    return np.asarray(np.round(np.asarray(color) * 255.0), dtype=np.uint8)


def _finite_point_cloud(points, colors=None, max_points=200000):
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        pts = pts.reshape(-1, 3)
    keep = np.isfinite(pts).all(axis=1)
    pts = pts[keep]

    cols = None
    if colors is not None:
        cols = np.asarray(colors)
        if cols.ndim == 2 and cols.shape[0] == keep.shape[0]:
            cols = cols[keep]
            if np.issubdtype(cols.dtype, np.floating):
                cols = np.clip(cols, 0.0, 1.0) * 255.0
            cols = np.asarray(np.clip(cols, 0, 255), dtype=np.uint8)
        else:
            cols = None

    max_points = int(max_points)
    if max_points > 0 and pts.shape[0] > max_points:
        idx = np.linspace(0, pts.shape[0] - 1, max_points).astype(np.int64)
        pts = pts[idx]
        if cols is not None:
            cols = cols[idx]
    if cols is None:
        cols = np.tile(np.array([190, 190, 190], dtype=np.uint8), (pts.shape[0], 1))
    return pts.astype(np.float32), cols


def _point_size_from_points(points, fallback=0.01):
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return float(fallback)
    bbox_min = np.nanmin(pts, axis=0)
    bbox_max = np.nanmax(pts, axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    if not np.isfinite(diag) or diag <= 0.0:
        return float(fallback)
    return float(np.clip(0.003 * diag, 0.002, 0.05))


def _camera_segments_from_extrinsics(extrinsics, scale):
    segments = []
    colors = []
    centers = []
    center_colors = []
    width, height = 640.0, 480.0
    fx, fy, cx, cy = 525.0, 525.0, 320.0, 240.0
    z = float(scale)
    local_corners = np.array(
        [
            [(-cx / fx) * z, (-cy / fy) * z, z],
            [((width - cx) / fx) * z, (-cy / fy) * z, z],
            [((width - cx) / fx) * z, ((height - cy) / fy) * z, z],
            [(-cx / fx) * z, ((height - cy) / fy) * z, z],
        ],
        dtype=np.float64,
    )
    edge_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for cam_idx, extrinsic in enumerate(extrinsics):
        E = np.asarray(extrinsic, dtype=np.float64)
        if E.shape != (4, 4) or not np.isfinite(E).all():
            continue
        T_wc = np.linalg.inv(E)
        R_wc = T_wc[:3, :3]
        t_wc = T_wc[:3, 3]
        corners = (R_wc @ local_corners.T).T + t_wc.reshape(1, 3)
        color = _hsv_u8(cam_idx, len(extrinsics))
        centers.append(t_wc)
        center_colors.append(color)
        for corner in corners:
            segments.append([t_wc, corner])
            colors.append(color)
        for i0, i1 in edge_pairs:
            segments.append([corners[i0], corners[i1]])
            colors.append(color)
    if not segments:
        return (
            np.zeros((0, 2, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.uint8),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.uint8),
        )
    return (
        np.asarray(segments, dtype=np.float32),
        np.asarray(colors, dtype=np.uint8),
        np.asarray(centers, dtype=np.float32),
        np.asarray(center_colors, dtype=np.uint8),
    )


def _scene_remove(server, path):
    try:
        server.scene.remove(path)
    except Exception:
        pass


def _wait_for_ctrl_c(message):
    print(message)
    stop = {"flag": False}
    prev_handler = signal.getsignal(signal.SIGINT)

    def _handle_sigint(signum, frame):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _handle_sigint)
    try:
        while not stop["flag"]:
            time.sleep(0.5)
    finally:
        signal.signal(signal.SIGINT, prev_handler)


def _viser_line_colors(colors, n_segments):
    arr = np.asarray(colors, dtype=np.uint8)
    if arr.shape == (3,):
        return arr
    if arr.shape == (int(n_segments), 2, 3):
        return arr
    if arr.shape == (int(n_segments), 3):
        return np.repeat(arr[:, None, :], 2, axis=1)
    return np.array([255, 255, 255], dtype=np.uint8)


def _visualize_teaser_alignment_viser(target, dst):
    from utils.viser_vis import get_viser_server

    target_pts, target_colors = _finite_point_cloud(target.T, max_points=args.viser_max_points)
    dst_pts, dst_colors = _finite_point_cloud(dst.T, max_points=args.viser_max_points)
    target_colors[:] = np.array([70, 120, 255], dtype=np.uint8)
    dst_colors[:] = np.array([255, 70, 70], dtype=np.uint8)

    server = get_viser_server(host=args.viser_host, port=args.viser_port)
    teaser_paths = ("/static_solver/teaser/estimate", "/static_solver/teaser/gt", "/static_solver/teaser/lines")
    for path in teaser_paths:
        _scene_remove(server, path)

    point_size = args.viser_point_size if args.viser_point_size > 0 else _point_size_from_points(np.vstack([target_pts, dst_pts]))
    if target_pts.size > 0:
        server.scene.add_point_cloud(
            "/static_solver/teaser/estimate",
            points=target_pts,
            colors=target_colors,
            point_size=point_size,
            point_shape="circle",
        )
    if dst_pts.size > 0:
        server.scene.add_point_cloud(
            "/static_solver/teaser/gt",
            points=dst_pts,
            colors=dst_colors,
            point_size=point_size,
            point_shape="circle",
        )
    n_lines = min(target_pts.shape[0], dst_pts.shape[0], 5000)
    if n_lines > 0:
        idx = np.linspace(0, min(target_pts.shape[0], dst_pts.shape[0]) - 1, n_lines).astype(np.int64)
        segments = np.stack([target_pts[idx], dst_pts[idx]], axis=1)
        server.scene.add_line_segments(
            "/static_solver/teaser/lines",
            points=segments,
            colors=np.array([80, 220, 80], dtype=np.uint8),
            line_width=1.0,
        )
    print(f"[viser] TEASER alignment preview served on http://localhost:{args.viser_port}")
    print(f"[viser] SSH forwarding example: ssh -L {args.viser_port}:localhost:{args.viser_port} <host>")
    _wait_for_ctrl_c("[viser] TEASER view active. Press Ctrl-C to continue to the reconstruction view.")
    for path in teaser_paths:
        _scene_remove(server, path)
    print("[viser] TEASER view cleared; continuing.")


def visualize_static_solver_viser(extrinsics, points, colors=None):
    from utils.viser_vis import get_viser_server

    pts, point_colors = _finite_point_cloud(points, colors=colors, max_points=args.viser_max_points)

    # Compute camera centers before centering so we can measure spread.
    _cam_scale_tmp = 1.0
    _, _, centers_raw, _ = _camera_segments_from_extrinsics(extrinsics, _cam_scale_tmp)

    # Scene center: median of camera positions, excluding the anchor frame which
    # is always at (0,0,0).  Subtracting the center shifts the whole scene so
    # cameras land near the Viser origin regardless of solver coordinate offset.
    if centers_raw.shape[0] > 0:
        valid_centers = centers_raw[np.linalg.norm(centers_raw, axis=1) > 1e-6]
        scene_center = np.median(valid_centers, axis=0) if valid_centers.shape[0] > 0 else np.zeros(3)
    else:
        scene_center = np.zeros(3)

    # cam_scale driven by camera trajectory spread, not point cloud size.
    if centers_raw.shape[0] >= 2:
        cam_spread = float(np.linalg.norm(centers_raw.max(axis=0) - centers_raw.min(axis=0)))
    else:
        cam_spread = 0.0
    scene_diag_pts = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) if pts.shape[0] >= 2 else 0.0
    scene_diag = max(cam_spread, scene_diag_pts, 1e-6)
    cam_scale = float(np.clip(0.05 * scene_diag, 0.02, 2.0))
    point_size = args.viser_point_size if args.viser_point_size > 0 else _point_size_from_points(pts)

    print(
        f"[viser] scene_center={scene_center}  cam_spread={cam_spread:.4g}  "
        f"scene_diag={scene_diag:.4g}  cam_scale={cam_scale:.4g}  point_size={point_size:.4g}"
    )

    cam_segments, cam_segment_colors, centers, center_colors = _camera_segments_from_extrinsics(extrinsics, cam_scale)

    # Shift everything to be centered at the origin.
    if pts.shape[0] > 0:
        pts = pts - scene_center.astype(pts.dtype)
    if centers.shape[0] > 0:
        centers = centers - scene_center.astype(centers.dtype)
    if cam_segments.shape[0] > 0:
        cam_segments = cam_segments - scene_center.astype(cam_segments.dtype)

    server = get_viser_server(host=args.viser_host, port=args.viser_port)

    for path in (
        "/static_solver/teaser/estimate",
        "/static_solver/teaser/gt",
        "/static_solver/teaser/lines",
        "/static_solver/reconstruction/points",
        "/static_solver/reconstruction/camera_centers",
        "/static_solver/reconstruction/camera_frustums",
    ):
        _scene_remove(server, path)

    if pts.size > 0:
        server.scene.add_point_cloud(
            "/static_solver/reconstruction/points",
            points=pts,
            colors=point_colors,
            point_size=point_size,
            point_shape="circle",
        )
    if centers.size > 0:
        server.scene.add_point_cloud(
            "/static_solver/reconstruction/camera_centers",
            points=centers,
            colors=center_colors,
            point_size=max(point_size * 2.5, 0.01),
            point_shape="circle",
        )
    if cam_segments.size > 0:
        try:
            server.scene.add_line_segments(
                "/static_solver/reconstruction/camera_frustums",
                points=cam_segments,
                colors=_viser_line_colors(cam_segment_colors, cam_segments.shape[0]),
                line_width=1.5,
            )
        except Exception as exc:
            print(f"[viser] camera frustum lines skipped: {exc}")

    print(
        f"[viser] Static solver view: points={pts.shape[0]} cameras={centers.shape[0]} "
        f"at http://localhost:{args.viser_port}"
    )
    print(f"[viser] SSH forwarding example: ssh -L {args.viser_port}:localhost:{args.viser_port} <host>")
    _wait_for_ctrl_c("[viser] Reconstruction view active. Press Ctrl-C to stop.")


def run_ceres_refine(
    edges,
    landmarks_2d,
    R_real,
    t_est,
    p_est,
    penalty_t_rel,
    neighbor_window,
    penalty_t_acc,
):
    try:
        from utils.ceresforXM import XM_Ceres_interface
    except ModuleNotFoundError as exc:
        if exc.name in {"pyceres", "pycolmap"}:
            print(f"[Ceres] skipped: optional dependency '{exc.name}' is not installed.")
            return R_real, t_est, p_est, 0.0, None
        raise

    return XM_Ceres_interface(
        edges,
        landmarks_2d,
        R_real,
        t_est,
        p_est,
        penalty_t_rel=penalty_t_rel,
        neighbor_window=neighbor_window,
        penalty_t_acc=penalty_t_acc,
        use_regularization=USE_CERES_REGULARIZATION,
        regularization_threshold=CERES_REGULARIZATION_THERESHOLD,
        optimize_intrinsics=CERES_OPTIMIZE_INTRINSICS,
        intrinsic_scale_min=CERES_INTRINSIC_SCALE_MIN,
        intrinsic_scale_max=CERES_INTRINSIC_SCALE_MAX,
        loss=CERES_LOSS,
        loss_scale=_ceres_loss_scale(),
    )


def _ceres_loss_scale():
    """CERES_LOSS_PX in normalized landmark units (px / median focal), or None without intrinsics.npy."""
    K_all = _load_intrinsics_npy(1)
    if K_all is None or CERES_LOSS == "trivial":
        return None
    return CERES_LOSS_PX / float(np.median(K_all[:, 0, 0]))


def build_sR_real(R_real, s_real, N):
    sR_real = np.zeros((3, 3 * N), dtype=np.float64)
    for i in range(N):
        sR_real[:, 3 * i:3 * i + 3] = s_real[i] * R_real[:, 3 * i:3 * i + 3]
    return sR_real


def _fit_scalar_scale(src, tgt, scale_min, scale_max):
    denom = float(np.sum(src * src))
    numer = float(np.sum(src * tgt))
    if denom <= 1e-12 or not np.isfinite(denom) or not np.isfinite(numer):
        return 1.0
    return float(np.clip(numer / denom, scale_min, scale_max))


def fit_frame_scales_to_ceres_points(edges, landmarks, R_real, t_est, p_ceres, N, M):
    """Fit one positive scale per frame against Ceres' 3D points.

    For frame i and observation k of point j:
        p_ceres[j] - t_i ~= s_i * R_i * x_ij

    The scalar s_i is solved in closed form, then refit after trimming large
    residual observations. Ceres points with consistently high residual are
    removed from the saved point cloud.
    """
    edges = np.asarray(edges, dtype=np.int64)
    landmarks = np.asarray(landmarks, dtype=np.float64)
    p_ceres = np.asarray(p_ceres, dtype=np.float64)
    t_est = np.asarray(t_est, dtype=np.float64)

    scale_min = max(env_float("CERES_POINT_SCALE_MIN", 0.5), 1e-8)
    scale_max = env_float("CERES_POINT_SCALE_MAX", 2.0)
    trim_percentile = env_float("CERES_POINT_SCALE_TRIM_PERCENTILE", 85.0)
    point_drop_percentile = env_float("CERES_POINT_OUTLIER_KEEP_PERCENTILE", 90.0)
    point_norm_max = env_float("CERES_POINT_MAX_NORM", 20.0)
    min_obs = env_int("CERES_POINT_SCALE_MIN_OBS", 6)
    if scale_max <= scale_min:
        print(f"[Ceres point scale] invalid bounds [{scale_min}, {scale_max}]; using [0.8, 1.2].")
        scale_min, scale_max = 0.8, 1.2
    trim_percentile = float(np.clip(trim_percentile, 50.0, 100.0))
    point_drop_percentile = float(np.clip(point_drop_percentile, 50.0, 100.0))

    point_keep = np.isfinite(p_ceres).all(axis=0) & (np.linalg.norm(p_ceres, axis=0) < point_norm_max)
    s_fit = np.ones((N,), dtype=np.float64)
    frame_rmse = np.full((N,), np.nan, dtype=np.float64)
    frame_obs_count = np.zeros((N,), dtype=np.int64)
    point_err_sum = np.zeros((M,), dtype=np.float64)
    point_err_cnt = np.zeros((M,), dtype=np.int64)

    for i in range(N):
        obs = np.flatnonzero(edges[:, 0] == i + 1)
        if obs.size == 0:
            continue
        pid = edges[obs, 1] - 1
        valid = (
            (pid >= 0)
            & (pid < min(M, p_ceres.shape[1]))
            & point_keep[pid]
            & np.isfinite(landmarks[obs]).all(axis=1)
        )
        obs = obs[valid]
        pid = pid[valid]
        if obs.size < min_obs:
            print(f"[Ceres point scale] frame={i}: only {obs.size} valid observations; using s=1.")
            continue

        Ri = R_real[:, 3 * i:3 * i + 3]
        src = (Ri @ landmarks[obs].T).T
        tgt = p_ceres[:, pid].T - t_est[:, i].reshape(1, 3)
        finite = np.isfinite(src).all(axis=1) & np.isfinite(tgt).all(axis=1)
        src = src[finite]
        tgt = tgt[finite]
        pid = pid[finite]
        if src.shape[0] < min_obs:
            print(f"[Ceres point scale] frame={i}: only {src.shape[0]} finite observations; using s=1.")
            continue

        s0 = _fit_scalar_scale(src, tgt, scale_min, scale_max)
        residual = np.linalg.norm(s0 * src - tgt, axis=1)
        cutoff = float(np.percentile(residual, trim_percentile))
        inlier = residual <= cutoff
        if np.count_nonzero(inlier) >= min_obs:
            s1 = _fit_scalar_scale(src[inlier], tgt[inlier], scale_min, scale_max)
            residual = np.linalg.norm(s1 * src - tgt, axis=1)
            s_fit[i] = s1
        else:
            s_fit[i] = s0

        frame_obs_count[i] = int(src.shape[0])
        frame_rmse[i] = float(np.sqrt(np.mean(residual ** 2)))
        np.add.at(point_err_sum, pid, residual)
        np.add.at(point_err_cnt, pid, 1)

    point_mean_err = np.full((M,), np.inf, dtype=np.float64)
    valid_point_err = point_err_cnt > 0
    point_mean_err[valid_point_err] = point_err_sum[valid_point_err] / point_err_cnt[valid_point_err]
    if np.any(valid_point_err):
        err_cutoff = float(np.percentile(point_mean_err[valid_point_err], point_drop_percentile))
        point_keep &= valid_point_err & (point_mean_err <= err_cutoff)
    else:
        err_cutoff = float("nan")
        point_keep &= False

    print(
        "[Ceres point scale] fitted per-frame s from Ceres p_est: "
        f"min={float(np.min(s_fit)):.8f}, max={float(np.max(s_fit)):.8f}, "
        f"mean={float(np.mean(s_fit)):.8f}, bounds=[{scale_min:.4f}, {scale_max:.4f}]"
    )
    print(
        "[Ceres point scale] residual filtering: "
        f"frame_trim_percentile={trim_percentile:.1f}, point_keep_percentile={point_drop_percentile:.1f}, "
        f"point_err_cutoff={err_cutoff:.8f}, kept_points={int(np.count_nonzero(point_keep))}/{M}"
    )
    valid_frames = np.isfinite(frame_rmse)
    mean_rmse = float(np.mean(frame_rmse[valid_frames])) if np.any(valid_frames) else float("nan")
    max_rmse = float(np.max(frame_rmse[valid_frames])) if np.any(valid_frames) else float("nan")
    return s_fit, point_keep, frame_rmse, frame_obs_count, mean_rmse, max_rmse


def _compute_depth_grad(depth):
    depth = np.asarray(depth, dtype=np.float64)
    finite = np.isfinite(depth)
    depth_safe = depth.copy()
    depth_safe[~finite] = 0.0
    gy, gx = np.gradient(depth_safe)
    grad = np.sqrt(gx * gx + gy * gy)
    denom = np.clip(np.abs(depth), 1e-6, None)
    rel_grad = grad / denom
    rel_grad[~finite] = np.inf
    return rel_grad


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

    get_depths_clean.py writes this file with the chosen intrinsic already applied
    (GT when USE_DA3_INTRINSICS is off, DA3's own when on), so the solver just reads
    it -- no flag, no second decision. Returns K_all (T, 3, 3), or None if absent.
    """
    intrinsics_path = Path(utils.definitions.da3_depth_dir) / "intrinsics.npy"
    if not intrinsics_path.exists():
        return None
    K_all = np.load(str(intrinsics_path)).astype(np.float64)
    if K_all.ndim == 2 and K_all.shape == (3, 3):
        K_all = np.repeat(K_all[None], int(t_expected), axis=0)
    if K_all.ndim != 3 or K_all.shape[1:] != (3, 3):
        raise ValueError(f"Expected intrinsics shape (T,3,3), got {K_all.shape}")
    return K_all


def _load_selected_depth_maps_and_intrinsics(t_expected):
    depth_source = utils.definitions.depth_source.strip().lower()
    try:
        depth_maps, _, depth_path = load_depth_source(depth_source, utils.definitions)
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


def _is_da3_depth_source():
    return utils.definitions.depth_source.strip().lower() in {
        "da3",
        "depth_anything",
        "depthanything",
        "depth_anything_3",
    }


def _filter_point_outliers(points, keep_percentile):
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 2 and pts.shape[0] == 3:
        pts = pts.T
    pts = pts.reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] == 0 or float(keep_percentile) >= 100.0:
        return pts
    center = np.median(pts, axis=0)
    radius = np.linalg.norm(pts - center.reshape(1, 3), axis=1)
    threshold = np.percentile(radius, np.clip(float(keep_percentile), 1.0, 100.0))
    return pts[radius <= threshold]


def _downsample_points(points, max_points, seed):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if int(max_points) <= 0 or pts.shape[0] <= int(max_points):
        return pts
    rng = np.random.default_rng(int(seed))
    return pts[rng.choice(pts.shape[0], int(max_points), replace=False)]


def _trimmed_mean_square(distances, keep_percentile):
    d = np.asarray(distances, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("inf")
    keep_percentile = float(np.clip(keep_percentile, 1.0, 100.0))
    if keep_percentile < 100.0 and d.size > 4:
        threshold = np.percentile(d, keep_percentile)
        d = d[d <= threshold]
    if d.size == 0:
        return float("inf")
    return float(np.mean(d * d))


def _trimmed_ms_rows(distances, keep_percentile):
    """Vectorized per-row trimmed mean-of-squares. distances: (K, N) -> (K,)."""
    d = np.asarray(distances, dtype=np.float64)
    keep = float(np.clip(keep_percentile, 1.0, 100.0))
    if keep < 100.0 and d.shape[1] > 4:
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
    if float(DA3_SCALE_DEPTH_MAX) > 0.0:
        valid &= z <= float(DA3_SCALE_DEPTH_MAX)
    if dynamic_mask is not None:
        valid &= ~np.asarray(dynamic_mask, dtype=bool)[yy, xx]
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float64)

    yy = yy[valid].astype(np.float64)
    xx = xx[valid].astype(np.float64)
    z = z[valid]
    pts_cam = _unproject(xx, yy, z, K)
    if int(max_points) > 0 and pts_cam.shape[0] > int(max_points):
        rng = np.random.default_rng(1000 + int(frame_idx))
        pts_cam = pts_cam[rng.choice(pts_cam.shape[0], int(max_points), replace=False)]
    return pts_cam


def _da3_scale_search(center, base_dirs, target_tree, static_ref_f, log_lo, log_hi, levels, candidates):
    """Grid then closed-form refine over log-scale.

    The grid does a batched forward-Chamfer NN query per level (one call for all
    candidate scales) purely to locate the basin; the (optional) reverse term is
    NOT used for selection — it is a flat regularizer that would only add a KD-tree
    rebuild per level. A closed-form ICP refinement then pins the sub-grid scale,
    and the reverse term is evaluated once at the end for the reported loss.
    `center` must be shaped (1, 1, 3).
    """
    from scipy.spatial import cKDTree

    fwd_keep = float(DA3_SCALE_FORWARD_KEEP_PERCENTILE)
    rev_w = float(DA3_SCALE_REVERSE_WEIGHT)
    rev_keep = float(DA3_SCALE_REVERSE_KEEP_PERCENTILE)
    reg = float(DA3_SCALE_LOG_SCALE_REG)
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
        if reg > 0.0:
            totals = totals + reg * grid * grid
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
        if fwd_keep < 100.0 and d.size > 4:
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
    if rev_w > 0.0:
        rev, _ = cKDTree(src).query(static_ref_f, workers=workers)
        final_loss += rev_w * _trimmed_mean_square(rev, rev_keep)
    if reg > 0.0:
        final_loss += reg * float(np.log(max(s, 1e-8)) ** 2)
    return float(s), float(final_loss)


def estimate_da3_frame_scales_static_chamfer(R_real, t_est, static_points, frame_index=None):
    """frame_index[i] = original frame of solver frame i (the solver may drop frames); None = the same order."""
    T_expected = t_est.shape[1]
    frame_index = np.arange(T_expected) if frame_index is None else np.asarray(frame_index, dtype=np.int64)
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
    if not _is_da3_depth_source():
        stats["reason"] = f"depth_source={utils.definitions.depth_source!r} is not DA3"
        return scales, losses, stats

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

    try:
        from scipy.spatial import cKDTree
    except Exception as exc:
        stats["reason"] = f"scipy unavailable: {exc}"
        print(f"[da3-scale] skipped: scipy unavailable: {exc}")
        return scales, losses, stats

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


def _read_depth_any(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".dpt":
        return depth_read(str(path)).astype(np.float64)
    if suffix == ".npy":
        return np.load(str(path)).astype(np.float64)
    if suffix == ".npz":
        data = np.load(str(path))
        for key in ("depth", "depths", "depth_map", "depth_maps", "pred_depth", "arr_0"):
            if key in data.files:
                return np.asarray(data[key], dtype=np.float64)
        raise KeyError(f"No depth key found in {path}")
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Could not read depth image: {path}")
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = depth.astype(np.float64)
    if suffix in {".png", ".tif", ".tiff"} and float(np.nanmax(depth)) > 255.0:
        depth = depth / 1000.0
    return depth


def _load_gt_depth_maps_for_eval(t_expected):
    gt_dir = Path(utils.definitions.gt_depth_dir)
    candidates = []
    for pattern in ("*.dpt", "*.npy", "*.npz", "*.png", "*.tif", "*.tiff", "*.jpg", "*.jpeg"):
        candidates.extend(sorted(gt_dir.glob(pattern)))
    if not candidates:
        print(f"[depth-bend] no GT depth files found in {gt_dir}; skipping evaluation.")
        return None, None
    maps = []
    for path in candidates[: int(t_expected)]:
        try:
            maps.append(_read_depth_any(path))
        except Exception as exc:
            print(f"[depth-bend] skipped GT depth {path}: {exc}")
    if not maps:
        return None, None
    return maps, gt_dir


def _scaled_depth_l1(pred, gt):
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape:
        pred = cv2.resize(pred.astype(np.float32), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float64)
    valid = np.isfinite(pred) & np.isfinite(gt) & (pred > 1e-8) & (gt > 1e-8)
    if np.count_nonzero(valid) < 32:
        return None
    ratio = gt[valid] / np.clip(pred[valid], 1e-8, None)
    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    if ratio.size == 0:
        return None
    scale = float(np.median(ratio))
    normalizer = float(np.median(gt[valid]))
    if not np.isfinite(normalizer) or normalizer <= 1e-8:
        normalizer = 1.0
    l1 = float(np.mean(np.abs(pred[valid] * scale - gt[valid]) / normalizer))
    return l1, scale, int(np.count_nonzero(valid))


def _depth_to_color(depth, vmin, vmax):
    depth = np.asarray(depth, dtype=np.float64)
    valid = np.isfinite(depth) & (depth > 1e-8)
    denom = max(float(vmax) - float(vmin), 1e-8)
    norm = np.zeros(depth.shape, dtype=np.float32)
    norm[valid] = np.clip((depth[valid] - float(vmin)) / denom, 0.0, 1.0)
    img = cv2.applyColorMap((norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~valid] = 0
    return img


def _add_panel_label(img, label):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(
        img,
        label,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return img


def _sparse_depth_to_color(shape, u, v, depth, vmin, vmax, radius):
    panel = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    denom = max(float(vmax) - float(vmin), 1e-8)
    for x, y, z in zip(u, v, depth, strict=False):
        if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(z) or z <= 1e-8:
            continue
        xi = int(round(x))
        yi = int(round(y))
        if xi < 0 or xi >= shape[1] or yi < 0 or yi >= shape[0]:
            continue
        value = int(np.clip((z - float(vmin)) / denom, 0.0, 1.0) * 255.0)
        color = cv2.applyColorMap(np.array([[value]], dtype=np.uint8), cv2.COLORMAP_TURBO)[0, 0].tolist()
        cv2.circle(panel, (xi, yi), int(radius), color, -1, lineType=cv2.LINE_AA)
    return panel


def _save_ceres_depth_png(
    *,
    out_dir,
    frame_idx,
    gt_depth,
    source_depth,
    source_scale,
    ceres_u,
    ceres_v,
    ceres_depth,
    ceres_l1,
    source_l1,
):
    valid_gt = np.isfinite(gt_depth) & (gt_depth > 1e-8)
    if np.count_nonzero(valid_gt) < 32:
        return
    vmin, vmax = np.percentile(gt_depth[valid_gt], [2.0, 98.0])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.min(gt_depth[valid_gt]))
        vmax = float(np.max(gt_depth[valid_gt]))
    if vmax <= vmin:
        return

    gt_panel = _depth_to_color(gt_depth, vmin, vmax)
    source_l1_text = "nan" if source_l1 is None else f"{source_l1:.4f}"
    if source_depth is not None and source_scale is not None:
        source_scaled = np.asarray(source_depth, dtype=np.float64) * float(source_scale)
        if source_scaled.shape != gt_depth.shape:
            source_scaled = cv2.resize(
                source_scaled.astype(np.float32),
                (gt_depth.shape[1], gt_depth.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.float64)
        source_panel = _depth_to_color(source_scaled, vmin, vmax)
    else:
        source_panel = np.zeros_like(gt_panel)

    radius = env_int("CERES_DEPTH_VIZ_RADIUS", 5)
    ceres_panel = _sparse_depth_to_color(
        gt_depth.shape,
        ceres_u,
        ceres_v,
        ceres_depth,
        vmin,
        vmax,
        radius=max(1, radius),
    )

    gt_panel = _add_panel_label(gt_panel, "GT depth")
    source_panel = _add_panel_label(source_panel, f"Source depth scaled, L1={source_l1_text}")
    ceres_panel = _add_panel_label(ceres_panel, f"Ceres points scaled, L1={ceres_l1:.4f}")
    png = np.concatenate([gt_panel, source_panel, ceres_panel], axis=1)
    cv2.imwrite(str(out_dir / f"frame{int(frame_idx):06d}_gt_source_ceres_depth.png"), png)


def evaluate_ceres_point_depths(
    *,
    edges,
    landmarks,
    p_ceres_full,
    ceres_point_ids,
    R_real,
    t_est,
    s_real,
    N,
    output_path,
):
    gt_maps, gt_path = _load_gt_depth_maps_for_eval(N)
    if gt_maps is None:
        return {}
    source_depth_maps, source_K_all, source_depth_path = _load_selected_depth_maps_and_intrinsics(N)

    image_paths = sorted(Path(utils.definitions.video_dir).glob(utils.definitions.image_pattern))
    kept_id_to_col = {int(pid): i for i, pid in enumerate(np.asarray(ceres_point_ids, dtype=np.int64).tolist())}
    edges0 = np.asarray(edges, dtype=np.int64) - 1
    landmarks = np.asarray(landmarks, dtype=np.float64)
    p_ceres_full = np.asarray(p_ceres_full, dtype=np.float64)

    out_dir = Path(output_path) / "ceres_depth_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    frames = []
    frame_limit = min(N, len(gt_maps), len(image_paths))
    for f in range(frame_limit):
        image_name = image_paths[f].name
        if image_name not in gt:
            continue
        gt_depth = np.asarray(gt_maps[f], dtype=np.float64)
        K = np.asarray(gt[image_name]["K"], dtype=np.float64).copy()
        image = cv2.imread(str(image_paths[f]), cv2.IMREAD_UNCHANGED)
        if image is not None:
            H_img, W_img = image.shape[:2]
            K[0, :] *= gt_depth.shape[1] / float(W_img)
            K[1, :] *= gt_depth.shape[0] / float(H_img)

        pred_z = []
        gt_z = []
        source_z = []
        source_gt_z = []
        frame_rows = []
        obs = np.where(edges0[:, 0] == f)[0]
        Ri = R_real[:, 3 * f : 3 * f + 3]
        for obs_idx in obs:
            pid = int(edges0[obs_idx, 1])
            col = kept_id_to_col.get(pid)
            if col is None or col < 0 or col >= p_ceres_full.shape[1]:
                continue
            Xw = p_ceres_full[:, col]
            Xc = Ri.T @ (Xw - t_est[:, f])
            z_ceres = float(Xc[2])
            if not np.isfinite(z_ceres) or z_ceres <= 1e-8:
                continue
            x_norm = float(landmarks[obs_idx, 0] / max(landmarks[obs_idx, 2], 1e-8))
            y_norm = float(landmarks[obs_idx, 1] / max(landmarks[obs_idx, 2], 1e-8))
            u = K[0, 0] * x_norm + K[0, 2]
            v = K[1, 1] * y_norm + K[1, 2]
            ui = int(round(u))
            vi = int(round(v))
            if ui < 0 or ui >= gt_depth.shape[1] or vi < 0 or vi >= gt_depth.shape[0]:
                continue
            z_gt = float(gt_depth[vi, ui])
            if not np.isfinite(z_gt) or z_gt <= 1e-8:
                continue
            pred_z.append(z_ceres)
            gt_z.append(z_gt)
            frame_rows.append((f, pid, u, v, z_ceres, z_gt))
            if (
                source_depth_maps is not None
                and source_K_all is not None
                and f < source_depth_maps.shape[0]
                and f < source_K_all.shape[0]
            ):
                src_depth = source_depth_maps[f]
                Ks = source_K_all[f]
                us = Ks[0, 0] * x_norm + Ks[0, 2]
                vs = Ks[1, 1] * y_norm + Ks[1, 2]
                usi = int(round(us))
                vsi = int(round(vs))
                if 0 <= usi < src_depth.shape[1] and 0 <= vsi < src_depth.shape[0]:
                    z_source = float(src_depth[vsi, usi])
                    if np.isfinite(z_source) and z_source > 1e-8:
                        source_z.append(z_source)
                        source_gt_z.append(z_gt)

        pred_z = np.asarray(pred_z, dtype=np.float64)
        gt_z = np.asarray(gt_z, dtype=np.float64)
        
        if pred_z.size < 8:
            continue
        ratio = gt_z / np.clip(pred_z, 1e-8, None)
        ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
        if ratio.size == 0:
            continue
        scale = float(np.median(ratio))
        normalizer = float(np.median(gt_z))
        if not np.isfinite(normalizer) or normalizer <= 1e-8:
            normalizer = 1.0
        abs_rel_l1 = np.abs(scale * pred_z - gt_z) / normalizer
        frame_l1 = float(np.mean(abs_rel_l1))
        frame_item = {
            "frame": int(f),
            "l1": frame_l1,
            "scale": scale,
            "valid": int(pred_z.size),
            "median_abs_error": float(np.median(np.abs(scale * pred_z - gt_z))),
            "median_gt_depth": float(np.median(gt_z)),
            "p90_l1": float(np.percentile(abs_rel_l1, 90.0)),
        }
        source_z = np.asarray(source_z, dtype=np.float64)
        source_gt_z = np.asarray(source_gt_z, dtype=np.float64)
        source_scale = None
        source_l1_value = None
        if source_z.size >= 8:
            source_ratio = source_gt_z / np.clip(source_z, 1e-8, None)
            source_ratio = source_ratio[np.isfinite(source_ratio) & (source_ratio > 0)]
            if source_ratio.size > 0:
                source_scale = float(np.median(source_ratio))
                source_normalizer = float(np.median(source_gt_z))
                if not np.isfinite(source_normalizer) or source_normalizer <= 1e-8:
                    source_normalizer = 1.0
                print(source_scale, s_real[f])
                source_l1 = np.abs(source_scale * source_z - source_gt_z) / source_normalizer
                source_l1_value = float(np.mean(source_l1))
                frame_item.update({
                    "source_l1_at_samples": source_l1_value,
                    "source_scale": source_scale,
                    "source_valid": int(source_z.size),
                    "source_p90_l1": float(np.percentile(source_l1, 90.0)),
                })
        source_depth_for_frame = (
            source_depth_maps[f]
            if source_depth_maps is not None and f < source_depth_maps.shape[0]
            else None
        )
        _save_ceres_depth_png(
            out_dir=out_dir,
            frame_idx=f,
            gt_depth=gt_depth,
            source_depth=source_depth_for_frame,
            source_scale=source_scale,
            ceres_u=np.asarray([row[2] for row in frame_rows], dtype=np.float64),
            ceres_v=np.asarray([row[3] for row in frame_rows], dtype=np.float64),
            ceres_depth=scale * pred_z,
            ceres_l1=frame_l1,
            source_l1=source_l1_value,
        )
        frames.append(frame_item)
        for row in frame_rows:
            rows.append((*row, scale))

    if not frames:
        print("[ceres-depth] skipped: no valid Ceres/GT depth samples.")
        return {}

    l1 = np.asarray([item["l1"] for item in frames], dtype=np.float64)
    valid = np.asarray([item["valid"] for item in frames], dtype=np.float64)
    weighted_mean_l1 = float(np.sum(l1 * valid) / max(float(np.sum(valid)), 1e-8))
    source_frame_l1 = np.asarray(
        [item["source_l1_at_samples"] for item in frames if "source_l1_at_samples" in item],
        dtype=np.float64,
    )
    source_valid = np.asarray(
        [item["source_valid"] for item in frames if "source_l1_at_samples" in item],
        dtype=np.float64,
    )
    source_mean_l1 = float(np.mean(source_frame_l1)) if source_frame_l1.size else None
    source_weighted_mean_l1 = (
        float(np.sum(source_frame_l1 * source_valid) / max(float(np.sum(source_valid)), 1e-8))
        if source_frame_l1.size
        else None
    )
    summary = {
        "gt_depth": str(gt_path),
        "source_depth": None if source_depth_path is None else str(source_depth_path),
        "metric": "kept Ceres point depth at observed pixels, per-frame median scale, mean(abs(scale*z_ceres-z_gt)/median(z_gt))",
        "mean_l1": float(np.mean(l1)),
        "weighted_mean_l1": weighted_mean_l1,
        "median_l1": float(np.median(l1)),
        "max_l1": float(np.max(l1)),
        "source_mean_l1_at_samples": source_mean_l1,
        "source_weighted_mean_l1_at_samples": source_weighted_mean_l1,
        "frames_evaluated": int(len(frames)),
        "total_samples": int(np.sum(valid)),
        "frames": frames,
    }
    with open(out_dir / "ceres_depth_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    rows_arr = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 7), dtype=np.float64)
    np.savez_compressed(
        out_dir / "ceres_depth_samples.npz",
        samples=rows_arr,
        columns=np.asarray(["frame", "point_id", "u", "v", "z_ceres", "z_gt", "frame_scale"]),
    )
    print(
        "[ceres-depth] "
        f"mean_l1={summary['mean_l1']:.6f}, weighted_mean_l1={weighted_mean_l1:.6f}, "
        f"source_weighted_l1={source_weighted_mean_l1 if source_weighted_mean_l1 is not None else float('nan'):.6f}, "
        f"median_l1={summary['median_l1']:.6f}, frames={len(frames)}, samples={int(np.sum(valid))}"
    )
    return summary


def _ratio_field_from_anchors(shape, anchor_y, anchor_x, anchor_ratio, sigma, strength):
    H, W = shape
    ratio_img = np.zeros((H, W), dtype=np.float32)
    known = np.zeros((H, W), dtype=np.uint8)
    if anchor_ratio.size == 0:
        return np.ones((H, W), dtype=np.float32)
    yi = np.clip(np.rint(anchor_y).astype(np.int64), 0, H - 1)
    xi = np.clip(np.rint(anchor_x).astype(np.int64), 0, W - 1)
    ratio = np.clip(anchor_ratio.astype(np.float64), 0.05, 20.0)
    order = np.argsort(ratio)
    for y, x, r in zip(yi[order], xi[order], ratio[order], strict=False):
        ratio_img[y, x] = float(r)
        known[y, x] = 255
    unknown = (known == 0).astype(np.uint8) * 255
    if np.any(known):
        ratio_filled = cv2.inpaint(ratio_img, unknown, 5.0, cv2.INPAINT_TELEA)
        median_ratio = float(np.median(ratio))
        ratio_filled[~np.isfinite(ratio_filled)] = median_ratio
        ratio_filled[ratio_filled <= 0] = median_ratio
    else:
        ratio_filled = np.ones((H, W), dtype=np.float32)
    if float(sigma) > 0:
        ratio_filled = gaussian_filter(ratio_filled, sigma=float(sigma)).astype(np.float32)
    ratio_filled = np.clip(ratio_filled, 0.05, 20.0)
    return (1.0 + float(strength) * (ratio_filled - 1.0)).astype(np.float32)


def bend_depth_maps_with_ceres(
    *,
    edges,
    landmarks,
    p_ceres_full,
    ceres_point_ids,
    R_real,
    t_est,
    s_real,
    N,
    output_path,
):
    depth_maps, K_all, depth_path = _load_selected_depth_maps_and_intrinsics(N)
    if depth_maps is None or K_all is None:
        return {}, np.zeros((0, 0, 0), dtype=np.float32)
    gt_maps, gt_path = _load_gt_depth_maps_for_eval(depth_maps.shape[0])
    if gt_maps is None:
        return {}, np.zeros((0, 0, 0), dtype=np.float32)

    kept_id_to_col = {int(pid): i for i, pid in enumerate(np.asarray(ceres_point_ids, dtype=np.int64).tolist())}
    candidates = [(0.25, 5.0), (0.5, 5.0), (0.75, 10.0), (1.0, 10.0), (0.75, 25.0), (1.0, 25.0), (1.0, 50.0)]
    per_candidate = {
        f"strength{strength:g}_sigma{sigma:g}": {"l1": [], "frames": [], "bent": []}
        for strength, sigma in candidates
    }
    original_l1 = []
    original_frames = []
    anchors_per_frame = []

    edges0 = np.asarray(edges, dtype=np.int64) - 1
    landmarks = np.asarray(landmarks, dtype=np.float64)
    p_ceres_full = np.asarray(p_ceres_full, dtype=np.float64)
    out_dir = Path(output_path) / "depth_bend_ceres"
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in range(min(N, depth_maps.shape[0], len(gt_maps))):
        src_depth = np.asarray(depth_maps[f], dtype=np.float64)
        gt_depth = np.asarray(gt_maps[f], dtype=np.float64)
        if gt_depth.shape != src_depth.shape:
            gt_eval = cv2.resize(gt_depth.astype(np.float32), (src_depth.shape[1], src_depth.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float64)
        else:
            gt_eval = gt_depth
        orig_eval = _scaled_depth_l1(src_depth, gt_eval)
        if orig_eval is not None:
            original_l1.append(orig_eval[0])
            original_frames.append({"frame": int(f), "l1": orig_eval[0], "scale": orig_eval[1], "valid": orig_eval[2]})

        obs = np.where(edges0[:, 0] == f)[0]
        anchor_x = []
        anchor_y = []
        anchor_ratio = []
        K = K_all[f].astype(np.float64)
        for obs_idx in obs:
            pid = int(edges0[obs_idx, 1])
            col = kept_id_to_col.get(pid)
            if col is None or col < 0 or col >= p_ceres_full.shape[1]:
                continue
            Ri = R_real[:, 3 * f : 3 * f + 3]
            Xw = p_ceres_full[:, col]
            Xc = Ri.T @ (Xw - t_est[:, f]) / max(float(s_real[f]), 1e-8)
            z_ceres = float(Xc[2])
            if not np.isfinite(z_ceres) or z_ceres <= 1e-8:
                continue
            x_norm = float(landmarks[obs_idx, 0] / max(landmarks[obs_idx, 2], 1e-8))
            y_norm = float(landmarks[obs_idx, 1] / max(landmarks[obs_idx, 2], 1e-8))
            u = K[0, 0] * x_norm + K[0, 2]
            v = K[1, 1] * y_norm + K[1, 2]
            ui = int(round(u))
            vi = int(round(v))
            if ui < 0 or ui >= src_depth.shape[1] or vi < 0 or vi >= src_depth.shape[0]:
                continue
            z_src = float(src_depth[vi, ui])
            if not np.isfinite(z_src) or z_src <= 1e-8:
                continue
            anchor_x.append(u)
            anchor_y.append(v)
            anchor_ratio.append(z_ceres / z_src)
        anchors_per_frame.append(int(len(anchor_ratio)))

        for strength, sigma in candidates:
            key = f"strength{strength:g}_sigma{sigma:g}"
            field = _ratio_field_from_anchors(
                src_depth.shape,
                np.asarray(anchor_y, dtype=np.float64),
                np.asarray(anchor_x, dtype=np.float64),
                np.asarray(anchor_ratio, dtype=np.float64),
                sigma=sigma,
                strength=strength,
            )
            bent = src_depth * field
            eval_item = _scaled_depth_l1(bent, gt_eval)
            if eval_item is not None:
                per_candidate[key]["l1"].append(eval_item[0])
                per_candidate[key]["frames"].append({"frame": int(f), "l1": eval_item[0], "scale": eval_item[1], "valid": eval_item[2]})
            per_candidate[key]["bent"].append(bent.astype(np.float32))

    if not original_l1:
        return {}, np.zeros((0, 0, 0), dtype=np.float32)

    orig_mean = float(np.mean(original_l1))
    original_by_frame = {int(item["frame"]): item for item in original_frames}
    best_key = None
    best_mean = float("inf")
    for key, item in per_candidate.items():
        if item["l1"]:
            mean_l1 = float(np.mean(item["l1"]))
            item["mean_l1"] = mean_l1
            item["improved_frames"] = int(sum(
                f_b["l1"] < original_by_frame[int(f_b["frame"])]["l1"]
                for f_b in item["frames"]
                if int(f_b["frame"]) in original_by_frame
            ))
            if mean_l1 < best_mean:
                best_mean = mean_l1
                best_key = key
    if best_key is None:
        return {}, np.zeros((0, 0, 0), dtype=np.float32)

    frame_limit = min(N, depth_maps.shape[0], len(gt_maps))
    candidate_frame_eval = {
        key: {int(frame_item["frame"]): frame_item for frame_item in item["frames"]}
        for key, item in per_candidate.items()
    }
    selected_frames = []
    selected_maps = []
    for f in range(frame_limit):
        src = depth_maps[f].astype(np.float32)
        orig = original_by_frame.get(f)
        best_frame_key = "original"
        best_frame_l1 = orig["l1"] if orig is not None else float("inf")
        best_frame_scale = orig["scale"] if orig is not None else float("nan")
        best_frame_valid = orig["valid"] if orig is not None else 0
        best_map = src
        for key, item in per_candidate.items():
            frame_item = candidate_frame_eval[key].get(f)
            if frame_item is None or f >= len(item["bent"]):
                continue
            if frame_item["l1"] < best_frame_l1:
                best_frame_key = key
                best_frame_l1 = frame_item["l1"]
                best_frame_scale = frame_item["scale"]
                best_frame_valid = frame_item["valid"]
                best_map = item["bent"][f]
        selected_maps.append(best_map.astype(np.float32))
        selected_frames.append({
            "frame": int(f),
            "selected_key": best_frame_key,
            "l1": float(best_frame_l1),
            "scale": float(best_frame_scale),
            "valid": int(best_frame_valid),
            "delta": float(best_frame_l1 - orig["l1"]) if orig is not None else float("nan"),
        })

    bent_maps = np.stack(selected_maps, axis=0)
    selected_l1 = [item["l1"] for item in selected_frames if np.isfinite(item["l1"])]
    selected_mean = float(np.mean(selected_l1)) if selected_l1 else float("nan")
    selected_improved = int(sum(item["delta"] < 0.0 for item in selected_frames if np.isfinite(item["delta"])))
    selected_unchanged = int(sum(item["selected_key"] == "original" for item in selected_frames))
    np.save(out_dir / "depth_original.npy", depth_maps.astype(np.float32))
    np.save(out_dir / "depth_bent.npy", bent_maps.astype(np.float32))
    summary = {
        "source_depth": str(depth_path),
        "gt_depth": str(gt_path),
        "metric": "per-frame median scale, mean(abs(scale*depth-gt)/median(gt))",
        "original_mean_l1": orig_mean,
        "global_best_key": best_key,
        "global_best_mean_l1": best_mean,
        "global_best_delta": best_mean - orig_mean,
        "selected_mode": "per_frame_best_nonworse",
        "selected_mean_l1": selected_mean,
        "selected_delta": selected_mean - orig_mean,
        "selected_improved_frames": selected_improved,
        "selected_original_frames": selected_unchanged,
        "anchors_per_frame": anchors_per_frame,
        "original_frames": original_frames,
        "selected_frames": selected_frames,
        "candidates": {
            key: {k: v for k, v in item.items() if k != "bent"}
            for key, item in per_candidate.items()
        },
    }
    with open(out_dir / "depth_bend_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(
        "[depth-bend] "
        f"original_mean_l1={orig_mean:.6f}, selected_mean_l1={selected_mean:.6f}, "
        f"delta={selected_mean - orig_mean:.6f}, improved_frames={selected_improved}, "
        f"kept_original_frames={selected_unchanged}, anchors/frame mean={float(np.mean(anchors_per_frame)):.1f}"
    )
    return summary, bent_maps


def lift_2d_tracks_to_solver_input(
    data,
    *,
    gt,
    dataset_cat,
    image_dir,
    image_pattern,
    output_path,
    depth_max,
    depth_grad_max,
    depth_jump_threshold=None,
):
    tracks_xy = np.asarray(data["tracks_xy"], dtype=np.float64)
    occluded = np.asarray(data["occluded"], dtype=bool)
    if tracks_xy.ndim != 4 or tracks_xy.shape[0] != 1 or tracks_xy.shape[-1] != 2:
        raise ValueError(f"Expected tracks_xy shape (1,Q,T,2), got {tracks_xy.shape}")
    if occluded.shape[:3] != tracks_xy.shape[:3]:
        raise ValueError(f"occluded shape {occluded.shape} does not match tracks_xy {tracks_xy.shape}")

    depth_source = utils.definitions.depth_source.strip().lower()
    if depth_source == "gt":
        gt_depth_dir = Path(utils.definitions.gt_depth_dir)
        if dataset_cat == "Replica":
            depth_paths = sorted(gt_depth_dir.glob("*.png"))
        else:
            depth_paths = sorted(gt_depth_dir.glob("*.dpt"))
        if not depth_paths:
            raise FileNotFoundError(f"No GT depth files found in {gt_depth_dir}")
        depth_maps = None
        conf_maps = None
        depth_path = gt_depth_dir
        T_depth = len(depth_paths)
    elif depth_source in {"da3", "depth_anything", "depthanything", "depth_anything_3", "vggt", "spatracker", "spa_tracker", "spatrack", "unidepth", "uni_depth"}:
        depth_maps, conf_maps, depth_path = load_depth_source(depth_source, utils.definitions)
        depth_maps = np.asarray(depth_maps, dtype=np.float64)
        if depth_maps.ndim != 3:
            raise ValueError(f"Expected depth maps shape (T,H,W), got {depth_maps.shape}")
        T_depth = depth_maps.shape[0]
    else:
        raise ValueError(
            f"Unsupported depth source '{utils.definitions.depth_source}'. "
            "Use gt, da3, vggt, spatracker, or unidepth."
        )

    # Single intrinsic source: for DA3-family depth, lift with intrinsics.npy
    # (get_depths_clean.py already baked the GT-vs-DA3 choice into it). Only the "gt"
    # depth source keeps GT colmap K (handled per-frame below).
    da3_K_all = None
    if depth_source in {"da3", "depth_anything", "depthanything", "depth_anything_3"}:
        da3_K_all = _load_intrinsics_npy(T_depth)
        if da3_K_all is not None:
            print(f"[track-lift] using intrinsics.npy, shape {da3_K_all.shape}")
        else:
            print("[track-lift] intrinsics.npy missing; falling back to GT colmap intrinsics.")

    image_paths = sorted(Path(image_dir).glob(image_pattern))
    T = min(tracks_xy.shape[2], len(image_paths), T_depth)
    Q = tracks_xy.shape[1]
    if T <= 0:
        raise RuntimeError("No frames available for 2D-to-3D lifting.")

    tracks_xyz = np.full((1, Q, T, 3), np.nan, dtype=np.float64)
    conf_xyz = np.full((Q, T), np.nan, dtype=np.float64)
    depth_grad_deleted_obs_mask = np.zeros((1, Q, T), dtype=bool)
    depth_max_deleted_obs_mask = np.zeros((1, Q, T), dtype=bool)
    prev_depths = np.full(Q, np.nan, dtype=np.float64)

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

        if depth_source == "gt":
            if dataset_cat == "Replica":
                depth = read_replica_depth(str(depth_paths[t]))
            else:
                depth = depth_read(str(depth_paths[t]))
            conf_vals_map = None
        else:
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
        max_ok = z <= float(depth_max) if float(depth_max) > 0 else np.ones_like(z, dtype=bool)
        basic_depth_ok = finite_positive & max_ok
        grad_ok = grad <= float(depth_grad_max) if float(depth_grad_max) > 0 else np.ones_like(z, dtype=bool)

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
        if depth_jump_threshold is not None and float(depth_jump_threshold) > 0:
            prev = prev_depths[idx]
            jump_ok = np.isnan(prev) | (np.abs(zv - prev) <= float(depth_jump_threshold))
            if not np.any(jump_ok):
                continue
            idx = idx[jump_ok]
            zv = zv[jump_ok]

        xv = x_depth[idx]
        yv = y_depth[idx]
        Pc = _unproject(xv, yv, zv, K_depth)
        tracks_xyz[0, idx, t, :] = Pc
        prev_depths[idx] = zv

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
            depth_source=np.array(depth_source),
        )
    print(
        "[track-lift] lifted 2D tracks in static_solver: "
        f"obs={edges.shape[0]}, depth_grad_deleted={int(np.sum(depth_grad_deleted_obs_mask))}, "
        f"depth_max_deleted={int(np.sum(depth_max_deleted_obs_mask))}, source={depth_path}"
    )
    return edges, weights, landmarks


def ate_teaser_c2w_headless(R_est, t_est, R_gt, t_gt):
    def rigid_align(src_pts, dst_pts):
        src_mean = np.mean(src_pts, axis=1, keepdims=True)
        dst_mean = np.mean(dst_pts, axis=1, keepdims=True)
        src_centered = src_pts - src_mean
        dst_centered = dst_pts - dst_mean

        H = dst_centered @ src_centered.T
        U, _, Vt = np.linalg.svd(H)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        t = dst_mean - R @ src_mean
        return R, t

    N = int(R_est.shape[1] / 3)
    assert R_est.shape == R_gt.shape
    assert t_est.shape == t_gt.shape
    assert N == t_est.shape[1]

    t_cam_gt = np.zeros((3, N))
    t_cam_est = np.zeros((3, N))
    for i in range(N):
        t_cam_gt[:, i] = R_gt[:, 3 * i : 3 * i + 3].T @ (-t_gt[:, i])
        t_cam_est[:, i] = t_est[:, i]

    src = t_cam_est
    dst = t_cam_gt

    finite_mask = np.isfinite(src).all(axis=0) & np.isfinite(dst).all(axis=0)
    src = src[:, finite_mask]
    dst = dst[:, finite_mask]
    if src.shape[1] < 3:
        print("[ATE] Too few finite correspondences for TEASER; using identity fallback.")
        return 1.0, np.eye(3), np.zeros((3, 1))

    dst_avg = trim_mean(dst, proportiontocut=0.05, axis=1)
    src_avg = trim_mean(src, proportiontocut=0.05, axis=1)
    dst_dis = np.linalg.norm(dst - dst_avg.reshape(3, 1), axis=0)
    src_dis = np.linalg.norm(src - src_avg.reshape(3, 1), axis=0)
    index = (src_dis < np.percentile(src_dis, 90)) & (dst_dis < np.percentile(dst_dis, 90))
    if np.count_nonzero(index) >= 3:
        src = src[:, index]
        dst = dst[:, index]
    dst_avg = np.mean(dst, axis=1)
    src_avg = np.mean(src, axis=1)

    scale1 = np.mean(np.linalg.norm(dst - dst_avg.reshape(3, 1), axis=0))
    scale2 = np.mean(np.linalg.norm(src - src_avg.reshape(3, 1), axis=0))

    print(f"ATE TEASER: scale1={scale1}, scale2={scale2}, ratio={scale1/scale2}")

    if (
        not np.isfinite(scale1)
        or not np.isfinite(scale2)
        or scale1 <= 1e-12
        or scale2 <= 1e-12
        or src.shape[1] < 3
    ):
        print("[ATE] Degenerate TEASER inputs; using SVD rigid-alignment fallback.")
        scale_ratio = scale1 / max(scale2, 1e-12)
        src_norm = src * scale_ratio
        dst_norm = dst
        rotation, translation = rigid_align(src_norm, dst_norm)
        target = rotation @ src_norm + translation
        error = np.linalg.norm(target - dst_norm, axis=0)
        print("Fallback alignment error:", np.linalg.norm(error))
        return scale_ratio, rotation, translation

    src_norm = src / scale2
    dst_norm = dst / scale1

    if not np.isfinite(src_norm).all() or not np.isfinite(dst_norm).all():
        print("[ATE] Non-finite normalized inputs; using SVD rigid-alignment fallback.")
        rotation, translation = rigid_align(src, dst)
        target = rotation @ src + translation
        error = np.linalg.norm(target - dst, axis=0)
        print("Fallback alignment error:", np.linalg.norm(error))
        return 1.0, rotation, translation

    if src_norm.shape[1] > 5000:
        idx = np.random.choice(src_norm.shape[1], 5000, replace=False)
        src_norm = src_norm[:, idx]
        dst_norm = dst_norm[:, idx]

    if teaserpp_python is None:
        print("[ATE] teaserpp_python is not installed; using SVD rigid-alignment fallback.")
        rotation, translation = rigid_align(src_norm, dst_norm)
        target = rotation @ src_norm + translation
        error = np.linalg.norm(target - dst_norm, axis=0)
        print("Fallback alignment error:", np.linalg.norm(error))
        return scale1 / scale2, rotation, scale1 * translation.reshape(3, 1)

    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1
    solver_params.noise_bound = 0.3
    solver_params.estimate_scaling = False
    solver_params.rotation_estimation_algorithm = (
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    )
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12

    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    solver.solve(src_norm, dst_norm)
    solution = solver.getSolution()

    src = t_cam_est * scale1 / scale2
    dst = t_cam_gt
    target = solution.rotation @ src + scale1 * solution.translation.reshape(3, 1)
    error = np.linalg.norm(target - dst, axis=0)
    print("TEASER error:", np.linalg.norm(error))

    if args.visualize:
        try:
            _visualize_teaser_alignment_viser(target, dst)
        except Exception as exc:
            print(f"[viser] TEASER alignment preview skipped: {exc}")

    return scale1 / scale2, solution.rotation, scale1 * solution.translation.reshape(3, 1)


def eval_mosca_style_with_evo(R_real, t_est, s_real, R_gt, t_gt, apply_coordinate_fix=True):
    """MoSca-style camera trajectory evaluation (ATE/RPE) using evo.

    - Builds GT and estimated trajectories in c2w convention.
    - Centers both trajectories (as in MoSca eval).
    - Runs APE (translation RMSE), RPE translation RMSE, RPE rotation RMSE (deg).
    """
    try:
        import importlib
        from scipy.spatial.transform import Rotation

        PoseTrajectory3D = importlib.import_module("evo.core.trajectory").PoseTrajectory3D
        sync = importlib.import_module("evo.core.sync")
        main_ape = importlib.import_module("evo.main_ape")
        main_rpe = importlib.import_module("evo.main_rpe")
        metrics_mod = importlib.import_module("evo.core.metrics")
        PoseRelation = metrics_mod.PoseRelation
        Unit = metrics_mod.Unit
    except Exception as e:
        print(f"[MoSca-eval] Skipped: evo/scipy import failed: {e}")
        print("[MoSca-eval] Install with: pip install evo scipy")
        return None

    N = t_est.shape[1]
    tstamps = np.arange(N, dtype=np.float64)

    # Build GT trajectory in c2w from w2c (R,t)
    xyz_ref = []
    quat_ref_wxyz = []
    try:
        for i in range(N):
            R_w2c = np.asarray(R_gt[:, 3 * i:3 * i + 3], dtype=np.float64)
            t_w2c = np.asarray(t_gt[:, i], dtype=np.float64).reshape(3)

            R_c2w = R_w2c.T
            t_c2w = -R_w2c.T @ t_w2c

            xyz_ref.append(t_c2w)
            xyzw = Rotation.from_matrix(R_c2w).as_quat()
            quat_ref_wxyz.append(np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64))
    except ValueError as e:
        print(f"[MoSca-eval] Skipped: invalid GT rotation matrix: {e}")
        return None

    xyz_ref = np.stack(xyz_ref, axis=0)
    quat_ref_wxyz = np.stack(quat_ref_wxyz, axis=0)
    xyz_ref = xyz_ref - np.mean(xyz_ref, axis=0, keepdims=True)

    traj_ref = PoseTrajectory3D(
        positions_xyz=xyz_ref,
        orientations_quat_wxyz=quat_ref_wxyz,
        timestamps=tstamps,
    )

    xyz_est = []
    quat_est_wxyz = []
    try:
        for i in range(N):
            R_c2w = np.asarray(R_real[:, 3 * i:3 * i + 3], dtype=np.float64)
            t_c2w = np.asarray(t_est[:, i], dtype=np.float64).reshape(3)

            xyz_est.append(t_c2w)
            xyzw = Rotation.from_matrix(R_c2w).as_quat()
            quat_est_wxyz.append(np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64))
    except ValueError as e:
        print(f"[MoSca-eval] Skipped: invalid estimated rotation matrix: {e}")
        return None

    xyz_est = np.stack(xyz_est, axis=0)
    quat_est_wxyz = np.stack(quat_est_wxyz, axis=0)
    xyz_est = xyz_est - np.mean(xyz_est, axis=0, keepdims=True)

    traj_est = PoseTrajectory3D(
        positions_xyz=xyz_est,
        orientations_quat_wxyz=quat_est_wxyz,
        timestamps=tstamps,
    )

    # Synchronize and evaluate
    # traj_ref_sync, traj_est_sync = sync.associate_trajectories(traj_ref, traj_est)
    traj_ref_sync, traj_est_sync = traj_ref, traj_est

    try:
        result_ate = main_ape.ape(
            traj_ref_sync,
            traj_est_sync,
            est_name="traj",
            pose_relation=PoseRelation.translation_part,
            align=True,
            correct_scale=True,
        )
        ate = float(result_ate.stats["rmse"])

        result_rpe_trans = main_rpe.rpe(
            traj_ref_sync,
            traj_est_sync,
            est_name="traj",
            pose_relation=PoseRelation.translation_part,
            align=True,
            correct_scale=True,
            delta=1,
            delta_unit=Unit.frames,
            rel_delta_tol=0.01,
            all_pairs=True,
        )
        rpe_trans = float(result_rpe_trans.stats["rmse"])

        result_rpe_rot = main_rpe.rpe(
            traj_ref_sync,
            traj_est_sync,
            est_name="traj",
            pose_relation=PoseRelation.rotation_angle_deg,
            align=True,
            correct_scale=True,
            delta=1,
            delta_unit=Unit.frames,
            rel_delta_tol=0.01,
            all_pairs=True,
        )
        rpe_rot = float(result_rpe_rot.stats["rmse"])
    except Exception as e:
        print(f"[MoSca-eval] Skipped: evo evaluation failed: {e}")
        return None

    print("\n" + "=" * 60)
    print("MoSca-style eval (evo)")
    print("=" * 60)
    print(f"ATE ↓:        {ate:.6f} m")
    print(f"RPE trans ↓:  {rpe_trans:.6f} m")
    print(f"RPE rot ↓:    {rpe_rot:.6f}°")
    print("=" * 60)

    return ate, rpe_trans, rpe_rot

############################
# For experiment in paper only
# download the dataset from:
# https://drive.google.com/drive/folders/13_2mcKGKVU0ibWck2n4ajUrN2MaDfR7y?usp=sharing
# and put it in the folder './assets/Experiment/**'
############################

### S0: CONFIG
dataset_cat = utils.definitions.dataset_cat
dataset_path = str(utils.definitions.rootpath)
output_path = str(utils.definitions.output_dir)
Path(output_path).mkdir(parents=True, exist_ok=True)
install_script_logger("static_solver", output_path)

image_dir = utils.definitions.video_dir
image_pattern = utils.definitions.image_pattern
image_list = sorted([p.name for p in Path(image_dir).glob(image_pattern)])

if dataset_cat == "Sintel":
    gt = load_sintel_gt(dataset_path)
    EVLUATION = True
elif dataset_cat == "TUM":
    gt = load_tum_gt(dataset_path)
    EVLUATION = True
elif dataset_cat == "DAVIS":
    gt = load_davis_gt(dataset_path)
    EVLUATION = False
elif dataset_cat == "Replica":
    gt = load_replica_gt(dataset_path)
    EVLUATION = True
elif dataset_cat == "Syn4D":
    gt = load_colmap_gt(os.path.join(dataset_path, "colmap"))
    EVLUATION = True
elif dataset_cat == "Lab":
    gt = load_colmap_gt(os.path.join(dataset_path, "colmap"))
    EVLUATION = True
elif dataset_cat == "IMC":
    gt = load_imc_gt(dataset_path)
    EVLUATION = True
elif dataset_cat == "Custom":
    gt = load_custom_gt(dataset_path)
    EVLUATION = False
elif dataset_cat == "NVIDIA":
    gt = load_nvidia_gt(dataset_path)
    EVLUATION = True
elif dataset_cat == "Dycheck":
    gt = load_dycheck_gt(dataset_path)
    EVLUATION = True
else:
    raise ValueError(f"Unsupported dataset_cat '{dataset_cat}'")

data = np.load(os.path.join(output_path, f'output_tapnet_{utils.definitions.depth_source}.npz'))
if "landmarks" in data.files and "edges" in data.files and "weights" in data.files:
    edges = data['edges']
    weights = data['weights'][:, 0].astype(np.float64, copy=False)
    landmarks = data['landmarks'].astype(np.float64, copy=False)
    print("[track-lift] using pre-lifted 3D observations from tracker output.")
elif "tracks_xy" in data.files and "occluded" in data.files:
    print("[track-lift] lifting 2D tracks to 3D observations for solver input.")
    edges, weights_full, landmarks = lift_2d_tracks_to_solver_input(
        data,
        gt=gt,
        dataset_cat=dataset_cat,
        image_dir=image_dir,
        image_pattern=image_pattern,
        output_path=output_path,
        depth_max=STATIC_TRACK_DEPTH_MAX,
        depth_grad_max=STATIC_TRACK_DEPTH_GRAD_MAX,
        depth_jump_threshold=STATIC_TRACK_DEPTH_JUMP_THRESHOLD,
    )
    weights = weights_full[:, 0].astype(np.float64, copy=False)
else:
    raise KeyError(
        f"{data.files} does not contain either old 3D keys "
        "(edges, weights, landmarks) or new 2D keys (tracks_xy, occluded)."
    )

frame_ids = edges[:, 0].astype(np.int64)
weight_per_frame = np.bincount(
    frame_ids,
    weights=weights,
    minlength=frame_ids.max() + 1,
)

if weight_mode == 2:
    denom = np.maximum(weight_per_frame[frame_ids], 1.0)
    weights = weights / denom
elif weight_mode == 3:
    denom = np.maximum(np.sqrt(weight_per_frame[frame_ids]), 1.0)
    weights = weights / denom

# dalete the row that contains the same edges in view-graph
_, unique_indices = np.unique(edges, axis=0, return_index=True)
edges = edges[unique_indices,:]
weights = weights[unique_indices]
landmarks = landmarks[unique_indices,:]

edges = edges[:, [1,0]] + 1
rgbs = np.zeros((edges.shape[0], 3))

# delete_depth = landmarks[:, 2] > np.percentile(landmarks[:, 2], 90)  # remove points with depth larger than 99.5 percentile
# edges = edges[~delete_depth]
# weights = weights[~delete_depth]
# landmarks = landmarks[~delete_depth,:]
# rgbs = rgbs[~delete_depth,:]

# assert np.max(edges[:,0]) == len(image_list)

# # only preserve camera < 3
# indices = np.isin(edges[:, 0], np.linspace(41, 50, 10))
# edges = edges[indices]
# rgbs = rgbs[indices]
# weights = weights[indices]
# landmarks = landmarks[indices]

N = int(np.max(edges[:, 0]))
M = int(np.max(edges[:, 1]))

# Drop frames whose observation count is <= FRAME_MIN_POINTS in checklandmarks.
# Raise it to remove weakly-observed cameras (often the largest-error ones).
FRAME_MIN_POINTS = dataset_config.get_int("FRAME_MIN_POINTS", 10, env="XM_FRAME_MIN_POINTS")

PENALTY_T_MULT = dataset_config.get_float("PENALTY_T_MULT", 0.0, env="XM_PENALTY_T_MULT")
PENALTY_T = env_float(
    "XM_PENALTY_T",
    N * PENALTY_T_MULT,
)  # set >0 to regularize nearby-camera relative translations in XM
PENALTY_T_CERES_REL_MULT = dataset_config.get_float(
    "PENALTY_T_CERES_REL_MULT", 1.0, env="XM_PENALTY_T_CERES_REL_MULT"
)
PENALTY_T_CERES_REL = env_float(
    "XM_PENALTY_T_CERES_REL",
    PENALTY_T * PENALTY_T_CERES_REL_MULT,
)  # set >0 to regularize nearby-camera relative translations in Ceres
PENALTY_T_CERES_WINDOW = env_int("XM_PENALTY_T_CERES_WINDOW", 3)
PENALTY_T_CERES_ACC_MULT = dataset_config.get_float(
    "PENALTY_T_CERES_ACC_MULT", 1.0, env="XM_PENALTY_T_CERES_ACC_MULT"
)
PENALTY_T_CERES_ACC = env_float(
    "XM_PENALTY_T_CERES_ACC",
    PENALTY_T * PENALTY_T_CERES_ACC_MULT,
)  # set >0 to penalize (t_{k+1} - t_k) - (t_k - t_{k-1})

print(
    "[Penalty] "
    f"PENALTY_T={PENALTY_T:.6f}, "
    f"PENALTY_T_CERES_REL={PENALTY_T_CERES_REL:.6f}, "
    f"PENALTY_T_CERES_WINDOW={PENALTY_T_CERES_WINDOW}, "
    f"PENALTY_T_CERES_ACC={PENALTY_T_CERES_ACC:.6f}"
)

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

# Landmark filter (config-driven, default 0 = keep all): drop observations whose
# landmark (edges[:,1]) is seen in fewer than LANDMARK_MIN_VIEWS frames. Keeping
# only longer tracks shrinks M (number of landmarks) -> smaller Q -> faster solve.
LANDMARK_MIN_VIEWS = dataset_config.get_int("LANDMARK_MIN_VIEWS", 0, env="XM_LANDMARK_MIN_VIEWS")
if LANDMARK_MIN_VIEWS > 1:
    lm_counts = np.bincount(edges[:, 1], minlength=int(edges[:, 1].max()) + 1)
    keep = lm_counts[edges[:, 1]] >= LANDMARK_MIN_VIEWS
    print(f"[landmark-filter] LANDMARK_MIN_VIEWS={LANDMARK_MIN_VIEWS}: keep {int(keep.sum())}/{keep.shape[0]} observations")
    edges, landmarks, weights, rgbs = edges[keep], landmarks[keep], weights[keep], rgbs[keep]

if MODE == 1:
    edges, landmarks, weights, rgbs, indices_all = checklandmarks(edges, landmarks, weights, rgbs, N, M, use_max_frame=False, frame_min_points=FRAME_MIN_POINTS)
elif MODE == 2:
    edges, landmarks, weights, rgbs, indices_all = checklandmarks(edges, landmarks, weights, rgbs, N, M, frame_min_points=FRAME_MIN_POINTS)


N = int(np.max(edges[:, 0]))
M = int(np.max(edges[:, 1]))

# interface to create the matrix given the edges, weights and landmarks
Abar = create_matrix(weights, edges, landmarks, output_path, penalty_t=PENALTY_T, debug=DEBUG)
LAMBDA = dataset_config.get_float("LAMBDA", 10.0, env="LAMBDA")
# Solver-acceleration knobs (config-driven; defaults reproduce the original behaviour):
#   RANK     : max rank searched by the mixed solver (lower = faster; IMC static rank is 3).
#   USE_XM2  : run the second XM^2 solve (outlier-removal + resolve). False ~halves solver time.
RANK = dataset_config.get_int("RANK", 5, env="XM_RANK")
USE_XM2 = dataset_config.get_bool("USE_XM2", True, env="XM_USE_XM2")
tol = 1e-4
solver_time = 0.0
solver_start = time.perf_counter()
if MODE == 1:
    XM.solve_mixed(output_path,RANK,1e-6,LAMBDA,30,sphere_ratio)
elif MODE == 2:
    XM.solve(output_path, RANK, 1e-4, LAMBDA, 30)
solver_time += time.perf_counter() - solver_start

# visualize the camera poses
R,_ = load_matrix_from_bin(output_path + '/R.bin')
s,_ = load_matrix_from_bin(output_path + '/s.bin')
Q,_ = load_matrix_from_bin(output_path + '/Q.bin')    

print("S1 is", s[0])

# recover p and t
# details refer to our paper
R_real, s_real, p_est, t_est, optimal_rank = recover_XM(Q, R, s, Abar, LAMBDA, return_rank=True)
R_real = np.asarray(R_real, dtype=np.float64)
s_real = np.asarray(s_real, dtype=np.float64)
p_est = np.asarray(p_est, dtype=np.float64)
t_est = np.asarray(t_est, dtype=np.float64)

if USE_XM2:
  sR_real = build_sR_real(R_real, s_real, N)

  # XM^2
  # some times the result is not good, so we need to remove some outliers
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
  # Lower it (e.g. 85) to reject more imperfect-depth observations.
  XM2_KEEP_PERCENTILE = dataset_config.get_float("XM2_KEEP_PERCENTILE", 95.0, env="XM_XM2_KEEP_PERCENTILE")
  threshold = np.percentile(error, XM2_KEEP_PERCENTILE)

  indices_to_remove = np.where(error > threshold)[0]

  # threshold = np.percentile(error, 100)  # 90th percentile

  # indices_to_remove_2 = np.where(error > threshold)[0] 
  # print("different: ", np.setxor1d(indices_to_remove, indices_to_remove_2))

  edges = np.delete(edges, indices_to_remove, axis=0)
  weights = np.delete(weights, indices_to_remove)
  rgbs = np.delete(rgbs, indices_to_remove, axis=0)
  landmarks = np.delete(landmarks, indices_to_remove, axis=0)

  # # Bake the first-pass per-frame scale into each observation before the second
  # # pass. Each landmark is scaled by its own frame's s_real so the second solve
  # # starts from scale-consistent observations (expected s ~= 1). This must happen
  # # before checklandmarks below, which renumbers frames and would break the
  # # edges[:,0] -> s_real correspondence.
  # scale_src_idx = edges[:, 0] - 1  # 1-based frame id -> 0-based, matches first-pass s_real
  # landmarks = s_real[scale_src_idx, None] * landmarks
  # print(
  #     "[xm-rescale] applied first-pass per-frame scale to observations: "
  #     f"s_real min={float(np.min(s_real)):.6g} median={float(np.median(s_real)):.6g} "
  #     f"max={float(np.max(s_real)):.6g}"
  # )

  # second run
  if MODE == 1:
      edges, landmarks, weights, rgbs, indices = checklandmarks(edges, landmarks, weights, rgbs, N, M, use_max_frame=False, frame_min_points=FRAME_MIN_POINTS)
  elif MODE == 2:
      edges, landmarks, weights, rgbs, indices = checklandmarks(edges, landmarks, weights, rgbs, N, M, frame_min_points=FRAME_MIN_POINTS)
  N_old = np.where(indices_all > -1)[0].shape[0]
  indices_all_copy = indices_all.copy()
  for i in range(N_old):
      indices_all[np.where(indices_all_copy == i)[0]] = indices[i]

  N = int(np.max(edges[:, 0]))  
  M = int(np.max(edges[:, 1]))  

  Abar = create_matrix(weights, edges, landmarks, output_path, penalty_t=PENALTY_T, debug=DEBUG)

  solver_start = time.perf_counter()
  if MODE == 1:
      XM.solve_mixed(output_path, RANK,1e-4, 0.0, 30, sphere_ratio)
  elif MODE == 2:
      XM.solve(output_path, RANK, 1e-4, 0.0, 30)
  solver_time += time.perf_counter() - solver_start

  R,_ = load_matrix_from_bin(output_path + '/R.bin')
  s,_ = load_matrix_from_bin(output_path + '/s.bin')
  Q,_ = load_matrix_from_bin(output_path + '/Q.bin')        

  # recover p and t
  R_real, s_real, p_est, t_est, optimal_rank = recover_XM(Q, R, s, Abar, LAMBDA, return_rank=True)
  R_real = np.asarray(R_real, dtype=np.float64)
  s_real = np.asarray(s_real, dtype=np.float64)
  p_est = np.asarray(p_est, dtype=np.float64)
  t_est = np.asarray(t_est, dtype=np.float64)

# If you still do not have a good result, consider using Ceres to refine using 2D observations.
# But make sure your 2D observation is good enough -- otherwise this won't help

landmarks_2D = landmarks[:, :2] / landmarks[:, 2, None]
nan_mask = np.isnan(landmarks_2D).any(axis=1)
ceres_time = 0.0
ceres_cost = None
ceres_point_scale_fit_rmse = None
ceres_point_scale_fit_max = None
ceres_point_scale_frame_rmse = None
ceres_point_scale_frame_obs_count = None
ceres_point_keep_mask = None
depth_bend_summary = {}
depth_bend_maps = np.zeros((0, 0, 0), dtype=np.float32)
ceres_depth_summary = {}

if args.use_ceres:
    R_real, t_est, p_est, ceres_time, ceres_cost = run_ceres_refine(
        edges[~nan_mask],
        landmarks_2D[~nan_mask],
        R_real,
        t_est,
        p_est,
        penalty_t_rel=PENALTY_T_CERES_REL,
        neighbor_window=PENALTY_T_CERES_WINDOW,
        penalty_t_acc=PENALTY_T_CERES_ACC,
    )
    R_real = np.asarray(R_real, dtype=np.float64)
    t_est = np.asarray(t_est, dtype=np.float64)
    p_est = np.asarray(p_est, dtype=np.float64)
else:
    print("Skipping Ceres refinement (--no-use-ceres).")

# R1 = R_real[:, :3].copy()
# for i in range(N):
#     R_real[:,3*i:3*i+3] = R1.T @ R_real[:,3*i:3*i+3]

t1 = t_est[:, 0:1].copy()
t_est = t_est - t1
p_est = p_est - t1
if args.use_ceres:
    # (
    #     s_real,
    #     ceres_point_keep_mask,
    #     ceres_point_scale_frame_rmse,
    #     ceres_point_scale_frame_obs_count,
    #     ceres_point_scale_fit_rmse,
    #     ceres_point_scale_fit_max,
    # ) = fit_frame_scales_to_ceres_points(edges, landmarks, R_real, t_est, p_est, N, M)
    # s_real = s_real / s_real[0]
    # ceres_kept_point_ids = np.where(ceres_point_keep_mask)[0]
    # if ceres_kept_point_ids.size > 0:
    #     ceres_depth_summary = evaluate_ceres_point_depths(
    #         edges=edges,
    #         landmarks=landmarks,
    #         p_ceres_full=p_est[:, ceres_point_keep_mask],
    #         ceres_point_ids=ceres_kept_point_ids,
    #         R_real=R_real,
    #         t_est=t_est,
    #         s_real=s_real,
    #         N=N,
    #         output_path=output_path,
    #     )
    # else:
    #     print("[ceres-depth] skipped: no kept Ceres points.")
    # if args.bend_depth_with_ceres:
    #     if ceres_kept_point_ids.size > 0:
    #         depth_bend_summary, depth_bend_maps = bend_depth_maps_with_ceres(
    #             edges=edges,
    #             landmarks=landmarks,
    #             p_ceres_full=p_est[:, ceres_point_keep_mask],
    #             ceres_point_ids=ceres_kept_point_ids,
    #             R_real=R_real,
    #             t_est=t_est,
    #             s_real=s_real,
    #             N=N,
    #             output_path=output_path,
    #         )
    #     else:
    #         print("[depth-bend] skipped: no kept Ceres points.")
    if np.any(ceres_point_keep_mask):
        p_est = p_est[:, ceres_point_keep_mask]
    else:
        print("[Ceres point scale] no Ceres points survived residual filtering; keeping finite/norm-filtered points.")
        finite_p = np.isfinite(p_est).all(axis=0)
        p_est = p_est[:, finite_p]
    sR_real = build_sR_real(R_real, s_real, N)
else:
    sR_real = build_sR_real(R_real, s_real, N)
    ybar_est = Abar @ sR_real.T
    y_est = np.hstack((np.zeros((3, 1)), ybar_est.T))
    p_est = y_est[:, N:N + M]
    finite_p = np.isfinite(p_est).all(axis=0)
    if RESULT_POINT_MAX_NORM > 0.0:
        p_norm = np.linalg.norm(p_est, axis=0)
        bounded_p = finite_p & (p_norm < RESULT_POINT_MAX_NORM)
        if np.any(bounded_p):
            finite_p = bounded_p
        else:
            print(
                "[point-filter] XM_RESULT_POINT_MAX_NORM would remove all points; "
                "keeping finite points instead."
            )
    p_est = p_est[:, finite_p]

if args.fit_da3_scale:
    solver_frame_orig = np.full(N, -1, dtype=np.int64)          # original frame of each solver frame
    for orig_idx, new_idx in enumerate(np.asarray(indices_all, dtype=np.int64)):
        if 0 <= new_idx < N:
            solver_frame_orig[new_idx] = orig_idx
    da3_frame_scale, da3_frame_scale_loss, da3_frame_scale_stats = estimate_da3_frame_scales_static_chamfer(
        R_real,
        t_est,
        p_est,
        frame_index=solver_frame_orig,
    )
    s_real = da3_frame_scale
else:
    print("[da3-scale] skipped (--no-da3-scale); using scale=1.0 for all frames.")
    da3_frame_scale = np.ones(N, dtype=np.float64)
    da3_frame_scale_loss = np.full(N, np.nan, dtype=np.float64)
    da3_frame_scale_stats = {}
sR_real = build_sR_real(R_real, s_real, N)

extrinsics = []
for i in range(N):
    # extrinsic is 4 * 4
    # extrinsic is world 2 camera, while R_real is camera 2 world
    extrinsics.append(np.vstack((np.hstack((R_real[:,3*i:3*i+3].T, -R_real[:,3*i:3*i+3].T @ t_est[:,i].reshape([3,1]))), np.array([0, 0, 0, 1]))))

# only visualize the camera poses
# print(extrinsics)

# uncomment this if you need to check the accuracy
# calculate accuracy

result_metrics = {
    "RPE_R": None,
    "RPE_T": None,
    "ATE_R": None,
    "ATE_T": None,
}

if EVLUATION:
    data_RPE_R = []
    data_RPE_T = []
    data_ATE_R = []
    data_ATE_T = []
    t_gt = np.zeros((3, N))
    camera_pos_gt = np.zeros((3, N))
    R_gt = np.zeros((3, 3 * N))
    ordered_orig_idx = np.where(indices_all >= 0)[0]
    if ordered_orig_idx.shape[0] != N:
        print(f"[EVLUATION] warning: expected {N} frames, got {ordered_orig_idx.shape[0]}")
    for orig_idx in ordered_orig_idx:
        new_idx = int(indices_all[orig_idx])
        if new_idx < 0 or new_idx >= N:
            continue
        R_gt[:, 3 * new_idx:3 * new_idx + 3] = gt[image_list[orig_idx]]["R"]
        t_gt[:, new_idx] = gt[image_list[orig_idx]]["t"]
        camera_pos_gt[:, new_idx] = -R_gt[:, 3 * new_idx:3 * new_idx + 3].T @ t_gt[:, new_idx]

    order_new = indices_all[ordered_orig_idx].astype(np.int64)
    order_new = order_new[(order_new >= 0) & (order_new < N)]
    if order_new.shape[0] != N:
        print(f"[EVLUATION] warning: expected {N} ordered indices, got {order_new.shape[0]}")

    R_real_ordered = np.zeros_like(R_real)
    t_est_ordered = np.zeros_like(t_est)
    R_gt_ordered = np.zeros_like(R_gt)
    t_gt_ordered = np.zeros_like(t_gt)
    camera_pos_gt_ordered = np.zeros_like(camera_pos_gt)
    for j, new_idx in enumerate(order_new):
        R_real_ordered[:, 3 * j:3 * j + 3] = R_real[:, 3 * new_idx:3 * new_idx + 3]
        t_est_ordered[:, j] = t_est[:, new_idx]
        R_gt_ordered[:, 3 * j:3 * j + 3] = R_gt[:, 3 * new_idx:3 * new_idx + 3]
        t_gt_ordered[:, j] = t_gt[:, new_idx]
        camera_pos_gt_ordered[:, j] = camera_pos_gt[:, new_idx]

    R_real = R_real_ordered
    t_est = t_est_ordered
    R_gt = R_gt_ordered
    t_gt = t_gt_ordered
    camera_pos_gt = camera_pos_gt_ordered
    if da3_frame_scale.shape[0] >= N and order_new.shape[0] == N:
        da3_frame_scale = da3_frame_scale[order_new]
        da3_frame_scale_loss = da3_frame_scale_loss[order_new]


    avg_t_gt = np.mean(camera_pos_gt,axis=1)
    cov_t_gt = np.mean(np.linalg.norm(camera_pos_gt - avg_t_gt.reshape(3,1),axis=0))

    s_g, R_g, t_g = ate_teaser_c2w_headless(R_real, t_est, R_gt, t_gt)

    # Align estimates before MoSca-style evaluation.
    # Keep the same representation used throughout this file:
    # - R_real: stacked c2w rotation blocks, shape (3, 3N)
    # - t_est:  c2w camera centers, shape (3, N)
    #
    # The conversion -R^T t is only for building a per-frame w2c extrinsic,
    # not for the trajectory arrays consumed by the evaluator.
    R_real_aligned = R_g @ R_real
    t_est_aligned = s_g * (R_g @ t_est) + t_g.reshape(3, 1)

    ATE_R = np.zeros(N)
    ATE_T = np.zeros(N)

    for i in range(N):
        cosvalue = (np.trace(R_g @ R_real[:,3*i:3*i+3] @ R_gt[:,3*i:3*i+3])-1)/2
        ATE_R[i] = np.abs(np.arccos(min(max(cosvalue,-1),1)))
        ATE_T[i] = np.linalg.norm((s_g * R_g @ t_est[:,i] + t_g.flatten())-R_gt[:,3*i:3*i+3].T @ (-t_gt[:,i]))

    RPE_R = []
    RPE_t = []
    for i in range(N):
        if N > 1000:
            for j in random.sample(list(np.arange(N)), 100):
                cosvalue = (np.trace(R_gt[:,3*i:3*i+3] @ R_gt[:,3*j:3*j+3].T @ R_real[:,3*j:3*j+3].T @  R_real[:,3*i:3*i+3])-1)/2
                RPE_R.append(np.abs(np.arccos(min(max(cosvalue,-1),1))))
                RPE_t.append(np.linalg.norm(- R_gt[:,3*i:3*i+3].T @ t_gt[:,i] + R_gt[:,3*j:3*j+3].T @ t_gt[:,j] - s_g * R_g @ (t_est[:,i] - t_est[:,j])))
        else:
            for j in range(i):
                cosvalue = (np.trace(R_gt[:,3*i:3*i+3] @ R_gt[:,3*j:3*j+3].T @ R_real[:,3*j:3*j+3].T @  R_real[:,3*i:3*i+3])-1)/2
                RPE_R.append(np.abs(np.arccos(min(max(cosvalue,-1),1))))
                RPE_t.append(np.linalg.norm(- R_gt[:,3*i:3*i+3].T @ t_gt[:,i] + R_gt[:,3*j:3*j+3].T @ t_gt[:,j] - s_g * R_g @ (t_est[:,i] - t_est[:,j])))

    result_metrics = {
        "RPE_R": float(np.average(RPE_R)),
        "RPE_T": float(np.average(RPE_t) / (cov_t_gt + 1.0)),
        "ATE_R": float(np.average(ATE_R)),
        "ATE_T": float(np.average(ATE_T) / (cov_t_gt + 1.0)),
    }
    print(cov_t_gt)
    print('RPE-R: ', result_metrics["RPE_R"],'RPE-T: ', result_metrics["RPE_T"],'ATE-R: ', result_metrics["ATE_R"],'ATE-T: ', result_metrics["ATE_T"])
    data_RPE_R.append(result_metrics["RPE_R"])
    data_RPE_T.append(result_metrics["RPE_T"])
    data_ATE_R.append(result_metrics["ATE_R"])
    data_ATE_T.append(result_metrics["ATE_T"])

    # MoSca-style metrics (APE/RPE via evo)
    eval_mosca_style_with_evo(
        R_real=R_real_aligned,
        t_est=t_est_aligned,
        s_real=None,
        R_gt=R_gt,
        t_gt=t_gt,
        apply_coordinate_fix=False,
    )

total_time = time.perf_counter() - total_start

# save the results together with timing / rank metadata
npz_path = os.path.join(output_path, 'result_XM.npz')
np.savez(
    npz_path,
    R=R_real,
    s=s_real,
    p=p_est,
    t=t_est,
    # original-frame -> solved-frame index (-1 for frames dropped by checklandmarks/XM2);
    # purely additive (already computed above) so downstream (EDG) can scatter the N
    # solved poses back onto the full T-frame timeline.
    indices_all=np.asarray(indices_all, dtype=np.int64),
    total_time=np.array(total_time, dtype=np.float64),
    solve_time=np.array(solver_time, dtype=np.float64),
    optimal_rank=np.array(-1 if optimal_rank is None else optimal_rank, dtype=np.int64),
    ceres_point_scale_frame_rmse=np.asarray(
        np.full((N,), np.nan, dtype=np.float64)
        if ceres_point_scale_frame_rmse is None
        else ceres_point_scale_frame_rmse,
        dtype=np.float64,
    ),
    ceres_point_scale_frame_obs_count=np.asarray(
        np.zeros((N,), dtype=np.int64)
        if ceres_point_scale_frame_obs_count is None
        else ceres_point_scale_frame_obs_count,
        dtype=np.int64,
    ),
    ceres_point_scale_fit_rmse=np.array(
        np.nan if ceres_point_scale_fit_rmse is None else ceres_point_scale_fit_rmse,
        dtype=np.float64,
    ),
    ceres_point_scale_fit_max=np.array(
        np.nan if ceres_point_scale_fit_max is None else ceres_point_scale_fit_max,
        dtype=np.float64,
    ),
    ceres_point_keep_mask=np.asarray(
        np.zeros((M,), dtype=bool) if ceres_point_keep_mask is None else ceres_point_keep_mask,
        dtype=bool,
    ),
    da3_frame_scale=np.asarray(da3_frame_scale, dtype=np.float64),
    da3_frame_scale_loss=np.asarray(da3_frame_scale_loss, dtype=np.float64),
    da3_frame_scale_stats=np.array(da3_frame_scale_stats, dtype=object),
    ceres_depth_summary=np.array(ceres_depth_summary, dtype=object),
    depth_bend_summary=np.array(depth_bend_summary, dtype=object),
    depth_bend_maps=np.asarray(depth_bend_maps, dtype=np.float32),
)

print('Total time: ', total_time)
print('Solve time: ', solver_time)
print('Optimal rank: ', optimal_rank)

result = {
    "dataset_cat": dataset_cat,
    "dataset_name": Path(dataset_path).name,
    "dataset_path": dataset_path,
    "depth_source": utils.definitions.depth_source,
    "mode": MODE,
    "sphere_ratio": sphere_ratio if MODE == 1 else None,
    "weight_mode": weight_mode,
    "use_ceres": bool(args.use_ceres),
    "penalty_t": float(PENALTY_T),
    "penalty_t_ceres_rel": float(PENALTY_T_CERES_REL),
    "penalty_t_ceres_window": int(PENALTY_T_CERES_WINDOW),
    "penalty_t_ceres_acc": float(PENALTY_T_CERES_ACC),
    "ceres_time": float(ceres_time),
    "ceres_cost": None if ceres_cost is None else float(ceres_cost),
    "ceres_point_scale_fit_rmse": None if ceres_point_scale_fit_rmse is None else float(ceres_point_scale_fit_rmse),
    "ceres_point_scale_fit_max": None if ceres_point_scale_fit_max is None else float(ceres_point_scale_fit_max),
    "ceres_point_scale_kept_points": (
        None if ceres_point_keep_mask is None else int(np.count_nonzero(ceres_point_keep_mask))
    ),
    "da3_frame_scale_stats": da3_frame_scale_stats,
    "ceres_depth": ceres_depth_summary,
    "depth_bend": depth_bend_summary,
    "RPE_R": result_metrics["RPE_R"],
    "RPE_T": result_metrics["RPE_T"],
    "ATE_R": result_metrics["ATE_R"],
    "ATE_T": result_metrics["ATE_T"],
    "total_time": float(total_time),
    "solve_time": float(solver_time),
    "optimal_rank": int(-1 if optimal_rank is None else optimal_rank),
}

if args.json:
    print("RESULT_JSON:", json.dumps(result, sort_keys=True))

finish_script_logger()

# visualize after logging so viewer time is not counted as solver runtime
if args.visualize:
    visualize_static_solver_viser(extrinsics, p_est.T)
    print(p_est.shape)
