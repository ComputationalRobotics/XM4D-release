#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pycolmap
import teaserpp_python
from scipy.stats import trim_mean

from utils.readgt_replica import load_replica_gt
from utils.readgt_sintel import load_sintel_gt

import open3d as o3d

ROOT_DIR = Path(__file__).parent.resolve()
SCENE_DIR = ROOT_DIR / "../datasets/Replica/office1_seq1/"
SPARSE_DIR = SCENE_DIR / "output_colmap/sparse"
DATASET_CAT = "replica"  # replica, sintel, syn4d

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate COLMAP reconstruction poses against GT."
    )
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=SCENE_DIR,
        help="Scene directory containing GT. Default: %(default)s",
    )
    parser.add_argument(
        "--sparse-dir",
        type=Path,
        default=SPARSE_DIR,
        help="COLMAP sparse directory. Default: <scene-dir>/sparse.",
    )
    parser.add_argument(
        "--gt-type",
        default=DATASET_CAT,
        help="Which GT loader to use.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print RESULT_JSON line for machine parsing.",
    )
    parser.add_argument(
        "--log-json",
        type=Path,
        default=None,
        help="Optional explicit log.json path. Default: same COLMAP output folder.",
    )
    return parser.parse_args()


def rotation_angle_rad(R: np.ndarray) -> float:
    cos_value = (np.trace(R) - 1.0) / 2.0
    cos_value = np.clip(cos_value, -1.0, 1.0)
    return float(abs(np.arccos(cos_value)))


def umeyama_sim3(
    src: np.ndarray,
    dst: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate Sim(3): dst ~= s * R @ src + t

    src, dst: shape (3, N)
    Returns:
        scale s
        rotation R, shape (3, 3)
        translation t, shape (3, 1)
    """
    assert src.shape == dst.shape
    assert src.shape[0] == 3

    n = src.shape[1]
    if n < 3:
        raise ValueError(f"Need at least 3 points for Sim(3), got {n}")

    src_mean = np.mean(src, axis=1, keepdims=True)
    dst_mean = np.mean(dst, axis=1, keepdims=True)

    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    cov = dst_centered @ src_centered.T / n

    U, singular_values, Vt = np.linalg.svd(cov)

    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[-1, -1] = -1.0

    R = U @ S @ Vt

    src_var = np.sum(src_centered**2) / n
    if src_var <= 1e-12:
        raise ValueError("Degenerate source trajectory for Sim(3) alignment.")

    scale = float(np.sum(singular_values * np.diag(S)) / src_var)
    t = dst_mean - scale * R @ src_mean

    return scale, R, t


def rigid_align_no_scale(
    src: np.ndarray,
    dst: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate rigid alignment: dst ~= R @ src + t."""
    src_mean = np.mean(src, axis=1, keepdims=True)
    dst_mean = np.mean(dst, axis=1, keepdims=True)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    H = dst_centered @ src_centered.T
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0.0:
        U[:, -1] *= -1.0
        R = U @ Vt
    t = dst_mean - R @ src_mean
    return R, t


def teaser_align_camera_centers(
    C_est: np.ndarray,
    C_gt: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Align estimated camera centers to GT with the previous TEASER style.

    TEASER estimates a robust rigid transform on normalized trajectories. The
    final returned transform is Sim(3)-style:
        C_est_aligned = s * R @ C_est + t
    where s comes from the GT/source trajectory scale ratio, matching the
    earlier eval_replica_colmap_vs_gt.py alignment logic.
    """
    assert C_est.shape == C_gt.shape
    assert C_est.shape[0] == 3

    finite = np.isfinite(C_est).all(axis=0) & np.isfinite(C_gt).all(axis=0)
    src_all = C_est[:, finite]
    dst_all = C_gt[:, finite]

    if src_all.shape[1] < 3:
        raise ValueError(f"Too few valid camera centers for TEASER: {src_all.shape[1]}")

    src = src_all
    dst = dst_all

    # Same trim-mean + 90th-percentile distance filtering as the previous code.
    dst_avg = trim_mean(dst, proportiontocut=0.05, axis=1)
    src_avg = trim_mean(src, proportiontocut=0.05, axis=1)
    dst_dis = np.linalg.norm(dst - dst_avg.reshape(3, 1), axis=0)
    src_dis = np.linalg.norm(src - src_avg.reshape(3, 1), axis=0)
    keep = (src_dis < np.percentile(src_dis, 90)) & (
        dst_dis < np.percentile(dst_dis, 90)
    )
    if np.count_nonzero(keep) >= 3:
        src = src[:, keep]
        dst = dst[:, keep]

    dst_avg = np.mean(dst, axis=1)
    src_avg = np.mean(src, axis=1)

    scale_gt = np.mean(np.linalg.norm(dst - dst_avg.reshape(3, 1), axis=0))
    scale_est = np.mean(np.linalg.norm(src - src_avg.reshape(3, 1), axis=0))

    if (
        not np.isfinite(scale_gt)
        or not np.isfinite(scale_est)
        or scale_gt <= 1e-12
        or scale_est <= 1e-12
        or src.shape[1] < 3
    ):
        print("[TEASER] Degenerate inputs; using no-scale SVD fallback.")
        R, t = rigid_align_no_scale(src_all, dst_all)
        return 1.0, R, t

    src_norm = src / scale_est
    dst_norm = dst / scale_gt

    if not np.isfinite(src_norm).all() or not np.isfinite(dst_norm).all():
        print("[TEASER] Non-finite normalized inputs; using no-scale SVD fallback.")
        R, t = rigid_align_no_scale(src_all, dst_all)
        return 1.0, R, t

    if src_norm.shape[1] > 5000:
        rng = np.random.default_rng(0)
        idx = rng.choice(src_norm.shape[1], 5000, replace=False)
        src_norm = src_norm[:, idx]
        dst_norm = dst_norm[:, idx]

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

    s = float(scale_gt / scale_est)
    R = np.asarray(solution.rotation, dtype=np.float64)
    t = scale_gt * np.asarray(solution.translation, dtype=np.float64).reshape(3, 1)

    target = s * R @ src_all + t
    teaser_rmse = float(np.sqrt(np.mean(np.sum((target - dst_all) ** 2, axis=0))))
    print(f"[TEASER] camera-center alignment RMSE: {teaser_rmse:.6f}")

    # visualization
    src_o3d = o3d.geometry.PointCloud()
    src_o3d.points = o3d.utility.Vector3dVector(target.T)
    src_o3d.paint_uniform_color([0, 0, 1])  # 蓝色

    dst_o3d = o3d.geometry.PointCloud()
    dst_o3d.points = o3d.utility.Vector3dVector(dst_all.T)
    dst_o3d.paint_uniform_color([1, 0, 0]) 

    # Draw correspondence lines
    src_pts = target.T
    dst_pts = dst_all.T
    points = np.vstack([src_pts, dst_pts])
    n = src_pts.shape[0]
    lines = [[i, i + n] for i in range(n)]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(np.tile([0.2, 0.8, 0.2], (n, 1)))

    o3d.visualization.draw_geometries([src_o3d, dst_o3d, line_set]) 

    return s, R, t


def apply_similarity_to_cameras(
    R_est_c2w: np.ndarray,
    C_est: np.ndarray,
    s: float,
    R_align: np.ndarray,
    t_align: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply C' = s * R_align @ C + t and R' = R_align @ R_c2w."""
    N = C_est.shape[1]
    R_aligned = np.zeros_like(R_est_c2w)
    C_aligned = s * R_align @ C_est + t_align.reshape(3, 1)

    for i in range(N):
        R_aligned[:, 3 * i : 3 * i + 3] = (
            R_align @ R_est_c2w[:, 3 * i : 3 * i + 3]
        )

    return R_aligned, C_aligned


def frame_id_from_name(name: str) -> int:
    stem = Path(name).stem
    match = re.search(r"(?<!\d)(\d+)$", stem)
    if match is None:
        raise ValueError(f"Could not extract frame id from image name: {name}")
    return int(match.group(1))


def load_colmap_gt(scene_dir: Path) -> dict[str, Any]:
    sparse_dir = scene_dir / "colmap" / "sparse"
    if not sparse_dir.is_dir():
        raise FileNotFoundError(f"Missing Syn4D GT COLMAP sparse directory: {sparse_dir}")

    rec = pycolmap.Reconstruction(str(sparse_dir))
    results: dict[str, Any] = {}

    for image_id in rec.images:
        image = rec.images[image_id]
        pose = image_cam_from_world(image)

        results[image.name] = {
            "id": frame_id_from_name(image.name),
            "K": None,
            "R": np.asarray(pose.rotation.matrix(), dtype=np.float64),
            "t": np.asarray(pose.translation, dtype=np.float64).reshape(3),
            "camera_id": image.camera_id,
        }

    return results


def image_cam_from_world(image: Any) -> Any:
    pose = image.cam_from_world
    if callable(pose):
        return pose()
    return pose


def load_gt(scene_dir: Path, gt_type: str) -> dict[str, Any]:
    gt_type = gt_type.lower()
    if gt_type == "replica":
        return load_replica_gt(str(scene_dir))
    if gt_type == "sintel":
        return load_sintel_gt(str(scene_dir))
    if gt_type in {"syn4d", "sync4d"}:
        return load_colmap_gt(scene_dir)
    raise ValueError(f"Unknown gt_type: {gt_type}")


def load_colmap_c2w_by_name(
    sparse_dir: Path,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Returns:
        image_name -> (R_c2w, C_c2w)
    """
    rec = pycolmap.Reconstruction(str(sparse_dir))
    by_name: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for image_id in rec.images:
        image = rec.images[image_id]
        pose = image_cam_from_world(image)

        R_w2c = np.asarray(pose.rotation.matrix(), dtype=np.float64)
        t_w2c = np.asarray(pose.translation, dtype=np.float64).reshape(3)

        R_c2w = R_w2c.T
        C_c2w = -R_w2c.T @ t_w2c

        by_name[image.name] = (R_c2w, C_c2w)

    return by_name


def build_pose_arrays(
    gt: dict[str, Any],
    colmap: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    common_pairs: list[tuple[str, str, str]] = []
    common_names = sorted(
        set(gt.keys()).intersection(colmap.keys()),
        key=lambda name: gt[name]["id"],
    )
    if common_names:
        common_pairs = [(name, name, name) for name in common_names]
    else:
        gt_by_base: dict[str, tuple[str, Any]] = {}
        for name, value in gt.items():
            base = Path(name).name
            if base not in gt_by_base:
                gt_by_base[base] = (name, value)

        colmap_by_base: dict[str, tuple[str, tuple[np.ndarray, np.ndarray]]] = {}
        for name, value in colmap.items():
            base = Path(name).name
            if base not in colmap_by_base:
                colmap_by_base[base] = (name, value)

        common_bases = sorted(
            set(gt_by_base.keys()).intersection(colmap_by_base.keys()),
            key=lambda base: gt_by_base[base][1]["id"],
        )
        common_pairs = [
            (base, gt_by_base[base][0], colmap_by_base[base][0])
            for base in common_bases
        ]

    if len(common_pairs) < 3:
        gt_sample = list(gt.keys())[:3]
        colmap_sample = list(colmap.keys())[:3]
        raise ValueError(
            f"Need at least 3 matched frames, got {len(common_pairs)}. "
            f"GT={len(gt)} COLMAP={len(colmap)}. "
            f"GT sample={gt_sample} COLMAP sample={colmap_sample}"
        )

    N = len(common_pairs)
    common_names = [name for name, _, _ in common_pairs]

    R_gt_w2c = np.zeros((3, 3 * N), dtype=np.float64)
    t_gt_w2c = np.zeros((3, N), dtype=np.float64)

    R_est_c2w = np.zeros((3, 3 * N), dtype=np.float64)
    C_est = np.zeros((3, N), dtype=np.float64)

    for i, (name, gt_key, colmap_key) in enumerate(common_pairs):
        Rg_w2c = np.asarray(gt[gt_key]["R"], dtype=np.float64)
        tg_w2c = np.asarray(gt[gt_key]["t"], dtype=np.float64).reshape(3)

        Re_c2w, Ce = colmap[colmap_key]

        R_gt_w2c[:, 3 * i : 3 * i + 3] = Rg_w2c
        t_gt_w2c[:, i] = tg_w2c

        R_est_c2w[:, 3 * i : 3 * i + 3] = Re_c2w
        C_est[:, i] = Ce

    return common_names, R_gt_w2c, t_gt_w2c, R_est_c2w, C_est


def compute_our_metrics(
    R_gt_w2c: np.ndarray,
    t_gt_w2c: np.ndarray,
    R_est_c2w: np.ndarray,
    C_est: np.ndarray,
) -> tuple[dict[str, float], tuple[float, np.ndarray, np.ndarray]]:
    N = C_est.shape[1]

    C_gt = np.zeros((3, N), dtype=np.float64)
    R_gt_c2w = np.zeros((3, 3 * N), dtype=np.float64)

    for i in range(N):
        Rg_w2c = R_gt_w2c[:, 3 * i : 3 * i + 3]
        tg_w2c = t_gt_w2c[:, i]

        R_gt_c2w[:, 3 * i : 3 * i + 3] = Rg_w2c.T
        C_gt[:, i] = -Rg_w2c.T @ tg_w2c

    cov_t_gt = np.mean(
        np.linalg.norm(C_gt - np.mean(C_gt, axis=1, keepdims=True), axis=0)
    )

    s, R_align, t_align = teaser_align_camera_centers(C_est, C_gt)

    ATE_R = []
    ATE_T = []

    for i in range(N):
        Re_c2w = R_est_c2w[:, 3 * i : 3 * i + 3]
        Rg_c2w = R_gt_c2w[:, 3 * i : 3 * i + 3]

        Re_aligned = R_align @ Re_c2w
        Ce_aligned = s * R_align @ C_est[:, i] + t_align.reshape(3)

        R_error = Re_aligned @ Rg_c2w.T
        ATE_R.append(rotation_angle_rad(R_error))
        ATE_T.append(np.linalg.norm(Ce_aligned - C_gt[:, i]))

    RPE_R = []
    RPE_T = []

    for i in range(N):
        Rg_i_w2c = R_gt_w2c[:, 3 * i : 3 * i + 3]
        Re_i_c2w = R_est_c2w[:, 3 * i : 3 * i + 3]

        for j in range(i):
            Rg_j_w2c = R_gt_w2c[:, 3 * j : 3 * j + 3]
            Re_j_c2w = R_est_c2w[:, 3 * j : 3 * j + 3]

            R_error = (
                Rg_i_w2c
                @ Rg_j_w2c.T
                @ Re_j_c2w.T
                @ Re_i_c2w
            )
            RPE_R.append(rotation_angle_rad(R_error))

            gt_delta = C_gt[:, i] - C_gt[:, j]
            est_delta = s * R_align @ (C_est[:, i] - C_est[:, j])
            RPE_T.append(np.linalg.norm(gt_delta - est_delta))

    metrics = {
        "RPE_R": float(np.mean(RPE_R)),
        "RPE_T": float(np.mean(RPE_T) / (cov_t_gt + 1.0)),
        "ATE_R": float(np.mean(ATE_R)),
        "ATE_T": float(np.mean(ATE_T) / (cov_t_gt + 1.0)),
        "scale_align": float(s),
        "gt_motion_scale": float(cov_t_gt),
    }

    return metrics, (s, R_align, t_align)

def load_existing_log(log_json_path: Path) -> dict[str, Any]:
    if not log_json_path.exists():
        return {}

    try:
        data = json.loads(log_json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {
            "previous_log_parse_error": True,
            "previous_log_raw": log_json_path.read_text(encoding="utf-8"),
        }

    if isinstance(data, dict):
        return data

    return {
        "previous_log_non_dict": data,
    }


def update_log_json(
    log_json_path: Path,
    eval_result: dict[str, Any],
) -> None:
    log_json_path.parent.mkdir(parents=True, exist_ok=True)

    log_data = load_existing_log(log_json_path)

    log_data["evaluation"] = eval_result

    log_json_path.write_text(
        json.dumps(log_data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def eval_mosca_style_with_evo(
    R_est_c2w: np.ndarray,
    C_est: np.ndarray,
    R_gt_w2c: np.ndarray,
    t_gt_w2c: np.ndarray,
) -> dict[str, float] | None:
    try:
        import importlib
        from scipy.spatial.transform import Rotation

        PoseTrajectory3D = importlib.import_module(
            "evo.core.trajectory"
        ).PoseTrajectory3D
        main_ape = importlib.import_module("evo.main_ape")
        main_rpe = importlib.import_module("evo.main_rpe")
        sync = importlib.import_module("evo.core.sync")
        metrics_mod = importlib.import_module("evo.core.metrics")
        PoseRelation = metrics_mod.PoseRelation
        Unit = metrics_mod.Unit
        geometry_mod = importlib.import_module("evo.core.geometry")
        GeometryException = geometry_mod.GeometryException
    except Exception as exc:
        print(f"[warn] MoSca/evo metrics skipped: {exc}")
        return None

    N = C_est.shape[1]
    timestamps = np.arange(N, dtype=np.float64)

    xyz_ref = []
    quat_ref_wxyz = []

    for i in range(N):
        Rg_w2c = R_gt_w2c[:, 3 * i : 3 * i + 3]
        tg_w2c = t_gt_w2c[:, i]

        Rg_c2w = Rg_w2c.T
        Cg = -Rg_w2c.T @ tg_w2c

        xyz_ref.append(Cg)
        xyzw = Rotation.from_matrix(Rg_c2w).as_quat()
        quat_ref_wxyz.append([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])

    xyz_est = []
    quat_est_wxyz = []

    for i in range(N):
        Re_c2w = R_est_c2w[:, 3 * i : 3 * i + 3]
        Ce = C_est[:, i]

        xyz_est.append(Ce)
        xyzw = Rotation.from_matrix(Re_c2w).as_quat()
        quat_est_wxyz.append([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])

    traj_ref = PoseTrajectory3D(
        positions_xyz=np.asarray(xyz_ref, dtype=np.float64),
        orientations_quat_wxyz=np.asarray(quat_ref_wxyz, dtype=np.float64),
        timestamps=timestamps,
    )

    traj_est = PoseTrajectory3D(
        positions_xyz=np.asarray(xyz_est, dtype=np.float64),
        orientations_quat_wxyz=np.asarray(quat_est_wxyz, dtype=np.float64),
        timestamps=timestamps,
    )

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    try:
        result_ate = main_ape.ape(
            traj_ref,
            traj_est,
            est_name="traj",
            pose_relation=PoseRelation.translation_part,
            align=True,
            correct_scale=True,
        )
    except GeometryException:
        print("[warn] Degenerate covariance: skipping ATE alignment")
        result_ate = main_ape.ape(
            traj_ref,
            traj_est,
            est_name="traj",
            pose_relation=PoseRelation.translation_part,
            align=False,
            correct_scale=False,
        )

    try:
        result_rpe_trans = main_rpe.rpe(
            traj_ref,
            traj_est,
            est_name="traj",
            pose_relation=PoseRelation.translation_part,
            align=True,
            correct_scale=True,
            delta=1,
            delta_unit=Unit.frames,
            rel_delta_tol=0.01,
            all_pairs=True,
        )
    except GeometryException:
        print("[warn] Degenerate covariance: skipping RPE(trans) alignment")
        result_rpe_trans = main_rpe.rpe(
            traj_ref,
            traj_est,
            est_name="traj",
            pose_relation=PoseRelation.translation_part,
            align=False,
            correct_scale=False,
            delta=1,
            delta_unit=Unit.frames,
            rel_delta_tol=0.01,
            all_pairs=True,
        )

    try:
        result_rpe_rot = main_rpe.rpe(
            traj_ref,
            traj_est,
            est_name="traj",
            pose_relation=PoseRelation.rotation_angle_deg,
            align=True,
            correct_scale=True,
            delta=1,
            delta_unit=Unit.frames,
            rel_delta_tol=0.01,
            all_pairs=True,
        )
    except GeometryException:
        print("[warn] Degenerate covariance: skipping RPE(rot) alignment")
        result_rpe_rot = main_rpe.rpe(
            traj_ref,
            traj_est,
            est_name="traj",
            pose_relation=PoseRelation.rotation_angle_deg,
            align=False,
            correct_scale=False,
            delta=1,
            delta_unit=Unit.frames,
            rel_delta_tol=0.01,
            all_pairs=True,
        )

    return {
        "MOSCA_ATE_M": float(result_ate.stats["rmse"]),
        "MOSCA_RPE_TRANS_M": float(result_rpe_trans.stats["rmse"]),
        "MOSCA_RPE_ROT_DEG": float(result_rpe_rot.stats["rmse"]),
    }


def default_log_json_path(sparse_dir: Path) -> Path:
    if sparse_dir.name == "sparse":
        return sparse_dir.parent / "log.json"
    return sparse_dir / "log.json"


def main() -> None:
    args = parse_args()

    scene_dir = args.scene_dir.resolve()
    sparse_dir = (
        args.sparse_dir.resolve()
        if args.sparse_dir is not None
        else (scene_dir / "sparse").resolve()
    )

    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
    if not sparse_dir.exists():
        raise FileNotFoundError(f"Sparse directory not found: {sparse_dir}")

    gt = load_gt(scene_dir, args.gt_type)
    colmap = load_colmap_c2w_by_name(sparse_dir)

    (
        common_names,
        R_gt_w2c,
        t_gt_w2c,
        R_est_c2w,
        C_est,
    ) = build_pose_arrays(gt, colmap)

    our_metrics, align = compute_our_metrics(
        R_gt_w2c=R_gt_w2c,
        t_gt_w2c=t_gt_w2c,
        R_est_c2w=R_est_c2w,
        C_est=C_est,
    )

    s_align, R_align, t_align = align
    R_est_c2w_aligned, C_est_aligned = apply_similarity_to_cameras(
        R_est_c2w=R_est_c2w,
        C_est=C_est,
        s=s_align,
        R_align=R_align,
        t_align=t_align,
    )

    mosca_metrics = eval_mosca_style_with_evo(
        R_est_c2w=R_est_c2w_aligned,
        C_est=C_est_aligned,
        R_gt_w2c=R_gt_w2c,
        t_gt_w2c=t_gt_w2c,
    )

    if mosca_metrics is None:
        mosca_metrics = {
            "MOSCA_ATE_M": None,
            "MOSCA_RPE_TRANS_M": None,
            "MOSCA_RPE_ROT_DEG": None,
        }

    result: dict[str, Any] = {
        "scene_dir": str(scene_dir),
        "sparse_dir": str(sparse_dir),
        "gt_type": args.gt_type,
        "matched_frames": int(len(common_names)),
        "gt_frames": int(len(gt)),
        "colmap_frames": int(len(colmap)),
        "alignment_method": "teaser",
        **our_metrics,
        **mosca_metrics,
    }

    log_json_path = (
        args.log_json.resolve()
        if args.log_json is not None
        else default_log_json_path(sparse_dir)
    )
    update_log_json(log_json_path, result)

    print(f"Matched frames: {len(common_names)} / GT={len(gt)} / COLMAP={len(colmap)}")
    print(f"RPE-R: {result['RPE_R']}")
    print(f"RPE-T: {result['RPE_T']}")
    print(f"ATE-R: {result['ATE_R']}")
    print(f"ATE-T: {result['ATE_T']}")
    print(f"MoSca ATE: {result['MOSCA_ATE_M']}")
    print(f"MoSca RPE trans: {result['MOSCA_RPE_TRANS_M']}")
    print(f"MoSca RPE rot: {result['MOSCA_RPE_ROT_DEG']}")
    print(f"Wrote log.json: {log_json_path}")

    if args.json:
        print("RESULT_JSON:", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
