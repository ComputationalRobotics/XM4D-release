#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scipy.stats import trim_mean

import teaserpp_python
import pycolmap

import open3d as o3d

from utils.readgt_replica import load_replica_gt
from utils.readgt_sintel import load_sintel_gt

import torch

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate COLMAP reconstruction poses against Replica GT using solver-style metrics."
    )
    parser.add_argument(
        "--scene-dir",
        type=Path,
        required=True,
        help="Replica scene directory (contains traj.txt and images/).",
    )
    parser.add_argument(
        "--sparse-dir",
        type=Path,
        default=None,
        help="COLMAP sparse directory (default: <scene-dir>/sparse).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print RESULT_JSON line for machine parsing.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Optional markdown file to save the report.",
    )
    return parser.parse_args()


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

    if (
        not np.isfinite(scale1)
        or not np.isfinite(scale2)
        or scale1 <= 1e-12
        or scale2 <= 1e-12
        or src.shape[1] < 3
    ):
        print("[ATE] Degenerate TEASER inputs; using SVD rigid-alignment fallback.")
        rotation, translation = rigid_align(src, dst)
        return 1.0, rotation, translation

    src_norm = src / scale2
    dst_norm = dst / scale1
    if not np.isfinite(src_norm).all() or not np.isfinite(dst_norm).all():
        print("[ATE] Non-finite normalized inputs; using SVD rigid-alignment fallback.")
        rotation, translation = rigid_align(src, dst)
        return 1.0, rotation, translation

    if src_norm.shape[1] > 5000:
        idx = np.random.choice(src_norm.shape[1], 5000, replace=False)
        src_norm = src_norm[:, idx]
        dst_norm = dst_norm[:, idx]

    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    solver.solve(src_norm, dst_norm)
    solution = solver.getSolution()

    src = t_cam_est
    dst = t_cam_gt

    src = src * scale1 / scale2
    dst = dst 

    target = solution.rotation @ src + scale1 * solution.translation.reshape(3,1)
    error = np.linalg.norm(target - dst, axis=0)
    print('TEASER error:',np.linalg.norm(error))
    
    # visualization
    src_o3d = o3d.geometry.PointCloud()
    src_o3d.points = o3d.utility.Vector3dVector(target.T)
    src_o3d.paint_uniform_color([0, 0, 1])  # 蓝色

    dst_o3d = o3d.geometry.PointCloud()
    dst_o3d.points = o3d.utility.Vector3dVector(dst.T)
    dst_o3d.paint_uniform_color([1, 0, 0]) 

    # Draw correspondence lines
    src_pts = target.T
    dst_pts = dst.T
    points = np.vstack([src_pts, dst_pts])
    n = src_pts.shape[0]
    lines = [[i, i + n] for i in range(n)]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(np.tile([0.2, 0.8, 0.2], (n, 1)))

    o3d.visualization.draw_geometries([src_o3d, dst_o3d, line_set]) 

    return scale1 / scale2, solution.rotation, scale1 * solution.translation.reshape(3, 1)


def eval_mosca_style_with_evo(R_real, t_est, R_gt, t_gt):
    """MoSca-style camera trajectory evaluation (ATE/RPE) using evo."""
    try:
        import importlib
        from scipy.spatial.transform import Rotation

        PoseTrajectory3D = importlib.import_module("evo.core.trajectory").PoseTrajectory3D
        main_ape = importlib.import_module("evo.main_ape")
        main_rpe = importlib.import_module("evo.main_rpe")
        sync = importlib.import_module("evo.core").sync
        metrics_mod = importlib.import_module("evo.core.metrics")
        PoseRelation = metrics_mod.PoseRelation
        Unit = metrics_mod.Unit
    except Exception as e:
        print(f"[MoSca-eval] Skipped: evo/scipy import failed: {e}")
        print("[MoSca-eval] Install with: pip install evo scipy")
        return None
    
    N = t_est.shape[1]
    tstamps = np.arange(N, dtype=np.float64)

    xyz_ref = []
    quat_ref_wxyz = []
    for i in range(N):
        R_w2c = np.asarray(R_gt[:, 3 * i : 3 * i + 3], dtype=np.float64)
        t_w2c = np.asarray(t_gt[:, i], dtype=np.float64).reshape(3)

        R_c2w = R_w2c.T
        t_c2w = -R_w2c.T @ t_w2c

        xyz_ref.append(t_c2w)
        xyzw = Rotation.from_matrix(R_c2w).as_quat()
        quat_ref_wxyz.append(np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64))

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
    for i in range(N):
        R_c2w = np.asarray(R_real[:, 3 * i : 3 * i + 3], dtype=np.float64)
        t_c2w = np.asarray(t_est[:, i], dtype=np.float64).reshape(3)

        xyz_est.append(t_c2w)
        xyzw = Rotation.from_matrix(R_c2w).as_quat()
        quat_est_wxyz.append(np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64))

    xyz_est = np.stack(xyz_est, axis=0)
    quat_est_wxyz = np.stack(quat_est_wxyz, axis=0)
    xyz_est = xyz_est - np.mean(xyz_est, axis=0, keepdims=True)

    traj_est = PoseTrajectory3D(
        positions_xyz=xyz_est,
        orientations_quat_wxyz=quat_est_wxyz,
        timestamps=tstamps,
    )

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    result_ate = main_ape.ape(
        traj_ref,
        traj_est,
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=False,
        correct_scale=False,
    )

    ate = float(result_ate.stats["rmse"])

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
    rpe_trans = float(result_rpe_trans.stats["rmse"])

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
    rpe_rot = float(result_rpe_rot.stats["rmse"])

    print("\n" + "=" * 60)
    print("MoSca-style eval (evo)")
    print("=" * 60)
    print(f"ATE ↓:        {ate:.6f} m")
    print(f"RPE trans ↓:  {rpe_trans:.6f} m")
    print(f"RPE rot ↓:    {rpe_rot:.6f}°")
    print("=" * 60)

    return ate, rpe_trans, rpe_rot


def load_colmap_w2c_by_name(sparse_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rec = pycolmap.Reconstruction(str(sparse_dir))
    by_name: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for image_id in rec.images:
        image = rec.images[image_id]
        pose = image.cam_from_world()
        R_w2c = np.asarray(pose.rotation.matrix(), dtype=np.float64)
        t_w2c = np.asarray(pose.translation, dtype=np.float64).reshape(3)
        by_name[image.name] = (R_w2c, t_w2c)
    return by_name


def main() -> None:
    args = parse_args()

    scene_dir = args.scene_dir.resolve()
    sparse_dir = (args.sparse_dir.resolve() if args.sparse_dir is not None else (scene_dir / "sparse").resolve())

    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
    if not sparse_dir.exists():
        raise FileNotFoundError(f"Sparse directory not found: {sparse_dir}")

    gt = load_replica_gt(str(scene_dir))
    # gt = load_sintel_gt(str(scene_dir))
    colmap = load_colmap_w2c_by_name(sparse_dir)

    common_names = sorted(set(gt.keys()).intersection(colmap.keys()), key=lambda n: gt[n]["id"])
    if len(common_names) < 3:
        raise ValueError(
            f"Need at least 3 matched frames, got {len(common_names)}. "
            f"GT={len(gt)} COLMAP={len(colmap)}"
        )

    N = len(common_names)
    print(f"Matched frames: {N} / GT={len(gt)} / COLMAP={len(colmap)}")

    R_gt = np.zeros((3, 3 * N), dtype=np.float64)   # world->camera
    t_gt = np.zeros((3, N), dtype=np.float64)       # world->camera
    R_est = np.zeros((3, 3 * N), dtype=np.float64)  # camera->world (solver convention)
    t_est = np.zeros((3, N), dtype=np.float64)      # camera center in world

    camera_pos_gt = np.zeros((3, N), dtype=np.float64)

    ## for mega_sam
    for i, name in enumerate(common_names):
        Rg = np.asarray(gt[name]["R"], dtype=np.float64)
        tg = np.asarray(gt[name]["t"], dtype=np.float64).reshape(3)

        Rp_w2c, tp_w2c = colmap[name]

        Rp_c2w = Rp_w2c.T
        tp_c2w = -Rp_w2c.T @ tp_w2c

        R_gt[:, 3 * i : 3 * i + 3] = Rg
        t_gt[:, i] = tg
        F = np.diag([1, 1, -1])
        R_est[:, 3 * i : 3 * i + 3] = Rp_w2c
        t_est[:, i] = tp_w2c
        # R_est[:, 3 * i : 3 * i + 3] = F @ Rp_w2c @ F  # MoSca-style reorientation
        # t_est[:, i] = F @ tp_w2c  # MoSca-style reorientation

        camera_pos_gt[:, i] = -Rg.T @ tg

    avg_t_gt = np.mean(camera_pos_gt, axis=1)
    cov_t_gt = np.mean(np.linalg.norm(camera_pos_gt - avg_t_gt.reshape(3, 1), axis=0))

    s_g, R_g, t_g = ate_teaser_c2w_headless(R_est, t_est, R_gt, t_gt)

    # aligned estimate for MoSca-style eval
    R_est_aligned = R_g @ R_est
    t_est_aligned = s_g * R_g @ t_est + t_g.reshape(3, 1)

    ATE_R = np.zeros(N)
    ATE_T = np.zeros(N)

    for i in range(N):
        cosvalue = (np.trace(R_g @ R_est[:, 3 * i : 3 * i + 3] @ R_gt[:, 3 * i : 3 * i + 3]) - 1) / 2
        ATE_R[i] = np.abs(np.arccos(min(max(cosvalue, -1), 1)))
        ATE_T[i] = np.linalg.norm(
            (s_g * R_g @ t_est[:, i] + t_g.flatten())
            - R_gt[:, 3 * i : 3 * i + 3].T @ (-t_gt[:, i])
        )

    RPE_R = []
    RPE_t = []
    for i in range(N):
        for j in range(i):
            cosvalue = (
                np.trace(
                    R_gt[:, 3 * i : 3 * i + 3]
                    @ R_gt[:, 3 * j : 3 * j + 3].T
                    @ R_est[:, 3 * j : 3 * j + 3].T
                    @ R_est[:, 3 * i : 3 * i + 3]
                )
                - 1
            ) / 2
            RPE_R.append(np.abs(np.arccos(min(max(cosvalue, -1), 1))))
            RPE_t.append(
                np.linalg.norm(
                    -R_gt[:, 3 * i : 3 * i + 3].T @ t_gt[:, i]
                    + R_gt[:, 3 * j : 3 * j + 3].T @ t_gt[:, j]
                    - s_g * R_g @ (t_est[:, i] - t_est[:, j])
                )
            )

    result = {
        "scene_dir": str(scene_dir),
        "sparse_dir": str(sparse_dir),
        "matched_frames": int(N),
        "RPE_R": float(np.average(RPE_R)),
        "RPE_T": float(np.average(RPE_t) / (cov_t_gt + 0.01)),
        "ATE_R": float(np.average(ATE_R)),
        "ATE_T": float(np.average(ATE_T) / (cov_t_gt + 0.01)),
        "scale_align": float(s_g),
    }
    print(cov_t_gt)
    print(
        "RPE-R: ",
        result["RPE_R"],
        "RPE-T: ",
        result["RPE_T"],
        "ATE-R: ",
        result["ATE_R"],
        "ATE-T: ",
        result["ATE_T"],
    )

    mosca = eval_mosca_style_with_evo(R_est_aligned, t_est_aligned, R_gt, t_gt)
    if mosca is not None:
        result["MOSCA_ATE_M"] = float(mosca[0])
        result["MOSCA_RPE_TRANS_M"] = float(mosca[1])
        result["MOSCA_RPE_ROT_DEG"] = float(mosca[2])

    if args.output_md is not None:
        md = [
            f"# Replica COLMAP vs GT Evaluation",
            "",
            f"- Scene: `{scene_dir}`",
            f"- Sparse: `{sparse_dir}`",
            f"- Matched frames: `{N}`",
            "",
            "```text",
            (
                f"RPE-R:  {result['RPE_R']} "
                f"RPE-T:  {result['RPE_T']} "
                f"ATE-R:  {result['ATE_R']} "
                f"ATE-T:  {result['ATE_T']}"
            ),
            "",
            "============================================================",
            "MoSca-style eval (evo)",
            "============================================================",
            f"ATE ↓:        {result.get('MOSCA_ATE_M', 'N/A')} m",
            f"RPE trans ↓:  {result.get('MOSCA_RPE_TRANS_M', 'N/A')} m",
            f"RPE rot ↓:    {result.get('MOSCA_RPE_ROT_DEG', 'N/A')}°",
            "============================================================",
            "```",
            "",
        ]
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text("\n".join(md), encoding="utf-8")
        print(f"Wrote markdown report: {args.output_md}")

    if args.json:
        print("RESULT_JSON:", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
