#!/usr/bin/env python3
"""Visualize solved vs GT cameras and static scene for Syn4D with Open3D."""

from __future__ import annotations

import numpy as np
import open3d as o3d
from scipy.stats import trim_mean

import utils.definitions
from utils.readgt_colmap import load_colmap_gt
from pathlib import Path
try:
    import teaserpp_python
except ImportError:
    teaserpp_python = None


ROOT = utils.definitions.rootpath
COLMAP_ROOT = ROOT / "colmap"
SCENE_PLY = ROOT / "scene_pointcloud_world.ply"
SOLVE_NPZ = ROOT / "outputs" / "result_XM.npz"
GT_OBJECT_DIR = ROOT / "object_pointcloud_world"

GT_STATIC_VOXEL_SIZE = 0.01
SOLVED_STATIC_VOXEL_SIZE = 0.01
CAMERA_AXIS_SCALE = 0.08


def _camera_center_world_from_w2c(R_wc: np.ndarray, t_wc: np.ndarray) -> np.ndarray:
    R_wc = np.asarray(R_wc, dtype=np.float64).reshape(3, 3)
    t_wc = np.asarray(t_wc, dtype=np.float64).reshape(3)
    return -R_wc.T @ t_wc


def _camera_pose_world_from_w2c(R_wc: np.ndarray, t_wc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(R_wc, dtype=np.float64).T, _camera_center_world_from_w2c(R_wc, t_wc)


def _camera_axes_lines(cam_center: np.ndarray, cam_R_cw: np.ndarray, axis_len: float):
    cam_center = np.asarray(cam_center, dtype=np.float64).reshape(3)
    cam_R_cw = np.asarray(cam_R_cw, dtype=np.float64).reshape(3, 3)
    axes = cam_R_cw @ (axis_len * np.eye(3, dtype=np.float64))
    pts = np.vstack([cam_center, cam_center + axes.T])
    lines = np.array([[0, 1], [0, 2], [0, 3]], dtype=np.int32)
    colors = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return pts, lines, colors


def _make_camera_trajectory_lines(centers: np.ndarray, color: tuple[float, float, float]) -> o3d.geometry.LineSet:
    ls = o3d.geometry.LineSet()
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape[0] == 0:
        return ls
    ls.points = o3d.utility.Vector3dVector(centers)
    if centers.shape[0] >= 2:
        lines = np.array([[i, i + 1] for i in range(centers.shape[0] - 1)], dtype=np.int32)
        ls.lines = o3d.utility.Vector2iVector(lines)
        ls.paint_uniform_color(color)
    return ls


def _normalize_rotation(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64))
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _rotation_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    n = np.linalg.norm(axis)
    if n <= 1e-12 or abs(angle) <= 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = axis / n
    K = _skew(axis)
    return np.eye(3, dtype=np.float64) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _rotation_align_vector_to_vector(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = np.asarray(src, dtype=np.float64).reshape(3)
    dst = np.asarray(dst, dtype=np.float64).reshape(3)
    src_n = np.linalg.norm(src)
    dst_n = np.linalg.norm(dst)
    if src_n <= 1e-12 or dst_n <= 1e-12:
        return np.eye(3, dtype=np.float64)
    src = src / src_n
    dst = dst / dst_n
    dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    if dot > 1.0 - 1e-10:
        return np.eye(3, dtype=np.float64)
    if dot < -1.0 + 1e-10:
        basis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(src[0]) > 0.9:
            basis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = np.cross(src, basis)
        axis /= np.linalg.norm(axis)
        return _rotation_from_axis_angle(axis, np.pi)
    axis = np.cross(src, dst)
    angle = np.arccos(dot)
    return _rotation_from_axis_angle(axis, angle)


def _principal_line_direction(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    center = np.mean(pts, axis=0)
    pts_c = pts - center.reshape(1, 3)
    _, svals, vh = np.linalg.svd(pts_c, full_matrices=False)
    direction = vh[0]
    return center, direction / np.linalg.norm(direction)


def _line_like(points: np.ndarray, ratio_thresh: float = 1e-3) -> bool:
    pts = np.asarray(points, dtype=np.float64)
    pts_c = pts - np.mean(pts, axis=0, keepdims=True)
    _, svals, _ = np.linalg.svd(pts_c, full_matrices=False)
    if svals.size < 2 or svals[0] <= 1e-12:
        return True
    return float(svals[1] / svals[0]) < ratio_thresh


def _project_to_plane(v: np.ndarray, axis: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return v - axis * np.dot(axis, v)


def _line_pose_align(
    solved_cam_centers: np.ndarray,
    solved_cam_rot: np.ndarray,
    gt_cam_centers: np.ndarray,
    gt_cam_rot: np.ndarray,
):
    valid = np.isfinite(solved_cam_centers).all(axis=1) & np.isfinite(gt_cam_centers).all(axis=1)
    if int(np.sum(valid)) < 3:
        print("[Align] Not enough valid poses for line fallback.")
        return None

    src_pts = solved_cam_centers[valid]
    dst_pts = gt_cam_centers[valid]
    src_center, src_dir = _principal_line_direction(src_pts)
    dst_center, dst_dir = _principal_line_direction(dst_pts)

    if np.dot(src_dir, dst_dir) < 0.0:
        dst_dir = -dst_dir

    src_scale = np.mean(np.linalg.norm(src_pts - src_center.reshape(1, 3), axis=1))
    dst_scale = np.mean(np.linalg.norm(dst_pts - dst_center.reshape(1, 3), axis=1))
    if src_scale <= 1e-12 or dst_scale <= 1e-12:
        print("[Align] Degenerate line scales for fallback.")
        return None
    scale = float(dst_scale / src_scale)

    R_line = _rotation_align_vector_to_vector(src_dir, dst_dir)
    axis_world = dst_dir / np.linalg.norm(dst_dir)

    sum_cross = 0.0
    sum_dot = 0.0
    basis_axes = (0, 1, 2)
    for Re, Rg in zip(solved_cam_rot[valid], gt_cam_rot[valid]):
        Re0 = R_line @ Re
        for k in basis_axes:
            ve = _project_to_plane(Re0[:, k], axis_world)
            vg = _project_to_plane(Rg[:, k], axis_world)
            ne = np.linalg.norm(ve)
            ng = np.linalg.norm(vg)
            if ne <= 1e-8 or ng <= 1e-8:
                continue
            ve /= ne
            vg /= ng
            sum_dot += float(np.dot(ve, vg))
            sum_cross += float(np.dot(axis_world, np.cross(ve, vg)))

    twist = np.arctan2(sum_cross, sum_dot) if abs(sum_cross) > 0.0 or abs(sum_dot) > 0.0 else 0.0
    R_twist = _rotation_from_axis_angle(axis_world, twist)
    R_align = _normalize_rotation(R_twist @ R_line)
    t_align = dst_center - scale * (R_align @ src_center)

    solved_cam_centers_aligned = _apply_similarity_points(solved_cam_centers, scale, R_align, t_align)
    solved_cam_rot_aligned = np.einsum("ij,tjk->tik", R_align, solved_cam_rot)
    err = np.linalg.norm(solved_cam_centers_aligned[valid] - gt_cam_centers[valid], axis=1)
    print(
        f"[Align] line fallback: frames={int(np.sum(valid))} scale={scale:.6f} "
        f"twist_deg={np.degrees(twist):.3f} mean_t_err={np.mean(err):.6f}"
    )
    return scale, R_align, t_align, solved_cam_centers_aligned, solved_cam_rot_aligned


def _rigid_align_cols(src_pts: np.ndarray, dst_pts: np.ndarray):
    src_mean = np.mean(src_pts, axis=1, keepdims=True)
    dst_mean = np.mean(dst_pts, axis=1, keepdims=True)
    src_centered = src_pts - src_mean
    dst_centered = dst_pts - dst_mean

    H = dst_centered @ src_centered.T
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    t = dst_mean - R @ src_mean
    return R, t


def _teaser_align_camera_centers(src_pts: np.ndarray, dst_pts: np.ndarray):
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"Expected matching Nx3 arrays, got {src.shape} and {dst.shape}")
    if src.shape[0] < 3:
        raise ValueError("Need at least 3 points for TEASER alignment")

    src = src.T.copy()
    dst = dst.T.copy()
    finite_mask = np.isfinite(src).all(axis=0) & np.isfinite(dst).all(axis=0)
    src = src[:, finite_mask]
    dst = dst[:, finite_mask]
    if src.shape[1] < 3:
        raise RuntimeError("Too few finite correspondences for TEASER alignment.")

    dst_avg = trim_mean(dst, proportiontocut=0.05, axis=1)
    src_avg = trim_mean(src, proportiontocut=0.05, axis=1)
    dst_dis = np.linalg.norm(dst - dst_avg.reshape(3, 1), axis=0)
    src_dis = np.linalg.norm(src - src_avg.reshape(3, 1), axis=0)
    keep = (src_dis < np.percentile(src_dis, 90)) & (dst_dis < np.percentile(dst_dis, 90))
    if np.count_nonzero(keep) >= 3:
        src = src[:, keep]
        dst = dst[:, keep]

    dst_avg = np.mean(dst, axis=1)
    src_avg = np.mean(src, axis=1)
    scale_dst = np.mean(np.linalg.norm(dst - dst_avg.reshape(3, 1), axis=0))
    scale_src = np.mean(np.linalg.norm(src - src_avg.reshape(3, 1), axis=0))
    scale = float(scale_dst / scale_src)
    print(
        f"[Align] TEASER prep: pairs={src.shape[1]} scale_dst={scale_dst:.6f} "
        f"scale_src={scale_src:.6f} ratio={scale:.6f}"
    )

    if (
        teaserpp_python is None
        or not np.isfinite(scale_dst)
        or not np.isfinite(scale_src)
        or scale_dst <= 1e-12
        or scale_src <= 1e-12
        or src.shape[1] < 3
    ):
        if teaserpp_python is None:
            print("[Align] teaserpp_python not available; using rigid fallback after scale normalization.")
        else:
            print("[Align] Degenerate TEASER inputs; using rigid fallback after scale normalization.")
        scaled_src = src * scale
        R_fallback, t_fallback = _rigid_align_cols(scaled_src, dst)
        return scale, R_fallback, t_fallback.reshape(3)

    src_norm = src / scale_src
    dst_norm = dst / scale_dst
    if not np.isfinite(src_norm).all() or not np.isfinite(dst_norm).all():
        print("[Align] Non-finite normalized inputs; using rigid fallback after scale normalization.")
        scaled_src = src * scale
        R_fallback, t_fallback = _rigid_align_cols(scaled_src, dst)
        return scale, R_fallback, t_fallback.reshape(3)

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
    translation = scale_dst * solution.translation.reshape(3)

    target = solution.rotation @ (src * scale) + translation.reshape(3, 1)
    error = np.linalg.norm(target - dst, axis=0)
    print(f"[Align] TEASER error norm={np.linalg.norm(error):.6f}, mean={np.mean(error):.6f}")
    return scale, solution.rotation, translation


def _mosca_align_with_evo(
    solved_cam_centers: np.ndarray,
    solved_cam_rot: np.ndarray,
    gt_cam_centers: np.ndarray,
    gt_cam_rot: np.ndarray,
):
    try:
        import importlib
        from scipy.spatial.transform import Rotation

        PoseTrajectory3D = importlib.import_module("evo.core.trajectory").PoseTrajectory3D
    except Exception as e:
        print(f"[Align] evo import failed: {e}")
        return None

    T = min(
        solved_cam_centers.shape[0],
        solved_cam_rot.shape[0],
        gt_cam_centers.shape[0],
        gt_cam_rot.shape[0],
    )
    valid = (
        np.isfinite(solved_cam_centers[:T]).all(axis=1)
        & np.isfinite(gt_cam_centers[:T]).all(axis=1)
    )
    if int(np.sum(valid)) < 3:
        print("[Align] Not enough valid poses for evo alignment.")
        return None

    est_xyz = solved_cam_centers[:T][valid]
    ref_xyz = gt_cam_centers[:T][valid]
    est_quat = []
    ref_quat = []
    for Re, Rg in zip(solved_cam_rot[:T][valid], gt_cam_rot[:T][valid]):
        est_xyzw = Rotation.from_matrix(Re).as_quat()
        ref_xyzw = Rotation.from_matrix(Rg).as_quat()
        est_quat.append(np.array([est_xyzw[3], est_xyzw[0], est_xyzw[1], est_xyzw[2]], dtype=np.float64))
        ref_quat.append(np.array([ref_xyzw[3], ref_xyzw[0], ref_xyzw[1], ref_xyzw[2]], dtype=np.float64))

    timestamps = np.arange(est_xyz.shape[0], dtype=np.float64)
    if _line_like(est_xyz) or _line_like(ref_xyz):
        print("[Align] Camera trajectory is line-like; using line-aware pose fallback.")
        return _line_pose_align(
            solved_cam_centers=solved_cam_centers[:T],
            solved_cam_rot=solved_cam_rot[:T],
            gt_cam_centers=gt_cam_centers[:T],
            gt_cam_rot=gt_cam_rot[:T],
        )

    traj_ref = PoseTrajectory3D(
        positions_xyz=ref_xyz,
        orientations_quat_wxyz=np.asarray(ref_quat, dtype=np.float64),
        timestamps=timestamps,
    )
    traj_est = PoseTrajectory3D(
        positions_xyz=est_xyz,
        orientations_quat_wxyz=np.asarray(est_quat, dtype=np.float64),
        timestamps=timestamps,
    )

    try:
        align_result = traj_est.align(traj_ref, correct_scale=True)
    except Exception as e:
        msg = str(e).lower()
        print(f"[Align] evo align failed: {e}")
        if "singular" in msg or "covariance" in msg:
            print("[Align] Falling back to line-aware pose alignment.")
            return _line_pose_align(
                solved_cam_centers=solved_cam_centers[:T],
                solved_cam_rot=solved_cam_rot[:T],
                gt_cam_centers=gt_cam_centers[:T],
                gt_cam_rot=gt_cam_rot[:T],
            )
        return None
    if align_result is None:
        print("[Align] evo align returned no transform.")
        return None

    if len(align_result) == 3:
        R_align, t_align, scale = align_result
    else:
        raise RuntimeError(f"Unexpected evo align return value: {align_result}")

    R_align = np.asarray(R_align, dtype=np.float64).reshape(3, 3)
    t_align = np.asarray(t_align, dtype=np.float64).reshape(3)
    scale = float(scale)

    solved_cam_centers_aligned = _apply_similarity_points(solved_cam_centers, scale, R_align, t_align)
    solved_cam_rot_aligned = np.einsum("ij,tjk->tik", R_align, solved_cam_rot)

    errors = np.linalg.norm(solved_cam_centers_aligned[:T][valid] - gt_cam_centers[:T][valid], axis=1)
    print(
        f"[Align] evo pose alignment: frames={int(np.sum(valid))} scale={scale:.6f} "
        f"mean_t_err={np.mean(errors):.6f}"
    )
    return scale, R_align, t_align, solved_cam_centers_aligned, solved_cam_rot_aligned


def _apply_similarity_points(pts: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray):
    pts = np.asarray(pts, dtype=np.float64)
    if pts.size == 0:
        return pts.reshape(-1, 3)
    return scale * (pts @ R.T) + t.reshape(1, 3)


def _load_gt_cameras(colmap_root):
    gt = load_colmap_gt(str(colmap_root))
    gt_items = sorted(gt.items(), key=lambda kv: int(kv[1]["id"]))
    names = []
    centers = []
    rotations = []
    for name, item in gt_items:
        R_cw, c = _camera_pose_world_from_w2c(item["R"], item["t"])
        names.append(name)
        centers.append(c)
        rotations.append(R_cw)
    return names, np.asarray(centers, dtype=np.float64), np.asarray(rotations, dtype=np.float64)


def _load_solved():
    rt = np.load(str(SOLVE_NPZ))
    R_raw = np.asarray(rt["R"], dtype=np.float64)
    t_raw = np.asarray(rt["t"], dtype=np.float64)
    p_raw = np.asarray(rt["p"], dtype=np.float64).T
    s_raw = np.asarray(rt["s"], dtype=np.float64) if "s" in rt else None

    T = t_raw.shape[1]
    cam_centers = t_raw.T.copy()
    cam_rot = np.zeros((T, 3, 3), dtype=np.float64)
    for i in range(T):
        Ri = R_raw[:, 3 * i : 3 * (i + 1)]
        if s_raw is not None and i < len(s_raw) and abs(float(s_raw[i])) > 1e-12:
            Ri = Ri / float(s_raw[i])
        cam_rot[i] = _normalize_rotation(Ri)
    return cam_centers, cam_rot, p_raw


def _load_static_scene(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing static scene cloud: {path}")
    pcd = o3d.io.read_point_cloud(str(path))
    if GT_STATIC_VOXEL_SIZE > 0:
        pcd = pcd.voxel_down_sample(GT_STATIC_VOXEL_SIZE)
    if not pcd.has_colors():
        pcd.paint_uniform_color([0.7, 0.7, 0.7])
    return pcd


def _load_gt_object_cloud_paths(root: str | Path, T_expected: int):
    obj_dir = Path(root) / "object_pointcloud_world"
    if not obj_dir.exists():
        return None
    paths = []
    for i in range(T_expected):
        p = obj_dir / f"frame{i:06d}.ply"
        paths.append(p if p.exists() else None)
    return paths


def _make_point_cloud(points: np.ndarray, color: tuple[float, float, float], voxel_size: float = 0.0):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    pcd.paint_uniform_color(color)
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
        pcd.paint_uniform_color(color)
    return pcd


def main():
    gt_names, gt_cam_centers, gt_cam_rot = _load_gt_cameras(COLMAP_ROOT)
    solved_cam_centers, solved_cam_rot, solved_static = _load_solved()

    T = min(len(gt_names), solved_cam_centers.shape[0], gt_cam_centers.shape[0], solved_cam_rot.shape[0], gt_cam_rot.shape[0])
    gt_names = gt_names[:T]
    gt_cam_centers = gt_cam_centers[:T]
    gt_cam_rot = gt_cam_rot[:T]
    solved_cam_centers = solved_cam_centers[:T]
    solved_cam_rot = solved_cam_rot[:T]

    evo_align = _mosca_align_with_evo(
        solved_cam_centers=solved_cam_centers,
        solved_cam_rot=solved_cam_rot,
        gt_cam_centers=gt_cam_centers,
        gt_cam_rot=gt_cam_rot,
    )
    if evo_align is not None:
        scale, R_align, t_align, solved_cam_centers_aligned, solved_cam_rot_aligned = evo_align
    else:
        valid = np.isfinite(solved_cam_centers).all(axis=1) & np.isfinite(gt_cam_centers).all(axis=1)
        if int(np.sum(valid)) < 3:
            raise RuntimeError("Not enough valid camera centers to align solved and GT trajectories.")
        scale, R_align, t_align = _teaser_align_camera_centers(
            solved_cam_centers[valid], gt_cam_centers[valid]
        )
        print(f"[Align] fallback TEASER alignment: frames={int(np.sum(valid))} scale={scale:.6f}")
        solved_cam_centers_aligned = _apply_similarity_points(solved_cam_centers, scale, R_align, t_align)
        solved_cam_rot_aligned = np.einsum("ij,tjk->tik", R_align, solved_cam_rot)

    solved_static_aligned = _apply_similarity_points(solved_static, scale, R_align, t_align)

    gt_static_pcd = _load_static_scene(SCENE_PLY)
    solved_static_pcd = _make_point_cloud(
        solved_static_aligned, color=(1.0, 0.75, 0.1), voxel_size=SOLVED_STATIC_VOXEL_SIZE
    )

    gt_traj = _make_camera_trajectory_lines(gt_cam_centers, color=(0.05, 0.7, 0.1))
    solved_traj = _make_camera_trajectory_lines(solved_cam_centers_aligned, color=(0.9, 0.2, 0.1))
    gt_object_cloud_paths = _load_gt_object_cloud_paths(ROOT, T)

    state = {"idx": 0}

    gt_cam_pts0, gt_cam_lines0, gt_cam_colors0 = _camera_axes_lines(
        gt_cam_centers[0], gt_cam_rot[0], CAMERA_AXIS_SCALE
    )
    solved_cam_pts0, solved_cam_lines0, solved_cam_colors0 = _camera_axes_lines(
        solved_cam_centers_aligned[0], solved_cam_rot_aligned[0], CAMERA_AXIS_SCALE
    )

    gt_cam_axes = o3d.geometry.LineSet()
    gt_cam_axes.points = o3d.utility.Vector3dVector(gt_cam_pts0)
    gt_cam_axes.lines = o3d.utility.Vector2iVector(gt_cam_lines0)
    gt_cam_axes.colors = o3d.utility.Vector3dVector(gt_cam_colors0)

    solved_cam_axes = o3d.geometry.LineSet()
    solved_cam_axes.points = o3d.utility.Vector3dVector(solved_cam_pts0)
    solved_cam_axes.lines = o3d.utility.Vector2iVector(solved_cam_lines0)
    solved_cam_axes.colors = o3d.utility.Vector3dVector(solved_cam_colors0)

    gt_object_pcd = o3d.geometry.PointCloud()
    gt_object_pcd.paint_uniform_color([0.0, 0.8, 0.0])
    if gt_object_cloud_paths is not None and gt_object_cloud_paths[0] is not None:
        gt_obj0 = o3d.io.read_point_cloud(str(gt_object_cloud_paths[0]))
        gt_object_pcd.points = o3d.utility.Vector3dVector(np.asarray(gt_obj0.points, dtype=np.float64))

    def refresh(vis, frame_id: int):
        gt_cam_pts, gt_cam_lines, gt_cam_colors = _camera_axes_lines(
            gt_cam_centers[frame_id], gt_cam_rot[frame_id], CAMERA_AXIS_SCALE
        )
        gt_cam_axes.points = o3d.utility.Vector3dVector(gt_cam_pts)
        gt_cam_axes.lines = o3d.utility.Vector2iVector(gt_cam_lines)
        gt_cam_axes.colors = o3d.utility.Vector3dVector(gt_cam_colors)

        solved_cam_pts, solved_cam_lines, solved_cam_colors = _camera_axes_lines(
            solved_cam_centers_aligned[frame_id], solved_cam_rot_aligned[frame_id], CAMERA_AXIS_SCALE
        )
        solved_cam_axes.points = o3d.utility.Vector3dVector(solved_cam_pts)
        solved_cam_axes.lines = o3d.utility.Vector2iVector(solved_cam_lines)
        solved_cam_axes.colors = o3d.utility.Vector3dVector(solved_cam_colors)

        gt_obj_pts = np.zeros((0, 3), dtype=np.float64)
        if gt_object_cloud_paths is not None and gt_object_cloud_paths[frame_id] is not None:
            gt_obj = o3d.io.read_point_cloud(str(gt_object_cloud_paths[frame_id]))
            gt_obj_pts = np.asarray(gt_obj.points, dtype=np.float64)
        gt_object_pcd.points = o3d.utility.Vector3dVector(gt_obj_pts)

        vis.update_geometry(gt_cam_axes)
        vis.update_geometry(solved_cam_axes)
        vis.update_geometry(gt_object_pcd)
        print(
            f"[Solved-vs-GT] frame={frame_id} image={gt_names[frame_id]} "
            f"gt_cam={gt_cam_centers[frame_id]} solved_cam={solved_cam_centers_aligned[frame_id]} "
            f"gt_obj={len(gt_obj_pts)}"
        )

    def on_space(vis):
        state["idx"] = (state["idx"] + 1) % T
        refresh(vis, state["idx"])
        return False

    def on_b(vis):
        state["idx"] = (state["idx"] - 1) % T
        refresh(vis, state["idx"])
        return False

    print("Open3D controls: SPACE=next frame, B=prev frame.")
    print("Colors: GT static=gray, solved static=yellow, GT camera traj=green, solved camera traj=red.")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Solved vs GT Syn4D", width=1440, height=900)
    vis.add_geometry(gt_static_pcd)
    vis.add_geometry(solved_static_pcd)
    vis.add_geometry(gt_traj)
    vis.add_geometry(solved_traj)
    vis.add_geometry(gt_cam_axes)
    vis.add_geometry(solved_cam_axes)
    vis.add_geometry(gt_object_pcd)
    opt = vis.get_render_option()
    opt.point_size = 3.0
    vis.register_key_callback(ord(" "), on_space)
    vis.register_key_callback(ord("B"), on_b)
    vis.register_key_callback(ord("b"), on_b)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
