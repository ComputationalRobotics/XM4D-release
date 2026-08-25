import numpy as np
import teaserpp_python
from scipy.stats import chi2
from scipy.spatial.transform import Rotation as Rot
import open3d as o3d
from scipy.stats import trim_mean

def average_rotation_from_3x3n(rotations):

    N = len(rotations)

    # find global rotation from quaternion

    quaternions = np.array([Rot.from_matrix(rot).as_quat() for rot in rotations])
    # change sign to all positive for w
    quaternions = np.array([q if q[3] > 0 else -q for q in quaternions])

    # avg_quaternion = np.mean(quaternions, axis=0)
    # avg_quaternion /= np.linalg.norm(avg_quaternion)

    # 初始化众数（从第一个四元数开始）
    mode_quaternion = quaternions[0]
    
    # # 迭代寻找众数
    # for _ in range(10):  # 设定最大迭代次数
    #     distances = np.linalg.norm(quaternions - mode_quaternion, axis=1)  # 计算每个四元数到 mode_quaternion 的欧氏距离
    #     weights = np.exp(-distances**2)  # 根据距离生成权重，高斯加权
    #     mode_quaternion = np.average(quaternions, axis=0, weights=weights)  # 加权平均
    #     mode_quaternion /= np.linalg.norm(mode_quaternion)  # 归一化为单位四元数

    R_avg = Rot.from_quat(mode_quaternion).as_matrix()

    return R_avg

def rad_error(R_est, R_gt):
    cosvalue = (np.trace(R_est @ R_gt.T) - 1.0) / 2.0
    cosvalue = np.clip(cosvalue, -1.0, 1.0)
    return np.arccos(cosvalue)

def _normalize_rotation(R):
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64))
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn

def _skew(v):
    x, y, z = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)

def _pca_axes(points):
    """
    points: 3xN
    Returns eigenvalues/eigenvectors sorted descending.
    """
    X = points - np.mean(points, axis=1, keepdims=True)
    C = X @ X.T / max(points.shape[1] - 1, 1)
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


def _orientation_alignment_targets_w2c(R_est, R_gt):
    """
    R_est, R_gt are assumed world-to-camera rotations.

    We want a global world transform R_align such that:

        C_gt ~= R_align C_est + t

    For w2c camera rotations, the corresponding orientation relation is:

        R_align ~= R_gt_i.T @ R_est_i
    """
    N = int(R_est.shape[1] / 3)
    targets = []
    for i in range(N):
        Re = R_est[:, 3 * i : 3 * i + 3]
        Rg = R_gt[:, 3 * i : 3 * i + 3]
        targets.append(Rg.T @ Re)
    return targets


def _mean_rotation_error(R_align, targets):
    errs = [rad_error(R_align, Rt) for Rt in targets]
    return float(np.mean(errs))


def _refine_rotation_about_axis(R_base, axis_dst, R_est, R_gt):
    """
    Refine R_base by left-multiplying a rotation around axis_dst:

        R_new = Rot(axis_dst, theta) @ R_base

    This preserves the already-aligned degenerate direction, e.g. line direction,
    while choosing theta to minimize camera orientation error.
    """
    axis_dst = np.asarray(axis_dst, dtype=np.float64)
    axis_dst /= max(np.linalg.norm(axis_dst), 1e-12)

    targets = _orientation_alignment_targets_w2c(R_est, R_gt)

    def objective(theta):
        R_delta = Rot.from_rotvec(theta * axis_dst).as_matrix()
        R_new = R_delta @ R_base
        errs = [rad_error(R_new, Rt) for Rt in targets]
        return float(np.mean(np.square(errs)))

    result = minimize_scalar(
        objective,
        bounds=(-np.pi, np.pi),
        method="bounded",
        options={"xatol": 1e-8},
    )

    R_delta = Rot.from_rotvec(float(result.x) * axis_dst).as_matrix()
    R_new = R_delta @ R_base

    return R_new, float(result.x), _mean_rotation_error(R_base, targets), _mean_rotation_error(R_new, targets)


def _degenerate_rotation_fix(
    src_points,
    dst_points,
    R_base,
    R_est,
    R_gt,
    eig_ratio_thresh=1e-3,
):
    """
    src_points, dst_points: 3xN normalized or unnormalized camera centers
    R_base: TEASER rotation mapping src -> dst

    If trajectory is line-like, refine rotation around the aligned line direction.
    If trajectory is plane-like, optionally refine around the plane normal.
    """
    vals_src, axes_src = _pca_axes(src_points)
    vals_dst, axes_dst = _pca_axes(dst_points)

    if vals_src[0] < 1e-12 or vals_dst[0] < 1e-12:
        return R_base, {
            "degenerate": False,
            "reason": "trajectory has near-zero baseline",
            "eig_src": vals_src.tolist(),
            "eig_dst": vals_dst.tolist(),
        }

    src_ratio_2 = vals_src[1] / vals_src[0]
    src_ratio_3 = vals_src[2] / vals_src[0]
    dst_ratio_2 = vals_dst[1] / vals_dst[0]
    dst_ratio_3 = vals_dst[2] / vals_dst[0]

    # Line-like: only first eigenvalue is significant.
    if src_ratio_2 < eig_ratio_thresh and dst_ratio_2 < eig_ratio_thresh:
        axis_src = axes_src[:, 0]  # line direction in source
        axis_dst = axes_dst[:, 0]  # line direction in target
        kind = "line"

    # Plane-like: third eigenvalue is tiny.
    elif src_ratio_3 < eig_ratio_thresh and dst_ratio_3 < eig_ratio_thresh:
        axis_src = axes_src[:, 2]  # plane normal in source
        axis_dst = axes_dst[:, 2]  # plane normal in target
        kind = "plane"

    else:
        return R_base, {
            "degenerate": False,
            "reason": "full-rank enough",
            "eig_src": vals_src.tolist(),
            "eig_dst": vals_dst.tolist(),
        }

    # Make axis sign consistent with the current TEASER rotation.
    if np.dot(R_base @ axis_src, axis_dst) < 0:
        axis_dst = -axis_dst

    R_new, theta, before, after = _refine_rotation_about_axis(
        R_base,
        axis_dst,
        R_est,
        R_gt,
    )

    return R_new, {
        "degenerate": True,
        "kind": kind,
        "theta_rad": theta,
        "theta_deg": float(np.rad2deg(theta)),
        "rot_error_before_rad": before,
        "rot_error_after_rad": after,
        "eig_src": vals_src.tolist(),
        "eig_dst": vals_dst.tolist(),
    }

def _rotation_from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    n = np.linalg.norm(axis)
    if n <= 1e-12 or abs(angle) <= 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = axis / n
    K = _skew(axis)
    return np.eye(3, dtype=np.float64) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)

def _rotation_align_vector_to_vector(src, dst):
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
        axis /= max(np.linalg.norm(axis), 1e-12)
        return _rotation_from_axis_angle(axis, np.pi)
    axis = np.cross(src, dst)
    return _rotation_from_axis_angle(axis, np.arccos(dot))

def _project_to_plane(v, axis):
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return v - axis * np.dot(axis, v)

def _trajectory_svals(points):
    pts = np.asarray(points, dtype=np.float64)
    pts_c = pts - np.mean(pts, axis=1, keepdims=True)
    return np.linalg.svd(pts_c, full_matrices=False)[1]

def _is_degenerate_trajectory(src, dst, eig_ratio_th=1e-3):
    s_src = _trajectory_svals(src)
    s_dst = _trajectory_svals(dst)
    if s_src.size < 3 or s_dst.size < 3:
        return True, s_src, s_dst
    src_rank3 = s_src[2] / max(s_src[0], 1e-12)
    dst_rank3 = s_dst[2] / max(s_dst[0], 1e-12)
    return min(src_rank3, dst_rank3) < eig_ratio_th, s_src, s_dst

def _degenerate_twist_axis(dst, line_ratio_th=1e-2):
    dst_c = np.asarray(dst, dtype=np.float64) - np.mean(dst, axis=1, keepdims=True)
    _, svals, vh = np.linalg.svd(dst_c.T, full_matrices=False)
    if svals[0] <= 1e-12:
        return None, svals, "point"
    if svals.size > 1 and svals[1] / svals[0] < line_ratio_th:
        axis = vh[0]
        mode = "line"
    else:
        axis = vh[-1]
        mode = "rank2"
    return axis / max(np.linalg.norm(axis), 1e-12), svals, mode

from scipy.optimize import minimize_scalar
def _optimize_degenerate_twist_after_teaser(src_scaled, dst, R_est, R_gt, R_teaser):
    """
    src_scaled: 3xN source camera centers after scale has already been applied
    dst:        3xN GT camera centers
    R_est:      3x3N source/solver world-to-camera rotations
    R_gt:       3x3N GT world-to-camera rotations
    R_teaser:   source-world -> GT-world rotation from TEASER

    Returns:
        R_refined, t_refined
    """
    axis_world, svals, mode = _degenerate_twist_axis(dst)
    if axis_world is None:
        t_refined = np.mean(dst - R_teaser @ src_scaled, axis=1, keepdims=True)
        return R_teaser, t_refined

    axis_world = axis_world / max(np.linalg.norm(axis_world), 1e-12)
    N = src_scaled.shape[1]

    def objective(theta):
        R_twist = _rotation_from_axis_angle(axis_world, theta)
        R_candidate = _normalize_rotation(R_twist @ R_teaser)

        errs = []
        for i in range(N):
            Re_w2c = R_est[:, 3 * i:3 * i + 3]
            Rg_w2c = R_gt[:, 3 * i:3 * i + 3]

            # Convert w2c -> c2w.
            Re_c2w = Re_w2c.T
            Rg_c2w = Rg_w2c.T

            # Correct source-world -> GT-world orientation relation:
            # Rg_c2w ~= R_candidate @ Re_c2w
            R_pred_c2w = R_candidate @ Re_c2w

            errs.append(rad_error(R_pred_c2w, Rg_c2w))

        return float(np.mean(np.square(errs)))

    result = minimize_scalar(
        objective,
        bounds=(-np.pi, np.pi),
        method="bounded",
        options={"xatol": 1e-8},
    )

    twist = float(result.x)
    R_twist = _rotation_from_axis_angle(axis_world, twist)
    R_refined = _normalize_rotation(R_twist @ R_teaser)

    # Recompute translation after changing rotation.
    t_refined = np.mean(dst - R_refined @ src_scaled, axis=1, keepdims=True)

    target = R_refined @ src_scaled + t_refined
    trans_err = np.linalg.norm(target - dst, axis=0)

    rot_err = []
    for i in range(N):
        Re_c2w = R_est[:, 3 * i:3 * i + 3].T
        Rg_c2w = R_gt[:, 3 * i:3 * i + 3].T
        R_pred_c2w = R_refined @ Re_c2w
        rot_err.append(rad_error(R_pred_c2w, Rg_c2w))

    print(
        "Degenerate ATE post-TEASER twist: "
        f"mode={mode}, svals={svals}, twist_deg={np.degrees(twist):.3f}, "
        f"mean_t_err={np.mean(trans_err):.6f}, "
        f"mean_rot_err_deg={np.degrees(np.mean(rot_err)):.3f}"
    )

    return R_refined, t_refined

def ATE_TEASER(R_est,t_est,R_gt,t_gt):
    # R_est: 3x3N
    # t_est: 3xN
    # R_gt: 3x3N
    # t_gt: 3xN
    # return: ATE
    N = int(R_est.shape[1]/3)
    assert R_est.shape == R_gt.shape
    assert t_est.shape == t_gt.shape
    assert N == t_est.shape[1]

    # find global transformation
    dof = 3
    MeasurementNoiseStd = 0.1
    epsilon_square = chi2.ppf(0.9999, dof) * (MeasurementNoiseStd ** 2)
    epsilon = np.sqrt(epsilon_square)

    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1
    solver_params.noise_bound = 1
    solver_params.estimate_scaling = False
    solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12

    t_cam_gt = np.zeros((3, N))
    t_cam_est = np.zeros((3, N))
    for i in range(N):
        t_cam_gt[:, i] = R_gt[:, 3 * i:3 * i + 3].T @ (-t_gt[:, i])
        t_cam_est[:, i] = R_est[:, 3 * i:3 * i + 3].T @ (-t_est[:, i])
        
    src = t_cam_est
    dst = t_cam_gt

    is_degenerate, s_src, s_dst = _is_degenerate_trajectory(src, dst)
    if is_degenerate:
        print(f"ATE_TEASER will refine TEASER twist for degenerate trajectory: src={s_src}, dst={s_dst}")

    # estimate scale
    dst_avg = trim_mean(dst, proportiontocut=0.05, axis=1)
    src_avg = trim_mean(src, proportiontocut=0.05, axis=1)
    dst_dis = np.linalg.norm(dst - dst_avg.reshape(3, 1), axis=0)
    src_dis = np.linalg.norm(src - src_avg.reshape(3, 1), axis=0)
    # delete 10% outliers
    index = src_dis < np.percentile(src_dis, 90)
    src = src[:, index]
    dst = dst[:, index]
    dst_avg = np.mean(dst, axis=1)
    src_avg = np.mean(src, axis=1)
    
    scale1 = np.mean(np.linalg.norm(dst - dst_avg.reshape(3, 1), axis=0))
    scale2 = np.mean(np.linalg.norm(src - src_avg.reshape(3, 1), axis=0))

    print(scale1, scale2)

    src = src / np.mean(np.linalg.norm(src - src_avg.reshape(3, 1), axis=0))
    dst = dst / np.mean(np.linalg.norm(dst - dst_avg.reshape(3, 1), axis=0))

    # randomly choose 5000
    if src.shape[1] > 5000:
        idx = np.random.choice(src.shape[1], 5000, replace=False)
        src = src[:, idx]
        dst = dst[:, idx]

    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    solver.solve(src, dst)
    solution = solver.getSolution()

    src = t_cam_est
    dst = t_cam_gt

    src = src * scale1 / scale2
    dst = dst 

    R_solution = solution.rotation
    t_solution = scale1 * solution.translation.reshape(3,1)
    if is_degenerate:
        R_solution, t_solution = _optimize_degenerate_twist_after_teaser(
            src, dst, R_est, R_gt, R_solution
        )

        # # If trajectory is degenerate, also align rotation using camera orientations.
        # targets = _orientation_alignment_targets_w2c(R_est, R_gt)
        # if targets:
        #     R_orient = average_rotation_from_3x3n(targets)
        #     err_solution = _mean_rotation_error(R_solution, targets)
        #     err_orient = _mean_rotation_error(R_orient, targets)
        #     if err_orient < err_solution:
        #         R_solution = _normalize_rotation(R_orient)
        #         t_solution = np.mean(dst - R_solution @ src, axis=1, keepdims=True)
        #         print(
        #             "[ATE_TEASER] Using orientation-aligned rotation for degenerate case: "
        #             f"err_solution={err_solution:.6f} err_orient={err_orient:.6f}"
        #         )

    target = R_solution @ src + t_solution
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

    return scale1 / scale2 , R_solution, t_solution

def ATE_TEASER_C2W(R_est,t_est,R_gt,t_gt):
    # R_est: 3x3N
    # t_est: 3xN
    # R_gt: 3x3N
    # t_gt: 3xN
    # return: ATE
    N = int(R_est.shape[1]/3)
    assert R_est.shape == R_gt.shape
    assert t_est.shape == t_gt.shape
    assert N == t_est.shape[1]

    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1
    solver_params.noise_bound = 0.3
    solver_params.estimate_scaling = False
    solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12

    t_cam_gt = np.zeros((3, N))
    t_cam_est = np.zeros((3, N))
    for i in range(N):
        t_cam_gt[:, i] = R_gt[:, 3 * i:3 * i + 3].T @ (-t_gt[:, i])
        t_cam_est[:, i] = t_est[:, i]

    src = t_cam_est
    dst = t_cam_gt

    is_degenerate, s_src, s_dst = _is_degenerate_trajectory(src, dst)
    if is_degenerate:
        print(f"ATE_TEASER_C2W will refine TEASER twist for degenerate trajectory: src={s_src}, dst={s_dst}")

    # estimate scale
    dst_avg = trim_mean(dst, proportiontocut=0.05, axis=1)
    src_avg = trim_mean(src, proportiontocut=0.05, axis=1)
    dst_dis = np.linalg.norm(dst - dst_avg.reshape(3, 1), axis=0)
    src_dis = np.linalg.norm(src - src_avg.reshape(3, 1), axis=0)
    # delete 10% outliers
    index = (src_dis < np.percentile(src_dis, 90)) & (dst_dis < np.percentile(dst_dis, 90))
    src = src[:, index]
    dst = dst[:, index]
    dst_avg = np.mean(dst, axis=1)
    src_avg = np.mean(src, axis=1)
    
    scale1 = np.mean(np.linalg.norm(dst - dst_avg.reshape(3, 1), axis=0))
    scale2 = np.mean(np.linalg.norm(src - src_avg.reshape(3, 1), axis=0))

    src = src / scale2
    dst = dst / scale1
    # randomly choose 5000
    if src.shape[1] > 5000:
        idx = np.random.choice(src.shape[1], 5000, replace=False)
        src = src[:, idx]
        dst = dst[:, idx]

    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    solver.solve(src, dst)
    solution = solver.getSolution()

    src = t_cam_est
    dst = t_cam_gt

    src = src * scale1 / scale2
    dst = dst 

    R_solution = solution.rotation
    t_solution = scale1 * solution.translation.reshape(3,1)
    if is_degenerate:
        R_solution, t_solution = _optimize_degenerate_twist_after_teaser(
            src, dst, R_est, R_gt, R_solution
        )

    target = R_solution @ src + t_solution
    error = np.linalg.norm(target - dst, axis=0)
    print('TEASER error:',np.linalg.norm(error))
    
    # visualization
    # DRAW LINE BETWEEN CORRESPONDING POINTS
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


    return scale1 / scale2 , R_solution, t_solution


def ATE_LEASTSQUARE(R_est,t_est,R_gt,t_gt):
    # R_est: 3x3N
    # t_est: 3xN
    # R_gt: 3x3N
    # t_gt: 3xN
    # return: ATE
    N = int(R_est.shape[1]/3)
    assert R_est.shape == R_gt.shape
    assert t_est.shape == t_gt.shape
    assert N == t_est.shape[1]

    # find global transformation
    # find transformation
    rotations_target = [R_gt[:, 3 * i:3 * (i + 1)] @ R_est[:, 3 * i:3 * (i + 1)].T for i in range(N)]
    R = average_rotation_from_3x3n(rotations_target)
    rotation_error = [np.linalg.norm(R @ R_est[:, 3 * i:3 * (i + 1)] - R_gt[:, 3 * i:3 * (i + 1)]) for i in range(N)]
    print('rotation error:',np.mean(rotation_error))

    target = R @ t_est 
    target_avg = np.mean(target, axis=1)
    target = target - target_avg.reshape(3, 1)


    # find scale
    t_gt_avg = np.mean(t_gt, axis=1)
    cov_t_gt = np.mean(np.linalg.norm(t_gt - t_gt_avg.reshape(3, 1), axis=0))
    cov_t_est = np.mean(np.linalg.norm(target, axis=0))
    s = cov_t_gt / cov_t_est

    target = s * target

    # find translation
    t = t_gt - target
    t_avg = np.mean(t, axis=1)

    # (R * t_est - target_avg) * s + t_avg
    return s, R, t_avg.reshape(3, 1) - target_avg.reshape(3 , 1) * s

if __name__ == "__main__":
    N = 5
    
