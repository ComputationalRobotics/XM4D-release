import os

import numpy as np
import pyceres
import pycolmap
import pycolmap.cost_functions

import utils.dataset_config as dataset_config


class TranslationDistancePriorCost(pyceres.CostFunction):
    """Penalize deviation from a target translation distance.

    Residual:
        sqrt(weight) * (||t_j - t_i|| - target_distance)

    This is intentionally weight-only: there is no extra scale normalization.
    """

    def __init__(self, target_distance, weight):
        super().__init__()
        self.target_distance = float(target_distance)
        self.sqrt_w = float(np.sqrt(max(0.0, weight)))
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([3, 3])

    def Evaluate(self, parameters, residuals, jacobians):
        ti = np.asarray(parameters[0], dtype=np.float64).reshape(3)
        tj = np.asarray(parameters[1], dtype=np.float64).reshape(3)

        diff = tj - ti
        diff_norm = float(np.linalg.norm(diff))
        residuals[0] = self.sqrt_w * (diff_norm - self.target_distance)

        if jacobians is not None:
            grad = np.zeros(3, dtype=np.float64)
            if diff_norm > 1e-12:
                grad = self.sqrt_w * diff / diff_norm
            if jacobians[0] is not None:
                jacobians[0][:] = (-grad).reshape(-1)
            if jacobians[1] is not None:
                jacobians[1][:] = grad.reshape(-1)
        return True


class SecondDifferenceTranslationCost(pyceres.CostFunction):
    """Penalize second-order finite difference on camera translations.

    Residual:
        sqrt(weight) * (t_{k+1} - 2 t_k + t_{k-1})
    """

    def __init__(self, weight):
        super().__init__()
        self.sqrt_w = float(np.sqrt(max(0.0, weight)))
        self.set_num_residuals(3)
        self.set_parameter_block_sizes([3, 3, 3])

    def Evaluate(self, parameters, residuals, jacobians):
        tm1 = np.asarray(parameters[0], dtype=np.float64).reshape(3)
        t0 = np.asarray(parameters[1], dtype=np.float64).reshape(3)
        tp1 = np.asarray(parameters[2], dtype=np.float64).reshape(3)

        r = self.sqrt_w * (tp1 - 2.0 * t0 + tm1)
        residuals[:] = r

        if jacobians is not None:
            eye = np.eye(3, dtype=np.float64)
            if jacobians[0] is not None:
                jacobians[0][:] = (self.sqrt_w * eye).reshape(-1)
            if jacobians[1] is not None:
                jacobians[1][:] = (-2.0 * self.sqrt_w * eye).reshape(-1)
            if jacobians[2] is not None:
                jacobians[2][:] = (self.sqrt_w * eye).reshape(-1)
        return True


def XM_Ceres_interface(
    edges,
    landmarks2D,
    R_XM,
    t_XM,
    p_XM,
    only_landmarks=False,
    penalty_t_rel=0.0,
    neighbor_window=1,
    penalty_t_acc=0.0,
    use_regularization=True,
    regularization_threshold=0.1,
    optimize_intrinsics=False,
    intrinsic_scale_min=0.25,
    intrinsic_scale_max=4.0,
    loss="trivial",
    loss_scale=None,
):
    edges = np.asarray(edges, dtype=np.int64)
    landmarks2D = np.asarray(landmarks2D, dtype=np.float64)
    R_XM = np.asarray(R_XM, dtype=np.float64)
    t_XM = np.asarray(t_XM, dtype=np.float64)
    p_XM = np.asarray(p_XM, dtype=np.float64)

    N = int(np.max(edges[:, 0]))
    M = int(np.max(edges[:, 1]))

    cam = pycolmap.Camera(
        model="SIMPLE_PINHOLE",
        width=1,
        height=1,
        params=np.array([1.0, 0.0, 0.0]),
        camera_id=0,
    )

    # Initial guess from XM.
    quat0 = []
    t0 = []
    points0 = []
    for i in range(N):
        Ri = pycolmap.Rotation3d(R_XM[:, 3 * i : 3 * i + 3].T)
        quat0.append(Ri.quat)
        t0.append(-R_XM[:, 3 * i : 3 * i + 3].T @ t_XM[:, i])

    for i in range(M):
        points0.append(p_XM[:, i])

    options = pyceres.SolverOptions()
    # Linear solver: DENSE_SCHUR for short videos with many landmarks per frame (<= 100 frames, >= 60 landmarks per
    # frame), ITERATIVE_SCHUR otherwise (long videos, where the dense reduced camera system grows with frames^2, and
    # few landmarks per frame). XM_CERES_LINEAR_SOLVER forces one.
    linear_solver = os.environ.get("XM_CERES_LINEAR_SOLVER") or (
        "DENSE_SCHUR" if N <= 100 and M >= 60 * N else "ITERATIVE_SCHUR"
    )
    print(f"[Ceres] linear solver {linear_solver} ({N} frames, {M} landmarks)")
    options.linear_solver_type = getattr(pyceres.LinearSolverType, linear_solver)
    options.preconditioner_type = pyceres.PreconditionerType.SCHUR_JACOBI
    options.use_nonmonotonic_steps = True
    options.minimizer_progress_to_stdout = True
    options.num_threads = int(os.environ.get("XM_CERES_NUM_THREADS", "100"))
    options.max_num_iterations = 1000
    options.eta = 1e-2
    # config.json values, env-overridable. Defaults keep Ceres' convergence tolerance (1e-6)
    # and a 60s cap.
    options.function_tolerance = dataset_config.get_float(
        "CERES_FUNCTION_TOLERANCE", 1e-6, env="CERES_FUNCTION_TOLERANCE"
    )
    options.max_solver_time_in_seconds = dataset_config.get_float(
        "CERES_MAX_SOLVE_TIME", 60.0, env="CERES_MAX_SOLVE_TIME"
    )

    def add_translation_regularization(prob, t_blocks, t_ref):
        # 1. Anchor the first and last frame distance:
        #    ||t_0 - t_end|| = ||t_0_ref - t_end_ref||
        scale = np.linalg.norm(t_ref[-1] - t_ref[0])
        if float(penalty_t_rel) > 0.0 and N >= 2:
            anchor_target = float(
                np.linalg.norm(
                    np.asarray(t_ref[1], dtype=np.float64)
                    - np.asarray(t_ref[0], dtype=np.float64)
                )
            )
            anchor_cost = TranslationDistancePriorCost(anchor_target, float(penalty_t_rel))
            prob.add_residual_block(anchor_cost, None, [t_blocks[0], t_blocks[1]])

        # 2. Relative distance regularization inside a temporal window:
        #    ||t_i - t_j|| matches the XM initialization distance.
        if float(penalty_t_rel) > 0.0 and int(neighbor_window) > 0:
            win = int(neighbor_window)
            for i in range(N):
                for j in range(i + 1, min(N, i + win + 1)):
                    target_distance = float(
                        np.linalg.norm(
                            np.asarray(t_ref[i], dtype=np.float64)
                            - np.asarray(t_ref[j], dtype=np.float64)
                        )
                    )
                    rel_cost = TranslationDistancePriorCost(
                        target_distance,
                        float(penalty_t_rel),
                    )
                    prob.add_residual_block(rel_cost, None, [t_blocks[i], t_blocks[j]])

        # 3. Acceleration regularization:
        #    t_{k+1} - 2 t_k + t_{k-1} -> 0
        if float(penalty_t_acc) > 0.0 and N >= 3:
            for k in range(1, N - 1):
                smooth_cost = SecondDifferenceTranslationCost(float(penalty_t_acc))
                prob.add_residual_block(
                    smooth_cost, None, [t_blocks[k - 1], t_blocks[k], t_blocks[k + 1]]
                )

    # Robust loss on the reprojection residuals; loss_scale is in the normalized landmark units.
    if loss == "trivial" or loss_scale is None:
        if loss != "trivial":
            print(f"[Ceres] no intrinsics for the {loss} loss scale; using least squares")
        make_loss = pyceres.TrivialLoss
    else:
        loss_cls = {"huber": pyceres.HuberLoss, "cauchy": pyceres.CauchyLoss, "softl1": pyceres.SoftLOneLoss}[loss]
        make_loss = lambda: loss_cls(float(loss_scale))
        print(f"[Ceres] {loss} loss, scale {float(loss_scale):.3g} (normalized)")

    def _solve(obs_keep_mask, quat_init, t_init, points_init, use_regularization=False):
        prob = pyceres.Problem()
        loss = make_loss()

        quat = [q.copy() for q in quat_init]
        t = [ti.copy() for ti in t_init]
        points = [pi.copy() for pi in points_init]

        # Observations grouped by camera, in their original order (stable sort = the per-camera np.where order).
        kept = np.flatnonzero(obs_keep_mask)
        kept = kept[np.argsort(edges[kept, 0], kind="stable")]
        bounds = np.searchsorted(edges[kept, 0], np.arange(1, N + 2))
        make_cost = pycolmap.cost_functions.ReprojErrorCost
        add_block = prob.add_residual_block
        cam_model, cam_params = cam.model, cam.params
        for i in range(N):
            index = kept[bounds[i]:bounds[i + 1]]
            if len(index) == 0:
                continue
            qi, ti = quat[i], t[i]
            for xy, pid in zip(landmarks2D[index, :], (edges[index, 1] - 1).tolist()):
                add_block(make_cost(cam_model, xy), loss, [qi, ti, points[pid], cam_params])
            prob.set_manifold(quat[i], pyceres.EigenQuaternionManifold())

        if optimize_intrinsics:
            # The 2D landmarks are already normalized by the (per-frame) intrinsic,
            # so cam.params[0] is a *normalized focal* initialized at 1.0. Letting it
            # vary refines a single shared global focal correction (f_true ~= f0 *
            # cam.params[0]); cx,cy stay pinned at 0 via a subset manifold, and the
            # focal is bounded to keep self-calibration from running away.
            prob.set_manifold(cam.params, pyceres.SubsetManifold(3, [1, 2]))
            prob.set_parameter_lower_bound(cam.params, 0, float(intrinsic_scale_min))
            prob.set_parameter_upper_bound(cam.params, 0, float(intrinsic_scale_max))
        else:
            prob.set_parameter_block_constant(cam.params)

        if only_landmarks:
            for i in range(N):
                prob.set_parameter_block_constant(quat[i])

        # Regularization is only added in the first pass.
        if use_regularization:
            add_translation_regularization(prob, t, t_init)

        print(
            prob.num_parameter_blocks(),
            prob.num_parameters(),
            prob.num_residual_blocks(),
            prob.num_residuals(),
        )
        summary = pyceres.SolverSummary()
        pyceres.solve(options, prob, summary)
        print(summary.FullReport())
        return quat, t, points, summary

    def compute_point_reprojection_stats(quat, t, points, obs_keep_mask):
        point_err_sum = np.zeros(M, dtype=np.float64)
        point_err_cnt = np.zeros(M, dtype=np.int64)

        for k in range(edges.shape[0]):
            if not obs_keep_mask[k]:
                continue

            cam_id = int(edges[k, 0]) - 1
            pid = int(edges[k, 1]) - 1
            if cam_id < 0 or cam_id >= N or pid < 0 or pid >= M:
                continue

            Rcw = pycolmap.Rotation3d(quat[cam_id]).matrix()
            tcw = t[cam_id]
            Xw = points[pid]
            Xc = Rcw @ Xw + tcw
            if Xc[2] <= 1e-12:
                continue

            uv = Xc[:2] / Xc[2]
            e = float(np.linalg.norm(uv - landmarks2D[k]))
            if np.isfinite(e):
                point_err_sum[pid] += e
                point_err_cnt[pid] += 1

        return point_err_sum, point_err_cnt

    # First pass: reprojection only.
    obs_keep = np.ones(edges.shape[0], dtype=bool)
    quat, t, points, summary = _solve(
        obs_keep, quat0, t0, points0, use_regularization=False
    )

    regularization_loss = 0.0
    for i in range(N):
        smooth_cost = SecondDifferenceTranslationCost(float(penalty_t_acc))
        r = np.zeros(3, dtype=np.float64)
        if i >= 1 and i < N - 1:
            tm1 = np.asarray(t[i - 1], dtype=np.float64).reshape(3)
            t1 = np.asarray(t[i], dtype=np.float64).reshape(3)
            tp1 = np.asarray(t[i + 1], dtype=np.float64).reshape(3)
            smooth_cost.Evaluate([tm1, t1, tp1], r, None)
            regularization_loss += float(np.dot(r, r))

    t_scale = np.linalg.norm(t[-1] - t[0]) if N >= 2 else 1.0
    regularization_loss /= max(1, N - 2)
    regularization_loss /= max(1e-12, t_scale * t_scale)
    print(f"[Ceres] Regularization loss after pass 1: {regularization_loss:.6f}")
    
    # The second pass only runs with regularization, so its per-point error statistics are computed only then.
    if regularization_loss > regularization_threshold and use_regularization:
        # Second pass:
        # 1. Measure per-point reprojection error from the first-pass solution.
        # 2. Remove the top 10% worst 3D points.
        # 3. Solve again without any regularization.
        point_err_sum, point_err_cnt = compute_point_reprojection_stats(
            quat, t, points, obs_keep
        )
        valid_pid = point_err_cnt > 0
        num_valid = int(np.sum(valid_pid))
        num_drop = max(1, int(np.ceil(0.10 * num_valid))) if num_valid > 0 else 0

        if num_drop > 0:
            point_err_mean = np.full(M, -np.inf, dtype=np.float64)
            point_err_mean[valid_pid] = point_err_sum[valid_pid] / point_err_cnt[valid_pid]
            worst_pid = np.argsort(point_err_mean)[-num_drop:]
            obs_keep = ~np.isin(edges[:, 1] - 1, worst_pid)
            print(
                f"[Ceres] Removing top {num_drop}/{num_valid} points by reprojection error (~10%) before pass 2."
            )
        else:
            print("[Ceres] No valid points available for reprojection-based filtering before pass 2.")

        quat, t, points, summary = _solve(
            obs_keep, quat0, t0, points0, use_regularization=True
        )

    if optimize_intrinsics:
        print(
            f"[Ceres] optimize_intrinsics: refined normalized focal = "
            f"{float(cam.params[0]):.6f} (i.e. f_true ~= f0 * {float(cam.params[0]):.6f})"
        )

    p_est = np.array(points).T.copy()

    R_est = np.zeros((3, 3 * N))
    for i in range(N):
        R_est[:, 3 * i : 3 * i + 3] = pycolmap.Rotation3d(quat[i]).matrix().T

    t_est = np.zeros((3, N))
    for i in range(N):
        t_est[:, i] = -R_est[:, 3 * i : 3 * i + 3] @ t[i]

    return R_est, t_est, p_est, summary.total_time_in_seconds, summary.final_cost
