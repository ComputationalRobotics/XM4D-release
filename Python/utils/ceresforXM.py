import os

import numpy as np
import pyceres
import pycolmap
import pycolmap.cost_functions

import utils.dataset_config as dataset_config

# Ceres stops after this many seconds.
CERES_MAX_SOLVE_TIME = 60.0


def XM_Ceres_interface(
    edges,
    landmarks2D,
    R_XM,
    t_XM,
    p_XM,
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
    options.num_threads = 100
    options.max_num_iterations = 1000
    options.eta = 1e-2
    # config.json value; the default keeps Ceres' convergence tolerance (1e-6).
    options.function_tolerance = dataset_config.get_float(
        "CERES_FUNCTION_TOLERANCE", 1e-6
    )
    options.max_solver_time_in_seconds = CERES_MAX_SOLVE_TIME

    # Robust loss on the reprojection residuals; loss_scale is in the normalized landmark units.
    if loss == "trivial" or loss_scale is None:
        if loss != "trivial":
            print(f"[Ceres] no intrinsics for the {loss} loss scale; using least squares")
        make_loss = pyceres.TrivialLoss
    else:
        loss_cls = {"huber": pyceres.HuberLoss, "cauchy": pyceres.CauchyLoss, "softl1": pyceres.SoftLOneLoss}[loss]
        make_loss = lambda: loss_cls(float(loss_scale))
        print(f"[Ceres] {loss} loss, scale {float(loss_scale):.3g} (normalized)")

    def _solve(obs_keep_mask, quat_init, t_init, points_init):
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

        print(
            prob.num_parameter_blocks(),
            prob.num_parameters(),
            prob.num_residual_blocks(),
            prob.num_residuals(),
        )
        summary = pyceres.SolverSummary()
        pyceres.solve(options, prob, summary)
        print(summary.FullReport())
        return quat, t, points

    obs_keep = np.ones(edges.shape[0], dtype=bool)
    quat, t, points = _solve(obs_keep, quat0, t0, points0)

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

    # the refined normalized focal: the K the 2D landmarks were normalized with, times this, is the refined K
    return R_est, t_est, p_est, float(cam.params[0])
