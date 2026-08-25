import numpy as np
import utils.definitions
from pathlib import Path
import open3d as o3d
import trimesh
from utils.sam3d_utils import find_radial_correspondences, lift_depth_points, load_depth_and_intrinsics, load_mask_for_frame

def _load_teaser_helpers():
    from sys import path
    path.append("/home/fenchel/4DReconstruction/deps/TEASER-plusplus/examples/teaser_python_fpfh_icp")
    from helpers import pcd2xyz, get_teaser_solver, Rt2T
    return pcd2xyz, get_teaser_solver, Rt2T

def _apply_transform(verts_np, T):
    R = T[:3, :3]
    t = T[:3, 3]
    return (verts_np @ R.T) + t

def _make_lines(A_pts, B_pts, corrs_A, corrs_B, max_lines=2000):
    if len(corrs_A) == 0:
        return None
    A = A_pts[corrs_A]
    B = B_pts[corrs_B]
    if A.shape[0] > max_lines:
        idx = np.linspace(0, A.shape[0] - 1, max_lines).astype(np.int64)
        A = A[idx]
        B = B[idx]
    points = np.vstack([A, B]).astype(np.float64)
    n = A.shape[0]
    lines = [[i, i + n] for i in range(n)]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    colors = np.tile(np.array([[1.0, 1.0, 0.0]]), (n, 1))
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

# CONFIG
FRAME_INDEX = 0
MESH_PATH = Path(utils.definitions.output_dir) / "sam3d" / f"sam3d_{FRAME_INDEX:05d}.ply"
VOXEL = 0.02
NOISE_BOUND = 0.02
ICP_MAX_ITER = 500

if not MESH_PATH.exists():
    raise FileNotFoundError(f"Missing mesh: {MESH_PATH}")

depth_map, intrinsics = load_depth_and_intrinsics(frame_idx=FRAME_INDEX)
mask = None

mesh = trimesh.load_mesh(str(MESH_PATH), process=False)
verts_scene = np.asarray(mesh.vertices, dtype=np.float64)

depth_pts = lift_depth_points(depth_map, intrinsics, mask=mask, stride=4)
if depth_pts.shape[0] == 0:
    raise RuntimeError("No valid depth points after lifting")

A_pcd_raw = o3d.geometry.PointCloud()
A_pcd_raw.points = o3d.utility.Vector3dVector(verts_scene)
A_pcd_raw.paint_uniform_color([0.2, 0.9, 0.2])
B_pcd_raw = o3d.geometry.PointCloud()
B_pcd_raw.points = o3d.utility.Vector3dVector(depth_pts.astype(np.float64))
B_pcd_raw.paint_uniform_color([0.2, 0.4, 1.0])

A_pcd = A_pcd_raw.voxel_down_sample(voxel_size=VOXEL)
B_pcd = B_pcd_raw.voxel_down_sample(voxel_size=VOXEL)

pcd2xyz, get_teaser_solver, Rt2T = _load_teaser_helpers()
A_xyz = pcd2xyz(A_pcd)
B_xyz = pcd2xyz(B_pcd)

corrs_A, corrs_B = find_radial_correspondences(np.asarray(A_pcd.points), np.asarray(B_pcd.points))
if len(corrs_A) < 3:
    raise RuntimeError("Too few correspondences for TEASER")

A_corr = A_xyz[:, corrs_A]
B_corr = B_xyz[:, corrs_B]

line_set = _make_lines(np.asarray(A_pcd.points), np.asarray(B_pcd.points), corrs_A, corrs_B)

print("[1] Raw point clouds + correspondences")
if line_set is not None:
    o3d.visualization.draw_geometries([A_pcd, B_pcd, line_set])
else:
    o3d.visualization.draw_geometries([A_pcd, B_pcd])

teaser_solver = get_teaser_solver(NOISE_BOUND)
teaser_solver.solve(A_corr, B_corr)
solution = teaser_solver.getSolution()
T_teaser = Rt2T(solution.rotation, solution.translation)

A_teaser = A_pcd.transform(T_teaser)
print("[2] After TEASER")
if line_set is not None:
    o3d.visualization.draw_geometries([A_teaser, B_pcd, line_set])
else:
    o3d.visualization.draw_geometries([A_teaser, B_pcd])

icp_sol = o3d.pipelines.registration.registration_icp(
    A_teaser,
    B_pcd,
    NOISE_BOUND,
    np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=ICP_MAX_ITER),
)
T_icp = icp_sol.transformation
A_icp = A_teaser.transform(T_icp)
print("[3] After ICP")
o3d.visualization.draw_geometries([A_icp, B_pcd])
print("ICP fitness:", icp_sol.fitness)
print("ICP rmse:", icp_sol.inlier_rmse)
