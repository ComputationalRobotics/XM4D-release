"""Run SAM3D inference then align mesh with depth points.

Pipeline:
1) Load image + mask and run SAM3D inference.
2) Chamfer scale search with pruning.
3) TEASER + ICP alignment.
4) Visualize mesh, depth points, and camera frame after updates.
"""
from __future__ import annotations

import glob
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial import cKDTree
from pytorch3d.transforms import quaternion_to_matrix
os.environ.setdefault("LIDRA_SKIP_INIT", "1")
PY_ROOT = Path(__file__).resolve().parent
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform

import utils.definitions
from utils.sam3d_utils import (
    find_radial_correspondences,
    lift_depth_points,
    load_depth_and_intrinsics,
    load_mask_for_frame,
)
from utils.track_tapnet import dilate_dynamic_masks
from utils.viser_vis import block_until_ctrl_c, visualize_alignment, visualize_correspondences, visualize_scene

# CONFIG
FRAME_INDEX = 0            # first frame of the (trimmed) sequence
NUM_COMPONENTS = 1         # run SAM3D on the N largest connected mask components
RESCALE_TO_MASK = True     # after depth alignment, rescale model so its 2D projection fills the mask
RECENTER_TO_MASK = True    # also shift so the projection centers on the mask region
OUTPUT_DIR = Path(utils.definitions.output_dir) / "sam3d"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
POINT_STRIDE = 4
MAX_POINTS = 200000

SCALE_MIN = 0.8
SCALE_MAX = 1.2
SCALE_STEPS = 21
PRUNE_ROUNDS = 3
PRUNE_FRACTION = 0.2

VOXEL = 0.02
NOISE_BOUND = 0.2
ICP_MAX_ITER = 5000
MAX_LINES = 2000
SHOW_VIS = True
SHRINK_MASK_PX = 0
RANDOM_SEED = 100
VISER_HOST = "0.0.0.0"
VISER_PORT = 8080
FG_DEPTH_TOL = 1e-5
MAX_FOREGROUND_VERTS = 200000

def _require_open3d():
    try:
        import open3d as o3d
    except Exception as exc:
        raise RuntimeError("open3d is required for visualization.") from exc
    return o3d


def _require_teaser():
    try:
        import teaserpp_python  # noqa: F401
    except Exception as exc:
        raise RuntimeError("teaserpp_python is required for TEASER alignment.") from exc


def _load_teaser_helpers():
    from sys import path

    path.append("/home/fenchel/4DReconstruction/deps/TEASER-plusplus/examples/teaser_python_fpfh_icp")
    from helpers import pcd2xyz, get_teaser_solver, Rt2T

    return pcd2xyz, get_teaser_solver, Rt2T


def _load_inference():
    root = Path(utils.definitions.rootpath)
    cur = root
    repo_root = None
    for _ in range(6):
        candidate = cur / "deps" / "sam-3d-objects"
        if candidate.exists():
            repo_root = cur
            break
        if cur.parent == cur:
            break
        cur = cur.parent
    if repo_root is None:
        raise FileNotFoundError("Could not locate deps/sam-3d-objects from utils.definitions.rootpath")

    sam3d_root = repo_root / "deps" / "sam-3d-objects"
    inference_path = sam3d_root / "notebook" / "inference.py"
    if not inference_path.exists():
        raise FileNotFoundError(f"Missing {inference_path}")
    spec = importlib.util.spec_from_file_location("sam3d_inference", str(inference_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules["sam3d_inference"] = module
    spec.loader.exec_module(module)
    config_path = sam3d_root / "checkpoints" / "hf" / "pipeline.yaml"
    return module.Inference, config_path


def _load_image_and_mask():
    image_dir = Path(utils.definitions.video_dir)
    mask_dir = Path(utils.definitions.mask_dir)
    image_pattern = utils.definitions.image_pattern
    mask_pattern = utils.definitions.mask_pattern

    image_paths = sorted(glob.glob(str(image_dir / image_pattern)))
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in {image_dir} with pattern {image_pattern}")

    if FRAME_INDEX < 0 or FRAME_INDEX >= len(image_paths):
        raise IndexError(f"FRAME_INDEX={FRAME_INDEX} out of range [0, {len(image_paths) - 1}]")

    image_path = Path(image_paths[FRAME_INDEX])
    mask_paths = sorted(glob.glob(str(mask_dir / mask_pattern))) if mask_dir.exists() else []
    mask_by_stem = {Path(p).stem: p for p in mask_paths}
    mask_path = mask_by_stem.get(image_path.stem, None)
    if mask_path is None and len(mask_paths) == len(image_paths):
        # Masks may use a different naming scheme than images (e.g. SAM2
        # 'frame_00000.png' vs image 'frame000000.png'); fall back to sorted-index
        # pairing, the same convention the tracking/solver pipeline uses.
        mask_path = mask_paths[FRAME_INDEX]
    if mask_path is None:
        raise RuntimeError(f"No mask found for {image_path.name} in {mask_dir}")

    image = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L")) > 0
    return image_path, image, mask


def _sample_points(points: np.ndarray, max_points: int | None) -> np.ndarray:
    if max_points is None or points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(points.shape[0], max_points, replace=False)
    return points[idx]


def _largest_components(mask: np.ndarray, k: int) -> list[np.ndarray]:
    """Return up to k binary masks for the k largest connected components of `mask`."""
    from scipy import ndimage

    binary = np.asarray(mask) > 0
    labels, n = ndimage.label(binary)
    if n == 0:
        return []
    sizes = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, n + 1))
    order = np.argsort(sizes)[::-1][:k]  # largest first
    return [(labels == (int(lab) + 1)) for lab in order]


def _chamfer_distances(src: np.ndarray, tgt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tree_tgt = cKDTree(tgt)
    dist_src, _ = tree_tgt.query(src, k=1)
    tree_src = cKDTree(src)
    dist_tgt, _ = tree_src.query(tgt, k=1)
    return dist_src, dist_tgt


def _prune_by_chamfer(
    src: np.ndarray,
    tgt: np.ndarray,
    prune_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    if src.shape[0] == 0 or tgt.shape[0] == 0:
        return src, tgt, np.zeros(0, dtype=bool), np.zeros(0, dtype=bool), np.inf
    dist_src, dist_tgt = _chamfer_distances(src, tgt)
    if dist_src.size == 0 or dist_tgt.size == 0:
        return src, tgt, np.zeros(0, dtype=bool), np.zeros(0, dtype=bool), np.inf

    keep_src = dist_src <= np.quantile(dist_src, 1.0 - prune_fraction)
    keep_tgt = dist_tgt <= np.quantile(dist_tgt, 1.0 - prune_fraction)
    if not np.any(keep_src) or not np.any(keep_tgt):
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            keep_src,
            keep_tgt,
            np.inf,
        )

    src = src[keep_src]
    tgt = tgt[keep_tgt]
    chamfer = float(dist_src[keep_src].mean() + dist_tgt[keep_tgt].mean())
    return src, tgt, keep_src, keep_tgt, chamfer


def _evaluate_scale(scale: float, src: np.ndarray, tgt: np.ndarray) -> float:
    if src.shape[0] == 0 or tgt.shape[0] == 0:
        return np.inf
    src_scaled = src * scale
    dist_src, dist_tgt = _chamfer_distances(src_scaled, tgt)
    if dist_src.size == 0 or dist_tgt.size == 0:
        return np.inf
    return float(dist_src.mean() + dist_tgt.mean())


def _project_points(points_cam: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    z = points_cam[:, 2].clamp(min=1e-6)
    x = points_cam[:, 0] / z * fx + cx
    y = points_cam[:, 1] / z * fy + cy
    return torch.stack([x, y], dim=-1)


def _estimate_scale_from_depth(points_cam, intrinsics, depth_map, mask=None, max_points=200000):
    if points_cam.numel() == 0:
        return 1.0

    points_cam = points_cam.cpu()
    intrinsics = intrinsics.cpu()
    depth_map = depth_map.cpu()
    if mask is not None:
        mask = np.asarray(mask)

    if points_cam.shape[0] > max_points:
        idx = torch.randperm(points_cam.shape[0])[:max_points]
        points_cam = points_cam[idx]

    H, W = depth_map.shape
    proj = _project_points(points_cam, intrinsics)
    xs = proj[:, 0].round().long()
    ys = proj[:, 1].round().long()
    valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H) & (points_cam[:, 2] > 1e-6)
    if mask is not None:
        mask_t = torch.as_tensor(mask, dtype=torch.uint8)
        xs_v = xs[valid]
        ys_v = ys[valid]
        if xs_v.numel() == 0:
            return 1.0
        mask_valid = mask_t[ys_v, xs_v] > 0
        valid_idx = valid.nonzero(as_tuple=False).squeeze(1)
        valid = valid_idx[mask_valid]
        xs = xs[valid]
        ys = ys[valid]
    else:
        xs = xs[valid]
        ys = ys[valid]
    if xs.numel() < 50:
        return 1.0

    depth_vals = depth_map[ys, xs].cpu()
    z_cam = points_cam[:, 2][valid].cpu()
    valid_depth = (torch.isfinite(depth_vals) & (depth_vals > 0)).cpu()
    if valid_depth.sum() < 50:
        return 1.0

    depth_vals = depth_vals[valid_depth]
    z_cam = z_cam[valid_depth]
    denom = (z_cam * z_cam).sum()
    if denom <= 0:
        return 1.0

    s = (z_cam * depth_vals).sum() / denom
    s = torch.clamp(s, min=0.1, max=10.0)
    return float(s.item())


def _apply_transform(verts_np, T):
    R = T[:3, :3]
    t = T[:3, 3]
    return (verts_np @ R.T) + t


def _rescale_to_mask_projection(verts, intrinsics, comp_mask):
    """Scale the aligned model (about its centroid) so its 2D projection fills the
    component mask's image region, optionally recentering onto the mask.

    The uniform scale is the geometric mean of the per-axis ratios of the mask's and the
    projected model's 2D bounding boxes; recentering moves the projected mean onto the mean
    of the mask pixels. Returns (verts_new, scale).
    """
    K = np.asarray(intrinsics, dtype=np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    verts = np.asarray(verts, dtype=np.float64)
    z = verts[:, 2]
    front = z > 1e-6
    ys, xs = np.where(comp_mask)
    if int(front.sum()) < 10 or xs.size == 0:
        return verts, 1.0

    u = verts[front, 0] / z[front] * fx + cx
    v = verts[front, 1] / z[front] * fy + cy
    proj_w = max(float(u.max() - u.min()), 1e-6)
    proj_h = max(float(v.max() - v.min()), 1e-6)
    mask_w = float(xs.max() - xs.min())
    mask_h = float(ys.max() - ys.min())
    # uniform factor = geometric mean of per-axis bbox ratios (same statistic both sides)
    scale = float(np.sqrt((mask_w / proj_w) * (mask_h / proj_h)))

    centroid = verts.mean(axis=0)  # projection of the object is scaled about this point
    verts_new = centroid + scale * (verts - centroid)

    if RECENTER_TO_MASK:
        # align the projected centre-of-mass to the mask centre-of-mass (better mask
        # overlap than bbox-centre when the mask is asymmetric)
        z2 = verts_new[:, 2]
        f2 = z2 > 1e-6
        u2 = verts_new[f2, 0] / z2[f2] * fx + cx
        v2 = verts_new[f2, 1] / z2[f2] * fy + cy
        cz = float(max(np.median(z2[f2]), 1e-6))
        verts_new[:, 0] += (float(xs.mean()) - float(u2.mean())) * cz / fx
        verts_new[:, 1] += (float(ys.mean()) - float(v2.mean())) * cz / fy

    return verts_new, scale


def _foreground_mask_from_mesh(
    verts_cam: np.ndarray,
    faces: np.ndarray,
    fg_depth_tol: float,
    max_verts: int | None = None,
) -> tuple[np.ndarray, float]:
    if verts_cam.shape[0] == 0:
        return np.zeros((0,), dtype=bool), 0.0

    if max_verts is not None and verts_cam.shape[0] > max_verts:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(verts_cam.shape[0], max_verts, replace=False)
        verts_eval = verts_cam[idx]
    else:
        verts_eval = verts_cam

    mesh_cam = trimesh.Trimesh(vertices=verts_cam, faces=faces, process=False)

    front_mask = verts_eval[:, 2] > 0
    verts_eval = verts_eval[front_mask]
    if verts_eval.shape[0] == 0:
        return np.zeros((verts_cam.shape[0],), dtype=bool), 0.0

    ray_dirs = verts_eval / (np.linalg.norm(verts_eval, axis=1, keepdims=True) + 1e-8)
    ray_origins = np.zeros_like(ray_dirs)

    locations, index_ray, _ = mesh_cam.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_dirs,
        multiple_hits=False,
    )

    hit_dist = np.full(len(ray_dirs), np.nan, dtype=np.float32)
    if locations.shape[0] > 0:
        hit_dist[index_ray] = np.linalg.norm(locations - ray_origins[index_ray], axis=1)

    vertex_depth = np.linalg.norm(verts_eval, axis=1)
    valid = np.isfinite(hit_dist)
    if np.any(valid):
        diff = np.abs(hit_dist[valid] - vertex_depth[valid])
        adaptive_tol = float(np.percentile(diff, 10))
        tol = max(fg_depth_tol, adaptive_tol)
        foreground_eval = valid & (np.abs(hit_dist - vertex_depth) <= tol)
    else:
        adaptive_tol = 0.0
        foreground_eval = np.zeros_like(valid, dtype=bool)

    fg_mask = np.zeros((verts_cam.shape[0],), dtype=bool)
    if max_verts is not None and verts_cam.shape[0] > max_verts:
        fg_mask[idx[front_mask]] = foreground_eval
    else:
        fg_mask[front_mask] = foreground_eval

    return fg_mask, adaptive_tol



def debug_export_aligned_mesh(mesh_save, verts_all_scaled, T_icp, out_mesh_path):
    out_mesh_path = Path(out_mesh_path)

    # 1) Build arrays explicitly
    verts_aligned = _apply_transform(
        verts_all_scaled.astype(np.float64), T_icp
    ).astype(np.float64)

    faces = np.asarray(mesh_save.faces, dtype=np.int64)

    colors = None
    if hasattr(mesh_save.visual, "vertex_colors"):
        vc = np.asarray(mesh_save.visual.vertex_colors)
        if len(vc) == len(verts_aligned):
            colors = vc
        else:
            print(f"[WARN] vertex_colors length mismatch: {len(vc)} vs {len(verts_aligned)}")
            colors = None

    # 2) Hard checks before creating any mesh object
    print("\n========== PRE-EXPORT CHECKS ==========")
    print("verts shape:", verts_aligned.shape, "dtype:", verts_aligned.dtype)
    print("faces shape:", faces.shape, "dtype:", faces.dtype)

    assert verts_aligned.ndim == 2 and verts_aligned.shape[1] == 3, \
        f"verts must be (N,3), got {verts_aligned.shape}"
    assert faces.ndim == 2 and faces.shape[1] == 3, \
        f"faces must be (M,3), got {faces.shape}"

    verts_finite = np.isfinite(verts_aligned).all()
    print("verts finite:", verts_finite)
    assert verts_finite, "NaN/Inf found in transformed vertices"

    if len(faces) > 0:
        fmin = int(faces.min())
        fmax = int(faces.max())
        print("face min index:", fmin)
        print("face max index:", fmax)
        print("vertex count   :", len(verts_aligned))
        assert fmin >= 0, "Negative face index found"
        assert fmax < len(verts_aligned), "Face index out of range"

        degenerate = (
            (faces[:, 0] == faces[:, 1]) |
            (faces[:, 1] == faces[:, 2]) |
            (faces[:, 0] == faces[:, 2])
        )
        print("degenerate face count:", int(degenerate.sum()))
    else:
        print("[WARN] faces array is empty")

    print("vertex min:", verts_aligned.min(axis=0))
    print("vertex max:", verts_aligned.max(axis=0))

    # 3) Export a RAW mesh exactly as-is
    raw_mesh = trimesh.Trimesh(
        vertices=verts_aligned,
        faces=faces,
        vertex_colors=colors,
        process=False,
    )

    raw_path = out_mesh_path.with_name(out_mesh_path.stem + "_raw_ascii.ply")
    raw_mesh.export(raw_path, file_type="ply", encoding="ascii")
    print("wrote raw   :", raw_path)

    # 4) Export a CHECKED mesh with trimesh cleanup/validation enabled
    checked_mesh = trimesh.Trimesh(
        vertices=verts_aligned,
        faces=faces,
        vertex_colors=colors,
        process=True,
        validate=True,
    )

    checked_path = out_mesh_path.with_name(out_mesh_path.stem + "_checked_ascii.ply")
    checked_mesh.export(checked_path, file_type="ply", encoding="ascii")
    print("wrote checked:", checked_path)

    print("\n========== TRIMESH SUMMARY ==========")
    print("raw    vertices:", len(raw_mesh.vertices), "faces:", len(raw_mesh.faces))
    print("checked vertices:", len(checked_mesh.vertices), "faces:", len(checked_mesh.faces))

    # Optional trimesh-side normals test
    try:
        vn = checked_mesh.vertex_normals
        print("trimesh vertex_normals OK:", vn.shape, vn.dtype)
    except Exception as e:
        print("trimesh vertex_normals FAILED:", repr(e))

    return raw_path, checked_path


def _process_component(comp_id, image, comp_mask, inference, depth_map, intrinsics, teaser_helpers):
    """Run SAM3D + depth alignment for a single mask component.

    Returns (verts_aligned [N,3], faces [M,3], colors [N,4], fg_mask [N]) in camera coords.
    """
    o3d, pcd2xyz, get_teaser_solver, Rt2T = teaser_helpers
    tag = f"[comp{comp_id}]"

    output = inference(image, comp_mask, seed=42)
    mesh = output.get("mesh", None)
    mesh = mesh[0] if isinstance(mesh, (list, tuple)) else mesh
    if mesh is None:
        raise RuntimeError(f"{tag} SAM3D output did not contain a mesh")

    vattr = getattr(mesh, "vertex_attrs", None)
    if vattr is None:
        raise RuntimeError(f"{tag} Mesh has no vertex_attrs for colors")
    rgb = vattr[..., :3].detach().cpu().numpy()
    mesh_save = trimesh.Trimesh(
        vertices=torch.as_tensor(mesh.vertices, device="cpu", dtype=torch.float32).numpy(),
        faces=mesh.faces.long().cpu().numpy(),
        vertex_colors=rgb,
        process=False,
    )

    # Depth target for THIS component only (its own dilated mask region).
    mask_for_depth = dilate_dynamic_masks(comp_mask[None, ...], SHRINK_MASK_PX)[0]
    depth_pts_full = lift_depth_points(depth_map, intrinsics, mask=mask_for_depth, stride=POINT_STRIDE)
    if depth_pts_full.shape[0] == 0:
        raise RuntimeError(f"{tag} No valid depth points after lifting")

    verts_local = torch.as_tensor(mesh_save.vertices, device="cpu", dtype=torch.float32)
    if verts_local.ndim == 2 and verts_local.shape[0] == 3 and verts_local.shape[1] != 3:
        verts_local = verts_local.T
    quat_l2c = output["rotation"].detach().cpu()
    trans_l2c = output["translation"].detach().cpu()
    scale_l2c = output["scale"].detach().cpu()
    rotation = quaternion_to_matrix(quat_l2c)
    l2c_transform = compose_transform(scale=scale_l2c, rotation=rotation, translation=trans_l2c)
    verts_scene = l2c_transform.transform_points(verts_local).reshape(-1, 3)

    depth_map_t = torch.as_tensor(depth_map)
    intrinsics_t = torch.as_tensor(intrinsics)
    s_align = _estimate_scale_from_depth(verts_scene, intrinsics_t, depth_map_t, mask=mask_for_depth)
    print(f"{tag} Initial scale from depth: s={s_align:.4f}")
    verts_scene = verts_scene * s_align
    verts_scene[:, 0:2] = -verts_scene[:, 0:2]

    verts_scene_np = verts_scene.detach().cpu().numpy().astype(np.float32)
    mesh_faces = mesh_save.faces.astype(np.int64)
    fg_mask, adaptive_tol = _foreground_mask_from_mesh(
        verts_scene_np, mesh_faces, FG_DEPTH_TOL, max_verts=MAX_FOREGROUND_VERTS,
    )
    print(f"{tag} Foreground mask (adaptive tol={adaptive_tol:.6f}). fg={fg_mask.sum()} / {len(fg_mask)} verts.")

    fg_points = verts_scene_np[fg_mask]
    verts = fg_points if fg_points.shape[0] > 0 else verts_scene_np
    verts_all = verts_scene_np
    verts = _sample_points(verts, MAX_POINTS)
    depth_pts = _sample_points(depth_pts_full, MAX_POINTS)

    if SHOW_VIS:
        visualize_scene(f"{tag} After initial transform/scale", verts, depth_pts, VISER_HOST, VISER_PORT)

    current_scale = 1.0
    src_base = verts
    tgt_work = depth_pts

    for round_idx in range(PRUNE_ROUNDS):
        if src_base.shape[0] == 0 or tgt_work.shape[0] == 0:
            raise RuntimeError(f"{tag} No points left after pruning")

        scales = current_scale * np.linspace(SCALE_MIN, SCALE_MAX, SCALE_STEPS)
        best_scale = None
        best_score = np.inf
        print(f"{tag} Round {round_idx + 1}: scale search {scales[0]:.3f} -> {scales[-1]:.3f} ({SCALE_STEPS} steps)")
        for scale in scales:
            score = _evaluate_scale(scale, src_base, tgt_work)
            if score < best_score:
                best_score = score
                best_scale = scale
        if best_scale is None or not np.isfinite(best_score):
            raise RuntimeError(f"{tag} Failed to find a valid scale")

        current_scale = best_scale
        src_scaled = src_base * current_scale
        src_scaled, tgt_work, keep_src, keep_tgt, chamfer = _prune_by_chamfer(
            src_scaled, tgt_work, PRUNE_FRACTION
        )
        src_base = src_base[keep_src]
        if SHOW_VIS:
            visualize_scene(
                f"{tag} Round {round_idx + 1} - after scale/prune (chamfer={chamfer:.6f})",
                src_scaled, tgt_work, VISER_HOST, VISER_PORT,
            )

    print(f"{tag} Final scale: {current_scale:.6f}")

    verts_scaled = verts * current_scale
    verts_all_scaled = verts_all * current_scale

    A_pcd_raw = o3d.geometry.PointCloud()
    A_pcd_raw.points = o3d.utility.Vector3dVector(verts_scaled.astype(np.float64))
    B_pcd_raw = o3d.geometry.PointCloud()
    B_pcd_raw.points = o3d.utility.Vector3dVector(depth_pts.astype(np.float64))

    A_pcd = A_pcd_raw.voxel_down_sample(voxel_size=VOXEL)
    B_pcd = B_pcd_raw.voxel_down_sample(voxel_size=VOXEL)

    A_xyz = pcd2xyz(A_pcd)
    B_xyz = pcd2xyz(B_pcd)

    corrs_A, corrs_B = find_radial_correspondences(np.asarray(A_pcd.points), np.asarray(B_pcd.points))
    if len(corrs_A) < 3:
        raise RuntimeError(f"{tag} Too few correspondences for TEASER")

    A_corr = A_xyz[:, corrs_A]
    B_corr = B_xyz[:, corrs_B]

    teaser_solver = get_teaser_solver(NOISE_BOUND)
    teaser_solver.solve(A_corr, B_corr)
    solution = teaser_solver.getSolution()
    T_teaser = Rt2T(solution.rotation, solution.translation)

    icp_sol = o3d.pipelines.registration.registration_icp(
        A_pcd, B_pcd, NOISE_BOUND, T_teaser,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=ICP_MAX_ITER),
    )
    T_icp = icp_sol.transformation
    print(f"{tag} ICP fitness={icp_sol.fitness:.4f} rmse={icp_sol.inlier_rmse:.6f}")

    verts_aligned = _apply_transform(verts_all_scaled.astype(np.float64), T_icp)

    if RESCALE_TO_MASK:
        verts_aligned, mask_scale = _rescale_to_mask_projection(verts_aligned, intrinsics, comp_mask)
        print(f"{tag} Rescaled model to fill mask projection: scale={mask_scale:.3f}"
              + (" (+recenter)" if RECENTER_TO_MASK else ""))

    return verts_aligned, np.asarray(mesh_save.faces, dtype=np.int64), np.asarray(mesh_save.visual.vertex_colors), fg_mask


def main() -> None:
    image_path, image, mask = _load_image_and_mask()

    components = _largest_components(mask, NUM_COMPONENTS)
    if len(components) == 0:
        raise RuntimeError("No connected components found in mask")
    if len(components) < NUM_COMPONENTS:
        print(f"[WARN] mask has only {len(components)} component(s); requested {NUM_COMPONENTS}.")
    print(f"Processing {len(components)} mask component(s) with sizes {[int(c.sum()) for c in components]}")

    Inference, config_path = _load_inference()
    inference = Inference(str(config_path), compile=False)

    depth_map, intrinsics = load_depth_and_intrinsics(frame_idx=FRAME_INDEX)

    _require_teaser()
    o3d = _require_open3d()
    pcd2xyz, get_teaser_solver, Rt2T = _load_teaser_helpers()
    teaser_helpers = (o3d, pcd2xyz, get_teaser_solver, Rt2T)

    all_verts, all_faces, all_colors, all_fg = [], [], [], []
    vert_offset = 0
    for comp_id, comp_mask in enumerate(components):
        ys, xs = np.where(comp_mask)
        print(f"[comp{comp_id}] input mask: area={int(comp_mask.sum())} "
              f"bbox=[x {int(xs.min())}..{int(xs.max())}, y {int(ys.min())}..{int(ys.max())}]")
        try:
            verts_c, faces_c, colors_c, fg_c = _process_component(
                comp_id, image, comp_mask, inference, depth_map, intrinsics, teaser_helpers
            )
        except Exception as exc:
            # SAM3D could not lift this object (e.g. degenerate internal pointmap/crop).
            # Fall back to DA3 depth-lifted points for this component so the combined
            # prior still contains it (same camera frame as the SAM3D-aligned output).
            print(f"[WARN] [comp{comp_id}] SAM3D failed ({type(exc).__name__}: {exc}); "
                  f"using depth-lifted points for this component instead.")
            pts = np.asarray(
                lift_depth_points(depth_map, intrinsics, mask=comp_mask, stride=POINT_STRIDE),
                dtype=np.float64,
            )
            if pts.shape[0] == 0:
                print(f"[WARN] [comp{comp_id}] no depth points either; skipping.")
                continue
            verts_c = pts
            faces_c = np.zeros((0, 3), dtype=np.int64)
            colors_c = np.tile(np.array([160, 160, 160, 255], np.uint8), (pts.shape[0], 1))
            fg_c = np.ones((pts.shape[0],), dtype=bool)
            print(f"[comp{comp_id}] depth-lift fallback: {pts.shape[0]} points.")
        all_verts.append(verts_c)
        all_faces.append(faces_c + vert_offset)  # offset faces into the merged vertex array
        all_colors.append(colors_c)
        all_fg.append(fg_c)
        vert_offset += verts_c.shape[0]

    if not all_verts:
        raise RuntimeError("No component produced any points (SAM3D and depth-lift both failed).")

    verts_aligned = np.concatenate(all_verts, axis=0)
    faces_merged = np.concatenate(all_faces, axis=0)
    colors_merged = np.concatenate(all_colors, axis=0)
    fg_mask = np.concatenate(all_fg, axis=0)

    # Combined aligned mesh (both components) -> single ply.
    aligned_mesh = trimesh.Trimesh(
        vertices=verts_aligned,
        faces=faces_merged,
        vertex_colors=colors_merged,
        process=False,
    )
    out_mesh_path = OUTPUT_DIR / f"sam3d_{image_path.stem}_aligned.ply"
    aligned_mesh.export(str(out_mesh_path))
    print(f"Saved combined aligned mesh ({len(components)} components, {verts_aligned.shape[0]} verts):", out_mesh_path)

    # Combined foreground mask (indices into the merged vertex array).
    fg_mask_path = OUTPUT_DIR / f"sam3d_{image_path.stem}_foreground_mask.npz"
    np.savez_compressed(
        fg_mask_path,
        foreground_mask=fg_mask,
        foreground_indices=np.flatnonzero(fg_mask).astype(np.int64),
    )
    print(f"Saved combined foreground mask (fg={int(fg_mask.sum())}/{len(fg_mask)}):", fg_mask_path)

    fg_points = verts_aligned[fg_mask]
    if fg_points.shape[0] > 0:
        fg_pc_path = OUTPUT_DIR / f"sam3d_{image_path.stem}_foreground_points.ply"
        trimesh.PointCloud(fg_points).export(str(fg_pc_path))
        print("Saved combined foreground point cloud:", fg_pc_path)

    # 2D projection of the combined aligned points onto the image.
    max_proj_points = 20000
    proj_pts = verts_aligned.astype(np.float64)
    if proj_pts.shape[0] > max_proj_points:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(proj_pts.shape[0], size=max_proj_points, replace=False)
        proj_pts = proj_pts[idx]
    K = intrinsics.astype(np.float64)
    z = proj_pts[:, 2]
    valid = z > 1e-6
    x = proj_pts[valid, 0] / z[valid]
    y = proj_pts[valid, 1] / z[valid]
    u = x * K[0, 0] + K[0, 2]
    v = y * K[1, 1] + K[1, 2]
    h, w = image.shape[:2]
    in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u = u[in_bounds]
    v = v[in_bounds]

    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.scatter(u, v, s=1, c="cyan", alpha=0.6)
    plt.title("Combined aligned mesh points projected to image")
    plt.axis("off")
    plt.show()

    if SHOW_VIS:
        visualize_scene(
            "Final combined aligned",
            verts_aligned.astype(np.float32),
            lift_depth_points(depth_map, intrinsics, mask=None, stride=POINT_STRIDE).astype(np.float32),
            VISER_HOST,
            VISER_PORT,
        )
        block_until_ctrl_c()


if __name__ == "__main__":
    main()
