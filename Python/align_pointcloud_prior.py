import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

import utils.definitions

from utils.edg import scatter_solved_poses
import open3d as o3d
from utils.readgt_colmap import load_colmap_gt
from utils.readgt_DAVIS import load_davis_gt
from utils.readgt_replica import load_replica_gt
from utils.readgt_sintel import load_sintel_gt
from utils.readgt_TUM import load_tum_gt
from utils.readgt_custom import load_custom_gt

FRAME_INDEX = 0
MAX_OBS_POINTS = 30000
MAX_PRIOR_POINTS = 3000
MAX_COMPONENTS = 3
MIN_COMPONENT_POINTS = 50
MAX_POINTS_PER_COMPONENT = 12000
REGION_MARGIN_PX = 20.0
CLUSTER_EPS = 0.5  # <=0 means auto from nearest-neighbor spacing
CLUSTER_EPS_MULT = 8.0
PRIOR_VISIBILITY_PIXEL_SIZE = 3.0
PRIOR_VISIBILITY_DEPTH_TOL_ABS = 1e-3
PRIOR_VISIBILITY_DEPTH_TOL_REL = 2e-3
VOXEL_SIZE = 0.0  # <=0 means auto from scene bbox
AUTO_VOXEL_DIVISOR = 350.0
MESH_SAMPLE_SURFACE = True
MESH_SAMPLE_SPACING = 0.0  # <=0 means auto from mesh bbox using AUTO_VOXEL_DIVISOR
MESH_MIN_SAMPLE_POINTS = 2000
MESH_MAX_SAMPLE_POINTS = 3000
SCALE_MIN = 0.2
SCALE_MAX = 5.0
SCALE_STEPS = 21
SCALE_ROUNDS = 4
ICP_ITERS = 20
ICP_TOL = 1e-5
RANDOM_SEED = 123
SHOW_PLOT = True
SHOW_OPEN3D = (
    os.environ.get("ENABLE_OPEN3D_WINDOWS", "0").lower() in {"1", "true", "yes", "on"}
    or os.environ.get("ALLOW_OPEN3D_WINDOWS", "0").lower() in {"1", "true", "yes", "on"}
)
ICP_OPEN3D_EVERY = 1
SHOW_VISER = os.environ.get("SHOW_VISER", "0").lower() in {"1", "true", "yes", "on"}
VISER_PORT = int(os.environ.get("VISER_PORT", "8080"))


def _parse_args():
    parser = argparse.ArgumentParser(description="Align object prior point cloud to lifted dynamic points.")
    parser.add_argument("--frame-index", type=int, default=FRAME_INDEX)
    parser.add_argument("--max-components", "--max-component", dest="max_components", type=int, default=MAX_COMPONENTS)
    parser.add_argument(
        "--prior-path",
        type=Path,
        default=None,
        help=(
            "Point-cloud/mesh file or directory. If a directory is passed, "
            "frame%06d.ply is selected from it. Defaults to <dataset>/mesh_camera.ply "
            "for Custom, otherwise <dataset>/object_pointcloud_world."
        ),
    )
    parser.add_argument(
        "--prior-coordinate-space",
        choices=["auto", "world", "camera"],
        default="auto",
        help="Whether prior points are in dataset world coordinates or already in camera coordinates.",
    )
    parser.add_argument(
        "--prior-in-camera-frame",
        action="store_true",
        help="Shortcut for --prior-coordinate-space camera; no world-to-camera transform is applied to the prior.",
    )
    parser.add_argument(
        "--camera-pose-source",
        choices=["auto", "gt", "xm"],
        default="auto",
        help="Camera pose used to convert world prior points into the current camera frame.",
    )
    parser.add_argument("--show-open3d", action=argparse.BooleanOptionalAction, default=SHOW_OPEN3D)
    parser.add_argument("--show-plot", action=argparse.BooleanOptionalAction, default=SHOW_PLOT)
    parser.add_argument("--show-viser", action=argparse.BooleanOptionalAction, default=SHOW_VISER,
                        help="Serve the alignment result (observations + aligned prior) in a viser web viewer.")
    return parser.parse_args()


def _apply_args(args):
    global FRAME_INDEX, MAX_COMPONENTS, SHOW_OPEN3D, SHOW_PLOT, SHOW_VISER
    FRAME_INDEX = int(args.frame_index)
    MAX_COMPONENTS = int(args.max_components)
    SHOW_OPEN3D = bool(args.show_open3d)
    SHOW_PLOT = bool(args.show_plot)
    SHOW_VISER = bool(args.show_viser)


def _sample_points(points, max_points):
    if points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def _finite_points(points):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return points[np.isfinite(points).all(axis=1)]


def _voxel_downsample(points, voxel_size):
    points = _finite_points(points)
    if points.shape[0] == 0 or voxel_size <= 0.0:
        return points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(float(voxel_size))
    return np.asarray(pcd.points, dtype=np.float64)


def _auto_voxel_size(*clouds):
    if VOXEL_SIZE > 0.0:
        return float(VOXEL_SIZE)
    pts = [c for c in clouds if c is not None and len(c) > 0]
    if not pts:
        return 0.0
    all_pts = np.concatenate(pts, axis=0)
    mins = np.min(all_pts, axis=0)
    maxs = np.max(all_pts, axis=0)
    diag = float(np.linalg.norm(maxs - mins))
    if not np.isfinite(diag) or diag <= 1e-12:
        return 0.0
    return diag / float(AUTO_VOXEL_DIVISOR)


def _auto_cluster_eps(points):
    points = _finite_points(points)
    if CLUSTER_EPS > 0.0:
        return float(CLUSTER_EPS)
    if points.shape[0] < 3:
        return 0.0
    sample = _sample_points(points, min(points.shape[0], 5000))
    tree = cKDTree(sample)
    d, _ = tree.query(sample, k=min(2, sample.shape[0]))
    nn = d[:, 1] if d.ndim == 2 and d.shape[1] > 1 else d
    nn = nn[np.isfinite(nn) & (nn > 0.0)]
    if nn.size == 0:
        return 0.0
    return float(np.median(nn) * CLUSTER_EPS_MULT)


def _component_centroid(points):
    points = _finite_points(points)
    if points.shape[0] == 0:
        return np.zeros((3,), dtype=np.float64)
    return np.mean(points, axis=0)


def _merge_components_to_count(components, target_count):
    components = [_finite_points(c) for c in components if _finite_points(c).shape[0] > 0]
    target_count = max(1, int(target_count))
    if len(components) <= target_count:
        return components

    print(
        f"[Cluster] merging {len(components)} prior fragments into {target_count} "
        "spatial components instead of dropping smaller fragments."
    )
    while len(components) > target_count:
        centers = np.stack([_component_centroid(c) for c in components], axis=0)
        best_pair = None
        best_dist = np.inf
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                d = float(np.linalg.norm(centers[i] - centers[j]))
                if d < best_dist:
                    best_dist = d
                    best_pair = (i, j)
        if best_pair is None:
            break
        i, j = best_pair
        merged = np.concatenate([components[i], components[j]], axis=0)
        print(
            f"[Cluster] merge fragments {i}+{j}: dist={best_dist:.6f}, "
            f"sizes=({components[i].shape[0]}, {components[j].shape[0]}) -> {merged.shape[0]}"
        )
        components[i] = merged
        del components[j]

    components.sort(key=lambda x: x.shape[0], reverse=True)
    return components


def _dbscan_components(points, eps):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    labels = np.asarray(
        pcd.cluster_dbscan(eps=float(eps), min_points=int(MIN_COMPONENT_POINTS), print_progress=False),
        dtype=np.int64,
    )
    components = []
    for label in sorted(set(labels.tolist())):
        if label < 0:
            continue
        comp = points[labels == label]
        if comp.shape[0] >= MIN_COMPONENT_POINTS:
            components.append(comp)
    components.sort(key=lambda x: x.shape[0], reverse=True)
    return components, labels


def _choose_prior_components(points, base_eps):
    target_count = max(1, int(MAX_COMPONENTS))
    multipliers = [0.35, 0.45, 0.55, 0.65, 0.8, 1.0, 1.25, 1.6]
    candidates = []
    for mult in multipliers:
        eps = float(base_eps) * float(mult)
        components, labels = _dbscan_components(points, eps)
        coverage = float(sum(c.shape[0] for c in components)) / max(float(points.shape[0]), 1.0)
        sizes = [int(c.shape[0]) for c in components]
        print(
            f"[Cluster] prior eps_try={eps:.6f} mult={mult:.2f} "
            f"components={len(components)} coverage={coverage:.3f} sizes={sizes[:8]}"
        )
        candidates.append((eps, components, labels, coverage))

    enough = [c for c in candidates if len(c[1]) >= target_count and c[3] >= 0.75]
    if enough:
        # Prefer the smallest radius that covers most prior points without collapsing fragments.
        oversegmented = [c for c in enough if len(c[1]) > target_count]
        chosen = oversegmented[0] if oversegmented else enough[0]
    else:
        valid = [c for c in candidates if len(c[1]) >= target_count]
        if valid:
            chosen = max(valid, key=lambda c: (len(c[1]), c[3]))
        else:
            chosen = max(candidates, key=lambda c: c[3])

    eps, components, _, coverage = chosen
    print(
        f"[Cluster] prior selected eps={eps:.6f}, raw_components={len(components)}, "
        f"coverage={coverage:.3f}"
    )
    if not components:
        return [points]
    return _merge_components_to_count(components, target_count)


def _cluster_components(points, name):
    points = _finite_points(points)
    if points.shape[0] == 0:
        return []
    eps = _auto_cluster_eps(points)
    if eps <= 0.0:
        return [points]
    if name == "prior" and MAX_COMPONENTS > 1:
        components = _choose_prior_components(points, eps)
    else:
        components, _ = _dbscan_components(points, eps)
    if not components:
        print(f"[Cluster] {name}: DBSCAN found no components; using full cloud.")
        components = [points]
    elif name != "prior":
        components = _merge_components_to_count(components, MAX_COMPONENTS)
    print(
        f"[Cluster] {name}: eps={eps:.6f}, components={len(components)}, "
        f"sizes={[int(c.shape[0]) for c in components]}"
    )
    for ci, comp in enumerate(components):
        center = _component_centroid(comp)
        span = np.max(comp, axis=0) - np.min(comp, axis=0)
        print(
            f"[Cluster] {name} component={ci} center={np.round(center, 5).tolist()} "
            f"span={np.round(span, 5).tolist()}"
        )
    return components


def _sample_component(points, voxel_size):
    sampled = _voxel_downsample(points, voxel_size)
    return _sample_points(sampled, MAX_POINTS_PER_COMPONENT)


def _mesh_sample_spacing_from_vertices(vertices):
    if MESH_SAMPLE_SPACING > 0.0:
        return float(MESH_SAMPLE_SPACING)
    vertices = _finite_points(vertices)
    if vertices.shape[0] == 0:
        return 0.0
    mins = np.min(vertices, axis=0)
    maxs = np.max(vertices, axis=0)
    diag = float(np.linalg.norm(maxs - mins))
    if not np.isfinite(diag) or diag <= 1e-12:
        return 0.0
    return diag / float(AUTO_VOXEL_DIVISOR)


def _load_geometry_points(path):
    if MESH_SAMPLE_SURFACE:
        mesh = o3d.io.read_triangle_mesh(str(path))
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        triangles = np.asarray(mesh.triangles, dtype=np.int64)
        if vertices.shape[0] > 0 and triangles.shape[0] > 0:
            spacing = _mesh_sample_spacing_from_vertices(vertices)
            area = float(mesh.get_surface_area())
            if spacing > 0.0 and np.isfinite(area) and area > 0.0:
                n_samples = int(np.ceil(area / (spacing * spacing)))
            else:
                n_samples = int(vertices.shape[0])
            n_samples = int(np.clip(n_samples, MESH_MIN_SAMPLE_POINTS, MESH_MAX_SAMPLE_POINTS))
            mesh.compute_vertex_normals()
            pcd = mesh.sample_points_uniformly(number_of_points=n_samples)
            pts = np.asarray(pcd.points, dtype=np.float64)
            print(
                f"[Mesh] sampled {pts.shape[0]} surface points from {path.name} "
                f"(vertices={vertices.shape[0]}, triangles={triangles.shape[0]}, "
                f"area={area:.6f}, spacing={spacing:.6f})"
            )
            return _finite_points(pts)

    # Point-cloud prior (no triangles): load fast via open3d and subsample for
    # the alignment/scale compute. The full cloud is loaded separately in
    # _load_geometry_points_full for saving/passing to the next stage.
    pcd = o3d.io.read_point_cloud(str(path))
    pts = _finite_points(np.asarray(pcd.points, dtype=np.float64))
    if pts.shape[0] == 0:
        pts = _finite_points(_load_ascii_ply_points(path))
    n_before = pts.shape[0]
    pts = _sample_points(pts, MESH_MAX_SAMPLE_POINTS)
    print(f"[Mesh] loaded {n_before} prior points from {path.name}, using {pts.shape[0]} for alignment")
    return pts


def _load_geometry_points_full(path):
    mesh = o3d.io.read_triangle_mesh(str(path))
    vertices = _finite_points(np.asarray(mesh.vertices, dtype=np.float64))
    if vertices.shape[0] > 0:
        print(f"[Mesh] loaded full prior vertices from {path.name}: {vertices.shape[0]}")
        return vertices
    pts = _load_ascii_ply_points(path)
    print(f"[Mesh] loaded full prior points from {path.name}: {pts.shape[0]}")
    return pts


def _load_ascii_ply_points(path):
    with open(path, "r", encoding="utf-8") as f:
        header = []
        while True:
            line = f.readline()
            if line == "":
                raise RuntimeError(f"Unexpected EOF while reading PLY header: {path}")
            header.append(line.strip())
            if line.strip() == "end_header":
                break

        if not header or header[0] != "ply":
            raise RuntimeError(f"Not a PLY file: {path}")
        if "format ascii 1.0" not in header:
            raise RuntimeError(f"Only ASCII PLY is supported here: {path}")

        vertex_count = None
        property_names = []
        in_vertex = False
        for line in header:
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
                in_vertex = True
                continue
            if line.startswith("element ") and not line.startswith("element vertex "):
                in_vertex = False
            if in_vertex and line.startswith("property "):
                property_names.append(line.split()[-1])

        if vertex_count is None:
            raise RuntimeError(f"No vertex element found in PLY: {path}")

        try:
            ix = property_names.index("x")
            iy = property_names.index("y")
            iz = property_names.index("z")
        except ValueError as exc:
            raise RuntimeError(f"PLY missing x/y/z vertex properties: {path}") from exc

        pts = np.zeros((vertex_count, 3), dtype=np.float64)
        for i in range(vertex_count):
            line = f.readline()
            if line == "":
                raise RuntimeError(f"Unexpected EOF while reading vertices from {path}")
            vals = line.strip().split()
            pts[i, 0] = float(vals[ix])
            pts[i, 1] = float(vals[iy])
            pts[i, 2] = float(vals[iz])
    return pts


def _chamfer(src, tgt):
    if src.shape[0] == 0 or tgt.shape[0] == 0:
        return np.inf
    tree_t = cKDTree(tgt)
    d_st, _ = tree_t.query(src, k=1)
    tree_s = cKDTree(src)
    d_ts, _ = tree_s.query(tgt, k=1)
    return float(np.mean(d_ts))


def _rigid_fit(src, tgt):
    src_mean = np.mean(src, axis=0)
    tgt_mean = np.mean(tgt, axis=0)
    src_c = src - src_mean[None, :]
    tgt_c = tgt - tgt_mean[None, :]
    H = src_c.T @ tgt_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = tgt_mean - R @ src_mean
    return R, t


def _apply_similarity(points, scale, R, t, center):
    centered = points - center[None, :]
    scaled = scale * centered
    return (scaled @ R.T) + t[None, :]


def _set_axes_equal(ax, points):
    if points.shape[0] == 0:
        return
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    if not np.isfinite(radius) or radius <= 1e-12:
        radius = 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _make_pcd(o3d, points, color):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    pcd.paint_uniform_color(np.asarray(color, dtype=np.float64))
    return pcd


def _show_open3d_step(title, prior_points=None, obs_points=None, aligned_points=None):
    if not SHOW_OPEN3D:
        return
    geoms = []
    if prior_points is not None and len(prior_points) > 0:
        geoms.append(_make_pcd(o3d, prior_points, [0.1, 0.7, 0.1]))
    if obs_points is not None and len(obs_points) > 0:
        geoms.append(_make_pcd(o3d, obs_points, [0.1, 0.4, 1.0]))
    if aligned_points is not None and len(aligned_points) > 0:
        geoms.append(_make_pcd(o3d, aligned_points, [0.9, 0.1, 0.1]))
    if not geoms:
        return
    print(f"[Open3D] {title}")
    o3d.visualization.draw_geometries(geoms, window_name=title, width=1440, height=900)


def _show_viser_alignment(obs_points, aligned_components, full_aligned_prior=None, port=VISER_PORT):
    """Serve the alignment result in a viser web viewer.

    obs_points        : (N,3) lifted dynamic observations (camera frame), blue.
    aligned_components: list of (Mi,3) aligned prior components, distinct colors.
    full_aligned_prior: optional (K,3) full-res aligned prior (hidden by default).
    """
    import time
    import viser

    obs = np.asarray(obs_points, dtype=np.float32).reshape(-1, 3)
    obs = obs[np.isfinite(obs).all(axis=1)]
    ref = obs if obs.shape[0] > 0 else np.concatenate(
        [np.asarray(c).reshape(-1, 3) for c in aligned_components if len(c) > 0] or [np.zeros((1, 3))], axis=0
    )
    diag = float(np.linalg.norm(ref.max(0) - ref.min(0))) if ref.shape[0] > 1 else 1.0
    psize = max(diag / 400.0, 1e-4)

    palette = [
        (230, 40, 40), (40, 200, 60), (230, 150, 30), (170, 40, 220),
        (30, 200, 200), (230, 70, 130), (140, 140, 30), (70, 70, 230),
    ]
    server = viser.ViserServer(port=int(port))
    server.scene.add_point_cloud(
        "/observations", points=obs, colors=(60, 120, 255), point_size=psize
    )
    for i, comp in enumerate(aligned_components):
        c = np.asarray(comp, dtype=np.float32).reshape(-1, 3)
        c = c[np.isfinite(c).all(axis=1)]
        if c.shape[0] == 0:
            continue
        server.scene.add_point_cloud(
            f"/aligned_prior/component_{i}", points=c, colors=palette[i % len(palette)], point_size=psize
        )
    if full_aligned_prior is not None:
        f = np.asarray(full_aligned_prior, dtype=np.float32).reshape(-1, 3)
        f = f[np.isfinite(f).all(axis=1)]
        if f.shape[0] > 0:
            server.scene.add_point_cloud(
                "/aligned_prior_full", points=f, colors=(200, 200, 200), point_size=psize, visible=False
            )
    print(f"[viser] Alignment viewer at http://localhost:{int(port)}  "
          "(blue=observations, colored=aligned prior).  Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[viser] closing.")


def _load_camera_pose():
    rt = np.load(str(utils.definitions.rootpath / "outputs" / "result_XM.npz"))
    tracks_path = np.load(utils.definitions.tracks_dynamics_path, allow_pickle=True)
    tracks_xyz = tracks_path["tracks_xyz"]
    T = tracks_xyz.shape[2]
    if FRAME_INDEX < 0 or FRAME_INDEX >= T:
        raise IndexError(f"FRAME_INDEX={FRAME_INDEX} out of range [0, {T - 1}]")

    # The static solver may drop frames (checklandmarks / XM^2), so R/t are stored in
    # solved order (N<=T). Scatter them onto the full T-frame timeline via indices_all;
    # dropped frames get identity R + NaN t.
    R_raw, t_raw = scatter_solved_poses(rt, T)
    if not np.isfinite(t_raw[FRAME_INDEX]).all():
        raise RuntimeError(
            f"Frame {FRAME_INDEX} was dropped by the static solver (no pose in result_XM.npz); "
            f"choose a different --frame-index whose camera was solved."
        )

    # Prior alignment happens in raw DA3 camera-depth coordinates.  Static_solver's
    # DA3 frame scale is applied later in EDG_prior when lifting into world space.
    R_for_edg = R_raw.copy()
    return R_raw[FRAME_INDEX], R_for_edg[FRAME_INDEX], t_raw[FRAME_INDEX]

def _load_gt_camera_pose():
    dataset_cat = utils.definitions.dataset_cat
    rootpath = utils.definitions.rootpath

    gt = _load_gt_dict(dataset_cat, rootpath)

    id = FRAME_INDEX
    frame_name = None
    for name, info in gt.items():
        if info["id"] == id:
            frame_name = name
            break
    if frame_name is None:
        raise RuntimeError(f"Could not find GT pose for frame index {FRAME_INDEX} in GT data")
    R_gt = gt[frame_name]["R"]
    t_gt = gt[frame_name]["t"]
    K_gt = gt[frame_name]["K"]
    if R_gt is None or t_gt is None:
        raise RuntimeError(f"GT loader for {dataset_cat} provides intrinsics but no camera pose")
    return R_gt, t_gt, K_gt


def _load_gt_dict(dataset_cat, rootpath):
    if dataset_cat in {"Sintel", "Sintel_debug"}:
        gt = load_sintel_gt(str(rootpath))
    elif dataset_cat == "DAVIS":
        gt = load_davis_gt(str(rootpath))
    elif dataset_cat == "Syn4D":
        gt = load_colmap_gt(str(rootpath / "colmap"))
    elif dataset_cat == "TUM":
        gt = load_tum_gt(str(rootpath))
    elif dataset_cat == "Replica":
        gt = load_replica_gt(str(rootpath))
    elif dataset_cat == "Custom":
        gt = load_custom_gt(rootpath)
    else:
        raise ValueError(f"No GT camera loader configured for dataset_cat={dataset_cat}")
    return gt


def _load_camera_intrinsics_from_gt():
    dataset_cat = utils.definitions.dataset_cat
    rootpath = utils.definitions.rootpath
    gt = _load_gt_dict(dataset_cat, rootpath)
    id = FRAME_INDEX
    frame_name = None
    for name, info in gt.items():
        if info["id"] == id:
            frame_name = name
            break
    if frame_name is None:
        raise RuntimeError(f"Could not find GT intrinsics for frame index {FRAME_INDEX} in GT data")
    return gt[frame_name]["K"]


def _load_camera_intrinsics():
    # Single intrinsic source: prefer intrinsics.npy (written by get_depths_clean.py
    # with the chosen GT-or-DA3 intrinsic already applied); fall back to GT only if
    # that file is absent.
    intrinsics_path = Path(utils.definitions.rootpath) / "outputs" / "depth_anything_3" / "intrinsics.npy"
    if intrinsics_path.is_file():
        intrinsics = np.load(intrinsics_path)
        if intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
            raise ValueError(f"Expected intrinsics shape (T,3,3), got {intrinsics.shape}")
        if FRAME_INDEX < 0 or FRAME_INDEX >= intrinsics.shape[0]:
            raise IndexError(f"FRAME_INDEX={FRAME_INDEX} out of range for {intrinsics_path}")
        return intrinsics[FRAME_INDEX].astype(np.float64)
    return _load_camera_intrinsics_from_gt()


def _load_dynamic_camera_points():
    data = np.load(utils.definitions.tracks_dynamics_path, allow_pickle=True)
    tracks_xy = data["tracks_xy"]
    tracks_xyz = data["tracks_xyz"]
    occluded = data["occluded"]
    dynamic_obs_mask = data["dynamic_obs_mask"][:, :]
    T = tracks_xyz.shape[2]
    tracks_xy = tracks_xy[:, :, :T]
    tracks_xyz = tracks_xyz[:, :, :T]
    occluded = occluded[:, :, :T]
    dynamic_obs_mask = dynamic_obs_mask[:, :T]
    if FRAME_INDEX < 0 or FRAME_INDEX >= T:
        raise IndexError(f"FRAME_INDEX={FRAME_INDEX} out of range [0, {T - 1}]")
    valid_mask = np.isfinite(tracks_xyz[0]).all(axis=-1) & (occluded[0] == 0) & dynamic_obs_mask.astype(bool)
    keep = valid_mask[:, FRAME_INDEX]
    pts = tracks_xyz[0, keep, FRAME_INDEX, :].astype(np.float64)
    xy = tracks_xy[0, keep, FRAME_INDEX, :].astype(np.float64)
    if pts.shape[0] == 0:
        raise RuntimeError(f"No lifted dynamic camera-frame points found at frame {FRAME_INDEX}")
    return pts, xy


def _load_prior_cloud(prior_path_arg=None):
    if prior_path_arg is None:
        root = Path(utils.definitions.rootpath)
        if utils.definitions.dataset_cat == "Custom":
            prior_path_arg = root / "mesh_camera.ply"
        else:
            prior_path_arg = root / "object_pointcloud_world"
    prior_path_arg = Path(prior_path_arg)

    if prior_path_arg.is_dir():
        prior_dir = prior_path_arg
        prior_path = prior_dir / f"frame{FRAME_INDEX:06d}.ply"
    else:
        prior_dir = prior_path_arg.parent
        prior_path = prior_path_arg

    if not prior_path.exists():
        all_paths = sorted(prior_dir.glob("*.ply"))
        if not all_paths:
            raise FileNotFoundError(f"No .ply point clouds found in {prior_dir}")
        raise FileNotFoundError(f"Missing prior file for frame {FRAME_INDEX}: {prior_path}")
    pts = _load_geometry_points(prior_path)
    pts_full = _load_geometry_points_full(prior_path)
    if pts.shape[0] == 0:
        raise RuntimeError(f"Empty prior point cloud: {prior_path}")
    if pts_full.shape[0] == 0:
        pts_full = pts
    return pts, prior_path, pts_full


def _world_to_camera(points_world, R_frame, t_frame):
    delta = points_world - t_frame[None, :]
    return np.linalg.solve(R_frame, delta.T).T


def _default_prior_coordinate_space(dataset_cat):
    if dataset_cat in {"Syn4D", "Sintel", "Sintel_debug", "TUM", "Replica"}:
        return "world"
    if dataset_cat in {"DAVIS", "Custom"}:
        return "camera"
    return "world"


def _default_camera_pose_source(dataset_cat):
    if dataset_cat in {"Syn4D", "Sintel", "Sintel_debug", "TUM", "Replica"}:
        return "gt"
    return "xm"


def _prior_world_to_camera(points_world, pose_source):
    if pose_source == "gt":
        R_gt_w2c, t_gt_w2c, _ = _load_gt_camera_pose()
        return (R_gt_w2c @ points_world.T).T + t_gt_w2c[None, :]

    R_frame, _, t_frame = _load_camera_pose()
    return _world_to_camera(points_world, R_frame, t_frame)


def _hierarchical_scale_search(prior_points, obs_points, base_scale):
    lo = float(SCALE_MIN * base_scale)
    hi = float(SCALE_MAX * base_scale)
    best_scale = float(base_scale)
    best_score = np.inf
    best_init = None

    for round_idx in range(int(SCALE_ROUNDS)):
        scale_grid = np.linspace(lo, hi, int(SCALE_STEPS))
        round_best_scale = None
        round_best_score = np.inf
        round_best_init = None

        for scale in scale_grid:
            src = float(scale) * prior_points
            score = _chamfer(src, obs_points)
            print(f"round={round_idx} scale={scale:.6f} chamfer={score:.6f}")
            if score < round_best_score:
                round_best_score = float(score)
                round_best_scale = float(scale)
                round_best_init = src.copy()

        if round_best_init is None:
            break

        best_scale = round_best_scale
        best_score = round_best_score
        best_init = round_best_init

        if len(scale_grid) <= 1:
            break
        step = float(scale_grid[1] - scale_grid[0])
        lo = max(best_scale - step, 1e-8)
        hi = best_scale + step

    return best_scale, best_score, best_init


def _project_camera_points(points_camera, K):
    pts = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    z = pts[:, 2]
    valid = np.isfinite(pts).all(axis=1) & np.isfinite(z) & (np.abs(z) > 1e-8)
    uv = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
    if np.any(valid):
        x = pts[valid, 0] / z[valid]
        y = pts[valid, 1] / z[valid]
        uv[valid, 0] = K[0, 0] * x + K[0, 2]
        uv[valid, 1] = K[1, 1] * y + K[1, 2]
    return uv, valid


def _visible_prior_mask(points_camera, occluder_points_camera, K):
    """Return points visible in a coarse z-buffer built from the full prior cloud."""
    pts = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
    occ = np.asarray(occluder_points_camera, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0 or occ.shape[0] == 0:
        return np.zeros((pts.shape[0],), dtype=bool)

    uv_occ, valid_occ = _project_camera_points(occ, K)
    z_occ = occ[:, 2]
    valid_occ &= np.isfinite(uv_occ).all(axis=1) & np.isfinite(z_occ) & (np.abs(z_occ) > 1e-8)
    if not np.any(valid_occ):
        return np.ones((pts.shape[0],), dtype=bool)

    z_valid = z_occ[valid_occ]
    positive_depth = float(np.median(z_valid)) >= 0.0
    depth_occ = z_occ if positive_depth else -z_occ
    valid_occ &= depth_occ > 1e-8
    if not np.any(valid_occ):
        return np.ones((pts.shape[0],), dtype=bool)

    pix = max(float(PRIOR_VISIBILITY_PIXEL_SIZE), 1e-6)
    cells_occ = np.floor(uv_occ[valid_occ] / pix).astype(np.int64)
    depth_occ_valid = depth_occ[valid_occ]
    zbuf: dict[tuple[int, int], float] = {}
    for cell, depth in zip(cells_occ, depth_occ_valid, strict=False):
        key = (int(cell[0]), int(cell[1]))
        prev = zbuf.get(key)
        if prev is None or float(depth) < prev:
            zbuf[key] = float(depth)

    uv, valid = _project_camera_points(pts, K)
    z = pts[:, 2]
    depth = z if positive_depth else -z
    valid &= np.isfinite(uv).all(axis=1) & np.isfinite(depth) & (depth > 1e-8)
    visible = np.zeros((pts.shape[0],), dtype=bool)
    if not np.any(valid):
        return visible

    tol = max(
        float(PRIOR_VISIBILITY_DEPTH_TOL_ABS),
        float(PRIOR_VISIBILITY_DEPTH_TOL_REL) * float(np.median(depth_occ_valid)),
    )
    cells = np.floor(uv[valid] / pix).astype(np.int64)
    valid_indices = np.flatnonzero(valid)
    for idx, cell in zip(valid_indices, cells, strict=False):
        front_depth = zbuf.get((int(cell[0]), int(cell[1])))
        if front_depth is None:
            continue
        visible[idx] = float(depth[idx]) <= front_depth + tol
    return visible


def _effective_prior_component_points(component_points, all_prior_points, K, obs_xy_region=None):
    visible = _visible_prior_mask(component_points, all_prior_points, K)
    pts = np.asarray(component_points, dtype=np.float64).reshape(-1, 3)
    effective_mask = visible.copy()

    if obs_xy_region is not None and np.asarray(obs_xy_region).size > 0 and np.any(visible):
        pix = max(float(PRIOR_VISIBILITY_PIXEL_SIZE), 1e-6)
        obs_xy_region = np.asarray(obs_xy_region, dtype=np.float64).reshape(-1, 2)
        obs_xy_region = obs_xy_region[np.isfinite(obs_xy_region).all(axis=1)]
        if obs_xy_region.shape[0] > 0:
            obs_cells = np.floor(obs_xy_region / pix).astype(np.int64)
            obs_cell_set = set()
            for cell in obs_cells:
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        obs_cell_set.add((int(cell[0] + dx), int(cell[1] + dy)))

            uv, uv_valid = _project_camera_points(pts, K)
            prior_cells = np.floor(uv / pix).astype(np.int64)
            in_observed_cells = np.zeros((pts.shape[0],), dtype=bool)
            for idx in np.flatnonzero(visible & uv_valid & np.isfinite(uv).all(axis=1)):
                in_observed_cells[idx] = (int(prior_cells[idx, 0]), int(prior_cells[idx, 1])) in obs_cell_set
            if np.any(in_observed_cells):
                effective_mask &= in_observed_cells
            else:
                print("[Visibility] observed-cell filter removed all visible prior points; using visible prior surface.")

    effective = pts[effective_mask]
    if effective.shape[0] == 0:
        print("[Visibility] no visible points found for component; falling back to full prior component.")
        return pts, effective_mask
    return effective, effective_mask


def _component_region_from_projection(component_points, K, margin_px):
    uv, valid = _project_camera_points(component_points, K)
    uv_valid = uv[valid]
    uv_valid = uv_valid[np.isfinite(uv_valid).all(axis=1)]
    if uv_valid.shape[0] == 0:
        return None, uv
    mn = np.min(uv_valid, axis=0) - float(margin_px)
    mx = np.max(uv_valid, axis=0) + float(margin_px)
    return (float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1])), uv


def _points_in_bbox(xy, bbox):
    if bbox is None:
        return np.zeros((xy.shape[0],), dtype=bool)
    x0, y0, x1, y1 = bbox
    xy = np.asarray(xy, dtype=np.float64)
    return (
        np.isfinite(xy).all(axis=1)
        & (xy[:, 0] >= x0)
        & (xy[:, 0] <= x1)
        & (xy[:, 1] >= y0)
        & (xy[:, 1] <= y1)
    )


def _bbox_to_array(bbox):
    if bbox is None:
        return np.full((4,), np.nan, dtype=np.float64)
    return np.asarray(bbox, dtype=np.float64).reshape(4)


def _assign_observations_to_prior_components(prior_components, obs_points, obs_xy, K, margin_px):
    """Assign observations by the front-most prior component in a coarse z-buffer."""
    bboxes = []
    zbuf: dict[tuple[int, int], tuple[float, int]] = {}
    pix = max(float(PRIOR_VISIBILITY_PIXEL_SIZE), 1e-6)

    all_depth_values = []
    for prior_comp in prior_components:
        bbox, uv = _component_region_from_projection(prior_comp, K, margin_px)
        bboxes.append(bbox)

    all_prior = np.concatenate([np.asarray(c, dtype=np.float64).reshape(-1, 3) for c in prior_components], axis=0)
    all_z = all_prior[:, 2]
    positive_depth = float(np.median(all_z[np.isfinite(all_z)])) >= 0.0

    for pi, prior_comp in enumerate(prior_components):
        pts = np.asarray(prior_comp, dtype=np.float64).reshape(-1, 3)
        uv, valid = _project_camera_points(pts, K)
        depth = pts[:, 2] if positive_depth else -pts[:, 2]
        valid &= np.isfinite(uv).all(axis=1) & np.isfinite(depth) & (depth > 1e-8)
        if not np.any(valid):
            continue
        all_depth_values.append(depth[valid])
        cells = np.floor(uv[valid] / pix).astype(np.int64)
        depths = depth[valid]
        for cell, dep in zip(cells, depths, strict=False):
            key = (int(cell[0]), int(cell[1]))
            prev = zbuf.get(key)
            if prev is None or float(dep) < prev[0]:
                zbuf[key] = (float(dep), int(pi))

    if not zbuf:
        print("[Region] prior z-buffer is empty; no observations assigned to components.")
        return np.full((obs_xy.shape[0],), -1, dtype=np.int64), bboxes

    labels = np.full((obs_xy.shape[0],), -1, dtype=np.int64)
    obs_xy = np.asarray(obs_xy, dtype=np.float64)
    for obs_idx, xy in enumerate(obs_xy):
        if not np.isfinite(xy).all():
            continue
        cell = np.floor(xy / pix).astype(np.int64)

        best = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                key = (int(cell[0] + dx), int(cell[1] + dy))
                entry = zbuf.get(key)
                if entry is None:
                    continue
                front_depth, comp_id = entry
                dist = abs(dx) + abs(dy)
                candidate = (dist, front_depth, comp_id)
                if best is None or candidate < best:
                    best = candidate
        if best is not None:
            labels[obs_idx] = int(best[2])

    print(
        "[Region] z-buffer prior observation assignment: "
        f"prior_components={len(prior_components)}, assigned={int(np.count_nonzero(labels >= 0))}/"
        f"{int(labels.shape[0])}, counts="
        f"{[int(np.count_nonzero(labels == pi)) for pi in range(len(prior_components))]}"
    )
    return labels, bboxes


def _align_prior_components_by_projected_regions(prior_components, obs_points, obs_xy, K, voxel_size, all_prior_points):
    obs_prior_labels, bboxes = _assign_observations_to_prior_components(
        prior_components,
        obs_points,
        obs_xy,
        K,
        REGION_MARGIN_PX,
    )
    matches = []
    for pi, prior_comp in enumerate(prior_components):
        bbox = bboxes[pi]
        region_mask = obs_prior_labels == int(pi)
        obs_region = obs_points[region_mask]
        obs_xy_region = obs_xy[region_mask]

        effective_prior, effective_mask = _effective_prior_component_points(
            prior_comp,
            all_prior_points,
            K,
            obs_xy_region=obs_xy_region,
        )
        prior_sample = _sample_component(effective_prior, voxel_size)
        obs_sample = _sample_component(obs_region, voxel_size)
        if prior_sample.shape[0] == 0:
            continue

        print(
            f"[Region] prior={pi} bbox={bbox} obs_points={obs_region.shape[0]} "
            f"prior_full={prior_comp.shape[0]} prior_effective={effective_prior.shape[0]} "
            f"prior_sample={prior_sample.shape[0]} obs_sample={obs_sample.shape[0]}"
        )
        if obs_sample.shape[0] > 0:
            scale, score, aligned_sample = _hierarchical_scale_search(
                prior_points=prior_sample,
                obs_points=obs_sample,
                base_scale=1.0,
            )
            if aligned_sample is None or not np.isfinite(score):
                scale = 1.0
                score = np.nan
                aligned_sample = prior_sample.copy()
        else:
            scale = 1.0
            score = np.nan
            aligned_sample = prior_sample.copy()
            print(
                f"[Region] prior={pi} has no observation points in its projected region; "
                "keeping prior component with scale=1."
            )
        matches.append(
            {
                "prior_index": int(pi),
                "obs_index": int(pi),
                "scale": float(scale),
                "chamfer": float(score),
                "prior_sample": prior_sample,
                "prior_effective": effective_prior,
                "prior_effective_mask": effective_mask,
                "obs_sample": obs_sample,
                "aligned_sample": aligned_sample,
                "bbox": bbox,
                "obs_count": int(obs_region.shape[0]),
                "obs_global_indices": np.flatnonzero(region_mask).astype(np.int64),
            }
        )

    if not matches:
        raise RuntimeError("No valid prior component was available after prior clustering.")

    print("[Region] selected prior-driven component alignments:")
    for m in matches:
        print(
            f"  prior={m['prior_index']} scale={m['scale']:.6f} "
            f"chamfer={m['chamfer']} obs_count={m['obs_count']} bbox={m['bbox']}"
        )
    return matches, obs_prior_labels


def main():
    args = _parse_args()
    _apply_args(args)

    dataset_cat = utils.definitions.dataset_cat
    prior_coordinate_space = "camera" if args.prior_in_camera_frame else args.prior_coordinate_space
    if prior_coordinate_space == "auto":
        prior_coordinate_space = _default_prior_coordinate_space(dataset_cat)

    camera_pose_source = args.camera_pose_source
    if camera_pose_source == "auto":
        camera_pose_source = _default_camera_pose_source(dataset_cat)

    print(
        f"[Config] frame={FRAME_INDEX} max_components={MAX_COMPONENTS} "
        f"prior_space={prior_coordinate_space} camera_pose_source={camera_pose_source}"
    )

    obs_points_full, obs_xy_full = _load_dynamic_camera_points()
    prior_points_raw_full, prior_path, prior_points_raw_complete = _load_prior_cloud(args.prior_path)

    R_frame, _, t_frame = _load_camera_pose()
    try:
        R_frame_gt, t_frame_gt, _ = _load_gt_camera_pose()
    except Exception as exc:
        if camera_pose_source == "gt":
            raise
        print(f"[warn] GT camera unavailable: {exc}")
        R_frame_gt = np.eye(3, dtype=np.float64)
        t_frame_gt = np.zeros(3, dtype=np.float64)
    # Intrinsics always from the single source (intrinsics.npy); GT is used only
    # for the camera pose above, never for K.
    K_gt = _load_camera_intrinsics()

    if prior_coordinate_space == "world":
        prior_points_full = _prior_world_to_camera(prior_points_raw_full, camera_pose_source)
        prior_points_complete = _prior_world_to_camera(prior_points_raw_complete, camera_pose_source)
        prior_points_world_save = prior_points_raw_full
        prior_points_world_full_save = prior_points_raw_complete
    else:
        prior_points_full = prior_points_raw_full
        prior_points_complete = prior_points_raw_complete
        prior_points_world_save = np.empty((0, 3), dtype=np.float64)
        prior_points_world_full_save = np.empty((0, 3), dtype=np.float64)

    if SHOW_OPEN3D:
        points_cloud = o3d.geometry.PointCloud()
        points_cloud.points = o3d.utility.Vector3dVector(obs_points_full.astype(np.float64))

        points_cloud_prior = o3d.geometry.PointCloud()
        points_cloud_prior.points = o3d.utility.Vector3dVector(prior_points_full.astype(np.float64))
        points_cloud_prior.paint_uniform_color([0.1, 0.7, 0.1])
        points_cloud.paint_uniform_color([0.1, 0.4, 1.0])

        geoms = [points_cloud, points_cloud_prior]
        if prior_coordinate_space == "world":
            points_cloud_world = o3d.geometry.PointCloud()
            points_cloud_world.points = o3d.utility.Vector3dVector(prior_points_raw_full.astype(np.float64))
            points_cloud_world.paint_uniform_color([0.7, 0.1, 0.1])
            geoms.append(points_cloud_world)

        camera_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        camera_axes.transform(np.linalg.inv(np.vstack([np.hstack([R_frame, t_frame[:, None]]), [0, 0, 0, 1]])))
        geoms.append(camera_axes)
        o3d.visualization.draw_geometries(geoms, window_name="Observed Camera-frame Points", width=1440, height=900)

    prior_components = _cluster_components(prior_points_full, "prior")
    voxel_size = _auto_voxel_size(obs_points_full, prior_points_full)
    print(f"[Sample] shared voxel_size={voxel_size:.8f}")

    matches, obs_prior_labels = _align_prior_components_by_projected_regions(
        prior_components=prior_components,
        obs_points=obs_points_full,
        obs_xy=obs_xy_full,
        K=K_gt,
        voxel_size=voxel_size,
        all_prior_points=prior_points_full,
    )

    fit_scales = np.asarray([m["scale"] for m in matches], dtype=np.float64)
    fit_chamfers = np.asarray([m["chamfer"] for m in matches], dtype=np.float64)
    obs_counts = np.asarray([m["obs_count"] for m in matches], dtype=np.float64)
    valid_fit = np.isfinite(fit_scales) & (fit_scales > 0.0) & (obs_counts > 0.0)
    if np.any(valid_fit):
        max_obs = float(np.max(obs_counts[valid_fit]))
        min_required_obs = max(50.0, 0.1 * max_obs)
        reliable = valid_fit & (obs_counts >= min_required_obs)
        finite_chamfer_fit = fit_chamfers[np.isfinite(fit_chamfers) & reliable]
        if finite_chamfer_fit.size > 0:
            chamfer_gate = max(float(np.min(finite_chamfer_fit)) * 4.0, float(np.median(finite_chamfer_fit)) * 3.0)
            reliable &= np.isfinite(fit_chamfers) & (fit_chamfers <= chamfer_gate)
        if not np.any(reliable):
            reliable = valid_fit
        best_scale = float(np.average(fit_scales[reliable], weights=obs_counts[reliable]))
        reliable_indices = [int(matches[i]["prior_index"]) for i in np.flatnonzero(reliable)]
    else:
        best_scale = 1.0
        reliable_indices = []
    print(
        "[Scale] applying one shared prior scale to all components: "
        f"scale={best_scale:.6f}, reliable_prior_components={reliable_indices}"
    )

    aligned_prior_parts = []
    aligned_prior_sample_parts = []
    obs_sample_parts = []
    aligned_prior_component_labels = []
    component_prior_indices = []
    component_obs_indices = []
    component_scales = []
    component_fit_scales = []
    component_chamfers = []
    component_fit_chamfers = []
    component_bboxes = []
    component_obs_counts = []
    effective_prior_parts = []
    effective_prior_component_labels = []

    for m in matches:
        pi = m["prior_index"]
        oi = m["obs_index"]
        scale = best_scale
        aligned_full = scale * prior_components[pi]
        aligned_sample = scale * m["prior_sample"]
        applied_chamfer = (
            _chamfer(aligned_sample, m["obs_sample"])
            if m["obs_sample"].shape[0] > 0 and aligned_sample.shape[0] > 0
            else np.nan
        )
        aligned_prior_parts.append(aligned_full)
        aligned_prior_sample_parts.append(aligned_sample)
        effective_prior_parts.append(float(scale) * m["prior_effective"])
        effective_prior_component_labels.append(np.full(m["prior_effective"].shape[0], pi, dtype=np.int64))
        obs_sample_parts.append(m["obs_sample"])
        aligned_prior_component_labels.append(np.full(aligned_full.shape[0], pi, dtype=np.int64))
        component_prior_indices.append(pi)
        component_obs_indices.append(oi)
        component_scales.append(scale)
        component_fit_scales.append(m["scale"])
        component_chamfers.append(applied_chamfer)
        component_fit_chamfers.append(m["chamfer"])
        component_bboxes.append(_bbox_to_array(m["bbox"]))
        component_obs_counts.append(m["obs_count"])

    aligned_prior_points_full = np.concatenate(aligned_prior_parts, axis=0)
    aligned_full_prior_points_camera = float(best_scale) * prior_points_complete
    aligned_prior_points_sample = np.concatenate(aligned_prior_sample_parts, axis=0)
    aligned_prior_labels = np.concatenate(aligned_prior_component_labels, axis=0)
    obs_points_sample = np.concatenate(obs_sample_parts, axis=0)
    effective_prior_points = (
        np.concatenate(effective_prior_parts, axis=0)
        if effective_prior_parts
        else np.zeros((0, 3), dtype=np.float64)
    )
    effective_prior_labels = (
        np.concatenate(effective_prior_component_labels, axis=0)
        if effective_prior_component_labels
        else np.zeros((0,), dtype=np.int64)
    )
    finite_chamfers = np.asarray(component_chamfers, dtype=np.float64)
    finite_chamfers = finite_chamfers[np.isfinite(finite_chamfers)]
    best_score = float(np.mean(finite_chamfers)) if finite_chamfers.size > 0 else float("nan")

    print("[Region] applied prior-driven component alignments:")
    for pi, applied_scale, fit_scale, applied_chamfer, fit_chamfer, obs_count in zip(
        component_prior_indices,
        component_scales,
        component_fit_scales,
        component_chamfers,
        component_fit_chamfers,
        component_obs_counts,
        strict=False,
    ):
        print(
            f"  prior={pi} applied_scale={applied_scale:.6f} fit_scale={fit_scale:.6f} "
            f"applied_chamfer={applied_chamfer} fit_chamfer={fit_chamfer} obs_count={obs_count}"
        )

    if SHOW_OPEN3D:
        aligned_geoms = [_make_pcd(o3d, obs_points_full, [0.1, 0.4, 1.0])]
        palette = np.asarray(
            [
                [0.9, 0.1, 0.1],
                [0.1, 0.8, 0.2],
                [0.9, 0.6, 0.1],
                [0.7, 0.1, 0.9],
                [0.1, 0.8, 0.8],
                [0.9, 0.3, 0.5],
                [0.5, 0.5, 0.1],
                [0.3, 0.3, 0.9],
            ],
            dtype=np.float64,
        )
        for m in matches:
            pi = m["prior_index"]
            aligned_geoms.append(_make_pcd(o3d, float(best_scale) * prior_components[pi], palette[pi % len(palette)]))
        print("[Open3D] Step 2: full input cloud plus every aligned prior component")
        o3d.visualization.draw_geometries(
            aligned_geoms,
            window_name="Input points + aligned prior components",
            width=1440,
            height=900,
        )

    save_path = Path(utils.definitions.output_dir) / f"aligned_pointcloud_prior_frame{FRAME_INDEX:06d}.npz"
    np.savez_compressed(
        save_path,
        frame_index=np.int64(FRAME_INDEX),
        prior_path=str(prior_path),
        prior_coordinate_space=prior_coordinate_space,
        camera_pose_source=camera_pose_source,
        prior_points_world=prior_points_world_save.astype(np.float32),
        prior_points_world_full=prior_points_world_full_save.astype(np.float32),
        prior_points_camera=prior_points_full.astype(np.float32),
        prior_points_camera_full=prior_points_complete.astype(np.float32),
        observation_points_camera=obs_points_full.astype(np.float32),
        observation_points_image=obs_xy_full.astype(np.float32),
        observation_prior_component_labels=obs_prior_labels.astype(np.int64),
        aligned_full_prior_points_camera=aligned_full_prior_points_camera.astype(np.float32),
        aligned_prior_points_camera=aligned_prior_points_full.astype(np.float32),
        aligned_prior_points_camera_sample=aligned_prior_points_sample.astype(np.float32),
        aligned_prior_component_labels=aligned_prior_labels.astype(np.int64),
        aligned_effective_prior_points_camera=effective_prior_points.astype(np.float32),
        aligned_effective_prior_component_labels=effective_prior_labels.astype(np.int64),
        observation_points_camera_sample=obs_points_sample.astype(np.float32),
        component_prior_indices=np.asarray(component_prior_indices, dtype=np.int64),
        component_obs_indices=np.asarray(component_obs_indices, dtype=np.int64),
        component_scales=np.asarray(component_scales, dtype=np.float64),
        component_fit_scales=np.asarray(component_fit_scales, dtype=np.float64),
        component_chamfer_losses=np.asarray(component_chamfers, dtype=np.float64),
        component_fit_chamfer_losses=np.asarray(component_fit_chamfers, dtype=np.float64),
        component_projected_bboxes=np.asarray(component_bboxes, dtype=np.float64),
        component_region_obs_counts=np.asarray(component_obs_counts, dtype=np.int64),
        component_count_from_prior=np.int64(len(prior_components)),
        matched_component_count=np.int64(len(matches)),
        component_voxel_size=np.float64(voxel_size),
        camera_rotation=R_frame.astype(np.float64),
        camera_translation=t_frame.astype(np.float64),
        camera_rotation_gt=R_frame_gt.astype(np.float64),
        camera_translation_gt=t_frame_gt.astype(np.float64),
        scale=np.float64(best_scale),
        chamfer_loss=np.float64(best_score),
    )
    print(f"Saved aligned prior to {save_path}")

    if SHOW_VISER:
        aligned_components_viser = [float(best_scale) * prior_components[m["prior_index"]] for m in matches]
        _show_viser_alignment(
            obs_points_full,
            aligned_components_viser,
            full_aligned_prior=aligned_full_prior_points_camera,
            port=VISER_PORT,
        )


if __name__ == "__main__":
    main()
