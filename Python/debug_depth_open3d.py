#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "Python"
MOSCA_DIR = REPO_ROOT / "deps" / "MoSca"
for path in (PYTHON_DIR, MOSCA_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import visualize_depth_comparison as vis  # noqa: E402
from eval_depth_depthanything import _load_da3_cameras, _load_gt, _scale_K  # noqa: E402


METHOD_COLORS = {
    "GT": (0.95, 0.95, 0.95),
    "Ours": (0.15, 0.55, 1.0),
    "MoSca": (1.0, 0.45, 0.1),
    "SpaTracker": (0.9, 0.1, 0.9),
    "DA3 direct": (0.1, 0.8, 0.25),
    "DA3 train-render": (0.05, 0.65, 0.2),
    "DA3 novel-render": (0.05, 0.65, 0.2),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open3D debug view of the exact scaled depth maps/masks used by "
            "visualize_depth_comparison.py for PNG visualization and L1."
        )
    )
    parser.add_argument("--scene-dir", type=Path, default=Path("datasets/Syn4D/bear1"))
    parser.add_argument("--split", choices=["train", "novel"], default="train")
    parser.add_argument("--frame-index", type=int, default=19)
    parser.add_argument("--novel-id", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--scale-mode", choices=["median", "gt_camera"], default="median")
    parser.add_argument("--ours-scale-mode", choices=["median", "gt_camera"], default=None)
    parser.add_argument("--mosca-scale-mode", choices=["median", "gt_camera"], default=None)
    parser.add_argument("--spatracker-scale-mode", choices=["median", "gt_camera"], default=None)
    parser.add_argument("--da3-scale-mode", choices=["median", "gt_camera"], default=None)
    parser.add_argument("--only", choices=["ours", "mosca", "spatracker", "da3"], action="append", default=None)
    parser.add_argument("--no-ours", action="store_true")
    parser.add_argument("--no-mosca", action="store_true")
    parser.add_argument("--no-spatracker", action="store_true")
    parser.add_argument("--no-da3", action="store_true")
    parser.add_argument("--gt-scale", type=float, default=1000.0)
    parser.add_argument("--min-depth", type=float, default=1e-6)
    parser.add_argument("--max-depth", type=float, default=-1.0)
    parser.add_argument("--alpha-threshold", type=float, default=0.05)
    parser.add_argument("--mask-threshold", type=float, default=127.0)
    parser.add_argument("--point-stride", type=int, default=4)
    parser.add_argument("--dynamic-point-stride", type=int, default=1)
    parser.add_argument("--spatracker-point-stride", type=int, default=0)
    parser.add_argument("--spatracker-point-radius", type=int, default=5)
    parser.add_argument("--spatracker-max-points", type=int, default=200000)
    parser.add_argument("--spatracker-use-gt-intrinsics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--spatracker-align", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ours-camera-source", choices=["colmap", "xm"], default="colmap")
    parser.add_argument("--ours-gaussians", type=Path, default=None)
    parser.add_argument("--mosca-npz", type=Path, default=None)
    parser.add_argument("--spatracker-npz", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--camera-stride", type=int, default=10)
    parser.add_argument("--camera-scale", type=float, default=0.35)
    parser.add_argument("--cloud-stride", type=int, default=3, help="Pixel stride for lifting displayed clouds.")
    parser.add_argument("--max-points-per-cloud", type=int, default=120000)
    parser.add_argument("--voxel-size", type=float, default=0.0, help="Open3D voxel downsample size; 0 disables.")
    parser.add_argument("--hide-gt-cloud", action="store_true")
    parser.add_argument("--hide-method-clouds", action="store_true")
    parser.add_argument("--write-ply", action="store_true")
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _image_files(scene_dir: Path) -> list[Path]:
    return sorted(
        p for p in (scene_dir / "images").iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


def _image_size(path: Path) -> tuple[int, int]:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return int(img.shape[1]), int(img.shape[0])


def _gt_c2w(info: dict[str, Any]) -> np.ndarray:
    R_w2c = np.asarray(info["R"], dtype=np.float64).reshape(3, 3)
    t_w2c = np.asarray(info["t"], dtype=np.float64).reshape(3)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R_w2c.T
    c2w[:3, 3] = -R_w2c.T @ t_w2c
    return c2w


def _target_camera(scene_dir: Path, args: argparse.Namespace, frame: int, gt_depth: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int, str]:
    gt = _load_gt(scene_dir)
    images = _image_files(scene_dir)
    if args.split == "train":
        info = next(info for info in gt.values() if int(info["id"]) == int(frame))
        base_frame = int(frame)
        label = f"GT train frame {base_frame}"
        c2w = _gt_c2w(info)
    else:
        rec = vis._selected_novel_record(scene_dir, int(args.novel_id))
        base_info = next(info for info in gt.values() if int(info["id"]) == int(rec.base_frame))
        base_frame = int(rec.base_frame)
        label = f"GT novel {int(rec.novel_id)} from base {base_frame}"
        c2w = np.asarray(rec.c2w, dtype=np.float64)
        info = base_info
    src_size = _image_size(images[base_frame])
    K = _scale_K(np.asarray(info["K"], dtype=np.float64), src_size, (gt_depth.shape[1], gt_depth.shape[0]))
    return c2w, K, gt_depth.shape[1], gt_depth.shape[0], label


def _lift_depth(depth: np.ndarray, valid: np.ndarray, c2w: np.ndarray, K: np.ndarray, stride: int) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    H, W = depth.shape
    stride = max(1, int(stride))
    ys, xs = np.meshgrid(np.arange(0, H, stride), np.arange(0, W, stride), indexing="ij")
    keep = valid[ys, xs] & np.isfinite(depth[ys, xs]) & (depth[ys, xs] > 0)
    if not np.any(keep):
        return np.zeros((0, 3), dtype=np.float64)
    xs_f = xs[keep].astype(np.float64)
    ys_f = ys[keep].astype(np.float64)
    z = depth[ys, xs][keep].astype(np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    cam = np.stack(
        [
            (xs_f - cx) / max(fx, 1e-12) * z,
            (ys_f - cy) / max(fy, 1e-12) * z,
            z,
        ],
        axis=1,
    )
    return (c2w[:3, :3] @ cam.T).T + c2w[:3, 3]


def _sample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if int(max_points) > 0 and points.shape[0] > int(max_points):
        rng = np.random.default_rng(seed)
        points = points[rng.choice(points.shape[0], int(max_points), replace=False)]
    return points


def _make_pcd(points: np.ndarray, color: tuple[float, float, float], max_points: int, seed: int, voxel: float):
    import open3d as o3d

    points = _sample_points(points, max_points, seed)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(color, dtype=np.float64), (points.shape[0], 1)))
    if float(voxel) > 0 and points.shape[0] > 0:
        pcd = pcd.voxel_down_sample(float(voxel))
    return pcd


def _frustum_points(c2w: np.ndarray, K: np.ndarray, width: int, height: int, scale: float) -> tuple[np.ndarray, list[tuple[int, int]]]:
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    z = float(scale)
    corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float64)
    cam = np.column_stack(
        [
            (corners[:, 0] - cx) / max(fx, 1e-12) * z,
            (corners[:, 1] - cy) / max(fy, 1e-12) * z,
            np.full(4, z),
        ]
    )
    pts = np.concatenate([c2w[:3, 3][None], cam @ c2w[:3, :3].T + c2w[:3, 3]], axis=0)
    lines = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
    return pts, lines


def _transform_c2w_similarity(c2w: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = np.asarray(c2w, dtype=np.float64).copy()
    out[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3) @ out[:3, :3]
    out[:3, 3] = float(scale) * (np.asarray(R, dtype=np.float64).reshape(3, 3) @ out[:3, 3]) + np.asarray(t, dtype=np.float64).reshape(3)
    return out


def _inverse_transform_c2w_similarity(c2w: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    out = np.asarray(c2w, dtype=np.float64).copy()
    out[:3, :3] = R.T @ out[:3, :3]
    out[:3, 3] = R.T @ ((out[:3, 3] - t) / max(float(scale), 1e-12))
    return out


def _make_lineset(frustums: list[tuple[np.ndarray, list[tuple[int, int]], tuple[float, float, float]]]):
    import open3d as o3d

    pts_all = []
    lines_all = []
    colors = []
    offset = 0
    for pts, lines, color in frustums:
        pts_all.extend(pts.tolist())
        lines_all.extend([(a + offset, b + offset) for a, b in lines])
        colors.extend([color] * len(lines))
        offset += len(pts)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.asarray(pts_all, dtype=np.float64))
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines_all, dtype=np.int32))
    line_set.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return line_set


def _sampled_camera_frustums(
    c2ws: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    scale: float,
    stride: int,
    color: tuple[float, float, float],
) -> list[tuple[np.ndarray, list[tuple[int, int]], tuple[float, float, float]]]:
    frustums = []
    for idx, c2w in enumerate(np.asarray(c2ws, dtype=np.float64)):
        if idx % max(1, int(stride)) != 0:
            continue
        frustums.append((*_frustum_points(c2w, K, width, height, scale), color))
    return frustums


def _method_key(name: str) -> str:
    lower = name.lower()
    if lower.startswith("da3"):
        return "da3"
    if lower == "spatracker":
        return "spatracker"
    if lower == "mosca":
        return "mosca"
    return "ours"


def _build_vis_args(args: argparse.Namespace) -> argparse.Namespace:
    return SimpleNamespace(
        scene_dir=args.scene_dir,
        split=args.split,
        frame_index=args.frame_index,
        novel_id=args.novel_id,
        output_dir=args.output_dir,
        gt_scale=args.gt_scale,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        point_stride=args.point_stride,
        dynamic_point_stride=args.dynamic_point_stride,
        point_radius=1,
        spatracker_point_stride=args.spatracker_point_stride,
        spatracker_point_radius=args.spatracker_point_radius,
        spatracker_max_points=args.spatracker_max_points,
        spatracker_use_gt_intrinsics=args.spatracker_use_gt_intrinsics,
        alpha_threshold=args.alpha_threshold,
        alpha_density_radius=1,
        alpha_density_scale=1.0,
        mask_threshold=args.mask_threshold,
        device=args.device,
        ours_camera_source=args.ours_camera_source,
        ours_gaussians=args.ours_gaussians,
        mosca_npz=args.mosca_npz,
        spatracker_npz=args.spatracker_npz,
        spatracker_align=args.spatracker_align,
        error_clip=2.0,
        scale_mode=args.scale_mode,
        ours_scale_mode=args.ours_scale_mode,
        mosca_scale_mode=args.mosca_scale_mode,
        spatracker_scale_mode=args.spatracker_scale_mode,
        da3_scale_mode=args.da3_scale_mode,
        only=args.only,
        combine_existing=False,
        no_ours=args.no_ours,
        no_mosca=args.no_mosca,
        no_spatracker=args.no_spatracker,
        no_da3=args.no_da3,
    )


def _add_reference_cameras(scene_dir: Path, target_c2w: np.ndarray, K: np.ndarray, width: int, height: int, args: argparse.Namespace):
    gt = _load_gt(scene_dir)
    frustums = []
    infos = sorted(gt.values(), key=lambda x: int(x["id"]))
    for idx, info in enumerate(infos):
        if idx % max(1, int(args.camera_stride)) != 0:
            continue
        frustums.append((*_frustum_points(_gt_c2w(info), K, width, height, args.camera_scale), (0.5, 0.5, 0.5)))
    frustums.append((*_frustum_points(target_c2w, K, width, height, args.camera_scale * 1.8), (1.0, 1.0, 1.0)))
    return frustums


def _add_method_camera_fallbacks(scene_dir: Path, target_c2w: np.ndarray, K: np.ndarray, width: int, height: int, items: list[dict[str, Any]], args: argparse.Namespace):
    # The displayed clouds are lifted from the exact target-view depth arrays, so the
    # exact render/metric camera for each depth map is the target camera. Draw one
    # colored duplicate per method to make the active method set explicit.
    frustums = []
    names = []
    offset = 0.0
    right = target_c2w[:3, 0]
    for item in items:
        name = str(item["name"])
        c2w = target_c2w.copy()
        c2w[:3, 3] = c2w[:3, 3] + right * offset
        offset += float(args.camera_scale) * 0.08
        frustums.append((*_frustum_points(c2w, K, width, height, args.camera_scale * 1.25), METHOD_COLORS.get(name, (0.8, 0.8, 0.2))))
        names.append(name)
    return frustums, names


def _add_aligned_method_camera_paths(
    scene_dir: Path,
    K: np.ndarray,
    width: int,
    height: int,
    args: argparse.Namespace,
) -> tuple[list[tuple[np.ndarray, list[tuple[int, int]], tuple[float, float, float]]], list[str]]:
    frustums: list[tuple[np.ndarray, list[tuple[int, int]], tuple[float, float, float]]] = []
    labels: list[str] = []

    if not args.no_ours:
        try:
            import eval_depth_gaussian_ours as ours

            est_c2ws, _gt_c2ws, _names = ours._load_training_gt_pose_pairs(scene_dir, args.ours_camera_source)
            s, R, t, method = ours._alignment_est_to_gt(scene_dir, args.ours_camera_source)
            aligned = np.stack([_transform_c2w_similarity(pose, s, R, t) for pose in est_c2ws], axis=0)
            frustums.extend(
                _sampled_camera_frustums(aligned, K, width, height, args.camera_scale, args.camera_stride, METHOD_COLORS["Ours"])
            )
            labels.append(f"Ours aligned train cameras ({method})")
        except Exception as exc:
            print(f"[open3d-depth] skip Ours aligned cameras: {exc}")

    if not args.no_mosca:
        try:
            from estimate_mosca_chamfer import _find_latest_mosca_npz
            from visualize_mosca_edg import _ate_gt_to_solver_similarity, _get_c2w, _load_gt_c2w_from_colmap

            npz_path = _find_latest_mosca_npz(args.mosca_npz or scene_dir)
            data = np.load(npz_path, allow_pickle=True)
            mosca_c2w = _get_c2w(data)
            gt_c2w = _load_gt_c2w_from_colmap(scene_dir)
            if mosca_c2w is None or gt_c2w is None:
                raise RuntimeError("missing MoSca or GT camera c2w")
            _scale_gt_to_solver, _R_gt_to_solver, _t_gt_to_solver, _gt_aligned, camera_align = _ate_gt_to_solver_similarity(
                mosca_c2w, gt_c2w
            )
            s_solver_to_gt, R_solver_to_gt, t_solver_to_gt = camera_align
            aligned = np.stack(
                [_transform_c2w_similarity(pose, s_solver_to_gt, R_solver_to_gt, t_solver_to_gt) for pose in mosca_c2w],
                axis=0,
            )
            frustums.extend(
                _sampled_camera_frustums(aligned, K, width, height, args.camera_scale, args.camera_stride, METHOD_COLORS["MoSca"])
            )
            labels.append("MoSca aligned train cameras")
        except Exception as exc:
            print(f"[open3d-depth] skip MoSca aligned cameras: {exc}")

    if not args.no_spatracker:
        try:
            import eval_spatracker as spa

            result_npz = args.spatracker_npz
            if result_npz is None:
                candidates = [
                    scene_dir / "outputs" / "spatracker_rgb_dynamic_adaptive_input" / "results" / "result.npz",
                    scene_dir / "outputs" / "spatracker_rgb_dynamic_input" / "results" / "result.npz",
                    scene_dir / "outputs" / "spatracker_input" / "results" / "result.npz",
                ]
                result_npz = next((p for p in candidates if p.is_file()), None)
            if result_npz is None:
                raise FileNotFoundError("SpaTracker result.npz")
            data = spa._load_result(Path(result_npz))
            gt = _load_gt(scene_dir)
            _camera, align, c2w_est = spa._camera_eval(scene_dir, gt, data)
            aligned = np.stack([spa._transform_c2w_similarity(pose, *align) for pose in c2w_est], axis=0)
            frustums.extend(
                _sampled_camera_frustums(aligned, K, width, height, args.camera_scale, args.camera_stride, METHOD_COLORS["SpaTracker"])
            )
            labels.append("SpaTracker aligned train cameras")
        except Exception as exc:
            print(f"[open3d-depth] skip SpaTracker aligned cameras: {exc}")

    if not args.no_da3:
        try:
            gt = _load_gt(scene_dir)
            _intrs, da_c2ws = _load_da3_cameras(scene_dir / "outputs" / "depth_anything_3")
            gt_to_da, R_gt_to_da, t_gt_to_da = __import__("eval_depth_depthanything")._gt_to_da_similarity(gt, da_c2ws)
            aligned = np.stack(
                [_inverse_transform_c2w_similarity(pose, gt_to_da, R_gt_to_da, t_gt_to_da) for pose in da_c2ws],
                axis=0,
            )
            frustums.extend(
                _sampled_camera_frustums(aligned, K, width, height, args.camera_scale, args.camera_stride, METHOD_COLORS["DA3 direct"])
            )
            labels.append("DA3 aligned train cameras")
        except Exception as exc:
            print(f"[open3d-depth] skip DA3 aligned cameras: {exc}")

    return frustums, labels


def main() -> None:
    args = _parse_args()
    scene_dir = args.scene_dir.resolve()
    out_dir = (args.output_dir or (scene_dir / "outputs" / "depth_open3d_debug")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    vis_args = _build_vis_args(args)
    gt_depth, frame, gt_path, label = vis._target_info(scene_dir, vis_args)
    items = vis._load_eval_items(scene_dir, gt_depth, frame, vis_args)
    target_c2w, K, width, height, camera_label = _target_camera(scene_dir, args, frame, gt_depth)

    import open3d as o3d

    geoms = []
    if not args.hide_gt_cloud:
        gt_valid = np.isfinite(gt_depth) & (gt_depth > float(args.min_depth))
        if float(args.max_depth) > 0:
            gt_valid &= gt_depth <= float(args.max_depth)
        gt_points = _lift_depth(gt_depth, gt_valid, target_c2w, K, int(args.cloud_stride))
        gt_pcd = _make_pcd(gt_points, METHOD_COLORS["GT"], int(args.max_points_per_cloud), 0, float(args.voxel_size))
        geoms.append(gt_pcd)
        if args.write_ply:
            o3d.io.write_point_cloud(str(out_dir / f"{label}_GT_cloud.ply"), gt_pcd)
        print(f"[open3d-depth] GT cloud points={len(gt_pcd.points)}")

    if not args.hide_method_clouds:
        for idx, item in enumerate(items):
            name = str(item["name"])
            points = _lift_depth(item["depth"], item["valid"], target_c2w, K, int(args.cloud_stride))
            pcd = _make_pcd(points, METHOD_COLORS.get(name, (0.8, 0.8, 0.2)), int(args.max_points_per_cloud), idx + 10, float(args.voxel_size))
            geoms.append(pcd)
            if args.write_ply:
                safe_name = name.replace(" ", "_").replace("/", "_")
                o3d.io.write_point_cloud(str(out_dir / f"{label}_{safe_name}_cloud.ply"), pcd)
            print(
                "[open3d-depth] "
                f"{name}: cloud_points={len(pcd.points)} valid_px={item['valid_px']} "
                f"l1={item.get('relative_l1')} scale={item.get('est_to_gt_scale')} "
                f"median_scale={item.get('median_est_to_gt_scale')} "
                f"gt_camera_scale={item.get('gt_camera_est_to_gt_scale')}"
            )

    frustums = _add_reference_cameras(scene_dir, target_c2w, K, width, height, args)
    aligned_method_frustums, aligned_method_labels = _add_aligned_method_camera_paths(scene_dir, K, width, height, args)
    frustums.extend(aligned_method_frustums)
    method_frustums, method_names = _add_method_camera_fallbacks(scene_dir, target_c2w, K, width, height, items, args)
    frustums.extend(method_frustums)
    geoms.append(_make_lineset(frustums))

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(args.camera_scale), origin=target_c2w[:3, 3])
    geoms.append(coord)

    print(f"[open3d-depth] scene={scene_dir}")
    print(f"[open3d-depth] split={args.split} frame={frame} label={label}")
    print(f"[open3d-depth] camera={camera_label}")
    print(f"[open3d-depth] GT depth={gt_path}")
    print(f"[open3d-depth] aligned method camera paths: {aligned_method_labels}")
    print(f"[open3d-depth] method cameras are target-view cameras used by the exact saved depth arrays: {method_names}")
    print("[open3d-depth] colors: GT=white/gray, Ours=blue, MoSca=orange, SpaTracker=magenta, DA3=green")

    if args.show:
        o3d.visualization.draw_geometries(geoms, window_name=f"Depth debug: {label}")


if __name__ == "__main__":
    main()
