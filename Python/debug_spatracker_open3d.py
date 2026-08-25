#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from eval_depth import _prefer_npy_depth_path, _read_depth
from eval_depth_depthanything import _lift_depth_to_world, _load_gt, _read_dynamic_mask, _scale_K
from eval_spatracker import _c2w_from_w2c, _gt_by_id, _gt_c2w
from utils.error import ATE_TEASER_C2W


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open3D debug view for SpaTracker camera alignment and lifted pointmap clouds."
    )
    parser.add_argument("--scene-dir", type=Path, default=Path("datasets/Syn4D/bear1"))
    parser.add_argument("--result-npz", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--frame", type=int, default=0, help="Start frame id for dynamic debug view.")
    parser.add_argument("--camera-stride", type=int, default=10)
    parser.add_argument("--camera-scale", type=float, default=0.15)
    parser.add_argument("--point-stride", type=int, default=2, help="Stride for lifted SpaTracker pointmap.")
    parser.add_argument("--gt-point-stride", type=int, default=4)
    parser.add_argument("--max-points", type=int, default=200000)
    parser.add_argument("--gt-scale", type=float, default=1000.0)
    parser.add_argument("--min-depth", type=float, default=1e-6)
    parser.add_argument("--max-depth", type=float, default=-1.0)
    parser.add_argument("--mask-dir", type=Path, default=None)
    parser.add_argument("--mask-threshold", type=float, default=127.0)
    parser.add_argument("--write-gt-pointcloud", action="store_true", help="Also export the GT depth point cloud.")
    parser.add_argument("--write-raw-pointcloud", action="store_true", help="Also export the raw pre-alignment SpaTracker point cloud.")
    parser.add_argument("--render-train-views", action="store_true", help="Render and save a PNG for each train view.")
    parser.add_argument("--render-point-size", type=float, default=2.5)
    parser.add_argument("--show", action="store_true", help="Open an Open3D window after writing debug geometry.")
    return parser.parse_args()


def _image_files(scene_dir: Path) -> list[Path]:
    return sorted(
        p for p in (scene_dir / "images").iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


def _image_size(path: Path) -> tuple[int, int]:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return int(img.shape[1]), int(img.shape[0])


def _load_result(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    required = ["extrinsics", "intrinsics", "depths"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"{path} missing keys: {missing}")
    return {key: data[key] for key in data.files}


def _umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or src.shape[0] < 3:
        return 1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src0 = src - mu_src
    dst0 = dst - mu_dst
    cov = (dst0.T @ src0) / float(src.shape[0])
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3, dtype=np.float64)
    if np.linalg.det(U @ Vt) < 0:
        D[-1, -1] = -1.0
    R = U @ D @ Vt
    var_src = float(np.mean(np.sum(src0 * src0, axis=1)))
    scale = float(np.trace(np.diag(S) @ D) / max(var_src, 1e-12))
    t = mu_dst - scale * (R @ mu_src)
    return scale, R, t


def _apply_similarity_points(points: np.ndarray, sim: tuple[float, np.ndarray, np.ndarray]) -> np.ndarray:
    if points.size == 0:
        return points.reshape(0, 3)
    s, R, t = sim
    return s * (R @ points.T).T + t.reshape(1, 3)


def _apply_similarity_c2w(c2w: np.ndarray, sim: tuple[float, np.ndarray, np.ndarray]) -> np.ndarray:
    s, R, t = sim
    out = np.asarray(c2w, dtype=np.float64).copy()
    out[:3, :3] = R @ out[:3, :3]
    out[:3, 3] = s * (R @ out[:3, 3]) + np.asarray(t, dtype=np.float64).reshape(3)
    return out


def _camera_frustum(c2w: np.ndarray, K: np.ndarray, width: int, height: int, scale: float) -> tuple[np.ndarray, list[tuple[int, int]]]:
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    z = float(scale)
    corners_px = np.array(
        [
            [0.0, 0.0],
            [float(width - 1), 0.0],
            [float(width - 1), float(height - 1)],
            [0.0, float(height - 1)],
        ],
        dtype=np.float64,
    )
    corners_cam = np.column_stack(
        [
            (corners_px[:, 0] - cx) / max(fx, 1e-12) * z,
            (corners_px[:, 1] - cy) / max(fy, 1e-12) * z,
            np.full(4, z, dtype=np.float64),
        ]
    )
    origin = c2w[:3, 3][None]
    corners_world = corners_cam @ c2w[:3, :3].T + c2w[:3, 3]
    points = np.concatenate([origin, corners_world], axis=0)
    lines = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
    return points, lines


def _make_lineset(frustums: list[tuple[np.ndarray, list[tuple[int, int]], tuple[float, float, float]]]):
    import open3d as o3d

    points = []
    lines = []
    colors = []
    offset = 0
    for pts, ls, color in frustums:
        points.extend(pts.tolist())
        lines.extend([(a + offset, b + offset) for a, b in ls])
        colors.extend([color] * len(ls))
        offset += len(pts)
    out = o3d.geometry.LineSet()
    out.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    out.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    out.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return out


def _make_pcd(points: np.ndarray, color: tuple[float, float, float], max_points: int, seed: int):
    import open3d as o3d

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if int(max_points) > 0 and points.shape[0] > int(max_points):
        rng = np.random.default_rng(int(seed))
        keep = rng.choice(points.shape[0], int(max_points), replace=False)
        points = points[keep]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(color, dtype=np.float64), (points.shape[0], 1)))
    return pcd


def _lift_depth_to_world_with_color(
    depth: np.ndarray,
    rgb: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    min_depth: float,
    max_depth: float,
    stride: int,
    pixel_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth, dtype=np.float64)
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    H, W = depth.shape
    stride = max(1, int(stride))
    ys, xs = np.meshgrid(np.arange(0, H, stride), np.arange(0, W, stride), indexing="ij")
    z = depth[ys, xs]
    valid = np.isfinite(z) & (z > float(min_depth))
    if float(max_depth) > 0:
        valid &= z <= float(max_depth)
    if pixel_mask is not None:
        valid &= pixel_mask[ys, xs]
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
    xs = xs[valid].astype(np.float64)
    ys = ys[valid].astype(np.float64)
    z = z[valid].astype(np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (xs - cx) / max(fx, 1e-12) * z
    y = (ys - cy) / max(fy, 1e-12) * z
    cam = np.stack([x, y, z], axis=1)
    world = (c2w[:3, :3] @ cam.T).T + c2w[:3, 3]
    colors = rgb[ys.astype(int), xs.astype(int), :3]
    return world, colors


def _write_pcd(path: Path, points: np.ndarray, color: tuple[float, float, float], max_points: int, seed: int):
    import open3d as o3d

    pcd = _make_pcd(points, color, max_points, seed)
    o3d.io.write_point_cloud(str(path), pcd)
    return pcd


def _local_index_for_frame(frame_ids: np.ndarray, frame: int) -> int:
    matches = np.where(frame_ids == int(frame))[0]
    if matches.size:
        return int(matches[0])
    if 0 <= int(frame) < frame_ids.shape[0]:
        return int(frame)
    raise IndexError(f"frame {frame} not available in result frame_ids")


def main() -> None:
    args = _parse_args()
    scene_dir = args.scene_dir.resolve()
    result_npz = (args.result_npz or (scene_dir / "outputs" / "spatracker_eval" / "result.npz")).resolve()
    if not result_npz.is_file():
        result_npz = (scene_dir / "outputs" / "spatracker_rgb_dynamic_input" / "results" / "result.npz").resolve()
    out_dir = (args.output_dir or (scene_dir / "outputs" / "spatracker_open3d_debug")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load_result(result_npz)
    gt = _load_gt(scene_dir)
    by_id = _gt_by_id(gt)
    images = _image_files(scene_dir)

    extrinsics = np.asarray(data["extrinsics"], dtype=np.float64)
    intrs = np.asarray(data["intrinsics"], dtype=np.float64)
    depths = np.asarray(data["depths"], dtype=np.float64)
    frame_ids = np.asarray(data.get("frame_ids", np.arange(extrinsics.shape[0])), dtype=np.int64).reshape(-1)
    if frame_ids.shape[0] != extrinsics.shape[0]:
        frame_ids = np.arange(extrinsics.shape[0], dtype=np.int64)

    gt_centers = []
    spa_centers = []
    R_gt_w2c = []
    t_gt_w2c = []
    R_est_c2w = []
    used = []
    gt_c2ws: dict[int, np.ndarray] = {}
    spa_c2ws: dict[int, np.ndarray] = {}
    for local_idx, frame_id in enumerate(frame_ids):
        if int(frame_id) not in by_id:
            continue
        _, info = by_id[int(frame_id)]
        gt_c2w = _gt_c2w(info)
        spa_c2w = _c2w_from_w2c(extrinsics[local_idx])
        gt_c2ws[int(frame_id)] = gt_c2w
        spa_c2ws[int(frame_id)] = spa_c2w
        gt_centers.append(gt_c2w[:3, 3])
        spa_centers.append(spa_c2w[:3, 3])
        R_gt_w2c.append(np.asarray(info["R"], dtype=np.float64).reshape(3, 3))
        t_gt_w2c.append(np.asarray(info["t"], dtype=np.float64).reshape(3))
        R_est_c2w.append(spa_c2w[:3, :3])
        used.append(int(frame_id))
    if len(used) < 3:
        raise RuntimeError("Need at least 3 matched cameras for similarity alignment.")
    gt_centers_arr = np.stack(gt_centers, axis=0)
    spa_centers_arr = np.stack(spa_centers, axis=0)
    R_gt_w2c_arr = np.concatenate(R_gt_w2c, axis=1)
    t_gt_w2c_arr = np.stack(t_gt_w2c, axis=1)
    R_est_c2w_arr = np.concatenate(R_est_c2w, axis=1)
    t_est_c2w_arr = np.stack(spa_centers, axis=1)
    scale_align, R_align, t_align = ATE_TEASER_C2W(
        R_est_c2w_arr,
        t_est_c2w_arr,
        R_gt_w2c_arr,
        t_gt_w2c_arr,
    )
    sim = (float(scale_align), np.asarray(R_align, dtype=np.float64), np.asarray(t_align, dtype=np.float64).reshape(3))
    spa_centers_aligned = _apply_similarity_points(spa_centers_arr, sim)
    center_rmse = float(np.sqrt(np.mean(np.sum((spa_centers_aligned - gt_centers_arr) ** 2, axis=1))))

    frame = int(args.frame)
    local_idx = _local_index_for_frame(frame_ids, frame)
    if frame not in by_id:
        raise KeyError(f"GT frame {frame} is not available.")
    image_size = _image_size(images[frame])
    mask_dir = (args.mask_dir or (scene_dir / "masks")).resolve()

    spa_points_raw = np.zeros((0, 3), dtype=np.float64)
    if args.write_raw_pointcloud:
        depth = depths[local_idx]
        K_spa = intrs[local_idx if intrs.shape[0] > 1 else 0]
        spa_points_raw = _lift_depth_to_world(
            depth,
            K_spa,
            spa_c2ws[frame],
            args.min_depth,
            args.max_depth,
            args.point_stride,
            0,
            pixel_mask=None,
        )
    spa_points_aligned = (
        _apply_similarity_points(spa_points_raw, sim)
        if args.write_raw_pointcloud
        else np.zeros((0, 3), dtype=np.float64)
    )

    static_points_raw = []
    for frame_id in used:
        local_i = _local_index_for_frame(frame_ids, frame_id)
        depth = depths[local_i]
        K_spa = intrs[local_i if intrs.shape[0] > 1 else 0]
        image_name = next((name for name, info in gt.items() if int(info.get("id", -1)) == frame_id), None)
        dynamic_mask = _read_dynamic_mask(mask_dir, frame_id, depth.shape, args.mask_threshold, image_name=image_name)
        static_mask = ~dynamic_mask
        static_pts = _lift_depth_to_world(
            depth,
            K_spa,
            spa_c2ws[frame_id],
            args.min_depth,
            args.max_depth,
            args.point_stride,
            0,
            pixel_mask=static_mask,
        )
        if static_pts.size > 0:
            static_points_raw.append(static_pts)
    spa_static_raw = np.concatenate(static_points_raw, axis=0) if static_points_raw else np.zeros((0, 3), dtype=np.float64)
    spa_static_aligned = _apply_similarity_points(spa_static_raw, sim)

    def _dynamic_points_for_frame(frame_id: int) -> np.ndarray:
        local_i = _local_index_for_frame(frame_ids, frame_id)
        depth = depths[local_i]
        K_spa = intrs[local_i if intrs.shape[0] > 1 else 0]
        image_name = next((name for name, info in gt.items() if int(info.get("id", -1)) == frame_id), None)
        dynamic_mask = _read_dynamic_mask(mask_dir, frame_id, depth.shape, args.mask_threshold, image_name=image_name)
        dyn_raw = _lift_depth_to_world(
            depth,
            K_spa,
            spa_c2ws[frame_id],
            args.min_depth,
            args.max_depth,
            args.point_stride,
            0,
            pixel_mask=dynamic_mask,
        )
        return _apply_similarity_points(dyn_raw, sim)

    spa_dynamic_aligned = _dynamic_points_for_frame(frame)

    if args.render_train_views:
        try:
            import open3d as o3d
            from open3d.visualization import rendering as o3d_render
        except Exception as exc:
            print(f"[render] Open3D offscreen renderer not available: {exc}")
        else:
            render_dir = out_dir / "train_renders"
            render_dir.mkdir(parents=True, exist_ok=True)
            point_size = float(args.render_point_size)
            static_cols = np.tile(np.array([[0.65, 0.65, 0.65]], dtype=np.float64), (spa_static_aligned.shape[0], 1))
            for frame_id in used:
                local_i = _local_index_for_frame(frame_ids, frame_id)
                depth = depths[local_i]
                K_spa = intrs[local_i if intrs.shape[0] > 1 else 0]
                image_name = next((name for name, info in gt.items() if int(info.get("id", -1)) == frame_id), None)
                if image_name is None:
                    continue
                rgb_path = scene_dir / "images" / Path(image_name).name
                rgb = cv2.imread(str(rgb_path), cv2.IMREAD_UNCHANGED)
                if rgb is None:
                    continue
                dyn_pts = _dynamic_points_for_frame(frame_id)
                dyn_cols = np.tile(np.array([[1.0, 0.85, 0.05]], dtype=np.float64), (dyn_pts.shape[0], 1))
                if spa_static_aligned.size and dyn_pts.size:
                    pts = np.concatenate([spa_static_aligned, dyn_pts], axis=0)
                    cols = np.concatenate([static_cols, dyn_cols], axis=0)
                elif spa_static_aligned.size:
                    pts = spa_static_aligned
                    cols = static_cols
                else:
                    pts = dyn_pts
                    cols = dyn_cols

                if pts.shape[0] == 0:
                    continue

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts)
                pcd.colors = o3d.utility.Vector3dVector(np.clip(cols, 0.0, 1.0))

                height, width = rgb.shape[0], rgb.shape[1]
                renderer = o3d_render.OffscreenRenderer(width, height)
                mat = o3d_render.MaterialRecord()
                mat.shader = "defaultUnlit"
                mat.point_size = point_size
                renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])
                renderer.scene.add_geometry("pcd", pcd, mat)
                intrinsic = o3d.camera.PinholeCameraIntrinsic(
                    width,
                    height,
                    float(K_spa[0, 0]),
                    float(K_spa[1, 1]),
                    float(K_spa[0, 2]),
                    float(K_spa[1, 2]),
                )
                c2w_aligned = _apply_similarity_c2w(spa_c2ws[frame_id], sim)
                w2c = np.linalg.inv(c2w_aligned)
                renderer.setup_camera(intrinsic, w2c)
                img = renderer.render_to_image()
                out_path = render_dir / f"frame{frame_id:06d}_spatracker.png"
                o3d.io.write_image(str(out_path), img)
                renderer.scene.clear_geometry()
                renderer.release()

    _, gt_info = by_id[frame]
    gt_depth_path = _prefer_npy_depth_path(scene_dir / "depths" / f"depth{frame:06d}.png")
    gt_depth = _read_depth(gt_depth_path, args.gt_scale)
    K_gt = _scale_K(np.asarray(gt_info["K"], dtype=np.float64), image_size, (gt_depth.shape[1], gt_depth.shape[0]))
    gt_points = np.zeros((0, 3), dtype=np.float64)
    if args.write_gt_pointcloud:
        gt_points = _lift_depth_to_world(
            gt_depth,
            K_gt,
            gt_c2ws[frame],
            args.min_depth,
            args.max_depth,
            args.gt_point_stride,
            0,
            pixel_mask=None,
        )

    frame_stride = max(1, int(args.camera_stride))
    frustums = []
    local_by_frame = {int(fid): i for i, fid in enumerate(frame_ids)}
    for i, frame_id in enumerate(used):
        if i % frame_stride != 0 and frame_id != frame:
            continue
        local_i = local_by_frame[frame_id]
        _, info_i = by_id[frame_id]
        gt_image_size = _image_size(images[frame_id])
        K_gt_i = np.asarray(info_i["K"], dtype=np.float64)
        K_spa_i = intrs[local_i if intrs.shape[0] > 1 else 0]
        spa_h, spa_w = depths[local_i].shape
        frustums.append((_camera_frustum(gt_c2ws[frame_id], K_gt_i, gt_image_size[0], gt_image_size[1], args.camera_scale), (0.05, 0.35, 1.0)))
        frustums.append((_camera_frustum(spa_c2ws[frame_id], K_spa_i, spa_w, spa_h, args.camera_scale), (1.0, 0.15, 0.05)))
        frustums.append((_camera_frustum(_apply_similarity_c2w(spa_c2ws[frame_id], sim), K_spa_i, spa_w, spa_h, args.camera_scale), (0.05, 0.9, 0.15)))
    frustum_input = [(pts, lines, color) for (pts, lines), color in frustums]
    cam_lines = _make_lineset(frustum_input)

    import open3d as o3d

    o3d.io.write_line_set(str(out_dir / "camera_frustums_gt_blue_spa_raw_red_spa_aligned_green.ply"), cam_lines)
    pcds = []
    if args.write_gt_pointcloud:
        pcds.append(_write_pcd(out_dir / f"frame{frame:06d}_gt_depth_cloud_blue.ply", gt_points, (0.05, 0.35, 1.0), args.max_points, 1))
    if args.write_raw_pointcloud:
        pcds.append(_write_pcd(out_dir / f"frame{frame:06d}_spa_pointmap_raw_red.ply", spa_points_raw, (1.0, 0.15, 0.05), args.max_points, 2))
        pcds.append(_write_pcd(out_dir / f"frame{frame:06d}_spa_pointmap_aligned_green.ply", spa_points_aligned, (0.05, 0.9, 0.15), args.max_points, 5))
    pcds.extend(
        [
            _write_pcd(out_dir / "spa_static_aligned_gray.ply", spa_static_aligned, (0.65, 0.65, 0.65), args.max_points, 3),
            _write_pcd(out_dir / f"frame{frame:06d}_spa_dynamic_aligned_yellow.ply", spa_dynamic_aligned, (1.0, 0.85, 0.05), args.max_points, 4),
        ]
    )
    summary = {
        "scene_dir": str(scene_dir),
        "result_npz": str(result_npz),
        "frame": frame,
        "matched_cameras": len(used),
        "similarity_est_to_gt": {
            "scale": float(sim[0]),
            "R": sim[1].tolist(),
            "t": sim[2].tolist(),
            "center_rmse": center_rmse,
        },
        "points": {
            "gt_depth": int(gt_points.shape[0]),
            "spa_pointmap_raw": int(spa_points_raw.shape[0]),
            "spa_static_raw": int(spa_static_raw.shape[0]),
            "spa_dynamic_raw": int(spa_dynamic_aligned.shape[0]),
        },
        "outputs": {
            "camera_frustums": str(out_dir / "camera_frustums_gt_blue_spa_raw_red_spa_aligned_green.ply"),
            "gt_depth_cloud": str(out_dir / f"frame{frame:06d}_gt_depth_cloud_blue.ply") if args.write_gt_pointcloud else None,
            "spa_pointmap_raw": str(out_dir / f"frame{frame:06d}_spa_pointmap_raw_red.ply") if args.write_raw_pointcloud else None,
            "spa_pointmap_aligned": str(out_dir / f"frame{frame:06d}_spa_pointmap_aligned_green.ply") if args.write_raw_pointcloud else None,
            "spa_static_aligned": str(out_dir / "spa_static_aligned_gray.ply"),
            "spa_dynamic_aligned": str(out_dir / f"frame{frame:06d}_spa_dynamic_aligned_yellow.ply"),
        },
        "legend": {
            "blue": "GT cameras; GT depth cloud only when --write-gt-pointcloud is set",
            "red": "raw SpaTracker cameras; raw pointmap cloud only when --write-raw-pointcloud is set",
            "green": "SpaTracker cameras/pointmap after evaluator Sim3 alignment to GT",
            "gray": "aligned SpaTracker static-mask pointmap points",
            "yellow": "aligned SpaTracker dynamic-mask pointmap points",
        },
    }
    (out_dir / "debug_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.show:
        static_pcd = _make_pcd(spa_static_aligned, (0.65, 0.65, 0.65), args.max_points, 42)
        dynamic_pcd = _make_pcd(spa_dynamic_aligned, (1.0, 0.85, 0.05), args.max_points, 43)
        geoms = [cam_lines, static_pcd, dynamic_pcd]
        if args.write_gt_pointcloud:
            geoms.append(_make_pcd(gt_points, (0.05, 0.35, 1.0), args.max_points, 44))

        state = {"frame": frame}

        def _refresh(vis):
            dyn_pts = _dynamic_points_for_frame(state["frame"])
            dynamic_pcd.points = o3d.utility.Vector3dVector(dyn_pts)
            dynamic_pcd.paint_uniform_color([1.0, 0.85, 0.05])
            vis.update_geometry(dynamic_pcd)
            print(f"[Open3D] frame={state['frame']} dynamic_points={dyn_pts.shape[0]}")
            return False

        def _on_space(vis):
            idx = used.index(state["frame"]) if state["frame"] in used else 0
            state["frame"] = used[(idx + 1) % len(used)]
            return _refresh(vis)

        def _on_b(vis):
            idx = used.index(state["frame"]) if state["frame"] in used else 0
            state["frame"] = used[(idx - 1) % len(used)]
            return _refresh(vis)

        def _on_q(vis):
            vis.close()
            return False

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name=f"SpaTracker debug {scene_dir.name} (static all frames)")
        for geom in geoms:
            vis.add_geometry(geom)
        vis.register_key_callback(ord(" "), _on_space)
        vis.register_key_callback(ord("b"), _on_b)
        vis.register_key_callback(ord("B"), _on_b)
        vis.register_key_callback(ord("q"), _on_q)
        vis.register_key_callback(ord("Q"), _on_q)
        vis.register_key_callback(256, _on_q)
        _refresh(vis)
        print("[Open3D] space=next dynamic frame, b=prev frame, q=quit")
        vis.run()
        vis.destroy_window()


if __name__ == "__main__":
    main()
