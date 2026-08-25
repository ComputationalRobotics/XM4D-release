#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from eval_depth import _prefer_npy_depth_path, _read_depth, _normalizer
from eval_depth_depthanything import (
    _gt_valid_mask,
    _lift_depth_to_world,
    _load_gt,
    _load_novel_metadata,
    _read_dynamic_mask,
    _render_depth_from_points,
    _scale_K,
)
from eval_xm import compute_our_metrics, covariance_offset_for_gt_type, eval_mosca_style_with_evo

REPLICA_RENDER_DEPTH_SCALE = 65535.0 * 0.1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate SpatialTrackerV2 camera and depth outputs.")
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--result-npz", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--gt-scale", type=float, default=REPLICA_RENDER_DEPTH_SCALE)
    parser.add_argument("--normalize-stat", choices=["median", "mean"], default="median")
    parser.add_argument("--min-depth", type=float, default=1e-6)
    parser.add_argument("--max-depth", type=float, default=-1.0)
    parser.add_argument("--point-stride", type=int, default=1)
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument("--static-point-stride", type=int, default=1)
    parser.add_argument("--static-frame-stride", type=int, default=1)
    parser.add_argument("--static-max-points", type=int, default=0)
    parser.add_argument("--mask-dir", type=Path, default=None)
    parser.add_argument("--mask-threshold", type=float, default=127.0)
    parser.add_argument("--alpha-threshold", type=float, default=0.1)
    parser.add_argument("--alpha-density-radius", type=int, default=20)
    parser.add_argument("--alpha-density-scale", type=float, default=1.0)
    parser.add_argument("--max-points", type=int, default=0)
    parser.add_argument("--visibility-threshold", type=float, default=0.5)
    parser.add_argument(
        "--use-gt-intrinsics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Recompute GT intrinsics scaled to SpaTracker pointmap resolution when lifting static pointmaps. Default uses result.npz intrinsics so static pointmap holes match dynamic tracks.",
    )
    parser.add_argument("--json", action="store_true")
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


def _gt_by_id(gt: dict[str, dict[str, Any]]) -> dict[int, tuple[str, dict[str, Any]]]:
    return {int(v["id"]): (k, v) for k, v in gt.items()}


def _c2w_from_w2c(extrinsic: np.ndarray) -> np.ndarray:
    ext = np.asarray(extrinsic, dtype=np.float64).reshape(4, 4)
    return np.linalg.inv(ext)


def _load_result(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    required = ["extrinsics", "intrinsics", "coords", "depths"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"{path} missing keys: {missing}")
    return {k: data[k] for k in data.files}


def _gt_c2w(info: dict[str, Any]) -> np.ndarray:
    R_w2c = np.asarray(info["R"], dtype=np.float64).reshape(3, 3)
    t_w2c = np.asarray(info["t"], dtype=np.float64).reshape(3)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R_w2c.T
    c2w[:3, 3] = -R_w2c.T @ t_w2c
    return c2w


def _cap_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if int(max_points) <= 0 or points.shape[0] <= int(max_points):
        return points
    rng = np.random.default_rng(int(seed))
    keep = rng.choice(points.shape[0], size=int(max_points), replace=False)
    return points[keep]


def _transform_c2w_similarity(c2w: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = np.asarray(c2w, dtype=np.float64).copy()
    out[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3) @ out[:3, :3]
    out[:3, 3] = float(scale) * (np.asarray(R, dtype=np.float64).reshape(3, 3) @ out[:3, 3]) + np.asarray(t, dtype=np.float64).reshape(3)
    return out


def _unit_scale_alignment_from_centers(src_centers: np.ndarray, dst_centers: np.ndarray, R: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(src_centers, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst_centers, dtype=np.float64).reshape(-1, 3)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.mean(dst - (R @ src.T).T, axis=0)
    return 1.0, R, t


def _spatracker_novel_c2w_from_base(rec: Any, gt_base_c2w: np.ndarray, spa_base_c2w_aligned: np.ndarray, offset_scale: float) -> np.ndarray:
    delta_cam = gt_base_c2w[:3, :3].T @ (rec.c2w[:3, 3] - gt_base_c2w[:3, 3])
    rel_rot = gt_base_c2w[:3, :3].T @ rec.c2w[:3, :3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = spa_base_c2w_aligned[:3, :3] @ rel_rot
    out[:3, 3] = spa_base_c2w_aligned[:3, 3] + float(offset_scale) * (spa_base_c2w_aligned[:3, :3] @ delta_cam)
    return out


def _track_points_for_frame(data: dict[str, np.ndarray], local_idx: int, align, args: argparse.Namespace) -> np.ndarray:
    coords = np.asarray(data["coords"], dtype=np.float64)
    if local_idx < 0 or local_idx >= coords.shape[0]:
        return np.zeros((0, 3), dtype=np.float64)
    points = coords[int(local_idx), :, :3]
    valid = _track_valid_mask(data, args)[int(local_idx)]
    points = points[valid]
    if points.shape[0] == 0:
        return points
    s_align, R_align, t_align = align
    points = s_align * (R_align @ points.T).T + t_align.reshape(1, 3)
    return _cap_points(points, int(args.max_points), seed=int(local_idx))


def _track_valid_mask(data: dict[str, np.ndarray], args: argparse.Namespace) -> np.ndarray:
    coords = np.asarray(data["coords"], dtype=np.float64)
    if coords.ndim != 3 or coords.shape[-1] < 3:
        return np.zeros((0, 0), dtype=bool)
    valid = np.isfinite(coords[..., :3]).all(axis=2)
    if "visibs" in data:
        vis = np.asarray(data["visibs"])
        if vis.ndim == 3:
            vis = vis[..., 0]
        if vis.ndim == 2:
            rows = min(valid.shape[0], vis.shape[0])
            cols = min(valid.shape[1], vis.shape[1])
            valid[:rows, :cols] &= vis[:rows, :cols] >= float(args.visibility_threshold)
            if cols < valid.shape[1]:
                valid[:rows, cols:] = False
    if "conf_pred" in data:
        conf = np.asarray(data["conf_pred"])
        if conf.ndim == 3:
            conf = conf[..., 0]
        if conf.ndim == 2:
            rows = min(valid.shape[0], conf.shape[0])
            cols = min(valid.shape[1], conf.shape[1])
            valid[:rows, :cols] &= conf[:rows, :cols] >= float(args.visibility_threshold)
            if cols < valid.shape[1]:
                valid[:rows, cols:] = False
    return valid


def _aligned_track_tensor(data: dict[str, np.ndarray], align) -> np.ndarray:
    coords = np.asarray(data["coords"], dtype=np.float64)[..., :3]
    if coords.ndim != 3:
        return np.zeros((0, 0, 3), dtype=np.float64)
    s_align, R_align, t_align = align
    flat = coords.reshape(-1, 3)
    out = np.full_like(flat, np.nan, dtype=np.float64)
    valid = np.isfinite(flat).all(axis=1)
    if np.any(valid):
        out[valid] = float(s_align) * (np.asarray(R_align, dtype=np.float64) @ flat[valid].T).T + np.asarray(t_align, dtype=np.float64).reshape(1, 3)
    return out.reshape(coords.shape)


def _image_names_by_frame(gt: dict[str, dict[str, Any]]) -> dict[int, str]:
    return {int(info["id"]): name for name, info in gt.items() if "id" in info}


def _mask_dir(scene_dir: Path, args: argparse.Namespace) -> Path:
    return (args.mask_dir or (scene_dir / "masks")).resolve()


def _static_world_points(
    scene_dir: Path,
    gt: dict[str, dict[str, Any]],
    data: dict[str, np.ndarray],
    align,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[str, Any]]:
    depths = np.asarray(data["depths"], dtype=np.float64)
    intrs = np.asarray(data["intrinsics"], dtype=np.float64)
    extrinsics = np.asarray(data["extrinsics"], dtype=np.float64)
    frame_ids = np.asarray(data.get("frame_ids", np.arange(depths.shape[0])), dtype=np.int64).reshape(-1)
    if frame_ids.shape[0] != depths.shape[0]:
        frame_ids = np.arange(depths.shape[0], dtype=np.int64)
    image_names = _image_names_by_frame(gt)
    by_id = _gt_by_id(gt)
    images = _image_files(scene_dir)
    mask_dir = _mask_dir(scene_dir, args)
    s_align, R_align, t_align = align
    chunks = []
    per_frame: dict[int, np.ndarray] = {}
    frames_used = 0
    frame_stride = max(1, int(args.static_frame_stride))
    for local_idx in range(0, min(depths.shape[0], extrinsics.shape[0]), frame_stride):
        frame_id = int(frame_ids[local_idx])
        depth = depths[local_idx]
        if bool(args.use_gt_intrinsics) and frame_id in by_id and 0 <= frame_id < len(images):
            _, info = by_id[frame_id]
            K = _scale_K(np.asarray(info["K"], dtype=np.float64), _image_size(images[frame_id]), (depth.shape[1], depth.shape[0]))
        else:
            K = intrs[local_idx if intrs.shape[0] > 1 else 0]
        dynamic_mask = _read_dynamic_mask(
            mask_dir,
            frame_id,
            depth.shape,
            args.mask_threshold,
            image_name=image_names.get(frame_id),
        )
        static_mask = ~dynamic_mask
        c2w_est = _c2w_from_w2c(extrinsics[local_idx])
        pts = _lift_depth_to_world(
            depth,
            K,
            c2w_est,
            args.min_depth,
            args.max_depth,
            args.static_point_stride,
            0,
            pixel_mask=static_mask,
        )
        if pts.size:
            pts = s_align * (R_align @ pts.T).T + t_align.reshape(1, 3)
            chunks.append(pts)
            per_frame[frame_id] = pts
        else:
            per_frame[frame_id] = np.zeros((0, 3), dtype=np.float64)
        frames_used += 1
    if not chunks:
        points = np.zeros((0, 3), dtype=np.float64)
    else:
        points = np.concatenate(chunks, axis=0)
        points = _cap_points(points, int(args.static_max_points), seed=0)
    return points, per_frame, {
        "frames_used": int(frames_used),
        "points": int(points.shape[0]),
        "per_frame_points_min": int(min((p.shape[0] for p in per_frame.values()), default=0)),
        "per_frame_points_max": int(max((p.shape[0] for p in per_frame.values()), default=0)),
        "point_stride": int(args.static_point_stride),
        "frame_stride": int(args.static_frame_stride),
        "max_points": int(args.static_max_points),
        "mask_dir": str(mask_dir),
        "intrinsics_source": "gt_scaled_to_spatracker_depth" if bool(args.use_gt_intrinsics) else "spatracker_predicted",
    }


def _render_combined_depth(
    static_points: np.ndarray,
    dynamic_points: np.ndarray,
    target_c2w: np.ndarray,
    K: np.ndarray,
    shape: tuple[int, int],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    if static_points.size and dynamic_points.size:
        points = np.concatenate([static_points, dynamic_points], axis=0)
    elif static_points.size:
        points = static_points
    else:
        points = dynamic_points
    if points.size == 0:
        empty = np.zeros(shape, dtype=np.float64)
        return empty, empty.copy(), 0, 0
    est_depth, alpha = _render_depth_from_points(
        points,
        target_c2w,
        K,
        shape[0],
        shape[1],
        args.min_depth,
        args.max_depth,
        args.point_radius,
        args.alpha_density_radius,
        args.alpha_density_scale,
    )
    return est_depth, alpha, int(static_points.shape[0]), int(dynamic_points.shape[0])


def _full_pointmap_training_check(scene_dir: Path, gt: dict[str, dict[str, Any]], data: dict[str, np.ndarray], align, args: argparse.Namespace) -> dict[str, Any]:
    depths = np.asarray(data["depths"], dtype=np.float64)
    intrs = np.asarray(data["intrinsics"], dtype=np.float64)
    extrinsics = np.asarray(data["extrinsics"], dtype=np.float64)
    frame_ids = np.asarray(data.get("frame_ids", np.arange(depths.shape[0])), dtype=np.int64).reshape(-1)
    if depths.shape[0] == 0:
        return {"passed": False, "reason": "no_depths"}
    local_idx = 0
    frame_id = int(frame_ids[local_idx]) if frame_ids.shape[0] == depths.shape[0] else 0
    by_id = _gt_by_id(gt)
    if frame_id not in by_id:
        return {"passed": False, "reason": f"missing_gt_frame_{frame_id}"}
    depth = depths[local_idx]
    pts = _lift_depth_to_world(
        depth,
        intrs[local_idx if intrs.shape[0] > 1 else 0],
        _c2w_from_w2c(extrinsics[local_idx]),
        args.min_depth,
        args.max_depth,
        1,
        0,
        pixel_mask=None,
    )
    est_depth, _ = _render_depth_from_points(
        pts,
        _c2w_from_w2c(extrinsics[local_idx]),
        intrs[local_idx if intrs.shape[0] > 1 else 0],
        depth.shape[0],
        depth.shape[1],
        args.min_depth,
        args.max_depth,
        0,
        0,
        1.0,
    )
    gt_valid = np.isfinite(depth) & (depth > float(args.min_depth))
    rendered_valid = np.isfinite(est_depth) & (est_depth > float(args.min_depth))
    coverage = float(np.count_nonzero(rendered_valid & gt_valid) / max(np.count_nonzero(gt_valid), 1))
    return {
        "passed": bool(coverage >= 0.999),
        "frame": int(frame_id),
        "coverage": coverage,
        "valid_pixels": int(np.count_nonzero(rendered_valid & gt_valid)),
        "gt_valid_pixels": int(np.count_nonzero(gt_valid)),
        "points": int(pts.shape[0]),
        "point_radius": 0,
        "resolution": [int(depth.shape[0]), int(depth.shape[1])],
    }


def _camera_eval(scene_dir: Path, gt: dict[str, dict[str, Any]], data: dict[str, np.ndarray]) -> tuple[dict[str, Any], tuple[float, np.ndarray, np.ndarray], np.ndarray]:
    extrinsics = np.asarray(data["extrinsics"], dtype=np.float64)
    T = int(extrinsics.shape[0])
    frame_ids = np.asarray(data.get("frame_ids", np.arange(T)), dtype=np.int64).reshape(-1)
    if frame_ids.shape[0] != T:
        frame_ids = np.arange(T, dtype=np.int64)
    by_id = _gt_by_id(gt)

    R_gt_w2c = []
    t_gt_w2c = []
    R_est_c2w = []
    C_est = []
    C_gt = []
    used_ids = []
    c2w_est = []
    for local_idx, frame_id in enumerate(frame_ids):
        if int(frame_id) not in by_id:
            continue
        _, info = by_id[int(frame_id)]
        est_c2w = _c2w_from_w2c(extrinsics[local_idx])
        R_gt_w2c.append(np.asarray(info["R"], dtype=np.float64).reshape(3, 3))
        t_gt_w2c.append(np.asarray(info["t"], dtype=np.float64).reshape(3))
        R_est_c2w.append(est_c2w[:3, :3])
        C_est.append(est_c2w[:3, 3])
        C_gt.append(_gt_c2w(info)[:3, 3])
        c2w_est.append(est_c2w)
        used_ids.append(int(frame_id))

    if not used_ids:
        raise RuntimeError("No SpaTracker frames matched GT frame ids.")
    R_gt_arr = np.concatenate(R_gt_w2c, axis=1)
    t_gt_arr = np.stack(t_gt_w2c, axis=1)
    R_est_arr = np.concatenate(R_est_c2w, axis=1)
    C_est_arr = np.stack(C_est, axis=1)
    metrics, align = compute_our_metrics(
        R_gt_arr,
        t_gt_arr,
        R_est_arr,
        C_est_arr,
        covariance_offset=covariance_offset_for_gt_type("sintel"),
        visualize_alignment=False,
    )
    s_align, R_align, t_align = align
    render_align = _unit_scale_alignment_from_centers(np.asarray(C_est), np.asarray(C_gt), R_align)
    R_est_aligned = R_align @ R_est_arr
    C_est_aligned = s_align * (R_align @ C_est_arr) + t_align.reshape(3, 1)
    mosca_metrics = eval_mosca_style_with_evo(R_est_aligned, C_est_aligned, R_gt_arr, t_gt_arr) or {}
    result = {
        "scene_dir": str(scene_dir),
        "matched_frames": int(len(used_ids)),
        "frame_ids": used_ids,
        **metrics,
        **mosca_metrics,
        "render_alignment_raw_scale": float(s_align),
        "render_alignment_scale_used": float(render_align[0]),
    }
    return result, render_align, np.stack(c2w_est, axis=0)


def _training_depth_eval(
    scene_dir: Path,
    data: dict[str, np.ndarray],
    align,
    static_points: np.ndarray,
    static_meta: dict[str, Any],
    args: argparse.Namespace,
    est_to_gt_scale: float,
) -> dict[str, Any]:
    coords = np.asarray(data["coords"], dtype=np.float64)
    intrs = np.asarray(data["intrinsics"], dtype=np.float64)
    extrinsics = np.asarray(data["extrinsics"], dtype=np.float64)
    frame_ids = np.asarray(data.get("frame_ids", np.arange(coords.shape[0])), dtype=np.int64).reshape(-1)
    if frame_ids.shape[0] != coords.shape[0]:
        frame_ids = np.arange(coords.shape[0], dtype=np.int64)
    gt_meta = _load_gt(scene_dir)
    by_id = _gt_by_id(gt_meta)
    images = _image_files(scene_dir)
    gt_chunks = []
    est_chunks = []
    pairs = []
    out_dir = (args.output_dir or (scene_dir / "outputs" / "spatracker_eval")).resolve()
    train_depth_dir = out_dir / "training_depth"
    train_depth_dir.mkdir(parents=True, exist_ok=True)
    track_dir = out_dir / "dynamic_tracks"
    track_dir.mkdir(parents=True, exist_ok=True)
    coords_spatracker = coords[..., :3]
    coords_gt_world = _aligned_track_tensor(data, align)
    track_valid = _track_valid_mask(data, args)
    np.savez_compressed(
        track_dir / "dynamic_tracks_all.npz",
        coords_spatracker=coords_spatracker.astype(np.float32),
        coords_gt_world=coords_gt_world.astype(np.float32),
        valid=track_valid.astype(bool),
        frame_ids=frame_ids.astype(np.int64),
        queries_xyt=np.asarray(data["queries_xyt"], dtype=np.float32) if "queries_xyt" in data else np.empty((0, 3), dtype=np.float32),
        visibs=np.asarray(data["visibs"], dtype=np.float32) if "visibs" in data else np.empty((0,), dtype=np.float32),
        conf_pred=np.asarray(data["conf_pred"], dtype=np.float32) if "conf_pred" in data else np.empty((0,), dtype=np.float32),
        query_counts_per_frame=np.asarray(data["query_counts_per_frame"], dtype=np.int64) if "query_counts_per_frame" in data else np.empty((0,), dtype=np.int64),
        query_candidate_counts_per_frame=np.asarray(data["query_candidate_counts_per_frame"], dtype=np.int64) if "query_candidate_counts_per_frame" in data else np.empty((0,), dtype=np.int64),
        first_query_points=np.asarray(data["first_query_points"]) if "first_query_points" in data else np.array(0, dtype=np.int64),
        later_query_points=np.asarray(data["later_query_points"]) if "later_query_points" in data else np.array(0, dtype=np.int64),
    )
    for local_idx, frame_id in enumerate(frame_ids):
        gt_path = _prefer_npy_depth_path(scene_dir / "depths" / f"depth{int(frame_id):06d}.png")
        if not gt_path.is_file() or int(frame_id) not in by_id or int(frame_id) >= len(images):
            continue
        gt_depth = _read_depth(gt_path, args.gt_scale)
        points_world = _track_points_for_frame(data, local_idx, align, args)
        _, info = by_id[int(frame_id)]
        src_size = _image_size(images[int(frame_id)])
        K_target = _scale_K(np.asarray(info["K"], dtype=np.float64), src_size, (gt_depth.shape[1], gt_depth.shape[0]))
        target_c2w = _transform_c2w_similarity(_c2w_from_w2c(extrinsics[local_idx]), *align)
        # Use the aggregated all-frames static cloud (lifted via SpaTracker's own
        # extrinsics), rendered into this frame's camera -- matching the DA3 evaluator
        # and SpaTracker's own novel path. (Previously used only this frame's own static
        # points, which is trivial self-rendering of the input depth.)
        est_depth, alpha, static_count, dynamic_count = _render_combined_depth(
            static_points,
            points_world,
            target_c2w,
            K_target,
            gt_depth.shape,
            args,
        )
        valid = _gt_valid_mask(gt_depth, est_depth, args.min_depth, args.max_depth, alpha, args.alpha_threshold)
        pairs.append((int(frame_id), gt_depth, est_depth, alpha, valid, gt_path, static_count, dynamic_count))
        np.savez_compressed(
            track_dir / f"frame{int(frame_id):06d}_dynamic_track.npz",
            frame_id=np.array(int(frame_id), dtype=np.int64),
            local_idx=np.array(int(local_idx), dtype=np.int64),
            points_spatracker=coords_spatracker[local_idx].astype(np.float32),
            points_gt_world_full=coords_gt_world[local_idx].astype(np.float32),
            dynamic_valid_mask=track_valid[local_idx].astype(bool),
            points_gt_world=points_world.astype(np.float32),
            visibility=(
                np.asarray(data["visibs"])[local_idx].astype(np.float32)
                if "visibs" in data and local_idx < np.asarray(data["visibs"]).shape[0]
                else np.empty((0,), dtype=np.float32)
            ),
            confidence=(
                np.asarray(data["conf_pred"])[local_idx].astype(np.float32)
                if "conf_pred" in data and local_idx < np.asarray(data["conf_pred"]).shape[0]
                else np.empty((0,), dtype=np.float32)
            ),
        )
        if np.any(valid):
            gt_chunks.append(gt_depth[valid])
            est_chunks.append(est_depth[valid])
    if gt_chunks:
        gt_normer = _normalizer(np.concatenate(gt_chunks), args.normalize_stat)
        est_normer = _normalizer(np.concatenate(est_chunks), args.normalize_stat)
    else:
        gt_normer = 1.0
        est_normer = 1.0
    median_est_to_gt_scale = float(gt_normer / max(est_normer, 1e-12))
    est_to_gt_scale = float(est_to_gt_scale)
    total_abs = 0.0
    total_raw_abs = 0.0
    total_valid = 0
    frame_l1 = []
    frame_raw_l1 = []
    rows = []
    coverage_rows = []
    for frame_id, gt_depth, est_depth, alpha, valid, gt_path, static_count, dynamic_count in pairs:
        est_depth_scaled = est_depth * est_to_gt_scale
        valid = _gt_valid_mask(gt_depth, est_depth_scaled, args.min_depth, args.max_depth, alpha, args.alpha_threshold)
        np.save(train_depth_dir / f"frame{frame_id:06d}_pred.npy", est_depth.astype(np.float32))
        np.save(train_depth_dir / f"frame{frame_id:06d}_pred_unscaled.npy", est_depth.astype(np.float32))
        np.save(train_depth_dir / f"frame{frame_id:06d}_pred_camera_scaled.npy", est_depth_scaled.astype(np.float32))
        np.save(train_depth_dir / f"frame{frame_id:06d}_gt.npy", gt_depth.astype(np.float32))
        np.save(train_depth_dir / f"frame{frame_id:06d}_alpha.npy", alpha.astype(np.float32))
        np.save(train_depth_dir / f"frame{frame_id:06d}_valid.npy", valid.astype(bool))
        if not np.any(valid):
            coverage_rows.append({"frame": int(frame_id), "valid_px": 0, "coverage": 0.0, "static_points": int(static_count), "dynamic_points": int(dynamic_count)})
            continue
        raw_diff = np.abs(gt_depth[valid] - est_depth_scaled[valid])
        gt_median = _normalizer(gt_depth[valid], args.normalize_stat)
        diff = raw_diff / max(gt_median, 1e-12)
        total_abs += float(np.sum(diff))
        total_raw_abs += float(np.sum(raw_diff))
        total_valid += int(raw_diff.size)
        l1_frame = float(np.mean(diff))
        raw_l1_frame = float(np.mean(raw_diff))
        frame_l1.append(l1_frame)
        frame_raw_l1.append(raw_l1_frame)
        coverage = float(diff.size / max(np.count_nonzero(np.isfinite(gt_depth) & (gt_depth > float(args.min_depth))), 1))
        rows.append({
            "frame": int(frame_id),
            "l1": l1_frame,
            "raw_l1": raw_l1_frame,
            "gt_median": float(gt_median),
            "valid_px": int(diff.size),
            "coverage": coverage,
            "static_points": int(static_count),
            "dynamic_points": int(dynamic_count),
        })
        coverage_rows.append(rows[-1])
    return {
        "num_frames": int(len(frame_l1)),
        "num_valid_pixels": int(total_valid),
        "normalize_stat": args.normalize_stat,
        "gt_normalizer": float(gt_normer),
        "est_normalizer": float(est_normer),
        "median_est_to_gt_scale_for_debug": float(median_est_to_gt_scale),
        "est_to_gt_scale": float(est_to_gt_scale),
        "scale_source": "camera alignment scale: SpaTracker camera units -> GT camera units",
        "metric": "mean(abs(gt_depth - pred_scaled) / median(gt_depth_per_frame))",
        "l1": None if total_valid == 0 else float(total_abs / total_valid),
        "raw_l1": None if total_valid == 0 else float(total_raw_abs / total_valid),
        "l1_avg_frame": float(np.mean(frame_l1)) if frame_l1 else None,
        "raw_l1_avg_frame": float(np.mean(frame_raw_l1)) if frame_raw_l1 else None,
        "depth_dir": str(train_depth_dir.resolve()),
        "dynamic_track_dir": str(track_dir.resolve()),
        "dynamic_track_file": str((track_dir / "dynamic_tracks_all.npz").resolve()),
        "dynamic_track_shape": [int(coords_spatracker.shape[0]), int(coords_spatracker.shape[1]), int(coords_spatracker.shape[2])],
        "dynamic_track_valid_observations": int(np.count_nonzero(track_valid)),
        "visibility_threshold": float(args.visibility_threshold),
        "point_radius": int(args.point_radius),
        "alpha_threshold": float(args.alpha_threshold),
        "rows": rows,
        "coverage_rows": coverage_rows,
        "mean_coverage": float(np.mean([r["coverage"] for r in coverage_rows])) if coverage_rows else 0.0,
        "depth_source": "raw rendered SpaTracker camera-frame depth; apply est_to_gt_scale before comparing to GT",
        "static_cloud": static_meta,
}


def _novel_depth_eval(
    scene_dir: Path,
    gt: dict[str, dict[str, Any]],
    data: dict[str, np.ndarray],
    align,
    static_points: np.ndarray,
    static_meta: dict[str, Any],
    args: argparse.Namespace,
    out_dir: Path,
    est_to_gt_scale: float,
) -> dict[str, Any]:
    coords = np.asarray(data["coords"], dtype=np.float64)
    extrinsics = np.asarray(data["extrinsics"], dtype=np.float64)
    frame_ids = np.asarray(data.get("frame_ids", np.arange(coords.shape[0])), dtype=np.int64).reshape(-1)
    if frame_ids.shape[0] != coords.shape[0]:
        frame_ids = np.arange(coords.shape[0], dtype=np.int64)
    local_by_frame = {int(fid): i for i, fid in enumerate(frame_ids)}
    by_id = _gt_by_id(gt)
    records = _load_novel_metadata(scene_dir / "novel")
    alpha_density_radius = args.point_radius if int(args.alpha_density_radius) < 0 else int(args.alpha_density_radius)

    rendered = []
    novel_depth_dir = out_dir / "novel_depth"
    novel_depth_dir.mkdir(parents=True, exist_ok=True)
    for rec in records:
        if int(rec.base_frame) not in local_by_frame or int(rec.base_frame) not in by_id:
            continue
        local_idx = local_by_frame[int(rec.base_frame)]
        points_world = _track_points_for_frame(data, local_idx, align, args)
        gt_depth = _read_depth(rec.depth_path, args.gt_scale)
        _, base_info = by_id[int(rec.base_frame)]
        base_image = _image_files(scene_dir)[int(rec.base_frame)]
        base_size = _image_size(base_image)
        K_target = _scale_K(np.asarray(base_info["K"], dtype=np.float64), base_size, (gt_depth.shape[1], gt_depth.shape[0]))
        spa_base_aligned = _transform_c2w_similarity(_c2w_from_w2c(extrinsics[local_idx]), *align)
        target_c2w = _spatracker_novel_c2w_from_base(rec, _gt_c2w(base_info), spa_base_aligned, align[0])
        est_depth, alpha, static_count, dynamic_count = _render_combined_depth(
            static_points,
            points_world,
            target_c2w,
            K_target,
            gt_depth.shape,
            args,
        )
        valid = _gt_valid_mask(gt_depth, est_depth, args.min_depth, args.max_depth, alpha, args.alpha_threshold)
        rendered.append((int(rec.novel_id), gt_depth, est_depth, alpha, valid, static_count, dynamic_count))
    total_abs = 0.0
    total_raw_abs = 0.0
    total_valid = 0
    frame_l1 = []
    frame_raw_l1 = []
    rows = []
    for novel_id, gt_depth, est_depth, alpha, valid, static_count, dynamic_count in rendered:
        est_depth_scaled = est_depth * float(est_to_gt_scale)
        valid = _gt_valid_mask(gt_depth, est_depth_scaled, args.min_depth, args.max_depth, alpha, args.alpha_threshold)
        np.save(novel_depth_dir / f"novel{novel_id:06d}_pred.npy", est_depth.astype(np.float32))
        np.save(novel_depth_dir / f"novel{novel_id:06d}_pred_unscaled.npy", est_depth.astype(np.float32))
        np.save(novel_depth_dir / f"novel{novel_id:06d}_pred_camera_scaled.npy", est_depth_scaled.astype(np.float32))
        np.save(novel_depth_dir / f"novel{novel_id:06d}_gt.npy", gt_depth.astype(np.float32))
        np.save(novel_depth_dir / f"novel{novel_id:06d}_alpha.npy", alpha.astype(np.float32))
        np.save(novel_depth_dir / f"novel{novel_id:06d}_valid.npy", valid.astype(bool))
        if not np.any(valid):
            continue
        raw_diff = np.abs(gt_depth[valid] - est_depth_scaled[valid])
        gt_median = _normalizer(gt_depth[valid], args.normalize_stat)
        diff = raw_diff / max(gt_median, 1e-12)
        total_abs += float(np.sum(diff))
        total_raw_abs += float(np.sum(raw_diff))
        total_valid += int(raw_diff.size)
        l1_frame = float(np.mean(diff))
        raw_l1_frame = float(np.mean(raw_diff))
        frame_l1.append(l1_frame)
        frame_raw_l1.append(raw_l1_frame)
        rows.append({
            "novel_id": int(novel_id),
            "l1": l1_frame,
            "raw_l1": raw_l1_frame,
            "gt_median": float(gt_median),
            "valid_px": int(diff.size),
            "static_points": int(static_count),
            "dynamic_points": int(dynamic_count),
        })
    return {
        "num_frames": int(len(records)),
        "num_frames_used": int(len(frame_l1)),
        "num_valid_pixels": int(total_valid),
        "normalize_stat": args.normalize_stat,
        "est_to_gt_scale": float(est_to_gt_scale),
        "scale_source": "camera alignment scale reused for novel: SpaTracker camera units -> GT camera units",
        "metric": "mean(abs(gt_depth - pred_scaled) / median(gt_depth_per_view))",
        "l1": None if total_valid == 0 else float(total_abs / total_valid),
        "raw_l1": None if total_valid == 0 else float(total_raw_abs / total_valid),
        "l1_avg_frame": float(np.mean(frame_l1)) if frame_l1 else None,
        "raw_l1_avg_frame": float(np.mean(frame_raw_l1)) if frame_raw_l1 else None,
        "alpha_threshold": float(args.alpha_threshold),
        "depth_dir": str(novel_depth_dir.resolve()),
        "depth_source": "raw rendered SpaTracker camera-frame depth; apply est_to_gt_scale before comparing to GT",
        "visibility_threshold": float(args.visibility_threshold),
        "point_radius": int(args.point_radius),
        "rows": rows,
        "static_cloud": static_meta,
    }


def main() -> None:
    args = _parse_args()
    scene_dir = args.scene_dir.resolve()
    out_dir = (args.output_dir or (scene_dir / "outputs" / "spatracker_eval")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    result_npz = (args.result_npz or (scene_dir / "outputs" / "spatracker_input" / "results" / "result.npz")).resolve()
    data = _load_result(result_npz)
    gt = _load_gt(scene_dir)

    camera, align, c2w_est = _camera_eval(scene_dir, gt, data)
    camera_depth_scale = float(camera.get("scale_align") or camera.get("render_alignment_raw_scale") or 1.0)
    static_points, per_frame_static_points, static_meta = _static_world_points(scene_dir, gt, data, align, args)
    training_depth = _training_depth_eval(scene_dir, data, align, static_points, static_meta, args, camera_depth_scale)
    novel_depth = _novel_depth_eval(
        scene_dir,
        gt,
        data,
        align,
        static_points,
        static_meta,
        args,
        out_dir,
        camera_depth_scale,
    )
    synthetic_check = _full_pointmap_training_check(scene_dir, gt, data, align, args)
    result = {
        "scene_dir": str(scene_dir),
        "result_npz": str(result_npz),
        "camera": camera,
        "training_depth": training_depth,
        "novel_depth": novel_depth,
        "synthetic_full_pointmap_training_check": synthetic_check,
    }
    out_json = out_dir / "spatracker_eval_summary.json"
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (out_dir / "camera_accuracy.json").write_text(json.dumps({"evaluation": camera}, indent=2, sort_keys=True) + "\n")
    (out_dir / "depth_eval.json").write_text(json.dumps(training_depth, indent=2, sort_keys=True) + "\n")
    (out_dir / "novel_depth_eval.json").write_text(json.dumps(novel_depth, indent=2, sort_keys=True) + "\n")
    print(f"[spatracker-eval] saved {out_json}")
    if args.json:
        print("RESULT_JSON:", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
