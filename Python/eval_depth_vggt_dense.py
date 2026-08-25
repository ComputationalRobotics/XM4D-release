#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pycolmap

from eval_depth_depthanything import (
    NovelRecord,
    _cap_points,
    _c2w_from_gt,
    _colorize,
    _gt_valid_mask,
    _label,
    _load_gt,
    _load_novel_metadata,
    _mask_path_for_frame,
    _read_depth,
    _read_dynamic_mask,
    _render_depth_from_points,
    _scale_K,
    _train_records,
    _umeyama_similarity,
)


@dataclass(frozen=True)
class VGGTCamera:
    frame_id: int
    name: str
    c2w: np.ndarray
    K: np.ndarray
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Evaluate VGGT rendered depth by reading output_vggt/dense_point_cloud, "
            "aligning cameras, scaling GT depth into VGGT camera units, rendering the dense cloud, "
            "and comparing to GT depth."
        )
    )
    ap.add_argument("--scene-dir", type=Path, required=True)
    ap.add_argument("--vggt-dir", type=Path, default=None, help="Default: <scene-dir>/output_vggt")
    ap.add_argument("--dense-npz", type=Path, default=None, help="Default: <vggt-dir>/dense_point_cloud/dense_point_cloud.npz")
    ap.add_argument("--sparse-dir", type=Path, default=None, help="Default: <vggt-dir>/sparse")
    ap.add_argument("--novel-dir", type=Path, default=None, help="Default: <scene-dir>/novel")
    ap.add_argument("--eval-split", choices=["train", "novel"], default="novel")
    ap.add_argument("--gt-depth-dir", type=Path, default=None, help="Default: <scene-dir>/depths")
    ap.add_argument("--gt-scale", type=float, default=6553.5)
    ap.add_argument("--min-depth", type=float, default=1e-6)
    ap.add_argument("--max-depth", type=float, default=-1.0)
    ap.add_argument("--point-stride", type=int, default=4, help="Sample every Nth VGGT dense pixel.")
    ap.add_argument("--dynamic-point-stride", type=int, default=1)
    ap.add_argument("--static-frame-stride", type=int, default=1)
    ap.add_argument("--static-max-points", type=int, default=2000000)
    ap.add_argument("--max-points", type=int, default=0, help="Cap combined render cloud per target; <=0 disables.")
    ap.add_argument("--point-radius", type=int, default=1)
    ap.add_argument("--alpha-threshold", type=float, default=0.05)
    ap.add_argument("--alpha-density-radius", type=int, default=1)
    ap.add_argument("--alpha-density-scale", type=float, default=1.0)
    ap.add_argument("--mask-dir", type=Path, default=None)
    ap.add_argument("--mask-threshold", type=float, default=127.0)
    ap.add_argument("--output-json", type=Path, default=None)
    ap.add_argument("--vis-out", type=Path, default=None)
    ap.add_argument("--rendered-depth-dir", type=Path, default=None)
    ap.add_argument("--depth-png-dir", type=Path, default=None)
    ap.add_argument("--save-depth-png", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--vis-frame", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    return ap.parse_args()


def _image_cam_from_world(image: Any) -> Any:
    pose = image.cam_from_world
    return pose() if callable(pose) else pose


def _load_scene_gt(scene_dir: Path) -> dict[str, dict[str, Any]]:
    try:
        return _load_gt(scene_dir)
    except ValueError:
        if (scene_dir / "colmap" / "sparse").is_dir():
            from eval_colmap import load_colmap_gt

            return load_colmap_gt(scene_dir)
        raise


def _frame_id_from_name(name: str) -> int:
    from eval_depth_depthanything import _frame_id_from_name as impl

    return impl(name)


def _camera_K(camera: Any) -> np.ndarray:
    params = np.asarray(camera.params, dtype=np.float64)
    K = np.eye(3, dtype=np.float64)
    model_obj = camera.model
    model = str(getattr(model_obj, "name", model_obj))
    if model.endswith("SIMPLE_PINHOLE"):
        K[0, 0] = params[0]
        K[1, 1] = params[0]
        K[0, 2] = params[1]
        K[1, 2] = params[2]
    elif model.endswith("PINHOLE"):
        K[0, 0] = params[0]
        K[1, 1] = params[1]
        K[0, 2] = params[2]
        K[1, 2] = params[3]
    else:
        raise ValueError(f"Unsupported VGGT/COLMAP camera model: {camera.model}")
    return K


def _load_vggt_cameras(sparse_dir: Path) -> dict[int, VGGTCamera]:
    rec = pycolmap.Reconstruction(str(sparse_dir))
    cams: dict[int, VGGTCamera] = {}
    for image_id in rec.images:
        image = rec.images[image_id]
        pose = _image_cam_from_world(image)
        R_w2c = np.asarray(pose.rotation.matrix(), dtype=np.float64)
        t_w2c = np.asarray(pose.translation, dtype=np.float64).reshape(3)
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = R_w2c.T
        c2w[:3, 3] = -R_w2c.T @ t_w2c
        camera = rec.cameras[image.camera_id]
        frame_id = _frame_id_from_name(image.name)
        cams[frame_id] = VGGTCamera(
            frame_id=frame_id,
            name=image.name,
            c2w=c2w,
            K=_camera_K(camera),
            width=int(camera.width),
            height=int(camera.height),
        )
    if not cams:
        raise ValueError(f"No registered VGGT cameras found in {sparse_dir}")
    return cams


def _camera_maps(cams: dict[int, VGGTCamera]) -> tuple[dict[str, VGGTCamera], dict[str, VGGTCamera]]:
    by_name: dict[str, VGGTCamera] = {}
    by_base: dict[str, VGGTCamera] = {}
    for cam in cams.values():
        by_name.setdefault(cam.name, cam)
        by_base.setdefault(Path(cam.name).name, cam)
    return by_name, by_base


def _camera_for_name_or_frame(
    image_name: str | None,
    frame_id: int,
    cams: dict[int, VGGTCamera],
    cams_by_name: dict[str, VGGTCamera],
    cams_by_base: dict[str, VGGTCamera],
) -> VGGTCamera | None:
    if image_name is not None:
        if image_name in cams_by_name:
            return cams_by_name[image_name]
        base = Path(image_name).name
        if base in cams_by_base:
            return cams_by_base[base]
    return cams.get(int(frame_id))


def _load_dense_npz(path: Path) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    data = np.load(path, allow_pickle=True)
    if "points_3d" not in data:
        raise KeyError(f"Missing points_3d in {path}")
    points = np.asarray(data["points_3d"], dtype=np.float64)
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError(f"Expected points_3d shape (T,H,W,3), got {points.shape}")
    valid = np.asarray(data["valid_mask"]).astype(bool) if "valid_mask" in data else None
    names = [str(x) for x in data["image_names"].tolist()] if "image_names" in data else [f"frame{i:06d}.jpg" for i in range(points.shape[0])]
    return points, valid, names


def _dense_index_maps(image_names: list[str]) -> tuple[dict[str, int], dict[str, int], dict[int, int]]:
    by_name: dict[str, int] = {}
    by_base: dict[str, int] = {}
    by_frame: dict[int, int] = {}
    for idx, name in enumerate(image_names):
        by_name.setdefault(name, idx)
        by_base.setdefault(Path(name).name, idx)
        try:
            by_frame.setdefault(_frame_id_from_name(name), idx)
        except ValueError:
            pass
    return by_name, by_base, by_frame


def _dense_index_for_camera(
    cam: VGGTCamera,
    image_names: list[str],
    by_name: dict[str, int],
    by_base: dict[str, int],
    by_frame: dict[int, int],
) -> int:
    if cam.name in by_name:
        return by_name[cam.name]
    base = Path(cam.name).name
    if base in by_base:
        return by_base[base]
    if int(cam.frame_id) in by_frame:
        return by_frame[int(cam.frame_id)]
    if 0 <= int(cam.frame_id) < len(image_names):
        return int(cam.frame_id)
    raise KeyError(f"Could not map VGGT camera {cam.name} frame_id={cam.frame_id} to dense point index.")


def _gt_to_vggt_similarity(gt: dict[str, dict[str, Any]], cams: dict[int, VGGTCamera]) -> tuple[float, np.ndarray, np.ndarray]:
    cams_by_name, cams_by_base = _camera_maps(cams)
    src = []
    dst = []
    for name, info in gt.items():
        frame = int(info.get("id", -1))
        cam = _camera_for_name_or_frame(name, frame, cams, cams_by_name, cams_by_base)
        if cam is not None:
            src.append(_c2w_from_gt(info)[:3, 3])
            dst.append(cam.c2w[:3, 3])
    if len(src) < 3:
        print("[vggt-camera] Too few GT/VGGT camera pairs; using no alignment for novel offsets.")
        return 1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    src_arr = np.stack(src, axis=0)
    dst_arr = np.stack(dst, axis=0)
    src_radius = np.linalg.norm(src_arr - src_arr.mean(axis=0, keepdims=True), axis=1)
    if not np.isfinite(src_radius).all() or float(np.mean(src_radius)) <= 1e-9:
        print("[vggt-camera] Degenerate GT camera motion; using depth scale 1.0.")
        return 1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    scale, R, t = _umeyama_similarity(src_arr, dst_arr)
    err = scale * (src_arr @ R.T) + t.reshape(1, 3) - dst_arr
    print(
        "[vggt-camera] GT->VGGT camera alignment "
        f"pairs={len(src)} scale={scale:.6g} center_rmse={float(np.sqrt(np.mean(np.sum(err * err, axis=1)))):.6g}"
    )
    return scale, R, t


def _novel_c2w_from_vggt_base(
    rec: NovelRecord,
    gt: dict[str, dict[str, Any]],
    cams: dict[int, VGGTCamera],
) -> np.ndarray:
    base_info = None
    for info in gt.values():
        if int(info.get("id", -1)) == int(rec.base_frame):
            base_info = info
            break
    if base_info is None or int(rec.base_frame) not in cams:
        raise KeyError(f"Could not build VGGT novel pose for base frame {rec.base_frame}")
    gt_base = _c2w_from_gt(base_info)
    vggt_base = cams[int(rec.base_frame)].c2w
    delta_cam = gt_base[:3, :3].T @ (rec.c2w[:3, 3] - gt_base[:3, 3])
    rel_rot = gt_base[:3, :3].T @ rec.c2w[:3, :3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = vggt_base[:3, :3] @ rel_rot
    out[:3, 3] = vggt_base[:3, 3] + (vggt_base[:3, :3] @ delta_cam)
    return out


def _points_for_mask(points_frame: np.ndarray, mask: np.ndarray | None, stride: int, valid_frame: np.ndarray | None) -> np.ndarray:
    stride = max(1, int(stride))
    pts = points_frame[::stride, ::stride]
    valid = np.isfinite(pts).all(axis=-1)
    if valid_frame is not None:
        valid &= valid_frame[::stride, ::stride].astype(bool)
    if mask is not None:
        if mask.shape != points_frame.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (points_frame.shape[1], points_frame.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        valid &= mask[::stride, ::stride].astype(bool)
    return pts[valid].reshape(-1, 3)


def _build_static_points(
    points: np.ndarray,
    valid_mask: np.ndarray | None,
    image_names: list[str],
    mask_dir: Path,
    args: argparse.Namespace,
) -> tuple[np.ndarray, int]:
    chunks = []
    frame_stride = max(1, int(args.static_frame_stride))
    for frame_idx in range(0, points.shape[0], frame_stride):
        dyn = _read_dynamic_mask(
            mask_dir,
            frame_idx,
            points.shape[1:3],
            args.mask_threshold,
            image_name=image_names[frame_idx] if frame_idx < len(image_names) else None,
        )
        static = ~dyn
        vf = valid_mask[frame_idx] if valid_mask is not None and frame_idx < valid_mask.shape[0] else None
        pts = _points_for_mask(points[frame_idx], static, args.point_stride, vf)
        if pts.size:
            chunks.append(pts)
    if not chunks:
        return np.zeros((0, 3), dtype=np.float64), 0
    merged = np.concatenate(chunks, axis=0)
    merged = _cap_points(merged, int(args.static_max_points), seed=0)
    return merged, len(range(0, points.shape[0], frame_stride))


def _dynamic_points_for_frame(
    points: np.ndarray,
    valid_mask: np.ndarray | None,
    image_names: list[str],
    mask_dir: Path,
    frame_idx: int,
    args: argparse.Namespace,
) -> np.ndarray:
    if frame_idx < 0 or frame_idx >= points.shape[0]:
        return np.zeros((0, 3), dtype=np.float64)
    path = _mask_path_for_frame(mask_dir, frame_idx, image_name=image_names[frame_idx] if frame_idx < len(image_names) else None)
    if path is None:
        return np.zeros((0, 3), dtype=np.float64)
    dyn = _read_dynamic_mask(mask_dir, frame_idx, points.shape[1:3], args.mask_threshold, image_name=image_names[frame_idx] if frame_idx < len(image_names) else None)
    vf = valid_mask[frame_idx] if valid_mask is not None and frame_idx < valid_mask.shape[0] else None
    return _points_for_mask(points[frame_idx], dyn, args.dynamic_point_stride, vf)


def _safe_median(values: np.ndarray) -> float:
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return 1.0
    med = float(np.median(values))
    return med if np.isfinite(med) and med > 1e-12 else 1.0


def _save_depth_png(path: Path, gt_depth_scaled: np.ndarray, est_depth: np.ndarray, alpha: np.ndarray, valid: np.ndarray, gt_median: float, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gt_valid = np.isfinite(gt_depth_scaled) & (gt_depth_scaled > 0)
    est_valid = np.isfinite(est_depth) & (est_depth > 0)
    scale_valid = valid if np.any(valid) else (gt_valid | est_valid)
    vmin = float(np.percentile(gt_depth_scaled[scale_valid], 2)) if np.any(scale_valid) else 0.0
    vmax = float(np.percentile(gt_depth_scaled[scale_valid], 98)) if np.any(scale_valid) else 1.0
    err = np.zeros_like(gt_depth_scaled)
    err[valid] = np.abs(gt_depth_scaled[valid] - est_depth[valid]) / max(float(gt_median), 1e-12)
    emax = float(np.percentile(err[valid], 98)) if np.any(valid) else 1.0
    canvas = np.vstack(
        [
            np.hstack([
                _label(_colorize(gt_depth_scaled, gt_valid, vmin, vmax), f"GT scaled {label}"),
                _label(_colorize(est_depth, est_valid, vmin, vmax), "VGGT dense render"),
            ]),
            np.hstack([
                _label(_colorize(err, valid, 0.0, max(emax, 1e-6)), "abs error / median(scaled GT)"),
                _label(_colorize(alpha, np.isfinite(alpha) & (alpha > 0), 0.0, 1.0), "render alpha"),
            ]),
        ]
    )
    cv2.imwrite(str(path), canvas)


def _evaluate(args: argparse.Namespace) -> dict[str, Any]:
    scene_dir = args.scene_dir.resolve()
    vggt_dir = (args.vggt_dir or (scene_dir / "output_vggt")).resolve()
    sparse_dir = (args.sparse_dir or (vggt_dir / "sparse")).resolve()
    dense_npz = (args.dense_npz or (vggt_dir / "dense_point_cloud" / "dense_point_cloud.npz")).resolve()
    novel_dir = (args.novel_dir or (scene_dir / "novel")).resolve()
    gt_depth_dir = (args.gt_depth_dir or (scene_dir / "depths")).resolve()
    rendered_depth_dir = (args.rendered_depth_dir or (vggt_dir / f"{args.eval_split}_render_depth_arrays")).resolve()
    rendered_depth_dir.mkdir(parents=True, exist_ok=True)
    depth_png_dir = (args.depth_png_dir or (vggt_dir / f"{args.eval_split}_render_depth_png")).resolve() if args.save_depth_png else None
    if depth_png_dir is not None:
        depth_png_dir.mkdir(parents=True, exist_ok=True)

    gt = _load_scene_gt(scene_dir)
    cams = _load_vggt_cameras(sparse_dir)
    cams_by_name, cams_by_base = _camera_maps(cams)
    points, valid_mask, image_names = _load_dense_npz(dense_npz)
    mask_dir = (args.mask_dir or (scene_dir / "masks")).resolve()
    gt_to_vggt_scale, _R, _t = _gt_to_vggt_similarity(gt, cams)
    gt_depth_scale_to_vggt = float(gt_to_vggt_scale)
    print(f"[vggt-{args.eval_split}] scaling GT depth into VGGT units by {gt_depth_scale_to_vggt:.6g}")
    dense_by_name, dense_by_base, dense_by_frame = _dense_index_maps(image_names)
    static_points, static_frames = _build_static_points(points, valid_mask, image_names, mask_dir, args)
    print(f"[vggt-{args.eval_split}] dense static cloud: {static_points.shape[0]} points from {static_frames} frame(s)")

    if args.eval_split == "train":
        gt_frame_ids = [int(info["id"]) for info in gt.values() if "id" in info]
        n_frames = (max(gt_frame_ids) + 1) if gt_frame_ids else points.shape[0]
        records: list[Any] = _train_records(scene_dir, gt, gt_depth_dir, n_frames)
    else:
        records = _load_novel_metadata(novel_dir)
    if not records:
        raise ValueError(f"No {args.eval_split} records found.")

    gt_chunks = []
    rendered_items = []
    dynamic_cache: dict[int, np.ndarray] = {}
    alpha_density_radius = args.point_radius if int(args.alpha_density_radius) < 0 else int(args.alpha_density_radius)

    def combined_points(frame_idx: int, dense_idx: int) -> tuple[np.ndarray, np.ndarray]:
        if dense_idx not in dynamic_cache:
            dynamic_cache[dense_idx] = _dynamic_points_for_frame(points, valid_mask, image_names, mask_dir, dense_idx, args)
            print(
                f"[vggt-{args.eval_split}] dense dynamic cloud "
                f"frame={frame_idx} dense_idx={dense_idx}: {dynamic_cache[dense_idx].shape[0]} points"
            )
        dyn = dynamic_cache[dense_idx]
        if static_points.size and dyn.size:
            merged = np.concatenate([static_points, dyn], axis=0)
        elif static_points.size:
            merged = static_points
        else:
            merged = dyn
        return _cap_points(merged, int(args.max_points), seed=frame_idx), dyn

    for rec in records:
        if args.eval_split == "train":
            frame_idx = int(rec.frame_id)
            cam = _camera_for_name_or_frame(rec.image_name, frame_idx, cams, cams_by_name, cams_by_base)
            if cam is None:
                continue
            target_c2w = cam.c2w
            dense_idx = _dense_index_for_camera(cam, image_names, dense_by_name, dense_by_base, dense_by_frame)
            gt_depth_path = rec.depth_path
            rec_id = frame_idx
            label = f"train{frame_idx:06d}"
        else:
            frame_idx = int(rec.base_frame)
            base_name = None
            for name, info in gt.items():
                if int(info.get("id", -1)) == frame_idx:
                    base_name = name
                    break
            cam = _camera_for_name_or_frame(base_name, frame_idx, cams, cams_by_name, cams_by_base)
            if cam is None:
                continue
            target_c2w = _novel_c2w_from_vggt_base(rec, gt, cams)
            dense_idx = _dense_index_for_camera(cam, image_names, dense_by_name, dense_by_base, dense_by_frame)
            gt_depth_path = rec.depth_path
            rec_id = int(rec.novel_id)
            label = f"novel{rec_id:06d}_base{frame_idx:06d}"

        render_points, _dyn = combined_points(frame_idx, dense_idx)
        gt_depth_raw = _read_depth(gt_depth_path, args.gt_scale)
        gt_depth = gt_depth_raw * gt_depth_scale_to_vggt
        h, w = gt_depth.shape
        K = _scale_K(cam.K, (cam.width, cam.height), (w, h))
        est_depth, alpha = _render_depth_from_points(
            render_points,
            target_c2w,
            K,
            h,
            w,
            args.min_depth,
            args.max_depth,
            args.point_radius,
            alpha_density_radius,
            args.alpha_density_scale,
        )
        valid = _gt_valid_mask(gt_depth, est_depth, args.min_depth, args.max_depth, alpha=alpha, alpha_threshold=args.alpha_threshold)
        print(
            f"[vggt-{args.eval_split}] {label} points={render_points.shape[0]} "
            f"dense_idx={dense_idx} render_px={int(np.count_nonzero(est_depth > 0))} "
            f"alpha_px={int(np.count_nonzero(alpha >= args.alpha_threshold))}"
        )
        np.save(rendered_depth_dir / f"{args.eval_split}{rec_id:06d}_depth.npy", est_depth.astype(np.float32))
        np.save(rendered_depth_dir / f"{args.eval_split}{rec_id:06d}_alpha.npy", alpha.astype(np.float32))
        np.save(rendered_depth_dir / f"{args.eval_split}{rec_id:06d}_gt.npy", gt_depth.astype(np.float32))
        np.save(rendered_depth_dir / f"{args.eval_split}{rec_id:06d}_gt_raw.npy", gt_depth_raw.astype(np.float32))
        np.save(rendered_depth_dir / f"{args.eval_split}{rec_id:06d}_valid.npy", valid.astype(bool))
        rendered_items.append((rec_id, gt_depth, est_depth, alpha, valid, label))
        if np.any(valid):
            gt_chunks.append(gt_depth[valid])

    if not gt_chunks:
        raise ValueError(f"No valid pixels found between rendered VGGT {args.eval_split} depths and GT.")
    gt_all = np.concatenate(gt_chunks)
    gt_median_depth = _safe_median(gt_all)

    total_abs = 0.0
    total_valid = 0
    frame_l1s = []
    vis_item = None
    for item in rendered_items:
        rec_id, gt_depth, est_depth, alpha, valid, label = item
        frame_gt_median = _safe_median(gt_depth[valid])
        if depth_png_dir is not None:
            _save_depth_png(depth_png_dir / f"{args.eval_split}_depth_eval_{rec_id:06d}.png", gt_depth, est_depth, alpha, valid, frame_gt_median, label)
        if np.any(valid):
            abs_err = np.abs(gt_depth[valid] - est_depth[valid])
            total_abs += float(np.sum(abs_err))
            total_valid += int(abs_err.size)
            frame_l1s.append(float(np.mean(abs_err / max(frame_gt_median, 1e-12))))
            if vis_item is None or int(rec_id) == int(args.vis_frame):
                vis_item = item

    if vis_item is not None:
        rec_id, gt_depth, est_depth, alpha, valid, label = vis_item
        vis_out = (args.vis_out or (vggt_dir / f"{args.eval_split}_render_depth_eval_vis.png")).resolve()
        _save_depth_png(vis_out, gt_depth, est_depth, alpha, valid, _safe_median(gt_depth[valid]), label)
    else:
        vis_out = (args.vis_out or (vggt_dir / f"{args.eval_split}_render_depth_eval_vis.png")).resolve()

    result = {
        "scene_dir": str(scene_dir),
        "vggt_dir": str(vggt_dir),
        "dense_npz": str(dense_npz),
        "sparse_dir": str(sparse_dir),
        "eval_split": args.eval_split,
        "novel_dir": str(novel_dir) if args.eval_split == "novel" else None,
        "gt_depth_dir": str(gt_depth_dir) if args.eval_split == "train" else None,
        "mask_dir": str(mask_dir),
        "num_frames": len(records),
        "num_frames_used": len(frame_l1s),
        "num_valid_pixels": total_valid,
        "metric": "mean(abs(gt_depth * gt_to_vggt_camera_scale - rendered_depth) / median(gt_depth * gt_to_vggt_camera_scale))",
        "evaluation_mask": "gt_valid_pixels_and_render_alpha",
        "depth_source": "vggt_dense_point_cloud_static_all_frames_dynamic_same_or_base_frame",
        "camera_source": "output_vggt/sparse",
        "camera_alignment": {"gt_to_vggt_scale": float(gt_to_vggt_scale)},
        "gt_depth_scale_to_vggt": gt_depth_scale_to_vggt,
        "novel_offset_scale": 1.0,
        "dense_point_map_shape": list(points.shape),
        "static_points": int(static_points.shape[0]),
        "static_frames": int(static_frames),
        "dynamic_frames_cached": int(len(dynamic_cache)),
        "point_stride": int(args.point_stride),
        "dynamic_point_stride": int(args.dynamic_point_stride),
        "static_frame_stride": int(args.static_frame_stride),
        "static_max_points": int(args.static_max_points),
        "max_points": int(args.max_points),
        "point_radius": int(args.point_radius),
        "alpha_density_radius": int(alpha_density_radius),
        "alpha_density_scale": float(args.alpha_density_scale),
        "alpha_threshold": float(args.alpha_threshold),
        "mask_threshold": float(args.mask_threshold),
        "gt_median_depth_scaled": float(gt_median_depth),
        "l1_depth_units": float(total_abs / max(total_valid, 1)),
        "l1": float((total_abs / max(total_valid, 1)) / max(gt_median_depth, 1e-12)),
        "l1_avg_frame": float(np.mean(frame_l1s)) if frame_l1s else None,
        "rendered_depth_dir": str(rendered_depth_dir),
        "depth_png_dir": str(depth_png_dir) if depth_png_dir is not None else None,
        "vis_out": str(vis_out),
    }
    return result


def main() -> None:
    args = parse_args()
    result = _evaluate(args)
    out = args.output_json or (Path(result["vggt_dir"]) / f"{args.eval_split}_render_depth_eval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"VGGT {args.eval_split} rendered-depth avg L1: {result['l1_avg_frame']:.6f} "
        f"(global {result['l1']:.6f}, frames {result['num_frames_used']}/{result['num_frames']})"
    )
    print(f"Wrote {out}")
    if args.json:
        print("RESULT_JSON:", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
