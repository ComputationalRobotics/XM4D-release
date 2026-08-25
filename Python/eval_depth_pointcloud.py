#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SINTEL_DPT_TAG = 202021.25


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render solver/EDG point clouds to depth and evaluate normalized L1 against GT depth."
    )
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--static-result", type=Path, default=None, help="Default: <scene>/outputs/result_XM.npz")
    parser.add_argument("--edg-vis", type=Path, default=None, help="Default: <scene>/outputs/edg_solution_vis.npz")
    parser.add_argument("--edg-result", type=Path, default=None, help="Fallback: <scene>/outputs/results_all.npz")
    parser.add_argument("--gt-depth-dir", type=Path, default=None, help="Default: <scene>/depths")
    parser.add_argument("--intrinsics", type=Path, default=None, help="Default: <scene>/outputs/depth_anything_3/intrinsics.npy")
    parser.add_argument("--gt-scale", type=float, default=1000.0)
    parser.add_argument("--normalize-stat", choices=["median", "mean"], default="median")
    parser.add_argument("--min-depth", type=float, default=1e-6)
    parser.add_argument("--max-depth", type=float, default=-1.0)
    parser.add_argument("--point-radius", type=int, default=1, help="Pixel splat radius for z-buffer rendering.")
    parser.add_argument("--max-static-points", type=int, default=0, help="0 keeps all static points.")
    parser.add_argument("--max-dynamic-points", type=int, default=0, help="0 keeps all dynamic points.")
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--dynamic-only", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--vis-out", type=Path, default=None)
    parser.add_argument("--rendered-depth-out", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _read_depth(path: Path, scale: float) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path).astype(np.float64)
    if suffix == ".npz":
        data = np.load(path)
        for key in ("depth", "depths", "depth_map", "depth_maps", "pred_depth", "arr_0"):
            if key in data:
                return data[key].astype(np.float64)
        raise KeyError(f"No depth key found in {path}")
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Could not read depth image: {path}")
        depth = depth.astype(np.float64)
        if depth.ndim == 3:
            depth = depth[..., 0]
        if scale > 0:
            depth = depth / float(scale)
        return depth
    if suffix == ".dpt":
        with path.open("rb") as f:
            tag = np.fromfile(f, dtype=np.float32, count=1)[0]
            if tag != SINTEL_DPT_TAG:
                raise ValueError(f"Wrong .dpt tag in {path}: {tag}")
            width = int(np.fromfile(f, dtype=np.int32, count=1)[0])
            height = int(np.fromfile(f, dtype=np.int32, count=1)[0])
            depth = np.fromfile(f, dtype=np.float32, count=width * height)
        return depth.reshape((height, width)).astype(np.float64)
    raise ValueError(f"Unsupported depth file: {path}")


def _depth_files(depth_dir: Path) -> list[Path]:
    exts = {".npy", ".npz", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dpt"}
    skip = {"depth_maps.npy", "conf_maps.npy", "intrinsics.npy", "extrinsics_c2w.npy"}
    return sorted(p for p in depth_dir.iterdir() if p.is_file() and p.suffix.lower() in exts and p.name not in skip)


def _image_files(scene_dir: Path) -> list[Path]:
    image_dir = scene_dir / "images"
    return sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"})


def _gt_depth_path(scene_dir: Path, gt_depth_dir: Path, frame_index: int) -> Path:
    images = _image_files(scene_dir)
    gt_files = _depth_files(gt_depth_dir)
    if frame_index < 0 or frame_index >= len(images):
        raise IndexError(f"frame_index={frame_index} outside images [0,{len(images)-1}]")
    by_stem = {p.stem: p for p in gt_files}
    image_stem = images[frame_index].stem
    if image_stem in by_stem:
        return by_stem[image_stem]
    if frame_index < len(gt_files):
        return gt_files[frame_index]
    raise FileNotFoundError(f"No GT depth for frame {frame_index} in {gt_depth_dir}")


def _load_intrinsics(scene_dir: Path, intrinsics_path: Path | None, frame_index: int) -> np.ndarray:
    path = intrinsics_path or (scene_dir / "outputs" / "depth_anything_3" / "intrinsics.npy")
    if path.is_file():
        intr = np.load(path)
        if intr.ndim == 3:
            if frame_index >= intr.shape[0]:
                raise IndexError(f"frame_index={frame_index} outside intrinsics length {intr.shape[0]}")
            return intr[frame_index].astype(np.float64)
        if intr.shape == (3, 3):
            return intr.astype(np.float64)
        raise ValueError(f"Unexpected intrinsics shape {intr.shape} in {path}")

    # Last-resort dataset GT readers. Kept lazy so this script works without all optional deps.
    dataset_cat = scene_dir.parent.name
    if dataset_cat == "TUM":
        from utils.readgt_TUM import load_tum_gt

        gt = load_tum_gt(str(scene_dir))
    elif dataset_cat == "Sintel":
        from utils.readgt_sintel import load_sintel_gt

        gt = load_sintel_gt(str(scene_dir))
    elif dataset_cat == "Replica":
        from utils.readgt_replica import load_replica_gt

        gt = load_replica_gt(str(scene_dir))
    elif dataset_cat == "Syn4D":
        from utils.readgt_colmap import load_colmap_gt

        gt = load_colmap_gt(str(scene_dir / "colmap"))
    else:
        raise FileNotFoundError(f"Missing intrinsics file: {path}")

    for _, info in gt.items():
        if int(info["id"]) == int(frame_index):
            return np.asarray(info["K"], dtype=np.float64)
    raise RuntimeError(f"Could not find GT intrinsics for frame {frame_index}")


def _sample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=int(max_points), replace=False)
    return points[idx]


def _load_static(static_result: Path, frame_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    data = np.load(static_result)
    points = np.asarray(data["p"], dtype=np.float64).T
    R = np.asarray(data["R"], dtype=np.float64)
    t = np.asarray(data["t"], dtype=np.float64)
    n_frames = int(t.shape[1])
    if frame_index < 0 or frame_index >= n_frames:
        raise IndexError(f"frame_index={frame_index} outside static result [0,{n_frames-1}]")
    R_c2w = R[:, 3 * frame_index : 3 * frame_index + 3]
    C = t[:, frame_index]
    return points, R_c2w, C, n_frames


def _load_dynamic_from_vis(path: Path, frame_index: int) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if "dyn_mapped" not in data:
        return np.zeros((0, 3), dtype=np.float64)
    dyn = np.asarray(data["dyn_mapped"], dtype=np.float64)
    if dyn.ndim != 3 or dyn.shape[2] != 3:
        raise ValueError(f"Expected dyn_mapped shape (Q,T,3), got {dyn.shape}")
    if frame_index >= dyn.shape[1]:
        raise IndexError(f"frame_index={frame_index} outside dyn_mapped length {dyn.shape[1]}")
    pts = dyn[:, frame_index, :]
    return pts[np.isfinite(pts).all(axis=1)]


def _load_dynamic_from_results(path: Path, frame_index: int) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if "dynamics_points_reconstruct_array" in data:
        dyn = np.asarray(data["dynamics_points_reconstruct_array"], dtype=np.float64)
        pts = dyn[:, frame_index, :]
        return pts[np.isfinite(pts).all(axis=1)]
    if "dynamics_points_reconstruct" not in data:
        return np.zeros((0, 3), dtype=np.float64)
    obj = data["dynamics_points_reconstruct"]
    dyn_dict = obj.item() if getattr(obj, "shape", None) == () else obj
    pts = []
    for track in dyn_dict.values():
        arr = np.asarray(track, dtype=np.float64)
        if frame_index < arr.shape[0] and np.isfinite(arr[frame_index]).all():
            pts.append(arr[frame_index])
    if not pts:
        return np.zeros((0, 3), dtype=np.float64)
    return np.stack(pts, axis=0)


def _load_dynamic(scene_dir: Path, edg_vis: Path | None, edg_result: Path | None, frame_index: int) -> tuple[np.ndarray, Path | None]:
    candidates = [
        edg_vis or (scene_dir / "outputs" / "edg_solution_vis.npz"),
        edg_result or (scene_dir / "outputs" / "results_all.npz"),
    ]
    for path in candidates:
        if path is None or not path.is_file():
            continue
        if path.name == "edg_solution_vis.npz":
            return _load_dynamic_from_vis(path, frame_index), path
        return _load_dynamic_from_results(path, frame_index), path
    return np.zeros((0, 3), dtype=np.float64), None


def _render_depth(
    points_world: np.ndarray,
    R_c2w: np.ndarray,
    C: np.ndarray,
    K: np.ndarray,
    height: int,
    width: int,
    min_depth: float,
    max_depth: float,
    radius: int,
) -> np.ndarray:
    depth = np.full((height, width), np.inf, dtype=np.float64)
    if points_world.shape[0] == 0:
        return depth

    pts_cam = (np.asarray(R_c2w, dtype=np.float64).T @ (points_world - C[None, :]).T).T
    z = pts_cam[:, 2]
    valid = np.isfinite(pts_cam).all(axis=1) & (z > float(min_depth))
    if float(max_depth) > 0:
        valid &= z <= float(max_depth)
    pts_cam = pts_cam[valid]
    z = z[valid]
    if pts_cam.shape[0] == 0:
        return depth

    u = K[0, 0] * (pts_cam[:, 0] / z) + K[0, 2]
    v = K[1, 1] * (pts_cam[:, 1] / z) + K[1, 2]
    ui = np.rint(u).astype(np.int64)
    vi = np.rint(v).astype(np.int64)
    inside = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    ui = ui[inside]
    vi = vi[inside]
    z = z[inside]

    order = np.argsort(z)[::-1]  # far first; nearer points overwrite.
    radius = max(0, int(radius))
    for idx in order:
        x = int(ui[idx])
        y = int(vi[idx])
        zz = float(z[idx])
        if radius == 0:
            depth[y, x] = zz
            continue
        x0 = max(0, x - radius)
        x1 = min(width, x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)
        patch = depth[y0:y1, x0:x1]
        patch[...] = np.minimum(patch, zz)

    depth[~np.isfinite(depth)] = np.nan
    return depth


def _normalizer(values: np.ndarray, method: str) -> float:
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return 1.0
    value = float(np.mean(values) if method == "mean" else np.median(values))
    return value if np.isfinite(value) and value > 1e-12 else 1.0


def _evaluate(gt_depth: np.ndarray, est_depth: np.ndarray, min_depth: float, max_depth: float, normalize_stat: str) -> dict[str, Any]:
    valid = np.isfinite(gt_depth) & np.isfinite(est_depth)
    valid &= gt_depth > float(min_depth)
    valid &= est_depth > float(min_depth)
    if float(max_depth) > 0:
        valid &= gt_depth <= float(max_depth)
        valid &= est_depth <= float(max_depth)
    if not np.any(valid):
        raise ValueError("No overlapping valid rendered-depth and GT pixels.")

    gt_norm = _normalizer(gt_depth[valid], normalize_stat)
    est_norm = _normalizer(est_depth[valid], normalize_stat)
    diff = np.abs(est_depth[valid] / est_norm - gt_depth[valid] / gt_norm)
    return {
        "num_valid_pixels": int(diff.size),
        "normalize_stat": normalize_stat,
        "gt_normalizer": float(gt_norm),
        "est_normalizer": float(est_norm),
        "l1": float(np.mean(diff)),
        "coverage": float(diff.size / gt_depth.size),
    }


def _normalize_for_vis(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if not np.any(valid):
        return np.zeros_like(depth, dtype=np.float32)
    vals = depth[valid]
    lo = float(np.percentile(vals, 2.0))
    hi = float(np.percentile(vals, 98.0))
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((depth - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _colorize(depth01: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap((np.nan_to_num(depth01) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def _write_vis(path: Path, gt_depth: np.ndarray, est_depth: np.ndarray, metrics: dict[str, Any], min_depth: float, max_depth: float) -> None:
    valid = np.isfinite(gt_depth) & np.isfinite(est_depth)
    valid &= gt_depth > float(min_depth)
    valid &= est_depth > float(min_depth)
    if float(max_depth) > 0:
        valid &= gt_depth <= float(max_depth)
        valid &= est_depth <= float(max_depth)
    gt_scaled = gt_depth / max(float(metrics["gt_normalizer"]), 1e-12)
    est_scaled = est_depth / max(float(metrics["est_normalizer"]), 1e-12)
    err = np.full_like(gt_scaled, np.nan)
    err[valid] = np.abs(est_scaled[valid] - gt_scaled[valid])
    vis = np.concatenate(
        [
            _colorize(_normalize_for_vis(gt_scaled, valid)),
            _colorize(_normalize_for_vis(est_scaled, valid)),
            _colorize(_normalize_for_vis(err, valid)),
        ],
        axis=1,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), vis)


def main() -> None:
    args = _parse_args()
    scene_dir = args.scene_dir.resolve()
    output_dir = scene_dir / "outputs"
    static_result = (args.static_result or (output_dir / "result_XM.npz")).resolve()
    gt_depth_dir = (args.gt_depth_dir or (scene_dir / "depths")).resolve()
    output_json = (args.output_json or (output_dir / f"depth_pointcloud_eval_frame{args.frame_index:06d}.json")).resolve()
    vis_out = (args.vis_out or (output_dir / f"depth_pointcloud_eval_frame{args.frame_index:06d}.png")).resolve()
    rendered_out = (args.rendered_depth_out or (output_dir / f"rendered_pointcloud_depth_frame{args.frame_index:06d}.npy")).resolve()

    gt_path = _gt_depth_path(scene_dir, gt_depth_dir, args.frame_index)
    gt_depth = _read_depth(gt_path, args.gt_scale)
    height, width = gt_depth.shape
    K = _load_intrinsics(scene_dir, args.intrinsics, args.frame_index)

    static_points, R_c2w, C, n_frames = _load_static(static_result, args.frame_index)
    dynamic_points, dynamic_path = _load_dynamic(scene_dir, args.edg_vis, args.edg_result, args.frame_index)

    if args.dynamic_only:
        static_points = np.zeros((0, 3), dtype=np.float64)
    if args.static_only:
        dynamic_points = np.zeros((0, 3), dtype=np.float64)

    static_points = _sample_points(static_points, args.max_static_points, seed=11)
    dynamic_points = _sample_points(dynamic_points, args.max_dynamic_points, seed=17)
    points_world = np.concatenate([static_points, dynamic_points], axis=0)

    rendered = _render_depth(
        points_world=points_world,
        R_c2w=R_c2w,
        C=C,
        K=K,
        height=height,
        width=width,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        radius=args.point_radius,
    )

    metrics = _evaluate(gt_depth, rendered, args.min_depth, args.max_depth, args.normalize_stat)
    result: dict[str, Any] = {
        "scene_dir": str(scene_dir),
        "frame_index": int(args.frame_index),
        "gt_depth_path": str(gt_path),
        "static_result": str(static_result),
        "dynamic_result": str(dynamic_path) if dynamic_path is not None else None,
        "num_static_points": int(static_points.shape[0]),
        "num_dynamic_points": int(dynamic_points.shape[0]),
        "num_render_points": int(points_world.shape[0]),
        "point_radius": int(args.point_radius),
        "gt_scale": float(args.gt_scale),
        "min_depth": float(args.min_depth),
        "max_depth": float(args.max_depth),
        "rendered_depth_path": str(rendered_out),
        "vis_out": str(vis_out),
        **metrics,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    rendered_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(rendered_out, rendered.astype(np.float32))
    _write_vis(vis_out, gt_depth, rendered, metrics, args.min_depth, args.max_depth)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json:
        print("RESULT_JSON:", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
