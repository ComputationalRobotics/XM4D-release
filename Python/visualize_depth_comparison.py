#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

REPLICA_RENDER_DEPTH_SCALE = 65535.0 * 0.1

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "Python"
MOSCA_DIR = REPO_ROOT / "deps" / "MoSca"
for path in (PYTHON_DIR, MOSCA_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_depth import _prefer_npy_depth_path, _read_depth, _normalizer  # noqa: E402
from eval_depth_depthanything import (  # noqa: E402
    _build_static_world_points as _da_build_static_world_points,
    _dynamic_world_points_for_frame as _da_dynamic_world_points_for_frame,
    _gt_valid_mask,
    _lift_depth_to_world,
    _load_da3_cameras,
    _load_da3_depths,
    _load_gt as _da_load_gt,
    _load_novel_metadata as _da_load_novel_metadata,
    _novel_c2w_from_da_base,
    _read_dynamic_mask,
    _render_depth_from_points,
    _scale_K,
    _umeyama_similarity,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize one train or novel depth frame across Ours, MoSca, SpaTracker, and DepthAnything."
    )
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "novel"], default="train")
    parser.add_argument("--frame-index", type=int, default=0, help="Training frame index, or fallback novel base frame.")
    parser.add_argument("--novel-id", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <scene-dir>/outputs/depth_visualization")
    parser.add_argument("--gt-scale", type=float, default=REPLICA_RENDER_DEPTH_SCALE)
    parser.add_argument("--min-depth", type=float, default=1e-6)
    parser.add_argument("--max-depth", type=float, default=-1.0)
    parser.add_argument("--point-stride", type=int, default=4)
    parser.add_argument("--dynamic-point-stride", type=int, default=1)
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument("--spatracker-point-stride", type=int, default=1)
    parser.add_argument("--spatracker-point-radius", type=int, default=2)
    parser.add_argument("--spatracker-max-points", type=int, default=200000, help="Deterministic cap for live SpaTracker visualization points; <=0 disables.")
    parser.add_argument(
        "--spatracker-use-gt-intrinsics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use GT intrinsics, scaled to the SpaTracker depth resolution, when lifting/rendering SpaTracker.",
    )
    parser.add_argument("--alpha-threshold", type=float, default=0.05)
    parser.add_argument("--alpha-density-radius", type=int, default=1)
    parser.add_argument("--alpha-density-scale", type=float, default=1.0)
    parser.add_argument("--mask-threshold", type=float, default=127.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ours-camera-source", choices=["colmap", "xm"], default="colmap")
    parser.add_argument("--ours-gaussians", type=Path, default=None)
    parser.add_argument("--mosca-npz", type=Path, default=None)
    parser.add_argument("--spatracker-npz", type=Path, default=None)
    parser.add_argument(
        "--spatracker-align",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Align SpaTracker cameras to GT before rendering with a camera-center Sim3.",
    )
    parser.add_argument("--error-clip", type=float, default=0.2)
    parser.add_argument(
        "--scale-mode",
        choices=["median", "gt_camera"],
        default="median",
        help="Default depth scale policy for all methods.",
    )
    parser.add_argument("--ours-scale-mode", choices=["median", "gt_camera"], default=None)
    parser.add_argument("--mosca-scale-mode", choices=["median", "gt_camera"], default=None)
    parser.add_argument("--spatracker-scale-mode", choices=["median", "gt_camera"], default=None)
    parser.add_argument("--da3-scale-mode", choices=["median", "gt_camera"], default=None)
    parser.add_argument(
        "--depth-correction-range",
        type=float,
        default=0.1,
        help="Fit one global multiplicative depth correction per method on train frame 0, clamped to [1-r, 1+r]. Use 0 to disable.",
    )
    parser.add_argument(
        "--spatracker-depth-correction-range",
        type=float,
        default=0.0,
        help="SpaTracker-only correction range. Default 0 disables final correction and keeps only GT-camera scale.",
    )
    parser.add_argument(
        "--depth-correction-samples",
        type=int,
        default=401,
        help="Number of samples used to fit the bounded train-frame-0 correction.",
    )
    parser.add_argument(
        "--only",
        choices=["ours", "mosca", "spatracker", "da3"],
        action="append",
        default=None,
        help="Render only selected method group. May be passed multiple times.",
    )
    parser.add_argument("--combine-existing", action="store_true", help="Build the final panel from existing saved method arrays.")
    parser.add_argument("--no-ours", action="store_true")
    parser.add_argument("--no-mosca", action="store_true")
    parser.add_argument("--no-spatracker", action="store_true")
    parser.add_argument("--no-da3", action="store_true")
    return parser.parse_args()


def _image_files(scene_dir: Path) -> list[Path]:
    return sorted(
        p for p in (scene_dir / "images").iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


def _depth_path_for_train(scene_dir: Path, frame: int) -> Path:
    candidates = [
        scene_dir / "depths" / f"depth{frame:06d}.npy",
        scene_dir / "depths" / f"frame{frame:06d}.npy",
        scene_dir / "depths" / f"{frame:06d}.npy",
        _prefer_npy_depth_path(scene_dir / "depths" / f"depth{frame:06d}.png"),
    ]
    images = _image_files(scene_dir)
    if 0 <= frame < len(images):
        candidates.insert(0, scene_dir / "depths" / f"{images[frame].stem}.npy")
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"No GT depth found for frame {frame} in {scene_dir / 'depths'}")


def _selected_novel_record(scene_dir: Path, novel_id: int) -> Any:
    records = _da_load_novel_metadata(scene_dir / "novel")
    if not records:
        raise FileNotFoundError(f"No novel metadata records in {scene_dir / 'novel' / 'metadata.txt'}")
    for rec in records:
        if int(rec.novel_id) == int(novel_id):
            return rec
    if 0 <= int(novel_id) < len(records):
        return records[int(novel_id)]
    raise IndexError(f"novel_id={novel_id} not found; available ids: {[int(r.novel_id) for r in records]}")


def _target_info(scene_dir: Path, args: argparse.Namespace) -> tuple[np.ndarray, int, Path, str]:
    if args.split == "train":
        gt = _read_depth(_depth_path_for_train(scene_dir, int(args.frame_index)), args.gt_scale)
        return gt, int(args.frame_index), _depth_path_for_train(scene_dir, int(args.frame_index)), f"train{int(args.frame_index):06d}"
    rec = _selected_novel_record(scene_dir, int(args.novel_id))
    gt = _read_depth(rec.depth_path, args.gt_scale)
    return gt, int(rec.base_frame), rec.depth_path, f"novel{int(rec.novel_id):06d}_base{int(rec.base_frame):06d}"


def _valid_for_est(gt: np.ndarray, est: np.ndarray, alpha: np.ndarray | None, args: argparse.Namespace) -> np.ndarray:
    if est.shape != gt.shape:
        est = cv2.resize(est.astype(np.float32), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
    valid = np.isfinite(gt) & np.isfinite(est)
    valid &= gt > float(args.min_depth)
    valid &= est > float(args.min_depth)
    if float(args.max_depth) > 0:
        valid &= gt <= float(args.max_depth)
        valid &= est <= float(args.max_depth)
    if alpha is not None and float(args.alpha_threshold) > 0:
        if alpha.shape != gt.shape:
            alpha = cv2.resize(alpha.astype(np.float32), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        valid &= np.isfinite(alpha) & (alpha >= float(args.alpha_threshold))
    return valid


def _score_depth_pair(
    name: str,
    gt: np.ndarray,
    est: np.ndarray,
    valid: np.ndarray | None,
    args: argparse.Namespace,
    est_to_gt_scale: float = 1.0,
    scale_source: str = "none",
    metric_max_depth: float | None = None,
) -> dict[str, Any]:
    if est.shape != gt.shape:
        est = cv2.resize(est.astype(np.float32), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float64)
    est = est * float(est_to_gt_scale)
    if valid is not None and valid.shape != gt.shape:
        valid = cv2.resize(valid.astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    if metric_max_depth is not None or valid is None:
        valid = np.ones_like(gt, dtype=bool) if valid is None else valid.astype(bool)
        depth_max = args.max_depth if metric_max_depth is None else metric_max_depth
    else:
        valid = valid.astype(bool)
        depth_max = args.max_depth
    valid &= np.isfinite(gt) & np.isfinite(est)
    valid &= gt > float(args.min_depth)
    valid &= est > float(args.min_depth)
    if float(depth_max) > 0:
        valid &= gt <= float(depth_max)
        valid &= est <= float(depth_max)
    err = np.zeros_like(gt, dtype=np.float64)
    raw_l1 = None
    relative_l1 = None
    if np.any(valid):
        err[valid] = np.abs(gt[valid] - est[valid])
        l1 = float(np.mean(err[valid]))
        raw_l1 = float(np.mean(np.abs(gt[valid] - est[valid])))
        gt_median = _normalizer(gt[valid], "median")
        relative_l1 = float(raw_l1 / max(gt_median, 1e-12))
    else:
        l1 = None
    return {
        "name": name,
        "gt_depth": gt,
        "depth": est,
        "error": err,
        "valid": valid,
        "l1": l1,
        "raw_l1": raw_l1,
        "relative_l1": relative_l1,
        "valid_px": int(np.count_nonzero(valid)),
        "est_to_gt_scale": float(est_to_gt_scale),
        "scale_source": scale_source,
    }


def _raw_l1_for_scale(
    gt: np.ndarray,
    est: np.ndarray,
    valid: np.ndarray | None,
    args: argparse.Namespace,
    scale: float,
) -> float:
    if est.shape != gt.shape:
        est = cv2.resize(est.astype(np.float32), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float64)
    est = est * float(scale)
    if valid is None:
        mask = np.ones_like(gt, dtype=bool)
    else:
        if valid.shape != gt.shape:
            valid = cv2.resize(valid.astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        mask = valid.astype(bool)
    mask &= np.isfinite(gt) & np.isfinite(est)
    mask &= gt > float(args.min_depth)
    mask &= est > float(args.min_depth)
    if float(args.max_depth) > 0:
        mask &= gt <= float(args.max_depth)
        mask &= est <= float(args.max_depth)
    if not np.any(mask):
        return float("inf")
    return float(np.mean(np.abs(gt[mask] - est[mask])))


def _fit_first_train_depth_correction(
    name: str,
    gt: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray | None,
    args: argparse.Namespace,
    base_est_to_gt_scale: float,
    correction_range: float | None = None,
) -> tuple[float, str, float | None]:
    radius = float(args.depth_correction_range) if correction_range is None else float(correction_range)
    radius = max(radius, 0.0)
    if radius <= 0:
        return 1.0, "train-frame-0 correction disabled", None
    lo = max(1.0 - radius, 1e-12)
    hi = 1.0 + radius
    samples = max(int(args.depth_correction_samples), 3)
    candidates = np.linspace(lo, hi, samples, dtype=np.float64)
    losses = np.array(
        [_raw_l1_for_scale(gt, depth, valid, args, float(base_est_to_gt_scale) * float(c)) for c in candidates],
        dtype=np.float64,
    )
    if not np.isfinite(losses).any():
        return 1.0, f"{name}: train frame 0 correction unavailable, no valid pixels", None
    idx = int(np.nanargmin(losses))
    correction = float(candidates[idx])
    return correction, f"{name}: train frame 0 bounded depth correction [{lo:.3g}, {hi:.3g}]", float(losses[idx])


def _normalize_and_score(
    gt: np.ndarray,
    est: np.ndarray,
    alpha: np.ndarray | None,
    args: argparse.Namespace,
    name: str = "",
    est_to_gt_scale: float = 1.0,
    scale_source: str = "none",
) -> dict[str, Any]:
    est_scaled = np.asarray(est, dtype=np.float64)
    if est_scaled.shape != gt.shape:
        est_scaled = cv2.resize(est_scaled.astype(np.float32), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float64)
    est_scaled = est_scaled * float(est_to_gt_scale)
    valid = _valid_for_est(gt, est_scaled, alpha, args)
    scored = _score_depth_pair(
        name=name,
        gt=gt,
        est=est_scaled,
        valid=valid,
        args=args,
        est_to_gt_scale=1.0,
        scale_source=scale_source,
    )
    scored["est_to_gt_scale"] = float(est_to_gt_scale)
    return scored


def _colorize(values: np.ndarray, valid: np.ndarray, vmin: float, vmax: float, cmap: int) -> np.ndarray:
    vmax = max(float(vmax), float(vmin) + 1e-6)
    norm = np.clip((values - float(vmin)) / (float(vmax) - float(vmin)), 0.0, 1.0)
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cmap)
    img[~valid] = 0
    return img


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _write_panel(out_path: Path, gt: np.ndarray, items: list[dict[str, Any]], args: argparse.Namespace) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gt_panel = items[0].get("gt_depth", gt) if items else gt
    gt_valid = np.isfinite(gt_panel) & (gt_panel > float(args.min_depth))
    depth_values = [gt_panel[gt_valid]] if np.any(gt_valid) else []
    for item in items:
        valid = item["valid"]
        if np.any(valid):
            depth_values.append(item["gt_depth"][valid])
            depth_values.append(item["depth"][valid])
    if depth_values:
        depth_cat = np.concatenate([np.asarray(v).reshape(-1) for v in depth_values if np.asarray(v).size])
        vmin, vmax = np.percentile(depth_cat, [2, 98])
    else:
        vmin, vmax = 0.0, 1.0
    depth_row = [_label(_colorize(gt_panel, gt_valid, float(vmin), float(vmax), cv2.COLORMAP_TURBO), "GT depth")]
    err_row = [_label(np.zeros_like(depth_row[0]), "error")]
    for item in items:
        title = item["name"]
        if item.get("relative_l1") is not None:
            title += f"  rel {item['relative_l1']:.4f}"
        depth_row.append(
            _label(_colorize(item["depth"], item["valid"], float(vmin), float(vmax), cv2.COLORMAP_TURBO), title)
        )
        clipped = np.clip(item["error"], 0.0, float(args.error_clip))
        err_row.append(
            _label(_colorize(clipped, item["valid"], 0.0, float(args.error_clip), cv2.COLORMAP_MAGMA), f"{item['name']} err")
        )
    canvas = np.vstack([np.hstack(depth_row), np.hstack(err_row)])
    cv2.imwrite(str(out_path), canvas)


def _method_slug(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_").lower()


def _method_prefix(label: str, name: str) -> str:
    return f"{label}_{_method_slug(name)}"


def _method_json_path(out_dir: Path, label: str, name: str) -> Path:
    return out_dir / f"{_method_prefix(label, name)}_metrics.json"


def _cap_points(points: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if int(max_points) <= 0 or points.shape[0] <= int(max_points):
        return points
    rng = np.random.default_rng(int(seed))
    keep = rng.choice(points.shape[0], int(max_points), replace=False)
    return points[keep]


def _transform_c2w_similarity(c2w: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = np.asarray(c2w, dtype=np.float64).copy()
    out[:3, :3] = R @ out[:3, :3]
    out[:3, 3] = float(scale) * (R @ out[:3, 3]) + np.asarray(t, dtype=np.float64).reshape(3)
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


def _spatracker_tracked_points(data: np.lib.npyio.NpzFile, local_idx: int) -> tuple[np.ndarray, int]:
    if "coords" not in data:
        return np.zeros((0, 3), dtype=np.float64), 0
    coords = np.asarray(data["coords"], dtype=np.float64)
    if coords.ndim != 3 or coords.shape[-1] < 3:
        return np.zeros((0, 3), dtype=np.float64), 0
    local_idx = int(np.clip(local_idx, 0, coords.shape[0] - 1))
    pts = coords[local_idx, :, :3]
    valid = np.isfinite(pts).all(axis=1)
    if "visibs" in data:
        vis = np.asarray(data["visibs"])
        if vis.ndim >= 2 and local_idx < vis.shape[0]:
            vis_local = np.asarray(vis[local_idx]).reshape(-1)
            n = min(int(vis_local.shape[0]), int(pts.shape[0]))
            valid[:n] &= vis_local[:n] > 0.5
            if n < pts.shape[0]:
                valid[n:] = False
    return pts[valid], int(pts.shape[0])


def _save_spatracker_dynamic_debug(
    out_dir: Path,
    label: str,
    points_world: np.ndarray,
    views: list[tuple[str, np.ndarray, np.ndarray]],
    shape_hw: tuple[int, int],
    args: argparse.Namespace,
) -> None:
    if points_world.size == 0:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, c2w, K in views:
        depth, alpha = _render_depth_from_points(
            points_world,
            c2w,
            K,
            int(shape_hw[0]),
            int(shape_hw[1]),
            args.min_depth,
            args.max_depth,
            args.spatracker_point_radius,
            args.alpha_density_radius,
            args.alpha_density_scale,
        )
        valid = np.isfinite(depth) & (depth > float(args.min_depth))
        if float(args.alpha_threshold) > 0:
            valid &= np.isfinite(alpha) & (alpha >= float(args.alpha_threshold))
        if np.any(valid):
            normer = _normalizer(depth[valid], "median")
            img = _colorize(depth / max(normer, 1e-12), valid, 0.0, 4.0, cv2.COLORMAP_TURBO)
        else:
            img = np.zeros((int(shape_hw[0]), int(shape_hw[1]), 3), dtype=np.uint8)
        path = out_dir / f"{label}_spatracker_dynamic_{name}.png"
        cv2.imwrite(str(path), _label(img, f"SpaTracker dynamic {name} valid {int(np.count_nonzero(valid))}"))
        print(f"[depth-vis] SpaTracker dynamic-only {name}: valid_px={int(np.count_nonzero(valid))}")
        print(f"[depth-vis] saved {path}")


def _save_method_outputs(out_dir: Path, label: str, item: dict[str, Any]) -> None:
    name = str(item["name"])
    prefix = _method_prefix(label, name)
    np.save(out_dir / f"{prefix}_gt.npy", item["gt_depth"].astype(np.float32))
    np.save(out_dir / f"{prefix}_depth.npy", item["depth"].astype(np.float32))
    np.save(out_dir / f"{prefix}_error.npy", item["error"].astype(np.float32))
    np.save(out_dir / f"{prefix}_valid.npy", item["valid"].astype(bool))
    metrics = {
        "name": name,
        "l1": item["l1"],
        "raw_l1": item.get("raw_l1"),
        "relative_l1": item.get("relative_l1"),
        "valid_px": item["valid_px"],
        "est_to_gt_scale": item.get("est_to_gt_scale"),
        "scale_source": item.get("scale_source"),
        "base_est_to_gt_scale": item.get("base_est_to_gt_scale"),
        "depth_correction": item.get("depth_correction"),
        "depth_correction_source": item.get("depth_correction_source"),
        "depth_correction_train0_l1": item.get("depth_correction_train0_l1"),
    }
    _method_json_path(out_dir, label, name).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")


def _load_existing_items(out_dir: Path, label: str) -> list[dict[str, Any]]:
    items = []
    suffix = "_metrics.json"
    for metrics_path in sorted(out_dir.glob(f"{label}_*{suffix}")):
        metrics = json.loads(metrics_path.read_text())
        name = metrics["name"]
        prefix = _method_prefix(label, name)
        gt_path = out_dir / f"{prefix}_gt.npy"
        depth_path = out_dir / f"{prefix}_depth.npy"
        if not depth_path.is_file():
            depth_path = out_dir / f"{prefix}_depth_norm.npy"
        error_path = out_dir / f"{prefix}_error.npy"
        valid_path = out_dir / f"{prefix}_valid.npy"
        if not gt_path.is_file() or not depth_path.is_file() or not error_path.is_file() or not valid_path.is_file():
            continue
        items.append(
            {
                "name": name,
                "gt_depth": np.load(gt_path),
                "depth": np.load(depth_path),
                "error": np.load(error_path),
                "valid": np.load(valid_path).astype(bool),
                "l1": metrics.get("l1"),
                "raw_l1": metrics.get("raw_l1"),
                "relative_l1": metrics.get("relative_l1"),
                "valid_px": int(metrics.get("valid_px", 0)),
                "est_to_gt_scale": metrics.get("est_to_gt_scale", metrics.get("gt_scale_factor")),
                "scale_source": metrics.get("scale_source"),
                "base_est_to_gt_scale": metrics.get("base_est_to_gt_scale"),
                "depth_correction": metrics.get("depth_correction"),
                "depth_correction_source": metrics.get("depth_correction_source"),
                "depth_correction_train0_l1": metrics.get("depth_correction_train0_l1"),
            }
        )
    preferred = ["Ours", "MoSca", "SpaTracker", "DA3 direct", "DA3 train-render", "DA3 novel-render"]
    order = {name: i for i, name in enumerate(preferred)}
    return sorted(items, key=lambda item: order.get(str(item["name"]), 999))


def _render_ours(scene_dir: Path, gt: np.ndarray, frame: int, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    import torch
    import eval_depth_gaussian_ours as ours

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    paths = ours._resolve_gaussian_paths(scene_dir, args.ours_gaussians)
    splats, edg, app_state, _step = ours._load_gaussians(paths, device)
    app_module = ours._build_app_module(app_state, splats, device)
    render_args = SimpleNamespace(
        no_edg_warp=False,
        sh_degree=2,
        near_plane=0.01,
        far_plane=1e10,
        depth_smooth="bilateral",
        smooth_ksize=50,
        bilateral_sigma_color=0.2,
        bilateral_sigma_space=5.0,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        alpha_threshold=max(float(args.alpha_threshold), 0.8),
    )
    if args.split == "train":
        c2w, K, *_ = ours._load_camera_and_gt_intrinsics(scene_dir, int(frame), args.ours_camera_source)
    else:
        rec = _selected_novel_record(scene_dir, int(args.novel_id))
        s, R_align, t_align, _ = ours._alignment_est_to_gt(scene_dir, args.ours_camera_source)
        est_c2ws, gt_c2ws, _ = ours._load_training_gt_pose_pairs(scene_dir, args.ours_camera_source)
        _s_raw = float(s)
        s, R_align, t_align = _unit_scale_alignment_from_centers(
            np.stack([pose[:3, 3] for pose in est_c2ws], axis=0),
            np.stack([pose[:3, 3] for pose in gt_c2ws], axis=0),
            R_align,
        )
        print(f"[depth-vis] Ours camera alignment scale raw={_s_raw:.6g} used=1")
        c2w = ours._transform_gt_c2w_to_est(rec.c2w, s, R_align, t_align)
        K = ours._load_gt_intrinsics(scene_dir, ours._gt_key_for_frame(scene_dir, int(rec.base_frame)))
    depth, alpha, _rgb = ours._render_depth(splats, edg, app_module, int(frame), c2w, K, gt.shape[0], gt.shape[1], render_args)
    depth = ours._smooth_depth(depth, alpha, render_args)
    return depth, alpha


def _render_mosca(scene_dir: Path, gt: np.ndarray, frame: int, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from estimate_mosca_chamfer import _align, _find_latest_mosca_npz
    from visualize_mosca_edg import (
        _apply_similarity_to_transform,
        _c2w_to_w2c,
        _find_workspace_from_npz,
        _get_c2w,
        _load_gt_c2w_from_colmap,
        _render_depth_from_saved_gaussians,
        _scale_K_for_size,
    )

    npz_path = _find_latest_mosca_npz(args.mosca_npz or scene_dir)
    data = np.load(npz_path, allow_pickle=True)
    align = _align(data, data, npz_path, SimpleNamespace())
    raw_scale = float(align[0]) if align is not None else 1.0
    scale = 1.0
    if align is not None:
        solver_c2w = _get_c2w(data)
        gt_c2w = _load_gt_c2w_from_colmap(_find_workspace_from_npz(npz_path))
        if solver_c2w is not None and gt_c2w is not None:
            n = min(int(solver_c2w.shape[0]), int(gt_c2w.shape[0]))
            scale, R_mosca, t_mosca = _unit_scale_alignment_from_centers(
                gt_c2w[:n, :3, 3],
                solver_c2w[:n, :3, 3],
                align[1],
            )
            align = (scale, R_mosca, t_mosca)
    print(f"[depth-vis] MoSca render scale raw={raw_scale:.6g} used=1")
    src_h = int(np.asarray(data["camera_H"]).item())
    src_w = int(np.asarray(data["camera_W"]).item())
    K = _scale_K_for_size(np.asarray(data["camera_K"], dtype=np.float64), (src_h, src_w), gt.shape)
    if args.split == "train":
        w2c = np.asarray(data["camera_w2c"], dtype=np.float64)[int(frame)]
    else:
        rec = _selected_novel_record(scene_dir, int(args.novel_id))
        c2w = rec.c2w
        if align is not None:
            c2w = _apply_similarity_to_transform(c2w, *align)
        w2c = _c2w_to_w2c(c2w)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    depth, alpha = _render_depth_from_saved_gaussians(
        data, int(frame), gt.shape[0], gt.shape[1], K, w2c, float(args.alpha_threshold), device
    )
    return depth / max(scale, 1e-12), alpha


def _render_spatracker(scene_dir: Path, gt: np.ndarray, frame: int, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    result_npz = args.spatracker_npz
    if result_npz is None:
        candidates = [
            scene_dir / "outputs" / "spatracker_rgb_dynamic_input" / "results" / "result.npz",
            scene_dir / "outputs" / "spatracker_eval" / "result.npz",
            scene_dir / "outputs" / "spatracker_input" / "results" / "result.npz",
        ]
        for candidate in candidates:
            if candidate.is_file():
                result_npz = candidate
                break
        if result_npz is None:
            result_npz = candidates[0]
    result_npz = result_npz.resolve()
    data = np.load(result_npz, allow_pickle=True)
    for key in ("extrinsics", "intrinsics", "depths"):
        if key not in data:
            raise KeyError(f"{result_npz} missing {key}")
    gt_data = _da_load_gt(scene_dir)
    depths = np.asarray(data["depths"], dtype=np.float64)
    intrs = np.asarray(data["intrinsics"], dtype=np.float64)
    extrinsics = np.asarray(data["extrinsics"], dtype=np.float64)
    c2w_est = np.linalg.inv(extrinsics)
    frame_ids = np.asarray(data.get("frame_ids", np.arange(depths.shape[0])), dtype=np.int64).reshape(-1)
    local_by_frame = {int(fid): i for i, fid in enumerate(frame_ids)}
    image_paths = _image_files(scene_dir)
    gt_by_id_items = _gt_by_id(gt_data)

    def spatracker_K_for_lift(local_i: int, frame_id: int) -> np.ndarray:
        if bool(args.spatracker_use_gt_intrinsics) and int(frame_id) in gt_by_id_items and 0 <= int(frame_id) < len(image_paths):
            _, info = gt_by_id_items[int(frame_id)]
            return _scale_K(
                np.asarray(info["K"], dtype=np.float64),
                _image_size(image_paths[int(frame_id)]),
                (depths[local_i].shape[1], depths[local_i].shape[0]),
            )
        return intrs[local_i if intrs.shape[0] > 1 else 0]

    def spatracker_K_for_render(frame_id: int, shape_hw: tuple[int, int]) -> np.ndarray:
        if bool(args.spatracker_use_gt_intrinsics) and int(frame_id) in gt_by_id_items and 0 <= int(frame_id) < len(image_paths):
            _, info = gt_by_id_items[int(frame_id)]
            return _scale_K(
                np.asarray(info["K"], dtype=np.float64),
                _image_size(image_paths[int(frame_id)]),
                (shape_hw[1], shape_hw[0]),
            )
        local_i = local_by_frame.get(int(frame_id), int(frame_id))
        local_i = int(np.clip(local_i, 0, depths.shape[0] - 1))
        return _scale_K(
            intrs[local_i if intrs.shape[0] > 1 else 0],
            (depths[local_i].shape[1], depths[local_i].shape[0]),
            (shape_hw[1], shape_hw[0]),
        )

    local_idx = local_by_frame.get(int(frame), int(frame))
    local_idx = int(np.clip(local_idx, 0, depths.shape[0] - 1))
    source_depth = depths[local_idx]
    align = None
    by_gt_id = {int(info["id"]): info for info in gt_data.values() if "id" in info}
    if bool(args.spatracker_align):
        src_centers = []
        dst_centers = []
        for i, fid in enumerate(frame_ids):
            if int(fid) not in by_gt_id or i >= c2w_est.shape[0]:
                continue
            info = by_gt_id[int(fid)]
            src_centers.append(c2w_est[i, :3, 3])
            dst_centers.append(_gt_c2w(info)[:3, 3])
        if len(src_centers) >= 3:
            raw_scale, R_align, _t_align = _umeyama_similarity(np.asarray(src_centers), np.asarray(dst_centers))
            align = _unit_scale_alignment_from_centers(np.asarray(src_centers), np.asarray(dst_centers), R_align)
            print(f"[depth-vis] SpaTracker camera alignment scale raw={float(raw_scale):.6g} used=1")
    if args.split == "train":
        K_target = spatracker_K_for_render(int(frame), gt.shape)
        target_c2w = c2w_est[local_idx].copy()
        mask_dir = scene_dir / "masks"
        by_id = {int(info["id"]): name for name, info in gt_data.items() if "id" in info}
        static_chunks = []
        for i, fid in enumerate(frame_ids):
            if i >= depths.shape[0] or i >= c2w_est.shape[0]:
                continue
            depth_i = depths[i]
            K_i = spatracker_K_for_lift(i, int(fid))
            image_name = by_id.get(int(fid))
            dynamic_mask = _read_dynamic_mask(mask_dir, int(fid), depth_i.shape, args.mask_threshold, image_name=image_name)
            static_pts = _lift_depth_to_world(
                depth_i,
                K_i,
                c2w_est[i],
                args.min_depth,
                args.max_depth,
                args.spatracker_point_stride,
                0,
                pixel_mask=~dynamic_mask,
            )
            if static_pts.size:
                static_chunks.append(static_pts)
        dynamic_points, dynamic_track_total = _spatracker_tracked_points(data, local_idx)
        static_points = np.concatenate(static_chunks, axis=0) if static_chunks else np.zeros((0, 3), dtype=np.float64)
        static_before_cap = int(static_points.shape[0])
        dynamic_before_cap = int(dynamic_points.shape[0])
        static_points = _cap_points(static_points, int(args.spatracker_max_points), seed=int(frame))
        points_world = (
            np.concatenate([static_points, dynamic_points], axis=0)
            if static_points.size or dynamic_points.size
            else np.zeros((0, 3), dtype=np.float64)
        )
        if align is not None:
            scale, R_align, t_align = align
            points_world = scale * (R_align @ points_world.T).T + t_align.reshape(1, 3)
            target_c2w = _transform_c2w_similarity(target_c2w, scale, R_align, t_align)
            dynamic_points_debug = scale * (R_align @ dynamic_points.T).T + t_align.reshape(1, 3) if dynamic_points.size else dynamic_points
        else:
            dynamic_points_debug = dynamic_points
        print(
            "[depth-vis] SpaTracker render points: "
            f"static_before_cap={static_before_cap} "
            f"static_rendered={static_points.shape[0]} "
            f"dynamic_tracks_total={dynamic_track_total} "
            f"dynamic_visible_rendered={dynamic_before_cap} "
            f"total_rendered={points_world.shape[0]} "
            f"frame={int(frame)} "
            f"intrinsics={'gt' if args.spatracker_use_gt_intrinsics else 'spatracker'}"
        )
        out_dir = (args.output_dir or (scene_dir / "outputs" / "depth_visualization")).resolve()
        _save_spatracker_dynamic_debug(
            out_dir,
            f"train{int(frame):06d}",
            dynamic_points_debug,
            [("train_camera", target_c2w, K_target)],
            gt.shape,
            args,
        )
    else:
        rec = _selected_novel_record(scene_dir, int(args.novel_id))
        gt_by_id = _gt_by_id(gt_data)
        _, base_info = gt_by_id[int(rec.base_frame)]
        base_local_idx = local_by_frame.get(int(rec.base_frame), int(rec.base_frame))
        base_local_idx = int(np.clip(base_local_idx, 0, depths.shape[0] - 1))
        mask_dir = scene_dir / "masks"
        image_names = {int(info["id"]): name for name, info in gt_data.items() if "id" in info}
        static_chunks = []
        for i, fid in enumerate(frame_ids):
            if i >= depths.shape[0] or i >= c2w_est.shape[0]:
                continue
            depth_i = depths[i]
            K_i = spatracker_K_for_lift(i, int(fid))
            dynamic_mask = _read_dynamic_mask(mask_dir, int(fid), depth_i.shape, args.mask_threshold, image_name=image_names.get(int(fid)))
            static_pts = _lift_depth_to_world(
                depth_i,
                K_i,
                c2w_est[i],
                args.min_depth,
                args.max_depth,
                args.spatracker_point_stride,
                0,
                pixel_mask=~dynamic_mask,
            )
            if static_pts.size:
                static_chunks.append(static_pts)
        dynamic_points, dynamic_track_total = _spatracker_tracked_points(data, base_local_idx)
        static_points = np.concatenate(static_chunks, axis=0) if static_chunks else np.zeros((0, 3), dtype=np.float64)
        static_before_cap = int(static_points.shape[0])
        dynamic_before_cap = int(dynamic_points.shape[0])
        static_points = _cap_points(static_points, int(args.spatracker_max_points), seed=int(rec.novel_id))
        points_world = (
            np.concatenate([static_points, dynamic_points], axis=0)
            if static_points.size or dynamic_points.size
            else np.zeros((0, 3), dtype=np.float64)
        )
        target_c2w = rec.c2w
        if align is not None:
            scale, R_align, t_align = align
            points_world = scale * (R_align @ points_world.T).T + t_align.reshape(1, 3)
            spa_base_aligned = _transform_c2w_similarity(c2w_est[base_local_idx], scale, R_align, t_align)
            target_c2w = _spatracker_novel_c2w_from_base(rec, _gt_c2w(base_info), spa_base_aligned, scale)
            dynamic_points_debug = scale * (R_align @ dynamic_points.T).T + t_align.reshape(1, 3) if dynamic_points.size else dynamic_points
            base_debug_c2w = spa_base_aligned
        else:
            dynamic_points_debug = dynamic_points
            base_debug_c2w = c2w_est[base_local_idx]
        K_target = spatracker_K_for_render(int(rec.base_frame), gt.shape)
        near_debug_c2w = base_debug_c2w.copy()
        near_debug_c2w[:3, 3] = base_debug_c2w[:3, 3] + 0.1 * (target_c2w[:3, 3] - base_debug_c2w[:3, 3])
        print(
            "[depth-vis] SpaTracker novel render points: "
            f"static_before_cap={static_before_cap} "
            f"static_rendered={static_points.shape[0]} "
            f"dynamic_tracks_total={dynamic_track_total} "
            f"dynamic_visible_rendered={dynamic_before_cap} "
            f"total_rendered={points_world.shape[0]} "
            f"novel={int(rec.novel_id)} base_frame={int(rec.base_frame)} "
            f"camera={'aligned_spa_base_delta' if align is not None else 'raw_gt_novel'} "
            f"intrinsics={'gt' if args.spatracker_use_gt_intrinsics else 'spatracker'}"
        )
        out_dir = (args.output_dir or (scene_dir / "outputs" / "depth_visualization")).resolve()
        _save_spatracker_dynamic_debug(
            out_dir,
            f"novel{int(rec.novel_id):06d}_base{int(rec.base_frame):06d}",
            dynamic_points_debug,
            [
                ("base_train_camera", base_debug_c2w, K_target),
                ("near_base_camera", near_debug_c2w, K_target),
                ("novel_camera", target_c2w, K_target),
            ],
            gt.shape,
            args,
        )
    depth, alpha = _render_depth_from_points(
        points_world,
        target_c2w,
        K_target,
        gt.shape[0],
        gt.shape[1],
        args.min_depth,
        args.max_depth,
        args.spatracker_point_radius,
        args.alpha_density_radius,
        args.alpha_density_scale,
    )
    return depth, alpha


def _gt_by_id(gt_data: dict[str, dict[str, Any]]) -> dict[int, tuple[str, dict[str, Any]]]:
    out = {}
    for name, info in gt_data.items():
        if "id" not in info:
            continue
        c2w = _gt_c2w(info)
        item = dict(info)
        item["c2w"] = c2w
        out[int(info["id"])] = (name, item)
    return out


def _gt_c2w(info: dict[str, Any]) -> np.ndarray:
    R_w2c = np.asarray(info["R"], dtype=np.float64).reshape(3, 3)
    t_w2c = np.asarray(info["t"], dtype=np.float64).reshape(3)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R_w2c.T
    c2w[:3, 3] = -R_w2c.T @ t_w2c
    return c2w


def _image_size(path: Path) -> tuple[int, int]:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return int(img.shape[1]), int(img.shape[0])


def _render_da3(scene_dir: Path, gt: np.ndarray, frame: int, args: argparse.Namespace) -> list[tuple[str, np.ndarray, np.ndarray]]:
    da_dir = scene_dir / "outputs" / "depth_anything_3"
    depths = _load_da3_depths(da_dir)
    intrs, c2ws = _load_da3_cameras(da_dir)
    mask_dir = scene_dir / "masks"
    da_args = SimpleNamespace(
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        point_stride=args.point_stride,
        dynamic_point_stride=args.dynamic_point_stride,
        static_frame_stride=1,
        static_max_points=0,
        max_points=0,
        mask_threshold=args.mask_threshold,
        point_radius=args.point_radius,
        alpha_density_radius=args.alpha_density_radius,
        alpha_density_scale=args.alpha_density_scale,
    )
    results: list[tuple[str, np.ndarray, np.ndarray]] = []
    if args.split == "train" and 0 <= int(frame) < depths.shape[0]:
        direct = depths[int(frame)]
        if direct.shape != gt.shape:
            direct = cv2.resize(direct.astype(np.float32), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        results.append(("DA3 direct", direct.astype(np.float64), np.ones_like(gt, dtype=np.float64)))
    da_gt = _da_load_gt(scene_dir)
    static_points, _static_frames = _da_build_static_world_points(scene_dir, depths, intrs, c2ws, da_gt, mask_dir, da_args)
    dynamic_points = _da_dynamic_world_points_for_frame(depths, intrs, c2ws, da_gt, mask_dir, int(frame), da_args)
    points = np.concatenate([p for p in (static_points, dynamic_points) if p.size], axis=0)
    source_depth = depths[int(frame)] if 0 <= int(frame) < depths.shape[0] else depths[0]
    K_base = intrs[int(frame) if intrs.shape[0] > 1 and int(frame) < intrs.shape[0] else 0]
    K_target = _scale_K(K_base, (source_depth.shape[1], source_depth.shape[0]), (gt.shape[1], gt.shape[0]))
    if args.split == "train":
        target_c2w = c2ws[int(frame)]
        name = "DA3 train-render"
    else:
        rec = _selected_novel_record(scene_dir, int(args.novel_id))
        gt_to_da_scale, _R, _t = __import__("eval_depth_depthanything")._gt_to_da_similarity(da_gt, c2ws)
        gt_src = []
        da_dst = []
        for info in da_gt.values():
            if "id" not in info:
                continue
            idx = int(info["id"])
            if 0 <= idx < c2ws.shape[0]:
                gt_src.append(_gt_c2w(info)[:3, 3])
                da_dst.append(c2ws[idx, :3, 3])
        if len(gt_src) >= 3:
            _scale, _R, _t = _unit_scale_alignment_from_centers(np.asarray(gt_src), np.asarray(da_dst), _R)
        print(f"[depth-vis] DA3 camera alignment scale raw={float(gt_to_da_scale):.6g} used=1")
        target_c2w = _novel_c2w_from_da_base(rec, da_gt, c2ws, 1.0)
        name = "DA3 novel-render"
    depth, alpha = _render_depth_from_points(
        points,
        target_c2w,
        K_target,
        gt.shape[0],
        gt.shape[1],
        args.min_depth,
        args.max_depth,
        args.point_radius,
        args.alpha_density_radius,
        args.alpha_density_scale,
    )
    results.append((name, depth, alpha))
    return results


def _add_method(
    items: list[dict[str, Any]],
    name: str,
    gt: np.ndarray,
    depth: np.ndarray,
    alpha: np.ndarray | None,
    args: argparse.Namespace,
    est_to_gt_scale: float = 1.0,
    scale_source: str = "none",
) -> None:
    scored = _normalize_and_score(
        gt,
        depth,
        alpha,
        args,
        name=name,
        est_to_gt_scale=est_to_gt_scale,
        scale_source=scale_source,
    )
    items.append(scored)


def _score_saved_arrays(
    name: str,
    gt: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray | None,
    args: argparse.Namespace,
    metric_max_depth: float | None = None,
    est_to_gt_scale: float = 1.0,
    scale_source: str = "none",
    gt_scale_factor: float | None = None,
) -> dict[str, Any]:
    if gt_scale_factor is not None and scale_source == "none":
        scale_source = f"legacy_gt_scale_factor={gt_scale_factor}"
    return _score_depth_pair(
        name=name,
        gt=gt,
        est=depth,
        valid=valid,
        args=args,
        est_to_gt_scale=float(est_to_gt_scale),
        scale_source=scale_source,
        metric_max_depth=metric_max_depth,
    )


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _load_npz_item(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    gt = np.asarray(data["gt_depth"], dtype=np.float64)
    depth_key = "rendered_depth" if "rendered_depth" in data else "pred_depth"
    depth = np.asarray(data[depth_key], dtype=np.float64)
    valid = np.asarray(data["valid_mask"], dtype=bool)
    return gt, depth, valid


def _first_train_depth_correction(
    scene_dir: Path,
    args: argparse.Namespace,
    name: str,
    base_est_to_gt_scale: float,
    loader,
    correction_range: float | None = None,
) -> tuple[float, str, float, float | None]:
    try:
        gt0, depth0, valid0 = loader()
        correction, correction_source, correction_l1 = _fit_first_train_depth_correction(
            name,
            gt0,
            depth0,
            valid0,
            args,
            base_est_to_gt_scale,
            correction_range,
        )
    except FileNotFoundError as exc:
        correction, correction_source, correction_l1 = 1.0, f"{name}: train frame 0 correction unavailable ({exc})", None
    total_scale = float(base_est_to_gt_scale) * float(correction)
    return total_scale, correction_source, correction, correction_l1


def _attach_depth_correction_report(
    item: dict[str, Any],
    base_est_to_gt_scale: float,
    correction: float,
    correction_source: str,
    correction_l1: float | None,
) -> dict[str, Any]:
    item["base_est_to_gt_scale"] = float(base_est_to_gt_scale)
    item["depth_correction"] = float(correction)
    item["depth_correction_source"] = correction_source
    item["depth_correction_train0_l1"] = correction_l1
    return item


def _scale_from_normalizers(summary: dict[str, Any], source: str) -> tuple[float, str]:
    gt_norm = summary.get("gt_normalizer")
    est_norm = summary.get("est_normalizer")
    if gt_norm is None or est_norm is None:
        return 1.0, f"{source}: no saved normalizers"
    gt_norm = float(gt_norm)
    est_norm = float(est_norm)
    if not np.isfinite(gt_norm) or not np.isfinite(est_norm) or est_norm <= 1e-12:
        return 1.0, f"{source}: invalid saved normalizers"
    return float(gt_norm / est_norm), f"{source}: train global median gt/est"


def _scale_from_named_normalizers(
    summary: dict[str, Any],
    gt_key: str,
    est_key: str,
    source: str,
) -> tuple[float, str]:
    gt_norm = summary.get(gt_key)
    est_norm = summary.get(est_key)
    if gt_norm is None or est_norm is None:
        return 1.0, f"{source}: no saved normalizers"
    gt_norm = float(gt_norm)
    est_norm = float(est_norm)
    if not np.isfinite(gt_norm) or not np.isfinite(est_norm) or est_norm <= 1e-12:
        return 1.0, f"{source}: invalid saved normalizers"
    return float(gt_norm / est_norm), f"{source}: global median gt/est"


def _scale_from_array_prefixes(
    directory: Path,
    prefix_glob: str,
    pred_suffix: str,
    gt_suffixes: tuple[str, ...],
    valid_suffix: str,
    args: argparse.Namespace,
    source: str,
) -> tuple[float, str]:
    gt_chunks = []
    pred_chunks = []
    prefixes = sorted(directory.glob(prefix_glob))
    for marker in prefixes:
        stem = marker.name[: -len(pred_suffix)] if marker.name.endswith(pred_suffix) else marker.stem
        prefix = marker.with_name(stem)
        pred_path = prefix.with_name(prefix.name + pred_suffix)
        if not pred_path.is_file():
            continue
        gt_path = next((prefix.with_name(prefix.name + suffix) for suffix in gt_suffixes if prefix.with_name(prefix.name + suffix).is_file()), None)
        if gt_path is None:
            continue
        valid_path = prefix.with_name(prefix.name + valid_suffix)
        pred = np.load(pred_path).astype(np.float64)
        gt = np.load(gt_path).astype(np.float64)
        if pred.shape != gt.shape:
            pred = cv2.resize(pred.astype(np.float32), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float64)
        valid = np.isfinite(gt) & np.isfinite(pred)
        valid &= gt > float(args.min_depth)
        valid &= pred > float(args.min_depth)
        if float(args.max_depth) > 0:
            valid &= gt <= float(args.max_depth)
            valid &= pred <= float(args.max_depth)
        if valid_path.is_file():
            saved_valid = np.load(valid_path).astype(bool)
            if saved_valid.shape != gt.shape:
                saved_valid = cv2.resize(saved_valid.astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            valid &= saved_valid
        if np.any(valid):
            gt_chunks.append(gt[valid])
            pred_chunks.append(pred[valid])
    if not gt_chunks or not pred_chunks:
        return 1.0, f"{source}: no valid saved arrays, scale=1"
    gt_med = _normalizer(np.concatenate(gt_chunks), "median")
    pred_med = _normalizer(np.concatenate(pred_chunks), "median")
    if pred_med <= 1e-12:
        return 1.0, f"{source}: invalid prediction median, scale=1"
    return float(gt_med / pred_med), f"{source}: novel global median gt/est"


def _collect_array_prefix_chunks(
    directory: Path,
    prefix_glob: str,
    pred_suffix: str,
    gt_suffixes: tuple[str, ...],
    valid_suffix: str,
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    gt_chunks = []
    pred_chunks = []
    for marker in sorted(directory.glob(prefix_glob)):
        stem = marker.name[: -len(pred_suffix)] if marker.name.endswith(pred_suffix) else marker.stem
        prefix = marker.with_name(stem)
        pred_path = prefix.with_name(prefix.name + pred_suffix)
        if not pred_path.is_file():
            continue
        gt_path = next((prefix.with_name(prefix.name + suffix) for suffix in gt_suffixes if prefix.with_name(prefix.name + suffix).is_file()), None)
        if gt_path is None:
            continue
        valid_path = prefix.with_name(prefix.name + valid_suffix)
        pred = np.load(pred_path).astype(np.float64)
        gt = np.load(gt_path).astype(np.float64)
        if pred.shape != gt.shape:
            pred = cv2.resize(pred.astype(np.float32), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float64)
        valid = np.isfinite(gt) & np.isfinite(pred)
        valid &= gt > float(args.min_depth)
        valid &= pred > float(args.min_depth)
        if float(args.max_depth) > 0:
            valid &= gt <= float(args.max_depth)
            valid &= pred <= float(args.max_depth)
        if valid_path.is_file():
            saved_valid = np.load(valid_path).astype(bool)
            if saved_valid.shape != gt.shape:
                saved_valid = cv2.resize(saved_valid.astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            valid &= saved_valid
        if np.any(valid):
            gt_chunks.append(gt[valid])
            pred_chunks.append(pred[valid])
    return gt_chunks, pred_chunks


def _scale_from_array_specs(
    specs: list[tuple[Path, str, str, tuple[str, ...], str]],
    args: argparse.Namespace,
    source: str,
) -> tuple[float, str]:
    gt_chunks = []
    pred_chunks = []
    for directory, prefix_glob, pred_suffix, gt_suffixes, valid_suffix in specs:
        gt_part, pred_part = _collect_array_prefix_chunks(
            directory,
            prefix_glob,
            pred_suffix,
            gt_suffixes,
            valid_suffix,
            args,
        )
        gt_chunks.extend(gt_part)
        pred_chunks.extend(pred_part)
    if not gt_chunks or not pred_chunks:
        return 1.0, f"{source}: no valid saved arrays, scale=1"
    gt_med = _normalizer(np.concatenate(gt_chunks), "median")
    pred_med = _normalizer(np.concatenate(pred_chunks), "median")
    if pred_med <= 1e-12:
        return 1.0, f"{source}: invalid prediction median, scale=1"
    return float(gt_med / pred_med), f"{source}: train+novel global median gt/est"


def _scale_from_npz_specs(
    specs: list[tuple[Path, str]],
    args: argparse.Namespace,
    source: str,
) -> tuple[float, str]:
    gt_chunks = []
    pred_chunks = []
    for directory, pattern in specs:
        for path in sorted(directory.glob(pattern)):
            try:
                gt, pred, valid = _load_npz_item(path)
            except (FileNotFoundError, KeyError):
                continue
            if pred.shape != gt.shape:
                pred = cv2.resize(pred.astype(np.float32), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float64)
            if valid.shape != gt.shape:
                valid = cv2.resize(valid.astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            valid = valid.astype(bool)
            valid &= np.isfinite(gt) & np.isfinite(pred)
            valid &= gt > float(args.min_depth)
            valid &= pred > float(args.min_depth)
            if float(args.max_depth) > 0:
                valid &= gt <= float(args.max_depth)
                valid &= pred <= float(args.max_depth)
            if np.any(valid):
                gt_chunks.append(gt[valid])
                pred_chunks.append(pred[valid])
    if not gt_chunks or not pred_chunks:
        return 1.0, f"{source}: no valid saved arrays, scale=1"
    gt_med = _normalizer(np.concatenate(gt_chunks), "median")
    pred_med = _normalizer(np.concatenate(pred_chunks), "median")
    if pred_med <= 1e-12:
        return 1.0, f"{source}: invalid prediction median, scale=1"
    return float(gt_med / pred_med), f"{source}: train+novel global median gt/est"


def _ours_union_median_scale(scene_dir: Path, args: argparse.Namespace) -> tuple[float, str]:
    return _scale_from_npz_specs(
        [
            (scene_dir / "outputs" / "gaussian_depth_eval" / "train_depth_arrays", "train_depth_*.npz"),
            (scene_dir / "outputs" / "gaussian_depth_eval" / "novel" / "depth_arrays", "novel_depth_*.npz"),
        ],
        args,
        "Ours",
    )


def _mosca_union_median_scale(scene_dir: Path, args: argparse.Namespace) -> tuple[float, str]:
    return _scale_from_array_specs(
        [
            (
                scene_dir / "outputs_mosca" / "depth_eval" / "training_depth",
                "frame*_pred.npy",
                "_pred.npy",
                ("_gt_unscaled.npy", "_gt.npy"),
                "_valid.npy",
            ),
            (
                scene_dir / "outputs_mosca" / "depth_eval" / "novel_depth",
                "novel*_pred.npy",
                "_pred.npy",
                ("_gt_unscaled.npy", "_gt.npy"),
                "_valid.npy",
            ),
        ],
        args,
        "MoSca",
    )


def _spatracker_union_median_scale(scene_dir: Path, args: argparse.Namespace) -> tuple[float, str]:
    return _scale_from_array_specs(
        [
            (
                scene_dir / "outputs" / "spatracker_eval" / "training_depth",
                "frame*_pred_unscaled.npy",
                "_pred_unscaled.npy",
                ("_gt.npy",),
                "_valid.npy",
            ),
            (
                scene_dir / "outputs" / "spatracker_eval" / "novel_depth",
                "novel*_pred_unscaled.npy",
                "_pred_unscaled.npy",
                ("_gt.npy",),
                "_valid.npy",
            ),
        ],
        args,
        "SpaTracker",
    )


def _da3_render_union_median_scale(scene_dir: Path, args: argparse.Namespace) -> tuple[float, str]:
    return _scale_from_array_specs(
        [
            (
                scene_dir / "outputs" / "depth_anything_3" / "train_render_depth_arrays",
                "train*_depth.npy",
                "_depth.npy",
                ("_gt.npy",),
                "_valid.npy",
            ),
            (
                scene_dir / "outputs" / "depth_anything_3" / "novel_render_depth_arrays",
                "novel*_depth.npy",
                "_depth.npy",
                ("_gt.npy",),
                "_valid.npy",
            ),
        ],
        args,
        "DA3 render",
    )


def _scale_from_current_pair(gt: np.ndarray, depth: np.ndarray, valid: np.ndarray | None, args: argparse.Namespace, source: str) -> tuple[float, str]:
    if depth.shape != gt.shape:
        depth = cv2.resize(depth.astype(np.float32), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float64)
    if valid is None:
        valid = _valid_for_est(gt, depth, None, args)
    elif valid.shape != gt.shape:
        valid = cv2.resize(valid.astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    valid = np.asarray(valid).astype(bool)
    if not np.any(valid):
        return 1.0, f"{source}: no valid pixels, scale=1"
    gt_med = _normalizer(gt[valid], "median")
    est_med = _normalizer(depth[valid], "median")
    if est_med <= 1e-12:
        return 1.0, f"{source}: invalid current median, scale=1"
    return float(gt_med / est_med), f"{source}: current-frame median fallback"


def _json_scale(path: Path, source: str) -> tuple[float, str]:
    if not path.is_file():
        return 1.0, f"{source}: missing {path.name}, scale=1"
    return _scale_from_normalizers(_json(path), source)


def _method_scale_mode(args: argparse.Namespace, method: str) -> str:
    override = getattr(args, f"{method.lower()}_scale_mode", None)
    return str(override or args.scale_mode)


def _select_scale(mode: str, median: tuple[float, str], gt_camera: tuple[float, str]) -> tuple[float, str]:
    return gt_camera if mode == "gt_camera" else median


def _with_scale_report(
    item: dict[str, Any],
    median: tuple[float, str],
    gt_camera: tuple[float, str],
) -> dict[str, Any]:
    item["median_est_to_gt_scale"] = float(median[0])
    item["median_scale_source"] = median[1]
    item["gt_camera_est_to_gt_scale"] = float(gt_camera[0])
    item["gt_camera_scale_source"] = gt_camera[1]
    return item


def _ours_camera_est_to_gt_scale(scene_dir: Path, args: argparse.Namespace) -> tuple[float, str]:
    try:
        import eval_depth_gaussian_ours as ours

        scale, _R, _t, method = ours._alignment_est_to_gt(scene_dir, args.ours_camera_source)
        return float(scale), f"Ours GT-camera scale est->GT ({method})"
    except Exception as exc:
        return 1.0, f"Ours GT-camera scale unavailable ({exc}), scale=1"


def _mosca_est_to_gt_scale(summary: dict[str, Any]) -> tuple[float, str]:
    scale = summary.get("depth_scale_render_to_gt")
    if scale is None:
        gt_to_render = summary.get("depth_scale_gt_to_render", summary.get("scale_gt_to_render"))
        if gt_to_render is not None and float(gt_to_render) > 1e-12:
            scale = 1.0 / float(gt_to_render)
    if scale is None:
        return 1.0, "MoSca: missing depth scale, scale=1"
    return float(scale), "MoSca: train global median render->GT"


def _mosca_camera_est_to_gt_scale(summary: dict[str, Any]) -> tuple[float, str]:
    scale = summary.get("camera_scale_render_to_gt")
    if scale is None:
        gt_to_render = summary.get("camera_scale_gt_to_render")
        if gt_to_render is not None and float(gt_to_render) > 1e-12:
            scale = 1.0 / float(gt_to_render)
    if scale is None:
        return 1.0, "MoSca: missing GT-camera scale, scale=1"
    return float(scale), "MoSca GT-camera scale render->GT"


def _spatracker_est_to_gt_scale(scene_dir: Path, split: str, mode: str = "median") -> tuple[float, str]:
    candidates = [
        scene_dir / "outputs_spatracker" / "spatracker_eval" / "depth_eval.json",
        scene_dir / "outputs" / "spatracker_eval" / "depth_eval.json",
        scene_dir / "outputs" / "spatracker_eval_static_dynamic_smoke" / "depth_eval.json",
    ]
    for path in candidates:
        if path.is_file():
            summary = _json(path)
            if mode == "gt_camera":
                camera_path = path.parent / "camera_accuracy.json"
                if camera_path.is_file():
                    camera = _json(camera_path).get("evaluation", {})
                    scale = camera.get("scale_align", camera.get("render_alignment_raw_scale"))
                    if scale is not None:
                        return float(scale), f"SpaTracker GT-camera scale est->GT from {camera_path.relative_to(scene_dir)}"
                scale = summary.get("est_to_gt_scale")
                if scale is not None:
                    return float(scale), f"SpaTracker GT-camera scale est->GT from {path.relative_to(scene_dir)}"
                return 1.0, "SpaTracker: missing GT-camera scale, scale=1"
            scale = summary.get("median_est_to_gt_scale_for_debug")
            if scale is not None:
                return float(scale), f"SpaTracker train global median gt/est from {path.relative_to(scene_dir)}"
            return _scale_from_normalizers(summary, f"SpaTracker {path.relative_to(scene_dir)}")
    return 1.0, "SpaTracker: no saved global normalizers, scale=1"


def _da3_camera_est_to_gt_scale(scene_dir: Path) -> tuple[float, str]:
    try:
        da_dir = scene_dir / "outputs" / "depth_anything_3"
        _intrs, c2ws = _load_da3_cameras(da_dir)
        gt = _da_load_gt(scene_dir)
        gt_to_da, _R, _t = __import__("eval_depth_depthanything")._gt_to_da_similarity(gt, c2ws)
        if float(gt_to_da) <= 1e-12:
            return 1.0, "DA3 GT-camera scale invalid, scale=1"
        return float(1.0 / float(gt_to_da)), "DA3 GT-camera scale DA->GT"
    except Exception as exc:
        return 1.0, f"DA3 GT-camera scale unavailable ({exc}), scale=1"


def _load_spatracker_depth(scene_dir: Path, split: str, frame: int, novel_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any]]:
    candidates = [
        scene_dir / "outputs_spatracker" / "spatracker_eval",
        scene_dir / "outputs" / "spatracker_eval",
    ]
    out_dir = next((p for p in candidates if p.is_dir()), candidates[0])
    if split == "train":
        summary_path = out_dir / "depth_eval.json"
        prefix = out_dir / "training_depth" / f"frame{int(frame):06d}"
    else:
        summary_path = out_dir / "novel_depth_eval.json"
        prefix = out_dir / "novel_depth" / f"novel{int(novel_id):06d}"
    summary = _json(summary_path)
    gt = np.load(prefix.with_name(prefix.name + "_gt.npy")).astype(np.float64)
    unscaled_path = prefix.with_name(prefix.name + "_pred_unscaled.npy")
    pred_path = unscaled_path if unscaled_path.is_file() else prefix.with_name(prefix.name + "_pred.npy")
    pred = np.load(pred_path).astype(np.float64)
    summary = dict(summary)
    summary["loaded_depth_path"] = str(pred_path)
    summary["loaded_depth_is_unscaled"] = bool(pred_path == unscaled_path)
    valid_path = prefix.with_name(prefix.name + "_valid.npy")
    valid = np.load(valid_path).astype(bool) if valid_path.is_file() else None
    return gt, pred, valid, summary


def _load_sidecar_depth(prefix: Path, fallback_gt: np.ndarray, prefer_unscaled_gt: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    pred = np.load(prefix.with_name(prefix.name + "_pred.npy")).astype(np.float64)
    valid_path = prefix.with_name(prefix.name + "_valid.npy")
    valid = np.load(valid_path).astype(bool) if valid_path.is_file() else None
    gt_path = prefix.with_name(prefix.name + ("_gt_unscaled.npy" if prefer_unscaled_gt else "_gt.npy"))
    if not gt_path.is_file() and prefer_unscaled_gt:
        gt_path = prefix.with_name(prefix.name + "_gt.npy")
    if gt_path.is_file():
        gt = np.load(gt_path).astype(np.float64)
        if gt.shape == fallback_gt.shape:
            fallback_gt = gt
    return fallback_gt, pred, valid


def _load_eval_items(scene_dir: Path, gt: np.ndarray, frame: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    only = set(args.only or [])

    def should_load(key: str) -> bool:
        key = key.lower()
        if args.no_ours and key == "ours":
            return False
        if args.no_mosca and key == "mosca":
            return False
        if args.no_spatracker and key == "spatracker":
            return False
        if args.no_da3 and key == "da3":
            return False
        if not only:
            return True
        if key == "da3":
            return "da3" in only
        return key in only

    if args.split == "train":
        if should_load("ours"):
            try:
                ours_summary = _json(scene_dir / "outputs" / "gaussian_depth_eval" / "gaussian_depth_eval.json")
                ours_median = _ours_union_median_scale(scene_dir, args)
                ours_camera = _ours_camera_est_to_gt_scale(scene_dir, args)
                ours_scale, ours_scale_source = _select_scale(_method_scale_mode(args, "ours"), ours_median, ours_camera)
                ours_base_scale = ours_scale
                ours_scale, ours_correction_source, ours_correction, ours_correction_l1 = _first_train_depth_correction(
                    scene_dir,
                    args,
                    "Ours",
                    ours_base_scale,
                    lambda: _load_npz_item(scene_dir / "outputs" / "gaussian_depth_eval" / "train_depth_arrays" / "train_depth_0000.npz"),
                )
                ours_gt, ours_depth, ours_valid = _load_npz_item(
                    scene_dir / "outputs" / "gaussian_depth_eval" / "train_depth_arrays" / f"train_depth_{int(frame):04d}.npz"
                )
                items.append(
                    _attach_depth_correction_report(
                        _with_scale_report(
                            _score_saved_arrays(
                            "Ours",
                            ours_gt,
                            ours_depth,
                            ours_valid,
                            args,
                            est_to_gt_scale=ours_scale,
                            scale_source=f"{ours_scale_source}; {ours_correction_source}",
                            ),
                            ours_median,
                            ours_camera,
                        ),
                        ours_base_scale,
                        ours_correction,
                        ours_correction_source,
                        ours_correction_l1,
                    )
                )
            except FileNotFoundError as exc:
                print(f"[depth-vis] skip Ours: {exc}")

        if should_load("mosca"):
            try:
                mosca_json = _json(scene_dir / "outputs_mosca" / "depth_eval" / "depth_l1_summary.json")
                mosca_median = _mosca_union_median_scale(scene_dir, args)
                mosca_camera = _mosca_camera_est_to_gt_scale(mosca_json)
                mosca_scale, mosca_scale_source = _select_scale(_method_scale_mode(args, "mosca"), mosca_median, mosca_camera)
                mosca_base_scale = mosca_scale
                mosca_scale, mosca_correction_source, mosca_correction, mosca_correction_l1 = _first_train_depth_correction(
                    scene_dir,
                    args,
                    "MoSca",
                    mosca_base_scale,
                    lambda: _load_sidecar_depth(
                        scene_dir / "outputs_mosca" / "depth_eval" / "training_depth" / "frame000000",
                        _read_depth(_depth_path_for_train(scene_dir, 0), args.gt_scale),
                        prefer_unscaled_gt=True,
                    ),
                )
                mosca_gt, pred, valid = _load_sidecar_depth(
                    scene_dir / "outputs_mosca" / "depth_eval" / "training_depth" / f"frame{int(frame):06d}",
                    gt,
                    prefer_unscaled_gt=True,
                )
                items.append(
                    _attach_depth_correction_report(
                        _with_scale_report(
                            _score_saved_arrays(
                            "MoSca",
                            mosca_gt,
                            pred,
                            valid,
                            args,
                            est_to_gt_scale=mosca_scale,
                            scale_source=f"{mosca_scale_source}; {mosca_correction_source}",
                            ),
                            mosca_median,
                            mosca_camera,
                        ),
                        mosca_base_scale,
                        mosca_correction,
                        mosca_correction_source,
                        mosca_correction_l1,
                    )
                )
            except FileNotFoundError as exc:
                print(f"[depth-vis] skip MoSca: {exc}")

        if should_load("spatracker"):
            try:
                spa_gt, spa_depth, spa_valid, spa_summary = _load_spatracker_depth(scene_dir, args.split, frame, int(args.novel_id))
                spa_median = (1.0, "SpaTracker median scale disabled; using GT-camera scale")
                spa_camera = _spatracker_est_to_gt_scale(scene_dir, args.split, "gt_camera")
                spa_base_scale = float(spa_camera[0])
                spa_scale, spa_correction_source, spa_correction, spa_correction_l1 = _first_train_depth_correction(
                    scene_dir,
                    args,
                    "SpaTracker",
                    spa_base_scale,
                    lambda: _load_spatracker_depth(scene_dir, "train", 0, int(args.novel_id))[:3],
                    correction_range=float(args.spatracker_depth_correction_range),
                )
                items.append(
                    _attach_depth_correction_report(
                        _with_scale_report(
                            _score_saved_arrays(
                            "SpaTracker",
                            spa_gt,
                            spa_depth,
                            spa_valid,
                            args,
                            est_to_gt_scale=spa_scale,
                            scale_source=f"{spa_camera[1]}; {spa_correction_source}",
                            ),
                            spa_median,
                            spa_camera,
                        ),
                        spa_base_scale,
                        spa_correction,
                        spa_correction_source,
                        spa_correction_l1,
                    )
                )
            except FileNotFoundError as exc:
                print(f"[depth-vis] skip SpaTracker: {exc}")

        if should_load("da3"):
            try:
                da_direct_median = _da3_render_union_median_scale(scene_dir, args)
                da_camera = _da3_camera_est_to_gt_scale(scene_dir)
                da_direct_scale, da_direct_scale_source = _select_scale(_method_scale_mode(args, "da3"), da_direct_median, da_camera)
                da_depths = _load_da3_depths(scene_dir / "outputs" / "depth_anything_3")
                da_direct_base_scale = da_direct_scale
                da_direct_scale, da_direct_correction_source, da_direct_correction, da_direct_correction_l1 = _first_train_depth_correction(
                    scene_dir,
                    args,
                    "DA3 direct",
                    da_direct_base_scale,
                    lambda: (_read_depth(_depth_path_for_train(scene_dir, 0), args.gt_scale), da_depths[0], None),
                )
                da_depth = da_depths[int(frame)]
                items.append(
                    _attach_depth_correction_report(
                        _with_scale_report(
                            _score_saved_arrays(
                            "DA3 direct",
                            gt,
                            da_depth,
                            None,
                            args,
                            est_to_gt_scale=da_direct_scale,
                            scale_source=f"{da_direct_scale_source}; {da_direct_correction_source}",
                            ),
                            da_direct_median,
                            da_camera,
                        ),
                        da_direct_base_scale,
                        da_direct_correction,
                        da_direct_correction_source,
                        da_direct_correction_l1,
                    )
                )
            except FileNotFoundError as exc:
                print(f"[depth-vis] skip DA3 direct: {exc}")

            try:
                da_render_json = _json(scene_dir / "outputs" / "depth_anything_3" / "train_render_depth_eval.json")
                da_render_median = _da3_render_union_median_scale(scene_dir, args)
                da_camera = _da3_camera_est_to_gt_scale(scene_dir)
                da_render_scale, da_render_scale_source = _select_scale(_method_scale_mode(args, "da3"), da_render_median, da_camera)
                da_dir = Path(da_render_json.get("rendered_depth_dir") or (scene_dir / "outputs" / "depth_anything_3" / "train_render_depth_arrays"))
                da_render_base_scale = da_render_scale
                da_render_scale, da_render_correction_source, da_render_correction, da_render_correction_l1 = _first_train_depth_correction(
                    scene_dir,
                    args,
                    "DA3 train-render",
                    da_render_base_scale,
                    lambda: (
                        np.load(da_dir / "train000000_gt.npy").astype(np.float64)
                        if (da_dir / "train000000_gt.npy").is_file()
                        else _read_depth(_depth_path_for_train(scene_dir, 0), args.gt_scale),
                        np.load(da_dir / "train000000_depth.npy").astype(np.float64),
                        np.load(da_dir / "train000000_valid.npy").astype(bool) if (da_dir / "train000000_valid.npy").is_file() else None,
                    ),
                )
                da_gt_path = da_dir / f"train{int(frame):06d}_gt.npy"
                da_gt = np.load(da_gt_path).astype(np.float64) if da_gt_path.is_file() else gt
                da_depth = np.load(da_dir / f"train{int(frame):06d}_depth.npy").astype(np.float64)
                da_valid_path = da_dir / f"train{int(frame):06d}_valid.npy"
                da_valid = np.load(da_valid_path).astype(bool) if da_valid_path.is_file() else None
                items.append(
                    _attach_depth_correction_report(
                        _with_scale_report(
                            _score_saved_arrays(
                            "DA3 train-render",
                            da_gt,
                            da_depth,
                            da_valid,
                            args,
                            est_to_gt_scale=da_render_scale,
                            scale_source=f"{da_render_scale_source}; {da_render_correction_source}",
                            ),
                            da_render_median,
                            da_camera,
                        ),
                        da_render_base_scale,
                        da_render_correction,
                        da_render_correction_source,
                        da_render_correction_l1,
                    )
                )
            except (FileNotFoundError, KeyError) as exc:
                print(f"[depth-vis] skip DA3 train-render: {exc}")
    else:
        novel_id = int(args.novel_id)
        if should_load("ours"):
            try:
                ours_summary = _json(scene_dir / "outputs" / "gaussian_depth_eval" / "gaussian_depth_eval.json")
                ours_median = _ours_union_median_scale(scene_dir, args)
                ours_camera = _ours_camera_est_to_gt_scale(scene_dir, args)
                ours_scale, ours_scale_source = _select_scale(_method_scale_mode(args, "ours"), ours_median, ours_camera)
                ours_base_scale = ours_scale
                ours_scale, ours_correction_source, ours_correction, ours_correction_l1 = _first_train_depth_correction(
                    scene_dir,
                    args,
                    "Ours",
                    ours_base_scale,
                    lambda: _load_npz_item(scene_dir / "outputs" / "gaussian_depth_eval" / "train_depth_arrays" / "train_depth_0000.npz"),
                )
                ours_gt, ours_depth, ours_valid = _load_npz_item(
                    scene_dir / "outputs" / "gaussian_depth_eval" / "novel" / "depth_arrays" / f"novel_depth_{novel_id:04d}.npz"
                )
                items.append(
                    _attach_depth_correction_report(
                        _with_scale_report(
                            _score_saved_arrays(
                            "Ours",
                            ours_gt,
                            ours_depth,
                            ours_valid,
                            args,
                            est_to_gt_scale=ours_scale,
                            scale_source=f"{ours_scale_source}; {ours_correction_source}",
                            ),
                            ours_median,
                            ours_camera,
                        ),
                        ours_base_scale,
                        ours_correction,
                        ours_correction_source,
                        ours_correction_l1,
                    )
                )
            except FileNotFoundError as exc:
                print(f"[depth-vis] skip Ours: {exc}")

        if should_load("mosca"):
            try:
                mosca_json = _json(scene_dir / "outputs_mosca" / "depth_eval" / "depth_l1_summary.json")
                mosca_median = _mosca_union_median_scale(scene_dir, args)
                mosca_camera = _mosca_camera_est_to_gt_scale(mosca_json)
                mosca_scale, mosca_scale_source = _select_scale(_method_scale_mode(args, "mosca"), mosca_median, mosca_camera)
                mosca_base_scale = mosca_scale
                mosca_scale, mosca_correction_source, mosca_correction, mosca_correction_l1 = _first_train_depth_correction(
                    scene_dir,
                    args,
                    "MoSca",
                    mosca_base_scale,
                    lambda: _load_sidecar_depth(
                        scene_dir / "outputs_mosca" / "depth_eval" / "training_depth" / "frame000000",
                        _read_depth(_depth_path_for_train(scene_dir, 0), args.gt_scale),
                        prefer_unscaled_gt=True,
                    ),
                )
                mosca_gt, pred, valid = _load_sidecar_depth(
                    scene_dir / "outputs_mosca" / "depth_eval" / "novel_depth" / f"novel{novel_id:06d}",
                    gt,
                    prefer_unscaled_gt=True,
                )
                items.append(
                    _attach_depth_correction_report(
                        _with_scale_report(
                            _score_saved_arrays(
                            "MoSca",
                            mosca_gt,
                            pred,
                            valid,
                            args,
                            est_to_gt_scale=mosca_scale,
                            scale_source=f"{mosca_scale_source}; {mosca_correction_source}",
                            ),
                            mosca_median,
                            mosca_camera,
                        ),
                        mosca_base_scale,
                        mosca_correction,
                        mosca_correction_source,
                        mosca_correction_l1,
                    )
                )
            except FileNotFoundError as exc:
                print(f"[depth-vis] skip MoSca: {exc}")

        if should_load("spatracker"):
            try:
                spa_gt, spa_depth, spa_valid, spa_summary = _load_spatracker_depth(scene_dir, args.split, frame, novel_id)
                spa_median = (1.0, "SpaTracker median scale disabled; using GT-camera scale")
                spa_camera = _spatracker_est_to_gt_scale(scene_dir, args.split, "gt_camera")
                spa_base_scale = float(spa_camera[0])
                spa_scale, spa_correction_source, spa_correction, spa_correction_l1 = _first_train_depth_correction(
                    scene_dir,
                    args,
                    "SpaTracker",
                    spa_base_scale,
                    lambda: _load_spatracker_depth(scene_dir, "train", 0, novel_id)[:3],
                    correction_range=float(args.spatracker_depth_correction_range),
                )
                items.append(
                    _attach_depth_correction_report(
                        _with_scale_report(
                            _score_saved_arrays(
                            "SpaTracker",
                            spa_gt,
                            spa_depth,
                            spa_valid,
                            args,
                            est_to_gt_scale=spa_scale,
                            scale_source=f"{spa_camera[1]}; {spa_correction_source}",
                            ),
                            spa_median,
                            spa_camera,
                        ),
                        spa_base_scale,
                        spa_correction,
                        spa_correction_source,
                        spa_correction_l1,
                    )
                )
            except FileNotFoundError as exc:
                print(f"[depth-vis] skip SpaTracker: {exc}")

        if should_load("da3"):
            da_render_json = _json(scene_dir / "outputs" / "depth_anything_3" / "novel_depth_eval.json")
            da_render_median = _da3_render_union_median_scale(scene_dir, args)
            da_camera = _da3_camera_est_to_gt_scale(scene_dir)
            da_render_scale, da_render_scale_source = _select_scale(_method_scale_mode(args, "da3"), da_render_median, da_camera)
            da_dir = Path(da_render_json.get("rendered_depth_dir") or (scene_dir / "outputs" / "depth_anything_3" / "novel_render_depth_arrays"))
            train_da_dir = scene_dir / "outputs" / "depth_anything_3" / "train_render_depth_arrays"
            da_render_base_scale = da_render_scale
            da_render_scale, da_render_correction_source, da_render_correction, da_render_correction_l1 = _first_train_depth_correction(
                scene_dir,
                args,
                "DA3 novel-render",
                da_render_base_scale,
                lambda: (
                    np.load(train_da_dir / "train000000_gt.npy").astype(np.float64)
                    if (train_da_dir / "train000000_gt.npy").is_file()
                    else _read_depth(_depth_path_for_train(scene_dir, 0), args.gt_scale),
                    np.load(train_da_dir / "train000000_depth.npy").astype(np.float64),
                    np.load(train_da_dir / "train000000_valid.npy").astype(bool) if (train_da_dir / "train000000_valid.npy").is_file() else None,
                ),
            )
            da_gt_path = da_dir / f"novel{novel_id:06d}_gt.npy"
            da_gt = np.load(da_gt_path).astype(np.float64) if da_gt_path.is_file() else gt
            da_depth = np.load(da_dir / f"novel{novel_id:06d}_depth.npy").astype(np.float64)
            da_valid_path = da_dir / f"novel{novel_id:06d}_valid.npy"
            da_valid = np.load(da_valid_path).astype(bool) if da_valid_path.is_file() else None
            items.append(
                _attach_depth_correction_report(
                    _with_scale_report(
                        _score_saved_arrays(
                        "DA3 novel-render",
                        da_gt,
                        da_depth,
                        da_valid,
                        args,
                        est_to_gt_scale=da_render_scale,
                        scale_source=f"{da_render_scale_source}; {da_render_correction_source}",
                        ),
                        da_render_median,
                        da_camera,
                    ),
                    da_render_base_scale,
                    da_render_correction,
                    da_render_correction_source,
                    da_render_correction_l1,
                )
            )
    return items


def main() -> None:
    args = _parse_args()
    scene_dir = args.scene_dir.resolve()
    out_dir = (args.output_dir or (scene_dir / "outputs" / "depth_visualization")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    gt, frame, gt_path, label = _target_info(scene_dir, args)

    items: list[dict[str, Any]] = []
    errors: dict[str, Any] = {
        "scene_dir": str(scene_dir),
        "split": args.split,
        "frame": int(frame),
        "gt_depth_path": str(gt_path),
        "normalization": "prediction scaled to GT units with the selected scale policy; PNGs and L1 use the same scaled arrays",
        "scale_policy": {
            "default": args.scale_mode,
            "ours": _method_scale_mode(args, "ours"),
            "mosca": _method_scale_mode(args, "mosca"),
            "spatracker": _method_scale_mode(args, "spatracker"),
            "da3": _method_scale_mode(args, "da3"),
        },
        "depth_correction": {
            "fit_split": "train",
            "fit_frame": 0,
            "range": [max(1.0 - max(float(args.depth_correction_range), 0.0), 1e-12), 1.0 + max(float(args.depth_correction_range), 0.0)],
            "spatracker_range": [
                max(1.0 - max(float(args.spatracker_depth_correction_range), 0.0), 1e-12),
                1.0 + max(float(args.spatracker_depth_correction_range), 0.0),
            ],
            "samples": max(int(args.depth_correction_samples), 3),
            "disabled": max(float(args.depth_correction_range), 0.0) <= 0,
            "spatracker_disabled": max(float(args.spatracker_depth_correction_range), 0.0) <= 0,
        },
        "error_visualization_clip": float(args.error_clip),
        "methods": {},
    }

    if args.combine_existing:
        items = _load_existing_items(out_dir, label)
        for item in items:
            errors["methods"][str(item["name"])] = {
                "l1": item.get("l1"),
                "raw_l1": item.get("raw_l1"),
                "relative_l1": item.get("relative_l1"),
                "valid_px": item.get("valid_px"),
                "est_to_gt_scale": item.get("est_to_gt_scale"),
                "scale_source": item.get("scale_source"),
                "median_est_to_gt_scale": item.get("median_est_to_gt_scale"),
                "median_scale_source": item.get("median_scale_source"),
                "gt_camera_est_to_gt_scale": item.get("gt_camera_est_to_gt_scale"),
                "gt_camera_scale_source": item.get("gt_camera_scale_source"),
                "base_est_to_gt_scale": item.get("base_est_to_gt_scale"),
                "depth_correction": item.get("depth_correction"),
                "depth_correction_source": item.get("depth_correction_source"),
                "depth_correction_train0_l1": item.get("depth_correction_train0_l1"),
            }
        panel_path = out_dir / f"{label}_depth_comparison.png"
        _write_panel(panel_path, gt, items, args)
        json_path = out_dir / f"{label}_depth_comparison.json"
        json_path.write_text(json.dumps(errors, indent=2, sort_keys=True) + "\n")
        print(f"[depth-vis] combined {len(items)} method(s)")
        print(f"[depth-vis] saved {panel_path}")
        print(f"[depth-vis] saved {json_path}")
        return

    only = set(args.only or [])
    try:
        loaded_items = _load_eval_items(scene_dir, gt, frame, args)
    except Exception as exc:
        raise RuntimeError(
            "Could not load evaluation arrays. Re-run the depth evaluators first so they save "
            "per-frame train/novel depth arrays."
        ) from exc

    def keep_item(name: str) -> bool:
        key = name.lower()
        if args.no_ours and key == "ours":
            return False
        if args.no_mosca and key == "mosca":
            return False
        if args.no_spatracker and key == "spatracker":
            return False
        if args.no_da3 and key.startswith("da3"):
            return False
        if not only:
            return True
        if "da3" in only and key.startswith("da3"):
            return True
        return key in only

    for item in loaded_items:
        if not keep_item(str(item["name"])):
            continue
        items.append(item)
        errors["methods"][str(item["name"])] = {
            "l1": item["l1"],
            "raw_l1": item.get("raw_l1"),
            "relative_l1": item.get("relative_l1"),
            "valid_px": item["valid_px"],
            "est_to_gt_scale": item.get("est_to_gt_scale"),
            "scale_source": item.get("scale_source"),
            "median_est_to_gt_scale": item.get("median_est_to_gt_scale"),
            "median_scale_source": item.get("median_scale_source"),
            "gt_camera_est_to_gt_scale": item.get("gt_camera_est_to_gt_scale"),
            "gt_camera_scale_source": item.get("gt_camera_scale_source"),
            "base_est_to_gt_scale": item.get("base_est_to_gt_scale"),
            "depth_correction": item.get("depth_correction"),
            "depth_correction_source": item.get("depth_correction_source"),
            "depth_correction_train0_l1": item.get("depth_correction_train0_l1"),
        }
        print(
            "[depth-vis] L1 "
            f"{item['name']}: raw={item.get('raw_l1')} "
            f"relative={item.get('relative_l1')} "
            f"valid_px={item['valid_px']} "
            f"est_to_gt_scale={item.get('est_to_gt_scale')} "
            f"depth_correction={item.get('depth_correction')} "
            f"median_scale={item.get('median_est_to_gt_scale')} "
            f"gt_camera_scale={item.get('gt_camera_est_to_gt_scale')} "
            f"source={item.get('scale_source')}"
        )
        _save_method_outputs(out_dir, label, item)

    panel_path = out_dir / f"{label}_depth_comparison.png"
    _write_panel(panel_path, gt, items, args)
    json_path = out_dir / f"{label}_depth_comparison.json"
    json_path.write_text(json.dumps(errors, indent=2, sort_keys=True) + "\n")
    print(f"[depth-vis] saved {panel_path}")
    print(f"[depth-vis] saved {json_path}")


if __name__ == "__main__":
    main()
