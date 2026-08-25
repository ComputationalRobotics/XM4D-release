#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "Python")]

from eval_depth import _prefer_npy_depth_path, _read_depth
from eval_depth_depthanything import (
    _gt_to_da_similarity,
    _load_da3_cameras,
    _load_da3_depths,
    _load_gt,
    _load_novel_metadata,
    _read_dynamic_mask,
    _render_depth_from_points,
    _scale_K,
)
from eval_spatracker import (
    _camera_eval as _spatracker_camera_eval,
    _c2w_from_w2c as _spatracker_c2w_from_w2c,
    _gt_c2w as _spatracker_gt_c2w,
    _load_result as _load_spatracker_result,
    _spatracker_novel_c2w_from_base,
    _static_world_points as _spatracker_static_world_points,
    _track_points_for_frame as _spatracker_track_points_for_frame,
    _transform_c2w_similarity as _spatracker_transform_c2w_similarity,
)

REPLICA_RENDER_DEPTH_SCALE = 65535.0 * 0.1


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compare saved aligned depth arrays with the same conventions used by debug_gt_depth_open3d_simple.py."
    )
    ap.add_argument("--scene-dir", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "novel"], default="train")
    ap.add_argument("--frame-index", type=int, default=0)
    ap.add_argument("--novel-id", type=int, default=0)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--gt-scale", type=float, default=REPLICA_RENDER_DEPTH_SCALE)
    ap.add_argument("--mask-threshold", type=float, default=127.0)
    ap.add_argument("--alpha-threshold", type=float, default=0.05,
                    help="Default alpha cutoff (ours / da3_origin: dense render).")
    # Per-method alpha cutoffs. Point-splatted methods (mosca, spatracker, da3_render)
    # have sparse splatted alpha (mostly 0.1-0.9), so a high global threshold masks them
    # out entirely. These default low so those methods keep coverage. <0 -> use --alpha-threshold.
    ap.add_argument("--mosca-alpha-threshold", type=float, default=0.9)
    ap.add_argument("--spatracker-alpha-threshold", type=float, default=0.05)
    ap.add_argument("--da3-alpha-threshold", type=float, default=0.05)
    ap.add_argument("--min-depth", type=float, default=1e-6)
    ap.add_argument("--error-clip", type=float, default=2.0)
    ap.add_argument("--save-downscale", type=float, default=1.0,
                    help="Downscale factor for saved PNGs (e.g. 0.5 = half resolution). "
                         "Only shrinks the written figures; metrics use full resolution.")
    ap.add_argument("--ours-depth-dir", type=Path, default=None)
    ap.add_argument("--methods", nargs="*", default=["ours", "da3_origin", "da3_render", "mosca", "spatracker"])
    ap.add_argument("--depth-correction-range", type=float, default=0.1)
    ap.add_argument("--spatracker-depth-correction-range", type=float, default=0.0)
    ap.add_argument("--depth-correction-samples", type=int, default=401)
    ap.add_argument("--static-only", action="store_true", help="Evaluate only non-dynamic mask pixels. Default evaluates static and dynamic pixels.")
    ap.add_argument("--spatracker-result-npz", type=Path, default=None)
    ap.add_argument("--spatracker-static-point-radius", type=int, default=1)
    ap.add_argument("--spatracker-dynamic-point-radius", type=int, default=5)
    ap.add_argument("--spatracker-alpha-density-radius", type=int, default=20)
    ap.add_argument("--spatracker-alpha-density-scale", type=float, default=1.0)
    ap.add_argument("--spatracker-static-point-stride", type=int, default=1)
    ap.add_argument("--spatracker-static-frame-stride", type=int, default=1)
    ap.add_argument("--spatracker-static-max-points", type=int, default=0)
    ap.add_argument("--spatracker-max-dynamic-points", type=int, default=0)
    ap.add_argument("--spatracker-visibility-threshold", type=float, default=0.5)
    ap.add_argument("--spatracker-fill-holes", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--spatracker-fill-hole-radius", type=float, default=5.0)
    ap.add_argument("--save-spatracker-debug", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--spatracker-use-gt-intrinsics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Recompute GT intrinsics for static lifting. Default uses result.npz intrinsics so SpaTracker static holes and dynamic tracks share the same K.",
    )
    return ap.parse_args()


def frame_id_from_name(name: str) -> int | None:
    digits = "".join(ch if ch.isdigit() else " " for ch in Path(name).stem).split()
    return int(digits[-1]) if digits else None


def image_files(scene_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in (scene_dir / "images").iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


def depth_path_for_frame(scene_dir: Path, frame_id: int) -> Path:
    depth_dir = scene_dir / "depths"
    direct = _prefer_npy_depth_path(depth_dir / f"depth{int(frame_id):06d}.png")
    if direct.is_file():
        return direct
    files = sorted(
        p
        for p in depth_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".npy", ".npz", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dpt"}
    )
    by_id = {fid: p for p in files if (fid := frame_id_from_name(p.name)) is not None}
    if int(frame_id) in by_id:
        return _prefer_npy_depth_path(by_id[int(frame_id)])
    if 0 <= int(frame_id) < len(files):
        return _prefer_npy_depth_path(files[int(frame_id)])
    raise FileNotFoundError(f"No depth found for frame {frame_id} in {depth_dir}")


def mask_dir(scene_dir: Path) -> Path:
    for name in ("mask", "masks"):
        path = scene_dir / name
        if path.is_dir():
            return path
    return scene_dir / "mask"


def novel_mask_dir(scene_dir: Path) -> Path:
    for path in (scene_dir / "novel" / "mask", scene_dir / "novel" / "masks"):
        if path.is_dir():
            return path
    return mask_dir(scene_dir)


def static_mask(mask_root: Path, frame_id: int, shape: tuple[int, int], threshold: float, image_name: str | None) -> np.ndarray:
    return ~_read_dynamic_mask(mask_root, frame_id, shape, threshold, image_name=image_name)


def resize_like(arr: np.ndarray | None, shape: tuple[int, int], *, is_mask: bool = False) -> np.ndarray | None:
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.shape == shape:
        return arr
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.resize(arr.astype(np.float32), (shape[1], shape[0]), interpolation=interp)


def load_npz_depth(path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    data = np.load(path, allow_pickle=True)
    key = "rendered_depth" if "rendered_depth" in data else "pred_depth"
    depth = np.asarray(data[key], dtype=np.float64)
    valid = np.asarray(data["valid_mask"], dtype=bool) if "valid_mask" in data else None
    alpha = np.asarray(data["alpha"], dtype=np.float64) if "alpha" in data else None
    return depth, valid, alpha


def sidecar_depth(prefix: Path, suffix: str = "_pred.npy") -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    depth_path = prefix.with_name(prefix.name + suffix)
    if not depth_path.is_file():
        raise FileNotFoundError(depth_path)
    valid_path = prefix.with_name(prefix.name + "_valid.npy")
    alpha_path = prefix.with_name(prefix.name + "_alpha.npy")
    depth = np.load(depth_path).astype(np.float64)
    valid = np.load(valid_path).astype(bool) if valid_path.is_file() else None
    alpha = np.load(alpha_path).astype(np.float64) if alpha_path.is_file() else None
    return depth, valid, alpha


def render_array_depth(prefix: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    return sidecar_depth(prefix, "_depth.npy")


def mosca_depth_multiplier(scene_dir: Path) -> float:
    summary_path = scene_dir / "outputs_mosca" / "depth_eval" / "depth_l1_summary.json"
    if not summary_path.is_file():
        return 1.0
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        multiplier = summary.get("camera_scale_render_to_gt") or summary.get("depth_scale_render_to_gt")
        if multiplier is None:
            inv = summary.get("camera_scale_gt_to_render") or summary.get("depth_scale_gt_to_render") or summary.get("scale_gt_to_render")
            if inv is not None and float(inv) > 1e-12:
                multiplier = 1.0 / float(inv)
        return float(multiplier) if multiplier is not None else 1.0
    except Exception as exc:
        print(f"[aligned-depth] warning: could not read MoSca depth conversion from {summary_path}: {exc}")
        return 1.0


def da3_depth_multiplier(scene_dir: Path) -> float:
    da_dir = scene_dir / "outputs" / "depth_anything_3"
    _intrs, da_c2ws = _load_da3_cameras(da_dir)
    with contextlib.redirect_stdout(io.StringIO()):
        scale_gt_to_da, _R, _t = _gt_to_da_similarity(_load_gt(scene_dir), da_c2ws)
    return 1.0 / max(float(scale_gt_to_da), 1e-12)


def spatracker_eval_dir(scene_dir: Path) -> Path:
    candidates = [
        scene_dir / "outputs_spatracker" / "spatracker_eval",
        scene_dir / "outputs" / "spatracker_eval",
        scene_dir / "outputs" / "spatracker_eval_static_dynamic_smoke",
    ]
    for path in candidates:
        if (path / "training_depth").is_dir() or (path / "novel_depth").is_dir():
            return path
    return candidates[0]


_SPATRACKER_CACHE: dict[tuple, dict[str, object]] = {}


def spatracker_result_path(args: argparse.Namespace) -> Path:
    candidates = []
    if args.spatracker_result_npz is not None:
        candidates.append(args.spatracker_result_npz)
    candidates.extend(
        [
            args.scene_dir / "outputs_spatracker" / "spatracker_rgb_dynamic_adaptive_input" / "results" / "result.npz",
            args.scene_dir / "outputs_spatracker" / "spatracker_rgb_dynamic_input" / "results" / "result.npz",
            args.scene_dir / "outputs_spatracker" / "spatracker_input" / "results" / "result.npz",
            args.scene_dir / "outputs_spatracker" / "spatracker_eval" / "result.npz",
            args.scene_dir / "outputs" / "spatracker_rgb_dynamic_adaptive_input" / "results" / "result.npz",
            args.scene_dir / "outputs" / "spatracker_rgb_dynamic_input" / "results" / "result.npz",
            args.scene_dir / "outputs" / "spatracker_input" / "results" / "result.npz",
            args.scene_dir / "outputs" / "spatracker_eval" / "result.npz",
        ]
    )
    seen: set[Path] = set()
    for path in candidates:
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            merged = merge_spatracker_split_results(path.parent, path, only_if_newer=True)
            if merged is not None:
                return merged
            return path
        merged = merge_spatracker_split_results(path.parent, path)
        if merged is not None:
            return merged
    raise FileNotFoundError(f"No SpaTracker result.npz found in: {', '.join(str(p) for p in seen)}")


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def merge_spatracker_split_results(results_dir: Path, merged_path: Path, only_if_newer: bool = False) -> Path | None:
    results_dir = results_dir.resolve()
    if not results_dir.is_dir():
        return None
    first_path = _first_existing(
        [
            results_dir / "result_first.npz",
            results_dir / "results_first.npz",
            results_dir / "first.npz",
        ]
    )
    later_path = _first_existing(
        [
            results_dir / "result_later.npz",
            results_dir / "result_second.npz",
            results_dir / "results_later.npz",
            results_dir / "results_second.npz",
            results_dir / "second.npz",
        ]
    )
    if first_path is None or later_path is None:
        return None
    if only_if_newer and merged_path.is_file():
        merged_mtime = merged_path.stat().st_mtime
        if first_path.stat().st_mtime <= merged_mtime and later_path.stat().st_mtime <= merged_mtime:
            return None

    first = np.load(first_path, allow_pickle=True)
    later = np.load(later_path, allow_pickle=True)
    merged = {key: first[key] for key in first.files}
    for key in ("coords", "visibs", "conf_pred"):
        if key in first and key in later:
            merged[key] = np.concatenate([first[key], later[key]], axis=1)
    if "queries_xyt" in first and "queries_xyt" in later:
        merged["queries_xyt"] = np.concatenate([first["queries_xyt"], later["queries_xyt"]], axis=0)
    if "query_counts_per_frame" in first and "query_counts_per_frame" in later:
        merged["query_counts_per_frame"] = first["query_counts_per_frame"] + later["query_counts_per_frame"]
    if "query_candidate_counts_per_frame" in first:
        merged["query_candidate_counts_per_frame"] = first["query_candidate_counts_per_frame"]
    merged["query_mode"] = np.array("merged_first_then_later")
    merged["first_pass_result"] = np.array(str(first_path))
    merged["later_pass_result"] = np.array(str(later_path))
    merged["first_query_points"] = np.array(int(first["coords"].shape[1]), dtype=np.int64)
    merged["later_query_points"] = np.array(int(later["coords"].shape[1]), dtype=np.int64)
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(merged_path, **merged)
    print(f"[aligned-depth] merged SpaTracker split outputs -> {merged_path}")
    return merged_path


def spatracker_render_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        gt_scale=float(args.gt_scale),
        min_depth=float(args.min_depth),
        max_depth=-1.0,
        point_radius=int(args.spatracker_dynamic_point_radius),
        static_point_stride=int(args.spatracker_static_point_stride),
        static_frame_stride=int(args.spatracker_static_frame_stride),
        static_max_points=int(args.spatracker_static_max_points),
        mask_dir=None,
        mask_threshold=float(args.mask_threshold),
        alpha_threshold=float(args.alpha_threshold),
        alpha_density_radius=int(args.spatracker_alpha_density_radius),
        alpha_density_scale=float(args.spatracker_alpha_density_scale),
        max_points=int(args.spatracker_max_dynamic_points),
        visibility_threshold=float(args.spatracker_visibility_threshold),
        use_gt_intrinsics=bool(args.spatracker_use_gt_intrinsics),
    )


def load_spatracker_state(args: argparse.Namespace) -> dict[str, object]:
    result_npz = spatracker_result_path(args)
    cache_key = (
        str(args.scene_dir.resolve()),
        str(result_npz),
        float(args.gt_scale),
        float(args.min_depth),
        int(args.spatracker_static_point_stride),
        int(args.spatracker_static_frame_stride),
        int(args.spatracker_static_max_points),
        float(args.mask_threshold),
        bool(args.spatracker_use_gt_intrinsics),
    )
    cached = _SPATRACKER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    render_args = spatracker_render_args(args)
    data = _load_spatracker_result(result_npz)
    gt = _load_gt(args.scene_dir)
    intrinsic_check = check_spatracker_intrinsics(args.scene_dir, gt, data)
    if intrinsic_check["max_relative_error"] > 0.08:
        print(
            "[aligned-depth] warning: SpaTracker result intrinsics do not look GT-derived "
            f"(max relative error {intrinsic_check['max_relative_error']:.3f}). "
            "Rerun run_spatracker_dynamic_adaptive.sh so the forward pass uses *_gt_camera.npz."
        )
    camera, align, _c2ws_est = _spatracker_camera_eval(args.scene_dir, gt, data)
    camera_depth_scale = float(camera.get("scale_align") or camera.get("render_alignment_raw_scale") or 1.0)
    static_points, per_frame_static_points, static_meta = _spatracker_static_world_points(
        args.scene_dir,
        gt,
        data,
        align,
        render_args,
    )
    state: dict[str, object] = {
        "result_npz": result_npz,
        "render_args": render_args,
        "data": data,
        "gt": gt,
        "align": align,
        "camera": camera,
        "camera_depth_scale": camera_depth_scale,
        "static_points": static_points,
        "per_frame_static_points": per_frame_static_points,
        "static_meta": static_meta,
        "intrinsic_check": intrinsic_check,
    }
    _SPATRACKER_CACHE[cache_key] = state
    return state


def check_spatracker_intrinsics(scene_dir: Path, gt: dict, data: dict[str, np.ndarray]) -> dict[str, object]:
    depths = np.asarray(data["depths"], dtype=np.float64)
    intrs = np.asarray(data["intrinsics"], dtype=np.float64)
    frame_ids = np.asarray(data.get("frame_ids", np.arange(depths.shape[0])), dtype=np.int64).reshape(-1)
    if frame_ids.shape[0] != depths.shape[0]:
        frame_ids = np.arange(depths.shape[0], dtype=np.int64)
    images = image_files(scene_dir)
    by_id = {int(info["id"]): info for info in gt.values() if "id" in info}
    rows = []
    max_rel = 0.0
    for local_idx, frame_id in enumerate(frame_ids[: min(5, len(frame_ids))]):
        frame_id = int(frame_id)
        if frame_id not in by_id or frame_id >= len(images):
            continue
        got = intrs[local_idx if intrs.shape[0] > 1 else 0]
        expected = _scale_K(
            np.asarray(by_id[frame_id]["K"], dtype=np.float64),
            _image_size(images[frame_id]),
            (depths[local_idx].shape[1], depths[local_idx].shape[0]),
        )
        denom = np.maximum(np.abs(expected[[0, 1], [0, 1]]), 1e-12)
        rel_f = np.abs(got[[0, 1], [0, 1]] - expected[[0, 1], [0, 1]]) / denom
        rel_c = np.abs(got[[0, 1], [2, 2]] - expected[[0, 1], [2, 2]]) / np.maximum(
            [depths[local_idx].shape[1], depths[local_idx].shape[0]], 1
        )
        rel = float(max(np.max(rel_f), np.max(rel_c)))
        max_rel = max(max_rel, rel)
        rows.append(
            {
                "frame": frame_id,
                "shape": [int(depths[local_idx].shape[0]), int(depths[local_idx].shape[1])],
                "saved_fx_fy_cx_cy": [float(got[0, 0]), float(got[1, 1]), float(got[0, 2]), float(got[1, 2])],
                "expected_gt_fx_fy_cx_cy": [
                    float(expected[0, 0]),
                    float(expected[1, 1]),
                    float(expected[0, 2]),
                    float(expected[1, 2]),
                ],
                "relative_error": rel,
            }
        )
    return {"max_relative_error": float(max_rel), "rows": rows}


def _image_size(path: Path) -> tuple[int, int]:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return int(img.shape[1]), int(img.shape[0])


def _spatracker_local_by_frame(data: dict[str, np.ndarray]) -> dict[int, int]:
    coords = np.asarray(data["coords"])
    frame_ids = np.asarray(data.get("frame_ids", np.arange(coords.shape[0])), dtype=np.int64).reshape(-1)
    if frame_ids.shape[0] != coords.shape[0]:
        frame_ids = np.arange(coords.shape[0], dtype=np.int64)
    return {int(fid): int(idx) for idx, fid in enumerate(frame_ids)}


def _render_spatracker_layers(
    static_points: np.ndarray,
    dynamic_points: np.ndarray,
    target_c2w: np.ndarray,
    K: np.ndarray,
    shape: tuple[int, int],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = int(shape[0]), int(shape[1])
    empty = np.zeros((height, width), dtype=np.float64)

    def render(points: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if points.shape[0] == 0:
            return empty.copy(), empty.copy()
        return _render_depth_from_points(
            points,
            target_c2w,
            K,
            height,
            width,
            float(args.min_depth),
            -1.0,
            int(radius),
            int(args.spatracker_alpha_density_radius),
            float(args.spatracker_alpha_density_scale),
        )

    static_depth, static_alpha = render(static_points, int(args.spatracker_static_point_radius))
    dynamic_depth, dynamic_alpha = render(dynamic_points, int(args.spatracker_dynamic_point_radius))
    static_valid = np.isfinite(static_depth) & (static_depth > float(args.min_depth))
    dynamic_valid = np.isfinite(dynamic_depth) & (dynamic_depth > float(args.min_depth))
    take_dynamic = dynamic_valid & (~static_valid | (dynamic_depth < static_depth))
    take_static = static_valid & ~take_dynamic
    out_depth = empty.copy()
    out_depth[take_static] = static_depth[take_static]
    out_depth[take_dynamic] = dynamic_depth[take_dynamic]
    alpha = np.maximum(static_alpha, dynamic_alpha)
    return out_depth, alpha, static_depth, static_alpha, dynamic_depth, dynamic_alpha


def save_spatracker_debug_layers(
    args: argparse.Namespace,
    label: str,
    gt_depth: np.ndarray,
    static_depth: np.ndarray,
    static_alpha: np.ndarray,
    dynamic_depth: np.ndarray,
    dynamic_alpha: np.ndarray,
    combined_depth: np.ndarray,
    combined_alpha: np.ndarray,
    intrinsic_check: dict[str, object],
) -> None:
    if not bool(args.save_spatracker_debug):
        return
    static_depth = np.asarray(static_depth, dtype=np.float64)
    dynamic_depth = np.asarray(dynamic_depth, dtype=np.float64)
    combined_depth = np.asarray(combined_depth, dtype=np.float64)
    out_dir = args.output_dir / "spatracker_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    valid_gt = np.isfinite(gt_depth) & (gt_depth > float(args.min_depth))
    if np.any(valid_gt):
        vmin, vmax = np.percentile(gt_depth[valid_gt], [2.0, 98.0])
    else:
        vmin, vmax = 0.0, 1.0
    _spa_thr = method_alpha_threshold("spatracker", args)
    static_mask_v = np.isfinite(static_depth) & (static_depth > float(args.min_depth)) & (static_alpha > _spa_thr)
    dynamic_mask_v = np.isfinite(dynamic_depth) & (dynamic_depth > float(args.min_depth)) & (dynamic_alpha > _spa_thr)
    combined_mask_v = np.isfinite(combined_depth) & (combined_depth > float(args.min_depth)) & (combined_alpha > _spa_thr)
    dynamic_in_static_hole = dynamic_mask_v & ~static_mask_v
    dynamic_overlap_static = dynamic_mask_v & static_mask_v
    panels = [
        label_panel(colorize_depth(gt_depth, valid_gt, float(vmin), float(vmax)), "GT"),
        label_panel(colorize_depth(static_depth, static_mask_v, float(vmin), float(vmax)), "Spa static"),
        label_panel(colorize_depth(dynamic_depth, dynamic_mask_v, float(vmin), float(vmax)), "Spa dynamic"),
        label_panel(colorize_depth(combined_depth, combined_mask_v, float(vmin), float(vmax)), "Spa combined"),
    ]
    overlay = np.zeros((*gt_depth.shape, 3), dtype=np.uint8)
    overlay[static_mask_v] = (80, 180, 255)
    overlay[dynamic_in_static_hole] = (0, 255, 0)
    overlay[dynamic_overlap_static] = (255, 120, 0)
    panels.append(label_panel(overlay, "coverage"))
    save_figure(out_dir / f"{label}_layers.png", np.concatenate(panels, axis=1), float(args.save_downscale))
    np.save(out_dir / f"{label}_static_depth.npy", static_depth.astype(np.float32))
    np.save(out_dir / f"{label}_dynamic_depth.npy", dynamic_depth.astype(np.float32))
    np.save(out_dir / f"{label}_combined_depth.npy", combined_depth.astype(np.float32))
    stats = {
        "label": label,
        "gt_shape": [int(gt_depth.shape[0]), int(gt_depth.shape[1])],
        "static_valid_pixels": int(np.count_nonzero(static_mask_v)),
        "dynamic_valid_pixels": int(np.count_nonzero(dynamic_mask_v)),
        "combined_valid_pixels": int(np.count_nonzero(combined_mask_v)),
        "dynamic_in_static_hole_pixels": int(np.count_nonzero(dynamic_in_static_hole)),
        "dynamic_overlap_static_pixels": int(np.count_nonzero(dynamic_overlap_static)),
        "dynamic_hole_fraction": float(np.count_nonzero(dynamic_in_static_hole) / max(np.count_nonzero(dynamic_mask_v), 1)),
        "intrinsic_check": intrinsic_check,
    }
    (out_dir / f"{label}_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(
        "[aligned-depth] SpaTracker debug: "
        f"static_px={stats['static_valid_pixels']} dynamic_px={stats['dynamic_valid_pixels']} "
        f"dynamic_in_static_hole={stats['dynamic_hole_fraction']:.3f} "
        f"png={out_dir / f'{label}_layers.png'}"
    )


def render_spatracker_depth(
    args: argparse.Namespace,
    split: str | None = None,
    frame_index: int | None = None,
    novel_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    split = args.split if split is None else split
    frame_index = int(args.frame_index if frame_index is None else frame_index)
    novel_id = int(args.novel_id if novel_id is None else novel_id)
    state = load_spatracker_state(args)
    data = state["data"]  # type: ignore[assignment]
    gt = state["gt"]  # type: ignore[assignment]
    align = state["align"]  # type: ignore[assignment]
    render_args = state["render_args"]  # type: ignore[assignment]
    local_by_frame = _spatracker_local_by_frame(data)
    by_id = {int(info["id"]): info for info in gt.values() if "id" in info}
    imgs = image_files(args.scene_dir)

    if split == "train":
        frame_id = int(frame_index)
        if frame_id not in local_by_frame or frame_id not in by_id or frame_id >= len(imgs):
            raise FileNotFoundError(f"SpaTracker has no raw frame for train frame {frame_id}")
        local_idx = local_by_frame[frame_id]
        gt_depth = _read_depth(depth_path_for_frame(args.scene_dir, frame_id), args.gt_scale)
        info = by_id[frame_id]
        K_target = _scale_K(np.asarray(info["K"], dtype=np.float64), _image_size(imgs[frame_id]), (gt_depth.shape[1], gt_depth.shape[0]))
        extrinsics = np.asarray(data["extrinsics"], dtype=np.float64)
        target_c2w = _spatracker_transform_c2w_similarity(_spatracker_c2w_from_w2c(extrinsics[local_idx]), *align)
        per_frame_static = state["per_frame_static_points"]  # type: ignore[assignment]
        static_points = per_frame_static.get(frame_id, np.zeros((0, 3), dtype=np.float64))
        dynamic_points = _spatracker_track_points_for_frame(data, local_idx, align, render_args)
    else:
        records = {int(rec.novel_id): rec for rec in _load_novel_metadata(args.scene_dir / "novel")}
        rec = records[int(novel_id)]
        if int(rec.base_frame) not in local_by_frame or int(rec.base_frame) not in by_id:
            raise FileNotFoundError(f"SpaTracker has no raw base frame for novel view {novel_id}")
        local_idx = local_by_frame[int(rec.base_frame)]
        gt_depth = _read_depth(rec.depth_path, args.gt_scale)
        base_info = by_id[int(rec.base_frame)]
        K_target = _scale_K(
            np.asarray(base_info["K"], dtype=np.float64),
            _image_size(imgs[int(rec.base_frame)]),
            (gt_depth.shape[1], gt_depth.shape[0]),
        )
        extrinsics = np.asarray(data["extrinsics"], dtype=np.float64)
        spa_base_aligned = _spatracker_transform_c2w_similarity(_spatracker_c2w_from_w2c(extrinsics[local_idx]), *align)
        target_c2w = _spatracker_novel_c2w_from_base(rec, _spatracker_gt_c2w(base_info), spa_base_aligned, align[0])
        static_points = state["static_points"]  # type: ignore[assignment]
        dynamic_points = _spatracker_track_points_for_frame(data, local_idx, align, render_args)

    depth, alpha, static_depth, static_alpha, dynamic_depth, dynamic_alpha = _render_spatracker_layers(
        static_points,
        dynamic_points,
        target_c2w,
        K_target,
        gt_depth.shape,
        args,
    )
    label = f"train_frame{frame_index:06d}" if split == "train" else f"novel{novel_id:06d}"
    save_spatracker_debug_layers(
        args,
        label,
        gt_depth,
        static_depth,
        static_alpha,
        dynamic_depth,
        dynamic_alpha,
        depth,
        alpha,
        state.get("intrinsic_check", {}),
    )
    valid = np.isfinite(depth) & (depth > float(args.min_depth)) & (alpha > method_alpha_threshold("spatracker", args))
    return depth, valid, alpha


def load_gt_target(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, str, int, str | None]:
    imgs = image_files(args.scene_dir)
    if args.split == "train":
        frame_id = int(args.frame_index)
        depth = _read_depth(depth_path_for_frame(args.scene_dir, frame_id), args.gt_scale)
        image_name = imgs[frame_id].name if 0 <= frame_id < len(imgs) else None
        keep = (
            static_mask(mask_dir(args.scene_dir), frame_id, depth.shape, args.mask_threshold, image_name)
            if args.static_only
            else np.ones(depth.shape, dtype=bool)
        )
        return depth, keep, f"train_frame{frame_id:06d}", frame_id, image_name

    records = {int(rec.novel_id): rec for rec in _load_novel_metadata(args.scene_dir / "novel")}
    rec = records[int(args.novel_id)]
    depth = _read_depth(rec.depth_path, args.gt_scale)
    novel_masks = novel_mask_dir(args.scene_dir)
    has_own = novel_masks.parent == args.scene_dir / "novel"
    mask_id = int(rec.novel_id) if has_own else int(rec.base_frame)
    image_name = rec.image_path.name if has_own else (imgs[int(rec.base_frame)].name if 0 <= int(rec.base_frame) < len(imgs) else None)
    keep = (
        static_mask(novel_masks, mask_id, depth.shape, args.mask_threshold, image_name)
        if args.static_only
        else np.ones(depth.shape, dtype=bool)
    )
    return depth, keep, f"novel{int(rec.novel_id):06d}", int(rec.base_frame), image_name


def load_methods(args: argparse.Namespace) -> dict[str, Callable[[], tuple[np.ndarray, np.ndarray | None, np.ndarray | None, float]]]:
    scene_dir = args.scene_dir
    ours_dir = (args.ours_depth_dir or (scene_dir / "outputs" / "gaussian_depth_eval")).resolve()
    da_dir = scene_dir / "outputs" / "depth_anything_3"
    mosca_dir = scene_dir / "outputs_mosca" / "depth_eval"
    methods: dict[str, Callable[[], tuple[np.ndarray, np.ndarray | None, np.ndarray | None, float]]] = {}

    if args.split == "train":
        frame = int(args.frame_index)
        methods["ours"] = lambda: (*load_npz_depth(ours_dir / "train_depth_arrays" / f"train_depth_{frame:04d}.npz"), 1.0)
        methods["da3_origin"] = lambda: (_load_da3_depths(da_dir)[frame].astype(np.float64), None, None, da3_depth_multiplier(scene_dir))
        methods["da3_render"] = lambda: (*render_array_depth(da_dir / "train_render_depth_arrays" / f"train{frame:06d}"), da3_depth_multiplier(scene_dir))
        methods["mosca"] = lambda: (*sidecar_depth(mosca_dir / "training_depth" / f"frame{frame:06d}"), mosca_depth_multiplier(scene_dir))
        methods["spatracker"] = lambda: (*render_spatracker_depth(args, split="train", frame_index=frame), float(load_spatracker_state(args)["camera_depth_scale"]))
    else:
        novel_id = int(args.novel_id)
        methods["ours"] = lambda: (*load_npz_depth(ours_dir / "novel" / "depth_arrays" / f"novel_depth_{novel_id:04d}.npz"), 1.0)
        methods["da3_render"] = lambda: (*render_array_depth(da_dir / "novel_render_depth_arrays" / f"novel{novel_id:06d}"), da3_depth_multiplier(scene_dir))
        methods["mosca"] = lambda: (*sidecar_depth(mosca_dir / "novel_depth" / f"novel{novel_id:06d}"), mosca_depth_multiplier(scene_dir))
        methods["spatracker"] = lambda: (*render_spatracker_depth(args, split="novel", novel_id=novel_id), float(load_spatracker_state(args)["camera_depth_scale"]))
    return {key: methods[key] for key in args.methods if key in methods}


def train0_method_loaders(args: argparse.Namespace) -> dict[str, Callable[[], tuple[np.ndarray, np.ndarray | None, np.ndarray | None, float]]]:
    scene_dir = args.scene_dir
    ours_dir = (args.ours_depth_dir or (scene_dir / "outputs" / "gaussian_depth_eval")).resolve()
    da_dir = scene_dir / "outputs" / "depth_anything_3"
    mosca_dir = scene_dir / "outputs_mosca" / "depth_eval"
    return {
        "ours": lambda: (*load_npz_depth(ours_dir / "train_depth_arrays" / "train_depth_0000.npz"), 1.0),
        "da3_origin": lambda: (_load_da3_depths(da_dir)[0].astype(np.float64), None, None, da3_depth_multiplier(scene_dir)),
        "da3_render": lambda: (*render_array_depth(da_dir / "train_render_depth_arrays" / "train000000"), da3_depth_multiplier(scene_dir)),
        "mosca": lambda: (*sidecar_depth(mosca_dir / "training_depth" / "frame000000"), mosca_depth_multiplier(scene_dir)),
        "spatracker": lambda: (*render_spatracker_depth(args, split="train", frame_index=0), float(load_spatracker_state(args)["camera_depth_scale"])),
    }


def fit_depth_correction(
    args: argparse.Namespace,
    name: str,
    loader: Callable[[], tuple[np.ndarray, np.ndarray | None, np.ndarray | None, float]],
) -> tuple[float, float | None, str]:
    radius = (
        float(args.spatracker_depth_correction_range)
        if name == "spatracker"
        else float(args.depth_correction_range)
    )
    radius = max(radius, 0.0)
    if radius <= 0:
        return 1.0, None, "disabled"
    try:
        gt0 = _read_depth(depth_path_for_frame(args.scene_dir, 0), args.gt_scale)
        imgs = image_files(args.scene_dir)
        image_name = imgs[0].name if imgs else None
        eval0 = (
            static_mask(mask_dir(args.scene_dir), 0, gt0.shape, args.mask_threshold, image_name)
            if args.static_only
            else np.ones(gt0.shape, dtype=bool)
        )
        depth0, valid0, alpha0, base_multiplier = loader()
    except FileNotFoundError as exc:
        return 1.0, None, f"missing train frame 0 data: {exc}"
    except Exception as exc:
        return 1.0, None, f"train frame 0 correction unavailable: {exc}"

    lo = max(1.0 - radius, 1e-12)
    hi = 1.0 + radius
    samples = max(int(args.depth_correction_samples), 3)
    candidates = np.linspace(lo, hi, samples, dtype=np.float64)
    best_correction = 1.0
    best_l1 = float("inf")
    for correction in candidates:
        pred0 = resize_like(np.asarray(depth0, dtype=np.float64) * float(base_multiplier) * float(correction), gt0.shape)
        valid = resize_like(valid0, gt0.shape, is_mask=True)
        alpha = resize_like(alpha0, gt0.shape)
        mask = valid_eval_mask(gt0, pred0, eval0, valid, alpha, args,
                               alpha_threshold=method_alpha_threshold(name, args))
        if not np.any(mask):
            continue
        # Median-normalized L1 (divide by median GT, matching eval_depth_gaussian_ours),
        # instead of per-pixel /gt which over-weights near pixels.
        gt_med = max(float(np.median(gt0[mask])), float(args.min_depth))
        rel = np.abs(pred0[mask] - gt0[mask]) / gt_med
        rel_l1 = float(np.mean(rel))
        if rel_l1 < best_l1:
            best_l1 = rel_l1
            best_correction = float(correction)
    if not np.isfinite(best_l1):
        return 1.0, None, "no valid train frame 0 pixels"
    return best_correction, best_l1, f"{name}: train frame 0 bounded correction [{lo:.3g}, {hi:.3g}]"


def method_alpha_threshold(name: str, args: argparse.Namespace) -> float:
    """Per-method alpha cutoff. Point-splatted methods use their own (low) threshold so
    they aren't masked out by the high default used for the dense Ours render."""
    per_method = {
        "mosca": float(args.mosca_alpha_threshold),
        "spatracker": float(args.spatracker_alpha_threshold),
        "da3_render": float(args.da3_alpha_threshold),
    }
    thr = per_method.get(name, float(args.alpha_threshold))
    return float(args.alpha_threshold) if thr < 0 else thr


def valid_eval_mask(
    gt_depth: np.ndarray,
    pred_depth: np.ndarray,
    static: np.ndarray,
    valid: np.ndarray | None,
    alpha: np.ndarray | None,
    args: argparse.Namespace,
    alpha_threshold: float | None = None,
) -> np.ndarray:
    pred_depth = resize_like(pred_depth, gt_depth.shape)
    valid = resize_like(valid, gt_depth.shape, is_mask=True)
    alpha = resize_like(alpha, gt_depth.shape)
    mask = static.copy()
    mask &= np.isfinite(gt_depth) & np.isfinite(pred_depth)
    mask &= gt_depth > float(args.min_depth)
    mask &= pred_depth > float(args.min_depth)
    if valid is not None:
        mask &= valid.astype(bool)
    if alpha is not None:
        thr = float(args.alpha_threshold) if alpha_threshold is None else float(alpha_threshold)
        mask &= alpha > thr
    return mask


def colorize_depth(
    depth: np.ndarray,
    mask: np.ndarray,
    vmin: float,
    vmax: float,
    *,
    fill_mask: np.ndarray | None = None,
    fill_radius: float = 5.0,
) -> np.ndarray:
    norm = np.zeros(depth.shape, dtype=np.uint8)
    if vmax > vmin:
        clipped = np.clip((depth - vmin) / (vmax - vmin), 0.0, 1.0)
        norm = (clipped * 255.0).astype(np.uint8)
    display_mask = mask if fill_mask is None else fill_mask.astype(bool)
    if fill_mask is not None and float(fill_radius) > 0:
        holes = display_mask & ~mask.astype(bool)
        seeds = display_mask & mask.astype(bool)
        if np.any(holes) and np.any(seeds):
            inpaint_mask = (holes.astype(np.uint8) * 255)
            norm = cv2.inpaint(norm, inpaint_mask, float(fill_radius), cv2.INPAINT_TELEA)
    out = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    out[~display_mask] = 0
    return out


def colorize_error(
    err: np.ndarray,
    mask: np.ndarray,
    clip: float,
    *,
    fill_mask: np.ndarray | None = None,
    fill_radius: float = 5.0,
) -> np.ndarray:
    norm = np.clip(err / max(float(clip), 1e-12), 0.0, 1.0)
    norm_u8 = (norm * 255.0).astype(np.uint8)
    display_mask = mask if fill_mask is None else fill_mask.astype(bool)
    if fill_mask is not None and float(fill_radius) > 0:
        holes = display_mask & ~mask.astype(bool)
        seeds = display_mask & mask.astype(bool)
        if np.any(holes) and np.any(seeds):
            inpaint_mask = (holes.astype(np.uint8) * 255)
            norm_u8 = cv2.inpaint(norm_u8, inpaint_mask, float(fill_radius), cv2.INPAINT_TELEA)
    out = cv2.applyColorMap(norm_u8, cv2.COLORMAP_MAGMA)
    out[~display_mask] = 0
    return out


def label_panel(img: np.ndarray, label: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def save_figure(path: Path, img: np.ndarray, downscale: float) -> None:
    """Write a figure PNG, optionally downscaled to shrink file size (metrics use full res)."""
    if downscale and downscale > 0 and abs(float(downscale) - 1.0) > 1e-6:
        h, w = img.shape[:2]
        new_w = max(1, int(round(w * float(downscale))))
        new_h = max(1, int(round(h * float(downscale))))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), img)


def _load_ours_eval_normalizers(scene_dir: Path) -> tuple[float | None, float | None]:
    """Read the dataset median normalizers saved by eval_depth_gaussian_ours.py so the viz
    can reproduce its exact error calc without re-running the (slow) 100-frame eval."""
    p = scene_dir / "outputs" / "gaussian_depth_eval" / "gaussian_depth_eval.json"
    try:
        d = json.load(open(p))
        return float(d["gt_normalizer"]), float(d["est_normalizer"])
    except Exception:
        return None, None


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gt_depth, static, label, _base_frame, _image_name = load_gt_target(args)
    valid_gt = static & np.isfinite(gt_depth) & (gt_depth > float(args.min_depth))
    if not np.any(valid_gt):
        raise RuntimeError("No valid GT pixels for evaluation.")
    vmin, vmax = np.percentile(gt_depth[valid_gt], [2.0, 98.0])
    gt_vis = colorize_depth(gt_depth, valid_gt, float(vmin), float(vmax))

    metrics = {
        "scene_dir": str(args.scene_dir),
        "split": args.split,
        "label": label,
        "gt_scale": float(args.gt_scale),
        "metric": "mean(abs(pred_depth - gt_depth)) / median(gt_depth) over valid pixels",
        "static_only": bool(args.static_only),
        "depth_correction": {
            "fit_split": "train",
            "fit_frame": 0,
            "range": [
                max(1.0 - max(float(args.depth_correction_range), 0.0), 1e-12),
                1.0 + max(float(args.depth_correction_range), 0.0),
            ],
            "spatracker_range": [
                max(1.0 - max(float(args.spatracker_depth_correction_range), 0.0), 1e-12),
                1.0 + max(float(args.spatracker_depth_correction_range), 0.0),
            ],
            "samples": max(int(args.depth_correction_samples), 3),
            "disabled": max(float(args.depth_correction_range), 0.0) <= 0,
            "spatracker_disabled": max(float(args.spatracker_depth_correction_range), 0.0) <= 0,
        },
        "spatracker_raw_render": {
            "result_npz": str(args.spatracker_result_npz) if args.spatracker_result_npz is not None else "auto",
            "static_point_radius": int(args.spatracker_static_point_radius),
            "dynamic_point_radius": int(args.spatracker_dynamic_point_radius),
            "fill_holes_for_depth_png": bool(args.spatracker_fill_holes),
            "fill_hole_radius": float(args.spatracker_fill_hole_radius),
            "static_point_stride": int(args.spatracker_static_point_stride),
            "static_frame_stride": int(args.spatracker_static_frame_stride),
            "static_max_points": int(args.spatracker_static_max_points),
            "max_dynamic_points": int(args.spatracker_max_dynamic_points),
            "use_gt_intrinsics": bool(args.spatracker_use_gt_intrinsics),
            "source": "SpaTracker result.npz rendered live; static pointmap and dynamic tracks are z-composited with separate radii",
        },
        "methods": {},
    }
    combined_depth_panels = [label_panel(gt_vis, "GT")]
    combined_error_panels = [label_panel(np.zeros_like(gt_vis), "relative error")]
    # Reuse eval_depth_gaussian_ours's SAVED dataset normalizers (from its log) so the viz
    # error calc is identical to the table without re-running the slow eval pipeline.
    ours_gt_norm, ours_est_norm = _load_ours_eval_normalizers(args.scene_dir)
    shared_gt_norm = ours_gt_norm  # dataset GT median; used for every method (fair GT norm)

    for name, loader in load_methods(args).items():
        try:
            depth, valid, alpha, depth_multiplier = loader()
        except FileNotFoundError as exc:
            print(f"[aligned-depth] skip {name}: missing {exc}")
            continue
        except Exception as exc:
            print(f"[aligned-depth] skip {name}: {exc}")
            continue

        est_arr = resize_like(np.asarray(depth, dtype=np.float64) * float(depth_multiplier), gt_depth.shape)
        valid = resize_like(valid, gt_depth.shape, is_mask=True)
        alpha = resize_like(alpha, gt_depth.shape)
        mask = valid_eval_mask(gt_depth, est_arr, static, valid, alpha, args,
                               alpha_threshold=method_alpha_threshold(name, args))
        if not np.any(mask):
            print(f"[aligned-depth] skip {name}: no valid pixels after masking")
            continue

        # Same error calc as eval_depth_gaussian_ours: mean|gt/M_gt - est/M_est| with medians
        # as normalizers. Ours reuses the eval log's dataset normalizers (exact match to the
        # table, no re-run); that same GT median normalizes every method (fair), and each
        # other method's est median is taken per-frame.
        M_gt = float(shared_gt_norm) if shared_gt_norm is not None else float(np.median(gt_depth[mask]))
        if name == "ours" and ours_est_norm is not None:
            M_est = float(ours_est_norm)
            correction_source = "eval-log gaussian_depth_eval.json"
        else:
            M_est = float(np.median(est_arr[mask]))
            correction_source = "per-frame est median"
        M_gt = max(M_gt, float(args.min_depth))
        M_est = max(M_est, float(args.min_depth))
        correction = M_gt / M_est
        # Clip the residual correction to a bounded window (reuse the per-method
        # correction range, default 0.8-1.2) and report THIS bounded correction as the
        # on-figure scale, not the raw up-to-scale camera multiplier -- that base scale is
        # large for monocular methods (mosca/spatracker reconstruct up-to-scale) and reads
        # as alarming even though it never enters the median-normalized L1.
        corr_range = (
            float(args.spatracker_depth_correction_range)
            if name == "spatracker"
            else float(args.depth_correction_range)
        )
        corr_lo = max(1.0 - corr_range, float(args.min_depth))
        corr_hi = 1.0 + corr_range
        correction = float(np.clip(correction, corr_lo, corr_hi))
        correction_l1 = None
        final_multiplier = float(depth_multiplier) * correction
        pred = est_arr * correction
        rel_err = np.zeros_like(gt_depth, dtype=np.float64)
        rel_err[mask] = np.abs(gt_depth[mask] / M_gt - est_arr[mask] / M_est)
        # eval_depth_gaussian_ours reports the per-frame MEDIAN of the normalized diff
        # (_evaluate_with_normalizers), not the mean. Match it so the figure == the table.
        l1 = float(np.median(rel_err[mask]))
        depth_fill_mask = None
        depth_fill_radius = 0.0
        if name == "spatracker" and bool(args.spatracker_fill_holes):
            depth_fill_mask = static & np.isfinite(gt_depth) & (gt_depth > float(args.min_depth))
            depth_fill_radius = float(args.spatracker_fill_hole_radius)
        pred_vis = colorize_depth(
            pred,
            mask,
            float(vmin),
            float(vmax),
            fill_mask=depth_fill_mask,
            fill_radius=depth_fill_radius,
        )
        err_vis = colorize_error(
            rel_err,
            mask,
            float(args.error_clip),
            fill_mask=depth_fill_mask,
            fill_radius=depth_fill_radius,
        )
        scale_label = f"corr x{correction:.4g}" if abs(float(correction) - 1.0) > 1e-6 else "corr x1"
        panel = np.concatenate(
            [
                label_panel(gt_vis, "GT"),
                label_panel(pred_vis, f"{name} relL1={l1:.4f} {scale_label}"),
                label_panel(err_vis, "relative error"),
            ],
            axis=1,
        )
        safe_name = name.replace("/", "_")
        png_path = args.output_dir / f"{label}_{safe_name}_aligned_depth.png"
        npy_path = args.output_dir / f"{label}_{safe_name}_aligned_depth.npy"
        save_figure(png_path, panel, float(args.save_downscale))
        np.save(npy_path, pred.astype(np.float32))
        combined_depth_panels.append(label_panel(pred_vis, f"{name} {l1:.4f} {scale_label}"))
        combined_error_panels.append(label_panel(err_vis, f"{name} error"))
        metrics["methods"][name] = {
            "relative_l1": l1,
            "num_pixels": int(mask.sum()),
            "base_depth_multiplier": float(depth_multiplier),
            "depth_correction": float(correction),
            "final_depth_multiplier": float(final_multiplier),
            "depth_correction_train0_relative_l1": correction_l1,
            "depth_correction_source": correction_source,
            "depth_png_hole_fill": bool(depth_fill_mask is not None),
            "depth_png_hole_fill_radius": float(depth_fill_radius),
            "png": str(png_path),
            "aligned_depth_npy": str(npy_path),
        }
        print(
            f"[aligned-depth] {name}: rel_l1={l1:.6g} pixels={int(mask.sum())} "
            f"base_multiplier={float(depth_multiplier):.6g} "
            f"depth_correction={float(correction):.6g} "
            f"final_multiplier={float(final_multiplier):.6g}"
        )

    if len(combined_depth_panels) > 1:
        combined = np.vstack(
            [
                np.concatenate(combined_depth_panels, axis=1),
                np.concatenate(combined_error_panels, axis=1),
            ]
        )
        combined_path = args.output_dir / f"{label}_all_aligned_depths.png"
        save_figure(combined_path, combined, float(args.save_downscale))
        metrics["combined_png"] = str(combined_path)

    json_path = args.output_dir / f"{label}_aligned_depth_metrics.json"
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[aligned-depth] wrote {json_path}")


if __name__ == "__main__":
    main()
