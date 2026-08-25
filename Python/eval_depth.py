#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT_DIR = Path(__file__).parent.resolve()
SINTEL_DPT_TAG = 202021.25

SCENE_DIR = ROOT_DIR / "../datasets/TUM/tum1"
GT_DEPTH_DIR = SCENE_DIR / "depths"
EST_DEPTH_DIR = SCENE_DIR / "outputs" / "depth_anything_3"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate globally normalized depth L1 against GT.")
    parser.add_argument("--scene-dir", type=Path, default=SCENE_DIR)
    parser.add_argument("--gt-depth-dir", type=Path, default=None)
    parser.add_argument("--est-depth-dir", type=Path, default=EST_DEPTH_DIR)
    parser.add_argument("--gt-scale", type=float, default=1000.0, help="Divide GT png depth values by this scale")
    parser.add_argument("--min-depth", type=float, default=1e-6)
    parser.add_argument("--max-depth", type=float, default=-1.0, help="Optional max depth; <=0 disables")
    parser.add_argument(
        "--normalize-stat",
        "--scale-align",
        dest="normalize_stat",
        choices=["median", "mean"],
        default="median",
        help="Normalize GT and estimate independently so this statistic is 1 before L1.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--vis-out", type=Path, default=None)
    parser.add_argument("--vis-frame", type=int, default=-1, help="Matched frame index for visualization; -1 uses middle")
    parser.add_argument("--depth-array-dir", type=Path, default=None, help="Default: <est-depth-dir>/depth_arrays")
    parser.add_argument("--gt-normalizer", type=float, default=None, help="Use this GT depth normalizer instead of fitting one from this split.")
    parser.add_argument("--est-normalizer", type=float, default=None, help="Use this estimated depth normalizer instead of fitting one from this split.")
    parser.add_argument("--normalizer-scope", type=str, default="dataset", help="Label recorded in JSON for the normalizer source.")
    parser.add_argument("--json", action="store_true", help="Print RESULT_JSON line for scripts.")
    return parser.parse_args()


def _read_depth(path: Path, value_scale: float = 1.0) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path).astype(np.float64)
    if suffix == ".npz":
        data = np.load(path)
        for key in ("depth", "depths", "depth_map", "depth_maps", "pred_depth", "arr_0"):
            if key in data:
                return data[key].astype(np.float64)
        raise KeyError(f"No depth key found in {path.name}")
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Could not read depth image: {path}")
        depth = depth.astype(np.float64)
        if depth.ndim == 3:
            depth = depth[..., 0]
        if value_scale > 0:
            depth = depth / float(value_scale)
        return depth
    if suffix == ".dpt":
        with path.open("rb") as f:
            tag = np.fromfile(f, dtype=np.float32, count=1)[0]
            if tag != SINTEL_DPT_TAG:
                raise ValueError(f"Wrong Sintel .dpt tag in {path}: expected {SINTEL_DPT_TAG}, got {tag}")
            width = int(np.fromfile(f, dtype=np.int32, count=1)[0])
            height = int(np.fromfile(f, dtype=np.int32, count=1)[0])
            size = width * height
            if width <= 0 or height <= 0 or size <= 1 or size >= 100000000:
                raise ValueError(f"Invalid Sintel .dpt size in {path}: width={width}, height={height}")
            depth = np.fromfile(f, dtype=np.float32, count=size)
        if depth.size != size:
            raise ValueError(f"Truncated Sintel .dpt file {path}: expected {size} values, got {depth.size}")
        return depth.reshape((height, width)).astype(np.float64)
    raise ValueError(f"Unsupported depth file: {path}")


def _prefer_npy_depth_path(path: Path) -> Path:
    if path.suffix.lower() == ".npy":
        return path
    npy_path = path.with_suffix(".npy")
    if npy_path.is_file():
        return npy_path
    return path


def _depth_files(depth_dir: Path) -> list[Path]:
    exts = {".npy", ".npz", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dpt"}
    skip_names = {"depth_maps.npy", "conf_maps.npy", "intrinsics.npy", "extrinsics_c2w.npy"}
    by_stem: dict[str, Path] = {}
    suffix_priority = {".npy": 0, ".npz": 1, ".dpt": 2, ".png": 3, ".jpg": 4, ".jpeg": 4, ".tif": 5, ".tiff": 5}
    for path in depth_dir.iterdir():
        suffix = path.suffix.lower()
        if not path.is_file() or suffix not in exts or path.name in skip_names:
            continue
        old = by_stem.get(path.stem)
        if old is None or suffix_priority[suffix] < suffix_priority[old.suffix.lower()]:
            by_stem[path.stem] = path
    return sorted(by_stem.values())


def _numeric_key(path: Path) -> int | None:
    match = re.search(r"(\d+)$", path.stem)
    if match is None:
        return None
    return int(match.group(1))


def _load_stacked_estimates(est_dir: Path) -> np.ndarray | None:
    for name in ("depth_maps.npy", "depths.npy"):
        path = est_dir / name
        if path.is_file():
            arr = np.load(path).astype(np.float64)
            if arr.ndim != 3:
                raise ValueError(f"Expected stacked depth array (N,H,W), got {arr.shape} from {path}")
            return arr
    return None


def _build_pairs(
    scene_dir: Path,
    gt_dir: Path,
    est_dir: Path,
    gt_scale: float,
) -> list[tuple[str, np.ndarray, np.ndarray, Path | None, Path | None]]:
    gt_files = _depth_files(gt_dir)
    if not gt_files:
        raise FileNotFoundError(f"No GT depth files found in {gt_dir}")

    stacked_est = _load_stacked_estimates(est_dir)
    if stacked_est is not None:
        image_files = sorted(
            path
            for path in (scene_dir / "images").iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if len(image_files) != stacked_est.shape[0]:
            raise ValueError(
                f"Stacked estimates have {stacked_est.shape[0]} frames but "
                f"{scene_dir / 'images'} has {len(image_files)} images"
            )
        gt_by_stem = {path.stem: path for path in gt_files}
        pairs = []
        for i, image_path in enumerate(image_files):
            gt_path = gt_by_stem.get(image_path.stem)
            if gt_path is None and i < len(gt_files):
                gt_path = gt_files[i]
            if gt_path is None:
                continue
            pairs.append((image_path.name, _read_depth(gt_path, gt_scale), stacked_est[i], gt_path, None))
        return pairs

    est_files = _depth_files(est_dir)
    if not est_files:
        raise FileNotFoundError(f"No estimated depth files found in {est_dir}")

    gt_by_stem = {path.stem: path for path in gt_files}
    est_by_stem = {path.stem: path for path in est_files}
    common_stems = sorted(set(gt_by_stem) & set(est_by_stem))
    if common_stems:
        return [
            (stem, _read_depth(gt_by_stem[stem], gt_scale), _read_depth(est_by_stem[stem]), gt_by_stem[stem], est_by_stem[stem])
            for stem in common_stems
        ]

    gt_by_num = {key: path for path in gt_files if (key := _numeric_key(path)) is not None}
    est_by_num = {key: path for path in est_files if (key := _numeric_key(path)) is not None}
    common_nums = sorted(set(gt_by_num) & set(est_by_num))
    return [
        (str(key), _read_depth(gt_by_num[key], gt_scale), _read_depth(est_by_num[key]), gt_by_num[key], est_by_num[key])
        for key in common_nums
    ]


def _valid_mask(gt: np.ndarray, est: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    if gt.shape != est.shape:
        raise ValueError(f"Shape mismatch: GT {gt.shape} vs EST {est.shape}")
    valid = np.isfinite(gt) & np.isfinite(est)
    valid &= gt > float(min_depth)
    valid &= est > float(min_depth)
    if float(max_depth) > 0:
        valid &= gt <= float(max_depth)
        valid &= est <= float(max_depth)
    return valid


def _normalizer(values: np.ndarray, method: str) -> float:
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return 1.0
    if method == "mean":
        scale = float(np.mean(values))
    else:
        scale = float(np.median(values))
    if not np.isfinite(scale) or scale <= 1e-12:
        return 1.0
    return scale


def _evaluate(
    pairs: list[tuple[str, np.ndarray, np.ndarray, Path | None, Path | None]],
    min_depth: float,
    max_depth: float,
    normalize_stat: str,
    gt_normalizer_override: float | None = None,
    est_normalizer_override: float | None = None,
    normalizer_scope: str = "dataset",
) -> dict[str, Any]:
    gt_chunks = []
    est_chunks = []
    masks = []
    for _, gt, est, _, _ in pairs:
        valid = _valid_mask(gt, est, min_depth, max_depth)
        masks.append(valid)
        if np.any(valid):
            gt_chunks.append(gt[valid])
            est_chunks.append(est[valid])

    if not gt_chunks:
        raise ValueError("No valid depth pixels found.")

    gt_all = np.concatenate(gt_chunks)
    est_all = np.concatenate(est_chunks)
    gt_normalizer = float(gt_normalizer_override) if gt_normalizer_override is not None else _normalizer(gt_all, normalize_stat)
    est_normalizer = float(est_normalizer_override) if est_normalizer_override is not None else _normalizer(est_all, normalize_stat)
    if not np.isfinite(gt_normalizer) or gt_normalizer <= 1e-12:
        raise ValueError(f"Invalid GT normalizer: {gt_normalizer}")
    if not np.isfinite(est_normalizer) or est_normalizer <= 1e-12:
        raise ValueError(f"Invalid estimated-depth normalizer: {est_normalizer}")

    total_abs = 0.0
    total_valid = 0
    frames_used = 0

    for (_, gt, est, _, _), valid in zip(pairs, masks, strict=True):
        if not np.any(valid):
            continue
        gt_norm = gt[valid] / gt_normalizer
        est_norm = est[valid] / est_normalizer
        diff = np.abs(est_norm - gt_norm)
        total_abs += float(np.sum(diff))
        total_valid += int(diff.size)
        frames_used += 1

    return {
        "num_frames": len(pairs),
        "num_frames_used": frames_used,
        "num_valid_pixels": total_valid,
        "normalize_stat": normalize_stat,
        "normalizer_scope": normalizer_scope,
        "gt_normalizer": float(gt_normalizer),
        "est_normalizer": float(est_normalizer),
        "l1": total_abs / max(total_valid, 1),
    }


def _normalize_for_vis(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return np.zeros_like(depth, dtype=np.float32)
    vals = depth[mask]
    vmin = float(np.percentile(vals, 2.0))
    vmax = float(np.percentile(vals, 98.0))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return np.clip((depth - vmin) / (vmax - vmin), 0.0, 1.0).astype(np.float32)


def _colorize(gray: np.ndarray) -> np.ndarray:
    img = (np.clip(gray, 0.0, 1.0) * 255).astype(np.uint8)
    return cv2.applyColorMap(img, cv2.COLORMAP_TURBO)


def _write_depth_vis(
    pair: tuple[str, np.ndarray, np.ndarray, Path | None, Path | None],
    gt_normalizer: float,
    est_normalizer: float,
    min_depth: float,
    max_depth: float,
    vis_out: Path,
) -> None:
    name, gt, est, _, _ = pair
    valid = _valid_mask(gt, est, min_depth, max_depth)
    gt_scaled = gt / max(float(gt_normalizer), 1e-12)
    est_scaled = est / max(float(est_normalizer), 1e-12)
    err = np.zeros_like(gt, dtype=np.float64)
    err[valid] = np.abs(est_scaled[valid] - gt_scaled[valid])

    gt_img = _colorize(_normalize_for_vis(gt_scaled, valid))
    est_img = _colorize(_normalize_for_vis(est_scaled, valid))
    err_img = _colorize(_normalize_for_vis(err, valid))
    label_h = 28
    labels = np.zeros((label_h, gt_img.shape[1] * 3, 3), dtype=np.uint8)
    for i, text in enumerate(("GT normalized", "Estimated normalized", f"|error| {name}")):
        cv2.putText(
            labels,
            text,
            (i * gt_img.shape[1] + 10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    vis = np.concatenate([gt_img, est_img, err_img], axis=1)
    vis = np.concatenate([labels, vis], axis=0)
    vis_out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(vis_out), vis)


def _save_depth_arrays(
    out_dir: Path,
    pair: tuple[str, np.ndarray, np.ndarray, Path | None, Path | None],
    valid: np.ndarray,
    gt_normalizer: float,
    est_normalizer: float,
    frame_index: int,
) -> None:
    name, gt, est, gt_path, est_path = pair
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"frame{frame_index:06d}.npz",
        gt_depth=gt.astype(np.float32),
        rendered_depth=est.astype(np.float32),
        pred_depth=est.astype(np.float32),
        valid_mask=valid.astype(bool),
        alpha=np.ones_like(gt, dtype=np.float32),
        gt_normalizer=np.array(float(gt_normalizer), dtype=np.float32),
        est_normalizer=np.array(float(est_normalizer), dtype=np.float32),
        meta=np.array(
            {
                "frame_index": int(frame_index),
                "name": name,
                "gt_depth": None if gt_path is None else str(gt_path),
                "est_depth": None if est_path is None else str(est_path),
            },
            dtype=object,
        ),
    )


def _default_gt_dir(scene_dir: Path) -> Path:
    return scene_dir / "depths"


def _default_output_json(est_dir: Path) -> Path:
    return est_dir / "depth_eval.json"


def main() -> None:
    args = _parse_args()
    scene_dir = args.scene_dir.resolve()
    gt_dir = (args.gt_depth_dir if args.gt_depth_dir is not None else _default_gt_dir(scene_dir)).resolve()
    est_dir = args.est_depth_dir.resolve()

    if not gt_dir.is_dir():
        raise FileNotFoundError(f"Missing GT depth dir: {gt_dir}")
    if not est_dir.is_dir():
        raise FileNotFoundError(f"Missing estimated depth dir: {est_dir}")

    pairs = _build_pairs(scene_dir, gt_dir, est_dir, gt_scale=float(args.gt_scale))
    if not pairs:
        raise FileNotFoundError(f"No matching depth frames in {gt_dir} and {est_dir}")

    summary = _evaluate(
        pairs,
        min_depth=float(args.min_depth),
        max_depth=float(args.max_depth),
        normalize_stat=args.normalize_stat,
        gt_normalizer_override=args.gt_normalizer,
        est_normalizer_override=args.est_normalizer,
        normalizer_scope=args.normalizer_scope,
    )

    result: dict[str, Any] = {
        "scene_dir": str(scene_dir),
        "gt_depth_dir": str(gt_dir),
        "est_depth_dir": str(est_dir),
        "gt_scale": float(args.gt_scale),
        "min_depth": float(args.min_depth),
        "max_depth": float(args.max_depth),
        **summary,
    }
    depth_array_dir = (args.depth_array_dir.resolve() if args.depth_array_dir is not None else est_dir / "depth_arrays")
    for frame_index, pair in enumerate(pairs):
        _, gt, est, _, _ = pair
        valid = _valid_mask(gt, est, float(args.min_depth), float(args.max_depth))
        _save_depth_arrays(
            depth_array_dir,
            pair,
            valid,
            float(summary["gt_normalizer"]),
            float(summary["est_normalizer"]),
            frame_index,
        )
    result["depth_arrays_dir"] = str(depth_array_dir.resolve())

    output_json = args.output_json.resolve() if args.output_json is not None else _default_output_json(est_dir)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    vis_out = args.vis_out.resolve() if args.vis_out is not None else est_dir / "depth_eval_vis.png"
    vis_index = len(pairs) // 2 if args.vis_frame < 0 else int(args.vis_frame)
    vis_index = max(0, min(vis_index, len(pairs) - 1))
    _write_depth_vis(
        pairs[vis_index],
        gt_normalizer=float(summary["gt_normalizer"]),
        est_normalizer=float(summary["est_normalizer"]),
        min_depth=float(args.min_depth),
        max_depth=float(args.max_depth),
        vis_out=vis_out,
    )
    result["vis_out"] = str(vis_out)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json:
        print("RESULT_JSON:", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
