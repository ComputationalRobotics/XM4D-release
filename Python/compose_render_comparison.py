#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Compose ours-vs-MoSca RGB render comparison PNG for one frame.")
    parser.add_argument("--ours-npz", type=Path, required=True)
    parser.add_argument("--mosca-npz", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--frame-index", type=int, default=None)
    parser.add_argument("--error-clip", type=float, default=0.25)
    parser.add_argument(
        "--hole-fill",
        choices=["gt", "raw", "black", "white"],
        default="gt",
        help="Display fill for render pixels outside the alpha-valid mask.",
    )
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def resize_like(arr: np.ndarray, shape: tuple[int, int], *, is_mask: bool = False) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.shape[:2] == shape:
        return arr
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.resize(arr.astype(np.float32), (shape[1], shape[0]), interpolation=interp)


def to_rgb(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    return np.clip(arr[..., :3], 0.0, 1.0)


def label_panel(img: np.ndarray, label: str) -> np.ndarray:
    out = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.putText(out, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def colorize_error(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, clip: float) -> tuple[np.ndarray, float, int]:
    valid = np.asarray(valid, dtype=bool)
    valid &= np.isfinite(pred).all(axis=-1) & np.isfinite(gt).all(axis=-1)
    err = np.zeros(pred.shape[:2], dtype=np.float32)
    if np.any(valid):
        err[valid] = np.mean(np.abs(pred[valid] - gt[valid]), axis=-1)
    norm = np.clip(err / max(float(clip), 1e-12), 0.0, 1.0)
    out = cv2.applyColorMap((norm * 255.0).astype(np.uint8), cv2.COLORMAP_MAGMA)
    out[~valid] = 0
    mean_err = float(np.mean(err[valid])) if np.any(valid) else float("nan")
    return out, mean_err, int(np.count_nonzero(valid))


def fill_render_holes(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, mode: str) -> np.ndarray:
    out = np.asarray(pred, dtype=np.float32).copy()
    invalid = ~np.asarray(valid, dtype=bool)
    if not np.any(invalid) or mode == "raw":
        return out
    if mode == "gt":
        out[invalid] = gt[invalid]
    elif mode == "white":
        out[invalid] = 1.0
    elif mode == "black":
        out[invalid] = 0.0
    else:
        raise ValueError(f"Unsupported hole-fill mode: {mode}")
    return out


def main() -> None:
    args = parse_args()
    ours = load_npz(args.ours_npz)
    mosca = load_npz(args.mosca_npz)

    gt = to_rgb(ours["gt"])
    shape = gt.shape[:2]
    ours_pred = to_rgb(resize_like(ours["pred"], shape))
    mosca_pred = to_rgb(resize_like(mosca["pred"], shape))
    ours_valid = resize_like(ours["valid"], shape, is_mask=True).astype(bool)
    mosca_valid = resize_like(mosca["valid"], shape, is_mask=True).astype(bool)

    ours_show = fill_render_holes(ours_pred, gt, ours_valid, args.hole_fill)
    mosca_show = fill_render_holes(mosca_pred, gt, mosca_valid, args.hole_fill)

    ours_err, ours_mean_err, ours_px = colorize_error(ours_pred, gt, ours_valid, float(args.error_clip))
    mosca_err, mosca_mean_err, mosca_px = colorize_error(mosca_pred, gt, mosca_valid, float(args.error_clip))
    blank = np.zeros_like(gt)

    frame_index = args.frame_index
    if frame_index is None and "frame_index" in ours:
        frame_index = int(np.asarray(ours["frame_index"]).item())
    split = str(np.asarray(ours.get("split", np.array("train"))).item())
    novel_id = int(np.asarray(ours.get("novel_id", np.array(-1))).item())
    if split == "novel" and novel_id >= 0:
        gt_label = f"GT novel {novel_id:06d}"
    else:
        gt_label = "GT" if frame_index is None else f"GT frame {frame_index:06d}"

    top = np.concatenate(
        [
            label_panel(gt, gt_label),
            label_panel(ours_show, "Ours render"),
            label_panel(mosca_show, "MoSca render"),
        ],
        axis=1,
    )
    bottom = np.concatenate(
        [
            label_panel(blank, "mean abs RGB error"),
            label_panel(cv2.cvtColor(ours_err, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0, f"Ours error {ours_mean_err:.4f}"),
            label_panel(cv2.cvtColor(mosca_err, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0, f"MoSca error {mosca_mean_err:.4f}"),
        ],
        axis=1,
    )
    canvas = np.concatenate([top, bottom], axis=0)

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output_png), canvas)
    print(f"[render-compare-viz] wrote {args.output_png}")

    summary = {
        "output_png": str(args.output_png),
        "ours_npz": str(args.ours_npz),
        "mosca_npz": str(args.mosca_npz),
        "frame_index": frame_index,
        "split": split,
        "novel_id": novel_id,
        "error_metric": "mean absolute RGB error over valid alpha pixels",
        "error_clip": float(args.error_clip),
        "render_hole_fill": args.hole_fill,
        "ours_mean_abs_rgb_error": ours_mean_err,
        "mosca_mean_abs_rgb_error": mosca_mean_err,
        "ours_valid_pixels": ours_px,
        "mosca_valid_pixels": mosca_px,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
