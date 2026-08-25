#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from eval_depth_gaussian_ours import (
    AppearanceOptModule,
    _alignment_est_to_gt,
    _build_app_module,
    _dataset_frame_count,
    _gt_key_for_frame,
    _image_files,
    _load_camera_and_gt_intrinsics,
    _load_gaussians,
    _load_gt_intrinsics,
    _load_novel_metadata,
    _render_depth,
    _resolve_gaussian_paths,
    _transform_gt_c2w_to_est,
)
from utils import definitions


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render saved ours/gsplat Gaussians on training and novel RGB views, "
            "then report mPSNR/mSSIM/mLPIPS while skipping alpha holes."
        )
    )
    parser.add_argument("--scene-dir", "--scene-path", type=Path, default=Path(definitions.rootpath))
    parser.add_argument("--gaussians", type=Path, default=None, help="Final .pt file or checkpoint directory.")
    parser.add_argument("--camera-source", choices=["colmap", "xm"], default="colmap")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-views", type=int, default=0, help="<=0 evaluates all selected views.")
    parser.add_argument("--alpha-threshold", "--alpha_threshold", dest="alpha_threshold", type=float, default=0.8)
    parser.add_argument("--near-plane", type=float, default=0.01)
    parser.add_argument("--far-plane", type=float, default=1e10)
    parser.add_argument("--sh-degree", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-edg-warp", action="store_true")
    parser.add_argument("--eval-novel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--novel-dir", type=Path, default=None, help="Default: <scene>/novel")
    parser.add_argument("--save-images", action="store_true", help="Save every per-view GT/render/error preview.")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read RGB image: {path}")
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    elif img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)
    rgb = img.astype(np.float32)
    if rgb.max() > 1.0:
        rgb /= 255.0
    return np.clip(rgb, 0.0, 1.0)


def _find_image_path(scene_dir: Path, image_name: str) -> Path:
    image_dir = scene_dir / "images"
    candidates = [image_dir / image_name, image_dir / Path(image_name).name]
    by_name = {path.name: path for path in _image_files(scene_dir)}
    if Path(image_name).name in by_name:
        candidates.append(by_name[Path(image_name).name])
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Could not find image {image_name} under {image_dir}")


def _scale_K_for_size(K: np.ndarray, src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> np.ndarray:
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    K_scaled = np.asarray(K, dtype=np.float32).copy()
    if src_w > 0 and src_h > 0 and (src_h != dst_h or src_w != dst_w):
        K_scaled[0, :] *= float(dst_w) / float(src_w)
        K_scaled[1, :] *= float(dst_h) / float(src_h)
    return K_scaled


def _psnr_masked(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float | None:
    if not np.any(valid):
        return None
    diff = pred[valid] - gt[valid]
    mse = float(np.mean(diff * diff))
    if mse <= 1e-12:
        return float("inf")
    return float(-10.0 * np.log10(mse))


def _make_metric_tensors(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    pred_filled = np.asarray(pred, dtype=np.float32).copy()
    gt = np.asarray(gt, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    pred_filled[~valid] = gt[~valid]
    pred_t = torch.from_numpy(pred_filled).permute(2, 0, 1).unsqueeze(0).to(device)
    gt_t = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(device)
    return pred_t.clamp(0.0, 1.0), gt_t.clamp(0.0, 1.0)


def _build_metrics(device: torch.device):
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    try:
        lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
    except TypeError:
        lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)
    return ssim, lpips


def _eval_rgb_metric(
    pred: np.ndarray,
    gt: np.ndarray,
    alpha: np.ndarray,
    device: torch.device,
    ssim_metric,
    lpips_metric,
    alpha_threshold: float,
) -> dict[str, Any] | None:
    if pred.shape[:2] != gt.shape[:2]:
        return None
    valid = np.isfinite(pred).all(axis=-1) & np.isfinite(gt).all(axis=-1) & np.isfinite(alpha)
    valid &= alpha > float(alpha_threshold)
    if not np.any(valid):
        return None
    psnr = _psnr_masked(pred, gt, valid)
    pred_t, gt_t = _make_metric_tensors(pred, gt, valid, device)
    ssim = float(ssim_metric(pred_t, gt_t).detach().cpu().item())
    lpips = float(lpips_metric(pred_t, gt_t).detach().cpu().item())
    return {
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips,
        "valid_px": int(np.count_nonzero(valid)),
        "valid_ratio": float(np.mean(valid)),
    }


def _write_rgb_preview(path: Path, gt: np.ndarray, pred: np.ndarray, alpha: np.ndarray, alpha_threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    valid = np.isfinite(alpha) & (alpha > float(alpha_threshold))
    pred_show = pred.copy()
    pred_show[~valid] = 0.0
    err = np.abs(pred - gt).mean(axis=-1)
    err[~valid] = 0.0
    denom = float(np.percentile(err[valid], 98.0)) if np.any(valid) else 1.0
    err_rgb = np.repeat(np.clip(err / max(denom, 1e-6), 0.0, 1.0)[..., None], 3, axis=-1)
    mask_rgb = np.repeat(valid[..., None], 3, axis=-1).astype(np.float32)
    canvas = np.concatenate([gt, pred_show, err_rgb, mask_rgb], axis=1)
    cv2.imwrite(str(path), cv2.cvtColor((np.clip(canvas, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "views": 0,
            "mPSNR": None,
            "mSSIM": None,
            "mLPIPS": None,
            "mean_valid_ratio": None,
            "rows": [],
        }
    psnrs = [float(r["psnr"]) for r in rows if r["psnr"] is not None and np.isfinite(r["psnr"])]
    return {
        "views": len(rows),
        "mPSNR": float(np.mean(psnrs)) if psnrs else None,
        "mSSIM": float(np.mean([r["ssim"] for r in rows])),
        "mLPIPS": float(np.mean([r["lpips"] for r in rows])),
        "mean_valid_ratio": float(np.mean([r["valid_ratio"] for r in rows])),
        "rows": rows,
    }


def _default_output_dir(scene_dir: Path) -> Path:
    return scene_dir / "outputs" / "ours_render_eval"


@torch.no_grad()
def _evaluate_training(
    scene_dir: Path,
    splats: dict[str, torch.Tensor],
    edg: dict[str, torch.Tensor] | None,
    app_module: AppearanceOptModule | None,
    args: argparse.Namespace,
    out_dir: Path,
    device: torch.device,
    ssim_metric,
    lpips_metric,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    num_frames = _dataset_frame_count(scene_dir, args.camera_source)
    stride = max(1, int(args.frame_stride))
    preview_frames = {0, max(0, num_frames // 2), max(0, num_frames - 1)}
    train_out = out_dir / "training_render"
    train_out.mkdir(parents=True, exist_ok=True)

    for frame_index in range(0, num_frames, stride):
        if int(args.max_views) > 0 and len(rows) >= int(args.max_views):
            break
        c2w, K, camera_source, sparse_dir, image_name = _load_camera_and_gt_intrinsics(
            scene_dir,
            int(frame_index),
            args.camera_source,
        )
        image_path = _find_image_path(scene_dir, image_name)
        gt = _read_rgb(image_path)
        _, alpha, pred = _render_depth(
            splats,
            edg,
            app_module,
            int(frame_index),
            c2w,
            K,
            gt.shape[0],
            gt.shape[1],
            args,
        )
        metric = _eval_rgb_metric(pred, gt, alpha, device, ssim_metric, lpips_metric, args.alpha_threshold)
        if metric is None:
            continue
        row = {
            "frame": int(frame_index),
            "image_name": image_name,
            "gt_image": str(image_path),
            "camera_source": camera_source,
            "sparse_dir": sparse_dir,
            **metric,
        }
        rows.append(row)
        if args.save_images or frame_index in preview_frames:
            cv2.imwrite(
                str(train_out / f"frame{frame_index:06d}_pred.png"),
                cv2.cvtColor((np.clip(pred, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(str(train_out / f"frame{frame_index:06d}_alpha.png"), (np.clip(alpha, 0.0, 1.0) * 255).astype(np.uint8))
            _write_rgb_preview(train_out / f"frame{frame_index:06d}_compare.png", gt, pred, alpha, args.alpha_threshold)
    return rows


@torch.no_grad()
def _evaluate_novel(
    scene_dir: Path,
    splats: dict[str, torch.Tensor],
    edg: dict[str, torch.Tensor] | None,
    app_module: AppearanceOptModule | None,
    args: argparse.Namespace,
    out_dir: Path,
    device: torch.device,
    ssim_metric,
    lpips_metric,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not args.eval_novel:
        return [], None
    novel_dir = (args.novel_dir or (scene_dir / "novel")).resolve()
    if not novel_dir.is_dir():
        return [], None
    records = _load_novel_metadata(scene_dir, novel_dir)
    if not records:
        return [], None

    s, R_align, t_align, align_method = _alignment_est_to_gt(scene_dir, args.camera_source)
    novel_out = out_dir / "novel_render"
    novel_out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    stride = max(1, int(args.frame_stride))

    for rec_idx, rec in enumerate(records):
        if rec_idx % stride != 0:
            continue
        if int(args.max_views) > 0 and len(rows) >= int(args.max_views):
            break
        image_path = Path(rec["image_path"])
        if not image_path.is_file():
            continue
        gt = _read_rgb(image_path)
        base_frame = int(rec["base_frame"])
        base_key = _gt_key_for_frame(scene_dir, base_frame)
        base_image = _read_rgb(_find_image_path(scene_dir, base_key))
        K = _scale_K_for_size(_load_gt_intrinsics(scene_dir, base_key), base_image.shape[:2], gt.shape[:2])
        c2w_est = _transform_gt_c2w_to_est(rec["c2w_gt"], s, R_align, t_align)
        _, alpha, pred = _render_depth(
            splats,
            edg,
            app_module,
            base_frame,
            c2w_est,
            K,
            gt.shape[0],
            gt.shape[1],
            args,
        )
        metric = _eval_rgb_metric(pred, gt, alpha, device, ssim_metric, lpips_metric, args.alpha_threshold)
        if metric is None:
            continue
        novel_id = int(rec["novel_id"])
        rows.append(
            {
                "novel_id": novel_id,
                "base_frame": base_frame,
                "gt_image": str(image_path),
                **metric,
            }
        )
        if args.save_images or novel_id < 5:
            cv2.imwrite(
                str(novel_out / f"novel{novel_id:06d}_pred.png"),
                cv2.cvtColor((np.clip(pred, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(str(novel_out / f"novel{novel_id:06d}_alpha.png"), (np.clip(alpha, 0.0, 1.0) * 255).astype(np.uint8))
            _write_rgb_preview(novel_out / f"novel{novel_id:06d}_compare.png", gt, pred, alpha, args.alpha_threshold)

    return rows, {
        "novel_dir": str(novel_dir),
        "novel_alignment_scale": float(s),
        "novel_alignment_method": align_method,
    }


def main() -> None:
    args = _parse_args()
    scene_dir = args.scene_dir.resolve()
    out_dir = (args.output_dir or _default_output_dir(scene_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    gaussian_paths = _resolve_gaussian_paths(scene_dir, args.gaussians)
    splats, edg, app_state, step = _load_gaussians(gaussian_paths, device)
    app_module = _build_app_module(app_state, splats, device)
    ssim_metric, lpips_metric = _build_metrics(device)

    train_rows = _evaluate_training(scene_dir, splats, edg, app_module, args, out_dir, device, ssim_metric, lpips_metric)
    novel_rows, novel_meta = _evaluate_novel(scene_dir, splats, edg, app_module, args, out_dir, device, ssim_metric, lpips_metric)

    result: dict[str, Any] = {
        "scene_dir": str(scene_dir),
        "gaussians": [str(path) for path in gaussian_paths],
        "step": int(step),
        "camera_source": args.camera_source,
        "alpha_threshold": float(args.alpha_threshold),
        "uses_app_module": app_module is not None,
        "uses_edg_warp": edg is not None and not args.no_edg_warp,
        "training": _summarize(train_rows),
        "novel": _summarize(novel_rows),
    }
    if novel_meta is not None:
        result["novel"].update(novel_meta)

    json_out = (args.output_json or (out_dir / "ours_render_metrics_summary.json")).resolve()
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[ours-render] saved summary to {json_out}")
    for key in ("training", "novel"):
        summary = result[key]
        if summary["views"] > 0:
            print(
                f"[ours-render] {key} views={summary['views']} "
                f"mPSNR={summary['mPSNR']:.4f} mSSIM={summary['mSSIM']:.4f} "
                f"mLPIPS={summary['mLPIPS']:.4f}"
            )
    if args.json:
        print("RESULT_JSON:", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
