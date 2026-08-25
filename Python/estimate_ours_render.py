#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import copy
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
    _load_training_gt_pose_pairs,
    _project_rotation,
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
    parser.add_argument(
        "--view-split",
        choices=["val", "train", "all"],
        default="val",
        help="Frames for RGB metrics. 'val' matches simple_trainer.py final evaluation.",
    )
    parser.add_argument("--test-every", type=int, default=0, help="Validation stride used by simple_trainer.py.")
    parser.add_argument("--render-scale", type=float, default=1.0)
    parser.add_argument(
        "--metric-mask",
        choices=["full", "alpha", "covisible"],
        default="covisible",
        help="'covisible' = HONEST DyCheck/MoSca protocol: score ALL co-visible pixels over the "
             "opaque (black-bg) render, holes count as error (matches deps/MoSca/eval_utils/"
             "dycheck_metrics.py). 'alpha' additionally drops covisible pixels with rendered "
             "alpha<=threshold (holes are free -> OPTIMISTIC, NOT comparable to MoSca); kept for "
             "reference only. 'full' = whole image (matches simple_trainer.py).",
    )
    parser.add_argument("--novel-app", choices=["base", "zero", "mean"], default="base",
                        help="Appearance embedding for NOVEL views when app_opt is on. 'base' uses "
                             "the base training frame's exposure embedding (can impose that frame's "
                             "exposure on the test camera); 'zero'/'mean' render a neutral appearance "
                             "(no test-GT used). 'mean' = average of all training embeddings.")
    parser.add_argument("--alpha-threshold", "--alpha_threshold", dest="alpha_threshold", type=float, default=0.99)
    parser.add_argument("--use-covisible-mask", action=argparse.BooleanOptionalAction, default=True,
                        help="Restrict novel-view metrics to co-visible pixels (DyCheck protocol); "
                             "reads the co-visible mask from each novel record's depth_path.")
    parser.add_argument("--psnr-full-denom", action=argparse.BooleanOptionalAction, default=False,
                        help="PSNR mean over the WHOLE image with BOTH pred and gt zeroed outside the "
                             "mask (out-of-mask -> (0-0)=0 error, but counted in the denominator): "
                             "mse = sum_in-mask(pred-gt)^2 / (all pixels). Mask still defines the region.")
    parser.add_argument("--color-calib", choices=["none", "scene", "per-view"], default="none",
                        help="Novel-view appearance/exposure calibration. 'scene' fits ONE per-channel "
                             "affine (gain+bias) across ALL novel views' masked pixels (defensible: "
                             "corrects the physical test-camera-vs-training-camera color offset). "
                             "'per-view' fits per image (test-time; report as such). 'none' = raw.")
    parser.add_argument("--antialiased", action="store_true",
                        help="Render with antialiased rasterization (match a model trained with --antialiased).")
    parser.add_argument("--novel-align", choices=["current", "adjusted-rotavg", "local"], default="current",
                        help="est->GT Sim3 used to place the novel cameras. 'current' = _alignment_est_to_gt "
                             "(default protocol). 'adjusted-rotavg' = align the pose_adjust-refined training "
                             "cameras (the ones the model was actually trained against) with rotation-averaged R "
                             "and scale refit for that R; still a single GLOBAL Sim3 from training-side info only. "
                             "'local' = rotation-avg + scale/t over the --novel-align-k training cameras nearest "
                             "to the test camera (DIAGNOSTIC upper bound; uses the test-cam position to weight "
                             "training frames -- report separately, not protocol-comparable).")
    parser.add_argument("--novel-align-k", type=int, default=20, help="k nearest training cams for --novel-align local.")
    parser.add_argument("--near-plane", type=float, default=0.01)
    parser.add_argument("--far-plane", type=float, default=1e10)
    parser.add_argument("--sh-degree", type=int, default=1)
    parser.add_argument("--lpips-net", choices=["alex", "vgg"], default="alex")
    parser.add_argument(
        "--sync-train-cfg",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Adopt sh-degree/near/far from training cfg.yml when available.",
    )
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


def _load_novel_intrinsics(novel_dir: Path, image_name: str, dst_hw: tuple[int, int]) -> np.ndarray | None:
    """Per-test-camera intrinsics for a novel view, from novel/cameras/<stem>.json.

    Each DyCheck test camera has its own K (focal/principal-point differ from the handheld
    training frames). Rendering novel views with the training frame's K misregisters the
    render over the wide-baseline static background and deflates the metric. Returns K at the
    render resolution dst_hw=(H,W), or None if the camera json is unavailable (caller falls back).
    """
    cam_path = novel_dir / "cameras" / (Path(image_name).stem + ".json")
    if not cam_path.is_file():
        return None
    try:
        cam = json.loads(cam_path.read_text(encoding="utf-8"))
        focal = float(cam["focal_length"])
        cx, cy = (float(v) for v in cam["principal_point"])
        par = float(cam.get("pixel_aspect_ratio", 1.0))
        skew = float(cam.get("skew", 0.0))
        json_w = float(cam["image_size"][0])  # DyCheck image_size = [W, H] at json (full) res
    except (KeyError, ValueError, TypeError):
        return None
    if json_w <= 0:
        return None
    scale = float(dst_hw[1]) / json_w
    return np.array(
        [[focal * scale, skew * scale, cx * scale],
         [0.0, focal * par * scale, cy * scale],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


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


def _build_metrics(device: torch.device, lpips_net: str):
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    if lpips_net == "vgg":
        lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(device)
    else:
        lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
    return ssim, lpips


def _build_old_metrics(device: torch.device):
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
    metric_mask: str,
    psnr_full_denom: bool = False,
    cov_mask: np.ndarray | None = None,
) -> dict[str, Any] | None:
    if pred.shape[:2] != gt.shape[:2]:
        return None
    finite = np.isfinite(pred).all(axis=-1) & np.isfinite(gt).all(axis=-1)
    # honest all-covisible mask (MoSca protocol): every co-visible pixel scored over the opaque
    # black-bg render, holes count as error. alpha here is already co-visibility-folded.
    covis = finite & (cov_mask if cov_mask is not None else np.ones_like(finite))
    valid = finite.copy()
    if metric_mask == "alpha":
        valid &= np.isfinite(alpha) & (alpha > float(alpha_threshold))
    elif metric_mask == "covisible":
        valid = covis
    if not np.any(valid):
        return None
    if psnr_full_denom:
        # PSNR mean over the WHOLE image, with BOTH pred and gt zeroed outside the mask,
        # so out-of-mask pixels contribute (0-0)=0 error but still count in the denominator:
        #   mse = sum_{in-mask}(pred-gt)^2 / (H*W*3)
        mask3 = valid[..., None]
        pred_zero = np.where(mask3, pred, 0.0)
        gt_zero = np.where(mask3, gt, 0.0)
        diff = pred_zero - gt_zero
        mse = float(np.mean(diff * diff))
        psnr = float("inf") if mse <= 1e-12 else float(-10.0 * np.log10(mse))
    else:
        psnr = _psnr_masked(pred, gt, valid)
    pred_t, gt_t = _make_metric_tensors(pred, gt, valid, device)
    ssim = float(ssim_metric(pred_t, gt_t).detach().cpu().item())
    lpips = float(lpips_metric(pred_t, gt_t).detach().cpu().item())
    # Always report the honest all-covisible PSNR (MoSca-comparable) for reference, whatever the
    # primary metric_mask is. Uses simple masked MSE (holes=black count as error), no full-denom.
    psnr_covis = _psnr_masked(pred, gt, covis) if np.any(covis) else None
    # Alpha-gated PSNR for reference (optimistic; drops holes).
    alpha_valid = finite & np.isfinite(alpha) & (alpha > float(alpha_threshold))
    psnr_alpha = _psnr_masked(pred, gt, alpha_valid) if np.any(alpha_valid) else None
    return {
        "psnr": psnr,
        "psnr_covis": psnr_covis,
        "psnr_alpha": psnr_alpha,
        "ssim": ssim,
        "lpips": lpips,
        "valid_px": int(np.count_nonzero(valid)),
        "valid_ratio": float(np.mean(valid)),
        "valid_ratio_covis": float(np.mean(covis)),
    }


def _frame_indices(num_frames: int, args: argparse.Namespace) -> list[int]:
    indices = list(range(int(num_frames)))
    if args.view_split == "val":
        test_every = int(args.test_every)
        if test_every <= 0:
            indices = []
        else:
            indices = [idx for idx in indices if idx % test_every == 0]
    elif args.view_split == "train":
        indices = indices
    elif args.view_split == "all":
        indices = indices
    else:
        raise ValueError(f"Unsupported view split: {args.view_split}")
    stride = max(1, int(args.frame_stride))
    return indices[::stride]


def _resize_for_render_scale(
    gt: np.ndarray,
    K: np.ndarray,
    render_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    scale = float(render_scale)
    if abs(scale - 1.0) <= 1e-8:
        return gt, np.asarray(K, dtype=np.float32)
    if scale <= 0.0:
        raise ValueError(f"render_scale must be > 0, got {scale}")
    src_h, src_w = gt.shape[:2]
    dst_w = max(1, int(round(src_w * scale)))
    dst_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(gt, (dst_w, dst_h), interpolation=cv2.INTER_AREA)
    K_scaled = np.asarray(K, dtype=np.float32).copy()
    K_scaled[0, :] *= float(dst_w) / float(max(src_w, 1))
    K_scaled[1, :] *= float(dst_h) / float(max(src_h, 1))
    return resized.astype(np.float32), K_scaled


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
            "mPSNR_covis": None,
            "mPSNR_alpha": None,
            "mSSIM": None,
            "mLPIPS": None,
            "mean_valid_ratio": None,
            "mean_valid_ratio_covis": None,
            "rows": [],
        }
    psnrs = [float(r["psnr"]) for r in rows if r["psnr"] is not None and np.isfinite(r["psnr"])]
    psnrs_covis = [float(r["psnr_covis"]) for r in rows if r.get("psnr_covis") is not None and np.isfinite(r.get("psnr_covis", float("nan")))]
    psnrs_alpha = [float(r["psnr_alpha"]) for r in rows if r.get("psnr_alpha") is not None and np.isfinite(r.get("psnr_alpha", float("nan")))]
    return {
        "views": len(rows),
        "mPSNR": float(np.mean(psnrs)) if psnrs else None,
        "mPSNR_covis": float(np.mean(psnrs_covis)) if psnrs_covis else None,
        "mPSNR_alpha": float(np.mean(psnrs_alpha)) if psnrs_alpha else None,
        "mSSIM": float(np.mean([r["ssim"] for r in rows])),
        "mLPIPS": float(np.mean([r["lpips"] for r in rows])),
        "mean_valid_ratio": float(np.mean([r["valid_ratio"] for r in rows])),
        "mean_valid_ratio_covis": float(np.mean([r.get("valid_ratio_covis", 0.0) for r in rows])),
        "rows": rows,
    }


def _default_output_dir(scene_dir: Path) -> Path:
    return scene_dir / "outputs" / "ours_render_eval"


def _load_yaml(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        import yaml
    except Exception:
        return None
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _infer_result_dir_from_gaussians(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    path = paths[0]
    if path.name.endswith(".pt"):
        if path.parent.name == "ckpts":
            return path.parent.parent
        return path.parent
    return None


def _load_pose_adjust_state(paths: list[Path], device: torch.device) -> dict[str, torch.Tensor] | None:
    if not paths:
        return None
    try:
        ckpt = torch.load(str(paths[0]), map_location=device)
    except Exception:
        return None
    state = ckpt.get("pose_adjust")
    if state is None:
        return None
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in state.items()}


def _load_gsplat_utils_module() -> Any | None:
    module_path = Path(__file__).resolve().parents[1] / "deps" / "gsplat" / "examples" / "utils.py"
    if not module_path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("gsplat_examples_utils", module_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_pose_adjust_module(
    pose_state: dict[str, torch.Tensor] | None,
    device: torch.device,
) -> torch.nn.Module | None:
    if pose_state is None or "embeds.weight" not in pose_state:
        return None
    gsplat_utils = _load_gsplat_utils_module()
    if gsplat_utils is None or not hasattr(gsplat_utils, "CameraOptModule"):
        return None
    n = int(pose_state["embeds.weight"].shape[0])
    module = gsplat_utils.CameraOptModule(n).to(device)
    module.load_state_dict({key.removeprefix("module."): value for key, value in pose_state.items()})
    module.eval()
    return module


def _apply_pose_adjust(c2w: np.ndarray, frame_index: int, pose_adjust: torch.nn.Module | None, device: torch.device) -> np.ndarray:
    if pose_adjust is None:
        return c2w
    with torch.no_grad():
        cam = torch.from_numpy(c2w).float().to(device)[None]
        embed_ids = torch.tensor([int(frame_index)], device=device, dtype=torch.long)
        cam = pose_adjust(cam, embed_ids)
        return cam[0].detach().cpu().numpy()


@torch.no_grad()
def _evaluate_training(
    scene_dir: Path,
    splats: dict[str, torch.Tensor],
    edg: dict[str, torch.Tensor] | None,
    app_module: AppearanceOptModule | None,
    pose_adjust: torch.nn.Module | None,
    args: argparse.Namespace,
    out_dir: Path,
    device: torch.device,
    ssim_metric,
    lpips_metric,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    num_frames = _dataset_frame_count(scene_dir, args.camera_source)
    train_out = out_dir / "training_render"
    train_out.mkdir(parents=True, exist_ok=True)

    frame_indices = _frame_indices(num_frames, args)
    preview_frames = {frame_indices[0], frame_indices[len(frame_indices) // 2], frame_indices[-1]} if frame_indices else set()
    for frame_index in frame_indices:
        if int(args.max_views) > 0 and len(rows) >= int(args.max_views):
            break
        c2w, K, camera_source, sparse_dir, image_name = _load_camera_and_gt_intrinsics(
            scene_dir,
            int(frame_index),
            args.camera_source,
        )
        c2w = _apply_pose_adjust(c2w, int(frame_index), pose_adjust, device)
        image_path = _find_image_path(scene_dir, image_name)
        gt = _read_rgb(image_path)
        gt, K = _resize_for_render_scale(gt, K, float(args.render_scale))
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
        metric = _eval_rgb_metric(
            pred,
            gt,
            alpha,
            device,
            ssim_metric,
            lpips_metric,
            args.alpha_threshold,
            args.metric_mask,
            args.psnr_full_denom,
        )
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
def _metric_valid_mask(pred, gt, alpha, alpha_threshold, metric_mask):
    valid = np.isfinite(pred).all(axis=-1) & np.isfinite(gt).all(axis=-1)
    if metric_mask == "alpha":
        valid &= np.isfinite(alpha) & (alpha > float(alpha_threshold))
    return valid


def _fit_color_affine(pairs, mode):
    """Fit a per-channel affine (gain, bias) mapping pred->gt over masked pixels.

    pairs: list of (pred[H,W,3], gt[H,W,3], valid[H,W]).  mode 'scene' -> one shared
    affine from all views; anything else -> identity. Returns (gain[3], bias[3]).
    """
    gain = np.ones(3, np.float64)
    bias = np.zeros(3, np.float64)
    if mode != "scene":
        return gain, bias
    for c in range(3):
        xs, ys = [], []
        for pred, gt, valid in pairs:
            if not np.any(valid):
                continue
            xs.append(pred[valid, c].astype(np.float64))
            ys.append(gt[valid, c].astype(np.float64))
        if xs:
            x = np.concatenate(xs)
            y = np.concatenate(ys)
            A = np.vstack([x, np.ones_like(x)]).T
            sol = np.linalg.lstsq(A, y, rcond=None)[0]
            gain[c], bias[c] = float(sol[0]), float(sol[1])
    return gain, bias


def _apply_color_affine(pred, gain, bias):
    out = pred.astype(np.float64) * gain[None, None, :] + bias[None, None, :]
    return np.clip(out, 0.0, 1.0).astype(pred.dtype)


def _rotavg_similarity(est_c2ws, gt_c2ws, sub=None):
    """Global Sim3 est->GT: R = rotation average of gt_R @ est_R^T, then LSQ scale+t for that R.

    Unlike _alignment_est_to_gt (which keeps the scale fitted with the POSITIONAL SVD rotation when
    it swaps in the rotation-averaged R), the scale here is refit for the R actually used.
    `sub` restricts the fit to a subset of frame indices (used by --novel-align local).
    """
    idx = list(range(len(est_c2ws))) if sub is None else list(sub)
    mats = [gt_c2ws[i][:3, :3] @ est_c2ws[i][:3, :3].T for i in idx]
    R = _project_rotation(np.sum(mats, axis=0))
    src = np.stack([est_c2ws[i][:3, 3] for i in idx])
    dst = np.stack([gt_c2ws[i][:3, 3] for i in idx])
    src_c = src - src.mean(0)
    dst_c = dst - dst.mean(0)
    s = float(np.sum(dst_c * (src_c @ R.T)) / max(np.sum(src_c * src_c), 1e-12))
    if not np.isfinite(s) or s <= 1e-12:
        s = 1.0
    t = dst.mean(0) - s * R @ src.mean(0)
    return s, R, t.reshape(3, 1)


def _novel_alignment(
    scene_dir: Path,
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    pose_adjust: torch.nn.Module | None,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, str]:
    mode = getattr(args, "novel_align", "current")
    if mode == "current":
        return _alignment_est_to_gt(scene_dir, args.camera_source)
    est_c2ws, gt_c2ws, _ = _load_training_gt_pose_pairs(scene_dir, args.camera_source)
    method = mode
    if mode == "adjusted-rotavg" and pose_adjust is None:
        method += "(no pose_adjust; raw cams)"
    if mode == "adjusted-rotavg" and pose_adjust is not None:
        # embed ids follow simple_trainer's train-frame ordering (val = idx % test_every == 0).
        te = int(args.test_every)
        train_ids = [i for i in range(len(est_c2ws)) if te <= 0 or i % te != 0]
        n_embed = int(pose_adjust.embeds.weight.shape[0])
        if n_embed == len(train_ids):
            est_c2ws = list(est_c2ws)
            for embed_id, frame_id in enumerate(train_ids):
                est_c2ws[frame_id] = _apply_pose_adjust(
                    np.asarray(est_c2ws[frame_id], np.float64), embed_id, pose_adjust, device
                )
            method += "(pose_adjusted)"
        else:
            method += f"(WARN embeds {n_embed}!={len(train_ids)} train frames; raw cams)"
    sub = None
    if mode == "local":
        centers = np.stack([rec["c2w_gt"][:3, 3] for rec in records]).mean(0)
        gt_centers = np.stack([pose[:3, 3] for pose in gt_c2ws])
        k = max(4, int(getattr(args, "novel_align_k", 20)))
        sub = np.argsort(np.linalg.norm(gt_centers - centers[None], axis=1))[:k]
        method += f"(k={k})"
    s, R, t = _rotavg_similarity(est_c2ws, gt_c2ws, sub=sub)
    return s, R, t, method


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
    pose_adjust: torch.nn.Module | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not args.eval_novel:
        return [], None
    novel_dir = (args.novel_dir or (scene_dir / "novel")).resolve()
    if not novel_dir.is_dir():
        return [], None
    records = _load_novel_metadata(scene_dir, novel_dir)
    if not records:
        return [], None

    s, R_align, t_align, align_method = _novel_alignment(scene_dir, args, records, pose_adjust, device)
    novel_out = out_dir / "novel_render"
    novel_out.mkdir(parents=True, exist_ok=True)
    stride = max(1, int(args.frame_stride))
    color_calib = getattr(args, "color_calib", "none")

    # Phase 1: render + mask every novel view, keep pred/gt/alpha/valid for calibration.
    collected: list[dict[str, Any]] = []
    for rec_idx, rec in enumerate(records):
        if rec_idx % stride != 0:
            continue
        if int(args.max_views) > 0 and len(collected) >= int(args.max_views):
            break
        image_path = Path(rec["image_path"])
        if not image_path.is_file():
            continue
        gt = _read_rgb(image_path)
        base_frame = int(rec["base_frame"])
        # Use each test camera's OWN intrinsics (novel/cameras/<stem>.json); only fall back to
        # the training base-frame K if the per-view camera json is missing. Rendering with the
        # training K misregisters the render over the wide-baseline static bg and deflates mPSNR.
        K = _load_novel_intrinsics(novel_dir, rec["image_name"], gt.shape[:2])
        if K is None:
            base_key = _gt_key_for_frame(scene_dir, base_frame)
            base_image = _read_rgb(_find_image_path(scene_dir, base_key))
            K = _scale_K_for_size(_load_gt_intrinsics(scene_dir, base_key), base_image.shape[:2], gt.shape[:2])
        c2w_est = _transform_gt_c2w_to_est(rec["c2w_gt"], s, R_align, t_align)
        _, alpha, pred = _render_depth(
            splats, edg, app_module, base_frame, c2w_est, K, gt.shape[0], gt.shape[1], args,
            app_embed_mode=getattr(args, "novel_app", "base"),
        )
        cov_mask = None
        if getattr(args, "use_covisible_mask", False) and rec.get("depth_path"):
            cov_path = Path(rec["depth_path"])
            if cov_path.is_file():
                cov = cv2.imread(str(cov_path), cv2.IMREAD_GRAYSCALE)
                if cov is not None:
                    if cov.shape[:2] != alpha.shape[:2]:
                        cov = cv2.resize(cov, (alpha.shape[1], alpha.shape[0]), interpolation=cv2.INTER_NEAREST)
                    cov_mask = (cov > 127)
                    alpha = alpha * cov_mask.astype(alpha.dtype)
        if pred.shape[:2] != gt.shape[:2]:
            continue
        valid = _metric_valid_mask(pred, gt, alpha, args.alpha_threshold, args.metric_mask)
        collected.append({
            "novel_id": int(rec["novel_id"]), "base_frame": base_frame, "gt_image": str(image_path),
            "pred": pred.astype(np.float32), "gt": gt.astype(np.float32),
            "alpha": alpha.astype(np.float32), "valid": valid,
            "cov_mask": cov_mask,
        })

    if not collected:
        return [], None

    # Fit the color calibration (scene-global by default per the DyCheck sensor-offset argument).
    gain, bias = _fit_color_affine([(c["pred"], c["gt"], c["valid"]) for c in collected], color_calib)

    # Phase 2: apply calibration, score, save.
    rows: list[dict[str, Any]] = []
    for c in collected:
        pred = c["pred"]
        if color_calib == "per-view":
            g_i, b_i = _fit_color_affine([(pred, c["gt"], c["valid"])], "scene")
            pred = _apply_color_affine(pred, g_i, b_i)
        elif color_calib == "scene":
            pred = _apply_color_affine(pred, gain, bias)
        metric = _eval_rgb_metric(
            pred, c["gt"], c["alpha"], device, ssim_metric, lpips_metric, args.alpha_threshold,
            args.metric_mask, args.psnr_full_denom, cov_mask=c.get("cov_mask"),
        )
        if metric is None:
            continue
        novel_id = c["novel_id"]
        rows.append({"novel_id": novel_id, "base_frame": c["base_frame"], "gt_image": c["gt_image"], **metric})
        if args.save_images or novel_id < 5:
            cv2.imwrite(
                str(novel_out / f"novel{novel_id:06d}_pred.png"),
                cv2.cvtColor((np.clip(pred, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(str(novel_out / f"novel{novel_id:06d}_alpha.png"), (np.clip(c["alpha"], 0.0, 1.0) * 255).astype(np.uint8))
            _write_rgb_preview(novel_out / f"novel{novel_id:06d}_compare.png", c["gt"], pred, c["alpha"], args.alpha_threshold)

    return rows, {
        "novel_dir": str(novel_dir),
        "novel_alignment_scale": float(s),
        "novel_alignment_method": align_method,
        "color_calib": color_calib,
        "color_calib_gain": [float(x) for x in gain],
        "color_calib_bias": [float(x) for x in bias],
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
    result_dir = _infer_result_dir_from_gaussians(gaussian_paths)
    if args.sync_train_cfg and result_dir is not None:
        cfg_path = result_dir / "cfg.yml"
        cfg = _load_yaml(cfg_path)
        if cfg is not None:
            if "sh_degree" in cfg:
                args.sh_degree = int(cfg["sh_degree"])
            if "near_plane" in cfg:
                args.near_plane = float(cfg["near_plane"])
            if "far_plane" in cfg:
                args.far_plane = float(cfg["far_plane"])
            if "render_scale" in cfg:
                args.render_scale = float(cfg["render_scale"])
            if "test_every" in cfg:
                args.test_every = int(cfg["test_every"])
            if "lpips_net" in cfg:
                args.lpips_net = str(cfg["lpips_net"])

    # Copy AFTER cfg sync so novel renders with the same sh_degree/near/far as train.
    novel_args = copy.copy(args)

    pose_state = _load_pose_adjust_state(gaussian_paths, device)
    pose_adjust = _build_pose_adjust_module(pose_state, device)
    ssim_metric, lpips_metric = _build_metrics(device, args.lpips_net)
    novel_ssim_metric, novel_lpips_metric = _build_old_metrics(device)

    train_rows = _evaluate_training(
        scene_dir,
        splats,
        edg,
        app_module,
        pose_adjust,
        args,
        out_dir,
        device,
        ssim_metric,
        lpips_metric,
    )
    novel_rows, novel_meta = _evaluate_novel(
        scene_dir,
        splats,
        edg,
        app_module,
        novel_args,
        out_dir,
        device,
        novel_ssim_metric,
        novel_lpips_metric,
        pose_adjust=pose_adjust,
    )

    result: dict[str, Any] = {
        "scene_dir": str(scene_dir),
        "gaussians": [str(path) for path in gaussian_paths],
        "step": int(step),
        "camera_source": args.camera_source,
        "view_split": args.view_split,
        "test_every": int(args.test_every),
        "render_scale": float(args.render_scale),
        "metric_mask": args.metric_mask,
        "alpha_threshold": float(args.alpha_threshold),
        "lpips_net": args.lpips_net,
        "training_convention": "current_trainer_synced",
        "novel_convention": "estimate_our_render_old.py",
        "novel_metric_mask": args.metric_mask,
        "novel_lpips_net": "alex",
        "uses_app_module": app_module is not None,
        "uses_pose_adjust": pose_adjust is not None,
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
    print(
        f"[ours-render] convention split={args.view_split} metric_mask={args.metric_mask} "
        f"render_scale={float(args.render_scale):g} test_every={int(args.test_every)} lpips={args.lpips_net}"
    )
    print("[ours-render] novel convention matches estimate_our_render_old.py")
    for key in ("training", "novel"):
        summary = result[key]
        if summary["views"] > 0:
            mpc = summary.get("mPSNR_covis")
            mpa = summary.get("mPSNR_alpha")
            vrc = summary.get("mean_valid_ratio_covis")
            print(
                f"[ours-render] {key} views={summary['views']} "
                f"mPSNR={summary['mPSNR']:.4f} mSSIM={summary['mSSIM']:.4f} "
                f"mLPIPS={summary['mLPIPS']:.4f}"
                + (f" | mPSNR_covis={mpc:.4f}" if mpc is not None else "")
                + (f" mPSNR_alpha={mpa:.4f}" if mpa is not None else "")
                + (f" covis_ratio={vrc:.3f}" if vrc is not None else "")
            )
    if args.json:
        print("RESULT_JSON:", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
