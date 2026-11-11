#!/usr/bin/env python3
"""Novel-view eval with MoSca-style test-time pose optimization (TTO).

MoSca's published DyCheck eval does NOT place test cameras with the global trajectory
alignment alone: mosca_reconstruct.py calls test_main(tto_flag=True), which runs
mosca_evaluate.render_test_tto — after a global alignment of the solved training
trajectory to GT, each test camera's SE(3) is photometrically optimized against the
test image itself (Gaussians frozen; Adam on translation+quaternion; loss = -PSNR over
the rendered-alpha mask; each view warm-started from the previous solved pose).

This script applies the same protocol to our gsplat checkpoints so novel-view numbers
are MoSca-comparable. Scoring (covisible mask, GT-fill, metrics) is identical to
estimate_ours_render.py.

Methods (--novel-align):
  sim3    : global Sim3 only (== estimate_ours_render.py 'current' protocol; baseline)
  tto     : MoSca parity — per-view SE(3) TTO from the sim3 init, chained warm start
            (restarts the long schedule at each physical-camera change)
  tto-cam : one shared SE(3) per physical test camera (DyCheck tripods are static),
            optimized jointly over sampled timestamps — fewer DoF than per-view TTO
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from eval_depth_gaussian_ours import (
    AppearanceOptModule,
    _alignment_est_to_gt,
    _build_app_module,
    _gt_key_for_frame,
    _load_camera_and_gt_intrinsics,
    _load_gaussians,
    _load_gt_intrinsics,
    _load_novel_metadata,
    _render_depth,
    _resolve_gaussian_paths,
    _splat_colors,
    _transform_gt_c2w_to_est,
    _warp_means_edg,
    _warp_quats_edg,
)
from estimate_ours_render import (
    _apply_color_affine,
    _apply_pose_adjust,
    _build_old_metrics,
    _build_pose_adjust_module,
    _eval_rgb_metric,
    _find_image_path,
    _fit_color_affine,
    _infer_result_dir_from_gaussians,
    _load_novel_intrinsics,
    _load_pose_adjust_state,
    _load_yaml,
    _metric_valid_mask,
    _read_rgb,
    _scale_K_for_size,
    _summarize,
    _write_rgb_preview,
)

from gsplat.rendering import rasterization
from utils import definitions


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene-dir", "--scene-path", type=Path, default=Path(definitions.rootpath))
    parser.add_argument("--gaussians", type=Path, default=None)
    parser.add_argument("--camera-source", choices=["colmap", "xm"], default="colmap")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--novel-dir", type=Path, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-views", type=int, default=0)
    parser.add_argument("--metric-mask", choices=["full", "alpha", "covisible"], default="alpha")
    parser.add_argument("--alpha-threshold", type=float, default=0.99)
    parser.add_argument("--use-covisible-mask", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--color-calib", choices=["none", "scene", "per-view"], default="none")
    parser.add_argument("--novel-app", choices=["base", "zero", "mean", "frame0", "split0"], default="base")
    parser.add_argument("--sh-degree", type=int, default=1)
    parser.add_argument("--near-plane", type=float, default=0.01)
    parser.add_argument("--far-plane", type=float, default=1e10)
    parser.add_argument("--antialiased", action="store_true")
    parser.add_argument("--sync-train-cfg", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-edg-warp", action="store_true")
    parser.add_argument("--edg-warp-quats", action="store_true",
                        help="Rotate dynamic Gaussian orientations by the skinned EDG rotation (match a "
                             "model trained with --edg-warp-quats). Requires --edg-target-frame.")
    parser.add_argument("--edg-target-frame", type=int, default=0,
                        help="Target/canonical frame the dynamic Gaussian orientations are defined in "
                             "(for --edg-warp-quats; must match training).")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--device", default="cuda")
    # pose method
    parser.add_argument("--novel-align", choices=["sim3", "tto", "tto-cam"], default="tto")
    parser.add_argument("--novel-fixed-cam", type=int, default=-1,
                        help="If >=0, place EVERY test view at this est TRAINING camera (pose-refined) with "
                             "its training K, bypassing Sim3/TTO. For NVIDIA's fixed-v000 protocol "
                             "(--novel-fixed-cam 0) only time is novel, so frame-0 reproduces the training "
                             "render exactly (t=0 PSNR -> training quality). Overrides --novel-align.")
    # MoSca iphone TTO defaults (mosca_evaluate.test_main iphone branch)
    parser.add_argument("--tto-steps", type=int, default=200)
    parser.add_argument("--tto-decay-start", type=int, default=15)
    parser.add_argument("--tto-lr", type=float, default=0.001)
    parser.add_argument("--tto-lr-final", type=float, default=0.0001)
    parser.add_argument("--tto-alpha-th", type=float, default=0.99)
    parser.add_argument("--tto-loss-mask", choices=["alpha", "covisible", "full"], default="alpha",
                        help="Pixels the TTO photometric loss sees. 'alpha' = rendered alpha > "
                             "--tto-alpha-th (MoSca parity). 'covisible' = the full DyCheck "
                             "covisible region (holes count as loss, matching the metric). "
                             "'full' = every pixel.")
    parser.add_argument("--tto-first-factor", type=int, default=10,
                        help="Step multiplier for the FIRST view of each physical camera.")
    parser.add_argument("--tto-chain-lr-factor", type=float, default=0.1,
                        help="LR multiplier for chained (non-first) views.")
    parser.add_argument("--tto-cam-steps", type=int, default=100)
    parser.add_argument("--tto-cam-samples", type=int, default=24,
                        help="Timestamps sampled per physical camera for tto-cam.")
    parser.add_argument("--camera-group-tol", type=float, default=1e-3,
                        help="Translation tolerance for grouping novel views into physical cameras.")
    parser.add_argument("--skip-pre-window", action=argparse.BooleanOptionalAction, default=True,
                        help="Skip novel views whose source time predates a trimmed train window "
                             "(clamped to base_frame 0 in metadata; un-renderable + poison chained TTO).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# SE(3) parameterization helpers (torch, differentiable)
# ---------------------------------------------------------------------------

def _rotmat_to_quat_np(R: np.ndarray) -> np.ndarray:
    """3x3 -> wxyz quaternion (robust branch method)."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / max(np.linalg.norm(q), 1e-12)


def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """wxyz quaternion (4,) -> 3x3 rotation, differentiable, normalizes q."""
    q = q / q.norm().clamp_min(1e-8)
    w, x, y, z = q[0], q[1], q[2], q[3]
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)]),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)]),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]),
    ])


def _compose_w2c(q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    R = _quat_to_rotmat(q)
    top = torch.cat([R, t[:, None]], dim=1)
    bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=q.device, dtype=top.dtype)
    return torch.cat([top, bottom], dim=0)


def _pose_delta(w2c_a: np.ndarray, w2c_b: np.ndarray) -> tuple[float, float]:
    """(rotation deg, translation norm) between two w2c poses (camera-center based)."""
    Ra, Rb = w2c_a[:3, :3], w2c_b[:3, :3]
    ca = -Ra.T @ w2c_a[:3, 3]
    cb = -Rb.T @ w2c_b[:3, 3]
    cosang = (np.trace(Ra @ Rb.T) - 1.0) / 2.0
    ang = math.degrees(math.acos(min(1.0, max(-1.0, cosang))))
    return float(ang), float(np.linalg.norm(ca - cb))


# ---------------------------------------------------------------------------
# Frozen-Gaussian prep + differentiable pose render
# ---------------------------------------------------------------------------

@torch.no_grad()
def _prep_frozen_gaussians(
    splats: dict[str, torch.Tensor],
    edg: dict[str, torch.Tensor] | None,
    app_module: AppearanceOptModule | None,
    frame_index: int,
    c2w_init: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    device = splats["means"].device
    means = splats["means"].float()
    quats_use = splats["quats"].float()
    if not args.no_edg_warp:
        means = _warp_means_edg(means, edg, int(frame_index))
        if getattr(args, "edg_warp_quats", False):
            quats_use = _warp_quats_edg(quats_use, edg, int(frame_index),
                                        int(getattr(args, "edg_target_frame", 0)))
    if app_module is not None:
        camtoworld = torch.from_numpy(np.asarray(c2w_init, np.float32)).to(device)[None]
        image_ids: torch.Tensor | None = torch.tensor([int(frame_index)], device=device, dtype=torch.long)
        embed_override = None
        if args.novel_app == "zero":
            image_ids = None
        elif args.novel_app == "mean":
            image_ids = None
            embed_override = app_module.embeds.weight.detach().mean(dim=0)
        elif args.novel_app == "frame0":
            # fixed-v000 protocol: the novel camera is physical camera 0 -> frame-0 embedding
            image_ids = torch.tensor([0], device=device, dtype=torch.long)
        elif args.novel_app == "split0":
            # hybrid: static -> camera-0 embedding, dynamic -> frame-t embedding
            image_ids = torch.tensor([0], device=device, dtype=torch.long)
        colors = app_module(
            features=splats["features"].float(),
            embed_ids=image_ids,
            dirs=means[None, :, :] - camtoworld[:, None, :3, 3],
            sh_degree=min(int(args.sh_degree), app_module.sh_degree),
            embed_override=embed_override,
        )
        if (args.novel_app == "split0" and edg is not None
                and edg.get("dyn_indices") is not None and int(frame_index) != 0):
            colors_t = app_module(
                features=splats["features"].float(),
                embed_ids=torch.tensor([int(frame_index)], device=device, dtype=torch.long),
                dirs=means[None, :, :] - camtoworld[:, None, :3, 3],
                sh_degree=min(int(args.sh_degree), app_module.sh_degree),
            )
            dyn_i = edg["dyn_indices"].to(device).long()
            colors = colors.clone()
            colors[:, dyn_i] = colors_t[:, dyn_i]
        colors = torch.sigmoid(colors + splats["colors"].float())
        sh_degree = None
    else:
        colors, sh_degree = _splat_colors(splats, int(args.sh_degree))
    return {
        "means": means,
        "quats": quats_use,
        "scales": torch.exp(splats["scales"].float()),
        "opacities": torch.sigmoid(splats["opacities"].float()),
        "colors": colors.float(),
        "sh_degree": sh_degree,
    }


def _render_pose(gs: dict[str, Any], w2c: torch.Tensor, K: torch.Tensor, H: int, W: int,
                 args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    renders, alphas, _ = rasterization(
        means=gs["means"],
        quats=gs["quats"],
        scales=gs["scales"],
        opacities=gs["opacities"],
        colors=gs["colors"],
        viewmats=w2c[None],
        Ks=K[None],
        width=int(W),
        height=int(H),
        sh_degree=gs["sh_degree"],
        near_plane=float(args.near_plane),
        far_plane=float(args.far_plane),
        render_mode="RGB",
        rasterize_mode="antialiased" if getattr(args, "antialiased", False) else "classic",
    )
    return renders[0, ..., :3], alphas[0, ..., 0]


def _load_covis_mask(rec: dict[str, Any], shape_hw: tuple[int, int]) -> np.ndarray | None:
    if not rec.get("depth_path"):
        return None
    cov_path = Path(rec["depth_path"])
    if not cov_path.is_file():
        return None
    cov = cv2.imread(str(cov_path), cv2.IMREAD_GRAYSCALE)
    if cov is None:
        return None
    if cov.shape[:2] != tuple(shape_hw):
        cov = cv2.resize(cov, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return cov > 127


def _tto_loss_mask_tensor(rec: dict[str, Any], shape_hw: tuple[int, int], args: argparse.Namespace,
                          device: torch.device) -> torch.Tensor | None:
    """Fixed loss mask for TTO, or None to fall back to the rendered-alpha mask."""
    if args.tto_loss_mask == "covisible":
        cm = _load_covis_mask(rec, shape_hw)
        if cm is not None:
            return torch.from_numpy(cm).to(device)
        return None
    if args.tto_loss_mask == "full":
        return torch.ones(shape_hw, dtype=torch.bool, device=device)
    return None


def _tto_refine_pose(
    w2c_init: np.ndarray,
    view_batches: list[tuple[dict[str, Any], torch.Tensor, torch.Tensor, int, int, torch.Tensor | None]],
    steps: int,
    decay_start: int,
    lr: float,
    lr_final: float,
    alpha_th: float,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, float, float]:
    """MoSca-style photometric pose refinement (Gaussians frozen).

    view_batches: list of (frozen_gs, gt_tensor[H,W,3], K_tensor[3,3], H, W, loss_mask);
    each step cycles through them (len==1 for per-view TTO). Loss follows render_test_tto:
    -PSNR over loss_mask if given, else over rendered alpha > alpha_th.
    Returns (w2c_refined, first_loss, last_loss).
    """
    t_param = torch.nn.Parameter(torch.from_numpy(w2c_init[:3, 3].astype(np.float32)).to(device))
    q_param = torch.nn.Parameter(torch.from_numpy(_rotmat_to_quat_np(w2c_init[:3, :3]).astype(np.float32)).to(device))
    optimizer = torch.optim.Adam([
        {"params": t_param, "lr": lr},
        {"params": q_param, "lr": lr},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, steps - decay_start), eta_min=lr_final,
    )
    first_loss, last_loss = None, None
    for step in range(steps):
        gs, gt_t, K_t, H, W, loss_mask = view_batches[step % len(view_batches)]
        optimizer.zero_grad()
        w2c = _compose_w2c(q_param, t_param)
        pred, alpha = _render_pose(gs, w2c, K_t, H, W, args)
        mask = loss_mask if loss_mask is not None else (alpha.detach() > alpha_th)
        if not bool(mask.any()):
            break
        mse = ((pred - gt_t) ** 2)[mask].mean()
        loss = 10.0 * torch.log10(mse.clamp_min(1e-12))  # == -PSNR
        loss.backward()
        optimizer.step()
        if step >= decay_start:
            scheduler.step()
        if first_loss is None:
            first_loss = float(loss.item())
        last_loss = float(loss.item())
    with torch.no_grad():
        w2c_final = _compose_w2c(q_param, t_param).detach().cpu().numpy().astype(np.float64)
    return w2c_final, (first_loss if first_loss is not None else float("nan")), (last_loss if last_loss is not None else float("nan"))


def _pre_window_novel_ids(scene_dir: Path, novel_dir: Path) -> set[int]:
    """Novel views whose SOURCE time predates the (possibly trimmed) train window.

    Trimmed scenes (e.g. block, source frames 150..446) clamp such views to base_frame=0 in
    novel/metadata.txt, so they are scored against GT captured before the reconstruction even
    starts (rendered frozen at t=0) and poison chained TTO. Detected via frame_map.txt's minimum
    source index vs each novel record's source_stem time; empty set for untrimmed scenes."""
    fm = scene_dir / "frame_map.txt"
    meta = novel_dir / "metadata.txt"
    if not fm.is_file() or not meta.is_file():
        return set()
    try:
        offset = min(int(line.split()[-1].split("_")[1])
                     for line in fm.read_text().splitlines()
                     if line.strip() and not line.startswith("#"))
    except (ValueError, IndexError):
        return set()
    if offset <= 0:
        return set()
    skip: set[int] = set()
    for line in meta.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        try:
            if int(parts[-1].split("_")[1]) < offset:
                skip.add(int(parts[0]))
        except (ValueError, IndexError):
            continue
    return skip


@torch.no_grad()
def _pose_loss(gs: dict[str, Any], w2c_np: np.ndarray, gt_t: torch.Tensor, K_t: torch.Tensor,
               H: int, W: int, loss_mask: torch.Tensor | None, alpha_th: float,
               args: argparse.Namespace, device: torch.device) -> float:
    """-PSNR of a fixed pose under the same mask rule as the TTO loss (for the divergence guard)."""
    w2c = torch.from_numpy(np.asarray(w2c_np, np.float32)).to(device)
    pred, alpha = _render_pose(gs, w2c, K_t, H, W, args)
    mask = loss_mask if loss_mask is not None else (alpha > alpha_th)
    if not bool(mask.any()):
        return float("inf")
    mse = ((pred - gt_t) ** 2)[mask].mean().clamp_min(1e-12)
    return float(10.0 * torch.log10(mse).item())


def _group_by_camera(records: list[dict[str, Any]], tol: float) -> list[int]:
    """Greedy cluster on GT camera centers -> group id per record (DyCheck static tripods)."""
    centers: list[np.ndarray] = []
    gids: list[int] = []
    for rec in records:
        c = np.asarray(rec["c2w_gt"][:3, 3], np.float64)
        for gi, gc in enumerate(centers):
            if np.linalg.norm(c - gc) <= tol:
                gids.append(gi)
                break
        else:
            centers.append(c)
            gids.append(len(centers) - 1)
    return gids


# ---------------------------------------------------------------------------
# Main eval
# ---------------------------------------------------------------------------

def _evaluate_novel_tto(
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
    novel_dir = (args.novel_dir or (scene_dir / "novel")).resolve()
    records = _load_novel_metadata(scene_dir, novel_dir)
    if not records:
        return [], None
    records = sorted(records, key=lambda r: int(r["novel_id"]))
    if getattr(args, "skip_pre_window", True):
        pre = _pre_window_novel_ids(scene_dir, novel_dir)
        if pre:
            records = [r for r in records if int(r["novel_id"]) not in pre]
            print(f"[tto] skipping {len(pre)} pre-window novel views (source time predates the "
                  f"trimmed train window); {len(records)} remain")
    stride = max(1, int(args.frame_stride))
    records = records[::stride]
    if int(args.max_views) > 0:
        records = records[: int(args.max_views)]

    s, R_align, t_align, align_method = _alignment_est_to_gt(scene_dir, args.camera_source)
    gids = _group_by_camera(records, float(args.camera_group_tol))
    n_groups = len(set(gids))
    print(f"[tto] {len(records)} novel views in {n_groups} physical camera group(s); "
          f"method={args.novel_align} sim3_init={align_method}")

    novel_out = out_dir / "novel_render"
    novel_out.mkdir(parents=True, exist_ok=True)

    # Pre-load per-view data lazily inside the loop; keep per-group chained poses.
    chained_w2c: dict[int, np.ndarray] = {}
    group_pose: dict[int, np.ndarray] = {}
    pose_rows: list[dict[str, Any]] = []

    # tto-cam: solve one pose per group first, over sampled timestamps.
    if args.novel_align == "tto-cam":
        for gi in sorted(set(gids)):
            g_idx = [i for i, g in enumerate(gids) if g == gi]
            samples = g_idx[:: max(1, len(g_idx) // max(1, int(args.tto_cam_samples)))][: int(args.tto_cam_samples)]
            batches = []
            w2c_init = None
            for i in samples:
                rec = records[i]
                image_path = Path(rec["image_path"])
                if not image_path.is_file():
                    continue
                gt = _read_rgb(image_path)
                K = _load_novel_intrinsics(novel_dir, rec["image_name"], gt.shape[:2])
                if K is None:
                    base_key = _gt_key_for_frame(scene_dir, int(rec["base_frame"]))
                    base_image = _read_rgb(_find_image_path(scene_dir, base_key))
                    K = _scale_K_for_size(_load_gt_intrinsics(scene_dir, base_key), base_image.shape[:2], gt.shape[:2])
                c2w_est = _transform_gt_c2w_to_est(rec["c2w_gt"], s, R_align, t_align)
                if w2c_init is None:
                    w2c_init = np.linalg.inv(np.asarray(c2w_est, np.float64))
                gs = _prep_frozen_gaussians(splats, edg, app_module, int(rec["base_frame"]), c2w_est, args)
                gt_t = torch.from_numpy(gt.astype(np.float32)).to(device)
                K_t = torch.from_numpy(np.asarray(K, np.float32)).to(device)
                lmask = _tto_loss_mask_tensor(rec, gt.shape[:2], args, device)
                batches.append((gs, gt_t, K_t, gt.shape[0], gt.shape[1], lmask))
            if not batches or w2c_init is None:
                continue
            w2c_ref, l0, l1 = _tto_refine_pose(
                w2c_init, batches, int(args.tto_cam_steps), int(args.tto_decay_start),
                float(args.tto_lr), float(args.tto_lr_final), float(args.tto_alpha_th), args, device,
            )
            rot_d, tr_d = _pose_delta(w2c_init, w2c_ref)
            print(f"[tto-cam] group {gi}: {len(batches)} samples, loss {l0:.3f}->{l1:.3f}, "
                  f"delta rot {rot_d:.3f}deg trans {tr_d:.4f}")
            group_pose[gi] = w2c_ref
            del batches
            torch.cuda.empty_cache()

    # --novel-fixed-cam: place ALL views at this est training camera (pose-refined) with its training K,
    # bypassing Sim3/TTO. For the fixed-v000 protocol the novel viewpoint IS a training camera, so t=0
    # reproduces the training render (PSNR -> training quality).
    fixed_c2w = None
    fixed_K = None
    if int(getattr(args, "novel_fixed_cam", -1)) >= 0:
        fc = int(args.novel_fixed_cam)
        c2w_fc, K_fc, _, _, _ = _load_camera_and_gt_intrinsics(scene_dir, fc, args.camera_source)
        fixed_c2w = _apply_pose_adjust(c2w_fc, fc, pose_adjust, device)
        fixed_K = np.asarray(K_fc, dtype=np.float64)
        print(f"[tto] novel-fixed-cam={fc}: placing ALL test views at est training camera {fc} "
              f"(pose-refined), training K; Sim3/TTO bypassed.")

    # Phase 1: place cameras (+ per-view TTO), render, collect.
    collected: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        image_path = Path(rec["image_path"])
        if not image_path.is_file():
            continue
        gt = _read_rgb(image_path)
        base_frame = int(rec["base_frame"])
        if fixed_c2w is not None:
            src_hw = (int(round(2 * fixed_K[1, 2])), int(round(2 * fixed_K[0, 2])))   # (H,W) from cx,cy
            K = _scale_K_for_size(fixed_K, src_hw, gt.shape[:2])                       # est frame-k training K
        else:
            K = _load_novel_intrinsics(novel_dir, rec["image_name"], gt.shape[:2])
            if K is None:
                base_key = _gt_key_for_frame(scene_dir, base_frame)
                base_image = _read_rgb(_find_image_path(scene_dir, base_key))
                K = _scale_K_for_size(_load_gt_intrinsics(scene_dir, base_key), base_image.shape[:2], gt.shape[:2])
        c2w_sim3 = np.asarray(_transform_gt_c2w_to_est(rec["c2w_gt"], s, R_align, t_align), np.float64)
        w2c_sim3 = np.linalg.inv(c2w_sim3)
        gi = gids[i]

        if fixed_c2w is not None:            # fixed est training camera -> reproduces training at t=0
            c2w_final = fixed_c2w
            rot_d, tr_d = 0.0, 0.0
        elif args.novel_align == "sim3":
            c2w_final = c2w_sim3
            rot_d, tr_d = 0.0, 0.0
        elif args.novel_align == "tto-cam":
            w2c_final = group_pose.get(gi, w2c_sim3)
            c2w_final = np.linalg.inv(w2c_final)
            rot_d, tr_d = _pose_delta(w2c_sim3, w2c_final)
        else:  # tto (MoSca parity)
            first_of_group = gi not in chained_w2c
            if first_of_group:
                w2c_init = w2c_sim3
                steps = int(args.tto_steps) * int(args.tto_first_factor)
                lr = float(args.tto_lr)
            else:
                w2c_init = chained_w2c[gi]
                steps = int(args.tto_steps)
                lr = float(args.tto_lr) * float(args.tto_chain_lr_factor)
            gs = _prep_frozen_gaussians(splats, edg, app_module, base_frame, np.linalg.inv(w2c_init), args)
            gt_t = torch.from_numpy(gt.astype(np.float32)).to(device)
            K_t = torch.from_numpy(np.asarray(K, np.float32)).to(device)
            lmask = _tto_loss_mask_tensor(rec, gt.shape[:2], args, device)
            w2c_ref, l0, l1 = _tto_refine_pose(
                w2c_init, [(gs, gt_t, K_t, gt.shape[0], gt.shape[1], lmask)], steps,
                int(args.tto_decay_start), lr, float(args.tto_lr_final) * (1.0 if first_of_group else float(args.tto_chain_lr_factor)),
                float(args.tto_alpha_th), args, device,
            )
            # Divergence guard: never accept a pose whose loss is worse than its candidates.
            # Candidates: refined (l1), the init it started from (l0), and the sim3 placement.
            cand = [(l1, w2c_ref), (l0, w2c_init)]
            if not first_of_group:
                cand.append((_pose_loss(gs, w2c_sim3, gt_t, K_t, gt.shape[0], gt.shape[1],
                                        lmask, float(args.tto_alpha_th), args, device), w2c_sim3))
            best_l, w2c_best = min(cand, key=lambda x: x[0])
            if w2c_best is not w2c_ref:
                print(f"[tto-guard] view {rec['novel_id']:4d}: refined loss {l1:.3f} worse than "
                      f"candidate {best_l:.3f}; keeping the better pose")
            chained_w2c[gi] = w2c_best
            c2w_final = np.linalg.inv(w2c_best)
            rot_d, tr_d = _pose_delta(w2c_sim3, w2c_best)
            if first_of_group or (i % 50 == 0):
                print(f"[tto] view {rec['novel_id']:4d} (group {gi}{' FIRST' if first_of_group else ''}): "
                      f"loss {l0:.3f}->{l1:.3f}, delta-vs-sim3 rot {rot_d:.3f}deg trans {tr_d:.4f}")

        _, alpha, pred = _render_depth(
            splats, edg, app_module, base_frame, c2w_final, np.asarray(K, np.float32),
            gt.shape[0], gt.shape[1], args, app_embed_mode=args.novel_app,
        )
        cov_mask = None
        if args.use_covisible_mask and rec.get("depth_path"):
            cov_path = Path(rec["depth_path"])
            if cov_path.is_file():
                cov = cv2.imread(str(cov_path), cv2.IMREAD_GRAYSCALE)
                if cov is not None:
                    if cov.shape[:2] != alpha.shape[:2]:
                        cov = cv2.resize(cov, (alpha.shape[1], alpha.shape[0]), interpolation=cv2.INTER_NEAREST)
                    cov_mask = cov > 127
                    alpha = alpha * cov_mask.astype(alpha.dtype)
        if pred.shape[:2] != gt.shape[:2]:
            continue
        valid = _metric_valid_mask(pred, gt, alpha, args.alpha_threshold, args.metric_mask)
        pose_rows.append({"novel_id": int(rec["novel_id"]), "group": int(gi),
                          "tto_rot_deg": rot_d, "tto_trans": tr_d})
        collected.append({
            "novel_id": int(rec["novel_id"]), "base_frame": base_frame, "gt_image": str(image_path),
            "pred": pred.astype(np.float32), "gt": gt.astype(np.float32),
            "alpha": alpha.astype(np.float32), "valid": valid, "cov_mask": cov_mask,
        })

    if not collected:
        return [], None

    gain, bias = _fit_color_affine([(c["pred"], c["gt"], c["valid"]) for c in collected], args.color_calib)

    # Phase 2: score + save (identical to estimate_ours_render).
    rows: list[dict[str, Any]] = []
    video_frames: dict[int, list[np.ndarray]] = {}
    for c, prow in zip(collected, pose_rows):
        pred = c["pred"]
        if args.color_calib == "per-view":
            g_i, b_i = _fit_color_affine([(pred, c["gt"], c["valid"])], "scene")
            pred = _apply_color_affine(pred, g_i, b_i)
        elif args.color_calib == "scene":
            pred = _apply_color_affine(pred, gain, bias)
        metric = _eval_rgb_metric(
            pred, c["gt"], c["alpha"], device, ssim_metric, lpips_metric, args.alpha_threshold,
            args.metric_mask, False, cov_mask=c.get("cov_mask"),
        )
        if metric is None:
            continue
        novel_id = c["novel_id"]
        rows.append({"novel_id": novel_id, "base_frame": c["base_frame"], "gt_image": c["gt_image"],
                     **prow, **metric})
        video_frames.setdefault(int(prow["group"]), []).append(
            (np.clip(pred, 0.0, 1.0) * 255).astype(np.uint8)
        )
        if args.save_images or novel_id < 5:
            cv2.imwrite(
                str(novel_out / f"novel{novel_id:06d}_pred.png"),
                cv2.cvtColor((np.clip(pred, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(str(novel_out / f"novel{novel_id:06d}_alpha.png"), (np.clip(c["alpha"], 0.0, 1.0) * 255).astype(np.uint8))
            _write_rgb_preview(novel_out / f"novel{novel_id:06d}_compare.png", c["gt"], pred, c["alpha"], args.alpha_threshold)

    # Per-frame TEST PSNR after alignment (v000_t000 == training frame 0, so t=0 should be highest).
    print(f"[tto] per-frame test PSNR (novel-align={args.novel_align}, color-calib={args.color_calib}):")
    for r in sorted(rows, key=lambda x: int(x["novel_id"])):
        print(f"[tto]   view {int(r['novel_id']):3d} (base_frame {int(r['base_frame']):3d}): "
              f"psnr={r.get('psnr', float('nan')):6.2f}  ssim={r.get('ssim', float('nan')):.3f}  "
              f"lpips={r.get('lpips', float('nan')):.3f}")

    # One render video per physical test camera (MoSca's tto_test_cam{i}.mp4 convention).
    for gi, frames in sorted(video_frames.items()):
        h, w = frames[0].shape[:2]
        vw = cv2.VideoWriter(str(out_dir / f"novel_cam{gi}.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (w, h))
        for fr in frames:
            vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        vw.release()
        print(f"[tto] wrote {out_dir / f'novel_cam{gi}.mp4'} ({len(frames)} frames)")

    rot_all = [r["tto_rot_deg"] for r in pose_rows]
    tr_all = [r["tto_trans"] for r in pose_rows]
    meta = {
        "novel_dir": str(novel_dir),
        "novel_align": args.novel_align,
        "novel_alignment_scale": float(s),
        "sim3_init_method": align_method,
        "camera_groups": int(len(set(gids))),
        "color_calib": args.color_calib,
        "color_calib_gain": [float(x) for x in gain],
        "color_calib_bias": [float(x) for x in bias],
        "tto_params": {
            "steps": int(args.tto_steps), "decay_start": int(args.tto_decay_start),
            "lr": float(args.tto_lr), "lr_final": float(args.tto_lr_final),
            "alpha_th": float(args.tto_alpha_th), "loss_mask": args.tto_loss_mask,
            "first_factor": int(args.tto_first_factor),
            "chain_lr_factor": float(args.tto_chain_lr_factor),
            "cam_steps": int(args.tto_cam_steps), "cam_samples": int(args.tto_cam_samples),
        },
        "tto_rot_deg_mean": float(np.mean(rot_all)) if rot_all else None,
        "tto_rot_deg_max": float(np.max(rot_all)) if rot_all else None,
        "tto_trans_mean": float(np.mean(tr_all)) if tr_all else None,
        "tto_trans_max": float(np.max(tr_all)) if tr_all else None,
    }
    return rows, meta


def main() -> None:
    args = _parse_args()
    scene_dir = args.scene_dir.resolve()
    out_dir = (args.output_dir or (scene_dir / "outputs" / f"ours_render_eval_tto_{args.novel_align}")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    gaussian_paths = _resolve_gaussian_paths(scene_dir, args.gaussians)
    splats, edg, app_state, step = _load_gaussians(gaussian_paths, device)
    app_module = _build_app_module(app_state, splats, device)
    pose_adjust = _build_pose_adjust_module(_load_pose_adjust_state(gaussian_paths, device), device)
    result_dir = _infer_result_dir_from_gaussians(gaussian_paths)
    if args.sync_train_cfg and result_dir is not None:
        cfg = _load_yaml(result_dir / "cfg.yml")
        if cfg is not None:
            if "sh_degree" in cfg:
                args.sh_degree = int(cfg["sh_degree"])
            if "near_plane" in cfg:
                args.near_plane = float(cfg["near_plane"])
            if "far_plane" in cfg:
                args.far_plane = float(cfg["far_plane"])

    ssim_metric, lpips_metric = _build_old_metrics(device)
    rows, meta = _evaluate_novel_tto(
        scene_dir, splats, edg, app_module, args, out_dir, device, ssim_metric, lpips_metric,
        pose_adjust=pose_adjust,
    )
    result: dict[str, Any] = {
        "scene_dir": str(scene_dir),
        "gaussians": [str(p) for p in gaussian_paths],
        "step": int(step),
        "camera_source": args.camera_source,
        "metric_mask": args.metric_mask,
        "alpha_threshold": float(args.alpha_threshold),
        "uses_app_module": app_module is not None,
        "uses_edg_warp": edg is not None and not args.no_edg_warp,
        "novel": _summarize(rows),
    }
    if meta is not None:
        result["novel"].update(meta)
    json_out = (args.output_json or (out_dir / "metrics_tto.json")).resolve()
    json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[tto] saved summary to {json_out}")
    summary = result["novel"]
    if summary["views"] > 0:
        print(
            f"[tto] novel-align={args.novel_align} views={summary['views']} "
            f"mPSNR={summary['mPSNR']:.4f} mSSIM={summary['mSSIM']:.4f} mLPIPS={summary['mLPIPS']:.4f}"
            + (f" | TTO delta rot mean {summary['tto_rot_deg_mean']:.3f}deg max {summary['tto_rot_deg_max']:.3f}deg"
               if summary.get("tto_rot_deg_mean") is not None else "")
        )


if __name__ == "__main__":
    main()
