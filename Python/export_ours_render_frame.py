#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from estimate_ours_render import (
    _alignment_est_to_gt,
    _apply_pose_adjust,
    _build_app_module,
    _build_pose_adjust_module,
    _find_image_path,
    _infer_result_dir_from_gaussians,
    _load_pose_adjust_state,
    _load_yaml,
    _read_rgb,
    _resize_for_render_scale,
    _scale_K_for_size,
)
from eval_depth_gaussian_ours import (
    _gt_c2w_from_info,
    _gt_key_for_frame,
    _load_camera_and_gt_intrinsics,
    _load_gaussians,
    _load_gt,
    _load_gt_intrinsics,
    _load_novel_metadata,
    _render_depth,
    _resolve_gaussian_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Export one ours RGB render frame for render comparison visualization.")
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "novel"], default="train")
    parser.add_argument("--frame-index", type=int, required=True)
    parser.add_argument("--novel-id", type=int, default=0)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--gaussians", type=Path, default=None)
    parser.add_argument("--camera-source", choices=["colmap", "xm"], default="colmap")
    parser.add_argument("--render-scale", type=float, default=1.0)
    parser.add_argument("--alpha-threshold", type=float, default=0.8)
    parser.add_argument("--near-plane", type=float, default=0.01)
    parser.add_argument("--far-plane", type=float, default=1e10)
    parser.add_argument("--sh-degree", type=int, default=2)
    parser.add_argument("--sync-train-cfg", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-edg-warp", action="store_true")
    return parser.parse_args()


def _novel_c2w_from_base(
    novel_c2w_gt: np.ndarray,
    gt_base_c2w: np.ndarray,
    est_base_c2w: np.ndarray,
    offset_scale: float,
) -> np.ndarray:
    delta_cam = gt_base_c2w[:3, :3].T @ (novel_c2w_gt[:3, 3] - gt_base_c2w[:3, 3])
    rel_rot = gt_base_c2w[:3, :3].T @ novel_c2w_gt[:3, :3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = est_base_c2w[:3, :3] @ rel_rot
    out[:3, 3] = est_base_c2w[:3, 3] + float(offset_scale) * (est_base_c2w[:3, :3] @ delta_cam)
    return out


def main() -> None:
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    gaussian_paths = _resolve_gaussian_paths(scene_dir, args.gaussians)
    splats, edg, app_state, step = _load_gaussians(gaussian_paths, device)
    app_module = _build_app_module(app_state, splats, device)
    novel_render_defaults = {
        "sh_degree": int(args.sh_degree),
        "near_plane": float(args.near_plane),
        "far_plane": float(args.far_plane),
        "render_scale": float(args.render_scale),
    }

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
            if "render_scale" in cfg:
                args.render_scale = float(cfg["render_scale"])

    if args.split == "novel":
        # estimate_ours_render.py intentionally evaluates novel views with the
        # pre-cfg-sync convention from estimate_our_render_old.py.
        args.sh_degree = novel_render_defaults["sh_degree"]
        args.near_plane = novel_render_defaults["near_plane"]
        args.far_plane = novel_render_defaults["far_plane"]
        args.render_scale = novel_render_defaults["render_scale"]

    pose_state = _load_pose_adjust_state(gaussian_paths, device)
    pose_adjust = _build_pose_adjust_module(pose_state, device)

    if args.split == "train":
        c2w, K, camera_source, _sparse_dir, image_name = _load_camera_and_gt_intrinsics(
            scene_dir,
            int(args.frame_index),
            args.camera_source,
        )
        render_frame = int(args.frame_index)
        c2w = _apply_pose_adjust(c2w, render_frame, pose_adjust, device)
        gt = _read_rgb(_find_image_path(scene_dir, image_name))
        gt, K = _resize_for_render_scale(gt, K, float(args.render_scale))
        novel_id = -1
    else:
        novel_dir = scene_dir / "novel"
        records = {int(rec["novel_id"]): rec for rec in _load_novel_metadata(scene_dir, novel_dir)}
        if int(args.novel_id) not in records:
            raise KeyError(f"novel-id {args.novel_id} not found in {novel_dir / 'metadata.txt'}")
        rec = records[int(args.novel_id)]
        render_frame = int(rec["base_frame"])
        gt = _read_rgb(Path(rec["image_path"]))
        base_key = _gt_key_for_frame(scene_dir, render_frame)
        base_image = _read_rgb(_find_image_path(scene_dir, base_key))
        K = _scale_K_for_size(_load_gt_intrinsics(scene_dir, base_key), base_image.shape[:2], gt.shape[:2])
        s, R_align, t_align, _align_method = _alignment_est_to_gt(scene_dir, args.camera_source)
        base_c2w, _base_K, camera_source, _sparse_dir, _base_image_name = _load_camera_and_gt_intrinsics(
            scene_dir,
            render_frame,
            args.camera_source,
        )
        base_c2w = _apply_pose_adjust(base_c2w, render_frame, pose_adjust, device)
        gt_base_c2w = _gt_c2w_from_info(_load_gt(scene_dir)[base_key])
        c2w = _novel_c2w_from_base(
            rec["c2w_gt"],
            gt_base_c2w,
            base_c2w,
            1.0 / max(float(s), 1e-12),
        )
        camera_source = args.camera_source
        image_name = Path(rec["image_path"]).name
        novel_id = int(rec["novel_id"])

    _depth, alpha, pred = _render_depth(
        splats,
        edg,
        app_module,
        render_frame,
        c2w,
        K,
        gt.shape[0],
        gt.shape[1],
        args,
    )
    valid = np.isfinite(pred).all(axis=-1) & np.isfinite(gt).all(axis=-1) & np.isfinite(alpha)
    valid &= alpha > float(args.alpha_threshold)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        pred=pred.astype(np.float32),
        gt=gt.astype(np.float32),
        alpha=alpha.astype(np.float32),
        valid=valid.astype(bool),
        split=np.array(args.split),
        frame_index=np.array(int(args.frame_index), dtype=np.int64),
        render_frame=np.array(int(render_frame), dtype=np.int64),
        novel_id=np.array(int(novel_id), dtype=np.int64),
        step=np.array(int(step), dtype=np.int64),
        camera_source=np.array(camera_source),
        image_name=np.array(image_name),
        render_convention=np.array("base_relative_novel_pose" if args.split == "novel" else "current_trainer_synced"),
        method=np.array("ours"),
    )
    print(f"[ours-render-frame] wrote {args.output_npz}")


if __name__ == "__main__":
    main()
