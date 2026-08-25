#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

try:
    import imageio.v2 as imageio
except ModuleNotFoundError:
    imageio = None

from eval_depth import _prefer_npy_depth_path, _read_depth
from eval_depth_depthanything import _c2w_from_gt, _load_gt, _scale_K

REPLICA_RENDER_DEPTH_SCALE = 65535.0 * 0.1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Prepare Syn4D scene inputs for deps/SpaTrackerV2/inference.py.")
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--video-name", type=str, default=None)
    parser.add_argument("--gt-scale", type=float, default=REPLICA_RENDER_DEPTH_SCALE)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-side", type=int, default=0, help="Resize frames before SpaTracker; <=0 keeps original.")
    parser.add_argument("--write-rgbd", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--write-mp4", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-masks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--write-gt-camera",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save GT intrinsics/extrinsics scaled to the prepared RGB frames for RGB SpaTracker inference.",
    )
    return parser.parse_args()


def _image_files(scene_dir: Path) -> list[Path]:
    return sorted(
        p for p in (scene_dir / "images").iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


def _resize_rgb_depth(rgb: np.ndarray, depth: np.ndarray, max_side: int) -> tuple[np.ndarray, np.ndarray]:
    if max_side <= 0:
        return rgb, depth
    h, w = rgb.shape[:2]
    scale = min(float(max_side) / max(h, w), 1.0)
    if scale >= 1.0:
        return rgb, depth
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    depth = cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST)
    return rgb, depth


def _write_mp4(path: Path, frames_rgb: list[np.ndarray], fps: int) -> None:
    if imageio is not None:
        try:
            imageio.mimsave(path, frames_rgb, fps=int(fps))
            return
        except Exception as exc:
            print(f"[spatracker-prepare] imageio mp4 writer failed ({exc}); falling back to OpenCV.")
    if not frames_rgb:
        raise ValueError("No frames to write.")
    h, w = frames_rgb[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open OpenCV VideoWriter for {path}")
    try:
        for frame in frames_rgb:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def main() -> None:
    args = _parse_args()
    scene_dir = args.scene_dir.resolve()
    out_dir = (args.output_dir or (scene_dir / "outputs" / "spatracker_input")).resolve()
    video_name = args.video_name or scene_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = _load_gt(scene_dir) if (args.write_rgbd or args.write_gt_camera) else None
    image_paths = _image_files(scene_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {scene_dir / 'images'}")

    rgbs = []
    depths = []
    intrinsics = []
    extrinsics = []
    frame_ids = []
    for idx, image_path in enumerate(image_paths):
        rgb_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        if args.write_rgbd:
            assert gt is not None
            depth_path = _prefer_npy_depth_path(scene_dir / "depths" / f"depth{idx:06d}.png")
            if not depth_path.is_file():
                depth_path = _prefer_npy_depth_path(scene_dir / "depths" / image_path.with_suffix(".png").name.replace("frame", "depth"))
            depth = _read_depth(depth_path, args.gt_scale)
            rgb, depth = _resize_rgb_depth(rgb, depth, int(args.max_side))

            info = gt.get(image_path.name) or gt.get(image_path.stem) or gt.get(f"frame{idx:06d}.png")
            if info is None:
                by_id = [v for v in gt.values() if int(v["id"]) == idx]
                if not by_id:
                    raise KeyError(f"No GT camera found for {image_path.name}")
                info = by_id[0]
            src_size = (int(info.get("width", rgb_bgr.shape[1])), int(info.get("height", rgb_bgr.shape[0])))
            K = _scale_K(np.asarray(info["K"], dtype=np.float64), src_size, (rgb.shape[1], rgb.shape[0]))
            c2w = _c2w_from_gt(info)
            depths.append(depth.astype(np.float32))
            intrinsics.append(K.astype(np.float32))
            extrinsics.append(np.linalg.inv(c2w).astype(np.float32))
        else:
            rgb, _ = _resize_rgb_depth(rgb, np.zeros(rgb.shape[:2], dtype=np.float32), int(args.max_side))
            if args.write_gt_camera:
                assert gt is not None
                info = gt.get(image_path.name) or gt.get(image_path.stem) or gt.get(f"frame{idx:06d}.png")
                if info is None:
                    by_id = [v for v in gt.values() if int(v["id"]) == idx]
                    if not by_id:
                        raise KeyError(f"No GT camera found for {image_path.name}")
                    info = by_id[0]
                src_size = (int(info.get("width", rgb_bgr.shape[1])), int(info.get("height", rgb_bgr.shape[0])))
                K = _scale_K(np.asarray(info["K"], dtype=np.float64), src_size, (rgb.shape[1], rgb.shape[0]))
                c2w = _c2w_from_gt(info)
                intrinsics.append(K.astype(np.float32))
                extrinsics.append(np.linalg.inv(c2w).astype(np.float32))

        rgbs.append(rgb.astype(np.float32) / 255.0)
        frame_ids.append(idx)

    video = np.stack(rgbs, axis=0)

    if args.write_rgbd:
        depths_arr = np.stack(depths, axis=0)
        intrinsics_arr = np.stack(intrinsics, axis=0)
        extrinsics_arr = np.stack(extrinsics, axis=0)
        np.savez_compressed(
            out_dir / f"{video_name}.npz",
            video=video.transpose(0, 3, 1, 2),
            depths=depths_arr,
            intrinsics=intrinsics_arr,
            extrinsics=extrinsics_arr,
            frame_ids=np.asarray(frame_ids, dtype=np.int64),
        )
    if args.write_gt_camera:
        np.savez_compressed(
            out_dir / f"{video_name}_gt_camera.npz",
            intrinsics=np.stack(intrinsics, axis=0).astype(np.float32),
            extrinsics=np.stack(extrinsics, axis=0).astype(np.float32),
            frame_ids=np.asarray(frame_ids, dtype=np.int64),
            image_size=np.asarray([video.shape[2], video.shape[1]], dtype=np.int64),
        )
    if args.write_mp4:
        _write_mp4(out_dir / f"{video_name}.mp4", [(np.clip(v, 0, 1) * 255).astype(np.uint8) for v in video], int(args.fps))
    if args.write_masks:
        mask_src_dir = scene_dir / "masks"
        mask_out_dir = out_dir / "masks"
        if mask_src_dir.is_dir():
            mask_out_dir.mkdir(parents=True, exist_ok=True)
            for idx, image_path in enumerate(image_paths):
                candidates = [
                    mask_src_dir / image_path.name,
                    mask_src_dir / f"frame{idx:06d}.png",
                    mask_src_dir / f"mask{idx:06d}.png",
                    mask_src_dir / f"{idx:06d}.png",
                ]
                src = next((p for p in candidates if p.is_file()), None)
                if src is None:
                    continue
                mask = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
                if mask is None:
                    shutil.copy2(src, mask_out_dir / f"frame{idx:06d}.png")
                    continue
                if int(args.max_side) > 0:
                    h, w = mask.shape[:2]
                    scale = min(float(args.max_side) / max(h, w), 1.0)
                    if scale < 1.0:
                        size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
                        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(mask_out_dir / f"frame{idx:06d}.png"), mask)
    print(f"[spatracker-prepare] wrote inputs to {out_dir}")


if __name__ == "__main__":
    main()
