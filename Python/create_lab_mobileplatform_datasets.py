#!/usr/bin/env python3
"""Create 100-frame Lab-MobilePlatform datasets from MOV captures.

Each output dataset contains:
  images/frame000000.jpg ...
  colmap/sparse/cameras.txt
  colmap/sparse/images.txt
  colmap/sparse/points3D.txt
  metadata.json

The camera intrinsics in cameras.txt are placeholders. Runtime intrinsics should
come from outputs/depth_anything_3/intrinsics.npy via readgt_lab_mobileplatform.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


IMAGE_EXTENSIONS = {".mov", ".MOV"}


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    frames: int
    fps: float


def _read_pose_csv(path: Path, value_names: list[str]) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        skiprows=3,
        header=None,
        usecols=[0, 2, 3, 4],
        names=["frame", *value_names],
        encoding="utf-8-sig",
    )
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No pose rows parsed from {path}")
    return df


def load_capture_poses(root: Path, capture_name: str) -> tuple[np.ndarray, np.ndarray]:
    position = _read_pose_csv(root / f"{capture_name}_position.csv", ["tx", "ty", "tz"])
    rotation = _read_pose_csv(root / f"{capture_name}_rotation.csv", ["rx", "ry", "rz"])
    n = min(len(position), len(rotation))
    if n == 0:
        raise ValueError(f"No aligned pose rows for {capture_name}")

    centers_m = position.loc[: n - 1, ["tx", "ty", "tz"]].to_numpy(np.float64) / 1000.0
    euler_xyz_deg = rotation.loc[: n - 1, ["rx", "ry", "rz"]].to_numpy(np.float64)

    # CSV says "Euler Angle XYZ". Treat these as camera-to-world XYZ rotations.
    c2w_rot = Rotation.from_euler("XYZ", euler_xyz_deg, degrees=True).as_matrix()
    return centers_m, c2w_rot


def video_info(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return VideoInfo(width=width, height=height, frames=frames, fps=fps)


def rotmat_to_colmap_qvec(rot: np.ndarray) -> np.ndarray:
    q_xyzw = Rotation.from_matrix(rot).as_quat()
    qvec = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)
    if qvec[0] < 0:
        qvec *= -1.0
    return qvec


def write_cameras_txt(path: Path, width: int, height: int) -> None:
    focal = float(max(width, height))
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0
    text = (
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        "# Number of cameras: 1\n"
        f"1 PINHOLE {width} {height} {focal:.12g} {focal:.12g} {cx:.12g} {cy:.12g}\n"
    )
    path.write_text(text)


def write_points3d_txt(path: Path) -> None:
    path.write_text(
        "# 3D point list with one line of data per point:\n"
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        "# Number of points: 0, mean track length: 0\n"
    )


def write_images_txt(
    path: Path,
    image_names: list[str],
    centers_m: np.ndarray,
    c2w_rot: np.ndarray,
) -> None:
    lines = [
        "# Image list with two lines of data per image:\n",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)\n",
        f"# Number of images: {len(image_names)}, mean observations per image: 0\n",
    ]

    for i, name in enumerate(image_names):
        image_id = i + 1
        r_w2c = c2w_rot[i].T
        t_w2c = -r_w2c @ centers_m[i]
        qvec = rotmat_to_colmap_qvec(r_w2c)
        values = [
            str(image_id),
            *(f"{v:.17g}" for v in qvec),
            *(f"{v:.17g}" for v in t_w2c),
            "1",
            name,
        ]
        lines.append(" ".join(values) + "\n")
        lines.append("\n")

    path.write_text("".join(lines))


def clip_starts(frame_count: int, clip_len: int, clips_per_video: int) -> list[int]:
    if frame_count < clip_len:
        raise ValueError(f"Video has only {frame_count} frames, need {clip_len}")
    if clips_per_video == 1:
        return [0]
    max_start = frame_count - clip_len
    starts = np.linspace(0, max_start, clips_per_video)
    return [int(round(x)) for x in starts]


def map_video_frames_to_pose_rows(frame_indices: np.ndarray, video_frames: int, pose_rows: int) -> np.ndarray:
    if video_frames <= 1:
        return np.zeros_like(frame_indices)
    mapped = np.rint(frame_indices * (pose_rows - 1) / (video_frames - 1)).astype(int)
    return np.clip(mapped, 0, pose_rows - 1)


def jpeg_quality_to_ffmpeg_qscale(jpeg_quality: int) -> int:
    quality = int(np.clip(jpeg_quality, 1, 100))
    return int(round(np.interp(quality, [1, 100], [31, 2])))


def _select_filter(frame_indices: np.ndarray) -> str:
    if frame_indices.size == 0:
        raise ValueError("No frame indices requested")
    start = int(frame_indices[0])
    last = int(frame_indices[-1])
    if frame_indices.size == 1:
        return f"select='eq(n\\,{start})'"

    diffs = np.diff(frame_indices)
    if not np.all(diffs == diffs[0]):
        raise ValueError("ffmpeg extractor expects evenly spaced frame indices")

    stride = int(diffs[0])
    if stride == 1:
        return f"select='between(n\\,{start}\\,{last})'"
    return f"select='between(n\\,{start}\\,{last})*not(mod(n-{start}\\,{stride}))'"


def extract_frames_ffmpeg(
    video_path: Path,
    frame_indices: np.ndarray,
    images_dir: Path,
    jpeg_quality: int,
    tonemap_hdr: bool,
) -> list[str]:
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg not found on PATH")

    select = _select_filter(frame_indices)
    filters = [select]
    if tonemap_hdr:
        # iPhone MOVs here are 10-bit HLG/BT.2020. Convert to ordinary SDR JPEGs.
        filters.extend(
            [
                "zscale=t=linear:npl=100",
                "format=gbrpf32le",
                "zscale=p=bt709",
                "tonemap=tonemap=hable:desat=0",
                "zscale=t=bt709:m=bt709:r=tv",
                "format=yuv420p",
            ]
        )
    vf = ",".join(filters)

    output_pattern = images_dir / "frame%06d.jpg"
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-vsync",
        "0",
        "-frames:v",
        str(len(frame_indices)),
        "-q:v",
        str(jpeg_quality_to_ffmpeg_qscale(jpeg_quality)),
        "-start_number",
        "0",
        str(output_pattern),
    ]
    subprocess.run(cmd, check=True)

    image_names = [f"frame{i:06d}.jpg" for i in range(len(frame_indices))]
    missing = [name for name in image_names if not (images_dir / name).is_file()]
    if missing:
        raise RuntimeError(
            f"ffmpeg extracted {len(image_names) - len(missing)}/{len(image_names)} frames; "
            f"first missing: {missing[0]}"
        )
    return image_names


def extract_frames_opencv(video_path: Path, frame_indices: np.ndarray, images_dir: Path, jpeg_quality: int) -> list[str]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    image_names: list[str] = []
    for out_i, frame_i in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_i))
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"Could not read frame {frame_i} from {video_path}")
        name = f"frame{out_i:06d}.jpg"
        out_path = images_dir / name
        cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        image_names.append(name)

    cap.release()
    return image_names


def extract_frames(
    video_path: Path,
    frame_indices: np.ndarray,
    images_dir: Path,
    jpeg_quality: int,
    extractor: str,
    tonemap_hdr: bool,
) -> list[str]:
    if extractor == "opencv":
        return extract_frames_opencv(video_path, frame_indices, images_dir, jpeg_quality)
    if extractor == "ffmpeg":
        return extract_frames_ffmpeg(video_path, frame_indices, images_dir, jpeg_quality, tonemap_hdr)
    raise ValueError(f"Unknown extractor: {extractor}")


def write_mp4_from_images(images_dir: Path, output_path: Path, fps: float) -> None:
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg not found on PATH")

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        f"{float(fps):.12g}",
        "-start_number",
        "0",
        "-i",
        str(images_dir / "frame%06d.jpg"),
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def create_dataset(
    root: Path,
    output_root: Path,
    video_path: Path,
    capture_name: str,
    clip_index: int,
    start_frame: int,
    clip_len: int,
    stride: int,
    jpeg_quality: int,
    overwrite: bool,
    extractor: str,
    tonemap_hdr: bool,
    write_mp4: bool,
    mp4_fps: float | None,
    mp4_name: str,
) -> Path:
    info = video_info(video_path)
    frame_indices = start_frame + np.arange(clip_len, dtype=int) * stride
    if frame_indices[-1] >= info.frames:
        raise ValueError(
            f"{capture_name} clip {clip_index}: last frame {frame_indices[-1]} "
            f"is outside video with {info.frames} frames"
        )

    centers_all, c2w_rot_all = load_capture_poses(root, capture_name)
    pose_rows = map_video_frames_to_pose_rows(frame_indices, info.frames, len(centers_all))

    dataset_dir = output_root / f"{capture_name}_clip{clip_index + 1:02d}"
    images_dir = dataset_dir / "images"
    sparse_dir = dataset_dir / "colmap" / "sparse"
    if dataset_dir.exists() and not overwrite:
        raise FileExistsError(f"Output exists, pass --overwrite to replace files: {dataset_dir}")
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    for old in images_dir.glob("*"):
        if old.is_file():
            old.unlink()

    image_names = extract_frames(video_path, frame_indices, images_dir, jpeg_quality, extractor, tonemap_hdr)
    preview_mp4_path = dataset_dir / mp4_name
    preview_fps = float(mp4_fps) if mp4_fps is not None else float(info.fps) / max(1, int(stride))
    if write_mp4:
        write_mp4_from_images(images_dir, preview_mp4_path, preview_fps)

    centers = centers_all[pose_rows]
    c2w_rot = c2w_rot_all[pose_rows]

    write_cameras_txt(sparse_dir / "cameras.txt", info.width, info.height)
    write_images_txt(sparse_dir / "images.txt", image_names, centers, c2w_rot)
    write_points3d_txt(sparse_dir / "points3D.txt")

    metadata = {
        "source_video": str(video_path),
        "capture": capture_name,
        "clip_index": clip_index + 1,
        "start_frame": int(start_frame),
        "stride": int(stride),
        "num_frames": int(clip_len),
        "video_num_frames": int(info.frames),
        "fps": info.fps,
        "width": info.width,
        "height": info.height,
        "pose_source": {
            "position_csv": str(root / f"{capture_name}_position.csv"),
            "rotation_csv": str(root / f"{capture_name}_rotation.csv"),
            "units": "meters",
            "rotation_convention": "Euler XYZ degrees interpreted as camera-to-world; COLMAP stores world-to-camera",
        },
        "frame_indices": frame_indices.tolist(),
        "pose_rows": pose_rows.tolist(),
        "extractor": extractor,
        "tonemap_hdr": bool(tonemap_hdr),
        "mp4": str(preview_mp4_path) if write_mp4 else None,
        "mp4_fps": preview_fps if write_mp4 else None,
        "intrinsics": "Run Depth Anything and save outputs/depth_anything_3/intrinsics.npy for this dataset.",
    }
    (dataset_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return dataset_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("../datasets/Lab-MobilePlatform"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--clip-len", type=int, default=100)
    parser.add_argument("--clips-per-video", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--no-mp4",
        action="store_true",
        help="Do not write a preview MP4 into each generated dataset root.",
    )
    parser.add_argument(
        "--mp4-fps",
        type=float,
        default=None,
        help="Preview MP4 frame rate. Defaults to source FPS divided by stride.",
    )
    parser.add_argument(
        "--mp4-name",
        default="video.mp4",
        help="Preview MP4 filename written in each dataset root.",
    )
    parser.add_argument(
        "--extractor",
        choices=["ffmpeg", "opencv"],
        default="ffmpeg",
        help="ffmpeg handles iPhone HDR/HLG color better; opencv is kept for debugging.",
    )
    parser.add_argument(
        "--no-tonemap-hdr",
        action="store_true",
        help="Disable ffmpeg HLG/BT.2020 to SDR/BT.709 tone mapping.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.root
    output_root = args.output_root or (root / "datasets_100f")
    output_root.mkdir(parents=True, exist_ok=True)

    videos = sorted(path for path in root.iterdir() if path.suffix in IMAGE_EXTENSIONS)
    if not videos:
        raise FileNotFoundError(f"No MOV videos found in {root}")

    created: list[Path] = []
    for video_path in videos:
        capture_name = video_path.stem
        info = video_info(video_path)
        effective_len = (args.clip_len - 1) * args.stride + 1
        for clip_index, start in enumerate(clip_starts(info.frames, effective_len, args.clips_per_video)):
            created.append(
                create_dataset(
                    root=root,
                    output_root=output_root,
                    video_path=video_path,
                    capture_name=capture_name,
                    clip_index=clip_index,
                    start_frame=start,
                    clip_len=args.clip_len,
                    stride=args.stride,
                    jpeg_quality=args.jpeg_quality,
                    overwrite=args.overwrite,
                    extractor=args.extractor,
                    tonemap_hdr=not args.no_tonemap_hdr,
                    write_mp4=not args.no_mp4,
                    mp4_fps=args.mp4_fps,
                    mp4_name=args.mp4_name,
                )
            )

    print("Created datasets:")
    for path in created:
        print(f"  {path}")


if __name__ == "__main__":
    main()
