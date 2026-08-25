#!/usr/bin/env python3
"""Create 100-frame TUM fr3 clips from the raw RGB-D folders.

The output layout matches the project TUM datasets:
  images/<rgb timestamp>.png
  depths/<rgb timestamp>.png
  masks/
  cameras/
  outputs/
  groundtruth.txt
  metadata.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TumEntry:
    timestamp: float
    path: str


@dataclass(frozen=True)
class ClipSpec:
    source: str
    name: str
    start_timestamp: float | None


CLIPS = [
    ClipSpec("fr3_sitting_static", "fr3_sitting_static_first100", None),
    ClipSpec("fr3_sitting_static", "fr3_sitting_static_5691_945997", 1341845691.945997),
    ClipSpec("fr3_sitting_xyz", "fr3_sitting_xyz_5831_369742", 1341845831.369742),
    ClipSpec("fr3_sitting_xyz", "fr3_sitting_xyz_5858_753674", 1341845858.753674),
    ClipSpec("fr3_walking_static", "fr3_walking_static_first100", None),
    ClipSpec("fr3_walking_static", "fr3_walking_static_6240_089841", 1341846240.089841),
]


def read_association_file(path: Path) -> list[TumEntry]:
    entries: list[TumEntry] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        timestamp, rel_path = line.split(maxsplit=1)
        entries.append(TumEntry(float(timestamp), rel_path))
    if not entries:
        raise ValueError(f"No entries parsed from {path}")
    return entries


def find_start_index(entries: list[TumEntry], start_timestamp: float | None) -> int:
    if start_timestamp is None:
        return 0
    for i, entry in enumerate(entries):
        if abs(entry.timestamp - start_timestamp) < 5e-7:
            return i
    nearest = min(entries, key=lambda entry: abs(entry.timestamp - start_timestamp))
    raise ValueError(
        f"Timestamp {start_timestamp:.6f} not found; nearest RGB timestamp is {nearest.timestamp:.6f}"
    )


def nearest_entry(entries: list[TumEntry], timestamp: float) -> TumEntry:
    return min(entries, key=lambda entry: abs(entry.timestamp - timestamp))


def copy_clip(
    tum_root: Path,
    spec: ClipSpec,
    clip_len: int,
    stride: int,
    overwrite: bool,
) -> Path:
    source_dir = tum_root / spec.source
    rgb_entries = read_association_file(source_dir / "rgb.txt")
    depth_entries = read_association_file(source_dir / "depth.txt")

    start_index = find_start_index(rgb_entries, spec.start_timestamp)
    rgb_indices = [start_index + i * stride for i in range(clip_len)]
    if rgb_indices[-1] >= len(rgb_entries):
        raise ValueError(
            f"{spec.name}: last RGB index {rgb_indices[-1]} exceeds "
            f"{len(rgb_entries)} entries in {source_dir / 'rgb.txt'}"
        )

    out_dir = tum_root / spec.name
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists, pass --overwrite: {out_dir}")
        shutil.rmtree(out_dir)

    images_dir = out_dir / "images"
    depths_dir = out_dir / "depths"
    for path in (images_dir, depths_dir, out_dir / "masks", out_dir / "cameras", out_dir / "outputs"):
        path.mkdir(parents=True, exist_ok=True)

    selected_rgb = [rgb_entries[i] for i in rgb_indices]
    selected_depth = [nearest_entry(depth_entries, entry.timestamp) for entry in selected_rgb]

    rgb_lines = ["# color images\n", f"# source: {source_dir}\n", "# timestamp filename\n"]
    depth_lines = ["# depth maps\n", f"# source: {source_dir}\n", "# timestamp filename\n"]
    associations = []

    for rgb_entry, depth_entry in zip(selected_rgb, selected_depth, strict=True):
        image_name = Path(rgb_entry.path).name
        depth_name = image_name
        shutil.copy2(source_dir / rgb_entry.path, images_dir / image_name)
        shutil.copy2(source_dir / depth_entry.path, depths_dir / depth_name)
        rgb_lines.append(f"{rgb_entry.timestamp:.6f} images/{image_name}\n")
        depth_lines.append(f"{rgb_entry.timestamp:.6f} depths/{depth_name}\n")
        associations.append(
            {
                "rgb_timestamp": rgb_entry.timestamp,
                "rgb_source": rgb_entry.path,
                "depth_timestamp": depth_entry.timestamp,
                "depth_source": depth_entry.path,
                "depth_time_delta": depth_entry.timestamp - rgb_entry.timestamp,
            }
        )

    shutil.copy2(source_dir / "groundtruth.txt", out_dir / "groundtruth.txt")
    (out_dir / "rgb.txt").write_text("".join(rgb_lines))
    (out_dir / "depth.txt").write_text("".join(depth_lines))
    metadata = {
        "source_dataset": str(source_dir),
        "start_timestamp": selected_rgb[0].timestamp,
        "start_rgb_index": start_index,
        "stride": stride,
        "num_frames": clip_len,
        "rgb_indices": rgb_indices,
        "associations": associations,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tum-root", type=Path, default=Path("../datasets/TUM"))
    parser.add_argument("--clip-len", type=int, default=100)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    created = [
        copy_clip(args.tum_root, spec, args.clip_len, args.stride, args.overwrite)
        for spec in CLIPS
    ]

    print("Created datasets:")
    for path in created:
        print(f"  {path}")


if __name__ == "__main__":
    main()
