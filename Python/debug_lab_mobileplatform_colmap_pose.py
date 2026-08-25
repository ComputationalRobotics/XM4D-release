#!/usr/bin/env python3
"""Compare Lab-MobilePlatform CSV camera centers with stored COLMAP poses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from create_lab_mobileplatform_datasets import load_capture_poses  # noqa: E402
from utils.readgt_colmap import load_colmap_gt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scene_dir",
        type=Path,
        help="Generated clip directory, e.g. datasets/Lab-MobilePlatform/datasets_100f/Capture1_clip01",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("datasets/Lab-MobilePlatform"),
        help="Directory containing Capture*_position.csv and Capture*_rotation.csv.",
    )
    parser.add_argument("--limit", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_dir = args.scene_dir
    metadata = json.loads((scene_dir / "metadata.json").read_text())

    centers_all, _ = load_capture_poses(args.source_root, metadata["capture"])
    print(centers_all)
    pose_rows = np.asarray(metadata["pose_rows"], dtype=int)
    expected_centers = centers_all[pose_rows]

    gt = load_colmap_gt(str(scene_dir / "colmap"))
    names = sorted(gt.keys(), key=lambda name: gt[name]["id"])
    recovered_centers = []

    print("COLMAP stores t_w2c, not camera position. Camera center is C = -R.T @ t.\n")
    for i, name in enumerate(names):
        R_w2c = np.asarray(gt[name]["R"], dtype=np.float64)
        t_w2c = np.asarray(gt[name]["t"], dtype=np.float64)
        center = -R_w2c.T @ t_w2c
        recovered_centers.append(center)

        if i < args.limit:
            print(name)
            print(f"  csv camera center:   {expected_centers[i]}")
            print(f"  colmap t_w2c:        {t_w2c}")
            print(f"  recovered center:    {center}")
            print(f"  center error norm:   {np.linalg.norm(center - expected_centers[i]):.6e}")

    recovered_centers = np.asarray(recovered_centers)
    errors = np.linalg.norm(recovered_centers - expected_centers[: len(recovered_centers)], axis=1)
    print(f"\nframes checked: {len(recovered_centers)}")
    print(f"max center error: {errors.max():.6e}")
    print(f"mean center error: {errors.mean():.6e}")


if __name__ == "__main__":
    main()
