#!/usr/bin/env python3
"""Visualize Lab-MobilePlatform camera positions as Open3D dots."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "datasets" / "Lab-MobilePlatform"


COLORS = np.array(
    [
        [0.90, 0.20, 0.18],
        [0.12, 0.47, 0.71],
        [0.20, 0.63, 0.17],
        [0.58, 0.40, 0.74],
        [1.00, 0.50, 0.05],
        [0.09, 0.75, 0.81],
    ],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read Lab-MobilePlatform position CSVs and draw 3D trajectories in Open3D."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Folder containing Capture*_position.csv files.",
    )
    parser.add_argument(
        "--captures",
        nargs="*",
        default=None,
        help="Optional capture names, e.g. Capture1 Capture3. Defaults to all position CSVs.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="Draw every Nth pose sample. Use 1 for all dots.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=5.0,
        help="Open3D render point size.",
    )
    parser.add_argument(
        "--unit-scale",
        type=float,
        default=0.001,
        help="Scale CSV positions by this value. Default converts mm to meters.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print loaded trajectory stats; do not open the Open3D viewer.",
    )
    return parser.parse_args()


def resolve_root(root: Path) -> Path:
    if root.is_dir():
        return root

    parent = root.parent if root.parent != Path("") else Path(".")
    matches = [path for path in parent.iterdir() if path.is_dir() and path.name.lower() == root.name.lower()]
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Could not find Lab-MobilePlatform folder: {root}")


def read_position_csv(path: Path, unit_scale: float) -> np.ndarray:
    df = pd.read_csv(
        path,
        skiprows=3,
        header=None,
        usecols=[2, 3, 4],
        names=["tx", "ty", "tz"],
        encoding="utf-8-sig",
    )
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No positions parsed from {path}")
    return df[["tx", "ty", "tz"]].to_numpy(np.float64) * float(unit_scale)


def discover_position_files(root: Path, captures: list[str] | None) -> list[Path]:
    if captures:
        files = [root / f"{capture}_position.csv" for capture in captures]
    else:
        files = sorted(root.glob("*_position.csv"))

    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing position CSV(s): " + ", ".join(map(str, missing)))
    if not files:
        raise FileNotFoundError(f"No *_position.csv files found in {root}")
    return files


def make_point_cloud(points: np.ndarray, color: np.ndarray):
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(np.tile(color.reshape(1, 3), (points.shape[0], 1)))
    return cloud


def main() -> None:
    args = parse_args()
    root = resolve_root(args.root)
    stride = max(1, int(args.stride))
    files = discover_position_files(root, args.captures)

    trajectories: list[tuple[str, np.ndarray]] = []
    for path in files:
        name = path.name.removesuffix("_position.csv")
        points = read_position_csv(path, args.unit_scale)
        points = points[::stride]
        trajectories.append((name, points))

        start = points[0]
        end = points[-1]
        dist = float(np.linalg.norm(end - start))
        print(
            f"{name}: {points.shape[0]} dots, "
            f"start={np.array2string(start, precision=4)}, "
            f"end={np.array2string(end, precision=4)}, "
            f"start_end_distance={dist:.4f}"
        )

    if args.summary_only:
        return

    import open3d as o3d

    geometries = []
    all_points = []
    for i, (_, points) in enumerate(trajectories):
        color = COLORS[i % len(COLORS)]
        geometries.append(make_point_cloud(points, color))
        all_points.append(points)

    stacked = np.concatenate(all_points, axis=0)
    extent = np.ptp(stacked, axis=0)
    axis_size = max(float(np.max(extent)) * 0.15, 0.2)
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size))

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Lab-MobilePlatform trajectories")
    for geom in geometries:
        vis.add_geometry(geom)
    render = vis.get_render_option()
    render.point_size = float(args.point_size)
    render.background_color = np.array([0.02, 0.02, 0.02])
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
