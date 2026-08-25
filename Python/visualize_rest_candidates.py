#!/usr/bin/env python3
"""Visualize saved rest-frame candidate 3D points from EDG_old outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

import utils.definitions


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize EDG rest-frame candidate 3D points.")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(utils.definitions.output_dir) / "results_all.npz",
        help="Path to results_all.npz produced by EDG_old.py",
    )
    parser.add_argument(
        "--start-segment",
        type=int,
        default=0,
        help="Segment index to show first.",
    )
    parser.add_argument(
        "--window-name",
        type=str,
        default="EDG Rest Candidates",
        help="Open3D window title.",
    )
    return parser.parse_args()


def _safe_axis_length(points: np.ndarray, fallback: float = 0.05):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 3:
        return fallback, fallback
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if points.shape[0] < 2:
        return fallback, fallback
    bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(points))
    diag = float(np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound()))
    if not np.isfinite(diag) or diag <= 1e-12:
        return fallback, fallback
    return diag, 0.02 * diag


def _load_rest_candidates(results_path: Path):
    if not results_path.exists():
        raise FileNotFoundError(f"results file not found: {results_path}")
    data = np.load(str(results_path), allow_pickle=True)
    if "rest_candidates" not in data:
        raise KeyError(f"'rest_candidates' not found in {results_path}")
    rest_candidates = list(data["rest_candidates"])
    if len(rest_candidates) == 0:
        raise RuntimeError("No rest_candidates entries found.")
    return rest_candidates


def main() -> None:
    args = _parse_args()
    rest_candidates = _load_rest_candidates(args.results)
    start_idx = int(np.clip(args.start_segment, 0, len(rest_candidates) - 1))
    state = {"idx": start_idx}

    first = rest_candidates[start_idx].item() if hasattr(rest_candidates[start_idx], "item") else rest_candidates[start_idx]
    pts0 = np.asarray(first["points_world"], dtype=np.float64)
    edg_ids0 = set(np.asarray(first["edg_global_ids"], dtype=int).tolist())
    gids0 = np.asarray(first["global_ids"], dtype=int)
    is_edg0 = np.array([int(gid) in edg_ids0 for gid in gids0], dtype=bool)

    diag, _ = _safe_axis_length(pts0)
    pt_size = 3.0 if diag <= 0 else float(np.clip(800.0 / (diag * 1000.0 + 1e-6), 2.0, 8.0))

    pcd_all = o3d.geometry.PointCloud()
    pcd_edg = o3d.geometry.PointCloud()
    pcd_all.paint_uniform_color([0.2, 0.5, 1.0])
    pcd_edg.paint_uniform_color([1.0, 0.2, 0.2])

    def _entry(idx: int):
        entry = rest_candidates[idx]
        return entry.item() if hasattr(entry, "item") else entry

    def _refresh(vis, idx: int):
        entry = _entry(idx)
        pts = np.asarray(entry["points_world"], dtype=np.float64)
        gids = np.asarray(entry["global_ids"], dtype=int)
        edg_ids = set(np.asarray(entry["edg_global_ids"], dtype=int).tolist())
        is_edg = np.array([int(gid) in edg_ids for gid in gids], dtype=bool)
        candidate_global_ids = np.asarray(entry.get("candidate_global_ids", gids), dtype=int)
        candidate_visible_rest_mask = np.asarray(
            entry.get("candidate_visible_rest_mask", np.ones(candidate_global_ids.shape[0], dtype=bool)),
            dtype=bool,
        )

        pcd_all.points = o3d.utility.Vector3dVector(pts)
        pcd_edg.points = o3d.utility.Vector3dVector(pts[is_edg] if np.any(is_edg) else np.zeros((0, 3), dtype=np.float64))

        vis.update_geometry(pcd_all)
        vis.update_geometry(pcd_edg)
        print(
            f"[RestCandidates] segment={entry['segment_id']} rest_frame={entry['rest_frame']} "
            f"segment_candidates={candidate_global_ids.shape[0]} "
            f"visible_at_rest={int(np.sum(candidate_visible_rest_mask))} "
            f"edg_nodes={int(np.sum(is_edg))}"
        )
        return False

    pcd_all.points = o3d.utility.Vector3dVector(pts0)
    pcd_edg.points = o3d.utility.Vector3dVector(pts0[is_edg0] if np.any(is_edg0) else np.zeros((0, 3), dtype=np.float64))

    def on_space(vis):
        state["idx"] = (state["idx"] + 1) % len(rest_candidates)
        return _refresh(vis, state["idx"])

    def on_b(vis):
        state["idx"] = (state["idx"] - 1) % len(rest_candidates)
        return _refresh(vis, state["idx"])

    def on_q(vis):
        vis.close()
        return False

    print("Open3D controls: SPACE=next segment, B=prev segment, Q/ESC=close.")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=args.window_name, width=1280, height=720)
    vis.add_geometry(pcd_all)
    vis.add_geometry(pcd_edg)
    opt = vis.get_render_option()
    opt.point_size = pt_size
    vis.register_key_callback(ord(" "), on_space)
    vis.register_key_callback(ord("b"), on_b)
    vis.register_key_callback(ord("B"), on_b)
    vis.register_key_callback(ord("q"), on_q)
    vis.register_key_callback(ord("Q"), on_q)
    vis.register_key_callback(256, on_q)

    _refresh(vis, state["idx"])
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
