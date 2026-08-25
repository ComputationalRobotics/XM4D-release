#!/usr/bin/env python3
"""Visualize transformed dynamic point observations in Open3D.

This script loads the original tracking/dynamics file, transforms the observed
3D dynamic points into the EDG/world frame using the XM camera output, and
visualizes the valid observations frame-by-frame.

It intentionally does not load EDG solver outputs such as results_all.npz.
It visualizes the transformed observations:

    X_world[t, q] = R_for_edg[t] @ tracks_xyz[q, t] + t_raw[t]

where R_for_edg optionally includes the per-frame scale stored in result_XM.npz.

Controls:
    SPACE : next frame
    B     : previous frame
    Q/ESC : close Open3D window
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import open3d as o3d

# Make the script runnable from either the project root or the scripts folder.
_THIS_DIR = Path(__file__).resolve().parent
_CANDIDATE_PROJECT_ROOTS = [Path.cwd(), _THIS_DIR, _THIS_DIR.parent]
for _root in _CANDIDATE_PROJECT_ROOTS:
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import utils.definitions  # noqa: E402
from utils.edg import build_world_trajectories_from_tracks  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize transformed dynamic point observations in Open3D."
    )
    parser.add_argument(
        "--tracks-npz",
        type=Path,
        default=Path(utils.definitions.tracks_dynamics_path),
        help="Path to tracks_dynamics npz. Default: utils.definitions.tracks_dynamics_path",
    )
    parser.add_argument(
        "--xm-result",
        type=Path,
        default=Path(utils.definitions.rootpath) / "outputs" / "result_XM.npz",
        help="Path to XM pose result npz. Default: rootpath/outputs/result_XM.npz",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Initial frame index shown in the viewer.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame stride for the viewer frame list.",
    )
    parser.add_argument(
        "--min-valid-per-track",
        type=int,
        default=1,
        help="Only keep dynamic tracks with at least this many valid transformed observations.",
    )
    parser.add_argument(
        "--no-scale-in-r",
        action="store_true",
        help="Do not multiply R by s from result_XM.npz even if s exists.",
    )
    parser.add_argument(
        "--no-landmarks",
        action="store_true",
        help="Do not visualize static landmarks from result_XM.npz, even if present.",
    )
    parser.add_argument(
        "--window-name",
        type=str,
        default="Transformed dynamic observations",
        help="Open3D window name.",
    )
    parser.add_argument(
        "--delete-smoothness-percentile",
        type=float,
        default=0.1,
        help="Same global smoothness filter used by EDG_old.py. Set <0 to disable.",
    )
    return parser.parse_args()


def _load_tracks(tracks_npz: Path):
    if not tracks_npz.exists():
        raise FileNotFoundError(f"tracks npz not found: {tracks_npz}")

    data = np.load(str(tracks_npz), allow_pickle=True)
    required = ["tracks_xyz", "occluded", "dynamic_obs_mask"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing keys in {tracks_npz}: {missing}")

    tracks_xyz = np.asarray(data["tracks_xyz"], dtype=np.float64)
    occluded = np.asarray(data["occluded"])
    dynamic_obs_mask = np.asarray(data["dynamic_obs_mask"])

    if tracks_xyz.ndim != 4 or tracks_xyz.shape[0] != 1 or tracks_xyz.shape[-1] != 3:
        raise ValueError(f"Expected tracks_xyz shape [1,Q,T,3], got {tracks_xyz.shape}")
    if occluded.shape[:3] != tracks_xyz.shape[:3]:
        raise ValueError(f"Expected occluded shape compatible with tracks_xyz, got {occluded.shape}")

    T = tracks_xyz.shape[2]
    dynamic_obs_mask = dynamic_obs_mask[:, :T]
    if dynamic_obs_mask.shape != (tracks_xyz.shape[1], T):
        raise ValueError(
            "Expected dynamic_obs_mask shape [Q,T] after slicing, "
            f"got {dynamic_obs_mask.shape}, expected {(tracks_xyz.shape[1], T)}"
        )

    return tracks_xyz[:, :, :T, :], occluded[:, :, :T], dynamic_obs_mask.astype(bool)


def _load_xm_result(xm_result: Path, T: int, use_scale: bool):
    if not xm_result.exists():
        raise FileNotFoundError(f"XM result npz not found: {xm_result}")

    rt = np.load(str(xm_result), allow_pickle=True)
    if "R" not in rt or "t" not in rt:
        raise KeyError(f"Expected keys 'R' and 't' in {xm_result}")

    R_blob = np.asarray(rt["R"], dtype=np.float64)
    t_blob = np.asarray(rt["t"], dtype=np.float64)

    if R_blob.shape[0] != 3 or R_blob.shape[1] < 3 * T:
        raise ValueError(f"Expected R shape [3, >=3*T], got {R_blob.shape}, T={T}")
    if t_blob.shape[0] != 3 or t_blob.shape[1] < T:
        raise ValueError(f"Expected t shape [3, >=T], got {t_blob.shape}, T={T}")

    R_raw = np.zeros((T, 3, 3), dtype=np.float64)
    t_raw = np.zeros((T, 3), dtype=np.float64)
    for i in range(T):
        R_raw[i] = R_blob[:, 3 * i : 3 * (i + 1)]
        t_raw[i] = t_blob[:, i]

    R_for_obs = R_raw.copy()
    print(R_raw)

    if use_scale and "s" in rt:
        s_raw = np.asarray(rt["s"], dtype=np.float64).reshape(-1)
        print(s_raw)
        if s_raw.shape[0] < T:
            raise ValueError(f"Expected s to have at least T entries, got {s_raw.shape[0]}, T={T}")
        R_for_obs = R_for_obs * s_raw[:T, None, None]
        print("[Load] Applied per-frame scale s to R_for_obs.")
    else:
        print("[Load] Using unscaled R_for_obs.")

    landmarks = None
    if "p" in rt:
        landmarks = np.asarray(rt["p"], dtype=np.float64).T
        if landmarks.ndim != 2 or landmarks.shape[1] != 3:
            print(f"[Load] Ignoring landmarks with unexpected shape {landmarks.shape}")
            landmarks = None

    return R_raw, R_for_obs, t_raw, landmarks


def _build_transformed_observation_tracks(
    tracks_xyz: np.ndarray,
    occluded: np.ndarray,
    dynamic_obs_mask: np.ndarray,
    R_for_obs: np.ndarray,
    t_raw: np.ndarray,
    min_valid_per_track: int,
) -> Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """Return {global_track_id: [T,3]} for valid transformed dynamic observations."""
    Q = tracks_xyz.shape[1]
    T = tracks_xyz.shape[2]

    glob_all_full = (
        np.einsum("tij,qtj->qti", R_for_obs, tracks_xyz[0].astype(np.float64))
        + t_raw[None, :, :]
    )

    valid_mask = (
        np.isfinite(glob_all_full).all(axis=-1)
        & (occluded[0] == 0)
        & dynamic_obs_mask.astype(bool)
    )

    dyn_tracks: Dict[int, np.ndarray] = {}
    for gid in range(Q):
        valid_count = int(np.count_nonzero(valid_mask[gid]))
        if valid_count < int(min_valid_per_track):
            continue
        arr = np.full((T, 3), np.nan, dtype=np.float64)
        arr[valid_mask[gid]] = glob_all_full[gid, valid_mask[gid]]
        dyn_tracks[int(gid)] = arr

    return dyn_tracks, glob_all_full, valid_mask


def _build_track_dict_from_global_ids(
    global_ids: np.ndarray,
    glob_all_full: np.ndarray,
    valid_mask: np.ndarray,
    min_valid_per_track: int,
) -> Dict[int, np.ndarray]:
    T = glob_all_full.shape[1]
    out: Dict[int, np.ndarray] = {}
    for gid in np.asarray(global_ids, dtype=int).reshape(-1).tolist():
        gid = int(gid)
        valid_gid = np.asarray(valid_mask[gid], dtype=bool)
        valid_count = int(np.count_nonzero(valid_gid))
        if valid_count < int(min_valid_per_track):
            continue
        arr = np.full((T, 3), np.nan, dtype=np.float64)
        arr[valid_gid] = glob_all_full[gid, valid_gid]
        out[gid] = arr
    return out


def _frame_points_from_tracks(track_dict, gids, frame_id):
    pts = []
    idx_map = {}
    for gid in gids:
        arr = track_dict.get(gid)
        if arr is None:
            continue
        p = arr[int(frame_id), :]
        if np.isfinite(p).all():
            idx_map[int(gid)] = len(pts)
            pts.append(p)
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float64), idx_map
    return np.asarray(pts, dtype=np.float64), idx_map


def _camera_axes_lines(cam_center, cam_R, axis_len):
    cam_center = np.asarray(cam_center, dtype=np.float64).reshape(3,)
    cam_R = np.asarray(cam_R, dtype=np.float64).reshape(3, 3)
    axes = cam_R @ (axis_len * np.eye(3))
    pts = np.vstack([cam_center, cam_center + axes.T])
    lines = np.array([[0, 1], [0, 2], [0, 3]], dtype=np.int32)
    colors = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return pts, lines, colors


def _safe_axis_length(points: np.ndarray, fallback: float = 0.05) -> Tuple[float, float]:
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
    return diag, 0.05 * diag


def _make_sphere_marker(center: np.ndarray, radius: float, color: Iterable[float]):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh.compute_vertex_normals()
    mesh.translate(np.asarray(center, dtype=np.float64).reshape(3))
    mesh.paint_uniform_color(list(color))
    return mesh


def main() -> None:
    args = _parse_args()

    if args.stride <= 0:
        raise ValueError("--stride must be positive")

    tracks_xyz, occluded, dynamic_obs_mask = _load_tracks(args.tracks_npz)
    T = tracks_xyz.shape[2]
    Q = tracks_xyz.shape[1]
    R_raw, R_for_obs, t_raw, landmarks = _load_xm_result(
        args.xm_result,
        T=T,
        use_scale=not args.no_scale_in_r,
    )

    dyn_tracks, glob_all_full, valid_mask = _build_transformed_observation_tracks(
        tracks_xyz=tracks_xyz,
        occluded=occluded,
        dynamic_obs_mask=dynamic_obs_mask,
        R_for_obs=R_for_obs,
        t_raw=t_raw,
        min_valid_per_track=args.min_valid_per_track,
    )

    keep_percentile = None
    if args.delete_smoothness_percentile is not None and args.delete_smoothness_percentile >= 0:
        keep_percentile = float(args.delete_smoothness_percentile)

    kept_global_ids = np.array(sorted(dyn_tracks.keys()), dtype=int)
    deleted_tracks: Dict[int, np.ndarray] = {}
    deleted_global_ids = np.zeros((0,), dtype=int)
    if keep_percentile is not None:
        global_filter_output = build_world_trajectories_from_tracks(
            glob_all=glob_all_full,
            valid_mask_global=valid_mask,
            frames=np.arange(T, dtype=int),
            delete_smoothness_percentile=keep_percentile,
        )
        global_keep = np.array(global_filter_output["selected_edg_idx"], dtype=int)
        global_keep_set = set(global_keep.tolist())
        deleted_global_ids = np.array(
            [gid for gid in kept_global_ids.tolist() if int(gid) not in global_keep_set],
            dtype=int,
        )
        kept_global_ids = np.array(
            [gid for gid in kept_global_ids.tolist() if int(gid) in global_keep_set],
            dtype=int,
        )
        dyn_tracks = _build_track_dict_from_global_ids(
            kept_global_ids,
            glob_all_full,
            valid_mask,
            min_valid_per_track=args.min_valid_per_track,
        )
        deleted_tracks = _build_track_dict_from_global_ids(
            deleted_global_ids,
            glob_all_full,
            valid_mask,
            min_valid_per_track=args.min_valid_per_track,
        )

    total_valid_obs = int(np.count_nonzero(valid_mask))
    dyn_gids = sorted(dyn_tracks.keys())
    deleted_gids = sorted(deleted_tracks.keys())
    if len(dyn_gids) == 0 and len(deleted_gids) == 0:
        raise RuntimeError("No valid transformed dynamic observations to visualize.")

    frame_list = list(range(0, T, args.stride))
    start_frame = int(np.clip(args.start_frame, 0, T - 1))
    if start_frame not in frame_list:
        nearest_idx = int(np.argmin(np.abs(np.asarray(frame_list) - start_frame)))
    else:
        nearest_idx = frame_list.index(start_frame)
    state = {"idx": nearest_idx}

    print(
        f"[Load] frames={T}, tracks={Q}, kept_dyn_tracks={len(dyn_gids)}, "
        f"deleted_global_tracks={len(deleted_gids)}, valid_dynamic_observations={total_valid_obs}"
    )

    def get_frame_geoms(frame_id: int):
        dyn_pts, dyn_idx = _frame_points_from_tracks(dyn_tracks, dyn_gids, frame_id)
        deleted_pts, deleted_idx = _frame_points_from_tracks(deleted_tracks, deleted_gids, frame_id)
        return dyn_pts, dyn_idx, deleted_pts, deleted_idx

    dyn_pts0, _, deleted_pts0, _ = get_frame_geoms(frame_list[state["idx"]])

    all_for_scale = dyn_pts0 if dyn_pts0.size > 0 else deleted_pts0
    if all_for_scale.shape[0] < 2:
        sample_frames = frame_list[:: max(1, len(frame_list) // 20)]
        sampled = []
        for f in sample_frames:
            pts_keep, _, pts_del, _ = get_frame_geoms(f)
            if pts_keep.size > 0:
                sampled.append(pts_keep)
            if pts_del.size > 0:
                sampled.append(pts_del)
        if sampled:
            all_for_scale = np.vstack(sampled)
    diag, cam_axis_len = _safe_axis_length(all_for_scale)
    pt_size = 3.0 if diag <= 0 else float(np.clip(800.0 / (diag * 1000.0 + 1e-6), 2.0, 6.0))
    cam_marker_radius = 0.015 * diag if diag > 0 else 0.015

    pcd_dyn = o3d.geometry.PointCloud()
    pcd_dyn.points = o3d.utility.Vector3dVector(dyn_pts0)
    pcd_dyn.paint_uniform_color([0.0, 0.4, 1.0])

    pcd_deleted = o3d.geometry.PointCloud()
    pcd_deleted.points = o3d.utility.Vector3dVector(deleted_pts0)
    pcd_deleted.paint_uniform_color([1.0, 0.2, 0.2])

    pcd_landmarks = o3d.geometry.PointCloud()
    if landmarks is not None and not args.no_landmarks:
        pcd_landmarks.points = o3d.utility.Vector3dVector(landmarks.astype(np.float64))
    else:
        pcd_landmarks.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
    pcd_landmarks.paint_uniform_color([1.0, 0.8, 0.2])

    frame0 = frame_list[state["idx"]]
    cam_pts0, cam_lines0, cam_colors0 = _camera_axes_lines(t_raw[frame0], R_for_obs[frame0], cam_axis_len)
    cam_axes = o3d.geometry.LineSet()
    cam_axes.points = o3d.utility.Vector3dVector(cam_pts0)
    cam_axes.lines = o3d.utility.Vector2iVector(cam_lines0)
    cam_axes.colors = o3d.utility.Vector3dVector(cam_colors0)

    cam_marker = _make_sphere_marker(np.zeros(3, dtype=np.float64), cam_marker_radius, [0.8, 0.0, 0.8])
    cam_marker_base_vertices = np.asarray(cam_marker.vertices, dtype=np.float64).copy()
    cam_marker.vertices = o3d.utility.Vector3dVector(cam_marker_base_vertices + t_raw[frame0].reshape(1, 3))

    def refresh(vis, frame_id: int):
        dyn_pts, _, deleted_pts, _ = get_frame_geoms(frame_id)
        pcd_dyn.points = o3d.utility.Vector3dVector(dyn_pts)
        pcd_deleted.points = o3d.utility.Vector3dVector(deleted_pts)

        cam_pts, cam_lines, cam_colors = _camera_axes_lines(t_raw[frame_id], R_for_obs[frame_id], cam_axis_len)
        cam_axes.points = o3d.utility.Vector3dVector(cam_pts)
        cam_axes.lines = o3d.utility.Vector2iVector(cam_lines)
        cam_axes.colors = o3d.utility.Vector3dVector(cam_colors)

        cam_marker.vertices = o3d.utility.Vector3dVector(
            cam_marker_base_vertices + t_raw[frame_id].reshape(1, 3)
        )

        vis.update_geometry(pcd_dyn)
        vis.update_geometry(pcd_deleted)
        vis.update_geometry(pcd_landmarks)
        vis.update_geometry(cam_axes)
        vis.update_geometry(cam_marker)
        print(f"[Open3D] frame={frame_id} kept={len(dyn_pts)} deleted_global={len(deleted_pts)}")
        return False

    def on_space(vis):
        state["idx"] = (state["idx"] + 1) % len(frame_list)
        return refresh(vis, frame_list[state["idx"]])

    def on_b(vis):
        state["idx"] = (state["idx"] - 1) % len(frame_list)
        return refresh(vis, frame_list[state["idx"]])

    def on_q(vis):
        vis.close()
        return False

    print(
        "Open3D controls: SPACE=next frame, B=prev frame, Q/ESC=close. "
        f"Start at frame={frame_list[state['idx']]}"
    )

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=args.window_name, width=1280, height=720)
    vis.add_geometry(pcd_dyn)
    vis.add_geometry(pcd_deleted)
    vis.add_geometry(pcd_landmarks)
    vis.add_geometry(cam_axes)
    vis.add_geometry(cam_marker)

    opt = vis.get_render_option()
    opt.point_size = pt_size

    vis.register_key_callback(ord(" "), on_space)
    vis.register_key_callback(ord("b"), on_b)
    vis.register_key_callback(ord("B"), on_b)
    vis.register_key_callback(ord("q"), on_q)
    vis.register_key_callback(ord("Q"), on_q)
    vis.register_key_callback(256, on_q)  # ESC on many Open3D/GLFW builds.

    refresh(vis, frame_list[state["idx"]])
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
