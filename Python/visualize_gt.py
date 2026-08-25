#!/usr/bin/env python3
"""Visualize GT camera poses and lifted depth points incrementally (Open3D).

Controls:
    Space: add next frame's points to the current point cloud and update camera.
    Esc/Q: close window.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import open3d as o3d

from utils.readgt_sintel import load_sintel_gt, depth_read


def lift_depth_to_points(depth: np.ndarray, K: np.ndarray, stride: int = 8) -> np.ndarray:
    """Lift a depth map to 3D points in camera coordinates.

    Args:
        depth: [H, W] float32 depth
        K: 3x3 intrinsics
        stride: sampling stride
    Returns:
        points: [N, 3]
    """
    H, W = depth.shape
    ys = np.arange(0, H, stride)
    xs = np.arange(0, W, stride)
    xv, yv = np.meshgrid(xs, ys)
    z = depth[yv, xv].astype(np.float32)
    valid = np.isfinite(z) & (z > 0)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)

    xv = xv[valid].astype(np.float32)
    yv = yv[valid].astype(np.float32)
    z = z[valid]

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    X = (xv - cx) / fx * z
    Y = (yv - cy) / fy * z
    pts = np.stack([X, Y, z], axis=1)
    return pts


def camera_frame_mesh(size: float = 0.1) -> o3d.geometry.TriangleMesh:
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    return frame


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--stride", type=int, default=8, help="Depth sampling stride")
    parser.add_argument("--fps", type=float, default=10.0, help="Playback FPS (unused in manual mode)")
    parser.add_argument("--camera_size", type=float, default=1.0, help="Camera frame size")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(script_dir, '../datasets/Sintel_debug/temple_2'))
    depth_dir = os.path.join(root, "depths")
    image_dir = os.path.join(root, "images")
    
    gt = load_sintel_gt(str(root))
    depth_paths = sorted(glob.glob(os.path.join(str(depth_dir), "*.dpt")))
    image_paths = sorted(glob.glob(os.path.join(str(image_dir), "*.png")))

    if not depth_paths:
        raise FileNotFoundError(f"No depth files found in {depth_dir}")
    if not image_paths:
        raise FileNotFoundError(f"No image files found in {image_dir}")

    T = min(len(depth_paths), len(image_paths))

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="GT Camera + Depth", width=960, height=720)

    pcd = o3d.geometry.PointCloud()
    cam_frame = camera_frame_mesh(args.camera_size)

    vis.add_geometry(pcd)
    vis.add_geometry(cam_frame)

    state = {
        "t": 0,
        "points": np.zeros((0, 3), dtype=np.float64),
    }

    def update_frame(vis):
        if state["t"] >= T:
            print("Reached last frame.")
            return False

        img_name = os.path.basename(image_paths[state["t"]])
        if img_name not in gt:
            state["t"] += 1
            return True

        K = gt[img_name]["K"]
        R = gt[img_name]["R"]
        tvec = gt[img_name]["t"]

        # Camera center in world: C = -R.T @ t (R is world->camera)
        C = -R.T @ tvec

        depth = depth_read(depth_paths[state["t"]])
        pts_cam = lift_depth_to_points(depth, K, stride=args.stride)

        # Transform points to world coordinates: Xw = R.T * Xc + C
        if pts_cam.size > 0:
            pts_world = (R.T @ pts_cam.T).T + C.reshape(1, 3)
        else:
            pts_world = np.zeros((0, 3), dtype=np.float64)

        # Accumulate points over time
        if pts_world.size > 0:
            state["points"] = np.vstack([state["points"], pts_world])

        pcd.points = o3d.utility.Vector3dVector(pts_world)

        # Update camera frame pose
        cam_frame = camera_frame_mesh(args.camera_size)
        cam_frame.rotate(R.T, center=(0, 0, 0))
        cam_frame.translate(C)

        vis.add_geometry(pcd)
        vis.add_geometry(cam_frame)

        state["t"] += 1
        return True

    def on_space(vis):
        update_frame(vis)
        return False

    vis.register_key_callback(ord(" "), on_space)

    # Initialize first frame
    update_frame(vis)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
