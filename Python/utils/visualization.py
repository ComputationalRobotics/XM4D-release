import colorsys
import os
import open3d as o3d
import numpy as np


def _open3d_windows_allowed() -> bool:
    return (
        os.environ.get("ENABLE_OPEN3D_WINDOWS", "0").lower() in {"1", "true", "yes", "on"}
        or os.environ.get("ALLOW_OPEN3D_WINDOWS", "0").lower() in {"1", "true", "yes", "on"}
    ) and os.environ.get("DISABLE_OPEN3D_WINDOWS", "0").lower() not in {"1", "true", "yes", "on"}

def visualize_camera(extrinsics):
    if not _open3d_windows_allowed():
        print("[Open3D] visualize_camera skipped. Set ALLOW_OPEN3D_WINDOWS=1 to open a window.")
        return
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(
        width=640,
        height=480,
        fx=525.0,
        fy=525.0,
        cx=320.0,
        cy=240.0
    )
    geometries = []
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for extrinsic in extrinsics:
        camera_visualization = o3d.geometry.LineSet.create_camera_visualization(
            intrinsic, extrinsic, scale=0.02
        )
        num_lines = len(camera_visualization.lines)
        camera_visualization.colors = o3d.utility.Vector3dVector(
            [[1,0,0]] * num_lines
        )
        geometries.append(camera_visualization)
        
    for geometry in geometries:
        vis.add_geometry(geometry)  # Add one geometry at a time
    vis.run()
    
def visualize(extrinsics, points, colors = None):
    if not _open3d_windows_allowed():
        print("[Open3D] visualize skipped. Set ALLOW_OPEN3D_WINDOWS=1 to open a window.")
        return
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(
        width=640,
        height=480,
        fx=525.0,
        fy=525.0,
        cx=320.0,
        cy=240.0
    )
    geometries = []
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_size = 3.0
    
    for cam_idx, extrinsic in enumerate(extrinsics):
        camera_visualization = o3d.geometry.LineSet.create_camera_visualization(
            intrinsic, extrinsic, scale=0.1
        )
        num_lines = len(camera_visualization.lines)
        # different color for each camera
        h = cam_idx / max(1, len(extrinsics))
        color_cam = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        camera_visualization.colors = o3d.utility.Vector3dVector(
            [[color_cam[0], color_cam[1], color_cam[2]] for i in range(num_lines)]
        )
        geometries.append(camera_visualization)
        
    for geometry in geometries:
        vis.add_geometry(geometry)  # Add one geometry at a time
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(pcd)
    vis.run()
