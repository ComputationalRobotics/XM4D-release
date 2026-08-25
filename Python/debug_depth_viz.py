"""Debug DA3 depth outputs: mask, depth map, and lifted point cloud."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import imageio.v2 as imageio

from utils import definitions

FRAME_INDEX = 49
POINT_STRIDE = 4
MAX_POINTS = 200000
SHOW_2D = True
SHOW_3D = True
MASK_DILATE_PX = 0
MASK_OVERLAY_ALPHA = 0.35


def _load_depth_and_intrinsics(frame_index: int) -> tuple[np.ndarray, np.ndarray]:
    depth_path = Path(definitions.da3_depth_dir) / "depth_maps.npy"
    intr_path = Path(definitions.da3_depth_dir) / "intrinsics.npy"
    if not depth_path.exists():
        raise FileNotFoundError(f"Missing depth map file: {depth_path}")
    if not intr_path.exists():
        raise FileNotFoundError(f"Missing intrinsics file: {intr_path}")

    depth_all = np.load(depth_path)
    intr_all = np.load(intr_path)
    depth = depth_all[frame_index]
    intrinsics = intr_all[frame_index]
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise RuntimeError(f"Unexpected depth shape: {depth.shape}")
    if intrinsics.shape != (3, 3):
        raise RuntimeError(f"Unexpected intrinsics shape: {intrinsics.shape}")
    return depth.astype(np.float32), intrinsics.astype(np.float32)


def _load_mask(frame_index: int) -> np.ndarray | None:
    mask_dir = Path(definitions.mask_dir)
    if not mask_dir.exists():
        return None
    mask_files = sorted(mask_dir.glob(definitions.mask_pattern))
    if not mask_files:
        return None
    if frame_index >= len(mask_files):
        raise IndexError(f"Frame index {frame_index} out of range for masks ({len(mask_files)}).")
    mask = imageio.imread(mask_files[frame_index])
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask > 0).astype(np.uint8)


def lift_depth_points(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    mask: np.ndarray | None = None,
    stride: int = 4,
    max_points: int | None = None,
) -> np.ndarray:
    height, width = depth.shape
    ys = np.arange(0, height, stride)
    xs = np.arange(0, width, stride)
    xv, yv = np.meshgrid(xs, ys)

    z = depth[yv, xv]
    valid = np.isfinite(z) & (z > 0)
    if mask is not None:
        valid &= mask[yv, xv] > 0
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)

    xv = xv[valid].astype(np.float32)
    yv = yv[valid].astype(np.float32)
    z = z[valid].astype(np.float32)

    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])

    x = (xv - cx) / fx * z
    y = (yv - cy) / fy * z
    pts = np.stack([x, y, z], axis=1)

    if max_points is not None and pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]
    return pts


def _show_2d(mask: np.ndarray | None, depth: np.ndarray, dilated_mask: np.ndarray | None) -> None:
    import matplotlib.pyplot as plt

    figs = 2 if mask is not None else 1
    fig, axes = plt.subplots(1, figs, figsize=(6 * figs, 5))
    if figs == 1:
        axes = [axes]

    if mask is not None:
        axes[0].imshow(mask, cmap="gray")
        title = "Mask"
        if dilated_mask is not None:
            title += " (dilated)"
        axes[0].set_title(title)
        axes[0].axis("off")
        depth_ax = axes[1]
    else:
        depth_ax = axes[0]

    finite_depth = depth[np.isfinite(depth)]
    vmin, vmax = np.percentile(finite_depth, [2, 98])
    im = depth_ax.imshow(depth, cmap="magma", vmin=vmin, vmax=vmax)

    if dilated_mask is not None:
        overlay = np.zeros((*dilated_mask.shape, 4), dtype=np.float32)
        overlay[..., 0] = 1.0
        overlay[..., 3] = (dilated_mask > 0).astype(np.float32) * MASK_OVERLAY_ALPHA
        depth_ax.imshow(overlay)

    depth_ax.set_title("Depth")
    depth_ax.axis("off")
    fig.colorbar(im, ax=depth_ax, fraction=0.046, pad=0.04)

    # Text box for mouse depth display
    info_text = depth_ax.text(
        0.02,
        0.98,
        "",
        transform=depth_ax.transAxes,
        va="top",
        ha="left",
        color="white",
        bbox=dict(facecolor="black", alpha=0.65, edgecolor="none"),
    )

    marker, = depth_ax.plot([], [], "wo", markersize=5)

    def on_mouse_move(event):
        if event.inaxes != depth_ax or event.xdata is None or event.ydata is None:
            info_text.set_text("")
            marker.set_data([], [])
            fig.canvas.draw_idle()
            return

        x = int(round(event.xdata))
        y = int(round(event.ydata))

        h, w = depth.shape
        if x < 0 or x >= w or y < 0 or y >= h:
            info_text.set_text("")
            marker.set_data([], [])
            fig.canvas.draw_idle()
            return

        z = float(depth[y, x])
        mask_val = None
        if dilated_mask is not None:
            mask_val = int(dilated_mask[y, x])
        elif mask is not None:
            mask_val = int(mask[y, x])

        if np.isfinite(z):
            msg = f"x={x}, y={y}\ndepth={z:.6f}"
        else:
            msg = f"x={x}, y={y}\ndepth=NaN/Inf"

        if mask_val is not None:
            msg += f"\nmask={mask_val}"

        info_text.set_text(msg)
        marker.set_data([x], [y])
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != depth_ax or event.xdata is None or event.ydata is None:
            return

        x = int(round(event.xdata))
        y = int(round(event.ydata))

        h, w = depth.shape
        if 0 <= x < w and 0 <= y < h:
            z = float(depth[y, x])
            print(f"Clicked pixel: x={x}, y={y}, depth={z}")

    fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)
    fig.canvas.mpl_connect("button_press_event", on_click)

    plt.tight_layout()
    plt.show()


def _show_pointcloud(points: np.ndarray) -> None:
    import open3d as o3d

    if points.shape[0] == 0:
        print("No points to display.")
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.paint_uniform_color([0.2, 0.6, 1.0])
    o3d.visualization.draw_geometries([pcd], window_name="Depth point cloud")


def self_test() -> None:
    depth, intrinsics = _load_depth_and_intrinsics(FRAME_INDEX)
    mask = _load_mask(FRAME_INDEX)
    points = lift_depth_points(depth, intrinsics, mask=mask, stride=POINT_STRIDE, max_points=MAX_POINTS)
    print(f"Depth shape: {depth.shape}")
    print(f"Intrinsics:\n{intrinsics}")
    print(f"Mask present: {mask is not None}")
    print(f"Point cloud points: {points.shape[0]}")


if __name__ == "__main__":
    depth_map, intrinsics = _load_depth_and_intrinsics(FRAME_INDEX)
    mask_map = _load_mask(FRAME_INDEX)
    dilated_mask = None
    if mask_map is not None:
        try:
            from utils.track_tapnet import dilate_dynamic_masks

            dilated_mask = dilate_dynamic_masks(mask_map[None, ...], MASK_DILATE_PX)[0]
        except ModuleNotFoundError as exc:
            print(f"Skipping dilate_dynamic_masks (missing dependency): {exc}")
            dilated_mask = mask_map

    if SHOW_2D:
        _show_2d(mask_map, depth_map, dilated_mask)

    points_3d = lift_depth_points(
        depth_map,
        intrinsics,
        mask=dilated_mask if dilated_mask is not None else mask_map,
        stride=POINT_STRIDE,
        max_points=MAX_POINTS,
    )

    if SHOW_3D:
        _show_pointcloud(points_3d)
