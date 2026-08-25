#!/usr/bin/env python3
"""Visualize the DA3-lifted dynamic points in world coordinates, frame by frame.

Starts at the first frame; press SPACE to step to the next frame (B = previous, S = save
the current frame as a PLY, Q/ESC = close). The lift matches EDG_prior.py exactly:
    X_world = da3_frame_scale[f] * R_raw[f] @ tracks_xyz[:, f] + t_raw[f]
    valid   = finite(X) & (occluded[f] == 0) & dynamic_obs_mask[:, f]

Run (from Python/):
    XM_DATASET_NAME=<scene> conda run -n XM python visualize_lifted_dynamic_frame0.py
Flags: --start N (default 0), --no-scale, --color {image,gray}
"""
import argparse
import glob
from pathlib import Path

import numpy as np

import utils.definitions
from utils.edg import scatter_solved_poses, scatter_solved_frame_vector


def _load_all():
    root = Path(utils.definitions.rootpath)
    data = np.load(utils.definitions.tracks_dynamics_path, allow_pickle=True)
    tracks_xyz = data["tracks_xyz"]          # (1, Q, T, 3)
    tracks_xy = data["tracks_xy"]            # (1, Q, T, 2)  image px
    occluded = data["occluded"]              # (1, Q, T)
    dyn_mask = data["dynamic_obs_mask"]      # (Q, T)
    T = tracks_xyz.shape[2]
    rt = np.load(str(root / "outputs" / "result_XM.npz"))
    R_raw, t_raw = scatter_solved_poses(rt, T)
    return dict(tracks_xyz=tracks_xyz, tracks_xy=tracks_xy, occluded=occluded,
                dyn_mask=dyn_mask, T=T, rt=rt, R_raw=R_raw, t_raw=t_raw,
                image_paths=sorted(glob.glob(str(Path(utils.definitions.video_dir) /
                                                 utils.definitions.image_pattern))))


def _frame_scale(rt, T, no_scale):
    if ("da3_frame_scale" in rt.files) and not no_scale:
        s = scatter_solved_frame_vector(rt, "da3_frame_scale", T, default=1.0)
        return np.where(np.isfinite(s) & (s > 0), s, 1.0)
    return np.ones((T,), dtype=np.float64)


def _valid_at(D, s, frame):
    """world-lift + validity for ALL tracks at `frame`."""
    Rf = float(s[frame]) * D["R_raw"][frame].astype(np.float64)
    P = (Rf @ D["tracks_xyz"][0, :, frame, :].astype(np.float64).T).T + D["t_raw"][frame][None, :]
    valid = (
        np.isfinite(P).all(axis=1)
        & (D["occluded"][0, :, frame] == 0)
        & D["dyn_mask"][:, frame].astype(bool)
    )
    return P, valid


def _lift(D, s, frame, subset):
    """Lift only the fixed `subset` of tracks (e.g. those from the first frame) at `frame`,
    keeping the ones still valid there -- so you follow the SAME points over time."""
    P, valid = _valid_at(D, s, frame)
    keep = subset[valid[subset]]
    return P[keep], D["tracks_xy"][0, keep, frame, :]


def _colors_from_image(D, frame, xy, mode):
    if mode != "image":
        return None
    try:
        import cv2
    except Exception:
        return None
    if frame >= len(D["image_paths"]):
        return None
    img = cv2.cvtColor(cv2.imread(D["image_paths"][frame]), cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    xi = np.clip(np.round(xy[:, 0]).astype(int), 0, W - 1)
    yi = np.clip(np.round(xy[:, 1]).astype(int), 0, H - 1)
    return np.clip(img[yi, xi].astype(np.float64) / 255.0, 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=0, help="starting frame (default 0)")
    ap.add_argument("--no-scale", action="store_true", help="do not apply da3 per-frame scale")
    ap.add_argument("--color", choices=["image", "gray"], default="image")
    args = ap.parse_args()

    import open3d as o3d

    D = _load_all()
    s = _frame_scale(D["rt"], D["T"], args.no_scale)
    T = D["T"]
    start = int(np.clip(args.start, 0, T - 1))
    state = {"idx": start}

    # Fix the point SET to the tracks visible at the start frame, then follow THOSE points
    # across frames (they move / disappear as they get occluded), rather than re-selecting a
    # fresh set every frame.
    _, valid0 = _valid_at(D, s, start)
    subset = np.where(valid0)[0]
    print(f"tracking {subset.size} points from the start frame ({start}) over time")

    pcd = o3d.geometry.PointCloud()
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    def _set_pcd(frame):
        pts, xy = _lift(D, s, frame, subset)
        pcd.points = o3d.utility.Vector3dVector(pts)
        cols = _colors_from_image(D, frame, xy, args.color)
        if cols is not None and len(cols) == len(pts):
            pcd.colors = o3d.utility.Vector3dVector(cols)
        elif len(pts) > 0:
            pcd.paint_uniform_color([0.3, 0.55, 1.0])
        return len(pts)

    def refresh(vis, frame):
        n = _set_pcd(frame)
        vis.update_geometry(pcd)
        # keep the camera fixed across frames so you can see the points move/disappear
        print(f"[Open3D] frame={frame}  start-frame points still visible={n}")
        return False

    def on_space(vis):
        state["idx"] = (state["idx"] + 1) % T
        return refresh(vis, state["idx"])

    def on_b(vis):
        state["idx"] = (state["idx"] - 1) % T
        return refresh(vis, state["idx"])

    def on_s(vis):
        import trimesh
        pts, xy = _lift(D, s, state["idx"], subset)
        cols = _colors_from_image(D, state["idx"], xy, args.color)
        vc = (cols * 255).astype(np.uint8) if cols is not None and len(cols) == len(pts) else None
        out = Path(utils.definitions.output_dir) / f"lifted_dynamic_frame{state['idx']:06d}.ply"
        trimesh.PointCloud(pts.astype(np.float64), colors=vc).export(str(out))
        print(f"[Open3D] saved {out}")
        return False

    def on_q(vis):
        vis.close()
        return False

    print("Open3D controls: SPACE=next frame, B=prev frame, S=save PLY, Q/ESC=close. "
          f"Start at frame={state['idx']} (T={T}).")
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Lifted dynamic points (per frame)", width=1280, height=720)
    n0 = _set_pcd(state["idx"])  # fill points BEFORE add_geometry so the camera fits them
    vis.add_geometry(pcd)
    vis.add_geometry(axes)
    print(f"[Open3D] frame={state['idx']}  lifted dynamic points={n0}")
    vis.get_render_option().point_size = 4.0
    vis.register_key_callback(ord(" "), on_space)
    vis.register_key_callback(ord("b"), on_b)
    vis.register_key_callback(ord("B"), on_b)
    vis.register_key_callback(ord("s"), on_s)
    vis.register_key_callback(ord("S"), on_s)
    vis.register_key_callback(ord("q"), on_q)
    vis.register_key_callback(ord("Q"), on_q)
    vis.register_key_callback(256, on_q)  # ESC

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
