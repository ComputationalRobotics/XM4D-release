#!/usr/bin/env python3
"""Visualize 4DGS training input: static point cloud + EDG-warped dynamic points.

Serves a Viser viewer for SSH/browser-friendly visualization. It
loads data entirely from:
  - gs_edg_target_xxxx/sparse/0/points3D.txt (static + all canonical points)
  - gs_edg_target_xxxx/dynamic_edg.npz   (dynamic binding: indices, knn, weights, rel offsets)
  - outputs/result_XM.npz                (camera R, t per frame)
  - outputs/results_all.npz              (edg_reconstruct, edg_rotation, edg_edges_union)

Use the frame slider or Next/Prev buttons in the Viser GUI.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Helpers: data loading
# ---------------------------------------------------------------------------

def _find_latest_gs_dir(output_dir: Path) -> Path:
    cands = sorted(output_dir.glob("gs_edg_target_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"No gs_edg_target_* directory in {output_dir}")
    return cands[0]


def _parse_target_from_name(path: Path):
    m = re.search(r"gs_edg_target_(\d+)$", path.name)
    return int(m.group(1)) if m else None


def _read_ascii_ply_xyz_rgb(ply_path: Path):
    """Read xyz + rgb from ASCII PLY (format: x y z nx ny nz r g b ...)."""
    with open(ply_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    n_verts, end_idx = None, None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("element vertex"):
            n_verts = int(s.split()[-1])
        if s == "end_header":
            end_idx = i
            break
    if n_verts is None or end_idx is None:
        raise RuntimeError(f"Invalid PLY header: {ply_path}")

    body = lines[end_idx + 1 : end_idx + 1 + n_verts]
    xyz = np.zeros((n_verts, 3), dtype=np.float64)
    rgb = np.zeros((n_verts, 3), dtype=np.uint8)
    for i, ln in enumerate(body):
        vals = ln.strip().split()
        xyz[i] = [float(vals[0]), float(vals[1]), float(vals[2])]
        rgb[i] = [int(vals[6]), int(vals[7]), int(vals[8])]
    return xyz, rgb


def _read_colmap_points3d_xyz_rgb(points3d_path: Path):
    xyz_rows = []
    rgb_rows = []
    with open(points3d_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            vals = s.split()
            if len(vals) < 8:
                continue
            xyz_rows.append([float(vals[1]), float(vals[2]), float(vals[3])])
            rgb_rows.append([int(vals[4]), int(vals[5]), int(vals[6])])
    if not xyz_rows:
        raise RuntimeError(f"No points found in {points3d_path}")
    return np.asarray(xyz_rows, dtype=np.float64), np.asarray(rgb_rows, dtype=np.uint8)


def _read_colmap_cameras_txt(path: Path):
    """Read the single PINHOLE camera from a COLMAP cameras.txt -> (W, H, K)."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            v = s.split()
            # CAMERA_ID MODEL WIDTH HEIGHT PARAMS...  (PINHOLE: fx fy cx cy)
            W, H = int(v[2]), int(v[3])
            fx, fy, cx, cy = float(v[4]), float(v[5]), float(v[6]), float(v[7])
            K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
            return W, H, K
    raise RuntimeError(f"No camera found in {path}")


def _read_colmap_images_first_c2w(path: Path):
    """Read the first frame (lowest NAME) from images.txt -> (name, c2w 4x4)."""
    from scipy.spatial.transform import Rotation

    entries = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip() and not ln.strip().startswith("#")]
    # Each image is one data line here (points2D line is empty and thus filtered out).
    for ln in lines:
        v = ln.split()
        if len(v) < 10:
            continue
        qw, qx, qy, qz = map(float, v[1:5])
        tx, ty, tz = map(float, v[5:8])
        name = v[9]
        R_w2c = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        t_w2c = np.array([tx, ty, tz], dtype=np.float64)
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = R_w2c
        w2c[:3, 3] = t_w2c
        entries.append((name, np.linalg.inv(w2c)))
    if not entries:
        raise RuntimeError(f"No images found in {path}")
    entries.sort(key=lambda e: e[0])
    return entries[0]


def save_first_frame_render(gs_dir: Path, output_dir: Path):
    """Render the first frame of the 4DGS training input the same way simple_trainer.py
    does (gsplat rasterization, identical Gaussian init from points3D.txt), and save the
    image to outputs/. Uses the exact dataset files simple_trainer consumes, so the result
    matches its frame-0 render at initialization.
    """
    import torch
    import imageio.v2 as imageio
    from scipy.spatial import cKDTree
    from gsplat.rendering import rasterization

    sparse = gs_dir / "sparse" / "0"
    points3d_path = sparse / "points3D.txt"
    cameras_path = sparse / "cameras.txt"
    images_path = sparse / "images.txt"
    for p in (points3d_path, cameras_path, images_path):
        if not p.exists():
            print(f"[render] skip: missing {p}")
            return

    xyz, rgb = _read_colmap_points3d_xyz_rgb(points3d_path)
    W, H, K = _read_colmap_cameras_txt(cameras_path)
    name, c2w = _read_colmap_images_first_c2w(images_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    means = torch.from_numpy(xyz).float().to(device)
    rgbs = torch.from_numpy(rgb.astype(np.float64) / 255.0).float().to(device)

    # GS init identical to simple_trainer.create_splats_with_optimizers (init_scale=1.0,
    # init_opacity=0.1): scales = log(mean dist to 3 nearest neighbors); isotropic quats.
    tree = cKDTree(xyz)
    dist, _ = tree.query(xyz, k=4)                       # includes self at col 0
    dist_avg = np.sqrt((dist[:, 1:] ** 2).mean(axis=1))
    scales = torch.from_numpy(dist_avg).float().to(device).clamp_min(1e-8).unsqueeze(-1).repeat(1, 3)
    N = means.shape[0]
    quats = torch.zeros((N, 4), device=device); quats[:, 0] = 1.0
    opacities = torch.full((N,), 0.1, device=device)

    C0 = 0.28209479177387814                             # rgb_to_sh: DC SH coefficient
    colors = torch.zeros((N, 16, 3), device=device)      # sh_degree=3 -> (3+1)^2 = 16
    colors[:, 0, :] = (rgbs - 0.5) / C0

    Ks = torch.from_numpy(K).float().to(device)[None]
    camtoworlds = torch.from_numpy(c2w).float().to(device)[None]

    with torch.no_grad():
        render_colors, _, _ = rasterization(
            means=means, quats=quats, scales=scales, opacities=opacities,
            colors=colors, viewmats=torch.linalg.inv(camtoworlds), Ks=Ks,
            width=W, height=H, sh_degree=3, camera_model="pinhole",
        )
    img = (render_colors[0, ..., :3].clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "training_input_render_frame000000.png"
    imageio.imwrite(str(out_path), img)
    print(f"[render] saved first-frame render ({name}, {N} gaussians) -> {out_path}")


def save_dynamic_projection_overlay(gs_dir: Path, output_dir: Path, rootpath: Path,
                                    mask_subdir: str = ""):
    """Project the canonical (target-frame) dynamic points to 2D with the frame's
    intrinsic and overlay them on the image + dynamic mask. Diagnoses scale/alignment
    mismatch between the prior-derived dynamic points and the true mask silhouette.
    Saves outputs/dynamic_projection_overlay_frame%06d.png.
    """
    import cv2

    sparse = gs_dir / "sparse" / "0"
    for p in (sparse / "points3D.txt", sparse / "cameras.txt", gs_dir / "dynamic_edg.npz"):
        if not p.exists():
            print(f"[proj] skip overlay: missing {p}")
            return

    xyz, _rgb = _read_colmap_points3d_xyz_rgb(sparse / "points3D.txt")
    d = np.load(str(gs_dir / "dynamic_edg.npz"), allow_pickle=True)
    dyn_idx = d["dyn_indices"].astype(np.int64)
    tgt = int(d["target_frame"]) if "target_frame" in d else 0
    valid = (dyn_idx >= 0) & (dyn_idx < xyz.shape[0])
    dyn = xyz[dyn_idx[valid]]                      # target-frame (= frame `tgt` camera) coords
    W, H, K = _read_colmap_cameras_txt(sparse / "cameras.txt")

    z = dyn[:, 2]
    ok = np.isfinite(z) & (z > 1e-8)
    u = K[0, 0] * dyn[:, 0] / np.clip(z, 1e-8, None) + K[0, 2]
    v = K[1, 1] * dyn[:, 1] / np.clip(z, 1e-8, None) + K[1, 2]

    # frame image
    img_paths = sorted((rootpath / "images").glob("*"))
    img = cv2.imread(str(img_paths[tgt])) if tgt < len(img_paths) else None
    if img is None:
        img = np.full((H, W, 3), 60, np.uint8)
    elif img.shape[:2] != (H, W):
        img = cv2.resize(img, (W, H))

    # dynamic mask (prefer given subdir, else masks_sam, else masks)
    mask = None
    for md in ([mask_subdir] if mask_subdir else []) + ["masks_sam", "masks"]:
        cands = sorted((rootpath / md).glob("*")) if (rootpath / md).is_dir() else []
        if cands and tgt < len(cands):
            m = cv2.imread(str(cands[tgt]), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                mask = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 127
                break

    canvas = img.copy()
    if mask is not None:
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, cnts, -1, (0, 255, 255), 2)   # yellow mask outline

    ui = np.rint(u[ok]).astype(np.int64)
    vi = np.rint(v[ok]).astype(np.int64)
    inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    frac_in = float("nan")
    if mask is not None and np.any(inb):
        inside = mask[vi[inb], ui[inb]]
        frac_in = 100.0 * inside.mean()
        red = np.zeros(inb.sum(), bool); red[~inside] = True
    for j, (uu, vv) in enumerate(zip(ui[inb], vi[inb])):
        col = (0, 0, 255) if (mask is not None and not mask[vv, uu]) else (0, 255, 0)
        cv2.circle(canvas, (int(uu), int(vv)), 1, col, -1)      # green=in-mask, red=outside

    txt = f"dyn pts={dyn.shape[0]}  in-mask={frac_in:.1f}%  (green=in, red=out)"
    cv2.putText(canvas, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"dynamic_projection_overlay_frame{tgt:06d}.png"
    cv2.imwrite(str(out_path), canvas)
    print(f"[proj] saved dynamic projection overlay (in-mask {frac_in:.1f}%) -> {out_path}")


def _load_xm(xm_path: Path, T: int):
    """Load raw solver c2w R/t from result_XM.npz.

    The static solver may drop frames (checklandmarks / XM^2), so R/t are stored in
    solved order (N<=T). Scatter them onto the full timeline via indices_all and fill
    any dropped frame from its nearest solved neighbor (this viz renders every frame).
    """
    from utils.edg import scatter_solved_poses
    rt = np.load(str(xm_path))
    R_raw, t_raw = scatter_solved_poses(rt, T)
    finite = np.isfinite(t_raw).all(axis=1)
    if not finite.all():
        kept = np.where(finite)[0]
        if kept.size:
            for i in np.where(~finite)[0]:
                j = int(kept[np.argmin(np.abs(kept - i))])
                R_raw[i] = R_raw[j]
                t_raw[i] = t_raw[j]
    return R_raw, t_raw


def _load_results_all(npz_path: Path):
    """Load edg_reconstruct, edg_rotation, edg_edges_union from results_all.npz. edg_edges_union is a
    pickled SET of (gid_a, gid_b) in EDG_old files, but a plain (E,2) int array in the new-pipeline
    files written by completed_to_results.py -- accept both."""
    res = np.load(str(npz_path), allow_pickle=True)
    edg_tracks  = res["edg_reconstruct"].item()   # dict: gid -> (T,3)
    edg_rots    = res["edg_rotation"].item()       # dict: gid -> (T,3,3)
    eu = res["edg_edges_union"]
    if eu.ndim == 0:                               # EDG_old: 0-d object array wrapping a set
        edges_union = eu.item()
    else:                                          # new pipeline: (E,2) int array
        edges_union = {(int(a), int(b)) for a, b in np.asarray(eu, dtype=np.int64).reshape(-1, 2)}
    return edg_tracks, edg_rots, edges_union


def _load_dynamic_edg(npz_path: Path, n_total_pts: int):
    """Load dynamic_edg.npz and validate indices against n_total_pts."""
    d = np.load(str(npz_path), allow_pickle=True)
    dyn_indices = d["dyn_indices"].astype(np.int64)
    dyn_knn_idx = d["dyn_knn_idx"].astype(np.int64)   # (Nd, K)
    dyn_knn_w   = d["dyn_knn_w"].astype(np.float64)   # (Nd, K)
    dyn_rel     = d["dyn_rel"].astype(np.float64)      # (Nd, K, 3)
    dyn_rgb     = d["dyn_rgb"].astype(np.uint8) if "dyn_rgb" in d else None
    edg_gids    = d["edg_gids"].astype(np.int64) if "edg_gids" in d else None
    target_frame = int(d["target_frame"]) if "target_frame" in d else 0

    valid = (dyn_indices >= 0) & (dyn_indices < n_total_pts)
    if not np.all(valid):
        print(f"[WARN] {int((~valid).sum())} dyn_indices out of range, dropping.")
        dyn_indices = dyn_indices[valid]
        dyn_knn_idx = dyn_knn_idx[valid]
        dyn_knn_w   = dyn_knn_w[valid]
        dyn_rel     = dyn_rel[valid]
        if dyn_rgb is not None:
            dyn_rgb = dyn_rgb[valid]

    return dyn_indices, dyn_knn_idx, dyn_knn_w, dyn_rel, dyn_rgb, edg_gids, target_frame


# ---------------------------------------------------------------------------
# Helpers: per-frame geometry (mirrors visualize_dynamic_tracks_3.py)
# ---------------------------------------------------------------------------

def _frame_points_from_tracks(track_dict, gids, frame_id):
    pts, idx_map = [], {}
    for gid in gids:
        arr = track_dict.get(gid)
        if arr is None:
            continue
        p = arr[int(frame_id), :]
        if np.isfinite(p).all():
            idx_map[int(gid)] = len(pts)
            pts.append(p)
    if not pts:
        return np.zeros((0, 3), dtype=np.float64), idx_map
    return np.asarray(pts, dtype=np.float64), idx_map


def _frame_rotations_from_tracks(rot_dict, gids, frame_id, idx_map):
    rots = [np.eye(3, dtype=np.float64) for _ in range(len(idx_map))]
    for gid, idx in idx_map.items():
        arr = rot_dict.get(int(gid))
        if arr is None:
            continue
        R = arr[int(frame_id), :, :]
        if np.isfinite(R).all():
            rots[idx] = R
    if not rots:
        return np.zeros((0, 3, 3), dtype=np.float64)
    return np.asarray(rots, dtype=np.float64)


def _camera_axes_lines(cam_center, cam_R, axis_len):
    cam_center = np.asarray(cam_center, dtype=np.float64).reshape(3)
    cam_R      = np.asarray(cam_R, dtype=np.float64).reshape(3, 3)
    axes = cam_R @ (axis_len * np.eye(3))
    pts  = np.vstack([cam_center, cam_center + axes.T])
    lns  = np.array([[0, 1], [0, 2], [0, 3]], dtype=np.int32)
    cols = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    return pts, lns, cols


def _rotation_axes_lines(pts, rots, axis_len):
    if pts.size == 0 or rots.size == 0:
        return (np.zeros((0, 3)), np.zeros((0, 2), dtype=np.int32), np.zeros((0, 3)))
    pts  = np.asarray(pts,  dtype=np.float64)
    rots = np.asarray(rots, dtype=np.float64)
    all_pts, lines_out, colors = [], [], []
    for i, p in enumerate(pts):
        R = rots[i]
        if not np.isfinite(R).all():
            continue
        s = len(all_pts)
        axes = R @ (axis_len * np.eye(3))
        all_pts += [p, p + axes[:, 0], p + axes[:, 1], p + axes[:, 2]]
        lines_out += [[s, s+1], [s, s+2], [s, s+3]]
        colors    += [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    if not all_pts:
        return (np.zeros((0, 3)), np.zeros((0, 2), dtype=np.int32), np.zeros((0, 3)))
    return (np.asarray(all_pts, dtype=np.float64),
            np.asarray(lines_out, dtype=np.int32),
            np.asarray(colors, dtype=np.float64))


def _warp_dynamic_xyz(edg_tracks, edg_rots, edg_gids_sorted, dyn_knn_idx,
                      dyn_knn_w, dyn_rel, frame_id):
    """Warp dynamic points at frame_id using EDG skinning.

    dyn_knn_idx: (Nd, K) indices into edg_gids_sorted
    dyn_knn_w:   (Nd, K) skinning weights
    dyn_rel:     (Nd, K, 3) local offsets from each knn EDG node
    Returns: (Nd, 3)
    """
    M = len(edg_gids_sorted)
    edg_pos = np.zeros((M, 3), dtype=np.float64)
    edg_rot = np.tile(np.eye(3, dtype=np.float64), (M, 1, 1))
    for j, gid in enumerate(edg_gids_sorted):
        arr = edg_tracks.get(gid)
        if arr is not None and np.isfinite(arr[frame_id]).all():
            edg_pos[j] = arr[frame_id]
        rot_arr = edg_rots.get(gid)
        if rot_arr is not None and np.isfinite(rot_arr[frame_id]).all():
            edg_rot[j] = rot_arr[frame_id]

    # For each dynamic point: weighted sum of node position + rotated local offset.
    nb_pos = edg_pos[dyn_knn_idx]                # (Nd, K, 3)
    nb_rot = edg_rot[dyn_knn_idx]                # (Nd, K, 3, 3)
    rotated_rel = np.einsum("nkij,nkj->nki", nb_rot, dyn_rel)
    warped = nb_pos + rotated_rel
    return np.einsum("nk,nkd->nd", dyn_knn_w, warped)  # (Nd, 3)


def _world_to_target_transform(R_c2w: np.ndarray, t_c2w: np.ndarray) -> np.ndarray:
    R = np.asarray(R_c2w, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t_c2w, dtype=np.float64).reshape(3)
    w2t = np.eye(4, dtype=np.float64)
    w2t[:3, :3] = R.T
    w2t[:3, 3] = -R.T @ t
    return w2t


def _apply_transform(points: np.ndarray, mat: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    return (mat @ np.concatenate([pts, ones], axis=1).T).T[:, :3]


def _transform_track_dict(track_dict, mat: np.ndarray):
    out = {}
    for gid, arr in track_dict.items():
        shape = arr.shape
        out[int(gid)] = _apply_transform(np.asarray(arr).reshape(-1, 3), mat).reshape(shape)
    return out


def _transform_rot_dict(rot_dict, R: np.ndarray):
    out = {}
    for gid, arr in rot_dict.items():
        out[int(gid)] = np.einsum("ij,tjk->tik", R, np.asarray(arr, dtype=np.float64))
    return out


# ---------------------------------------------------------------------------
# Main Viser visualizer
# ---------------------------------------------------------------------------

def _block_until_ctrl_c():
    print("[VIS] Press Ctrl-C to stop the Viser server.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("[VIS] Stopped.")


def _clear_paths(server, paths):
    for path in paths:
        try:
            server.scene.remove(path)
        except Exception:
            pass


def _bbox_diag(*arrays):
    mins, maxs = [], []
    for arr in arrays:
        pts = np.asarray(arr, dtype=np.float64).reshape(-1, 3)
        if pts.size == 0:
            continue
        valid = np.isfinite(pts).all(axis=1)
        if not np.any(valid):
            continue
        pts = pts[valid]
        mins.append(np.min(pts, axis=0))
        maxs.append(np.max(pts, axis=0))
    if not mins:
        return 1.0
    diag = float(np.linalg.norm(np.max(np.vstack(maxs), axis=0) - np.min(np.vstack(mins), axis=0)))
    return diag if np.isfinite(diag) and diag > 0.0 else 1.0


def _downsample_points(points, colors, max_points):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    colors = np.asarray(colors)
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    if colors.ndim == 2 and colors.shape[0] == valid.shape[0]:
        colors = colors[valid]
    else:
        colors = np.tile(np.array([180, 180, 180], dtype=np.uint8), (points.shape[0], 1))
    if np.issubdtype(colors.dtype, np.floating):
        colors = np.clip(colors, 0.0, 1.0) * 255.0
    colors = np.asarray(np.clip(colors, 0, 255), dtype=np.uint8)

    max_points = int(max_points)
    if max_points > 0 and points.shape[0] > max_points:
        idx = np.linspace(0, points.shape[0] - 1, max_points).astype(np.int64)
        points = points[idx]
        colors = colors[idx]
    return points.astype(np.float32), colors


def _line_segments(points, lines):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    lines = np.asarray(lines, dtype=np.int64).reshape(-1, 2)
    if points.size == 0 or lines.size == 0:
        return np.zeros((0, 2, 3), dtype=np.float32)
    valid = (
        (lines[:, 0] >= 0)
        & (lines[:, 1] >= 0)
        & (lines[:, 0] < points.shape[0])
        & (lines[:, 1] < points.shape[0])
    )
    lines = lines[valid]
    if lines.size == 0:
        return np.zeros((0, 2, 3), dtype=np.float32)
    return points[lines].astype(np.float32)


def _line_colors(colors, n_segments):
    colors = np.asarray(colors)
    if colors.shape == (3,):
        if np.issubdtype(colors.dtype, np.floating):
            colors = np.clip(colors, 0.0, 1.0) * 255.0
        return np.asarray(np.clip(colors, 0, 255), dtype=np.uint8)
    if colors.ndim == 2 and colors.shape[0] == int(n_segments) and colors.shape[1] == 3:
        if np.issubdtype(colors.dtype, np.floating):
            colors = np.clip(colors, 0.0, 1.0) * 255.0
        colors = np.asarray(np.clip(colors, 0, 255), dtype=np.uint8)
        return np.repeat(colors[:, None, :], 2, axis=1)
    return np.array([255, 255, 255], dtype=np.uint8)


def _empty_points():
    return np.zeros((1, 3), dtype=np.float32), np.array([[0, 0, 0]], dtype=np.uint8)


def run_visualizer(
    *,
    xyz_static,
    rgb_static,
    rgb_dyn,
    dyn_knn_idx,
    dyn_knn_w,
    dyn_rel,
    edg_tracks,
    edg_rots,
    edg_gids_sorted,
    edges_union,
    R_display,
    t_raw,
    T,
    host,
    port,
    max_static_points,
    point_size,
    edg_axis_len_frac=0.02,
    edg_point_mult=1.0,
):
    from utils.viser_vis import get_viser_server

    frame_list = list(range(T))
    state = {"idx": 0}

    def _build_frame(fid):
        edg_pts, edg_idx = _frame_points_from_tracks(edg_tracks, edg_gids_sorted, fid)
        edg_r = _frame_rotations_from_tracks(edg_rots, edg_gids_sorted, fid, edg_idx)
        li = []
        for ga, gb in edges_union:
            if ga in edg_idx and gb in edg_idx:
                li.append([edg_idx[ga], edg_idx[gb]])
        li = np.asarray(li, dtype=np.int32)
        dyn_xyz = _warp_dynamic_xyz(edg_tracks, edg_rots, edg_gids_sorted,
                                    dyn_knn_idx, dyn_knn_w, dyn_rel, fid)
        return edg_pts, edg_r, li, dyn_xyz

    edg_pts0, edg_r0, li0, dyn_xyz0 = _build_frame(frame_list[0])
    diag = _bbox_diag(xyz_static, edg_pts0, dyn_xyz0)
    cam_axis_len = 0.05 * diag if diag > 0 else 0.05
    rot_axis_len = (edg_axis_len_frac * diag) if diag > 0 else edg_axis_len_frac   # EDG pose-frame size
    pt_size = float(point_size) if point_size > 0 else float(np.clip(0.0012 * diag, 0.0008, 0.02))

    server = get_viser_server(host=host, port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    server.gui.add_markdown("### Training Input Dynamics")
    frame_slider = server.gui.add_slider("Frame", min=0, max=T - 1, step=1, initial_value=0)
    btn_prev = server.gui.add_button("Prev")
    btn_next = server.gui.add_button("Next")

    static_pts, static_cols = _downsample_points(xyz_static, rgb_static, max_static_points)
    _clear_paths(server, ["/training/static"])
    if static_pts.shape[0] > 0:
        server.scene.add_point_cloud(
            "/training/static",
            points=static_pts,
            colors=static_cols,
            point_size=pt_size,
            point_shape="circle",
            visible=True,
        )

    dynamic_cols = (
        np.asarray(rgb_dyn, dtype=np.uint8)
        if rgb_dyn is not None
        else np.tile(np.array([0, 102, 255], dtype=np.uint8), (dyn_knn_idx.shape[0], 1))
    )

    dynamic_paths = [
        "/training/dynamic",
        "/training/edg_nodes",
        "/training/edg_edges",
        "/training/edg_rotation_axes",
        "/training/camera_axes",
    ]

    def refresh(fid):
        fid = int(fid) % T
        state["idx"] = fid
        edg_pts, edg_r, li, dyn_xyz = _build_frame(fid)

        _clear_paths(server, dynamic_paths)

        dyn_pts, dyn_cols = _downsample_points(dyn_xyz, dynamic_cols, max_static_points)
        if dyn_pts.shape[0] > 0:
            server.scene.add_point_cloud(
                "/training/dynamic",
                points=dyn_pts,
                colors=dyn_cols,
                point_size=max(pt_size * 0.7, 0.0012),
                point_shape="circle",
                visible=True,
            )
        else:
            empty_pts, empty_cols = _empty_points()
            server.scene.add_point_cloud(
                "/training/dynamic",
                points=empty_pts,
                colors=empty_cols,
                point_size=0.001,
                visible=False,
            )

        edg_valid = np.isfinite(edg_pts).all(axis=1) if edg_pts.size else np.zeros((0,), dtype=bool)
        if np.any(edg_valid):
            edg_cols = np.tile(np.array([255, 64, 64], dtype=np.uint8), (int(np.count_nonzero(edg_valid)), 1))
            server.scene.add_point_cloud(
                "/training/edg_nodes",
                points=edg_pts[edg_valid].astype(np.float32),
                colors=edg_cols,
                point_size=max(pt_size * float(edg_point_mult), 0.002),
                point_shape="circle",
                visible=True,
            )
        else:
            empty_pts, empty_cols = _empty_points()
            server.scene.add_point_cloud(
                "/training/edg_nodes",
                points=empty_pts,
                colors=empty_cols,
                point_size=0.001,
                visible=False,
            )

        edge_segments = _line_segments(edg_pts, li)
        if edge_segments.shape[0] > 0:
            server.scene.add_line_segments(
                "/training/edg_edges",
                points=edge_segments,
                colors=np.array([0, 0, 0], dtype=np.uint8),
                line_width=1.5,
                visible=True,
            )

        rp, rl, rc = _rotation_axes_lines(edg_pts, edg_r, rot_axis_len)
        rot_segments = _line_segments(rp, rl)
        if rot_segments.shape[0] > 0:
            server.scene.add_line_segments(
                "/training/edg_rotation_axes",
                points=rot_segments,
                colors=_line_colors(rc, rot_segments.shape[0]),
                line_width=1.2,
                visible=True,
            )

        cp, cl, cc = _camera_axes_lines(t_raw[fid], R_display[fid], cam_axis_len)
        cam_segments = _line_segments(cp, cl)
        if cam_segments.shape[0] > 0:
            server.scene.add_line_segments(
                "/training/camera_axes",
                points=cam_segments,
                colors=_line_colors(cc, cam_segments.shape[0]),
                line_width=2.0,
                visible=True,
            )

        print(f"[VIS] frame={fid}  edg={len(edg_pts)}  dyn={len(dyn_xyz)}")

    def set_frame(fid):
        fid = int(fid) % T
        if int(frame_slider.value) != fid:
            frame_slider.value = fid
        refresh(fid)

    @frame_slider.on_update
    def _(_event):
        refresh(frame_slider.value)

    @btn_next.on_click
    def _(_event):
        set_frame(state["idx"] + 1)

    @btn_prev.on_click
    def _(_event):
        set_frame(state["idx"] - 1)

    refresh(0)
    print(f"[VIS] Viser running at http://localhost:{port}")
    print(f"[VIS] SSH forwarding example: ssh -L {port}:localhost:{port} <host>")
    print("[VIS] Use the Frame slider or Next/Prev buttons to inspect frames.")
    _block_until_ctrl_c()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize 4DGS training input: static cloud + EDG-warped dynamics")
    parser.add_argument("--gs_dir",   default="",
                        help="Path to gs_edg_target_xxxx directory (default: latest)")
    parser.add_argument("--rootpath", default="",
                        help="Dataset root (contains outputs/). Default: from utils.definitions")
    parser.add_argument("--viser-host", default=os.environ.get("VISER_HOST", "0.0.0.0"))
    parser.add_argument("--viser-port", type=int, default=int(os.environ.get("VISER_PORT", "8080")))
    parser.add_argument(
        "--viser-max-points",
        type=int,
        default=int(os.environ.get("VISER_MAX_POINTS", "200000")),
        help="Maximum points for large point-cloud layers; <=0 disables downsampling.",
    )
    parser.add_argument(
        "--viser-point-size",
        type=float,
        default=float(os.environ.get("VISER_POINT_SIZE", "0.0")),
        help="Override Viser point size. <=0 chooses from scene scale.",
    )
    parser.add_argument(
        "--edg-axis-len",
        type=float,
        default=0.02,
        help="EDG pose-frame (per-node RGB rotation triad) length as a fraction of the scene bbox diagonal "
             "(was 0.04; lower = smaller pose frames).",
    )
    parser.add_argument(
        "--edg-point-mult",
        type=float,
        default=1.0,
        help="EDG node point size as a multiple of the base point size (was 2.0; lower = smaller EDG dots).",
    )
    parser.add_argument(
        "--display-space",
        choices=["solver", "target"],
        default="solver",
        help=(
            "solver: show points/cameras in raw solver-world gauge, matching debug_gt_depth_lift_viser.py. "
            "target: show canonical target-frame coordinates written into the 4DGS dataset."
        ),
    )
    parser.add_argument(
        "--save-first-frame-render",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render the first training frame like simple_trainer.py and save to outputs/.",
    )
    args = parser.parse_args()

    # --- resolve paths ---
    if args.rootpath:
        rootpath = Path(args.rootpath).resolve()
    else:
        import utils.definitions
        rootpath = Path(utils.definitions.rootpath).resolve()

    output_dir = rootpath / "outputs"

    if args.gs_dir:
        gs_dir = Path(args.gs_dir).resolve()
    else:
        gs_dir = _find_latest_gs_dir(output_dir)

    print(f"[INFO] gs_dir   : {gs_dir}")
    print(f"[INFO] rootpath : {rootpath}")

    if args.save_first_frame_render:
        try:
            save_first_frame_render(gs_dir, output_dir)
        except Exception as exc:
            print(f"[render] first-frame render failed (continuing to viser): {exc}")
        try:
            save_dynamic_projection_overlay(gs_dir, output_dir, rootpath,
                                            mask_subdir=os.environ.get("XM_MASK_SUBDIR", ""))
        except Exception as exc:
            print(f"[proj] projection overlay failed (continuing to viser): {exc}")

    points3d_path = gs_dir / "sparse" / "0" / "points3D.txt"
    ply_path      = gs_dir / "points3d.ply"
    dyn_npz_path = gs_dir / "dynamic_edg.npz"
    xm_path      = output_dir / "result_XM.npz"
    res_path     = output_dir / "results_all.npz"

    for p in [dyn_npz_path, xm_path, res_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    # --- load points ---
    print("[INFO] Loading point cloud...")
    if points3d_path.exists():
        xyz_all, rgb_all = _read_colmap_points3d_xyz_rgb(points3d_path)
        point_source = points3d_path
    elif ply_path.exists():
        xyz_all, rgb_all = _read_ascii_ply_xyz_rgb(ply_path)
        point_source = ply_path
    else:
        raise FileNotFoundError(f"Missing point cloud: {points3d_path} or {ply_path}")
    n_pts = xyz_all.shape[0]
    print(f"[INFO] {n_pts} pts from {point_source}")

    # --- dynamic binding ---
    print("[INFO] Loading dynamic_edg.npz...")
    dyn_indices, dyn_knn_idx, dyn_knn_w, dyn_rel, dyn_rgb, edg_gids_from_dyn, target_frame = \
        _load_dynamic_edg(dyn_npz_path, n_pts)

    name_target = _parse_target_from_name(gs_dir)
    if name_target is not None and name_target != target_frame:
        print(f"[WARN] dir name target={name_target} != dynamic_edg target_frame={target_frame}")
    print(f"[INFO] target_frame={target_frame}  dyn_pts={dyn_indices.shape[0]}")

    # --- EDG tracks/rotations ---
    print("[INFO] Loading results_all.npz...")
    edg_tracks, edg_rots, edges_union = _load_results_all(res_path)
    edg_gids_sorted = sorted(edg_tracks.keys())
    print(f"[INFO] EDG nodes={len(edg_gids_sorted)}  edges={len(edges_union)}")

    # Remap dyn_knn_idx from dynamic_edg ordering to edg_gids_sorted ordering
    if edg_gids_from_dyn is not None:
        gid_to_sorted_idx = {int(g): i for i, g in enumerate(edg_gids_sorted)}
        remap = np.array([gid_to_sorted_idx.get(int(g), -1) for g in edg_gids_from_dyn],
                         dtype=np.int64)
        bad = int(np.sum(remap < 0))
        if bad:
            print(f"[WARN] {bad} gids in dynamic_edg not found in results_all.")
        dyn_knn_idx = remap[dyn_knn_idx]   # (Nd, K)

    # --- camera poses ---
    T = edg_tracks[edg_gids_sorted[0]].shape[0]
    print(f"[INFO] T={T} frames")
    R_raw, t_raw = _load_xm(xm_path, T)

    # --- split static / dynamic ---
    static_mask = np.ones(n_pts, dtype=bool)
    static_mask[dyn_indices] = False
    xyz_static_target = xyz_all[static_mask]
    rgb_static = rgb_all[static_mask]
    rgb_dyn_use = dyn_rgb if dyn_rgb is not None else rgb_all[dyn_indices]

    target_w2t = _world_to_target_transform(R_raw[target_frame], t_raw[target_frame])
    target_t2w = np.linalg.inv(target_w2t)

    if args.display_space == "solver":
        xyz_static = _apply_transform(xyz_static_target, target_t2w)
        edg_tracks_display = edg_tracks
        edg_rots_display = edg_rots
        R_display = R_raw
        t_display = t_raw
        print("[INFO] display_space=solver: transformed PLY target-frame points back to solver world.")
    else:
        xyz_static = xyz_static_target
        edg_tracks_display = _transform_track_dict(edg_tracks, target_w2t)
        edg_rots_display = _transform_rot_dict(edg_rots, target_w2t[:3, :3])
        R_display = np.einsum("ij,tjk->tik", target_w2t[:3, :3], R_raw)
        t_display = _apply_transform(t_raw, target_w2t)
        print("[INFO] display_space=target: visualizing canonical target-frame coordinates.")

    print(f"[INFO] static={xyz_static.shape[0]}  dynamic={dyn_indices.shape[0]}")

    run_visualizer(
        xyz_static=xyz_static,
        rgb_static=rgb_static,
        rgb_dyn=rgb_dyn_use,
        dyn_knn_idx=dyn_knn_idx,
        dyn_knn_w=dyn_knn_w,
        dyn_rel=dyn_rel,
        edg_tracks=edg_tracks_display,
        edg_rots=edg_rots_display,
        edg_gids_sorted=edg_gids_sorted,
        edges_union=edges_union,
        R_display=R_display,
        t_raw=t_display,
        T=T,
        host=args.viser_host,
        port=args.viser_port,
        max_static_points=args.viser_max_points,
        point_size=args.viser_point_size,
        edg_axis_len_frac=args.edg_axis_len,
        edg_point_mult=args.edg_point_mult,
    )


if __name__ == "__main__":
    main()
