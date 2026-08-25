import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "Python")]

from eval_depth import _prefer_npy_depth_path, _read_depth
from eval_depth_depthanything import (
    _gt_to_da_similarity,
    _load_da3_cameras,
    _load_da3_depths,
    _load_gt,
    _load_novel_metadata,
    _read_dynamic_mask,
    _scale_K,
)

REPLICA_RENDER_DEPTH_SCALE = 65535.0 * 0.1


def quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("Zero-length quaternion")
    qw, qx, qy, qz = q / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def c2w(info):
    """Convert stored R/t into a 4x4 camera-to-world matrix.

    The input format stores R,t such that X_cam = R @ X_world + t. We invert
    that to produce a c2w matrix.
    """
    R = np.asarray(info["R"], float).reshape(3, 3)
    t = np.asarray(info["t"], float).reshape(3)

    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = -R.T @ t

    return T


def w2c_to_c2w(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3).T
    T[:3, 3] = -T[:3, :3] @ np.asarray(t, dtype=np.float64).reshape(3)
    return T


def transform_c2w_similarity(c2w: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = np.asarray(c2w, dtype=np.float64).copy()
    out[:3, :3] = R @ out[:3, :3]
    out[:3, 3] = float(scale) * (R @ out[:3, 3]) + t.reshape(3)
    return out


def inverse_transform_c2w_similarity(c2w: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = np.asarray(c2w, dtype=np.float64).copy()
    inv_R = R.T
    inv_scale = 1.0 / max(float(scale), 1e-12)
    out[:3, :3] = inv_R @ out[:3, :3]
    out[:3, 3] = inv_scale * (inv_R @ (out[:3, 3] - t.reshape(3)))
    return out


def transform_points_similarity(points: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if points.size == 0:
        return points
    return float(scale) * (R @ points.T).T + t.reshape(1, 3)


def inverse_transform_points_similarity(points: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if points.size == 0:
        return points
    return (R.T @ (points - t.reshape(1, 3)).T).T / max(float(scale), 1e-12)


def _pose_file(directory: Path) -> Path | None:
    for name in ("camera.txt", "caemra.txt", "camera_poses_world.txt"):
        path = directory / name
        if path.is_file():
            return path
    return None


def _frame_id_from_name(name: str) -> int | None:
    digits = "".join(ch if ch.isdigit() else " " for ch in Path(name).stem).split()
    return int(digits[-1]) if digits else None


def _pose_from_matrix(flat: list[float], convention: str) -> np.ndarray:
    if len(flat) == 12:
        T = np.eye(4, dtype=np.float64)
        T[:3, :4] = np.asarray(flat, dtype=np.float64).reshape(3, 4)
    elif len(flat) == 16:
        T = np.asarray(flat, dtype=np.float64).reshape(4, 4)
    else:
        raise ValueError(f"Expected 12 or 16 matrix values, got {len(flat)}")
    if convention == "w2c":
        return np.linalg.inv(T)
    return T


def _pose_from_tokens(parts: list[str], convention: str) -> tuple[int | None, np.ndarray] | None:
    """Parse one camera.txt/caemra.txt row into a c2w pose.

    Supported rows:
    - metadata-like: novel_id base image depth tx ty tz qw qx qy qz (c2w)
    - renderer camera_poses_world: frame_id image_name tx ty tz qw qx qy qz (c2w)
    - COLMAP-like: image_id qw qx qy qz tx ty tz camera_id image_name (w2c)
    - TUM-like: id/name tx ty tz qx qy qz qw (c2w)
    - matrix-like: optional id/name followed by 12 or 16 matrix values (c2w by default)
    """
    if len(parts) == 11 and not _is_float(parts[2]) and not _is_float(parts[3]):
        novel_id = int(parts[0])
        tx, ty, tz = [float(x) for x in parts[4:7]]
        qw, qx, qy, qz = [float(x) for x in parts[7:11]]
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = quat_to_rot(qw, qx, qy, qz)
        T[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
        return novel_id, T

    if len(parts) == 9 and _is_float(parts[0]) and not _is_float(parts[1]):
        frame_id = int(float(parts[0]))
        tx, ty, tz = [float(x) for x in parts[2:5]]
        qw, qx, qy, qz = [float(x) for x in parts[5:9]]
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = quat_to_rot(qw, qx, qy, qz)
        T[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
        if convention == "w2c":
            T = np.linalg.inv(T)
        return frame_id, T

    if len(parts) >= 10 and all(_is_float(x) for x in parts[1:9]):
        # COLMAP images.txt style stores world-to-camera q/t.
        try:
            image_id = int(parts[0])
            camera_id = int(parts[8])
            _ = camera_id
        except ValueError:
            image_id = _frame_id_from_name(parts[-1])
        qw, qx, qy, qz = [float(x) for x in parts[1:5]]
        tx, ty, tz = [float(x) for x in parts[5:8]]
        name_id = _frame_id_from_name(" ".join(parts[9:]))
        return name_id if name_id is not None else image_id, w2c_to_c2w(
            quat_to_rot(qw, qx, qy, qz), np.array([tx, ty, tz], dtype=np.float64)
        )

    lead_id: int | None = None
    numeric_start = 0
    if parts and not _is_float(parts[0]):
        lead_id = _frame_id_from_name(parts[0])
        numeric_start = 1
    elif parts and len(parts) in {8, 13, 17} and _is_float(parts[0]):
        # A leading integer id plus 7 pose values or 12/16 matrix values.
        lead_id = int(float(parts[0]))
        numeric_start = 1

    nums = [float(x) for x in parts[numeric_start:] if _is_float(x)]
    if len(nums) in {12, 16}:
        return lead_id, _pose_from_matrix(nums, convention)
    if len(nums) == 7:
        tx, ty, tz = nums[:3]
        # TUM convention is qx qy qz qw after translation.
        qx, qy, qz, qw = nums[3:]
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = quat_to_rot(qw, qx, qy, qz)
        T[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
        if convention == "w2c":
            T = np.linalg.inv(T)
        return lead_id, T
    return None


def _is_float(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def load_camera_txt_poses(directory: Path, convention: str = "c2w") -> dict[int, np.ndarray]:
    path = _pose_file(directory)
    if path is None:
        return {}

    cleaned: list[tuple[int, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.split("#", 1)[0].strip()
            if line:
                cleaned.append((line_no, line))

    all_parts = [part for _line_no, line in cleaned for part in line.split()]
    if all_parts and all(_is_float(part) for part in all_parts):
        nums = [float(part) for part in all_parts]
        if len(nums) in {12, 16}:
            return {0: _pose_from_matrix(nums, convention)}
        if len(nums) % 16 == 0:
            return {
                idx: _pose_from_matrix(nums[16 * idx : 16 * (idx + 1)], convention)
                for idx in range(len(nums) // 16)
            }
        if len(nums) % 12 == 0:
            return {
                idx: _pose_from_matrix(nums[12 * idx : 12 * (idx + 1)], convention)
                for idx in range(len(nums) // 12)
            }

    rows: list[tuple[int | None, np.ndarray]] = []
    for line_no, line in cleaned:
        parsed = _pose_from_tokens(line.split(), convention)
        if parsed is None:
            raise ValueError(f"Could not parse pose row {line_no} in {path}: {line}")
        rows.append(parsed)

    if len(rows) == 1 and rows[0][0] is None:
        return {0: rows[0][1]}
    return {idx if pose_id is None else int(pose_id): pose for idx, (pose_id, pose) in enumerate(rows)}


def pose_for_id(poses: dict[int, np.ndarray], pose_id: int, label: str) -> np.ndarray | None:
    if not poses:
        return None
    if pose_id in poses:
        return poses[pose_id]
    keys = sorted(poses)
    if 0 <= pose_id < len(keys):
        return poses[keys[pose_id]]
    raise KeyError(f"{label} pose {pose_id} was not found in camera.txt/caemra.txt; available ids: {keys[:10]}...")


def image_files(scene_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in (scene_dir / "images").iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


def img_size(path: Path) -> tuple[int, int]:
    """Return image size as (width, height)."""
    im = cv2.imread(str(path))
    if im is None:
        raise FileNotFoundError(path)
    return (im.shape[1], im.shape[0])


def depth_path_for_frame(scene_dir: Path, frame_id: int) -> Path:
    depth_dir = scene_dir / "depths"
    direct = _prefer_npy_depth_path(depth_dir / f"depth{int(frame_id):06d}.png")
    if direct.is_file():
        return direct

    files = sorted(
        p
        for p in depth_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".npy", ".npz", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dpt"}
    )
    by_id = {fid: p for p in files if (fid := _frame_id_from_name(p.name)) is not None}
    if int(frame_id) in by_id:
        return _prefer_npy_depth_path(by_id[int(frame_id)])
    if 0 <= int(frame_id) < len(files):
        return _prefer_npy_depth_path(files[int(frame_id)])
    raise FileNotFoundError(f"No depth found for frame {frame_id} in {depth_dir}")


def mask_dir(scene_dir: Path) -> Path:
    for name in ("mask", "masks"):
        path = scene_dir / name
        if path.is_dir():
            return path
    return scene_dir / "mask"


def novel_mask_dir(scene_dir: Path) -> Path:
    for path in (scene_dir / "novel" / "mask", scene_dir / "novel" / "masks"):
        if path.is_dir():
            return path
    return mask_dir(scene_dir)


def static_mask(
    mask_root: Path,
    frame_id: int,
    shape: tuple[int, int],
    threshold: float,
    image_name: str | None = None,
) -> np.ndarray:
    dynamic = _read_dynamic_mask(mask_root, frame_id, shape, threshold, image_name=image_name)
    return ~dynamic


def cap_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if int(max_points) > 0 and points.shape[0] > int(max_points):
        rng = np.random.default_rng(seed)
        points = points[rng.choice(points.shape[0], int(max_points), replace=False)]
    return points


def project_rotation(mat: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(mat)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def svd_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    src_mean = np.mean(src, axis=1, keepdims=True)
    dst_mean = np.mean(dst, axis=1, keepdims=True)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    R = project_rotation(dst_c @ src_c.T)
    denom = float(np.sum(src_c * src_c))
    scale = float(np.sum(dst_c * (R @ src_c)) / max(denom, 1e-12))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    t = dst_mean - scale * R @ src_mean
    return scale, R, t.reshape(3)


def load_xm_training_poses(scene_dir: Path, n_images: int) -> dict[int, np.ndarray]:
    path = scene_dir / "outputs" / "result_XM.npz"
    if not path.is_file():
        path = scene_dir / "result_XM.npz"
    if not path.is_file():
        return {}
    data = np.load(path)
    R_all = np.asarray(data["R"], dtype=np.float64)
    t_all = np.asarray(data["t"], dtype=np.float64)
    n_frames = min(int(t_all.shape[1]), n_images, int(R_all.shape[1] // 3))
    poses = {}
    for frame_id in range(n_frames):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_all[:, 3 * frame_id : 3 * (frame_id + 1)]
        T[:3, 3] = t_all[:, frame_id]
        poses[frame_id] = T
    return poses


def xm_to_gt_similarity(xm_poses: dict[int, np.ndarray], gt_poses: dict[int, np.ndarray]) -> tuple[float, np.ndarray, np.ndarray]:
    ids = sorted(set(xm_poses) & set(gt_poses))
    if len(ids) < 2:
        raise RuntimeError(f"Need at least two XM/GT pose pairs for alignment, got {len(ids)}")
    src = np.stack([xm_poses[i][:3, 3] for i in ids], axis=1)
    dst = np.stack([gt_poses[i][:3, 3] for i in ids], axis=1)
    scale, R, t = svd_similarity(src, dst)
    R_rot = project_rotation(np.sum([gt_poses[i][:3, :3] @ xm_poses[i][:3, :3].T for i in ids], axis=0))
    pos_err = np.mean([np.linalg.norm((R @ xm_poses[i][:3, :3]) - gt_poses[i][:3, :3]) for i in ids])
    rot_err = np.mean([np.linalg.norm((R_rot @ xm_poses[i][:3, :3]) - gt_poses[i][:3, :3]) for i in ids])
    if rot_err + 1e-6 < pos_err:
        R = R_rot
        t = np.mean(dst - scale * R @ src, axis=1)
    return scale, R, t


def load_npz_depth(path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    data = np.load(path, allow_pickle=True)
    key = "rendered_depth" if "rendered_depth" in data else "pred_depth"
    depth = np.asarray(data[key], dtype=np.float64)
    valid = np.asarray(data["valid_mask"], dtype=bool) if "valid_mask" in data else None
    alpha = np.asarray(data["alpha"], dtype=np.float64) if "alpha" in data else None
    return depth, valid, alpha


def ours_train_depth_path(ours_dir: Path, frame_id: int) -> Path:
    return ours_dir / "train_depth_arrays" / f"train_depth_{int(frame_id):04d}.npz"


def ours_novel_depth_path(ours_dir: Path, novel_id: int) -> Path:
    return ours_dir / "novel" / "depth_arrays" / f"novel_depth_{int(novel_id):04d}.npz"


def sidecar_depth(prefix: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    pred_path = prefix.with_name(prefix.name + "_pred.npy")
    if not pred_path.is_file():
        raise FileNotFoundError(pred_path)
    depth = np.load(pred_path).astype(np.float64)
    valid_path = prefix.with_name(prefix.name + "_valid.npy")
    alpha_path = prefix.with_name(prefix.name + "_alpha.npy")
    valid = np.load(valid_path).astype(bool) if valid_path.is_file() else None
    alpha = np.load(alpha_path).astype(np.float64) if alpha_path.is_file() else None
    return depth, valid, alpha


def render_array_depth(prefix: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    depth_path = prefix.with_name(prefix.name + "_depth.npy")
    if not depth_path.is_file():
        raise FileNotFoundError(depth_path)
    depth = np.load(depth_path).astype(np.float64)
    valid_path = prefix.with_name(prefix.name + "_valid.npy")
    alpha_path = prefix.with_name(prefix.name + "_alpha.npy")
    valid = np.load(valid_path).astype(bool) if valid_path.is_file() else None
    alpha = np.load(alpha_path).astype(np.float64) if alpha_path.is_file() else None
    return depth, valid, alpha


def lift(
    depth: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    stride: int,
    pixel_mask: np.ndarray | None = None,
    max_points: int = 0,
    seed: int = 0,
) -> np.ndarray:
    """Backproject a depth map into world-space points.

    Samples pixels with the given stride to keep the pointcloud lightweight.

    The provided matrix must be a camera-to-world transform (c2w). The
    backprojected camera-frame points are rotated and then translated by
    the camera-to-world translation to produce world coordinates.
    """
    h, w = depth.shape
    y, x = np.mgrid[0:h:stride, 0:w:stride]

    z = depth[y, x]
    mask = np.isfinite(z) & (z > 1e-6)
    if pixel_mask is not None:
        if pixel_mask.shape != depth.shape:
            pixel_mask = cv2.resize(
                pixel_mask.astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        mask &= pixel_mask[y, x].astype(bool)

    x = x[mask].astype(float)
    y = y[mask].astype(float)
    z = z[mask].astype(float)

    cam = np.c_[
        (x - K[0, 2]) / K[0, 0] * z,
        (y - K[1, 2]) / K[1, 1] * z,
        z,
    ]

    # Apply rotation then translation: X_world = R_c2w @ X_cam + t_c2w
    rotated = (c2w[:3, :3] @ cam.T).T
    world = rotated + c2w[:3, 3]
    return cap_points(world, max_points, seed)


def pcd(points: np.ndarray, color: tuple[float, float, float]) -> o3d.geometry.PointCloud:
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(points)
    p.colors = o3d.utility.Vector3dVector(np.tile(color, (len(points), 1)))
    return p


def frustum(T: np.ndarray, K: np.ndarray, w: int, h: int, s: float, color: tuple[float, float, float]) -> o3d.geometry.LineSet:
    """Create a small camera frustum geometry in world coordinates."""
    pts = np.array(
        [
            [0, 0, 0],
            [-(K[0, 2] / K[0, 0]) * s, -(K[1, 2] / K[1, 1]) * s, s],
            [((w - K[0, 2]) / K[0, 0]) * s, -(K[1, 2] / K[1, 1]) * s, s],
            [((w - K[0, 2]) / K[0, 0]) * s, ((h - K[1, 2]) / K[1, 1]) * s, s],
            [-(K[0, 2] / K[0, 0]) * s, ((h - K[1, 2]) / K[1, 1]) * s, s],
        ]
    )

    pts = (T[:3, :3] @ pts.T).T + T[:3, 3]

    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    ls = o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(pts), o3d.utility.Vector2iVector(lines)
    )
    ls.colors = o3d.utility.Vector3dVector(np.tile(color, (len(lines), 1)))
    return ls


def make_cloud_geoms(
    train_cloud: np.ndarray,
    novel_cloud: np.ndarray,
    frustums: list,
    train_color: tuple[float, float, float],
    novel_color: tuple[float, float, float],
    voxel_size: float,
) -> list:
    train_pcd = pcd(train_cloud, train_color)
    novel_pcd = pcd(novel_cloud, novel_color)
    if float(voxel_size) > 0:
        train_pcd = train_pcd.voxel_down_sample(float(voxel_size))
        novel_pcd = novel_pcd.voxel_down_sample(float(voxel_size))
    return [train_pcd, novel_pcd] + frustums


def build_xm_geoms(
    args: argparse.Namespace,
    ours_dir: Path,
    imgs: list[Path],
    by_id: dict[int, dict],
    gt_train_poses: dict[int, np.ndarray],
    novel_dir: Path,
    train_mask_dir: Path,
    novel_masks: Path,
    novel_has_own_masks: bool,
) -> list:
    xm_train_poses = load_xm_training_poses(args.scene_dir, len(imgs))
    if not xm_train_poses:
        print("[xm-open3d] skip XM: missing outputs/result_XM.npz")
        return []
    if not ours_dir.is_dir():
        print(f"[xm-open3d] skip XM: missing ours depth dir {ours_dir}")
        return []

    summary_path = ours_dir / "gaussian_depth_eval.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            source = summary.get("camera_source")
            if source is not None and source != "xm":
                print(f"[xm-open3d] warning: saved Ours depth arrays report camera_source={source!r}, expected 'xm'")
        except Exception as exc:
            print(f"[xm-open3d] warning: could not read {summary_path}: {exc}")

    try:
        scale, R_align, t_align = xm_to_gt_similarity(xm_train_poses, gt_train_poses)
    except RuntimeError as exc:
        print(f"[xm-open3d] skip XM: {exc}")
        return []
    train_points = []
    novel_points = []
    frustums = []

    for frame_id in sorted(by_id):
        info = by_id[frame_id]
        if frame_id not in xm_train_poses or not (0 <= frame_id < len(imgs)):
            continue
        depth_path = ours_train_depth_path(ours_dir, frame_id)
        if not depth_path.is_file():
            continue
        depth, valid, alpha = load_npz_depth(depth_path)
        K = _scale_K(np.asarray(info["K"], float), img_size(imgs[frame_id]), (depth.shape[1], depth.shape[0]))
        T_xm = xm_train_poses[frame_id]
        keep_static = static_mask(train_mask_dir, frame_id, depth.shape, args.mask_threshold, image_name=imgs[frame_id].name)
        if valid is not None:
            keep_static &= valid
        if alpha is not None:
            keep_static &= alpha > float(args.alpha_threshold)
        pts_xm = lift(depth, K, T_xm, args.stride, keep_static, args.max_points_per_view, seed=200000 + frame_id)
        pts = transform_points_similarity(pts_xm, scale, R_align, t_align)
        if pts.size:
            train_points.append(pts)
        if int(args.camera_stride) > 0 and frame_id % int(args.camera_stride) == 0:
            T_gt = transform_c2w_similarity(T_xm, scale, R_align, t_align)
            frustums.append(frustum(T_gt, K, depth.shape[1], depth.shape[0], args.camera_scale, (0.95, 0.75, 0.05)))

    if novel_dir.is_dir():
        for rec in _load_novel_metadata(novel_dir):
            depth_path = ours_novel_depth_path(ours_dir, int(rec.novel_id))
            if not depth_path.is_file():
                continue
            base_frame = int(rec.base_frame)
            if base_frame not in by_id or not (0 <= base_frame < len(imgs)):
                continue
            depth, valid, alpha = load_npz_depth(depth_path)
            base = by_id[base_frame]
            K = _scale_K(np.asarray(base["K"], float), img_size(imgs[base_frame]), (depth.shape[1], depth.shape[0]))
            T_gt_novel = np.asarray(rec.c2w, dtype=np.float64)
            T_xm_novel = inverse_transform_c2w_similarity(T_gt_novel, scale, R_align, t_align)
            keep_static = static_mask(
                novel_masks,
                int(rec.novel_id) if novel_has_own_masks else base_frame,
                depth.shape,
                args.mask_threshold,
                image_name=rec.image_path.name if novel_has_own_masks else imgs[base_frame].name,
            )
            if valid is not None:
                keep_static &= valid
            if alpha is not None:
                keep_static &= alpha > float(args.alpha_threshold)
            pts_xm = lift(depth, K, T_xm_novel, args.stride, keep_static, args.max_points_per_view, seed=300000 + int(rec.novel_id))
            pts = transform_points_similarity(pts_xm, scale, R_align, t_align)
            if pts.size:
                novel_points.append(pts)
            # Novel camera aligns back to the GT novel pose after the XM->GT similarity.
            # Keep the same camera-stride behavior as training.
            if int(args.camera_stride) > 0 and int(rec.novel_id) % int(args.camera_stride) == 0:
                frustums.append(frustum(T_gt_novel, K, depth.shape[1], depth.shape[0], args.camera_scale, (0.95, 0.75, 0.05)))

    train_cloud = np.concatenate(train_points, axis=0) if train_points else np.zeros((0, 3), dtype=np.float64)
    novel_cloud = np.concatenate(novel_points, axis=0) if novel_points else np.zeros((0, 3), dtype=np.float64)
    total = train_cloud.shape[0] + novel_cloud.shape[0]
    if int(args.max_total_points) > 0 and total > int(args.max_total_points):
        train_budget = int(args.max_total_points * train_cloud.shape[0] / max(total, 1))
        novel_budget = int(args.max_total_points) - train_budget
        train_cloud = cap_points(train_cloud, train_budget, seed=3)
        novel_cloud = cap_points(novel_cloud, novel_budget, seed=4)

    geoms = make_cloud_geoms(
        train_cloud,
        novel_cloud,
        frustums,
        (0.15, 0.45, 1.0),
        (0.05, 0.95, 1.0),
        args.voxel_size,
    )
    print(
        f"[xm-open3d] train_views={len(train_points)} train_points={len(geoms[0].points)} "
        f"novel_views={len(novel_points)} novel_points={len(geoms[1].points)} "
        f"align_scale={scale:.6g} ours_dir={ours_dir}"
    )
    return geoms


def build_da3_direct_geoms(
    args: argparse.Namespace,
    imgs: list[Path],
    by_id: dict[int, dict],
    gt_train_poses: dict[int, np.ndarray],
    novel_dir: Path,
    train_mask_dir: Path,
    novel_masks: Path,
    novel_has_own_masks: bool,
) -> list:
    da_dir = args.scene_dir / "outputs" / "depth_anything_3"
    if not da_dir.is_dir():
        print(f"[da3-origin-open3d] skip: missing {da_dir}")
        return []
    try:
        depths = _load_da3_depths(da_dir)
        intrs, da_c2ws = _load_da3_cameras(da_dir)
        scale_gt_to_da, R_gt_to_da, t_gt_to_da = _gt_to_da_similarity(_load_gt(args.scene_dir), da_c2ws)
    except Exception as exc:
        print(f"[da3-origin-open3d] skip: {exc}")
        return []
    train_points = []
    novel_points = []
    frustums = []

    for frame_id in sorted(by_id):
        if frame_id >= min(depths.shape[0], da_c2ws.shape[0]) or frame_id not in gt_train_poses or not (0 <= frame_id < len(imgs)):
            continue
        depth = depths[frame_id] / max(float(scale_gt_to_da), 1e-12)
        K = _scale_K(intrs[min(frame_id, intrs.shape[0] - 1)], (depth.shape[1], depth.shape[0]), (depth.shape[1], depth.shape[0]))
        T = gt_train_poses[frame_id]
        keep_static = static_mask(train_mask_dir, frame_id, depth.shape, args.mask_threshold, image_name=imgs[frame_id].name)
        pts = lift(depth, K, T, args.stride, keep_static, args.max_points_per_view, seed=400000 + frame_id)
        if pts.size:
            train_points.append(pts)
        if int(args.camera_stride) > 0 and frame_id % int(args.camera_stride) == 0:
            frustums.append(frustum(T, K, depth.shape[1], depth.shape[0], args.camera_scale, (0.2, 0.85, 0.25)))

    if novel_dir.is_dir():
        for rec in _load_novel_metadata(novel_dir):
            base_frame = int(rec.base_frame)
            if base_frame >= min(depths.shape[0], da_c2ws.shape[0]) or base_frame not in by_id or not (0 <= base_frame < len(imgs)):
                continue
            depth = _read_depth(rec.depth_path, args.gt_scale)
            K = _scale_K(intrs[min(base_frame, intrs.shape[0] - 1)], (depths[base_frame].shape[1], depths[base_frame].shape[0]), (depth.shape[1], depth.shape[0]))
            T_gt_novel = np.asarray(rec.c2w, dtype=np.float64)
            keep_static = static_mask(
                novel_masks,
                int(rec.novel_id) if novel_has_own_masks else base_frame,
                depth.shape,
                args.mask_threshold,
                image_name=rec.image_path.name if novel_has_own_masks else imgs[base_frame].name,
            )
            # DA3 origin has no novel direct depth; use the GT novel depth only to draw the aligned DA3-origin novel camera context.
            # The actual DA3-origin cloud is from training frames.
            if int(args.camera_stride) > 0 and int(rec.novel_id) % int(args.camera_stride) == 0:
                _ = keep_static
                frustums.append(frustum(T_gt_novel, K, depth.shape[1], depth.shape[0], args.camera_scale, (0.2, 0.85, 0.25)))

    train_cloud = np.concatenate(train_points, axis=0) if train_points else np.zeros((0, 3), dtype=np.float64)
    novel_cloud = np.concatenate(novel_points, axis=0) if novel_points else np.zeros((0, 3), dtype=np.float64)
    geoms = make_cloud_geoms(train_cloud, novel_cloud, frustums, (0.1, 0.8, 0.25), (0.1, 0.8, 0.25), args.voxel_size)
    print(f"[da3-origin-open3d] train_views={len(train_points)} train_points={len(geoms[0].points)} depth_scale_da_to_gt={1.0 / max(float(scale_gt_to_da), 1e-12):.6g}")
    return geoms


def build_saved_depth_geoms(
    args: argparse.Namespace,
    name: str,
    train_loader,
    novel_loader,
    imgs: list[Path],
    by_id: dict[int, dict],
    gt_train_poses: dict[int, np.ndarray],
    novel_dir: Path,
    train_mask_dir: Path,
    novel_masks: Path,
    novel_has_own_masks: bool,
    train_color: tuple[float, float, float],
    novel_color: tuple[float, float, float],
    depth_scale: float = 1.0,
) -> list:
    train_points = []
    novel_points = []
    frustums = []
    loaded_any = False

    for frame_id in sorted(by_id):
        if frame_id not in gt_train_poses or not (0 <= frame_id < len(imgs)):
            continue
        try:
            depth, valid, alpha = train_loader(frame_id)
        except FileNotFoundError:
            continue
        loaded_any = True
        depth = depth * float(depth_scale)
        info = by_id[frame_id]
        K = _scale_K(np.asarray(info["K"], float), img_size(imgs[frame_id]), (depth.shape[1], depth.shape[0]))
        T = gt_train_poses[frame_id]
        keep_static = static_mask(train_mask_dir, frame_id, depth.shape, args.mask_threshold, image_name=imgs[frame_id].name)
        if valid is not None:
            keep_static &= valid
        if alpha is not None:
            keep_static &= alpha > float(args.alpha_threshold)
        pts = lift(depth, K, T, args.stride, keep_static, args.max_points_per_view, seed=500000 + frame_id)
        if pts.size:
            train_points.append(pts)
        if int(args.camera_stride) > 0 and frame_id % int(args.camera_stride) == 0:
            frustums.append(frustum(T, K, depth.shape[1], depth.shape[0], args.camera_scale, train_color))

    if novel_dir.is_dir():
        for rec in _load_novel_metadata(novel_dir):
            base_frame = int(rec.base_frame)
            if base_frame not in by_id or not (0 <= base_frame < len(imgs)):
                continue
            try:
                depth, valid, alpha = novel_loader(int(rec.novel_id))
            except FileNotFoundError:
                continue
            loaded_any = True
            depth = depth * float(depth_scale)
            base = by_id[base_frame]
            K = _scale_K(np.asarray(base["K"], float), img_size(imgs[base_frame]), (depth.shape[1], depth.shape[0]))
            T = np.asarray(rec.c2w, dtype=np.float64)
            keep_static = static_mask(
                novel_masks,
                int(rec.novel_id) if novel_has_own_masks else base_frame,
                depth.shape,
                args.mask_threshold,
                image_name=rec.image_path.name if novel_has_own_masks else imgs[base_frame].name,
            )
            if valid is not None:
                keep_static &= valid
            if alpha is not None:
                keep_static &= alpha > float(args.alpha_threshold)
            pts = lift(depth, K, T, args.stride, keep_static, args.max_points_per_view, seed=600000 + int(rec.novel_id))
            if pts.size:
                novel_points.append(pts)
            if int(args.camera_stride) > 0 and int(rec.novel_id) % int(args.camera_stride) == 0:
                frustums.append(frustum(T, K, depth.shape[1], depth.shape[0], args.camera_scale, novel_color))

    if not loaded_any:
        print(f"[{name}-open3d] skip: no saved depth arrays found")
        return []

    train_cloud = np.concatenate(train_points, axis=0) if train_points else np.zeros((0, 3), dtype=np.float64)
    novel_cloud = np.concatenate(novel_points, axis=0) if novel_points else np.zeros((0, 3), dtype=np.float64)
    total = train_cloud.shape[0] + novel_cloud.shape[0]
    if int(args.max_total_points) > 0 and total > int(args.max_total_points):
        train_budget = int(args.max_total_points * train_cloud.shape[0] / max(total, 1))
        novel_budget = int(args.max_total_points) - train_budget
        train_cloud = cap_points(train_cloud, train_budget, seed=7)
        novel_cloud = cap_points(novel_cloud, novel_budget, seed=8)
    geoms = make_cloud_geoms(train_cloud, novel_cloud, frustums, train_color, novel_color, args.voxel_size)
    print(f"[{name}-open3d] train_views={len(train_points)} train_points={len(geoms[0].points)} novel_views={len(novel_points)} novel_points={len(geoms[1].points)}")
    return geoms


def build_da3_render_geoms(args, imgs, by_id, gt_train_poses, novel_dir, train_mask_dir, novel_masks, novel_has_own_masks) -> list:
    da_dir = args.scene_dir / "outputs" / "depth_anything_3"
    train_dir = da_dir / "train_render_depth_arrays"
    novel_dir_arrays = da_dir / "novel_render_depth_arrays"
    depth_scale = 1.0
    try:
        _intrs, da_c2ws = _load_da3_cameras(da_dir)
        scale_gt_to_da, _R, _t = _gt_to_da_similarity(_load_gt(args.scene_dir), da_c2ws)
        depth_scale = 1.0 / max(float(scale_gt_to_da), 1e-12)
    except Exception as exc:
        print(f"[da3-render-open3d] warning: camera scale unavailable ({exc}); using scale=1")
    return build_saved_depth_geoms(
        args,
        "da3-render",
        lambda frame_id: render_array_depth(train_dir / f"train{int(frame_id):06d}"),
        lambda novel_id: render_array_depth(novel_dir_arrays / f"novel{int(novel_id):06d}"),
        imgs,
        by_id,
        gt_train_poses,
        novel_dir,
        train_mask_dir,
        novel_masks,
        novel_has_own_masks,
        (0.0, 0.65, 0.2),
        (0.0, 0.95, 0.35),
        depth_scale,
    )


def build_mosca_geoms(args, imgs, by_id, gt_train_poses, novel_dir, train_mask_dir, novel_masks, novel_has_own_masks) -> list:
    mosca_dir = args.scene_dir / "outputs_mosca" / "depth_eval"
    train_dir = mosca_dir / "training_depth"
    novel_depth_dir = mosca_dir / "novel_depth"
    depth_scale = 1.0
    summary_path = mosca_dir / "depth_l1_summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            scale = summary.get("camera_scale_render_to_gt") or summary.get("depth_scale_render_to_gt")
            if scale is None:
                inv = summary.get("camera_scale_gt_to_render") or summary.get("depth_scale_gt_to_render") or summary.get("scale_gt_to_render")
                if inv is not None and float(inv) > 1e-12:
                    scale = 1.0 / float(inv)
            if scale is not None:
                depth_scale = float(scale)
        except Exception as exc:
            print(f"[mosca-open3d] warning: could not read scale from {summary_path}: {exc}")
    return build_saved_depth_geoms(
        args,
        "mosca",
        lambda frame_id: sidecar_depth(train_dir / f"frame{int(frame_id):06d}"),
        lambda novel_id: sidecar_depth(novel_depth_dir / f"novel{int(novel_id):06d}"),
        imgs,
        by_id,
        gt_train_poses,
        novel_dir,
        train_mask_dir,
        novel_masks,
        novel_has_own_masks,
        (1.0, 0.45, 0.05),
        (1.0, 0.65, 0.05),
        depth_scale,
    )


def spatracker_eval_dir(scene_dir: Path) -> Path:
    candidates = [
        scene_dir / "outputs_spatracker" / "spatracker_eval",
        scene_dir / "outputs" / "spatracker_eval",
        scene_dir / "outputs" / "spatracker_eval_static_dynamic_smoke",
    ]
    for path in candidates:
        if (path / "training_depth").is_dir() or (path / "novel_depth").is_dir():
            return path
    return candidates[0]


def build_spatracker_geoms(args, imgs, by_id, gt_train_poses, novel_dir, train_mask_dir, novel_masks, novel_has_own_masks) -> list:
    spa_dir = spatracker_eval_dir(args.scene_dir)
    train_dir = spa_dir / "training_depth"
    novel_depth_dir = spa_dir / "novel_depth"
    geoms = build_saved_depth_geoms(
        args,
        "spatracker",
        lambda frame_id: sidecar_depth(train_dir / f"frame{int(frame_id):06d}"),
        lambda novel_id: sidecar_depth(novel_depth_dir / f"novel{int(novel_id):06d}"),
        imgs,
        by_id,
        gt_train_poses,
        novel_dir,
        train_mask_dir,
        novel_masks,
        novel_has_own_masks,
        (0.72, 0.2, 1.0),
        (0.92, 0.35, 1.0),
        1.0,
    )
    if geoms:
        print(f"[spatracker-open3d] depth_dir={spa_dir}")
    return geoms


def draw_toggle(modes: dict[str, list]) -> None:
    modes = {key: value for key, value in modes.items() if value}
    if "GT" not in modes:
        raise RuntimeError("GT geometry is required")
    order = list(modes)
    state = {"idx": 0, "key": order[0], "geoms": modes[order[0]]}

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Static depth clouds (press l to cycle)")
    for geom in state["geoms"]:
        vis.add_geometry(geom)
    print("[open3d] loaded modes: " + " -> ".join(order))
    print(f"[open3d] showing {state['key']}")

    def toggle(_vis):
        if len(order) <= 1:
            print("[open3d] no alternate geometry loaded")
            return False
        for geom in state["geoms"]:
            _vis.remove_geometry(geom, reset_bounding_box=False)
        state["idx"] = (state["idx"] + 1) % len(order)
        state["key"] = order[state["idx"]]
        state["geoms"] = modes[state["key"]]
        for geom in state["geoms"]:
            _vis.add_geometry(geom, reset_bounding_box=False)
        _vis.update_renderer()
        print(f"[open3d] showing {state['key']}")
        return False

    vis.register_key_callback(ord("L"), toggle)
    vis.register_key_callback(ord("l"), toggle)
    vis.run()
    vis.destroy_window()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", type=Path, default=Path("../datasets/Syn4D/bear1"))
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--max-points-per-view", type=int, default=20000)
    ap.add_argument("--max-total-points", type=int, default=800000)
    ap.add_argument("--voxel-size", type=float, default=0.0)
    ap.add_argument("--mask-threshold", type=float, default=127.0)
    ap.add_argument("--alpha-threshold", type=float, default=0.05)
    ap.add_argument("--camera-stride", type=int, default=1)
    ap.add_argument("--camera-scale", type=float, default=0.4)
    ap.add_argument(
        "--ours-depth-dir",
        type=Path,
        default=None,
        help="Default: <scene-dir>/outputs/gaussian_depth_eval.",
    )
    ap.add_argument(
        "--gt-scale",
        type=float,
        default=REPLICA_RENDER_DEPTH_SCALE,
        help="PNG depth divisor. ReplicaRendererMoving saves uint16(round(z * 65535 * 0.1)).",
    )
    ap.add_argument(
        "--camera-txt-convention",
        choices=["c2w", "w2c"],
        default="c2w",
        help="Convention for matrix/TUM/camera_poses_world rows. COLMAP-like rows are always inverted.",
    )

    a = ap.parse_args()

    gt = _load_gt(a.scene_dir)
    imgs = image_files(a.scene_dir)

    by_id = {int(v["id"]): v for v in gt.values()}
    train_txt_poses = load_camera_txt_poses(a.scene_dir, a.camera_txt_convention)
    novel_txt_poses = load_camera_txt_poses(a.scene_dir / "novel", a.camera_txt_convention)

    train_points = []
    novel_points = []
    gt_frustums = []
    novel_dir = a.scene_dir / "novel"
    train_mask_dir = mask_dir(a.scene_dir)
    novel_masks = novel_mask_dir(a.scene_dir)
    novel_has_own_masks = novel_masks.parent == novel_dir
    gt_train_poses: dict[int, np.ndarray] = {}

    for frame_id in sorted(by_id):
        info = by_id[frame_id]
        if not (0 <= frame_id < len(imgs)):
            continue
        depth = _read_depth(depth_path_for_frame(a.scene_dir, frame_id), a.gt_scale)
        K = _scale_K(np.asarray(info["K"], float), img_size(imgs[frame_id]), (depth.shape[1], depth.shape[0]))
        T = pose_for_id(train_txt_poses, frame_id, "training")
        if T is None:
            T = c2w(info)
        gt_train_poses[frame_id] = T
        keep_static = static_mask(train_mask_dir, frame_id, depth.shape, a.mask_threshold, image_name=imgs[frame_id].name)
        pts = lift(depth, K, T, a.stride, keep_static, a.max_points_per_view, seed=frame_id)
        if pts.size:
            train_points.append(pts)
        if int(a.camera_stride) > 0 and frame_id % int(a.camera_stride) == 0:
            gt_frustums.append(frustum(T, K, depth.shape[1], depth.shape[0], a.camera_scale, (0.0, 1.0, 0.0)))

    if novel_dir.is_dir():
        for rec in _load_novel_metadata(novel_dir):
            depth = _read_depth(rec.depth_path, a.gt_scale)
            base_frame = int(rec.base_frame)
            if base_frame not in by_id or not (0 <= base_frame < len(imgs)):
                continue
            base = by_id[base_frame]
            K = _scale_K(np.asarray(base["K"], float), img_size(imgs[base_frame]), (depth.shape[1], depth.shape[0]))
            T = pose_for_id(novel_txt_poses, int(rec.novel_id), "novel")
            if T is None:
                T = np.asarray(rec.c2w, dtype=np.float64)
            keep_static = static_mask(
                novel_masks,
                int(rec.novel_id) if novel_has_own_masks else base_frame,
                depth.shape,
                a.mask_threshold,
                image_name=rec.image_path.name if novel_has_own_masks else imgs[base_frame].name,
            )
            pts = lift(depth, K, T, a.stride, keep_static, a.max_points_per_view, seed=100000 + int(rec.novel_id))
            if pts.size:
                novel_points.append(pts)
            # if int(a.camera_stride) > 0 and int(rec.novel_id) % int(a.camera_stride) == 0:
            #     gt_frustums.append(frustum(T, K, depth.shape[1], depth.shape[0], a.camera_scale, (0.0, 1.0, 0.0)))

    train_cloud = np.concatenate(train_points, axis=0) if train_points else np.zeros((0, 3), dtype=np.float64)
    novel_cloud = np.concatenate(novel_points, axis=0) if novel_points else np.zeros((0, 3), dtype=np.float64)
    total = train_cloud.shape[0] + novel_cloud.shape[0]
    if int(a.max_total_points) > 0 and total > int(a.max_total_points):
        train_budget = int(a.max_total_points * train_cloud.shape[0] / max(total, 1))
        novel_budget = int(a.max_total_points) - train_budget
        train_cloud = cap_points(train_cloud, train_budget, seed=1)
        novel_cloud = cap_points(novel_cloud, novel_budget, seed=2)

    gt_geoms = make_cloud_geoms(
        train_cloud,
        novel_cloud,
        gt_frustums,
        (1.0, 0.15, 0.1),
        (1.0, 0.15, 0.1),
        a.voxel_size,
    )

    print(
        f"[gt-open3d] train_views={len(train_points)} train_points={len(gt_geoms[0].points)} "
        f"novel_views={len(novel_points)} novel_points={len(gt_geoms[1].points)} "
        f"stride={a.stride} mask_dir={train_mask_dir} novel_mask_dir={novel_masks}"
    )

    ours_dir = (a.ours_depth_dir or (a.scene_dir / "outputs" / "gaussian_depth_eval")).resolve()
    xm_geoms = build_xm_geoms(
        a,
        ours_dir,
        imgs,
        by_id,
        gt_train_poses,
        novel_dir,
        train_mask_dir,
        novel_masks,
        novel_has_own_masks,
    )
    da3_origin_geoms = build_da3_direct_geoms(
        a,
        imgs,
        by_id,
        gt_train_poses,
        novel_dir,
        train_mask_dir,
        novel_masks,
        novel_has_own_masks,
    )
    da3_render_geoms = build_da3_render_geoms(
        a,
        imgs,
        by_id,
        gt_train_poses,
        novel_dir,
        train_mask_dir,
        novel_masks,
        novel_has_own_masks,
    )
    mosca_geoms = build_mosca_geoms(
        a,
        imgs,
        by_id,
        gt_train_poses,
        novel_dir,
        train_mask_dir,
        novel_masks,
        novel_has_own_masks,
    )
    spatracker_geoms = build_spatracker_geoms(
        a,
        imgs,
        by_id,
        gt_train_poses,
        novel_dir,
        train_mask_dir,
        novel_masks,
        novel_has_own_masks,
    )
    draw_toggle(
        {
            "GT": gt_geoms,
            "XM/Ours": xm_geoms,
            "DA3 origin": da3_origin_geoms,
            "DA3 re-render": da3_render_geoms,
            "MoSca": mosca_geoms,
            "SpaTracker": spatracker_geoms,
        }
    )


if __name__ == "__main__":
    main()
