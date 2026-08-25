

from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2
import numpy as np

import utils.definitions
from utils.readgt_DAVIS import load_davis_gt
from utils.readgt_sintel import load_sintel_gt
from utils.readgt_colmap import load_colmap_gt
from utils.readgt_custom import load_custom_gt


def write_cameras_text(cameras, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    HEADER = "# Camera list with one line of data per camera:\n" + \
             "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n" + \
             "# Number of cameras: {}\n".format(len(cameras))
    with open(path, 'w', encoding='utf-8') as f:
        f.write(HEADER)
        for camera_id in sorted(cameras.keys()):
            camera = cameras[camera_id]
            to_write = [camera_id, camera['model'], camera['width'], camera['height']] + camera['params']
            line = " ".join([str(elem) for elem in to_write])
            f.write(line + "\n")

def write_images_text(images, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    mean_observations = 0
    # if len(images) == 0:
    #     mean_observations = 0
    # else:
    #     mean_observations = sum((len(img.point3D_ids) for _, img in images.items()))/len(images)
    HEADER = "# Image list with two lines of data per image:\n" + \
             "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n" + \
             "#   POINTS2D[] as (X, Y, POINT3D_ID)\n" + \
             "# Number of images: {}, mean observations per image: {}\n".format(len(images), mean_observations)
 
    with open(path, "w", encoding='utf-8') as fid:
        fid.write(HEADER)
        for image_id in sorted(images.keys()):
            image = images[image_id]
            image_header = [
                image['id'],
                *image['qvec'], 
                *image['tvec'], 
                image['camera_id'], 
                image['name']
            ]
            first_line = " ".join(map(str, image_header))
            fid.write(first_line + "\n")
 
            points_strings = []
            # for xy, point3D_id in zip(img.xys, img.point3D_ids):
            #     points_strings.append(" ".join(map(str, [*xy, point3D_id])))
            fid.write(" ".join(points_strings) + "\n")

def write_points3D_text(points3D, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    mean_track_length = 0
    # if len(points3D) == 0:
    #     mean_track_length = 0
    # else:
    #     mean_track_length = sum((len(pt.image_ids) for _, pt in points3D.items()))/len(points3D)
    HEADER = "# 3D point list with one line of data per point:\n" + \
             "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n" + \
             "# Number of points: {}, mean track length: {}\n".format(len(points3D), mean_track_length)
 
    with open(path, "w", encoding='utf-8') as fid:
        fid.write(HEADER)
        for pt_id in sorted(points3D.keys()):
            pt = points3D[pt_id]
            point_header = [pt["id"], *pt["xyz"], *pt["rgb"], pt["error"]]
            fid.write(" ".join(map(str, point_header)) + " ")
            track_strings = []
            # for image_id, point2D in zip(pt.image_ids, pt.point2D_idxs):
            #     track_strings.append(" ".join(map(str, [image_id, point2D])))
            fid.write(" ".join(track_strings) + "\n")


def _rotation_matrix_to_qvec(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to COLMAP quaternion [qw, qx, qy, qz]."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    K = np.array(
        [
            [R[0, 0] - R[1, 1] - R[2, 2], 0.0, 0.0, 0.0],
            [R[1, 0] + R[0, 1], R[1, 1] - R[0, 0] - R[2, 2], 0.0, 0.0],
            [R[2, 0] + R[0, 2], R[2, 1] + R[1, 2], R[2, 2] - R[0, 0] - R[1, 1], 0.0],
            [R[1, 2] - R[2, 1], R[2, 0] - R[0, 2], R[0, 1] - R[1, 0], R[0, 0] + R[1, 1] + R[2, 2]],
        ],
        dtype=np.float64,
    )
    K /= 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec = -qvec
    return qvec.astype(np.float64)


def _xm_pose_c2w_to_colmap_w2c(R_c2w: np.ndarray, t_c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert XM pose (camera->world) to COLMAP image pose (world->camera)."""
    R_c2w = np.asarray(R_c2w, dtype=np.float64).reshape(3, 3)
    t_c2w = np.asarray(t_c2w, dtype=np.float64).reshape(3)
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ t_c2w
    return R_w2c, t_w2c


def _read_ascii_ply_xyz_rgb(ply_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not ply_path.exists():
        return None
    try:
        with open(ply_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return None

    header_end = None
    prop_names = []
    n_verts = None
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("element vertex"):
            parts = line.split()
            if len(parts) >= 3:
                n_verts = int(parts[2])
        elif line.startswith("property"):
            parts = line.split()
            if len(parts) >= 3:
                prop_names.append(parts[2])
        elif line == "end_header":
            header_end = i + 1
            break

    if header_end is None or n_verts is None:
        return None

    data = lines[header_end : header_end + n_verts]
    xyz = []
    rgb = []
    for row in data:
        parts = row.strip().split()
        if len(parts) < 3:
            continue
        vals = [float(v) for v in parts[: len(prop_names)]]
        prop_map = dict(zip(prop_names, vals))
        xyz.append([prop_map.get("x", 0.0), prop_map.get("y", 0.0), prop_map.get("z", 0.0)])
        rgb.append([
            int(prop_map.get("red", 127)),
            int(prop_map.get("green", 127)),
            int(prop_map.get("blue", 127)),
        ])

    if not xyz:
        return None
    return np.asarray(xyz, dtype=np.float64), np.asarray(rgb, dtype=np.uint8)


def convertXM2COLMAP(
    rootpath: str | Path | None = None,
    result_npz_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    points3d_path: str | Path | None = None,
) -> Path:
    """Export XM reconstruction to COLMAP text model (cameras/images/points3D)."""
    root = Path(rootpath) if rootpath is not None else Path(utils.definitions.rootpath)
    result_path = Path(result_npz_path) if result_npz_path is not None else (root / "outputs" / "result_XM.npz")
    out_dir = Path(output_dir) if output_dir is not None else (root / "outputs" / "colmap_sparse_txt")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not result_path.exists():
        raise FileNotFoundError(f"Missing XM result file: {result_path}")

    dataset_cat = utils.definitions.dataset_cat
    if dataset_cat == "DAVIS":
        gt = load_davis_gt(str(root))
    elif dataset_cat in {"Sintel", "Sintel_debug"}:
        gt = load_sintel_gt(str(root))
    elif dataset_cat in {"Syn4D", "Lab"}:
        gt = load_colmap_gt(str(root / "colmap"))
    elif dataset_cat == "Custom":
        gt = load_custom_gt(root)
    else:
        raise ValueError(f"Unsupported dataset_cat '{dataset_cat}' for COLMAP export")
    if len(gt) == 0:
        raise RuntimeError("" \
        "GT/intrinsics dictionary is empty.")

    gt_items = sorted(gt.items(), key=lambda kv: int(kv[1]["id"]))
    image_names = [name for name, _ in gt_items]

    xm = np.load(str(result_path), allow_pickle=True)
    R_block = np.asarray(xm["R"], dtype=np.float64)
    t = np.asarray(xm["t"], dtype=np.float64)

    n_pose_R = R_block.shape[1] // 3
    n_pose_t = t.shape[1] if t.ndim == 2 else 0
    N = min(len(image_names), n_pose_R, n_pose_t)
    if N <= 0:
        raise RuntimeError("No valid poses found in XM result.")

    # Cameras (one camera per frame; ids must be positive for COLMAP)
    cameras: Dict[int, dict] = {}
    image_dir = root / "images"
    for i in range(N):
        rec = gt[image_names[i]]
        K = np.asarray(rec["K"], dtype=np.float64)
        cam_id = int(rec["camera_id"]) + 1
        im = cv2.imread(str(image_dir / image_names[i]), cv2.IMREAD_COLOR)
        if im is not None:
            h, w = int(im.shape[0]), int(im.shape[1])
        else:
            h = int(2.0 * K[1, 2])
            w = int(2.0 * K[0, 2])
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])
        cameras[cam_id] = {
            "model": "PINHOLE",
            "width": w,
            "height": h,
            "params": [fx, fy, cx, cy],
        }

    # Images (COLMAP expects w2c quaternion+translation)
    images: Dict[int, dict] = {}
    for i in range(N):
        R_c2w = R_block[:, 3 * i : 3 * (i + 1)]
        t_c2w = t[:, i]
        R_w2c, t_w2c = _xm_pose_c2w_to_colmap_w2c(R_c2w, t_c2w)
        qvec = _rotation_matrix_to_qvec(R_w2c)
        images[i + 1] = {
            "id": i + 1,
            "qvec": qvec,
            "tvec": t_w2c.astype(np.float64),
            "camera_id": int(gt[image_names[i]]["camera_id"]) + 1,
            "name": image_names[i],
        }

    # Sparse points
    points3D: Dict[int, dict] = {}
    ply_xyz_rgb = None
    if points3d_path is not None:
        ply_xyz_rgb = _read_ascii_ply_xyz_rgb(Path(points3d_path))

    if ply_xyz_rgb is not None:
        xyz, rgb = ply_xyz_rgb
        for i in range(xyz.shape[0]):
            points3D[i + 1] = {
                "id": i + 1,
                "xyz": xyz[i].astype(np.float64),
                "rgb": rgb[i].astype(np.uint8),
                "error": 0.0,
            }
    elif "p" in xm:
        p = np.asarray(xm["p"], dtype=np.float64)
        if p.ndim == 2 and p.shape[0] == 3:
            p = p.T
        if p.ndim == 2 and p.shape[1] == 3:
            for i in range(p.shape[0]):
                points3D[i + 1] = {
                    "id": i + 1,
                    "xyz": p[i].astype(np.float64),
                    "rgb": np.array([127, 127, 127], dtype=np.uint8),
                    "error": 0.0,
                }

    write_cameras_text(cameras, str(out_dir / "cameras.txt"))
    write_images_text(images, str(out_dir / "images.txt"))
    write_points3D_text(points3D, str(out_dir / "points3D.txt"))

    print(f"Saved COLMAP text model to: {out_dir}")
    return out_dir


if __name__ == "__main__":
    convertXM2COLMAP()
