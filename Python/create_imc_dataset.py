import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from utils.readgt_colmap import load_camera_data, quat2rot


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare an IMC COLMAP/GLOMAP scene for Python/static_solver.py."
    )
    parser.add_argument(
        "--scene",
        type=Path,
        default=Path("datasets/IMC/temple"),
        help="IMC scene root containing images/ and sparse/.",
    )
    parser.add_argument(
        "--source",
        choices=["sparse", "legacy-output"],
        default="sparse",
        help=(
            "sparse: build observations from sparse/images.txt + points3D.txt. "
            "legacy-output: convert existing output/{edges,weights,landmarks,filename}.npy."
        ),
    )
    parser.add_argument(
        "--legacy-output",
        type=Path,
        default=None,
        help="Legacy XM-code output directory. Defaults to SCENE/output.",
    )
    parser.add_argument(
        "--depth-source-name",
        default=None,
        help="Name used in outputs/output_tapnet_<name>.npz. Defaults to sparse or unidepth.",
    )
    parser.add_argument("--max-images", type=int, default=0, help="Use only the first N sorted images.")
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth sorted image.")
    parser.add_argument("--min-depth", type=float, default=1e-6, help="Drop observations behind the camera.")
    parser.add_argument(
        "--weight-mode",
        choices=["uniform", "inverse_error", "track_length"],
        default="inverse_error",
        help="Observation weights for --source sparse.",
    )
    parser.add_argument("--weight-min", type=float, default=0.05)
    parser.add_argument("--weight-max", type=float, default=20.0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite outputs/output_tapnet_<name>.npz if it already exists.",
    )
    return parser.parse_args()


def _image_paths(scene_root):
    image_dir = scene_root / "images"
    paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return paths


def _select_names(paths, stride, max_images):
    stride = max(1, int(stride))
    selected = paths[::stride]
    if int(max_images) > 0:
        selected = selected[: int(max_images)]
    return [path.name for path in selected]


def _iter_colmap_data_lines(path):
    lines = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            lines.append(line.rstrip("\n"))
    return lines


def load_images_txt(path):
    lines = _iter_colmap_data_lines(path)
    image_by_id = {}
    image_by_name = {}
    i = 0
    while i < len(lines):
        header = lines[i].strip()
        i += 1
        if not header:
            continue
        parts = header.split()
        if len(parts) < 10:
            raise ValueError(f"Malformed COLMAP image header in {path}: {header}")

        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        camera_id = int(parts[8])
        name = " ".join(parts[9:])

        points_line = lines[i].strip() if i < len(lines) else ""
        i += 1
        point_tokens = points_line.split()
        if len(point_tokens) % 3 != 0:
            raise ValueError(f"Malformed POINTS2D row for image {name}: {points_line[:120]}")
        observations = []
        for j in range(0, len(point_tokens), 3):
            x = float(point_tokens[j])
            y = float(point_tokens[j + 1])
            point3d_id = int(point_tokens[j + 2])
            observations.append((x, y, point3d_id))

        record = {
            "image_id": image_id,
            "name": name,
            "camera_id": camera_id,
            "R": quat2rot(qw, qx, qy, qz),
            "t": np.array([tx, ty, tz], dtype=np.float64),
            "q_wxyz": np.array([qw, qx, qy, qz], dtype=np.float64),
            "observations": observations,
        }
        image_by_id[image_id] = record
        image_by_name[name] = record
    return image_by_id, image_by_name


def load_points3d_txt(path):
    points = {}
    for line in _iter_colmap_data_lines(path):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        point_id = int(parts[0])
        xyz = np.array(list(map(float, parts[1:4])), dtype=np.float64)
        rgb = np.array(list(map(float, parts[4:7])), dtype=np.float64)
        error = float(parts[7])
        track_len = max(0, (len(parts) - 8) // 2)
        points[point_id] = {
            "xyz": xyz,
            "rgb": rgb,
            "error": error,
            "track_len": track_len,
        }
    return points


def _point_weight(point, mode, weight_min, weight_max):
    if mode == "uniform":
        weight = 1.0
    elif mode == "track_length":
        weight = np.sqrt(max(float(point["track_len"]), 1.0))
    else:
        error = max(float(point["error"]), 0.0)
        weight = 1.0 / (1.0 + error * error)
    return float(np.clip(weight, float(weight_min), float(weight_max)))


def build_from_sparse(scene_root, selected_names, args):
    sparse_dir = scene_root / "sparse"
    _, image_by_name = load_images_txt(sparse_dir / "images.txt")
    points = load_points3d_txt(sparse_dir / "points3D.txt")

    image_name_to_frame = {name: i for i, name in enumerate(selected_names)}
    point_id_to_solver = {}
    edges = []
    weights = []
    landmarks = []
    rgbs = []
    used_images = 0

    for name in selected_names:
        record = image_by_name.get(name)
        if record is None:
            continue
        frame_idx = image_name_to_frame[name]
        R = record["R"]
        t = record["t"]
        image_obs = 0
        for _x, _y, point3d_id in record["observations"]:
            if point3d_id < 0:
                continue
            point = points.get(point3d_id)
            if point is None:
                continue
            pc = R @ point["xyz"] + t
            if not np.isfinite(pc).all() or pc[2] <= float(args.min_depth):
                continue
            solver_point = point_id_to_solver.setdefault(point3d_id, len(point_id_to_solver))
            edges.append([solver_point, frame_idx])
            weights.append([_point_weight(point, args.weight_mode, args.weight_min, args.weight_max)])
            landmarks.append(pc.tolist())
            rgbs.append(point["rgb"].tolist())
            image_obs += 1
        used_images += int(image_obs > 0)

    return (
        np.asarray(edges, dtype=np.int64),
        np.asarray(weights, dtype=np.float64),
        np.asarray(landmarks, dtype=np.float64),
        np.asarray(rgbs, dtype=np.float64),
        {
            "source": "sparse",
            "sparse_dir": str(sparse_dir),
            "used_images": used_images,
            "source_points": len(points),
            "solver_points": len(point_id_to_solver),
        },
    )


def build_from_legacy(scene_root, selected_names, args):
    legacy_dir = args.legacy_output or (scene_root / "output")
    edges_old = np.load(legacy_dir / "edges.npy")
    weights_old = np.load(legacy_dir / "weights.npy")
    landmarks_old = np.load(legacy_dir / "landmarks.npy")
    filename = np.load(legacy_dir / "filename.npy", allow_pickle=True)
    rgbs_old_path = legacy_dir / "rgbs.npy"
    rgbs_old = np.load(rgbs_old_path) if rgbs_old_path.is_file() else None

    selected_set = set(selected_names)
    image_name_to_frame = {name: i for i, name in enumerate(selected_names)}
    old_point_to_solver = {}
    edges = []
    weights = []
    landmarks = []
    rgbs = []

    if weights_old.ndim == 1:
        weights_old = weights_old[:, None]

    for obs_idx, (frame_1, point_1) in enumerate(np.asarray(edges_old, dtype=np.int64)):
        old_frame = int(frame_1) - 1
        old_point = int(point_1) - 1
        if old_frame < 0 or old_frame >= filename.shape[0]:
            continue
        name = str(filename[old_frame])
        if name not in selected_set:
            continue
        pc = np.asarray(landmarks_old[obs_idx], dtype=np.float64)
        if not np.isfinite(pc).all() or pc[2] <= float(args.min_depth):
            continue
        solver_point = old_point_to_solver.setdefault(old_point, len(old_point_to_solver))
        edges.append([solver_point, image_name_to_frame[name]])
        weights.append(weights_old[obs_idx].astype(np.float64).reshape(-1)[:1].tolist())
        landmarks.append(pc.tolist())
        if rgbs_old is not None:
            rgbs.append(np.asarray(rgbs_old[obs_idx], dtype=np.float64).tolist())

    rgbs_arr = np.asarray(rgbs, dtype=np.float64) if rgbs else np.zeros((len(edges), 3), dtype=np.float64)
    return (
        np.asarray(edges, dtype=np.int64),
        np.asarray(weights, dtype=np.float64),
        np.asarray(landmarks, dtype=np.float64),
        rgbs_arr,
        {
            "source": "legacy-output",
            "legacy_output": str(legacy_dir),
            "legacy_observations": int(edges_old.shape[0]),
            "solver_points": len(old_point_to_solver),
        },
    )


def write_tum_like_manifests(scene_root, selected_names):
    sparse_dir = scene_root / "sparse"
    cameras = load_camera_data(str(sparse_dir / "cameras.txt"))
    _, image_by_name = load_images_txt(sparse_dir / "images.txt")

    with (scene_root / "rgb.txt").open("w", encoding="utf-8") as f:
        f.write("# color images\n")
        f.write(f"# source: {scene_root}\n")
        f.write("# timestamp filename\n")
        for idx, name in enumerate(selected_names):
            f.write(f"{idx:.6f} images/{name}\n")

    with (scene_root / "groundtruth.txt").open("w", encoding="utf-8") as f:
        f.write("# COLMAP camera trajectory exported in TUM format\n")
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for idx, name in enumerate(selected_names):
            record = image_by_name.get(name)
            if record is None:
                continue
            R_cw = record["R"]
            t_cw = record["t"]
            R_wc = R_cw.T
            center = -R_wc @ t_cw
            qx, qy, qz, qw = Rotation.from_matrix(R_wc).as_quat()
            f.write(
                f"{idx:.6f} {center[0]:.12g} {center[1]:.12g} {center[2]:.12g} "
                f"{qx:.12g} {qy:.12g} {qz:.12g} {qw:.12g}\n"
            )

    camera_json = {}
    for name in selected_names:
        record = image_by_name.get(name)
        if record is None or record["camera_id"] not in cameras.index:
            continue
        row = cameras.loc[record["camera_id"]]
        K = np.asarray(row["K"], dtype=np.float64)
        camera_json[name] = {
            "camera_id": int(record["camera_id"]),
            "width": int(row["width"]),
            "height": int(row["height"]),
            "K": K.tolist(),
        }
    with (scene_root / "cameras.json").open("w", encoding="utf-8") as f:
        json.dump(camera_json, f, indent=2)


def main():
    args = parse_args()
    scene_root = args.scene.resolve()
    image_paths = _image_paths(scene_root)
    selected_names = _select_names(image_paths, args.stride, args.max_images)
    if not selected_names:
        raise RuntimeError("No selected images after applying stride/max-images.")

    depth_source_name = args.depth_source_name
    if depth_source_name is None:
        depth_source_name = "unidepth" if args.source == "legacy-output" else "sparse"

    output_dir = scene_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_npz = output_dir / f"output_tapnet_{depth_source_name}.npz"
    if out_npz.exists() and not args.overwrite:
        raise FileExistsError(f"{out_npz} already exists. Pass --overwrite to replace it.")

    if args.source == "sparse":
        edges, weights, landmarks, rgbs, summary = build_from_sparse(scene_root, selected_names, args)
    else:
        edges, weights, landmarks, rgbs, summary = build_from_legacy(scene_root, selected_names, args)

    if edges.size == 0:
        raise RuntimeError("No valid IMC observations were produced.")
    if weights.ndim == 1:
        weights = weights[:, None]

    np.savez_compressed(
        out_npz,
        edges=edges,
        weights=weights,
        landmarks=landmarks,
        rgbs=rgbs,
        image_names=np.asarray(selected_names, dtype=object),
    )
    write_tum_like_manifests(scene_root, selected_names)

    summary.update(
        {
            "scene": str(scene_root),
            "depth_source_name": depth_source_name,
            "output_npz": str(out_npz),
            "images_selected": len(selected_names),
            "observations": int(edges.shape[0]),
            "weights_shape": list(weights.shape),
            "landmarks_shape": list(landmarks.shape),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    with (output_dir / f"imc_{depth_source_name}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
