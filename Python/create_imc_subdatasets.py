import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from utils.readgt_IMC import load_imc_gt
from utils.readgt_colmap import load_camera_data


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser(description="Create 200-image IMC subdatasets from IMC_raw.")
    parser.add_argument("--raw-root", type=Path, default=Path("datasets/IMC_raw"))
    parser.add_argument("--out-root", type=Path, default=Path("datasets/IMC"))
    parser.add_argument("--images-per-subset", type=int, default=200)
    parser.add_argument("--subsets-per-scene", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-projection-side", type=int, default=960)
    parser.add_argument("--max-points-per-check", type=int, default=2500)
    return parser.parse_args()


def iter_data_lines(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            yield line.rstrip("\n")


def read_images_txt(path):
    lines = list(iter_data_lines(path))
    records_by_name = {}
    records_by_id = {}
    i = 0
    while i < len(lines):
        header = lines[i]
        i += 1
        if i > len(lines):
            raise ValueError(f"Missing POINTS2D row after image header: {header}")
        points_line = lines[i] if i < len(lines) else ""
        i += 1

        parts = header.split()
        if len(parts) < 10:
            raise ValueError(f"Malformed image row in {path}: {header}")
        image_id = int(parts[0])
        camera_id = int(parts[8])
        name = " ".join(parts[9:])
        point_tokens = points_line.split()
        if len(point_tokens) % 3 != 0:
            raise ValueError(f"Malformed POINTS2D row for {name}")

        points2d = []
        point3d_ids = set()
        for j in range(0, len(point_tokens), 3):
            x = float(point_tokens[j])
            y = float(point_tokens[j + 1])
            point3d_id = int(point_tokens[j + 2])
            points2d.append((x, y, point3d_id))
            if point3d_id >= 0:
                point3d_ids.add(point3d_id)

        record = {
            "header": header,
            "image_id": image_id,
            "camera_id": camera_id,
            "name": name,
            "points2d": points2d,
            "point3d_ids": point3d_ids,
        }
        records_by_name[name] = record
        records_by_id[image_id] = record
    return records_by_name, records_by_id


def read_points3d_txt(path):
    points = {}
    for line in iter_data_lines(path):
        parts = line.split()
        if len(parts) < 8:
            continue
        point3d_id = int(parts[0])
        track = []
        for j in range(8, len(parts), 2):
            if j + 1 >= len(parts):
                break
            track.append((int(parts[j]), int(parts[j + 1])))
        points[point3d_id] = {
            "xyz": np.array(list(map(float, parts[1:4])), dtype=np.float64),
            "rgb": tuple(map(int, parts[4:7])),
            "error": float(parts[7]),
            "track": track,
            "raw": parts,
        }
    return points


def image_paths(scene_dir):
    image_dir = scene_dir / "images"
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)


def link_or_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_zero_mask(image_path, mask_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image for mask: {image_path}")
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mask_path), np.zeros(image.shape[:2], dtype=np.uint8))


def format_float(value):
    return f"{float(value):.17g}"


def filter_points_for_subset(points_by_id, selected_image_ids, observed_point_ids):
    kept_points = {}
    for point_id in observed_point_ids:
        point = points_by_id.get(point_id)
        if point is None:
            continue
        filtered_track = [(iid, pidx) for iid, pidx in point["track"] if iid in selected_image_ids]
        if not filtered_track:
            continue
        kept_points[point_id] = {**point, "track": filtered_track}
    return kept_points


def write_cameras_txt(src_path, dst_path, used_camera_ids):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with src_path.open("r", encoding="utf-8") as src, dst_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if line.startswith("#") or not line.strip():
                dst.write(line)
                continue
            camera_id = int(line.split()[0])
            if camera_id in used_camera_ids:
                dst.write(line)


def write_images_txt(dst_path, selected_records, kept_point_ids):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(selected_records)}\n")
        for record in selected_records:
            f.write(record["header"] + "\n")
            tokens = []
            for x, y, point3d_id in record["points2d"]:
                out_id = point3d_id if point3d_id in kept_point_ids else -1
                tokens.extend([format_float(x), format_float(y), str(out_id)])
            f.write(" ".join(tokens) + "\n")


def write_points3d_txt(dst_path, kept_points):
    with dst_path.open("w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(kept_points)}\n")
        for point_id in sorted(kept_points):
            point = kept_points[point_id]
            x, y, z = point["xyz"]
            r, g, b = point["rgb"]
            tokens = [
                str(point_id),
                format_float(x),
                format_float(y),
                format_float(z),
                str(r),
                str(g),
                str(b),
                format_float(point["error"]),
            ]
            for image_id, point2d_idx in point["track"]:
                tokens.extend([str(image_id), str(point2d_idx)])
            f.write(" ".join(tokens) + "\n")


def project_points_for_record(record, camera_df, points_by_id):
    pose = record["header"].split()
    qw, qx, qy, qz = map(float, pose[1:5])
    tx, ty, tz = map(float, pose[5:8])
    camera_id = int(pose[8])
    K = np.asarray(camera_df.loc[camera_id, "K"], dtype=np.float64)
    R = quat_to_rot(qw, qx, qy, qz)
    t = np.array([tx, ty, tz], dtype=np.float64)

    observed = []
    projected = []
    colors = []
    for x_obs, y_obs, point_id in record["points2d"]:
        point = points_by_id.get(point_id)
        if point is None:
            continue
        pc = R @ point["xyz"] + t
        if not np.isfinite(pc).all() or pc[2] <= 1e-8:
            continue
        uvw = K @ pc
        projected.append([uvw[0] / uvw[2], uvw[1] / uvw[2]])
        observed.append([x_obs, y_obs])
        colors.append(point["rgb"])
    return np.asarray(observed), np.asarray(projected), np.asarray(colors, dtype=np.uint8)


def quat_to_rot(qw, qx, qy, qz):
    return np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ],
        dtype=np.float64,
    )


def draw_projection_check(image_path, record, camera_df, points_by_id, dst_path, max_side, max_points):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    observed, projected, colors = project_points_for_record(record, camera_df, points_by_id)
    h, w = image.shape[:2]
    inside = (
        (projected[:, 0] >= 0)
        & (projected[:, 0] < w)
        & (projected[:, 1] >= 0)
        & (projected[:, 1] < h)
    ) if projected.size else np.zeros((0,), dtype=bool)
    observed = observed[inside]
    projected = projected[inside]
    colors = colors[inside]

    if projected.shape[0] > max_points:
        idx = np.linspace(0, projected.shape[0] - 1, max_points).astype(np.int64)
        observed = observed[idx]
        projected = projected[idx]
        colors = colors[idx]

    overlay = image.copy()
    for obs, proj, color in zip(observed, projected, colors, strict=False):
        color_bgr = tuple(int(v) for v in color[::-1])
        p = tuple(np.rint(proj).astype(int))
        o = tuple(np.rint(obs).astype(int))
        cv2.circle(overlay, p, 2, color_bgr, -1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, o, 2, (0, 0, 255), 1, lineType=cv2.LINE_AA)
        if np.linalg.norm(np.asarray(p) - np.asarray(o)) > 1.5:
            cv2.line(overlay, o, p, (0, 255, 255), 1, lineType=cv2.LINE_AA)

    scale = min(1.0, float(max_side) / max(h, w)) if max_side > 0 else 1.0
    if scale < 1.0:
        overlay = cv2.resize(overlay, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_path), overlay)

    reproj_err = np.linalg.norm(projected - observed, axis=1) if projected.size else np.zeros((0,))
    return {
        "image": image_path.name,
        "points": int(projected.shape[0]),
        "mean_reprojection_px": float(np.mean(reproj_err)) if reproj_err.size else None,
        "median_reprojection_px": float(np.median(reproj_err)) if reproj_err.size else None,
        "png": str(dst_path),
    }


def create_subset(raw_scene, out_scene, selected_names, args, image_records_by_name, points_by_id):
    if out_scene.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out_scene} exists; pass --overwrite to replace it.")
        shutil.rmtree(out_scene)

    (out_scene / "images").mkdir(parents=True)
    (out_scene / "masks").mkdir()
    (out_scene / "outputs").mkdir()
    (out_scene / "sparse").mkdir()

    selected_records = [image_records_by_name[name] for name in selected_names]
    selected_image_ids = {record["image_id"] for record in selected_records}
    observed_point_ids = set().union(*(record["point3d_ids"] for record in selected_records))
    kept_points = filter_points_for_subset(points_by_id, selected_image_ids, observed_point_ids)
    kept_point_ids = set(kept_points)
    used_camera_ids = {record["camera_id"] for record in selected_records}

    for name in selected_names:
        src = raw_scene / "images" / name
        dst = out_scene / "images" / name
        link_or_copy(src, dst)
        write_zero_mask(dst, out_scene / "masks" / f"{Path(name).stem}.png")

    write_cameras_txt(raw_scene / "sparse" / "cameras.txt", out_scene / "sparse" / "cameras.txt", used_camera_ids)
    write_images_txt(out_scene / "sparse" / "images.txt", selected_records, kept_point_ids)
    write_points3d_txt(out_scene / "sparse" / "points3D.txt", kept_points)

    gt = load_imc_gt(out_scene)
    if len(gt) != len(selected_names):
        raise RuntimeError(f"{out_scene}: readgt_IMC loaded {len(gt)} / {len(selected_names)} cameras.")

    camera_df = load_camera_data(str(out_scene / "sparse" / "cameras.txt"))
    check_dir = out_scene / "outputs" / "projection_checks"
    checks = []
    for idx, record in enumerate(selected_records):
        checks.append(
            draw_projection_check(
                out_scene / "images" / record["name"],
                record,
                camera_df,
                kept_points,
                check_dir / f"{idx:06d}_{Path(record['name']).stem}.png",
                args.max_projection_side,
                args.max_points_per_check,
            )
        )

    summary = {
        "raw_scene": str(raw_scene),
        "scene": str(out_scene),
        "images": len(selected_names),
        "masks": len(list((out_scene / "masks").glob("*.png"))),
        "cameras_loaded_by_readgt_IMC": len(gt),
        "points3D": len(kept_points),
        "projection_checks_dir": str(check_dir),
        "projection_check_example": checks[0]["png"] if checks else None,
        "mean_reprojection_px": float(np.mean([c["mean_reprojection_px"] for c in checks if c["mean_reprojection_px"] is not None])),
        "median_reprojection_px": float(np.median([c["median_reprojection_px"] for c in checks if c["median_reprojection_px"] is not None])),
    }
    with (out_scene / "outputs" / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump({**summary, "projection_checks": checks}, f, indent=2)
    return summary


def main():
    args = parse_args()
    raw_root = args.raw_root
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for raw_scene in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        image_records_by_name, _ = read_images_txt(raw_scene / "sparse" / "images.txt")
        points_by_id = read_points3d_txt(raw_scene / "sparse" / "points3D.txt")
        names = [path.name for path in image_paths(raw_scene) if path.name in image_records_by_name]
        for subset_idx in range(args.subsets_per_scene):
            start = subset_idx * args.images_per_subset
            end = start + args.images_per_subset
            selected = names[start:end]
            if len(selected) < args.images_per_subset:
                print(f"[skip] {raw_scene.name}_{subset_idx:02d}: only {len(selected)} images")
                continue
            out_scene = out_root / f"{raw_scene.name}_{subset_idx:02d}"
            summary = create_subset(raw_scene, out_scene, selected, args, image_records_by_name, points_by_id)
            summaries.append(summary)
            print(json.dumps(summary, indent=2))

    with (out_root / "imc_subdatasets_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"Wrote {len(summaries)} IMC subdatasets under {out_root}")


if __name__ == "__main__":
    main()
