#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any

# # Keep BLAS/OpenMP libraries from spawning hundreds of workers before numpy loads.
# for _thread_env in (
#     "OPENBLAS_NUM_THREADS",
#     "OMP_NUM_THREADS",
#     "MKL_NUM_THREADS",
#     "NUMEXPR_NUM_THREADS",
#     "VECLIB_MAXIMUM_THREADS",
# ):
#     os.environ.setdefault(_thread_env, "1")

import numpy as np
import pycolmap
from PIL import Image

from utils.readgt_replica import load_replica_camera
from utils.readgt_sintel import load_sintel_gt
from utils.readgt_colmap import load_colmap_camera, load_colmap_gt
from utils.readgt_TUM import load_tum_camera, load_tum_gt
from utils.readgt_dycheck import load_dycheck_camera

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_CAT = "Syn4D"  # "Replica", "Sintel", "Syn4D", "TUM" or "Lab", 
DATASET_ROOT_DEFAULT = REPO_ROOT / "datasets" / DATASET_CAT
LOG_DIR_DEFAULT = REPO_ROOT / "Python" / "logs" / "colmap_pycolmap"

# SCENES = [
#     "office1_seq1",
#     "office1_seq2",
#     "office2_seq1",
#     "office2_seq2",
#     "room1_seq1",
#     "room1_seq2",
#     "room2_seq1",
#     "room2_seq2",
# ]

# SCENES = [
#     "alley_1",
#     "alley_2",
#     "ambush_2",
#     "ambush_4",
#     "ambush_5",
#     "ambush_6",
#     "ambush_7",
#     "bamboo_1",
#     "bamboo_2",
#     "bandage_1",
#     "bandage_2",
#     "cave_2",
#     "cave_4",
#     "market_2",
#     "market_5",
#     "market_6",
#     "mountain_1",
#     "shaman_2",
#     "shaman_3",
#     "sleeping_1",
#     "sleeping_2",
#     "temple_2",
#     "temple_3"
# ]

SCENES = [
    "bear1",
    "bear2",
    "bear3",
    "bear4",
    "casa1",
    "casa2",
    "casa3",
    "casa4"
]

# SCENES = [
#     # "Capture1_clip01",
#     # "Capture1_clip02",
#     "Capture1_clip01_new",
#     "Capture2_clip01_new",
#     "Capture3_clip01_new",
#     "Capture4_clip01_new",
# ]

# SCENES = [
#     "tum1",
#     "tum2",
#     "tum3",
#     "tum4",
#     "tum5",
#     "tum6"
# ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pycolmap reconstruction on Replica, Sintel, Syn4D, Lab, or TUM scenes."
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="Scene names. Defaults to all scene folders with an images/ directory.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT_DEFAULT,
        help="Root directory containing scene folders.",
    )
    parser.add_argument(
        "--dataset-cat",
        choices=["Replica", "Sintel", "Syn4D", "Lab", "TUM", "Dycheck"],
        default=DATASET_CAT,
        help="Dataset type. Auto detects from dataset-root/scene layout.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Optional root output folder. If omitted, each scene writes to "
            "<scene>/output_colmap."
        ),
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=LOG_DIR_DEFAULT,
        help="Directory for per-scene logs.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Keep an existing database/mapper/sparse/mask output before running.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Maximum pycolmap worker threads for feature extraction, matching, and mapping.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _fmt(v: Any, digits: int = 6) -> str:
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.{digits}g}"
    except Exception:
        return str(v)


def _pick_best_reconstruction(reconstructions: dict[int, pycolmap.Reconstruction]) -> pycolmap.Reconstruction | None:
    if not reconstructions:
        return None
    # pick reconstruction with most registered images
    return max(reconstructions.values(), key=lambda r: r.num_reg_images())


def _image_files(image_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

def _scene_output_dir(scene_dir: Path, output_root: Path | None, scene_name: str) -> Path:
    if output_root is None:
        return scene_dir / "output_colmap"
    return output_root / scene_name


def _discover_scenes(dataset_root: Path) -> list[str]:
    # Use explicit SCENES list; do not auto-discover.
    if SCENES:
        return SCENES
    raise FileNotFoundError("SCENES is empty; provide --scenes or populate SCENES.")


def _clean_output_dir(output_dir: Path) -> None:
    for child in ("database.db", "mapper", "sparse", "colmap_masks", "log.json"):
        path = output_dir / child
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def _database_stats(database_path: Path) -> dict[str, Any]:
    if not database_path.is_file():
        return {}

    stats: dict[str, Any] = {}
    with sqlite3.connect(database_path) as con:
        for table in ("images", "keypoints", "matches", "two_view_geometries"):
            stats[f"{table}_count"] = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

        keypoint_rows = [
            row[0]
            for row in con.execute("SELECT rows FROM keypoints ORDER BY rows").fetchall()
        ]
        geometry_rows = [
            row[0]
            for row in con.execute(
                "SELECT rows FROM two_view_geometries WHERE rows > 0 ORDER BY rows"
            ).fetchall()
        ]

    if keypoint_rows:
        stats["keypoints_min"] = int(keypoint_rows[0])
        stats["keypoints_median"] = float(np.median(keypoint_rows))
        stats["keypoints_max"] = int(keypoint_rows[-1])
    if geometry_rows:
        stats["verified_pairs"] = len(geometry_rows)
        stats["verified_inliers_min"] = int(geometry_rows[0])
        stats["verified_inliers_median"] = float(np.median(geometry_rows))
        stats["verified_inliers_max"] = int(geometry_rows[-1])
    else:
        stats["verified_pairs"] = 0

    return stats


def _resolve_mask_path(
    image_path: Path,
    image_dir: Path,
    dynamic_mask_dir: Path,
    indexed_masks: list[Path],
    frame_index: int,
) -> Path | None:
    rel = image_path.relative_to(image_dir)
    candidates = [
        dynamic_mask_dir / rel,
        dynamic_mask_dir / rel.name,
        dynamic_mask_dir / f"{image_path.stem}.png",
        dynamic_mask_dir / f"{frame_index:05d}.png",
        dynamic_mask_dir / f"{frame_index:06d}.png",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if frame_index < len(indexed_masks):
        return indexed_masks[frame_index]
    return None


def prepare_dynamic_masks_for_colmap(
    image_dir: Path,
    dynamic_mask_dir: Path,
    colmap_mask_dir: Path,
    dynamic_is_white: bool = True,
) -> dict[str, Any]:
    """
    Convert dynamic-object masks to COLMAP masks.

    COLMAP expects:
        0   = ignore
        255 = use

    If dynamic_is_white=True:
        input dynamic mask: 255 dynamic, 0 static
        output COLMAP mask: 0 dynamic, 255 static
    """

    colmap_mask_dir.mkdir(parents=True, exist_ok=True)
    image_paths = _image_files(image_dir)
    indexed_masks = _image_files(dynamic_mask_dir)
    written = 0
    missing: list[str] = []

    for frame_index, image_path in enumerate(image_paths):
        rel = image_path.relative_to(image_dir)
        dynamic_mask_path = _resolve_mask_path(
            image_path=image_path,
            image_dir=image_dir,
            dynamic_mask_dir=dynamic_mask_dir,
            indexed_masks=indexed_masks,
            frame_index=frame_index,
        )

        if dynamic_mask_path is None:
            missing.append(str(rel))
            continue

        mask = Image.open(dynamic_mask_path).convert("L")
        with Image.open(image_path) as image:
            if mask.size != image.size:
                mask = mask.resize(image.size, Image.Resampling.NEAREST)
        mask_np = np.array(mask)

        # Binary dynamic mask
        dynamic = mask_np > 0

        if dynamic_is_white:
            # input: white = dynamic
            # COLMAP: black = ignore dynamic, white = keep static
            colmap_mask = np.where(dynamic, 0, 255).astype(np.uint8)
        else:
            # input: black = dynamic, white = static
            colmap_mask = np.where(dynamic, 255, 0).astype(np.uint8)

        # COLMAP expects mask path = image relative path + ".png"
        # image: 000000.png -> mask: 000000.png.png
        out_path = colmap_mask_dir / Path(str(rel) + ".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        Image.fromarray(colmap_mask).save(out_path)
        written += 1

    return {
        "source_dir": str(dynamic_mask_dir),
        "colmap_mask_dir": str(colmap_mask_dir),
        "images": len(image_paths),
        "written": written,
        "missing": missing,
    }


def _configure_reader_options(
    scene_dir: Path,
    dataset_cat: str,
    mask_dir: Path | None,
) -> tuple[pycolmap.ImageReaderOptions, pycolmap.CameraMode, str]:
    image_reader_options = pycolmap.ImageReaderOptions()
    camera_mode = pycolmap.CameraMode.AUTO
    camera_model = "SIMPLE_RADIAL"

    if mask_dir is not None:
        image_reader_options.mask_path = str(mask_dir)

    if dataset_cat == "Replica":
        camera_info = load_replica_camera(str(scene_dir))
        if len(camera_info) == 1:
            params = camera_info[1]["params"]
            image_reader_options.camera_params = ",".join(map(str, params))
            camera_mode = pycolmap.CameraMode.SINGLE
            camera_model = camera_info[1].get("model", "PINHOLE")
    elif dataset_cat == "Syn4D":
        camera_info = load_colmap_camera(str(scene_dir / "colmap"))
        if len(camera_info) == 1:
            params = camera_info[1]["params"]
            image_reader_options.camera_params = ",".join(map(str, params))
            camera_mode = pycolmap.CameraMode.SINGLE
            camera_model = camera_info[1].get("model", "PINHOLE")
    elif dataset_cat == "Dycheck":
        camera_info = load_dycheck_camera(str(scene_dir))
        if len(camera_info) == 1:
            params = camera_info[1]["params"]
            image_reader_options.camera_params = ",".join(map(str, params))
            camera_mode = pycolmap.CameraMode.SINGLE
            camera_model = camera_info[1].get("model", "PINHOLE")
    elif dataset_cat == "Sintel":
        gt = load_sintel_gt(str(scene_dir))
        if gt:
            first = next(iter(gt.values()))
            k = np.asarray(first["K"], dtype=float)
            image_reader_options.camera_params = ",".join(
                map(str, [k[0, 0], k[1, 1], k[0, 2], k[1, 2]])
            )
            camera_mode = pycolmap.CameraMode.SINGLE
            camera_model = "PINHOLE"
    elif dataset_cat == "TUM":
        camera_info = load_tum_camera(str(scene_dir))
        if len(camera_info) == 1:
            params = camera_info[1]["params"]
            image_reader_options.camera_params = ",".join(map(str, params))
            camera_mode = pycolmap.CameraMode.SINGLE
            camera_model = camera_info[1].get("model", "PINHOLE")

    return image_reader_options, camera_mode, camera_model


def run_pycolmap(
    scene_dir: Path,
    output_dir: Path,
    dataset_cat: str,
    dynamic_mask_dir: Path | None = None,
    clean: bool = True,
    num_threads: int = 4,
    allow_mask_fallback: bool = True,
) -> tuple[pycolmap.Reconstruction | None, dict[str, Any]]:
    image_dir = scene_dir / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if clean:
        _clean_output_dir(output_dir)

    database_path = output_dir / "database.db"
    sparse_dir = output_dir / "sparse"
    log: dict[str, Any] = {
        "scene_dir": str(scene_dir),
        "dataset_cat": dataset_cat,
        "image_dir": str(image_dir),
        "output_dir": str(output_dir),
        "database_path": str(database_path),
        "mask": None,
        "num_threads": num_threads,
        "times_s": {},
    }

    colmap_mask_dir: Path | None = None

    if dynamic_mask_dir is not None:
        colmap_mask_dir = output_dir / "colmap_masks"
        log["mask"] = prepare_dynamic_masks_for_colmap(
            image_dir=image_dir,
            dynamic_mask_dir=dynamic_mask_dir,
            colmap_mask_dir=colmap_mask_dir,
            dynamic_is_white=True,   # change this if your sign is opposite
        )
        if log["mask"]["written"] == 0:
            colmap_mask_dir = None
            log["mask"]["used"] = False
            log["mask"]["warning"] = "No masks matched image files; running without masks."
        else:
            log["mask"]["used"] = True

    image_reader_options, camera_mode, camera_model = _configure_reader_options(
        scene_dir=scene_dir,
        dataset_cat=dataset_cat,
        mask_dir=colmap_mask_dir,
    )
    log["camera_model"] = camera_model
    log["camera_mode"] = str(camera_mode)

    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.num_threads = num_threads
    if dataset_cat in ("Sintel", "Syn4D", "TUM", "Dycheck"):
        extraction_options.sift.max_num_features = 16384
        extraction_options.sift.peak_threshold = 0.003

    t_extract = time.perf_counter()
    pycolmap.extract_features(
        str(database_path),
        str(image_dir),
        camera_mode=camera_mode,
        camera_model=camera_model,
        reader_options=image_reader_options,
        extraction_options=extraction_options,
    )
    log["times_s"]["extract_features"] = time.perf_counter() - t_extract

    matching_options = pycolmap.FeatureMatchingOptions()
    matching_options.num_threads = num_threads
    if dataset_cat in ("Sintel", "Syn4D", "TUM", "Dycheck"):
        matching_options.guided_matching = True
        matching_options.sift.max_ratio = 0.9
        matching_options.sift.max_distance = 0.9

    t_match = time.perf_counter()
    pycolmap.match_exhaustive(str(database_path), matching_options=matching_options)
    log["times_s"]["match_exhaustive"] = time.perf_counter() - t_match
    log["times_s"]["preprocess"] = (
        log["times_s"]["extract_features"] + log["times_s"]["match_exhaustive"]
    )
    log["database_stats"] = _database_stats(database_path)

    mapping_output = output_dir / "mapper"
    mapping_output.mkdir(parents=True, exist_ok=True)
    mapping_options = pycolmap.IncrementalPipelineOptions()
    mapping_options.num_threads = num_threads
    mapping_options.mapper.num_threads = num_threads
    if dataset_cat in ("Sintel", "Syn4D", "TUM", "Dycheck"):
        mapping_options.min_num_matches = 8
        mapping_options.ba_refine_focal_length = False
        mapping_options.ba_refine_extra_params = False
        mapping_options.mapper.init_min_num_inliers = 20
        mapping_options.mapper.init_min_tri_angle = 0.5
        mapping_options.mapper.init_max_forward_motion = 0.999
        mapping_options.mapper.abs_pose_min_num_inliers = 15
        mapping_options.mapper.ba_local_min_tri_angle = 0.5
        mapping_options.mapper.filter_min_tri_angle = 0.1
        mapping_options.triangulation.ignore_two_view_tracks = False
        mapping_options.triangulation.min_angle = 0.1
    log["mapping_options"] = {
        "min_num_matches": mapping_options.min_num_matches,
        "init_min_num_inliers": mapping_options.mapper.init_min_num_inliers,
        "init_min_tri_angle": mapping_options.mapper.init_min_tri_angle,
        "init_max_forward_motion": mapping_options.mapper.init_max_forward_motion,
        "triangulation_min_angle": mapping_options.triangulation.min_angle,
    }

    t_mapping = time.perf_counter()
    reconstructions = pycolmap.incremental_mapping(
        str(database_path),
        str(image_dir),
        str(mapping_output),
        options=mapping_options,
    )
    log["times_s"]["incremental_mapping"] = time.perf_counter() - t_mapping
    log["times_s"]["solver"] = log["times_s"]["incremental_mapping"]

    reconstruction = _pick_best_reconstruction(reconstructions)
    if reconstruction is None:
        if colmap_mask_dir is not None and allow_mask_fallback:
            log["status"] = "fail_no_reconstruction_with_masks"
            log["mask"]["fallback"] = "retry_without_masks"
            return run_pycolmap(
                scene_dir=scene_dir,
                output_dir=output_dir,
                dataset_cat=dataset_cat,
                dynamic_mask_dir=None,
                clean=True,
                num_threads=num_threads,
                allow_mask_fallback=False,
            )
        log["status"] = "fail_no_reconstruction"
        return None, log

    sparse_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.write_text(str(sparse_dir))
    log["sparse_txt_dir"] = str(sparse_dir)
    log["num_reg_images"] = reconstruction.num_reg_images()
    log["num_images"] = len(_image_files(image_dir))
    log["num_points3D"] = reconstruction.num_points3D()
    log["status"] = "ok"
    return reconstruction, log


def main() -> None:
    args = parse_args()
    scenes = list(dict.fromkeys(args.scenes if args.scenes is not None else _discover_scenes(args.dataset_root)))
    args.log_dir.mkdir(parents=True, exist_ok=True)
    if args.output_root is not None:
        args.output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    for scene in scenes:
        scene_dir = args.dataset_root / scene
        dataset_cat = args.dataset_cat
        output_dir = _scene_output_dir(scene_dir, args.output_root, scene)
        log_colmap = args.log_dir / f"{scene}__colmap.log"
        log_json = output_dir / "log.json"
        dynamic_mask_dir = scene_dir / "masks"
        if not dynamic_mask_dir.is_dir():
            dynamic_mask_dir = None

        row: dict[str, Any] = {
            "scene": scene,
            "dataset_cat": dataset_cat,
            "output_dir": str(output_dir),
            "colmap_status": "planned",
        }

        if args.dry_run:
            row["colmap_status"] = "dry-run"
            rows.append(row)
            print(f"[dry-run] {dataset_cat}/{scene} -> {output_dir}")
            continue

        t0 = time.perf_counter()
        try:
            reconstruction, run_log = run_pycolmap(
                scene_dir=scene_dir,
                output_dir=output_dir,
                dataset_cat=dataset_cat,
                dynamic_mask_dir=dynamic_mask_dir,
                clean=not args.no_clean,
                num_threads=max(1, args.num_threads),
            )
            row["colmap_time_s"] = time.perf_counter() - t0
            if reconstruction is None:
                row["colmap_status"] = "fail (no reconstruction)"
            else:
                row["colmap_status"] = "ok"
            run_log["total_time_s"] = row["colmap_time_s"]
            run_log["status"] = row["colmap_status"]
        except Exception as exc:
            row["colmap_time_s"] = time.perf_counter() - t0
            row["colmap_status"] = f"fail ({exc})"
            output_dir.mkdir(parents=True, exist_ok=True)
            run_log = {
                "scene_dir": str(scene_dir),
                "dataset_cat": dataset_cat,
                "output_dir": str(output_dir),
                "status": row["colmap_status"],
                "total_time_s": row["colmap_time_s"],
                "error": repr(exc),
            }
            log_colmap.write_text(str(exc), encoding="utf-8")
            log_json.write_text(json.dumps(run_log, indent=2) + "\n", encoding="utf-8")
            rows.append(row)
            print(f"[{row['colmap_status']}] {scene}")
            continue

        log_colmap.write_text("ok" if row["colmap_status"] == "ok" else "failed", encoding="utf-8")
        log_json.write_text(json.dumps(run_log, indent=2) + "\n", encoding="utf-8")

        rows.append(row)
        print(
            f"[{row['colmap_status']}] {dataset_cat}/{scene} "
            f"preprocess={_fmt(run_log.get('times_s', {}).get('preprocess'), 6)}s "
            f"solver={_fmt(run_log.get('times_s', {}).get('solver'), 6)}s "
            f"output={output_dir}"
        )

    ok = sum(1 for row in rows if row["colmap_status"] == "ok")
    print(f"[done] {ok}/{len(rows)} scenes reconstructed")


if __name__ == "__main__":
    main()
