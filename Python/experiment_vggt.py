#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pycolmap


REPO_ROOT = Path(__file__).resolve().parents[1]
VGGT_DIR = REPO_ROOT / "deps" / "vggt"
DATASET_CAT = "Replica"
DATASET_ROOT_DEFAULT = REPO_ROOT / "datasets" / DATASET_CAT
LOG_DIR_DEFAULT = REPO_ROOT / "Python" / "logs" / "vggt"

SCENES = [
    "office1_seq1",
    "office1_seq2",
    "office2_seq1",
    "office2_seq2",
    "room1_seq1",
    "room1_seq2",
    "room2_seq1",
    "room2_seq2",
]

# SCENES = [
#     "bear_0",
#     "bear_1",
#     "bear_2",
#     "bear_3",
#     "casa1",
#     "casa2",
#     "casa3",
#     "casa4"
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

# SCENES = [
#     "Capture1_clip01",
#     # "Capture1_clip02",
# ]

# SCENES = [
#     "tum1",
#     "tum2",
#     "tum3",
#     "tum4",
#     "tum5",
#     "tum6"
# ]

DEPTH_OUTPUT_DIRNAME = "depths"
DENSE_POINT_CLOUD_DIRNAME = "dense_point_cloud"
DEPTH_FILE_EXTENSIONS = {
    ".npy",
    ".npz",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".exr",
    ".pfm",
    ".pt",
    ".pth",
}
DEPTH_NAME_MARKERS = {
    "depth",
    "depths",
    "depth_map",
    "depth_maps",
    "pred_depth",
    "pred_depths",
    "predicted_depth",
    "predicted_depths",
    "disp",
    "disparity",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VGGT reconstruction on Replica, Sintel, Syn4D, Lab, or TUM scenes."
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="Scene names. Defaults to the explicit SCENES list in this file.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT_DEFAULT,
        help="Root directory containing scene folders.",
    )
    parser.add_argument(
        "--dataset-cat",
        choices=["Replica", "Sintel", "Syn4D", "Lab", "TUM", "auto"],
        default=DATASET_CAT,
        help="Dataset type to use for all scenes, or auto-detect from each scene.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Optional root output folder. If omitted, each scene writes to "
            "<scene>/output_vggt."
        ),
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=LOG_DIR_DEFAULT,
        help="Directory for per-scene stdout/stderr logs.",
    )
    parser.add_argument(
        "--vggt-python",
        default=sys.executable,
        help="Python executable used to run deps/vggt/run_replica.py.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Keep an existing sparse/log/depth output before running.",
    )
    parser.add_argument(
        "--no-save-dense-point-cloud",
        action="store_true",
        help="Do not ask VGGT to persist dense_point_cloud/dense_point_cloud.npz and sampled PLY.",
    )
    parser.add_argument(
        "--dense-point-cloud-max-points",
        type=int,
        default=2000000,
        help="Maximum points in dense_point_cloud.ply. The dense NPZ still stores the full point map.",
    )
    parser.add_argument(
        "--dense-point-cloud-conf-thres",
        type=float,
        default=0.0,
        help="Minimum VGGT depth confidence for points written to dense_point_cloud.ply.",
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


def _image_files(image_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


def _detect_dataset_cat(dataset_root: Path, scene_dir: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    if (scene_dir / "colmap").is_dir() or dataset_root.name.lower() == "syn4d":
        return "Syn4D"
    if (scene_dir / "groundtruth.txt").is_file() or dataset_root.name.lower() == "tum":
        return "TUM"
    if (scene_dir / "traj.txt").is_file() or dataset_root.name.lower() == "replica":
        return "Replica"
    if (scene_dir / "cameras").is_dir() or "sintel" in dataset_root.name.lower():
        return "Sintel"
    if (scene_dir / "masks").is_dir():
        return "Sintel"
    return "Replica"


def _scene_output_dir(scene_dir: Path, output_root: Path | None, scene_name: str) -> Path:
    if output_root is None:
        return scene_dir / "output_vggt"
    return output_root / scene_name


def _discover_scenes(dataset_root: Path) -> list[str]:
    # Keep this synchronized with experiment_colmap.py.
    if SCENES:
        return SCENES
    raise FileNotFoundError("SCENES is empty; provide --scenes or populate SCENES.")


def _clean_output_dir(output_dir: Path) -> None:
    for child in ("sparse", "log.json", DEPTH_OUTPUT_DIRNAME, "depth_conf", DENSE_POINT_CLOUD_DIRNAME):
        path = output_dir / child
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def _run_cmd(cmd: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    out = proc.stdout or ""
    if proc.stderr:
        out += "\n\n[STDERR]\n" + proc.stderr
    return proc.returncode, out


def _sparse_has_reconstruction(sparse_dir: Path) -> bool:
    binary = ["cameras.bin", "images.bin", "points3D.bin"]
    text = ["cameras.txt", "images.txt", "points3D.txt"]
    return sparse_dir.is_dir() and (
        all((sparse_dir / name).is_file() for name in binary)
        or all((sparse_dir / name).is_file() for name in text)
    )


def _write_sparse_text(sparse_dir: Path) -> dict[str, Any]:
    reconstruction = pycolmap.Reconstruction(str(sparse_dir))
    reconstruction.write_text(str(sparse_dir))
    return {
        "sparse_txt_dir": str(sparse_dir),
        "num_reg_images": reconstruction.num_reg_images(),
        "num_points3D": reconstruction.num_points3D(),
    }


def _has_depth_marker(path: Path) -> bool:
    text_parts = [part.lower() for part in path.parts]
    stem = path.stem.lower()
    return any(marker in stem for marker in DEPTH_NAME_MARKERS) or any(
        part in DEPTH_NAME_MARKERS or any(marker in part for marker in DEPTH_NAME_MARKERS)
        for part in text_parts
    )


def _depth_candidate_files(output_dir: Path) -> list[Path]:
    """Find depth files produced by deps/vggt/run_replica.py.

    This keeps the launcher independent of the exact VGGT filename convention. It
    copies any depth/disparity-like file produced under output_dir into
    output_dir/depths.
    """
    skip_top_level = {"sparse", DEPTH_OUTPUT_DIRNAME, "depth_conf"}
    candidates: list[Path] = []

    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in DEPTH_FILE_EXTENSIONS:
            continue

        try:
            rel = path.relative_to(output_dir)
        except ValueError:
            continue

        if rel.parts and rel.parts[0] in skip_top_level:
            continue
        if not _has_depth_marker(rel):
            continue

        candidates.append(path)

    return sorted(candidates)


def _depth_destination_relpath(output_dir: Path, src: Path) -> Path:
    rel = src.relative_to(output_dir)
    if rel.parts and _has_depth_marker(Path(rel.parts[0])):
        rel = Path(*rel.parts[1:]) if len(rel.parts) > 1 else Path(src.name)
    return rel


def _save_depth_outputs(output_dir: Path) -> dict[str, Any]:
    """Copy VGGT depth outputs into output_dir/depths.

    If run_replica.py already writes a directory named depth/depth_maps/etc., this
    normalizes the location to output_vggt/depths while preserving subfolders.
    If no depth files were produced, the directory is still created and the log
    records num_depth_files=0.
    """
    depth_dir = output_dir / DEPTH_OUTPUT_DIRNAME
    depth_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for src in _depth_candidate_files(output_dir):
        rel = _depth_destination_relpath(output_dir, src)
        dst = depth_dir / rel
        if src.resolve() == dst.resolve():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(str(dst))

    existing = sorted(
        path
        for path in depth_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in DEPTH_FILE_EXTENSIONS
    )
    per_frame_depths = [
        path
        for path in existing
        if path.name not in {"depth_maps.npy", "metadata.json"}
    ]

    return {
        "depth_dir": str(depth_dir),
        "num_depth_files": len(existing),
        "num_depth_frames": len(per_frame_depths),
        "copied_depth_files": len(copied),
        "depth_saved": len(existing) > 0,
    }


def run_vggt(
    scene_dir: Path,
    output_dir: Path,
    dataset_cat: str,
    vggt_python: str,
    clean: bool = True,
    save_dense_point_cloud: bool = True,
    dense_point_cloud_max_points: int = 2000000,
    dense_point_cloud_conf_thres: float = 0.0,
) -> tuple[bool, dict[str, Any], str]:
    image_dir = scene_dir / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if clean:
        _clean_output_dir(output_dir)

    sparse_dir = output_dir / "sparse"
    log: dict[str, Any] = {
        "scene_dir": str(scene_dir),
        "dataset_cat": dataset_cat,
        "image_dir": str(image_dir),
        "output_dir": str(output_dir),
        "database_path": None,
        "mask": None,
        "times_s": {},
    }

    cmd = [
        vggt_python,
        "run_replica.py",
        f"--scene_dir={scene_dir}",
        f"--output_dir={output_dir}",
    ]
    if save_dense_point_cloud:
        cmd.extend(
            [
                "--save_dense_point_cloud",
                f"--dense_point_cloud_max_points={dense_point_cloud_max_points}",
                f"--dense_point_cloud_conf_thres={dense_point_cloud_conf_thres}",
            ]
        )
    log["command"] = cmd

    t_solver = time.perf_counter()
    rc, command_log = _run_cmd(cmd, cwd=VGGT_DIR)
    log["times_s"]["preprocess"] = 0.0
    log["times_s"]["solver"] = time.perf_counter() - t_solver
    log["times_s"]["vggt"] = log["times_s"]["solver"]

    if rc != 0:
        log["status"] = f"fail ({rc})"
        return False, log, command_log

    if not _sparse_has_reconstruction(sparse_dir):
        log["status"] = "fail_no_reconstruction"
        log.update(_save_depth_outputs(output_dir))
        log.update(_dense_point_cloud_outputs(output_dir))
        return False, log, command_log

    log.update(_write_sparse_text(sparse_dir))
    log.update(_save_depth_outputs(output_dir))
    log.update(_dense_point_cloud_outputs(output_dir))
    log["num_images"] = len(_image_files(image_dir))
    log["status"] = "ok"
    return True, log, command_log


def _dense_point_cloud_outputs(output_dir: Path) -> dict[str, Any]:
    dense_dir = output_dir / DENSE_POINT_CLOUD_DIRNAME
    metadata_path = dense_dir / "metadata.json"
    npz_path = dense_dir / "dense_point_cloud.npz"
    ply_path = dense_dir / "dense_point_cloud.ply"

    result: dict[str, Any] = {
        "dense_point_cloud_dir": str(dense_dir),
        "dense_point_cloud_saved": npz_path.is_file(),
        "dense_point_cloud_npz": str(npz_path) if npz_path.is_file() else None,
        "dense_point_cloud_ply": str(ply_path) if ply_path.is_file() else None,
    }
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            result.update(
                {
                    "dense_point_cloud_num_valid_points": metadata.get("num_valid_points"),
                    "dense_point_cloud_num_ply_points": metadata.get("num_ply_points"),
                    "dense_point_cloud_shape": metadata.get("point_map_shape"),
                }
            )
        except Exception:
            result["dense_point_cloud_metadata_error"] = str(metadata_path)
    return result


def main() -> None:
    args = parse_args()
    scenes = list(dict.fromkeys(args.scenes if args.scenes is not None else _discover_scenes(args.dataset_root)))
    args.log_dir.mkdir(parents=True, exist_ok=True)
    if args.output_root is not None:
        args.output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    for scene in scenes:
        scene_dir = args.dataset_root / scene
        dataset_cat = _detect_dataset_cat(args.dataset_root, scene_dir, args.dataset_cat)
        output_dir = _scene_output_dir(scene_dir, args.output_root, scene)
        log_vggt = args.log_dir / f"{scene}__vggt.log"
        log_json = output_dir / "log.json"

        row: dict[str, Any] = {
            "scene": scene,
            "dataset_cat": dataset_cat,
            "output_dir": str(output_dir),
            "vggt_status": "planned",
        }

        if args.dry_run:
            row["vggt_status"] = "dry-run"
            rows.append(row)
            print(f"[dry-run] {dataset_cat}/{scene} -> {output_dir}")
            continue

        t0 = time.perf_counter()
        try:
            ok, run_log, command_log = run_vggt(
                scene_dir=scene_dir,
                output_dir=output_dir,
                dataset_cat=dataset_cat,
                vggt_python=args.vggt_python,
                clean=not args.no_clean,
                save_dense_point_cloud=not args.no_save_dense_point_cloud,
                dense_point_cloud_max_points=args.dense_point_cloud_max_points,
                dense_point_cloud_conf_thres=args.dense_point_cloud_conf_thres,
            )
            run_log["command_log"] = command_log
            row["vggt_time_s"] = time.perf_counter() - t0
            row["vggt_status"] = "ok" if ok else run_log["status"]
            row["depth_saved"] = bool(run_log.get("depth_saved", False))
            row["num_depth_files"] = int(run_log.get("num_depth_files", 0))
            row["dense_point_cloud_saved"] = bool(run_log.get("dense_point_cloud_saved", False))
            run_log["total_time_s"] = row["vggt_time_s"]
            run_log["status"] = row["vggt_status"]
            log_vggt.write_text(command_log, encoding="utf-8")
        except Exception as exc:
            row["vggt_time_s"] = time.perf_counter() - t0
            row["vggt_status"] = f"fail ({exc})"
            output_dir.mkdir(parents=True, exist_ok=True)
            run_log = {
                "scene_dir": str(scene_dir),
                "dataset_cat": dataset_cat,
                "output_dir": str(output_dir),
                "status": row["vggt_status"],
                "total_time_s": row["vggt_time_s"],
                "error": repr(exc),
                "command_log": None,
            }
            log_vggt.write_text(str(exc), encoding="utf-8")

        log_json.write_text(json.dumps(run_log, indent=2) + "\n", encoding="utf-8")
        rows.append(row)
        print(
            f"[{row['vggt_status']}] {dataset_cat}/{scene} "
            f"preprocess={_fmt(run_log.get('times_s', {}).get('preprocess'), 6)}s "
            f"solver={_fmt(run_log.get('times_s', {}).get('solver'), 6)}s "
            f"depths={run_log.get('num_depth_files', 0)} "
            f"dense_cloud={run_log.get('dense_point_cloud_num_ply_points', 0)} "
            f"output={output_dir}"
        )

    ok = sum(1 for row in rows if row["vggt_status"] == "ok")
    print(f"[done] {ok}/{len(rows)} scenes reconstructed")


if __name__ == "__main__":
    main()
