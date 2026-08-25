#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import pycolmap

from utils.readgt_replica import load_replica_camera


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "datasets" / "Replica"
LOG_DIR_DEFAULT = REPO_ROOT / "Python" / "logs" / "replica_pycolmap"
REPORT_DEFAULT = REPO_ROOT / "Python" / "replica_pycolmap_accuracy.md"
EVAL_SCRIPT = REPO_ROOT / "Python" / "eval_replica_colmap_vs_gt.py"

SCENES = [
    "office1_seq1",
    # "office1_seq2",
    # "office2_seq1",
    # "office2_seq2",
    # "room1_seq1",
    # "room1_seq2",
    # "room2_seq1",
    # "room2_seq2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pycolmap reconstruction on Replica scenes and evaluate against GT."
    )
    parser.add_argument("--scenes", nargs="*", default=SCENES)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "pycolmap_replica",
        help="Root output folder for COLMAP reconstructions.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=LOG_DIR_DEFAULT,
        help="Directory for per-scene logs.",
    )
    parser.add_argument(
        "--eval-conda",
        default="/home/fenchel/miniconda3/bin/conda",
        help="Conda executable to run evaluator in XM env.",
    )
    parser.add_argument(
        "--eval-env",
        default="XM",
        help="Conda environment name for eval_replica_colmap_vs_gt.py.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORT_DEFAULT,
        help="Markdown report output path.",
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


def _parse_result_json(text: str) -> dict[str, Any] | None:
    m = re.search(r"RESULT_JSON:\s*(\{.*\})", text)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def _run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    out = proc.stdout or ""
    if proc.stderr:
        out += "\n\n[STDERR]\n" + proc.stderr
    return proc.returncode, out


def _pick_best_reconstruction(reconstructions: dict[int, pycolmap.Reconstruction]) -> pycolmap.Reconstruction | None:
    if not reconstructions:
        return None
    # pick reconstruction with most registered images
    return max(reconstructions.values(), key=lambda r: r.num_reg_images())


def run_pycolmap(scene_dir: Path, output_dir: Path) -> pycolmap.Reconstruction | None:
    image_dir = scene_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_dir / "database.db"
    sparse_dir = output_dir / "sparse"

    camera_info = load_replica_camera(str(scene_dir))
    image_reader_options = pycolmap.ImageReaderOptions()

    if len(camera_info) == 1:
        params = camera_info[1]["params"]
        image_reader_options.camera_params = ",".join(map(str, params))
        pycolmap.extract_features(
            str(database_path),
            str(image_dir),
            camera_mode=pycolmap.CameraMode.SINGLE,
            camera_model="PINHOLE",
            reader_options=image_reader_options,
        )
    else:
        pycolmap.extract_features(str(database_path), str(image_dir))

    pycolmap.match_exhaustive(str(database_path))

    mapping_output = output_dir / "mapper"
    mapping_output.mkdir(parents=True, exist_ok=True)
    reconstructions = pycolmap.incremental_mapping(
        str(database_path),
        str(image_dir),
        str(mapping_output),
    )
    reconstruction = _pick_best_reconstruction(reconstructions)
    if reconstruction is None:
        return None

    sparse_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.write(str(sparse_dir))
    return reconstruction


def write_report(rows: list[dict[str, Any]], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Replica pycolmap Accuracy Report",
        "",
        "Per-scene pipeline:",
        "1. pycolmap extract + exhaustive match + incremental mapping",
        "2. eval_replica_colmap_vs_gt.py against Replica GT",
        "",
        "## Summary",
        "",
        "| Scene | COLMAP status | Eval status | Matched | RPE-R | RPE-T | ATE-R | ATE-T | ATE (m) | RPE trans (m) | RPE rot (deg) | COLMAP time (s) | Eval time (s) | Output dir |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for row in rows:
        lines.append(
            "| {scene} | {c_status} | {e_status} | {matched} | {rpe_r} | {rpe_t} | {ate_r} | {ate_t} | {m_ate} | {m_rpe_t} | {m_rpe_r} | {c_t} | {e_t} | `{out_dir}` |".format(
                scene=row["scene"],
                c_status=row["colmap_status"],
                e_status=row["eval_status"],
                matched=row.get("matched_frames", "N/A"),
                rpe_r=_fmt(row.get("RPE_R"), 6),
                rpe_t=_fmt(row.get("RPE_T"), 6),
                ate_r=_fmt(row.get("ATE_R"), 6),
                ate_t=_fmt(row.get("ATE_T"), 6),
                m_ate=_fmt(row.get("MOSCA_ATE_M"), 6),
                m_rpe_t=_fmt(row.get("MOSCA_RPE_TRANS_M"), 6),
                m_rpe_r=_fmt(row.get("MOSCA_RPE_ROT_DEG"), 6),
                c_t=_fmt(row.get("colmap_time_s"), 6),
                e_t=_fmt(row.get("eval_time_s"), 6),
                out_dir=Path(row["output_dir"]).relative_to(REPO_ROOT),
            )
        )

    lines += ["", "## Per-scene metric blocks", ""]
    for row in rows:
        lines += [
            f"### {row['scene']}",
            "",
            f"COLMAP status: **{row['colmap_status']}**",
            f"Eval status: **{row['eval_status']}**",
            "",
            "```text",
            (
                f"RPE-R:  {row.get('RPE_R', 'N/A')} "
                f"RPE-T:  {row.get('RPE_T', 'N/A')} "
                f"ATE-R:  {row.get('ATE_R', 'N/A')} "
                f"ATE-T:  {row.get('ATE_T', 'N/A')}"
            ),
            "",
            "============================================================",
            "MoSca-style eval (evo)",
            "============================================================",
            f"ATE ↓:        {row.get('MOSCA_ATE_M', 'N/A')} m",
            f"RPE trans ↓:  {row.get('MOSCA_RPE_TRANS_M', 'N/A')} m",
            f"RPE rot ↓:    {row.get('MOSCA_RPE_ROT_DEG', 'N/A')}°",
            "============================================================",
            "```",
            "",
        ]

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    scenes = list(dict.fromkeys(args.scenes))
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    for scene in scenes:
        scene_dir = DATASET_ROOT / scene
        output_dir = scene_dir / "output_colmap"
        log_colmap = args.log_dir / f"{scene}__colmap.log"
        log_eval = args.log_dir / f"{scene}__eval.log"

        row: dict[str, Any] = {
            "scene": scene,
            "output_dir": str(output_dir),
            "colmap_status": "planned",
            "eval_status": "planned",
        }

        if args.dry_run:
            row["colmap_status"] = "dry-run"
            row["eval_status"] = "dry-run"
            rows.append(row)
            print(f"[dry-run] {scene} -> {output_dir}")
            continue

        t0 = time.perf_counter()
        try:
            reconstruction = run_pycolmap(scene_dir, output_dir)
            row["colmap_time_s"] = time.perf_counter() - t0
            if reconstruction is None:
                row["colmap_status"] = "fail (no reconstruction)"
            else:
                row["colmap_status"] = "ok"
        except Exception as exc:
            row["colmap_time_s"] = time.perf_counter() - t0
            row["colmap_status"] = f"fail ({exc})"
            log_colmap.write_text(str(exc), encoding="utf-8")
            row["eval_status"] = "skipped"
            rows.append(row)
            print(f"[{row['colmap_status']}] {scene}")
            continue

        log_colmap.write_text("ok" if row["colmap_status"] == "ok" else "failed", encoding="utf-8")

        if row["colmap_status"] != "ok":
            row["eval_status"] = "skipped"
            rows.append(row)
            continue

        eval_cmd = [
            args.eval_conda,
            "run",
            "-n",
            args.eval_env,
            "--no-capture-output",
            "python",
            str(EVAL_SCRIPT),
            "--scene-dir",
            str(scene_dir),
            "--sparse-dir",
            str(output_dir / "sparse"),
            "--json",
        ]
        
        print(eval_cmd)

        t1 = time.perf_counter()
        rc_eval, out_eval = _run(eval_cmd, cwd=REPO_ROOT)
        row["eval_time_s"] = time.perf_counter() - t1
        log_eval.write_text(out_eval, encoding="utf-8")
        row["eval_status"] = "ok" if rc_eval == 0 else f"fail ({rc_eval})"

        parsed = _parse_result_json(out_eval)
        if parsed is not None:
            row.update(parsed)

        rows.append(row)
        print(
            f"[{row['colmap_status']}|{row['eval_status']}] {scene} "
            f"RPE-R={_fmt(row.get('RPE_R'), 6)} ATE-T={_fmt(row.get('ATE_T'), 6)}"
        )

    write_report(rows, args.report)
    print(f"[done] Wrote report: {args.report}")


if __name__ == "__main__":
    main()
