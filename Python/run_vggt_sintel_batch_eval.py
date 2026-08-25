#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
VGGT_DIR = REPO_ROOT / "deps" / "vggt"
EVAL_SCRIPT = REPO_ROOT / "Python" / "eval_colmap.py"
LOG_DIR_DEFAULT = REPO_ROOT / "Python" / "logs" / "vggt_sintel_batch"
REPORT_DEFAULT = REPO_ROOT / "Python" / "vggt_sintel_accuracy.md"
SINTEL_ROOT_DEFAULT = REPO_ROOT / "datasets" / "Sintel"
VGGT_PYTHON_CANDIDATE = REPO_ROOT / "deps" / "mega-sam" / ".conda-env" / "bin" / "python"
VGGT_ENV_PREFIX = VGGT_PYTHON_CANDIDATE.parents[1] if VGGT_PYTHON_CANDIDATE.is_file() else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VGGT reconstruction for Sintel scenes and evaluate COLMAP poses vs GT."
    )
    parser.add_argument(
        "--sintel-root",
        type=Path,
        default=SINTEL_ROOT_DEFAULT,
        help="Root directory containing Sintel scene folders.",
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="Sintel scene names (default: all folders under sintel-root except 'template').",
    )
    default_python = str(VGGT_PYTHON_CANDIDATE) if VGGT_PYTHON_CANDIDATE.is_file() else "python"
    parser.add_argument(
        "--vggt-python",
        default=default_python,
        help="Python executable used to run deps/vggt/run_replica.py.",
    )
    parser.add_argument(
        "--vggt-conda",
        default=None,
        help="Optional conda executable to run VGGT in a conda environment.",
    )
    parser.add_argument(
        "--vggt-env-prefix",
        type=Path,
        default=VGGT_ENV_PREFIX,
        help="Conda env prefix for VGGT when using --vggt-conda.",
    )
    parser.add_argument(
        "--xm-conda",
        default="/home/fenchel/miniconda3/bin/conda",
        help="Path to conda executable used for XM eval environment.",
    )
    parser.add_argument("--xm-env", default="XM", help="Conda env name for evaluator script.")
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR_DEFAULT, help="Directory to store logs.")
    parser.add_argument("--report", type=Path, default=REPORT_DEFAULT, help="Markdown output report path.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
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


def _run_cmd(cmd: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    out = (proc.stdout or "")
    if proc.stderr:
        out = out + "\n\n[STDERR]\n" + proc.stderr
    return proc.returncode, out


def _discover_scenes(root: Path) -> list[str]:
    if not root.exists():
        return []
    scenes = [p.name for p in root.iterdir() if p.is_dir() and p.name != "template"]
    return sorted(scenes)


def _has_sparse_recon(output_dir: Path) -> bool:
    sparse_dir = output_dir / "sparse"
    required = ["cameras.bin", "images.bin", "points3D.bin"]
    return sparse_dir.is_dir() and all((sparse_dir / name).is_file() for name in required)


def write_report(rows: list[dict[str, Any]], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# VGGT Sintel Accuracy Report",
        "",
        "Pipeline per scene:",
        "1. Run `deps/vggt/run_replica.py` to generate COLMAP sparse reconstruction.",
        "2. Run `Python/eval_colmap.py` for solver-style metrics against Sintel GT.",
        "",
        "## Summary",
        "",
        "| Scene | Run status | Eval status | Matched | RPE-R | RPE-T | ATE-R | ATE-T | ATE (m) | RPE trans (m) | RPE rot (deg) | VGGT solver time (s) | Eval time (s) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in rows:
        lines.append(
            "| {scene} | {run_status} | {eval_status} | {matched} | {rpe_r} | {rpe_t} | {ate_r} | {ate_t} | {mosca_ate} | {mosca_rpe_t} | {mosca_rpe_r} | {vggt_t} | {eval_t} |".format(
                scene=row["scene"],
                run_status=row["run_status"],
                eval_status=row["eval_status"],
                matched=row.get("matched_frames", "N/A"),
                rpe_r=_fmt(row.get("RPE_R"), 6),
                rpe_t=_fmt(row.get("RPE_T"), 6),
                ate_r=_fmt(row.get("ATE_R"), 6),
                ate_t=_fmt(row.get("ATE_T"), 6),
                mosca_ate=_fmt(row.get("MOSCA_ATE_M"), 6),
                mosca_rpe_t=_fmt(row.get("MOSCA_RPE_TRANS_M"), 6),
                mosca_rpe_r=_fmt(row.get("MOSCA_RPE_ROT_DEG"), 6),
                vggt_t=_fmt(row.get("vggt_solver_time_s", row.get("vggt_time_s")), 6),
                eval_t=_fmt(row.get("eval_time_s"), 6),
            )
        )

    lines += ["", "## Per-scene metric blocks", ""]
    for row in rows:
        lines += [
            f"### {row['scene']}",
            "",
            f"- Run status: **{row['run_status']}**",
            f"- Eval status: **{row['eval_status']}**",
            f"- VGGT log: `{Path(row['vggt_log']).relative_to(REPO_ROOT)}`",
            f"- Eval log: `{Path(row['eval_log']).relative_to(REPO_ROOT)}`",
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
    args.log_dir.mkdir(parents=True, exist_ok=True)

    scenes = args.scenes
    if scenes is None:
        scenes = _discover_scenes(args.sintel_root)
    scenes = list(dict.fromkeys(scenes))

    rows: list[dict[str, Any]] = []

    for scene in scenes:
        scene_dir = args.sintel_root / scene
        vggt_log = args.log_dir / f"{scene}__vggt.log"
        eval_log = args.log_dir / f"{scene}__eval.log"
        eval_md = args.log_dir / f"{scene}__eval.md"
        output_dir = scene_dir / "output_vggt"

        row: dict[str, Any] = {
            "scene": scene,
            "run_status": "planned",
            "eval_status": "planned",
            "vggt_log": str(vggt_log),
            "eval_log": str(eval_log),
        }

        if args.vggt_conda is not None:
            vggt_cmd = [
                args.vggt_conda,
                "run",
                "-p",
                str(args.vggt_env_prefix),
                "python",
                "run_replica.py",
                f"--scene_dir=../../datasets/Sintel/{scene}/",
                f"--output_dir=../../datasets/Sintel/{scene}/output_vggt",
            ]
        else:
            vggt_cmd = [
                args.vggt_python,
                "run_replica.py",
                f"--scene_dir=../../datasets/Sintel/{scene}/",
                f"--output_dir=../../datasets/Sintel/{scene}/output_vggt",
            ]
        eval_cmd = [
            args.xm_conda,
            "run",
            "-n",
            args.xm_env,
            "--no-capture-output",
            "python",
            str(EVAL_SCRIPT),
            "--scene-dir",
            str(scene_dir),
            "--gt-type",
            "sintel",
            "--sparse-dir",
            str(output_dir / "sparse"),
            "--json",
            "--output-md",
            str(eval_md),
        ]

        if args.dry_run:
            row["run_status"] = "dry-run"
            row["eval_status"] = "dry-run"
            rows.append(row)
            print(f"[dry-run] {scene}: {' '.join(vggt_cmd)}")
            print(f"[dry-run] {scene}: {' '.join(eval_cmd)}")
            continue

        if _has_sparse_recon(output_dir):
            row["run_status"] = "skipped-existing"
            vggt_log.write_text("[skip] Existing sparse reconstruction found.\n", encoding="utf-8")
        else:
            t0 = time.perf_counter()
            rc_vggt, out_vggt = _run_cmd(vggt_cmd, cwd=VGGT_DIR)
            row["vggt_time_s"] = time.perf_counter() - t0
            row["vggt_solver_time_s"] = row["vggt_time_s"]
            vggt_log.write_text(out_vggt, encoding="utf-8")
            row["run_status"] = "ok" if rc_vggt == 0 else f"fail ({rc_vggt})"

        if row["run_status"].startswith("fail"):
            row["eval_status"] = "skipped"
            rows.append(row)
            print(f"[{row['run_status']}] {scene} -> {vggt_log}")
            continue

        t1 = time.perf_counter()
        rc_eval, out_eval = _run_cmd(eval_cmd, cwd=REPO_ROOT)
        row["eval_time_s"] = time.perf_counter() - t1
        eval_log.write_text(out_eval, encoding="utf-8")
        row["eval_status"] = "ok" if rc_eval == 0 else f"fail ({rc_eval})"

        parsed = _parse_result_json(out_eval)
        if parsed is not None:
            row.update(parsed)

        rows.append(row)
        print(
            f"[{row['run_status']}|{row['eval_status']}] {scene} "
            f"RPE-R={_fmt(row.get('RPE_R'), 6)} ATE-T={_fmt(row.get('ATE_T'), 6)}"
        )

    write_report(rows, args.report)
    print(f"[done] Wrote report: {args.report}")


if __name__ == "__main__":
    main()