#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "Python"
DATASET_ROOT = REPO_ROOT / "datasets" / "Replica"
LOG_ROOT = PYTHON_DIR / "logs" / "replica_solver_16"
REPORT_PATH = PYTHON_DIR / "replica_solver_16_results.md"

DATASETS = [
    "office1_seq1",
    "office1_seq2",
    "office2_seq1",
    "office2_seq2",
    "room1_seq1",
    "room1_seq2",
    "room2_seq1",
    "room2_seq2",
]


@dataclass(frozen=True)
class ExperimentMode:
    key: str
    label: str
    depth_source: str
    use_ceres: bool


DEFAULT_MODES = [
    ExperimentMode(
        key="with_depth",
        label="with depth (DA3, no Ceres)",
        depth_source="da3",
        use_ceres=False,
    ),
    ExperimentMode(
        key="without_depth",
        label="without depth (GT, with Ceres)",
        depth_source="gt",
        use_ceres=True,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run static_solver.py for Replica 8 scenes × 2 modes and generate a markdown summary."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=DATASETS,
        help="Replica datasets to run (default: all 8).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=LOG_ROOT,
        help="Directory to store per-run logs.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORT_PATH,
        help="Markdown output path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands only.",
    )
    parser.add_argument(
        "--depth-sources",
        nargs="+",
        choices=["da3", "gt"],
        default=["da3", "gt"],
        help="Depth source(s) to include in runs.",
    )
    parser.add_argument(
        "--ceres-mode",
        choices=["on", "off", "both"],
        default="both",
        help="Run with Ceres on/off/both.",
    )
    return parser.parse_args()


def _select_modes(depth_sources: list[str], ceres_mode: str) -> list[ExperimentMode]:
    depth_sources_set = set(depth_sources)
    selected: list[ExperimentMode] = []

    for mode in DEFAULT_MODES:
        if mode.depth_source not in depth_sources_set:
            continue
        if ceres_mode == "on" and not mode.use_ceres:
            continue
        if ceres_mode == "off" and mode.use_ceres:
            continue
        selected.append(mode)

    if ceres_mode in {"on", "off"}:
        use_ceres = ceres_mode == "on"
        for depth in sorted(depth_sources_set):
            if any(m.depth_source == depth and m.use_ceres == use_ceres for m in selected):
                continue
            key = f"{depth}_{'ceres' if use_ceres else 'noceres'}"
            label = f"depth={depth}, {'with' if use_ceres else 'no'} Ceres"
            selected.append(
                ExperimentMode(
                    key=key,
                    label=label,
                    depth_source=depth,
                    use_ceres=use_ceres,
                )
            )

    return selected


def _extract_float(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text, flags=re.MULTILINE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _parse_stdout_metrics(stdout_text: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "RPE_R": _extract_float(r"RPE-R:\s*([0-9eE+\-\.]+)", stdout_text),
        "RPE_T": _extract_float(r"RPE-T:\s*([0-9eE+\-\.]+)", stdout_text),
        "ATE_R": _extract_float(r"ATE-R:\s*([0-9eE+\-\.]+)", stdout_text),
        "ATE_T": _extract_float(r"ATE-T:\s*([0-9eE+\-\.]+)", stdout_text),
        "MOSCA_ATE_M": _extract_float(r"ATE\s*↓:\s*([0-9eE+\-\.]+)\s*m", stdout_text),
        "MOSCA_RPE_TRANS_M": _extract_float(r"RPE trans\s*↓:\s*([0-9eE+\-\.]+)\s*m", stdout_text),
        "MOSCA_RPE_ROT_DEG": _extract_float(r"RPE rot\s*↓:\s*([0-9eE+\-\.]+)°", stdout_text),
        "TOTAL_TIME": _extract_float(r"Total time:\s*([0-9eE+\-\.]+)", stdout_text),
        "SOLVE_TIME": _extract_float(r"Solve time:\s*([0-9eE+\-\.]+)", stdout_text),
        "OPTIMAL_RANK": _extract_float(r"Optimal rank:\s*([0-9eE+\-\.]+)", stdout_text),
    }

    result_json = None
    m = re.search(r"RESULT_JSON:\s*(\{.*\})", stdout_text)
    if m:
        try:
            result_json = json.loads(m.group(1))
        except Exception:
            result_json = None

    if result_json is not None:
        for key in ["RPE_R", "RPE_T", "ATE_R", "ATE_T", "total_time", "solve_time", "optimal_rank"]:
            if key in result_json:
                out_key = {
                    "total_time": "TOTAL_TIME",
                    "solve_time": "SOLVE_TIME",
                    "optimal_rank": "OPTIMAL_RANK",
                }.get(key, key)
                metrics[out_key] = result_json[key]

    return metrics


def _fmt(v: Any, digits: int = 12) -> str:
    if v is None:
        return "N/A"
    try:
        vf = float(v)
        return f"{vf:.{digits}g}"
    except Exception:
        return str(v)


def _build_metric_block(row: dict[str, Any]) -> str:
    return (
        f"RPE-R:  {_fmt(row.get('RPE_R'))} "
        f"RPE-T:  {_fmt(row.get('RPE_T'))} "
        f"ATE-R:  {_fmt(row.get('ATE_R'))} "
        f"ATE-T:  {_fmt(row.get('ATE_T'))}\n\n"
        "============================================================\n"
        "MoSca-style eval (evo)\n"
        "============================================================\n"
        f"ATE ↓:        {_fmt(row.get('MOSCA_ATE_M'), 6)} m\n"
        f"RPE trans ↓:  {_fmt(row.get('MOSCA_RPE_TRANS_M'), 6)} m\n"
        f"RPE rot ↓:    {_fmt(row.get('MOSCA_RPE_ROT_DEG'), 6)}°\n"
        "============================================================\n"
        f"Total time:  {_fmt(row.get('TOTAL_TIME'))}\n"
        f"Solve time:  {_fmt(row.get('SOLVE_TIME'))}\n"
        f"Optimal rank:  {int(float(row.get('OPTIMAL_RANK'))) if row.get('OPTIMAL_RANK') is not None else 'N/A'}"
    )


def write_report(rows: list[dict[str, Any]], report_path: Path, modes: list[ExperimentMode]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    n_datasets = len({row.get("dataset") for row in rows})
    mode_desc = ", ".join(m.label for m in modes) if modes else "(none)"

    lines = [
        f"# Replica Solver Experiments ({n_datasets} datasets × {len(modes)} mode(s))",
        "",
        "Run configuration:",
        f"- Modes: {mode_desc}",
        "",
        "## Summary table",
        "",
        "| Dataset | Mode | Status | RPE-R | RPE-T | ATE-R | ATE-T | ATE (m) | RPE trans (m) | RPE rot (deg) | Total (s) | Solve (s) | Rank | Log |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for row in rows:
        lines.append(
            "| {dataset} | {mode} | {status} | {rpe_r} | {rpe_t} | {ate_r} | {ate_t} | {mosca_ate} | {mosca_rpe_t} | {mosca_rpe_r} | {total_t} | {solve_t} | {rank} | `{log}` |".format(
                dataset=row["dataset"],
                mode=row["mode_label"],
                status=row["status"],
                rpe_r=_fmt(row.get("RPE_R"), 6),
                rpe_t=_fmt(row.get("RPE_T"), 6),
                ate_r=_fmt(row.get("ATE_R"), 6),
                ate_t=_fmt(row.get("ATE_T"), 6),
                mosca_ate=_fmt(row.get("MOSCA_ATE_M"), 6),
                mosca_rpe_t=_fmt(row.get("MOSCA_RPE_TRANS_M"), 6),
                mosca_rpe_r=_fmt(row.get("MOSCA_RPE_ROT_DEG"), 6),
                total_t=_fmt(row.get("TOTAL_TIME"), 6),
                solve_t=_fmt(row.get("SOLVE_TIME"), 6),
                rank=(int(float(row["OPTIMAL_RANK"])) if row.get("OPTIMAL_RANK") is not None else "N/A"),
                log=Path(row["log_path"]).relative_to(REPO_ROOT),
            )
        )

    lines += ["", "## Per-experiment metric blocks", ""]

    for row in rows:
        lines += [
            f"### {row['dataset']} — {row['mode_label']}",
            "",
            f"Status: **{row['status']}**",
            "",
            "```text",
            _build_metric_block(row),
            "```",
            "",
        ]

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    modes = _select_modes(args.depth_sources, args.ceres_mode)
    if not modes:
        raise SystemExit("No experiment modes selected. Check --depth-sources and --ceres-mode.")

    datasets = list(dict.fromkeys(args.datasets))
    args.log_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    print(f"Datasets: {', '.join(datasets)}")
    print(f"Modes: {', '.join(m.label for m in modes)}")

    for dataset in datasets:
        scene_root = DATASET_ROOT / dataset
        if not scene_root.exists():
            for mode in modes:
                rows.append(
                    {
                        "dataset": dataset,
                        "mode_key": mode.key,
                        "mode_label": mode.label,
                        "status": "missing-dataset",
                        "log_path": args.log_dir / f"{dataset}__{mode.key}.log",
                    }
                )
            print(f"[skip] dataset missing: {scene_root}")
            continue

        for mode in modes:
            log_path = args.log_dir / f"{dataset}__{mode.key}.log"
            env = os.environ.copy()
            env["XM_DATASET_CAT"] = "Replica"
            env["XM_DATASET_NAME"] = dataset
            env["XM_DEPTH_SOURCE"] = mode.depth_source

            cmd = [sys.executable, "static_solver.py", "--mode", "1", "--weight-mode", "2", "--json"]
            if mode.use_ceres:
                cmd.append("--use-ceres")
            else:
                cmd.append("--no-use-ceres")
            cmd.append("--no-visualize")

            if args.dry_run:
                rows.append(
                    {
                        "dataset": dataset,
                        "mode_key": mode.key,
                        "mode_label": mode.label,
                        "status": "planned",
                        "log_path": log_path,
                    }
                )
                print(
                    f"[dry-run] {dataset} [{mode.label}] depth={mode.depth_source}, "
                    f"use_ceres={mode.use_ceres}: {' '.join(cmd)}"
                )
                continue

            t0 = time.perf_counter()
            proc = subprocess.run(
                cmd,
                cwd=PYTHON_DIR,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            elapsed = time.perf_counter() - t0
            stdout_text = proc.stdout or ""
            stderr_text = proc.stderr or ""
            merged = stdout_text
            if stderr_text.strip():
                merged = merged + "\n\n[STDERR]\n" + stderr_text
            log_path.write_text(merged, encoding="utf-8")

            row: dict[str, Any] = {
                "dataset": dataset,
                "mode_key": mode.key,
                "mode_label": mode.label,
                "status": "ok" if proc.returncode == 0 else f"fail ({proc.returncode})",
                "runtime_wall_s": elapsed,
                "log_path": log_path,
            }
            row.update(_parse_stdout_metrics(stdout_text))
            rows.append(row)

            print(
                f"[{row['status']}] {dataset} [{mode.key}] "
                f"RPE-R={_fmt(row.get('RPE_R'), 6)} ATE-T={_fmt(row.get('ATE_T'), 6)} "
                f"log={log_path}"
            )

    write_report(rows, args.report, modes)
    print(f"[done] Wrote report: {args.report}")


if __name__ == "__main__":
    main()
