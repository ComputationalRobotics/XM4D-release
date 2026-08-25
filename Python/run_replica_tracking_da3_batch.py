#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "datasets" / "Replica"
LOG_ROOT = REPO_ROOT / "Python" / "logs" / "replica_tracking_da3"
REPORT_PATH = REPO_ROOT / "Python" / "replica_tracking_da3_batch.md"


DEFAULT_REPLICA_DATASETS = [
    "office1_seq1",
    "office1_seq2",
    "office2_seq1",
    "office2_seq2",
    "room1_seq1",
    "room1_seq2",
    "room2_seq1",
    "room2_seq2",
]


def _default_datasets() -> list[str]:
    if not DATASET_ROOT.exists():
        return DEFAULT_REPLICA_DATASETS.copy()

    existing = []
    for name in DEFAULT_REPLICA_DATASETS:
        scene = DATASET_ROOT / name
        if scene.is_dir() and (scene / "images").is_dir():
            existing.append(name)

    if existing:
        return existing

    auto = sorted(
        [
            path.name
            for path in DATASET_ROOT.iterdir()
            if path.is_dir() and (path / "images").is_dir()
        ]
    )
    return auto if auto else DEFAULT_REPLICA_DATASETS.copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TAPNet online tracking on Replica scenes with DA3/GT depth via XM_* env vars."
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=_default_datasets(),
        help="Replica dataset names to process.",
    )
    parser.add_argument(
        "--depth-source",
        choices=["da3", "gt"],
        default="da3",
        help="Depth source passed to XM_DEPTH_SOURCE.",
    )
    parser.add_argument(
        "--script",
        type=Path,
        default=REPO_ROOT / "Python" / "static_tracking_online.py",
        help="Tracking script to run once per dataset.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=LOG_ROOT,
        help="Directory for per-scene logs.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORT_PATH,
        help="Markdown report path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned runs without launching tracking.",
    )
    return parser.parse_args()


def write_report(
    rows: list[dict[str, object]],
    report_path: Path,
    script_path: Path,
    depth_source: str,
) -> None:
    lines = [
        "# Replica Tracking Batch Run",
        "",
        f"Tracking script: `{script_path.relative_to(REPO_ROOT)}`",
        f"Depth source: `{depth_source}`",
        "",
        "| Dataset | Status | Runtime (s) | Log |",
        "| --- | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {status} | {runtime:.2f} | `{log_path}` |".format(
                dataset=row["dataset"],
                status=row["status"],
                runtime=row["runtime_s"],
                log_path=Path(row["log_path"]).relative_to(REPO_ROOT),
            )
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    datasets = list(dict.fromkeys(args.datasets))
    if not datasets:
        raise SystemExit("No datasets specified.")

    args.log_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    print(f"Running {len(datasets)} dataset(s): {', '.join(datasets)}")
    print(f"Depth source: {args.depth_source}")

    for dataset in datasets:
        scene_root = DATASET_ROOT / dataset
        if not scene_root.exists():
            rows.append(
                {
                    "dataset": dataset,
                    "status": "missing-dataset",
                    "runtime_s": 0.0,
                    "log_path": args.log_dir / f"{dataset}.log",
                }
            )
            print(f"[skip] {dataset}: missing directory {scene_root}")
            continue

        log_path = args.log_dir / f"{dataset}.log"

        env = os.environ.copy()
        env["XM_DATASET_CAT"] = "Replica"
        env["XM_DATASET_NAME"] = dataset
        env["XM_DEPTH_SOURCE"] = args.depth_source

        cmd = [sys.executable, str(args.script)]
        if args.dry_run:
            print(
                f"[dry-run] {dataset}: XM_DATASET_CAT=Replica "
                f"XM_DATASET_NAME={dataset} XM_DEPTH_SOURCE={args.depth_source} -> {cmd}"
            )
            rows.append(
                {
                    "dataset": dataset,
                    "status": "planned",
                    "runtime_s": 0.0,
                    "log_path": log_path,
                }
            )
            continue

        t0 = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as handle:
            proc = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        runtime_s = time.perf_counter() - t0
        status = "ok" if proc.returncode == 0 else f"fail ({proc.returncode})"
        rows.append(
            {
                "dataset": dataset,
                "status": status,
                "runtime_s": runtime_s,
                "log_path": log_path,
            }
        )
        print(f"[{status}] {dataset} -> {log_path}")

    write_report(rows, args.report, args.script, args.depth_source)
    print(f"[done] Report written to {args.report}")


if __name__ == "__main__":
    main()
