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
LOG_ROOT = REPO_ROOT / "Python" / "logs" / "replica_tracking"
REPORT_PATH = REPO_ROOT / "Python" / "replica_tracking_smoke.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Replica tracking smoke tests.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=sorted(
            [
                path.name
                for path in DATASET_ROOT.iterdir()
                if path.is_dir() and (path / "images").is_dir() and (path / "depths").is_dir()
            ]
        ),
        help="Structured Replica datasets to track.",
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
    return parser.parse_args()


def write_report(rows: list[dict[str, object]], report_path: Path) -> None:
    lines = [
        "# Replica Tracking Smoke Test",
        "",
        "Tracking command: `Python/static_tracking_online.py` with `XM_DATASET_CAT=Replica` and `XM_DEPTH_SOURCE=gt`.",
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
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for dataset in args.datasets:
        log_path = args.log_dir / f"{dataset}.log"
        env = os.environ.copy()
        env["XM_DATASET_CAT"] = "Replica"
        env["XM_DATASET_NAME"] = dataset
        env["XM_DEPTH_SOURCE"] = "gt"
        cmd = [sys.executable, str(REPO_ROOT / "Python" / "static_tracking_online.py")]

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

    write_report(rows, args.report)
    print(f"[done] Report written to {args.report}")


if __name__ == "__main__":
    main()
