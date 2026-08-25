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
PYTHON_DIR = REPO_ROOT / "Python"
EVAL_SCRIPT = PYTHON_DIR / "eval_replica_colmap_vs_gt.py"
LOG_DIR_DEFAULT = PYTHON_DIR / "logs" / "replica_colmap_eval"
REPORT_DEFAULT = PYTHON_DIR / "replica_colmap_eval_all8.md"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run eval_replica_colmap_vs_gt.py for all Replica scenes and write a markdown report."
    )
    parser.add_argument("--scenes", nargs="*", default=SCENES, help="Replica scenes to evaluate.")
    parser.add_argument(
        "--python-exec",
        default="python",
        help="Python executable for running eval script (default: python).",
    )
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR_DEFAULT)
    parser.add_argument("--report", type=Path, default=REPORT_DEFAULT)
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


def write_report(rows: list[dict[str, Any]], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Replica COLMAP-vs-GT Evaluation (All 8 scenes)",
        "",
        "Command pattern used per scene:",
        "`python eval_replica_colmap_vs_gt.py --scene-dir ../datasets/Replica/<scene>/ --json`",
        "",
        "## Summary",
        "",
        "| Scene | Status | Matched | RPE-R | RPE-T | ATE-R | ATE-T | ATE (m) | RPE trans (m) | RPE rot (deg) | Eval time (s) | Log |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for row in rows:
        lines.append(
            "| {scene} | {status} | {matched} | {rpe_r} | {rpe_t} | {ate_r} | {ate_t} | {m_ate} | {m_rpe_t} | {m_rpe_r} | {t} | `{log}` |".format(
                scene=row["scene"],
                status=row["status"],
                matched=row.get("matched_frames", "N/A"),
                rpe_r=_fmt(row.get("RPE_R"), 6),
                rpe_t=_fmt(row.get("RPE_T"), 6),
                ate_r=_fmt(row.get("ATE_R"), 6),
                ate_t=_fmt(row.get("ATE_T"), 6),
                m_ate=_fmt(row.get("MOSCA_ATE_M"), 6),
                m_rpe_t=_fmt(row.get("MOSCA_RPE_TRANS_M"), 6),
                m_rpe_r=_fmt(row.get("MOSCA_RPE_ROT_DEG"), 6),
                t=_fmt(row.get("eval_time_s"), 6),
                log=Path(row["log_path"]).relative_to(REPO_ROOT),
            )
        )

    lines += ["", "## Per-scene metric blocks", ""]

    for row in rows:
        lines += [
            f"### {row['scene']}",
            "",
            f"Status: **{row['status']}**",
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

    rows: list[dict[str, Any]] = []

    for scene in scenes:
        scene_dir = REPO_ROOT / "datasets" / "Replica" / scene
        log_path = args.log_dir / f"{scene}.log"

        cmd = [
            args.python_exec,
            str(EVAL_SCRIPT),
            "--scene-dir",
            str(scene_dir),
            "--json",
        ]

        row: dict[str, Any] = {
            "scene": scene,
            "status": "planned",
            "log_path": str(log_path),
        }

        if args.dry_run:
            row["status"] = "dry-run"
            rows.append(row)
            print(f"[dry-run] {' '.join(cmd)}")
            continue

        t0 = time.perf_counter()
        rc, out = _run(cmd, cwd=PYTHON_DIR)
        row["eval_time_s"] = time.perf_counter() - t0
        log_path.write_text(out, encoding="utf-8")
        row["status"] = "ok" if rc == 0 else f"fail ({rc})"

        parsed = _parse_result_json(out)
        if parsed is not None:
            row.update(parsed)

        rows.append(row)
        print(f"[{row['status']}] {scene} RPE-R={_fmt(row.get('RPE_R'), 6)} ATE-T={_fmt(row.get('ATE_T'), 6)}")

    write_report(rows, args.report)
    print(f"[done] Wrote report: {args.report}")


if __name__ == "__main__":
    main()
