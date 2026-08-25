#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable

SCENES = [
    "mountain_1",
    "alley_1",
    "alley_2",
    "bamboo_1",
    "bamboo_2",
    "bandage_1",
    "bandage_2",
    "temple_2",
    "temple_3",
    "market_2",
    "market_5",
    "market_6",
    "cave_4",
    "ambush_2",
    "ambush_4",
    "ambush_5",
    "ambush_6",
    "ambush_7",
    "cave_2",
    "shaman_2",
    "shaman_3",
    "sleeping_1",
    "sleeping_2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-evaluate MegaSaM COLMAP exports against GT."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets" / "Sintel",
        help="Root Sintel dataset directory.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "megasam_colmap_eval.md",
        help="Markdown file to write the report.",
    )
    parser.add_argument(
        "--eval-script",
        type=Path,
        default=Path(__file__).resolve().parent / "eval_replica_colmap_vs_gt.py",
        help="Evaluation script path.",
    )
    return parser.parse_args()


def _extract_summary(output: str) -> list[str]:
    lines = output.strip().splitlines()
    summary = []
    rpe_idx = next((i for i, line in enumerate(lines) if line.strip().startswith("RPE-R:")), None)
    if rpe_idx is not None:
        summary.append(lines[rpe_idx])
        for j in range(rpe_idx + 1, len(lines)):
            if lines[j].strip().startswith("============================================================"):
                summary.append(lines[j])
                if j + 1 < len(lines):
                    summary.append(lines[j + 1])
                if j + 2 < len(lines):
                    summary.append(lines[j + 2])
                if j + 3 < len(lines):
                    summary.append(lines[j + 3])
                if j + 4 < len(lines):
                    summary.append(lines[j + 4])
                if j + 5 < len(lines):
                    summary.append(lines[j + 5])
                if j + 6 < len(lines):
                    summary.append(lines[j + 6])
                break
        return summary

    mosca_start = next(
        (i for i, line in enumerate(lines) if line.strip().startswith("MoSca-style eval")),
        None,
    )
    if mosca_start is not None:
        start = max(0, mosca_start - 2)
        end = min(len(lines), mosca_start + 6)
        return lines[start:end]

    return ["No summary found in output."]


def _run_scene(scene: str, eval_script: Path, dataset_root: Path) -> tuple[bool, list[str]]:
    scene_dir = dataset_root / scene
    sparse_dir = scene_dir / "output_megasam" / "sparse" / "0"
    cmd = [
        sys.executable,
        str(eval_script),
        "--scene-dir",
        str(scene_dir),
        "--sparse-dir",
        str(sparse_dir),
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        return False, ["Evaluation failed:", *output.strip().splitlines()[-20:]]
    return True, _extract_summary(output)


def _format_section(scene: str, ok: bool, summary_lines: Iterable[str]) -> str:
    status = "OK" if ok else "FAILED"
    lines = [f"## {scene}", "", f"Status: **{status}**", "", "```text"]
    lines.extend(summary_lines)
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_md = args.output_md
    output_md.parent.mkdir(parents=True, exist_ok=True)

    sections = ["# MegaSaM COLMAP vs GT", ""]
    for scene in SCENES:
        ok, summary = _run_scene(scene, args.eval_script, args.dataset_root)
        sections.append(_format_section(scene, ok, summary))

    output_md.write_text("\n".join(sections), encoding="utf-8")
    print(f"Wrote report: {output_md}")


if __name__ == "__main__":
    main()
