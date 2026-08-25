#!/usr/bin/env python3
"""Collect MegaSAM IMC results: ATE-T/RPE-T/RPE-R from eval logs + tracking time
from output_megasam/logs.json. Prints a compact table."""
import json
import re
from pathlib import Path

REPO = Path("/home/fenchel/4DReconstruction")
SCENES = ["gate_00", "gate_01", "rome_00", "rome_04", "temple_00", "temple_01"]
STAGE = REPO / "datasets" / "IMC" / "megasam_stage_logs"

print(f"{'scene':<11} {'ATE-T':>8} {'RPE-T':>8} {'RPE-R':>8} {'matched':>8} {'track_s':>8}")
acc = {"ATE_T": [], "RPE_T": [], "RPE_R": []}
times = []
for s in SCENES:
    ev = (STAGE / f"{s}.eval.log")
    ate = rpet = rper = matched = None
    if ev.exists():
        for line in ev.read_text().splitlines():
            if line.startswith("RESULT_JSON:"):
                d = json.loads(line.split("RESULT_JSON:", 1)[1])
                ate, rpet, rper = d["ATE_T"], d["RPE_T"], d["RPE_R"]
                matched = d["matched_frames"]
    lj = REPO / "datasets" / "IMC_msam" / s / "output_megasam" / "logs.json"
    t = None
    if lj.exists():
        d = json.loads(lj.read_text())
        t = d.get("stages", {}).get("tracking", {}).get("time_s")
    if ate is not None:
        acc["ATE_T"].append(ate); acc["RPE_T"].append(rpet); acc["RPE_R"].append(rper)
    if t is not None:
        times.append(t)
    print(f"{s:<11} {ate if ate is None else round(ate,4):>8} {rpet if rpet is None else round(rpet,4):>8} "
          f"{rper if rper is None else round(rper,4):>8} {str(matched):>8} {t if t is None else round(t,1):>8}")

if acc["ATE_T"]:
    n = len(acc["ATE_T"])
    print(f"{'AVERAGE':<11} {sum(acc['ATE_T'])/n:>8.4f} {sum(acc['RPE_T'])/n:>8.4f} {sum(acc['RPE_R'])/n:>8.4f} "
          f"{'':>8} {sum(times)/len(times) if times else 0:>8.1f}")
