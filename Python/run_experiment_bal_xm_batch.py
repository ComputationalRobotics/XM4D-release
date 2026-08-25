import argparse
import json
import os
import subprocess
import sys
from datetime import datetime


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DEFAULT_XM_PYTHON = "/home/fenchel/miniconda3/envs/XM/bin/python"

DATASETS = ["BAL-93", "BAL-394", "BAL-1936"]
EPS_VALUES = [0.0, 0.1, 0.3]
SETTINGS = [
    {"setting_id": 1, "mode": 1, "sphere_ratio": 0.1, "weight_mode": 1},
    {"setting_id": 2, "mode": 1, "sphere_ratio": 0.3, "weight_mode": 1},
    {"setting_id": 3, "mode": 1, "sphere_ratio": 0.5, "weight_mode": 1},
    {"setting_id": 4, "mode": 1, "sphere_ratio": 0.9, "weight_mode": 2},
    {"setting_id": 5, "mode": 1, "sphere_ratio": 0.9, "weight_mode": 3},
    {"setting_id": 6, "mode": 2, "sphere_ratio": None, "weight_mode": 1},
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the requested BAL/XM grid and save a markdown summary."
    )
    parser.add_argument(
        "--output",
        default=os.path.join(SCRIPT_DIR, "experiment_bal_xm_results.md"),
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--log-dir",
        default=os.path.join(SCRIPT_DIR, "logs", "experiment_bal_xm"),
        help="Directory for per-run logs.",
    )
    parser.add_argument(
        "--python",
        default=DEFAULT_XM_PYTHON if os.path.exists(DEFAULT_XM_PYTHON) else sys.executable,
        help="Python executable to use for each experiment subprocess.",
    )
    parser.add_argument(
        "--title",
        default="BAL XM Experiment Results",
        help="Markdown report title.",
    )
    parser.add_argument(
        "--refinement-note",
        default=None,
        help="Optional refinement note to include near the top of the report.",
    )
    return parser.parse_args()


def run_one(args, dataset, eps, setting):
    cmd = [
        args.python,
        os.path.join(SCRIPT_DIR, "experiment_bal_xm.py"),
        "--dataset",
        dataset,
        "--eps",
        str(eps),
        "--mode",
        str(setting["mode"]),
        "--weight-mode",
        str(setting["weight_mode"]),
        "--result-npz",
        os.path.join(SCRIPT_DIR, "result_XM.npz"),
        "--json",
        "--no-print-s",
    ]
    if setting["sphere_ratio"] is not None:
        cmd.extend(["--sphere-ratio", str(setting["sphere_ratio"])])

    tag = (
        f"{dataset}_eps{eps:.1f}_setting{setting['setting_id']}"
        f"_mode{setting['mode']}_w{setting['weight_mode']}"
    )
    if setting["sphere_ratio"] is not None:
        tag += f"_sr{setting['sphere_ratio']:.1f}"

    print(f"[RUN] {tag}", flush=True)
    completed = subprocess.run(
        cmd,
        cwd=SCRIPT_DIR,
        capture_output=True,
        text=True,
    )

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"{tag}.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        if completed.stdout:
            f.write("[stdout]\n")
            f.write(completed.stdout)
            if not completed.stdout.endswith("\n"):
                f.write("\n")
        if completed.stderr:
            f.write("\n[stderr]\n")
            f.write(completed.stderr)
            if not completed.stderr.endswith("\n"):
                f.write("\n")

    if completed.returncode != 0:
        raise RuntimeError(
            f"Run failed for {tag} with code {completed.returncode}. See {log_path}"
        )

    result = None
    for line in completed.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            result = json.loads(line.split("RESULT_JSON:", 1)[1].strip())
            break

    if result is None:
        raise RuntimeError(f"Missing RESULT_JSON for {tag}. See {log_path}")

    result["setting_id"] = setting["setting_id"]
    result["sphere_ratio"] = setting["sphere_ratio"]
    result["log_path"] = log_path
    return result


def write_markdown(output_path, results, title, refinement_note):
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        f"# {title}",
        "",
        f"Generated: `{created_at}`",
        "",
        "Requested sweep:",
        "",
        "- Datasets: `BAL-93`, `BAL-394`, `BAL-1936`",
        "- Eps: `0.0`, `0.1`, `0.3`",
        "- Settings 1-6 exactly as requested",
        "",
    ]
    if refinement_note:
        lines.extend([f"- Refinement: `{refinement_note}`", ""])
    lines.extend(
        [
            "| Dataset | Eps | Setting | Mode | Sphere Ratio | WEIGHT_MODE | RPE-R | RPE-T | ATE-R | ATE-T | Total time (s) | Solve time (s) | Optimal rank | Log |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )

    for result in results:
        sphere_ratio = (
            f"{result['sphere_ratio']:.1f}"
            if result["sphere_ratio"] is not None
            else "-"
        )
        rel_log = os.path.relpath(result["log_path"], REPO_ROOT)
        lines.append(
            "| "
            f"{result['dataset']} | "
            f"{result['eps']:.1f} | "
            f"{result['setting_id']} | "
            f"{result['mode']} | "
            f"{sphere_ratio} | "
            f"{result['weight_mode']} | "
            f"{result['RPE_R']:.16f} | "
            f"{result['RPE_T']:.16f} | "
            f"{result['ATE_R']:.16f} | "
            f"{result['ATE_T']:.16f} | "
            f"{result['total_time']:.6f} | "
            f"{result['solve_time']:.6f} | "
            f"{result['optimal_rank']} | "
            f"`{rel_log}` |"
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    results = []

    total_runs = len(DATASETS) * len(EPS_VALUES) * len(SETTINGS)
    run_index = 0
    for dataset in DATASETS:
        for eps in EPS_VALUES:
            for setting in SETTINGS:
                run_index += 1
                print(
                    f"[{run_index}/{total_runs}] dataset={dataset} eps={eps:.1f} setting={setting['setting_id']}",
                    flush=True,
                )
                results.append(run_one(args, dataset, eps, setting))
                write_markdown(args.output, results, args.title, args.refinement_note)

    write_markdown(args.output, results, args.title, args.refinement_note)
    print(f"[DONE] Wrote markdown report to {args.output}", flush=True)


if __name__ == "__main__":
    main()
