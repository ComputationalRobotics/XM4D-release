#!/usr/bin/env python3

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SINTEL_ROOT = REPO_ROOT / "datasets" / "Sintel"
MPI_SINTEL_ROOT = REPO_ROOT / "datasets" / "MPI-Sintel-depth-training-20150305" / "training"

DA3_PYTHON = Path("/home/fenchel/miniconda3/envs/da3/bin/python")
XM_SITE_PACKAGES = Path("/home/fenchel/miniconda3/envs/XM/lib/python3.10/site-packages")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a Sintel GT-depth solver sweep for one heuristic setting."
    )
    parser.add_argument(
        "--output",
        default=str(SCRIPT_DIR / "sintel_xm_results_gt_depth_setting4.md"),
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--log-dir",
        default=str(SCRIPT_DIR / "logs" / "sintel_gt_depth_setting4"),
        help="Directory for preparation/solver logs.",
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="Optional subset of Sintel scene names. Defaults to all scenes except template.",
    )
    parser.add_argument(
        "--force-convert",
        action="store_true",
        help="Rebuild output_tapnet_gt.npz even if it already exists.",
    )
    parser.add_argument(
        "--use-ceres",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the Ceres refinement stage inside static_solver.py.",
    )
    parser.add_argument("--setting-id", type=int, default=4, help="Heuristic setting label for the report.")
    parser.add_argument("--mode", type=int, choices=[1, 2], default=1, help="static_solver.py --mode")
    parser.add_argument(
        "--weight-mode",
        type=int,
        choices=[1, 2, 3],
        default=2,
        help="static_solver.py --weight-mode",
    )
    parser.add_argument(
        "--sphere-ratio",
        type=float,
        default=0.9,
        help="static_solver.py --sphere-ratio",
    )
    return parser.parse_args()


def list_scenes(selected):
    all_scenes = sorted(
        p.name for p in SINTEL_ROOT.iterdir() if p.is_dir() and p.name != "template"
    )
    if selected:
        selected_set = set(selected)
        missing = sorted(selected_set - set(all_scenes))
        if missing:
            raise RuntimeError(f"Unknown Sintel scenes requested: {missing}")
        return [scene for scene in all_scenes if scene in selected_set]
    return all_scenes


def base_env(scene):
    env = os.environ.copy()
    env["XM_DATASET_CAT"] = "Sintel"
    env["XM_DATASET_NAME"] = scene
    env["XM_DEPTH_SOURCE"] = "gt"
    env["XM_MPI_SINTEL_ROOT"] = str(MPI_SINTEL_ROOT)
    return env


def camera_source_for_scene(scene):
    local_dir = SINTEL_ROOT / scene / "cameras"
    local_count = len(list(local_dir.glob("*.cam"))) if local_dir.exists() else 0
    image_count = len(list((SINTEL_ROOT / scene / "images").glob("*.png")))
    return "local" if local_count >= image_count and image_count > 0 else "mpi_fallback"


def gt_tapnet_complete(scene):
    return (SINTEL_ROOT / scene / "outputs" / "output_tapnet_gt.npz").exists()


def run_and_log(cmd, env, cwd, log_path):
    completed = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
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
        raise RuntimeError(f"Command failed with code {completed.returncode}: {' '.join(cmd)}")
    return completed


def ensure_gt_conversion(scene, args):
    already_complete = gt_tapnet_complete(scene)
    if already_complete and not args.force_convert:
        return {"status": "reused", "log_path": None}

    cmd = [str(DA3_PYTHON), str(SCRIPT_DIR / "convert_sintel_tracks_da3_to_gt.py")]
    log_path = Path(args.log_dir) / "prep" / f"{scene}_gt_convert.log"
    print(f"[GT-CONVERT] {scene} ({'rerun' if already_complete else 'new'})", flush=True)
    run_and_log(cmd, env=base_env(scene), cwd=str(REPO_ROOT), log_path=log_path)
    return {"status": "rerun" if already_complete else "generated", "log_path": str(log_path)}


def run_solver(scene, args, setting):
    env = base_env(scene)
    env["PYTHONPATH"] = str(XM_SITE_PACKAGES) + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        str(DA3_PYTHON),
        str(SCRIPT_DIR / "static_solver.py"),
        "--mode",
        str(setting["mode"]),
        "--weight-mode",
        str(setting["weight_mode"]),
        "--sphere-ratio",
        str(setting["sphere_ratio"]),
        "--no-visualize",
        "--json",
    ]
    cmd.append("--use-ceres" if args.use_ceres else "--no-use-ceres")

    tag = (
        f"{scene}_setting{setting['setting_id']}"
        f"_mode{setting['mode']}_w{setting['weight_mode']}"
        f"_sr{setting['sphere_ratio']:.1f}"
    )
    tag += "_ceres" if args.use_ceres else "_no_ceres"

    log_path = Path(args.log_dir) / "solver" / f"{tag}.log"
    print(f"[SOLVER] {tag}", flush=True)
    completed = run_and_log(cmd, env=env, cwd=str(REPO_ROOT), log_path=log_path)

    result = None
    for line in completed.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            result = json.loads(line.split("RESULT_JSON:", 1)[1].strip())
            break
    if result is None:
        raise RuntimeError(f"Missing RESULT_JSON in solver log {log_path}")

    result["setting_id"] = setting["setting_id"]
    result["sphere_ratio"] = setting["sphere_ratio"]
    result["log_path"] = str(log_path)
    return result


def write_markdown(output_path, scene_rows, solver_results, use_ceres, setting):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    ceres_label = "with Ceres refinement" if use_ceres else "without Ceres refinement"
    lines = [
        f"# Sintel XM GT-Depth Results (setting {setting['setting_id']})",
        "",
        f"Generated: `{created_at}`",
        "",
        "Pipeline:",
        "",
        "- Dataset root: `datasets/Sintel`",
        "- Missing GT cameras are loaded from `datasets/MPI-Sintel-depth-training-20150305/training/camdata_left`",
        "- GT depth is taken from `datasets/MPI-Sintel-depth-training-20150305/training/depth/<scene>` when no local `depths/` folder exists",
        "- TAP tracking is reused from `outputs/tracks_debug.npz`; `output_tapnet_gt.npz` is rebuilt without rerunning tracking",
        (
            f"- Solver heuristic is setting {setting['setting_id']}: "
            f"`mode={setting['mode']}`, "
            f"`sphere_ratio={setting['sphere_ratio']:.1f}`, "
            f"`WEIGHT_MODE={setting['weight_mode']}`"
        ),
        f"- Refinement: `{ceres_label}`",
        "",
        "## Preparation",
        "",
        "| Scene | Frames | Camera source | GT-depth conversion | Conversion log |",
        "| --- | ---: | --- | --- | --- |",
    ]

    for row in scene_rows:
        convert_log = f"`{row['convert_log_rel']}`" if row["convert_log_rel"] else "-"
        lines.append(
            "| "
            f"{row['scene']} | "
            f"{row['frames']} | "
            f"{row['camera_source']} | "
            f"{row['convert_status']} | "
            f"{convert_log} |"
        )

    lines.extend(
        [
            "",
            "## Solver Results",
            "",
            "| Scene | Setting | Mode | Sphere Ratio | WEIGHT_MODE | Ceres refine | RPE-R | RPE-T | ATE-R | ATE-T | Total time (s) | Solve time (s) | Optimal rank | Log |",
            "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )

    for result in solver_results:
        rel_log = os.path.relpath(result["log_path"], REPO_ROOT)
        lines.append(
            "| "
            f"{result['dataset_name']} | "
            f"{result['setting_id']} | "
            f"{result['mode']} | "
            f"{result['sphere_ratio']:.1f} | "
            f"{result['weight_mode']} | "
            f"{'yes' if result.get('use_ceres', use_ceres) else 'no'} | "
            f"{result['RPE_R']:.16f} | "
            f"{result['RPE_T']:.16f} | "
            f"{result['ATE_R']:.16f} | "
            f"{result['ATE_T']:.16f} | "
            f"{result['total_time']:.6f} | "
            f"{result['solve_time']:.6f} | "
            f"{result['optimal_rank']} | "
            f"`{rel_log}` |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    setting = {
        "setting_id": args.setting_id,
        "mode": args.mode,
        "sphere_ratio": args.sphere_ratio,
        "weight_mode": args.weight_mode,
    }
    scenes = list_scenes(args.scenes)

    scene_rows = []
    solver_results = []

    for scene in scenes:
        image_count = len(list((SINTEL_ROOT / scene / "images").glob("*.png")))
        camera_source = camera_source_for_scene(scene)
        convert_info = ensure_gt_conversion(scene, args)

        scene_rows.append(
            {
                "scene": scene,
                "frames": image_count,
                "camera_source": camera_source,
                "convert_status": convert_info["status"],
                "convert_log_rel": os.path.relpath(convert_info["log_path"], REPO_ROOT)
                if convert_info["log_path"]
                else None,
            }
        )
        write_markdown(args.output, scene_rows, solver_results, args.use_ceres, setting)

        solver_results.append(run_solver(scene, args, setting))
        write_markdown(args.output, scene_rows, solver_results, args.use_ceres, setting)

    write_markdown(args.output, scene_rows, solver_results, args.use_ceres, setting)
    print(f"[DONE] Wrote markdown report to {args.output}", flush=True)


if __name__ == "__main__":
    main()
