import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SINTEL_ROOT = REPO_ROOT / "datasets" / "Sintel"
MPI_SINTEL_ROOT = REPO_ROOT / "datasets" / "MPI-Sintel-depth-training-20150305" / "training"

DA3_PYTHON = Path("/home/fenchel/miniconda3/envs/da3/bin/python")
TAPNET_PYTHON = Path("/home/fenchel/miniconda3/envs/tapnet/bin/python")
XM_SITE_PACKAGES = Path("/home/fenchel/miniconda3/envs/XM/lib/python3.10/site-packages")

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
        description="Run the Sintel static-tracking + XM heuristic sweep and save a markdown report."
    )
    parser.add_argument(
        "--output",
        default=str(SCRIPT_DIR / "sintel_xm_results.md"),
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--log-dir",
        default=str(SCRIPT_DIR / "logs" / "sintel_xm"),
        help="Directory for preparation/solver logs.",
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="Optional subset of Sintel scene names. Defaults to all scenes in datasets/Sintel except template.",
    )
    parser.add_argument(
        "--force-depth",
        action="store_true",
        help="Recompute DA3 depth even if cached outputs already exist.",
    )
    parser.add_argument(
        "--force-tracking",
        action="store_true",
        help="Recompute output_tapnet_da3.npz even if it already exists.",
    )
    parser.add_argument(
        "--use-ceres",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the Ceres refinement stage inside static_solver.py.",
    )
    return parser.parse_args()


def list_scenes(selected):
    all_scenes = sorted(
        p.name for p in SINTEL_ROOT.iterdir() if p.is_dir() and p.name != "template"
    )
    if selected:
        missing = sorted(set(selected) - set(all_scenes))
        if missing:
            raise RuntimeError(f"Unknown Sintel scenes requested: {missing}")
        return [scene for scene in all_scenes if scene in set(selected)]
    return all_scenes


def base_env(scene):
    env = os.environ.copy()
    env["XM_DATASET_CAT"] = "Sintel"
    env["XM_DATASET_NAME"] = scene
    env["XM_DEPTH_SOURCE"] = "da3"
    env["XM_MPI_SINTEL_ROOT"] = str(MPI_SINTEL_ROOT)
    return env


def camera_source_for_scene(scene):
    local_dir = SINTEL_ROOT / scene / "cameras"
    local_count = len(list(local_dir.glob("*.cam"))) if local_dir.exists() else 0
    image_count = len(list((SINTEL_ROOT / scene / "images").glob("*.png")))
    return "local" if local_count >= image_count and image_count > 0 else "mpi_fallback"


def da3_complete(scene):
    out_dir = SINTEL_ROOT / scene / "outputs" / "depth_anything_3"
    return all(
        (out_dir / name).exists()
        for name in ("depth_maps.npy", "conf_maps.npy", "intrinsics.npy")
    )


def tracking_complete(scene):
    return (SINTEL_ROOT / scene / "outputs" / "output_tapnet_da3.npz").exists()


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


def ensure_depth(scene, args):
    out_dir = SINTEL_ROOT / scene / "outputs" / "depth_anything_3"
    already_complete = da3_complete(scene)
    if already_complete and not args.force_depth:
        return {"status": "reused", "log_path": None}

    cmd = [str(DA3_PYTHON), str(SCRIPT_DIR / "get_depths_clean.py")]
    log_path = Path(args.log_dir) / "prep" / f"{scene}_depth.log"
    print(f"[DEPTH] {scene} ({'rerun' if already_complete else 'new'})", flush=True)
    run_and_log(cmd, env=base_env(scene), cwd=str(REPO_ROOT), log_path=log_path)
    return {"status": "rerun" if already_complete else "generated", "log_path": str(log_path)}


def ensure_tracking(scene, args):
    out_path = SINTEL_ROOT / scene / "outputs" / "output_tapnet_da3.npz"
    already_complete = tracking_complete(scene)
    if already_complete and not args.force_tracking:
        return {"status": "reused", "log_path": None}

    cmd = [str(TAPNET_PYTHON), str(SCRIPT_DIR / "static_tracking_online.py")]
    log_path = Path(args.log_dir) / "prep" / f"{scene}_tracking.log"
    print(f"[TRACK] {scene} ({'rerun' if already_complete else 'new'})", flush=True)
    run_and_log(cmd, env=base_env(scene), cwd=str(REPO_ROOT), log_path=log_path)
    return {"status": "rerun" if already_complete else "generated", "log_path": str(log_path)}


def run_solver(scene, setting, args):
    env = base_env(scene)
    env["PYTHONPATH"] = str(XM_SITE_PACKAGES) + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        str(DA3_PYTHON),
        str(SCRIPT_DIR / "static_solver.py"),
        "--mode",
        str(setting["mode"]),
        "--weight-mode",
        str(setting["weight_mode"]),
        "--no-visualize",
        "--json",
    ]
    cmd.append("--use-ceres" if args.use_ceres else "--no-use-ceres")
    if setting["sphere_ratio"] is not None:
        cmd.extend(["--sphere-ratio", str(setting["sphere_ratio"])])

    tag = (
        f"{scene}_setting{setting['setting_id']}"
        f"_mode{setting['mode']}_w{setting['weight_mode']}"
    )
    if setting["sphere_ratio"] is not None:
        tag += f"_sr{setting['sphere_ratio']:.1f}"
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


def write_markdown(output_path, scene_rows, solver_results, use_ceres):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    ceres_label = "with Ceres refinement" if use_ceres else "without Ceres refinement"
    lines = [
        f"# Sintel XM Experiment Results ({ceres_label})",
        "",
        f"Generated: `{created_at}`",
        "",
        "Pipeline:",
        "",
        "- Dataset root: `datasets/Sintel`",
        "- Missing GT cameras are loaded from `datasets/MPI-Sintel-depth-training-20150305/training/camdata_left`",
        "- Missing DA3 depth is generated with `Python/get_depths_clean.py` in the `da3` environment",
        "- Missing `output_tapnet_da3.npz` is generated with `Python/static_tracking_online.py` in the `tapnet` environment",
        "- Solver sweep uses the same 6 heuristics as the BAL experiment, with no added noise",
        f"- Refinement: `{ceres_label}`",
        "",
        "## Preparation",
        "",
        "| Scene | Frames | Camera source | DA3 depth | Tracking | Depth log | Tracking log |",
        "| --- | ---: | --- | --- | --- | --- | --- |",
    ]

    for row in scene_rows:
        depth_log = f"`{row['depth_log_rel']}`" if row["depth_log_rel"] else "-"
        track_log = f"`{row['tracking_log_rel']}`" if row["tracking_log_rel"] else "-"
        lines.append(
            "| "
            f"{row['scene']} | "
            f"{row['frames']} | "
            f"{row['camera_source']} | "
            f"{row['depth_status']} | "
            f"{row['tracking_status']} | "
            f"{depth_log} | "
            f"{track_log} |"
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
        sphere_ratio = (
            f"{result['sphere_ratio']:.1f}"
            if result["sphere_ratio"] is not None
            else "-"
        )
        rel_log = os.path.relpath(result["log_path"], REPO_ROOT)
        lines.append(
            "| "
            f"{result['dataset_name']} | "
            f"{result['setting_id']} | "
            f"{result['mode']} | "
            f"{sphere_ratio} | "
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
    scenes = list_scenes(args.scenes)

    scene_rows = []
    solver_results = []

    for scene in scenes:
        image_count = len(list((SINTEL_ROOT / scene / "images").glob("*.png")))
        camera_source = camera_source_for_scene(scene)
        depth_info = ensure_depth(scene, args)
        tracking_info = ensure_tracking(scene, args)

        scene_rows.append(
            {
                "scene": scene,
                "frames": image_count,
                "camera_source": camera_source,
                "depth_status": depth_info["status"],
                "tracking_status": tracking_info["status"],
                "depth_log_rel": os.path.relpath(depth_info["log_path"], REPO_ROOT)
                if depth_info["log_path"]
                else None,
                "tracking_log_rel": os.path.relpath(tracking_info["log_path"], REPO_ROOT)
                if tracking_info["log_path"]
                else None,
            }
        )
        write_markdown(args.output, scene_rows, solver_results, args.use_ceres)

        for setting in SETTINGS:
            solver_results.append(run_solver(scene, setting, args))
            write_markdown(args.output, scene_rows, solver_results, args.use_ceres)

    write_markdown(args.output, scene_rows, solver_results, args.use_ceres)
    print(f"[DONE] Wrote markdown report to {args.output}", flush=True)


if __name__ == "__main__":
    main()
