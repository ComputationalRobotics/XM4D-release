"""COLMAP (incremental) and GLOMAP (global) baselines for one IMC scene.

Uses the official conda-forge COLMAP 3.13 + GLOMAP 1.2 (env `glomap-official`).
Shared preprocessing: COLMAP GPU feature extraction + exhaustive matching, with
the scene's ground-truth PINHOLE intrinsics injected (same information Ours
gets). Then two solvers on the same database:
  - COLMAP: `colmap mapper`  -> outputs/baseline/colmap/0
  - GLOMAP: `glomap mapper`  -> outputs/baseline/glomap/0
Times (preprocess / colmap solver / glomap solver) -> outputs/baseline/times.json.
Pose accuracy is evaluated separately with eval_colmap.py (gt-type imc).
"""
import argparse
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imc_common import load_pinhole_by_name

ENV = "glomap-official"
CONDA = "/home/fenchel/miniconda3/bin/conda"
PINHOLE_MODEL_ID = 1


def run(cmd, **kw):
    print("[run]", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def patch_intrinsics(database_path, scene_dir):
    pin = load_pinhole_by_name(scene_dir)
    con = sqlite3.connect(str(database_path))
    cur = con.cursor()
    patched = 0
    for _img_id, name, camera_id in cur.execute("SELECT image_id, name, camera_id FROM images").fetchall():
        info = pin.get(name)
        if info is None:
            continue
        params = np.asarray(info["params"], dtype=np.float64).tobytes()
        cur.execute(
            "UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=1 WHERE camera_id=?",
            (PINHOLE_MODEL_ID, info["width"], info["height"], params, camera_id),
        )
        patched += 1
    con.commit()
    con.close()
    print(f"[intrinsics] patched {patched} cameras with GT PINHOLE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir")
    ap.add_argument("--skip-match", action="store_true")
    ap.add_argument("--glomap-only", action="store_true", help="Skip the slow COLMAP incremental solver.")
    args = ap.parse_args()
    scene = Path(args.scene_dir)
    image_dir = scene / "images"
    work = scene / "outputs" / "baseline"
    work.mkdir(parents=True, exist_ok=True)
    db = work / "database.db"
    times = {}

    if not args.skip_match:
        for p in [db, db.with_suffix(".db-wal"), db.with_suffix(".db-shm")]:
            if p.exists():
                p.unlink()
        t = time.perf_counter()
        run([CONDA, "run", "-n", ENV, "colmap", "feature_extractor",
             "--database_path", db, "--image_path", image_dir,
             "--ImageReader.camera_model", "PINHOLE",
             "--ImageReader.single_camera_per_image", "1",
             "--FeatureExtraction.use_gpu", "1"])
        patch_intrinsics(db, scene)
        run([CONDA, "run", "-n", ENV, "colmap", "exhaustive_matcher",
             "--database_path", db, "--FeatureMatching.use_gpu", "1"])
        times["preprocess"] = time.perf_counter() - t

    # COLMAP incremental
    if not args.glomap_only:
        colmap_out = work / "colmap"
        colmap_out.mkdir(exist_ok=True)
        t = time.perf_counter()
        run([CONDA, "run", "-n", ENV, "colmap", "mapper",
             "--database_path", db, "--image_path", image_dir, "--output_path", colmap_out])
        times["colmap_solver"] = time.perf_counter() - t

    # GLOMAP global
    glomap_out = work / "glomap"
    glomap_out.mkdir(exist_ok=True)
    t = time.perf_counter()
    run([CONDA, "run", "-n", ENV, "glomap", "mapper",
         "--database_path", db, "--image_path", image_dir, "--output_path", glomap_out,
         "--TrackEstablishment.max_num_view_per_track", "1000000"])
    times["glomap_solver"] = time.perf_counter() - t

    # merge with any prior preprocess time if we skipped matching
    times_path = work / "times.json"
    if args.skip_match and times_path.exists():
        prev = json.loads(times_path.read_text())
        times["preprocess"] = prev.get("preprocess", times.get("preprocess"))
    times_path.write_text(json.dumps(times, indent=2))
    print("[times]", json.dumps(times))


if __name__ == "__main__":
    main()
