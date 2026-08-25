#!/usr/bin/env python3
"""COLMAP (incremental SfM) camera-pose baseline for one DyCheck scene.

DyCheck scenes are ordered monocular *video* (single moving iPhone camera), so we
use sequential matching + vocab-tree loop closure (far faster than exhaustive for
hundreds of frames, and closes the handheld orbit loops). Ground-truth PINHOLE
intrinsics are injected into the DB and held FIXED in the mapper (the IMC lesson:
letting COLMAP refine known intrinsics drifts / folds the reconstruction).

Output COLMAP model -> <scene>/output_colmap/sparse/0 (eval with
eval_colmap.py --gt-type dycheck). Uses the isolated conda env `glomap-official`
(official COLMAP), never the modified fork on PATH.
"""
import argparse
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "imc"))
from imc_common import load_pinhole_by_name

ENV = "glomap-official"
CONDA = "/home/fenchel/miniconda3/bin/conda"
PINHOLE_MODEL_ID = 1
VOCAB_TREE = str(Path.home() / ".cache" / "colmap" /
                 "96ca8ec8ea60b1f73465aaf2c401fd3b3ca75cdba2d3c50d6a2f6f760f275ddc-"
                 "vocab_tree_faiss_flickr100K_words256K.bin")


def run(cmd):
    print("[run]", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def patch_intrinsics(db, scene_dir):
    pin = load_pinhole_by_name(scene_dir)
    con = sqlite3.connect(str(db))
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
    ap.add_argument("--matcher", choices=["sequential", "exhaustive"], default="exhaustive",
                    help="exhaustive (gold standard, robust to fast/looping motion) or "
                         "sequential+loop (faster). DyCheck images are small so exhaustive is cheap.")
    args = ap.parse_args()
    scene = Path(args.scene_dir).resolve()
    image_dir = scene / "images"
    work = scene / "output_colmap"
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
             "--ImageReader.single_camera", "1",
             "--FeatureExtraction.use_gpu", "1"])
        patch_intrinsics(db, scene)
        if args.matcher == "exhaustive":
            run([CONDA, "run", "-n", ENV, "colmap", "exhaustive_matcher",
                 "--database_path", db, "--FeatureMatching.use_gpu", "1"])
        else:
            run([CONDA, "run", "-n", ENV, "colmap", "sequential_matcher",
                 "--database_path", db, "--FeatureMatching.use_gpu", "1",
                 "--SequentialMatching.loop_detection", "1",
                 "--SequentialMatching.vocab_tree_path", VOCAB_TREE])
        times["preprocess"] = time.perf_counter() - t

    out = work / "sparse"
    out.mkdir(exist_ok=True)
    t = time.perf_counter()
    run([CONDA, "run", "-n", ENV, "colmap", "mapper",
         "--database_path", db, "--image_path", image_dir, "--output_path", out,
         "--Mapper.ba_refine_focal_length", "0",
         "--Mapper.ba_refine_principal_point", "0",
         "--Mapper.ba_refine_extra_params", "0"])
    times["colmap_solver"] = time.perf_counter() - t

    (work / "times.json").write_text(json.dumps(times, indent=2))
    print("[times]", json.dumps(times))


if __name__ == "__main__":
    main()
