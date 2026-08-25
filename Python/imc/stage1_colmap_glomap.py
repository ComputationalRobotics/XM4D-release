"""Stage 1: COLMAP feature matching + GLOMAP view-graph for an IMC scene.

Uses the CUDA-enabled colmap binary built from the same source as the GLOMAP
fork (the pip pycolmap is CPU-only and the system colmap 3.14 writes an
incompatible DB schema) for GPU feature extraction + exhaustive matching, then
runs the GLOMAP fork and parses its view graph.

Intrinsics: NO ground-truth is passed. COLMAP initialises a PINHOLE camera per
image and GLOMAP calibrates the focal lengths during global SfM. The estimated
per-image intrinsics are read back from GLOMAP's output model and recorded to
outputs/glomap_work/glomap_intrinsics.json -- this is the single intrinsic
source consumed by stage2 (UniDepth) and stage3 (lift). Pass --gt-intrinsics to
instead inject the scene's ground-truth PINHOLE intrinsics (for A/B comparison).

Artifacts are written to <scene>/outputs/glomap_work/. GLOMAP (our fork) always
dumps its result to ./assets/tempdata/{output,filename,relative_pose}.txt
relative to the working directory, so it is launched with cwd = XM-code root.
"""
import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, save_npz

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imc_common import load_pinhole_by_name, sorted_image_names

# This script lives in <repo>/Python/imc/, but the GLOMAP/COLMAP binaries and the
# assets/tempdata dir GLOMAP writes to live under <repo>/XM-code/. Anchor to the
# repo root (parents[2]) so the paths resolve regardless of this file's location.
REPO_ROOT = Path(__file__).resolve().parents[2]
XM_CODE_ROOT = REPO_ROOT / "XM-code"
GLOMAP_BUILD = XM_CODE_ROOT / "deps" / "glomap" / "build"
GLOMAP_BIN = GLOMAP_BUILD / "glomap" / "glomap"
# Use the COLMAP built from the SAME source GLOMAP links against (commit
# 32542632, CUDA). The system colmap 3.14 writes a newer DB schema the fork's
# embedded COLMAP cannot read; pip pycolmap is CPU-only.
COLMAP_BIN = GLOMAP_BUILD / "_deps" / "colmap-build" / "src" / "colmap" / "exe" / "colmap"
TEMPDATA = XM_CODE_ROOT / "assets" / "tempdata"
COLMAP_PINHOLE_MODEL_ID = 1


def run(cmd, **kw):
    print("[run]", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def patch_intrinsics(database_path, scene_dir):
    """Overwrite each image's camera with the GT PINHOLE intrinsics by name."""
    pin = load_pinhole_by_name(scene_dir)
    con = sqlite3.connect(str(database_path))
    cur = con.cursor()
    rows = cur.execute("SELECT image_id, name, camera_id FROM images").fetchall()
    patched = missing = 0
    for _img_id, name, camera_id in rows:
        info = pin.get(name)
        if info is None:
            missing += 1
            continue
        params = np.asarray(info["params"], dtype=np.float64).tobytes()
        cur.execute(
            "UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=1 "
            "WHERE camera_id=?",
            (COLMAP_PINHOLE_MODEL_ID, info["width"], info["height"], params, camera_id),
        )
        patched += 1
    con.commit()
    con.close()
    print(f"[intrinsics] patched {patched} cameras with GT PINHOLE, {missing} without GT")


def clear_tempdata():
    # GLOMAP writes to the shared ./assets/tempdata path; clear it first so a run
    # that fails to emit fresh files can't silently reuse a previous scene's data.
    for name in ("output.txt", "filename.txt", "relative_pose.txt"):
        p = TEMPDATA / name
        if p.exists():
            p.unlink()


def run_glomap(database_path, image_dir, glomap_out):
    cmd = [
        GLOMAP_BIN, "mapper",
        "--database_path", database_path,
        "--output_path", glomap_out,
        "--image_path", image_dir,
        "--TrackEstablishment.max_num_view_per_track", "1000000",
    ]
    run(cmd, cwd=str(XM_CODE_ROOT))


def parse_and_save(work, scene_dir):
    out_txt = TEMPDATA / "output.txt"
    if not out_txt.exists():
        raise RuntimeError(
            f"GLOMAP did not write {out_txt} (it likely failed to register/emit tracks). "
            "Refusing to parse; re-run this scene."
        )
    matches = np.loadtxt(out_txt)
    file_lines = [
        l.split() for l in (TEMPDATA / "filename.txt").read_text().splitlines() if l.strip()
    ]
    ids = np.array([int(p[0]) for p in file_lines])
    names = np.array([p[1] for p in file_lines], dtype=object)
    order = np.argsort(ids)
    ids, names = ids[order], names[order]
    max_index = int(ids.max())
    filename = np.empty(max_index, dtype=object)
    filename[ids - 1] = names

    # Guard against stale/mismatched tempdata: the parsed names must belong to this scene.
    scene_names = set(sorted_image_names(scene_dir))
    overlap = sum(1 for n in names if str(n).strip() in scene_names)
    if overlap == 0:
        raise RuntimeError(
            f"GLOMAP tempdata does not match scene {scene_dir}: 0/{len(names)} names overlap "
            "(stale/contaminated tempdata). Re-run this scene."
        )

    N = int(np.max(matches[:, 0]))
    M = int(np.max(matches[:, 3]))
    print(f"[glomap] registered frames N={N}, tracks M={M}, observations={matches.shape[0]}")

    edges = matches[:, [0, 3]].copy()
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, matches = edges[order], matches[order]
    _, unique_idx = np.unique(edges, axis=0, return_index=True)
    dropped = matches.shape[0] - unique_idx.shape[0]
    matches = matches[unique_idx]
    print(f"[glomap] dropped {dropped} duplicate observations, kept {matches.shape[0]}")

    rows = matches[:, 0].astype(int) - 1
    cols = matches[:, 3].astype(int) - 1
    vis = coo_matrix((np.ones(matches.shape[0], dtype=int), (rows, cols))).tocsr()
    landmarkx = coo_matrix((matches[:, 1], (rows, cols))).tocsr()
    landmarky = coo_matrix((matches[:, 2], (rows, cols))).tocsr()

    save_npz(work / "vis.npz", vis)
    save_npz(work / "landmarkx.npz", landmarkx)
    save_npz(work / "landmarky.npz", landmarky)
    np.save(work / "filename.npy", filename)
    print(f"[glomap] saved view graph to {work}")


def dump_glomap_intrinsics(work, glomap_out, database_path):
    """Read GLOMAP's estimated per-image intrinsics and record them to JSON.

    GLOMAP's calibrated cameras live in glomap_output/0/cameras.bin. The
    image-name -> camera_id mapping comes from the COLMAP database (single camera
    per image), because GLOMAP may write an empty images.bin when its global
    positioning does not converge -- the view graph observations are still valid.
    """
    import pycolmap

    model_dir = Path(glomap_out) / "0"
    rec = pycolmap.Reconstruction(str(model_dir))
    con = sqlite3.connect(str(database_path))
    name2cam = dict(con.execute("SELECT name, camera_id FROM images").fetchall())
    con.close()

    intrinsics = {}
    for name, cam_id in name2cam.items():
        if cam_id not in rec.cameras:
            continue
        cam = rec.cameras[cam_id]
        K = np.asarray(cam.calibration_matrix(), dtype=np.float64)
        intrinsics[name] = {
            "K": K.tolist(),
            "model": cam.model.name if hasattr(cam.model, "name") else str(cam.model),
            "params": [float(p) for p in cam.params],
            "width": int(cam.width),
            "height": int(cam.height),
        }
    out = Path(work) / "glomap_intrinsics.json"
    with open(out, "w") as f:
        json.dump(intrinsics, f, indent=2)
    fx = [v["K"][0][0] for v in intrinsics.values()]
    print(f"[glomap] recorded estimated intrinsics for {len(intrinsics)} images -> {out} "
          f"(focal min={min(fx):.1f} median={np.median(fx):.1f} max={max(fx):.1f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir")
    ap.add_argument("--skip-match", action="store_true",
                    help="Reuse existing database (features+matches); only glomap + parse.")
    ap.add_argument("--gt-intrinsics", action="store_true",
                    help="Inject ground-truth PINHOLE intrinsics before matching (default: let "
                         "COLMAP/GLOMAP estimate them).")
    ap.add_argument("--no-gpu", action="store_true")
    args = ap.parse_args()

    scene = Path(args.scene_dir)
    image_dir = scene / "images"
    work = scene / "outputs" / "glomap_work"
    work.mkdir(parents=True, exist_ok=True)
    database_path = work / "database.db"
    glomap_out = work / "glomap_output"
    use_gpu = "0" if args.no_gpu else "1"

    if not args.skip_match:
        for p in [database_path, database_path.with_suffix(".db-wal"),
                  database_path.with_suffix(".db-shm")]:
            if p.exists():
                p.unlink()
        run([COLMAP_BIN, "feature_extractor",
             "--database_path", database_path,
             "--image_path", image_dir,
             "--ImageReader.camera_model", "PINHOLE",
             "--ImageReader.single_camera_per_image", "1",
             "--SiftExtraction.use_gpu", use_gpu])
        if args.gt_intrinsics:
            patch_intrinsics(database_path, scene)
        run([COLMAP_BIN, "exhaustive_matcher",
             "--database_path", database_path,
             "--SiftMatching.use_gpu", use_gpu])
    elif args.gt_intrinsics:
        patch_intrinsics(database_path, scene)

    clear_tempdata()
    run_glomap(database_path, image_dir, glomap_out)
    parse_and_save(work, scene)
    dump_glomap_intrinsics(work, glomap_out, database_path)


if __name__ == "__main__":
    main()
