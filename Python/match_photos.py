"""Step 2 for a photo collection (config.json "input": "photos"): the 2D tracks of the photos, by feature matching.

COLMAP extracts SIFT features of every photo and matches every pair of photos on the GPU (exhaustive matching, with
two-view geometric verification); our GLOMAP fork (glomap/, built by install/glomap.sh) then estimates the relative
poses, drops the pairs that disagree with rotation averaging, keeps the largest connected set of photos and joins the
verified matches into tracks. It stops there: the cameras are solved by XM in step 3.

Every photo's camera is set to its intrinsics from step 1 (the depth model's, or config.json INTRINSICS) before
matching, so step 1 runs first: the matching verifies calibrated two-view geometry, and step 3 lifts with the same K.

Writes outputs/photo_tracks.npz: names (the photos, in the order of utils.definitions.photo_paths(); a photo's index
is its camera index), frame (n,) photo index, track (n,) track index, xy (n, 2) pixel position (COLMAP convention), one
row per observation. The COLMAP database and GLOMAP's files stay in outputs/matching/.
"""
import sqlite3
import subprocess
from pathlib import Path

import numpy as np

import utils.definitions as D
from utils.script_logger import finish_script_logger, install_script_logger

BUILD = D.REPO_ROOT / "build" / "glomap"
GLOMAP = BUILD / "glomap" / "glomap"
COLMAP = BUILD / "_deps" / "colmap-build" / "src" / "colmap" / "exe" / "colmap"   # the COLMAP GLOMAP is built with
PINHOLE = 1   # COLMAP camera model id


def run(cmd, **kw):
    print("[run]", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def set_intrinsics(db):
    """Every photo's camera (one per photo) becomes the pinhole K of step 1, marked as known."""
    con = sqlite3.connect(str(db))
    for name, cam_id in con.execute("SELECT name, camera_id FROM images").fetchall():
        f = D.photo_depth_dir / f"{name}.npz"
        if not f.exists():
            raise SystemExit(f"{f} is missing: run step 1 first (the photos are matched with its intrinsics)")
        with np.load(f) as z:
            K = z["K"]
        params = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], np.float64)
        con.execute("UPDATE cameras SET model=?, params=?, prior_focal_length=1 WHERE camera_id=?",
                    (PINHOLE, params.tobytes(), cam_id))
    con.commit()
    con.close()


def main():
    if not D.photos:
        raise SystemExit('match_photos.py is step 2 for photos ("input": "photos" in config.json); '
                         "videos use static_tracking_online.py")
    for exe in (GLOMAP, COLMAP):
        if not exe.exists():
            raise SystemExit(f"{exe} is missing: build it with bash install/glomap.sh")
    work = D.output_dir / "matching"
    tempdata = work / "assets" / "tempdata"   # the GLOMAP fork writes its tracks to ./assets/tempdata/ of its CWD
    tempdata.mkdir(parents=True, exist_ok=True)
    install_script_logger("match_photos", D.output_dir)

    photos = D.photo_paths()
    names = [p.name for p in photos]
    if len(names) < 2:
        raise SystemExit(f"Need at least 2 photos in {D.video_dir}, found {len(names)}")
    db = work / "database.db"
    for f in (db, Path(f"{db}-wal"), Path(f"{db}-shm"), *tempdata.glob("*.txt")):
        f.unlink(missing_ok=True)
    (work / "image_list.txt").write_text("\n".join(names) + "\n")

    run([COLMAP, "feature_extractor", "--database_path", db, "--image_path", D.video_dir,
         "--image_list_path", work / "image_list.txt",
         "--ImageReader.camera_model", "PINHOLE", "--ImageReader.single_camera_per_image", "1",
         "--SiftExtraction.use_gpu", "1"])
    set_intrinsics(db)
    print(f"[match] every photo's camera set to its intrinsics from step 1 ({D.photo_depth_dir})")
    run([COLMAP, "exhaustive_matcher", "--database_path", db, "--SiftMatching.use_gpu", "1"])
    run([GLOMAP, "mapper", "--database_path", db, "--output_path", work / "glomap", "--image_path", D.video_dir,
         "--TrackEstablishment.max_num_view_per_track", "1000000"], cwd=str(work))

    if not (tempdata / "output.txt").exists():
        raise SystemExit("GLOMAP wrote no tracks (too few matched photos?); see the log above")
    obs = np.loadtxt(tempdata / "output.txt", ndmin=2)              # image_id x y track (1-based)
    id_name = dict(line.split(maxsplit=1) for line in (tempdata / "filename.txt").read_text().splitlines() if line)
    index = {n: i for i, n in enumerate(names)}
    frame = np.array([index.get(id_name.get(str(int(i)), "").strip(), -1) for i in obs[:, 0]], np.int64)
    keep = frame >= 0
    frame, xy, track = frame[keep], obs[keep, 1:3], obs[keep, 3].astype(np.int64) - 1
    # one observation per (track, photo): GLOMAP can put two features of one photo in a track; keep the first
    _, first = np.unique(np.stack([track, frame], 1), axis=0, return_index=True)
    first.sort()
    frame, xy, track = frame[first], xy[first], track[first]
    np.savez(D.photo_tracks_path, names=np.array(names), frame=frame.astype(np.int32), track=track.astype(np.int32),
             xy=xy.astype(np.float32))
    n_used = len(np.unique(frame))
    print(f"[match] {len(np.unique(track))} tracks, {len(frame)} observations in {n_used} of {len(names)} photos "
          f"-> {D.photo_tracks_path}")
    finish_script_logger()


if __name__ == "__main__":
    main()
