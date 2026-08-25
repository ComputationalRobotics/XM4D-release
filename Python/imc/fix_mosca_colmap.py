#!/usr/bin/env python3
"""MoSca exports its COLMAP images.txt with a camera-to-world convention
(qvec from R_wc, tvec = camera center C), whereas standard COLMAP / eval_colmap.py
expect world-to-camera (R_w2c, t_w2c). This rewrites MoSca's model into standard
convention so it can be evaluated with the same eval_colmap.py as every other
IMC method.

Conversion:  R_w2c = R_c2w^T ;  t_w2c = -R_c2w^T @ C
Writes a text-only model (no .bin) so pycolmap reads the corrected .txt.
"""
import argparse
from pathlib import Path

import numpy as np


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)


def rotmat2qvec(R):
    # standard COLMAP conversion
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
    ]) / 3.0
    vals, vecs = np.linalg.eigh(K)
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    if q[0] < 0:
        q = -q
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="MoSca sparse dir with cameras.txt/images.txt (c2w convention)")
    ap.add_argument("dst", help="output dir for standardized w2c text model")
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    # cameras.txt copied verbatim
    (dst / "cameras.txt").write_text((src / "cameras.txt").read_text())
    (dst / "points3D.txt").write_text(
        "# 3D point list\n# Number of points: 0, mean track length: 0\n")

    lines = [l for l in (src / "images.txt").read_text().splitlines()
             if not l.startswith("#") and l.strip()]
    out = ["# Image list with two lines of data per image:",
           "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
           "#   POINTS2D[] as (X, Y, POINT3D_ID)",
           f"# Number of images: {len(lines)}, mean observations per image: 0"]
    for l in lines:
        p = l.split()
        iid = p[0]
        q = np.array(list(map(float, p[1:5])))     # R_c2w quaternion
        C = np.array(list(map(float, p[5:8])))     # camera center (world)
        cam_id, name = p[8], p[9]
        R_c2w = qvec2rotmat(q)
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ C
        qn = rotmat2qvec(R_w2c)
        out.append(f"{iid} {qn[0]:.12f} {qn[1]:.12f} {qn[2]:.12f} {qn[3]:.12f} "
                   f"{t_w2c[0]:.12f} {t_w2c[1]:.12f} {t_w2c[2]:.12f} {cam_id} {name}")
        out.append("")  # empty POINTS2D line
    (dst / "images.txt").write_text("\n".join(out) + "\n")
    print(f"wrote standardized model -> {dst} ({len(lines)} images)")


if __name__ == "__main__":
    main()
