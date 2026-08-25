"""Stage 3: lift the GLOMAP view graph to 3D solver observations.

Runs in the `XM` conda env (only needs numpy/scipy/cv2). Combines:
  - GLOMAP view graph  (outputs/glomap_work/{vis,landmarkx,landmarky}.npz, filename.npy)
  - UniDepth depth     (outputs/unidepth_depth/<stem>.npy)
  - GLOMAP-estimated K (outputs/glomap_work/glomap_intrinsics.json)  [no ground truth]

and writes outputs/output_tapnet_unidepth.npz with the arrays
Python/static_solver.py consumes via its pre-lifted "old 3D keys" branch:

  edges     (num_obs, 2) int   -> [landmark_index, frame_index], 0-based
  weights   (num_obs, 1) float
  landmarks (num_obs, 3) float -> camera-frame 3D point  K^-1 [u,v,1] * depth

Frame indices are the position of each image name in the *sorted* name list,
matching static_solver.py's image_list and utils/readgt_IMC.load_imc_gt.
Frames GLOMAP did not register, or whose observations all fail depth/bounds
checks, simply produce no edges (handled downstream by checklandmarks).
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.sparse import load_npz

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imc_common import INTRINSIC_SOURCES, load_intrinsics, name_to_frame_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir")
    ap.add_argument("--intrinsics", choices=INTRINSIC_SOURCES, default="glomap",
                    help="Intrinsic source used to unproject 2D observations to 3D.")
    ap.add_argument("--weight", choices=("uniform", "confidence"), default="uniform",
                    help="Per-observation solver weight: uniform (1.0) or UniDepth "
                         "per-pixel confidence (deals with imperfect depth).")
    args = ap.parse_args()

    scene = Path(args.scene_dir)
    out = scene / "outputs"
    work = out / "glomap_work"
    depth_dir = out / "unidepth_depth"
    conf_dir = out / "unidepth_conf"

    vis = load_npz(work / "vis.npz").tocsr()
    landmarkx = load_npz(work / "landmarkx.npz").tocsr()
    landmarky = load_npz(work / "landmarky.npz").tocsr()
    filename = np.load(work / "filename.npy", allow_pickle=True)

    name2frame = name_to_frame_index(scene)
    K_by_name = load_intrinsics(scene, args.intrinsics)  # gt | glomap | unidepth

    edges = []
    weights = []
    landmarks = []

    n_frames = vis.shape[0]
    frames_with_points = 0
    empty_frames = 0
    skipped_no_gt = 0
    depth_cache = {}
    conf_cache = {}
    use_conf = args.weight == "confidence"

    for i in range(n_frames):
        name = filename[i]
        if name is None:
            empty_frames += 1
            continue
        name = str(name).strip()
        if name not in name2frame or name not in K_by_name:
            skipped_no_gt += 1
            continue
        frame_idx = name2frame[name]

        col_indices = vis[i].nonzero()[1]
        if col_indices.size == 0:
            empty_frames += 1
            continue

        u = np.rint(landmarkx.getrow(i).data).astype(np.int64)
        v = np.rint(landmarky.getrow(i).data).astype(np.int64)

        stem = Path(name).stem
        if stem not in depth_cache:
            depth_path = depth_dir / f"{stem}.npy"
            if not depth_path.exists():
                empty_frames += 1
                continue
            depth_cache[stem] = np.load(depth_path).astype(np.float64)
        depth = depth_cache[stem]
        h, w = depth.shape

        in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u, v, col = u[in_bounds], v[in_bounds], col_indices[in_bounds]
        if u.size == 0:
            empty_frames += 1
            continue

        d = depth[v, u]
        good = np.isfinite(d) & (d > 0)
        u, v, col, d = u[good], v[good], col[good], d[good]
        if u.size == 0:
            empty_frames += 1
            continue

        K = K_by_name[name]["K"]
        Kinv = np.linalg.inv(K)
        pix = np.vstack((u.astype(np.float64), v.astype(np.float64), np.ones_like(u, dtype=np.float64)))
        cam = (Kinv @ pix).T * d[:, None]  # (n, 3) camera-frame points

        if use_conf:
            if stem not in conf_cache:
                cpath = conf_dir / f"{stem}.npy"
                conf_cache[stem] = (
                    np.load(cpath).astype(np.float64) if cpath.exists() else None
                )
            cmap = conf_cache[stem]
            if cmap is not None and cmap.shape == depth.shape:
                w_obs = np.clip(cmap[v, u], 1e-6, None)[:, None]
            else:
                w_obs = np.ones((col.shape[0], 1), dtype=np.float64)
        else:
            w_obs = np.ones((col.shape[0], 1), dtype=np.float64)

        edges.append(np.column_stack((col.astype(np.int64),
                                      np.full(col.shape[0], frame_idx, dtype=np.int64))))
        weights.append(w_obs)
        landmarks.append(cam)
        frames_with_points += 1

    if not edges:
        raise RuntimeError("No valid 3D observations were produced.")

    edges = np.concatenate(edges, axis=0)
    weights = np.concatenate(weights, axis=0)
    landmarks = np.concatenate(landmarks, axis=0)

    if use_conf:
        # UniDepth confidence is heavy-tailed (a few pixels >> the rest); clip to
        # [p1, p99] so no single observation dominates after per-frame weight
        # normalization in the solver.
        lo, hi = np.percentile(weights, [1.0, 99.0])
        weights = np.clip(weights, max(lo, 1e-6), hi)

    # Canonical name read by static_solver.py (depth_source=unidepth), plus a
    # source-tagged copy so the three intrinsic variants can be kept side by side.
    npz_path = out / "output_tapnet_unidepth.npz"
    np.savez(npz_path, edges=edges, weights=weights, landmarks=landmarks)
    np.savez(out / f"output_tapnet_unidepth__{args.intrinsics}_{args.weight}.npz",
             edges=edges, weights=weights, landmarks=landmarks)
    np.save(out / "edges.npy", edges)
    np.save(out / "weights.npy", weights)
    np.save(out / "landmarks.npy", landmarks)

    summary = {
        "scene": str(scene),
        "intrinsics_source": args.intrinsics,
        "weight_mode": args.weight,
        "num_observations": int(edges.shape[0]),
        "num_frames_total": len(name2frame),
        "num_glomap_frames": int(n_frames),
        "frames_with_points": int(frames_with_points),
        "empty_frames": int(empty_frames),
        "skipped_no_gt": int(skipped_no_gt),
        "num_landmarks": int(edges[:, 0].max() + 1),
        "max_frame_index": int(edges[:, 1].max()),
        "solver_npz": str(npz_path),
    }
    with open(out / "lift_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
