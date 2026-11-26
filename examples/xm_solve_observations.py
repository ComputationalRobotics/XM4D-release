"""XM from 3D observations: what step 3 (Python/static_solver.py) does after lifting the tracks to 3D, here on given
observations with known ground truth.

Every observation is a landmark seen by a camera, as a 3D point in that camera's frame (in a pipeline: the pixel
lifted with the depth and the intrinsics). examples/data/SIMPLE2/landmark.bin holds one row per observation:
camera index, landmark index (both 1-based), x, y, z, weight. XM finds, for every camera i, a rotation R_i, a
translation t_i and a scale s_i (the depth's unknown scale), and every landmark's position p_j, minimizing
sum_ij w_ij || s_i R_i x_ij + t_i - p_j ||^2, globally. Steps:
  1. checklandmarks: drop cameras with few observations and landmarks seen once, keep the largest connected part;
  2. create_matrix: build the problem's matrices (Q.bin and the others) in a folder;
  3. XM.solve_mixed: solve the relaxation (R.bin, s.bin);
  4. recover_XM: the rotations, scales, translations and landmarks of the solution.
The result is then compared with the ground truth (gtR.bin, gtt.bin) after a similarity alignment.

Run from the repo root in the XM environment: python examples/xm_solve_observations.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path += [str(ROOT / "XM" / "build"), str(ROOT / "Python")]
import XM  # noqa: E402
from utils.checkconnection import checklandmarks  # noqa: E402
from utils.creatematrix import create_matrix  # noqa: E402
from utils.io import load_matrix_from_bin  # noqa: E402
from utils.recoversolution import recover_XM  # noqa: E402

DATA = ROOT / "examples" / "data" / "SIMPLE2"
LAMBDA = 0.0   # regularization of the scales (step 3 uses LAMBDA of config.json in its first solve)

data, _ = load_matrix_from_bin(str(DATA / "landmark.bin"))
_, first = np.unique(data[:, :2].astype(int), axis=0, return_index=True)   # one observation per (camera, landmark)
data = data[np.sort(first)]
edges, landmarks, weights = data[:, :2].astype(int), data[:, 2:5], data[:, 5]
N, M = int(edges[:, 0].max()), int(edges[:, 1].max())
print(f"{N} cameras, {M} landmarks, {len(edges)} observations")

edges, landmarks, weights, indices_all = checklandmarks(edges, landmarks, weights, N, M, frame_min_points=0)
N = int(edges[:, 0].max())

work = Path(tempfile.mkdtemp(prefix="xm_"))
Abar = create_matrix(weights, edges, landmarks, str(work))
XM.solve_mixed(str(work), 5, 1e-6, LAMBDA, 30, 0.9)
R, _ = load_matrix_from_bin(str(work / "R.bin"))
s, _ = load_matrix_from_bin(str(work / "s.bin"))
Q, _ = load_matrix_from_bin(str(work / "Q.bin"))
R, s, p, t, rank = recover_XM(Q, R, s, Abar, LAMBDA)   # R (3, 3N) camera-to-world, t (3, N) camera centres
print(f"solved at rank {rank}: {N} cameras, {p.shape[1]} landmarks")

# compare with the ground truth: gtR, gtt are world-to-camera, so the true camera centre is -R^T t
gtR, _ = load_matrix_from_bin(str(DATA / "gtR.bin"))
gtt, _ = load_matrix_from_bin(str(DATA / "gtt.bin"))
orig = np.array([int(np.flatnonzero(indices_all == i)[0]) for i in range(N)])   # solved camera -> input camera
C_gt = np.stack([-gtR[:, 3 * k:3 * k + 3].T @ gtt[:, k] for k in orig], 1)
C = np.asarray(t)
mu, mu_gt = C.mean(1, keepdims=True), C_gt.mean(1, keepdims=True)
U, S, Vt = np.linalg.svd((C_gt - mu_gt) @ (C - mu).T)
D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
Rg = U @ D @ Vt                                                     # similarity: C_gt ~ c Rg C + b
c = np.trace(np.diag(S) @ D) / np.sum((C - mu) ** 2)
b = mu_gt - c * Rg @ mu
ate = np.linalg.norm(c * Rg @ C + b - C_gt, axis=0) / np.mean(np.linalg.norm(C_gt - mu_gt, axis=0))
rot = [np.degrees(np.arccos(np.clip((np.trace(Rg @ R[:, 3 * i:3 * i + 3] @ gtR[:, 3 * k:3 * k + 3]) - 1) / 2, -1, 1)))
       for i, k in enumerate(orig)]
print(f"camera centre error (median, relative to the scene size): {np.median(ate):.2e}; "
      f"rotation error (median): {np.median(rot):.2e} deg")
print(f"outputs in {work}")
