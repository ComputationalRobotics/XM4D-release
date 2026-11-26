"""The XM solver alone: the matrix Q of the problem in, the camera rotations and scales out.

Q is the data matrix of XM's semidefinite relaxation of scaled bundle adjustment (see the XM paper,
arxiv.org/abs/2502.04640); examples/data/SIMPLE1/Q.bin has 149 cameras. The solver writes R.bin (3N x r: the N
rotations, stacked, at the rank r where the solution was certified, 3 when the relaxation is tight) and s.bin (the N
scales) next to Q.bin. To get the camera positions and the 3D
points too, the solver needs the matrices that Q is built from: see xm_solve_observations.py.

Run from the repo root in the XM environment: python examples/xm_solve_Q.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path += [str(ROOT / "XM" / "build"), str(ROOT / "Python")]
import XM  # noqa: E402
from utils.io import load_matrix_from_bin  # noqa: E402

work = Path(tempfile.mkdtemp(prefix="xm_"))
shutil.copy(ROOT / "examples" / "data" / "SIMPLE1" / "Q.bin", work)

# XM.solve_mixed(folder with Q.bin, max rank, gradient tolerance, scale regularization lambda, time limit per rank in s,
#                sphere ratio): Riemannian staircase from rank 3 up to the max rank until the solution is certified
XM.solve_mixed(str(work), 5, 1e-6, 0.0, 30, 0.9)

R, _ = load_matrix_from_bin(str(work / "R.bin"))
s, _ = load_matrix_from_bin(str(work / "s.bin"))
print(f"{R.shape[0] // 3} cameras, solved at rank {R.shape[1]}; scales s[:5] = {s.ravel()[:5]}")
print(f"outputs in {work}")
