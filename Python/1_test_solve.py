import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../XM_old/build/'))
import numpy as np

import XM

from utils.io import save_matrix_to_bin, load_matrix_from_bin

# The XM solver takes in the Q matrix for the SDP problem 
# (more details refer to our paper: https://arxiv.org/abs/2502.04640)
# and it output the R matrix and the s vector. Note these may in rank higher than 3
# so you will need more information for previous matrix process to recover the XM solution

"""
    Solves the XM problem using the provided Q matrix.

    This function reads the Q matrix from a .bin file (I/O operations are handled by utils.io)
    located in the specified dataset path. It then applies the XM algorithm to compute the R matrix 
    and the s vector. The computed results are saved back as .bin files in the same dataset directory.

    Parameters:
        dataset_path (str): The path to the dataset containing the Q.bin file.
        max_rank (int, optional): The maximum allowed rank for the solution. 
                                  For a full XM solution, a larger value (e.g., 10 or 5) may be used to achieve certifiable global minimum.
                                  Defaults to 10.
        tol (float, optional): The convergence tolerance. The algorithm will stop iterating when the norm of gradient falls below this threshold.
                               Defaults to 1e-6.
        lam (float, optional): The regularization parameter on scale.
                               Defaults to 0.0.
        max_time (float, optional): The maximum allowed computation time (s) for the algorithm.
                                    Defaults to 1000.

    Returns:
        NoneRPE-R:  0.0008987949744307115 RPE-T:  0.0017212043110791696 ATE-R:  0.0006041749206983403 ATE-T:  0.0009824154662642717
        
    C++ function:
    void solve(const std::string& dataset_path, size_t max_rank = 10, double tol = 1e-6, double lam = 0.0, double max_time = 1000)
"""

# full XM
dataset_path = "../datasets/BAL/BAL-394/"
# XM.solve(dataset_path,3,1e-16,0.0,60,False,100)
XM.solve(dataset_path,3,1e-6,0.0,20)
# XM.solve_mixed(dataset_path,5,1e-6,0.0,600,0.1)

s_real, _ = load_matrix_from_bin(f"{dataset_path}/s.bin")
print(s_real)
Q, _ = load_matrix_from_bin(f"{dataset_path}/Q.bin")
print(np.diag(Q), np.max(np.diag(Q)), np.min(np.diag(Q)))
max_index = np.argmax(s_real)
print(Q[max_index*3:max_index*3+3, max_index*3:max_index*3+3])
print(np.max(s_real), np.min(s_real), np.mean(s_real), np.median(s_real))




