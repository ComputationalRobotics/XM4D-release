import os
import numpy as np
import time

from concurrent.futures import ThreadPoolExecutor
from scipy.linalg import solve
from scipy.sparse import coo_matrix, csr_matrix, bmat, hstack, diags

from utils.io import save_matrix_to_bin

from scipy.io import savemat


def process_observation(i, ind_l, landmark, weight):
        # Compute weighted landmarks
        v = landmark * weight[:, None]

        # Calculate blocks to update the sparse matrices
        Q1_block = v.T @ landmark
        V1_block = v.sum(axis=0)[:, None]
        V2_block = v.T

        # Return the computed blocks with their indices
        return i, Q1_block, V1_block, V2_block, ind_l


def _sparse_times_dense(S, D, transpose_result=False):
    """S @ D, or D.T @ S with transpose_result, for sparse S and dense D, split over column blocks of D in threads
    (scipy releases the GIL here). Every output entry is summed in the same order as the one-call product, so the
    result is bit-identical; ~10x faster than the single-threaded call on a big machine."""
    nt = max(1, min(32, os.cpu_count() or 1, D.shape[1]))
    bounds = np.linspace(0, D.shape[1], nt + 1).astype(int)
    out = np.empty((D.shape[1], S.shape[1]) if transpose_result else (S.shape[0], D.shape[1]), dtype=np.float64)

    def block(k):
        a, b = bounds[k], bounds[k + 1]
        Dk = np.ascontiguousarray(D[:, a:b])
        if transpose_result:
            out[a:b] = Dk.T @ S
        else:
            out[:, a:b] = S @ Dk

    with ThreadPoolExecutor(nt) as executor:
        list(executor.map(block, range(nt)))
    return out


# create matrix function; returns Abar (written to Abar.bin, and Q.mat, only with debug)
def create_matrix(weight, edges, landmarks, output_path, debug = False):
    weight = np.asarray(weight, dtype=np.float64)
    landmarks = np.asarray(landmarks, dtype=np.float64)

    M = int(edges[:,1].max())
    N = int(edges[:,0].max())
    print(f"M: {M}, N: {N}")
    V3 = coo_matrix((weight, (edges[:, 0]-1, edges[:, 1]-1)), shape=(N, M)).tocsr()

    observation_index = coo_matrix((np.arange(1, len(edges) + 1), (edges[:, 0]-1, edges[:, 1]-1)), shape=(N, M)).tocsr()

    Q2 = diags(V3.sum(axis=1).A1, format='csr')
    Q3 = diags(V3.sum(axis=0).A1, format='csr')  

    # Prepare lists to collect the indices and data for assembly
    Q1_data = []
    V1_data = []
    V2_data = []
    
    print("Start reading data")

    # In-process: a pool of worker processes cost more to start than these tiny per-frame blocks take
    for i in range(N):
        row = observation_index.getrow(i)
        i, Q1_block, V1_block, V2_block, ind_l = process_observation(i, row.indices, landmarks[row.data-1,:], weight[row.data-1])

        # Collect data and indices for Q1
        row_indices_Q1 = np.repeat(np.arange(3*i, 3*i+3), 3)
        col_indices_Q1 = np.tile(np.arange(3*i, 3*i+3), 3)
        Q1_values = Q1_block.flatten()
        Q1_data.append((Q1_values, row_indices_Q1, col_indices_Q1))

        # Collect data and indices for V1
        row_indices_V1 = np.arange(3*i, 3*i+3)
        col_indices_V1 = np.array([i]*3)
        V1_values = V1_block.flatten()
        V1_data.append((V1_values, row_indices_V1, col_indices_V1))

        # Collect data and indices for V2
        rows_V2 = np.repeat(np.arange(3*i, 3*i+3), len(ind_l))
        cols_V2 = np.tile(ind_l, 3)
        V2_values = V2_block.flatten()
        V2_data.append((V2_values, rows_V2, cols_V2))

    print("Finished processing data")

    # Combine collected data for Q1
    Q1_values = np.concatenate([data for data, _, _ in Q1_data])
    Q1_row_indices = np.concatenate([rows for _, rows, _ in Q1_data])
    Q1_col_indices = np.concatenate([cols for _, _, cols in Q1_data])

    # Combine collected data for V1
    V1_values = np.concatenate([data for data, _, _ in V1_data])
    V1_row_indices = np.concatenate([rows for _, rows, _ in V1_data])
    V1_col_indices = np.concatenate([cols for _, _, cols in V1_data])

    # Combine collected data for V2
    V2_values = np.concatenate([data for data, _, _ in V2_data])
    V2_row_indices = np.concatenate([rows for _, rows, _ in V2_data])
    V2_col_indices = np.concatenate([cols for _, _, cols in V2_data])

    # Create sparse matrices
    Q1 = coo_matrix((Q1_values, (Q1_row_indices, Q1_col_indices)), shape=(3*N, 3*N)).tocsr()
    V1 = coo_matrix((V1_values, (V1_row_indices, V1_col_indices)), shape=(3*N, N)).tocsr()
    V2 = coo_matrix((V2_values, (V2_row_indices, V2_col_indices)), shape=(3*N, M)).tocsr()
    
    print("Finished constructing basic matrices")

    # Construct Qtp
    Qtp = bmat([[Q2, -V3],
                [-V3.transpose(), Q3]], format='csr')

    # Construct Vtp
    Vtp = hstack([V1, -V2], format='csr')

    # Extract Qtpbar (all columns from the second to the end)
    Qtpbar = Qtp[:, 1:]

    # Compute a0 (transpose of the first row from the second element to the end)
    a0 = Qtp[0, 1:].transpose()

    # Extract Q2_bar (submatrix from the second row and column to the end)
    Q2_bar = Q2[1:, 1:]

    # Extract V3_bar (all rows from the second to the end)
    V3_bar = V3[1:, :]

    # Compute sqrt_Q3
    sqrt_Q3 = Q3.copy()
    sqrt_Q3.data = np.sqrt(sqrt_Q3.data)

    # Get the diagonal elements
    sqrt_Q3_diag = sqrt_Q3.diagonal()

    # Compute inverse sqrt_Q3_diag and reshape for broadcasting
    inv_sqrt_Q3_diag = (1 / sqrt_Q3_diag).reshape(1, -1)
    inv_sqrt_Q3_diag_T = inv_sqrt_Q3_diag.transpose()

    # Perform element-wise division on V3_bar
    V3_bar_F = V3_bar.multiply(inv_sqrt_Q3_diag)

    # sparse-sparse
    VT = Q2_bar - V3_bar_F @ V3_bar_F.transpose()
    
    print("Finished constructing VT matrices")
    
    # if SLAM this is sparse matrix
    VT = VT.todense()

    # sparse-sparse
    RHS_left = Qtpbar.transpose() @ Vtp.transpose()

    RHS = hstack([RHS_left, a0], format='csr')

    RHS_A = RHS[:N-1, :]  # Rows 0 to N-2
    RHS_B = RHS[N-1:, :]

    RHS_A = RHS_A.todense()
    RHS_B = RHS_B.todense()

    start_time_mul = time.perf_counter()
    
    # Convert inputs to NumPy and SciPy objects
    RHS_B = np.array(RHS_B, dtype=np.float64)
    RHS_A = np.array(RHS_A, dtype=np.float64)
    VT = np.array(VT, dtype=np.float64)
    V3_bar_F = V3_bar_F.toarray().astype(np.float64)
    inv_sqrt_Q3_diag_T = np.array(inv_sqrt_Q3_diag_T, dtype=np.float64)
    a0 = csr_matrix(a0, dtype=np.float64)
    
    # Main computation loop
    for i in range(2):
        print(f"iteration {i}")
        RHS_B = np.multiply(RHS_B, inv_sqrt_Q3_diag_T)
        RHS_A = solve(VT, RHS_A + V3_bar_F @ RHS_B)
        RHS_B += np.dot(V3_bar_F.T,RHS_A)
        RHS_B = np.multiply(RHS_B, inv_sqrt_Q3_diag_T)

    # Initialize matrix A
    A = np.zeros((N + M, 3 * N), dtype=np.float64)

    # Update A
    np.negative(RHS_A[:, :-1], out=A[1:N, :])
    np.negative(RHS_B[:, :-1], out=A[N:, :])

    v2 = np.hstack([RHS_A[:, -1], RHS_B[:, -1]])

    del RHS_A, RHS_B, VT

    # Compute S
    S = 1 + a0.T @ v2
    
    print("updating A")

    if S == 0:
        raise ValueError("S is 0")

    # all columns at once: A[1:, col] -= v2 * (a0.T @ A[1:, col]) / S (same per-entry arithmetic)
    A[1:, :] -= v2[:, None] * (a0.T @ A[1:, :]) / S

    # Convert A to dense NumPy array
    Abar = A[1:, :]

    # Save Abar to a binary file
    if debug:
        save_matrix_to_bin(output_path + "/Abar.bin", Abar)
    Abar = np.asfortranarray(Abar)    # the layout Abar.bin was read back with

    middle = _sparse_times_dense(Qtp, A, transpose_result=True)    # A.T @ Qtp

    # Compute matrix C
    C = np.dot(middle , A)

    del Qtp

    T = _sparse_times_dense(Vtp, A)    # Vtp @ A
    C += T + T.T
    
    C += Q1

    if np.linalg.norm(C - C.T, 'fro')/N/N >= 1e-8:
        print("Matrix C is not symmetric within the tolerance")
        C = (C + C.T) / 2 

    end_time_mul = time.perf_counter()
    elapsed_time = (end_time_mul - start_time_mul)*1e6
    print(f"Multiplication time: {elapsed_time} microseconds")

    filename_C = output_path + "/Q.bin"
    with open(filename_C, "wb") as file:
        file.write(np.array(3*N, dtype=np.int32).tobytes())
        file.write(np.array(3*N, dtype=np.int32).tobytes())
        
        C.astype(np.float64).tofile(file)

    print(f"Matrix saved to {filename_C}\n")
    
    # save as .mat (MAT v5 has a 4GB limit on a single variable); nothing reads it, so only with debug
    if debug:
        try:
            q_nbytes = C.nbytes
            if q_nbytes >= (2**32 - 1):
                print(f"[WARN] Skip Q.mat (size={q_nbytes} bytes exceeds MAT v5 limit).")
            else:
                savemat(output_path + "/Q.mat", {'Q': C})
        except OverflowError as exc:
            print(f"[WARN] Failed to save Q.mat: {exc}")

    del C, Vtp, Q1, A, V3, Q2, Q3, V3_bar, V3_bar_F, Q2_bar, a0, inv_sqrt_Q3_diag_T, sqrt_Q3_diag, sqrt_Q3, RHS, RHS_left, V1, V2, Qtpbar

    return Abar
