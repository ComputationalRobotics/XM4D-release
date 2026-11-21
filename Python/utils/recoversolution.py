import numpy as np
from scipy.linalg import eigh, svd

def recover_XM(Q, R, s, Abar, lam, return_rank: bool = False):
    N = s.shape[0]
    optimal_rank = None
        
    sR = np.zeros((3*N, R.shape[1]))
    for i in range(N):
        sR[3*i:3*i+3,:] = s[i] * R[3*i:3*i+3,:]
        
    if R.shape[1] > 3:
        X = sR @ sR.T

        eig_vals, eig_vecs = eigh(X)

        idx = np.argsort(eig_vals)[::-1]
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:, idx]

        top_3_eig_vecs = eig_vecs[:, :3]

        sR_real = top_3_eig_vecs * np.sqrt(eig_vals[:3])
        sR_real = sR_real.T
        
        if(abs(eig_vals[3]/eig_vals[2]) < 1e-3):
            optimal_rank = 3
            print("Optimal rank is 3", " with eigenvalue ratio ", abs(eig_vals[3]/eig_vals[2]))
        else:
            optimal_rank = int(R.shape[1])
            print("Optimal rank is ", optimal_rank, " with eigenvalue ratio ", abs(eig_vals[3]/eig_vals[2]))
            X_new = sR_real.T @ sR_real
            suboptimality = np.sum(Q * (X_new - X)) + lam * np.sum((np.diag(X_new)-1)**2)/3 - lam * np.sum((np.diag(X) - 1)**2)/3
            print("suboptimality: ", suboptimality)

    else:
        optimal_rank = int(R.shape[1])
        s_real = s.squeeze()
        R_real = R.T
        sR_real = np.zeros((3, 3*N))
        for i in range(N):
            sR_real[:,3*i:3*i+3] = s_real[i] * R_real[:,3*i:3*i+3]

        
    s_real = np.zeros(N)
    R_real = np.zeros((3, 3*N))

    for i in range(N):
        s_real[i] = np.linalg.norm(sR_real[:, 3*i:3*i+3], 'fro') / np.sqrt(3)
        R_real[:, 3*i:3*i+3] = sR_real[:, 3*i:3*i+3] / s_real[i]

    print("s_real: ", s_real)

    # rank > 3: the rank-3 rounding leaves blocks that are not orthogonal; replace each by its nearest orthogonal
    # matrix (determinant sign kept, so a global reflection still cancels in the anchoring below)
    if optimal_rank != 3:
        for i in range(N):
            U, S, Vt = svd(R_real[:, 3*i:3*i+3])
            R_real[:, 3*i:3*i+3] = U @ Vt

    # because of anchoring
    R1 = R_real[:,:3]
    R_real = R1.T @ R_real

    negative_R = 0
    for i in range(N):
        U, S, Vt = svd(R_real[:,3*i:3*i+3])

        if np.linalg.det(U @ Vt) < 0:
            negative_R = negative_R + 1

    # blocks with det(R) < 0 are projected back to SO(3) below
    if negative_R > 0:
        print("warning: some det(R) < 0")


    # judge which is the largest component
    if negative_R == N:
        R_real = -R_real

    # normalize s and R
    s_real /= s_real[0]
    R1 = R_real[:, :3].copy()          # the anchor before the loop overwrites block 0
    for i in range(N):
        R_real[:, 3*i:3*i+3] = R1.T @ R_real[:, 3*i:3*i+3]

    if negative_R > 0 and negative_R < N:
        for i in range(N):
            A = R_real[:, 3*i:3*i+3]
            U, S, Vt = svd(A)

            R = U @ Vt
            if np.linalg.det(R) < 0:
                # flip the singular direction with the smallest singular value (minimal change)
                k = int(np.argmin(S))          # index of smallest singular value
                D = np.eye(3)
                D[k, k] = -1.0
                R = U @ D @ Vt                 # now det(R) = +1

            R_real[:, 3*i:3*i+3] = R
            sR_real[:, 3*i:3*i+3] = s_real[i] * R

    # Compute ybar_est
    # scale to s = 1
    sR_real = np.zeros((3, 3*N))
    for i in range(N):
        sR_real[:,3*i:3*i+3] = s_real[i] * R_real[:,3*i:3*i+3]
    ybar_est = Abar @ sR_real.T

    # Add a column of zeros to the left
    y_est = np.hstack((np.zeros((3, 1)), ybar_est.T))

    # Compute final outputs
    t_est = y_est[:, :N]
    p_est = y_est[:, N:]
    
    # previous 2 lines changed to:
    #t_est = y_est[:, 1:N+1]   # Skip the zero column
    #p_est = y_est[:, N+1:]

    if return_rank:
        return R_real, s_real, p_est, t_est, optimal_rank
    return R_real, s_real, p_est, t_est

