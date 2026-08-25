import os
import sys
import time

from utils.creatematrix import create_matrix
from utils.io import save_matrix_to_bin, load_matrix_from_bin
from utils.recoversolution import recover_XM
from utils.visualization import visualize_camera, visualize
from utils.checkconnection import checklandmarks
from utils.readgt_BAL import load_BAL_gt
from utils.readgt_colmap import load_colmap_gt
from utils.readgt_replica import load_replica_gt
from utils.error import ATE_TEASER_C2W

import numpy as np

total_start = time.perf_counter()
import networkx as nx
import random

############################
# For experiment in paper only
# download the dataset from:
# https://drive.google.com/drive/folders/13_2mcKGKVU0ibWck2n4ajUrN2MaDfR7y?usp=sharing
# and put it in the folder './assets/Experiment/**'
############################

dataset_path = "../datasets/BAL/BAL-394/"
output_path = dataset_path 
# data = np.loadtxt('../assets/train/xm_observations_tapnet_static.txt')

# # dalete the row that contains the same edges in view-graph
# edges = data[:,:2].astype(int)
# _, unique_indices = np.unique(edges, axis=0, return_index=True)
# edges = edges[unique_indices,:]
# data = data[unique_indices,:]

# weights = np.ones((edges.shape[0]))
# landmarks = data[:,2:5]

# edges = edges[:, [1,0]] + 1

# # only preserve camera < 3
# indices = np.isin(edges[:, 0], [1, 35, 70])
# edges = edges[indices]
# weights = weights[indices]
# landmarks = landmarks[indices]

# data, n_observation = load_matrix_from_bin(dataset_path + '/landmark.bin')
# edges = data[:, :2].astype(int) + 1
# weights = data[:, 5]
# landmarks = data[:, 2:5]

gt = load_BAL_gt(dataset_path)
# gt = load_replica_gt(dataset_path)
# gt = load_colmap_gt(dataset_path)

# visualize the camera poses
Abar,_ = load_matrix_from_bin(output_path + '/Abar.bin')
R,_ = load_matrix_from_bin(output_path + '/R.bin')
s,_ = load_matrix_from_bin(output_path + '/s.bin')
Q,_ = load_matrix_from_bin(output_path + '/Q.bin')  

N = R.shape[0] // 3
M = Abar.shape[1]

print("Number of cameras: ", N)
print("Number of landmarks: ", M)

# recover p and t
# details refer to our paper
R_real, s_real, p_est, t_est, optimal_rank = recover_XM(Q, R, s, Abar, 0.0, return_rank=True)

# save the results
npz_path = './result_XM.npz'
np.savez(npz_path, R=R_real, s=s_real, p=p_est, t=t_est)

extrinsics = []
for i in range(N):
    # extrinsic is 4 * 4
    # extrinsic is world 2 camera, while R_real is camera 2 world
    extrinsics.append(np.vstack((np.hstack((R_real[:,3*i:3*i+3].T, -R_real[:,3*i:3*i+3].T @ t_est[:,i].reshape([3,1]))), np.array([0, 0, 0, 1]))))

# only visualize the camera poses
# visualize_camera(extrinsics)
print(s_real, np.max(s_real), np.min(s_real))

# visualize all
p_est = p_est - np.median(p_est, axis=1, keepdims=True)
far_points = np.linalg.norm(p_est, axis=0) > 1.5 * np.median(np.linalg.norm(p_est, axis=0))
# p_est = p_est[:, ~far_points]
visualize(extrinsics, p_est.T)

# uncomment this if you need to check the accuracy
# calculate accuracy
data_RPE_R = []
data_RPE_T = []
data_ATE_R = []
data_ATE_T = []
t_gt = np.zeros((3, N))
R_gt = np.zeros((3, 3 * N))

for i in range(N):
    t_gt[:,i] = gt[i]["t"]  
    R_gt[:,3*i:3*i+3] = gt[i]["R"]

avg_t_gt = np.mean(t_gt,axis=1)
cov_t_gt = np.mean(np.linalg.norm(t_gt - avg_t_gt.reshape(3,1),axis=0))

s_g, R_g, t_g = ATE_TEASER_C2W(R_real,t_est,R_gt,t_gt)

ATE_R = np.zeros(N)
ATE_T = np.zeros(N)

for i in range(N):
    cosvalue = (np.trace(R_g @ R_real[:,3*i:3*i+3] @ R_gt[:,3*i:3*i+3])-1)/2
    ATE_R[i] = np.abs(np.arccos(min(max(cosvalue,-1),1)))
    ATE_T[i] = np.linalg.norm((s_g * R_g @ t_est[:,i] + t_g.flatten())-R_gt[:,3*i:3*i+3].T @ (-t_gt[:,i]))
    
RPE_R = []
RPE_t = []
for i in range(N):
    if N > 1000:
        for j in random.sample(list(np.arange(N)), 100):
            cosvalue = (np.trace(R_gt[:,3*i:3*i+3] @ R_gt[:,3*j:3*j+3].T @ R_real[:,3*j:3*j+3].T @  R_real[:,3*i:3*i+3])-1)/2
            RPE_R.append(np.abs(np.arccos(min(max(cosvalue,-1),1))))
            RPE_t.append(np.linalg.norm(- R_gt[:,3*i:3*i+3].T @ t_gt[:,i] + R_gt[:,3*j:3*j+3].T @ t_gt[:,j] - s_g * R_g @ (t_est[:,i] - t_est[:,j])))
    else:
        for j in range(i):
            cosvalue = (np.trace(R_gt[:,3*i:3*i+3] @ R_gt[:,3*j:3*j+3].T @ R_real[:,3*j:3*j+3].T @  R_real[:,3*i:3*i+3])-1)/2
            RPE_R.append(np.abs(np.arccos(min(max(cosvalue,-1),1))))
            RPE_t.append(np.linalg.norm(- R_gt[:,3*i:3*i+3].T @ t_gt[:,i] + R_gt[:,3*j:3*j+3].T @ t_gt[:,j] - s_g * R_g @ (t_est[:,i] - t_est[:,j])))
    
def _load_scalar_bin(path):
    if not os.path.exists(path):
        return np.nan
    mat, _ = load_matrix_from_bin(path)
    if mat is None or mat.size == 0:
        return np.nan
    return float(mat.reshape(-1)[0])

opt_gap = _load_scalar_bin(os.path.join(dataset_path, "opt_gap.bin"))
min_eig = _load_scalar_bin(os.path.join(dataset_path, "min_eig.bin"))
total_time = time.perf_counter() - total_start

print('RPE-R: ', np.median(RPE_R),'RPE-T: ', np.median(RPE_t)/cov_t_gt,'ATE-R: ', np.median(ATE_R),'ATE-T: ', np.median(ATE_T)/cov_t_gt)
print('Total time: ', total_time)
print('Optimality gap: ', opt_gap)
print('Min eig: ', min_eig)
print('Optimal rank: ', optimal_rank)
data_RPE_R.append(np.median(RPE_R))   
data_RPE_T.append(np.median(RPE_t)/cov_t_gt)
data_ATE_R.append(np.median(ATE_R))
data_ATE_T.append(np.median(ATE_T)/cov_t_gt)
