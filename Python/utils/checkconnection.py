import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

def delete_thereshold(min_threshold, M, data):
    valid_frames = np.bincount(data, minlength=M)
    valid_indices = valid_frames > min_threshold
    wrong_indices = valid_frames <= min_threshold
    num_valid_frames = np.sum(valid_indices)
    frame_index = np.zeros(M, dtype=int)
    frame_index[valid_indices] = np.arange(num_valid_frames)
    frame_index[wrong_indices] = -1
    max_frame = np.argmax(valid_frames)
    return max_frame, num_valid_frames, frame_index

def checklandmarks(edges, landmarks, weights, rgbs, N, M, use_max_frame=True,
                   frame_min_points=10):
    # delete and reindex the frames whose number of observations is <= frame_min_points.
    # Raise frame_min_points to drop weakly-observed frames (e.g. IMC gate).
    max_frame, N, indices_frame = delete_thereshold(frame_min_points, N, edges[:,0]-1)

    # exchange the first frame
    if use_max_frame:
        if indices_frame[max_frame] != 0:
            indices_frame[indices_frame == 0] = indices_frame[max_frame]
            indices_frame[max_frame] = 0

    indices_all = indices_frame.copy()  
    edges[:,0] = indices_frame[edges[:,0]-1].copy() + 1
    # delete the row that edges contain -1
    indices = np.any(edges == 0, axis=1)
    edges = edges[~indices]
    weights = weights[~indices]
    landmarks = landmarks[~indices]
    rgbs = rgbs[~indices]

    # delete and reindex the landmarks that contains one frame
    # Note: raise the threshold to also drop landmarks seen in few frames, e.g. IMC gate
    _, M, indices_landmarks = delete_thereshold(0, M, edges[:,1]-1)
    edges[:,1] = indices_landmarks[edges[:,1]-1].copy() + 1
    
    # delete the row that edges contain -1
    indices = np.any(edges == 0, axis=1)
    print("delete:", np.sum(indices), "edges")
    edges = edges[~indices]
    weights = weights[~indices]
    rgbs = rgbs[~indices]
    landmarks = landmarks[~indices]
    
    max_frame, N, indices_frame = delete_thereshold(0, N, edges[:,0]-1)
    edges[:,0] = indices_frame[edges[:,0]-1].copy() + 1

    N_old = np.where(indices_all > -1)[0].shape[0]
    indices_all_copy = indices_all.copy()
    for i in range(N_old):
        indices_all[np.where(indices_all_copy == i)[0]] = indices_frame[i]
    # delete the row that edges contain -1
    indices = np.any(edges == 0, axis=1)
    edges = edges[~indices]
    weights = weights[~indices]
    landmarks = landmarks[~indices]
    rgbs = rgbs[~indices]

    # Components of the frame-landmark graph (nodes u and v + N). Same choice as the networkx version: the largest
    # component by node count, ties going to the component met first in edge order; an edge's two nodes always share
    # a component, so an edge is kept when its frame node is in the chosen one.
    n_nodes = N + M + 1
    graph = coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1] + N)), shape=(n_nodes, n_nodes))
    _, labels = connected_components(graph, directed=False)
    present = np.zeros(n_nodes, dtype=bool)
    present[edges[:, 0]] = True
    present[edges[:, 1] + N] = True
    sizes = np.bincount(labels[present], minlength=labels.max() + 1)
    edge_label = labels[edges[:, 0]]
    first_edge = np.full(sizes.shape[0], edges.shape[0], dtype=np.int64)
    np.minimum.at(first_edge, edge_label, np.arange(edges.shape[0]))
    print("Number of connected components: ", int(np.count_nonzero(sizes)))
    tied = np.flatnonzero(sizes == sizes.max())
    largest_label = tied[np.argmin(first_edge[tied])]
    filtered_indices = np.flatnonzero(edge_label == largest_label)
    if filtered_indices.shape[0] < edges.shape[0]:
        print("Not connected, Choose Largest Component")
        edges = edges[filtered_indices]
        weights = weights[filtered_indices]
        rgbs = rgbs[filtered_indices]
        landmarks = landmarks[filtered_indices]
        _, N, indices_frame = delete_thereshold(0, N, edges[:,0]-1)
        edges[:,0] = indices_frame[edges[:,0]-1].copy() + 1

        N_old = np.where(indices_all > -1)[0].shape[0]
        indices_all_copy = indices_all.copy()
        for i in range(N_old):
            indices_all[np.where(indices_all_copy == i)[0]] = indices_frame[i]
        
        _, M, indices_landmarks = delete_thereshold(0, M, edges[:,1]-1)
        edges[:,1] = indices_landmarks[edges[:,1]-1].copy() + 1
        # here do not need to delete edges because we already know the graph is connected
        
    return edges, landmarks, weights, rgbs, indices_all
