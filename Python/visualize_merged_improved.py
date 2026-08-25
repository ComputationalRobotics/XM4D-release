#!/usr/bin/env python3
"""
visualize_merged.py

Merge per-segment 4D reconstructions (saved as reconstruction_results_seg_XX.npz)
into one continuous trajectory by rigidly aligning consecutive segments at
their shared boundary frame, then render ONE merged MP4 video.

Usage:
    python visualize_merged.py --videoname train --outputs_root ../deps/tapnet/outputs --fps 5

FIXES:
- Correct segment count (uses actual number of segments, not size-1)
- Improved rigid alignment with fallback strategies
- Smooth blending at segment boundaries
- Proper handling of all segments including the last one
- Camera trajectory properly indexed
- Better edge case handling
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from typing import Tuple, Dict, List, Optional, Any

# ----------------------------- utilities -----------------------------

def _get_num_segments(base_path: str) -> int:
    """Get the actual number of segments from segment files."""
    seg_starts_path = os.path.join(base_path, "segment_starts.txt")
    seg_ends_path = os.path.join(base_path, "segment_ends.txt")
    
    if not os.path.isfile(seg_starts_path):
        raise FileNotFoundError(f"segment_starts.txt not found at: {seg_starts_path}")
    if not os.path.isfile(seg_ends_path):
        raise FileNotFoundError(f"segment_ends.txt not found at: {seg_ends_path}")
    
    seg_starts = np.loadtxt(seg_starts_path, dtype=int, ndmin=1)
    seg_ends = np.loadtxt(seg_ends_path, dtype=int, ndmin=1)
    
    # Number of segments is the number of entries in segment_starts (or segment_ends)
    num_segs = seg_starts.size
    
    if num_segs != seg_ends.size:
        raise ValueError(f"Mismatch: segment_starts has {seg_starts.size} entries, "
                        f"segment_ends has {seg_ends.size} entries")
    
    print(f"[info] Found {num_segs} segments: starts={seg_starts.tolist()}, ends={seg_ends.tolist()}")
    return num_segs


def _load_static_from_result_xm(base_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load camera trajectory + optional static points from result_XM.npz.
    
    Returns:
        B_list: (T, 3, 3) rotation matrices
        h_list: (T, 3) camera centers
        static_landmarks: (Ns, 3) static points
    """
    p = os.path.join(base_path, "result_XM.npz")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Missing static reconstruction file: {p}")

    data = np.load(p)
    R_real = data["R"]       # 3 x (3T)
    t_est = data["t"]        # 3 x T
    s_est = data["s"].astype(float).squeeze()
    p_est = data["p"].T if "p" in data.files else None  # (Ns, 3) if exists

    T = len(s_est)
    B_list = np.stack([R_real[:, 3*i:3*i+3] for i in range(T)], axis=0).astype(float)
    h_list = t_est.T.astype(float)

    static_landmarks = np.zeros((0, 3), dtype=float)
    if p_est is not None and isinstance(p_est, np.ndarray) and p_est.ndim == 2 and p_est.shape[1] == 3:
        static_landmarks = p_est.astype(float)

    return B_list, h_list, static_landmarks


def _rigid_fit_svd(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find R, t minimizing || (R @ P_i + t) - Q_i ||^2 over corresponding points.
    
    Args:
        P: (N, 3) source points
        Q: (N, 3) target points
    
    Returns:
        R: (3, 3) rotation matrix
        t: (3,) translation vector
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    
    if P.shape != Q.shape or P.shape[1] != 3:
        raise ValueError(f"P, Q must both be (N, 3). Got {P.shape} and {Q.shape}.")
    
    if P.shape[0] < 3:
        # Not enough points for a unique solution
        return np.eye(3), np.zeros(3)

    pbar = P.mean(axis=0)
    qbar = Q.mean(axis=0)
    X = P - pbar
    Y = Q - qbar
    H = X.T @ Y
    
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    t = qbar - (R @ pbar)
    return R, t


def _apply_rigid_transform(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Apply rigid transform x -> R @ x + t to points.
    Handles NaN values by preserving them.
    
    Args:
        points: (..., 3) array of 3D points
        R: (3, 3) rotation matrix
        t: (3,) translation vector
    
    Returns:
        Transformed points with same shape as input
    """
    points = np.asarray(points, dtype=float)
    if points.shape[-1] != 3:
        raise ValueError(f"Points must have last dimension 3, got {points.shape}")

    original_shape = points.shape
    X = points.reshape(-1, 3)
    
    # Find valid (finite) points
    valid_mask = np.all(np.isfinite(X), axis=1)
    
    result = X.copy()
    if valid_mask.any():
        result[valid_mask] = (X[valid_mask] @ R.T) + t.reshape(1, 3)
    
    return result.reshape(original_shape)


def _load_segment_npz(base_path: str, seg: int) -> Tuple[Optional[Dict], str]:
    """Load a segment's reconstruction results."""
    npz_path = os.path.join(base_path, f"reconstruction_results_seg_{seg:02d}.npz")
    if not os.path.isfile(npz_path):
        return None, npz_path
    
    data = np.load(npz_path, allow_pickle=True)
    return {k: data[k] for k in data.files}, npz_path


def _get_segment_bounds(base_path: str, seg: int) -> Tuple[int, int]:
    """Get start and end frame for a segment."""
    seg_starts = np.loadtxt(os.path.join(base_path, "segment_starts.txt"), dtype=int, ndmin=1)
    seg_ends = np.loadtxt(os.path.join(base_path, "segment_ends.txt"), dtype=int, ndmin=1)
    
    if seg < 0 or seg >= seg_starts.size:
        raise IndexError(f"seg={seg} out of range [0, {seg_starts.size})")
    
    return int(seg_starts[seg]), int(seg_ends[seg])


def _get_frames_sorted(seg_data: Dict) -> np.ndarray:
    """Extract and sort frame indices from segment data."""
    if "frames_vis" in seg_data:
        frames = np.asarray(seg_data["frames_vis"], dtype=np.int64).reshape(-1)
        return np.sort(frames)
    
    if "frames_to_keep" in seg_data:
        frames = np.asarray(seg_data["frames_to_keep"], dtype=np.int64).reshape(-1)
        return np.sort(frames)
    
    raise KeyError("Segment NPZ missing both 'frames_vis' and 'frames_to_keep'.")


def _get_trajectories_sorted(seg_data: Dict, key: str) -> np.ndarray:
    """
    Get trajectory data sorted by frame order.
    
    Returns:
        (N, T, 3) array with trajectories in chronological frame order
    """
    if key not in seg_data:
        raise KeyError(f"Segment NPZ missing '{key}'")
    
    X = np.asarray(seg_data[key], dtype=float)
    if X.ndim != 3 or X.shape[2] != 3:
        raise ValueError(f"{key} expected (N, T, 3), got {X.shape}")
    
    if "frames_to_keep" not in seg_data:
        return X
    
    # Reorder to chronological
    solver_frames = np.asarray(seg_data["frames_to_keep"], dtype=np.int64).reshape(-1)
    chrono_frames = np.sort(solver_frames)
    
    frame_to_idx = {int(f): i for i, f in enumerate(solver_frames.tolist())}
    perm = [frame_to_idx[int(f)] for f in chrono_frames.tolist()]
    
    return X[:, perm, :]


# ----------------------------- segment structure -----------------------------

def build_segment_struct(base_path: str, seg: int) -> Optional[Dict[str, Any]]:
    """
    Build a standardized segment structure from NPZ file.
    
    Returns:
        Dictionary with standardized keys or None if segment not found
    """
    seg_data, npz_path = _load_segment_npz(base_path, seg)
    if seg_data is None:
        print(f"[load] Segment {seg:02d} not found: {npz_path}")
        return None
    
    seg_start, seg_end = _get_segment_bounds(base_path, seg)
    frames = _get_frames_sorted(seg_data)
    
    # Get trajectories
    edg_deformed = _get_trajectories_sorted(seg_data, "edg_deformed")
    dyn_mapped = _get_trajectories_sorted(seg_data, "dyn_mapped")
    
    # Handle dimension mismatches
    T_frames = frames.size
    T_edg = edg_deformed.shape[1]
    T_dyn = dyn_mapped.shape[1]
    
    if T_frames != T_edg or T_frames != T_dyn:
        print(f"[load] WARNING seg={seg:02d}: frames={T_frames}, edg_T={T_edg}, dyn_T={T_dyn}. Truncating to min.")
        T_min = min(T_frames, T_edg, T_dyn)
        frames = frames[:T_min]
        edg_deformed = edg_deformed[:, :T_min, :]
        dyn_mapped = dyn_mapped[:, :T_min, :]
    
    # Get node/point IDs
    edg_node_ids = np.asarray(
        seg_data.get("edg_ids", seg_data.get("edg_node_ids", np.array([]))),
        dtype=int
    )
    if edg_node_ids.size == 0:
        raise KeyError(f"seg={seg:02d} missing EDG node IDs")
    
    dyn_kps = np.asarray(
        seg_data.get("unique_kps_final", seg_data.get("unique_kps", np.array([]))),
        dtype=int
    )
    if dyn_kps.size == 0:
        raise KeyError(f"seg={seg:02d} missing dynamic keypoint IDs")
    
    # Get edges
    edges = np.asarray(
        seg_data.get("edges", seg_data.get("edg_edges", np.zeros((0, 2)))),
        dtype=int
    )
    
    result = {
        "seg": seg,
        "seg_start": seg_start,
        "seg_end": seg_end,
        "frames": frames.astype(int),
        "edg_node_ids": edg_node_ids,
        "edg_deformed": edg_deformed.astype(float),
        "edg_edges": edges,
        "dyn_kps": dyn_kps,
        "dyn_mapped": dyn_mapped.astype(float),
    }
    
    print(f"[load] Segment {seg:02d} OK | frames [{frames[0]}..{frames[-1]}] ({frames.size}) "
          f"| EDG nodes={edg_deformed.shape[0]} | DYN points={dyn_mapped.shape[0]}")
    
    return result


# ----------------------------- merging -----------------------------

def find_shared_frame(A: Dict, B: Dict) -> int:
    """
    Find the shared boundary frame between consecutive segments.
    This should be A's end frame which equals B's start frame.
    """
    # The shared frame is where segment A ends
    shared = int(A["seg_end"])
    
    # Verify it's in both frame arrays
    if shared not in A["frames"]:
        print(f"[warn] Shared frame {shared} not in segment A frames. Using A's last frame.")
        shared = int(A["frames"][-1])
    
    if shared not in B["frames"]:
        print(f"[warn] Shared frame {shared} not in segment B frames. Using B's first frame.")
        # Find closest frame in B
        closest_idx = np.argmin(np.abs(B["frames"] - shared))
        shared = int(B["frames"][closest_idx])
    
    return shared


def compute_alignment_transform(
    A: Dict, B: Dict, shared_frame: int,
    blend_window: int = 3
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute rigid transform to align segment A to segment B at shared frame.
    
    Uses multiple strategies:
    1. Shared EDG nodes at boundary frame
    2. Shared dynamic points at boundary frame
    3. Average over a small window around boundary
    
    Returns:
        R: (3, 3) rotation matrix
        t: (3,) translation vector  
        alignment_error: float, mean alignment error
    """
    framesA = A["frames"].astype(int)
    framesB = B["frames"].astype(int)
    
    # Find frame indices
    idxA = np.where(framesA == shared_frame)[0]
    idxB = np.where(framesB == shared_frame)[0]
    
    if idxA.size == 0 or idxB.size == 0:
        print(f"[align] Shared frame {shared_frame} not found in both segments. Using identity.")
        return np.eye(3), np.zeros(3), float('inf')
    
    idxA = int(idxA[0])
    idxB = int(idxB[0])
    
    # Build correspondence from shared EDG nodes
    A_ids = set(A["edg_node_ids"].tolist())
    B_ids = set(B["edg_node_ids"].tolist())
    shared_edg_ids = A_ids.intersection(B_ids)
    
    mapA = {int(nid): i for i, nid in enumerate(A["edg_node_ids"].tolist())}
    mapB = {int(nid): i for i, nid in enumerate(B["edg_node_ids"].tolist())}
    
    # Collect corresponding points
    PA, PB = [], []
    
    for nid in shared_edg_ids:
        ia = mapA[nid]
        ib = mapB[nid]
        
        pA = A["edg_deformed"][ia, idxA, :]
        pB = B["edg_deformed"][ib, idxB, :]
        
        if np.all(np.isfinite(pA)) and np.all(np.isfinite(pB)):
            PA.append(pA)
            PB.append(pB)
    
    # Also try shared dynamic points
    A_kps = set(A["dyn_kps"].tolist())
    B_kps = set(B["dyn_kps"].tolist())
    shared_kps = A_kps.intersection(B_kps)
    
    mapA_kp = {int(k): i for i, k in enumerate(A["dyn_kps"].tolist())}
    mapB_kp = {int(k): i for i, k in enumerate(B["dyn_kps"].tolist())}
    
    for kp in shared_kps:
        ia = mapA_kp[kp]
        ib = mapB_kp[kp]
        
        pA = A["dyn_mapped"][ia, idxA, :]
        pB = B["dyn_mapped"][ib, idxB, :]
        
        if np.all(np.isfinite(pA)) and np.all(np.isfinite(pB)):
            PA.append(pA)
            PB.append(pB)
    
    print(f"[align] Shared frame {shared_frame}: {len(shared_edg_ids)} shared EDG nodes, "
          f"{len(shared_kps)} shared DYN points, {len(PA)} usable correspondences")
    
    if len(PA) < 3:
        print("[align] Not enough correspondences. Using identity transform.")
        return np.eye(3), np.zeros(3), float('inf')
    
    PA = np.array(PA, dtype=float)
    PB = np.array(PB, dtype=float)
    
    # Compute rigid alignment: transform A -> B
    # We want R, t such that R @ pA + t ≈ pB
    R, t = _rigid_fit_svd(PA, PB)
    
    # Check for NaN
    if not (np.all(np.isfinite(R)) and np.all(np.isfinite(t))):
        print("[align] NaN in transform. Using identity.")
        return np.eye(3), np.zeros(3), float('inf')
    
    # Compute alignment error
    PA_transformed = (PA @ R.T) + t
    error = np.mean(np.linalg.norm(PA_transformed - PB, axis=1))
    print(f"[align] Alignment error: {error:.6f}")
    
    return R, t, error


def merge_two_segments(A: Dict, B: Dict, blend_frames: int = 1) -> Dict:
    """
    Merge two consecutive segments with smooth blending at boundary.
    
    Args:
        A: First segment (will be transformed to align with B)
        B: Second segment (reference)
        blend_frames: Number of frames around boundary to blend
    
    Returns:
        Merged segment dictionary
    """
    shared_frame = find_shared_frame(A, B)
    print(f"\n[merge] Merging segments {A['seg']} and {B['seg']} at frame {shared_frame}")
    
    # Compute alignment transform
    R, t, error = compute_alignment_transform(A, B, shared_frame)
    
    # Transform A's trajectories
    A_edg_tf = _apply_rigid_transform(A["edg_deformed"], R, t)
    A_dyn_tf = _apply_rigid_transform(A["dyn_mapped"], R, t)
    
    # Build merged frame list
    framesA = A["frames"].astype(int)
    framesB = B["frames"].astype(int)
    
    # Frames from A before shared (exclusive), shared frame, frames from B after shared (exclusive)
    frames_before = framesA[framesA < shared_frame]
    frames_after = framesB[framesB > shared_frame]
    merged_frames = np.concatenate([
        frames_before,
        np.array([shared_frame], dtype=int),
        frames_after
    ])
    
    Tm = merged_frames.size
    frame_to_tm = {int(f): i for i, f in enumerate(merged_frames.tolist())}
    
    # --- Merge EDG nodes ---
    all_edg_ids = np.union1d(A["edg_node_ids"], B["edg_node_ids"])
    M = all_edg_ids.size
    
    mapA_edg = {int(nid): i for i, nid in enumerate(A["edg_node_ids"].tolist())}
    mapB_edg = {int(nid): i for i, nid in enumerate(B["edg_node_ids"].tolist())}
    mapM_edg = {int(nid): i for i, nid in enumerate(all_edg_ids.tolist())}
    
    merged_edg = np.full((M, Tm, 3), np.nan, dtype=float)
    
    # Fill from A (before shared frame)
    for f in frames_before.tolist():
        tA = int(np.where(framesA == f)[0][0])
        tM = frame_to_tm[f]
        for nid in A["edg_node_ids"].tolist():
            ia = mapA_edg[nid]
            im = mapM_edg[nid]
            merged_edg[im, tM, :] = A_edg_tf[ia, tA, :]
    
    # Fill from B (after shared frame)
    for f in frames_after.tolist():
        tB = int(np.where(framesB == f)[0][0])
        tM = frame_to_tm[f]
        for nid in B["edg_node_ids"].tolist():
            ib = mapB_edg[nid]
            im = mapM_edg[nid]
            merged_edg[im, tM, :] = B["edg_deformed"][ib, tB, :]
    
    # Handle shared frame with blending for shared nodes
    tM_shared = frame_to_tm[shared_frame]
    idxA_shared = int(np.where(framesA == shared_frame)[0][0]) if shared_frame in framesA else None
    idxB_shared = int(np.where(framesB == shared_frame)[0][0]) if shared_frame in framesB else None
    
    for nid in all_edg_ids.tolist():
        posA = None
        posB = None
        
        if nid in mapA_edg and idxA_shared is not None:
            ia = mapA_edg[nid]
            posA = A_edg_tf[ia, idxA_shared, :]
            if not np.all(np.isfinite(posA)):
                posA = None
        
        if nid in mapB_edg and idxB_shared is not None:
            ib = mapB_edg[nid]
            posB = B["edg_deformed"][ib, idxB_shared, :]
            if not np.all(np.isfinite(posB)):
                posB = None
        
        im = mapM_edg[nid]
        if posA is not None and posB is not None:
            # Blend: average
            merged_edg[im, tM_shared, :] = 0.5 * (posA + posB)
        elif posB is not None:
            merged_edg[im, tM_shared, :] = posB
        elif posA is not None:
            merged_edg[im, tM_shared, :] = posA
    
    # --- Merge edges ---
    A_edges = A.get("edg_edges", np.zeros((0, 2), dtype=int))
    B_edges = B.get("edg_edges", np.zeros((0, 2), dtype=int))
    
    all_edges = []
    edge_set = set()
    
    for edges in [A_edges, B_edges]:
        if edges.size == 0:
            continue
        for e in edges.tolist():
            a, b = int(e[0]), int(e[1])
            if a > b:
                a, b = b, a
            if (a, b) not in edge_set and a in mapM_edg and b in mapM_edg:
                edge_set.add((a, b))
                all_edges.append([a, b])
    
    merged_edges = np.array(all_edges, dtype=int) if all_edges else np.zeros((0, 2), dtype=int)
    
    # --- Merge dynamic points ---
    all_kps = np.union1d(A["dyn_kps"], B["dyn_kps"])
    Nd = all_kps.size
    
    mapA_kp = {int(k): i for i, k in enumerate(A["dyn_kps"].tolist())}
    mapB_kp = {int(k): i for i, k in enumerate(B["dyn_kps"].tolist())}
    mapM_kp = {int(k): i for i, k in enumerate(all_kps.tolist())}
    
    merged_dyn = np.full((Nd, Tm, 3), np.nan, dtype=float)
    
    # Fill from A
    for f in frames_before.tolist():
        tA = int(np.where(framesA == f)[0][0])
        tM = frame_to_tm[f]
        for kp in A["dyn_kps"].tolist():
            ia = mapA_kp[kp]
            im = mapM_kp[kp]
            merged_dyn[im, tM, :] = A_dyn_tf[ia, tA, :]
    
    # Fill from B
    for f in frames_after.tolist():
        tB = int(np.where(framesB == f)[0][0])
        tM = frame_to_tm[f]
        for kp in B["dyn_kps"].tolist():
            ib = mapB_kp[kp]
            im = mapM_kp[kp]
            merged_dyn[im, tM, :] = B["dyn_mapped"][ib, tB, :]
    
    # Handle shared frame
    for kp in all_kps.tolist():
        posA = None
        posB = None
        
        if kp in mapA_kp and idxA_shared is not None:
            ia = mapA_kp[kp]
            posA = A_dyn_tf[ia, idxA_shared, :]
            if not np.all(np.isfinite(posA)):
                posA = None
        
        if kp in mapB_kp and idxB_shared is not None:
            ib = mapB_kp[kp]
            posB = B["dyn_mapped"][ib, idxB_shared, :]
            if not np.all(np.isfinite(posB)):
                posB = None
        
        im = mapM_kp[kp]
        if posA is not None and posB is not None:
            merged_dyn[im, tM_shared, :] = 0.5 * (posA + posB)
        elif posB is not None:
            merged_dyn[im, tM_shared, :] = posB
        elif posA is not None:
            merged_dyn[im, tM_shared, :] = posA
    
    result = {
        "seg": int(B["seg"]),
        "seg_start": int(A["seg_start"]),
        "seg_end": int(B["seg_end"]),
        "frames": merged_frames,
        "edg_node_ids": all_edg_ids,
        "edg_deformed": merged_edg,
        "edg_edges": merged_edges,
        "dyn_kps": all_kps,
        "dyn_mapped": merged_dyn,
        "last_transform": {"R": R, "t": t, "error": error},
    }
    
    print(f"[merge] Result: frames [{merged_frames[0]}..{merged_frames[-1]}] ({Tm}) "
          f"| EDG={M} | DYN={Nd}")
    
    return result


def merge_all_segments(base_path: str, limit_segs: Optional[int] = None) -> Tuple[Dict, List[int]]:
    """
    Load and merge all available segments.
    
    Returns:
        merged_all: Merged segment dictionary
        ok_segs: List of successfully loaded segment indices
    """
    num_segs = _get_num_segments(base_path)
    
    if limit_segs is not None:
        num_segs = min(num_segs, int(limit_segs))
    
    # Load all available segments
    seg_structs = {}
    ok_segs = []
    
    for seg in range(num_segs):
        struct = build_segment_struct(base_path, seg)
        if struct is not None:
            seg_structs[seg] = struct
            ok_segs.append(seg)
    
    if not ok_segs:
        raise RuntimeError("No segments could be loaded.")
    
    ok_segs = sorted(ok_segs)
    print(f"\n[merge] Successfully loaded segments: {ok_segs}")
    
    # Start with first segment
    merged = seg_structs[ok_segs[0]]
    print(f"\n=== Starting merge from segment {merged['seg']} ===")
    print(f"    Frames: [{merged['frames'][0]}..{merged['frames'][-1]}]")
    
    # Merge consecutive segments
    for i in range(1, len(ok_segs)):
        prev_seg = ok_segs[i - 1]
        curr_seg = ok_segs[i]
        
        if curr_seg != prev_seg + 1:
            print(f"\n[merge] WARNING: Gap between segments {prev_seg} and {curr_seg}. Skipping merge.")
            continue
        
        merged = merge_two_segments(merged, seg_structs[curr_seg])
    
    print(f"\n=== Merge complete ===")
    print(f"    Final frames: [{merged['frames'][0]}..{merged['frames'][-1]}] ({merged['frames'].size})")
    print(f"    EDG nodes: {merged['edg_node_ids'].size}")
    print(f"    DYN points: {merged['dyn_kps'].size}")
    
    return merged, ok_segs


# ----------------------------- rendering -----------------------------

def _edges_to_local_indices(node_ids: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert global node IDs in edges to local array indices."""
    if edges.size == 0:
        return np.zeros((0,), dtype=int), np.zeros((0,), dtype=int)
    
    id_to_idx = {int(nid): i for i, nid in enumerate(node_ids.tolist())}
    
    Ei, Ej = [], []
    for a, b in edges.tolist():
        if int(a) in id_to_idx and int(b) in id_to_idx:
            Ei.append(id_to_idx[int(a)])
            Ej.append(id_to_idx[int(b)])
    
    return np.array(Ei, dtype=int), np.array(Ej, dtype=int)


def _set_equal_aspect_3d(ax, points: np.ndarray, pad_frac: float = 0.05):
    """Set equal aspect ratio for 3D axes."""
    valid = np.all(np.isfinite(points), axis=1)
    if not valid.any():
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        return
    
    pts = points[valid]
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    
    # Add padding
    rng = mx - mn
    mn = mn - pad_frac * rng
    mx = mx + pad_frac * rng
    
    # Equal aspect
    center = (mn + mx) / 2
    span = max(mx - mn)
    if span <= 0:
        span = 1.0
    half = span / 2
    
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def render_merged_video(
    merged: Dict,
    cam_centers: np.ndarray,
    out_path: str,
    fps: int = 5,
    point_size_edg: float = 18.0,
    point_size_dyn: float = 10.0,
    cam_path_lw: float = 1.5,
    cam_dot_size: float = 120.0,
    elev: float = -54.0,
    azim: float = -5.5,
    show_camera_trajectory: bool = True,
    static_landmarks: Optional[np.ndarray] = None,
):
    """
    Render the merged reconstruction as an MP4 video.
    """
    frames = merged["frames"].astype(int)
    Tf = frames.size
    
    edg = merged["edg_deformed"]  # (M, T, 3)
    dyn = merged["dyn_mapped"]    # (Nd, T, 3)
    M = edg.shape[0]
    Nd = dyn.shape[0]
    
    Tcam = cam_centers.shape[0]
    
    # Collect all points for axis limits
    all_pts = [
        edg.reshape(-1, 3),
        dyn.reshape(-1, 3),
        cam_centers.reshape(-1, 3),
    ]
    if static_landmarks is not None and static_landmarks.size:
        all_pts.append(static_landmarks.reshape(-1, 3))
    
    all_pts = np.vstack(all_pts)
    valid = np.all(np.isfinite(all_pts), axis=1)
    all_pts_valid = all_pts[valid]
    
    if all_pts_valid.size == 0:
        raise RuntimeError("All points are NaN/Inf. Nothing to render.")
    
    # Get edge indices
    Ei, Ej = _edges_to_local_indices(merged["edg_node_ids"], merged["edg_edges"])
    
    # Setup figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)
    _set_equal_aspect_3d(ax, all_pts_valid)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    title = ax.set_title("")
    
    # Static landmarks
    if static_landmarks is not None and static_landmarks.size:
        valid_static = np.all(np.isfinite(static_landmarks), axis=1)
        if valid_static.any():
            s = static_landmarks[valid_static]
            ax.scatter(s[:, 0], s[:, 1], s[:, 2], s=5, c="gray", alpha=0.3)
    
    # Camera trajectory
    cam_valid = np.all(np.isfinite(cam_centers), axis=1)
    if show_camera_trajectory and cam_valid.any():
        cam_line, = ax.plot(
            cam_centers[cam_valid, 0],
            cam_centers[cam_valid, 1],
            cam_centers[cam_valid, 2],
            "b-", linewidth=cam_path_lw, alpha=0.5
        )
    else:
        cam_line, = ax.plot([], [], [], linewidth=cam_path_lw)
    
    # Camera dot
    cam_dot = ax.scatter([], [], [], s=cam_dot_size, c="red", marker="o")
    
    # EDG and DYN scatter
    scat_edg = ax.scatter([], [], [], s=point_size_edg, c="blue", marker="o")
    scat_dyn = ax.scatter([], [], [], s=point_size_dyn, c="green", marker=".", alpha=0.6)
    
    # Edge lines
    edge_line, = ax.plot([], [], [], "b-", linewidth=0.5, alpha=0.5)
    
    def update(ti: int):
        frame_g = int(frames[ti])
        
        # Camera index (clamp to valid range)
        cam_idx = min(max(0, frame_g), Tcam - 1)
        
        # Get current positions
        edg_t = edg[:, ti, :]
        dyn_t = dyn[:, ti, :]
        
        valid_edg = np.all(np.isfinite(edg_t), axis=1)
        valid_dyn = np.all(np.isfinite(dyn_t), axis=1)
        
        # Update EDG scatter
        if valid_edg.any():
            scat_edg._offsets3d = (edg_t[valid_edg, 0], edg_t[valid_edg, 1], edg_t[valid_edg, 2])
        else:
            scat_edg._offsets3d = ([], [], [])
        
        # Update DYN scatter
        if valid_dyn.any():
            scat_dyn._offsets3d = (dyn_t[valid_dyn, 0], dyn_t[valid_dyn, 1], dyn_t[valid_dyn, 2])
        else:
            scat_dyn._offsets3d = ([], [], [])
        
        # Update edges
        if Ei.size > 0:
            valid_edges = valid_edg[Ei] & valid_edg[Ej]
            ii = Ei[valid_edges]
            jj = Ej[valid_edges]
            
            if ii.size > 0:
                X = np.vstack([edg_t[ii, 0], edg_t[jj, 0], np.full(ii.size, np.nan)]).T.ravel()
                Y = np.vstack([edg_t[ii, 1], edg_t[jj, 1], np.full(ii.size, np.nan)]).T.ravel()
                Z = np.vstack([edg_t[ii, 2], edg_t[jj, 2], np.full(ii.size, np.nan)]).T.ravel()
                edge_line.set_data(X, Y)
                edge_line.set_3d_properties(Z)
            else:
                edge_line.set_data([], [])
                edge_line.set_3d_properties([])
        
        # Update camera dot
        cam_pos = cam_centers[cam_idx]
        if np.all(np.isfinite(cam_pos)):
            cam_dot._offsets3d = ([cam_pos[0]], [cam_pos[1]], [cam_pos[2]])
        else:
            cam_dot._offsets3d = ([], [], [])
        
        # Update title
        title.set_text(
            f"Merged | Frame {frame_g} (t={ti}/{Tf}) | "
            f"EDG: {valid_edg.sum()}/{M} | DYN: {valid_dyn.sum()}/{Nd}"
        )
        
        return scat_edg, scat_dyn, edge_line, cam_dot, title
    
    print(f"[render] Creating animation: {Tf} frames at {fps} fps")
    anim = animation.FuncAnimation(fig, update, frames=Tf, interval=1000//fps, blit=False)
    
    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=fps, bitrate=1800)
    anim.save(out_path, writer=writer)
    plt.close(fig)
    
    print(f"[render] Saved: {out_path}")


# ----------------------------- main -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Merge and visualize reconstructed segments")
    parser.add_argument("--videoname", type=str, default="train")
    parser.add_argument("--outputs_root", type=str, default="../deps/tapnet/outputs")
    parser.add_argument("--out_mp4", type=str, default=None)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--limit_segs", type=int, default=None,
                       help="Only process first N segments (for debugging)")
    parser.add_argument("--no_camera", action="store_true",
                       help="Don't show camera trajectory")
    args = parser.parse_args()
    
    base_path = os.path.join(args.outputs_root, args.videoname)
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"Output directory not found: {base_path}")
    
    # Merge segments
    merged, ok_segs = merge_all_segments(base_path, limit_segs=args.limit_segs)
    
    # Load camera trajectory
    B_list, h_list, static_landmarks = _load_static_from_result_xm(base_path)
    
    # Output path
    out_mp4 = args.out_mp4
    if out_mp4 is None:
        out_mp4 = os.path.join(base_path, f"merged_reconstruction_{args.videoname}.mp4")
    
    # Render
    render_merged_video(
        merged=merged,
        cam_centers=h_list,
        out_path=out_mp4,
        fps=args.fps,
        show_camera_trajectory=not args.no_camera,
        static_landmarks=None,  # Set to static_landmarks to show them
    )
    
    print(f"\nDone! Output saved to: {out_mp4}")


if __name__ == "__main__":
    main()
