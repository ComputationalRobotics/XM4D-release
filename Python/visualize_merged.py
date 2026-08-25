# visualize_merged.py
#
# Merge per-segment 4D reconstructions (saved as reconstruction_results_seg_XX.npz)
# into one continuous trajectory by rigidly aligning consecutive segments at
# their shared boundary frame, then render ONE merged MP4 video.
#
# Usage:
#   python visualize_merged.py --videoname train --outputs_root ../deps/tapnet/outputs --fps 5
#
# Notes:
# - Expects files:
#     {outputs_root}/{videoname}/segment_starts.txt
#     {outputs_root}/{videoname}/segment_ends.txt
#     {outputs_root}/{videoname}/result_XM.npz
#     {outputs_root}/{videoname}/reconstruction_results_seg_XX.npz
#
# IMPORTANT FIXES:
# - Axis limits include CAMERA CENTERS, so camera trajectory isn't clipped.
# - Adds a big RED DOT at the current camera center each frame.

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# ----------------------------- utilities -----------------------------

def _num_segments_from_segment_starts(base_path: str) -> int:
    segStartsPath = os.path.join(base_path, "segment_starts.txt")
    if not os.path.isfile(segStartsPath):
        raise FileNotFoundError(f"segment_starts.txt not found at: {segStartsPath}")
    seg_starts = np.loadtxt(segStartsPath, dtype=int, ndmin=1)
    num_segs = int(seg_starts.size - 1)
    if num_segs <= 0:
        raise ValueError(f"Computed num_segs={num_segs} from {segStartsPath}. Check file contents.")
    return num_segs


def _load_static_from_result_xm(base_path: str):
    """
    Loads camera trajectory + optional static points from result_XM.npz.
    Returns:
      B_list: (T,3,3)
      h_list: (T,3)
      static_landmarks: (Ns,3)
    """
    p = os.path.join(base_path, "result_XM.npz")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Missing static reconstruction file: {p}")

    data = np.load(p)
    R_real = data["R"]       # 3 x (3T)
    t_est  = data["t"]       # 3 x T
    s_est  = data["s"].astype(float).squeeze()
    p_est  = data["p"].T if "p" in data.files else None  # (Ns,3) if exists

    T = len(s_est)
    B_list = np.stack([R_real[:, 3*i:3*i+3] for i in range(T)], axis=0).astype(float)  # (T,3,3)
    h_list = t_est.T.astype(float)  # (T,3)

    static_landmarks = np.zeros((0, 3), dtype=float)
    if p_est is not None and isinstance(p_est, np.ndarray) and p_est.ndim == 2 and p_est.shape[1] == 3:
        static_landmarks = p_est.astype(float)

    return B_list, h_list, static_landmarks


def _rigid_fit_svd(P: np.ndarray, Q: np.ndarray):
    """
    Find R,t minimizing || (R P_i + t) - Q_i ||^2 over corresponding points.
    P,Q: (N,3)
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if P.shape != Q.shape or P.shape[1] != 3:
        raise ValueError(f"P,Q must both be (N,3). Got {P.shape} and {Q.shape}.")

    pbar = P.mean(axis=0)
    qbar = Q.mean(axis=0)
    X = P - pbar
    Y = Q - qbar
    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = qbar - (R @ pbar)
    return R, t


def _apply_rigid_to_traj(traj: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Apply x -> R x + t to last-dimension-3 trajectories (....,3)
    Preserves shape, keeps NaNs.
    """
    traj = np.asarray(traj, dtype=float)
    if traj.shape[-1] != 3:
        raise ValueError(f"traj last dim must be 3, got {traj.shape}")

    X = traj.reshape(-1, 3)
    mask = np.all(np.isfinite(X), axis=1)
    Xtf = X.copy()
    Xtf[mask] = (X[mask] @ R.T) + t.reshape(1, 3)
    return Xtf.reshape(traj.shape)


def _load_seg_npz(base_path: str, seg: int):
    npz_path = os.path.join(base_path, f"reconstruction_results_seg_{seg:02d}.npz")
    if not os.path.isfile(npz_path):
        return None, npz_path
    d = np.load(npz_path, allow_pickle=True)
    return {k: d[k] for k in d.files}, npz_path


def _infer_seg_start_end_from_segment_files(base_path: str, seg: int):
    seg_starts = np.loadtxt(os.path.join(base_path, "segment_starts.txt"), dtype=int, ndmin=1)
    seg_ends   = np.loadtxt(os.path.join(base_path, "segment_ends.txt"), dtype=int, ndmin=1)
    if seg < 0 or seg >= seg_starts.size:
        raise IndexError(f"seg={seg} out of range for segment_starts size={seg_starts.size}")
    return int(seg_starts[seg]), int(seg_ends[seg])


def _frames_chronological_from_saved(seg_data: dict) -> np.ndarray:
    if "frames_vis" in seg_data:
        frames = np.asarray(seg_data["frames_vis"], dtype=np.int64).reshape(-1)
        return np.sort(frames)

    if "frames_to_keep" in seg_data:
        f = np.asarray(seg_data["frames_to_keep"], dtype=np.int64).reshape(-1)
        return np.sort(f)

    raise KeyError("Segment NPZ missing both 'frames_vis' and 'frames_to_keep'.")


def _traj_chronological(seg_data: dict, key: str) -> np.ndarray:
    if key not in seg_data:
        raise KeyError(f"Segment NPZ missing '{key}'")

    X = np.asarray(seg_data[key], dtype=float)
    if X.ndim != 3 or X.shape[2] != 3:
        raise ValueError(f"{key} expected (N,T,3), got {X.shape}")

    if "frames_to_keep" not in seg_data:
        return X

    solver_frames = np.asarray(seg_data["frames_to_keep"], dtype=np.int64).reshape(-1)
    chrono_frames = np.sort(solver_frames)
    frame_to_solver_idx = {int(f): i for i, f in enumerate(solver_frames.tolist())}
    perm = [frame_to_solver_idx[int(f)] for f in chrono_frames.tolist()]
    perm = np.asarray(perm, dtype=int)
    return X[:, perm, :]


# ----------------------------- merging -----------------------------

def merge_two_structs_accum_to_next(A: dict, B: dict) -> dict:
    shared_frame = int(A["seg_end"])
    framesA = A["frames"].astype(int)
    framesB = B["frames"].astype(int)

    idxA = np.where(framesA == shared_frame)[0]
    idxB = np.where(framesB == shared_frame)[0]
    if idxA.size == 0 or idxB.size == 0:
        raise ValueError(f"Shared frame {shared_frame} not found in A/B frames.")
    idxA = int(idxA[0])
    idxB = int(idxB[0])

    A_ids = A["edg_node_ids"].astype(int)
    B_ids = B["edg_node_ids"].astype(int)
    mapA = {int(nid): i for i, nid in enumerate(A_ids.tolist())}
    mapB = {int(nid): i for i, nid in enumerate(B_ids.tolist())}

    shared_nodes = np.intersect1d(A_ids, B_ids)
    PA, PB = [], []
    for nid in shared_nodes.tolist():
        ia = mapA[int(nid)]
        ib = mapB[int(nid)]
        a = A["edg_deformed"][ia, idxA, :]
        b = B["edg_deformed"][ib, idxB, :]
        if np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
            PA.append(a)
            PB.append(b)
    PA = np.asarray(PA, dtype=float)
    PB = np.asarray(PB, dtype=float)

    print(f"shared_frame={shared_frame} | shared_nodes={shared_nodes.size} |std | usable={PA.shape[0]}")
    if PA.shape[0] < 3:
        print("[MERGE] Not enough finite shared EDG nodes. Using identity transform.")
        Rcorr = np.eye(3)
        tcorr = np.zeros(3)
    else:
        Rcorr, tcorr = _rigid_fit_svd(PA, PB)

    if not np.all(np.isfinite(Rcorr)) or not np.all(np.isfinite(tcorr)):
        print("[MERGE] NaN transform detected. Using identity.")
        Rcorr = np.eye(3)
        tcorr = np.zeros(3)

    A_dyn_tf = _apply_rigid_to_traj(A["dyn_mapped"], Rcorr, tcorr)
    A_edg_tf = _apply_rigid_to_traj(A["edg_deformed"], Rcorr, tcorr)

    framesA_pre  = framesA[framesA < shared_frame]
    framesB_post = framesB[framesB > shared_frame]
    merged_frames = np.concatenate([framesA_pre, np.array([shared_frame], dtype=int), framesB_post], axis=0)
    Tm = merged_frames.size

    # EDG merge
    all_nodes = np.union1d(A_ids, B_ids)
    mapM_node = {int(nid): i for i, nid in enumerate(all_nodes.tolist())}
    M = all_nodes.size
    merged_edg = np.full((M, Tm, 3), np.nan, dtype=float)

    A_edges = A.get("edg_edges", np.zeros((0, 2), dtype=int))
    B_edges = B.get("edg_edges", np.zeros((0, 2), dtype=int))
    edgesAB = np.vstack([A_edges, B_edges]).astype(int) if (A_edges.size or B_edges.size) else np.zeros((0, 2), dtype=int)
    if edgesAB.size:
        node_set = set(all_nodes.tolist())
        keep = [(a in node_set and b in node_set) for a, b in edgesAB.tolist()]
        edgesAB = edgesAB[np.array(keep, dtype=bool)]
        if edgesAB.size:
            edgesAB = np.sort(edgesAB, axis=1)
            _, ia = np.unique(edgesAB, axis=0, return_index=True)
            edgesAB = edgesAB[np.sort(ia)]
    merged_edges = edgesAB

    def _tM_for_frame(f: int) -> int:
        return int(np.where(merged_frames == f)[0][0])

    # fill A pre
    for f in framesA_pre.tolist():
        tA = int(np.where(framesA == f)[0][0])
        tM = _tM_for_frame(f)
        for i, nid in enumerate(A_ids.tolist()):
            merged_edg[mapM_node[int(nid)], tM, :] = A_edg_tf[i, tA, :]

    # boundary
    tM_shared = _tM_for_frame(shared_frame)
    for i, nid in enumerate(B_ids.tolist()):
        merged_edg[mapM_node[int(nid)], tM_shared, :] = B["edg_deformed"][i, idxB, :]

    for nid in shared_nodes.tolist():
        ia = mapA[int(nid)]
        ib = mapB[int(nid)]
        a_pos = A_edg_tf[ia, idxA, :]
        b_pos = B["edg_deformed"][ib, idxB, :]
        if np.all(np.isfinite(a_pos)) and np.all(np.isfinite(b_pos)):
            merged_edg[mapM_node[int(nid)], tM_shared, :] = 0.5 * (a_pos + b_pos)

    # fill B post
    for f in framesB_post.tolist():
        tB = int(np.where(framesB == f)[0][0])
        tM = _tM_for_frame(f)
        for i, nid in enumerate(B_ids.tolist()):
            merged_edg[mapM_node[int(nid)], tM, :] = B["edg_deformed"][i, tB, :]

    # DYN merge
    A_kps = A["dyn_kps"].astype(int)
    B_kps = B["dyn_kps"].astype(int)
    all_kps = np.union1d(A_kps, B_kps)
    mapA_kp = {int(k): i for i, k in enumerate(A_kps.tolist())}
    mapB_kp = {int(k): i for i, k in enumerate(B_kps.tolist())}
    mapM_kp = {int(k): i for i, k in enumerate(all_kps.tolist())}
    Nd = all_kps.size
    merged_dyn = np.full((Nd, Tm, 3), np.nan, dtype=float)

    for f in framesA_pre.tolist():
        tA = int(np.where(framesA == f)[0][0])
        tM = _tM_for_frame(f)
        for i, kid in enumerate(A_kps.tolist()):
            merged_dyn[mapM_kp[int(kid)], tM, :] = A_dyn_tf[i, tA, :]

    for i, kid in enumerate(B_kps.tolist()):
        merged_dyn[mapM_kp[int(kid)], tM_shared, :] = B["dyn_mapped"][i, idxB, :]

    shared_kps = np.intersect1d(A_kps, B_kps)
    for kid in shared_kps.tolist():
        ia = mapA_kp[int(kid)]
        ib = mapB_kp[int(kid)]
        a_pos = A_dyn_tf[ia, idxA, :]
        b_pos = B["dyn_mapped"][ib, idxB, :]
        if np.all(np.isfinite(a_pos)) and np.all(np.isfinite(b_pos)):
            merged_dyn[mapM_kp[int(kid)], tM_shared, :] = 0.5 * (a_pos + b_pos)

    for f in framesB_post.tolist():
        tB = int(np.where(framesB == f)[0][0])
        tM = _tM_for_frame(f)
        for i, kid in enumerate(B_kps.tolist()):
            merged_dyn[mapM_kp[int(kid)], tM, :] = B["dyn_mapped"][i, tB, :]

    return {
        "seg": int(B["seg"]),
        "seg_start": int(A["seg_start"]),
        "seg_end": int(B["seg_end"]),
        "frames": merged_frames.astype(int),
        "edg_node_ids": all_nodes.astype(int),
        "edg_deformed": merged_edg,
        "edg_edges": merged_edges.astype(int),
        "dyn_kps": all_kps.astype(int),
        "dyn_mapped": merged_dyn,
        "Rcorr": Rcorr,
        "tcorr": tcorr,
    }


def merge_all_segments(base_path: str, limit_segs: int | None = None):
    num_segs = _num_segments_from_segment_starts(base_path)
    if limit_segs is not None:
        num_segs = min(num_segs, int(limit_segs))

    ok_segs = []
    seg_structs = {}

    for seg in range(num_segs):
        seg_data, npz_path = _load_seg_npz(base_path, seg)
        if seg_data is None:
            print(f"[merge] seg {seg:02d} missing: {npz_path}")
            continue

        seg_start, seg_end = _infer_seg_start_end_from_segment_files(base_path, seg)
        frames = _frames_chronological_from_saved(seg_data)

        edg_deformed = _traj_chronological(seg_data, "edg_deformed")
        dyn_mapped   = _traj_chronological(seg_data, "dyn_mapped")

        if edg_deformed.shape[1] != frames.size:
            print(f"[merge] WARNING seg={seg:02d}: edg T={edg_deformed.shape[1]} vs frames={frames.size}. Using min.")
            Tm = min(edg_deformed.shape[1], frames.size, dyn_mapped.shape[1])
            frames = frames[:Tm]
            edg_deformed = edg_deformed[:, :Tm, :]
            dyn_mapped   = dyn_mapped[:, :Tm, :]

        edg_node_ids = np.asarray(seg_data.get("edg_ids", seg_data.get("edg_node_ids", None)), dtype=int)
        if edg_node_ids is None or edg_node_ids.size == 0:
            raise KeyError(f"seg={seg:02d} missing edg ids ('edg_ids' or 'edg_node_ids')")

        edges = np.asarray(seg_data.get("edges", seg_data.get("edg_edges", np.zeros((0, 2), dtype=int))), dtype=int)

        dyn_kps = np.asarray(seg_data.get("unique_kps_final", seg_data.get("unique_kps", None)), dtype=int)
        if dyn_kps is None or dyn_kps.size == 0:
            raise KeyError(f"seg={seg:02d} missing dynamic kp ids ('unique_kps_final' or 'unique_kps')")

        seg_structs[seg] = {
            "seg": seg,
            "seg_start": seg_start,
            "seg_end": seg_end,
            "frames": frames.astype(int),
            "edg_node_ids": edg_node_ids.astype(int),
            "edg_deformed": edg_deformed.astype(float),
            "edg_edges": edges.astype(int),
            "dyn_kps": dyn_kps.astype(int),
            "dyn_mapped": dyn_mapped.astype(float),
        }
        ok_segs.append(seg)

        print(f"[merge] seg {seg:02d} OK | frames [{frames[0]}..{frames[-1]}] ({frames.size}) "
              f"| edg M={edg_deformed.shape[0]} dyn Nd={dyn_mapped.shape[0]}")

    if not ok_segs:
        raise RuntimeError("No successful segments found (no reconstruction_results_seg_XX.npz).")

    ok_segs = sorted(ok_segs)
    print("[merge] Successful segments:", ok_segs)

    merged_all = seg_structs[ok_segs[0]]
    print(f"\n=== Start merge at seg {merged_all['seg']} | frames "
          f"[{merged_all['frames'][0]}..{merged_all['frames'][-1]}] ({merged_all['frames'].size}) ===")

    for k in range(1, len(ok_segs)):
        prev_seg = ok_segs[k - 1]
        cur_seg  = ok_segs[k]
        if cur_seg != prev_seg + 1:
            print(f"[merge] WARNING skipping merge between seg {prev_seg} and {cur_seg} (not consecutive).")
            continue
        print(f"\n--- Merging accumulated with segment {cur_seg} ---")
        merged_all = merge_two_structs_accum_to_next(merged_all, seg_structs[cur_seg])
        print(f"[merge] Merged now frames [{merged_all['frames'][0]}..{merged_all['frames'][-1]}] "
              f"({merged_all['frames'].size})")

    return merged_all, ok_segs


# ----------------------------- rendering -----------------------------

def _edges_to_indices(node_ids: np.ndarray, edges: np.ndarray):
    node_ids = node_ids.astype(int)
    edges = edges.astype(int)
    if edges.size == 0:
        return np.zeros((0,), dtype=int), np.zeros((0,), dtype=int)

    id2i = {int(nid): i for i, nid in enumerate(node_ids.tolist())}
    Ei, Ej = [], []
    for a, b in edges.tolist():
        if int(a) in id2i and int(b) in id2i:
            Ei.append(id2i[int(a)])
            Ej.append(id2i[int(b)])
    return np.asarray(Ei, dtype=int), np.asarray(Ej, dtype=int)


def _set_equal_axes(ax, pts: np.ndarray, pad_frac: float = 0.05):
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    rng3 = mx - mn
    mn = mn - pad_frac * rng3
    mx = mx + pad_frac * rng3

    cx, cy, cz = (mn + mx) / 2
    span = max(mx - mn)
    if span <= 0:
        span = 1.0
    half = span / 2

    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_zlim(cz - half, cz + half)


def render_merged_video(
    merged_all: dict,
    B_list: np.ndarray,
    h_list: np.ndarray,
    static_landmarks: np.ndarray,
    out_mp4: str,
    fps: int = 5,
    point_size_edg: float = 18.0,
    point_size_dyn: float = 10.0,
    cam_path_lw: float = 1.5,
    cam_dot_size: float = 120.0,   # <-- big dot size
    elev: float = -54.1034,
    azim: float = -5.4855,
    show_camera_up_to_current_frame: bool = False,
):
    frames_full = merged_all["frames"].astype(int)
    Tf = frames_full.size

    edg = merged_all["edg_deformed"]  # (M,T,3)
    dyn = merged_all["dyn_mapped"]    # (Nd,T,3)
    M = edg.shape[0]
    Nd = dyn.shape[0]

    cam_centers = np.asarray(h_list, dtype=float)  # (Tcam,3)
    Tcam = cam_centers.shape[0]

    # bounds include EDG + DYN + CAMERA (+ optional static)
    E = edg.reshape(-1, 3)
    P = dyn.reshape(-1, 3)
    all_pts = [E, P, cam_centers.reshape(-1, 3)]
    if static_landmarks is not None and static_landmarks.size:
        all_pts.append(static_landmarks.reshape(-1, 3))

    all_pts = np.vstack(all_pts)
    valid = np.all(np.isfinite(all_pts), axis=1)
    all_pts = all_pts[valid]
    if all_pts.size == 0:
        raise RuntimeError("All merged points are NaN/Inf; nothing to render.")

    Ei, Ej = _edges_to_indices(
        merged_all["edg_node_ids"],
        merged_all.get("edg_edges", np.zeros((0, 2), dtype=int))
    )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)
    _set_equal_axes(ax, all_pts, pad_frac=0.05)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    title = ax.set_title("")

    # Static landmarks
    scat_static = None
    if static_landmarks is not None and static_landmarks.size:
        s = static_landmarks
        ms = np.all(np.isfinite(s), axis=1)
        scat_static = ax.scatter(s[ms, 0], s[ms, 1], s[ms, 2], s=5)

    # Camera trajectory line
    cam_ok = np.all(np.isfinite(cam_centers), axis=1)
    if cam_ok.any():
        if show_camera_up_to_current_frame:
            cam_line, = ax.plot(
                cam_centers[cam_ok][:1, 0],
                cam_centers[cam_ok][:1, 1],
                cam_centers[cam_ok][:1, 2],
                linewidth=cam_path_lw,
            )
        else:
            cam_line, = ax.plot(
                cam_centers[cam_ok, 0],
                cam_centers[cam_ok, 1],
                cam_centers[cam_ok, 2],
                linewidth=cam_path_lw,
            )
    else:
        cam_line, = ax.plot([], [], [], linewidth=cam_path_lw)

    # Big red dot for current camera center
    cam_dot = ax.scatter([], [], [], s=cam_dot_size, c="red")

    # Dynamic scatter + EDG scatter
    edg0 = edg[:, 0, :]
    dyn0 = dyn[:, 0, :]
    okE0 = np.all(np.isfinite(edg0), axis=1)
    okD0 = np.all(np.isfinite(dyn0), axis=1)

    scat_edg = ax.scatter(edg0[okE0, 0], edg0[okE0, 1], edg0[okE0, 2], s=point_size_edg)
    scat_dyn = ax.scatter(dyn0[okD0, 0], dyn0[okD0, 1], dyn0[okD0, 2], s=point_size_dyn)

    # EDG edges
    edge_line, = ax.plot([], [], [], linewidth=0.5)

    def _update(ti: int):
        frame_g = int(frames_full[ti])

        # camera index: clamp to valid range in case merged frames exceed camera length
        cam_idx = frame_g
        if cam_idx < 0:
            cam_idx = 0
        if cam_idx >= Tcam:
            cam_idx = Tcam - 1

        edg_t = edg[:, ti, :]
        dyn_t = dyn[:, ti, :]

        okE = np.all(np.isfinite(edg_t), axis=1)
        okD = np.all(np.isfinite(dyn_t), axis=1)

        # update scatters
        scat_edg._offsets3d = (edg_t[okE, 0], edg_t[okE, 1], edg_t[okE, 2]) if okE.any() else ([], [], [])
        scat_dyn._offsets3d = (dyn_t[okD, 0], dyn_t[okD, 1], dyn_t[okD, 2]) if okD.any() else ([], [], [])

        # update edges
        if Ei.size:
            okEdge = okE[Ei] & okE[Ej]
            ii = Ei[okEdge]
            jj = Ej[okEdge]
            if ii.size:
                X = np.vstack([edg_t[ii, 0], edg_t[jj, 0], np.full(ii.size, np.nan)]).T.reshape(-1)
                Y = np.vstack([edg_t[ii, 1], edg_t[jj, 1], np.full(ii.size, np.nan)]).T.reshape(-1)
                Z = np.vstack([edg_t[ii, 2], edg_t[jj, 2], np.full(ii.size, np.nan)]).T.reshape(-1)
                edge_line.set_data(X, Y)
                edge_line.set_3d_properties(Z)
            else:
                edge_line.set_data([], [])
                edge_line.set_3d_properties([])
        else:
            edge_line.set_data([], [])
            edge_line.set_3d_properties([])

        # update camera line if growing
        if show_camera_up_to_current_frame and cam_ok.any():
            upto = cam_idx + 1
            cc = cam_centers[:upto]
            okc = np.all(np.isfinite(cc), axis=1)
            cc = cc[okc]
            if cc.shape[0] >= 1:
                cam_line.set_data(cc[:, 0], cc[:, 1])
                cam_line.set_3d_properties(cc[:, 2])
            else:
                cam_line.set_data([], [])
                cam_line.set_3d_properties([])

        # update big camera dot
        cpos = cam_centers[cam_idx]
        if np.all(np.isfinite(cpos)):
            cam_dot._offsets3d = ([cpos[0]], [cpos[1]], [cpos[2]])
        else:
            cam_dot._offsets3d = ([], [], [])

        title.set_text(
            f"FULL (merged) | Global Frame {frame_g} | dyn={int(okD.sum())}/{Nd} | edg={int(okE.sum())}/{M}"
        )

        return scat_edg, scat_dyn, edge_line, title, cam_line, cam_dot

    print(f"[render] Writing MP4: {out_mp4} | frames={Tf} | fps={fps}")
    anim = animation.FuncAnimation(fig, _update, frames=Tf, interval=int(1000 / fps), blit=False)

    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=fps, bitrate=1800)
    anim.save(out_mp4, writer=writer)
    plt.close(fig)
    print(f"[render] Done. Saved: {out_mp4}")


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videoname", type=str, default="train")
    ap.add_argument("--outputs_root", type=str, default="../deps/tapnet/outputs")
    ap.add_argument("--out_mp4", type=str, default=None)
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--limit_segs", type=int, default=None, help="debug: only load/merge first N segments")
    ap.add_argument("--camera_growing_path", action="store_true",
                    help="If set, show camera path only up to current frame (instead of full path).")
    args = ap.parse_args()

    base_path = os.path.join(args.outputs_root, args.videoname)
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"Base path not found: {base_path}")

    merged_all, _ = merge_all_segments(base_path, limit_segs=args.limit_segs)

    B_list, h_list, static_landmarks = _load_static_from_result_xm(base_path)
    static_landmarks = None  # keep your original behavior

    out_mp4 = args.out_mp4
    if out_mp4 is None:
        out_mp4 = os.path.join(base_path, f"full_reconstructed_motion_MERGED_{args.videoname}.mp4")

    render_merged_video(
        merged_all=merged_all,
        B_list=B_list,
        h_list=h_list,
        static_landmarks=static_landmarks,
        out_mp4=out_mp4,
        fps=args.fps,
        show_camera_up_to_current_frame=bool(args.camera_growing_path),
    )


if __name__ == "__main__":
    main()
