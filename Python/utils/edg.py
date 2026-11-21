from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def scatter_solved_poses(rt, T):
	"""Scatter static_solver's per-solved-frame poses onto the full T-frame timeline.

	The solver may drop frames (checklandmarks / XM^2), so result_XM.npz stores N<=T
	poses in *solved* order. `indices_all` (saved by static_solver) maps each original
	frame -> its solved index, or -1 if that frame was dropped. Dropped frames get an
	identity rotation and NaN translation so a downstream np.isfinite valid_mask drops them.

	Returns R_raw (T,3,3), t_raw (T,3).
	"""
	N = rt["R"].shape[1] // 3
	R_all = np.asarray(rt["R"], dtype=np.float64)
	t_all = np.asarray(rt["t"], dtype=np.float64)
	R_raw = np.tile(np.eye(3), (int(T), 1, 1)).astype(np.float64)
	t_raw = np.full((int(T), 3), np.nan, dtype=np.float64)
	if "indices_all" in rt.files:
		idx = np.asarray(rt["indices_all"]).astype(np.int64)
		for oi in range(min(int(T), idx.shape[0])):
			j = int(idx[oi])
			if 0 <= j < N:
				R_raw[oi] = R_all[:, 3 * j : 3 * (j + 1)]
				t_raw[oi] = t_all[:, j]
	else:
		if N != int(T):
			print(
				f"[EDG] WARNING: result_XM.npz has {N} solved frames vs {int(T)} track frames "
				f"and no 'indices_all' mapping; assuming the first {min(int(T), N)} align. "
				f"Re-run static_solver to embed the mapping."
			)
		for oi in range(min(int(T), N)):
			R_raw[oi] = R_all[:, 3 * oi : 3 * (oi + 1)]
			t_raw[oi] = t_all[:, oi]
	return R_raw, t_raw


def scatter_solved_frame_vector(rt, name, T, default=1.0):
	"""Scatter a per-solved-frame 1-D array (e.g. 's', 'da3_frame_scale') onto the full
	T-frame timeline via `indices_all`. Returns None if `name` is absent; otherwise a
	(T,) array with `default` at dropped/absent frames."""
	if name not in rt.files:
		return None
	raw = np.asarray(rt[name], dtype=np.float64).reshape(-1)
	out = np.full((int(T),), float(default), dtype=np.float64)
	if "indices_all" in rt.files:
		idx = np.asarray(rt["indices_all"]).astype(np.int64)
		for oi in range(min(int(T), idx.shape[0])):
			j = int(idx[oi])
			if 0 <= j < raw.shape[0]:
				out[oi] = raw[j]
	else:
		n = min(int(T), raw.shape[0])
		out[:n] = raw[:n]
	return out


def farthest_point_sample(points: np.ndarray, num_samples: int) -> np.ndarray:
	"""Farthest point sampling on NxD points, returning selected indices."""
	if points.shape[0] <= num_samples:
		return np.arange(points.shape[0])
	chosen = np.zeros(num_samples, dtype=int)
	chosen[0] = np.random.randint(points.shape[0])
	dist = np.full(points.shape[0], np.inf)
	for i in range(1, num_samples):
		last_pt = points[chosen[i - 1]]
		dist = np.minimum(dist, np.sum((points - last_pt) ** 2, axis=1))
		chosen[i] = np.argmax(dist)
	return chosen


def draw_dynamic_graph(
	frame: np.ndarray,
	pts: np.ndarray,
	valid_mask: np.ndarray,
	edges: list[tuple[int, int]] | set[tuple[int, int]],
	node_color: tuple[int, int, int] = (0, 0, 255),
	edge_color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
	"""Draw 2D graph nodes/edges on an image frame."""
	import cv2

	out = frame.copy()
	if pts.size == 0:
		return out
	for i, j in edges:
		if not (valid_mask[i] and valid_mask[j]):
			continue
		x1, y1 = pts[i]
		x2, y2 = pts[j]
		if np.isfinite(x1) and np.isfinite(y1) and np.isfinite(x2) and np.isfinite(y2):
			cv2.line(out, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), edge_color, 1)
	for (x, y) in pts[valid_mask]:
		if np.isfinite(x) and np.isfinite(y):
			cv2.circle(out, (int(round(x)), int(round(y))), 2, node_color, -1)
	return out


def visualize_2d_trajectories_on_images(
	images: Sequence[np.ndarray],
	tracks_xy: np.ndarray,
	valid_mask_global: np.ndarray,
	global_node_indices: np.ndarray | Sequence[int],
	frames: np.ndarray | Sequence[int],
	*,
	point_radius: int = 2,
	color: tuple[int, int, int] = (0, 0, 255),
) -> list[np.ndarray]:
	"""Overlay selected 2D points on image frames.

	Args:
		images: Sequence of BGR images indexed by global frame id.
		tracks_xy: Track array with shape [1, Q, T, 2].
		global_node_indices: Global query ids to visualize.
		frames: Frame ids to render.
		valid_mask_global: Boolean validity mask with shape [Q, T].

	Returns:
		List of rendered BGR frames in the same order as `frames`.
	"""
	if tracks_xy.ndim != 4 or tracks_xy.shape[0] != 1 or tracks_xy.shape[-1] != 2:
		raise ValueError(f"Expected tracks_xy shape [1,Q,T,2], got {tracks_xy.shape}")
	if valid_mask_global.ndim != 2:
		raise ValueError(f"Expected valid_mask_global shape [Q,T], got {valid_mask_global.shape}")
	if valid_mask_global.shape[0] != tracks_xy.shape[1] or valid_mask_global.shape[1] != tracks_xy.shape[2]:
		raise ValueError(
			f"Shape mismatch: valid_mask_global {valid_mask_global.shape} vs tracks_xy {tracks_xy.shape}"
		)

	idx = np.asarray(global_node_indices, dtype=int).reshape(-1)
	fr = np.asarray(frames, dtype=int).reshape(-1)
	if fr.size == 0:
		return []

	if np.any(fr < 0) or np.any(fr >= tracks_xy.shape[2]):
		raise ValueError("Some frame ids are out of tracks_xy time range.")
	if np.any(idx < 0) or np.any(idx >= tracks_xy.shape[1]):
		raise ValueError("Some global_node_indices are out of tracks_xy query range.")

	if len(images) <= int(fr.max()):
		raise ValueError("Not enough images for the requested frame ids.")

	import cv2

	out_frames: list[np.ndarray] = []
	for f in fr.tolist():
		img = np.asarray(images[f]).copy()

		for q in idx.tolist():
			if not bool(valid_mask_global[q, f]):
				continue

			x, y = tracks_xy[0, q, f, :]
			if np.isfinite(x) and np.isfinite(y):
				cv2.circle(img, (int(round(x)), int(round(y))), int(point_radius), color, -1)

		out_frames.append(img)

	return out_frames


def build_initial_edg_graph(
	tracks_xy_candidate: np.ndarray,
	valid_mask_global: np.ndarray,
	seed_frame: int,
	graph_samples: int = 30,
	graph_knn: int = 6,
) -> dict[str, Any]:
	"""Build initial EDG nodes/edges at a selected seed frame.

	Returns only candidate-local graph node indices and edge list.
	"""
	if tracks_xy_candidate.ndim != 4 or tracks_xy_candidate.shape[0] != 1:
		raise ValueError(
			f"Expected tracks_xy_candidate shape [1,N,T,2], got {tracks_xy_candidate.shape}"
		)
	if valid_mask_global.ndim != 2:
		raise ValueError(f"Expected valid_mask_global shape [N,T], got {valid_mask_global.shape}")
	if valid_mask_global.shape[0] != tracks_xy_candidate.shape[1]:
		raise ValueError(
			f"Shape mismatch: valid_mask_global {valid_mask_global.shape} vs tracks_xy_candidate {tracks_xy_candidate.shape}"
		)
	if seed_frame < 0 or seed_frame >= tracks_xy_candidate.shape[2]:
		raise ValueError(f"seed_frame {seed_frame} out of range for T={tracks_xy_candidate.shape[2]}")

	pts0 = tracks_xy_candidate[0, :, seed_frame, :]
	mask0 = valid_mask_global[:, seed_frame]
	init_idx = np.where(mask0)[0]
	if init_idx.size == 0:
		raise RuntimeError("No dynamic points in selected seed frame for initial EDG.")

	n0 = min(graph_samples, init_idx.size)
	seed_local = farthest_point_sample(pts0[init_idx], n0)
	graph_indices = init_idx[seed_local].tolist()  # candidate-local ids

	edges = set()
	if len(graph_indices) > 1:
		P0 = tracks_xy_candidate[0, graph_indices, seed_frame, :]
		d2 = np.sum((P0[:, None, :] - P0[None, :, :]) ** 2, axis=2)
		k0 = min(graph_knn, len(graph_indices) - 1)
		for i in range(len(graph_indices)):
			nn = np.argsort(d2[i])[1 : k0 + 1]
			for j in nn:
				a, b = (i, j) if i < j else (j, i)
				edges.add((a, b))

	return {
		"graph_indices": graph_indices,
		"graph_edges": sorted(edges),
	}


def smoothness_metric(traj_LD: np.ndarray, mask_L: np.ndarray, frame_ids: np.ndarray) -> float:
	"""Compute mean acceleration magnitude over contiguous visible segments.

	Lower values indicate smoother motion.
	"""
	idxv = np.where(mask_L)[0]
	if idxv.size < 3:
		return np.nan

	runs = np.split(idxv, np.where(np.diff(idxv) != 1)[0] + 1)

	seg_vals, seg_w = [], []
	for r in runs:
		if r.size < 3:
			continue

		pos = traj_LD[r]
		tt = frame_ids[r].astype(np.float64)
		dt = np.diff(tt)
		if np.any(dt <= 0):
			continue

		vel = np.diff(pos, axis=0) / dt[:, None]
		tmid = 0.5 * (tt[1:] + tt[:-1])
		dt2 = np.diff(tmid)
		if np.any(dt2 <= 0):
			continue

		acc = np.diff(vel, axis=0) / dt2[:, None]
		acc_norm = np.linalg.norm(acc, axis=1)

		seg_vals.append(np.mean(acc_norm))
		seg_w.append(r.size)

	if not seg_vals:
		return np.nan

	return float(np.average(seg_vals, weights=seg_w))


def build_world_trajectories_from_tracks(
	glob_all: np.ndarray,
	valid_mask_global: np.ndarray,
	frames: np.ndarray,
	delete_smoothness_percentile: float | None = None,
) -> dict[str, Any]:
	"""Filter trajectories in a frame window from precomputed world trajectories.

	Expected input shapes:
	- `glob_all`: [Q, T, D] world-frame trajectories for the full video
	- `valid_mask_global`: [Q, T] global validity mask
	- `frames`: [L] frame ids in the current segment window
	"""
	if glob_all.ndim != 3:
		raise ValueError(f"Expected glob_all shape [Q,T,D], got {glob_all.shape}")
	if valid_mask_global.ndim != 2:
		raise ValueError(f"Expected valid_mask_global shape [Q,T], got {valid_mask_global.shape}")
	if valid_mask_global.shape[:2] != glob_all.shape[:2]:
		raise ValueError(
			f"Shape mismatch: valid_mask_global {valid_mask_global.shape} vs glob_all {glob_all.shape}"
		)

	frames = np.asarray(frames, dtype=int).reshape(-1)
	if frames.size == 0:
		raise ValueError("Empty frame window.")

	pts_win = np.take(glob_all, frames, axis=1)  # (Q,L,D)
	valid_win = np.take(valid_mask_global.astype(bool), frames, axis=1)  # (Q,L)

	# Segment-local selection: only points that appear in the first frame of this window.
	# This enforces a consistent anchor set across the segment.
	idx_segment = np.where(valid_win[:, 0])[0]
	M_segment = len(idx_segment)

	glob_win = pts_win[idx_segment, :, :].astype(np.float64)  # (M_segment,L,D)
	mask_sel = valid_win[idx_segment]  # (M_segment,L)

	global_sel = np.full_like(glob_win, np.nan)
	global_sel[mask_sel] = glob_win[mask_sel]

	smooth_raw = np.array(
		[smoothness_metric(global_sel[i], mask_sel[i], frames) for i in range(M_segment)]
	)

	if delete_smoothness_percentile is None:
		keep_order = np.arange(M_segment, dtype=int)
	else:
		keep_num = int(M_segment * (1.0 - float(delete_smoothness_percentile)))
		keep_num = max(0, min(M_segment, keep_num))
		order = np.argsort(np.nan_to_num(smooth_raw, nan=np.inf))
		keep_order = order[:keep_num]

	selected_edg_idx = idx_segment[keep_order]

	return {
		"selected_edg_idx": selected_edg_idx,
		"smooth_raw": smooth_raw,
	}


def draw_world_trajectories_from_glob_all(
	glob_all: np.ndarray,
	smooth_scaled: np.ndarray | None = None,
	max_show: int = 1000,
) -> None:
	"""Draw 3D trajectories over time in world frame from `glob_all` using Open3D."""
	import matplotlib.pyplot as plt
	import open3d as o3d
	import open3d.web_visualizer as webvis

	traj = np.asarray(glob_all)
	if traj.ndim != 3:
		raise ValueError(f"Expected glob_all shape (M,L,D), got {traj.shape}")

	M_plot, L_plot, D_plot = traj.shape
	if D_plot < 2:
		raise ValueError(f"Need at least 2 dims in glob_all, got D={D_plot}")

	if D_plot == 2:
		traj3 = np.concatenate([traj, np.zeros((M_plot, L_plot, 1), dtype=traj.dtype)], axis=-1)
	else:
		traj3 = traj[..., :3]

	if smooth_scaled is not None and np.asarray(smooth_scaled).shape[0] == M_plot:
		order = np.argsort(np.nan_to_num(np.asarray(smooth_scaled), nan=-np.inf))[::-1]
	else:
		order = np.arange(M_plot)
	sel = order[: min(max_show, M_plot)]

	line_points = []
	line_indices = []
	line_colors = []
	start_pts = []
	end_pts = []

	for i in sel:
		tr = traj3[i]
		ok = np.isfinite(tr).all(axis=1)
		if np.sum(ok) < 2:
			continue

		vv = np.where(ok)[0]
		runs = np.split(vv, np.where(np.diff(vv) != 1)[0] + 1)

		for r in runs:
			if r.size < 2:
				continue
			seg = tr[r]
			tt = np.linspace(0.1, 1.0, len(r))
			for k in range(len(r) - 1):
				p0 = seg[k]
				p1 = seg[k + 1]
				i0 = len(line_points)
				i1 = i0 + 1
				line_points.extend([p0, p1])
				line_indices.append([i0, i1])
				c = np.array(plt.cm.viridis(tt[k])[:3], dtype=np.float64)
				line_colors.append(c)

		ok_idx = np.where(ok)[0]
		if ok_idx.size > 0:
			start_pts.append(tr[ok_idx[0]])
			end_pts.append(tr[ok_idx[-1]])

	geoms = []
	if len(line_points) > 0:
		lines = o3d.geometry.LineSet()
		lines.points = o3d.utility.Vector3dVector(np.asarray(line_points, dtype=np.float64))
		lines.lines = o3d.utility.Vector2iVector(np.asarray(line_indices, dtype=np.int32))
		lines.colors = o3d.utility.Vector3dVector(np.asarray(line_colors, dtype=np.float64))
		geoms.append(lines)

	if len(start_pts) > 0:
		pcd_start = o3d.geometry.PointCloud()
		pcd_start.points = o3d.utility.Vector3dVector(np.asarray(start_pts, dtype=np.float64))
		pcd_start.paint_uniform_color([0.0, 1.0, 0.0])
		geoms.append(pcd_start)

	if len(end_pts) > 0:
		pcd_end = o3d.geometry.PointCloud()
		pcd_end.points = o3d.utility.Vector3dVector(np.asarray(end_pts, dtype=np.float64))
		pcd_end.paint_uniform_color([1.0, 0.0, 0.0])
		geoms.append(pcd_end)

	if not geoms:
		raise RuntimeError("No valid trajectories to visualize.")

	print(f"Open3D world-frame trajectories from glob_all (showing {len(sel)}/{M_plot})")
	webvis.draw(geoms)


def _assemble_normal_eqs_vec(*, edges, edg_idx_map, g, edg_obs, edg_obs_mask, N_dyn_obs, dyn_kp_idx, dyn_frame, dyn_global,
							 frame_to_trev, kp_neighbors, kp_weights, kp_rest, R_start, p_start, T_rev, nz, w_obs, w_arap, w_smooth, w_edg_obs):
	"""Batched version of the normal-equation loops of solve_qcqp_and_recover_until_part5 (speed only): the same
	blocks (2 w C^T C, -2 w C^T b, w b^T b) of the observation, ARAP, EDG-observation and smoothness terms, built per batch
	instead of one 3x3 / 9x9 block at a time. Returns Q_I_list, Q_J_list, Q_V_list, q_accum, c_accum."""
	I_l, J_l, V_l = [], [], []
	q = np.zeros(nz, dtype=float)
	ar3, ar9 = np.arange(3), np.arange(9)

	def add(idx, Qb, qb):                                          # idx (N,k), Qb (N,k,k), qb (N,k)
		k = idx.shape[1]
		I_l.append(np.repeat(idx, k, axis=1).ravel()); J_l.append(np.tile(idx, (1, k)).ravel()); V_l.append(np.ascontiguousarray(Qb).reshape(-1))
		np.add.at(q, idx.ravel(), qb.ravel())

	c = 0.0
	# ---- observations: x_obs ~ sum_n w_n (R_{j_n,t} (x_rest - g_{j_n}) + g_{j_n} + p_{j_n,t})
	# observations used (t_rev > 1, keypoint known) grouped by neighbour count K, in observation order; the per-keypoint
	# rows (node indices, weights, rest point) are built once per keypoint and gathered per observation (speed only)
	ukp, inv = np.unique(np.asarray(dyn_kp_idx[:N_dyn_obs]).astype(np.int64), return_inverse=True)
	ufr, finv = np.unique(np.asarray(dyn_frame[:N_dyn_obs]).astype(np.int64), return_inverse=True)
	tr_all = np.array([int(frame_to_trev[int(f)]) for f in ufr.tolist()], dtype=np.int64)[finv.ravel()]
	inv = inv.ravel()
	K_kp = np.array([len(kp_neighbors[kp]) if kp in kp_neighbors and kp in kp_weights and kp in kp_rest else -1
	                 for kp in ukp.tolist()], dtype=np.int64)
	sel = np.nonzero((tr_all != 1) & (K_kp[inv] >= 0))[0]
	K_sel = K_kp[inv[sel]]
	uK, first = np.unique(K_sel, return_index=True)
	by_k = {int(K): sel[K_sel == K] for K in uK[np.argsort(first)].tolist()}
	dyn_global = np.asarray(dyn_global, dtype=float)
	M_o = g.shape[0]; keys, Ds, us, vs, ss = [], [], [], [], []       # observation blocks summed per (node, node, frame)
	for K, kk_all in by_k.items():
		rows = np.unique(inv[kk_all]); kpsK = ukp[rows].tolist()
		pos = np.full(ukp.size, -1, dtype=np.int64); pos[rows] = np.arange(rows.size)
		js_K = np.array([[edg_idx_map[int(n)] for n in kp_neighbors[x]] for x in kpsK], dtype=np.int64).reshape(-1, K)
		w_K = np.array([np.asarray(kp_weights[x], dtype=float) for x in kpsK]).reshape(-1, K)
		xr_K = np.array([np.asarray(kp_rest[x], dtype=float).reshape(3) for x in kpsK])
		step = max(1, int(2e7 // (12 * K) ** 2))
		for c0 in range(0, len(kk_all), step):
			kk = kk_all[c0:c0 + step]; rk = pos[inv[kk]]
			tr = tr_all[kk]
			js, w, xr = js_K[rk], w_K[rk], xr_K[rk]
			d = xr[:, None, :] - g[js]                             # (N,K,3)
			C = np.zeros((len(kk), 3, 12 * K))
			for n in range(K):
				for a_ in range(3):
					for r in range(3):
						C[:, r, 12 * n + 3 * a_ + r] = -w[:, n] * d[:, n, a_]   # -w kron(d^T, I3)
				for r in range(3):
					C[:, r, 12 * n + 9 + r] = -w[:, n]                   # -w I3
			b = -(dyn_global[kk, :3] - np.einsum("nk,nkc->nc", w, g[js]))
			idx = np.concatenate([np.concatenate([R_start[js[:, n], tr][:, None] + ar9, p_start[js[:, n], tr][:, None] + ar3], 1) for n in range(K)], 1)
			np.add.at(q, idx.ravel(), (-2.0 * w_obs * np.einsum("nri,nr->ni", C, b)).ravel())
			c += float(w_obs * (b ** 2).sum())
			for n in range(K):                                     # C_n^T C_m = w_n w_m [[kron(d_n d_m^T, I3), kron(d_n, I3)], [kron(d_m^T, I3), I3]]
				for m in range(K):
					ww = w[:, n] * w[:, m]
					keys.append((js[:, n] * M_o + js[:, m]) * (T_rev + 1) + tr)
					Ds.append((ww[:, None, None] * d[:, n, :, None] * d[:, m, None, :]).reshape(-1, 9))
					us.append(ww[:, None] * d[:, n]); vs.append(ww[:, None] * d[:, m]); ss.append(ww)
	if keys:                                                       # one 12x12 block per
		uk, inv = np.unique(np.concatenate(keys), return_inverse=True)   # (node, node, frame) instead of one per observation
		U = uk.size; inv = inv.ravel()
		acc = lambda X: np.stack([np.bincount(inv, weights=X[:, i], minlength=U) for i in range(X.shape[1])], 1)
		D, u_, v_, s_ = acc(np.concatenate(Ds)), acc(np.concatenate(us)), acc(np.concatenate(vs)), np.bincount(inv, weights=np.concatenate(ss), minlength=U)
		Qb = np.zeros((U, 12, 12))
		for a_ in range(3):
			for r in range(3):
				for b_ in range(3):
					Qb[:, 3 * a_ + r, 3 * b_ + r] = D[:, 3 * a_ + b_]
				Qb[:, 3 * a_ + r, 9 + r] = u_[:, a_]; Qb[:, 9 + r, 3 * a_ + r] = v_[:, a_]
		for r in range(3):
			Qb[:, 9 + r, 9 + r] = s_
		ta = uk % (T_rev + 1); ab = uk // (T_rev + 1); na, nb = ab // M_o, ab % M_o
		ir = np.concatenate([R_start[na, ta][:, None] + ar9, p_start[na, ta][:, None] + ar3], 1)
		ic = np.concatenate([R_start[nb, ta][:, None] + ar9, p_start[nb, ta][:, None] + ar3], 1)
		I_l.append(np.repeat(ir, 12, axis=1).ravel()); J_l.append(np.tile(ic, (1, 12)).ravel()); V_l.append((2.0 * w_obs * Qb).reshape(-1))
	# ---- ARAP: R_{i,t} e_ij + p_{i,t} - p_{j,t} ~ e_ij (and the same with R_{j,t})
	if edges.size:
		ei = np.array([edg_idx_map[int(x)] for x in edges[:, 0]], dtype=np.int64)
		ej = np.array([edg_idx_map[int(x)] for x in edges[:, 1]], dtype=np.int64)
		e = g[ej] - g[ei]
		C = np.zeros((len(e), 3, 15))
		for a_ in range(3):
			for r in range(3):
				C[:, r, 3 * a_ + r] = e[:, a_]
		for r in range(3):
			C[:, r, 9 + r] = 1.0; C[:, r, 12 + r] = -1.0
		Qb = 2.0 * w_arap * np.einsum("nri,nrj->nij", C, C); qb = -2.0 * w_arap * np.einsum("nri,nr->ni", C, e); cb = float(w_arap * (e ** 2).sum())
		for t in range(2, T_rev + 1):
			pi = p_start[ei, t][:, None] + ar3; pj = p_start[ej, t][:, None] + ar3
			for Rk in (R_start[ei, t], R_start[ej, t]):
				add(np.concatenate([Rk[:, None] + ar9, pi, pj], 1), Qb, qb); c += cb
	# ---- EDG observations: p_{j,t} ~ edg_{j,t} - g_j
	if edg_obs is not None:
		edg_obs = np.asarray(edg_obs, dtype=float)
		M_ = g.shape[0]
		if edg_obs.shape != (M_, T_rev, 3):
			raise ValueError(f"Expected edg_obs shape {(M_, T_rev, 3)}, got {edg_obs.shape}")
		edg_obs_mask = np.isfinite(edg_obs).all(axis=-1) if edg_obs_mask is None else np.asarray(edg_obs_mask, dtype=bool)
		jj, ti = np.nonzero(edg_obs_mask[:, 1:T_rev])
		if jj.size:
			tgt = edg_obs[jj, ti + 1] - g[jj]
			add(p_start[jj, ti + 2][:, None] + ar3, np.broadcast_to(2.0 * w_edg_obs * np.eye(3), (jj.size, 3, 3)), -2.0 * w_edg_obs * tgt)
			c += float(w_edg_obs * (tgt ** 2).sum())
		print(f"  EDG observation terms: {jj.size}")
	# ---- smoothness: R_{j,2} ~ I, p_{j,2} ~ 0, R_{j,t} ~ R_{j,t+1}, p_{j,t} ~ p_{j,t+1}
	M_ = g.shape[0]; jall = np.arange(M_)
	I_vec = np.eye(3).reshape(-1, order="F")
	add(R_start[jall, 2][:, None] + ar9, np.broadcast_to(2.0 * w_smooth * np.eye(9), (M_, 9, 9)), np.broadcast_to(-2.0 * w_smooth * I_vec, (M_, 9)))
	c += float(w_smooth * 3.0 * M_)
	add(p_start[jall, 2][:, None] + ar3, np.broadcast_to(2.0 * w_smooth * np.eye(3), (M_, 3, 3)), np.zeros((M_, 3)))
	if T_rev > 2:
		jt, tt = np.meshgrid(jall, np.arange(2, T_rev), indexing="ij"); jt, tt = jt.ravel(), tt.ravel()
		for st, n in ((R_start, 9), (p_start, 3)):
			CC = np.hstack([np.eye(n), -np.eye(n)])
			add(np.concatenate([st[jt, tt][:, None] + np.arange(n), st[jt, tt + 1][:, None] + np.arange(n)], 1),
			    np.broadcast_to(2.0 * w_smooth * CC.T @ CC, (jt.size, 2 * n, 2 * n)), np.zeros((jt.size, 2 * n)))
	return I_l, J_l, V_l, q, c


_WORK = {}                                                         # dense work arrays kept across solves (speed only)


def _work(name, shape, order="C"):
	"""Uninitialised float64 array of `shape` on memory kept from earlier solves: no fresh pages per segment."""
	n = int(np.prod(shape))
	if name not in _WORK or _WORK[name].size < n:
		_WORK[name] = np.empty(n)
	return _WORK[name][:n].reshape(shape, order=order)


def solve_qcqp_and_recover_until_part5(
	*,
	edges: np.ndarray,
	edg_idx_map: dict[int, int],
	g: np.ndarray,
	edg_obs: np.ndarray | None = None,
	edg_obs_mask: np.ndarray | None = None,
	N_dyn_obs: int,
	dyn_kp_idx: np.ndarray,
	dyn_frame: np.ndarray,
	dyn_global: np.ndarray,
	frame_to_trev: dict[int, int],
	kp_neighbors: dict[int, np.ndarray],
	kp_weights: dict[int, np.ndarray],
	kp_rest: dict[int, np.ndarray],
	output_dir: str,
	solve_Ronly_fn: Any,
	load_matrix_from_bin_fn: Any,
	w_obs: float = 1.0,
	w_arap: float = 0.2,
	w_smooth: float = 0.02,
	w_edg_obs: float = 10.0,
	c_float: float = 200.0,
	solver_rank: int = 5,
	solver_tol: float = 1e-4,
	solver_mu: float = 0.0,
	solver_max_iter: float = 1000.0,
) -> dict[str, Any]:
	"""Run QCQP build/solve and recover EDG motion/state up to Part 5 boundary.

	This wraps the block from "# QCQP weights" through recovery of `R_edg`, `p_edg`,
	`dyn_mapped`, and `edg_deformed`.
	"""
	import os
	import scipy.sparse as sp
	from scipy.sparse import coo_matrix, eye as speye
	from scipy.sparse.linalg import spsolve, splu

	M = int(len(edg_idx_map))
	if len(frame_to_trev) == 0:
		raise ValueError("frame_to_trev is empty.")
	T_rev = int(max(frame_to_trev.values()))

	T_var = T_rev - 1
	ny = M * T_var * 9
	nw = M * T_var * 3
	nz = ny + nw

	R_start = -np.ones((M, T_rev + 1), dtype=np.int64)
	p_start = -np.ones((M, T_rev + 1), dtype=np.int64)

	ofs = 0
	for j in range(M):
		for t in range(2, T_rev + 1):
			R_start[j, t] = ofs
			ofs += 9
	assert ofs == ny

	ofs = ny
	for j in range(M):
		for t in range(2, T_rev + 1):
			p_start[j, t] = ofs
			ofs += 3
	assert ofs == nz

	def R_idx(j: int, t: int) -> np.ndarray | None:
		s = R_start[j, t]
		if s < 0:
			return None
		return np.arange(s, s + 9, dtype=np.int64)

	def p_idx(j: int, t: int) -> np.ndarray | None:
		s = p_start[j, t]
		if s < 0:
			return None
		return np.arange(s, s + 3, dtype=np.int64)

	Q_I_list, Q_J_list, Q_V_list = [], [], []
	q_accum = np.zeros(nz, dtype=float)
	c_accum = 0.0

	def append_normal_eq_block(idx: np.ndarray, C: np.ndarray, b: np.ndarray, weight: float):
		idx = np.asarray(idx, dtype=np.int64)
		C = np.asarray(C, dtype=float)
		b = np.asarray(b, dtype=float).reshape(-1, 1)

		Qblk = (2.0 * weight) * (C.T @ C)
		qblk = (-2.0 * weight) * (C.T @ b)
		cinc = float(weight * (b.T @ b))

		I, J = np.meshgrid(idx, idx, indexing="ij")
		return I.ravel(), J.ravel(), Qblk.ravel(), idx, qblk.ravel(), cinc

	_vec = os.environ.get("EDG_ASSEMBLY", "vec") != "loop"            # batched assembly (identical Q, q); "loop" = the original
	if _vec:
		print(f"Building all residual blocks batched ({N_dyn_obs} observations, {edges.shape[0]} edges, {M} nodes)...")
		Q_I_list, Q_J_list, Q_V_list, q_accum, c_accum = _assemble_normal_eqs_vec(
			edges=edges, edg_idx_map=edg_idx_map, g=g, edg_obs=edg_obs, edg_obs_mask=edg_obs_mask, N_dyn_obs=N_dyn_obs,
			dyn_kp_idx=dyn_kp_idx, dyn_frame=dyn_frame, dyn_global=dyn_global, frame_to_trev=frame_to_trev,
			kp_neighbors=kp_neighbors, kp_weights=kp_weights, kp_rest=kp_rest, R_start=R_start, p_start=p_start,
			T_rev=T_rev, nz=nz, w_obs=w_obs, w_arap=w_arap, w_smooth=w_smooth, w_edg_obs=w_edg_obs)
	else:
		print(f"Building observation residuals ({N_dyn_obs} observations)...")
		obs_skipped_t1 = 0
		I3 = np.eye(3)

		for k in range(N_dyn_obs):
			kp = int(dyn_kp_idx[k])
			frame_t = int(dyn_frame[k])
			t_rev = int(frame_to_trev[frame_t])
			if t_rev == 1:
				obs_skipped_t1 += 1
				continue
			x_obs = dyn_global[k, :].reshape(3, 1)
			if kp not in kp_neighbors or kp not in kp_weights or kp not in kp_rest:
				continue

			neighbors = np.asarray(kp_neighbors[kp], dtype=int)
			weights_kp = np.asarray(kp_weights[kp], dtype=float)
			x_rest = kp_rest[kp].reshape(3, 1)

			js = np.array([edg_idx_map[int(nid)] for nid in neighbors], dtype=int)
			g_js = g[js, :].T

			const_part = x_obs - (g_js * weights_kp.reshape(1, -1)).sum(axis=1, keepdims=True)

			idx_all = []
			C_parts = []
			for n in range(len(neighbors)):
				j = int(js[n])
				wj = float(weights_kp[n])
				gj = g[j, :].reshape(3, 1)

				d = (x_rest - gj).reshape(3,)
				Rind = R_idx(j, t_rev)
				pind = p_idx(j, t_rev)

				R_coeff = -wj * np.kron(d.reshape(1, 3), I3)
				p_coeff = -wj * I3

				idx_all.extend(Rind.tolist())
				idx_all.extend(pind.tolist())
				C_parts.append(R_coeff)
				C_parts.append(p_coeff)

			C_all = np.hstack(C_parts)
			b = -const_part

			II, JJ, VV, q_idx, q_val, c_inc = append_normal_eq_block(idx_all, C_all, b, w_obs)
			Q_I_list.append(II)
			Q_J_list.append(JJ)
			Q_V_list.append(VV)
			q_accum[q_idx] += q_val
			c_accum += c_inc

		print(f"  Done: {N_dyn_obs - obs_skipped_t1} processed, {obs_skipped_t1} skipped at t=1.")

		print(f"\nBuilding ARAP loss ({edges.shape[0]} edges x {T_var} times)...")
		edge_i = np.array([edg_idx_map[int(a)] for a in edges[:, 0]], dtype=int) if edges.size else np.array([], dtype=int)
		edge_j = np.array([edg_idx_map[int(b)] for b in edges[:, 1]], dtype=int) if edges.size else np.array([], dtype=int)

		for e in range(edges.shape[0]):
			i = int(edge_i[e])
			j = int(edge_j[e])
			g_i = g[i, :].reshape(3, 1)
			g_j = g[j, :].reshape(3, 1)
			e_ij = (g_j - g_i).reshape(3, 1)

			for t in range(2, T_rev + 1):
				Ri = R_idx(i, t)
				Rj = R_idx(j, t)
				pi = p_idx(i, t)
				pj = p_idx(j, t)

				C_common = np.hstack([np.kron(e_ij.reshape(1, 3), I3), I3, -I3])
				b_common = e_ij

				idx1 = np.concatenate([Ri, pi, pj])
				II, JJ, VV, q_idx, q_val, c_inc = append_normal_eq_block(idx1, C_common, b_common, w_arap)
				Q_I_list.append(II)
				Q_J_list.append(JJ)
				Q_V_list.append(VV)
				q_accum[q_idx] += q_val
				c_accum += c_inc

				idx2 = np.concatenate([Rj, pi, pj])
				II, JJ, VV, q_idx, q_val, c_inc = append_normal_eq_block(idx2, C_common, b_common, w_arap)
				Q_I_list.append(II)
				Q_J_list.append(JJ)
				Q_V_list.append(VV)
				q_accum[q_idx] += q_val
				c_accum += c_inc

		print("  Done.")

		# EDG observation consistency: g_i + p_{i,t} ~= edg_{i,t}
		# => p_{i,t} ~= edg_{i,t} - g_i
		if edg_obs is not None:
			edg_obs = np.asarray(edg_obs, dtype=float)
			if edg_obs.shape != (M, T_rev, 3):
				raise ValueError(f"Expected edg_obs shape {(M, T_rev, 3)}, got {edg_obs.shape}")
			if edg_obs_mask is None:
				edg_obs_mask = np.isfinite(edg_obs).all(axis=-1)
			else:
				edg_obs_mask = np.asarray(edg_obs_mask, dtype=bool)
				if edg_obs_mask.shape != (M, T_rev):
					raise ValueError(f"Expected edg_obs_mask shape {(M, T_rev)}, got {edg_obs_mask.shape}")

			print("\nBuilding EDG observation loss...")
			n_edg_terms = 0
			for j in range(M):
				g_j = g[j, :].reshape(3,)
				for t in range(2, T_rev + 1):
					t_idx = t - 1
					if not edg_obs_mask[j, t_idx]:
						continue

					target_p = (edg_obs[j, t_idx, :] - g_j).reshape(3, 1)
					pind = p_idx(j, t)
					C = np.eye(3)
					b = target_p

					II, JJ, VV, q_idx, q_val, c_inc = append_normal_eq_block(pind, C, b, w_edg_obs)
					Q_I_list.append(II)
					Q_J_list.append(JJ)
					Q_V_list.append(VV)
					q_accum[q_idx] += q_val
					c_accum += c_inc
					n_edg_terms += 1
			print(f"  Done: {n_edg_terms} EDG observation terms.")

		print(f"\nBuilding motion smoothness loss ({M} nodes x {T_rev-1} time pairs)...")
		I_vec = np.eye(3).reshape(-1, order="F").reshape(9, 1)

		for j in range(M):
			Rj2 = R_idx(j, 2)
			pj2 = p_idx(j, 2)

			C_R_12 = -np.eye(9)
			b_R_12 = -I_vec
			II, JJ, VV, q_idx, q_val, c_inc = append_normal_eq_block(Rj2, C_R_12, b_R_12, w_smooth)
			Q_I_list.append(II)
			Q_J_list.append(JJ)
			Q_V_list.append(VV)
			q_accum[q_idx] += q_val
			c_accum += c_inc

			C_p_12 = -np.eye(3)
			b_p_12 = np.zeros((3, 1))
			II, JJ, VV, q_idx, q_val, c_inc = append_normal_eq_block(pj2, C_p_12, b_p_12, w_smooth)
			Q_I_list.append(II)
			Q_J_list.append(JJ)
			Q_V_list.append(VV)
			q_accum[q_idx] += q_val
			c_accum += c_inc

			for t in range(2, T_rev):
				Rjt = R_idx(j, t)
				Rjt1 = R_idx(j, t + 1)
				pjt = p_idx(j, t)
				pjt1 = p_idx(j, t + 1)

				idxR = np.concatenate([Rjt, Rjt1])
				CR = np.hstack([np.eye(9), -np.eye(9)])
				bR = np.zeros((9, 1))
				II, JJ, VV, q_idx, q_val, c_inc = append_normal_eq_block(idxR, CR, bR, w_smooth)
				Q_I_list.append(II)
				Q_J_list.append(JJ)
				Q_V_list.append(VV)
				q_accum[q_idx] += q_val
				c_accum += c_inc

				idxp = np.concatenate([pjt, pjt1])
				Cp = np.hstack([np.eye(3), -np.eye(3)])
				bp = np.zeros((3, 1))
				II, JJ, VV, q_idx, q_val, c_inc = append_normal_eq_block(idxp, Cp, bp, w_smooth)
				Q_I_list.append(II)
				Q_J_list.append(JJ)
				Q_V_list.append(VV)
				q_accum[q_idx] += q_val
				c_accum += c_inc

		print("  Done.")

	print("\nAssembling sparse Q matrix...")
	Q_I = np.concatenate(Q_I_list) if Q_I_list else np.zeros((0,), dtype=np.int64)
	Q_J = np.concatenate(Q_J_list) if Q_J_list else np.zeros((0,), dtype=np.int64)
	Q_V = np.concatenate(Q_V_list) if Q_V_list else np.zeros((0,), dtype=float)
	Q = coo_matrix((Q_V, (Q_I, Q_J)), shape=(nz, nz)).tocsr()
	Q = 0.5 * (Q + Q.T)
	q = q_accum

	y_idx = slice(0, ny)
	w_idx = slice(ny, nz)
	Qyy = Q[y_idx, y_idx]
	Qyw = Q[y_idx, w_idx]
	Qwy = Q[w_idx, y_idx]
	Qww = Q[w_idx, w_idx]
	qy = q[:ny]
	qw = q[ny:]

	print("\nPerforming Schur complement to eliminate translations...")
	Qww_reg = Qww.tocsc()
	Qwy_csc = Qwy.tocsc()

	# Optional dense path for small systems
	max_dense_n = 6000
	max_dense_mb = 2048
	qww_n = Qww.shape[0]
	est_dense_mb = (qww_n * qww_n * 8) / (1024**2)
	use_dense = qww_n <= max_dense_n and est_dense_mb <= max_dense_mb

	def _solve_sparse_rhs(lu_factor, rhs_csc, block_size: int = 256):
		"""Solve lu_factor * X = rhs_csc for sparse RHS in column blocks."""
		cols = []
		for start in range(0, rhs_csc.shape[1], block_size):
			end = min(start + block_size, rhs_csc.shape[1])
			rhs_block = rhs_csc[:, start:end].toarray()
			sol_block = lu_factor.solve(rhs_block)
			cols.append(sp.csc_matrix(sol_block))
		if len(cols) == 1:
			return cols[0]
		return sp.hstack(cols, format="csc")

	if use_dense:
		print(f"  Using dense solve for Qww (n={qww_n}, ~{est_dense_mb:.1f} MB)")
		Qww_dense = Qww_reg.toarray(out=_work("Qww", Qww_reg.shape, "F"))      # Fortran (csc) order, as toarray()
		Qwy3 = Qwy_csc[:, 0::3]                                  # XM takes Qred[0::3, 0::3] only: solve for those columns
		Qwy_dense = Qwy3.toarray(out=_work("Qwy", Qwy3.shape, "F"))           # both factored / solved in place below
		try:                                                     # Qww is symmetric positive definite: Cholesky
			from scipy.linalg import cho_factor, cho_solve
			cf = cho_factor(Qww_dense, lower=True, overwrite_a=True, check_finite=False)
			X = cho_solve(cf, Qwy_dense, overwrite_b=True, check_finite=False)
			v = cho_solve(cf, qw, check_finite=False).astype(float)
		except np.linalg.LinAlgError:
			Qww_dense, Qwy_dense = Qww_reg.toarray(), Qwy_csc[:, 0::3].toarray()   # (may be overwritten above)
			X = np.linalg.solve(Qww_dense, Qwy_dense).astype(float)
			v = np.linalg.solve(Qww_dense, qw).astype(float)
	else:
		try:
			lu = splu(Qww_reg)
			X = _solve_sparse_rhs(lu, Qwy_csc[:, 0::3].tocsc()).astype(float)
			v = lu.solve(qw).astype(float)
		except Exception:
			X = spsolve(Qww_reg, Qwy_csc[:, 0::3].tocsc()).astype(float)
			v = spsolve(Qww_reg, qw).astype(float)
	if use_dense:                                                  # dense: the same values as the csr round trip (speed only)
		Qyy3 = Qyy[0::3, 0::3]
		Qred = Qyy3.toarray(out=_work("Qred", Qyy3.shape))
		Xc = _work("Xc", X.shape)                                  # X in row order (the sparse product would copy it
		Xc[...] = X                                                # to a fresh array)
		Qred -= Qyw[0::3, :] @ Xc
	else:
		Qred = (Qyy[0::3, 0::3] - Qyw[0::3, :] @ X).tocsr()
	qred = (qy - (Qyw @ v)).astype(float)
	q_XM = qred.reshape(-1, 3)

	n3 = q_XM.shape[0]                                             # Q_all = [[c I3, q_XM^T], [q_XM, Q_XM]], built dense
	Q_all = _work("Q_all", (n3 + 3, n3 + 3))
	Q_all[:3, :3] = c_float * np.eye(3)
	Q_all[:3, 3:] = q_XM.T
	Q_all[3:, :3] = q_XM
	Q_XM = Q_all[3:, 3:]                                           # = the full Schur complement's [0::3, 0::3]
	if use_dense:
		np.add(Qred, Qred.T, out=Q_XM)                             # = 0.5 * (Qred + Qred.T)
		Q_XM *= 0.5
	else:
		Q_XM[...] = (0.5 * (Qred + Qred.T)).toarray()
	Q_all += 0.0                                                   # -0.0 -> +0.0, as the sparse bmat -> toarray gave

	os.makedirs(output_dir, exist_ok=True)
	filename_C = os.path.join(output_dir, "Q.bin")
	# rewritten in place and then cut to size: the same bytes without freeing + reallocating the file per solve (speed only)
	with os.fdopen(os.open(filename_C, os.O_WRONLY | os.O_CREAT, 0o666), "wb") as file:
		file.write(np.array(Q_all.shape[0], dtype=np.int32).tobytes())
		file.write(np.array(Q_all.shape[1], dtype=np.int32).tobytes())
		Q_all.tofile(file)
		file.truncate()

	solve_Ronly_fn(output_dir, int(solver_rank), float(solver_tol), float(solver_mu), float(solver_max_iter))

	R, _ = load_matrix_from_bin_fn(os.path.join(output_dir, "R.bin"))
	R1 = R[:3, :]
	R_real = (R1 @ R[3:, :].T).T

	print(
		"The original objective value:",
		np.trace(R_real.T @ Q_XM @ R_real) + 2 * np.trace(R_real.T @ q_XM) + c_float * 3,
	)

	print("\n=== Recovering Translations ===")
	n_poses = M * (T_rev - 1)

	def _ensure_R_real_shape(R_real_in: np.ndarray, n_poses_local: int) -> np.ndarray:
		Rr = np.asarray(R_real_in, dtype=float)
		if Rr.shape == (3, 3 * n_poses_local):
			Rr = Rr.T
		if Rr.shape != (3 * n_poses_local, 3):
			raise ValueError(
				f"Expected R_real shape (3*n_poses,3)=({3*n_poses_local},3) (or its transpose), got {Rr.shape}"
			)
		return Rr

	R_real = _ensure_R_real_shape(R_real, n_poses)

	# R_start[j, t] = 9 * pose (pose = j * (T_rev - 1) + t - 2): y_est[R_start[j, t]:+9] = R_jt in column order = the pose's
	# 3x3 block of R_real in row order -> one reshape (speed only)
	y_est = R_real.reshape(-1).copy()
	assert y_est.size == ny
	print(f"Converted XM rotations to y_est with ny={ny} (poses={n_poses})")

	rhs = (Qwy @ y_est) + qw
	Qww_sym = 0.5 * (Qww + Qww.T)

	try:
		lu = splu(Qww_sym.tocsc())
		w_est = -lu.solve(rhs)
	except Exception:
		w_est = -spsolve(Qww_sym.tocsc(), rhs)

	w_est = np.asarray(w_est, dtype=float).reshape(-1)
	print(f"Recovered {w_est.size} translation variables (nw={nw})")

	p_edg = np.zeros((M, T_rev, 3), dtype=float)
	R_edg = np.zeros((M, T_rev, 3, 3), dtype=float)
	p_edg[:, 0, :] = 0.0
	R_edg[:, 0, :, :] = np.eye(3)[None, :, :]

	p_edg[:, 1:, :] = w_est.reshape(M, T_rev - 1, 3)              # p_start[j, t] - ny = 3 * pose (same layout)
	R_edg[:, 1:, :, :] = R_real.reshape(M, T_rev - 1, 3, 3).transpose(0, 1, 3, 2)

	print(f"Organized rotations/translations: R_edg {R_edg.shape}, p_edg {p_edg.shape}")

	print("\n=== Mapping Dynamic Keypoints via EDG ===")
	unique_kps_final = np.unique(dyn_kp_idx).astype(int)
	Nd_final = len(unique_kps_final)
	dyn_mapped = np.zeros((Nd_final, T_rev, 3), dtype=float)

	# batched over the keypoints with the same neighbour count (speed only): the same per-element operations in the
	# same order (x_def = 0; x_def += w_n (R_jt @ (x_rest - g_j) + g_j + p_jt) for n = 0..K-1; the stacked matmul runs
	# the same BLAS gemv per 3x3 block)
	missing_rest = 0
	by_K = {}
	for kp_i, kp in enumerate(unique_kps_final.tolist()):
		if kp not in kp_rest:
			missing_rest += 1
			continue
		by_K.setdefault(len(kp_neighbors[kp]), []).append(kp_i)
	for K, rows in by_K.items():
		if K == 0:                                                     # no neighbours -> stays 0
			continue
		kps = unique_kps_final[rows].tolist()
		x_rest = np.array([np.asarray(kp_rest[kp], dtype=float).reshape(3,) for kp in kps]).reshape(-1, 3)
		nids = np.array([np.asarray(kp_neighbors[kp], dtype=int) for kp in kps], dtype=int).reshape(-1, K)
		wts = np.array([np.asarray(kp_weights[kp], dtype=float) for kp in kps], dtype=float).reshape(-1, K)
		js = np.array([[edg_idx_map.get(int(nid), -1) for nid in r] for r in nids.tolist()], dtype=np.int64).reshape(-1, K)
		x_def = np.zeros((len(rows), T_rev, 3), dtype=float)
		for n in range(K):
			j = js[:, n]; has = j >= 0                               # neighbours outside the graph are skipped
			if not has.any():
				continue
			jn, g_j = j[has], g[j[has]]
			Rv = np.matmul(R_edg[jn], (x_rest[has] - g_j)[:, None, :, None])[..., 0]       # (N,T_rev,3)
			x_def[has] += wts[has, n][:, None, None] * (Rv + g_j[:, None, :] + p_edg[jn])
		dyn_mapped[rows] = x_def

	print(f"Mapped {Nd_final} keypoints across {T_rev} time steps (missing rest={missing_rest})")
	edg_deformed = g[:, None, :] + p_edg

	return {
		"R_edg": R_edg,
		"p_edg": p_edg,
		"dyn_mapped": dyn_mapped,
		"edg_deformed": edg_deformed,
	}

