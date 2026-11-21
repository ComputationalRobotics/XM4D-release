from __future__ import annotations

from typing import Any

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
	idx = np.asarray(rt["indices_all"]).astype(np.int64)
	for oi in range(min(int(T), idx.shape[0])):
		j = int(idx[oi])
		if 0 <= j < N:
			R_raw[oi] = R_all[:, 3 * j : 3 * (j + 1)]
			t_raw[oi] = t_all[:, j]
	return R_raw, t_raw


def scatter_solved_frame_vector(rt, name, T, default=1.0):
	"""Scatter a per-solved-frame 1-D array (e.g. 's', 'da3_frame_scale') onto the full
	T-frame timeline via `indices_all`: a (T,) array with `default` at dropped frames."""
	raw = np.asarray(rt[name], dtype=np.float64).reshape(-1)
	out = np.full((int(T),), float(default), dtype=np.float64)
	idx = np.asarray(rt["indices_all"]).astype(np.int64)
	for oi in range(min(int(T), idx.shape[0])):
		j = int(idx[oi])
		if 0 <= j < raw.shape[0]:
			out[oi] = raw[j]
	return out


def _assemble_normal_eqs_vec(*, edges, edg_idx_map, g, edg_obs, edg_obs_mask, N_dyn_obs, dyn_kp_idx, dyn_frame, dyn_global,
							 frame_to_trev, kp_neighbors, kp_weights, kp_rest, R_start, p_start, T_rev, nz, w_obs, w_arap, w_smooth, w_edg_obs):
	"""The normal equations of solve_qcqp_and_recover_until_part5: the blocks (2 w C^T C, -2 w C^T b, w b^T b) of the
	observation, ARAP, EDG-observation and smoothness terms, built per batch. Returns Q_I_list, Q_J_list, Q_V_list,
	q_accum, c_accum."""
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
	edg_obs = np.asarray(edg_obs, dtype=float)
	M_ = g.shape[0]
	if edg_obs.shape != (M_, T_rev, 3):
		raise ValueError(f"Expected edg_obs shape {(M_, T_rev, 3)}, got {edg_obs.shape}")
	edg_obs_mask = np.asarray(edg_obs_mask, dtype=bool)
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
	edg_obs: np.ndarray,
	edg_obs_mask: np.ndarray,
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
	from scipy.sparse import coo_matrix
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

	print(f"Building all residual blocks batched ({N_dyn_obs} observations, {edges.shape[0]} edges, {M} nodes)...")
	Q_I_list, Q_J_list, Q_V_list, q_accum, c_accum = _assemble_normal_eqs_vec(
		edges=edges, edg_idx_map=edg_idx_map, g=g, edg_obs=edg_obs, edg_obs_mask=edg_obs_mask, N_dyn_obs=N_dyn_obs,
		dyn_kp_idx=dyn_kp_idx, dyn_frame=dyn_frame, dyn_global=dyn_global, frame_to_trev=frame_to_trev,
		kp_neighbors=kp_neighbors, kp_weights=kp_weights, kp_rest=kp_rest, R_start=R_start, p_start=p_start,
		T_rev=T_rev, nz=nz, w_obs=w_obs, w_arap=w_arap, w_smooth=w_smooth, w_edg_obs=w_edg_obs)

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

