from pathlib import Path
import numpy as np
import utils.definitions

try:
    import torch
except Exception:
    torch = None

def find_repo_root():
    root = Path(utils.definitions.rootpath)
    cur = root
    for _ in range(6):
        if (cur / "deps" / "sam-3d-objects").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise FileNotFoundError("Could not locate deps/sam-3d-objects from utils.definitions.rootpath")

def load_depth_and_intrinsics(frame_idx=0):
    da3_dir = Path(utils.definitions.da3_depth_dir)
    depth_path = da3_dir / "depth_maps.npy"
    intr_path = da3_dir / "intrinsics.npy"
    if not depth_path.exists() or not intr_path.exists():
        raise FileNotFoundError(f"Missing depth_anything_3 outputs in {da3_dir}")
    depth_maps = np.load(depth_path)
    intrinsics = np.load(intr_path)
    if depth_maps.ndim != 3:
        raise ValueError(f"depth_maps.npy should be (T,H,W), got {depth_maps.shape}")
    if intrinsics.ndim == 2:
        intr = intrinsics
    else:
        intr = intrinsics[frame_idx]
    depth = depth_maps[frame_idx]
    return depth, intr

def load_mask_for_frame(frame_idx=0):
    mask_dir = Path(utils.definitions.mask_dir)
    if not mask_dir.exists():
        return None
    mask_paths = sorted(mask_dir.glob(utils.definitions.mask_pattern))
    if not mask_paths:
        return None
    if frame_idx < 0 or frame_idx >= len(mask_paths):
        return None
    from PIL import Image
    mask = np.array(Image.open(mask_paths[frame_idx]).convert("L")) > 0
    return mask

def _is_torch_tensor(val) -> bool:
    return torch is not None and torch.is_tensor(val)

def lift_depth_points(depth_map, intrinsics, mask=None, stride=4):
    depth = depth_map.cpu().numpy() if _is_torch_tensor(depth_map) else depth_map
    K = intrinsics.cpu().numpy() if _is_torch_tensor(intrinsics) else intrinsics
    H, W = depth.shape
    ys = np.arange(0, H, stride)
    xs = np.arange(0, W, stride)
    xv, yv = np.meshgrid(xs, ys)
    if mask is not None:
        m = np.asarray(mask)
        m = m[yv, xv] > 0
    else:
        m = np.ones_like(xv, dtype=bool)
    z = depth[yv, xv]
    valid = np.isfinite(z) & (z > 0) & m
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)
    xv = xv[valid].astype(np.float32)
    yv = yv[valid].astype(np.float32)
    z = z[valid].astype(np.float32)
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    X = (xv - cx) / fx * z
    Y = (yv - cy) / fy * z
    pts = np.stack([X, Y, z], axis=1)
    return pts

def filter_correspondences_by_error(
    A_pts,
    B_pts,
    corrs_A,
    corrs_B,
    max_percentage,
    w_line=1.0,
    w_origin=0.05,
):
    if max_percentage is None:
        return corrs_A, corrs_B
    if len(corrs_A) == 0 or len(corrs_B) == 0:
        return corrs_A, corrs_B
    A_pts = np.asarray(A_pts, dtype=np.float64)
    B_pts = np.asarray(B_pts, dtype=np.float64)
    corrs_A = np.asarray(corrs_A, dtype=np.int64)
    corrs_B = np.asarray(corrs_B, dtype=np.int64)

    A_sel = A_pts[corrs_A]
    B_sel = B_pts[corrs_B]
    B_norm = np.linalg.norm(B_sel, axis=1)
    valid = np.isfinite(B_norm) & (B_norm > 1e-8)
    if not np.any(valid):
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    A_sel = A_sel[valid]
    B_sel = B_sel[valid]
    corrs_A = corrs_A[valid]
    corrs_B = corrs_B[valid]

    A_norm = np.linalg.norm(A_sel, axis=1)
    u = B_sel / B_norm[valid][:, None]
    proj = np.sum(A_sel * u, axis=1)
    proj_vec = proj[:, None] * u
    d_line = np.linalg.norm(A_sel - proj_vec, axis=1)
    d_origin = np.abs(A_norm - B_norm[valid])
    loss = w_line * d_line + w_origin * d_origin
    max_error = np.percentile(loss, max_percentage * 100)
    keep = loss <= max_error
    return corrs_A[keep], corrs_B[keep]


def find_radial_correspondences(
    A_pts,
    B_pts,
    w_line=1.0,
    w_origin=0.05,
    max_corrs=5000,
    top_k=5,
    max_percentage=None,
):
    A_pts = np.asarray(A_pts, dtype=np.float64)
    B_pts = np.asarray(B_pts, dtype=np.float64)
    if A_pts.size == 0 or B_pts.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    N = A_pts.shape[0]
    K = max(1, min(int(top_k), N))
    A_norm = np.linalg.norm(A_pts, axis=1)
    b_idx = np.arange(B_pts.shape[0])
    if max_corrs is not None and B_pts.shape[0] > max_corrs:
        b_idx = np.linspace(0, B_pts.shape[0] - 1, max_corrs).astype(np.int64)
    best_b_idx = -np.ones((N, K), dtype=np.int64)
    best_b_loss = np.full((N, K), np.inf, dtype=np.float64)
    min_loss_A = np.full(N, np.inf, dtype=np.float64)
    min_loss_B = np.full(B_pts.shape[0], np.inf, dtype=np.float64)
    b_topk = {}
    b_topk_loss = {}
    for j in b_idx:
        b = B_pts[j]
        b_norm = np.linalg.norm(b)
        if not np.isfinite(b_norm) or b_norm <= 1e-8:
            continue
        u = b / b_norm
        proj = A_pts @ u
        proj_vec = np.outer(proj, u)
        d_line = np.linalg.norm(A_pts - proj_vec, axis=1)
        d_origin = np.abs(A_norm - b_norm)
        loss = w_line * d_line + w_origin * d_origin
        min_loss_A = np.minimum(min_loss_A, loss)
        min_loss_B[int(j)] = float(np.min(loss))
        k = min(K, N)
        idx = np.argpartition(loss, k - 1)[:k]
        idx = idx[np.argsort(loss[idx])]
        b_topk[int(j)] = idx
        b_topk_loss[int(j)] = {int(a): float(loss[a]) for a in idx}
        for a in range(N):
            worst_pos = int(np.argmax(best_b_loss[a]))
            if loss[a] < best_b_loss[a, worst_pos]:
                best_b_loss[a, worst_pos] = loss[a]
                best_b_idx[a, worst_pos] = int(j)
    valid_A = np.ones(N, dtype=bool)
    valid_B = np.ones(B_pts.shape[0], dtype=bool)
    if max_percentage is not None:
        valid_A = min_loss_A <= np.percentile(min_loss_A, max_percentage * 100)
        valid_B = min_loss_B <= np.percentile(min_loss_B, max_percentage * 100)

    a_topk = []
    for a in range(N):
        if not valid_A[a]:
            a_topk.append(set())
            continue
        valid = best_b_idx[a][best_b_idx[a] >= 0]
        a_topk.append(set(int(x) for x in valid))
    pairs = []
    for b, a_list in b_topk.items():
        if not valid_B[int(b)]:
            continue
        for a in a_list:
            if b in a_topk[int(a)]:
                pairs.append((b_topk_loss[b][int(a)], int(a), int(b)))
    pairs.sort(key=lambda x: x[0])
    corrs_A = []
    corrs_B = []
    used_a = set()
    used_b = set()
    for loss, a, b in pairs:
        if a in used_a or b in used_b:
            continue
        used_a.add(a)
        used_b.add(b)
        corrs_A.append(a)
        corrs_B.append(b)
    corrs_A = np.asarray(corrs_A, dtype=np.int64)
    corrs_B = np.asarray(corrs_B, dtype=np.int64)
    return filter_correspondences_by_error(
        A_pts,
        B_pts,
        corrs_A,
        corrs_B,
        max_percentage,
        w_line=w_line,
        w_origin=w_origin,
    )
