#!/usr/bin/env python3
"""Partition the solved EDG into K rigid parts from the RELATIVE POSE between node pairs.

Two nodes belong to the same rigid body exactly when their relative pose is constant in time. For nodes
i, j with world pose (R_i(t), c_i(t)) the relative pose expressed in i's frame is

    R_ij(t) = R_i(t)^T R_j(t)              (relative rotation)
    t_ij(t) = R_i(t)^T (c_j(t) - c_i(t))   (relative translation)

so the pair dissimilarity is how much those WANDER over the frames both nodes are valid:

    d_rot(i,j) = mean geodesic angle of R_ij(t) about its own Karcher-style mean      [deg]
    d_tra(i,j) = mean ||t_ij(t) - mean_t t_ij|| , divided by the scene scale          [relative]
    d(i,j)     = d_rot / rot_scale + w_tra * d_tra / tra_scale

Nodes are then grouped by spectral clustering on the affinity exp(-d^2), which needs no threshold -- only K.
Using the relative POSE rather than only the relative DISTANCE separates parts that rotate about a shared
centre (a spinning blade against its mount keeps its distance but not its orientation).

Writes <outputs>/edg_test/edg_rigid_parts.npz (labels per node id) and prints the partition.

Run: XM_DATASET_CAT=Dycheck XM_DATASET_NAME=paper-windmill \
       conda run -n XM python edg_rigid_parts.py --k 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import utils.definitions as D


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, required=True, help="number of rigid parts")
    ap.add_argument("--edg", default=None, help="edg_completed.npz (default: <outputs>/edg_test/)")
    ap.add_argument("--w-tra", type=float, default=1.0, help="weight of the translation term vs rotation")
    ap.add_argument("--w-rot-change", type=float, default=1.0,
                    help="weight of the per-frame RELATIVE-ROTATION CHANGE term. A rigid pair has R_ij(t) "
                         "constant, so its frame-to-frame change is 0; a pair on different bodies changes "
                         "every frame. This is a cleaner rigidity test than the spread about the mean, "
                         "which also fires when a pair is rigid but the pose estimate is noisy.")
    ap.add_argument("--min-covis", type=int, default=5, help="pairs valid in fewer frames are unconstrained")
    ap.add_argument("--reduce", choices=["sum", "mean"], default="sum",
                    help="how each pair's per-frame deviation from its rest pose is accumulated. 'sum' is the "
                         "TOTAL deviation over the co-visible frames, so a pair that stays together for a "
                         "long time accrues more evidence than one briefly seen; 'mean' normalises that away.")
    ap.add_argument("--out", default="edg_rigid_parts.npz")
    ap.add_argument("--viz", action="store_true",
                    help="also render the EDG nodes coloured by part onto the frames (mp4; --frame for a PNG)")
    ap.add_argument("--tracks", default="tracks_dynamic_merged.npz",
                    help="2D positions used by --viz; node_ids index into this file")
    ap.add_argument("--frame", type=int, default=None, help="--viz: write this single frame as a PNG")
    ap.add_argument("--axes", action="store_true",
                    help="--viz: also draw each node's ROTATION as a projected coordinate frame "
                         "(x=red, y=green, z=blue), so a node whose orientation is wrong is visible directly")
    ap.add_argument("--axis-len", type=float, default=0.03,
                    help="--axes: axis length in solver-world units")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--scale", type=float, default=2.0)
    return ap.parse_args()


def relative_pose_dissimilarity(R, c, valid, min_covis, reduce="sum"):
    """-> (rot[M,M] deg spread about the mean, chg[M,M] deg per-frame change, tra[M,M], ok[M,M])."""
    red = (lambda v: float(np.sum(v))) if reduce == "sum" else (lambda v: float(np.mean(v)))
    M, T = R.shape[0], R.shape[1]
    rot = np.full((M, M), np.nan)
    chg = np.full((M, M), np.nan)
    tra = np.full((M, M), np.nan)
    ok = np.zeros((M, M), bool)
    for i in range(M):
        for j in range(i + 1, M):
            m = valid[i] & valid[j]
            n = int(m.sum())
            if n < min_covis:
                continue
            Ri, Rj = R[i, m], R[j, m]                                  # (n,3,3)
            Rij = np.einsum("tba,tbc->tac", Ri, Rj)                    # R_i^T R_j
            tij = np.einsum("tba,tb->ta", Ri, c[j, m] - c[i, m])       # R_i^T (c_j - c_i)
            # rotation spread: angle of R_mean^T R(t), with R_mean the projection of the arithmetic mean
            U, _, Vt = np.linalg.svd(Rij.mean(0))
            Rm = U @ Vt
            if np.linalg.det(Rm) < 0:
                U[:, -1] *= -1.0
                Rm = U @ Vt
            dR = np.einsum("ba,tbc->tac", Rm, Rij)
            cos = np.clip((np.trace(dR, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
            rot[i, j] = rot[j, i] = red(np.degrees(np.arccos(cos)))      # sum_t |R_ij(t) - R_0|
            # per-frame CHANGE of the relative rotation: 0 for a rigid pair, whatever its absolute offset
            if n >= 2:
                step = np.einsum("tba,tbc->tac", Rij[:-1], Rij[1:])
                cs = np.clip((np.trace(step, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
                chg[i, j] = chg[j, i] = red(np.degrees(np.arccos(cs)))
            else:
                chg[i, j] = chg[j, i] = 0.0
            tra[i, j] = tra[j, i] = red(np.linalg.norm(tij - tij.mean(0), axis=1))   # sum_t |p(t) - p_0|
            ok[i, j] = ok[j, i] = True
    return rot, chg, tra, ok


def main():
    a = parse_args()
    edg = Path(a.edg) if a.edg else Path(D.output_dir) / "edg_test" / "edg_completed.npz"
    d = np.load(str(edg), allow_pickle=True)
    ids = np.asarray(d["node_ids"]).reshape(-1)
    c = np.asarray(d["edg_reconstruct"], np.float64)                   # (M,T,3)
    R = np.asarray(d["edg_rotation"], np.float64)                      # (M,T,3,3)
    valid = np.isfinite(c).all(-1) & np.isfinite(R).all((-1, -2))
    M = len(ids)
    nco = np.array([[int((valid[i] & valid[j]).sum()) for j in range(M)] for i in range(M)])
    off = nco[~np.eye(M, dtype=bool)]
    print(f"[parts] {edg.name}: {M} nodes, T={c.shape[1]}, valid {valid.mean():.2f}; "
          f"pair co-visibility min={off.min()} max={off.max()}"
          + ("  (uniform -> sum and mean give the SAME partition)" if off.min() == off.max() else ""))
    if a.k > M:
        raise SystemExit(f"--k {a.k} exceeds the {M} available nodes")

    rot, chg, tra, ok = relative_pose_dissimilarity(R, c, valid, a.min_covis, a.reduce)
    if not ok.any():
        raise SystemExit("no node pair is co-visible often enough; lower --min-covis")
    # scale each term by its own median so the two are comparable without hand-tuned units
    rs = np.nanmedian(rot[ok]) or 1.0
    cs_ = np.nanmedian(chg[ok]) or 1.0
    ts = np.nanmedian(tra[ok]) or 1.0
    dis = rot / rs + a.w_rot_change * chg / cs_ + a.w_tra * tra / ts
    big = np.nanmax(dis[ok]) * 2.0 if ok.any() else 1.0
    dis = np.where(ok, dis, big)                                       # unconstrained pairs: assume unrelated
    np.fill_diagonal(dis, 0.0)
    print(f"[parts] reduce={a.reduce} (p_0 = mean relative pose over co-visible frames)\n"
          f"[parts] pair dissimilarity: rot-spread median {rs:.2f} deg, rot-CHANGE median {cs_:.2f} "
          f"deg/frame, translation median {ts:.4f} world units; "
          f"{int(ok.sum()) // 2}/{M * (M - 1) // 2} pairs co-visible >= {a.min_covis} frames")

    sigma = np.median(dis[ok]) or 1.0
    A = np.exp(-(dis / sigma) ** 2)
    np.fill_diagonal(A, 1.0)
    from sklearn.cluster import SpectralClustering
    lab = SpectralClustering(n_clusters=a.k, affinity="precomputed", assign_labels="kmeans",
                             random_state=0).fit_predict(A)

    print(f"[parts] K={a.k} partition:")
    for p in range(a.k):
        sel = np.where(lab == p)[0]
        intra = dis[np.ix_(sel, sel)][np.triu_indices(len(sel), 1)]
        other = np.where(lab != p)[0]
        inter = dis[np.ix_(sel, other)]
        print(f"  part {p}: {len(sel):3d} nodes  ids={ids[sel].tolist()}")
        if intra.size:
            print(f"           intra-part dissimilarity median {np.median(intra):.2f} "
                  f"vs inter-part {np.median(inter):.2f}")
    out = edg.parent / a.out
    np.savez(out, node_ids=ids, label=lab.astype(np.int64), dis=dis, rot=rot, chg=chg, tra=tra,
             ok=ok, k=a.k)
    print(f"[parts] wrote {out}")
    if a.viz:
        render_parts(a, ids, lab, c, R)


# one strongly separated colour per part (BGR)
PART_COLORS = [(60, 220, 60), (60, 130, 255), (255, 200, 0), (200, 80, 255), (60, 220, 220),
               (0, 165, 255), (255, 80, 160), (120, 255, 120)]


def render_parts(a, ids, lab, c3d=None, Rn=None):
    """Draw the EDG nodes on the frames, one colour per rigid part, from their 2D track positions.
    With --axes, also project each node's own rotation as a little coordinate frame."""
    import glob

    import cv2

    tp = Path(D.output_dir) / a.tracks
    d = np.load(str(tp), allow_pickle=True)
    xy = np.asarray(d["tracks_xy"], np.float64)[0]
    occ = np.asarray(d["occluded"])[0].astype(bool)
    images = sorted(glob.glob(str(Path(D.video_dir) / D.image_pattern)))
    T = min(xy.shape[1], len(images))

    cam = None
    if a.axes:                                                         # solver cameras + intrinsics, so the
        from build_edg import _load_all, _frame_scale                  # node rotations can be projected
        from train_4dgs_edg import _proj_uv_z
        DD = _load_all()
        K = np.load(str(D.active_intrinsics))
        K = np.asarray(K, np.float64)
        K = K if K.ndim == 3 else np.repeat(K[None], T, 0)
        cam = dict(R=DD["R_raw"], t=DD["t_raw"], K=K, s=_frame_scale(DD["rt"], DD["T"], False),
                   proj=_proj_uv_z)
        print(f"[parts] --axes: projecting node rotations with the solver cameras "
              f"(K {K.shape}, {len(cam['R'])} frames)")
    if ids.max() >= xy.shape[0]:
        raise SystemExit(f"{tp.name} has {xy.shape[0]} tracks but node id {ids.max()} was requested -- "
                         "the EDG was built from a different tracks file")

    def draw(t):
        img = cv2.imread(images[t])
        for n, g in enumerate(ids.tolist()):
            if occ[g, t] or not np.isfinite(xy[g, t]).all():
                continue
            col = PART_COLORS[int(lab[n]) % len(PART_COLORS)]
            pt = (int(round(xy[g, t, 0])), int(round(xy[g, t, 1])))
            cv2.circle(img, pt, 5, col, -1, cv2.LINE_AA)
            cv2.circle(img, pt, 5, (0, 0, 0), 1, cv2.LINE_AA)
            if cam is not None and t < len(cam["R"]) and np.isfinite(c3d[n, t]).all():
                Rc = np.asarray(cam["R"][t], np.float64)[:3, :3]
                org = c3d[n, t]
                tips = org[None] + Rn[n, t].T * float(a.axis_len)      # rows of R^T = the node's x,y,z axes
                pw = np.vstack([org[None], tips])
                u, v, z = cam["proj"](pw, Rc, cam["t"][t], cam["K"][min(t, len(cam["K"]) - 1)])
                if np.isfinite(u).all() and np.isfinite(v).all() and (z > 1e-6).all():
                    o = (int(round(u[0])), int(round(v[0])))
                    for k_, acol in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):   # x r, y g, z b
                        cv2.line(img, o, (int(round(u[1 + k_])), int(round(v[1 + k_]))), acol, 2, cv2.LINE_AA)
        img = cv2.resize(img, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_NEAREST)
        cv2.putText(img, f"frame {t}/{T-1}  K={a.k} rigid parts", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        for p in range(a.k):                                           # legend
            cv2.circle(img, (16, 44 + 22 * p), 6, PART_COLORS[p % len(PART_COLORS)], -1, cv2.LINE_AA)
            cv2.putText(img, f"part {p}  ({int((lab == p).sum())} nodes)", (30, 50 + 22 * p),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    if a.frame is not None:
        f = int(np.clip(a.frame, 0, T - 1))
        png = Path(D.output_dir) / f"edg_parts_k{a.k}_frame{f:04d}.png"
        cv2.imwrite(str(png), draw(f))
        print(f"[parts] wrote {png}")
        return
    mp4 = Path(D.output_dir) / f"edg_parts_k{a.k}.mp4"
    h, w = draw(0).shape[:2]
    vw = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
    for t in range(T):
        vw.write(draw(t))
    vw.release()
    print(f"[parts] wrote {mp4}  ({T} frames)")


if __name__ == "__main__":
    main()
