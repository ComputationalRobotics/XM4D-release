#!/usr/bin/env python3
"""Diagnostic plot of a solved EDG's MOTION: which nodes move, which rotate, and about what axis.

Reads any npz in the EDG_test schema (node_ids / edg_reconstruct / edg_rotation) -- the holey
edg_test/edg_global.npz from EDG_test_fast.py or the dense edg_test/edg_completed.npz -- and writes a
4-panel figure:
  (a) distance of each node from its own mean position, per frame -- a node fixed in space (a hub, the
      pivot of a pinwheel) sits near 0 while the others trace their orbit radius;
  (b) each node's frame-to-frame rotation STEP, ang(R(t)^T R(t+1)), per frame. This is gauge-free (the
      absolute ang(R(t)) is not: every node is anchored at its OWN rest frame, so absolute curves are
      offset against each other for a reason that has nothing to do with the motion). On one rigid body
      every node must follow the SAME step curve, so spread between the lines is solver disagreement;
  (c) the node trajectories projected into the plane orthogonal to the dominant rotation axis -- a
      pinwheel shows concentric circles about the hub;
  (d) per node: mean orbit radius vs mean rotation step, which separates the static pivot from the blades
      and exposes the nodes whose rotation spins far faster than the body.
The console prints the same numbers and names the most static node.

Run:  conda run -n XM python visualize_edg_motion.py [--npz <outputs>/edg_test/edg_global.npz]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import utils.definitions as D


def _ang(R):
    """Rotation angle (deg) of each 3x3 in R[..., 3, 3]; NaN where R is not finite."""
    tr = np.trace(R, axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def _axis(R):
    """Unit rotation axis of each 3x3 (NaN for near-identity, where the axis is undefined)."""
    v = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(n > 1e-8, v / np.maximum(n, 1e-12), np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default=None, help="default: <outputs>/edg_test/edg_global.npz")
    ap.add_argument("--out", default=None, help="output PNG (default: <outputs>/edg_motion_<npz stem>.png)")
    a = ap.parse_args()

    p = Path(a.npz) if a.npz else Path(D.output_dir) / "edg_test" / "edg_global.npz"
    d = np.load(str(p), allow_pickle=True)
    ids = np.asarray(d["node_ids"]).reshape(-1)
    c = np.asarray(d["edg_reconstruct"], np.float64)                   # (M,T,3)
    R = np.asarray(d["edg_rotation"], np.float64)                      # (M,T,3,3)
    M, T = c.shape[:2]

    with np.errstate(invalid="ignore"):
        ctr = np.nanmean(c, axis=1, keepdims=True)
        rad = np.linalg.norm(c - ctr, axis=-1)                         # (M,T) distance from own mean
        okR = np.isfinite(R).all((-1, -2))
        ang = np.where(okR, _ang(np.where(okR[..., None, None], R, np.eye(3))), np.nan)
        ax = np.where(okR[..., None], _axis(np.where(okR[..., None, None], R, np.eye(3))), np.nan)

    # dominant rotation axis = principal direction of all solved axes (sign-free, so use the outer product)
    A = ax.reshape(-1, 3); A = A[np.isfinite(A).all(1)]
    if len(A):
        w, V = np.linalg.eigh(A.T @ A)
        axis = V[:, -1]
        spread = np.degrees(np.arccos(np.clip(np.abs(A @ axis), 0, 1)))
    else:
        axis, spread = np.array([0.0, 0.0, 1.0]), np.zeros(0)
    e1 = np.cross(axis, [1.0, 0.0, 0.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(axis, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1); e2 = np.cross(axis, e1)

    step = np.full((M, T - 1), np.nan)                                 # gauge-free frame-to-frame rotation
    for i in range(M):
        m = okR[i, :-1] & okR[i, 1:]
        if m.any():
            step[i, m] = _ang(np.einsum("tab,tbc->tac", R[i, :-1][m].transpose(0, 2, 1), R[i, 1:][m]))
    r_mean = np.nanmean(rad, axis=1)
    travel = np.nanmean(step, axis=1)                                  # mean rotation step (deg/frame)
    order = np.argsort(r_mean)
    print(f"[motion] {p.name}: {M} nodes, T={T}, rotations on {100 * okR.mean():.1f}% of node-frames")
    print(f"[motion] dominant rotation axis {np.round(axis, 3).tolist()}; "
          f"{len(A)} solved axes deviate from it by median {np.median(spread) if len(spread) else 0:.1f} deg")
    print("  node   orbit-radius(mean)   rot-step(deg/frame)   rot-step p90   frames-with-R")
    for i in order.tolist():
        p90 = np.nanpercentile(step[i], 90) if np.isfinite(step[i]).any() else np.nan
        print(f"  {ids[i]:5d} {r_mean[i]:18.4f} {travel[i]:21.2f} {p90:14.2f} {int(okR[i].sum()):15d}")
    print(f"[motion] MOST STATIC node: {ids[order[0]]} (orbit radius {r_mean[order[0]]:.4f}, "
          f"{r_mean[order[0]] / max(np.nanmedian(r_mean), 1e-9):.2f}x the median node)")
    cnt = np.isfinite(step).sum(0); sel = cnt >= 5
    if sel.any():
        sp = np.nanstd(step[:, sel], 0); md = np.nanmedian(step[:, sel])
        print(f"[motion] RIGIDITY: across-node spread of the rotation step is {np.nanmedian(sp):.2f} deg "
              f"vs a median step of {md:.2f} deg ({np.nanmedian(sp) / max(md, 1e-9):.2f}x). On one rigid "
              "body this ratio should be well under 1.")

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    cm = plt.get_cmap("turbo")(np.linspace(0, 1, M))
    for i in range(M):
        axs[0, 0].plot(rad[i], color=cm[i], lw=0.9)
        axs[0, 1].plot(step[i], color=cm[i], lw=0.9)
    axs[0, 0].set(title="(a) distance from the node's own mean position", xlabel="frame", ylabel="world units")
    axs[0, 1].set(title="(b) rotation STEP ang(R(t)$^T$R(t+1)) -- gauge-free; one rigid body = one curve",
                  xlabel="frame", ylabel="deg / frame")
    for i in range(M):
        u = (c[i] - ctr.mean(0)) @ e1; v = (c[i] - ctr.mean(0)) @ e2
        axs[1, 0].plot(u, v, color=cm[i], lw=0.9)
    axs[1, 0].set(title="(c) trajectories in the plane orthogonal to the dominant rotation axis",
                  xlabel="u", ylabel="v"); axs[1, 0].set_aspect("equal", "datalim")
    axs[1, 1].scatter(r_mean, travel, c=cm, s=60, edgecolor="k", lw=0.5)
    for i in range(M):
        axs[1, 1].annotate(str(ids[i]), (r_mean[i], travel[i]), fontsize=7,
                           xytext=(3, 3), textcoords="offset points")
    axs[1, 1].set(title="(d) orbit radius vs mean rotation step", xlabel="mean orbit radius",
                  ylabel="mean step (deg/frame)")
    for x in axs.ravel():
        x.grid(alpha=0.3)
    fig.suptitle(f"EDG motion -- {p.name}  ({M} nodes, {T} frames, "
                 f"{100 * okR.mean():.0f}% rotations solved)")
    fig.tight_layout()
    out = Path(a.out) if a.out else Path(D.output_dir) / f"edg_motion_{p.stem}.png"
    fig.savefig(out, dpi=130)
    print(f"[motion] wrote {out}")


if __name__ == "__main__":
    main()
