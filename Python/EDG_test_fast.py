#!/usr/bin/env python3
"""Fast multi-run EDG solve (approximation of EDG_test.py, much cheaper).

EDG_test.py chains 10-frame segments across the WHOLE video, so its node set grows to hundreds and the
LATE solves have huge dense systems. Instead, do MANY short RUNS:
  * each run picks a random gap g and a start s, giving ~n_seg*(seg_len-1)+1 STRIDED frames
    s, s+g, s+2g, ...  (gap 1 -> ~50 real frames, gap 2 -> ~100, gap 3 -> ~150). The start is biased
    toward windows that currently have MORE HOLES (--hole-bias): before each run we count, per frame, the
    EDG cells that are observable but still unfilled, and sample s ~ (windowed hole count)^bias -> later
    runs chase the gaps left by earlier ones instead of re-covering the same region (bias 0 = uniform);
  * within a run it does EXACTLY EDG_test's chaining: up to `n_seg` (=5) segments of `seg_len` (=11)
    strided frames with a 1-frame overlap, CARRYING each solved EDG node into the next segment and
    composing the global rotation (via EDG_test.build_segment_inputs + solve_segment). Capping a run to
    5 segments bounds the per-solve node count -> each solve stays small and fast;
  * think of the result as ONE table  table[node, frame] = 3D position, most cells empty at the start;
  * each run writes its solved positions into the table with a per-cell RELIABILITY weight: a cell is most
    trustworthy at a frame where that node was DIRECTLY observed and decays as the solve extrapolates away
    from it -- weight = 1/(1+d), d = #frames from this frame to the node's NEAREST direct observation in the
    run (observed to 147, solved at 150 -> d=3 -> w=1/4). A node never observed in a run contributes nothing;
  * every (node/dyn, frame) is that weighted AVERAGE over all runs that touched it (all runs solve in the
    same world frame -> directly averageable);
  * OPTIONALLY (--seed-obs, default OFF) a node's OWN direct observations can additionally be seeded into
    the table at weight 1 -- that fills EVERY observed cell across the whole video, so by default it is off
    and the table stays a pure average of the independently solved sampled segments (NaN elsewhere);
  * ROTATIONS: the solve only yields per-run-relative LOCAL rotations (each relative to that run's own rest),
    so they cannot be averaged across runs. Instead we fit a GLOBAL rotation per (node, frame) from the
    finished POSITION table -- the Kabsch rotation mapping the node's EDG-neighbourhood cloud from its rest
    frame to frame f (the ARAP local-rotation step), world-referenced because the positions are. R[node,rest]
    = I; NaN where the cell is a hole or < --rot-min-nbr neighbours are shared;
  * NO local optimization / fill -- cells no run ever anchored stay HOLES (NaN).

Outputs:
  * edg_test/edg_global.npz          -- same schema as EDG_test (holey now, real fitted edg_rotation)
  * edg_test/edg_table.npz           -- node x frame table + weights + filled mask + fitted rotation[.,.,3,3]
  * edg_test/edg_table_coverage.png  -- heatmap of the table: white = hole, color = accumulated weight

Run:  XM_DATASET_NAME=<scene> conda run -n XM python EDG_test_fast.py [--merged] [--n-runs 30]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

import utils.definitions
import EDG_test as ET          # reuse load_all / build_segment_inputs / solve_segment (imports XM once)


def _dist_to_obs(obs, frames):
    """obs: [n,F] bool of DIRECTLY-observed cells per id at the sorted `frames` [F]. Return [n,F] float = #frames
    from each frame to the nearest observed frame in that row (0 at an observed frame); inf for a row with no
    observation."""
    idx = np.asarray(frames)[None, :]
    T = int(idx.max(initial=-1)) + 1
    prev = np.maximum.accumulate(np.where(obs, idx, -1), axis=1)     # last obs index <= f  (-1 if none yet)
    df = np.where(prev >= 0, idx - prev, np.inf)                     # distance looking backward
    nxt = np.minimum.accumulate(np.where(obs, idx, T)[:, ::-1], axis=1)[:, ::-1]  # first obs index >= f
    db = np.where(nxt < T, nxt - idx, np.inf)                        # distance looking forward
    return np.minimum(df, db)


def save_table_heatmap(node_pos, node_w, node_ids, path):
    """Heatmap of the node x frame table: white = hole, color = accumulated reliability weight."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    filled = np.isfinite(node_pos).all(-1)                           # [n,T]
    T = node_pos.shape[1]
    order = np.argsort(np.where(filled.any(1), filled.argmax(1), T)) # rows by first filled frame
    W = node_w[order].astype(float)
    W[~filled[order]] = np.nan                                       # holes -> white
    fig, ax = plt.subplots(figsize=(12, max(3.0, node_ids.size * 0.03)), dpi=120)
    cmap = plt.cm.viridis.copy(); cmap.set_bad("white")
    vmax = np.nanmax(W) if np.isfinite(W).any() else 1.0
    im = ax.imshow(W, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0.0, vmax=vmax)
    ax.set_xlabel("frame"); ax.set_ylabel("EDG node (sorted by first filled frame)")
    ax.set_title(f"EDG table  {100 * filled.mean():.1f}% filled  |  white = hole, color = accumulated weight")
    fig.colorbar(im, ax=ax, label="sum of 1/(1+d) over runs")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def solve_run(D, run_frames, seg_len, n_seg, w_arap, w_smooth, out_dir, max_dyn, rng, run_id, n_runs, gap):
    """One run = EDG_test chaining over the ordered strided `run_frames`: up to n_seg segments of seg_len
    frames (1-frame overlap), carrying each EDG node forward + composing global rotation. Returns
    node_pos dict gid -> [T,3], finite only at the run's frames; the dynamic points as (gids [n], positions
    [n, run_frames.size, 3], NaN where not solved); and #segments solved."""
    T = D["T"]
    edg_rot, edg_pos, dyn_seg = {}, {}, []

    def ens_rot(g):
        if g not in edg_rot:
            edg_rot[g] = np.tile(np.eye(3, dtype=np.float64), (T, 1, 1))
        return edg_rot[g]

    def ens_pos(store, g):
        if g not in store:
            store[g] = np.full((T, 3), np.nan, dtype=np.float64)
        return store[g]

    ci, nseg = 0, 0
    for seg_i in range(1, n_seg + 1):
        seg_frames = run_frames[ci: ci + seg_len]
        if seg_frames.size < 2:
            break
        a, b = int(seg_frames[0]), int(seg_frames[-1])
        inp = ET.build_segment_inputs(D, seg_frames, edg_pos, max_dyn=max_dyn, rng=rng)
        if inp is None:
            print(f"  --> sample {run_id}/{n_runs} (gap {gap}) | iter {seg_i}/{n_seg}: "
                  f"segment frames {a}-{b}: <2 EDG nodes, skipped", flush=True)
        else:
            print(f"  --> sample {run_id}/{n_runs} (gap {gap}) | iter {seg_i}/{n_seg}: "
                  f"segment frames {a}-{b}  nodes={inp['M']} ({inp['n_carried']} carried) "
                  f"dyn-obs={inp['N_dyn_obs']}", flush=True)
            try:
                ed, R_edg, dyn, ukps = ET.solve_segment(inp, out_dir, w_arap, w_smooth)
            except Exception as e:
                print(f"  --> sample {run_id}/{n_runs} | iter {seg_i}: solve FAILED "
                      f"({type(e).__name__}); ending this sample early", flush=True)
                break
            present, frames, rest = inp["present"], inp["frames"], inp["rest"]
            for j, gid in enumerate(present.tolist()):                  # carry + compose (like EDG_test)
                rot_arr = ens_rot(gid)
                rest_global = rot_arr[rest] if np.isfinite(rot_arr[rest]).all() else np.eye(3)
                Rg = rest_global[None, :, :] @ R_edg[j]
                pos_arr = ens_pos(edg_pos, gid)
                rot_arr[frames] = Rg; pos_arr[frames] = ed[j]
            dyn_seg.append((ukps, np.searchsorted(run_frames, frames), dyn))
            nseg += 1
        if ci + seg_len >= run_frames.size:
            break
        ci += seg_len - 1                                              # 1-frame overlap
    # dynamic points over the run's frames; a later segment overwrites the shared overlap frame (like the nodes)
    dyn_ids = np.unique(np.concatenate([u for u, _, _ in dyn_seg])) if dyn_seg else np.zeros(0, dtype=np.int64)
    dyn_pos = np.full((dyn_ids.size, run_frames.size, 3), np.nan, dtype=np.float64)
    for u, fi, dyn in dyn_seg:
        dyn_pos[np.searchsorted(dyn_ids, u)[:, None], fi[None, :]] = dyn
    return edg_pos, (dyn_ids, dyn_pos), nseg


def _sample_start(rng, hole, g, span_frames, T, bias):
    """Pick a run START s biased toward windows that currently have many HOLES. A run at start s touches the
    strided frames s, s+g, ..., s+g*(span_frames-1); score(s) = sum of `hole` (per-frame hole count) over
    those frames. Sample s ~ (score+eps)^bias over the valid starts [0, T-span). bias<=0 -> uniform (old)."""
    span = g * (span_frames - 1)
    hi = max(1, T - span)
    if bias <= 0:
        return int(rng.integers(0, hi))
    score = np.zeros(T, dtype=np.float64)
    for k in range(span_frames):
        gk = g * k                                                   # add hole[s+gk] into score[s]
        if gk >= T:
            break
        score[:T - gk] += hole[gk:]
    p = np.power(score[:hi] + 1e-6, bias)
    tot = p.sum()
    if not np.isfinite(tot) or tot <= 0:
        return int(rng.integers(0, hi))
    return int(rng.choice(hi, p=p / tot))


def _accumulate(pos_sum, pos_w, gids, P, valid, frames):
    """Add a run's per-id trajectories P [n,F,3] (rows = gids, columns = the run's sorted `frames`; a run
    solves nothing elsewhere, so only these cells change) into the global weighted sums. Cell (id, frame f)
    gets weight 1/(1+d), d = #frames from f to that id's NEAREST direct observation this run (valid[id]=True
    & solved); f observed -> d=0 -> w=1; extrapolated k frames past the last obs -> w=1/(1+k). No obs this
    run -> w=0."""
    if not gids.size:
        return
    solved = np.isfinite(P).all(-1)                                   # [n,F] cells this run produced
    d = _dist_to_obs(valid[np.ix_(gids, frames)].astype(bool) & solved, frames)  # [n,F] dist to nearest direct obs
    w = np.where(solved & np.isfinite(d), 1.0 / (1.0 + d), 0.0)       # [n,F]
    pos_sum[np.ix_(gids, frames)] += w[..., None] * np.where(solved[..., None], P, 0.0)
    pos_w[np.ix_(gids, frames)] += w


def _kabsch_R(P, Q):
    """Proper rotation (det=+1) mapping the centred point cloud P onto Q (least squares)."""
    cP, cQ = P.mean(0), Q.mean(0)
    H = (P - cP).T @ (Q - cQ)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def _fit_global_rotations(table, node_ids, edg_edges, rest_frame, min_nbr=4):
    """Fit a GLOBAL rotation per (node, frame) from the (globally-consistent) POSITION table -- the runs only
    give per-run-relative local rotations, so instead we recover each node's world rotation the ARAP way:
    R[i,f] = proper rotation (Kabsch) mapping node i's EDG-neighbourhood cloud from its rest frame to frame f.
    R[i, rest_i] == I by construction. NaN where the node/frame is a hole or < min_nbr shared neighbours are
    filled at both rest and f (rotation undetermined). rest_frame[i] = node i's reference frame (its first
    direct observation)."""
    Ne, T = table.shape[0], table.shape[1]
    filled = np.isfinite(table).all(-1)                                  # [Ne,T]
    id2loc = {int(n): i for i, n in enumerate(node_ids.tolist())}
    adj = [[] for _ in range(Ne)]
    for a, b in edg_edges.tolist():
        ia, ib = id2loc.get(int(a)), id2loc.get(int(b))
        if ia is not None and ib is not None:
            adj[ia].append(ib); adj[ib].append(ia)
    R = np.full((Ne, T, 3, 3), np.nan, dtype=np.float64)
    for i in range(Ne):
        rest = int(rest_frame[i])
        if rest < 0 or not filled[i, rest]:
            continue
        grp = np.array([i] + adj[i], dtype=int)                         # the node itself + its EDG neighbours
        at_rest = grp[filled[grp, rest]]                               # of those, the ones known at rest
        if at_rest.size < min_nbr:
            R[i, rest] = np.eye(3)                                      # rest is identity even if unrotatable
            continue
        for f in np.where(filled[i])[0].tolist():
            usable = at_rest[filled[at_rest, f]]
            if usable.size < min_nbr:
                continue
            R[i, f] = _kabsch_R(table[usable, rest], table[usable, f])
    return R


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg-len", type=int, default=11, help="frames per segment (EDG_test: 11, 1-frame overlap)")
    ap.add_argument("--n-seg", type=int, default=5, help="chained segments per run (5 x 10 = ~50 frames)")
    ap.add_argument("--n-runs", type=int, default=10, help="number of random-strided runs to average")
    ap.add_argument("--gaps", type=int, nargs="+", default=[1, 2], help="candidate strides per run")
    ap.add_argument("--seed-obs", action="store_true",
                    help="also seed each node's direct observations into the table at weight 1 (fills ALL "
                         "observed cells, not just sampled segments); default OFF = pure sampled average")
    ap.add_argument("--hole-bias", type=float, default=2.0,
                    help="bias run starts toward windows with more current holes (0 = uniform, higher = greedier)")
    ap.add_argument("--rot-min-nbr", type=int, default=4,
                    help="min EDG neighbours (incl self) filled at both rest and a frame to fit that frame's rotation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w-arap", type=float, default=0.2)
    ap.add_argument("--w-smooth", type=float, default=0.2)
    ap.add_argument("--max-dyn", type=int, default=0, help="subsample dyn keypoints per segment (0 = all)")
    ap.add_argument("--merged", action="store_true", help="solve on the MERGED tracks (match build_edg.py --merged)")
    ap.add_argument("--smoothed", action=argparse.BooleanOptionalAction, default=True,
                    help="use tracks_dynamic_merged_smoothed.npz (holds the 3D; the raw merged file is 2D-only). Default on.")
    args = ap.parse_args()

    merged_path = None
    if args.merged:
        name = "tracks_dynamic_merged_smoothed.npz" if args.smoothed else "tracks_dynamic_merged.npz"
        merged_path = str(Path(utils.definitions.output_dir) / name)
        if args.smoothed and not Path(merged_path).exists():          # not smoothed yet -> warn + fall back
            merged_path = str(Path(utils.definitions.output_dir) / "tracks_dynamic_merged.npz")
            print("[merged] WARNING: smoothed file missing -- the raw merged file is 2D-only (tracks_xyz all-NaN). "
                  "The solve will find no valid 3D.")
        print(f"[merged] solving on consolidated tracks: {merged_path}")
    D = ET.load_all(tracks_path=merged_path)
    T, Q = D["T"], D["Q"]
    edg_nodes, edg_edges, valid = D["edg_nodes"], D["edg_edges"], D["valid"]
    out_dir = Path(utils.definitions.output_dir) / "edg_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    pos_sum = np.zeros((Q, T, 3), dtype=np.float64)
    pos_w = np.zeros((Q, T), dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    span_frames = args.n_seg * (args.seg_len - 1) + 1                 # strided frames per run (~51)
    vn = valid[edg_nodes].astype(bool)                               # [Ne,T] which node-cells are observable

    n_ok = 0
    for run in range(args.n_runs):
        g = int(rng.choice(args.gaps))
        hole = (vn & (pos_w[edg_nodes] <= 0)).sum(0).astype(np.float64)  # [T] observable-but-unfilled cells now
        s = _sample_start(rng, hole, g, span_frames, T, args.hole_bias)
        run_frames = np.unique(s + g * np.arange(span_frames))
        run_frames = run_frames[run_frames < T]
        if run_frames.size < 2:
            continue
        print(f"\n===== SAMPLE {run+1}/{args.n_runs}: gap={g}, {run_frames.size} strided frames "
              f"{run_frames[0]}..{run_frames[-1]}, up to {args.n_seg} iterations of {args.seg_len}-frame "
              f"segments (targeting {int(hole[run_frames].sum())} unfilled observable cells) =====", flush=True)
        node_pos, (dyn_gids, dyn_pos), nseg = solve_run(D, run_frames, args.seg_len, args.n_seg, args.w_arap,
                                                        args.w_smooth, out_dir, (args.max_dyn or None), rng,
                                                        run + 1, args.n_runs, g)
        node_gids = np.fromiter(node_pos.keys(), dtype=np.int64)
        P = np.stack([node_pos[gid][run_frames] for gid in node_gids.tolist()]) if node_gids.size else None
        _accumulate(pos_sum, pos_w, node_gids, P, valid, run_frames)
        _accumulate(pos_sum, pos_w, dyn_gids, dyn_pos, valid, run_frames)
        n_ok += 1
        print(f"[sample {run+1}/{args.n_runs} done] {nseg} segments solved; "
              f"{len(node_pos)} nodes + {dyn_gids.size} dyn points touched")

    glob = D["glob"]                                                  # [Q,T,3] world-lifted observations
    en = edg_nodes
    obs_mask = valid[en] & np.isfinite(glob[en]).all(-1)             # a node's OWN direct observations
    if args.seed_obs:
        # optional: also write each node's raw observations into the table at weight 1. This fills EVERY
        # observed cell across the whole video (not just the sampled segments), so the table is no longer
        # "sampled-segment average, NaN elsewhere" -- off by default.
        pos_sum[en] += np.where(obs_mask[..., None], glob[en], 0.0)
        pos_w[en] += obs_mask.astype(np.float64)
        print(f"[obs] seeded {int(obs_mask.sum())} direct EDG observations at weight 1 "
              f"({int((obs_mask & (pos_w[en] <= 1)).sum())} of them fill cells no run had solved)")
    else:
        print("[obs] observation seeding OFF: table = average of independently solved sampled segments, "
              "NaN everywhere else (pass --seed-obs to also fill all directly-observed cells)")

    m = pos_w > 0

    def _mean(ids):                                                   # weighted average of the rows `ids` (NaN = no weight)
        out = np.full((ids.size, T, 3), np.nan, dtype=np.float64)
        r, f = np.nonzero(m[ids])
        out[r, f] = pos_sum[ids[r], f] / pos_w[ids[r], f][:, None]
        return out

    node_pos = _mean(edg_nodes)                                       # the node x frame table (NaN = hole)
    node_w = pos_w[edg_nodes].copy()                                  # accumulated reliability per cell
    filled = np.isfinite(node_pos).all(-1)
    print(f"[table] {n_ok}/{args.n_runs} runs; node-frame table {100 * filled.mean():.1f}% filled "
          f"({int((~filled).sum())} of {filled.size} cells are HOLES; no local fill)")

    node_mask = np.zeros(Q, dtype=bool); node_mask[edg_nodes] = True
    dyn_ids = np.where(m.any(1) & ~node_mask)[0]
    dyn_reconstruct = _mean(dyn_ids)

    node_ids = edg_nodes
    node_set = set(node_ids.tolist())
    edg_edges_global = np.array([[a, b] for a, b in edg_edges.tolist()
                                 if a in node_set and b in node_set], dtype=np.int64).reshape(-1, 2)

    # global per-node rotation, fitted from the position table (the runs only give per-run-relative rotations)
    rest_frame = np.where(obs_mask.any(1), obs_mask.argmax(1),         # each node's reference = its 1st direct
                          np.where(filled.any(1), filled.argmax(1), -1))  # obs (fallback: 1st filled frame)
    edg_rotation = _fit_global_rotations(node_pos, node_ids, edg_edges_global, rest_frame, args.rot_min_nbr)
    rot_filled = np.isfinite(edg_rotation).all(axis=(-1, -2))          # [Ne,T] where a rotation was fit
    print(f"[rot] fitted global rotations for {100 * rot_filled.mean():.1f}% of node-frames "
          f"({int((filled & ~rot_filled).sum())} filled cells lacked >= {args.rot_min_nbr} shared neighbours)")

    out = out_dir / "edg_global.npz"
    np.savez_compressed(out, node_ids=node_ids, edg_reconstruct=node_pos, edg_rotation=edg_rotation,
                        edg_edges=edg_edges_global, dyn_ids=dyn_ids, dyn_reconstruct=dyn_reconstruct,
                        n_tracks=int(Q))                              # track-id space stamp (merged vs raw)

    tbl = out_dir / "edg_table.npz"                                   # the explicit table with holes
    np.savez_compressed(tbl, node_ids=node_ids, frames=np.arange(T), table=node_pos,
                        weight=node_w, filled=filled, rotation=edg_rotation, rot_filled=rot_filled)
    png = out_dir / "edg_table_coverage.png"
    save_table_heatmap(node_pos, node_w, node_ids, png)
    print(f"\n[done] {out}  ({int(node_ids.size)} nodes, {int(dyn_ids.size)} dyn points, T={T})")
    print(f"[done] {tbl}  (table [{int(node_ids.size)},{T},3] + weight + filled + rotation [.,.,3,3] + rot_filled)")
    print(f"[done] {png}  (white = hole, color = accumulated weight)")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
