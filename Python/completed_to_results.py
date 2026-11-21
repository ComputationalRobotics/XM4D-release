#!/usr/bin/env python3
"""Convert the NEW-pipeline EDG result (outputs/edg_test/edg_completed.npz, from complete_edg_table.py)
into the results_all.npz format consumed by train_4dgs_edg.py + deps/gsplat simple_trainer:
the dict-style `edg_reconstruct` / `edg_rotation` / `dynamics_points_reconstruct` are REPLACED
from the completed solve; every other key (static_landmarks, prior_*, da3_frame_scale, ...) is carried
over unchanged from a BASE results_all npz (the static/prior side of the pipeline is untouched).

Run:  XM_DATASET_NAME=<scene> conda run -n XM python completed_to_results.py \
          --completed <outputs>/edg_test/edg_completed.npz --base <outputs>/results_all.npz \
          --out <outputs>/results_all_completed.npz
"""
import argparse
from pathlib import Path

import numpy as np

import utils.definitions
from utils.io import savez_compressed_fast


def hold_fill(arr):
    """[T,...] nearest-finite fill (dyn tracks are NaN outside their observed lifetime; the 4DGS prep
    expects finite tracks). Returns (filled, has_any_finite)."""
    flat = arr.reshape(arr.shape[0], -1)
    fin = np.isfinite(flat).all(1)
    idx = np.where(fin)[0]
    if idx.size == 0:
        return arr, False
    if idx.size == arr.shape[0]:
        return arr, True
    out = arr.copy()
    pos = np.arange(arr.shape[0])
    j = np.searchsorted(idx, pos)                                     # idx[j-1] < pos <= idx[j]
    lo, hi = idx[np.maximum(j - 1, 0)], idx[np.minimum(j, idx.size - 1)]
    nearest = np.where(pos - lo <= hi - pos, lo, hi)                  # ties -> the earlier frame, as argmin did
    out[~fin] = arr[nearest[~fin]]
    return out, True


def convert(completed_path, base_path, out_path, comp=None):
    """Write `out_path` = the completed EDG (from edg_completed.npz `completed_path`) in the results_all.npz
    contract, carrying every non-EDG key over from `base_path` unchanged. Safe when out_path == base_path
    (the base is fully materialised into memory before writing). `comp`: the completed arrays when the caller
    still holds them (skips re-reading the file). Returns (n_edg_nodes, n_dyn_tracks)."""
    comp = np.load(str(completed_path)) if comp is None else comp
    node_ids = np.asarray(comp["node_ids"], dtype=int)
    er = np.asarray(comp["edg_reconstruct"], dtype=np.float64)        # [M,T,3]
    ero = np.asarray(comp["edg_rotation"], dtype=np.float64)          # [M,T,3,3]
    dyn_ids = np.asarray(comp["dyn_ids"], dtype=int)
    dyn = np.asarray(comp["dyn_reconstruct"], dtype=np.float64)       # [Nd,T,3]

    edg_tracks, edg_rots = {}, {}
    n_filled = 0
    for i, gid in enumerate(node_ids.tolist()):
        p, okp = hold_fill(er[i])
        r, okr = hold_fill(ero[i])
        if not (okp and okr):
            continue                                                  # node with no finite frame at all
        n_filled += int((~np.isfinite(er[i]).all(-1)).sum())
        edg_tracks[int(gid)] = p
        edg_rots[int(gid)] = r

    dyn_tracks = {}
    for i, gid in enumerate(dyn_ids.tolist()):
        p, ok = hold_fill(dyn[i])
        if ok:
            dyn_tracks[int(gid)] = p

    base = np.load(str(base_path), allow_pickle=True)
    n_base_keys = len(base.files)
    out = {k: base[k] for k in base.files}                            # materialise before any overwrite
    base.close()
    out["edg_reconstruct"] = np.array(edg_tracks, dtype=object)
    out["edg_rotation"] = np.array(edg_rots, dtype=object)
    out["dynamics_points_reconstruct"] = np.array(dyn_tracks, dtype=object)
    if "edg_edges" in comp:                                           # topology edges among the new nodes
        out["edg_edges_union"] = np.asarray(comp["edg_edges"], dtype=np.int64).reshape(-1, 2)

    savez_compressed_fast(str(out_path), **out)
    T = er.shape[1]
    print(f"[convert] {out_path}: {len(edg_tracks)} EDG nodes (hold-filled {n_filled} node-frames), "
          f"{len(dyn_tracks)} dyn tracks, T={T}; base keys carried: "
          f"{n_base_keys - 3} (from {Path(base_path).name})")
    return len(edg_tracks), len(dyn_tracks)


def main():
    out_dir = Path(utils.definitions.output_dir)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--completed", default=str(out_dir / "edg_test" / "edg_completed.npz"))
    ap.add_argument("--base", default=str(out_dir / "results_all.npz"),
                    help="base results_all npz supplying the non-EDG keys (static/prior/da3)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    convert(args.completed, args.base, args.out)


if __name__ == "__main__":
    main()
