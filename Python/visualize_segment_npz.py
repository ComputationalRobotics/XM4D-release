#!/usr/bin/env python3
"""Replay ONE saved segment solve in Open3D: the raw per-segment result, before any cross-run averaging.

EDG_test.py writes edg_test/segment_<id>.npz for every segment it chains; EDG_test_fast.py writes
edg_test/segments_fast/run<run>_seg<iter>.npz for every segment of every run when given --save-segments.
Both carry the same arrays, so this viewer replays either: red = solved EDG nodes + edges, RGB axes = the
chained global rotation, blue = the deformed dynamic points. SPACE/B step frames, Q/ESC closes.

Run:  conda run -n XM python visualize_segment_npz.py --npz <outputs>/edg_test/segments_fast/run001_seg01.npz
      conda run -n XM python visualize_segment_npz.py --list
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import EDG_test as ET
import utils.definitions


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", type=Path, default=None, help="segment npz (default: the first one found)")
    ap.add_argument("--list", action="store_true", help="list the saved segments and exit")
    ap.add_argument("--node-radius", type=float, default=0.0,
                    help="EDG node sphere radius in solver units (0 = 0.6%% of the segment's bbox diagonal)")
    ap.add_argument("--merged", action="store_true", help="observations from the MERGED tracks (match the solve)")
    ap.add_argument("--tracks", default="", help="explicit tracks npz for the observations (name under "
                                                 "outputs/ or a full path); overrides --merged/--smoothed")
    ap.add_argument("--smoothed", action=argparse.BooleanOptionalAction, default=True,
                    help="--merged: use the smoothed merged tracks (match the solve)")
    ap.add_argument("--tails", action=argparse.BooleanOptionalAction, default=True,
                    help="draw each track's history up to the current frame (T toggles it live)")
    ap.add_argument("--no-obs", action="store_true",
                    help="skip the green raw-observation spheres/tails (no track load)")
    a = ap.parse_args()

    edg_dir = Path(utils.definitions.output_dir) / "edg_test"
    found = sorted(edg_dir.glob("segments_fast/*.npz")) + sorted(edg_dir.glob("segment_*.npz"))
    if a.list or a.npz is None:
        for p in found:
            d = np.load(p)
            f = d["frames"]
            print(f"  {p.relative_to(edg_dir)}  frames {int(f[0])}..{int(f[-1])} ({f.size})  "
                  f"nodes={d['present'].size} edges={len(d['edges'])} dyn={d['dyn_mapped'].shape[0]}")
        if not found:
            print(f"no segment npz under {edg_dir} -- rerun EDG_test_fast.py with --save-segments")
        if a.list:
            return
        if not found:
            return
        a.npz = found[0]
        print(f"[seg] no --npz given, showing {a.npz.name}")

    d = np.load(a.npz)
    present = np.asarray(d["present"], np.int64)
    inp = dict(present=present, frames=np.asarray(d["frames"], np.int64), rest=int(d["rest_frame"]),
               edges=np.asarray(d["edges"], np.int64),
               edg_idx_map={int(g): i for i, g in enumerate(present.tolist())})
    ed = np.asarray(d["edg_deformed"], np.float64)
    node_r = a.node_radius
    if node_r <= 0:
        diag = float(np.linalg.norm(np.nanmax(ed, (0, 1)) - np.nanmin(ed, (0, 1))))
        node_r = max(1e-4, 0.006 * (diag or 1.0))
    obs = None
    if not a.no_obs:
        tracks_path = None
        if a.tracks:
            tracks_path = a.tracks if "/" in a.tracks else \
                str(Path(utils.definitions.output_dir) / a.tracks)
        elif a.merged:
            name = "tracks_dynamic_merged_smoothed.npz" if a.smoothed else "tracks_dynamic_merged.npz"
            tracks_path = str(Path(utils.definitions.output_dir) / name)
        D = ET.load_all(tracks_path=tracks_path)
        frames = inp["frames"]
        obs = D["glob"][present][:, frames, :].copy()
        obs[~D["valid"][present][:, frames]] = np.nan
    ET.visualize_segment(ed, np.asarray(d["R_global"], np.float64), np.asarray(d["dyn_mapped"], np.float64),
                         inp, node_r, a.npz.stem, obs, a.tails)


if __name__ == "__main__":
    main()
