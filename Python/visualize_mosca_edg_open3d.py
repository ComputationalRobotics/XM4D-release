#!/usr/bin/env python3
"""Interactive Open3D viewer for MoSca's Embedded Deformation Graph (EDG).

Loads MoSca's `mosca_edg_middle.npz` (edg_node_xyz (T,N,3), edg_node_rotmat (T,N,3,3),
edg_edges (E,2), node_rgb (N,3)) and shows the deformation graph animating over time:
nodes as a point cloud, graph edges as a LineSet, optional per-node orientation triads.

Controls (printed at startup):
  SPACE  play / pause
  . / ,  step one frame forward / back (also pauses)
  D / A  step +10 / -10 frames
  ] / [  faster / slower playback
  R      reset to frame 0
  C      cycle color mode: motion -> node-rgb -> uniform
  Q/ESC  quit

Optionally overlay OUR EDG (--ours-npz results_all.npz), transformed into MoSca's frame
by an inverse Sim(3) (--sim3 npz with s,R,t mapping MoSca->ours) so the two graphs can be
compared side by side (MoSca colored, ours drawn in solid magenta).

Example:
  conda run -n gsplat python visualize_mosca_edg_open3d.py \
    --npz datasets/Dycheck/wheel/outputs_mosca/logs/iphone_fit_colfree_native_add3_20260821_104610/mosca_edg_middle.npz
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import open3d as o3d


def turbo_colors(scalar: np.ndarray, vmax: float | None = None) -> np.ndarray:
    """Map a 1-D scalar to RGB via matplotlib's turbo (falls back to grayscale)."""
    s = np.asarray(scalar, dtype=np.float64)
    vmax = float(np.percentile(s, 95)) if vmax is None else vmax
    t = np.clip(s / (vmax + 1e-12), 0.0, 1.0)
    try:
        import matplotlib.cm as cm
        return cm.turbo(t)[:, :3]
    except Exception:
        return np.stack([t, np.zeros_like(t), 1.0 - t], axis=1)


def load_mosca_edg(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    xyz = d["edg_node_xyz"].astype(np.float64)          # (T,N,3)
    rot = d["edg_node_rotmat"].astype(np.float64) if "edg_node_rotmat" in d.files else None
    edges = d["edg_edges"].astype(np.int64) if "edg_edges" in d.files else np.zeros((0, 2), np.int64)
    rgb = np.clip(d["node_rgb"], 0, 1).astype(np.float64) if "node_rgb" in d.files else None
    disp = np.linalg.norm(xyz - xyz[0:1], axis=2).max(0)  # per-node max displacement
    print(f"[load] {path.name}: T={xyz.shape[0]} nodes={xyz.shape[1]} edges={len(edges)} "
          f"| displacement median={np.median(disp):.4f} max={disp.max():.4f}")
    return dict(xyz=xyz, rot=rot, edges=edges, rgb=rgb, disp=disp)


def load_ours_edg(path: Path) -> np.ndarray | None:
    """Return our node trajectories (T,N,3) from results_all.npz (edg_reconstruct dict)."""
    d = np.load(path, allow_pickle=True)
    if "edg_reconstruct" not in d.files:
        print(f"[ours] {path} has no edg_reconstruct; skipping overlay")
        return None
    rec = d["edg_reconstruct"].item()
    gids = sorted(rec.keys())
    xyz = np.stack([np.asarray(rec[g], dtype=np.float64) for g in gids], axis=1)  # (T,N,3)
    print(f"[ours] {path.name}: T={xyz.shape[0]} nodes={xyz.shape[1]}")
    return xyz


def sim3_inv_apply(pts: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply inverse of Sim(3) (MoSca->ours) to bring OUR pts back into MoSca's frame."""
    return ((pts - t) @ R) / s  # inverse of s*R@x + t  ==  R^T (x - t)/s


def make_axes_lineset(pos: np.ndarray, rot: np.ndarray, length: float) -> o3d.geometry.LineSet:
    """One RGB triad per node: 4 pts (origin,+x,+y,+z), 3 lines, colored r/g/b."""
    n = len(pos)
    pts = np.empty((4 * n, 3)); lines = np.empty((3 * n, 2), np.int64); cols = np.empty((3 * n, 3))
    base = np.eye(3) * length
    for i in range(n):
        o = pos[i]; axes = (rot[i] @ base.T).T  # world-space axis endpoints
        pts[4 * i] = o; pts[4 * i + 1:4 * i + 4] = o + axes
        lines[3 * i:3 * i + 3] = [[4 * i, 4 * i + 1], [4 * i, 4 * i + 2], [4 * i, 4 * i + 3]]
        cols[3 * i:3 * i + 3] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(cols)
    return ls


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", type=Path, required=True, help="MoSca mosca_edg_middle.npz")
    ap.add_argument("--color", choices=["motion", "rgb", "uniform"], default="motion")
    ap.add_argument("--point-size", type=float, default=6.0)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--no-edges", action="store_true")
    ap.add_argument("--node-frames", type=int, default=0,
                    help="draw orientation triads for this many (subsampled) nodes (0=off)")
    ap.add_argument("--triad-scale", type=float, default=0.01)
    ap.add_argument("--ours-npz", type=Path, default=None, help="overlay OUR EDG (results_all.npz)")
    ap.add_argument("--sim3", type=Path, default=None,
                    help="npz with s,R,t (MoSca->ours); needed to place --ours-npz in MoSca frame")
    args = ap.parse_args()

    edg = load_mosca_edg(args.npz)
    XYZ, ROT, EDGES, RGB, DISP = edg["xyz"], edg["rot"], edg["edges"], edg["rgb"], edg["disp"]
    T, N = XYZ.shape[:2]

    # optional overlay of OUR edg in MoSca's frame
    OURS = None
    if args.ours_npz is not None:
        ours = load_ours_edg(args.ours_npz)
        if ours is not None and args.sim3 is not None:
            s = np.load(args.sim3); OURS = sim3_inv_apply(ours.reshape(-1, 3), float(s["s"]), s["R"], s["t"]).reshape(ours.shape)
        elif ours is not None:
            print("[ours] no --sim3 given; drawing ours in its own coords (likely misaligned)")
            OURS = ours

    color_modes = ["motion", "rgb", "uniform"]
    st = {"t": int(args.frame) % T, "play": False, "speed": 1, "last": 0.0,
          "cmode": color_modes.index(args.color)}

    def node_colors() -> np.ndarray:
        m = color_modes[st["cmode"]]
        if m == "motion":
            return turbo_colors(DISP)
        if m == "rgb" and RGB is not None:
            return RGB
        return np.tile([0.1, 0.7, 1.0], (N, 1))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(XYZ[st["t"]])
    pcd.colors = o3d.utility.Vector3dVector(node_colors())

    lineset = None
    if not args.no_edges and len(EDGES):
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(XYZ[st["t"]])
        lineset.lines = o3d.utility.Vector2iVector(EDGES)
        lineset.colors = o3d.utility.Vector3dVector(np.tile([0.6, 0.6, 0.6], (len(EDGES), 1)))

    tri_idx = None
    triads = None
    if args.node_frames > 0 and ROT is not None:
        tri_idx = np.linspace(0, N - 1, min(args.node_frames, N)).astype(int)
        triads = make_axes_lineset(XYZ[st["t"]][tri_idx], ROT[st["t"]][tri_idx], args.triad_scale)

    ours_geoms = []
    if OURS is not None:
        opcd = o3d.geometry.PointCloud()
        opcd.points = o3d.utility.Vector3dVector(OURS[st["t"] % OURS.shape[0]])
        opcd.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0.0, 1.0], (OURS.shape[1], 1)))
        ours_geoms.append(("ours_pcd", opcd))

    def set_frame(t: int) -> None:
        st["t"] = t % T
        pcd.points = o3d.utility.Vector3dVector(XYZ[st["t"]])
        if lineset is not None:
            lineset.points = o3d.utility.Vector3dVector(XYZ[st["t"]])
        if triads is not None:
            nt = make_axes_lineset(XYZ[st["t"]][tri_idx], ROT[st["t"]][tri_idx], args.triad_scale)
            triads.points = nt.points
        for _, g in ours_geoms:
            g.points = o3d.utility.Vector3dVector(OURS[st["t"] % OURS.shape[0]])

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(f"MoSca EDG — {args.npz.name}", 1400, 1000)
    vis.add_geometry(pcd)
    if lineset is not None:
        vis.add_geometry(lineset)
    if triads is not None:
        vis.add_geometry(triads)
    for _, g in ours_geoms:
        vis.add_geometry(g)
    extent = float(np.linalg.norm(XYZ[0].max(0) - XYZ[0].min(0)))
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15 * extent))
    ro = vis.get_render_option(); ro.point_size = args.point_size; ro.background_color = np.array([0.05, 0.05, 0.08])

    def refresh() -> None:
        vis.update_geometry(pcd)
        if lineset is not None:
            vis.update_geometry(lineset)
        if triads is not None:
            vis.update_geometry(triads)
        for _, g in ours_geoms:
            vis.update_geometry(g)

    def recolor() -> None:
        pcd.colors = o3d.utility.Vector3dVector(node_colors())
        vis.update_geometry(pcd)

    def step(d: int, pause: bool = True):
        if pause:
            st["play"] = False
        set_frame(st["t"] + d); refresh(); _title()
        return True

    def _title():
        print(f"\r frame {st['t']:3d}/{T-1}  speed x{st['speed']}  "
              f"color={color_modes[st['cmode']]}  {'PLAY ' if st['play'] else 'PAUSE'}", end="", flush=True)

    def anim(_v):
        if not st["play"]:
            return False
        now = time.time()
        if now - st["last"] < 1.0 / max(args.fps, 1e-3):
            return False
        st["last"] = now
        set_frame(st["t"] + st["speed"]); refresh(); _title()
        return True

    vis.register_key_callback(32, lambda v: (st.update(play=not st["play"]), _title(), False)[-1])   # space
    vis.register_key_callback(ord("."), lambda v: step(+1))
    vis.register_key_callback(ord(","), lambda v: step(-1))
    vis.register_key_callback(ord("D"), lambda v: step(+10))
    vis.register_key_callback(ord("A"), lambda v: step(-10))
    vis.register_key_callback(ord("]"), lambda v: (st.update(speed=min(st["speed"] + 1, 20)), _title(), False)[-1])
    vis.register_key_callback(ord("["), lambda v: (st.update(speed=max(st["speed"] - 1, 1)), _title(), False)[-1])
    vis.register_key_callback(ord("R"), lambda v: step(-st["t"]))
    vis.register_key_callback(ord("C"), lambda v: (st.update(cmode=(st["cmode"] + 1) % 3), recolor(), _title(), False)[-1])
    vis.register_animation_callback(anim)

    print(__doc__.split("Example:")[0].split("Controls")[1])
    if OURS is not None:
        print(" [overlay] MoSca=colored, OURS=magenta")
    _title()
    vis.run()
    vis.destroy_window()
    print()


if __name__ == "__main__":
    main()
