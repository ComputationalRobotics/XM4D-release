#!/usr/bin/env python3
"""EDG-ONLY viewer (no static/dyn/GT geometry) with pre-visualization filters on the EDG tracks.

Filter 1 (--smooth-win / --smooth-sigma): temporally smooth each EDG track (NaN-aware moving average) and
measure the per-frame residual ||x[t] - smooth(x)[t]||. The SPECIFIC OBSERVATIONS whose residual is BIG
(r[t] > median + sigma * 1.4826*MAD of all residuals pooled over nodes+frames) are DELETED -- only those
(node, frame) cells, not the whole track.

Filter 2 (--reproj-del-px / --mask-dilate-px / --no-mask-del): based on the 2D PROJECTION of the solved EDG
(inverse XM lift + per-frame intrinsics), a table cell is deleted when its projected pixel is (a) farther
than --reproj-del-px from the node's initial 2D observation at that frame, or (b) OUTSIDE the (dilated)
dynamic mask.

Every deleted observation is shown in the 3D viewer as a BIG SPHERE (RED = filter 1, ORANGE = filter 2),
its previous-frame (pre-filter) position as a BLUE sphere, and a line joins the pair -- so you see exactly
the jump/drift that got the observation deleted. (More filters can be added in apply_filters.)

Run:  XM_DATASET_NAME=<scene> conda run -n XM python visualize_edg_filtered_open3d.py \
          --npz <outputs>/edg_test/edg_completed.npz --no-scale-search
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

import utils.definitions
from edg_viz_common import (_build_dyn_skin, _skin_dyn_frame, detect_render_color_outliers,
                            _load_2d_context, project_to_frame)
from utils.readgt_colmap import load_colmap_gt
from utils.readgt_DAVIS import load_davis_gt
from utils.readgt_sintel import load_sintel_gt


def _npz_has_persistent_skinning(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=True) as data:
            return data.get("dynamic_skinning_nodes_array") is not None
    except Exception:
        return False


def _default_results_path() -> Path:
    output_dir = Path(utils.definitions.output_dir)
    return output_dir / "results_all.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize solved EDG, static landmarks, skinned dynamic points, GT geometry, and cameras."
    )

    parser.add_argument(
        "--npz",
        type=Path,
        default=_default_results_path(),
        help="Path to results_all.npz or results_all_prior.npz produced by EDG.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index to visualize.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=4.0,
        help="Open3D point size.",
    )
    parser.add_argument(
        "--result-xm",
        type=Path,
        default=Path(utils.definitions.output_dir) / "result_XM.npz",
        help="XM pose result file containing R/t/s for alignment.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(utils.definitions.rootpath),
        help="Dataset root containing GT cameras and dynamic/object point clouds.",
    )
    parser.add_argument(
        "--gt-mesh",
        type=Path,
        default=None,
        help="Optional GT mesh/point-cloud file or directory.",
    )
    parser.add_argument(
        "--gt-mesh-voxel-size",
        type=float,
        default=0.0,
        help="Optional voxel downsample size for GT mesh/point-cloud display.",
    )
    parser.add_argument(
        "--gt-mesh-sample-points",
        type=int,
        default=1000,
        help="Number of surface points sampled when a GT triangle mesh is loaded.",
    )
    parser.add_argument(
        "--hide-gt-mesh",
        action="store_true",
        help="Hide aligned GT mesh/point-cloud geometry.",
    )
    parser.add_argument(
        "--coordinate-space",
        choices=("robocasa", "solver"),
        default="robocasa",
        help=(
            "Visualization coordinate convention. 'robocasa' shows geometry in raw "
            "COLMAP/world coordinates like visualize_moving_outputs_open3d.py; "
            "'solver' shows GT transformed into EDG/XM solver coordinates."
        ),
    )

    parser.add_argument(
        "--show-cameras",
        action="store_true",
        help="Show solver and GT camera poses as RGB axes.",
    )
    parser.add_argument(
        "--camera-axis-len",
        type=float,
        default=0.08,
        help="Camera axis length for visualization.",
    )
    parser.add_argument(
        "--edg-edge-radius",
        type=float,
        default=0.0,
        help="Draw EDG node-node edges as CYLINDERS of this world-space radius -- RELIABLE edge thickness "
             "(the legacy Open3D viewer ignores LineSet width). Try ~0.003-0.01. 0 = thin lines.",
    )
    parser.add_argument(
        "--edg-line-width",
        type=float,
        default=0.0,
        help="Set the Open3D render line width (quick, but IGNORED on most builds -- use --edg-edge-radius "
             "for guaranteed thickness). 0 = default.",
    )
    parser.add_argument(
        "--edg-axis-len",
        type=float,
        default=0.02,
        help="Length (solver units) of the RGB orientation triad drawn at each EDG node from its rotation.",
    )
    parser.add_argument(
        "--hide-edg-axes",
        action="store_true",
        help="Do not draw the per-node EDG rotation axes.",
    )
    parser.add_argument(
        "--smooth-win",
        type=int,
        default=9,
        help="Filter 1 window (frames) for the NaN-aware moving-average smoothing of each EDG track; "
             "0 disables the filter.",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=5.0,
        help="Filter 1 threshold: delete a track whose max |raw - smoothed| exceeds "
             "median + sigma * 1.4826*MAD of all residuals (pooled over nodes and frames).",
    )
    parser.add_argument(
        "--deleted-dot-radius",
        type=float,
        default=0.008,
        help="Radius (solver units) of the big red spheres marking DELETED EDG tracks.",
    )
    parser.add_argument(
        "--reproj-del-px",
        type=float,
        default=10.0,
        help="Filter 2 (2D projection): DELETE a table cell when its projected 2D position is farther than "
             "this many px from the node's initial 2D observation at that frame (0 disables this rule).",
    )
    parser.add_argument(
        "--no-mask-del",
        action="store_true",
        help="Filter 2: do NOT delete cells whose projection falls outside the (dilated) dynamic mask.",
    )
    parser.add_argument(
        "--mask-dilate-px",
        type=int,
        default=5,
        help="Filter 2: dilate the dynamic masks by this many px before the outside-mask test.",
    )
    # ---- Filter 3 (photometric render-colour outlier detection on the DYNAMIC tracks) ----
    parser.add_argument(
        "--no-render-outlier",
        action="store_true",
        help="Filter 3: disable the photometric render-colour outlier test on the dynamic point cloud "
             "(otherwise the skinned dynamic points are drawn: normal = blue, flagged OUTLIER = big RED).",
    )
    parser.add_argument(
        "--render-color-space",
        choices=("hue", "bgr"),
        default="hue",
        help="Filter 3: colour space of the reprojection-colour test. 'hue' (HSV, circular distance in "
             "degrees 0..90) is specular-robust on this scene; 'bgr' uses L2 BGR distance (0..441).",
    )
    parser.add_argument(
        "--render-color-thresh",
        type=float,
        default=30.0,
        help="Filter 3: a dynamic track is an OUTLIER when its render-colour score exceeds this. Units match "
             "--render-color-space (hue: degrees 0..90; bgr: L2 BGR 0..441).",
    )
    parser.add_argument(
        "--render-color-quantile",
        type=float,
        default=0.5,
        help="Filter 3: quantile of the per-frame colour drift used as the per-track score (0.5 = robust "
             "median; raise toward 0.75 for a stricter 'drifts on most frames' test).",
    )
    parser.add_argument(
        "--render-min-span",
        type=int,
        default=5,
        help="Filter 3: a track needs at least this many EXTRAPOLATED (non-observed) saturated frames to be "
             "scored; shorter tracks are left as normal (never flagged).",
    )
    parser.add_argument(
        "--render-min-obs",
        type=int,
        default=3,
        help="Filter 3: a track needs at least this many OBSERVED saturated frames to build a reference "
             "colour; otherwise it is left as normal (never flagged).",
    )
    parser.add_argument(
        "--render-patch",
        type=int,
        default=2,
        help="Filter 3: half-width of the (2p+1)x(2p+1) BGR patch median used to denoise the sampled colour "
             "(p=2 -> 5x5). 0 samples a single pixel.",
    )
    parser.add_argument(
        "--render-sat-min",
        type=int,
        default=40,
        help="Filter 3 (hue only): ignore frames whose sampled saturation is below this (specular/white "
             "pixels have unstable hue).",
    )
    parser.add_argument(
        "--render-dot-radius",
        type=float,
        default=0.01,
        help="Radius (solver units) of the big RED spheres marking dynamic render-colour OUTLIER points.",
    )
    parser.add_argument(
        "--edg-quality",
        action="store_true",
        help="Open a SEPARATE clean Open3D view showing ONLY the EDG, with each node classified by the "
             "skinning-weighted fraction of its attached dynamic points that are render-colour OUTLIERS: "
             "GREEN sphere = nearly all dynamic points good (frac <= --edg-good-frac), RED big sphere = "
             "large portion bad (frac >= --edg-bad-frac), ORANGE = in between, grey = no dynamic points. "
             "Nothing else is drawn (no dynamic cloud / deleted markers / axes). Off = the normal display.",
    )
    parser.add_argument(
        "--edg-good-frac",
        type=float,
        default=0.1,
        help="(--edg-quality) a node is GOOD when its weighted bad-dynamic fraction is <= this.",
    )
    parser.add_argument(
        "--edg-bad-frac",
        type=float,
        default=0.3,
        help="(--edg-quality) a node is BAD when its weighted bad-dynamic fraction is >= this.",
    )
    parser.add_argument(
        "--edg-only",
        action="store_true",
        help="Force EDG-only display: hide the dynamic point cloud and skip the render-colour outlier pass, "
             "overriding the automatic completed-npz detection.",
    )
    parser.add_argument(
        "--with-dyn",
        action="store_true",
        help="Force the dynamic point cloud + render-colour outlier pass ON, overriding the automatic "
             "completed-npz detection (which otherwise shows the dynamic cloud only for edg_completed.npz).",
    )
    parser.add_argument(
        "--edg-solo",
        action="store_true",
        help="Show ONLY the EDG (nodes + edges + orientation axes): a clean independent view that skips "
             "the dynamic cloud, deleted-observation markers, static landmarks, GT mesh, and cameras "
             "(like --edg-quality but with normal red nodes instead of quality colouring).",
    )
    parser.add_argument(
        "--init-first-cam",
        action="store_true",
        help="Initialize the Open3D viewpoint at the solver's first-frame camera pose (needs "
             "--result-xm; set XM_DATASET_NAME so it points at this scene). Falls back to the default "
             "fit-to-geometry view if the camera pose is unavailable.",
    )
    parser.add_argument(
        "--merged",
        action="store_true",
        help="The npz was solved on the MERGED tracks; load those for the 2D observations (filter 2 / "
             "dynamic-point context).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build one frame of geometry and print counts without opening Open3D.",
    )
    parser.add_argument(
        "--save-frame",
        type=Path,
        default=None,
        help="Headless: render the single --frame to this PNG via Open3D offscreen (EGL) and exit, instead "
             "of opening an interactive window. Useful to eyeball the dynamic outliers without a display.",
    )
    parser.add_argument(
        "--fill-edg",
        action="store_true",
        help="(EDG_test format) HOLD each EDG node across the whole interval (back/forward-fill holes with "
             "its nearest solved pose). Off by default: a node that is a HOLE at a frame is not drawn, so "
             "holey tables (EDG_test_fast.py) show empty where nothing was solved.",
    )

    parser.add_argument(
        "--scale-search",
        action="store_true",
        default=True,
        help="Optimize GT-to-solver scale by Chamfer loss between dynamic points and GT geometry.",
    )
    parser.add_argument(
        "--no-scale-search",
        dest="scale_search",
        action="store_false",
        help="Disable Chamfer scale search and use scale=1.0 after first-camera alignment.",
    )
    parser.add_argument(
        "--scale-min",
        type=float,
        default=0.2,
        help="Minimum scale for Chamfer line search.",
    )
    parser.add_argument(
        "--scale-max",
        type=float,
        default=5.0,
        help="Maximum scale for Chamfer line search.",
    )
    parser.add_argument(
        "--scale-steps",
        type=int,
        default=21,
        help="Number of scale candidates for Chamfer line search.",
    )
    parser.add_argument(
        "--scale-iters",
        type=int,
        default=3,
        help="Number of adaptive scale line-search refinement passes.",
    )
    parser.add_argument(
        "--scale-search-frame",
        type=int,
        default=None,
        help="Frame used for Chamfer scale search. Default: same as --frame.",
    )
    parser.add_argument(
        "--chamfer-max-points",
        type=int,
        default=20000,
        help="Maximum points used from each cloud during Chamfer scale search.",
    )

    return parser.parse_args()


def _load_dict(data, key):
    value = data.get(key)
    if value is None:
        return None
    return value.item()


def _load_tracks(track_dict, frame_id: int):
    points = []
    gids = []

    for key, arr in track_dict.items():
        gid = int(key)
        arr = np.asarray(arr, dtype=np.float64)

        if frame_id < 0 or frame_id >= arr.shape[0]:
            continue

        pt = arr[frame_id]
        if np.isfinite(pt).all():
            gids.append(gid)
            points.append(pt)

    if not points:
        return np.zeros((0, 3), dtype=np.float64), []

    return np.asarray(points, dtype=np.float64), gids


def _load_edges(edges_obj):
    if edges_obj is None:
        return []

    if isinstance(edges_obj, (list, tuple, set)):
        iterable = edges_obj
    else:
        iterable = np.asarray(edges_obj, dtype=object).tolist()

    edges = []
    for item in iterable:
        if item is None:
            continue

        a, b = item
        a = int(a)
        b = int(b)

        if a != b:
            edges.append((min(a, b), max(a, b)))

    return sorted(set(edges))


def _hold_fill_stack(arr: np.ndarray) -> np.ndarray:
    """Complete each track [T,...] over the whole interval so it is finite at every frame: hold the first
    finite value backward, the last finite value forward, and carry the last finite across interior gaps.
    (EDG_test solves a node forward from its first appearance, so this mainly back-fills the pre-appearance
    frames -> the WHOLE EDG shows across the whole time interval.)"""
    arr = np.asarray(arr, dtype=np.float64).copy()
    N, T = arr.shape[0], arr.shape[1]
    flat = arr.reshape(N, T, -1)
    for i in range(N):
        fin = np.isfinite(flat[i]).all(axis=1)
        idx = np.where(fin)[0]
        if idx.size == 0 or idx.size == T:
            continue
        arr[i, : idx[0]] = arr[i, idx[0]]
        arr[i, idx[-1] + 1:] = arr[i, idx[-1]]
        last = idx[0]
        for t in range(idx[0] + 1, idx[-1] + 1):
            if fin[t]:
                last = t
            else:
                arr[i, t] = arr[i, last]
    return arr


# ------------------------------------------------------------------- pre-visualization EDG track filters
def _smooth_traj_nan(arr: np.ndarray, win: int) -> np.ndarray:
    """NaN-aware centred moving average of a [T,3] trajectory: each finite frame gets the mean of the
    finite samples in its window; NaN frames stay NaN."""
    T = arr.shape[0]
    half = max(1, int(win)) // 2
    fin = np.isfinite(arr).all(axis=1)
    vals = np.where(fin[:, None], arr, 0.0)
    kernel = np.ones(2 * half + 1)
    num = np.stack([np.convolve(vals[:, d], kernel, mode="same") for d in range(3)], axis=1)
    cnt = np.convolve(fin.astype(np.float64), kernel, mode="same")
    out = np.full_like(arr, np.nan)
    ok = fin & (cnt > 0)
    out[ok] = num[ok] / cnt[ok, None]
    return out


def filter_smooth_error(edg_tracks: dict, win: int, sigma: float):
    """Filter 1: delete the SPECIFIC OBSERVATIONS (frames) where a track disagrees with its own smoothed
    trajectory -- NOT the whole track. Per cell, residual r[t] = ||x[t] - smooth(x)[t]||; a cell is deleted
    if r[t] > median + sigma * 1.4826*MAD of the residuals pooled over ALL tracks and frames. Returns
    {gid: bool mask [T]} of the cells to delete (tracks with no flagged cell are absent)."""
    res = {}
    for gid, arr in edg_tracks.items():
        arr = np.asarray(arr, dtype=np.float64)
        fin = np.isfinite(arr).all(axis=1)
        if fin.sum() < max(3, win):
            continue                                                  # too short to judge -> keep as is
        res[gid] = np.linalg.norm(arr - _smooth_traj_nan(arr, win), axis=1)   # NaN at holes
    if not res:
        return {}
    pooled = np.concatenate([r[np.isfinite(r)] for r in res.values()])
    med = float(np.median(pooled))
    mad = float(np.median(np.abs(pooled - med)))
    thresh = med + float(sigma) * 1.4826 * mad
    bad = {}
    for gid, r in res.items():
        mask = np.isfinite(r) & (r > thresh)
        if mask.any():
            bad[gid] = mask
    n_cells = int(sum(m.sum() for m in bad.values()))
    n_total = int(sum(np.isfinite(r).sum() for r in res.values()))
    print(f"[filter1 smooth-error] win={win} sigma={sigma}: per-frame residual median={med:.5f} "
          f"MAD={mad:.5f} -> threshold={thresh:.5f}; deleting {n_cells}/{n_total} observations "
          f"across {len(bad)}/{len(edg_tracks)} tracks")
    worst = sorted(((float(np.nanmax(res[gid][bad[gid]])), gid, int(bad[gid].sum())) for gid in bad),
                   reverse=True)[:8]
    for mx, gid, n in worst:
        print(f"    gid={gid}: {n} frames deleted, worst residual {mx:.5f} ({mx / max(thresh, 1e-12):.1f}x)")
    return bad


def _load_dyn_mask_images(ctx: dict, dilate_px: int):
    """(T,H,W) bool dynamic-mask images (dilated), or None when unavailable (filter falls back gracefully)."""
    mask_dir = getattr(utils.definitions, "mask_dir", None)
    if mask_dir is None or not Path(str(mask_dir)).exists():
        print(f"[filter2] no mask dir ({mask_dir}); skipping the outside-mask deletion")
        return None
    try:
        import cv2
        from utils.track_tapnet import load_dynamic_masks, dilate_dynamic_masks
        H, W = cv2.imread(ctx["image_paths"][0]).shape[:2]
        masks = load_dynamic_masks(str(mask_dir), utils.definitions.mask_pattern,
                                   t_expected=ctx["T"], mask_mode="auto", target_size=(W, H))
        if dilate_px > 0:
            masks = dilate_dynamic_masks(masks, int(dilate_px))
        masks = np.asarray(masks).astype(bool)
        print(f"[filter2] loaded {masks.shape[0]} dynamic masks ({masks.shape[2]}x{masks.shape[1]}, "
              f"dilated {dilate_px}px)")
        return masks
    except Exception as e:
        print(f"[filter2] cannot load dynamic masks ({e}); skipping the outside-mask deletion")
        return None


def filter_reproj_2d(edg_tracks: dict, ctx: dict, masks, del_px: float):
    """Filter 2: delete SPECIFIC table cells based on the 2D PROJECTION of the solved EDG: a cell is deleted
    when (a) its projected pixel is farther than `del_px` from the node's initial 2D observation at that
    frame (where the solver had an observation), or (b) its projected pixel falls OUTSIDE the (dilated)
    dynamic mask. Returns {gid: bool mask [T]} of cells to delete."""
    gids = sorted(int(g) for g in edg_tracks.keys())
    if not gids:
        return {}
    P = np.stack([np.asarray(edg_tracks[g], np.float64) for g in gids])   # [N,T',3]
    T = min(ctx["T"], P.shape[1])
    gid_arr = np.asarray(gids, int)
    bad = np.zeros((len(gids), P.shape[1]), bool)
    n_far = n_out = 0
    for f in range(T):
        u, v, solved = project_to_frame(P[:, f], ctx, f)               # R^T (x - t) / s, row form + pinhole
        if not solved.any():
            continue
        if del_px > 0:
            obs = ctx["xy"][gid_arr, f]
            ok_o = ctx["valid"][gid_arr, f] & np.isfinite(obs).all(1)
            far = solved & ok_o & (np.hypot(u - obs[:, 0], v - obs[:, 1]) > del_px)
            n_far += int(far.sum())
            bad[:, f] |= far
        if masks is not None:
            Hm, Wm = masks.shape[1], masks.shape[2]
            ui = np.round(np.nan_to_num(u, nan=-1.0)).astype(int)     # NaN (unsolved) -> out of bounds
            vi = np.round(np.nan_to_num(v, nan=-1.0)).astype(int)
            inb = solved & (ui >= 0) & (ui < Wm) & (vi >= 0) & (vi < Hm)
            mv = masks[f][np.clip(vi, 0, Hm - 1), np.clip(ui, 0, Wm - 1)]
            out = solved & ~(inb & mv)                                # out of bounds OR outside the mask
            n_out += int(out.sum())
            bad[:, f] |= out
    tot = int(np.isfinite(P).all(-1).sum())
    print(f"[filter2 reproj-2d] deleting {int(bad.sum())}/{tot} solved cells "
          f"({n_far} farther than {del_px:.0f}px from the observation, {n_out} outside the dynamic mask"
          f"{'; masks unavailable' if masks is None else ''})")
    return {int(gid_arr[i]): bad[i] for i in range(len(gids)) if bad[i].any()}


def apply_filters(edg_tracks: dict, edg_rot_tracks: dict, args, ctx=None, masks=None):
    """Run all filters. Flagged (track, frame) cells are set to NaN in the EDG dicts (so those observations
    vanish from points/edges/axes). Returns {group: (deleted, deleted_prev)} with {gid: [T,3]} finite ONLY
    at the deleted cells -- shown in the 3D viewer as big marker spheres (red = filter 1 smooth-error,
    ORANGE = filter 2 2D-projection) with the pre-filter t-1 position in blue and a joining line."""
    def _cut(mask_by_gid):
        deleted, prev = {}, {}
        for gid, mask in mask_by_gid.items():
            arr = np.asarray(edg_tracks[gid], dtype=np.float64).copy()
            rem = np.full_like(arr, np.nan)
            rem[mask] = arr[mask]                                     # the deleted observations (markers)
            prv = np.full_like(arr, np.nan)                           # previous-frame (pre-filter) positions
            idx = np.where(mask)[0]
            idx = idx[idx > 0]
            prv[idx] = arr[idx - 1]                                   # may be NaN if t-1 is a hole -> skipped
            arr[mask] = np.nan                                        # delete just those cells
            edg_tracks[gid] = arr
            deleted[gid] = rem
            prev[gid] = prv
        return deleted, prev

    groups = {}
    if args.smooth_win > 0:
        groups["smooth"] = _cut(filter_smooth_error(edg_tracks, args.smooth_win, args.smooth_sigma))
    if ctx is not None and (args.reproj_del_px > 0 or masks is not None):
        groups["reproj2d"] = _cut(filter_reproj_2d(edg_tracks, ctx, masks, args.reproj_del_px))
    if not any(d for d, _ in groups.values()):
        print("[filter] no EDG observations deleted")
    return groups


def _spheres_mesh(centers: np.ndarray, radius: float, color) -> "o3d.geometry.TriangleMesh":
    """One TriangleMesh of small spheres at `centers` (big-dot markers, independent of the global point size)."""
    mesh = o3d.geometry.TriangleMesh()
    for c in np.asarray(centers, dtype=np.float64).reshape(-1, 3):
        s = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius), resolution=8)
        s.translate(c)
        mesh += s
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def _edges_cylinders(points: np.ndarray, line_idx, radius: float, color, colors=None):
    """EDG edges as a merged TriangleMesh of thin CYLINDERS -- gives RELIABLE edge thickness, unlike a
    LineSet whose line width the legacy Open3D viewer ignores on most builds. `points` are the node
    positions, `line_idx` the (a,b) index pairs, `radius` the world-space cylinder radius. If `colors` is
    given (one RGB per edge, 0..1) each cylinder is painted individually (e.g. colour edges by max distance)."""
    points = np.asarray(points, np.float64).reshape(-1, 3)
    mesh = o3d.geometry.TriangleMesh()
    z = np.array([0.0, 0.0, 1.0])
    for e, (a, b) in enumerate(line_idx):
        p0, p1 = points[int(a)], points[int(b)]
        d = p1 - p0
        L = float(np.linalg.norm(d))
        if L < 1e-9:
            continue
        cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=float(radius), height=L, resolution=6, split=1)
        dz = d / L                                          # rotate +Z -> edge direction (Rodrigues)
        axis = np.cross(z, dz); s = float(np.linalg.norm(axis)); c = float(np.dot(z, dz))
        if s < 1e-9:
            R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
        else:
            k = axis / s
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            R = np.eye(3) + s * K + (1.0 - c) * (K @ K)
        cyl.rotate(R, center=(0, 0, 0))
        cyl.translate((p0 + p1) / 2.0)
        if colors is not None:
            cyl.paint_uniform_color([float(x) for x in colors[e]])   # per-edge colour (kept through +=)
        mesh += cyl
    if len(mesh.vertices) == 0:
        return None
    if colors is None:
        mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def _edges_from_topology(node_ids: np.ndarray):
    """Fallback EDG edges for EDG_test files that predate the saved `edg_edges`: read the global topology
    from output_dir/edg_topology.npz and keep pairs whose endpoints are both in node_ids."""
    p = Path(utils.definitions.output_dir) / "edg_topology.npz"
    if not p.exists():
        return None
    topo = np.load(p)
    if "edg_edges" not in topo.files:
        return None
    ns = {int(x) for x in np.asarray(node_ids).reshape(-1).tolist()}
    ee = np.asarray(topo["edg_edges"], dtype=np.int64).reshape(-1, 2)
    keep = [[a, b] for a, b in ee.tolist() if a in ns and b in ns]
    return np.asarray(keep, dtype=np.int64).reshape(-1, 2)


def _dynamic_point_at_frame(
    *,
    gid: int,
    frame_id: int,
    skin_nodes: np.ndarray,
    skin_weights: np.ndarray,
    bind_rest: np.ndarray,
    bind_frame: int,
    edg_tracks,
    edg_rot_tracks,
):
    if bind_frame < 0 or not np.isfinite(bind_rest).all():
        return None

    accum = np.zeros(3, dtype=np.float64)
    w_accum = 0.0

    for node_gid, weight in zip(skin_nodes.tolist(), skin_weights.tolist()):
        node_gid = int(node_gid)
        weight = float(weight)

        if node_gid < 0 or not np.isfinite(weight) or weight <= 0.0:
            continue

        if node_gid not in edg_tracks or node_gid not in edg_rot_tracks:
            continue                                                  # e.g. the node was deleted by a filter

        node_track = np.asarray(edg_tracks[node_gid], dtype=np.float64)
        node_rot = np.asarray(edg_rot_tracks[node_gid], dtype=np.float64)

        if (
            bind_frame >= node_track.shape[0]
            or frame_id >= node_track.shape[0]
            or bind_frame >= node_rot.shape[0]
            or frame_id >= node_rot.shape[0]
        ):
            continue

        x_bind = node_track[bind_frame]
        x_frame = node_track[frame_id]
        r_bind = node_rot[bind_frame]
        r_frame = node_rot[frame_id]

        if not (
            np.isfinite(x_bind).all()
            and np.isfinite(x_frame).all()
            and np.isfinite(r_bind).all()
            and np.isfinite(r_frame).all()
        ):
            continue

        rel = r_bind.T @ (bind_rest - x_bind)
        mapped = x_frame + r_frame @ rel

        accum += weight * mapped
        w_accum += weight

    if w_accum <= 1e-12:
        return None

    return accum / w_accum


def _load_dynamic_points(frame_id: int, data, edg_tracks, edg_rot_tracks):
    dynamic_global_ids = np.asarray(data.get("dynamic_global_ids"), dtype=np.int64)
    skin_nodes = np.asarray(data.get("dynamic_skinning_nodes_array"), dtype=np.int64)
    skin_weights = np.asarray(data.get("dynamic_skinning_weights_array"), dtype=np.float64)
    bind_rest = np.asarray(data.get("dynamic_skinning_rest_array"), dtype=np.float64)
    bind_frame = np.asarray(data.get("dynamic_skinning_bind_frame_array"), dtype=np.int64)
    assigned = np.asarray(data.get("dynamic_skinning_assigned_mask"), dtype=bool)

    if dynamic_global_ids.size == 0:
        return np.zeros((0, 3), dtype=np.float64), 0, 0, 0, 0

    points = []
    skinned_now = 0
    solver_now = 0
    bind_rest_now = 0
    bound_total = int(np.sum(assigned))

    for gid in dynamic_global_ids.tolist():
        gid = int(gid)

        if gid >= assigned.shape[0] or not bool(assigned[gid]):
            continue

        pt = _dynamic_point_at_frame(
            gid=gid,
            frame_id=frame_id,
            skin_nodes=skin_nodes[gid],
            skin_weights=skin_weights[gid],
            bind_rest=bind_rest[gid],
            bind_frame=int(bind_frame[gid]),
            edg_tracks=edg_tracks,
            edg_rot_tracks=edg_rot_tracks,
        )

        if pt is not None:
            points.append(pt)
            skinned_now += 1

    if not points:
        return np.zeros((0, 3), dtype=np.float64), skinned_now, solver_now, bind_rest_now, bound_total

    return np.asarray(points, dtype=np.float64), skinned_now, solver_now, bind_rest_now, bound_total


def _normalize_rotation(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64))
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


def _load_solved_c2w(result_xm: Path) -> np.ndarray:
    """
    Load solver/XM camera pose as c2w.

    Assumption:
        result_XM.npz stores R[:, 3*i:3*i+3] as c2w rotation up to optional scale,
        and t[:, i] as camera center / translation in solver world.
    """
    rt = np.load(result_xm, allow_pickle=True)

    R_raw = np.asarray(rt["R"], dtype=np.float64)
    t_raw = np.asarray(rt["t"], dtype=np.float64)
    s_raw = np.asarray(rt["s"], dtype=np.float64) if "s" in rt else None

    T = int(t_raw.shape[1])
    c2w = np.tile(np.eye(4, dtype=np.float64), (T, 1, 1))

    for i in range(T):
        R = R_raw[:, 3 * i : 3 * (i + 1)]

        if s_raw is not None and i < len(s_raw) and abs(float(s_raw[i])) > 1e-12:
            R = R / float(s_raw[i])

        c2w[i, :3, :3] = _normalize_rotation(R)
        c2w[i, :3, 3] = t_raw[:, i]

    return c2w


def _load_gt_c2w(dataset_root: Path) -> np.ndarray:
    dataset_cat = utils.definitions.dataset_cat

    if dataset_cat in {"Sintel", "Sintel_debug"}:
        gt = load_sintel_gt(str(dataset_root))
    elif dataset_cat == "DAVIS":
        gt = load_davis_gt(str(dataset_root))
    else:
        gt = load_colmap_gt(str(dataset_root / "colmap"))

    items = sorted(gt.items(), key=lambda kv: int(kv[1]["id"]))

    c2w = []
    for _, item in items:
        R_raw = item.get("R")
        t_raw = item.get("t")
        if R_raw is None or t_raw is None:
            continue

        R_w2c = np.asarray(R_raw, dtype=np.float64)
        t_w2c = np.asarray(t_raw, dtype=np.float64)
        if R_w2c.size != 9 or t_w2c.size != 3:
            continue

        R_w2c = R_w2c.reshape(3, 3)
        t_w2c = t_w2c.reshape(3)
        if not (np.isfinite(R_w2c).all() and np.isfinite(t_w2c).all()):
            continue

        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = R_w2c.T
        pose[:3, 3] = -R_w2c.T @ t_w2c
        c2w.append(pose)

    if not c2w:
        raise RuntimeError(f"No GT camera poses found under {dataset_root}")

    return np.stack(c2w, axis=0)


def _apply_similarity(points: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return float(scale) * (points @ R.T) + np.asarray(t, dtype=np.float64).reshape(1, 3)


def _apply_inverse_similarity(points: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return (points - np.asarray(t, dtype=np.float64).reshape(1, 3)) @ np.asarray(R, dtype=np.float64) / float(scale)


def _first_camera_aligned_similarity(
    solver_c2w: np.ndarray,
    gt_c2w: np.ndarray,
    scale: float,
):
    """
    Build similarity that maps GT/world geometry into solver/EDG world:

        x_solver = scale * R_align @ x_gt + t_align

    First-camera alignment constraints:
        R_align @ R_gt0 = R_solver0
        scale * R_align @ c_gt0 + t_align = c_solver0
    """
    R_solver = solver_c2w[0, :3, :3]
    c_solver = solver_c2w[0, :3, 3]

    R_gt = gt_c2w[0, :3, :3]
    c_gt = gt_c2w[0, :3, 3]

    R_align = _normalize_rotation(R_solver @ R_gt.T)
    t_align = c_solver - float(scale) * (R_align @ c_gt)

    return float(scale), R_align, t_align


def _load_geometry_points(path: Path, sample_points: int, voxel_size: float) -> np.ndarray:
    if not path.exists():
        return np.zeros((0, 3), dtype=np.float64)

    pts = np.zeros((0, 3), dtype=np.float64)
    mesh = None

    if path.suffix.lower() in {".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb"}:
        pcd = o3d.io.read_point_cloud(str(path))
        pts = np.asarray(pcd.points, dtype=np.float64)

    if pts.size == 0:
        mesh = o3d.io.read_triangle_mesh(str(path))

    if mesh is not None and len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        mesh.compute_vertex_normals()

        if sample_points > 0:
            pcd = mesh.sample_points_uniformly(number_of_points=int(sample_points))
            pts = np.asarray(pcd.points, dtype=np.float64)
        else:
            pts = np.asarray(mesh.vertices, dtype=np.float64)

    elif pts.size == 0:
        pcd = o3d.io.read_point_cloud(str(path))
        pts = np.asarray(pcd.points, dtype=np.float64)

    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.float64)

    pts = pts[np.isfinite(pts).all(axis=1)]

    if voxel_size > 0.0 and pts.shape[0] > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd = pcd.voxel_down_sample(float(voxel_size))
        pts = np.asarray(pcd.points, dtype=np.float64)

    return pts


def _resolve_gt_geometry_paths(gt_mesh: Path | None, dataset_root: Path, T: int):
    root = gt_mesh.resolve() if gt_mesh is not None else None

    if root is None:
        for auto_dir_name in ("dynamic_pointcloud_world", "object_pointcloud_world"):
            auto_dir = dataset_root / auto_dir_name
            if auto_dir.exists():
                root = auto_dir
                print(f"[gt] using {auto_dir_name} for GT object point clouds.")
                break

    if root is None:
        return None

    if root.is_file():
        return [root for _ in range(T)]

    if not root.is_dir():
        print(f"[warning] GT mesh path does not exist: {root}")
        return None

    paths = []
    for frame in range(T):
        candidates = [
            root / f"frame{frame:06d}.ply",
            root / f"frame{frame:06d}.obj",
            root / f"{frame:06d}.ply",
            root / f"{frame:06d}.obj",
        ]
        paths.append(next((p for p in candidates if p.exists()), None))

    if not any(p is not None for p in paths):
        print(f"[warning] no GT mesh/point-cloud files found in {root}")
        return None

    return paths


def _subsample_points(points: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]

    if max_points <= 0 or points.shape[0] <= max_points:
        return points

    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=int(max_points), replace=False)
    return points[idx]


def _point_cloud_std(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] < 2:
        return float("nan")
    centered = points - np.mean(points, axis=0, keepdims=True)
    return float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))


def _symmetric_chamfer(src: np.ndarray, dst: np.ndarray) -> float:
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)

    if src.shape[0] == 0 or dst.shape[0] == 0:
        return np.inf

    src_tree = cKDTree(src)
    dst_tree = cKDTree(dst)

    d_src_to_dst, _ = dst_tree.query(src, k=1)
    d_dst_to_src, _ = src_tree.query(dst, k=1)

    return float(np.mean(d_src_to_dst ** 2) + np.mean(d_dst_to_src ** 2))


def _normalized_chamfer(raw: float, gt_std: float) -> float:
    if not np.isfinite(raw) or not np.isfinite(gt_std) or gt_std <= 1e-12:
        return raw
    return float(raw / gt_std)


def _line_search_gt_to_dynamic_scale(
    *,
    dyn_pts_solver: np.ndarray,
    gt_pts: np.ndarray,
    solver_c2w: np.ndarray,
    gt_c2w: np.ndarray,
    scale_min: float,
    scale_max: float,
    scale_steps: int,
    scale_iters: int,
    max_points: int,
):
    """
    Search for scale in:

        x_solver = scale * R_align @ x_gt + t_align(scale)

    R_align is fixed by first-camera orientation.
    t_align changes with scale so the first GT camera center remains aligned
    with the first solver camera center.
    """
    dyn_pts_solver = _subsample_points(dyn_pts_solver, max_points, seed=0)
    gt_pts = _subsample_points(gt_pts, max_points, seed=1)

    if dyn_pts_solver.shape[0] == 0 or gt_pts.shape[0] == 0:
        print("[scale-search] skipped: empty dynamic or GT point cloud.")
        return None

    scale_steps = max(2, int(scale_steps))
    scale_iters = max(1, int(scale_iters))
    lo = min(float(scale_min), float(scale_max))
    hi = max(float(scale_min), float(scale_max))
    if lo <= 0.0:
        lo = 1e-6

    gt_std = _point_cloud_std(gt_pts)
    best = None

    for pass_idx in range(scale_iters):
        scales = np.geomspace(lo, hi, num=scale_steps) if lo < hi else np.asarray([lo])
        if lo <= 1.0 <= hi:
            scales = np.unique(np.concatenate([scales, np.asarray([1.0])]))

        losses = []
        for scale in scales.tolist():
            s, R, t = _first_camera_aligned_similarity(
                solver_c2w,
                gt_c2w,
                scale=float(scale),
            )

            gt_aligned = _apply_similarity(gt_pts, s, R, t)
            raw_loss = _symmetric_chamfer(dyn_pts_solver, gt_aligned)
            norm_loss = _normalized_chamfer(raw_loss, gt_std)
            loss_for_search = norm_loss if np.isfinite(norm_loss) else raw_loss

            if np.isfinite(loss_for_search):
                losses.append((s, raw_loss, norm_loss, R, t))
                if best is None or loss_for_search < best[0]:
                    best = (loss_for_search, raw_loss, norm_loss, s, R, t)

        if not losses:
            break

        losses.sort(key=lambda item: item[0])
        best_idx = min(
            range(len(losses)),
            key=lambda idx: losses[idx][2] if np.isfinite(losses[idx][2]) else losses[idx][1],
        )
        lo = losses[max(best_idx - 1, 0)][0]
        hi = losses[min(best_idx + 1, len(losses) - 1)][0]

        print(
            f"[scale-search] pass={pass_idx + 1}/{scale_iters} "
            f"best_scale={best[3]:.8f} "
            f"raw_chamfer={best[1]:.8f} "
            f"norm_chamfer={best[2]:.8f} "
            f"next_range=[{lo:.8f}, {hi:.8f}]"
        )

    if best is None:
        return None

    _, raw_loss, norm_loss, s, R, t = best

    print(
        f"[scale-search] best scale={s:.6f}, "
        f"raw_chamfer={raw_loss:.8f}, "
        f"norm_chamfer={norm_loss:.8f}, "
        f"gt_std={gt_std:.8f}, "
        f"dynamic_pts={dyn_pts_solver.shape[0]}, "
        f"gt_pts={gt_pts.shape[0]}"
    )

    return s, R, t, raw_loss, norm_loss


def _camera_axes_lines(
    c2w_pose: np.ndarray,
    axis_len: float,
):
    pose = np.asarray(c2w_pose, dtype=np.float64).reshape(4, 4)

    center = pose[:3, 3]
    R = pose[:3, :3]

    pts = np.vstack(
        [
            center,
            center + float(axis_len) * R[:, 0],
            center + float(axis_len) * R[:, 1],
            center + float(axis_len) * R[:, 2],
        ]
    )

    lines = np.asarray(
        [
            [0, 1],
            [0, 2],
            [0, 3],
        ],
        dtype=np.int32,
    )

    colors = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(pts)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def _edg_axes_lineset(centers_solver, gids, frame_id, edg_rot_tracks, axis_len, to_vis):
    """One LineSet of RGB orientation triads, one per EDG node: from the node's rotation R at this frame draw
    center -> center + axis_len * R[:,k] for k = x(red), y(green), z(blue). Built in SOLVER coords then mapped
    to vis coords together with the centers, so the axes stay consistent with the point cloud in either space."""
    if edg_rot_tracks is None:
        return None
    pts, lines, colors = [], [], []
    axis_c = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    base = 0
    for i, gid in enumerate(gids):
        R = edg_rot_tracks.get(int(gid))
        if R is None:
            continue
        R = np.asarray(R, dtype=np.float64)
        if frame_id < 0 or frame_id >= R.shape[0]:
            continue
        Rf = R[frame_id]
        if not np.isfinite(Rf).all():
            continue
        c = np.asarray(centers_solver[i], dtype=np.float64)
        pts.append(c)
        for k in range(3):
            pts.append(c + float(axis_len) * Rf[:, k])
            lines.append([base, base + 1 + k])
            colors.append(axis_c[k])
        base += 4

    if not lines:
        return None

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(to_vis(np.asarray(pts, dtype=np.float64)))
    ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return ls


def _transform_c2w_pose_by_similarity(
    c2w_pose: np.ndarray,
    scale: float,
    R_align: np.ndarray,
    t_align: np.ndarray,
):
    """
    Transform a c2w camera pose by x_new = scale * R_align @ x_old + t_align.

    Rotation is only rotated by R_align.
    Translation is scaled + rotated + translated.
    """
    pose = np.asarray(c2w_pose, dtype=np.float64).copy()
    pose[:3, :3] = _normalize_rotation(R_align @ pose[:3, :3])
    pose[:3, 3] = _apply_similarity(
        pose[:3, 3].reshape(1, 3),
        scale,
        R_align,
        t_align,
    ).reshape(3)
    return pose


def _transform_c2w_pose_by_inverse_similarity(
    c2w_pose: np.ndarray,
    scale: float,
    R_align: np.ndarray,
    t_align: np.ndarray,
):
    """
    Transform a c2w camera pose by inverse of x_new = scale * R_align @ x_old + t_align.

    This maps a solver-space camera back into raw GT/COLMAP world coordinates.
    """
    pose = np.asarray(c2w_pose, dtype=np.float64).copy()
    pose[:3, :3] = _normalize_rotation(R_align.T @ pose[:3, :3])
    pose[:3, 3] = _apply_inverse_similarity(
        pose[:3, 3].reshape(1, 3),
        scale,
        R_align,
        t_align,
    ).reshape(3)
    return pose


def main() -> None:
    args = parse_args()

    npz_path = args.npz.resolve()
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing results file: {npz_path}")

    print(f"Loading EDG result: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    # ADAPTIVE dynamic display: the dynamic point cloud + render-colour outlier pass are shown ONLY for the
    # COMPLETED table (edg_completed.npz), detected by "completed" in the npz filename (case-insensitive).
    # after-XM edg_global.npz (or any non-"completed" npz) -> EDG ONLY. --edg-only / --with-dyn override it.
    show_dyn = "completed" in npz_path.name.lower()
    if args.edg_only:
        show_dyn = False
    elif args.with_dyn:
        show_dyn = True
    forced = " [forced by --edg-only]" if args.edg_only else (" [forced by --with-dyn]" if args.with_dyn else "")
    print(f"[mode] npz='{npz_path.name}' -> "
          + ("EDG + dynamic points (blue) + render-colour outliers (red)" if show_dyn
             else "EDG ONLY (no dynamic cloud, no render-colour outlier pass)") + forced)

    static_landmarks = data.get("static_landmarks")
    # EDG_test.py saves edg_test/edg_global.npz (stacked arrays + already-deformed dynamic points) instead
    # of EDG_old's results_all.npz (dict tracks + per-point skinning arrays). Support both formats.
    edg_test_mode = ("node_ids" in data.files and "dyn_reconstruct" in data.files)
    dyn_ids = dyn_reconstruct = None
    if edg_test_mode:
        print("[edg_test] detected EDG_test edg_global.npz (direct EDG + dynamic tracks; no skinning re-warp).")
        node_ids = np.asarray(data["node_ids"], dtype=np.int64)
        er = np.asarray(data["edg_reconstruct"], dtype=np.float64)          # [Nn,T,3]
        ero = np.asarray(data["edg_rotation"], dtype=np.float64)            # [Nn,T,3,3]
        if args.fill_edg:                                                 # show the WHOLE EDG at every frame
            er = _hold_fill_stack(er)
            ero = _hold_fill_stack(ero)
            print(f"[edg_test] --fill-edg: held {int(node_ids.size)} EDG nodes across the whole "
                  f"{er.shape[1]}-frame interval.")
        else:                                                             # holes stay blank (the default)
            print("[edg_test] holes are blank: an EDG node is drawn only at frames it was actually solved "
                  "(pass --fill-edg to hold nodes across the whole interval).")
        edg_tracks = {int(node_ids[i]): er[i] for i in range(int(node_ids.size))}
        edg_rot_tracks = {int(node_ids[i]): ero[i] for i in range(int(node_ids.size))}
        edges_union = data["edg_edges"] if "edg_edges" in data.files else _edges_from_topology(node_ids)
        if edges_union is None or np.asarray(edges_union).size == 0:
            print("[edg_test] no EDG edges in the npz or edg_topology.npz; re-run EDG_test.py to save them.")
        dyn_ids = np.asarray(data["dyn_ids"], dtype=np.int64)
        dyn_reconstruct = np.asarray(data["dyn_reconstruct"], dtype=np.float64)  # [Nd,T,3]
    else:
        edg_tracks = _load_dict(data, "edg_reconstruct")
        edg_rot_tracks = _load_dict(data, "edg_rotation")
        edges_union = data.get("edg_edges_union")
        if edg_tracks is None or edg_rot_tracks is None:
            raise RuntimeError(f"Missing EDG tracks or EDG rotations in {npz_path}.")
        if data.get("dynamic_skinning_nodes_array") is None:
            raise RuntimeError(
                f"Missing persistent skinning arrays in {npz_path}. "
                "Run EDG_old.py or EDG_prior.py with the updated code, "
                "or pass --npz to EDG_test's edg_test/edg_global.npz."
            )

    # ---------------- 2D context (needed by FILTER 2 and the dynamic points): pick the tracks table by stamp
    stamp = int(data["n_tracks"]) if "n_tracks" in data.files else None
    if stamp is None and "node_ids" in data.files:                     # unstamped npz: fall back to the
        topo_p = Path(utils.definitions.output_dir) / "edg_topology.npz"    # topology's stamp when the
        if topo_p.exists():                                            # node ids come from that topology
            topo = np.load(topo_p)
            if "n_tracks" in topo.files:
                tn = set(np.asarray(topo["edg_nodes"], int).tolist())
                if set(np.asarray(data["node_ids"], int).tolist()) <= tn:
                    stamp = int(topo["n_tracks"])
                    print(f"[2d] npz is unstamped but its node_ids match edg_topology.npz -> using the "
                          f"topology's stamp n_tracks={stamp}")
    if stamp is None:
        print("[2d] npz has no n_tracks stamp (older solve): trusting --merged; if observations look "
              "missing/misconnected, the flag is probably wrong for this npz")
    ctx = masks = None
    if args.reproj_del_px > 0 or not args.no_mask_del:
        try:
            ctx = _load_2d_context(args.merged, stamp)
            if not args.no_mask_del:
                masks = _load_dyn_mask_images(ctx, args.mask_dilate_px)
        except Exception as e:
            print(f"[2d] context unavailable ({e}); 2D-based filtering disabled")
            ctx = None

    # ---------------- pre-visualization filters: delete bad EDG observations, keep them for the markers
    deleted_groups = apply_filters(edg_tracks, edg_rot_tracks, args, ctx, masks)

    max_frames = max(int(arr.shape[0]) for arr in edg_tracks.values()) if edg_tracks else 0

    if max_frames == 0:
        raise RuntimeError("No EDG tracks found to visualize.")

    # ---------------- Filter 3: photometric render-colour outlier detection on the DYNAMIC tracks
    # (skin every dynamic point across the whole span, then flag tracks whose reprojected colour drifts off
    #  their observed-frame reference colour -> drawn RED among the blue dynamic point cloud below).
    dyn_skin = dyn_pids = dyn_flag = None
    if (show_dyn or args.edg_quality) and edg_test_mode and not args.no_render_outlier:
        if ctx is None:
            print("[render-outlier] 2D context unavailable; skipping photometric detection.")
        else:
            dyn_skin, dyn_pids = _build_dyn_skin(dyn_ids, dyn_reconstruct, node_ids, er, ero)
            if dyn_skin is not None:
                flag, score = detect_render_color_outliers(
                    dyn_skin, dyn_pids, dyn_ids, dyn_reconstruct, ctx, args)
                dyn_flag = flag

    # ---------------- EDG-QUALITY: per-node skinning-weighted fraction of BAD (outlier) dynamic points.
    # A node driven mostly by colour-drifting dynamic points is itself suspect; a node whose attached
    # points are nearly all good is trustworthy. gid -> fraction (NaN = no dynamic points attached).
    edg_bad_frac_of = None
    if args.edg_quality:
        if dyn_skin is None or dyn_flag is None:
            print("[edg-quality] needs the render-colour scoring (completed npz, no --no-render-outlier); "
                  "falling back to the normal EDG display.")
        else:
            bad_w = np.zeros(node_ids.size); tot_w = np.zeros(node_ids.size)
            np.add.at(tot_w, dyn_skin["nl"], dyn_skin["w"])
            np.add.at(bad_w, dyn_skin["nl"], dyn_skin["w"] * dyn_flag[dyn_skin["p"]].astype(np.float64))
            frac = np.where(tot_w > 1e-9, bad_w / np.maximum(tot_w, 1e-9), np.nan)
            edg_bad_frac_of = {int(node_ids[i]): float(frac[i]) for i in range(node_ids.size)}
            fin = np.isfinite(frac)
            n_good = int((frac[fin] <= args.edg_good_frac).sum())
            n_bad = int((frac[fin] >= args.edg_bad_frac).sum())
            n_mix = int(fin.sum()) - n_good - n_bad
            print(f"[edg-quality] {node_ids.size} nodes: {n_good} GOOD (bad-frac <= {args.edg_good_frac:g}, "
                  f"GREEN) | {n_mix} MIXED (ORANGE) | {n_bad} BAD (>= {args.edg_bad_frac:g}, RED) | "
                  f"{int((~fin).sum())} no-dyn (grey)")
            worst = np.argsort(-np.nan_to_num(frac, nan=-1.0))[:8]
            print("[edg-quality] worst nodes: " +
                  ", ".join(f"gid={int(node_ids[i])}({100 * frac[i]:.0f}% bad)" for i in worst if fin[i]))

    edges = _load_edges(edges_union)

    solver_c2w = None
    gt_c2w = None
    align = None

    try:
        solver_c2w = _load_solved_c2w(args.result_xm)
        gt_c2w = _load_gt_c2w(args.dataset_root)

        n_pose = min(solver_c2w.shape[0], gt_c2w.shape[0], max_frames)
        solver_c2w = solver_c2w[:n_pose]
        gt_c2w = gt_c2w[:n_pose]

        if n_pose >= 1:
            align = _first_camera_aligned_similarity(
                solver_c2w,
                gt_c2w,
                scale=1.0,
            )
            print("[align] initialized GT-to-solver transform from first camera.")

    except Exception as exc:
        print(f"[warning] could not load/align camera poses: {exc}")
        solver_c2w = None
        gt_c2w = None
        align = None

    gt_mesh_paths = None
    if not args.hide_gt_mesh:
        gt_mesh_paths = _resolve_gt_geometry_paths(
            args.gt_mesh,
            args.dataset_root,
            max_frames,
        )

    if (
        args.scale_search
        and gt_mesh_paths is not None
        and solver_c2w is not None
        and gt_c2w is not None
    ):
        search_frame = args.frame if args.scale_search_frame is None else args.scale_search_frame
        search_frame = int(np.clip(search_frame, 0, min(max_frames, len(gt_mesh_paths)) - 1))

        if gt_mesh_paths[search_frame] is None:
            print(f"[scale-search] skipped: no GT geometry for frame={search_frame}.")
        else:
            if edg_test_mode:
                dyn_pts_search = dyn_reconstruct[:, search_frame, :]
                dyn_pts_search = dyn_pts_search[np.isfinite(dyn_pts_search).all(axis=1)]
            else:
                dyn_pts_search, *_ = _load_dynamic_points(
                    search_frame, data, edg_tracks, edg_rot_tracks,
                )

            gt_pts_search = _load_geometry_points(
                gt_mesh_paths[search_frame],
                sample_points=args.gt_mesh_sample_points,
                voxel_size=args.gt_mesh_voxel_size,
            )

            scale_result = _line_search_gt_to_dynamic_scale(
                dyn_pts_solver=dyn_pts_search,
                gt_pts=gt_pts_search,
                solver_c2w=solver_c2w,
                gt_c2w=gt_c2w,
                scale_min=args.scale_min,
                scale_max=args.scale_max,
                scale_steps=args.scale_steps,
                scale_iters=args.scale_iters,
                max_points=args.chamfer_max_points,
            )

            if scale_result is not None:
                scale, R, t, raw_loss, norm_loss = scale_result
                align = (scale, R, t)
                print(
                    f"[align] using Chamfer scale-search alignment: "
                    f"scale={scale:.6f}, "
                    f"raw_chamfer={raw_loss:.8f}, "
                    f"norm_chamfer={norm_loss:.8f}"
                )

    coordinate_space = args.coordinate_space
    if coordinate_space == "robocasa":
        if align is None:
            print(
                "[warning] --coordinate-space robocasa requested, but camera alignment "
                "is unavailable. Solver geometries will stay in solver coordinates; "
                "GT mesh/cameras will be hidden to avoid mixing coordinate frames."
            )
            gt_mesh_paths = None
        else:
            scale, _, _ = align
            print(
                "[coords] displaying in raw COLMAP/world coordinates "
                "to match robocasa visualize_moving_outputs_open3d.py "
                f"(inverse GT-to-solver scale={scale:.6f})."
            )
    else:
        if align is None and gt_mesh_paths is not None:
            print(
                "[warning] --coordinate-space solver requested, but GT-to-solver alignment "
                "is unavailable. GT mesh/cameras will be hidden."
            )
            gt_mesh_paths = None
        print("[coords] displaying in EDG/XM solver coordinates.")

    def _solver_points_to_vis(points: np.ndarray) -> np.ndarray:
        if coordinate_space == "robocasa" and align is not None:
            scale, R, t = align
            return _apply_inverse_similarity(points, scale, R, t)
        return points

    def _gt_points_to_vis(points: np.ndarray) -> np.ndarray:
        if coordinate_space == "solver" and align is not None:
            scale, R, t = align
            return _apply_similarity(points, scale, R, t)
        return points

    def _solver_pose_to_vis(pose: np.ndarray) -> np.ndarray:
        if coordinate_space == "robocasa" and align is not None:
            scale, R, t = align
            return _transform_c2w_pose_by_inverse_similarity(pose, scale, R, t)
        return pose

    def _gt_pose_to_vis(pose: np.ndarray) -> np.ndarray:
        if coordinate_space == "solver" and align is not None:
            scale, R, t = align
            return _transform_c2w_pose_by_similarity(pose, scale, R, t)
        return pose

    show_gt_cameras = gt_c2w is not None and (coordinate_space == "robocasa" or align is not None)
    if align is None:
        show_gt_cameras = False

    state = {"frame": int(np.clip(args.frame, 0, max_frames - 1))}

    def _make_geoms(frame_id: int):
        geoms = []                                                    # EDG-only viewer: no static/dyn/GT

        edg_pts, edg_gids = _load_tracks(edg_tracks, frame_id)

        if edg_pts.size > 0:
            edg_pts_solver = edg_pts                                   # keep solver coords for the rotation axes
            edg_pts = _solver_points_to_vis(edg_pts)
            gid_to_idx = {int(gid): i for i, gid in enumerate(edg_gids)}

            if edg_bad_frac_of is not None:
                # EDG-QUALITY view: nodes SEPARATED by the bad-dynamic fraction of their skinned points.
                fr = np.array([edg_bad_frac_of.get(int(g), np.nan) for g in edg_gids])
                fin_fr = np.isfinite(fr)
                for name, m, col, rmul in (
                        ("edg_bad", fin_fr & (fr >= args.edg_bad_frac), [1.0, 0.05, 0.05], 1.6),
                        ("edg_mixed", fin_fr & (fr > args.edg_good_frac) & (fr < args.edg_bad_frac),
                         [1.0, 0.6, 0.0], 1.0),
                        ("edg_good", fin_fr & (fr <= args.edg_good_frac), [0.0, 0.8, 0.15], 1.0),
                        ("edg_nodata", ~fin_fr, [0.55, 0.55, 0.55], 0.7)):
                    if m.any():
                        geoms.append((name, _spheres_mesh(edg_pts[m], args.deleted_dot_radius * rmul, col)))
            else:
                pcd_edg = o3d.geometry.PointCloud()
                pcd_edg.points = o3d.utility.Vector3dVector(edg_pts)
                pcd_edg.paint_uniform_color([0.9, 0.1, 0.1])
                geoms.append(("edg", pcd_edg))

            line_idx = []
            for a, b in edges:
                if a in gid_to_idx and b in gid_to_idx:
                    line_idx.append([gid_to_idx[a], gid_to_idx[b]])

            if line_idx:
                if float(getattr(args, "edg_edge_radius", 0.0)) > 0.0:   # thick edges as cylinders (reliable)
                    cyl = _edges_cylinders(edg_pts, line_idx, float(args.edg_edge_radius), [0.72, 0.72, 0.72])
                    if cyl is not None:
                        geoms.append(("edges", cyl))
                else:
                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(edg_pts)
                    line_set.lines = o3d.utility.Vector2iVector(np.asarray(line_idx, dtype=np.int32))
                    line_set.paint_uniform_color([0.72, 0.72, 0.72])       # weak light-grey edges
                    geoms.append(("edges", line_set))

            if edg_bad_frac_of is None and not args.hide_edg_axes and args.edg_axis_len > 0:
                axes = _edg_axes_lineset(edg_pts_solver, edg_gids, frame_id, edg_rot_tracks,
                                         args.edg_axis_len, _solver_points_to_vis)   # RGB triads (normal view)
                if axes is not None:
                    geoms.append(("edg_axes", axes))

        # EDG-QUALITY / --edg-solo are INDEPENDENT clean windows: ONLY the EDG (nodes + edges [+ axes]).
        # Skip everything else -- deleted markers, dynamic cloud, outlier spheres, cameras.
        if edg_bad_frac_of is not None or args.edg_solo:
            label = ("EDG-quality (GREEN=good ORANGE=mixed RED=bad grey=no-dyn)"
                     if edg_bad_frac_of is not None else "EDG-solo (red nodes + grey edges)")
            print(f"[Open3D] frame={frame_id} {label}: "
                  f"{int(edg_pts.shape[0]) if edg_pts.size else 0} nodes + edges only")
            if not geoms:
                origin = o3d.geometry.PointCloud()
                origin.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))
                origin.paint_uniform_color([0.15, 0.15, 0.15])
                geoms.append(("placeholder", origin))
            return geoms

        # DELETED observations (failed a filter): big marker spheres at the deleted position (red = filter 1
        # smooth-error, ORANGE = filter 2 2D-projection), a blue sphere at the track's previous-frame
        # position, and a line joining the pair (visualizes the jump).
        GROUP_COLOR = {"smooth": [1.0, 0.0, 0.0], "reproj2d": [1.0, 0.55, 0.0]}
        n_deleted_now = 0
        for gname, (dtr, dpr) in deleted_groups.items():
            mark, blue, pairs = [], [], []
            for gid, arr in dtr.items():
                if not (0 <= frame_id < arr.shape[0]):
                    continue
                xr = arr[frame_id]
                if not np.isfinite(xr).all():
                    continue
                mark.append(xr)
                pv = dpr.get(gid)
                if pv is not None and np.isfinite(pv[frame_id]).all():
                    blue.append(pv[frame_id])
                    pairs.append((len(mark) - 1, len(blue) - 1))
            if mark:
                n_deleted_now += len(mark)
                col = GROUP_COLOR.get(gname, [1.0, 0.0, 0.0])
                mark_v = _solver_points_to_vis(np.asarray(mark, dtype=np.float64))
                geoms.append((f"del_{gname}", _spheres_mesh(mark_v, args.deleted_dot_radius, col)))
                if blue:
                    blue_v = _solver_points_to_vis(np.asarray(blue, dtype=np.float64))
                    geoms.append((f"del_{gname}_prev",
                                  _spheres_mesh(blue_v, args.deleted_dot_radius, [0.0, 0.4, 1.0])))
                    link = o3d.geometry.LineSet()
                    link.points = o3d.utility.Vector3dVector(np.vstack([mark_v, blue_v]))
                    link.lines = o3d.utility.Vector2iVector(
                        np.asarray([[ri, len(mark) + bi] for ri, bi in pairs], dtype=np.int32))
                    link.paint_uniform_color(col)
                    geoms.append((f"del_{gname}_links", link))

        # DYNAMIC point cloud (skinned at this frame): normal = blue points, render-colour OUTLIER = big RED
        # spheres (Filter 3). Every dynamic track keeps its per-track flag across all frames.
        n_dyn_now = n_dyn_out_now = 0
        if show_dyn and dyn_skin is not None and 0 <= frame_id < dyn_skin["edg_pos"].shape[1]:
            dyn_pos = _skin_dyn_frame(dyn_skin, frame_id)
            fin = np.isfinite(dyn_pos).all(1)
            flags = dyn_flag if dyn_flag is not None else np.zeros(dyn_skin["P"], bool)
            norm_mask = fin & ~flags
            out_mask = fin & flags
            n_dyn_now = int(fin.sum()); n_dyn_out_now = int(out_mask.sum())
            if norm_mask.any():
                dn = _solver_points_to_vis(dyn_pos[norm_mask])
                pcd_dyn = o3d.geometry.PointCloud()
                pcd_dyn.points = o3d.utility.Vector3dVector(dn)
                pcd_dyn.paint_uniform_color([0.0, 0.4, 1.0])          # normal dynamic points = blue
                geoms.append(("dyn", pcd_dyn))
            if out_mask.any():
                do = _solver_points_to_vis(dyn_pos[out_mask])
                geoms.append(("dyn_outlier", _spheres_mesh(do, args.render_dot_radius, [1.0, 0.0, 0.0])))

        if args.show_cameras and solver_c2w is not None:
            if frame_id < solver_c2w.shape[0]:
                solver_cam = _camera_axes_lines(
                    _solver_pose_to_vis(solver_c2w[frame_id]),
                    axis_len=float(args.camera_axis_len),
                )
                geoms.append(("solver_camera", solver_cam))

            if show_gt_cameras and frame_id < gt_c2w.shape[0]:
                gt_pose_vis = _gt_pose_to_vis(gt_c2w[frame_id])

                gt_cam = _camera_axes_lines(
                    gt_pose_vis,
                    axis_len=float(args.camera_axis_len),
                )
                geoms.append(("gt_camera", gt_cam))

        print(f"[Open3D] frame={frame_id} edg={edg_pts.shape[0]} deleted_shown={n_deleted_now} "
              f"dyn_shown={n_dyn_now} dyn_outliers={n_dyn_out_now}")

        if not geoms:                                                     # all nodes are holes this frame:
            origin = o3d.geometry.PointCloud()                            # keep the viewer alive with a single
            origin.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # point at the origin -> the frame
            origin.paint_uniform_color([0.15, 0.15, 0.15])               # looks empty (no phantom geometry)
            geoms.append(("placeholder", origin))

        return geoms

    named_geoms = _make_geoms(state["frame"])

    if not named_geoms:
        raise RuntimeError("No geometry to visualize.")

    if args.dry_run:
        print("[dry-run] built geometries:", ", ".join(name for name, _ in named_geoms))
        return

    if args.save_frame is not None:
        # Headless offscreen render (EGL) of the single --frame -> PNG, so the dynamic outliers can be
        # eyeballed without a display. Uses a hidden legacy window (works with these legacy geometries).
        import cv2
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="EDG Filtered Viewer", width=1280, height=720, visible=False)
        for _, geom in named_geoms:
            vis.add_geometry(geom)
        vis.get_render_option().point_size = float(args.point_size)
        vis.poll_events(); vis.update_renderer()
        buf = vis.capture_screen_float_buffer(True)
        vis.destroy_window()
        rgb = (np.asarray(buf) * 255.0).clip(0, 255).astype(np.uint8)
        args.save_frame.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_frame), rgb[:, :, ::-1])
        print(f"[save-frame] wrote {args.save_frame}")
        return

    geom_map = {name: geom for name, geom in named_geoms}

    def _refresh(vis):
        new_named = _make_geoms(state["frame"])
        new_map = {name: geom for name, geom in new_named}

        # Add new geometry types that were absent at initialization.
        for name, new_geom in new_map.items():
            if name not in geom_map:
                vis.add_geometry(new_geom)
                geom_map[name] = new_geom

        # Update existing geometries; a type absent this frame (its nodes/points are holes) is CLEARED to
        # empty so it disappears from the screen instead of showing stale points from the previous frame.
        empty_pts = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
        for name, geom in list(geom_map.items()):
            new_geom = new_map.get(name)

            if isinstance(geom, o3d.geometry.PointCloud):
                if isinstance(new_geom, o3d.geometry.PointCloud):
                    geom.points = new_geom.points
                    geom.colors = new_geom.colors
                else:
                    geom.points = empty_pts
                    geom.colors = empty_pts
                vis.update_geometry(geom)

            elif isinstance(geom, o3d.geometry.LineSet):
                if isinstance(new_geom, o3d.geometry.LineSet):
                    geom.points = new_geom.points
                    geom.lines = new_geom.lines
                    geom.colors = new_geom.colors
                else:
                    geom.points = empty_pts
                    geom.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
                vis.update_geometry(geom)

            elif isinstance(geom, o3d.geometry.TriangleMesh):          # the deleted-track spheres
                if isinstance(new_geom, o3d.geometry.TriangleMesh):
                    geom.vertices = new_geom.vertices
                    geom.triangles = new_geom.triangles
                    geom.vertex_colors = new_geom.vertex_colors
                    geom.vertex_normals = new_geom.vertex_normals
                else:
                    geom.vertices = empty_pts
                    geom.triangles = o3d.utility.Vector3iVector(np.zeros((0, 3), dtype=np.int32))
                vis.update_geometry(geom)

        return False

    def on_space(vis):
        state["frame"] = (state["frame"] + 1) % max_frames
        return _refresh(vis)

    def on_b(vis):
        state["frame"] = (state["frame"] - 1) % max_frames
        return _refresh(vis)

    def on_q(vis):
        vis.close()
        return False

    print("Open3D controls: SPACE=next frame, B=prev frame, Q/ESC=close.")
    print(
        "Colors: EDG=red points, edges=grey, orientation=RGB axes; DELETED observations = BIG "
        "spheres (RED = filter 1 smooth-error, ORANGE = filter 2 2D-projection: too far from observation / "
        "outside the dynamic mask), previous-frame position = blue sphere, a line joins each pair; DYNAMIC "
        "points = blue, render-colour OUTLIERS (filter 3) = big RED spheres."
    )

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="EDG Filtered Viewer", width=1280, height=720)

    for _, geom in named_geoms:
        vis.add_geometry(geom)

    opt = vis.get_render_option()
    opt.point_size = float(args.point_size)
    if float(getattr(args, "edg_line_width", 0.0)) > 0.0:
        opt.line_width = float(args.edg_line_width)         # quick but ignored on many builds

    if args.init_first_cam:
        cam_c2w_vis = None
        if solver_c2w is not None and solver_c2w.shape[0] > 0:
            cam_c2w_vis = _solver_pose_to_vis(np.asarray(solver_c2w[0], dtype=np.float64))
        if cam_c2w_vis is not None and np.isfinite(cam_c2w_vis).all():
            ctr = vis.get_view_control()
            params = ctr.convert_to_pinhole_camera_parameters()
            params.extrinsic = np.linalg.inv(cam_c2w_vis)  # world-to-camera of frame 0
            try:
                ok = ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
            except TypeError:  # older Open3D without allow_arbitrary
                ok = ctr.convert_from_pinhole_camera_parameters(params)
            print(f"[init-first-cam] viewpoint set to solver frame-0 camera (applied={ok}).")
        else:
            print("[init-first-cam] no solver camera pose available (set XM_DATASET_NAME / --result-xm); "
                  "using default fit-to-geometry view.")

    vis.register_key_callback(ord(" "), on_space)
    vis.register_key_callback(ord("b"), on_b)
    vis.register_key_callback(ord("B"), on_b)
    vis.register_key_callback(ord("q"), on_q)
    vis.register_key_callback(ord("Q"), on_q)
    vis.register_key_callback(256, on_q)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
