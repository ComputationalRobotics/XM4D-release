import time
from pathlib import Path
import glob
import numpy as np
import viser
import utils.definitions
from utils.edg import build_world_trajectories_from_tracks
from utils.readgt_DAVIS import load_davis_gt

BUILD_EDG_DELETE_SMOOTHNESS_PERCENTILE = 0.0
BUILD_EDG_SEGMENR_LENGTH = utils.definitions.segment_length

track_data = np.load(utils.definitions.tracks_dynamics_path, allow_pickle=True)
tracks_xyz = track_data["tracks_xyz"]
occluded = track_data["occluded"]
dynamic_obs_mask = track_data["dynamic_obs_mask"]

T = tracks_xyz.shape[2]
tracks_xyz = tracks_xyz[:, :, :T]
occluded = occluded[:, :, :T]
dynamic_obs_mask = dynamic_obs_mask[:, :T]
print(f"Loaded {T} frames and {tracks_xyz.shape[1]} tracks")

GT = load_davis_gt(str(utils.definitions.rootpath))
if len(GT) == 0:
    raise RuntimeError("DAVIS GT/intrinsics dictionary is empty.")

gt_items = sorted(GT.items(), key=lambda kv: int(kv[1]["id"]))
rt = np.load(str(utils.definitions.rootpath / "outputs" / "result_XM.npz"))
R_raw = np.zeros((T, 3, 3))
t_raw = np.zeros((T, 3))
for i in range(T):
    R_raw[i] = rt["R"][:, 3 * i : 3 * (i + 1)]
    t_raw[i] = rt["t"][:, i]

s_raw = rt["s"] if "s" in rt else None
R_for_edg = R_raw.astype(np.float64).copy()
if s_raw is not None:
    R_for_edg *= s_raw[:, None, None]

glob_all_full = np.einsum("tij,qtj->qti", R_for_edg, tracks_xyz[0].astype(np.float64)) + t_raw[None, :, :]
valid_mask_global = (
    np.isfinite(glob_all_full).all(axis=-1)
    & (occluded[0] == 0)
    & dynamic_obs_mask.astype(bool)
)

current_frame = 0
end_frame_excl = int(min(T, current_frame + BUILD_EDG_SEGMENR_LENGTH))
frames_seg = np.arange(30, 50, dtype=int)

seg_filter_output = build_world_trajectories_from_tracks(
    glob_all=glob_all_full,
    valid_mask_global=valid_mask_global,
    frames=frames_seg,
    delete_smoothness_percentile=BUILD_EDG_DELETE_SMOOTHNESS_PERCENTILE,
)

selected_ids = np.array(seg_filter_output["selected_edg_idx"], dtype=int)
print(f"Segment dynamics keep {len(selected_ids)}/{glob_all_full.shape[0]} points")

tracks_sel = glob_all_full[selected_ids]  # (N,T,3)
valid_sel = valid_mask_global[selected_ids]

server = viser.ViserServer(host="0.0.0.0", port=8080)
frame_slider = server.gui.add_slider("Frame", min=0, max=T - 1, step=1, initial_value=0)

def _build_tail_segments(upto_t: int) -> np.ndarray:
    segments = []
    for i in range(tracks_sel.shape[0]):
        pts = tracks_sel[i]
        valid = valid_sel[i]
        for t in range(min(upto_t, T - 1)):
            if valid[t] and valid[t + 1] and np.isfinite(pts[t]).all() and np.isfinite(pts[t + 1]).all():
                segments.append([pts[t], pts[t + 1]])
    if len(segments) == 0:
        return np.zeros((0, 2, 3), dtype=np.float32)
    return np.asarray(segments, dtype=np.float32)

def _frame_points(frame_id: int) -> np.ndarray:
    pts = tracks_sel[:, frame_id, :]
    valid = valid_sel[:, frame_id]
    mask = valid & np.isfinite(pts).all(axis=1)
    if not np.any(mask):
        return np.zeros((0, 3), dtype=np.float32)
    return pts[mask].astype(np.float32)

def _update(frame_id: int):
    frame_id = int(frame_id)
    tail_seg = _build_tail_segments(frame_id)
    if tail_seg.shape[0] > 0:
        tail_colors = np.tile(np.array([255, 255, 0], dtype=np.uint8)[None, None, :], (tail_seg.shape[0], 2, 1))
        server.scene.add_line_segments(
            "/dyn/tails",
            points=tail_seg,
            colors=tail_colors,
            line_width=1.5,
            visible=True,
        )
    else:
        server.scene.add_line_segments(
            "/dyn/tails",
            points=np.zeros((1, 2, 3), dtype=np.float32),
            colors=np.array([[[0, 0, 0], [0, 0, 0]]], dtype=np.uint8),
            line_width=1.0,
            visible=False,
        )

    pts = _frame_points(frame_id)
    if pts.shape[0] > 0:
        colors = np.tile(np.array([0, 120, 255], dtype=np.uint8), (pts.shape[0], 1))
        server.scene.add_point_cloud(
            "/dyn/points",
            points=pts,
            colors=colors,
            point_size=0.01,
            point_shape="circle",
            visible=True,
        )
    else:
        server.scene.add_point_cloud(
            "/dyn/points",
            points=np.zeros((1, 3), dtype=np.float32),
            colors=np.array([[0, 0, 0]], dtype=np.uint8),
            point_size=0.001,
            visible=False,
        )

def _on_slider(event):
    _update(frame_slider.value)

frame_slider.on_update(_on_slider)
_update(0)
print("[VIS] Drag the Frame slider to inspect points and tails. Press Ctrl-C to exit.")
try:
    while True:
        time.sleep(0.5)
except KeyboardInterrupt:
    pass
