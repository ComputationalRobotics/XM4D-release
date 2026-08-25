import signal
import time
import numpy as np

_SERVER = None


def _require_viser():
    try:
        import viser
    except Exception as exc:
        raise RuntimeError("viser is required for visualization.") from exc
    return viser


def get_viser_server(host="0.0.0.0", port=8080):
    global _SERVER
    if _SERVER is None:
        viser = _require_viser()
        _SERVER = viser.ViserServer(host=host, port=port)
        print(f"[viser] Server running at http://localhost:{port}")
    return _SERVER


def _point_size_from_bbox(points_a, points_b):
    if points_a.size == 0 and points_b.size == 0:
        return 0.01
    pts = points_a if points_b.size == 0 else points_b if points_a.size == 0 else np.vstack([points_a, points_b])
    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    return float(np.clip(0.005 * diag, 0.002, 0.05))


def _axis_length_from_bbox(points_a, points_b):
    if points_a.size == 0 and points_b.size == 0:
        return 0.2
    pts = points_a if points_b.size == 0 else points_b if points_a.size == 0 else np.vstack([points_a, points_b])
    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    return float(0.05 * diag) if diag > 0 else 0.2


def _build_line_segments(src_pts, tgt_pts, corrs_A, corrs_B, max_lines):
    if len(corrs_A) == 0 or len(corrs_B) == 0:
        return np.zeros((0, 2, 3), dtype=np.float32)
    num = min(max_lines, len(corrs_A), len(corrs_B))
    segments = np.zeros((num, 2, 3), dtype=np.float32)
    for i in range(num):
        segments[i, 0] = src_pts[corrs_A[i]]
        segments[i, 1] = tgt_pts[corrs_B[i]]
    return segments


def _clear_paths(server, paths):
    for path in paths:
        try:
            server.scene.remove(path)
        except Exception:
            pass


def visualize_scene(label, mesh_pts, depth_pts, host="0.0.0.0", port=8080):
    server = get_viser_server(host, port)
    _clear_paths(server, ["/sam3d/mesh", "/sam3d/depth", "/sam3d/camera"])
    pt_size = _point_size_from_bbox(mesh_pts, depth_pts)
    axis_len = _axis_length_from_bbox(mesh_pts, depth_pts)
    if mesh_pts.size > 0:
        mesh_colors = np.tile(np.array([51, 230, 51], dtype=np.uint8), (mesh_pts.shape[0], 1))
        server.scene.add_point_cloud("/sam3d/mesh", points=mesh_pts.astype(np.float32), colors=mesh_colors, point_size=pt_size, point_shape="circle", visible=True)
    if depth_pts.size > 0:
        depth_colors = np.tile(np.array([51, 102, 255], dtype=np.uint8), (depth_pts.shape[0], 1))
        server.scene.add_point_cloud("/sam3d/depth", points=depth_pts.astype(np.float32), colors=depth_colors, point_size=pt_size, point_shape="circle", visible=True)
    server.scene.add_frame("/sam3d/camera", show_axes=True, axes_length=float(axis_len), axes_radius=float(axis_len * 0.05), wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64), position=np.array([0.0, 0.0, 0.0], dtype=np.float64), visible=True)
    print(f"[VIS] {label}")


def visualize_correspondences(src_pts, tgt_pts, corrs_A, corrs_B, max_lines, host="0.0.0.0", port=8080):
    if len(corrs_A) == 0 or len(corrs_B) == 0:
        return
    server = get_viser_server(host, port)
    _clear_paths(server, ["/corr/src", "/corr/tgt", "/corr/lines"])
    pt_size = _point_size_from_bbox(src_pts, tgt_pts)
    src_colors = np.tile(np.array([51, 230, 51], dtype=np.uint8), (src_pts.shape[0], 1))
    tgt_colors = np.tile(np.array([51, 102, 255], dtype=np.uint8), (tgt_pts.shape[0], 1))
    server.scene.add_point_cloud("/corr/src", points=src_pts.astype(np.float32), colors=src_colors, point_size=pt_size, point_shape="circle", visible=True)
    server.scene.add_point_cloud("/corr/tgt", points=tgt_pts.astype(np.float32), colors=tgt_colors, point_size=pt_size, point_shape="circle", visible=True)
    segments = _build_line_segments(src_pts, tgt_pts, corrs_A, corrs_B, max_lines)
    if len(segments) > 0:
        server.scene.add_line_segments("/corr/lines", points=segments, colors=np.array([255, 51, 51], dtype=np.uint8), line_width=2.0, visible=True)
    print("[VIS] Correspondences (red)")


def visualize_alignment(label, src_pts, tgt_pts, host="0.0.0.0", port=8080):
    server = get_viser_server(host, port)
    _clear_paths(server, ["/align/src", "/align/tgt"])
    pt_size = _point_size_from_bbox(src_pts, tgt_pts)
    src_colors = np.tile(np.array([51, 230, 51], dtype=np.uint8), (src_pts.shape[0], 1))
    tgt_colors = np.tile(np.array([51, 102, 255], dtype=np.uint8), (tgt_pts.shape[0], 1))
    server.scene.add_point_cloud("/align/src", points=src_pts.astype(np.float32), colors=src_colors, point_size=pt_size, point_shape="circle", visible=True)
    server.scene.add_point_cloud("/align/tgt", points=tgt_pts.astype(np.float32), colors=tgt_colors, point_size=pt_size, point_shape="circle", visible=True)
    print(f"[VIS] {label}")


def block_until_ctrl_c():
    print("[VIS] Press Ctrl-C to close the viewer.")
    stop = {"flag": False}
    prev_handler = signal.getsignal(signal.SIGINT)
    def _handle_sigint(signum, frame):
        stop["flag"] = True
    signal.signal(signal.SIGINT, _handle_sigint)
    try:
        while not stop["flag"]:
            time.sleep(0.5)
        print("[VIS] Closing viewer.")
    finally:
        signal.signal(signal.SIGINT, prev_handler)
