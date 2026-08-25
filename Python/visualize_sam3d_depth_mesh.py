import numpy as np
import utils.definitions
from pathlib import Path
import trimesh
import viser
from utils.sam3d_utils import lift_depth_points, load_depth_and_intrinsics, load_mask_for_frame

def _get_viser_url(server):
    url = None
    for obj in [server, getattr(server, "_server", None), getattr(server, "_http_server", None)]:
        if obj is None:
            continue
        for attr in ("http_server_url", "url", "base_url", "_base_url", "_url"):
            val = getattr(obj, attr, None)
            if isinstance(val, str) and val.startswith("http"):
                url = val
                break
        if url:
            break
    if url is None:
        host = getattr(server, "host", "localhost")
        port = getattr(server, "port", None)
        url = f"http://{host}:{port}" if port is not None else "http://localhost"
    return url

# CONFIG
FRAME_INDEX = 0
MESH_PATH = Path(utils.definitions.output_dir) / "sam3d" / f"sam3d_{FRAME_INDEX:05d}.ply"
POINT_SIZE_MESH = 0.003
POINT_SIZE_DEPTH = 0.002

if not MESH_PATH.exists():
    raise FileNotFoundError(f"Missing mesh: {MESH_PATH}")

depth_map, intrinsics = load_depth_and_intrinsics(frame_idx=FRAME_INDEX)
mask = None

mesh = trimesh.load_mesh(str(MESH_PATH), process=False)
verts_scene = np.asarray(mesh.vertices, dtype=np.float32)

# Depth points (camera coordinates)
depth_pts = lift_depth_points(depth_map, intrinsics, mask=mask, stride=4)

server = viser.ViserServer()
server.scene.add_point_cloud(
    name="mesh_vertices",
    points=verts_scene,
    colors=np.tile(np.array([[0.2, 0.9, 0.2]], dtype=np.float32), (verts_scene.shape[0], 1)),
    point_size=POINT_SIZE_MESH,
)
server.scene.add_point_cloud(
    name="depth_lifted",
    points=depth_pts,
    colors=np.tile(np.array([[0.2, 0.4, 1.0]], dtype=np.float32), (depth_pts.shape[0], 1)),
    point_size=POINT_SIZE_DEPTH,
)

print(depth_pts)

print("Viser running. Open the URL in a browser.")
print(_get_viser_url(server))

try:
    input("Press Enter to quit...")
except KeyboardInterrupt:
    pass
