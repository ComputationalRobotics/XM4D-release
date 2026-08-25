#!/usr/bin/env python3
"""Visualize EDG-skinned mesh with rotations using Open3D."""

from pathlib import Path
import glob
import os

import numpy as np
import trimesh
from scipy.spatial import cKDTree

import utils.definitions

try:
	import open3d as o3d
	OPEN3D_AVAILABLE = True
except Exception:
	OPEN3D_AVAILABLE = False


# ---- CONFIG ----
MESH_FRAME = 0
K_NEIGHBORS = 4
MIN_WEIGHT_DIST = 1e-6
CHUNK_SIZE = 5000
MAX_MESH_TRIANGLES = 60000
EXPORT_FRAMES = True
EXPORT_EVERY = 1
EXPORT_MAX_FRAMES = 200
RUN_VISUALIZER = True


def main() -> None:
	out_dir = Path(utils.definitions.output_dir)
	vis_path = out_dir / "edg_solution_vis.npz"
	if not vis_path.exists():
		raise FileNotFoundError(f"Missing EDG solution file: {vis_path}")

	data = np.load(vis_path)
	edg_deformed = np.asarray(data["edg_deformed"], dtype=np.float64)
	edg_rotation = data["edg_rotation"] if "edg_rotation" in data else None
	if edg_rotation is None or np.asarray(edg_rotation).size == 0:
		edg_rotation = np.tile(
			np.eye(3, dtype=np.float64)[None, None, :, :],
			(edg_deformed.shape[0], edg_deformed.shape[1], 1, 1),
		)
	else:
		edg_rotation = np.asarray(edg_rotation, dtype=np.float64)

	if edg_deformed.ndim != 3 or edg_deformed.shape[-1] != 3:
		raise ValueError(f"Expected edg_deformed (M,T,3), got {edg_deformed.shape}")
	if edg_rotation.ndim != 4 or edg_rotation.shape[-2:] != (3, 3):
		raise ValueError(f"Expected edg_rotation (M,T,3,3), got {edg_rotation.shape}")

	image_paths = sorted(glob.glob(str(utils.definitions.video_dir / utils.definitions.image_pattern)))
	if not image_paths:
		raise FileNotFoundError(f"No images found in {utils.definitions.video_dir}")
	frame_idx = int(np.clip(MESH_FRAME, 0, len(image_paths) - 1))
	image_path = Path(image_paths[frame_idx])

	mesh_candidates = [
		Path(utils.definitions.output_dir) / "sam3d" / f"sam3d_{image_path.stem}_aligned.ply",
		Path(utils.definitions.rootpath)
		/ "object_pointcloud_world"
		/ f"frame{frame_idx:06d}.ply",
		Path(utils.definitions.rootpath) / "object_pointcloud_world" / f"{frame_idx:06d}.ply",
	]
	mesh_candidates += [
		Path(p)
		for p in sorted(
			glob.glob(str(Path(utils.definitions.output_dir) / "sam3d" / "*.ply"))
		)
	]
	mesh_path = next((p for p in mesh_candidates if p.exists()), None)
	if mesh_path is None:
		raise FileNotFoundError(
			"Missing mesh: tried sam3d outputs and object_pointcloud_world for frame "
			f"{frame_idx:06d}"
		)

	mesh = trimesh.load(str(mesh_path), process=False)
	if isinstance(mesh, trimesh.Scene):
		mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
	if isinstance(mesh, trimesh.points.PointCloud):
		verts = np.asarray(mesh.vertices, dtype=np.float64)
		faces = np.zeros((0, 3), dtype=np.int64)
		mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
	if not isinstance(mesh, trimesh.Trimesh):
		raise RuntimeError(f"Expected Trimesh or PointCloud, got {type(mesh)}")
	if MAX_MESH_TRIANGLES is not None and mesh.faces.shape[0] > int(MAX_MESH_TRIANGLES):
		simplify_fn = getattr(mesh, "simplify_quadratic_decimation", None)
		if simplify_fn is not None:
			try:
				mesh = simplify_fn(int(MAX_MESH_TRIANGLES))
			except Exception as exc:
				print(f"[WARN] Mesh simplification failed: {exc}")
		else:
			print("[WARN] Mesh simplification not available; skipping.")
	verts = np.asarray(mesh.vertices, dtype=np.float64)
	if verts.ndim != 2 or verts.shape[1] != 3:
		raise ValueError(f"Expected mesh vertices [V,3], got {verts.shape}")
	faces = np.asarray(mesh.faces, dtype=np.int64)
	if faces.size > 0:
		if faces.max() >= verts.shape[0]:
			mask = (faces >= 0).all(axis=1) & (faces < verts.shape[0]).all(axis=1)
			faces = faces[mask]
	mesh.faces = faces
	has_faces = faces.size > 0

	edg_rest = edg_deformed[:, 0, :].astype(np.float64)
	edg_rot_rest = edg_rotation[:, 0, :, :].astype(np.float64)

	tree = cKDTree(edg_rest)
	nn_idx = np.zeros((verts.shape[0], K_NEIGHBORS), dtype=np.int64)
	nn_dist = np.zeros((verts.shape[0], K_NEIGHBORS), dtype=np.float64)
	for i, v in enumerate(verts):
		dist, idx = tree.query(v, k=int(K_NEIGHBORS))
		idx = np.atleast_1d(idx).tolist()
		dist = np.atleast_1d(dist).tolist()
		if len(idx) < K_NEIGHBORS:
			pad = K_NEIGHBORS - len(idx)
			idx = list(idx) + [idx[-1]] * pad
			dist = list(dist) + [dist[-1]] * pad
		nn_idx[i, :] = np.asarray(idx, dtype=np.int64)
		nn_dist[i, :] = np.asarray(dist, dtype=np.float64)

	w = 1.0 / (np.sqrt(nn_dist) + float(MIN_WEIGHT_DIST))
	w_sum = np.sum(w, axis=1, keepdims=True)
	w = w / np.maximum(w_sum, MIN_WEIGHT_DIST)

	rel = np.zeros((verts.shape[0], K_NEIGHBORS, 3), dtype=np.float64)
	for i in range(verts.shape[0]):
		idx = nn_idx[i]
		rel[i] = (edg_rot_rest[idx].transpose(0, 2, 1) @ (verts[i] - edg_rest[idx]).reshape(-1, 3, 1)).reshape(-1, 3)

	T = edg_deformed.shape[1]

	def _deform_vertices(t: int) -> np.ndarray:
		out = np.zeros_like(verts)
		for start in range(0, verts.shape[0], int(CHUNK_SIZE)):
			end = min(start + int(CHUNK_SIZE), verts.shape[0])
			idx = nn_idx[start:end]
			w_chunk = w[start:end]
			rel_chunk = rel[start:end]
			edg_pos = edg_deformed[idx, t, :]
			edg_rot = edg_rotation[idx, t, :, :]
			warped = edg_pos + np.einsum("...ij,...j->...i", edg_rot, rel_chunk)
			out[start:end] = np.sum(warped * w_chunk[:, :, None], axis=1)
		return out

	verts0 = _deform_vertices(0)

	has_display = bool(os.environ.get("DISPLAY"))
	if EXPORT_FRAMES:
		export_dir = Path(utils.definitions.output_dir) / "edg_mesh_frames"
		export_dir.mkdir(parents=True, exist_ok=True)
		export_count = 0
		for t in range(0, T, max(1, int(EXPORT_EVERY))):
			out_path = export_dir / f"mesh_{t:04d}.ply"
			verts_t = _deform_vertices(t)
			if has_faces:
				mesh_t = trimesh.Trimesh(vertices=verts_t, faces=faces, process=False)
				mesh_t.export(out_path)
			else:
				pcd = o3d.geometry.PointCloud()
				pcd.points = o3d.utility.Vector3dVector(verts_t)
				o3d.io.write_point_cloud(str(out_path), pcd)
			export_count += 1
			if export_count >= int(EXPORT_MAX_FRAMES):
				break
		print(f"[EXPORT] Wrote {export_count} mesh frames to {export_dir}")

	if not RUN_VISUALIZER or not has_display or not OPEN3D_AVAILABLE:
		print("[VIS] Skipping interactive Open3D window.")
		return

	if has_faces:
		mesh_o3d = o3d.geometry.TriangleMesh()
		mesh_o3d.vertices = o3d.utility.Vector3dVector(verts0)
		mesh_o3d.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
		mesh_o3d.compute_vertex_normals()
	else:
		mesh_o3d = o3d.geometry.PointCloud()
		mesh_o3d.points = o3d.utility.Vector3dVector(verts0)

	state = {"t": 0}

	def on_space(vis):
		state["t"] = (state["t"] + 1) % T
		verts_t = _deform_vertices(state["t"])
		if has_faces:
			mesh_o3d.vertices = o3d.utility.Vector3dVector(verts_t)
			mesh_o3d.compute_vertex_normals()
		else:
			mesh_o3d.points = o3d.utility.Vector3dVector(verts_t)
		vis.update_geometry(mesh_o3d)
		return False

	print("Press SPACE to step frames.")
	vis = o3d.visualization.VisualizerWithKeyCallback()
	vis.create_window(window_name="EDG Mesh Deformation")
	vis.add_geometry(mesh_o3d)
	vis.register_key_callback(ord(" "), on_space)
	vis.run()
	vis.destroy_window()


if __name__ == "__main__":
	main()
