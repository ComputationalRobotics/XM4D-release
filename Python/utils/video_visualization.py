from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile

import cv2
import numpy as np


def write_mp4_compat(
	frames_bgr: list[np.ndarray],
	out_path: str | Path,
	fps: float = 10.0,
) -> None:
	"""Write BGR frames to MP4, preferring ffmpeg/libx264 and falling back to OpenCV mp4v."""
	if len(frames_bgr) == 0:
		raise ValueError("frames_bgr is empty.")

	out_path = Path(out_path)
	out_path.parent.mkdir(parents=True, exist_ok=True)

	h, w = frames_bgr[0].shape[:2]
	norm_frames: list[np.ndarray] = []
	for frm in frames_bgr:
		if frm.dtype != np.uint8:
			frm = np.clip(frm, 0, 255).astype(np.uint8)
		if frm.ndim == 2:
			frm = cv2.cvtColor(frm, cv2.COLOR_GRAY2BGR)
		if frm.shape[0] != h or frm.shape[1] != w:
			frm = cv2.resize(frm, (w, h), interpolation=cv2.INTER_LINEAR)
		norm_frames.append(frm)

	ffmpeg_bin = shutil.which("ffmpeg")
	if ffmpeg_bin is not None:
		try:
			with tempfile.TemporaryDirectory(prefix="traj2d_viz_") as temp_dir:
				for t, frame in enumerate(norm_frames):
					cv2.imwrite(
						str(Path(temp_dir) / f"{t:06d}.jpg"),
						frame,
						[int(cv2.IMWRITE_JPEG_QUALITY), 95],
					)
				cmd = [
					ffmpeg_bin,
					"-y",
					"-hide_banner",
					"-loglevel",
					"error",
					"-framerate",
					str(int(round(float(fps)))),
					"-i",
					str(Path(temp_dir) / "%06d.jpg"),
					"-c:v",
					"libx264",
					"-crf",
					"20",
					"-pix_fmt",
					"yuv420p",
					str(out_path),
				]
				subprocess.run(cmd, check=True)
				return
		except Exception as e:
			print(f"ffmpeg export failed, fallback to OpenCV mp4v: {e}")

	writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h), True)
	if not writer.isOpened():
		raise RuntimeError("Failed to open MP4 writer fallback (mp4v).")
	for frm in norm_frames:
		writer.write(frm)
	writer.release()

