"""Import the NVIDIA Dynamic Scenes dataset into datasets/NVIDIA/, laid out
like Syn4D so the rest of the pipeline can consume it.

For each scene it produces:
    images/frame%06d.png                        (copied from images_original_size)
    poses_bounds.npy                            (native GT, copied as-is)
    camera_extrinsics_world_to_camera.txt       (w2c, COLMAP convention)
    camera_poses_world.txt                       (c2w, OpenCV axes)
    episode_info.json

Run:
    python Python/utils/prepare_nvidia.py \
        --src ~/Downloads/data/nvidia \
        --dst datasets/NVIDIA
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from readgt_nvidia import _llff_pose_to_opencv


def _quat_wxyz(R: np.ndarray):
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return w, x, y, z


def _write_pose_txt(path: Path, header2: str, rows):
    with open(path, "w") as f:
        f.write("# frame_id image_name tx ty tz qw qx qy qz\n")
        f.write(f"# {header2}\n")
        for i, name, t, q in rows:
            f.write(
                f"{i} {name} {t[0]:.17g} {t[1]:.17g} {t[2]:.17g} "
                f"{q[0]:.17g} {q[1]:.17g} {q[2]:.17g} {q[3]:.17g}\n"
            )


def prepare_scene(src_scene: Path, dst_scene: Path):
    img_src = src_scene / "images_original_size"
    if not img_src.is_dir():
        raise FileNotFoundError(f"Missing images_original_size in {src_scene}")
    poses_bounds = np.load(src_scene / "poses_bounds.npy")
    poses = poses_bounds[:, :15].reshape(-1, 3, 5)

    src_imgs = sorted(
        (p for p in img_src.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}),
        key=lambda p: int("".join(ch for ch in p.stem if ch.isdigit()) or -1),
    )
    if len(src_imgs) != poses.shape[0]:
        raise ValueError(
            f"{src_scene.name}: {len(src_imgs)} images vs {poses.shape[0]} poses"
        )

    dst_img = dst_scene / "images"
    dst_img.mkdir(parents=True, exist_ok=True)

    names = []
    w2c_rows = []
    c2w_rows = []
    for i, src_img in enumerate(src_imgs):
        name = f"frame{i:06d}.png"
        names.append(name)
        shutil.copy2(src_img, dst_img / name)

        _, R_wc, t_wc, R_cw, center = _llff_pose_to_opencv(poses[i])
        w2c_rows.append((i, name, t_wc, _quat_wxyz(R_wc)))
        c2w_rows.append((i, name, center, _quat_wxyz(R_cw)))

    shutil.copy2(src_scene / "poses_bounds.npy", dst_scene / "poses_bounds.npy")
    _write_pose_txt(
        dst_scene / "camera_extrinsics_world_to_camera.txt",
        "world-to-camera / w2c: T_camera_world, same convention used in COLMAP images.txt",
        w2c_rows,
    )
    _write_pose_txt(
        dst_scene / "camera_poses_world.txt",
        "camera-to-world / c2w: T_world_camera, OpenCV camera axes (x right, y down, z forward)",
        c2w_rows,
    )

    h, w, f = poses[0][:, 4]
    episode = {
        "dataset": "NVIDIA Dynamic Scenes",
        "scene": src_scene.name,
        "num_frames": int(poses.shape[0]),
        "camera_height": int(round(h)),
        "camera_width": int(round(w)),
        "focal": float(f),
        "image_source": "images_original_size",
    }
    with open(dst_scene / "episode_info.json", "w") as fp:
        json.dump(episode, fp, indent=2)

    print(f"[NVIDIA] {src_scene.name}: {len(names)} frames -> {dst_scene}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(Path.home() / "Downloads/data/nvidia"))
    ap.add_argument(
        "--dst",
        default=str(Path(__file__).resolve().parents[2] / "datasets" / "NVIDIA"),
    )
    args = ap.parse_args()

    src_root = Path(args.src).expanduser()
    dst_root = Path(args.dst).expanduser()
    scenes = sorted(p for p in src_root.iterdir() if p.is_dir())
    if not scenes:
        raise FileNotFoundError(f"No scene folders found in {src_root}")
    for scene in scenes:
        prepare_scene(scene, dst_root / scene.name)


if __name__ == "__main__":
    main()
