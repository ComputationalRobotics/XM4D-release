"""Convert the raw DyCheck iPhone dataset into the repo's dataset layout.

Source (per scene):  ~/Downloads/data/iphone/<scene>/
    images/0_*.png            handheld (moving) training camera
    sensor_depth/0_*.npz      LiDAR depth, key 'dep', float32 metres, 0 = invalid
    cameras/0_*.json          Nerfies camera json (full-res intrinsics)
    test_images/{1,2}_*.png   two static tripod (novel/test) cameras
    test_cameras/{1,2}_*.json
    test_covisible/{1,2}_*.png  co-visibility masks for the test views

Target:  datasets/Dycheck/<scene>/
    images/frameNNNNNN.png
    depth/depthNNNNNN.png     uint16 millimetres (metres * 1000), 0 = invalid
    masks/                    empty (dynamic masks come from SAM2 later)
    cameras/frameNNNNNN.json  copied verbatim (readgt_dycheck rescales K)
    novel/images/novelNNNNNN.png
    novel/cameras/novelNNNNNN.json
    novel/covisible/covisibleNNNNNN.png
    novel/metadata.txt
    frame_map.txt

Camera json is copied verbatim; the intrinsic rescale (json is full-res, images
are downscaled) is applied at load time in utils/readgt_dycheck.py.
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

SCENES = ["apple", "block", "paper-windmill", "space-out", "spin", "teddy", "wheel"]
DEPTH_SCALE = 1000.0  # metres -> millimetres, matches Syn4D uint16 depth PNGs
UINT16_MAX = 65535


def _sorted(paths):
    return sorted(paths, key=lambda p: p.name)


def _depth_npz_to_png(npz_path, png_path):
    dep = np.load(npz_path)["dep"].astype(np.float64)  # metres, 0 = invalid
    mm = np.round(dep * DEPTH_SCALE)
    clipped = int((mm > UINT16_MAX).sum())
    mm = np.clip(mm, 0, UINT16_MAX).astype(np.uint16)
    cv2.imwrite(str(png_path), mm)
    return clipped


def convert_scene(src_scene, dst_scene):
    for sub in ("images", "depth", "masks", "cameras",
                "novel/images", "novel/cameras", "novel/covisible"):
        (dst_scene / sub).mkdir(parents=True, exist_ok=True)

    # --- training views (camera 0) ---
    train_images = _sorted((src_scene / "images").glob("*.png"))
    total_clipped = 0
    with (dst_scene / "frame_map.txt").open("w") as fmap:
        fmap.write("# frame_id image_name depth_name camera_name source_stem\n")
        for i, img in enumerate(train_images):
            stem = img.stem  # e.g. 0_00000
            new_img = f"frame{i:06d}.png"
            new_depth = f"depth{i:06d}.png"
            new_cam = f"frame{i:06d}.json"
            shutil.copy2(img, dst_scene / "images" / new_img)
            shutil.copy2(src_scene / "cameras" / f"{stem}.json",
                         dst_scene / "cameras" / new_cam)
            total_clipped += _depth_npz_to_png(
                src_scene / "sensor_depth" / f"{stem}.npz",
                dst_scene / "depth" / new_depth,
            )
            fmap.write(f"{i} {new_img} {new_depth} {new_cam} {stem}\n")

    # --- novel / test views (cameras 1 and 2) ---
    test_images = _sorted((src_scene / "test_images").glob("*.png"))
    with (dst_scene / "novel" / "metadata.txt").open("w") as meta:
        meta.write("# novel_id image_name camera_name covisible_name source_stem\n")
        for i, img in enumerate(test_images):
            stem = img.stem  # e.g. 1_00008
            new_img = f"novel{i:06d}.png"
            new_cam = f"novel{i:06d}.json"
            new_covis = f"covisible{i:06d}.png"
            shutil.copy2(img, dst_scene / "novel" / "images" / new_img)
            shutil.copy2(src_scene / "test_cameras" / f"{stem}.json",
                         dst_scene / "novel" / "cameras" / new_cam)
            shutil.copy2(src_scene / "test_covisible" / f"{stem}.png",
                         dst_scene / "novel" / "covisible" / new_covis)
            meta.write(f"{i} {new_img} {new_cam} {new_covis} {stem}\n")

    return len(train_images), len(test_images), total_clipped


def main():
    parser = argparse.ArgumentParser(description="Build datasets/Dycheck from the raw iPhone data.")
    parser.add_argument("--src", type=Path, default=Path.home() / "Downloads" / "data" / "iphone")
    parser.add_argument("--dst", type=Path,
                        default=Path(__file__).resolve().parents[1] / "datasets" / "Dycheck")
    parser.add_argument("--scenes", nargs="*", default=SCENES)
    args = parser.parse_args()

    for scene in args.scenes:
        src_scene = args.src / scene
        if not src_scene.is_dir():
            raise FileNotFoundError(f"Source scene not found: {src_scene}")
        dst_scene = args.dst / scene
        n_train, n_novel, clipped = convert_scene(src_scene, dst_scene)
        note = f"  ({clipped} depth px clipped to {UINT16_MAX}mm)" if clipped else ""
        print(f"{scene:16s} train={n_train:4d}  novel={n_novel:4d}{note}")


if __name__ == "__main__":
    main()
