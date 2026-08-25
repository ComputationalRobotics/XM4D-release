from pathlib import Path
import shutil


DATASETS_ROOT = Path("/home/fenchel/4DReconstruction/datasets")
DEPTH_ROOT = DATASETS_ROOT / "MPI-Sintel-depth-training-20150305" / "training" / "depth"
IMAGE_ROOT = DATASETS_ROOT / "MPI-Sintel-training_images" / "training" / "final"
MASK_ROOT = DATASETS_ROOT / "Sintel-RGB-clean"
OUT_ROOT = DATASETS_ROOT / "Sintel"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEPTH_EXTS = {".dpt"}


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _collect_files(src_dir: Path, exts: set[str] | None = None):
    if not src_dir.exists():
        return []
    files = []
    for path in src_dir.iterdir():
        if not path.is_file():
            continue
        if exts is not None and path.suffix.lower() not in exts:
            continue
        files.append(path)
    return sorted(files)


def _copy_files(src_files: list[Path], dst_dir: Path):
    _ensure_dir(dst_dir)
    for src in src_files:
        shutil.copy2(src, dst_dir / src.name)


def _mask_scene_dir(scene: str) -> Path:
    return MASK_ROOT / f"{scene}_masks"


def _scene_names():
    image_scenes = {p.name for p in IMAGE_ROOT.iterdir() if p.is_dir()}
    depth_scenes = {p.name for p in DEPTH_ROOT.iterdir() if p.is_dir()}
    mask_scenes = {
        p.name[:-6]
        for p in MASK_ROOT.iterdir()
        if p.is_dir() and p.name.endswith("_masks")
    }
    return sorted(image_scenes & depth_scenes & mask_scenes)


def create_sintel_scene(scene: str):
    image_src = IMAGE_ROOT / scene
    depth_src = DEPTH_ROOT / scene
    mask_src = _mask_scene_dir(scene)

    image_files = _collect_files(image_src, IMAGE_EXTS)
    depth_files = _collect_files(depth_src, DEPTH_EXTS)
    mask_files = _collect_files(mask_src, IMAGE_EXTS)

    if not image_files or not depth_files or not mask_files:
        print(
            f"[skip] {scene}: "
            f"images={len(image_files)}, depths={len(depth_files)}, masks={len(mask_files)}"
        )
        return

    scene_root = OUT_ROOT / scene
    images_out = scene_root / "images"
    depths_out = scene_root / "depths"
    masks_out = scene_root / "masks"
    cameras_out = scene_root / "cameras"
    outputs_out = scene_root / "outputs"

    for path in (images_out, depths_out, masks_out, cameras_out, outputs_out):
        _ensure_dir(path)

    _copy_files(image_files, images_out)
    _copy_files(depth_files, depths_out)
    _copy_files(mask_files, masks_out)

    print(
        f"[done] {scene}: "
        f"{len(image_files)} images, {len(depth_files)} depths, {len(mask_files)} masks"
    )

def main():
    print("Building Sintel dataset layout")
    print(f"  images: {IMAGE_ROOT}")
    print(f"  depths: {DEPTH_ROOT}")
    print(f"  masks:  {MASK_ROOT}")
    print(f"  out:    {OUT_ROOT}")

    scenes = _scene_names()
    print(f"Found {len(scenes)} scenes with image/depth/mask data")

    for scene in scenes:
        create_sintel_scene(scene)


if __name__ == "__main__":
    main()
