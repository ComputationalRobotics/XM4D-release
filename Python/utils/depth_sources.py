from pathlib import Path
import re

import cv2
import numpy as np


def load_np_depth_like(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path).astype(np.float32)
    if suffix == ".npz":
        data = np.load(path)
        for key in ("depth", "depths", "depth_map", "depth_maps", "conf", "confidence", "arr_0"):
            if key in data:
                return data[key].astype(np.float32)
        raise KeyError(f"No depth/conf array key found in {path}")
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise FileNotFoundError(f"Could not read image array: {path}")
        if arr.ndim == 3:
            arr = arr[..., 0]
        return arr.astype(np.float32)
    raise ValueError(f"Unsupported depth/conf file: {path}")


def sorted_array_files(directory: Path) -> list[Path]:
    exts = {".npy", ".npz", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in exts
        and path.name not in {"depth_maps.npy", "conf_maps.npy", "metadata.json"}
    )


def frame_sort_key(path: Path) -> tuple[int, str]:
    nums = re.findall(r"\d+(?:\.\d+)?", path.stem)
    if not nums:
        return (10**12, path.name)
    try:
        value = float(nums[-1])
    except ValueError:
        value = 10**12
    return (int(round(value * 1_000_000)), path.name)


def sorted_frame_files(directory: Path, pattern: str = "*") -> list[Path]:
    if not directory.is_dir():
        return []
    exts = {".npy", ".npz", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    return sorted(
        (
            path
            for path in directory.glob(pattern)
            if path.is_file()
            and path.suffix.lower() in exts
            and path.name not in {"depth_maps.npy", "conf_maps.npy", "metadata.json"}
        ),
        key=frame_sort_key,
    )


def load_vggt_depth_and_conf(depth_dir: Path, conf_dir: Path) -> tuple[np.ndarray, np.ndarray | None]:
    stacked_depth_path = depth_dir / "depth_maps.npy"
    if stacked_depth_path.is_file():
        depth_maps = np.load(stacked_depth_path).astype(np.float32)
    else:
        depth_files = sorted_array_files(depth_dir)
        if not depth_files:
            raise FileNotFoundError(
                f"No VGGT depth files found. Tried {stacked_depth_path} and per-frame files in {depth_dir}"
            )
        depth_maps = np.stack([load_np_depth_like(path) for path in depth_files], axis=0)

    if depth_maps.ndim != 3:
        raise ValueError(f"Expected VGGT depth maps shape (T,H,W), got {depth_maps.shape}")

    conf_maps = None
    stacked_conf_candidates = [
        conf_dir / "conf_maps.npy",
        conf_dir / "depth_conf.npy",
        depth_dir / "conf_maps.npy",
    ]
    stacked_conf_path = next((path for path in stacked_conf_candidates if path.is_file()), None)
    if stacked_conf_path is not None:
        conf_maps = np.load(stacked_conf_path).astype(np.float32)
    else:
        conf_files = sorted_array_files(conf_dir)
        if conf_files:
            conf_maps = np.stack([load_np_depth_like(path) for path in conf_files], axis=0)

    if conf_maps is not None:
        if conf_maps.ndim != 3:
            raise ValueError(f"Expected VGGT confidence maps shape (T,H,W), got {conf_maps.shape}")
        if conf_maps.shape[1:] != depth_maps.shape[1:]:
            raise ValueError(
                f"VGGT confidence shape {conf_maps.shape} does not match depth shape {depth_maps.shape}"
            )
        if conf_maps.shape[0] < depth_maps.shape[0]:
            print(
                f"[warn] VGGT confidence has fewer frames than depth: "
                f"{conf_maps.shape[0]} < {depth_maps.shape[0]}"
            )

    return depth_maps, conf_maps


def load_spatracker_depth(depth_dir: Path) -> tuple[np.ndarray, None]:
    depth_files = sorted_frame_files(depth_dir, "*_pred.*")
    if not depth_files:
        depth_files = sorted_frame_files(depth_dir)
    if not depth_files:
        raise FileNotFoundError(f"No SpaTracker depth files found in {depth_dir}")
    depth_maps = np.stack([load_np_depth_like(path) for path in depth_files], axis=0).astype(np.float32)
    if depth_maps.ndim != 3:
        raise ValueError(f"Expected SpaTracker depth maps shape (T,H,W), got {depth_maps.shape}")
    return depth_maps, None


def load_unidepth_depth(depth_dir: Path) -> tuple[np.ndarray, None]:
    depth_files = sorted_frame_files(depth_dir)
    if not depth_files:
        raise FileNotFoundError(f"No UniDepth depth files found in {depth_dir}")
    depth_maps = np.stack([load_np_depth_like(path) for path in depth_files], axis=0).astype(np.float32)
    if depth_maps.ndim != 3:
        raise ValueError(f"Expected UniDepth depth maps shape (T,H,W), got {depth_maps.shape}")
    return depth_maps, None


def load_depth_source(depth_source: str, definitions) -> tuple[np.ndarray, np.ndarray | None, Path]:
    source = depth_source.strip().lower()
    if source in {"da3", "depth_anything", "depthanything", "depth_anything_3"}:
        # Honor the selected depth variant (definitions.active_depth_maps -> outputs/depths/<depth_variant>/
        # depth_maps.npy); falls back to the da3 slot (depth_anything_3/depth_maps.npy) when no variant
        # catalog exists, so non-Dycheck datasets are unchanged.
        depth_path = Path(getattr(definitions, "active_depth_maps",
                                  Path(definitions.da3_depth_dir) / "depth_maps.npy"))
        if not depth_path.exists():
            raise FileNotFoundError(f"No DepthAnything depth file found at {depth_path}")
        depth_maps = np.load(str(depth_path)).astype(np.float32)
        conf_path = Path(definitions.da3_depth_dir) / "conf_maps.npy"
        conf_maps = np.load(str(conf_path)).astype(np.float32) if conf_path.exists() else None
        return depth_maps, conf_maps, depth_path
    if source == "vggt":
        depth_maps, conf_maps = load_vggt_depth_and_conf(
            Path(definitions.vggt_depth_dir),
            Path(definitions.vggt_conf_dir),
        )
        return depth_maps, conf_maps, Path(definitions.vggt_depth_dir)
    if source in {"spatracker", "spa_tracker", "spatrack"}:
        depth_maps, conf_maps = load_spatracker_depth(Path(definitions.spatracker_depth_dir))
        return depth_maps, conf_maps, Path(definitions.spatracker_depth_dir)
    if source in {"unidepth", "uni_depth"}:
        depth_maps, conf_maps = load_unidepth_depth(Path(definitions.unidepth_depth_dir))
        return depth_maps, conf_maps, Path(definitions.unidepth_depth_dir)
    raise ValueError(
        f"Unsupported depth source '{depth_source}'. "
        "Use da3, vggt, spatracker, unidepth, or gt where supported."
    )
