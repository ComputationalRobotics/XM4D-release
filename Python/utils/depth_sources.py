from pathlib import Path
import re

import numpy as np


def frame_sort_key(path: Path) -> tuple[int, str]:
    nums = re.findall(r"\d+(?:\.\d+)?", path.stem)
    if not nums:
        return (10**12, path.name)
    try:
        value = float(nums[-1])
    except ValueError:
        value = 10**12
    return (int(round(value * 1_000_000)), path.name)


def load_depth_source(depth_source: str, definitions) -> tuple[np.ndarray, np.ndarray | None, Path]:
    """Step 1's depth of a video (<depth_dir>/depth_maps.npy: outputs/depth_anything_3/ or outputs/unidepth/, see
    DEPTH_MODEL) and its confidence (None if missing). depth_source is "da3" (kept for the callers)."""
    depth_path = Path(definitions.depth_dir) / "depth_maps.npy"
    if not depth_path.exists():
        raise FileNotFoundError(f"No step-1 depth file found at {depth_path}")
    depth_maps = np.load(str(depth_path)).astype(np.float32)
    conf_path = Path(definitions.depth_dir) / "conf_maps.npy"
    conf_maps = np.load(str(conf_path)).astype(np.float32) if conf_path.exists() else None
    return depth_maps, conf_maps, depth_path
