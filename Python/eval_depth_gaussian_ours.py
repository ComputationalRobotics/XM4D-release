#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
GSPLAT_ROOT = REPO_ROOT / "deps" / "gsplat"
if str(GSPLAT_ROOT) not in sys.path:
    sys.path.insert(0, str(GSPLAT_ROOT))

from gsplat.cuda._torch_impl import _eval_sh_bases_fast  # noqa: E402
from gsplat.rendering import rasterization  # noqa: E402
from utils import definitions as defs  # noqa: E402

SINTEL_DPT_TAG = 202021.25
DEFAULT_SCENE_DIR = Path(defs.rootpath)
DEFAULT_OUTPUT_DIR = Path(defs.output_dir)
DEFAULT_GT_DEPTH_DIR = Path(defs.gt_depth_dir)
DEFAULT_DA3_DEPTH_DIR = Path(defs.da3_depth_dir)
DEFAULT_GAUSSIAN_CKPT_DIR = DEFAULT_OUTPUT_DIR / "gsplat_edg_target_0000" / "ckpts"


class AppearanceOptModule(torch.nn.Module):
    def __init__(
        self,
        n: int,
        feature_dim: int,
        embed_dim: int,
        sh_degree: int,
        mlp_width: int,
        mlp_depth: int,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.sh_degree = int(sh_degree)
        self.embeds = torch.nn.Embedding(n, embed_dim)
        layers: list[torch.nn.Module] = [
            torch.nn.Linear(embed_dim + feature_dim + (sh_degree + 1) ** 2, mlp_width),
            torch.nn.ReLU(inplace=True),
        ]
        for _ in range(mlp_depth - 1):
            layers.extend([torch.nn.Linear(mlp_width, mlp_width), torch.nn.ReLU(inplace=True)])
        layers.append(torch.nn.Linear(mlp_width, 3))
        self.color_head = torch.nn.Sequential(*layers)

    def forward(
        self,
        features: torch.Tensor,
        embed_ids: torch.Tensor | None,
        dirs: torch.Tensor,
        sh_degree: int,
        embed_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        C, N = dirs.shape[:2]
        if embed_override is not None:
            embeds = embed_override.to(features.device).reshape(1, self.embed_dim).expand(C, -1)
        elif embed_ids is None:
            embeds = torch.zeros(C, self.embed_dim, device=features.device)
        else:
            embeds = self.embeds(embed_ids)
        embeds = embeds[:, None, :].expand(-1, N, -1)
        features = features[None, :, :].expand(C, -1, -1)
        dirs = F.normalize(dirs, dim=-1)
        num_bases_to_use = (int(sh_degree) + 1) ** 2
        num_bases = (self.sh_degree + 1) ** 2
        sh_bases = torch.zeros(C, N, num_bases, device=features.device)
        sh_bases[:, :, :num_bases_to_use] = _eval_sh_bases_fast(num_bases_to_use, dirs)
        if self.embed_dim > 0:
            h = torch.cat([embeds, features, sh_bases], dim=-1)
        else:
            h = torch.cat([features, sh_bases], dim=-1)
        return self.color_head(h)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render expected depth from saved gsplat gaussians and evaluate normalized L1 against GT depth."
    )
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE_DIR)
    parser.add_argument(
        "--gaussians",
        type=Path,
        default=None,
        help=f"Final .pt file or ckpt directory. Default: {DEFAULT_GAUSSIAN_CKPT_DIR}",
    )
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--gt-depth-dir", type=Path, default=None, help=f"Default: {DEFAULT_GT_DEPTH_DIR}")
    parser.add_argument(
        "--camera-source",
        choices=["colmap", "xm"],
        default="colmap",
        help="Camera pose source. Intrinsics always come from GT by image name.",
    )
    parser.add_argument("--gt-scale", type=float, default=1000.0)
    parser.add_argument("--normalize-stat", choices=["median", "mean"], default="median")
    parser.add_argument("--min-depth", type=float, default=1e-6)
    parser.add_argument("--max-depth", type=float, default=-1.0)
    parser.add_argument("--alpha_threshold", type=float, default=0.9)
    parser.add_argument(
        "--depth-smooth",
        choices=["none", "median", "bilateral"],
        default="bilateral",
        help="Post-filter rendered Gaussian depth before evaluation/visualization.",
    )
    parser.add_argument("--smooth-ksize", type=int, default=50, help="Odd filter window size for depth smoothing.")
    parser.add_argument("--bilateral-sigma-color", type=float, default=0.2)
    parser.add_argument("--bilateral-sigma-space", type=float, default=5.0)
    parser.add_argument("--near-plane", type=float, default=0.01)
    parser.add_argument("--far-plane", type=float, default=1e10)
    parser.add_argument("--sh-degree", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-edg-warp", action="store_true", help="Disable EDG mean warping when the checkpoint contains EDG state.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--vis-out", type=Path, default=None)
    parser.add_argument("--rendered-depth-out", type=Path, default=None)
    parser.add_argument(
        "--save-depth-png",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save PNG depth visualizations for every train/novel view.",
    )
    parser.add_argument(
        "--save-depth-arrays",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Optionally save rendered depth, GT depth, alpha, and normalized error arrays as .npz files.",
    )
    parser.add_argument("--eval-novel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--novel-dir", type=Path, default=None, help="Default: <scene>/novel")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _safe_torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _resolve_gaussian_paths(scene_dir: Path, gaussians: Path | None) -> list[Path]:
    candidates: list[Path] = []
    if gaussians is not None:
        target = gaussians.resolve()
        if target.is_dir():
            paths = sorted(target.glob("final_rank*.pt"))
            if paths:
                return paths
            paths = sorted(target.glob("ckpt_*_rank*.pt"), key=_ckpt_sort_key)
            if paths:
                last_step = _ckpt_sort_key(paths[-1])[0]
                return [p for p in paths if _ckpt_sort_key(p)[0] == last_step]
            raise FileNotFoundError(f"No final_rank*.pt or ckpt_*_rank*.pt files in {target}")
        if not target.is_file():
            raise FileNotFoundError(f"Missing gaussian checkpoint: {target}")
        return [target]

    candidates.extend(
        [
            scene_dir / "outputs" / "gsplat_edg_target_0000" / "final_gaussians.pt",
            scene_dir / "outputs" / "gsplat" / "final_gaussians.pt",
            scene_dir / "outputs" / "gaussians" / "final_gaussians.pt",
            scene_dir / "final_gaussians.pt",
        ]
    )
    if scene_dir.resolve() == DEFAULT_SCENE_DIR.resolve():
        candidates.append(DEFAULT_OUTPUT_DIR / "gsplat_edg_target_0000" / "final_gaussians.pt")
    for path in candidates:
        if path.is_file():
            return [path]
    ckpt_dirs = [
        scene_dir / "outputs" / "gsplat_edg_target_0000" / "ckpts",
        scene_dir / "outputs" / "gsplat" / "ckpts",
        scene_dir / "outputs" / "gaussians" / "ckpts",
        scene_dir / "ckpts",
    ]
    if scene_dir.resolve() == DEFAULT_SCENE_DIR.resolve():
        ckpt_dirs.append(DEFAULT_GAUSSIAN_CKPT_DIR)
    for ckpt_dir in ckpt_dirs:
        if ckpt_dir.is_dir():
            paths = sorted(ckpt_dir.glob("final_rank*.pt"))
            if paths:
                return paths
            paths = sorted(ckpt_dir.glob("ckpt_*_rank*.pt"), key=_ckpt_sort_key)
            if paths:
                last_step = _ckpt_sort_key(paths[-1])[0]
                return [p for p in paths if _ckpt_sort_key(p)[0] == last_step]
    raise FileNotFoundError("Could not find final gaussians; pass --gaussians.")


def _ckpt_sort_key(path: Path) -> tuple[int, int]:
    match = re.search(r"ckpt_(\d+)_rank(\d+)\.pt$", path.name)
    if match is None:
        return (-1, -1)
    return int(match.group(1)), int(match.group(2))


def _load_gaussians(
    paths: list[Path],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None, dict[str, torch.Tensor] | None, int]:
    ckpts = [_safe_torch_load(path, device) for path in paths]
    splat_keys = ckpts[0]["splats"].keys()
    splats = {
        key: torch.cat([ckpt["splats"][key].to(device) for ckpt in ckpts], dim=0)
        for key in splat_keys
    }
    edg = ckpts[0].get("edg_warp")
    if edg is not None:
        edg = {key: value.to(device) if torch.is_tensor(value) else value for key, value in edg.items()}
    app_state = ckpts[0].get("app_module")
    if app_state is not None:
        app_state = {key: value.to(device) if torch.is_tensor(value) else value for key, value in app_state.items()}
    return splats, edg, app_state, int(ckpts[0].get("step", -1))


def _build_app_module(
    app_state: dict[str, torch.Tensor] | None,
    splats: dict[str, torch.Tensor],
    device: torch.device,
) -> AppearanceOptModule | None:
    if app_state is None or "features" not in splats:
        return None
    state = {key.removeprefix("module."): value for key, value in app_state.items()}
    if "embeds.weight" not in state:
        return None

    n_images, embed_dim = state["embeds.weight"].shape
    feature_dim = int(splats["features"].shape[-1])
    linear_weights = sorted(
        (key for key in state if key.startswith("color_head.") and key.endswith(".weight")),
        key=lambda key: int(key.split(".")[1]),
    )
    if not linear_weights:
        return None
    first_weight = state[linear_weights[0]]
    mlp_width = int(first_weight.shape[0])
    input_dim = int(first_weight.shape[1])
    sh_bases = input_dim - int(embed_dim) - feature_dim
    sh_degree = int(round(sh_bases**0.5)) - 1
    if (sh_degree + 1) ** 2 != sh_bases:
        raise ValueError(f"Cannot infer app-module SH degree from input dim {input_dim}")
    mlp_depth = max(len(linear_weights) - 1, 1)

    module = AppearanceOptModule(
        n=int(n_images),
        feature_dim=feature_dim,
        embed_dim=int(embed_dim),
        sh_degree=sh_degree,
        mlp_width=mlp_width,
        mlp_depth=mlp_depth,
    ).to(device)
    module.load_state_dict(state)
    module.eval()
    return module


def _image_files(scene_dir: Path) -> list[Path]:
    image_dir = scene_dir / "images"
    return sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"})


def _depth_files(depth_dir: Path) -> list[Path]:
    skip = {"depth_maps.npy", "conf_maps.npy", "intrinsics.npy", "extrinsics_c2w.npy"}
    exts = {".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    rank = {".npy": 0, ".png": 1, ".tif": 2, ".tiff": 2, ".jpg": 3, ".jpeg": 3}
    by_stem: dict[str, Path] = {}
    for path in depth_dir.iterdir():
        suffix = path.suffix.lower()
        if not path.is_file() or suffix not in exts or path.name in skip:
            continue
        old = by_stem.get(path.stem)
        if old is None or rank[suffix] < rank[old.suffix.lower()]:
            by_stem[path.stem] = path
    return sorted(by_stem.values(), key=lambda p: p.stem)


def _gt_depth_path(scene_dir: Path, gt_depth_dir: Path, frame_index: int, image_name: str | None = None) -> Path:
    images = _image_files(scene_dir)
    gt_files = _depth_files(gt_depth_dir)
    if frame_index < 0 or frame_index >= len(images):
        raise IndexError(f"frame_index={frame_index} outside images [0,{len(images)-1}]")
    by_stem = {p.stem: p for p in gt_files}
    image_stem = Path(image_name).stem if image_name is not None else images[frame_index].stem
    if image_stem in by_stem:
        return by_stem[image_stem]
    if frame_index < len(gt_files):
        return gt_files[frame_index]
    raise FileNotFoundError(f"No GT depth for frame {frame_index} in {gt_depth_dir}")


def _read_depth(path: Path, scale: float) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path).astype(np.float64)
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Could not read depth image: {path}")
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth = depth.astype(np.float64)
        if scale > 0:
            depth /= float(scale)
        return depth
    raise ValueError(f"Unsupported GT depth file {path}")


def _resolve_depth_path(path: Path) -> Path:
    if path.is_file():
        return path
    npy_path = path.with_suffix(".npy")
    if npy_path.is_file():
        return npy_path
    if path.with_suffix(".png").is_file():
        return path.with_suffix(".png")
    if path.parent.is_dir():
        matches = sorted(
            p for p in path.parent.iterdir()
            if p.is_file() and p.suffix.lower() in {".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff"} and p.stem == path.stem
        )
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No per-frame GT depth found for {path}")


def _load_gt(scene_dir: Path) -> dict[str, Any]:
    dataset_cat = scene_dir.parent.name
    if dataset_cat == "TUM":
        from utils.readgt_TUM import load_tum_gt

        return load_tum_gt(str(scene_dir))
    elif dataset_cat == "Sintel":
        from utils.readgt_sintel import load_sintel_gt

        return load_sintel_gt(str(scene_dir))
    elif dataset_cat == "Replica":
        from utils.readgt_replica import load_replica_gt

        return load_replica_gt(str(scene_dir))
    elif dataset_cat == "Syn4D":
        return _load_syn4d_colmap_gt(scene_dir / "colmap")
    elif dataset_cat == "Dycheck":
        from utils.readgt_dycheck import load_dycheck_gt

        return load_dycheck_gt(str(scene_dir))
    else:
        raise RuntimeError(f"No GT reader configured for dataset category {dataset_cat}")


def _frame_id_from_name(name: str) -> int:
    match = re.search(r"(?<!\d)(\d+)$", Path(name).stem)
    if match is None:
        raise ValueError(f"Could not extract frame id from image name: {name}")
    return int(match.group(1))


def _load_syn4d_colmap_gt(colmap_dir: Path) -> dict[str, Any]:
    sparse_dir = colmap_dir / "sparse"
    if not (sparse_dir / "images.txt").is_file() and (sparse_dir / "0" / "images.txt").is_file():
        sparse_dir = sparse_dir / "0"
    cameras = _read_colmap_cameras_txt(sparse_dir / "cameras.txt")
    results: dict[str, Any] = {}
    with (sparse_dir / "images.txt").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            name = " ".join(parts[9:])
            camera_id = int(parts[8])
            if camera_id not in cameras:
                raise KeyError(f"Camera {camera_id} from {sparse_dir / 'images.txt'} missing in cameras.txt")
            R = _qvec_to_rot([float(v) for v in parts[1:5]])
            t = np.array([float(v) for v in parts[5:8]], dtype=np.float64)
            _, _, _, K = cameras[camera_id]
            results[name] = {
                "id": _frame_id_from_name(name),
                "K": K,
                "R": R,
                "t": t,
                "camera_id": camera_id,
            }
    return results


def _load_gt_info(scene_dir: Path, frame_index: int) -> dict[str, Any]:
    gt = _load_gt(scene_dir)
    for _, info in gt.items():
        if int(info["id"]) == int(frame_index):
            return info
    raise RuntimeError(f"Could not find GT camera for frame {frame_index}")


def _load_gt_intrinsics(scene_dir: Path, image_name: str) -> np.ndarray:
    gt = _load_gt(scene_dir)
    key = image_name
    if key not in gt:
        basename = Path(image_name).name
        if basename in gt:
            key = basename
        else:
            by_basename = {Path(name).name: name for name in gt}
            if basename in by_basename:
                key = by_basename[basename]
            else:
                raise KeyError(f"Could not find GT intrinsics for image {image_name}")
    K = gt[key]["K"]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _qvec_to_rot(qvec: list[float]) -> np.ndarray:
    qw, qx, qy, qz = [float(x) for x in qvec]
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _camera_K_from_colmap(model: str, params: list[float]) -> np.ndarray:
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params[:3]
        fx = fy = f
    elif model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE"}:
        fx, fy, cx, cy = params[:4]
    elif model in {"SIMPLE_RADIAL", "RADIAL"}:
        f, cx, cy = params[:3]
        fx = fy = f
    else:
        if len(params) < 4:
            raise ValueError(f"Unsupported COLMAP camera model {model} with params {params}")
        fx, fy, cx, cy = params[:4]
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _training_data_dir_from_cfg(scene_dir: Path) -> Path | None:
    cfg_path = scene_dir / "outputs" / "gsplat_edg_target_0000" / "cfg.yml"
    if not cfg_path.is_file() and scene_dir.resolve() == DEFAULT_SCENE_DIR.resolve():
        cfg_path = DEFAULT_OUTPUT_DIR / "gsplat_edg_target_0000" / "cfg.yml"
    if not cfg_path.is_file():
        return None
    for line in cfg_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("data_dir:"):
            value = line.split(":", 1)[1].strip()
            if value:
                return Path(value)
    return None


def _read_colmap_cameras_txt(path: Path) -> dict[int, tuple[str, int, int, np.ndarray]]:
    cameras: dict[int, tuple[str, int, int, np.ndarray]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(v) for v in parts[4:]]
            cameras[cam_id] = (model, width, height, _camera_K_from_colmap(model, params))
    return cameras


def _colmap_sparse_candidates(scene_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    # XM_SPARSE_OVERRIDE: highest-priority sparse dir (e.g. when evaluating a model trained on a
    # dataset copy with refined poses). Unset -> behavior unchanged.
    override = os.environ.get("XM_SPARSE_OVERRIDE")
    if override:
        candidates.append(Path(override))
    train_data_dir = _training_data_dir_from_cfg(scene_dir)
    if train_data_dir is not None:
        candidates.extend([train_data_dir / "sparse" / "0", train_data_dir / "sparse"])
    candidates.extend([
        scene_dir / "outputs" / "gs_edg_target_0000" / "sparse" / "0",
        scene_dir / "outputs" / "gs_edg_target_0000" / "sparse",
        scene_dir / "outputs" / "depth_anything_3" / "sparse",
        scene_dir / "colmap" / "sparse" / "0",
        scene_dir / "colmap" / "sparse",
        scene_dir / "sparse",
        scene_dir / "sparse" / "0",
        scene_dir / "output_colmap" / "sparse",
        scene_dir / "output_colmap" / "mapper" / "0",
        scene_dir / "output_vggt" / "sparse",
        scene_dir / "outputs" / "colmap" / "sparse",
    ])
    seen: set[Path] = set()
    unique = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _load_colmap_frame(scene_dir: Path, frame_index: int) -> tuple[np.ndarray, np.ndarray, Path, str] | None:
    for sparse_dir in _colmap_sparse_candidates(scene_dir):
        images_txt = sparse_dir / "images.txt"
        cameras_txt = sparse_dir / "cameras.txt"
        if not images_txt.is_file() or not cameras_txt.is_file():
            continue
        cameras = _read_colmap_cameras_txt(cameras_txt)
        records: list[list[str]] = []
        with images_txt.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 10:
                    continue
                records.append(parts)
        if frame_index < 0 or frame_index >= len(records):
            raise IndexError(f"frame_index={frame_index} outside COLMAP images [0,{len(records)-1}] in {images_txt}")
        parts = records[frame_index]
        name = " ".join(parts[9:])
        camera_id = int(parts[8])
        if camera_id not in cameras:
            raise KeyError(f"Camera {camera_id} from {images_txt} missing in {cameras_txt}")
        R = _qvec_to_rot([float(v) for v in parts[1:5]])
        t = np.array([float(v) for v in parts[5:8]], dtype=np.float64)
        w2c = np.eye(4, dtype=np.float64)
        # Match deps/gsplat/examples/datasets/colmap.py Parser.
        w2c[:3, :3] = R.T
        w2c[:3, 3] = t
        _, colmap_width, colmap_height, K = cameras[camera_id]
        image_path = scene_dir / "images" / name
        if not image_path.is_file():
            image_path = scene_dir / "images" / Path(name).name
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        actual_height, actual_width = image.shape[:2]
        K = K.copy()
        K[0, :] *= actual_width / float(colmap_width)
        K[1, :] *= actual_height / float(colmap_height)
        return np.linalg.inv(w2c).astype(np.float32), K.astype(np.float32), sparse_dir, name
    return None


def _load_colmap_training_poses(scene_dir: Path) -> tuple[list[np.ndarray], list[str]]:
    for sparse_dir in _colmap_sparse_candidates(scene_dir):
        images_txt = sparse_dir / "images.txt"
        cameras_txt = sparse_dir / "cameras.txt"
        if not images_txt.is_file() or not cameras_txt.is_file():
            continue
        c2ws: list[np.ndarray] = []
        names: list[str] = []
        with images_txt.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 10:
                    continue
                R = _qvec_to_rot([float(v) for v in parts[1:5]])
                t = np.array([float(v) for v in parts[5:8]], dtype=np.float64)
                w2c = np.eye(4, dtype=np.float64)
                w2c[:3, :3] = R.T
                w2c[:3, 3] = t
                c2ws.append(np.linalg.inv(w2c))
                names.append(" ".join(parts[9:]))
        if c2ws:
            return c2ws, names
    raise FileNotFoundError("Missing COLMAP text images.txt in known sparse locations.")


def _load_camera_colmap(scene_dir: Path, frame_index: int) -> tuple[np.ndarray, str, str] | None:
    loaded = _load_colmap_frame(scene_dir, frame_index)
    if loaded is None:
        return None
    c2w, _, sparse_dir, image_name = loaded
    return c2w, str(sparse_dir), image_name


def _load_camera_gt(scene_dir: Path, frame_index: int) -> np.ndarray:
    info = _load_gt_info(scene_dir, frame_index)
    R = np.asarray(info["R"], dtype=np.float64)
    t = np.asarray(info["t"], dtype=np.float64).reshape(3)
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R
    w2c[:3, 3] = t
    return np.linalg.inv(w2c).astype(np.float32)


def _load_camera_xm(scene_dir: Path, frame_index: int) -> tuple[np.ndarray, str] | None:
    path = scene_dir / "outputs" / "result_XM.npz"
    if not path.is_file():
        path = scene_dir / "result_XM.npz"
    if not path.is_file():
        return None
    images = _image_files(scene_dir)
    if frame_index < 0 or frame_index >= len(images):
        raise IndexError(f"frame_index={frame_index} outside images [0,{len(images)-1}]")
    data = np.load(path)
    R = np.asarray(data["R"], dtype=np.float64)[:, 3 * frame_index : 3 * (frame_index + 1)]
    C = np.asarray(data["t"], dtype=np.float64)[:, frame_index]
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R
    c2w[:3, 3] = C
    return c2w.astype(np.float32), images[frame_index].name


def _load_xm_training_poses(scene_dir: Path) -> tuple[list[np.ndarray], list[str]]:
    path = scene_dir / "outputs" / "result_XM.npz"
    if not path.is_file():
        path = scene_dir / "result_XM.npz"
    if not path.is_file():
        raise FileNotFoundError("Missing result_XM.npz for XM camera pose.")
    data = np.load(path)
    images = _image_files(scene_dir)
    n_frames = min(len(images), int(np.asarray(data["t"]).shape[1]))
    c2ws: list[np.ndarray] = []
    names: list[str] = []
    for frame_index in range(n_frames):
        R = np.asarray(data["R"], dtype=np.float64)[:, 3 * frame_index : 3 * (frame_index + 1)]
        C = np.asarray(data["t"], dtype=np.float64)[:, frame_index]
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = R
        c2w[:3, 3] = C
        c2ws.append(c2w)
        names.append(images[frame_index].name)
    return c2ws, names


def _load_camera_and_gt_intrinsics(
    scene_dir: Path,
    frame_index: int,
    source: str,
) -> tuple[np.ndarray, np.ndarray, str, str | None, str]:
    if source == "colmap":
        loaded = _load_colmap_frame(scene_dir, frame_index)
        if loaded is not None:
            # Use the intrinsic from the training sparse/cameras.txt (the one the
            # gaussians were trained with), NOT GT — they can differ (e.g. Syn4D
            # bears trained with DA3 intrinsics). Using GT here misaligns the render.
            c2w, K, sparse_dir, image_name = loaded
            return c2w, K, "colmap", str(sparse_dir), image_name
        raise FileNotFoundError("Missing COLMAP text cameras.txt/images.txt in known sparse locations.")
    if source == "xm":
        loaded_xm = _load_camera_xm(scene_dir, frame_index)
        if loaded_xm is None:
            raise FileNotFoundError("Missing result_XM.npz for XM camera pose.")
        c2w, image_name = loaded_xm
        K = _load_gt_intrinsics(scene_dir, image_name)
        return c2w, K, "xm", None, image_name
    raise ValueError(f"Unsupported camera source: {source}")


def _splat_colors(splats: dict[str, torch.Tensor], sh_degree: int) -> tuple[torch.Tensor, int | None]:
    if "sh0" in splats:
        sh0 = splats["sh0"]
        shn = splats.get("shN")
        if shn is None:
            shn = torch.empty((sh0.shape[0], 0, 3), device=sh0.device, dtype=sh0.dtype)
        degree = min(int(sh_degree), int(round((sh0.shape[1] + shn.shape[1]) ** 0.5)) - 1)
        return torch.cat([sh0, shn], dim=1), degree
    if "colors" in splats:
        colors = torch.sigmoid(splats["colors"])
        if colors.ndim == 2:
            return colors, None
        return colors, int(sh_degree)
    return torch.ones((splats["means"].shape[0], 3), device=splats["means"].device), None


def _warp_means_edg(means: torch.Tensor, edg: dict[str, torch.Tensor] | None, frame_index: int) -> torch.Tensor:
    if edg is None or "dyn_indices" not in edg or "edg_tracks" not in edg or "edg_rots" not in edg:
        return means
    tracks = edg["edg_tracks"]
    rots = edg["edg_rots"]
    if frame_index < 0 or frame_index >= tracks.shape[1]:
        return means
    dyn_indices = edg["dyn_indices"].long()
    valid = (dyn_indices >= 0) & (dyn_indices < means.shape[0])
    if not bool(valid.any()):
        return means
    dyn_indices = dyn_indices[valid]
    dyn_knn_idx = edg["dyn_knn_idx"].long()[valid]
    dyn_rel = edg["dyn_rel"].float()[valid]
    if "dyn_knn_logits" in edg:
        dyn_knn_w = torch.softmax(edg["dyn_knn_logits"].float()[valid], dim=1)
    else:
        dyn_knn_w = edg["dyn_knn_w"].float()[valid]
    edg_pos = tracks[:, frame_index, :][dyn_knn_idx]
    edg_rot = rots[:, frame_index, :, :][dyn_knn_idx]
    warped = edg_pos + torch.matmul(edg_rot, dyn_rel.unsqueeze(-1)).squeeze(-1)
    dyn_xyz = torch.sum(warped * dyn_knn_w.unsqueeze(-1), dim=1)
    out = means.clone()
    out[dyn_indices] = dyn_xyz.to(dtype=means.dtype)
    return out


def _quat_to_rotmat_b(q: torch.Tensor) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    N = q.shape[0]
    R = torch.empty((N, 3, 3), device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _rotmat_to_quat_b(R: torch.Tensor) -> torch.Tensor:
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    w = torch.sqrt(torch.clamp(1.0 + m00 + m11 + m22, min=1e-8)) * 0.5
    x = torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=1e-8)) * 0.5
    y = torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=1e-8)) * 0.5
    z = torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=1e-8)) * 0.5
    x = torch.copysign(x, R[:, 2, 1] - R[:, 1, 2])
    y = torch.copysign(y, R[:, 0, 2] - R[:, 2, 0])
    z = torch.copysign(z, R[:, 1, 0] - R[:, 0, 1])
    q = torch.stack([w, x, y, z], dim=-1)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _warp_quats_edg(quats: torch.Tensor, edg: dict[str, torch.Tensor] | None,
                    frame_index: int, target_frame: int) -> torch.Tensor:
    """Rotate dynamic Gaussian orientations by the EDG surface rotation from target_frame to frame_index
    (argmax-weight node's relative rotation). Mirrors simple_trainer.EDGWarp.warp_quats for eval parity."""
    if edg is None or "dyn_indices" not in edg or "edg_rots" not in edg:
        return quats
    rots = edg["edg_rots"]
    if frame_index < 0 or frame_index >= rots.shape[1]:
        return quats
    dyn_indices = edg["dyn_indices"].long()
    valid = (dyn_indices >= 0) & (dyn_indices < quats.shape[0])
    if not bool(valid.any()):
        return quats
    dyn_indices = dyn_indices[valid]
    dyn_knn_idx = edg["dyn_knn_idx"].long()[valid]
    if "dyn_knn_logits" in edg:
        dyn_knn_w = torch.softmax(edg["dyn_knn_logits"].float()[valid], dim=1)
    else:
        dyn_knn_w = edg["dyn_knn_w"].float()[valid]
    amax = torch.argmax(dyn_knn_w, dim=1)
    node = dyn_knn_idx[torch.arange(dyn_knn_idx.shape[0], device=dyn_knn_idx.device), amax]
    R_t = rots[node, frame_index].float()
    R_tg = rots[node, int(target_frame)].float()
    R_rel = torch.matmul(R_t, R_tg.transpose(1, 2))
    new_q = _rotmat_to_quat_b(torch.matmul(R_rel, _quat_to_rotmat_b(quats[dyn_indices].float())))
    out = quats.clone()
    out[dyn_indices] = new_q.to(out.dtype)
    return out


@torch.no_grad()
def _render_depth(
    splats: dict[str, torch.Tensor],
    edg: dict[str, torch.Tensor] | None,
    app_module: AppearanceOptModule | None,
    frame_index: int,
    c2w: np.ndarray,
    K: np.ndarray,
    height: int,
    width: int,
    args: argparse.Namespace,
    app_embed_mode: str = "base",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = splats["means"].device
    means = splats["means"].float()
    quats_use = splats["quats"].float()
    if not args.no_edg_warp:
        means = _warp_means_edg(means, edg, int(frame_index))
        if getattr(args, "edg_warp_quats", False):
            quats_use = _warp_quats_edg(quats_use, edg, int(frame_index),
                                        int(getattr(args, "edg_target_frame", 0)))
    camtoworld = torch.from_numpy(c2w).float().to(device)[None]
    if app_module is not None:
        # app_embed_mode: 'base' = this frame's learned exposure embedding (default);
        # 'zero' = neutral (no embedding); 'mean' = average over all training embeddings.
        # For NOVEL views, base imposes a training frame's exposure on a different physical
        # test camera -> a self-inflicted offset; 'zero'/'mean' render a neutral appearance.
        image_ids: torch.Tensor | None = torch.tensor([int(frame_index)], device=device, dtype=torch.long)
        embed_override = None
        if app_embed_mode == "zero":
            image_ids = None
        elif app_embed_mode == "mean":
            image_ids = None
            embed_override = app_module.embeds.weight.detach().mean(dim=0)
        colors = app_module(
            features=splats["features"].float(),
            embed_ids=image_ids,
            dirs=means[None, :, :] - camtoworld[:, None, :3, 3],
            sh_degree=min(int(args.sh_degree), app_module.sh_degree),
            embed_override=embed_override,
        )
        colors = torch.sigmoid(colors + splats["colors"].float())
        sh_degree = None
    else:
        colors, sh_degree = _splat_colors(splats, int(args.sh_degree))
    renders, alphas, _ = rasterization(
        means=means,
        quats=quats_use,
        scales=torch.exp(splats["scales"].float()),
        opacities=torch.sigmoid(splats["opacities"].float()),
        colors=colors.float(),
        viewmats=torch.linalg.inv(camtoworld),
        Ks=torch.from_numpy(K).float().to(device)[None],
        width=int(width),
        height=int(height),
        sh_degree=sh_degree,
        near_plane=float(args.near_plane),
        far_plane=float(args.far_plane),
        render_mode="RGB+ED",
        rasterize_mode="antialiased" if getattr(args, "antialiased", False) else "classic",
    )
    rgb = renders[0, ..., :3].clamp(0.0, 1.0).detach().cpu().numpy().astype(np.float32)
    depth = renders[0, ..., 3].detach().cpu().numpy().astype(np.float64)
    alpha = alphas[0, ..., 0].detach().cpu().numpy().astype(np.float64)
    return depth, alpha, rgb


def _normalizer(values: np.ndarray, method: str) -> float:
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return 1.0
    scale = float(np.mean(values) if method == "mean" else np.median(values))
    if not np.isfinite(scale) or scale <= 1e-12:
        return 1.0
    return scale


def _evaluate(gt: np.ndarray, est: np.ndarray, alpha: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    valid = _valid_mask_gaussian(gt, est, alpha, args)
    if not np.any(valid):
        raise ValueError("No valid depth pixels after masking.")

    gt_normalizer = _normalizer(gt[valid], args.normalize_stat)
    est_normalizer = _normalizer(est[valid], args.normalize_stat)
    diff = np.abs(gt[valid] / gt_normalizer - est[valid] / est_normalizer)
    return {
        "num_valid_pixels": int(diff.size),
        "normalize_stat": args.normalize_stat,
        "gt_normalizer": float(gt_normalizer),
        "est_normalizer": float(est_normalizer),
        "l1": float(np.mean(diff)),
    }


def _valid_mask_gaussian(gt: np.ndarray, est: np.ndarray, alpha: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    valid = np.isfinite(gt) & np.isfinite(est) & np.isfinite(alpha)
    valid &= gt > float(args.min_depth)
    valid &= est > float(args.min_depth)
    valid &= alpha > float(args.alpha_threshold)
    if float(args.max_depth) > 0:
        valid &= gt <= float(args.max_depth)
        valid &= est <= float(args.max_depth)
    return valid


def _evaluate_with_normalizers(
    gt: np.ndarray,
    est: np.ndarray,
    alpha: np.ndarray,
    gt_normalizer: float,
    est_normalizer: float,
    args: argparse.Namespace,
) -> tuple[float, int]:
    valid = _valid_mask_gaussian(gt, est, alpha, args)
    if not np.any(valid):
        return 0.0, 0
    diff = np.abs(gt[valid] / max(gt_normalizer, 1e-12) - est[valid] / max(est_normalizer, 1e-12))
    return float(np.median(diff))*int(diff.size), int(diff.size)


def _smooth_depth(depth: np.ndarray, alpha: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    method = args.depth_smooth
    if method == "none":
        return depth
    valid = np.isfinite(depth) & (depth > float(args.min_depth)) & (alpha > float(args.alpha_threshold))
    if float(args.max_depth) > 0:
        valid &= depth <= float(args.max_depth)
    if not np.any(valid):
        return depth

    ksize = max(1, int(args.smooth_ksize))
    if ksize % 2 == 0:
        ksize += 1

    fill_value = float(np.median(depth[valid]))
    depth_filled = depth.astype(np.float32).copy()
    depth_filled[~valid] = fill_value

    normalizer = max(fill_value, 1e-6)
    depth_norm = (depth_filled / normalizer).astype(np.float32)
    if method == "median":
        smoothed_norm = cv2.medianBlur(depth_norm, ksize)
    else:
        smoothed_norm = cv2.bilateralFilter(
            depth_norm,
            d=ksize,
            sigmaColor=float(args.bilateral_sigma_color),
            sigmaSpace=float(args.bilateral_sigma_space),
        )
    smoothed = smoothed_norm.astype(np.float64) * normalizer
    out = depth.copy()
    out[valid] = smoothed[valid]
    return out


def _normalize_for_vis(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return np.zeros_like(depth, dtype=np.float32)
    vals = depth[mask]
    vmin = float(np.percentile(vals, 2.0))
    vmax = float(np.percentile(vals, 98.0))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return np.clip((depth - vmin) / (vmax - vmin), 0.0, 1.0).astype(np.float32)


def _colorize(gray: np.ndarray) -> np.ndarray:
    img = (np.clip(gray, 0.0, 1.0) * 255).astype(np.uint8)
    return cv2.applyColorMap(img, cv2.COLORMAP_TURBO)


def _write_vis(
    gt: np.ndarray,
    est: np.ndarray,
    alpha: np.ndarray,
    rendered_rgb: np.ndarray,
    summary: dict[str, Any],
    args: argparse.Namespace,
    vis_out: Path,
) -> None:
    valid = np.isfinite(gt) & np.isfinite(est) & (gt > args.min_depth) & (est > args.min_depth) & (alpha > args.alpha_threshold)
    if args.max_depth > 0:
        valid &= (gt <= args.max_depth) & (est <= args.max_depth)
    gt_scaled = gt / max(float(summary["gt_normalizer"]), 1e-12)
    est_scaled = est / max(float(summary["est_normalizer"]), 1e-12)
    err = np.zeros_like(gt_scaled)
    err[valid] = np.abs(gt_scaled[valid] - est_scaled[valid])
    panels = [
        _colorize(_normalize_for_vis(gt_scaled, valid)),
        _colorize(_normalize_for_vis(est_scaled, valid)),
        _colorize(_normalize_for_vis(err, valid)),
        cv2.cvtColor((np.clip(rendered_rgb, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
    ]
    label_h = 28
    labels = np.zeros((label_h, gt.shape[1] * len(panels), 3), dtype=np.uint8)
    for i, text in enumerate(("GT normalized", "Gaussian normalized", "|error|", "Rendered RGB")):
        cv2.putText(labels, text, (i * gt.shape[1] + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    vis = np.concatenate([labels, np.concatenate(panels, axis=1)], axis=0)
    vis_out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(vis_out), vis)


def _save_depth_arrays(
    out_path: Path,
    *,
    gt: np.ndarray,
    rendered: np.ndarray,
    alpha: np.ndarray,
    valid: np.ndarray,
    gt_normalizer: float,
    est_normalizer: float,
    meta: dict[str, Any],
) -> None:
    gt_scaled = gt / max(float(gt_normalizer), 1e-12)
    rendered_scaled = rendered / max(float(est_normalizer), 1e-12)
    error = np.abs(gt_scaled - rendered_scaled)
    error = np.where(valid, error, np.nan)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        gt_depth=gt.astype(np.float32),
        rendered_depth=rendered.astype(np.float32),
        alpha=alpha.astype(np.float32),
        valid_mask=valid.astype(bool),
        error=error.astype(np.float32),
        gt_depth_normalized=gt_scaled.astype(np.float32),
        rendered_depth_normalized=rendered_scaled.astype(np.float32),
        gt_normalizer=np.array(float(gt_normalizer), dtype=np.float32),
        est_normalizer=np.array(float(est_normalizer), dtype=np.float32),
        meta=np.array(meta, dtype=object),
    )


def _dataset_frame_count(scene_dir: Path, source: str) -> int:
    if source == "colmap":
        for sparse_dir in _colmap_sparse_candidates(scene_dir):
            images_txt = sparse_dir / "images.txt"
            cameras_txt = sparse_dir / "cameras.txt"
            if not images_txt.is_file() or not cameras_txt.is_file():
                continue
            count = 0
            with images_txt.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and len(line.split()) >= 10:
                        count += 1
            if count:
                return count
        raise FileNotFoundError("Missing COLMAP text cameras.txt/images.txt in known sparse locations.")
    return len(_image_files(scene_dir))


def _render_eval_frame(
    scene_dir: Path,
    gt_dir: Path,
    frame_index: int,
    splats: dict[str, torch.Tensor],
    edg: dict[str, torch.Tensor] | None,
    app_module: AppearanceOptModule | None,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, Path, np.ndarray, str, str | None]:
    c2w, K, camera_source, sparse_dir, image_name = _load_camera_and_gt_intrinsics(
        scene_dir,
        int(frame_index),
        args.camera_source,
    )
    gt_path = _gt_depth_path(scene_dir, gt_dir, int(frame_index), image_name=image_name)
    gt = _read_depth(gt_path, float(args.gt_scale))
    rendered, alpha, rendered_rgb = _render_depth(
        splats,
        edg,
        app_module,
        int(frame_index),
        c2w,
        K,
        gt.shape[0],
        gt.shape[1],
        args,
    )
    rendered = _smooth_depth(rendered, alpha, args)
    return gt, rendered, alpha, rendered_rgb, image_name, gt_path, K, camera_source, sparse_dir


def _evaluate_dataset_gaussian(
    scene_dir: Path,
    gt_dir: Path,
    splats: dict[str, torch.Tensor],
    edg: dict[str, torch.Tensor] | None,
    app_module: AppearanceOptModule | None,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, Path, np.ndarray, str, str | None]]:
    num_frames = _dataset_frame_count(scene_dir, args.camera_source)
    gt_chunks: list[np.ndarray] = []
    est_chunks: list[np.ndarray] = []
    selected = None
    rendered_frames = []
    for frame_idx in range(num_frames):
        frame_data = _render_eval_frame(scene_dir, gt_dir, frame_idx, splats, edg, app_module, args)
        gt, est, alpha, *_ = frame_data
        valid = _valid_mask_gaussian(gt, est, alpha, args)
        if np.any(valid):
            gt_chunks.append(gt[valid])
            est_chunks.append(est[valid])
        if frame_idx == int(args.frame_index):
            selected = frame_data
        rendered_frames.append(frame_data)
    if selected is None:
        raise IndexError(f"frame_index={args.frame_index} outside dataset [0,{num_frames - 1}]")
    if not gt_chunks:
        if args.save_depth_png:
            train_png_dir = (args.output_dir or (scene_dir / "outputs" / "gaussian_depth_eval")).resolve() / "train_depth_png"
            for frame_idx, frame_data in enumerate(rendered_frames):
                gt, est, alpha, rendered_rgb, _, _, _, _, _ = frame_data
                _write_vis(
                    gt,
                    est,
                    alpha,
                    rendered_rgb,
                    {"gt_normalizer": 1.0, "est_normalizer": 1.0},
                    args,
                    train_png_dir / f"train_depth_eval_{frame_idx:04d}.png",
                )
        if args.save_depth_arrays:
            train_arrays_dir = (args.output_dir or (scene_dir / "outputs" / "gaussian_depth_eval")).resolve() / "train_depth_arrays"
            for frame_idx, frame_data in enumerate(rendered_frames):
                gt, est, alpha, _, image_name, gt_path, _, _, _ = frame_data
                valid = _valid_mask_gaussian(gt, est, alpha, args)
                _save_depth_arrays(
                    train_arrays_dir / f"train_depth_{frame_idx:04d}.npz",
                    gt=gt,
                    rendered=est,
                    alpha=alpha,
                    valid=valid,
                    gt_normalizer=1.0,
                    est_normalizer=1.0,
                    meta={
                        "frame_index": int(frame_idx),
                        "image_name": image_name,
                        "gt_depth": str(gt_path),
                        "l1_frame": None,
                        "num_valid_pixels": 0,
                    },
                )
        raise ValueError("No valid depth pixels found over the dataset.")
    gt_normalizer = _normalizer(np.concatenate(gt_chunks), args.normalize_stat)
    est_normalizer = _normalizer(np.concatenate(est_chunks), args.normalize_stat)
    total_abs = 0.0
    total_valid = 0
    frames_used = 0
    train_arrays_dir = None
    if args.save_depth_arrays:
        train_arrays_dir = (args.output_dir or (scene_dir / "outputs" / "gaussian_depth_eval")).resolve() / "train_depth_arrays"
    train_png_dir = None
    if args.save_depth_png:
        train_png_dir = (args.output_dir or (scene_dir / "outputs" / "gaussian_depth_eval")).resolve() / "train_depth_png"
    for frame_idx, frame_data in enumerate(rendered_frames):
        gt, est, alpha, rendered_rgb, image_name, gt_path, _, _, _ = frame_data
        diff_sum, n_valid = _evaluate_with_normalizers(gt, est, alpha, gt_normalizer, est_normalizer, args)
        if train_arrays_dir is not None:
            valid = _valid_mask_gaussian(gt, est, alpha, args)
            _save_depth_arrays(
                train_arrays_dir / f"train_depth_{frame_idx:04d}.npz",
                gt=gt,
                rendered=est,
                alpha=alpha,
                valid=valid,
                gt_normalizer=gt_normalizer,
                est_normalizer=est_normalizer,
                meta={
                    "frame_index": int(frame_idx),
                    "image_name": image_name,
                    "gt_depth": str(gt_path),
                    "l1_frame": None if n_valid == 0 else float(diff_sum / n_valid),
                    "num_valid_pixels": int(n_valid),
                },
            )
        if train_png_dir is not None:
            _write_vis(
                gt,
                est,
                alpha,
                rendered_rgb,
                {"gt_normalizer": gt_normalizer, "est_normalizer": est_normalizer},
                args,
                train_png_dir / f"train_depth_eval_{frame_idx:04d}.png",
            )
        if n_valid == 0:
            continue
        total_abs += diff_sum
        total_valid += n_valid
        frames_used += 1
    summary = {
        "num_frames": num_frames,
        "num_frames_used": frames_used,
        "num_valid_pixels": total_valid,
        "normalize_stat": args.normalize_stat,
        "normalizer_scope": "dataset",
        "gt_normalizer": float(gt_normalizer),
        "est_normalizer": float(est_normalizer),
        "l1": total_abs / max(total_valid, 1),
        "depth_arrays_dir": None if train_arrays_dir is None else str(train_arrays_dir),
        "depth_png_dir": None if train_png_dir is None else str(train_png_dir),
    }
    return summary, selected


def _gt_c2w_from_info(info: dict[str, Any]) -> np.ndarray:
    R = np.asarray(info["R"], dtype=np.float64)
    t = np.asarray(info["t"], dtype=np.float64).reshape(3)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t
    return c2w


def _load_training_gt_pose_pairs(scene_dir: Path, source: str) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    if source == "colmap":
        est_c2ws, names = _load_colmap_training_poses(scene_dir)
    else:
        est_c2ws, names = _load_xm_training_poses(scene_dir)
    gt = _load_gt(scene_dir)
    by_basename = {Path(name).name: name for name in gt}
    est_used: list[np.ndarray] = []
    gt_used: list[np.ndarray] = []
    names_used: list[str] = []
    for est_c2w, name in zip(est_c2ws, names, strict=False):
        key = name if name in gt else by_basename.get(Path(name).name)
        if key is None:
            continue
        est_used.append(np.asarray(est_c2w, dtype=np.float64))
        gt_used.append(_gt_c2w_from_info(gt[key]))
        names_used.append(name)
    if len(est_used) < 2:
        raise RuntimeError(f"Need at least two matched train/GT poses for alignment, got {len(est_used)}")
    return est_used, gt_used, names_used


def _project_rotation(mat: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(mat)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def _svd_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src_mean = np.mean(src, axis=1, keepdims=True)
    dst_mean = np.mean(dst, axis=1, keepdims=True)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    H = dst_c @ src_c.T
    R = _project_rotation(H)
    denom = float(np.sum(src_c * src_c))
    scale = float(np.sum(dst_c * (R @ src_c)) / max(denom, 1e-12))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    t = dst_mean - scale * R @ src_mean
    return scale, R, t


def _teaser_or_svd_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, str]:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    finite = np.isfinite(src).all(axis=0) & np.isfinite(dst).all(axis=0)
    src = src[:, finite]
    dst = dst[:, finite]
    if src.shape[1] < 3:
        s, R, t = _svd_similarity(src, dst)
        return s, R, t, "svd_too_few"
    src_center = np.mean(src, axis=1, keepdims=True)
    dst_center = np.mean(dst, axis=1, keepdims=True)
    src_spread = float(np.mean(np.linalg.norm(src - src_center, axis=0)))
    dst_spread = float(np.mean(np.linalg.norm(dst - dst_center, axis=0)))
    if src_spread <= 1e-12 or dst_spread <= 1e-12:
        s, R, t = _svd_similarity(src, dst)
        return s, R, t, "svd_degenerate"
    try:
        import teaserpp_python

        params = teaserpp_python.RobustRegistrationSolver.Params()
        params.cbar2 = 1
        params.noise_bound = 0.3
        params.estimate_scaling = False
        params.rotation_estimation_algorithm = (
            teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
        )
        params.rotation_gnc_factor = 1.4
        params.rotation_max_iterations = 100
        params.rotation_cost_threshold = 1e-12
        solver = teaserpp_python.RobustRegistrationSolver(params)
        solver.solve(src / src_spread, dst / dst_spread)
        solution = solver.getSolution()
        s = dst_spread / src_spread
        R = np.asarray(solution.rotation, dtype=np.float64)
        t = np.mean(dst - s * R @ src, axis=1, keepdims=True)
        if np.isfinite(R).all() and np.isfinite(t).all():
            return float(s), R, t, "teaser"
    except Exception:
        pass
    s, R, t = _svd_similarity(src, dst)
    return s, R, t, "svd_fallback"


def _rotation_alignment(est_c2ws: list[np.ndarray], gt_c2ws: list[np.ndarray]) -> np.ndarray:
    mats = [gt[:3, :3] @ est[:3, :3].T for est, gt in zip(est_c2ws, gt_c2ws, strict=True)]
    return _project_rotation(np.sum(mats, axis=0))


def _alignment_est_to_gt(scene_dir: Path, source: str) -> tuple[float, np.ndarray, np.ndarray, str]:
    est_c2ws, gt_c2ws, _ = _load_training_gt_pose_pairs(scene_dir, source)
    src = np.stack([pose[:3, 3] for pose in est_c2ws], axis=1)
    dst = np.stack([pose[:3, 3] for pose in gt_c2ws], axis=1)
    s, R, t, method = _teaser_or_svd_similarity(src, dst)
    R_rot = _rotation_alignment(est_c2ws, gt_c2ws)
    err_pos_rot = np.mean([np.linalg.norm((R @ est[:3, :3]) - gt[:3, :3]) for est, gt in zip(est_c2ws, gt_c2ws, strict=True)])
    err_rot = np.mean([np.linalg.norm((R_rot @ est[:3, :3]) - gt[:3, :3]) for est, gt in zip(est_c2ws, gt_c2ws, strict=True)])
    src_center = src - np.mean(src, axis=1, keepdims=True)
    rank = np.linalg.matrix_rank(src_center)
    if rank < 2 or err_rot + 1e-6 < err_pos_rot:
        R = R_rot
        t = np.mean(dst - s * R @ src, axis=1, keepdims=True)
        method = f"{method}+rotation"
    return float(s), R, t, method


def _quat_wxyz_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    return _qvec_to_rot([qw, qx, qy, qz])


def _load_novel_metadata(scene_dir: Path, novel_dir: Path) -> list[dict[str, Any]]:
    meta_path = novel_dir / "metadata.txt"
    if not meta_path.is_file():
        return []
    records = []
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 11:
            continue
        novel_id = int(parts[0])
        base_frame = int(parts[1])
        image_name = parts[2]
        depth_name = parts[3]
        tx, ty, tz = map(float, parts[4:7])
        qw, qx, qy, qz = map(float, parts[7:11])
        c2w_gt = np.eye(4, dtype=np.float64)
        c2w_gt[:3, :3] = _quat_wxyz_to_rot(qw, qx, qy, qz)
        c2w_gt[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
        depth_path = _resolve_depth_path(novel_dir / depth_name)
        records.append(
            {
                "novel_id": novel_id,
                "base_frame": base_frame,
                "image_name": image_name,
                "depth_path": depth_path,
                "image_path": novel_dir / image_name,
                "c2w_gt": c2w_gt,
            }
        )
    return records


def _gt_key_for_frame(scene_dir: Path, frame_index: int) -> str:
    gt = _load_gt(scene_dir)
    for name, info in gt.items():
        if int(info["id"]) == int(frame_index):
            return name
    images = _image_files(scene_dir)
    return images[frame_index].name


def _transform_gt_c2w_to_est(c2w_gt: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R.T @ c2w_gt[:3, :3]
    c2w[:3, 3] = (R.T @ ((c2w_gt[:3, 3].reshape(3, 1) - t) / max(float(s), 1e-12))).reshape(3)
    return c2w


def _evaluate_novel_gaussian(
    scene_dir: Path,
    splats: dict[str, torch.Tensor],
    edg: dict[str, torch.Tensor] | None,
    app_module: AppearanceOptModule | None,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any] | None:
    novel_dir = (args.novel_dir or (scene_dir / "novel")).resolve()
    if not args.eval_novel or not novel_dir.is_dir():
        return None
    records = _load_novel_metadata(scene_dir, novel_dir)
    if not records:
        return None

    s, R_align, t_align, align_method = _alignment_est_to_gt(scene_dir, args.camera_source)
    gt_chunks: list[np.ndarray] = []
    est_chunks: list[np.ndarray] = []
    rendered_frames: list[tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    for rec in records:
        base_key = _gt_key_for_frame(scene_dir, int(rec["base_frame"]))
        K = _load_gt_intrinsics(scene_dir, base_key)
        gt_depth = _read_depth(Path(rec["depth_path"]), float(args.gt_scale))
        c2w_est = _transform_gt_c2w_to_est(rec["c2w_gt"], s, R_align, t_align)
        rendered, alpha, rendered_rgb = _render_depth(
            splats,
            edg,
            app_module,
            int(rec["base_frame"]),
            c2w_est,
            K,
            gt_depth.shape[0],
            gt_depth.shape[1],
            args,
        )
        rendered = _smooth_depth(rendered, alpha, args)
        valid = _valid_mask_gaussian(gt_depth, rendered, alpha, args)
        if np.any(valid):
            gt_chunks.append(gt_depth[valid])
            est_chunks.append(rendered[valid])
        rendered_frames.append((rec, gt_depth, rendered, alpha, rendered_rgb))

    novel_out = out_dir / "novel"
    novel_out.mkdir(parents=True, exist_ok=True)
    novel_arrays_dir = novel_out / "depth_arrays" if args.save_depth_arrays else None
    novel_png_dir = novel_out / "depth_png" if args.save_depth_png else None
    if not gt_chunks:
        if novel_png_dir is not None:
            for rec, gt_depth, rendered, alpha, rendered_rgb in rendered_frames:
                _write_vis(
                    gt_depth,
                    rendered,
                    alpha,
                    rendered_rgb,
                    {"gt_normalizer": 1.0, "est_normalizer": 1.0},
                    args,
                    novel_png_dir / f"novel_depth_eval_{int(rec['novel_id']):04d}.png",
                )
        if novel_arrays_dir is not None:
            for rec, gt_depth, rendered, alpha, _ in rendered_frames:
                valid = _valid_mask_gaussian(gt_depth, rendered, alpha, args)
                _save_depth_arrays(
                    novel_arrays_dir / f"novel_depth_{int(rec['novel_id']):04d}.npz",
                    gt=gt_depth,
                    rendered=rendered,
                    alpha=alpha,
                    valid=valid,
                    gt_normalizer=1.0,
                    est_normalizer=1.0,
                    meta={
                        "novel_id": int(rec["novel_id"]),
                        "base_frame": int(rec["base_frame"]),
                        "depth_path": str(rec["depth_path"]),
                        "image_path": str(rec["image_path"]),
                        "l1_frame": None,
                        "num_valid_pixels": 0,
                    },
                )
        return {
            "novel_dir": str(novel_dir),
            "novel_num_frames": len(records),
            "novel_num_frames_used": 0,
            "novel_l1": None,
            "novel_l1_avg_frame": None,
            "novel_alignment_method": align_method,
            "novel_depth_arrays_dir": None if novel_arrays_dir is None else str(novel_arrays_dir),
            "novel_depth_png_dir": None if novel_png_dir is None else str(novel_png_dir),
        }

    gt_normalizer = _normalizer(np.concatenate(gt_chunks), args.normalize_stat)
    est_normalizer = _normalizer(np.concatenate(est_chunks), args.normalize_stat)
    total_abs = 0.0
    total_valid = 0
    frame_l1s: list[float] = []

    for rec, gt_depth, rendered, alpha, rendered_rgb in rendered_frames:
        diff_sum, n_valid = _evaluate_with_normalizers(
            gt_depth,
            rendered,
            alpha,
            gt_normalizer,
            est_normalizer,
            args,
        )
        if novel_arrays_dir is not None:
            valid = _valid_mask_gaussian(gt_depth, rendered, alpha, args)
            _save_depth_arrays(
                novel_arrays_dir / f"novel_depth_{int(rec['novel_id']):04d}.npz",
                gt=gt_depth,
                rendered=rendered,
                alpha=alpha,
                valid=valid,
                gt_normalizer=gt_normalizer,
                est_normalizer=est_normalizer,
                meta={
                    "novel_id": int(rec["novel_id"]),
                    "base_frame": int(rec["base_frame"]),
                    "depth_path": str(rec["depth_path"]),
                    "image_path": str(rec["image_path"]),
                    "l1_frame": None if n_valid == 0 else float(diff_sum / n_valid),
                    "num_valid_pixels": int(n_valid),
                },
            )
        if n_valid == 0:
            continue
        total_abs += diff_sum
        total_valid += n_valid
        frame_l1s.append(diff_sum / n_valid)
        if novel_png_dir is not None:
            _write_vis(
                gt_depth,
                rendered,
                alpha,
                rendered_rgb,
                {"gt_normalizer": gt_normalizer, "est_normalizer": est_normalizer},
                args,
                novel_png_dir / f"novel_depth_eval_{int(rec['novel_id']):04d}.png",
            )

    novel_l1 = total_abs / max(total_valid, 1)
    novel_l1_avg_frame = float(np.mean(frame_l1s)) if frame_l1s else None
    print(
        f"Novel-view avg L1: {novel_l1_avg_frame:.6f} "
        f"(global {novel_l1:.6f}, frames {len(frame_l1s)}/{len(records)})"
    )
    return {
        "novel_dir": str(novel_dir),
        "novel_num_frames": len(records),
        "novel_num_frames_used": len(frame_l1s),
        "novel_num_valid_pixels": total_valid,
        "novel_normalizer_scope": "novel_dataset",
        "novel_gt_normalizer": float(gt_normalizer),
        "novel_est_normalizer": float(est_normalizer),
        "novel_l1": float(novel_l1),
        "novel_l1_avg_frame": novel_l1_avg_frame,
        "novel_alignment_scale": float(s),
        "novel_alignment_method": align_method,
        "novel_depth_arrays_dir": None if novel_arrays_dir is None else str(novel_arrays_dir),
        "novel_depth_png_dir": None if novel_png_dir is None else str(novel_png_dir),
    }


def main() -> None:
    args = _parse_args()
    scene_dir = args.scene_dir.resolve()
    if args.gt_depth_dir is not None:
        gt_dir = args.gt_depth_dir.resolve()
    elif scene_dir == DEFAULT_SCENE_DIR.resolve():
        gt_dir = DEFAULT_GT_DEPTH_DIR.resolve()
    else:
        gt_dir = (scene_dir / "depths").resolve()
    out_dir = (args.output_dir or (scene_dir / "outputs" / "gaussian_depth_eval")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    gaussian_paths = _resolve_gaussian_paths(scene_dir, args.gaussians)
    splats, edg, app_state, step = _load_gaussians(gaussian_paths, device)
    app_module = _build_app_module(app_state, splats, device)

    summary, selected = _evaluate_dataset_gaussian(
        scene_dir,
        gt_dir,
        splats,
        edg,
        app_module,
        args,
    )
    gt, rendered, alpha, rendered_rgb, image_name, gt_path, K, camera_source, sparse_dir = selected

    vis_out = args.vis_out or (out_dir / f"gaussian_depth_eval_frame{args.frame_index:04d}.png")
    json_out = args.output_json or (out_dir / f"gaussian_depth_eval.json")
    rendered_out = None
    if args.rendered_depth_out is not None:
        rendered_out = args.rendered_depth_out
        rendered_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(rendered_out, rendered.astype(np.float32))
    if args.save_depth_png and args.vis_out is not None:
        _write_vis(gt, rendered, alpha, rendered_rgb, summary, args, vis_out)
    novel_summary = _evaluate_novel_gaussian(scene_dir, splats, edg, app_module, args, out_dir)

    result: dict[str, Any] = {
        "scene_dir": str(scene_dir),
        "frame_index": int(args.frame_index),
        "image_name": image_name,
        "gaussians": [str(p) for p in gaussian_paths],
        "step": step,
        "gt_depth": str(gt_path),
        "gt_scale": float(args.gt_scale),
        "camera_source": camera_source,
        "sparse_dir": sparse_dir,
        "uses_app_module": app_module is not None,
        "intrinsics": K.tolist(),
        "min_depth": float(args.min_depth),
        "max_depth": float(args.max_depth),
        "alpha_threshold": float(args.alpha_threshold),
        "depth_smooth": args.depth_smooth,
        "smooth_ksize": int(args.smooth_ksize),
        "rendered_depth_out": None if rendered_out is None else str(rendered_out),
        "vis_out": None if not (args.save_depth_png and args.vis_out is not None) else str(vis_out),
        **summary,
    }
    if novel_summary is not None:
        result.update(novel_summary)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json:
        print("RESULT_JSON:", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
