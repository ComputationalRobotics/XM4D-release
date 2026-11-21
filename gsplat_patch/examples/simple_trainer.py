import json
import math
import os
import re
import sys
import time
import atexit
import datetime as _dt
from collections import defaultdict
import imageio.v2 as imageio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import colorsys
import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from gsplat.color_correct import color_correct_affine, color_correct_quadratic
from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)
from fused_ssim import fused_ssim
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed

from gsplat import export_splats
from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.strategy.ops import remove
from gsplat_viewer import GsplatViewer, GsplatRenderTabState
from nerfview import CameraState, RenderTabState, apply_float_colormap

_SIMPLE_TRAINER_LOGGER_FINISH = None


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)


def _install_simple_trainer_logger(result_dir: str) -> Path:
    log_dir = Path(result_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "simple_trainer.log"
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file = log_path.open("w", encoding="utf-8", buffering=1)
    start = time.perf_counter()
    state = {"status": 0, "finished": False}

    print(f"[simple_trainer] log: {log_path}", file=original_stdout, flush=True)
    sys.stdout = _Tee(original_stdout, log_file)
    sys.stderr = _Tee(original_stderr, log_file)
    print("stage: simple_trainer")
    print(f"log_file: {log_path}")
    print(f"start_time: {_dt.datetime.now().isoformat(timespec='seconds')}")
    print(f"command: {' '.join(sys.argv)}")
    print()

    old_excepthook = sys.excepthook

    def _excepthook(exc_type, exc, tb):
        state["status"] = 1
        old_excepthook(exc_type, exc, tb)

    sys.excepthook = _excepthook

    def _finish() -> None:
        if state["finished"]:
            return
        state["finished"] = True
        runtime = time.perf_counter() - start
        print()
        print(f"end_time: {_dt.datetime.now().isoformat(timespec='seconds')}")
        print(f"runtime_seconds: {runtime:.3f}")
        print(f"exit_status: {state['status']}")
        print(f"[simple_trainer] runtime: {runtime:.3f}s")
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()

    atexit.register(_finish)
    global _SIMPLE_TRAINER_LOGGER_FINISH
    _SIMPLE_TRAINER_LOGGER_FINISH = _finish
    return log_path


def _finish_simple_trainer_logger() -> None:
    if _SIMPLE_TRAINER_LOGGER_FINISH is not None:
        _SIMPLE_TRAINER_LOGGER_FINISH()


@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path
    render_traj_path: str = "interp"

    # EDG warp inputs (optional)
    edg_results: Optional[str] = None
    edg_target_frame: int = 0
    # Rotate dynamic Gaussian orientations by the skinned EDG rotation each frame (keeps ellipsoids aligned
    # with the rotating surface -> less rotation blur). Uses rotation the EDG already computes; near-free.
    edg_warp_quats: bool = False
    edg_knn: int = 6
    edg_chunk: int = 4096
    edg_apply_rgb: bool = True
    # Learning rates for trainable EDG parameters (0 = frozen)
    edg_tracks_lr: float = 1e-4
    edg_rots_lr: float = 1e-4
    # Learning rates for dynamic-point skinning attachments (0 = frozen).
    # Weights are optimized as logits and normalized with softmax.
    edg_weights_lr: float = 1e-4
    edg_rel_lr: float = 1e-4
    # EDG learning-rate STEP DECAY: at EACH iteration in edg_lr_decay_steps, multiply every EDG
    # param-group lr (tracks/rots/skinning weights/rel) by edg_lr_decay_gamma. Gammas COMPOUND across
    # milestones (e.g. x0.1 at 15000 and again at 20000 -> 1/100 total). Empty list disables.
    edg_lr_decay_steps: List[int] = field(default_factory=lambda: [15000])
    edg_lr_decay_gamma: float = 0.1
    # EDG ARAP regularization weight (0 = disabled).
    # Penalises deviation from rigid edge lengths using the middle frame as rest pose.
    # NOTE: the ARAP/smooth losses are COUNT-NORMALIZED (means, not sums), so these lambdas are
    # stable across scene size.
    edg_arap_lambda: float = 5e-3
    # ARAP temporal gaps (complete_edg_table L_arap form: rigidity between ALL (t, t+delta) pairs
    # for delta in this list; rest-frame-free).
    edg_arap_gaps: List[int] = field(default_factory=lambda: [1, 2, 4])
    # Weight of the edge-LENGTH residual inside the ARAP loss (complete_edg_table eta_len).
    edg_arap_eta_len: float = 0.1
    # EDG temporal smoothness weight (0 = disabled).
    # Penalises large changes in node positions / rotations between consecutive frames.
    edg_smooth_lambda: float = 1e-1
    # Keep optimized skinning attachments close to EDG-precomputed values.
    edg_skinning_reg_lambda: float = 1e-2
    # OBSERVATION ANCHORS (training-time analogues of complete_edg_table's L_init / L_obs):
    #   edg_obs_lambda -- EDG observation loss: keep the LEARNED node tables (edg_tracks/edg_rots) near
    #                     their solved-table init (which complete_edg_table fit to the 3D track
    #                     observations), so photometric finetuning cannot drift the skeleton to overfit
    #                     training views. Positions whitened by the node-cloud scale, rotations by 0.1rad.
    #   dyn_obs_lambda -- 3D observation loss: keep the WARPED dynamic-Gaussian positions (per frame)
    #                     near their init-warp positions (init EDG + init skinning + init offsets --
    #                     transitively anchored to the 3D observations). Random subsample per step.
    edg_obs_lambda: float = 0.0
    dyn_obs_lambda: float = 0.0
    dyn_obs_samples: int = 16384
    # Delay EDG parameter updates while Gaussian geometry/color stabilizes.
    edg_warmup_steps: int = 2000
    # Clip EDG gradients and skinning offset drift for stability.
    edg_grad_clip: float = 0.1
    edg_rel_delta_max: float = 0.1
    # Let the DYNAMIC (EDG-warped) Gaussians grow as densely as possible:
    #   edg_dyn_no_prune  -- never prune dynamic Gaussians (opacity/too-big prune skips them), so once
    #                        created they persist and the dynamic cloud accumulates.
    #   edg_dyn_force_grow -- force every small dynamic Gaussian to duplicate (and large ones to split)
    #                        at EACH refine step regardless of its 2D gradient, so the dynamic cloud
    #                        densifies to the max_gaussians budget quickly. (Aggressive: can bloat count.)
    edg_dyn_no_prune: bool = False
    edg_dyn_force_grow: bool = False
    # Depth-error pruning: remove Gaussians whose rendered depth disagrees with the DA3
    # monocular depth (x da3_frame_scale) beyond a robust threshold. Cleans wrong-depth
    # floaters that corrupt novel views. Uses DA3 depth (NOT GT) -> no train leakage.
    depth_error_prune: bool = False
    depth_error_prune_start_iter: int = 2000
    depth_error_prune_every: int = 500
    # Single bound on the max fraction of Gaussians removed in ONE prune call. Applies to
    # BOTH the densification prune (opacity + too-big/prune-all-big, largest-scale-first)
    # AND this depth-error prune. Lower it to prune less aggressively.
    depth_error_prune_max_fraction: float = 0.02
    depth_error_prune_std_ratio: float = 3.0
    depth_error_prune_min_error: float = 0.5
    depth_error_prune_min_candidates: int = 100
    depth_error_prune_static_only: bool = False
    eval_depth_min_depth: float = 1e-6
    # Hard cap on every Gaussian's max 3D scale (as a fraction of scene_scale), enforced
    # by clamping EVERY step for the whole run -- not just during the prune phase. Without
    # this, large Gaussians accumulate after refine_stop_iter and corrupt novel-view depth.
    # <=0 disables the cap.
    max_scale3d: float = 0.1
    # Custom eval: render all frames from first camera pose
    custom_eval_first_frame: bool = False
    # Lightweight evaluation: report only PSNR (+ depth L1) and skip the
    # expensive SSIM/LPIPS metrics and per-frame depth-map PNG dumps.
    eval_lite: bool = True
    # Render an interpolated trajectory video at eval steps.
    eval_render_traj: bool = False

    # Dynamic segmentation loss (optional, requires EDG dynamic points).
    # Within the per-frame dynamic mask, penalize color contributed by *static*
    # Gaussians so that dynamic pixels are rendered from dynamic Gaussians only.
    seg_dyn_lambda: float = 0.1
    # Delay the loss so Gaussian geometry/color can stabilize first.
    seg_dyn_start_iter: int = 1000
    # Apply the loss every N steps to amortize its extra render pass (1 = every step).
    seg_dyn_every: int = 4
    # Directory of per-frame binary dynamic masks (frame<idx>.png, white=dynamic).
    # Auto-found by searching upwards from data_dir for a "masks" folder if None.
    seg_dyn_mask_dir: Optional[str] = None

    # --- Dynamic-appearance & mask options ---
    # Render dynamic Gaussians as a single view-independent color: bypass the appearance
    # module and any SH beyond DC for dynamic points (each dynamic Gaussian = one flat RGB).
    dyn_flat_color: bool = True
    # Penalize dynamic-Gaussian color contribution OUTSIDE the per-frame dynamic mask
    # (complement of seg_dyn_lambda, which penalizes static contribution inside the mask).
    # Reuses seg_dyn_start_iter / seg_dyn_every / the dynamic mask dir.
    dyn_outside_lambda: float = 0.1  
    # Supervise the static region (outside the dynamic mask) with a STATIC-ONLY render
    # (dynamic Gaussians excluded) against GT, so static Gaussians alone explain the background.
    static_only_region_lambda: float = 0.1

    debug_save_images = True
    debug_save_every = 800

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image (used for val set; train always uses all frames)
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = False
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"
    # Load EXIF exposure metadata from images (if available)
    load_exposure: bool = True

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [3_000, 7_000, 15_000, 30_000, 50_000, 70_000, 100_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [3_000, 7_000, 15_000, 30_000, 50_000, 70_000, 100_000])
    # Also write the last training checkpoint to stable final file names.
    save_final_gaussians: bool = True
    final_gaussians_name: str = "final_gaussians.pt"
    # Whether to save ply file (storage size can be large)
    save_ply: bool = False
    # Steps to save the model as ply
    ply_steps: List[int] = field(default_factory=lambda: [3_000, 7_000, 15_000, 30_000, 50_000, 70_000, 100_000])
    # Whether to disable video generation during training and evaluation
    disable_video: bool = True
    # Cap expensive evaluation/preview renders. <=0 means render all available frames.
    eval_max_frames: int = 0
    custom_eval_max_frames: int = 0
    traj_max_frames: int = 0
    # Render validation/custom/trajectory previews at this image scale. 1.0 keeps full resolution.
    render_scale: float = 1.0

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Density-based downsample of the SFM init point cloud: keep ~one averaged point per
    # voxel of size (init_voxel_size * scene_scale), so already-dense regions are thinned
    # while sparse regions are preserved. <=0 disables (keep all points). Ignored for random init.
    init_voxel_size: float = 0.0
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 1
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 2000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 0.5
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = True

    # LR for 3D point positions
    means_lr: float = 3e-4
    # LR for Gaussian scale factors
    scales_lr: float = 1e-3
    # LR for alpha blending weights
    opacities_lr: float = 5e-2
    # LR for orientation (quaternions)
    quats_lr: float = 1e-3
    # LR for SH band 0 (brightness)
    sh0_lr: float = 2.5e-3
    # LR for higher-order SH (detail)
    shN_lr: float = 2.5e-3 / 20

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0
    # Hard constraints for per-Gaussian anisotropy ratio max(scale)/min(scale).
    scale_ratio_min: float = 0.0
    scale_ratio_max: float = 100.0
    # One-time cleanup for isolated static initialization points before training.
    static_outlier_prune: bool = False
    static_outlier_k: int = 64
    static_outlier_std_ratio: float = 1.8
    static_outlier_min_points: int = 64
    static_outlier_max_fraction: float = 0.5
    # Gaussian budget (0 = no cap). Densification never grows past this, and if
    # the SfM init exceeds it, the excess is trimmed once at startup by dropping
    # the lowest-opacity *static* Gaussians (all dynamic Gaussians are kept).
    max_gaussians: int = 1000_000

    # Enable camera optimization. (Default ON: helps novel-view registration on DyCheck.)
    pose_opt: bool = True
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-4
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-5
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = True
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Post-processing method for appearance correction (experimental)
    post_processing: Optional[Literal["bilateral_grid", "ppisp"]] = "bilateral_grid"
    # Use fused implementation for bilateral grid (only applies when post_processing="bilateral_grid")
    bilateral_grid_fused: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)
    # Enable PPISP controller
    ppisp_use_controller: bool = True
    # Use controller distillation in PPISP (only applies when post_processing="ppisp" and ppisp_use_controller=True)
    ppisp_controller_distillation: bool = True
    # Controller activation ratio for PPISP (only applies when post_processing="ppisp" and ppisp_use_controller=True)
    ppisp_controller_activation_num_steps: int = 25_000
    # Color correction method for cc_* metrics (only applies when post_processing is set)
    color_correct_method: Literal["affine", "quadratic"] = "affine"
    # Compute color-corrected metrics (cc_psnr, cc_ssim, cc_lpips) during evaluation
    use_color_correction_metric: bool = False

    # Enable depth loss. (Default ON: the EDG dataset ships per-point depth; helps geometry.)
    depth_loss: bool = True
    # Weight for depth loss
    depth_lambda: float = 3e-4

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"

    # 3DGUT (uncented transform + eval 3D)
    with_ut: bool = False
    with_eval3d: bool = False

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
            if strategy.noise_injection_stop_iter >= 0:
                strategy.noise_injection_stop_iter = int(
                    strategy.noise_injection_stop_iter * factor
                )
        else:
            assert_never(strategy)


def _find_upwards(start: Path, rel_paths: List[str], max_depth: int = 4) -> Optional[Path]:
    cur = start.resolve()
    for _ in range(max_depth + 1):
        for rel in rel_paths:
            cand = (cur / rel).resolve()
            if cand.exists():
                return cand
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def _quat_to_rotmat_batch(q: torch.Tensor) -> torch.Tensor:
    """(N,4) wxyz quaternions -> (N,3,3) rotation matrices (normalizes q)."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    N = q.shape[0]
    R = torch.empty((N, 3, 3), device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _rotmat_to_quat_batch(R: torch.Tensor) -> torch.Tensor:
    """(N,3,3) rotation matrices -> (N,4) wxyz quaternions (batched, trace-stable)."""
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    tr = m00 + m11 + m22
    w = torch.sqrt(torch.clamp(1.0 + tr, min=1e-8)) * 0.5
    x = torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=1e-8)) * 0.5
    y = torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=1e-8)) * 0.5
    z = torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=1e-8)) * 0.5
    x = torch.copysign(x, R[:, 2, 1] - R[:, 1, 2])
    y = torch.copysign(y, R[:, 0, 2] - R[:, 2, 0])
    z = torch.copysign(z, R[:, 1, 0] - R[:, 0, 1])
    q = torch.stack([w, x, y, z], dim=-1)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


class EDGWarp:
    def __init__(
        self,
        *,
        data_dir: Path,
        results_npz: Path,
        target_frame: int,
        knn: int,
        chunk: int,
        device: torch.device,
    ) -> None:
        self.device = device
        self.knn = int(knn)
        self.chunk = int(chunk)

        results = np.load(str(results_npz), allow_pickle=True)
        edg_tracks_dict = results["edg_reconstruct"].item()
        edg_rot_dict = results.get("edg_rotation", None)
        if not edg_tracks_dict:
            raise RuntimeError("EDG results are empty; cannot run 4D warp training.")
        if edg_rot_dict is None:
            raise RuntimeError("Missing edg_rotation in results_all.npz; regenerate EDG rotations.")

        edg_tracks, edg_gids = self._as_track_array(edg_tracks_dict)
        self.edg_gids = edg_gids
        self.T = edg_tracks.shape[1]

        # Store edge pairs as index pairs into edg_gids (N_edges, 2). edg_edges_union may be saved either
        # as a 0-d object array wrapping a set/list of (ga,gb) pairs (older EDG) or as a plain (N,2) int
        # array (newer EDG); support both -- .item() on the latter raises "size 1 to a Python scalar".
        gid_to_idx = {int(g): i for i, g in enumerate(edg_gids.tolist())}
        if "edg_edges_union" in results:
            _eu = results["edg_edges_union"]
            edges_union = _eu.item() if (_eu.dtype == object or _eu.ndim == 0) else _eu
        else:
            edges_union = set()
        self.edge_pairs = np.array(
            [[gid_to_idx[int(ga)], gid_to_idx[int(gb)]] for ga, gb in edges_union
             if int(ga) in gid_to_idx and int(gb) in gid_to_idx],
            dtype=np.int64,
        ) if len(edges_union) else np.zeros((0, 2), dtype=np.int64)
        if target_frame < 0 or target_frame >= self.T:
            raise ValueError(f"target_frame {target_frame} out of range [0, {self.T-1}]")
        self.target_frame = int(target_frame)

        rt_path = self._find_xm_results(data_dir)
        if rt_path is None:
            raise FileNotFoundError(f"Missing result_XM.npz near {data_dir}")
        rt = np.load(str(rt_path))
        # The static solver may drop frames (checklandmarks / XM^2), so R/t are stored in
        # solved order (N<=self.T). Scatter them onto the full timeline via indices_all,
        # then fill any dropped frame from its nearest solved neighbor so the target-frame
        # transform and per-frame warps stay valid.
        _N = rt["R"].shape[1] // 3
        _R_all = np.asarray(rt["R"], dtype=np.float64)
        _t_all = np.asarray(rt["t"], dtype=np.float64)
        R_raw = np.tile(np.eye(3), (self.T, 1, 1)).astype(np.float64)
        t_raw = np.full((self.T, 3), np.nan, dtype=np.float64)
        if "indices_all" in rt.files:
            _idx = np.asarray(rt["indices_all"]).astype(np.int64)
            for _oi in range(min(self.T, _idx.shape[0])):
                _j = int(_idx[_oi])
                if 0 <= _j < _N:
                    R_raw[_oi] = _R_all[:, 3 * _j : 3 * (_j + 1)]
                    t_raw[_oi] = _t_all[:, _j]
        else:
            for _oi in range(min(self.T, _N)):
                R_raw[_oi] = _R_all[:, 3 * _oi : 3 * (_oi + 1)]
                t_raw[_oi] = _t_all[:, _oi]
        _finite = np.isfinite(t_raw).all(axis=1)
        if not _finite.all():
            _kept = np.where(_finite)[0]
            if _kept.size == 0:
                raise RuntimeError("result_XM.npz has no valid camera poses.")
            for _i in np.where(~_finite)[0]:
                _jj = int(_kept[np.argmin(np.abs(_kept - _i))])
                R_raw[_i] = R_raw[_jj]
                t_raw[_i] = t_raw[_jj]

        w2t = np.eye(4, dtype=np.float64)
        R = R_raw[self.target_frame]
        t = t_raw[self.target_frame]
        w2t[:3, :3] = R.T
        w2t[:3, 3] = -R.T @ t

        edg_rots, edg_rot_gids = self._as_rotation_array(edg_rot_dict.item())
        if not np.array_equal(edg_gids, edg_rot_gids):
            raise RuntimeError("Mismatch between edg_reconstruct and edg_rotation node ordering.")

        edg_tracks_target = self._apply_transform(edg_tracks.reshape(-1, 3), w2t)
        edg_tracks_target = edg_tracks_target.reshape(edg_tracks.shape)
        edg_rots_target = np.einsum("ij,ntjk->ntik", w2t[:3, :3], edg_rots)
        self.edg_tracks = torch.nn.Parameter(
            torch.from_numpy(edg_tracks_target).to(device=device, dtype=torch.float32)
        )
        self.edg_rots = torch.nn.Parameter(
            torch.from_numpy(edg_rots_target).to(device=device, dtype=torch.float32)
        )
        # Frozen reference copies to measure how much the parameters have moved
        self.edg_tracks_init = torch.from_numpy(edg_tracks_target).to(device=device, dtype=torch.float32)
        self.edg_rots_init   = torch.from_numpy(edg_rots_target).to(device=device, dtype=torch.float32)
        self.edg_target = self.edg_tracks[:, self.target_frame, :]

        self.dyn_indices = None
        self.dyn_knn_idx = None
        self.dyn_knn_w = None
        self.dyn_knn_logits = None
        self.dyn_rel = None
        self.dyn_knn_logits_init = None
        self.dyn_rel_init = None
        self.dyn_rgb = None
        self._dyn_filter_cache = None
        self._dyn_filter_warned = False
        self._edg_optimizer = None
        self.optimize_skinning_weights = False
        self.optimize_skinning_rel = False
        dyn_npz = self._find_dynamic_edg_npz(data_dir)
        if dyn_npz is not None:
            dyn_data = np.load(str(dyn_npz))
            dyn_indices = dyn_data["dyn_indices"].astype(np.int64)
            dyn_knn_idx = dyn_data["dyn_knn_idx"].astype(np.int64)
            dyn_knn_w = dyn_data["dyn_knn_w"].astype(np.float64)
            dyn_rel = dyn_data["dyn_rel"].astype(np.float64)
            dyn_rgb = dyn_data.get("dyn_rgb", None)
            dyn_edg_gids = dyn_data.get("edg_gids", None)
            dyn_target = int(dyn_data.get("target_frame", self.target_frame))
            if dyn_target != self.target_frame:
                print(f"[EDG] Warning: dynamic_edg target_frame={dyn_target} != {self.target_frame}")

            if dyn_edg_gids is not None:
                dyn_edg_gids = np.asarray(dyn_edg_gids, dtype=np.int64)
                if not np.array_equal(dyn_edg_gids, self.edg_gids):
                    gid_to_idx = {gid: i for i, gid in enumerate(self.edg_gids.tolist())}
                    remap = np.vectorize(lambda g: gid_to_idx.get(int(g), -1))(dyn_edg_gids)
                    if np.any(remap < 0):
                        raise RuntimeError("dynamic_edg contains unknown edg_gids; regenerate dynamic_edg.npz")
                    dyn_knn_idx = remap[dyn_knn_idx]

            self.dyn_indices = torch.from_numpy(dyn_indices).to(device=device, dtype=torch.long)
            self.dyn_knn_idx = torch.from_numpy(dyn_knn_idx).to(device=device, dtype=torch.long)
            self.dyn_knn_w = torch.from_numpy(dyn_knn_w).to(device=device, dtype=torch.float32)
            self.dyn_knn_w = self.dyn_knn_w / self.dyn_knn_w.sum(dim=1, keepdim=True).clamp_min(1e-8)
            self.dyn_knn_logits = torch.log(self.dyn_knn_w.clamp_min(1e-8))
            self.dyn_rel = torch.from_numpy(dyn_rel).to(device=device, dtype=torch.float32)
            self.dyn_knn_logits_init = self.dyn_knn_logits.detach().clone()
            self.dyn_rel_init = self.dyn_rel.detach().clone()
            if dyn_rgb is not None:
                dyn_rgb = np.asarray(dyn_rgb, dtype=np.uint8)
                self.dyn_rgb = torch.from_numpy(dyn_rgb).to(device=device, dtype=torch.float32)
            print(f"[EDG] Loaded dynamic_edg.npz: {self.dyn_indices.numel()} dynamic points")
        else:
            print("[EDG] dynamic_edg.npz not found; dynamic warping disabled.")

    @staticmethod
    def _as_track_array(track_dict):
        gids = sorted(track_dict.keys())
        tracks = np.stack([track_dict[g] for g in gids], axis=0).astype(np.float64)
        return tracks, np.array(gids, dtype=np.int64)

    @staticmethod
    def _as_rotation_array(rot_dict):
        gids = sorted(rot_dict.keys())
        rots = np.stack([rot_dict[g] for g in gids], axis=0).astype(np.float64)
        return rots, np.array(gids, dtype=np.int64)

    @staticmethod
    def _apply_transform(points: np.ndarray, mat: np.ndarray) -> np.ndarray:
        ones = np.ones((points.shape[0], 1), dtype=np.float64)
        hom = np.concatenate([points, ones], axis=1)
        out = (mat @ hom.T).T
        return out[:, :3]

    @staticmethod
    def _find_dynamic_edg_npz(data_dir: Path) -> Optional[Path]:
        return _find_upwards(data_dir, ["dynamic_edg.npz"], max_depth=4)

    @staticmethod
    def _find_xm_results(data_dir: Path) -> Optional[Path]:
        return _find_upwards(data_dir, ["result_XM.npz", "outputs/result_XM.npz"], max_depth=4)

    @staticmethod
    def find_results_npz(data_dir: Path) -> Optional[Path]:
        return _find_upwards(data_dir, ["results_all.npz", "outputs/results_all.npz"], max_depth=4)

    def _filter_dynamic(self, n_pts: int):
        if self.dyn_indices is None or self.dyn_indices.numel() == 0:
            return None
        use_cache = not (self.optimize_skinning_weights or self.optimize_skinning_rel)
        if (not use_cache) or (self._dyn_filter_cache is None) or (self._dyn_filter_cache[0] != n_pts):
            valid = (self.dyn_indices >= 0) & (self.dyn_indices < n_pts)
            if not bool(valid.all()) and not self._dyn_filter_warned:
                bad = int((~valid).sum().item())
                print(f"[EDG] Warning: {bad} dynamic indices out of bounds for {n_pts} gaussians; skipping.")
                self._dyn_filter_warned = True
            if bool(valid.any()):
                dyn_indices = self.dyn_indices[valid]
                dyn_knn_idx = self.dyn_knn_idx[valid]
                if self.optimize_skinning_weights and self.dyn_knn_logits is not None:
                    dyn_knn_w = torch.softmax(self.dyn_knn_logits[valid], dim=1)
                else:
                    dyn_knn_w = self.dyn_knn_w[valid]
                dyn_rel = self.dyn_rel[valid]
                dyn_rgb = self.dyn_rgb[valid] if self.dyn_rgb is not None else None
                if not use_cache:
                    return dyn_indices, dyn_knn_idx, dyn_knn_w, dyn_rel, dyn_rgb
                self._dyn_filter_cache = (n_pts, dyn_indices, dyn_knn_idx, dyn_knn_w, dyn_rel, dyn_rgb)
            else:
                if not use_cache:
                    return None
                self._dyn_filter_cache = (n_pts, None, None, None, None, None)

        _, dyn_indices, dyn_knn_idx, dyn_knn_w, dyn_rel, dyn_rgb = self._dyn_filter_cache
        if dyn_indices is None:
            return None
        return dyn_indices, dyn_knn_idx, dyn_knn_w, dyn_rel, dyn_rgb

    def warp(self, xyz_target: torch.Tensor, frame_idx: int) -> torch.Tensor:
        if frame_idx < 0 or frame_idx >= self.T:
            return xyz_target
        filtered = self._filter_dynamic(int(xyz_target.shape[0]))
        if filtered is None:
            return xyz_target
        dyn_indices, dyn_knn_idx, dyn_knn_w, dyn_rel, _ = filtered

        edg_t = self.edg_tracks[:, frame_idx, :]
        edg_rot_t = self.edg_rots[:, frame_idx, :, :]

        edg_pos = edg_t[dyn_knn_idx]
        edg_rot = edg_rot_t[dyn_knn_idx]
        rel = dyn_rel.unsqueeze(-1)
        warped = edg_pos + torch.matmul(edg_rot, rel).squeeze(-1)
        w = dyn_knn_w.unsqueeze(-1)
        dyn_xyz = torch.sum(warped * w, dim=1)

        xyz_out = xyz_target.clone()
        xyz_out[dyn_indices] = dyn_xyz
        return xyz_out

    def warp_quats(self, quats_target: torch.Tensor, frame_idx: int) -> torch.Tensor:
        """Rotate each dynamic Gaussian's orientation by the EDG rotation its surface underwent from the
        target frame to `frame_idx` (argmax-weight node's relative rotation R_m(t) R_m(target)^T applied to
        the base quat). The position warp already moves the centers; this keeps the ellipsoid aligned with
        the rotating surface so it need not inflate to cover all orientations (reduces rotation blur)."""
        if frame_idx < 0 or frame_idx >= self.T:
            return quats_target
        filtered = self._filter_dynamic(int(quats_target.shape[0]))
        if filtered is None:
            return quats_target
        dyn_indices, dyn_knn_idx, dyn_knn_w, _, _ = filtered
        amax = torch.argmax(dyn_knn_w, dim=1)                                # (Nd,) dominant node col
        node = dyn_knn_idx[torch.arange(dyn_knn_idx.shape[0], device=dyn_knn_idx.device), amax]
        R_t = self.edg_rots[node, frame_idx]                                 # (Nd,3,3)
        R_tg = self.edg_rots[node, self.target_frame]                        # (Nd,3,3)
        R_rel = torch.matmul(R_t, R_tg.transpose(1, 2))                      # target -> t surface rotation
        base_R = _quat_to_rotmat_batch(quats_target[dyn_indices])           # (Nd,3,3)
        new_q = _rotmat_to_quat_batch(torch.matmul(R_rel, base_R))          # (Nd,4) wxyz
        q_out = quats_target.clone()
        q_out[dyn_indices] = new_q.to(q_out.dtype)
        return q_out

    def edg_scene_arrays(self, frame_idx: int):
        """Return (node_points, line_points) as float32 numpy arrays for viser.

        node_points : (N, 3)
        line_points : (M, 2, 3)  – each row is a (start, end) pair
        """
        pts = self.edg_tracks[:, frame_idx, :].detach().cpu().numpy().astype(np.float32)  # (N,3)
        if self.edge_pairs.shape[0] > 0:
            line_pts = np.stack(
                [pts[self.edge_pairs[:, 0]], pts[self.edge_pairs[:, 1]]], axis=1
            ).astype(np.float32)  # (M, 2, 3)
        else:
            line_pts = np.zeros((0, 2, 3), dtype=np.float32)
        return pts, line_pts

    def edg_regularization_loss(
        self,
        arap_lambda: float = 1e-3,
        smooth_lambda: float = 1e-3,
        arap_gaps: Tuple[int, ...] = (1, 2, 4),
        arap_eta_len: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute ARAP and temporal-smoothness losses over all frames.

        ARAP (rest-frame-free, MULTI time-step pairs -- ALIGNED with complete_edg_table.py's L_arap):
            For each graph edge (i, j) and each temporal gap delta in `arap_gaps`, compare the edge
            between frames t and t+delta in node i's LOCAL frame, plus the edge LENGTH:

                d(t)  = g[j,t] - g[i,t]
                rc(t) = R[i,t]^T d(t) - R[i,t+delta]^T d(t+delta)      (local-coordinates residual)
                rl(t) = ||d(t)|| - ||d(t+delta)||                       (length residual)
                L_arap = arap_lambda * mean( ||rc||^2 + arap_eta_len * rl^2 )

            No fixed rest frame: rigidity is enforced between ALL (t, t+delta) pairs, so it does not
            pin the graph to any single frame. (complete_edg_table additionally sigma-whitens +
            Hubers these residuals; here arap_lambda is the single knob.)

        Temporal smoothness (COUNT-NORMALIZED like the ARAP):
            L_smooth = smooth_lambda * mean over (N,T-1) of ( ||dpos||^2 + ||drot||_F^2 )

        Both losses are MEANS (not sums), so the lambdas are stable across scene size; see the
        Config defaults.

        Returns:
            arap_loss  : scalar tensor (0 if no edges or arap_lambda==0)
            smooth_loss: scalar tensor (0 if T<2 or smooth_lambda==0)
        """
        device = self.edg_tracks.device
        zero = torch.zeros([], device=device)

        # ------------------------------------------------------------------
        # ARAP loss: rest-frame-free, multi-gap (complete_edg_table L_arap form)
        # ------------------------------------------------------------------
        if arap_lambda > 0.0 and self.edge_pairs.shape[0] > 0:
            ep = torch.from_numpy(self.edge_pairs).to(device=device, dtype=torch.long)  # [E, 2]
            ei = ep[:, 0]  # [E]
            ej = ep[:, 1]  # [E]

            d = self.edg_tracks[ej] - self.edg_tracks[ei]          # [E, T, 3]  edge vector per frame
            R_i = self.edg_rots[ei]                                # [E, T, 3, 3] node-i rotation per frame

            total = zero
            n_terms = 0
            for gap in arap_gaps:
                g_ = int(gap)
                if g_ <= 0 or self.T <= g_:
                    continue
                dt, dtd = d[:, : self.T - g_], d[:, g_:]           # [E, T-g, 3]
                Rt, Rtd = R_i[:, : self.T - g_], R_i[:, g_:]       # [E, T-g, 3, 3]
                rc = torch.einsum("etji,etj->eti", Rt, dt) \
                    - torch.einsum("etji,etj->eti", Rtd, dtd)      # R^T d difference (local coords)
                rl = dt.norm(dim=-1) - dtd.norm(dim=-1)            # edge-length difference
                total = total + (rc ** 2).sum() + float(arap_eta_len) * (rl ** 2).sum()
                n_terms += rc.numel() + rl.numel()
            arap_loss = arap_lambda * total / max(n_terms, 1)
        else:
            arap_loss = zero

        # ------------------------------------------------------------------
        # Temporal smoothness loss (count-normalized)
        # ------------------------------------------------------------------
        if smooth_lambda > 0.0 and self.T >= 2:
            # Position smoothness: edg_tracks[:, t+1, :] - edg_tracks[:, t, :]
            dpos = self.edg_tracks[:, 1:, :] - self.edg_tracks[:, :-1, :]   # [N, T-1, 3]
            # Rotation smoothness: Frobenius norm of rotation difference
            drot = self.edg_rots[:, 1:, :, :] - self.edg_rots[:, :-1, :, :]  # [N, T-1, 3, 3]
            smooth_loss = smooth_lambda * (
                (dpos ** 2).sum() + (drot ** 2).sum()
            ) / max(dpos.numel() + drot.numel(), 1)
        else:
            smooth_loss = zero

        return arap_loss, smooth_loss

    def observation_anchor_loss(self, frame_idx: int, edg_obs_lambda: float,
                                dyn_obs_lambda: float, n_sample: int = 16384):
        """Training-time analogues of complete_edg_table's L_init / L_obs (both 0 when disabled).

        edg_obs: whitened L2 pulling the LEARNED edg_tracks/edg_rots toward their solved-table init
                 (positions / node-cloud scale, rotations Frobenius / 0.1).
        dyn_obs: on a random subsample of dynamic Gaussians, the CURRENT warp at `frame_idx` (current
                 EDG + current skinning weights/offsets) must stay near the INIT warp (init EDG + init
                 skinning) -- the init warp is what complete_edg_table fit to the 3D observations."""
        device = self.edg_tracks.device
        zero = torch.zeros([], device=device)
        edg_obs = dyn_obs = zero
        if not hasattr(self, "_obs_scale"):
            with torch.no_grad():
                pts = self.edg_tracks_init.reshape(-1, 3)
                med = pts.median(dim=0).values
                self._obs_scale = float((pts - med).norm(dim=-1).median().clamp_min(1e-6))
        if edg_obs_lambda > 0.0:
            dp = (self.edg_tracks - self.edg_tracks_init).pow(2).sum(-1) / (self._obs_scale ** 2)
            dr = (self.edg_rots - self.edg_rots_init).pow(2).sum((-1, -2)) / (0.1 ** 2)
            edg_obs = edg_obs_lambda * (dp.mean() + dr.mean())
        if (dyn_obs_lambda > 0.0 and self.dyn_knn_idx is not None
                and 0 <= frame_idx < self.T and self.dyn_knn_idx.numel() > 0):
            Nd = self.dyn_knn_idx.shape[0]
            idx = torch.randint(0, Nd, (min(int(n_sample), Nd),), device=device)
            knn = self.dyn_knn_idx[idx]                                       # (S,K)
            if self.optimize_skinning_weights and self.dyn_knn_logits is not None:
                w = torch.softmax(self.dyn_knn_logits[idx], dim=1)
            else:
                w = self.dyn_knn_w[idx]
            rel = self.dyn_rel[idx]
            cur = ((self.edg_tracks[:, frame_idx][knn]
                    + torch.matmul(self.edg_rots[:, frame_idx][knn], rel.unsqueeze(-1)).squeeze(-1))
                   * w.unsqueeze(-1)).sum(1)                                  # (S,3)
            with torch.no_grad():
                w0 = torch.softmax(self.dyn_knn_logits_init[idx], dim=1) \
                    if self.dyn_knn_logits_init is not None else w
                rel0 = self.dyn_rel_init[idx] if self.dyn_rel_init is not None else rel
                ref = ((self.edg_tracks_init[:, frame_idx][knn]
                        + torch.matmul(self.edg_rots_init[:, frame_idx][knn],
                                       rel0.unsqueeze(-1)).squeeze(-1))
                       * w0.unsqueeze(-1)).sum(1)
            dyn_obs = dyn_obs_lambda * ((cur - ref).pow(2).sum(-1) / (self._obs_scale ** 2)).mean()
        return edg_obs, dyn_obs

    def create_optimizer(
        self,
        tracks_lr: float,
        rots_lr: float,
        weights_lr: float,
        rel_lr: float,
        batch_size: int = 1,
    ) -> Optional[torch.optim.Optimizer]:
        """Return an Adam optimizer for trainable EDG deformation and skinning tensors."""
        param_groups = []
        self.optimize_skinning_weights = bool(weights_lr > 0.0 and self.dyn_knn_logits is not None)
        self.optimize_skinning_rel = bool(rel_lr > 0.0 and self.dyn_rel is not None)

        if tracks_lr > 0.0:
            param_groups.append({"params": [self.edg_tracks], "lr": tracks_lr, "name": "edg_tracks"})
        if rots_lr > 0.0:
            param_groups.append({"params": [self.edg_rots], "lr": rots_lr, "name": "edg_rots"})
        if self.optimize_skinning_weights:
            self.dyn_knn_logits = torch.nn.Parameter(self.dyn_knn_logits.detach().clone())
            param_groups.append({"params": [self.dyn_knn_logits], "lr": weights_lr, "name": "edg_skinning_weights"})
        if self.optimize_skinning_rel:
            self.dyn_rel = torch.nn.Parameter(self.dyn_rel.detach().clone())
            param_groups.append({"params": [self.dyn_rel], "lr": rel_lr, "name": "edg_skinning_rel"})
        if not param_groups:
            return None
        self._edg_optimizer = torch.optim.Adam(param_groups, eps=1e-15)
        return self._edg_optimizer

    def skinning_regularization_loss(self, reg_lambda: float) -> torch.Tensor:
        device = self.edg_tracks.device
        if reg_lambda <= 0.0:
            return torch.zeros([], device=device)

        loss = torch.zeros([], device=device)
        if self.optimize_skinning_weights and self.dyn_knn_logits is not None and self.dyn_knn_logits_init is not None:
            w = torch.softmax(self.dyn_knn_logits, dim=1)
            w0 = torch.softmax(self.dyn_knn_logits_init[: w.shape[0]], dim=1)
            loss = loss + F.mse_loss(w, w0)
        if self.optimize_skinning_rel and self.dyn_rel is not None and self.dyn_rel_init is not None:
            rel0 = self.dyn_rel_init[: self.dyn_rel.shape[0]]
            loss = loss + F.mse_loss(self.dyn_rel, rel0)
        return reg_lambda * loss

    def trainable_parameters(self) -> List[torch.Tensor]:
        params = [self.edg_tracks, self.edg_rots]
        if self.optimize_skinning_weights and self.dyn_knn_logits is not None:
            params.append(self.dyn_knn_logits)
        if self.optimize_skinning_rel and self.dyn_rel is not None:
            params.append(self.dyn_rel)
        return params

    @torch.no_grad()
    def apply_skinning_constraints(self, rel_delta_max: float) -> None:
        if (
            rel_delta_max > 0.0
            and self.optimize_skinning_rel
            and self.dyn_rel is not None
            and self.dyn_rel_init is not None
            and self.dyn_rel.shape == self.dyn_rel_init.shape
        ):
            delta = self.dyn_rel.data - self.dyn_rel_init
            norm = delta.norm(dim=-1, keepdim=True)
            scale = torch.clamp(float(rel_delta_max) / norm.clamp_min(1e-8), max=1.0)
            self.dyn_rel.data.copy_(self.dyn_rel_init + delta * scale)

    def _refresh_edg_optimizer_params(self) -> None:
        if self._edg_optimizer is None:
            return
        for group in self._edg_optimizer.param_groups:
            name = group.get("name")
            if name == "edg_tracks":
                group["params"] = [self.edg_tracks]
            elif name == "edg_rots":
                group["params"] = [self.edg_rots]
            elif name == "edg_skinning_weights" and self.dyn_knn_logits is not None:
                group["params"] = [self.dyn_knn_logits]
            elif name == "edg_skinning_rel" and self.dyn_rel is not None:
                group["params"] = [self.dyn_rel]

    def _rewrap_skinning_params_after_topology_change(self) -> None:
        if self.optimize_skinning_weights and self.dyn_knn_logits is not None:
            self.dyn_knn_logits = torch.nn.Parameter(self.dyn_knn_logits.detach().clone())
        if self.optimize_skinning_rel and self.dyn_rel is not None:
            self.dyn_rel = torch.nn.Parameter(self.dyn_rel.detach().clone())
        self._refresh_edg_optimizer_params()

    def state_dict(self) -> Dict[str, torch.Tensor]:
        state = {
            "edg_tracks": self.edg_tracks.detach().cpu(),
            "edg_rots": self.edg_rots.detach().cpu(),
        }
        if self.dyn_indices is not None:
            state["dyn_indices"] = self.dyn_indices.detach().cpu()
            state["dyn_knn_idx"] = self.dyn_knn_idx.detach().cpu()
            state["dyn_knn_w"] = self.dyn_knn_w.detach().cpu()
            if self.dyn_knn_logits is not None:
                state["dyn_knn_logits"] = self.dyn_knn_logits.detach().cpu()
            if self.dyn_rel is not None:
                state["dyn_rel"] = self.dyn_rel.detach().cpu()
            if self.dyn_rgb is not None:
                state["dyn_rgb"] = self.dyn_rgb.detach().cpu()
        return state

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        device = self.edg_tracks.device
        if "edg_tracks" in state:
            self.edg_tracks.data = state["edg_tracks"].to(device=device, dtype=torch.float32)
        if "edg_rots" in state:
            self.edg_rots.data = state["edg_rots"].to(device=device, dtype=torch.float32)
        if "dyn_indices" in state:
            self.dyn_indices = state["dyn_indices"].to(device=device, dtype=torch.long)
        if "dyn_knn_idx" in state:
            self.dyn_knn_idx = state["dyn_knn_idx"].to(device=device, dtype=torch.long)
        if "dyn_knn_w" in state:
            self.dyn_knn_w = state["dyn_knn_w"].to(device=device, dtype=torch.float32)
        if "dyn_knn_logits" in state:
            logits = state["dyn_knn_logits"].to(device=device, dtype=torch.float32)
            self.dyn_knn_logits = torch.nn.Parameter(logits) if self.optimize_skinning_weights else logits
        if "dyn_rel" in state:
            rel = state["dyn_rel"].to(device=device, dtype=torch.float32)
            self.dyn_rel = torch.nn.Parameter(rel) if self.optimize_skinning_rel else rel
        if "dyn_rgb" in state:
            self.dyn_rgb = state["dyn_rgb"].to(device=device, dtype=torch.float32)
        self._dyn_filter_cache = None
        self._refresh_edg_optimizer_params()

    # ------------------------------------------------------------------
    # EDG-aware densification hooks
    # These must be called immediately after the gsplat duplicate/split/remove
    # ops so that the dynamic bookkeeping stays in sync with the GS arrays.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def edg_notify_remove(self, mask: torch.Tensor) -> None:
        """Update EDG dynamic bookkeeping after gaussians are removed.

        Args:
            mask: bool tensor of shape [N], True = GS is being removed.
        """
        if self.dyn_indices is None or self.dyn_indices.numel() == 0:
            return

        N = mask.shape[0]

        # Clip to valid range (handles pre-existing out-of-bounds indices)
        valid = (self.dyn_indices >= 0) & (self.dyn_indices < N)
        if not bool(valid.any()):
            return

        # Work only on the valid subset, then reconstruct full arrays
        # (invalid entries are dropped).
        vi = self.dyn_indices[valid]

        # 1. Which valid dynamic points survive? (their GS is NOT in the remove mask)
        dyn_gs_mask = mask[vi]            # [Dv], True = this dyn point's GS is pruned
        survive_valid = ~dyn_gs_mask      # [Dv], True = keep

        # 2. Build remapping: old_gs_idx -> new_gs_idx after removal
        keep = (~mask).long()                         # [N]
        new_idx = torch.cumsum(keep, dim=0) - 1       # [N]

        # 3. Update only the valid entries that survive
        kept_indices_into_valid = torch.where(valid)[0][survive_valid]

        self.dyn_indices = new_idx[vi[survive_valid]]
        self.dyn_knn_idx = self.dyn_knn_idx[kept_indices_into_valid]
        self.dyn_knn_w   = self.dyn_knn_w[kept_indices_into_valid]
        if self.dyn_knn_logits is not None:
            self.dyn_knn_logits = self.dyn_knn_logits[kept_indices_into_valid]
        if self.dyn_knn_logits_init is not None:
            self.dyn_knn_logits_init = self.dyn_knn_logits_init[kept_indices_into_valid]
        self.dyn_rel     = self.dyn_rel[kept_indices_into_valid]
        if self.dyn_rel_init is not None:
            self.dyn_rel_init = self.dyn_rel_init[kept_indices_into_valid]
        if self.dyn_rgb is not None:
            self.dyn_rgb = self.dyn_rgb[kept_indices_into_valid]

        # Invalidate filter cache
        self._dyn_filter_cache = None
        self._dyn_filter_warned = False
        self._rewrap_skinning_params_after_topology_change()

    @torch.no_grad()
    def edg_notify_duplicate(self, mask: torch.Tensor, orig_N: int) -> None:
        """Update EDG dynamic bookkeeping after gaussians are duplicated.

        After duplication the GS array is [original_params | duplicated_params],
        where duplicated_params = original_params[sel].

        Args:
            mask: bool tensor of shape [orig_N], True = GS was duplicated.
            orig_N: number of GS before duplication.
        """
        if self.dyn_indices is None or self.dyn_indices.numel() == 0:
            return

        sel = torch.where(mask)[0]   # indices of duplicated originals, shape [M]
        if sel.numel() == 0:
            return

        # Only consider valid (in-bounds) dyn_indices
        valid = (self.dyn_indices >= 0) & (self.dyn_indices < orig_N)
        if not bool(valid.any()):
            return

        valid_dyn_indices = self.dyn_indices[valid]   # [Dv]
        valid_rows = torch.where(valid)[0]            # positions in dyn_* tensors

        # Build a mapping: orig_idx -> position in sel
        sel_pos = torch.full((orig_N,), -1, dtype=torch.long, device=sel.device)
        sel_pos[sel] = torch.arange(sel.numel(), device=sel.device)

        # Which valid dynamic points have a GS that was duplicated?
        dup_pos = sel_pos[valid_dyn_indices]           # [Dv], -1 if not duplicated
        is_dup  = dup_pos >= 0                         # [Dv]

        if not bool(is_dup.any()):
            return

        dup_rows = valid_rows[is_dup]                 # positions in dyn_* tensors to copy
        new_dyn_indices = orig_N + dup_pos[is_dup]    # [M_dyn]

        # Append new entries
        self.dyn_indices = torch.cat([self.dyn_indices, new_dyn_indices])
        self.dyn_knn_idx = torch.cat([self.dyn_knn_idx, self.dyn_knn_idx[dup_rows]])
        self.dyn_knn_w   = torch.cat([self.dyn_knn_w,   self.dyn_knn_w[dup_rows]])
        if self.dyn_knn_logits is not None:
            self.dyn_knn_logits = torch.cat([self.dyn_knn_logits, self.dyn_knn_logits[dup_rows]])
        if self.dyn_knn_logits_init is not None:
            self.dyn_knn_logits_init = torch.cat([self.dyn_knn_logits_init, self.dyn_knn_logits_init[dup_rows]])
        self.dyn_rel     = torch.cat([self.dyn_rel,      self.dyn_rel[dup_rows]])
        if self.dyn_rel_init is not None:
            self.dyn_rel_init = torch.cat([self.dyn_rel_init, self.dyn_rel_init[dup_rows]])
        if self.dyn_rgb is not None:
            self.dyn_rgb = torch.cat([self.dyn_rgb, self.dyn_rgb[dup_rows]])

        # Invalidate filter cache
        self._dyn_filter_cache = None
        self._dyn_filter_warned = False
        self._rewrap_skinning_params_after_topology_change()

    @torch.no_grad()
    def edg_notify_split(self, mask: torch.Tensor, orig_N: int) -> None:
        """Update EDG dynamic bookkeeping after gaussians are split.

        After split the GS array layout is [rest_params | split1 | split2],
        where rest = ~mask, split1/split2 = mask (each copy).

        Args:
            mask: bool tensor of shape [orig_N], True = GS was split.
            orig_N: number of GS before split.
        """
        if self.dyn_indices is None or self.dyn_indices.numel() == 0:
            return

        sel  = torch.where(mask)[0]   # split originals [S]
        rest = torch.where(~mask)[0]  # kept originals  [R]

        R = rest.numel()
        S = sel.numel()

        # Only consider valid (in-bounds) dyn_indices
        valid = (self.dyn_indices >= 0) & (self.dyn_indices < orig_N)
        if not bool(valid.any()):
            return

        valid_dyn_indices = self.dyn_indices[valid]   # [Dv]
        valid_rows = torch.where(valid)[0]            # positions in dyn_* tensors

        # Build remapping for 'rest' indices: orig_idx -> new_idx
        rest_remap = torch.full((orig_N,), -1, dtype=torch.long, device=mask.device)
        rest_remap[rest] = torch.arange(R, device=mask.device)

        # Build mapping for split originals: orig_idx -> position in sel
        sel_pos = torch.full((orig_N,), -1, dtype=torch.long, device=mask.device)
        sel_pos[sel] = torch.arange(S, device=mask.device)

        is_rest_v = rest_remap[valid_dyn_indices] >= 0    # [Dv]
        is_sel_v  = sel_pos[valid_dyn_indices] >= 0       # [Dv]

        # Part 1: rest dynamic points
        rest_rows = valid_rows[is_rest_v]
        new_indices_rest = rest_remap[valid_dyn_indices[is_rest_v]]
        knn_idx_rest = self.dyn_knn_idx[rest_rows]
        knn_w_rest   = self.dyn_knn_w[rest_rows]
        logits_rest = self.dyn_knn_logits[rest_rows] if self.dyn_knn_logits is not None else None
        logits_init_rest = self.dyn_knn_logits_init[rest_rows] if self.dyn_knn_logits_init is not None else None
        rel_rest     = self.dyn_rel[rest_rows]
        rel_init_rest = self.dyn_rel_init[rest_rows] if self.dyn_rel_init is not None else None
        rgb_rest     = self.dyn_rgb[rest_rows] if self.dyn_rgb is not None else None

        # Part 2: split dynamic points — two copies
        sel_rows = valid_rows[is_sel_v]
        sp = sel_pos[valid_dyn_indices[is_sel_v]]     # position in sel
        new_indices_split1 = R + sp
        new_indices_split2 = R + S + sp
        knn_idx_split = self.dyn_knn_idx[sel_rows]
        knn_w_split   = self.dyn_knn_w[sel_rows]
        logits_split = self.dyn_knn_logits[sel_rows] if self.dyn_knn_logits is not None else None
        logits_init_split = self.dyn_knn_logits_init[sel_rows] if self.dyn_knn_logits_init is not None else None
        rel_split     = self.dyn_rel[sel_rows]
        rel_init_split = self.dyn_rel_init[sel_rows] if self.dyn_rel_init is not None else None
        rgb_split     = self.dyn_rgb[sel_rows] if self.dyn_rgb is not None else None

        self.dyn_indices = torch.cat([
            new_indices_rest,
            new_indices_split1,
            new_indices_split2,
        ])
        self.dyn_knn_idx = torch.cat([knn_idx_rest, knn_idx_split, knn_idx_split])
        self.dyn_knn_w   = torch.cat([knn_w_rest,   knn_w_split,   knn_w_split])
        if self.dyn_knn_logits is not None:
            self.dyn_knn_logits = torch.cat([logits_rest, logits_split, logits_split])
        if self.dyn_knn_logits_init is not None:
            self.dyn_knn_logits_init = torch.cat([logits_init_rest, logits_init_split, logits_init_split])
        self.dyn_rel     = torch.cat([rel_rest,      rel_split,     rel_split])
        if self.dyn_rel_init is not None:
            self.dyn_rel_init = torch.cat([rel_init_rest, rel_init_split, rel_init_split])
        if self.dyn_rgb is not None:
            self.dyn_rgb = torch.cat([rgb_rest, rgb_split, rgb_split])

        # Invalidate filter cache
        self._dyn_filter_cache = None
        self._dyn_filter_warned = False
        self._rewrap_skinning_params_after_topology_change()


def _voxel_downsample_points(points: np.ndarray, rgbs: np.ndarray, voxel_size: float):
    """Density-based downsample: keep one averaged point per occupied voxel.

    Already-dense regions collapse to a single representative point per voxel while
    sparse regions are preserved (each occupied voxel contributes exactly one point).
    Returns (points, rgbs) with the same dtypes. voxel_size<=0 or empty input is a no-op.
    """
    points = np.asarray(points)
    rgbs = np.asarray(rgbs)
    if voxel_size is None or voxel_size <= 0.0 or points.shape[0] == 0:
        return points, rgbs
    keys = np.floor(points / float(voxel_size)).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    n_vox = int(inverse.max()) + 1
    counts = np.bincount(inverse, minlength=n_vox).astype(np.float64)
    pts = np.zeros((n_vox, 3), dtype=np.float64)
    rgb = np.zeros((n_vox, 3), dtype=np.float64)
    np.add.at(pts, inverse, points.astype(np.float64))
    np.add.at(rgb, inverse, rgbs.astype(np.float64))
    pts /= counts[:, None]
    rgb /= counts[:, None]
    return pts.astype(points.dtype), rgb.astype(rgbs.dtype)


def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_voxel_size: float = 0.0,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    means_lr: float = 1.6e-4,
    scales_lr: float = 5e-3,
    opacities_lr: float = 5e-2,
    quats_lr: float = 1e-3,
    sh0_lr: float = 2.5e-3,
    shN_lr: float = 2.5e-3 / 20,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        pts_np = np.asarray(parser.points, dtype=np.float32)
        rgb_np = np.asarray(parser.points_rgb, dtype=np.float32) / 255.0
        if init_voxel_size and init_voxel_size > 0.0:
            v = float(init_voxel_size) * float(scene_scale)
            n0 = pts_np.shape[0]
            pts_np, rgb_np = _voxel_downsample_points(pts_np, rgb_np, v)
            print(f"[init] density voxel-downsample sfm points: {n0} -> {pts_np.shape[0]} "
                  f"(voxel={v:.5g} = {init_voxel_size}*scene_scale={scene_scale:.4g})")
        points = torch.from_numpy(np.ascontiguousarray(pts_np)).float()
        rgbs = torch.from_numpy(np.ascontiguousarray(rgb_np)).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), scales_lr),
        ("quats", torch.nn.Parameter(quats), quats_lr),
        ("opacities", torch.nn.Parameter(opacities), opacities_lr),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), sh0_lr))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), sh0_lr))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
            fused=True,
        )
        for name, _, lr in params
    }
    return splats, optimizers


class EDGAwareDefaultStrategy(DefaultStrategy):
    """DefaultStrategy subclass that keeps EDGWarp bookkeeping in sync after
    every grow (duplicate/split) and prune (remove) operation.

    Rules:
      - EDG *node* parameters (edg_tracks, edg_rots) are never touched here;
        they live in their own optimizer and are NOT Gaussians.
      - For *dynamic* Gaussians (those tracked by EDGWarp.dyn_*):
          - remove  → update dyn_indices / knn / weights / rel
          - duplicate → append copied entries for each new GS copy
          - split     → remap rest + append two copies per split GS
      - Static Gaussians (not in dyn_indices) pass through unchanged.
    """

    def __init__(
        self,
        edg_warp: Optional["EDGWarp"] = None,
        max_prune_fraction: Optional[float] = None,
        max_gaussians: int = 0,
        edg_dyn_no_prune: bool = False,
        edg_dyn_force_grow: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Set by Runner after EDGWarp is constructed; may be None.
        self.edg_warp: Optional["EDGWarp"] = edg_warp
        self.max_prune_fraction = max_prune_fraction
        self.max_gaussians = int(max_gaussians)
        self.edg_dyn_no_prune = bool(edg_dyn_no_prune)
        self.edg_dyn_force_grow = bool(edg_dyn_force_grow)

    def _dyn_index_mask(self, n_total, device):
        """Bool [n_total] mask, True where a Gaussian is a dynamic (EDG-warped) point."""
        m = torch.zeros(n_total, dtype=torch.bool, device=device)
        if (self.edg_warp is not None and self.edg_warp.dyn_indices is not None
                and self.edg_warp.dyn_indices.numel() > 0):
            di = self.edg_warp.dyn_indices
            di = di[(di >= 0) & (di < n_total)]
            if di.numel():
                m[di] = True
        return m

    @torch.no_grad()
    def _grow_gs(self, params, optimizers, state, step):
        """Grow gaussians, then update EDG bookkeeping."""
        # Capture N before grow
        orig_N = int(params["means"].shape[0])

        # Hard cap: stop densifying once the count reaches max_gaussians.
        if self.max_gaussians > 0 and orig_N >= self.max_gaussians:
            return 0, 0

        # Force dynamic Gaussians to densify every refine step: boost their accumulated grad2d above the
        # grow threshold in the SAME state the parent reads, so both our re-derived masks (for edg
        # bookkeeping) and the parent's actual grow see the identical decision -> stays in sync.
        if self.edg_dyn_force_grow:
            dm = self._dyn_index_mask(orig_N, state["grad2d"].device)
            if bool(dm.any()):
                state["grad2d"][dm] = state["count"][dm].clamp_min(1) * (self.grow_grad2d * 2.0 + 1e-12)

        # --- replicate the duplicate decision logic from DefaultStrategy ---
        count  = state["count"]
        grads  = state["grad2d"] / count.clamp_min(1)
        device = grads.device

        is_grad_high = grads > self.grow_grad2d
        is_small = (
            torch.exp(params["scales"]).max(dim=-1).values
            <= self.grow_scale3d * state["scene_scale"]
        )
        is_dupli = is_grad_high & is_small
        is_large = ~is_small
        is_split = is_grad_high & is_large
        if step < self.refine_scale2d_stop_iter:
            is_split |= state["radii"] > self.grow_scale2d

        # Block duplicating / splitting EDG node GS positions:
        # EDG nodes are *not* in the GS splats array — they live in edg_warp
        # separately — so nothing special is needed here for EDG nodes.
        # We just proceed with normal grow and then update dyn bookkeeping.

        # Let parent do the actual grow (returns n_dupli, n_split)
        n_dupli, n_split = super()._grow_gs(params, optimizers, state, step)

        if self.edg_warp is None or (n_dupli == 0 and n_split == 0):
            return n_dupli, n_split

        # Notify EDGWarp — we must do it in the same order that ops.py applies them.
        # Parent calls: duplicate(is_dupli) then split(is_split_extended)
        # After duplicate, splats have orig_N + n_dupli rows.
        # After split, splats have rest + 2*n_split rows.

        # Re-derive is_dupli and is_split from the saved grads/state
        # (state may have been reset, but we saved the boolean masks above)
        if n_dupli > 0:
            self.edg_warp.edg_notify_duplicate(is_dupli, orig_N)

        if n_split > 0:
            # After duplicate, total = orig_N + n_dupli
            N_after_dup = orig_N + n_dupli
            # is_split_extended: original is_split + zeros for duplicated rows
            is_split_ext = torch.cat([
                is_split,
                torch.zeros(n_dupli, dtype=torch.bool, device=device),
            ])
            self.edg_warp.edg_notify_split(is_split_ext, N_after_dup)

        return n_dupli, n_split

    @torch.no_grad()
    def _prune_gs(self, params, optimizers, state, step):
        """Prune gaussians, then update EDG bookkeeping."""
        # Replicate the prune mask logic from DefaultStrategy
        opacities = torch.sigmoid(params["opacities"].flatten())
        is_prune = opacities < self.prune_opa
        low_opa = is_prune.clone()
        if step > self.reset_every:
            scales_max = torch.exp(params["scales"]).max(dim=-1).values
            is_too_big = (
                scales_max
                > self.prune_scale3d * state["scene_scale"]
            )
            too_big_3d = is_too_big.clone()
            if step < self.refine_scale2d_stop_iter:
                is_too_big |= state["radii"] > self.prune_scale2d
                too_big_2d = state["radii"] > self.prune_scale2d
            else:
                too_big_2d = None
            # Prune ALL too-big Gaussians (big Gaussians corrupt novel-view depth). The
            # depth_error_prune_max_fraction cap + largest-3D-scale tie-break below bound
            # how many are removed per step.
            is_prune = is_prune | is_too_big
        else:
            scales_max = torch.exp(params["scales"]).max(dim=-1).values
            too_big_3d = torch.zeros_like(is_prune)
            too_big_2d = None

        # Never prune dynamic Gaussians -> the dynamic cloud accumulates as densely as possible.
        if self.edg_dyn_no_prune:
            dm = self._dyn_index_mask(is_prune.shape[0], is_prune.device)
            if bool(dm.any()):
                is_prune = is_prune & (~dm)

        n_prune_requested = int(is_prune.sum().item())
        if self.verbose:
            n_low_opa = int(low_opa.sum().item())
            n_too_big_3d = int(too_big_3d.sum().item())
            n_too_big_2d = int(too_big_2d.sum().item()) if too_big_2d is not None else 0
            scene_scale = float(state.get("scene_scale", 1.0))
            print(
                "[Prune] step={} total={} low_opa={} too_big_3d={} too_big_2d={} "
                "opa_med={:.4f} opa_min={:.4f} scale_max_med={:.4f} scene_scale={:.4f} "
                "prune_opa={:.4f} prune_scale3d={:.4f} prune_scale2d={:.4f}".format(
                    step,
                    int(is_prune.numel()),
                    n_low_opa,
                    n_too_big_3d,
                    n_too_big_2d,
                    float(opacities.median().item()),
                    float(opacities.min().item()),
                    float(scales_max.median().item()),
                    scene_scale,
                    float(self.prune_opa),
                    float(self.prune_scale3d),
                    float(self.prune_scale2d),
                )
            )
        if self.max_prune_fraction is not None and self.max_prune_fraction >= 0.0:
            n_total = int(params["means"].shape[0])
            max_prune = int(np.floor(n_total * float(self.max_prune_fraction)))
            max_prune = max(1, min(max_prune, n_total))
        else:
            max_prune = n_prune_requested

        if n_prune_requested > max_prune:
            prune_idx = torch.where(is_prune)[0]
            # When the cap is hit, remove the LARGEST-3D-scale candidates first: big
            # Gaussians are what corrupt novel-view depth.
            cand_scale3d = torch.exp(params["scales"]).max(dim=-1).values[prune_idx]
            keep_order = torch.argsort(cand_scale3d, descending=True)[:max_prune]
            capped_mask = torch.zeros_like(is_prune)
            capped_mask[prune_idx[keep_order]] = True
            if self.verbose:
                print(
                    f"Step {step}: prune guard capped {n_prune_requested} requested "
                    f"prunes to {max_prune} ({100.0 * float(self.max_prune_fraction):.1f}%)."
                )
            is_prune = capped_mask

        n_prune = int(is_prune.sum().item())
        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

        if self.edg_warp is not None and n_prune > 0:
            self.edg_warp.edg_notify_remove(is_prune)

        return n_prune


class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(int(os.environ.get("GSPLAT_SEED", "42")) + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/ply"
        os.makedirs(self.ply_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
            load_exposure=cfg.load_exposure,
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
        )
        self.valset = Dataset(self.parser, split="val")
        # Per-frame metric scale from the XM solver; aligns monocular training
        # depths to metric scale (replaces per-image median scale fitting).
        self.da3_frame_scale = None
        xm_path = _find_upwards(
            Path(cfg.data_dir), ["result_XM.npz", "outputs/result_XM.npz"], max_depth=4
        )
        if xm_path is not None:
            try:
                _rt_scale = np.load(str(xm_path))
                _raw_scale = np.asarray(_rt_scale["da3_frame_scale"], dtype=np.float64).reshape(-1)
                # The solver may drop frames (N<=T); da3_frame_scale is per-solved-frame but
                # indexed here by image_id (0..T-1). Scatter it onto the full timeline via
                # indices_all (dropped frames -> 1.0) so indexing stays aligned/in-bounds.
                if "indices_all" in _rt_scale.files:
                    _idx_map = np.asarray(_rt_scale["indices_all"]).astype(np.int64)
                    scale_arr = np.ones((_idx_map.shape[0],), dtype=np.float64)
                    for _oi in range(_idx_map.shape[0]):
                        _j = int(_idx_map[_oi])
                        if 0 <= _j < _raw_scale.shape[0]:
                            scale_arr[_oi] = _raw_scale[_j]
                else:
                    scale_arr = _raw_scale
                self.da3_frame_scale = torch.from_numpy(scale_arr).to(device=self.device)
            except Exception as exc:
                print(f"[depth] Could not load da3_frame_scale from {xm_path}: {exc}")
        else:
            print(f"[depth] result_XM.npz not found near {cfg.data_dir}; depth loss uses scale=1.0")

        # Full DA3 monocular depth maps [T,H,W] for depth-error pruning (kept on CPU,
        # moved per-frame). This is the SAME monocular depth used for the depth loss /
        # lift -> pruning against it is legitimate (no GT leakage).
        self.da3_depth_maps = None
        dm_path = _find_upwards(
            Path(cfg.data_dir),
            ["depth_anything_3/depth_maps.npy", "outputs/depth_anything_3/depth_maps.npy",
             "depth/depth_maps.npy", "outputs/depth/depth_maps.npy"],
            max_depth=4,
        )
        if dm_path is not None:
            try:
                dm = np.load(str(dm_path)).astype(np.float32)
                self.da3_depth_maps = torch.from_numpy(dm)  # CPU [T,H,W]
                print(f"[depth] Loaded DA3 depth maps {tuple(dm.shape)} for depth-error prune from {dm_path}")
            except Exception as exc:
                print(f"[depth] Could not load DA3 depth maps: {exc}")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        if self.parser.num_cameras > 1 and cfg.batch_size != 1:
            raise ValueError(
                f"When using multiple cameras ({self.parser.num_cameras} found), batch_size must be 1, "
                f"but got batch_size={cfg.batch_size}."
            )
        if cfg.post_processing == "ppisp" and cfg.batch_size != 1:
            raise ValueError(
                f"PPISP post-processing requires batch_size=1, got batch_size={cfg.batch_size}"
            )
        if cfg.post_processing is not None and world_size > 1:
            raise ValueError(
                f"Post-processing ({cfg.post_processing}) requires single-GPU training, "
                f"but world_size={world_size}."
            )
        if cfg.post_processing == "ppisp" and isinstance(cfg.strategy, DefaultStrategy):
            raise ValueError(
                f"PPISP post-processing requires MCMCStrategy at the moment."
            )

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_voxel_size=cfg.init_voxel_size,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            means_lr=cfg.means_lr,
            scales_lr=cfg.scales_lr,
            opacities_lr=cfg.opacities_lr,
            quats_lr=cfg.quats_lr,
            sh0_lr=cfg.sh0_lr,
            shN_lr=cfg.shN_lr,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        # EDG warp (optional)
        self.edg_warp = None
        self._edg_warned_multi_frame = False
        results_npz = Path(cfg.edg_results) if cfg.edg_results else EDGWarp.find_results_npz(Path(cfg.data_dir))
        if results_npz is not None:
            try:
                self.edg_warp = EDGWarp(
                    data_dir=Path(cfg.data_dir),
                    results_npz=results_npz,
                    target_frame=cfg.edg_target_frame,
                    knn=cfg.edg_knn,
                    chunk=cfg.edg_chunk,
                    device=torch.device(self.device),
                )
                print(f"[EDG] Warp enabled from {results_npz}")
            except Exception as exc:
                print(f"[EDG] Failed to initialize warp: {exc}")
                self.edg_warp = None

        # Dynamic segmentation loss: resolve per-frame dynamic mask directory.
        self.seg_mask_dir: Optional[Path] = None
        self._seg_mask_cache: Dict[int, Optional[Tensor]] = {}
        if (
            cfg.seg_dyn_lambda > 0.0
            and self.edg_warp is not None
            and self.edg_warp.dyn_indices is not None
            and self.edg_warp.dyn_indices.numel() > 0
        ):
            md = (
                Path(cfg.seg_dyn_mask_dir)
                if cfg.seg_dyn_mask_dir
                else _find_upwards(Path(cfg.data_dir), ["masks"], max_depth=4)
            )
            if md is not None and md.is_dir():
                self.seg_mask_dir = md
                # Sorted-by-index fallback list (same convention as utils/track_tapnet.py's
                # load_dynamic_masks): the mask filename pattern need NOT match the training
                # image stem (e.g. NVIDIA masks are "frame_00000.png" vs training image stem
                # "frame000000" -- exact-stem lookup below silently finds nothing and the
                # segmentation loss stays inactive with no error).
                self._seg_mask_paths_by_index = sorted(
                    md.glob("*.png"),
                    key=lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.stem)],
                )
                print(f"[SEG] Dynamic mask dir: {md} ({len(self._seg_mask_paths_by_index)} masks)")
            else:
                self._seg_mask_paths_by_index = []
                print("[SEG] Dynamic mask dir not found; segmentation loss disabled.")

        # EDG trainable optimizers
        self.edg_optimizers = []
        if self.edg_warp is not None:
            edg_opt = self.edg_warp.create_optimizer(
                tracks_lr=cfg.edg_tracks_lr,
                rots_lr=cfg.edg_rots_lr,
                weights_lr=cfg.edg_weights_lr,
                rel_lr=cfg.edg_rel_lr,
                batch_size=cfg.batch_size,
            )
            if edg_opt is not None:
                self.edg_optimizers = [edg_opt]
                print(
                    "[EDG] trainable parameters: "
                    f"edg_tracks_lr={cfg.edg_tracks_lr}, "
                    f"edg_rots_lr={cfg.edg_rots_lr}, "
                    f"edg_weights_lr={cfg.edg_weights_lr}, "
                    f"edg_rel_lr={cfg.edg_rel_lr}"
                )
            else:
                print("[EDG] Both EDG LRs are 0; edg_tracks/edg_rots frozen.")

        if self.edg_warp is not None and cfg.edg_apply_rgb:
            filtered = self.edg_warp._filter_dynamic(int(self.splats["means"].shape[0]))
            if filtered is not None:
                dyn_indices, _, _, _, dyn_rgb = filtered
                if dyn_rgb is not None:
                    with torch.no_grad():
                        rgb = dyn_rgb / 255.0
                        if "sh0" in self.splats:
                            self.splats["sh0"].data[dyn_indices, 0, :] = rgb_to_sh(rgb)
                        elif "colors" in self.splats:
                            rgb = rgb.clamp(1e-4, 1.0 - 1e-4)
                            self.splats["colors"].data[dyn_indices] = torch.log(rgb / (1.0 - rgb))
                    print(f"[EDG] Applied dynamic RGB to {dyn_indices.numel()} splats")

        # Upgrade strategy to EDG-aware variant when EDG warp is active
        # (only for DefaultStrategy; MCMCStrategy has its own densification path)
        if self.edg_warp is not None and isinstance(cfg.strategy, DefaultStrategy):
            base = cfg.strategy
            edg_strategy = EDGAwareDefaultStrategy(
                edg_warp=self.edg_warp,
                max_prune_fraction=cfg.depth_error_prune_max_fraction,
                max_gaussians=cfg.max_gaussians,
                edg_dyn_no_prune=cfg.edg_dyn_no_prune,
                edg_dyn_force_grow=cfg.edg_dyn_force_grow,
                # Copy every field from the base DefaultStrategy
                prune_opa=base.prune_opa,
                grow_grad2d=base.grow_grad2d,
                grow_scale3d=base.grow_scale3d,
                grow_scale2d=base.grow_scale2d,
                prune_scale3d=base.prune_scale3d,
                prune_scale2d=base.prune_scale2d,
                refine_scale2d_stop_iter=base.refine_scale2d_stop_iter,
                refine_start_iter=base.refine_start_iter,
                refine_stop_iter=base.refine_stop_iter,
                reset_every=base.reset_every,
                refine_every=base.refine_every,
                pause_refine_after_reset=base.pause_refine_after_reset,
                absgrad=base.absgrad,
                revised_opacity=base.revised_opacity,
                verbose=base.verbose,
                key_for_gradient=base.key_for_gradient,
            )
            cfg.strategy = edg_strategy
            print("[EDG] Strategy upgraded to EDGAwareDefaultStrategy")

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        self._prune_static_outlier_gaussians()
        self._trim_to_gaussian_budget()

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        self.post_processing_module = None
        if cfg.post_processing == "bilateral_grid":
            self.post_processing_module = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
        elif cfg.post_processing == "ppisp":
            ppisp_config = PPISPConfig(
                use_controller=cfg.ppisp_use_controller,
                controller_distillation=cfg.ppisp_controller_distillation,
                controller_activation_ratio=cfg.ppisp_controller_activation_num_steps
                / cfg.max_steps,
            )
            self.post_processing_module = PPISP(
                num_cameras=self.parser.num_cameras,
                num_frames=len(self.trainset),
                config=ppisp_config,
            ).to(self.device)

        self.post_processing_optimizers = []
        if cfg.post_processing == "bilateral_grid":
            self.post_processing_optimizers = [
                torch.optim.Adam(
                    self.post_processing_module.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]
        elif cfg.post_processing == "ppisp":
            self.post_processing_optimizers = (
                self.post_processing_module.create_optimizers()
            )

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        # LPIPS loads a perceptual network; skip it entirely in lite eval mode.
        self.lpips = None
        if not cfg.eval_lite:
            if cfg.lpips_net == "alex":
                self.lpips = LearnedPerceptualImagePatchSimilarity(
                    net_type="alex", normalize=True
                ).to(self.device)
            elif cfg.lpips_net == "vgg":
                # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
                self.lpips = LearnedPerceptualImagePatchSimilarity(
                    net_type="vgg", normalize=False
                ).to(self.device)
            else:
                raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training",
            )
            # EDG frame slider – broadcast to all clients (including late joiners)
            self._edg_frame_slider = None
            self._edg_show_checkbox = None
            if self.edg_warp is not None:
                self._edg_show_checkbox = self.server.gui.add_checkbox(
                    label="Show EDG",
                    initial_value=True,
                )
                init_frame = self.edg_warp.target_frame
                T = self.edg_warp.T
                self._edg_frame_slider = self.server.gui.add_slider(
                    label="EDG frame",
                    min=0,
                    max=T - 1,
                    step=1,
                    initial_value=init_frame,
                )

                @self._edg_frame_slider.on_update
                def _on_edg_frame_slider_change(_event):
                    # Update EDG scene overlay (nodes and edges)
                    self._update_edg_scene(frame_idx=self._edg_frame_slider.value)
                    # The viewer will automatically re-render using the new frame

                @self._edg_show_checkbox.on_update
                def _on_edg_show_checkbox_change(_event):
                    self._update_edg_scene(frame_idx=self._edg_frame_slider.value)

                # Also re-send slider + scene geometry to every new browser tab
                @self.server.on_client_connect
                def _on_client_connect(client: viser.ClientHandle):
                    self._update_edg_scene(frame_idx=self._edg_frame_slider.value)

            # Show initial EDG overlay
            self._update_edg_scene(frame_idx=self.edg_warp.target_frame if self.edg_warp is not None else 0)

        # Track if Gaussians are frozen (for controller distillation)
        self._gaussians_frozen = False

    def close_viewer(self):
        if self.cfg.disable_viewer:
            return
        viewer = getattr(self, "viewer", None)
        if viewer is not None:
            try:
                viewer.complete()
            except Exception as exc:
                print(f"[viewer] failed to mark viewer complete: {exc}")
        server = getattr(self, "server", None)
        if server is not None:
            for method_name in ("close", "stop"):
                method = getattr(server, method_name, None)
                if method is None:
                    continue
                try:
                    method()
                    action = "closed" if method_name == "close" else "stopped"
                    print(f"[viewer] viser server {action}; exiting.")
                    return
                except Exception as exc:
                    print(f"[viewer] failed to {method_name} viser server: {exc}")
        print("[viewer] no explicit viser close API found; exiting main process.")

    @torch.no_grad()
    def _remove_gaussians(self, mask: Tensor, reason: str) -> int:
        mask = mask.to(device=self.device, dtype=torch.bool).flatten()
        if mask.numel() == 0 or not bool(mask.any()):
            return 0
        n_before = int(mask.numel())
        if int(self.splats["means"].shape[0]) != n_before:
            raise ValueError(
                f"Remove mask length {n_before} does not match splat count {self.splats['means'].shape[0]}"
            )
        keep = torch.where(~mask)[0]
        n_remove = int(mask.sum().item())

        for name, param in list(self.splats.items()):
            old_param = param
            new_param = torch.nn.Parameter(old_param[keep], requires_grad=old_param.requires_grad)
            self.splats[name] = new_param
            opt = self.optimizers.get(name)
            if opt is None:
                continue
            old_state = opt.state.pop(old_param, {})
            for key, value in list(old_state.items()):
                if key != "step" and torch.is_tensor(value) and value.shape[:1] == (n_before,):
                    old_state[key] = value[keep]
            for group in opt.param_groups:
                group["params"] = [new_param if p is old_param else p for p in group["params"]]
            opt.state[new_param] = old_state

        state = getattr(self, "strategy_state", None)
        if isinstance(state, dict):
            for key, value in list(state.items()):
                if torch.is_tensor(value) and value.shape[:1] == (n_before,):
                    state[key] = value[keep]

        if self.edg_warp is not None:
            self.edg_warp.edg_notify_remove(mask)
        print(f"[prune] removed {n_remove}/{n_before} Gaussians ({reason}); remaining={len(keep)}")
        return n_remove

    @torch.no_grad()
    def _trim_to_gaussian_budget(self) -> None:
        """One-time startup trim so the SfM init respects cfg.max_gaussians.

        Drops the lowest-opacity *static* Gaussians (all dynamic Gaussians kept).
        Densification later refills up to the same budget as points get pruned.
        """
        cap = int(self.cfg.max_gaussians)
        n_total = int(self.splats["means"].shape[0])
        if cap <= 0 or n_total <= cap:
            return
        n_remove = n_total - cap
        static_mask = self._static_gaussian_mask(n_total, self.device)  # True = static
        static_idx = torch.where(static_mask)[0]
        if static_idx.numel() < n_remove:
            print(f"[init cap] only {static_idx.numel()} static Gaussians; cannot trim to {cap}")
            return
        opa = torch.sigmoid(self.splats["opacities"].detach())[static_idx]
        drop_idx = static_idx[torch.argsort(opa)[:n_remove]]
        mask = torch.zeros(n_total, dtype=torch.bool, device=self.device)
        mask[drop_idx] = True
        self._remove_gaussians(mask, f"init cap to {cap}")

    @torch.no_grad()
    def _static_gaussian_mask(self, n_total: int, device: torch.device) -> Tensor:
        static_mask = torch.ones((n_total,), dtype=torch.bool, device=device)
        dyn_indices = None
        if self.edg_warp is not None:
            filtered = self.edg_warp._filter_dynamic(n_total)
            if filtered is not None:
                dyn_indices = filtered[0]
        if dyn_indices is not None and dyn_indices.numel() > 0:
            dyn_indices = dyn_indices.to(device=device, dtype=torch.long)
            dyn_indices = dyn_indices[(dyn_indices >= 0) & (dyn_indices < n_total)]
            if dyn_indices.numel() > 0:
                static_mask[dyn_indices] = False
        return static_mask

    def _load_seg_mask(self, frame_index: int, height: int, width: int) -> Optional[Tensor]:
        """Load the per-frame binary dynamic mask (1=dynamic) at (height, width)."""
        if self.seg_mask_dir is None:
            return None
        m = self._seg_mask_cache.get(frame_index, "miss")
        if m == "miss":
            stem = Path(self.parser.image_paths[frame_index]).stem
            path = self.seg_mask_dir / f"{stem}.png"
            if not path.exists():
                # Mask filenames need not match the training image stem (e.g. NVIDIA:
                # "frame_00000.png" vs training stem "frame000000"); fall back to the
                # sorted-by-index mask list built in __init__.
                paths = getattr(self, "_seg_mask_paths_by_index", [])
                path = paths[frame_index] if 0 <= frame_index < len(paths) else None
            if path is None or not path.exists():
                self._seg_mask_cache[frame_index] = None
                return None
            arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if arr is None:
                self._seg_mask_cache[frame_index] = None
                return None
            if arr.ndim == 3:
                arr = arr[..., 0]
            m = torch.from_numpy((arr > 127).astype(np.float32)).to(self.device)
            self._seg_mask_cache[frame_index] = m
        if m is None:
            return None
        if m.shape[0] != height or m.shape[1] != width:
            m = F.interpolate(m[None, None], size=(height, width), mode="nearest")[0, 0]
        return m

    def _render_static_weight(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        override_means: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """Per-pixel STATIC accumulation and TOTAL alpha, each [C, H, W].

        Uses the same geometry/opacities as the RGB render but a constant
        per-Gaussian label (1=static, 0=dynamic). Returns (static_weight, total_alpha):
        static_weight = accumulated static-labelled alpha; total_alpha = accumulated
        alpha over ALL Gaussians. dynamic_weight = total_alpha - static_weight is then
        the true dynamic accumulation (0 on uncovered/background pixels).
        """
        means = override_means if override_means is not None else self.splats["means"]
        quats = self.splats["quats"]
        scales = torch.exp(self.splats["scales"])
        opacities = torch.sigmoid(self.splats["opacities"])
        static_label = self._static_gaussian_mask(means.shape[0], means.device).float().unsqueeze(-1)  # [N, 1]
        render_colors, render_alphas, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=static_label,
            viewmats=torch.linalg.inv(camtoworlds),
            Ks=Ks,
            width=width,
            height=height,
            packed=self.cfg.packed,
            render_mode="RGB",
            camera_model=self.cfg.camera_model,
        )
        return render_colors[..., 0], render_alphas[..., 0]  # each [C, H, W]

    @torch.no_grad()
    def _prune_static_outlier_gaussians(self) -> int:
        cfg = self.cfg
        if not bool(cfg.static_outlier_prune):
            return 0
        points = self.splats["means"].detach()
        n_total = int(points.shape[0])
        if n_total < int(cfg.static_outlier_min_points):
            return 0

        static_mask = self._static_gaussian_mask(n_total, points.device)
        static_idx = torch.where(static_mask)[0]
        if static_idx.numel() < int(cfg.static_outlier_min_points):
            return 0

        static_pts = points[static_idx].detach().cpu().numpy().astype(np.float64)
        finite = np.isfinite(static_pts).all(axis=1)
        outlier_local = ~finite
        finite_pts = static_pts[finite]
        if finite_pts.shape[0] >= int(cfg.static_outlier_min_points):
            try:
                from scipy.spatial import cKDTree

                k_eff = min(int(max(1, cfg.static_outlier_k)), finite_pts.shape[0] - 1)
                if k_eff > 0:
                    tree = cKDTree(finite_pts)
                    try:
                        dists, _ = tree.query(finite_pts, k=k_eff + 1, workers=-1)
                    except TypeError:
                        dists, _ = tree.query(finite_pts, k=k_eff + 1)
                    mean_knn = np.mean(dists[:, 1:], axis=1)
                    median = float(np.median(mean_knn))
                    mad = float(np.median(np.abs(mean_knn - median)))
                    robust_sigma = 1.4826 * mad
                    if not np.isfinite(robust_sigma) or robust_sigma <= 1e-12:
                        robust_sigma = float(np.std(mean_knn))
                    if np.isfinite(robust_sigma) and robust_sigma > 1e-12:
                        threshold = median + float(cfg.static_outlier_std_ratio) * robust_sigma
                        finite_outlier = mean_knn > threshold
                        outlier_local[np.where(finite)[0]] = finite_outlier
            except Exception as exc:
                print(f"[static-outlier] scipy density filter unavailable ({exc}); only removing non-finite points.")

        if not np.any(outlier_local):
            print(f"[static-outlier] no isolated static Gaussians found among {int(static_idx.numel())} static points.")
            return 0

        outlier_idx = static_idx[torch.from_numpy(outlier_local).to(device=points.device)]
        max_fraction = float(cfg.static_outlier_max_fraction)
        if max_fraction >= 0.0:
            max_remove = int(np.floor(n_total * max_fraction))
            if outlier_idx.numel() > max_remove:
                outlier_idx = outlier_idx[:max_remove]
        remove_mask = torch.zeros((n_total,), dtype=torch.bool, device=points.device)
        remove_mask[outlier_idx] = True
        return self._remove_gaussians(
            remove_mask,
            reason=(
                f"static density k={int(cfg.static_outlier_k)} "
                f"std_ratio={float(cfg.static_outlier_std_ratio):g}"
            ),
        )

    def freeze_gaussians(self):
        """Freeze all Gaussian parameters for controller distillation.

        This prevents Gaussians from being updated by any loss (including regularization)
        while the controller learns to predict per-frame corrections.
        """
        if self._gaussians_frozen:
            return

        for name, param in self.splats.items():
            param.requires_grad = False

        self._gaussians_frozen = True
        print("[Distillation] Gaussian parameters frozen")

    @torch.no_grad()
    def _prune_depth_error_gaussians(self, image_ids, camtoworlds, Ks, width, height, step) -> int:
        """Prune Gaussians whose rendered depth disagrees with the DA3 monocular depth
        (x da3_frame_scale) beyond a robust per-frame threshold. DA3 depth is the same
        monocular supervision used by the depth loss / lift, so this is NOT GT leakage."""
        cfg = self.cfg
        if not bool(cfg.depth_error_prune) or not bool(cfg.depth_loss):
            return 0
        if step < int(cfg.depth_error_prune_start_iter):
            return 0
        every = int(cfg.depth_error_prune_every)
        if every <= 0 or step % every != 0:
            return 0
        if self.da3_depth_maps is None:
            return 0
        n_total = int(self.splats["means"].shape[0])
        max_remove = int(np.floor(n_total * max(0.0, float(cfg.depth_error_prune_max_fraction))))
        if n_total <= 0 or max_remove <= 0:
            return 0

        means = self._edg_override_means(image_ids)
        if means is None:
            means = self.splats["means"]
        means = means.detach()
        candidate_mask = torch.ones((n_total,), dtype=torch.bool, device=means.device)
        if bool(cfg.depth_error_prune_static_only):
            candidate_mask &= self._static_gaussian_mask(n_total, means.device)

        T = int(self.da3_depth_maps.shape[0])
        eps = 1e-6
        camtoworlds = camtoworlds.detach()
        Ks = Ks.detach()
        all_idx, all_err, used_frames = [], [], []
        for b in range(int(camtoworlds.shape[0])):
            f = int(image_ids[b].item())
            if f < 0 or f >= T:
                continue
            scale = float(self.da3_frame_scale[f]) if self.da3_frame_scale is not None else 1.0
            gt = self.da3_depth_maps[f].to(device=means.device, dtype=means.dtype) * scale
            if gt.shape != (int(height), int(width)):
                gt = torch.nn.functional.interpolate(
                    gt[None, None], size=(int(height), int(width)), mode="bilinear", align_corners=False
                )[0, 0]
            w2c = torch.linalg.inv(camtoworlds[b])
            R, t = w2c[:3, :3], w2c[:3, 3]
            pts = means @ R.T + t.reshape(1, 3)
            z = pts[:, 2]
            K = Ks[b]
            x = K[0, 0] * (pts[:, 0] / z.clamp_min(eps)) + K[0, 2]
            y = K[1, 1] * (pts[:, 1] / z.clamp_min(eps)) + K[1, 2]
            xi, yi = torch.round(x).long(), torch.round(y).long()
            in_view = (
                candidate_mask & torch.isfinite(z) & (z > float(cfg.eval_depth_min_depth))
                & (xi >= 0) & (xi < int(width)) & (yi >= 0) & (yi < int(height))
            )
            if not bool(in_view.any()):
                continue
            idx = torch.where(in_view)[0]
            g = gt[yi[idx], xi[idx]]
            valid = torch.isfinite(g) & (g > float(cfg.eval_depth_min_depth))
            idx, g = idx[valid], g[valid]
            if idx.numel() < int(cfg.depth_error_prune_min_candidates):
                continue
            pred = z[idx]
            normalizer = torch.median(g).clamp_min(eps)
            err = torch.abs(pred - g) / normalizer
            finite = err[torch.isfinite(err)]
            if finite.numel() < int(cfg.depth_error_prune_min_candidates):
                continue
            median = torch.median(finite)
            mad = torch.median(torch.abs(finite - median))
            sigma = 1.4826 * mad
            if not bool(torch.isfinite(sigma)) or float(sigma) <= 1e-12:
                sigma = finite.std()
            thr = torch.maximum(
                median + float(cfg.depth_error_prune_std_ratio) * sigma,
                torch.tensor(float(cfg.depth_error_prune_min_error), device=means.device, dtype=means.dtype),
            )
            bad = torch.isfinite(err) & (err > thr)
            if bool(bad.any()):
                all_idx.append(idx[bad])
                all_err.append(err[bad])
                used_frames.append(f)

        if not all_idx:
            return 0
        prune_idx = torch.cat(all_idx)
        prune_err = torch.cat(all_err)
        order = torch.argsort(prune_err, descending=True)
        seen, selected = set(), []
        for local in order.tolist():
            i = int(prune_idx[local].item())
            if i in seen:
                continue
            seen.add(i)
            selected.append(i)
            if len(selected) >= max_remove:
                break
        if not selected:
            return 0
        mask = torch.zeros((n_total,), dtype=torch.bool, device=means.device)
        mask[torch.tensor(selected, dtype=torch.long, device=means.device)] = True
        return self._remove_gaussians(
            mask, reason=f"depth-error(DA3) step={step} frames={used_frames} n={len(selected)}"
        )

    def _edg_override_means(self, frame_idcs: Optional[Tensor]) -> Optional[Tensor]:
        if self.edg_warp is None or frame_idcs is None:
            return None
        frame_ids = frame_idcs.detach().flatten()
        if frame_ids.numel() == 0:
            return None
        frame_idx = int(frame_ids[0].item())
        if frame_ids.numel() > 1 and not torch.all(frame_ids == frame_ids[0]):
            if not self._edg_warned_multi_frame:
                print("[EDG] Warning: multiple frame ids in batch; using first frame for warp.")
                self._edg_warned_multi_frame = True
        return self.edg_warp.warp(self.splats["means"], frame_idx)

    def _edg_override_quats(self, frame_idcs: Optional[Tensor]) -> Optional[Tensor]:
        if self.edg_warp is None or frame_idcs is None or not self.cfg.edg_warp_quats:
            return None
        frame_ids = frame_idcs.detach().flatten()
        if frame_ids.numel() == 0:
            return None
        return self.edg_warp.warp_quats(self.splats["quats"], int(frame_ids[0].item()))

    def _edg_overlay_visible(self) -> bool:
        return bool(
            getattr(self, "_edg_show_checkbox", None) is None
            or self._edg_show_checkbox.value
        )

    @torch.no_grad()
    def _apply_hard_constraints(self) -> None:
        """Apply hard parameter constraints for training stability."""
        if "scales" not in self.splats:
            return

        rmin = float(max(self.cfg.scale_ratio_min, 1e-6))
        rmax = float(max(self.cfg.scale_ratio_max, rmin))
        # ratio = max/min is always >= 1; effective cap is max(rmax, 1).
        ratio_cap = max(rmax, 1.0)

        log_cap = float(math.log(ratio_cap))
        if log_cap <= 0.0:
            return

        s = self.splats["scales"]  # log-scales [N, 3]
        smin = torch.min(s, dim=1, keepdim=True).values
        smax = torch.max(s, dim=1, keepdim=True).values
        span = smax - smin
        too_wide = span > log_cap
        if bool(too_wide.any()):
            center = 0.5 * (smax + smin)
            half = 0.5 * log_cap
            lo = center - half
            hi = center + half
            self.splats["scales"].data = torch.clamp(s, min=lo, max=hi)

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        rasterize_mode: Optional[Literal["classic", "antialiased"]] = None,
        camera_model: Optional[Literal["pinhole", "ortho", "fisheye"]] = None,
        frame_idcs: Optional[Tensor] = None,
        camera_idcs: Optional[Tensor] = None,
        exposure: Optional[Tensor] = None,
        override_means: Optional[Tensor] = None,
        override_quats: Optional[Tensor] = None,
        gaussian_subset: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = override_means if override_means is not None else self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = override_quats if override_quats is not None else self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)  # [C, N, 3]
            if self.cfg.dyn_flat_color and self.edg_warp is not None:
                # Dynamic Gaussians: single view-independent color (bypass appearance/SH),
                # taken from their base color param only. torch.where is a no-op when there
                # are no dynamic Gaussians, so no .any() CUDA sync in the hot path.
                dyn_mask = ~self._static_gaussian_mask(means.shape[0], means.device)  # True = dynamic
                flat_all = torch.sigmoid(self.splats["colors"])  # [N, 3]
                colors = torch.where(
                    dyn_mask[None, :, None], flat_all[None].to(colors.dtype), colors
                )
        else:
            colors = torch.cat(
                [self.splats["sh0"], self.splats["shN"]], 1
            )  # [N, K, 3]
            if self.cfg.dyn_flat_color and self.edg_warp is not None and colors.shape[1] > 1:
                # sh_degree 0 for dynamic Gaussians: zero their higher-order SH (keep DC only).
                dyn_mask = ~self._static_gaussian_mask(means.shape[0], means.device)
                keep = torch.ones_like(colors)
                keep[dyn_mask, 1:, :] = 0.0
                colors = colors * keep

        # Optional render-time subset (e.g. static-only render). Applied AFTER color
        # computation so per-Gaussian colors stay aligned, then indexed down.
        if gaussian_subset is not None:
            sub = gaussian_subset
            means = means[sub]
            quats = quats[sub]
            scales = scales[sub]
            opacities = opacities[sub]
            colors = colors[:, sub] if colors.dim() == 3 else colors[sub]

        if rasterize_mode is None:
            rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        if camera_model is None:
            camera_model = self.cfg.camera_model
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            with_ut=self.cfg.with_ut,
            with_eval3d=self.cfg.with_eval3d,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0

        if self.cfg.post_processing is not None:
            # Create pixel coordinates [H, W, 2] with +0.5 center offset
            pixel_y, pixel_x = torch.meshgrid(
                torch.arange(height, device=self.device) + 0.5,
                torch.arange(width, device=self.device) + 0.5,
                indexing="ij",
            )
            pixel_coords = torch.stack([pixel_x, pixel_y], dim=-1)  # [H, W, 2]

            # Split RGB from extra channels (e.g. depth) for post-processing
            rgb = render_colors[..., :3]
            extra = render_colors[..., 3:] if render_colors.shape[-1] > 3 else None

            if self.cfg.post_processing == "bilateral_grid":
                if frame_idcs is not None:
                    grid_xy = (
                        pixel_coords / torch.tensor([width, height], device=self.device)
                    ).unsqueeze(0)
                    rgb = slice(
                        self.post_processing_module,
                        grid_xy.expand(rgb.shape[0], -1, -1, -1),
                        rgb,
                        frame_idcs.unsqueeze(-1),
                    )["rgb"]
            elif self.cfg.post_processing == "ppisp":
                camera_idx = camera_idcs.item() if camera_idcs is not None else None
                frame_idx = frame_idcs.item() if frame_idcs is not None else None
                rgb = self.post_processing_module(
                    rgb=rgb,
                    pixel_coords=pixel_coords,
                    resolution=(width, height),
                    camera_idx=camera_idx,
                    frame_idx=frame_idx,
                    exposure_prior=exposure,
                )

            render_colors = (
                torch.cat([rgb, extra], dim=-1) if extra is not None else rgb
            )

        return render_colors, render_alphas, info

    def _scaled_render_inputs(
        self,
        Ks: Tensor,
        width: int,
        height: int,
        scale: float,
        pixels: Optional[Tensor] = None,
        masks: Optional[Tensor] = None,
    ) -> Tuple[Tensor, int, int, Optional[Tensor], Optional[Tensor]]:
        """Downscale render-only inputs for faster previews/evaluation."""
        scale = float(scale)
        if scale <= 0.0:
            raise ValueError(f"render_scale must be > 0, got {scale}")
        if abs(scale - 1.0) <= 1e-8:
            return Ks, int(width), int(height), pixels, masks

        new_width = max(1, int(round(int(width) * scale)))
        new_height = max(1, int(round(int(height) * scale)))
        sx = float(new_width) / float(max(int(width), 1))
        sy = float(new_height) / float(max(int(height), 1))

        Ks_scaled = Ks.clone()
        Ks_scaled[..., 0, :] *= sx
        Ks_scaled[..., 1, :] *= sy

        pixels_scaled = pixels
        if pixels is not None:
            pixels_scaled = F.interpolate(
                pixels.permute(0, 3, 1, 2),
                size=(new_height, new_width),
                mode="area",
            ).permute(0, 2, 3, 1)

        masks_scaled = masks
        if masks is not None:
            masks_scaled = F.interpolate(
                masks[:, None].float(),
                size=(new_height, new_width),
                mode="nearest",
            )[:, 0].bool()

        return Ks_scaled, new_width, new_height, pixels_scaled, masks_scaled

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        # EDG learning-rate step decay: multiply ALL EDG param-group lrs (tracks/rots/skinning) by
        # edg_lr_decay_gamma at EACH milestone in edg_lr_decay_steps (gammas compound). Empty = off.
        if self.edg_optimizers and len(cfg.edg_lr_decay_steps) > 0:
            schedulers.append(
                torch.optim.lr_scheduler.MultiStepLR(
                    self.edg_optimizers[0],
                    milestones=sorted(int(s) for s in cfg.edg_lr_decay_steps if int(s) > 0),
                    gamma=float(cfg.edg_lr_decay_gamma),
                )
            )
        # Post-processing module has a learning rate schedule
        if cfg.post_processing == "bilateral_grid":
            # Linear warmup + exponential decay
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.post_processing_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.post_processing_optimizers[0],
                            gamma=0.01 ** (1.0 / max_steps),
                        ),
                    ]
                )
            )
        elif cfg.post_processing == "ppisp":
            ppisp_schedulers = self.post_processing_module.create_schedulers(
                self.post_processing_optimizers,
                max_optimization_iters=max_steps,
            )
            schedulers.extend(ppisp_schedulers)

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            # Freeze Gaussians when PPISP controller distillation starts
            if (
                cfg.post_processing == "ppisp"
                and cfg.ppisp_use_controller
                and cfg.ppisp_controller_distillation
                and step >= cfg.ppisp_controller_activation_num_steps
            ):
                self.freeze_gaussians()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            camera_idx = data["camera_idx"].to(device)
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
            exposure = (
                data["exposure"].to(device) if "exposure" in data else None
            )  # [B,]
            if cfg.depth_loss:
                points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            override_means = self._edg_override_means(image_ids)
            override_quats = self._edg_override_quats(image_ids)

            # forward
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED",
                masks=masks,
                frame_idcs=image_ids,
                camera_idcs=data["camera_idx"].to(device),
                exposure=exposure,
                override_means=override_means,
                override_quats=override_quats,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            # save image for debugging (GT + render)
            if (
                cfg.debug_save_images
                and world_rank == 0
                and cfg.debug_save_every > 0
                and step % cfg.debug_save_every == 0
            ):
                try:
                    # pixels and colors are [1, H, W, 3]
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    # reshape to (B*H, W, C) if batch>1; usually batch=1
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    out_path = f"{self.render_dir}/train_step{step:04d}_rank{self.world_rank}.png"
                    imageio.imwrite(out_path, (canvas * 255).astype(np.uint8))
                    if depths is not None:
                        valid_depth = torch.isfinite(depths) & (depths > 0.0)
                        if bool(valid_depth.any()):
                            vals = depths[valid_depth]
                            near = torch.quantile(vals, 0.02)
                            far = torch.quantile(vals, 0.98)
                            depth_norm = (depths - near) / (far - near + 1e-10)
                            depth_norm = torch.clamp(depth_norm, 0.0, 1.0)
                            depth_norm = torch.where(valid_depth, depth_norm, torch.zeros_like(depth_norm))
                        else:
                            depth_norm = torch.zeros_like(depths)
                        depth_rgb = depth_norm.repeat(1, 1, 1, 3)
                        depth_canvas = torch.cat([colors, depth_rgb], dim=2).detach().cpu().numpy()
                        depth_canvas = depth_canvas.reshape(-1, *depth_canvas.shape[2:])
                        depth_out_path = f"{self.render_dir}/train_step{step:04d}_rank{self.world_rank}_depth.png"
                        imageio.imwrite(depth_out_path, (depth_canvas * 255).astype(np.uint8))
                except Exception:
                    # don't fail training for debug saving errors
                    import traceback
                    traceback.print_exc()

            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            # loss
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            if cfg.depth_loss:
                frame_index = int(image_ids.flatten()[0].item())
                # query depths from depth map
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [1, M]
                # align monocular GT depth to metric scale using the solver's
                # per-frame scale, then normalized-L1 against the render.
                scale_t = (
                    float(self.da3_frame_scale[frame_index])
                    if self.da3_frame_scale is not None
                    else 1.0
                )
                gt_metric = depths_gt * scale_t
                valid = (
                    torch.isfinite(depths)
                    & torch.isfinite(gt_metric)
                    & (depths > 0.0)
                    & (gt_metric > 0.0)
                )
                if bool(valid.any()):
                    normalizer = torch.median(gt_metric[valid]).clamp_min(1e-6)
                    depthloss = torch.mean(
                        torch.abs(depths[valid] - gt_metric[valid]) / normalizer
                    )
                else:
                    depthloss = torch.zeros([], device=depths.device)
                loss += depthloss * cfg.depth_lambda            
                
            if cfg.post_processing == "bilateral_grid":
                post_processing_reg_loss = 10 * total_variation_loss(
                    self.post_processing_module.grids
                )
                loss += post_processing_reg_loss
            elif cfg.post_processing == "ppisp":
                post_processing_reg_loss = (
                    self.post_processing_module.get_regularization_loss()
                )
                loss += post_processing_reg_loss

            # regularizations
            if cfg.opacity_reg > 0.0:
                loss += cfg.opacity_reg * torch.sigmoid(self.splats["opacities"]).mean()
            if cfg.scale_reg > 0.0:
                loss += cfg.scale_reg * torch.exp(self.splats["scales"]).mean()

            # EDG regularization: ARAP + temporal smoothness.
            # Keep tensors and defer .item() to the logging step (avoids per-step syncs).
            edg_arap_t = edg_smooth_t = edg_skinning_reg_t = None
            if self.edg_warp is not None and (cfg.edg_arap_lambda > 0.0 or cfg.edg_smooth_lambda > 0.0):
                edg_arap_t, edg_smooth_t = self.edg_warp.edg_regularization_loss(
                    arap_lambda=cfg.edg_arap_lambda,
                    smooth_lambda=cfg.edg_smooth_lambda,
                    arap_gaps=tuple(cfg.edg_arap_gaps),
                    arap_eta_len=cfg.edg_arap_eta_len,
                )
                loss = loss + edg_arap_t + edg_smooth_t
            if self.edg_warp is not None and cfg.edg_skinning_reg_lambda > 0.0:
                edg_skinning_reg_t = self.edg_warp.skinning_regularization_loss(cfg.edg_skinning_reg_lambda)
                loss = loss + edg_skinning_reg_t
            # Observation anchors (complete_edg_table L_init/L_obs analogues): keep the learned EDG and
            # the warped dynamic positions near the observation-fitted solve (see Config docstrings).
            if self.edg_warp is not None and (cfg.edg_obs_lambda > 0.0 or cfg.dyn_obs_lambda > 0.0):
                _fi = int(image_ids.flatten()[0].item())
                edg_obs_t, dyn_obs_t = self.edg_warp.observation_anchor_loss(
                    _fi, cfg.edg_obs_lambda, cfg.dyn_obs_lambda, cfg.dyn_obs_samples)
                loss = loss + edg_obs_t + dyn_obs_t

            # Dynamic/static segmentation losses, all keyed off the per-frame dynamic mask.
            #   seg_dyn_lambda        -- penalize STATIC contribution INSIDE  the mask
            #   dyn_outside_lambda    -- penalize DYNAMIC contribution OUTSIDE the mask
            #   static_only_region_.. -- supervise a STATIC-ONLY render vs GT OUTSIDE the mask
            seg_loss_val = 0.0
            dyn_out_loss_val = 0.0
            static_only_loss_val = 0.0
            # These losses are meaningless without dynamic Gaussians (EDG warp).
            seg_active = (
                self.seg_mask_dir is not None
                and self.edg_warp is not None
                and step >= cfg.seg_dyn_start_iter
                and step % max(cfg.seg_dyn_every, 1) == 0
            )
            want_static_weight = seg_active and (
                cfg.seg_dyn_lambda > 0.0 or cfg.dyn_outside_lambda > 0.0
            )
            if seg_active and (want_static_weight or cfg.static_only_region_lambda > 0.0):
                frame_index = int(image_ids.flatten()[0].item())
                seg_mask = self._load_seg_mask(frame_index, height, width)  # [H, W], 1=dynamic
                if seg_mask is not None:
                    inside = (seg_mask > 0).float()
                    outside = 1.0 - inside
                    static_weight = None
                    dyn_weight = None
                    if want_static_weight:
                        static_weight, total_alpha = self._render_static_weight(
                            camtoworlds, Ks, width, height, override_means
                        )
                        static_weight = static_weight[0]  # [H, W]
                        # True dynamic accumulation (0 on uncovered/background pixels), not 1-static.
                        dyn_weight = (total_alpha[0] - static_weight).clamp_min(0.0)
                    if cfg.seg_dyn_lambda > 0.0 and static_weight is not None and bool(inside.any()):
                        seg_loss = (static_weight * inside).sum() / inside.sum().clamp_min(1.0)
                        loss = loss + cfg.seg_dyn_lambda * seg_loss
                        seg_loss_val = seg_loss.item()
                    if cfg.dyn_outside_lambda > 0.0 and dyn_weight is not None and bool(outside.any()):
                        dyn_out_loss = (dyn_weight * outside).sum() / outside.sum().clamp_min(1.0)
                        loss = loss + cfg.dyn_outside_lambda * dyn_out_loss
                        dyn_out_loss_val = dyn_out_loss.item()
                    if cfg.static_only_region_lambda > 0.0 and bool(outside.any()):
                        static_mask = self._static_gaussian_mask(
                            override_means.shape[0] if override_means is not None
                            else self.splats["means"].shape[0],
                            self.device,
                        )
                        static_rgb, _, _ = self.rasterize_splats(
                            camtoworlds=camtoworlds,
                            Ks=Ks,
                            width=width,
                            height=height,
                            sh_degree=sh_degree_to_use,
                            near_plane=cfg.near_plane,
                            far_plane=cfg.far_plane,
                            image_ids=image_ids,
                            render_mode="RGB",
                            override_means=override_means,
                            gaussian_subset=static_mask,
                        )
                        so_err = F.l1_loss(static_rgb[..., :3], pixels, reduction="none").mean(-1)  # [1, H, W]
                        so_loss = (so_err[0] * outside).sum() / outside.sum().clamp_min(1.0)
                        loss = loss + cfg.static_only_region_lambda * so_loss
                        static_only_loss_val = so_loss.item()

            loss.backward()

            # Logging is gated to tb_every so the per-step hot path has no
            # .item()/CUDA syncs. tqdm still shows time/iter every step.
            is_log = (
                world_rank == 0
                and cfg.tb_every > 0
                and (step % cfg.tb_every == 0 or step == max_steps - 1)
            )
            if is_log:
                with torch.no_grad():
                    mseloss = F.mse_loss(colors, pixels)
                    train_psnr = -10.0 * torch.log10(mseloss.clamp_min(1e-10))
                desc = f"psnr={train_psnr.item():.2f}| "
                if cfg.depth_loss:
                    desc += f"depth={depthloss.item():.6f}| "
                if seg_loss_val > 0.0:
                    desc += f"seg={seg_loss_val:.4f}| "
                if dyn_out_loss_val > 0.0:
                    desc += f"dynout={dyn_out_loss_val:.4f}| "
                if static_only_loss_val > 0.0:
                    desc += f"statonly={static_only_loss_val:.4f}| "
                pbar.set_description(desc)

                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/mseloss", mseloss.item(), step)
                self.writer.add_scalar("train/psnr", train_psnr.item(), step)
                self.writer.add_scalar("train/ssim", (1.0 - ssimloss).item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.post_processing is not None:
                    self.writer.add_scalar(
                        "train/post_processing_reg_loss",
                        post_processing_reg_loss.item(),
                        step,
                    )
                if self.edg_warp is not None:
                    with torch.no_grad():
                        _dtr = (self.edg_warp.edg_tracks - self.edg_warp.edg_tracks_init).norm(dim=-1)
                        _dro = (self.edg_warp.edg_rots - self.edg_warp.edg_rots_init).norm(dim=[-1, -2])
                        edg_tracks_dist = _dtr.mean().item()
                        edg_rots_dist = _dro.mean().item()
                        edg_tracks_max = _dtr.max().item()
                        edg_rots_max = _dro.max().item()
                    self.writer.add_scalar("train/edg_tracks_dist", edg_tracks_dist, step)
                    self.writer.add_scalar("train/edg_rots_dist",   edg_rots_dist,   step)
                    # PRINT the EDG pose change vs its INITIAL state (positions in solver units,
                    # rotations as Frobenius norm of R - R_init) so the EDG updates are visible live.
                    pbar.write(
                        f"[EDG-drift] step {step}: pos mean={edg_tracks_dist:.6f} max={edg_tracks_max:.6f} | "
                        f"rot(Frob) mean={edg_rots_dist:.6f} max={edg_rots_max:.6f}"
                        + ("  (warmup: EDG frozen)" if step < cfg.edg_warmup_steps else "")
                    )
                    if edg_skinning_reg_t is not None:
                        self.writer.add_scalar("train/edg_skinning_reg", edg_skinning_reg_t.item(), step)
                    if edg_arap_t is not None:
                        self.writer.add_scalar("train/edg_arap_loss",   edg_arap_t.item(),   step)
                        self.writer.add_scalar("train/edg_smooth_loss", edg_smooth_t.item(), step)
                if self.seg_mask_dir is not None:
                    self.writer.add_scalar("train/seg_dyn_loss", seg_loss_val, step)
                    self.writer.add_scalar("train/dyn_outside_loss", dyn_out_loss_val, step)
                    self.writer.add_scalar("train/static_only_loss", static_only_loss_val, step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            # save checkpoint before updating the model
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {"step": step, "splats": self.splats.state_dict()}
                if cfg.pose_opt:
                    if world_size > 1:
                        data["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        data["pose_adjust"] = self.pose_adjust.state_dict()
                if cfg.app_opt:
                    if world_size > 1:
                        data["app_module"] = self.app_module.module.state_dict()
                    else:
                        data["app_module"] = self.app_module.state_dict()
                if self.post_processing_module is not None:
                    data["post_processing"] = self.post_processing_module.state_dict()
                if self.edg_warp is not None:
                    data["edg_warp"] = self.edg_warp.state_dict()
                ckpt_path = f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                torch.save(data, ckpt_path)
                if step == max_steps - 1 and cfg.save_final_gaussians:
                    final_rank_path = f"{self.ckpt_dir}/final_rank{self.world_rank}.pt"
                    torch.save(data, final_rank_path)
                    if world_rank == 0:
                        final_path = f"{cfg.result_dir}/{cfg.final_gaussians_name}"
                        torch.save(data, final_path)
                        print(f"Final gaussians saved to {final_path}")
            if (
                step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1
            ) and cfg.save_ply:

                if self.cfg.app_opt:
                    # eval at origin to bake the appeareance into the colors
                    rgb = self.app_module(
                        features=self.splats["features"],
                        embed_ids=None,
                        dirs=torch.zeros_like(self.splats["means"][None, :, :]),
                        sh_degree=sh_degree_to_use,
                    )
                    rgb = rgb + self.splats["colors"]
                    rgb = torch.sigmoid(rgb).squeeze(0).unsqueeze(1)
                    sh0 = rgb_to_sh(rgb)
                    shN = torch.empty([sh0.shape[0], 0, 3], device=sh0.device)
                else:
                    sh0 = self.splats["sh0"]
                    shN = self.splats["shN"]

                means = self.splats["means"]
                scales = self.splats["scales"]
                quats = self.splats["quats"]
                opacities = self.splats["opacities"]
                export_splats(
                    means=means,
                    scales=scales,
                    quats=quats,
                    opacities=opacities,
                    sh0=sh0,
                    shN=shN,
                    format="ply",
                    save_to=f"{self.ply_dir}/point_cloud_{step}.ply",
                )

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            if cfg.visible_adam:
                gaussian_cnt = self.splats.means.shape[0]
                if cfg.packed:
                    visibility_mask = torch.zeros_like(
                        self.splats["opacities"], dtype=bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).all(-1).any(0)

            # Warm-up: freeze EDG while Gaussian geometry/color stabilizes.
            if self.edg_warp is not None and step < cfg.edg_warmup_steps:
                if self.edg_warp.edg_tracks.grad is not None:
                    self.edg_warp.edg_tracks.grad.zero_()
                if self.edg_warp.edg_rots.grad is not None:
                    self.edg_warp.edg_rots.grad.zero_()
                if self.edg_warp.dyn_knn_logits is not None and self.edg_warp.dyn_knn_logits.grad is not None:
                    self.edg_warp.dyn_knn_logits.grad.zero_()
                if self.edg_warp.dyn_rel is not None and self.edg_warp.dyn_rel.grad is not None:
                    self.edg_warp.dyn_rel.grad.zero_()
            elif self.edg_warp is not None and cfg.edg_grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    self.edg_warp.trainable_parameters(),
                    max_norm=float(cfg.edg_grad_clip),
                )

            # optimize
            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            # Hard scale cap for the WHOLE run (log-scale space). The too-big prune only
            # fires before refine_stop_iter; this clamp runs every step so large Gaussians
            # never accumulate and corrupt novel-view depth.
            if cfg.max_scale3d > 0.0:
                with torch.no_grad():
                    self.splats["scales"].clamp_(
                        max=math.log(float(cfg.max_scale3d) * self.scene_scale)
                    )
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.edg_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.post_processing_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            self._apply_hard_constraints()
            if self.edg_warp is not None:
                self.edg_warp.apply_skinning_constraints(cfg.edg_rel_delta_max)

            # Run post-backward steps after backward and optimizer
            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            # Depth-error prune (DA3-supervised): remove wrong-depth floaters that would
            # corrupt novel views. Self-gates on cfg.depth_error_prune + schedule.
            self._prune_depth_error_gaussians(image_ids, camtoworlds, Ks, width, height, step)

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)
                if cfg.eval_render_traj:
                    self.render_traj(step)
                if cfg.custom_eval_first_frame:
                    self.eval_first_frame_pose(step)

            # run compression
            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.render_tab_state.num_train_rays_per_sec = (
                    num_train_rays_per_sec
                )
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)
                # Refresh EDG overlay every 10 steps so it tracks optimised positions.
                if self.edg_warp is not None and step % 10 == 0:
                    fid = self._edg_frame_slider.value if self._edg_frame_slider is not None else self.edg_warp.target_frame
                    self._update_edg_scene(frame_idx=fid)

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation."""
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        if len(valloader) == 0:
            print(f"[eval] Val set is empty (test_every={cfg.test_every}); skipping evaluation.")
            return

        eval_limit = int(cfg.eval_max_frames)
        if eval_limit > 0:
            print(f"Running evaluation on at most {eval_limit} frame(s)...")
        else:
            print("Running evaluation...")
        if float(cfg.render_scale) != 1.0:
            print(f"[render] eval render_scale={float(cfg.render_scale):.3f}")
        ellipse_time = 0
        eval_count = 0
        metrics = defaultdict(list)
        for i, data in enumerate(valloader):
            if eval_limit > 0 and eval_count >= eval_limit:
                break
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]
            Ks, width, height, pixels, masks = self._scaled_render_inputs(
                Ks=Ks,
                width=width,
                height=height,
                scale=cfg.render_scale,
                pixels=pixels,
                masks=masks,
            )

            # Exposure metadata is available for any image with EXIF data (train or val)
            exposure = data["exposure"].to(device) if "exposure" in data else None
            image_ids = data["image_id"].to(device)
            camera_idx = data["camera_idx"].to(device)
            frame_idcs = image_ids
            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)
            override_means = self._edg_override_means(frame_idcs)

            torch.cuda.synchronize()
            tic = time.time()
            renders, alphas, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
                frame_idcs=frame_idcs,
                camera_idcs=camera_idx,
                exposure=exposure,
                image_ids=image_ids,
                override_means=override_means,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)
            eval_count += 1

            colors = renders[..., :3]
            depths = renders[..., 3:4]
            colors = torch.clamp(colors, 0.0, 1.0)
            canvas_list = [pixels, colors]

            if world_rank == 0:
                # write images
                canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                    canvas,
                )
                if not cfg.eval_lite:
                    valid_depth = torch.isfinite(depths) & (depths > 0.0)
                    if bool(valid_depth.any()):
                        vals = depths[valid_depth]
                        near = torch.quantile(vals, 0.02)
                        far = torch.quantile(vals, 0.98)
                        depth_norm = (depths - near) / (far - near + 1e-10)
                        depth_norm = torch.clamp(depth_norm, 0.0, 1.0)
                        depth_norm = torch.where(valid_depth, depth_norm, torch.zeros_like(depth_norm))
                    else:
                        depth_norm = torch.zeros_like(depths)
                    depth_rgb = depth_norm.repeat(1, 1, 1, 3)
                    depth_canvas = torch.cat([colors, depth_rgb], dim=2).squeeze(0).cpu().numpy()
                    depth_canvas = (depth_canvas * 255).astype(np.uint8)
                    imageio.imwrite(
                        f"{self.render_dir}/{stage}_step{step}_{i:04d}_depth.png",
                        depth_canvas,
                    )

                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                if not cfg.eval_lite:
                    metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                    metrics["lpips"].append(self.lpips(colors_p, pixels_p))
                # Compute color-corrected metrics for fair comparison across methods
                if cfg.use_color_correction_metric and not cfg.eval_lite:
                    if cfg.color_correct_method == "affine":
                        cc_colors = color_correct_affine(colors, pixels)
                    else:
                        cc_colors = color_correct_quadratic(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))
                    metrics["cc_ssim"].append(self.ssim(cc_colors_p, pixels_p))
                    metrics["cc_lpips"].append(self.lpips(cc_colors_p, pixels_p))

        if world_rank == 0:
            if eval_count == 0:
                print("[eval] No frames rendered after applying eval_max_frames; skipping stats.")
                return
            ellipse_time /= eval_count

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["means"]),
                }
            )
            if cfg.eval_lite:
                print(
                    f"PSNR: {stats['psnr']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
                )
            elif cfg.use_color_correction_metric:
                print(
                    f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"CC_PSNR: {stats['cc_psnr']:.3f}, CC_SSIM: {stats['cc_ssim']:.4f}, CC_LPIPS: {stats['cc_lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
                )
            else:
                print(
                    f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
                )
            # save stats as json
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                if isinstance(v, (int, float, np.integer, np.floating)):
                    self.writer.add_scalar(f"{stage}/{k}", float(v), step)
            self.writer.flush()

    @torch.no_grad()
    def eval_target_frame_pose(self, step: int, target_frame: int):
        """Render all frames using the target camera pose; save frames + video under test/."""
        cfg = self.cfg
        device = self.device

        camtoworlds_all = self.parser.camtoworlds
        if camtoworlds_all is None or len(camtoworlds_all) == 0:
            print("[EDG] No camera poses available for custom eval.")
            return
        target_frame = int(np.clip(target_frame, 0, len(camtoworlds_all) - 1))

        out_dir = Path(cfg.result_dir) / "test" / f"first_frame_eval_{step:04d}"
        frames_dir = out_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        cam0 = torch.from_numpy(camtoworlds_all[target_frame]).float().to(device)
        cam0 = cam0[None]
        target_camera_id = self.parser.camera_ids[target_frame]
        K = torch.from_numpy(self.parser.Ks_dict[target_camera_id]).float().to(device)
        width, height = self.parser.imsize_dict[target_camera_id]
        K_batched, width, height, _, _ = self._scaled_render_inputs(
            Ks=K[None],
            width=width,
            height=height,
            scale=cfg.render_scale,
        )
        K = K_batched[0]

        video_path = out_dir / f"first_frame_eval_{step:04d}.mp4"
        writer = None if cfg.disable_video else imageio.get_writer(str(video_path), fps=30)

        num_frames = len(camtoworlds_all)
        if int(cfg.custom_eval_max_frames) > 0:
            num_frames = min(num_frames, int(cfg.custom_eval_max_frames))
        if float(cfg.render_scale) != 1.0:
            print(f"[render] custom eval render_scale={float(cfg.render_scale):.3f}")
        for idx in tqdm.trange(num_frames, desc="Custom eval (first pose)"):
            frame_idcs = torch.tensor([idx], device=device, dtype=torch.long)
            camera_idx = target_frame
            if hasattr(self.parser, "camera_indices"):
                try:
                    camera_idx = int(self.parser.camera_indices[idx])
                except Exception:
                    camera_idx = 0
            camera_idcs = torch.tensor([camera_idx], device=device, dtype=torch.long)

            override_means = self._edg_override_means(frame_idcs)

            colors, _, _ = self.rasterize_splats(
                camtoworlds=cam0,
                Ks=K[None],
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB",
                frame_idcs=frame_idcs,
                camera_idcs=camera_idcs,
                image_ids=frame_idcs,
                override_means=override_means,
            )
            colors = torch.clamp(colors, 0.0, 1.0)
            frame = (colors[0].detach().cpu().numpy() * 255).astype(np.uint8)
            frame_path = frames_dir / f"frame_{idx:04d}.png"
            imageio.imwrite(frame_path, frame)
            if writer is not None:
                writer.append_data(frame)

        if writer is not None:
            writer.close()
        print(f"[EDG] Custom eval saved to {out_dir}")
        
    @torch.no_grad()
    def eval_first_frame_pose(self, step: int):
        return self.eval_target_frame_pose(step, self.edg_warp.target_frame if self.edg_warp is not None else 0)

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        if self.cfg.disable_video:
            return
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds_all = self.parser.camtoworlds[5:-5]
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 1
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        if int(cfg.traj_max_frames) > 0:
            camtoworlds_all = camtoworlds_all[: int(cfg.traj_max_frames)]

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        camera_id = self.parser.camera_ids[0]
        K = torch.from_numpy(self.parser.Ks_dict[camera_id]).float().to(device)
        width, height = self.parser.imsize_dict[camera_id]
        K_batched, width, height, _, _ = self._scaled_render_inputs(
            Ks=K[None],
            width=width,
            height=height,
            scale=cfg.render_scale,
        )
        K = K_batched[0]
        if float(cfg.render_scale) != 1.0:
            print(f"[render] trajectory render_scale={float(cfg.render_scale):.3f}")

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            depths = renders[..., 3:4]  # [1, H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas_list = [colors, depths.repeat(1, 1, 1, 3)]

            # write images
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def export_ppisp_reports(self) -> None:
        """Export PPISP visualization reports (PDF) and parameter JSON."""
        if self.cfg.post_processing != "ppisp":
            return
        print("Exporting PPISP reports...")

        # Compute frames per camera from training dataset
        num_cameras = self.parser.num_cameras
        frames_per_camera = [0] * num_cameras
        for idx in self.trainset.indices:
            cam_idx = self.parser.camera_indices[idx]
            frames_per_camera[cam_idx] += 1

        # Generate camera names from COLMAP camera IDs
        # camera_id_to_idx maps COLMAP ID -> 0-based index
        idx_to_camera_id = {v: k for k, v in self.parser.camera_id_to_idx.items()}
        camera_names = [f"camera_{idx_to_camera_id[i]}" for i in range(num_cameras)]

        # Export reports
        output_dir = Path(self.cfg.result_dir) / "ppisp_reports"
        pdf_paths = export_ppisp_report(
            self.post_processing_module,
            frames_per_camera,
            output_dir,
            camera_names=camera_names,
        )
        print(f"PPISP reports saved to {output_dir}")
        for path in pdf_paths:
            print(f"  - {path.name}")

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def _update_edg_scene(self, frame_idx: int = 0) -> None:
        """Push EDG node positions and graph edges to the viser scene overlay."""
        if self.cfg.disable_viewer or self.edg_warp is None:
            return
        if not self._edg_overlay_visible():
            empty_pts = np.zeros((0, 3), dtype=np.float32)
            empty_lines = np.zeros((0, 2, 3), dtype=np.float32)
            self.server.scene.add_point_cloud(
                name="edg/nodes",
                points=empty_pts,
                colors=np.zeros((0, 3), dtype=np.uint8),
                point_size=0.01,
                point_shape="circle",
            )
            self.server.scene.add_line_segments(
                name="edg/edges",
                points=empty_lines,
                colors=np.zeros((0, 2, 3), dtype=np.uint8),
                line_width=1,
            )
            self.server.scene.add_line_segments(
                name="edg/trajectories",
                points=empty_lines,
                colors=np.zeros((0, 2, 3), dtype=np.uint8),
                line_width=0.2,
            )
            return
        node_pts, line_pts = self.edg_warp.edg_scene_arrays(frame_idx)
        
        # Large red spherical markers for each EDG node
        self.server.scene.add_point_cloud(
            name="edg/nodes",
            points=node_pts,
            colors=np.tile(np.array([255, 30, 30], dtype=np.uint8), (len(node_pts), 1)),
            point_size=0.01,
            point_shape="circle",
        )
        
        # White lines connecting EDG graph edges
        if line_pts.shape[0] > 0:
            self.server.scene.add_line_segments(
                name="edg/edges",
                points=line_pts,
                colors=np.tile(np.array([220, 220, 220], dtype=np.uint8), (len(line_pts), 2, 1)),
                line_width=1,
            )
        
        # Add colorful trajectory tails for dynamic points
        self._add_dynamic_trajectories(frame_idx)

    def _add_dynamic_trajectories(self, current_frame: int) -> None:
        """Add colorful trajectory tails for 500 selected dynamic points (optimized)."""
        if self.edg_warp is None or self.edg_warp.dyn_indices is None:
            return
        
        # Get filtered dynamic data
        n_gaussians = int(self.splats["means"].shape[0])
        filtered = self.edg_warp._filter_dynamic(n_gaussians)
        if filtered is None:
            return
        
        dyn_indices, dyn_knn_idx, dyn_knn_w, dyn_rel, _ = filtered
        n_dynamic = dyn_indices.numel()
        if n_dynamic == 0:
            return
        
        # Validate dynamic indices are within bounds
        valid_mask = (dyn_indices >= 0) & (dyn_indices < n_gaussians)
        if not valid_mask.any():
            return
        
        # Filter to only valid indices
        valid_indices = torch.where(valid_mask)[0]
        n_valid = valid_indices.numel()
        
        # Select subset for trajectories (500 or less)
        n_show = min(500, n_valid)
        
        # Select evenly spaced indices
        if n_valid > n_show:
            step = n_valid // n_show
            selected_valid = valid_indices[::step][:n_show]
        else:
            selected_valid = valid_indices
        
        n_selected = selected_valid.numel()
        if n_selected == 0:
            return
        
        # Get the dynamic point data for selected points
        sel_dyn_knn_idx = dyn_knn_idx[selected_valid]  # [n_selected, k]
        sel_dyn_knn_w = dyn_knn_w[selected_valid]      # [n_selected, k]
        sel_dyn_rel = dyn_rel[selected_valid]          # [n_selected, k, 3]
        
        # Parameters for trajectory
        trail_length = min(10, self.edg_warp.T)
        start_frame = max(0, current_frame - trail_length + 1)
        end_frame = current_frame + 1
        n_frames = end_frame - start_frame
        
        if n_frames < 2:
            return
        
        # Pre-compute all trajectory positions at once using vectorized ops
        # Shape: [n_frames, n_selected, 3]
        trajectory_positions = []
        
        for frame in range(start_frame, end_frame):
            if frame < 0 or frame >= self.edg_warp.T:
                continue
            
            # Get EDG node positions and rotations for this frame
            edg_t = self.edg_warp.edg_tracks[:, frame, :]       # [N_nodes, 3]
            edg_rot_t = self.edg_warp.edg_rots[:, frame, :, :]  # [N_nodes, 3, 3]
            
            # Gather for selected points: [n_selected, k, 3] and [n_selected, k, 3, 3]
            edg_pos = edg_t[sel_dyn_knn_idx]      # [n_selected, k, 3]
            edg_rot = edg_rot_t[sel_dyn_knn_idx]  # [n_selected, k, 3, 3]
            
            # Compute warped positions: edg_pos + R @ rel
            rel = sel_dyn_rel.unsqueeze(-1)  # [n_selected, k, 3, 1]
            warped = edg_pos + torch.matmul(edg_rot, rel).squeeze(-1)  # [n_selected, k, 3]
            
            # Weighted sum
            w = sel_dyn_knn_w.unsqueeze(-1)  # [n_selected, k, 1]
            positions = torch.sum(warped * w, dim=1)  # [n_selected, 3]
            
            trajectory_positions.append(positions.detach().cpu().numpy())
        
        if len(trajectory_positions) < 2:
            return
        
        # Stack: [n_frames, n_selected, 3]
        trajectory_positions = np.stack(trajectory_positions, axis=0)
        n_frames_actual = trajectory_positions.shape[0]
        
        # Generate colors for each trajectory (rainbow spectrum)
        hue_values = np.linspace(0, 1, n_selected, endpoint=False)
        
        # Build line segments efficiently using numpy
        trajectory_lines = []
        trajectory_colors = []
        
        for t in range(n_frames_actual - 1):
            # Line segments from frame t to t+1 for all points
            starts = trajectory_positions[t]      # [n_selected, 3]
            ends = trajectory_positions[t + 1]    # [n_selected, 3]
            
            # Time-based alpha (brightness)
            time_alpha = (t + 1) / n_frames_actual
            
            for i in range(n_selected):
                line_segment = np.array([starts[i], ends[i]], dtype=np.float32)
                trajectory_lines.append(line_segment)
                
                # HSV to RGB color
                r, g, b = colorsys.hsv_to_rgb(hue_values[i], 1.0, time_alpha)
                color = np.array([int(r * 255), int(g * 255), int(b * 255)], dtype=np.uint8)
                trajectory_colors.append(np.tile(color, (2, 1)))
        
        # Add trajectory lines to scene
        if trajectory_lines:
            try:
                trajectory_lines = np.array(trajectory_lines, dtype=np.float32)
                trajectory_colors = np.array(trajectory_colors, dtype=np.uint8)
                
                self.server.scene.add_line_segments(
                    name="edg/trajectories",
                    points=trajectory_lines,
                    colors=trajectory_colors,
                    line_width=0.2,
                )
            except Exception as e:
                pass  # Silently skip on error

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: CameraState, render_tab_state: RenderTabState
    ):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        # Get EDG frame for dynamic Gaussian warping
        frame_idx_tensor = None
        if self.edg_warp is not None and self._edg_frame_slider is not None:
            edg_frame = self._edg_frame_slider.value
            frame_idx_tensor = torch.tensor([edg_frame], device=self.device)
            # Debug print
            if hasattr(self, '_last_edg_frame') and self._last_edg_frame != edg_frame:
                print(f"[EDG] Viewer rendering frame {edg_frame}")
            self._last_edg_frame = edg_frame

        # Warp dynamic Gaussians to the selected EDG frame
        override_means = self._edg_override_means(frame_idx_tensor)

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        render_colors, render_alphas, info = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device)
            / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
            override_means=override_means,  # Use warped positions
        )  # [1, H, W, 3]
        render_tab_state.total_gs_count = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            # colors represented with sh are not guranteed to be in [0, 1]
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            # normalize depth to [0, 1]
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            if render_tab_state.inverse:
                alpha = 1 - alpha
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        return renders


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    # Import post-processing modules based on configuration
    # These imports must be here (not in __main__) for distributed workers
    if cfg.post_processing == "bilateral_grid":
        global BilateralGrid, slice, total_variation_loss
        if cfg.bilateral_grid_fused:
            from fused_bilagrid import (
                BilateralGrid,
                slice,
                total_variation_loss,
            )
        else:
            from lib_bilagrid import (
                BilateralGrid,
                slice,
                total_variation_loss,
            )
    elif cfg.post_processing == "ppisp":
        global PPISP, PPISPConfig, export_ppisp_report
        from ppisp import PPISP, PPISPConfig
        from ppisp.report import export_ppisp_report

    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        if runner.post_processing_module is not None:
            pp_state = ckpts[0].get("post_processing")
            if pp_state is not None:
                runner.post_processing_module.load_state_dict(pp_state)
        if runner.edg_warp is not None:
            edg_state = ckpts[0].get("edg_warp")
            if edg_state is not None:
                runner.edg_warp.load_state_dict(edg_state)
        # Restore the trained appearance module and pose refinement too (app_opt=True models need the
        # trained app module for exposure/colour). The ckpt saves both (see save block above).
        if getattr(runner, "app_module", None) is not None:
            app_state = ckpts[0].get("app_module")
            if app_state is not None:
                app_mod = runner.app_module.module if hasattr(runner.app_module, "module") else runner.app_module
                app_mod.load_state_dict(app_state)
        if getattr(runner, "pose_adjust", None) is not None:
            pose_state = ckpts[0].get("pose_adjust")
            if pose_state is not None:
                pose_mod = runner.pose_adjust.module if hasattr(runner.pose_adjust, "module") else runner.pose_adjust
                pose_mod.load_state_dict(pose_state)
        step = ckpts[0]["step"]
        runner.eval(step=step)
        runner.render_traj(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        runner.train()
        runner.export_ppisp_reports()

    if not cfg.disable_viewer:
        runner.close_viewer()
        _finish_simple_trainer_logger()


# ---- training config from XM4D's config.json ----
# These trainer hyperparameters are set in the same config.json (repo root) as the rest of the XM4D pipeline. A JSON value
# overrides the hardcoded Config default; an explicit CLI flag still overrides the JSON (tyro is applied after).
_DATASET_CONFIG_FIELDS = (
    "edg_tracks_lr", "edg_rots_lr", "edg_weights_lr", "edg_rel_lr",
    "edg_lr_decay_steps", "edg_lr_decay_gamma", "edg_arap_lambda", "edg_smooth_lambda",
    "edg_skinning_reg_lambda", "edg_rel_delta_max", "seg_dyn_lambda", "dyn_flat_color",
    "dyn_outside_lambda", "static_only_region_lambda", "means_lr", "pose_opt", "depth_lambda",
    "app_opt", "app_opt_reg", "edg_obs_lambda", "dyn_obs_lambda",
    "max_steps", "steps_scaler",
)


def _xm_config_path():
    env = os.environ.get("XM_CONFIG")
    if env and os.path.isfile(env):
        return Path(env)
    # install/gs.sh puts this file at <repo>/build/gsplat_examples/simple_trainer.py; config.json is at <repo>
    return Path(__file__).resolve().parents[2] / "config.json"


def _apply_dataset_config(cfg):
    """Override whitelisted Config fields from XM4D's config.json."""
    path = _xm_config_path()
    merged = json.loads(path.read_text()) if path.exists() else {}
    applied = {}
    for fld in _DATASET_CONFIG_FIELDS:
        if fld in merged:
            setattr(cfg, fld, merged[fld])
            applied[fld] = merged[fld]
    if applied:
        print(f"[gsplat-config] {path} overrides: {applied}")
    return cfg


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=9 python -m examples.simple_trainer default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                # prune_scale3d 0.3 (gsplat's DefaultStrategy default is 0.1)
                strategy=DefaultStrategy(verbose=True, refine_stop_iter=15_000, prune_scale3d=0.3),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    # Training hyperparameters from config.json become the tyro defaults
    # (explicit CLI flags still override them).
    for _cfg_name in configs:
        _apply_dataset_config(configs[_cfg_name][1])
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)
    _install_simple_trainer_logger(cfg.result_dir)

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    if cfg.with_ut:
        assert cfg.with_eval3d, "Training with UT requires setting `with_eval3d` flag."

    cli(main, cfg, verbose=True)
