#!/usr/bin/env python3
"""NVIDIA Dynamic Scenes novel-view evaluation (headline protocol, one command).

Wraps estimate_ours_render_tto.py with the NVIDIA fixed-v000 protocol defaults:
  - GT: <scene>/novel_orig (Yoon multiview_GT cam0, 480x270), full-frame metric, no color calib
  - pose: MoSca-parity per-view TTO from the Sim3 init (--novel-align tto)
  - appearance: FIRST-FRAME SPLIT eval (--novel-app split0) -- static Gaussians rendered with
    the frame-0 embedding (the novel view is always physical camera 0, and NVIDIA training
    frame t is physical camera t, so embed-t encodes camera t's color response, not the
    scene's), dynamic Gaussians keep the frame-t embedding (their appearance at time t was
    only ever observed by camera t).

Scene comes from XM_DATASET_CAT/XM_DATASET_NAME (utils.definitions) unless --scene-dir is
given; the checkpoint defaults to the newest outputs/gsplat*/ckpts/final_rank0.pt.
Any explicit flag overrides the corresponding default.

Usage:
  XM_DATASET_CAT=NVIDIA XM_DATASET_NAME=Skating python estimate_nvidia.py
  python estimate_nvidia.py --scene-dir <path> --gaussians <ckpt> [extra tto flags]
"""
import sys
from pathlib import Path


def main() -> None:
    argv = list(sys.argv[1:])

    def has(flag: str) -> bool:
        return any(a == flag or a.startswith(flag + "=") for a in argv)

    if not has("--scene-dir") and not has("--scene-path"):
        from utils import definitions
        argv += ["--scene-dir", str(definitions.rootpath)]
    scene_dir = Path(argv[argv.index("--scene-dir") + 1] if has("--scene-dir")
                     else argv[argv.index("--scene-path") + 1]).resolve()

    if not has("--gaussians"):
        canonical = scene_dir / "outputs/gsplat_fx_dyndense/ckpts/final_rank0.pt"
        cands = ([canonical] if canonical.is_file() else
                 sorted(scene_dir.glob("outputs/gsplat*/ckpts/final_rank0.pt"),
                        key=lambda p: p.stat().st_mtime, reverse=True))
        if not cands:
            raise FileNotFoundError(f"No outputs/gsplat*/ckpts/final_rank0.pt under {scene_dir}; "
                                    f"pass --gaussians explicitly.")
        argv += ["--gaussians", str(cands[0])]
        print(f"[nvidia-eval] default gaussians: {cands[0]}")

    defaults = [
        ("--camera-source", "colmap"),
        ("--metric-mask", "full"),
        ("--color-calib", "none"),
        ("--novel-align", "tto"),
        ("--novel-app", "split0"),
        ("--novel-dir", str(scene_dir / "novel_orig")),
        ("--output-dir", str(scene_dir / "outputs" / "nvidia_eval")),
        ("--output-json", str(scene_dir / "outputs" / "nvidia_eval" / "summary.json")),
    ]
    for flag, val in defaults:
        if not has(flag):
            argv += [flag, val]
    if not has("--use-covisible-mask") and not has("--no-use-covisible-mask"):
        argv += ["--no-use-covisible-mask"]

    import estimate_ours_render_tto
    sys.argv = [sys.argv[0]] + argv
    estimate_ours_render_tto.main()


if __name__ == "__main__":
    main()
