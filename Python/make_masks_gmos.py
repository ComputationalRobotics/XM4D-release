"""Step 0 without a notebook (e.g. on a workstation without Jupyter): the moving-object masks found automatically by GMOS
(Xie et al., "GMOS: Grounding Moving Object Segmentation in 3D Space and Time", github.com/Jyxarthur/gmos).
GMOS finds every object that moves in the video and tracks it through all frames. Each one is saved as its own object,
masks/other_gmos<k>/ (category "other" for now), and merged as in make_masks_sam3.ipynb (check_masks.merge_masks).
Objects saved with make_masks_sam3.ipynb stay; the objects of an earlier GMOS run are replaced.
Run in the `gmos` environment (install/gmos.sh), for the scene in config.json.
usage: python make_masks_gmos.py [--fps 24] [--offload-cpu]
  --fps          frame rate of the frames in images/: GMOS looks at 5-frame windows max(1, fps // 8) frames apart
  --offload-cpu  keep GMOS's feature caches in CPU memory (long videos that run out of GPU memory)"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _numeric_key(p):   # XM's frame order (utils.track_tapnet._numeric_key)
    s = Path(p).stem
    try:
        return int(s)
    except ValueError:
        return s


def main():
    ap = argparse.ArgumentParser(description="moving-object masks with GMOS")
    ap.add_argument("--fps", type=float, default=24.0, help="frame rate of the frames (default 24)")
    ap.add_argument("--offload-cpu", action="store_true", help="GMOS feature caches in CPU memory (long videos)")
    args = ap.parse_args()
    from utils import definitions as D
    from check_masks import merge_masks, remove_object, save_object

    frames = sorted(glob.glob(str(D.video_dir / D.image_pattern)), key=_numeric_key)
    assert frames, f"no frames matching {D.image_pattern} in {D.video_dir}"
    work = D.output_dir / "gmos"
    shutil.rmtree(work, ignore_errors=True)
    (work / "frames").mkdir(parents=True)
    for i, f in enumerate(frames):   # GMOS reads the frames in name order: link them as <index>.<ext>
        (work / "frames" / f"{i:05d}{Path(f).suffix.lower()}").symlink_to(os.path.abspath(f))
    print(f"[gmos] {len(frames)} frames of {D.video_dir}, fps {args.fps:g} (window stride {max(1, int(args.fps / 8))})",
          flush=True)
    subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", str(work / "frames"), str(work / "objects"),
                    str(args.fps), str(int(args.offload_cpu))], cwd=D.REPO_ROOT / "deps" / "gmos", check=True)

    for d in sorted(D.mask_dir.glob("other_gmos*")):   # objects of an earlier GMOS run
        remove_object(d.name.split("_", 1)[1])
    objs = sorted((work / "objects").iterdir(), key=lambda d: min(d.iterdir()).name) if (work / "objects").is_dir() else []
    if not objs:
        sys.exit("[gmos] GMOS found no moving object. For a scene without any moving object, set \"no_masks\": true in "
                 "config.json.")
    W, H = Image.open(frames[0]).size
    for k, d in enumerate(objs, 1):   # numbered by first appearance
        save_object(f"gmos{k}", "other", frames,
                    (np.asarray(Image.open(p)) > 0 if p.exists() else np.zeros((H, W), bool)
                     for p in (d / f"{t:05d}.png" for t in range(len(frames)))))
    merge_masks()


def worker(frame_dir, out_dir, fps, offload_cpu):
    """GMOS itself, run in deps/gmos as a separate process: GMOS has its own `utils` module, which clashes with XM's.
    Writes <out_dir>/<object id>/<frame index>.png for every frame an object is on."""
    sys.path[0] = os.getcwd()   # deps/gmos instead of XM's Python/
    import torch
    from safetensors.torch import load_file

    import inference_gmos as G
    from utils import LoadedImageDirVideo, patch_predictor_cache_additive

    device = torch.device("cuda")   # the models are loaded as in inference_gmos.py
    pi3 = G.Pi3Encoder().to(device).eval()
    pi3.load_state_dict(load_file("checkpoints/pi3_model.safetensors"), strict=False)
    for p in pi3.parameters():
        p.requires_grad = False
    gmos = G.GMOSProposer().to(device)
    state = torch.load("checkpoints/gmos_proposer.pth")["model_state_dict"]
    gmos.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=False)
    predictor = G.build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_l.yaml", "checkpoints/sam2.1_hiera_large.pt",
                                             device=device)
    patch_predictor_cache_additive(predictor)

    # keep GMOS's per-object masks (the "mos" output: every object that moves, on all its frames) instead of its
    # indexed PNG, which adds the ids of overlapping objects
    found = {}
    G.save_results = lambda frames, on, off, *a, **kw: found.update(off.get("full_segments", {}))
    fps = float(fps)
    G.process_video(frame_dir, pi3, gmos, predictor, out_dir, batch_size=8, device=device, save_mode="",
                    stride=max(1, int(fps / 8)), offload_cpu=offload_cpu == "1",
                    preloaded_video=LoadedImageDirVideo(frame_dir, fps=fps), seq_name="scene", output_fps=fps)
    for t, objs in found.items():
        for oid, m in objs.items():
            m = np.asarray(m)[0] > 0
            if m.any():
                (Path(out_dir) / str(oid)).mkdir(parents=True, exist_ok=True)
                Image.fromarray(m.astype(np.uint8) * 255).save(Path(out_dir) / str(oid) / f"{t:05d}.png")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--worker"]:
        worker(*sys.argv[2:])
    else:
        main()
