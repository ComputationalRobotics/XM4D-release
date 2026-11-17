"""Check the conda environments of XM4D after installing them: for every stage, in its own env, the imports of the
stage's scripts, the GPU and the model weights. Run it with any Python 3 that has conda on the PATH (e.g. conda's base):
    python install/check_env.py                 # every stage in its env: sam3 (or gmos), da3, tapnet, XM, gs
    python install/check_env.py --env xm4d      # the one-environment install (install/xm4d.sh)
    python install/check_env.py tapnet=mytap    # a stage installed under another env name (bash install/tapnet.sh mytap)
Step 0 is checked with sam3, with gmos when sam3 is not installed (both when both are), and gives a warning when neither
is installed. Exits 1 if a stage fails or is missing."""
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# what the stage's scripts import (they add deps/tapnet, deps/Depth-Anything-3/src and XM/build to sys.path themselves)
MODULES = {
    "sam3": ["numpy", "PIL", "cv2", "matplotlib", "torch", "ipykernel", "ipympl", "sam3.model_builder"],
    "gmos": ["numpy", "PIL", "torch", "safetensors", "hydra", "mediapy", "inference_gmos"],
    "da3": ["numpy", "scipy", "PIL", "cv2", "pandas", "open3d", "torch", "depth_anything_3.api"],
    "tapnet": ["numpy", "PIL", "cv2", "pandas", "tqdm", "torch", "einops", "tapnet.tapnext.tapnext_torch"],
    "XM": ["numpy", "scipy", "cv2", "pandas", "matplotlib", "networkx", "open3d", "pyceres", "pycolmap",
           "teaserpp_python", "XM"],
    "gs": ["numpy", "scipy", "cv2", "PIL", "imageio", "yaml", "tqdm", "matplotlib", "sklearn", "piexif", "tyro", "viser",
           "nerfview", "torchmetrics", "torch", "fused_ssim", "gsplat"],
}
PATHS = {
    "gmos": [ROOT / "deps" / "gmos"],
    "da3": [ROOT / "deps" / "Depth-Anything-3" / "src"],
    "tapnet": [ROOT / "deps" / "tapnet"],
    "XM": [ROOT / "XM" / "build"],
}
TAPNEXT_CKPT = ROOT / "deps" / "tapnet" / "checkpoints" / "tapnextpp_ckpt.pt"
ALLTRACKER_CKPT = ROOT / "deps" / "alltracker" / "checkpoints" / "alltracker.pth"
GS_TRAINER = ROOT / "build" / "gsplat_examples" / "simple_trainer.py"
failed = []


def check(name, fn):
    try:
        note = fn()
        print(f"  ok    {name}" + (f"  ({note})" if note else ""))
    except Exception as exc:  # report every problem, not only the first
        print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
        failed.append(name)


def gpu():
    import torch
    assert torch.cuda.is_available(), "torch sees no CUDA GPU"
    assert (torch.ones(1, device="cuda") + 1).item() == 2
    return torch.cuda.get_device_name(0)


def sam3_weights():
    from huggingface_hub import hf_hub_download, try_to_load_from_cache
    try:
        hf_hub_download("facebook/sam3", "config.json")  # gated like the checkpoint
    except Exception as exc:
        raise RuntimeError("no access to the gated facebook/sam3 weights: request access at "
                           "https://huggingface.co/facebook/sam3, then run `hf auth login`") from exc
    cached = isinstance(try_to_load_from_cache("facebook/sam3", "sam3.pt"), str)
    return "access ok, sam3.pt cached" if cached else "access ok, sam3.pt (3.5 GB) downloads on first use"


def gmos_weights():
    import torch
    from safetensors import safe_open
    ck = ROOT / "deps" / "gmos" / "checkpoints"
    with safe_open(str(ck / "pi3_model.safetensors"), "pt") as f:
        assert len(list(f.keys())) > 0
    torch.load(ck / "sam2.1_hiera_large.pt", map_location="cpu", weights_only=False)["model"]
    torch.load(ck / "gmos_proposer.pth", map_location="cpu", weights_only=False)["model_state_dict"]
    return "Pi3, SAM 2.1 and GMOS proposer checkpoints load"


def tapnext_forward():
    import torch
    from tapnet.tapnext.tapnext_torch import TAPNext
    assert TAPNEXT_CKPT.exists(), f"missing {TAPNEXT_CKPT}"
    model = TAPNext(image_size=(256, 256))  # loaded as in static_tracking_online.py
    ckpt = torch.load(TAPNEXT_CKPT, map_location="cpu")
    model.load_state_dict({k.replace("tapnext.", ""): v for k, v in ckpt["state_dict"].items()})
    model.cuda().eval()
    video = torch.rand(1, 1, 256, 256, 3, device="cuda") * 2 - 1
    queries = torch.tensor([[[0.0, 64.0, 64.0], [0.0, 128.0, 192.0]]], device="cuda")  # (t, y, x)
    with torch.no_grad():
        tracks = model(video=video, query_points=queries)[0]
    assert torch.isfinite(tracks).all()
    return "TAPNext++ checkpoint loads and runs"


def alltracker_forward():
    import numpy as np
    sys.path.insert(0, str(ROOT / "Python"))
    from utils.track_alltracker import track_keyframes  # loaded as in the tracking steps
    assert ALLTRACKER_CKPT.exists(), f"missing {ALLTRACKER_CKPT}"
    frames = (np.random.rand(4, 96, 128, 3) * 255).astype(np.uint8)
    tracks, vis, _ = track_keyframes(frames, lambda t, cov: np.array([[32.0, 48.0], [96.0, 40.0]]) if t == 0 else [],
                                     keyframe=4, res=128, device="cuda", label="check")
    assert tracks.shape == (2, 4, 2) and np.isfinite(tracks).all()
    return "AllTracker checkpoint loads and runs"


def gsplat_render():
    import torch
    from gsplat import rasterization
    assert GS_TRAINER.exists(), f"missing {GS_TRAINER} (made by install/gs.sh)"
    n, dev = 64, "cuda"
    means = torch.rand(n, 3, device=dev) - 0.5 + torch.tensor([0.0, 0.0, 3.0], device=dev)
    quats = torch.nn.functional.normalize(torch.rand(n, 4, device=dev), dim=-1)
    img, alpha, _ = rasterization(means, quats, torch.full((n, 3), 0.05, device=dev), torch.full((n,), 0.8, device=dev),
                                  torch.rand(n, 3, device=dev), torch.eye(4, device=dev)[None],
                                  torch.tensor([[[64.0, 0, 32], [0, 64.0, 32], [0, 0, 1]]], device=dev), 64, 64)
    assert torch.isfinite(img).all() and alpha.max() > 0
    return "trainer in build/gsplat_examples, gsplat CUDA rasterizer runs"


def check_stage(stage):   # runs inside the stage's env
    sys.path[:0] = [str(p) for p in PATHS.get(stage, [])]
    print(f"[check_env] {stage}  (python {sys.version.split()[0]}, {sys.executable})")
    for mod in MODULES[stage]:
        check(f"import {mod}", lambda mod=mod: importlib.import_module(mod) and None)
    if stage != "XM":
        check("CUDA GPU", gpu)
    if stage == "sam3":
        check("SAM 3 weights", sam3_weights)
    if stage == "gmos":
        check("GMOS weights", gmos_weights)
    if stage == "tapnet":
        check("AllTracker", alltracker_forward)
        check("TAPNext++", tapnext_forward)
    if stage == "gs":
        check("gsplat", gsplat_render)
    if failed:
        print(f"[check_env] {stage}: NOT ready, {len(failed)} problem(s): {', '.join(failed)}")
        sys.exit(1)
    print(f"[check_env] {stage}: ready")


def main():
    envs = {stage: stage for stage in MODULES}   # stage -> conda env name
    one_env = None
    args = sys.argv[1:]
    while args:
        a = args.pop(0)
        if a == "--env" and args:
            one_env = args.pop(0)
            envs.update({stage: one_env for stage in ("sam3", "da3", "tapnet", "XM")})   # GMOS and gs are not in the one env
        elif a.split("=")[0] in MODULES and "=" in a:
            envs[a.split("=")[0]] = a.split("=", 1)[1]
        else:
            sys.exit(__doc__)
    conda = os.environ.get("CONDA_EXE", "conda")
    listed = json.loads(subprocess.run([conda, "env", "list", "--json"], capture_output=True, text=True,
                                       check=True).stdout)["envs"]
    installed = {Path(p).name for p in listed}
    masks = [stage for stage in ("sam3", "gmos") if envs[stage] in installed]
    summary = []
    for stage in masks + ["da3", "tapnet", "XM", "gs"]:
        if envs[stage] not in installed:
            script = "xm4d" if envs[stage] == one_env else stage
            summary.append(("FAIL", stage, f"env {envs[stage]} is not installed: bash install/{script}.sh"))
            continue
        r = subprocess.run([conda, "run", "--no-capture-output", "-n", envs[stage], "python", str(Path(__file__).resolve()),
                            "--stage", stage])
        summary.append(("ok" if r.returncode == 0 else "FAIL", stage, f"env {envs[stage]}"))
    print("\n[check_env] summary")
    for status, stage, note in summary:
        print(f"  {status:<5} {stage:<7} {note}")
    if not masks:
        print(f"  WARNING step 0: neither sam3 (env {envs['sam3']}) nor gmos (env {envs['gmos']}) is installed, so no "
              "masks can be made.\n          Install one: bash install/sam3.sh, or bash install/gmos.sh where no "
              "notebook runs.\n          Scenes that already have their masks, or set \"no_masks\", do not need it.")
    if any(status == "FAIL" for status, _, _ in summary):
        sys.exit(1)
    print("[check_env] all stages ready" + ("" if masks else ", except step 0 (see the warning)"))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--stage"]:
        check_stage(sys.argv[2])
    else:
        main()
