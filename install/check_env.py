"""Check that a conda env can run its XM4D stage: the imports of the stage's scripts, the GPU, and the model weights.
usage (inside the env):  python install/check_env.py {sam3,da3,tapnet,XM}
The install scripts run it at the end; run it again any time. Exits 1 if something is missing."""
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# what the stage's scripts import (they add deps/tapnet, deps/Depth-Anything-3/src and XM/build to sys.path themselves)
MODULES = {
    "sam3": ["numpy", "PIL", "cv2", "matplotlib", "torch", "ipykernel", "ipympl", "sam3.model_builder"],
    "da3": ["numpy", "scipy", "PIL", "cv2", "pandas", "open3d", "torch", "depth_anything_3.api"],
    "tapnet": ["numpy", "PIL", "cv2", "pandas", "tqdm", "torch", "einops", "tapnet.tapnext.tapnext_torch"],
    "XM": ["numpy", "scipy", "cv2", "pandas", "matplotlib", "networkx", "open3d", "pyceres", "pycolmap",
           "teaserpp_python", "XM"],
}
PATHS = {
    "da3": [ROOT / "deps" / "Depth-Anything-3" / "src"],
    "tapnet": [ROOT / "deps" / "tapnet"],
    "XM": [ROOT / "XM" / "build"],
}
TAPNEXT_CKPT = ROOT / "deps" / "tapnet" / "checkpoints" / "tapnextpp_ckpt.pt"
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


def main():
    env = sys.argv[1] if len(sys.argv) > 1 else ""
    if env not in MODULES:
        sys.exit(f"usage: python {sys.argv[0]} {{{','.join(MODULES)}}}")
    sys.path[:0] = [str(p) for p in PATHS.get(env, [])]
    print(f"[check_env] {env}  (python {sys.version.split()[0]}, {sys.executable})")
    for mod in MODULES[env]:
        check(f"import {mod}", lambda mod=mod: importlib.import_module(mod) and None)
    if env != "XM":
        check("CUDA GPU", gpu)
    if env == "sam3":
        check("SAM 3 weights", sam3_weights)
    if env == "tapnet":
        check("TAPNext++", tapnext_forward)
    if failed:
        print(f"[check_env] {env}: NOT ready, {len(failed)} problem(s): {', '.join(failed)}")
        sys.exit(1)
    print(f"[check_env] {env}: ready")


if __name__ == "__main__":
    main()
