"""Stage 2: run UniDepth on the GLOMAP-registered images of an IMC scene.

Runs in the `unidepth` conda env. Uses the intrinsics GLOMAP estimated (recorded
by stage1 to outputs/glomap_work/glomap_intrinsics.json) so UniDepth-V2 predicts
metric depth in the same camera the solver observations are built with. No
ground-truth intrinsics are used. Depth maps are resized back to the original
image resolution and saved as <scene>/outputs/unidepth_depth/<image_stem>.npy so
that pixel coordinates from GLOMAP index them directly.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imc_common import INTRINSIC_SOURCES, load_glomap_intrinsics, load_intrinsics

from unidepth.models import UniDepthV2
from unidepth.utils.camera import Pinhole


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir")
    ap.add_argument("--intrinsics", choices=INTRINSIC_SOURCES, default="glomap",
                    help="Camera intrinsics fed to UniDepth. 'unidepth' lets UniDepth estimate "
                         "them and records outputs/unidepth_intrinsics.json.")
    ap.add_argument("--model", default="lpiccinelli/unidepth-v2-vitl14")
    args = ap.parse_args()

    scene = Path(args.scene_dir)
    out_dir = scene / "outputs" / "unidepth_depth"
    conf_dir = scene / "outputs" / "unidepth_conf"
    out_dir.mkdir(parents=True, exist_ok=True)
    conf_dir.mkdir(parents=True, exist_ok=True)
    source = args.intrinsics

    # Frame set = images GLOMAP registered (exactly what stage3 lifts).
    names = sorted(load_glomap_intrinsics(scene).keys())
    K_src = load_intrinsics(scene, source) if source in ("gt", "glomap") else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("[unidepth] WARNING: CUDA not available, running on CPU (slow).")
    model = UniDepthV2.from_pretrained(args.model)
    model.interpolation_mode = "bilinear"
    model = model.to(device).eval()

    est_intrinsics = {}
    done = 0
    for name in names:
        if K_src is not None and name not in K_src:
            print(f"[unidepth] {name}: no {source} intrinsics, skipped.")
            continue
        img_path = scene / "images" / name
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[unidepth] {name}: unreadable, skipped.")
            continue
        H, W = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1)
        with torch.no_grad():
            if source == "unidepth":
                pred = model.infer(rgb_t)  # let UniDepth estimate the camera
            else:
                K_t = torch.from_numpy(K_src[name]["K"].astype(np.float32))
                pred = model.infer(rgb_t, Pinhole(K=K_t.unsqueeze(0)))
        depth = pred["depth"].squeeze().detach().cpu().numpy().astype(np.float32)
        if depth.shape != (H, W):
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
        np.save(out_dir / f"{Path(name).stem}.npy", depth)
        # Per-pixel confidence (used as observation weights in stage3).
        conf_t = pred.get("confidence")
        if conf_t is not None:
            conf = np.asarray(conf_t.squeeze().detach().cpu().numpy(), dtype=np.float32)
            if conf.shape != (H, W):
                conf = cv2.resize(conf, (W, H), interpolation=cv2.INTER_LINEAR)
            np.save(conf_dir / f"{Path(name).stem}.npy", conf)
        if source == "unidepth":
            K_pred = np.asarray(pred["intrinsics"].squeeze().detach().cpu().numpy(), dtype=np.float64)
            est_intrinsics[name] = {"K": K_pred.tolist(), "width": W, "height": H}
        done += 1
        if done % 25 == 0:
            print(f"[unidepth] {done}/{len(names)} done", flush=True)

    if source == "unidepth":
        import json
        ipath = scene / "outputs" / "unidepth_intrinsics.json"
        with open(ipath, "w") as f:
            json.dump(est_intrinsics, f, indent=2)
        fx = [v["K"][0][0] for v in est_intrinsics.values()]
        print(f"[unidepth] recorded estimated intrinsics for {len(est_intrinsics)} images -> {ipath} "
              f"(focal min={min(fx):.1f} median={float(np.median(fx)):.1f} max={max(fx):.1f})")

    print(f"[unidepth] finished ({source} intrinsics): {done} depth maps in {out_dir}")


if __name__ == "__main__":
    main()
