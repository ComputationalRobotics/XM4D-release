# XM4D — Reproduction & Certification (2026-10-07)

Every number below was produced **from this repository** with the runners in `bashes/`, on 2× RTX 6000 Ada
(49 GB), driver CUDA 12.8. Conda envs per stage: `da3` (depth), `tapnet` (2D tracking, EDG completion),
`XM` (merge/solver/EDG/prep), `gsplat` (training, eval), `SpaTrack2` (CoWTracker skeleton; DyCheck only).

## Solver certification
All EDG segment solves are certified by the XM dual certificate (checkeig min-eigenvalue / relative gap),
printed per solve by `EDG_test_fast.py`. This campaign: **1916 / 1916 solves certified (100%)**
(DyCheck 678, Syn4D 800, NVIDIA 438).

## Seeded inputs (the reproducibility contract)
Learned-model inference is nondeterministic (TAPNext static tracks, CoWTracker skeleton, DA3 depth).
The solver and everything downstream are deterministic given these inputs. Each scene therefore ships:
`output_tapnet_da3.npz` (static tracks), `result_XM.npz` (camera solve), `tracks_dynamic_only.npz`
(dynamic tracks), and for DyCheck `cow_skeleton.npz`. Running the runners on these seeds reproduces the
tables below to within TTO-eval noise (±0.9 dB per scene, unseeded). Regenerating the seeds from raw video
also works but solver-sensitive scenes can drift dB-scale — demonstrated twice in this campaign:
DyCheck/wheel 15.0→17.8 and NVIDIA/Truck 27.4→(record) purely by restoring the shipped camera solve.

## Commands
```bash
bash bashes/run_dycheck_fast.sh  <scene> [gpu]   # apple block paper-windmill space-out spin teddy wheel
bash bashes/run_syn4d.sh         <scene> [gpu]   # bear1-4 casa1-4
bash bashes/run_nvidia_seeded.sh <scene> [gpu]   # Balloon1 Balloon2 Jumping Playground Skating Truck Umbrella
```
Each prints per-stage `STAGE <scene> <name> <seconds>` lines and a final `RELDONE` metrics line.
All parameters live in `Python/configs/<dataset>.json`; the DyCheck json carries the fast 2026-10 recipe
(CoWTracker merge, lift-only 3D, rotation sync, ARAP pruning, node-2D weight), while `nvidia.json` and
`syn4d_*.json` pin the legacy recipe their recorded numbers certify (kabsch rotations, no ARAP pruning,
bad-node 0.3, dynamic off-mask filter off).

## DyCheck — novel-view alpha mPSNR (TTO, covisible mask, 25k steps)
| scene | reference (main repo) | this repo | Δ |
|---|---|---|---|
| apple | 19.17 | 19.28 | +0.11 |
| block | 18.59 | 18.81 | +0.22 |
| paper-windmill | 21.31 | 21.34 | +0.03 |
| space-out | 21.82 | 21.79 | −0.03 |
| spin | 20.98 | 20.86 | −0.12 |
| teddy | 16.37 | 16.41 | +0.04 |
| wheel | 17.80 | 17.76 | −0.04 |
| **mean** | **19.43** | **19.46** | **+0.03** |

## Syn4D — vs the paper table (8 scenes)
| metric | paper "Ours" | this repo | Δ |
|---|---|---|---|
| novel PSNR avg | 19.33 | **21.33** | +2.00 (every scene above) |
| train PSNR avg | 35.52 | **36.50** | +0.98 |
| camera ATE-T | bears 0.001, casa3/4 0.004 | bears 0.0005–0.0008, casa3 0.0058, casa4 0.0043 | ≤ paper except casa3 (+0.002) |

Per-scene novel (bear1→casa4): 20.80, 18.31, 18.72, 18.20, 22.43, 22.86, 24.78, 24.55.
The novel-PSNR gain over the paper reflects eval/trainer fixes that postdate it (TTO per-camera alignment,
per-frame depth-scale in the dyn lift) — the paper numbers were correct for their code vintage.

## NVIDIA — fixed-v000 novel view, split0 appearance (vs the 26.90 record)
| scene | record | this repo | Δ |
|---|---|---|---|
| Balloon1 | 23.53 | 23.58 | +0.05 |
| Balloon2 | 27.80 | 27.81 | +0.01 |
| Jumping | 25.72 | 25.40 | −0.32 |
| Playground | 26.36 | 26.62 | +0.26 |
| Skating | 32.80 | 34.39 | +1.59 (documented σ≈0.78 scene) |
| Truck | 29.38 | 29.36 | −0.02 |
| Umbrella | 22.68 | 22.68 | ±0.00 |

**mean 27.12 vs record 26.90 (+0.22).** Truck initially measured 27.44 on a camera solve left by a
later from-scratch pass; restoring the shipped record solve recovered 29.36 — together with
DyCheck/wheel (15.0 → 17.76 by the same fix) these are the two case studies behind the seeded-inputs
contract above.

## Per-scene runtime (stage medians, seeded inputs, one GPU)
| dataset | pre-train (merge→dataset) | train | eval | total |
|---|---|---|---|---|
| DyCheck | ~15 min (attach dominates; skeleton reused) | 20.5 min | 7.5 min | **~43 min** |
| Syn4D | ~3 min | 30 min | 2.3 min | **~35 min** |
| NVIDIA | ~6 min | 21 min | 0.6 min | **~28 min** |

## Noise bands for judging a reproduction
TTO eval is unseeded: ±0.9 dB per scene (~±0.34 on a 7-scene mean). gsplat densification is
atomic-add-nondeterministic (NVIDIA/Skating σ≈0.78). Camera accuracy and the EDG solve are deterministic
given the seeds; `optimize_dynamic_tracks` outputs are bitwise reproducible and stamped with their depth
variant (`depth_variant` key; loaders warn on mismatch — heed it, a wrong `XM_DEPTH_VARIANT` silently
changes the whole EDG).
