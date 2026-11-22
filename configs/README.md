# Tuned configs: DyCheck, NVIDIA and Syn4D

`config.json` at the repo root is the config for any video. This folder holds the configs we tuned on three benchmark
datasets:

| Config | Dataset | Scenes |
|---|---|---|
| `dycheck.json` | DyCheck (iPhone) | apple, block, paper-windmill, space-out, spin, teddy, wheel |
| `nvidia.json` | NVIDIA Dynamic Scenes | Balloon1, Balloon2, Jumping, Playground, Skating, Truck, Umbrella |
| `syn4d_bear.json` | Syn4D | bear1–4 |
| `syn4d_casa.json` | Syn4D | casa1–4 |

To use one, copy it over `config.json` and set `"scene"`. The scene folder keeps the dataset's scene name, e.g.
`datasets/apple`, so that the config finds that scene's intrinsics:

```bash
cp configs/dycheck.json config.json      # then set "scene": "apple"
```

Then run the steps as in the [README](../README.md). The preprocessed benchmark inputs and the runners that reproduce
the paper's tables are on the
[`reproduce` branch](https://github.com/ComputationalRobotics/XM4D-release/tree/reproduce).

What they change from `config.json` (the options are described in [docs/pipeline.md](../docs/pipeline.md)):

- All four set each scene's intrinsics from the dataset (`INTRINSICS`), with `"CERES_OPTIMIZE_INTRINSICS": false` so
  that the focal is not refined.
- DyCheck uses its LiDAR depth (`"LIDAR_DEPTH": true`): put it in `<scene>/depth/*.png`, one 16-bit PNG per frame in
  mm, 0 where there is none. Its 4D Gaussians train for 25000 steps (`max_steps` × `steps_scaler`; 30000 by default)
  with appearance optimisation (`app_opt`).
- NVIDIA trains the 4D Gaussians at half resolution (`"GS_DATA_FACTOR": 2`) with more of them on the moving objects
  (`"GS_DYN_MIN_POINTS": 100000`), with appearance optimisation and its own loss weights.
- Syn4D repairs depth spikes in the moving tracks (`"SMOOTH_DEPTH_REPAIR": 0.1`), holds a graph node's unsolved first
  and last frames rigid to its neighbours (`"EDG_RIGID_ENDS": true`), starts from fewer 4D Gaussians (200000, 20000 on
  the moving objects), and trains them for 12000 steps with lower learning rates. Camera refinement during training
  (`pose_opt`) is off for bear and on for casa.
