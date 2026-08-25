# Sintel XM Experiment Results

Generated: `2026-04-21T16:29:19-04:00`

Pipeline:

- Dataset root: `datasets/Sintel`
- Missing GT cameras are loaded from `datasets/MPI-Sintel-depth-training-20150305/training/camdata_left`
- Missing DA3 depth is generated with `Python/get_depths_clean.py` in the `da3` environment
- Missing `output_tapnet_da3.npz` is generated with `Python/static_tracking_online.py` in the `tapnet` environment
- Solver sweep uses the same 6 heuristics as the BAL experiment, with no added noise

## Preparation

| Scene | Frames | Camera source | DA3 depth | Tracking | Depth log | Tracking log |
| --- | ---: | --- | --- | --- | --- | --- |
| market_6 | 40 | mpi_fallback | reused | reused | - | - |

## Solver Results

| Scene | Setting | Mode | Sphere Ratio | WEIGHT_MODE | RPE-R | RPE-T | ATE-R | ATE-T | Total time (s) | Solve time (s) | Optimal rank | Log |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| market_6 | 1 | 1 | 0.1 | 1 | 0.0153179725775960 | 0.0197897865021878 | 2.1033956634084090 | 0.0146409412003699 | 4.479437 | 0.330050 | 3 | `Python/logs/sintel_xm_market6_test/solver/market_6_setting1_mode1_w1_sr0.1.log` |
| market_6 | 2 | 1 | 0.3 | 1 | 0.0171845255366349 | 0.0210767994860478 | 2.1906636658362864 | 0.0158993273643915 | 5.590612 | 0.364801 | 3 | `Python/logs/sintel_xm_market6_test/solver/market_6_setting2_mode1_w1_sr0.3.log` |
| market_6 | 3 | 1 | 0.5 | 1 | 0.0177824812682655 | 0.0206665313687428 | 2.2437293746013220 | 0.0155564044968997 | 5.356120 | 0.382917 | 3 | `Python/logs/sintel_xm_market6_test/solver/market_6_setting3_mode1_w1_sr0.5.log` |
| market_6 | 4 | 1 | 0.9 | 2 | 0.0328691944725408 | 0.0394072560543809 | 2.3346040359545457 | 0.0274059076724267 | 5.249836 | 0.507079 | 3 | `Python/logs/sintel_xm_market6_test/solver/market_6_setting4_mode1_w2_sr0.9.log` |
| market_6 | 5 | 1 | 0.9 | 3 | 0.0340230933072658 | 0.0280112870679107 | 2.5027203278677144 | 0.0197086320451160 | 4.653336 | 0.473493 | 3 | `Python/logs/sintel_xm_market6_test/solver/market_6_setting5_mode1_w3_sr0.9.log` |
| market_6 | 6 | 2 | - | 1 | 0.0181292816061293 | 0.0266234467623965 | 2.1384318019836877 | 0.0188008847029780 | 5.807014 | 0.255742 | 3 | `Python/logs/sintel_xm_market6_test/solver/market_6_setting6_mode2_w1.log` |
