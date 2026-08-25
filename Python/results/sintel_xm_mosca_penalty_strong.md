# Sintel MoSca-Style Eval Results (strong penalty schedule)

Generated: `2026-04-22T18:25:50-04:00`

Source:

- Metrics are extracted from the `MoSca-style eval (evo)` block in the logs referenced by `Python/sintel_xm_results_penalty_strong.md`.
- Columns correspond to `ATE`, `RPE trans`, and `RPE rot` exactly as printed there.
- Missing MoSca metrics are rendered as `-` and listed below.

Missing metrics:

- `bamboo_2` setting `1`: evo evaluation failed: Degenerate covariance rank, Umeyama alignment is not possible
- `bamboo_2` setting `2`: evo evaluation failed: Degenerate covariance rank, Umeyama alignment is not possible
- `bamboo_2` setting `3`: evo evaluation failed: Degenerate covariance rank, Umeyama alignment is not possible
- `bamboo_2` setting `4`: evo evaluation failed: Degenerate covariance rank, Umeyama alignment is not possible
- `bamboo_2` setting `5`: evo evaluation failed: Degenerate covariance rank, Umeyama alignment is not possible
- `bamboo_2` setting `6`: evo evaluation failed: Degenerate covariance rank, Umeyama alignment is not possible
- `bandage_1` setting `1`: evo evaluation failed: Degenerate covariance rank, Umeyama alignment is not possible
- `bandage_1` setting `2`: evo evaluation failed: Degenerate covariance rank, Umeyama alignment is not possible
- `bandage_1` setting `3`: evo evaluation failed: Degenerate covariance rank, Umeyama alignment is not possible
- `bandage_1` setting `4`: evo evaluation failed: Degenerate covariance rank, Umeyama alignment is not possible
- `bandage_1` setting `5`: evo evaluation failed: Degenerate covariance rank, Umeyama alignment is not possible
- `bandage_1` setting `6`: evo evaluation failed: Degenerate covariance rank, Umeyama alignment is not possible
- `bandage_2` setting `1`: evo evaluation failed: Degenerate covariance rank, Umeyama alignment is not possible
- `bandage_2` setting `2`: evo evaluation failed: Degenerate covariance rank, Umeyama alignment is not possible
- `bandage_2` setting `3`: evo evaluation failed: Degenerate covariance rank, Umeyama alignment is not possible
- `bandage_2` setting `4`: evo evaluation failed: Degenerate covariance rank, Umeyama alignment is not possible
- `bandage_2` setting `5`: evo evaluation failed: Degenerate covariance rank, Umeyama alignment is not possible
- `bandage_2` setting `6`: evo evaluation failed: Degenerate covariance rank, Umeyama alignment is not possible
- `shaman_2` setting `1`: evo evaluation failed: Degenerate covariance rank, Umeyama alignment is not possible
- `shaman_2` setting `2`: evo evaluation failed: Degenerate covariance rank, Umeyama alignment is not possible
- `shaman_2` setting `3`: evo evaluation failed: Degenerate covariance rank, Umeyama alignment is not possible
- `shaman_2` setting `4`: evo evaluation failed: Degenerate covariance rank, Umeyama alignment is not possible
- `shaman_2` setting `5`: evo evaluation failed: Degenerate covariance rank, Umeyama alignment is not possible
- `shaman_2` setting `6`: evo evaluation failed: Degenerate covariance rank, Umeyama alignment is not possible

| Scene | Setting | Mode | Sphere Ratio | WEIGHT_MODE | ATE (m) | RPE trans (m) | RPE rot (deg) | Optimal rank | Log |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| alley_1 | 1 | 1 | 0.1 | 1 | 0.002835 | 0.006146 | 0.087390 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/alley_1_setting1_mode1_w1_sr0.1_ceres.log` |
| alley_1 | 2 | 1 | 0.3 | 1 | 0.002491 | 0.006138 | 0.088775 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/alley_1_setting2_mode1_w1_sr0.3_ceres.log` |
| alley_1 | 3 | 1 | 0.5 | 1 | 0.002486 | 0.006122 | 0.087538 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/alley_1_setting3_mode1_w1_sr0.5_ceres.log` |
| alley_1 | 4 | 1 | 0.9 | 2 | 0.006353 | 0.001719 | 0.023095 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/alley_1_setting4_mode1_w2_sr0.9_ceres.log` |
| alley_1 | 5 | 1 | 0.9 | 3 | 0.003534 | 0.006127 | 0.119628 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/alley_1_setting5_mode1_w3_sr0.9_ceres.log` |
| alley_1 | 6 | 2 | - | 1 | 0.027953 | 0.016026 | 0.513918 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/alley_1_setting6_mode2_w1_ceres.log` |
| alley_2 | 1 | 1 | 0.1 | 1 | 0.010044 | 0.008093 | 0.387365 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/alley_2_setting1_mode1_w1_sr0.1_ceres.log` |
| alley_2 | 2 | 1 | 0.3 | 1 | 0.028501 | 0.006932 | 0.388652 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/alley_2_setting2_mode1_w1_sr0.3_ceres.log` |
| alley_2 | 3 | 1 | 0.5 | 1 | 0.030291 | 0.006951 | 0.392278 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/alley_2_setting3_mode1_w1_sr0.5_ceres.log` |
| alley_2 | 4 | 1 | 0.9 | 2 | 0.017320 | 0.009284 | 0.399818 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/alley_2_setting4_mode1_w2_sr0.9_ceres.log` |
| alley_2 | 5 | 1 | 0.9 | 3 | 0.025810 | 0.008666 | 0.383643 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/alley_2_setting5_mode1_w3_sr0.9_ceres.log` |
| alley_2 | 6 | 2 | - | 1 | 0.142996 | 0.205028 | 0.410921 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/alley_2_setting6_mode2_w1_ceres.log` |
| ambush_2 | 1 | 1 | 0.1 | 1 | 0.000002 | 0.000004 | 3.476674 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_2_setting1_mode1_w1_sr0.1_ceres.log` |
| ambush_2 | 2 | 1 | 0.3 | 1 | 0.000002 | 0.000004 | 5.116940 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_2_setting2_mode1_w1_sr0.3_ceres.log` |
| ambush_2 | 3 | 1 | 0.5 | 1 | 0.000002 | 0.000004 | 4.684046 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_2_setting3_mode1_w1_sr0.5_ceres.log` |
| ambush_2 | 4 | 1 | 0.9 | 2 | 0.000002 | 0.000004 | 1.004381 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_2_setting4_mode1_w2_sr0.9_ceres.log` |
| ambush_2 | 5 | 1 | 0.9 | 3 | 0.000002 | 0.000004 | 1.103765 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_2_setting5_mode1_w3_sr0.9_ceres.log` |
| ambush_2 | 6 | 2 | - | 1 | 0.000002 | 0.000004 | 4.632412 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_2_setting6_mode2_w1_ceres.log` |
| ambush_4 | 1 | 1 | 0.1 | 1 | 0.037258 | 0.018940 | 0.200791 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_4_setting1_mode1_w1_sr0.1_ceres.log` |
| ambush_4 | 2 | 1 | 0.3 | 1 | 0.072346 | 0.021678 | 0.256942 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_4_setting2_mode1_w1_sr0.3_ceres.log` |
| ambush_4 | 3 | 1 | 0.5 | 1 | 0.011955 | 0.011925 | 0.093748 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_4_setting3_mode1_w1_sr0.5_ceres.log` |
| ambush_4 | 4 | 1 | 0.9 | 2 | 0.009995 | 0.008000 | 0.084799 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_4_setting4_mode1_w2_sr0.9_ceres.log` |
| ambush_4 | 5 | 1 | 0.9 | 3 | 0.009534 | 0.009490 | 0.089189 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_4_setting5_mode1_w3_sr0.9_ceres.log` |
| ambush_4 | 6 | 2 | - | 1 | 0.044130 | 0.028605 | 0.513926 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_4_setting6_mode2_w1_ceres.log` |
| ambush_5 | 1 | 1 | 0.1 | 1 | 0.048685 | 0.017216 | 0.285866 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_5_setting1_mode1_w1_sr0.1_ceres.log` |
| ambush_5 | 2 | 1 | 0.3 | 1 | 0.180209 | 0.030008 | 2.005156 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_5_setting2_mode1_w1_sr0.3_ceres.log` |
| ambush_5 | 3 | 1 | 0.5 | 1 | 0.212804 | 0.027201 | 0.664841 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_5_setting3_mode1_w1_sr0.5_ceres.log` |
| ambush_5 | 4 | 1 | 0.9 | 2 | 0.144482 | 0.035746 | 0.247258 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_5_setting4_mode1_w2_sr0.9_ceres.log` |
| ambush_5 | 5 | 1 | 0.9 | 3 | 0.011259 | 0.008630 | 0.146632 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_5_setting5_mode1_w3_sr0.9_ceres.log` |
| ambush_5 | 6 | 2 | - | 1 | 0.212374 | 0.082972 | 0.616686 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_5_setting6_mode2_w1_ceres.log` |
| ambush_6 | 1 | 1 | 0.1 | 1 | 0.174028 | 0.100060 | 1.240305 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_6_setting1_mode1_w1_sr0.1_ceres.log` |
| ambush_6 | 2 | 1 | 0.3 | 1 | 0.063291 | 0.064401 | 0.966567 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_6_setting2_mode1_w1_sr0.3_ceres.log` |
| ambush_6 | 3 | 1 | 0.5 | 1 | 0.079573 | 0.076519 | 1.775642 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_6_setting3_mode1_w1_sr0.5_ceres.log` |
| ambush_6 | 4 | 1 | 0.9 | 2 | 0.098349 | 0.040891 | 0.766340 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_6_setting4_mode1_w2_sr0.9_ceres.log` |
| ambush_6 | 5 | 1 | 0.9 | 3 | 0.099014 | 0.045871 | 0.823680 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_6_setting5_mode1_w3_sr0.9_ceres.log` |
| ambush_6 | 6 | 2 | - | 1 | 0.281086 | 0.145398 | 1.246777 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_6_setting6_mode2_w1_ceres.log` |
| ambush_7 | 1 | 1 | 0.1 | 1 | 0.000003 | 0.000004 | 0.059923 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_7_setting1_mode1_w1_sr0.1_ceres.log` |
| ambush_7 | 2 | 1 | 0.3 | 1 | 0.000003 | 0.000004 | 0.059735 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_7_setting2_mode1_w1_sr0.3_ceres.log` |
| ambush_7 | 3 | 1 | 0.5 | 1 | 0.000003 | 0.000004 | 0.061082 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_7_setting3_mode1_w1_sr0.5_ceres.log` |
| ambush_7 | 4 | 1 | 0.9 | 2 | 0.000003 | 0.000004 | 0.058594 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_7_setting4_mode1_w2_sr0.9_ceres.log` |
| ambush_7 | 5 | 1 | 0.9 | 3 | 0.000003 | 0.000004 | 0.053217 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_7_setting5_mode1_w3_sr0.9_ceres.log` |
| ambush_7 | 6 | 2 | - | 1 | 0.000003 | 0.000004 | 0.061401 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_7_setting6_mode2_w1_ceres.log` |
| bamboo_1 | 1 | 1 | 0.1 | 1 | 0.001506 | 0.000843 | 0.012308 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bamboo_1_setting1_mode1_w1_sr0.1_ceres.log` |
| bamboo_1 | 2 | 1 | 0.3 | 1 | 0.001147 | 0.000605 | 0.011072 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bamboo_1_setting2_mode1_w1_sr0.3_ceres.log` |
| bamboo_1 | 3 | 1 | 0.5 | 1 | 0.001127 | 0.000607 | 0.011004 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bamboo_1_setting3_mode1_w1_sr0.5_ceres.log` |
| bamboo_1 | 4 | 1 | 0.9 | 2 | 0.001142 | 0.000666 | 0.011080 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bamboo_1_setting4_mode1_w2_sr0.9_ceres.log` |
| bamboo_1 | 5 | 1 | 0.9 | 3 | 0.001143 | 0.000648 | 0.011245 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bamboo_1_setting5_mode1_w3_sr0.9_ceres.log` |
| bamboo_1 | 6 | 2 | - | 1 | 0.001056 | 0.001004 | 0.012363 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bamboo_1_setting6_mode2_w1_ceres.log` |
| bamboo_2 | 1 | 1 | 0.1 | 1 | - | - | - | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bamboo_2_setting1_mode1_w1_sr0.1_ceres.log` |
| bamboo_2 | 2 | 1 | 0.3 | 1 | - | - | - | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bamboo_2_setting2_mode1_w1_sr0.3_ceres.log` |
| bamboo_2 | 3 | 1 | 0.5 | 1 | - | - | - | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bamboo_2_setting3_mode1_w1_sr0.5_ceres.log` |
| bamboo_2 | 4 | 1 | 0.9 | 2 | - | - | - | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bamboo_2_setting4_mode1_w2_sr0.9_ceres.log` |
| bamboo_2 | 5 | 1 | 0.9 | 3 | - | - | - | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bamboo_2_setting5_mode1_w3_sr0.9_ceres.log` |
| bamboo_2 | 6 | 2 | - | 1 | - | - | - | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bamboo_2_setting6_mode2_w1_ceres.log` |
| bandage_1 | 1 | 1 | 0.1 | 1 | - | - | - | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bandage_1_setting1_mode1_w1_sr0.1_ceres.log` |
| bandage_1 | 2 | 1 | 0.3 | 1 | - | - | - | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bandage_1_setting2_mode1_w1_sr0.3_ceres.log` |
| bandage_1 | 3 | 1 | 0.5 | 1 | - | - | - | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bandage_1_setting3_mode1_w1_sr0.5_ceres.log` |
| bandage_1 | 4 | 1 | 0.9 | 2 | - | - | - | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bandage_1_setting4_mode1_w2_sr0.9_ceres.log` |
| bandage_1 | 5 | 1 | 0.9 | 3 | - | - | - | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bandage_1_setting5_mode1_w3_sr0.9_ceres.log` |
| bandage_1 | 6 | 2 | - | 1 | - | - | - | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bandage_1_setting6_mode2_w1_ceres.log` |
| bandage_2 | 1 | 1 | 0.1 | 1 | - | - | - | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bandage_2_setting1_mode1_w1_sr0.1_ceres.log` |
| bandage_2 | 2 | 1 | 0.3 | 1 | - | - | - | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bandage_2_setting2_mode1_w1_sr0.3_ceres.log` |
| bandage_2 | 3 | 1 | 0.5 | 1 | - | - | - | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bandage_2_setting3_mode1_w1_sr0.5_ceres.log` |
| bandage_2 | 4 | 1 | 0.9 | 2 | - | - | - | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bandage_2_setting4_mode1_w2_sr0.9_ceres.log` |
| bandage_2 | 5 | 1 | 0.9 | 3 | - | - | - | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bandage_2_setting5_mode1_w3_sr0.9_ceres.log` |
| bandage_2 | 6 | 2 | - | 1 | - | - | - | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bandage_2_setting6_mode2_w1_ceres.log` |
| cave_2 | 1 | 1 | 0.1 | 1 | 0.862062 | 0.195667 | 18.952931 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/cave_2_setting1_mode1_w1_sr0.1_ceres.log` |
| cave_2 | 2 | 1 | 0.3 | 1 | 0.963883 | 0.302823 | 38.820529 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/cave_2_setting2_mode1_w1_sr0.3_ceres.log` |
| cave_2 | 3 | 1 | 0.5 | 1 | 0.776743 | 0.144071 | 5.985528 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/cave_2_setting3_mode1_w1_sr0.5_ceres.log` |
| cave_2 | 4 | 1 | 0.9 | 2 | 0.589659 | 0.159884 | 1.796699 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/cave_2_setting4_mode1_w2_sr0.9_ceres.log` |
| cave_2 | 5 | 1 | 0.9 | 3 | 0.606140 | 0.178520 | 2.003818 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/cave_2_setting5_mode1_w3_sr0.9_ceres.log` |
| cave_2 | 6 | 2 | - | 1 | 1.110739 | 0.141688 | 5.413203 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/cave_2_setting6_mode2_w1_ceres.log` |
| cave_4 | 1 | 1 | 0.1 | 1 | 0.007597 | 0.004126 | 0.033085 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/cave_4_setting1_mode1_w1_sr0.1_ceres.log` |
| cave_4 | 2 | 1 | 0.3 | 1 | 0.010443 | 0.004494 | 0.033311 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/cave_4_setting2_mode1_w1_sr0.3_ceres.log` |
| cave_4 | 3 | 1 | 0.5 | 1 | 0.004528 | 0.003721 | 0.031181 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/cave_4_setting3_mode1_w1_sr0.5_ceres.log` |
| cave_4 | 4 | 1 | 0.9 | 2 | 0.004250 | 0.003928 | 0.031728 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/cave_4_setting4_mode1_w2_sr0.9_ceres.log` |
| cave_4 | 5 | 1 | 0.9 | 3 | 0.004387 | 0.003615 | 0.029943 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/cave_4_setting5_mode1_w3_sr0.9_ceres.log` |
| cave_4 | 6 | 2 | - | 1 | 0.005761 | 0.005337 | 0.044335 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/cave_4_setting6_mode2_w1_ceres.log` |
| market_2 | 1 | 1 | 0.1 | 1 | 0.071724 | 0.018007 | 0.136468 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/market_2_setting1_mode1_w1_sr0.1_ceres.log` |
| market_2 | 2 | 1 | 0.3 | 1 | 0.005922 | 0.002003 | 0.011027 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/market_2_setting2_mode1_w1_sr0.3_ceres.log` |
| market_2 | 3 | 1 | 0.5 | 1 | 0.054332 | 0.029462 | 0.077710 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/market_2_setting3_mode1_w1_sr0.5_ceres.log` |
| market_2 | 4 | 1 | 0.9 | 2 | 0.007020 | 0.002494 | 0.005098 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/market_2_setting4_mode1_w2_sr0.9_ceres.log` |
| market_2 | 5 | 1 | 0.9 | 3 | 0.007633 | 0.002781 | 0.005683 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/market_2_setting5_mode1_w3_sr0.9_ceres.log` |
| market_2 | 6 | 2 | - | 1 | 0.004464 | 0.003613 | 0.006421 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/market_2_setting6_mode2_w1_ceres.log` |
| market_5 | 1 | 1 | 0.1 | 1 | 0.006026 | 0.007170 | 0.556525 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/market_5_setting1_mode1_w1_sr0.1_ceres.log` |
| market_5 | 2 | 1 | 0.3 | 1 | 0.005584 | 0.007191 | 0.528403 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/market_5_setting2_mode1_w1_sr0.3_ceres.log` |
| market_5 | 3 | 1 | 0.5 | 1 | 0.005003 | 0.007343 | 0.245274 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/market_5_setting3_mode1_w1_sr0.5_ceres.log` |
| market_5 | 4 | 1 | 0.9 | 2 | 0.005420 | 0.007168 | 0.233080 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/market_5_setting4_mode1_w2_sr0.9_ceres.log` |
| market_5 | 5 | 1 | 0.9 | 3 | 0.004248 | 0.007293 | 0.232889 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/market_5_setting5_mode1_w3_sr0.9_ceres.log` |
| market_5 | 6 | 2 | - | 1 | 0.019969 | 0.011974 | 2.824471 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/market_5_setting6_mode2_w1_ceres.log` |
| market_6 | 1 | 1 | 0.1 | 1 | 0.022617 | 0.011534 | 0.102916 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/market_6_setting1_mode1_w1_sr0.1_ceres.log` |
| market_6 | 2 | 1 | 0.3 | 1 | 0.020468 | 0.008882 | 0.064263 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/market_6_setting2_mode1_w1_sr0.3_ceres.log` |
| market_6 | 3 | 1 | 0.5 | 1 | 0.030886 | 0.013201 | 0.118240 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/market_6_setting3_mode1_w1_sr0.5_ceres.log` |
| market_6 | 4 | 1 | 0.9 | 2 | 0.032578 | 0.014600 | 0.094972 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/market_6_setting4_mode1_w2_sr0.9_ceres.log` |
| market_6 | 5 | 1 | 0.9 | 3 | 0.019461 | 0.016775 | 0.098393 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/market_6_setting5_mode1_w3_sr0.9_ceres.log` |
| market_6 | 6 | 2 | - | 1 | 0.039896 | 0.014278 | 0.115955 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/market_6_setting6_mode2_w1_ceres.log` |
| mountain_1 | 1 | 1 | 0.1 | 1 | 0.418945 | 0.102316 | 0.010648 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/mountain_1_setting1_mode1_w1_sr0.1_ceres.log` |
| mountain_1 | 2 | 1 | 0.3 | 1 | 0.407617 | 0.111343 | 0.010798 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/mountain_1_setting2_mode1_w1_sr0.3_ceres.log` |
| mountain_1 | 3 | 1 | 0.5 | 1 | 0.578257 | 0.155774 | 0.012851 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/mountain_1_setting3_mode1_w1_sr0.5_ceres.log` |
| mountain_1 | 4 | 1 | 0.9 | 2 | 0.503592 | 0.118199 | 0.011369 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/mountain_1_setting4_mode1_w2_sr0.9_ceres.log` |
| mountain_1 | 5 | 1 | 0.9 | 3 | 0.439370 | 0.104948 | 0.010868 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/mountain_1_setting5_mode1_w3_sr0.9_ceres.log` |
| mountain_1 | 6 | 2 | - | 1 | 0.346532 | 0.136866 | 0.015877 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/mountain_1_setting6_mode2_w1_ceres.log` |
| shaman_2 | 1 | 1 | 0.1 | 1 | - | - | - | 3 | `Python/logs/sintel_xm_penalty_strong/solver/shaman_2_setting1_mode1_w1_sr0.1_ceres.log` |
| shaman_2 | 2 | 1 | 0.3 | 1 | - | - | - | 3 | `Python/logs/sintel_xm_penalty_strong/solver/shaman_2_setting2_mode1_w1_sr0.3_ceres.log` |
| shaman_2 | 3 | 1 | 0.5 | 1 | - | - | - | 3 | `Python/logs/sintel_xm_penalty_strong/solver/shaman_2_setting3_mode1_w1_sr0.5_ceres.log` |
| shaman_2 | 4 | 1 | 0.9 | 2 | - | - | - | 3 | `Python/logs/sintel_xm_penalty_strong/solver/shaman_2_setting4_mode1_w2_sr0.9_ceres.log` |
| shaman_2 | 5 | 1 | 0.9 | 3 | - | - | - | 3 | `Python/logs/sintel_xm_penalty_strong/solver/shaman_2_setting5_mode1_w3_sr0.9_ceres.log` |
| shaman_2 | 6 | 2 | - | 1 | - | - | - | 3 | `Python/logs/sintel_xm_penalty_strong/solver/shaman_2_setting6_mode2_w1_ceres.log` |
| shaman_3 | 1 | 1 | 0.1 | 1 | 0.003023 | 0.000734 | 0.077597 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/shaman_3_setting1_mode1_w1_sr0.1_ceres.log` |
| shaman_3 | 2 | 1 | 0.3 | 1 | 0.002983 | 0.000735 | 0.077664 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/shaman_3_setting2_mode1_w1_sr0.3_ceres.log` |
| shaman_3 | 3 | 1 | 0.5 | 1 | 0.003000 | 0.000739 | 0.076212 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/shaman_3_setting3_mode1_w1_sr0.5_ceres.log` |
| shaman_3 | 4 | 1 | 0.9 | 2 | 0.003166 | 0.002424 | 0.162491 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/shaman_3_setting4_mode1_w2_sr0.9_ceres.log` |
| shaman_3 | 5 | 1 | 0.9 | 3 | 0.004679 | 0.002074 | 0.261880 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/shaman_3_setting5_mode1_w3_sr0.9_ceres.log` |
| shaman_3 | 6 | 2 | - | 1 | 0.003116 | 0.000774 | 0.077459 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/shaman_3_setting6_mode2_w1_ceres.log` |
| sleeping_1 | 1 | 1 | 0.1 | 1 | 0.004590 | 0.002898 | 0.186341 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/sleeping_1_setting1_mode1_w1_sr0.1_ceres.log` |
| sleeping_1 | 2 | 1 | 0.3 | 1 | 0.004705 | 0.002972 | 0.190170 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/sleeping_1_setting2_mode1_w1_sr0.3_ceres.log` |
| sleeping_1 | 3 | 1 | 0.5 | 1 | 0.004645 | 0.002925 | 0.187165 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/sleeping_1_setting3_mode1_w1_sr0.5_ceres.log` |
| sleeping_1 | 4 | 1 | 0.9 | 2 | 0.028484 | 0.006400 | 0.270010 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/sleeping_1_setting4_mode1_w2_sr0.9_ceres.log` |
| sleeping_1 | 5 | 1 | 0.9 | 3 | 0.015347 | 0.004184 | 0.211048 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/sleeping_1_setting5_mode1_w3_sr0.9_ceres.log` |
| sleeping_1 | 6 | 2 | - | 1 | 0.005315 | 0.004213 | 0.278245 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/sleeping_1_setting6_mode2_w1_ceres.log` |
| sleeping_2 | 1 | 1 | 0.1 | 1 | 0.000589 | 0.000415 | 0.009206 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/sleeping_2_setting1_mode1_w1_sr0.1_ceres.log` |
| sleeping_2 | 2 | 1 | 0.3 | 1 | 0.000590 | 0.000416 | 0.009226 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/sleeping_2_setting2_mode1_w1_sr0.3_ceres.log` |
| sleeping_2 | 3 | 1 | 0.5 | 1 | 0.000590 | 0.000416 | 0.009230 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/sleeping_2_setting3_mode1_w1_sr0.5_ceres.log` |
| sleeping_2 | 4 | 1 | 0.9 | 2 | 0.000626 | 0.000903 | 0.009468 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/sleeping_2_setting4_mode1_w2_sr0.9_ceres.log` |
| sleeping_2 | 5 | 1 | 0.9 | 3 | 0.000610 | 0.000466 | 0.009439 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/sleeping_2_setting5_mode1_w3_sr0.9_ceres.log` |
| sleeping_2 | 6 | 2 | - | 1 | 0.000649 | 0.000531 | 0.010911 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/sleeping_2_setting6_mode2_w1_ceres.log` |
| temple_2 | 1 | 1 | 0.1 | 1 | 0.123636 | 0.018859 | 0.230259 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/temple_2_setting1_mode1_w1_sr0.1_ceres.log` |
| temple_2 | 2 | 1 | 0.3 | 1 | 0.107697 | 0.087934 | 0.383293 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/temple_2_setting2_mode1_w1_sr0.3_ceres.log` |
| temple_2 | 3 | 1 | 0.5 | 1 | 0.137183 | 0.035291 | 0.211983 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/temple_2_setting3_mode1_w1_sr0.5_ceres.log` |
| temple_2 | 4 | 1 | 0.9 | 2 | 0.163628 | 0.024739 | 0.244359 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/temple_2_setting4_mode1_w2_sr0.9_ceres.log` |
| temple_2 | 5 | 1 | 0.9 | 3 | 0.107195 | 0.039736 | 0.255101 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/temple_2_setting5_mode1_w3_sr0.9_ceres.log` |
| temple_2 | 6 | 2 | - | 1 | 0.149078 | 0.064486 | 0.257658 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/temple_2_setting6_mode2_w1_ceres.log` |
| temple_3 | 1 | 1 | 0.1 | 1 | 0.286375 | 0.256797 | 0.085877 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/temple_3_setting1_mode1_w1_sr0.1_ceres.log` |
| temple_3 | 2 | 1 | 0.3 | 1 | 0.364675 | 0.294725 | 0.075727 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/temple_3_setting2_mode1_w1_sr0.3_ceres.log` |
| temple_3 | 3 | 1 | 0.5 | 1 | 0.350724 | 0.299362 | 0.076564 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/temple_3_setting3_mode1_w1_sr0.5_ceres.log` |
| temple_3 | 4 | 1 | 0.9 | 2 | 0.230334 | 0.202423 | 0.101318 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/temple_3_setting4_mode1_w2_sr0.9_ceres.log` |
| temple_3 | 5 | 1 | 0.9 | 3 | 0.620653 | 0.177611 | 0.418637 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/temple_3_setting5_mode1_w3_sr0.9_ceres.log` |
| temple_3 | 6 | 2 | - | 1 | 0.394269 | 0.361217 | 0.066669 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/temple_3_setting6_mode2_w1_ceres.log` |
