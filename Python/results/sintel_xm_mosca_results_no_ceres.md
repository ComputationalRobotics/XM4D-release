# Sintel MoSca-Style Eval Results (without Ceres refinement)

Generated: `2026-04-22T12:46:10-04:00`

Source:

- Metrics are extracted from the `MoSca-style eval (evo)` block in the logs referenced by `Python/sintel_xm_results_no_ceres.md`.
- Columns correspond to `ATE`, `RPE trans`, and `RPE rot` exactly as printed there.
- Missing MoSca metrics are rendered as `-` and listed below.

Missing metrics:

- `cave_2` setting `6`: invalid estimated rotation matrix: Non-positive determinant (left-handed or null coordinate frame) in rotation matrix 0: [[-5.61930254e-0...

| Scene | Setting | Mode | Sphere Ratio | WEIGHT_MODE | ATE (m) | RPE trans (m) | RPE rot (deg) | Optimal rank | Log |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| alley_1 | 1 | 1 | 0.1 | 1 | 0.009754 | 0.005837 | 0.077301 | 3 | `Python/logs/sintel_xm_no_ceres/solver/alley_1_setting1_mode1_w1_sr0.1_no_ceres.log` |
| alley_1 | 2 | 1 | 0.3 | 1 | 0.009589 | 0.005798 | 0.077472 | 3 | `Python/logs/sintel_xm_no_ceres/solver/alley_1_setting2_mode1_w1_sr0.3_no_ceres.log` |
| alley_1 | 3 | 1 | 0.5 | 1 | 0.009508 | 0.005800 | 0.077058 | 3 | `Python/logs/sintel_xm_no_ceres/solver/alley_1_setting3_mode1_w1_sr0.5_no_ceres.log` |
| alley_1 | 4 | 1 | 0.9 | 2 | 0.008153 | 0.005481 | 0.014684 | 3 | `Python/logs/sintel_xm_no_ceres/solver/alley_1_setting4_mode1_w2_sr0.9_no_ceres.log` |
| alley_1 | 5 | 1 | 0.9 | 3 | 0.009366 | 0.005447 | 0.049606 | 3 | `Python/logs/sintel_xm_no_ceres/solver/alley_1_setting5_mode1_w3_sr0.9_no_ceres.log` |
| alley_1 | 6 | 2 | - | 1 | 0.015961 | 0.009570 | 0.170024 | 3 | `Python/logs/sintel_xm_no_ceres/solver/alley_1_setting6_mode2_w1_no_ceres.log` |
| alley_2 | 1 | 1 | 0.1 | 1 | 0.115445 | 0.024335 | 0.301643 | 3 | `Python/logs/sintel_xm_no_ceres/solver/alley_2_setting1_mode1_w1_sr0.1_no_ceres.log` |
| alley_2 | 2 | 1 | 0.3 | 1 | 0.102611 | 0.023142 | 0.302682 | 3 | `Python/logs/sintel_xm_no_ceres/solver/alley_2_setting2_mode1_w1_sr0.3_no_ceres.log` |
| alley_2 | 3 | 1 | 0.5 | 1 | 0.091957 | 0.020651 | 0.303486 | 3 | `Python/logs/sintel_xm_no_ceres/solver/alley_2_setting3_mode1_w1_sr0.5_no_ceres.log` |
| alley_2 | 4 | 1 | 0.9 | 2 | 0.062923 | 0.014359 | 0.350614 | 3 | `Python/logs/sintel_xm_no_ceres/solver/alley_2_setting4_mode1_w2_sr0.9_no_ceres.log` |
| alley_2 | 5 | 1 | 0.9 | 3 | 0.086906 | 0.018143 | 0.313391 | 3 | `Python/logs/sintel_xm_no_ceres/solver/alley_2_setting5_mode1_w3_sr0.9_no_ceres.log` |
| alley_2 | 6 | 2 | - | 1 | 0.129554 | 0.042478 | 0.357846 | 3 | `Python/logs/sintel_xm_no_ceres/solver/alley_2_setting6_mode2_w1_no_ceres.log` |
| ambush_2 | 1 | 1 | 0.1 | 1 | 0.000013 | 0.000006 | 7.630334 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_2_setting1_mode1_w1_sr0.1_no_ceres.log` |
| ambush_2 | 2 | 1 | 0.3 | 1 | 0.000007 | 0.000004 | 8.598679 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_2_setting2_mode1_w1_sr0.3_no_ceres.log` |
| ambush_2 | 3 | 1 | 0.5 | 1 | 0.000005 | 0.000004 | 8.451006 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_2_setting3_mode1_w1_sr0.5_no_ceres.log` |
| ambush_2 | 4 | 1 | 0.9 | 2 | 0.000003 | 0.000004 | 2.864061 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_2_setting4_mode1_w2_sr0.9_no_ceres.log` |
| ambush_2 | 5 | 1 | 0.9 | 3 | 0.000003 | 0.000004 | 4.139416 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_2_setting5_mode1_w3_sr0.9_no_ceres.log` |
| ambush_2 | 6 | 2 | - | 1 | 0.000007 | 0.000004 | 8.393084 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_2_setting6_mode2_w1_no_ceres.log` |
| ambush_4 | 1 | 1 | 0.1 | 1 | 0.075100 | 0.026907 | 0.342880 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_4_setting1_mode1_w1_sr0.1_no_ceres.log` |
| ambush_4 | 2 | 1 | 0.3 | 1 | 0.077239 | 0.027311 | 0.456630 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_4_setting2_mode1_w1_sr0.3_no_ceres.log` |
| ambush_4 | 3 | 1 | 0.5 | 1 | 0.071308 | 0.027361 | 0.433725 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_4_setting3_mode1_w1_sr0.5_no_ceres.log` |
| ambush_4 | 4 | 1 | 0.9 | 2 | 0.189467 | 0.026245 | 0.117001 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_4_setting4_mode1_w2_sr0.9_no_ceres.log` |
| ambush_4 | 5 | 1 | 0.9 | 3 | 0.086370 | 0.025530 | 0.125290 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_4_setting5_mode1_w3_sr0.9_no_ceres.log` |
| ambush_4 | 6 | 2 | - | 1 | 0.148744 | 0.039868 | 0.251834 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_4_setting6_mode2_w1_no_ceres.log` |
| ambush_5 | 1 | 1 | 0.1 | 1 | 0.235803 | 0.100033 | 1.090970 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_5_setting1_mode1_w1_sr0.1_no_ceres.log` |
| ambush_5 | 2 | 1 | 0.3 | 1 | 0.238727 | 0.090428 | 1.028988 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_5_setting2_mode1_w1_sr0.3_no_ceres.log` |
| ambush_5 | 3 | 1 | 0.5 | 1 | 0.396194 | 0.102548 | 1.014464 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_5_setting3_mode1_w1_sr0.5_no_ceres.log` |
| ambush_5 | 4 | 1 | 0.9 | 2 | 0.190807 | 0.056139 | 0.248621 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_5_setting4_mode1_w2_sr0.9_no_ceres.log` |
| ambush_5 | 5 | 1 | 0.9 | 3 | 0.343978 | 0.103964 | 0.527199 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_5_setting5_mode1_w3_sr0.9_no_ceres.log` |
| ambush_5 | 6 | 2 | - | 1 | 0.848476 | 1.007358 | 1.198605 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_5_setting6_mode2_w1_no_ceres.log` |
| ambush_6 | 1 | 1 | 0.1 | 1 | 0.259052 | 0.143290 | 2.026080 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_6_setting1_mode1_w1_sr0.1_no_ceres.log` |
| ambush_6 | 2 | 1 | 0.3 | 1 | 0.288444 | 0.142303 | 2.023602 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_6_setting2_mode1_w1_sr0.3_no_ceres.log` |
| ambush_6 | 3 | 1 | 0.5 | 1 | 0.278839 | 0.144033 | 2.019461 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_6_setting3_mode1_w1_sr0.5_no_ceres.log` |
| ambush_6 | 4 | 1 | 0.9 | 2 | 0.275294 | 0.097316 | 1.728234 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_6_setting4_mode1_w2_sr0.9_no_ceres.log` |
| ambush_6 | 5 | 1 | 0.9 | 3 | 0.275603 | 0.116820 | 1.754777 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_6_setting5_mode1_w3_sr0.9_no_ceres.log` |
| ambush_6 | 6 | 2 | - | 1 | 0.438407 | 0.270133 | 2.037699 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_6_setting6_mode2_w1_no_ceres.log` |
| ambush_7 | 1 | 1 | 0.1 | 1 | 0.000004 | 0.000004 | 0.215112 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_7_setting1_mode1_w1_sr0.1_no_ceres.log` |
| ambush_7 | 2 | 1 | 0.3 | 1 | 0.000004 | 0.000004 | 0.215570 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_7_setting2_mode1_w1_sr0.3_no_ceres.log` |
| ambush_7 | 3 | 1 | 0.5 | 1 | 0.000004 | 0.000004 | 0.215910 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_7_setting3_mode1_w1_sr0.5_no_ceres.log` |
| ambush_7 | 4 | 1 | 0.9 | 2 | 0.000004 | 0.000004 | 0.047198 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_7_setting4_mode1_w2_sr0.9_no_ceres.log` |
| ambush_7 | 5 | 1 | 0.9 | 3 | 0.000004 | 0.000004 | 0.079925 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_7_setting5_mode1_w3_sr0.9_no_ceres.log` |
| ambush_7 | 6 | 2 | - | 1 | 0.000004 | 0.000004 | 0.740914 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_7_setting6_mode2_w1_no_ceres.log` |
| bamboo_1 | 1 | 1 | 0.1 | 1 | 0.038550 | 0.022025 | 0.053589 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bamboo_1_setting1_mode1_w1_sr0.1_no_ceres.log` |
| bamboo_1 | 2 | 1 | 0.3 | 1 | 0.030546 | 0.013673 | 0.054453 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bamboo_1_setting2_mode1_w1_sr0.3_no_ceres.log` |
| bamboo_1 | 3 | 1 | 0.5 | 1 | 0.029546 | 0.010444 | 0.054615 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bamboo_1_setting3_mode1_w1_sr0.5_no_ceres.log` |
| bamboo_1 | 4 | 1 | 0.9 | 2 | 0.035585 | 0.006040 | 0.056406 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bamboo_1_setting4_mode1_w2_sr0.9_no_ceres.log` |
| bamboo_1 | 5 | 1 | 0.9 | 3 | 0.030818 | 0.005582 | 0.039098 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bamboo_1_setting5_mode1_w3_sr0.9_no_ceres.log` |
| bamboo_1 | 6 | 2 | - | 1 | 0.109489 | 0.151395 | 0.081492 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bamboo_1_setting6_mode2_w1_no_ceres.log` |
| bamboo_2 | 1 | 1 | 0.1 | 1 | 0.100857 | 0.036890 | 0.098672 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bamboo_2_setting1_mode1_w1_sr0.1_no_ceres.log` |
| bamboo_2 | 2 | 1 | 0.3 | 1 | 0.075435 | 0.030079 | 0.097010 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bamboo_2_setting2_mode1_w1_sr0.3_no_ceres.log` |
| bamboo_2 | 3 | 1 | 0.5 | 1 | 0.062865 | 0.025678 | 0.093565 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bamboo_2_setting3_mode1_w1_sr0.5_no_ceres.log` |
| bamboo_2 | 4 | 1 | 0.9 | 2 | 0.066434 | 0.008131 | 0.075472 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bamboo_2_setting4_mode1_w2_sr0.9_no_ceres.log` |
| bamboo_2 | 5 | 1 | 0.9 | 3 | 0.064621 | 0.014360 | 0.068750 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bamboo_2_setting5_mode1_w3_sr0.9_no_ceres.log` |
| bamboo_2 | 6 | 2 | - | 1 | 0.132190 | 0.130261 | 0.103799 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bamboo_2_setting6_mode2_w1_no_ceres.log` |
| bandage_1 | 1 | 1 | 0.1 | 1 | 0.208208 | 0.018326 | 1.017392 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bandage_1_setting1_mode1_w1_sr0.1_no_ceres.log` |
| bandage_1 | 2 | 1 | 0.3 | 1 | 0.207148 | 0.017002 | 1.020886 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bandage_1_setting2_mode1_w1_sr0.3_no_ceres.log` |
| bandage_1 | 3 | 1 | 0.5 | 1 | 0.205930 | 0.016905 | 1.018597 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bandage_1_setting3_mode1_w1_sr0.5_no_ceres.log` |
| bandage_1 | 4 | 1 | 0.9 | 2 | 0.031293 | 0.002094 | 1.128533 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bandage_1_setting4_mode1_w2_sr0.9_no_ceres.log` |
| bandage_1 | 5 | 1 | 0.9 | 3 | 0.109854 | 0.007904 | 0.945514 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bandage_1_setting5_mode1_w3_sr0.9_no_ceres.log` |
| bandage_1 | 6 | 2 | - | 1 | 0.108853 | 0.031176 | 0.959253 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bandage_1_setting6_mode2_w1_no_ceres.log` |
| bandage_2 | 1 | 1 | 0.1 | 1 | 0.031988 | 0.009274 | 0.168382 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bandage_2_setting1_mode1_w1_sr0.1_no_ceres.log` |
| bandage_2 | 2 | 1 | 0.3 | 1 | 0.031428 | 0.009096 | 0.164715 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bandage_2_setting2_mode1_w1_sr0.3_no_ceres.log` |
| bandage_2 | 3 | 1 | 0.5 | 1 | 0.031206 | 0.009100 | 0.163394 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bandage_2_setting3_mode1_w1_sr0.5_no_ceres.log` |
| bandage_2 | 4 | 1 | 0.9 | 2 | 0.005528 | 0.000548 | 0.011382 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bandage_2_setting4_mode1_w2_sr0.9_no_ceres.log` |
| bandage_2 | 5 | 1 | 0.9 | 3 | 0.014033 | 0.002335 | 0.042452 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bandage_2_setting5_mode1_w3_sr0.9_no_ceres.log` |
| bandage_2 | 6 | 2 | - | 1 | 0.035685 | 0.022156 | 0.329003 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bandage_2_setting6_mode2_w1_no_ceres.log` |
| cave_2 | 1 | 1 | 0.1 | 1 | 1.147176 | 0.206422 | 4.016630 | 3 | `Python/logs/sintel_xm_no_ceres/solver/cave_2_setting1_mode1_w1_sr0.1_no_ceres.log` |
| cave_2 | 2 | 1 | 0.3 | 1 | 1.100138 | 0.193507 | 4.439885 | 3 | `Python/logs/sintel_xm_no_ceres/solver/cave_2_setting2_mode1_w1_sr0.3_no_ceres.log` |
| cave_2 | 3 | 1 | 0.5 | 1 | 1.038101 | 0.188870 | 3.894330 | 3 | `Python/logs/sintel_xm_no_ceres/solver/cave_2_setting3_mode1_w1_sr0.5_no_ceres.log` |
| cave_2 | 4 | 1 | 0.9 | 2 | 1.403257 | 0.183196 | 1.788815 | 3 | `Python/logs/sintel_xm_no_ceres/solver/cave_2_setting4_mode1_w2_sr0.9_no_ceres.log` |
| cave_2 | 5 | 1 | 0.9 | 3 | 1.215409 | 0.166232 | 2.268823 | 3 | `Python/logs/sintel_xm_no_ceres/solver/cave_2_setting5_mode1_w3_sr0.9_no_ceres.log` |
| cave_2 | 6 | 2 | - | 1 | - | - | - | 4 | `Python/logs/sintel_xm_no_ceres/solver/cave_2_setting6_mode2_w1_no_ceres.log` |
| cave_4 | 1 | 1 | 0.1 | 1 | 0.048332 | 0.013093 | 0.077060 | 3 | `Python/logs/sintel_xm_no_ceres/solver/cave_4_setting1_mode1_w1_sr0.1_no_ceres.log` |
| cave_4 | 2 | 1 | 0.3 | 1 | 0.051532 | 0.014605 | 0.077778 | 3 | `Python/logs/sintel_xm_no_ceres/solver/cave_4_setting2_mode1_w1_sr0.3_no_ceres.log` |
| cave_4 | 3 | 1 | 0.5 | 1 | 0.052373 | 0.014715 | 0.078812 | 3 | `Python/logs/sintel_xm_no_ceres/solver/cave_4_setting3_mode1_w1_sr0.5_no_ceres.log` |
| cave_4 | 4 | 1 | 0.9 | 2 | 0.043215 | 0.009263 | 0.128704 | 3 | `Python/logs/sintel_xm_no_ceres/solver/cave_4_setting4_mode1_w2_sr0.9_no_ceres.log` |
| cave_4 | 5 | 1 | 0.9 | 3 | 0.034389 | 0.008058 | 0.082893 | 3 | `Python/logs/sintel_xm_no_ceres/solver/cave_4_setting5_mode1_w3_sr0.9_no_ceres.log` |
| cave_4 | 6 | 2 | - | 1 | 0.190479 | 0.158895 | 0.697952 | 3 | `Python/logs/sintel_xm_no_ceres/solver/cave_4_setting6_mode2_w1_no_ceres.log` |
| market_2 | 1 | 1 | 0.1 | 1 | 0.085003 | 0.040095 | 0.021496 | 3 | `Python/logs/sintel_xm_no_ceres/solver/market_2_setting1_mode1_w1_sr0.1_no_ceres.log` |
| market_2 | 2 | 1 | 0.3 | 1 | 0.089585 | 0.039942 | 0.021280 | 3 | `Python/logs/sintel_xm_no_ceres/solver/market_2_setting2_mode1_w1_sr0.3_no_ceres.log` |
| market_2 | 3 | 1 | 0.5 | 1 | 0.068900 | 0.040028 | 0.021639 | 3 | `Python/logs/sintel_xm_no_ceres/solver/market_2_setting3_mode1_w1_sr0.5_no_ceres.log` |
| market_2 | 4 | 1 | 0.9 | 2 | 0.012177 | 0.005249 | 0.060939 | 3 | `Python/logs/sintel_xm_no_ceres/solver/market_2_setting4_mode1_w2_sr0.9_no_ceres.log` |
| market_2 | 5 | 1 | 0.9 | 3 | 0.023082 | 0.012309 | 0.025455 | 3 | `Python/logs/sintel_xm_no_ceres/solver/market_2_setting5_mode1_w3_sr0.9_no_ceres.log` |
| market_2 | 6 | 2 | - | 1 | 0.101058 | 0.060763 | 0.059493 | 3 | `Python/logs/sintel_xm_no_ceres/solver/market_2_setting6_mode2_w1_no_ceres.log` |
| market_5 | 1 | 1 | 0.1 | 1 | 0.006618 | 0.007271 | 0.308119 | 3 | `Python/logs/sintel_xm_no_ceres/solver/market_5_setting1_mode1_w1_sr0.1_no_ceres.log` |
| market_5 | 2 | 1 | 0.3 | 1 | 0.006727 | 0.007201 | 0.305446 | 3 | `Python/logs/sintel_xm_no_ceres/solver/market_5_setting2_mode1_w1_sr0.3_no_ceres.log` |
| market_5 | 3 | 1 | 0.5 | 1 | 0.006157 | 0.007184 | 0.304977 | 3 | `Python/logs/sintel_xm_no_ceres/solver/market_5_setting3_mode1_w1_sr0.5_no_ceres.log` |
| market_5 | 4 | 1 | 0.9 | 2 | 0.018230 | 0.007951 | 0.805526 | 3 | `Python/logs/sintel_xm_no_ceres/solver/market_5_setting4_mode1_w2_sr0.9_no_ceres.log` |
| market_5 | 5 | 1 | 0.9 | 3 | 0.004850 | 0.007273 | 0.401976 | 3 | `Python/logs/sintel_xm_no_ceres/solver/market_5_setting5_mode1_w3_sr0.9_no_ceres.log` |
| market_5 | 6 | 2 | - | 1 | 0.007748 | 0.009293 | 1.089768 | 3 | `Python/logs/sintel_xm_no_ceres/solver/market_5_setting6_mode2_w1_no_ceres.log` |
| market_6 | 1 | 1 | 0.1 | 1 | 0.047334 | 0.028382 | 0.081242 | 3 | `Python/logs/sintel_xm_no_ceres/solver/market_6_setting1_mode1_w1_sr0.1_no_ceres.log` |
| market_6 | 2 | 1 | 0.3 | 1 | 0.040446 | 0.016718 | 0.075716 | 3 | `Python/logs/sintel_xm_no_ceres/solver/market_6_setting2_mode1_w1_sr0.3_no_ceres.log` |
| market_6 | 3 | 1 | 0.5 | 1 | 0.046375 | 0.017057 | 0.073946 | 3 | `Python/logs/sintel_xm_no_ceres/solver/market_6_setting3_mode1_w1_sr0.5_no_ceres.log` |
| market_6 | 4 | 1 | 0.9 | 2 | 0.161289 | 0.035633 | 0.103430 | 3 | `Python/logs/sintel_xm_no_ceres/solver/market_6_setting4_mode1_w2_sr0.9_no_ceres.log` |
| market_6 | 5 | 1 | 0.9 | 3 | 0.077941 | 0.021594 | 0.061855 | 3 | `Python/logs/sintel_xm_no_ceres/solver/market_6_setting5_mode1_w3_sr0.9_no_ceres.log` |
| market_6 | 6 | 2 | - | 1 | 0.065953 | 0.047464 | 0.182091 | 3 | `Python/logs/sintel_xm_no_ceres/solver/market_6_setting6_mode2_w1_no_ceres.log` |
| mountain_1 | 1 | 1 | 0.1 | 1 | 1.682782 | 0.645642 | 0.053018 | 3 | `Python/logs/sintel_xm_no_ceres/solver/mountain_1_setting1_mode1_w1_sr0.1_no_ceres.log` |
| mountain_1 | 2 | 1 | 0.3 | 1 | 1.417158 | 0.547002 | 0.052574 | 3 | `Python/logs/sintel_xm_no_ceres/solver/mountain_1_setting2_mode1_w1_sr0.3_no_ceres.log` |
| mountain_1 | 3 | 1 | 0.5 | 1 | 1.325276 | 0.507790 | 0.052467 | 3 | `Python/logs/sintel_xm_no_ceres/solver/mountain_1_setting3_mode1_w1_sr0.5_no_ceres.log` |
| mountain_1 | 4 | 1 | 0.9 | 2 | 5.305029 | 1.272800 | 0.043687 | 3 | `Python/logs/sintel_xm_no_ceres/solver/mountain_1_setting4_mode1_w2_sr0.9_no_ceres.log` |
| mountain_1 | 5 | 1 | 0.9 | 3 | 1.993077 | 0.734545 | 0.047809 | 3 | `Python/logs/sintel_xm_no_ceres/solver/mountain_1_setting5_mode1_w3_sr0.9_no_ceres.log` |
| mountain_1 | 6 | 2 | - | 1 | 1.739007 | 2.171555 | 0.106981 | 3 | `Python/logs/sintel_xm_no_ceres/solver/mountain_1_setting6_mode2_w1_no_ceres.log` |
| shaman_2 | 1 | 1 | 0.1 | 1 | 0.044938 | 0.005382 | 0.165819 | 3 | `Python/logs/sintel_xm_no_ceres/solver/shaman_2_setting1_mode1_w1_sr0.1_no_ceres.log` |
| shaman_2 | 2 | 1 | 0.3 | 1 | 0.044433 | 0.005389 | 0.165006 | 3 | `Python/logs/sintel_xm_no_ceres/solver/shaman_2_setting2_mode1_w1_sr0.3_no_ceres.log` |
| shaman_2 | 3 | 1 | 0.5 | 1 | 0.044324 | 0.005394 | 0.164951 | 3 | `Python/logs/sintel_xm_no_ceres/solver/shaman_2_setting3_mode1_w1_sr0.5_no_ceres.log` |
| shaman_2 | 4 | 1 | 0.9 | 2 | 0.025382 | 0.001704 | 0.139061 | 3 | `Python/logs/sintel_xm_no_ceres/solver/shaman_2_setting4_mode1_w2_sr0.9_no_ceres.log` |
| shaman_2 | 5 | 1 | 0.9 | 3 | 0.039282 | 0.003048 | 0.149614 | 3 | `Python/logs/sintel_xm_no_ceres/solver/shaman_2_setting5_mode1_w3_sr0.9_no_ceres.log` |
| shaman_2 | 6 | 2 | - | 1 | 0.041171 | 0.007085 | 0.303404 | 3 | `Python/logs/sintel_xm_no_ceres/solver/shaman_2_setting6_mode2_w1_no_ceres.log` |
| shaman_3 | 1 | 1 | 0.1 | 1 | 0.004182 | 0.001629 | 0.124174 | 3 | `Python/logs/sintel_xm_no_ceres/solver/shaman_3_setting1_mode1_w1_sr0.1_no_ceres.log` |
| shaman_3 | 2 | 1 | 0.3 | 1 | 0.004184 | 0.001627 | 0.123868 | 3 | `Python/logs/sintel_xm_no_ceres/solver/shaman_3_setting2_mode1_w1_sr0.3_no_ceres.log` |
| shaman_3 | 3 | 1 | 0.5 | 1 | 0.004226 | 0.001622 | 0.124071 | 3 | `Python/logs/sintel_xm_no_ceres/solver/shaman_3_setting3_mode1_w1_sr0.5_no_ceres.log` |
| shaman_3 | 4 | 1 | 0.9 | 2 | 0.010408 | 0.002143 | 0.422843 | 3 | `Python/logs/sintel_xm_no_ceres/solver/shaman_3_setting4_mode1_w2_sr0.9_no_ceres.log` |
| shaman_3 | 5 | 1 | 0.9 | 3 | 0.007964 | 0.001939 | 0.256911 | 3 | `Python/logs/sintel_xm_no_ceres/solver/shaman_3_setting5_mode1_w3_sr0.9_no_ceres.log` |
| shaman_3 | 6 | 2 | - | 1 | 0.006048 | 0.004300 | 0.174670 | 3 | `Python/logs/sintel_xm_no_ceres/solver/shaman_3_setting6_mode2_w1_no_ceres.log` |
| sleeping_1 | 1 | 1 | 0.1 | 1 | 0.005563 | 0.002415 | 0.111236 | 3 | `Python/logs/sintel_xm_no_ceres/solver/sleeping_1_setting1_mode1_w1_sr0.1_no_ceres.log` |
| sleeping_1 | 2 | 1 | 0.3 | 1 | 0.005865 | 0.002380 | 0.112227 | 3 | `Python/logs/sintel_xm_no_ceres/solver/sleeping_1_setting2_mode1_w1_sr0.3_no_ceres.log` |
| sleeping_1 | 3 | 1 | 0.5 | 1 | 0.005862 | 0.002353 | 0.111750 | 3 | `Python/logs/sintel_xm_no_ceres/solver/sleeping_1_setting3_mode1_w1_sr0.5_no_ceres.log` |
| sleeping_1 | 4 | 1 | 0.9 | 2 | 0.030016 | 0.005134 | 0.250336 | 3 | `Python/logs/sintel_xm_no_ceres/solver/sleeping_1_setting4_mode1_w2_sr0.9_no_ceres.log` |
| sleeping_1 | 5 | 1 | 0.9 | 3 | 0.021378 | 0.004291 | 0.189069 | 3 | `Python/logs/sintel_xm_no_ceres/solver/sleeping_1_setting5_mode1_w3_sr0.9_no_ceres.log` |
| sleeping_1 | 6 | 2 | - | 1 | 0.019004 | 0.014354 | 0.286360 | 3 | `Python/logs/sintel_xm_no_ceres/solver/sleeping_1_setting6_mode2_w1_no_ceres.log` |
| sleeping_2 | 1 | 1 | 0.1 | 1 | 0.002108 | 0.001641 | 0.035872 | 3 | `Python/logs/sintel_xm_no_ceres/solver/sleeping_2_setting1_mode1_w1_sr0.1_no_ceres.log` |
| sleeping_2 | 2 | 1 | 0.3 | 1 | 0.002091 | 0.001584 | 0.035757 | 3 | `Python/logs/sintel_xm_no_ceres/solver/sleeping_2_setting2_mode1_w1_sr0.3_no_ceres.log` |
| sleeping_2 | 3 | 1 | 0.5 | 1 | 0.002077 | 0.001586 | 0.035719 | 3 | `Python/logs/sintel_xm_no_ceres/solver/sleeping_2_setting3_mode1_w1_sr0.5_no_ceres.log` |
| sleeping_2 | 4 | 1 | 0.9 | 2 | 0.018588 | 0.004940 | 0.128000 | 3 | `Python/logs/sintel_xm_no_ceres/solver/sleeping_2_setting4_mode1_w2_sr0.9_no_ceres.log` |
| sleeping_2 | 5 | 1 | 0.9 | 3 | 0.005091 | 0.002159 | 0.045774 | 3 | `Python/logs/sintel_xm_no_ceres/solver/sleeping_2_setting5_mode1_w3_sr0.9_no_ceres.log` |
| sleeping_2 | 6 | 2 | - | 1 | 0.004083 | 0.003083 | 0.050009 | 3 | `Python/logs/sintel_xm_no_ceres/solver/sleeping_2_setting6_mode2_w1_no_ceres.log` |
| temple_2 | 1 | 1 | 0.1 | 1 | 0.428868 | 0.359681 | 0.305804 | 3 | `Python/logs/sintel_xm_no_ceres/solver/temple_2_setting1_mode1_w1_sr0.1_no_ceres.log` |
| temple_2 | 2 | 1 | 0.3 | 1 | 0.403933 | 0.372819 | 0.295077 | 3 | `Python/logs/sintel_xm_no_ceres/solver/temple_2_setting2_mode1_w1_sr0.3_no_ceres.log` |
| temple_2 | 3 | 1 | 0.5 | 1 | 0.405060 | 0.370867 | 0.291799 | 3 | `Python/logs/sintel_xm_no_ceres/solver/temple_2_setting3_mode1_w1_sr0.5_no_ceres.log` |
| temple_2 | 4 | 1 | 0.9 | 2 | 0.233851 | 0.055744 | 0.334286 | 3 | `Python/logs/sintel_xm_no_ceres/solver/temple_2_setting4_mode1_w2_sr0.9_no_ceres.log` |
| temple_2 | 5 | 1 | 0.9 | 3 | 0.281362 | 0.140569 | 0.285173 | 3 | `Python/logs/sintel_xm_no_ceres/solver/temple_2_setting5_mode1_w3_sr0.9_no_ceres.log` |
| temple_2 | 6 | 2 | - | 1 | 0.845172 | 0.874008 | 1.670702 | 3 | `Python/logs/sintel_xm_no_ceres/solver/temple_2_setting6_mode2_w1_no_ceres.log` |
| temple_3 | 1 | 1 | 0.1 | 1 | 0.205930 | 0.043009 | 0.654290 | 3 | `Python/logs/sintel_xm_no_ceres/solver/temple_3_setting1_mode1_w1_sr0.1_no_ceres.log` |
| temple_3 | 2 | 1 | 0.3 | 1 | 0.243422 | 0.042363 | 0.615915 | 3 | `Python/logs/sintel_xm_no_ceres/solver/temple_3_setting2_mode1_w1_sr0.3_no_ceres.log` |
| temple_3 | 3 | 1 | 0.5 | 1 | 0.215159 | 0.043699 | 0.465953 | 3 | `Python/logs/sintel_xm_no_ceres/solver/temple_3_setting3_mode1_w1_sr0.5_no_ceres.log` |
| temple_3 | 4 | 1 | 0.9 | 2 | 0.255656 | 0.058896 | 0.156778 | 3 | `Python/logs/sintel_xm_no_ceres/solver/temple_3_setting4_mode1_w2_sr0.9_no_ceres.log` |
| temple_3 | 5 | 1 | 0.9 | 3 | 0.238988 | 0.057623 | 0.167400 | 3 | `Python/logs/sintel_xm_no_ceres/solver/temple_3_setting5_mode1_w3_sr0.9_no_ceres.log` |
| temple_3 | 6 | 2 | - | 1 | 0.251821 | 0.179405 | 0.692113 | 3 | `Python/logs/sintel_xm_no_ceres/solver/temple_3_setting6_mode2_w1_no_ceres.log` |
