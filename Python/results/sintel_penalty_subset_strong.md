# Sintel XM Experiment Results (with Ceres refinement)

Generated: `2026-04-22T17:37:02-04:00`

Pipeline:

- Dataset root: `datasets/Sintel`
- Missing GT cameras are loaded from `datasets/MPI-Sintel-depth-training-20150305/training/camdata_left`
- Missing DA3 depth is generated with `Python/get_depths_clean.py` in the `da3` environment
- Missing `output_tapnet_da3.npz` is generated with `Python/static_tracking_online.py` in the `tapnet` environment
- Solver sweep uses the same 6 heuristics as the BAL experiment, with no added noise
- Refinement: `with Ceres refinement`

## Preparation

| Scene | Frames | Camera source | DA3 depth | Tracking | Depth log | Tracking log |
| --- | ---: | --- | --- | --- | --- | --- |
| alley_1 | 50 | local | reused | reused | - | - |
| cave_2 | 50 | local | reused | reused | - | - |
| market_6 | 40 | mpi_fallback | reused | reused | - | - |
| shaman_3 | 50 | mpi_fallback | reused | reused | - | - |
| temple_3 | 50 | mpi_fallback | reused | reused | - | - |

## Solver Results

| Scene | Setting | Mode | Sphere Ratio | WEIGHT_MODE | Ceres refine | RPE-R | RPE-T | ATE-R | ATE-T | Total time (s) | Solve time (s) | Optimal rank | Log |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| alley_1 | 1 | 1 | 0.1 | 1 | yes | 0.0249189017810479 | 0.0035355713046854 | 3.1170061791916419 | 0.0025593090698046 | 13.291788 | 0.314116 | 3 | `Python/logs/sintel_penalty_subset_strong/solver/alley_1_setting1_mode1_w1_sr0.1_ceres.log` |
| alley_1 | 2 | 1 | 0.3 | 1 | yes | 0.0258899197857223 | 0.0030760036967413 | 3.1065102693025710 | 0.0022140563110753 | 11.851224 | 0.357427 | 3 | `Python/logs/sintel_penalty_subset_strong/solver/alley_1_setting2_mode1_w1_sr0.3_ceres.log` |
| alley_1 | 3 | 1 | 0.5 | 1 | yes | 0.0259486216321155 | 0.0030648846424706 | 3.0978651966728945 | 0.0022046893281008 | 12.475181 | 0.409811 | 3 | `Python/logs/sintel_penalty_subset_strong/solver/alley_1_setting3_mode1_w1_sr0.5_ceres.log` |
| alley_1 | 4 | 1 | 0.9 | 2 | yes | 0.0066869565864752 | 0.0075375900145275 | 2.5202382619393267 | 0.0055895555135130 | 28.176022 | 0.389196 | 3 | `Python/logs/sintel_penalty_subset_strong/solver/alley_1_setting4_mode1_w2_sr0.9_ceres.log` |
| alley_1 | 5 | 1 | 0.9 | 3 | yes | 0.0248585741432508 | 0.0035612668459638 | 2.9859867325521789 | 0.0025728555949315 | 65.988994 | 0.484440 | 3 | `Python/logs/sintel_penalty_subset_strong/solver/alley_1_setting5_mode1_w3_sr0.9_ceres.log` |
| alley_1 | 6 | 2 | - | 1 | yes | 0.0245244660582118 | 0.0037430638856782 | 2.9818925601669175 | 0.0026292422087331 | 16.729451 | 0.246852 | 3 | `Python/logs/sintel_penalty_subset_strong/solver/alley_1_setting6_mode2_w1_ceres.log` |
| cave_2 | 1 | 1 | 0.1 | 1 | yes | 1.2147497020726024 | 0.5748390486758417 | 2.9259925590794431 | 0.6208879042074247 | 22.551397 | 8.590294 | 3 | `Python/logs/sintel_penalty_subset_strong/solver/cave_2_setting1_mode1_w1_sr0.1_ceres.log` |
| cave_2 | 2 | 1 | 0.3 | 1 | yes | 1.3032705431327032 | 0.6270420217929835 | 2.8177793964118707 | 0.6200013513907264 | 42.152382 | 28.112023 | 3 | `Python/logs/sintel_penalty_subset_strong/solver/cave_2_setting2_mode1_w1_sr0.3_ceres.log` |
| cave_2 | 3 | 1 | 0.5 | 1 | yes | 0.9178424803219715 | 0.4696160422322676 | 2.8808384439698269 | 0.3853080539828869 | 22.361081 | 8.383776 | 3 | `Python/logs/sintel_penalty_subset_strong/solver/cave_2_setting3_mode1_w1_sr0.5_ceres.log` |
| cave_2 | 4 | 1 | 0.9 | 2 | yes | 0.4767554151357388 | 0.6578441363251047 | 3.0708245672351375 | 0.4703626084921589 | 72.531936 | 2.804391 | 3 | `Python/logs/sintel_penalty_subset_strong/solver/cave_2_setting4_mode1_w2_sr0.9_ceres.log` |
| cave_2 | 5 | 1 | 0.9 | 3 | yes | 0.5157861013705561 | 0.3190551254680556 | 2.7154626520105309 | 0.2276069864447137 | 26.430756 | 2.295677 | 3 | `Python/logs/sintel_penalty_subset_strong/solver/cave_2_setting5_mode1_w3_sr0.9_ceres.log` |
| cave_2 | 6 | 2 | - | 1 | yes | 0.9280234887714266 | 0.7618065782476930 | 2.7657512286676735 | 0.5320434654240964 | 13.639411 | 1.377689 | 3 | `Python/logs/sintel_penalty_subset_strong/solver/cave_2_setting6_mode2_w1_ceres.log` |
| market_6 | 1 | 1 | 0.1 | 1 | yes | 0.0176685583320691 | 0.0181243363764427 | 2.1099220911451981 | 0.0136224917206328 | 8.009181 | 0.327982 | 3 | `Python/logs/sintel_penalty_subset_strong/solver/market_6_setting1_mode1_w1_sr0.1_ceres.log` |
| market_6 | 2 | 1 | 0.3 | 1 | yes | 0.0096988592912040 | 0.0156140001801111 | 1.8111272047587914 | 0.0112965873971817 | 4.983705 | 0.363229 | 3 | `Python/logs/sintel_penalty_subset_strong/solver/market_6_setting2_mode1_w1_sr0.3_ceres.log` |
| market_6 | 3 | 1 | 0.5 | 1 | yes | 0.0160129690011100 | 0.0151307041613676 | 2.1709652950959026 | 0.0115087232547427 | 7.539986 | 0.390903 | 3 | `Python/logs/sintel_penalty_subset_strong/solver/market_6_setting3_mode1_w1_sr0.5_ceres.log` |
| market_6 | 4 | 1 | 0.9 | 2 | yes | 0.0173380941265777 | 0.0221155877500451 | 2.2331555129904359 | 0.0161868465988016 | 7.837170 | 0.478878 | 3 | `Python/logs/sintel_penalty_subset_strong/solver/market_6_setting4_mode1_w2_sr0.9_ceres.log` |
| market_6 | 5 | 1 | 0.9 | 3 | yes | 0.0168078424451811 | 0.0148975953757688 | 2.2515903925950314 | 0.0111421447356602 | 24.752986 | 0.471680 | 3 | `Python/logs/sintel_penalty_subset_strong/solver/market_6_setting5_mode1_w3_sr0.9_ceres.log` |
| market_6 | 6 | 2 | - | 1 | yes | 0.0136093662140936 | 0.0307804706902128 | 1.9773489738410706 | 0.0214819034519911 | 5.001442 | 0.255255 | 3 | `Python/logs/sintel_penalty_subset_strong/solver/market_6_setting6_mode2_w1_ceres.log` |
| shaman_3 | 1 | 1 | 0.1 | 1 | yes | 0.0111277031698550 | 0.0035466288940747 | 0.6685297722505590 | 0.0026987661672299 | 13.330840 | 0.354269 | 3 | `Python/logs/sintel_penalty_subset_strong/solver/shaman_3_setting1_mode1_w1_sr0.1_ceres.log` |
| shaman_3 | 2 | 1 | 0.3 | 1 | yes | 0.0112815540648864 | 0.0035053652290234 | 0.7198405744645955 | 0.0026632870513124 | 13.676083 | 0.436814 | 3 | `Python/logs/sintel_penalty_subset_strong/solver/shaman_3_setting2_mode1_w1_sr0.3_ceres.log` |
| shaman_3 | 3 | 1 | 0.5 | 1 | yes | 0.0113698594270560 | 0.0035273027667487 | 0.7544441699840031 | 0.0026805822294276 | 14.633818 | 0.471116 | 3 | `Python/logs/sintel_penalty_subset_strong/solver/shaman_3_setting3_mode1_w1_sr0.5_ceres.log` |
| shaman_3 | 4 | 1 | 0.9 | 2 | yes | 0.0089144901063481 | 0.0042038155629597 | 1.1385792229198382 | 0.0029987652339148 | 5.368636 | 0.638774 | 3 | `Python/logs/sintel_penalty_subset_strong/solver/shaman_3_setting4_mode1_w2_sr0.9_ceres.log` |
| shaman_3 | 5 | 1 | 0.9 | 3 | yes | 0.0112549633064147 | 0.0056105204314730 | 1.3555606650227454 | 0.0040500437452210 | 29.271025 | 0.590123 | 3 | `Python/logs/sintel_penalty_subset_strong/solver/shaman_3_setting5_mode1_w3_sr0.9_ceres.log` |
| shaman_3 | 6 | 2 | - | 1 | yes | 0.0120430223373451 | 0.0036279504110122 | 1.2609971903528916 | 0.0027527147046378 | 64.299721 | 0.253316 | 3 | `Python/logs/sintel_penalty_subset_strong/solver/shaman_3_setting6_mode2_w1_ceres.log` |
| temple_3 | 1 | 1 | 0.1 | 1 | yes | 0.0020990340004096 | 0.2251640770371710 | 0.7727476535878277 | 0.1543089764093791 | 10.371248 | 1.201228 | 3 | `Python/logs/sintel_penalty_subset_strong/solver/temple_3_setting1_mode1_w1_sr0.1_ceres.log` |
| temple_3 | 2 | 1 | 0.3 | 1 | yes | 0.0027553732628150 | 0.2668751072618006 | 1.6231036873465863 | 0.1790792410330083 | 8.838592 | 1.528835 | 3 | `Python/logs/sintel_penalty_subset_strong/solver/temple_3_setting2_mode1_w1_sr0.3_ceres.log` |
| temple_3 | 3 | 1 | 0.5 | 1 | yes | 0.0017935060029918 | 0.2811648242243088 | 0.6747631369194668 | 0.1932750771813937 | 30.958766 | 1.808604 | 3 | `Python/logs/sintel_penalty_subset_strong/solver/temple_3_setting3_mode1_w1_sr0.5_ceres.log` |
| temple_3 | 4 | 1 | 0.9 | 2 | yes | 0.0024381442618738 | 0.2855089001281117 | 1.6080103964100110 | 0.1953095777659020 | 67.100585 | 1.387569 | 3 | `Python/logs/sintel_penalty_subset_strong/solver/temple_3_setting4_mode1_w2_sr0.9_ceres.log` |
| temple_3 | 5 | 1 | 0.9 | 3 | yes | 0.0048025499680984 | 0.1879627629163390 | 2.4866368078077894 | 0.1314046119812091 | 19.738180 | 1.436340 | 3 | `Python/logs/sintel_penalty_subset_strong/solver/temple_3_setting5_mode1_w3_sr0.9_ceres.log` |
| temple_3 | 6 | 2 | - | 1 | yes | 0.0021241411040749 | 0.2853220521318540 | 1.3443146924955203 | 0.1925039789923529 | 8.143779 | 0.554389 | 3 | `Python/logs/sintel_penalty_subset_strong/solver/temple_3_setting6_mode2_w1_ceres.log` |
