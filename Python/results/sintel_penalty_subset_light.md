# Sintel XM Experiment Results (with Ceres refinement)

Generated: `2026-04-22T17:24:36-04:00`

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
| alley_1 | 1 | 1 | 0.1 | 1 | yes | 0.0067551240437117 | 0.0059131370132957 | 0.2862881828152423 | 0.0042806163874756 | 24.605521 | 0.278596 | 3 | `Python/logs/sintel_penalty_subset_light/solver/alley_1_setting1_mode1_w1_sr0.1_ceres.log` |
| alley_1 | 2 | 1 | 0.3 | 1 | yes | 0.0060385516819062 | 0.0139805056122640 | 0.9466707452744193 | 0.0098512487245757 | 10.281333 | 0.323569 | 3 | `Python/logs/sintel_penalty_subset_light/solver/alley_1_setting2_mode1_w1_sr0.3_ceres.log` |
| alley_1 | 3 | 1 | 0.5 | 1 | yes | 0.0062212941501605 | 0.0088726348686423 | 0.5064952731026946 | 0.0061674015109739 | 15.010445 | 0.354371 | 3 | `Python/logs/sintel_penalty_subset_light/solver/alley_1_setting3_mode1_w1_sr0.5_ceres.log` |
| alley_1 | 4 | 1 | 0.9 | 2 | yes | 0.0068116122293768 | 0.0076529828323706 | 0.2973194151846434 | 0.0056635089406894 | 8.840000 | 0.423036 | 3 | `Python/logs/sintel_penalty_subset_light/solver/alley_1_setting4_mode1_w2_sr0.9_ceres.log` |
| alley_1 | 5 | 1 | 0.9 | 3 | yes | 0.0082862720669099 | 0.0065411487103537 | 3.1246418260536171 | 0.0044688001815765 | 14.317995 | 0.490597 | 3 | `Python/logs/sintel_penalty_subset_light/solver/alley_1_setting5_mode1_w3_sr0.9_ceres.log` |
| alley_1 | 6 | 2 | - | 1 | yes | 0.0034765766833750 | 0.0258362821297600 | 3.0678954150862046 | 0.0172931351004991 | 25.476300 | 0.249690 | 3 | `Python/logs/sintel_penalty_subset_light/solver/alley_1_setting6_mode2_w1_ceres.log` |
| cave_2 | 1 | 1 | 0.1 | 1 | yes | 1.6498470522367275 | 0.5689530429314007 | 2.1815710646386992 | 0.4553886944452100 | 26.946044 | 13.761581 | 3 | `Python/logs/sintel_penalty_subset_light/solver/cave_2_setting1_mode1_w1_sr0.1_ceres.log` |
| cave_2 | 2 | 1 | 0.3 | 1 | yes | 1.3965988799624103 | 0.7225790543225709 | 2.4739936379269416 | 0.6984474127114870 | 25.461216 | 9.581700 | 3 | `Python/logs/sintel_penalty_subset_light/solver/cave_2_setting2_mode1_w1_sr0.3_ceres.log` |
| cave_2 | 3 | 1 | 0.5 | 1 | yes | 1.3694069208478628 | 0.6570463469566592 | 2.8571681360848449 | 0.5117051573355256 | 22.301194 | 7.490664 | 3 | `Python/logs/sintel_penalty_subset_light/solver/cave_2_setting3_mode1_w1_sr0.5_ceres.log` |
| cave_2 | 4 | 1 | 0.9 | 2 | yes | 0.4917824956772512 | 0.4524946275792257 | 2.9571552279248032 | 0.3487976591813872 | 75.441440 | 2.599431 | 3 | `Python/logs/sintel_penalty_subset_light/solver/cave_2_setting4_mode1_w2_sr0.9_ceres.log` |
| cave_2 | 5 | 1 | 0.9 | 3 | yes | 0.4993350318768514 | 0.4749306933596690 | 2.8072705351782923 | 0.3962817837763864 | 29.568404 | 2.462429 | 3 | `Python/logs/sintel_penalty_subset_light/solver/cave_2_setting5_mode1_w3_sr0.9_ceres.log` |
| cave_2 | 6 | 2 | - | 1 | yes | 0.9407968518032604 | 0.6242788457126868 | 2.2380538645925658 | 0.4499477237722817 | 15.388818 | 2.528345 | 3 | `Python/logs/sintel_penalty_subset_light/solver/cave_2_setting6_mode2_w1_ceres.log` |
| market_6 | 1 | 1 | 0.1 | 1 | yes | 0.0186278532493745 | 0.0204859557721113 | 2.1459318382443877 | 0.0149362128407085 | 12.022799 | 0.310827 | 3 | `Python/logs/sintel_penalty_subset_light/solver/market_6_setting1_mode1_w1_sr0.1_ceres.log` |
| market_6 | 2 | 1 | 0.3 | 1 | yes | 0.0152165169163316 | 0.0191096743489852 | 2.1799071318721581 | 0.0138616704741717 | 8.596615 | 0.360660 | 3 | `Python/logs/sintel_penalty_subset_light/solver/market_6_setting2_mode1_w1_sr0.3_ceres.log` |
| market_6 | 3 | 1 | 0.5 | 1 | yes | 0.0148985994011653 | 0.0168670641237158 | 2.1631561295197810 | 0.0122583930188653 | 64.720729 | 0.388423 | 3 | `Python/logs/sintel_penalty_subset_light/solver/market_6_setting3_mode1_w1_sr0.5_ceres.log` |
| market_6 | 4 | 1 | 0.9 | 2 | yes | 0.0187840544382241 | 0.0217538387738899 | 2.1088823889656112 | 0.0153703648022502 | 7.462233 | 0.509800 | 3 | `Python/logs/sintel_penalty_subset_light/solver/market_6_setting4_mode1_w2_sr0.9_ceres.log` |
| market_6 | 5 | 1 | 0.9 | 3 | yes | 0.0152274554428860 | 0.0182086047105510 | 2.1823432480220379 | 0.0132070509353951 | 64.474153 | 0.465628 | 3 | `Python/logs/sintel_penalty_subset_light/solver/market_6_setting5_mode1_w3_sr0.9_ceres.log` |
| market_6 | 6 | 2 | - | 1 | yes | 0.0202285597523819 | 0.0247987425027149 | 2.1236347294950475 | 0.0183979612633365 | 7.061972 | 0.304329 | 3 | `Python/logs/sintel_penalty_subset_light/solver/market_6_setting6_mode2_w1_ceres.log` |
| shaman_3 | 1 | 1 | 0.1 | 1 | yes | 0.0111911404899951 | 0.0042730468161835 | 1.0004916945136113 | 0.0031878645119578 | 5.218393 | 0.351390 | 3 | `Python/logs/sintel_penalty_subset_light/solver/shaman_3_setting1_mode1_w1_sr0.1_ceres.log` |
| shaman_3 | 2 | 1 | 0.3 | 1 | yes | 0.0111367689343811 | 0.0042751335416739 | 0.9500355395009608 | 0.0031883394923479 | 6.511387 | 1.644697 | 3 | `Python/logs/sintel_penalty_subset_light/solver/shaman_3_setting2_mode1_w1_sr0.3_ceres.log` |
| shaman_3 | 3 | 1 | 0.5 | 1 | yes | 0.0112136935162345 | 0.0042594353533286 | 0.9623436601421553 | 0.0031778815291358 | 7.105937 | 2.022033 | 3 | `Python/logs/sintel_penalty_subset_light/solver/shaman_3_setting3_mode1_w1_sr0.5_ceres.log` |
| shaman_3 | 4 | 1 | 0.9 | 2 | yes | 0.0109782624161296 | 0.0045771755900488 | 0.1945201093834782 | 0.0034608889302402 | 12.352521 | 3.057092 | 3 | `Python/logs/sintel_penalty_subset_light/solver/shaman_3_setting4_mode1_w2_sr0.9_ceres.log` |
| shaman_3 | 5 | 1 | 0.9 | 3 | yes | 0.0111112481688768 | 0.0048747743484247 | 1.2580064368697783 | 0.0036390006069381 | 7.754726 | 2.624714 | 3 | `Python/logs/sintel_penalty_subset_light/solver/shaman_3_setting5_mode1_w3_sr0.9_ceres.log` |
| shaman_3 | 6 | 2 | - | 1 | yes | 0.0109554800709577 | 0.0027160435722957 | 1.9469311243425924 | 0.0019180728266358 | 7.237023 | 0.257823 | 3 | `Python/logs/sintel_penalty_subset_light/solver/shaman_3_setting6_mode2_w1_ceres.log` |
| temple_3 | 1 | 1 | 0.1 | 1 | yes | 0.0063059769637938 | 0.3271769841772024 | 2.6669932828126317 | 0.2374629974974553 | 13.657031 | 1.075937 | 3 | `Python/logs/sintel_penalty_subset_light/solver/temple_3_setting1_mode1_w1_sr0.1_ceres.log` |
| temple_3 | 2 | 1 | 0.3 | 1 | yes | 0.0026489126919307 | 0.2673165437216479 | 2.6643019095003266 | 0.1912504241917058 | 12.635336 | 1.628300 | 3 | `Python/logs/sintel_penalty_subset_light/solver/temple_3_setting2_mode1_w1_sr0.3_ceres.log` |
| temple_3 | 3 | 1 | 0.5 | 1 | yes | 0.0017801795840017 | 0.2481306380469955 | 0.9689471993065929 | 0.1720216698043719 | 21.771061 | 2.018805 | 3 | `Python/logs/sintel_penalty_subset_light/solver/temple_3_setting3_mode1_w1_sr0.5_ceres.log` |
| temple_3 | 4 | 1 | 0.9 | 2 | yes | 0.0033828088863858 | 0.1732159746522494 | 1.9424634666581300 | 0.1180304483694221 | 20.349426 | 1.254262 | 3 | `Python/logs/sintel_penalty_subset_light/solver/temple_3_setting4_mode1_w2_sr0.9_ceres.log` |
| temple_3 | 5 | 1 | 0.9 | 3 | yes | 0.0030068985685867 | 0.2828890906614699 | 0.7622786479701469 | 0.1954126398059899 | 23.713960 | 1.432871 | 3 | `Python/logs/sintel_penalty_subset_light/solver/temple_3_setting5_mode1_w3_sr0.9_ceres.log` |
| temple_3 | 6 | 2 | - | 1 | yes | 0.0052583860745456 | 0.4071706801092857 | 2.4702699823359007 | 0.2945673075518422 | 13.015228 | 0.557650 | 3 | `Python/logs/sintel_penalty_subset_light/solver/temple_3_setting6_mode2_w1_ceres.log` |
