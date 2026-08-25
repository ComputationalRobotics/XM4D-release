# Sintel XM GT-Depth Results (setting 4)

Generated: `2026-04-22T21:22:52-04:00`

Pipeline:

- Dataset root: `datasets/Sintel`
- Missing GT cameras are loaded from `datasets/MPI-Sintel-depth-training-20150305/training/camdata_left`
- GT depth is taken from `datasets/MPI-Sintel-depth-training-20150305/training/depth/<scene>` when no local `depths/` folder exists
- TAP tracking is reused from `outputs/tracks_debug.npz`; `output_tapnet_gt.npz` is rebuilt without rerunning tracking
- Solver heuristic is setting 4: `mode=1`, `sphere_ratio=0.9`, `WEIGHT_MODE=2`
- Refinement: `with Ceres refinement`

## Preparation

| Scene | Frames | Camera source | GT-depth conversion | Conversion log |
| --- | ---: | --- | --- | --- |
| alley_1 | 50 | local | rerun | `Python/logs/sintel_gt_depth_setting4/prep/alley_1_gt_convert.log` |
| alley_2 | 50 | local | rerun | `Python/logs/sintel_gt_depth_setting4/prep/alley_2_gt_convert.log` |
| ambush_2 | 21 | local | rerun | `Python/logs/sintel_gt_depth_setting4/prep/ambush_2_gt_convert.log` |
| ambush_4 | 33 | local | rerun | `Python/logs/sintel_gt_depth_setting4/prep/ambush_4_gt_convert.log` |
| ambush_5 | 50 | local | rerun | `Python/logs/sintel_gt_depth_setting4/prep/ambush_5_gt_convert.log` |
| ambush_6 | 20 | local | rerun | `Python/logs/sintel_gt_depth_setting4/prep/ambush_6_gt_convert.log` |
| ambush_7 | 50 | local | rerun | `Python/logs/sintel_gt_depth_setting4/prep/ambush_7_gt_convert.log` |
| bamboo_1 | 50 | local | rerun | `Python/logs/sintel_gt_depth_setting4/prep/bamboo_1_gt_convert.log` |
| bamboo_2 | 50 | local | rerun | `Python/logs/sintel_gt_depth_setting4/prep/bamboo_2_gt_convert.log` |
| bandage_1 | 50 | local | rerun | `Python/logs/sintel_gt_depth_setting4/prep/bandage_1_gt_convert.log` |
| bandage_2 | 50 | local | rerun | `Python/logs/sintel_gt_depth_setting4/prep/bandage_2_gt_convert.log` |
| cave_2 | 50 | local | rerun | `Python/logs/sintel_gt_depth_setting4/prep/cave_2_gt_convert.log` |
| cave_4 | 50 | local | rerun | `Python/logs/sintel_gt_depth_setting4/prep/cave_4_gt_convert.log` |
| market_2 | 50 | local | rerun | `Python/logs/sintel_gt_depth_setting4/prep/market_2_gt_convert.log` |
| market_5 | 50 | local | rerun | `Python/logs/sintel_gt_depth_setting4/prep/market_5_gt_convert.log` |
| market_6 | 40 | mpi_fallback | rerun | `Python/logs/sintel_gt_depth_setting4/prep/market_6_gt_convert.log` |
| mountain_1 | 50 | mpi_fallback | rerun | `Python/logs/sintel_gt_depth_setting4/prep/mountain_1_gt_convert.log` |
| shaman_2 | 50 | mpi_fallback | generated | `Python/logs/sintel_gt_depth_setting4/prep/shaman_2_gt_convert.log` |
| shaman_3 | 50 | mpi_fallback | generated | `Python/logs/sintel_gt_depth_setting4/prep/shaman_3_gt_convert.log` |
| sleeping_1 | 50 | mpi_fallback | generated | `Python/logs/sintel_gt_depth_setting4/prep/sleeping_1_gt_convert.log` |
| sleeping_2 | 50 | mpi_fallback | generated | `Python/logs/sintel_gt_depth_setting4/prep/sleeping_2_gt_convert.log` |
| temple_2 | 50 | local | generated | `Python/logs/sintel_gt_depth_setting4/prep/temple_2_gt_convert.log` |
| temple_3 | 50 | mpi_fallback | generated | `Python/logs/sintel_gt_depth_setting4/prep/temple_3_gt_convert.log` |

## Solver Results

| Scene | Setting | Mode | Sphere Ratio | WEIGHT_MODE | Ceres refine | RPE-R | RPE-T | ATE-R | ATE-T | Total time (s) | Solve time (s) | Optimal rank | Log |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| alley_1 | 4 | 1 | 0.9 | 2 | yes | 0.0002617151142515 | 0.0009053228720747 | 2.1885045285597431 | 0.0006366804874139 | 4.861517 | 0.403985 | 3 | `Python/logs/sintel_gt_depth_setting4/solver/alley_1_setting4_mode1_w2_sr0.9_ceres.log` |
| alley_2 | 4 | 1 | 0.9 | 2 | yes | 0.0008259225164853 | 0.0018980418436470 | 0.7559112118592003 | 0.0013972444244112 | 6.708262 | 1.019524 | 3 | `Python/logs/sintel_gt_depth_setting4/solver/alley_2_setting4_mode1_w2_sr0.9_ceres.log` |
| ambush_2 | 4 | 1 | 0.9 | 2 | yes | 0.0490228887122379 | 0.0000055585308283 | 2.2031103768533158 | 0.0000036965069044 | 71.295871 | 65.467051 | 3 | `Python/logs/sintel_gt_depth_setting4/solver/ambush_2_setting4_mode1_w2_sr0.9_ceres.log` |
| ambush_4 | 4 | 1 | 0.9 | 2 | yes | 0.0090397371385959 | 0.0742458627997198 | 0.8966903831252101 | 0.0489953935418241 | 21.577601 | 0.789261 | 3 | `Python/logs/sintel_gt_depth_setting4/solver/ambush_4_setting4_mode1_w2_sr0.9_ceres.log` |
| ambush_5 | 4 | 1 | 0.9 | 2 | yes | 0.0037408640041119 | 0.0083331135136849 | 1.3023561086855171 | 0.0055907726470631 | 8.962051 | 2.448050 | 3 | `Python/logs/sintel_gt_depth_setting4/solver/ambush_5_setting4_mode1_w2_sr0.9_ceres.log` |
| ambush_6 | 4 | 1 | 0.9 | 2 | yes | 0.0174641206283499 | 0.0385853284069641 | 0.2374374868570314 | 0.0283127872484961 | 6.961664 | 0.440061 | 3 | `Python/logs/sintel_gt_depth_setting4/solver/ambush_6_setting4_mode1_w2_sr0.9_ceres.log` |
| ambush_7 | 4 | 1 | 0.9 | 2 | yes | 0.0079128673623600 | 0.0000053620436756 | 2.2516606824949781 | 0.0000038566587171 | 8.375190 | 0.424637 | 3 | `Python/logs/sintel_gt_depth_setting4/solver/ambush_7_setting4_mode1_w2_sr0.9_ceres.log` |
| bamboo_1 | 4 | 1 | 0.9 | 2 | yes | 0.0026904902723643 | 0.0011606966312603 | 2.3276139838261485 | 0.0008041570434337 | 5.827273 | 0.363774 | 3 | `Python/logs/sintel_gt_depth_setting4/solver/bamboo_1_setting4_mode1_w2_sr0.9_ceres.log` |
| bamboo_2 | 4 | 1 | 0.9 | 2 | yes | 0.0018628712482237 | 0.0012082264015774 | 0.5248349769065819 | 0.0008375934706839 | 5.109662 | 0.453483 | 3 | `Python/logs/sintel_gt_depth_setting4/solver/bamboo_2_setting4_mode1_w2_sr0.9_ceres.log` |
| bandage_1 | 4 | 1 | 0.9 | 2 | yes | 0.3791467375023018 | 0.0416354900760332 | 2.0871091910355783 | 0.0315225293034209 | 4.236790 | 0.436831 | 3 | `Python/logs/sintel_gt_depth_setting4/solver/bandage_1_setting4_mode1_w2_sr0.9_ceres.log` |
| bandage_2 | 4 | 1 | 0.9 | 2 | yes | 0.0008372883799399 | 0.0015425156657067 | 3.0588776251623289 | 0.0012252813351172 | 3.891455 | 0.368804 | 3 | `Python/logs/sintel_gt_depth_setting4/solver/bandage_2_setting4_mode1_w2_sr0.9_ceres.log` |
| cave_2 | 4 | 1 | 0.9 | 2 | yes | 0.0402939988347959 | 0.5591276618323054 | 3.1064096146743059 | 0.4590241087380568 | 43.886283 | 28.353288 | 3 | `Python/logs/sintel_gt_depth_setting4/solver/cave_2_setting4_mode1_w2_sr0.9_ceres.log` |
| cave_4 | 4 | 1 | 0.9 | 2 | yes | 0.0067579230304399 | 0.0369920213595404 | 0.2016170440286620 | 0.0275260130909333 | 13.546482 | 0.647126 | 3 | `Python/logs/sintel_gt_depth_setting4/solver/cave_4_setting4_mode1_w2_sr0.9_ceres.log` |
| market_2 | 4 | 1 | 0.9 | 2 | yes | 2.0877603099810869 | 0.0959074285188661 | 2.1773261508150679 | 0.0657304308316160 | 68.391538 | 62.025876 | 3 | `Python/logs/sintel_gt_depth_setting4/solver/market_2_setting4_mode1_w2_sr0.9_ceres.log` |
| market_5 | 4 | 1 | 0.9 | 2 | yes | 0.0740376349868800 | 0.0178190074200200 | 2.0734284334171118 | 0.0136659119946396 | 8.763756 | 1.721938 | 3 | `Python/logs/sintel_gt_depth_setting4/solver/market_5_setting4_mode1_w2_sr0.9_ceres.log` |
| market_6 | 4 | 1 | 0.9 | 2 | yes | 0.0195851790593754 | 0.0288622453412561 | 2.0917484681316934 | 0.0203510824743902 | 4.996377 | 0.656278 | 3 | `Python/logs/sintel_gt_depth_setting4/solver/market_6_setting4_mode1_w2_sr0.9_ceres.log` |
| mountain_1 | 4 | 1 | 0.9 | 2 | yes | 0.0028107533307638 | 0.1553885121509358 | 0.2403075700886799 | 0.1084606018498063 | 9.762219 | 0.608242 | 3 | `Python/logs/sintel_gt_depth_setting4/solver/mountain_1_setting4_mode1_w2_sr0.9_ceres.log` |
| shaman_2 | 4 | 1 | 0.9 | 2 | yes | 0.0289277339938511 | 0.0051141364117334 | 2.2368994420278550 | 0.0039552468539131 | 5.120602 | 0.392209 | 3 | `Python/logs/sintel_gt_depth_setting4/solver/shaman_2_setting4_mode1_w2_sr0.9_ceres.log` |
| shaman_3 | 4 | 1 | 0.9 | 2 | yes | 0.0117600925127745 | 0.0045141376020467 | 2.0692862179631755 | 0.0033484173115847 | 4.035133 | 0.455358 | 3 | `Python/logs/sintel_gt_depth_setting4/solver/shaman_3_setting4_mode1_w2_sr0.9_ceres.log` |
| sleeping_1 | 4 | 1 | 0.9 | 2 | yes | 0.0492896306638159 | 0.0046698964582336 | 2.5833839257294517 | 0.0033011563044236 | 4.069287 | 0.405552 | 3 | `Python/logs/sintel_gt_depth_setting4/solver/sleeping_1_setting4_mode1_w2_sr0.9_ceres.log` |
| sleeping_2 | 4 | 1 | 0.9 | 2 | yes | 0.0003875466019982 | 0.0006647731963097 | 0.0517225129969734 | 0.0004529788522055 | 5.660288 | 0.392975 | 3 | `Python/logs/sintel_gt_depth_setting4/solver/sleeping_2_setting4_mode1_w2_sr0.9_ceres.log` |
| temple_2 | 4 | 1 | 0.9 | 2 | yes | 0.0007299260406288 | 0.0015678779463898 | 0.0119253254113385 | 0.0011148340998619 | 11.079789 | 2.223627 | 3 | `Python/logs/sintel_gt_depth_setting4/solver/temple_2_setting4_mode1_w2_sr0.9_ceres.log` |
| temple_3 | 4 | 1 | 0.9 | 2 | yes | 0.0018779062271850 | 0.1056344019241888 | 3.1194602938946208 | 0.0768830291485656 | 17.400551 | 2.305327 | 3 | `Python/logs/sintel_gt_depth_setting4/solver/temple_3_setting4_mode1_w2_sr0.9_ceres.log` |
