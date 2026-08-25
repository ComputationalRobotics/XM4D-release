# Sintel XM Experiment Results (without Ceres refinement)

Generated: `2026-04-21T22:38:59-04:00`

Pipeline:

- Dataset root: `datasets/Sintel`
- Missing GT cameras are loaded from `datasets/MPI-Sintel-depth-training-20150305/training/camdata_left`
- Missing DA3 depth is generated with `Python/get_depths_clean.py` in the `da3` environment
- Missing `output_tapnet_da3.npz` is generated with `Python/static_tracking_online.py` in the `tapnet` environment
- Solver sweep uses the same 6 heuristics as the BAL experiment, with no added noise
- Refinement: `without Ceres refinement`

## Preparation

| Scene | Frames | Camera source | DA3 depth | Tracking | Depth log | Tracking log |
| --- | ---: | --- | --- | --- | --- | --- |
| alley_1 | 50 | local | reused | reused | - | - |
| alley_2 | 50 | local | reused | reused | - | - |
| ambush_2 | 21 | local | reused | reused | - | - |
| ambush_4 | 33 | local | reused | reused | - | - |
| ambush_5 | 50 | local | reused | reused | - | - |
| ambush_6 | 20 | local | reused | reused | - | - |
| ambush_7 | 50 | local | reused | reused | - | - |
| bamboo_1 | 50 | local | reused | reused | - | - |
| bamboo_2 | 50 | local | reused | reused | - | - |
| bandage_1 | 50 | local | reused | reused | - | - |
| bandage_2 | 50 | local | reused | reused | - | - |
| cave_2 | 50 | local | reused | reused | - | - |
| cave_4 | 50 | local | reused | reused | - | - |
| market_2 | 50 | local | reused | reused | - | - |
| market_5 | 50 | local | reused | reused | - | - |
| market_6 | 40 | mpi_fallback | reused | reused | - | - |
| mountain_1 | 50 | mpi_fallback | reused | reused | - | - |
| shaman_2 | 50 | mpi_fallback | reused | reused | - | - |
| shaman_3 | 50 | mpi_fallback | reused | reused | - | - |
| sleeping_1 | 50 | mpi_fallback | reused | reused | - | - |
| sleeping_2 | 50 | mpi_fallback | reused | reused | - | - |
| temple_2 | 50 | local | reused | reused | - | - |
| temple_3 | 50 | mpi_fallback | reused | reused | - | - |

## Solver Results

| Scene | Setting | Mode | Sphere Ratio | WEIGHT_MODE | Ceres refine | RPE-R | RPE-T | ATE-R | ATE-T | Total time (s) | Solve time (s) | Optimal rank | Log |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| alley_1 | 1 | 1 | 0.1 | 1 | no | 0.0182247281852324 | 0.0114826060020479 | 3.0121610344090932 | 0.0084113482140138 | 2.715956 | 0.293278 | 3 | `Python/logs/sintel_xm_no_ceres/solver/alley_1_setting1_mode1_w1_sr0.1_no_ceres.log` |
| alley_1 | 2 | 1 | 0.3 | 1 | no | 0.0183712987736154 | 0.0111303218463468 | 3.0748573839175481 | 0.0080504826453107 | 2.786558 | 0.357559 | 3 | `Python/logs/sintel_xm_no_ceres/solver/alley_1_setting2_mode1_w1_sr0.3_no_ceres.log` |
| alley_1 | 3 | 1 | 0.5 | 1 | no | 0.0183929900479805 | 0.0110402333861865 | 3.0230722335997644 | 0.0080708674424160 | 2.932589 | 0.393022 | 3 | `Python/logs/sintel_xm_no_ceres/solver/alley_1_setting3_mode1_w1_sr0.5_no_ceres.log` |
| alley_1 | 4 | 1 | 0.9 | 2 | no | 0.0024811797523253 | 0.0089100001720620 | 3.0716307101385341 | 0.0062884963935931 | 2.733985 | 0.405374 | 3 | `Python/logs/sintel_xm_no_ceres/solver/alley_1_setting4_mode1_w2_sr0.9_no_ceres.log` |
| alley_1 | 5 | 1 | 0.9 | 3 | no | 0.0142140104609715 | 0.0102563826069138 | 2.9593616632856810 | 0.0069510821122808 | 3.004368 | 0.480733 | 3 | `Python/logs/sintel_xm_no_ceres/solver/alley_1_setting5_mode1_w3_sr0.9_no_ceres.log` |
| alley_1 | 6 | 2 | - | 1 | no | 0.0160910690072550 | 0.0158965588677989 | 2.9920897826623309 | 0.0108898704561590 | 2.763450 | 0.241812 | 3 | `Python/logs/sintel_xm_no_ceres/solver/alley_1_setting6_mode2_w1_no_ceres.log` |
| alley_2 | 1 | 1 | 0.1 | 1 | no | 0.0833961583445085 | 0.0953320059537775 | 2.0691595101273479 | 0.0900947785235512 | 3.144556 | 0.381117 | 3 | `Python/logs/sintel_xm_no_ceres/solver/alley_2_setting1_mode1_w1_sr0.1_no_ceres.log` |
| alley_2 | 2 | 1 | 0.3 | 1 | no | 0.0835271100001130 | 0.0850475292052894 | 0.4380836717625332 | 0.0708244305631728 | 3.129720 | 0.426537 | 3 | `Python/logs/sintel_xm_no_ceres/solver/alley_2_setting2_mode1_w1_sr0.3_no_ceres.log` |
| alley_2 | 3 | 1 | 0.5 | 1 | no | 0.0836643149802288 | 0.0764983320398677 | 0.3954207101846976 | 0.0628950789970579 | 3.084531 | 0.488686 | 3 | `Python/logs/sintel_xm_no_ceres/solver/alley_2_setting3_mode1_w1_sr0.5_no_ceres.log` |
| alley_2 | 4 | 1 | 0.9 | 2 | no | 0.0954919395645395 | 0.0511506837647693 | 0.9749810985691866 | 0.0414984579884262 | 4.002109 | 1.295219 | 3 | `Python/logs/sintel_xm_no_ceres/solver/alley_2_setting4_mode1_w2_sr0.9_no_ceres.log` |
| alley_2 | 5 | 1 | 0.9 | 3 | no | 0.0864314529270915 | 0.0708821614609543 | 1.0539921609297804 | 0.0584297903357989 | 3.626357 | 0.869026 | 3 | `Python/logs/sintel_xm_no_ceres/solver/alley_2_setting5_mode1_w3_sr0.9_no_ceres.log` |
| alley_2 | 6 | 2 | - | 1 | no | 0.0889994840831194 | 0.1006944799076550 | 2.0705461800512235 | 0.0940035543370680 | 2.993229 | 0.256641 | 3 | `Python/logs/sintel_xm_no_ceres/solver/alley_2_setting6_mode2_w1_no_ceres.log` |
| ambush_2 | 1 | 1 | 0.1 | 1 | no | 0.4329211140409605 | 0.0000120213829224 | 2.4989043110268350 | 0.0000076547203914 | 4.433526 | 2.639808 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_2_setting1_mode1_w1_sr0.1_no_ceres.log` |
| ambush_2 | 2 | 1 | 0.3 | 1 | no | 0.5178342835790000 | 0.0000078302912199 | 1.3399300040920341 | 0.0000055625382805 | 5.745452 | 3.933357 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_2_setting2_mode1_w1_sr0.3_no_ceres.log` |
| ambush_2 | 3 | 1 | 0.5 | 1 | no | 0.4929051524866383 | 0.0000066084173219 | 2.7956092307734242 | 0.0000045309199085 | 6.425267 | 4.618487 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_2_setting3_mode1_w1_sr0.5_no_ceres.log` |
| ambush_2 | 4 | 1 | 0.9 | 2 | no | 0.0824629434451376 | 0.0000040720827934 | 1.7693614998739338 | 0.0000027334029261 | 2.500806 | 0.737092 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_2_setting4_mode1_w2_sr0.9_no_ceres.log` |
| ambush_2 | 5 | 1 | 0.9 | 3 | no | 0.1586072944436490 | 0.0000041114482778 | 2.1003877886796913 | 0.0000027707123782 | 3.177304 | 1.276591 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_2_setting5_mode1_w3_sr0.9_no_ceres.log` |
| ambush_2 | 6 | 2 | - | 1 | no | 0.4895838711987353 | 0.0000077281366664 | 1.2981906292149517 | 0.0000049773363499 | 3.052471 | 1.347162 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_2_setting6_mode2_w1_no_ceres.log` |
| ambush_4 | 1 | 1 | 0.1 | 1 | no | 0.0188390737284687 | 0.0717308930927819 | 2.4866264356647148 | 0.0500391145922865 | 2.775560 | 0.425072 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_4_setting1_mode1_w1_sr0.1_no_ceres.log` |
| ambush_4 | 2 | 1 | 0.3 | 1 | no | 0.0195569660976418 | 0.0750372162318482 | 1.5489324475592012 | 0.0511381340710726 | 2.961380 | 0.527828 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_4_setting2_mode1_w1_sr0.3_no_ceres.log` |
| ambush_4 | 3 | 1 | 0.5 | 1 | no | 0.0192675659608863 | 0.0691400159764856 | 2.3862782135218596 | 0.0502707836130967 | 3.087615 | 0.593282 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_4_setting3_mode1_w1_sr0.5_no_ceres.log` |
| ambush_4 | 4 | 1 | 0.9 | 2 | no | 0.0080755078127534 | 0.1657054156057023 | 2.7700564353347152 | 0.1090251597206261 | 3.069562 | 0.691903 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_4_setting4_mode1_w2_sr0.9_no_ceres.log` |
| ambush_4 | 5 | 1 | 0.9 | 3 | no | 0.0106608349052756 | 0.0880279108105103 | 0.2732232046953560 | 0.0685436088466619 | 3.121644 | 0.653116 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_4_setting5_mode1_w3_sr0.9_no_ceres.log` |
| ambush_4 | 6 | 2 | - | 1 | no | 0.0137955379646375 | 0.1356905148343004 | 2.3664293339951650 | 0.0860328202772424 | 2.560636 | 0.265675 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_4_setting6_mode2_w1_no_ceres.log` |
| ambush_5 | 1 | 1 | 0.1 | 1 | no | 0.0476474329363119 | 0.2172236425034224 | 0.1700689830652631 | 0.1506250787859348 | 3.038426 | 0.451327 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_5_setting1_mode1_w1_sr0.1_no_ceres.log` |
| ambush_5 | 2 | 1 | 0.3 | 1 | no | 0.0470765215069801 | 0.2157766200434061 | 1.6228386241836494 | 0.1440638452208620 | 2.986185 | 0.547684 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_5_setting2_mode1_w1_sr0.3_no_ceres.log` |
| ambush_5 | 3 | 1 | 0.5 | 1 | no | 0.0465274801981368 | 0.3701813649001611 | 2.4460725652442372 | 0.3402062582521167 | 3.180370 | 0.679288 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_5_setting3_mode1_w1_sr0.5_no_ceres.log` |
| ambush_5 | 4 | 1 | 0.9 | 2 | no | 0.0255280785943534 | 0.1774106407929683 | 0.4956484031883360 | 0.1239764936171673 | 3.170395 | 0.758987 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_5_setting4_mode1_w2_sr0.9_no_ceres.log` |
| ambush_5 | 5 | 1 | 0.9 | 3 | no | 0.0340382465211413 | 0.2791332003258589 | 2.6227056086274008 | 0.1923899675961039 | 3.281945 | 0.863061 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_5_setting5_mode1_w3_sr0.9_no_ceres.log` |
| ambush_5 | 6 | 2 | - | 1 | no | 0.0600234866494789 | 0.5188819717743659 | 2.5791047882169864 | 0.3531614564905857 | 2.565239 | 0.276049 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_5_setting6_mode2_w1_no_ceres.log` |
| ambush_6 | 1 | 1 | 0.1 | 1 | no | 0.0995295118885481 | 0.2492434438184399 | 2.9860778884270363 | 0.1695521004862884 | 2.361682 | 0.301565 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_6_setting1_mode1_w1_sr0.1_no_ceres.log` |
| ambush_6 | 2 | 1 | 0.3 | 1 | no | 0.0964736678602297 | 0.2785750126386786 | 1.0387874583639900 | 0.1986582946171062 | 2.260338 | 0.334687 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_6_setting2_mode1_w1_sr0.3_no_ceres.log` |
| ambush_6 | 3 | 1 | 0.5 | 1 | no | 0.0956337982488899 | 0.2754571890938941 | 1.2365412628121752 | 0.2168154389286235 | 2.390986 | 0.348142 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_6_setting3_mode1_w1_sr0.5_no_ceres.log` |
| ambush_6 | 4 | 1 | 0.9 | 2 | no | 0.0984212544588721 | 0.2612606593604040 | 1.4241511987073854 | 0.1921542361663051 | 2.361883 | 0.410622 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_6_setting4_mode1_w2_sr0.9_no_ceres.log` |
| ambush_6 | 5 | 1 | 0.9 | 3 | no | 0.0924268294662530 | 0.2689473200112866 | 1.0009229444021241 | 0.1948724379495977 | 2.413059 | 0.400145 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_6_setting5_mode1_w3_sr0.9_no_ceres.log` |
| ambush_6 | 6 | 2 | - | 1 | no | 0.1057813215352028 | 0.3865034650829842 | 1.7287969080966661 | 0.2669184239138406 | 2.330688 | 0.243449 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_6_setting6_mode2_w1_no_ceres.log` |
| ambush_7 | 1 | 1 | 0.1 | 1 | no | 0.0290843037173668 | 0.0000051269827219 | 1.7914179041433145 | 0.0000036367193894 | 2.646402 | 0.356466 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_7_setting1_mode1_w1_sr0.1_no_ceres.log` |
| ambush_7 | 2 | 1 | 0.3 | 1 | no | 0.0288137008072235 | 0.0000051977666149 | 1.1003321398157844 | 0.0000037199554154 | 2.757023 | 0.420561 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_7_setting2_mode1_w1_sr0.3_no_ceres.log` |
| ambush_7 | 3 | 1 | 0.5 | 1 | no | 0.0287807172443678 | 0.0000052209747831 | 1.0398415805477828 | 0.0000037381866411 | 2.834547 | 0.474353 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_7_setting3_mode1_w1_sr0.5_no_ceres.log` |
| ambush_7 | 4 | 1 | 0.9 | 2 | no | 0.0027752909805789 | 0.0000051871697865 | 2.8698926312526645 | 0.0000038404441524 | 2.814236 | 0.432971 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_7_setting4_mode1_w2_sr0.9_no_ceres.log` |
| ambush_7 | 5 | 1 | 0.9 | 3 | no | 0.0099491052519571 | 0.0000050734459222 | 2.6279713890533918 | 0.0000037375314264 | 2.884009 | 0.482098 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_7_setting5_mode1_w3_sr0.9_no_ceres.log` |
| ambush_7 | 6 | 2 | - | 1 | no | 0.0247007591696683 | 0.0000048990578662 | 1.8819567361017646 | 0.0000035104816466 | 2.687881 | 0.248763 | 3 | `Python/logs/sintel_xm_no_ceres/solver/ambush_7_setting6_mode2_w1_no_ceres.log` |
| bamboo_1 | 1 | 1 | 0.1 | 1 | no | 0.0054793829601972 | 0.0402701321475860 | 0.4857024216738227 | 0.0283073068874056 | 2.878257 | 0.259058 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bamboo_1_setting1_mode1_w1_sr0.1_no_ceres.log` |
| bamboo_1 | 2 | 1 | 0.3 | 1 | no | 0.0054874644817900 | 0.0326735812963169 | 2.3071140159042365 | 0.0239162325294314 | 2.888033 | 0.291443 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bamboo_1_setting2_mode1_w1_sr0.3_no_ceres.log` |
| bamboo_1 | 3 | 1 | 0.5 | 1 | no | 0.0055133290028777 | 0.0310093043682664 | 0.5790401675592950 | 0.0229121119895690 | 2.966883 | 0.338786 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bamboo_1_setting3_mode1_w1_sr0.5_no_ceres.log` |
| bamboo_1 | 4 | 1 | 0.9 | 2 | no | 0.0122705193382545 | 0.0328935643178939 | 0.5857495868107108 | 0.0249481571673684 | 2.939686 | 0.338622 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bamboo_1_setting4_mode1_w2_sr0.9_no_ceres.log` |
| bamboo_1 | 5 | 1 | 0.9 | 3 | no | 0.0057720398282452 | 0.0300010375518739 | 2.7532923877064337 | 0.0228111258093818 | 2.813261 | 0.368961 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bamboo_1_setting5_mode1_w3_sr0.9_no_ceres.log` |
| bamboo_1 | 6 | 2 | - | 1 | no | 0.0058815260340513 | 0.0649804510542694 | 2.7364472820018397 | 0.0414708880082274 | 2.926685 | 0.243272 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bamboo_1_setting6_mode2_w1_no_ceres.log` |
| bamboo_2 | 1 | 1 | 0.1 | 1 | no | 0.0128543289078765 | 0.1092369314273509 | 2.6466526736745504 | 0.0793019227119874 | 2.834198 | 0.278482 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bamboo_2_setting1_mode1_w1_sr0.1_no_ceres.log` |
| bamboo_2 | 2 | 1 | 0.3 | 1 | no | 0.0129019774437259 | 0.0791959211651071 | 1.0467722407350732 | 0.0568390521605247 | 2.868648 | 0.323721 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bamboo_2_setting2_mode1_w1_sr0.3_no_ceres.log` |
| bamboo_2 | 3 | 1 | 0.5 | 1 | no | 0.0128412061384276 | 0.0677875397113305 | 2.6312648703718047 | 0.0484071008192331 | 2.937317 | 0.327187 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bamboo_2_setting3_mode1_w1_sr0.5_no_ceres.log` |
| bamboo_2 | 4 | 1 | 0.9 | 2 | no | 0.0153641525494117 | 0.0639124932617900 | 2.5647577572829379 | 0.0463083742520033 | 3.080567 | 0.441703 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bamboo_2_setting4_mode1_w2_sr0.9_no_ceres.log` |
| bamboo_2 | 5 | 1 | 0.9 | 3 | no | 0.0132538183221547 | 0.0672963681735338 | 0.9281970831955494 | 0.0534485936873659 | 3.040101 | 0.397917 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bamboo_2_setting5_mode1_w3_sr0.9_no_ceres.log` |
| bamboo_2 | 6 | 2 | - | 1 | no | 0.0129755809760846 | 0.1232051202392587 | 2.4630805229718971 | 0.1000025886125737 | 3.017269 | 0.221696 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bamboo_2_setting6_mode2_w1_no_ceres.log` |
| bandage_1 | 1 | 1 | 0.1 | 1 | no | 0.1977983510560500 | 0.2398085228023760 | 1.8946641697084770 | 0.1894664098507078 | 2.501231 | 0.332815 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bandage_1_setting1_mode1_w1_sr0.1_no_ceres.log` |
| bandage_1 | 2 | 1 | 0.3 | 1 | no | 0.1977839225636844 | 0.2357942647658188 | 2.1542454991493956 | 0.1882344515758262 | 2.635034 | 0.383989 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bandage_1_setting2_mode1_w1_sr0.3_no_ceres.log` |
| bandage_1 | 3 | 1 | 0.5 | 1 | no | 0.1970599343804564 | 0.2325930994889865 | 0.7458178130347869 | 0.1874786624835245 | 2.553528 | 0.476908 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bandage_1_setting3_mode1_w1_sr0.5_no_ceres.log` |
| bandage_1 | 4 | 1 | 0.9 | 2 | no | 0.1432911350770573 | 0.0363175511445719 | 0.8683766676466965 | 0.0283766602479664 | 2.596200 | 0.409082 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bandage_1_setting4_mode1_w2_sr0.9_no_ceres.log` |
| bandage_1 | 5 | 1 | 0.9 | 3 | no | 0.1440624578088269 | 0.1254554599692929 | 1.2609041952501676 | 0.0998456999030404 | 2.622131 | 0.458854 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bandage_1_setting5_mode1_w3_sr0.9_no_ceres.log` |
| bandage_1 | 6 | 2 | - | 1 | no | 0.1962933561076316 | 0.1237361231744159 | 1.1826365718007141 | 0.0964958532774077 | 2.394858 | 0.257609 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bandage_1_setting6_mode2_w1_no_ceres.log` |
| bandage_2 | 1 | 1 | 0.1 | 1 | no | 0.0122669555006517 | 0.0390397411989079 | 1.9855293336702935 | 0.0286353393260159 | 2.400155 | 0.320904 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bandage_2_setting1_mode1_w1_sr0.1_no_ceres.log` |
| bandage_2 | 2 | 1 | 0.3 | 1 | no | 0.0120065891118080 | 0.0383544145621132 | 1.9403083604311790 | 0.0279714421738666 | 2.439141 | 0.348591 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bandage_2_setting2_mode1_w1_sr0.3_no_ceres.log` |
| bandage_2 | 3 | 1 | 0.5 | 1 | no | 0.0119305427804876 | 0.0381261893612729 | 2.8105949537381836 | 0.0276074531574841 | 2.377249 | 0.402181 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bandage_2_setting3_mode1_w1_sr0.5_no_ceres.log` |
| bandage_2 | 4 | 1 | 0.9 | 2 | no | 0.0022722473785392 | 0.0065060659765857 | 3.0063111374614380 | 0.0051324636556919 | 2.433338 | 0.361347 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bandage_2_setting4_mode1_w2_sr0.9_no_ceres.log` |
| bandage_2 | 5 | 1 | 0.9 | 3 | no | 0.0055619524514073 | 0.0170503524153481 | 3.0736730531022509 | 0.0129667273125721 | 2.413824 | 0.372266 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bandage_2_setting5_mode1_w3_sr0.9_no_ceres.log` |
| bandage_2 | 6 | 2 | - | 1 | no | 0.0125718427598802 | 0.0439202311151968 | 1.1628924347042073 | 0.0312643903864262 | 2.335625 | 0.252971 | 3 | `Python/logs/sintel_xm_no_ceres/solver/bandage_2_setting6_mode2_w1_no_ceres.log` |
| cave_2 | 1 | 1 | 0.1 | 1 | no | 0.9087587728667010 | 0.6531924795456561 | 2.5552267475902632 | 0.7845618414619929 | 5.194368 | 2.553319 | 3 | `Python/logs/sintel_xm_no_ceres/solver/cave_2_setting1_mode1_w1_sr0.1_no_ceres.log` |
| cave_2 | 2 | 1 | 0.3 | 1 | no | 0.9020980309056733 | 0.6247719796302867 | 2.5951517664853223 | 0.7915535635292420 | 5.860658 | 3.272830 | 3 | `Python/logs/sintel_xm_no_ceres/solver/cave_2_setting2_mode1_w1_sr0.3_no_ceres.log` |
| cave_2 | 3 | 1 | 0.5 | 1 | no | 0.8664858611463246 | 0.5896480245760832 | 2.6543722430564403 | 0.7510567400405675 | 4.539107 | 2.225144 | 3 | `Python/logs/sintel_xm_no_ceres/solver/cave_2_setting3_mode1_w1_sr0.5_no_ceres.log` |
| cave_2 | 4 | 1 | 0.9 | 2 | no | 0.4910297645034831 | 0.7151354654112561 | 1.3148053365085215 | 0.4812507473450670 | 4.270042 | 1.694055 | 3 | `Python/logs/sintel_xm_no_ceres/solver/cave_2_setting4_mode1_w2_sr0.9_no_ceres.log` |
| cave_2 | 5 | 1 | 0.9 | 3 | no | 0.6226673549166590 | 0.6430086182244724 | 1.5111664243371214 | 0.4352725442255406 | 4.372126 | 1.848837 | 3 | `Python/logs/sintel_xm_no_ceres/solver/cave_2_setting5_mode1_w3_sr0.9_no_ceres.log` |
| cave_2 | 6 | 2 | - | 1 | no | 0.7012775044196836 | 0.9527047132348039 | 2.6461663348857076 | 1.0790742469916890 | 3.286218 | 0.811817 | 4 | `Python/logs/sintel_xm_no_ceres/solver/cave_2_setting6_mode2_w1_no_ceres.log` |
| cave_4 | 1 | 1 | 0.1 | 1 | no | 0.0087574193389743 | 0.0429170792115697 | 0.2130680964133142 | 0.0278152091662757 | 2.489188 | 0.338736 | 3 | `Python/logs/sintel_xm_no_ceres/solver/cave_4_setting1_mode1_w1_sr0.1_no_ceres.log` |
| cave_4 | 2 | 1 | 0.3 | 1 | no | 0.0088520676107018 | 0.0425601432363638 | 0.1636495121742015 | 0.0269231362841215 | 2.499977 | 0.367738 | 3 | `Python/logs/sintel_xm_no_ceres/solver/cave_4_setting2_mode1_w1_sr0.3_no_ceres.log` |
| cave_4 | 3 | 1 | 0.5 | 1 | no | 0.0088547041876840 | 0.0441136175294645 | 0.2599466277972924 | 0.0282092687150475 | 2.451599 | 0.401205 | 3 | `Python/logs/sintel_xm_no_ceres/solver/cave_4_setting3_mode1_w1_sr0.5_no_ceres.log` |
| cave_4 | 4 | 1 | 0.9 | 2 | no | 0.0219130975453993 | 0.0423404330889298 | 0.6174229853063145 | 0.0385071305321201 | 2.561081 | 0.522833 | 3 | `Python/logs/sintel_xm_no_ceres/solver/cave_4_setting4_mode1_w2_sr0.9_no_ceres.log` |
| cave_4 | 5 | 1 | 0.9 | 3 | no | 0.0117607147828074 | 0.0321312170969900 | 0.2985815392320698 | 0.0228981148144785 | 2.423552 | 0.453138 | 3 | `Python/logs/sintel_xm_no_ceres/solver/cave_4_setting5_mode1_w3_sr0.9_no_ceres.log` |
| cave_4 | 6 | 2 | - | 1 | no | 0.0185075770238818 | 0.1581146077411429 | 0.4508718039398772 | 0.1037097311749142 | 2.472560 | 0.248625 | 3 | `Python/logs/sintel_xm_no_ceres/solver/cave_4_setting6_mode2_w1_no_ceres.log` |
| market_2 | 1 | 1 | 0.1 | 1 | no | 0.0020700905963953 | 0.0899420642273946 | 0.9064528261781913 | 0.0646075003853972 | 2.681382 | 0.343915 | 3 | `Python/logs/sintel_xm_no_ceres/solver/market_2_setting1_mode1_w1_sr0.1_no_ceres.log` |
| market_2 | 2 | 1 | 0.3 | 1 | no | 0.0020551354057359 | 0.0947276878013090 | 1.0134315195475547 | 0.0650156053795354 | 2.579892 | 0.411147 | 3 | `Python/logs/sintel_xm_no_ceres/solver/market_2_setting2_mode1_w1_sr0.3_no_ceres.log` |
| market_2 | 3 | 1 | 0.5 | 1 | no | 0.0020639254879137 | 0.0730490542061146 | 1.0204239409518243 | 0.0509562686600915 | 2.729778 | 0.442117 | 3 | `Python/logs/sintel_xm_no_ceres/solver/market_2_setting3_mode1_w1_sr0.5_no_ceres.log` |
| market_2 | 4 | 1 | 0.9 | 2 | no | 0.0012204640300590 | 0.0124250399645215 | 2.9287746951316906 | 0.0084089688240176 | 2.797718 | 0.456982 | 3 | `Python/logs/sintel_xm_no_ceres/solver/market_2_setting4_mode1_w2_sr0.9_no_ceres.log` |
| market_2 | 5 | 1 | 0.9 | 3 | no | 0.0015285930677421 | 0.0261995420858882 | 3.1153634190696522 | 0.0184274099000946 | 2.721049 | 0.499942 | 3 | `Python/logs/sintel_xm_no_ceres/solver/market_2_setting5_mode1_w3_sr0.9_no_ceres.log` |
| market_2 | 6 | 2 | - | 1 | no | 0.0020593538899235 | 0.1081564996747757 | 2.9657664669572337 | 0.0751386106154132 | 2.501031 | 0.257284 | 3 | `Python/logs/sintel_xm_no_ceres/solver/market_2_setting6_mode2_w1_no_ceres.log` |
| market_5 | 1 | 1 | 0.1 | 1 | no | 0.0845418677014860 | 0.0072703796243253 | 1.7020825294861419 | 0.0055407484979760 | 3.297598 | 0.997501 | 3 | `Python/logs/sintel_xm_no_ceres/solver/market_5_setting1_mode1_w1_sr0.1_no_ceres.log` |
| market_5 | 2 | 1 | 0.3 | 1 | no | 0.0845234524271864 | 0.0074096003640417 | 1.7382891216524581 | 0.0055076043719770 | 3.812340 | 1.505091 | 3 | `Python/logs/sintel_xm_no_ceres/solver/market_5_setting2_mode1_w1_sr0.3_no_ceres.log` |
| market_5 | 3 | 1 | 0.5 | 1 | no | 0.0843441062894304 | 0.0067664153416664 | 1.7986679640025625 | 0.0049546152342172 | 4.072893 | 1.678876 | 3 | `Python/logs/sintel_xm_no_ceres/solver/market_5_setting3_mode1_w1_sr0.5_no_ceres.log` |
| market_5 | 4 | 1 | 0.9 | 2 | no | 0.1427190663827360 | 0.0196187005495639 | 1.7516679311432737 | 0.0160565248528552 | 4.440912 | 2.017146 | 3 | `Python/logs/sintel_xm_no_ceres/solver/market_5_setting4_mode1_w2_sr0.9_no_ceres.log` |
| market_5 | 5 | 1 | 0.9 | 3 | no | 0.1016751836830673 | 0.0050834983724765 | 1.8266502065352710 | 0.0036806898121837 | 4.838617 | 2.413848 | 3 | `Python/logs/sintel_xm_no_ceres/solver/market_5_setting5_mode1_w3_sr0.9_no_ceres.log` |
| market_5 | 6 | 2 | - | 1 | no | 0.1058439494632584 | 0.0080093136055766 | 1.9521600861759734 | 0.0055713307035569 | 3.031356 | 0.463193 | 3 | `Python/logs/sintel_xm_no_ceres/solver/market_5_setting6_mode2_w1_no_ceres.log` |
| market_6 | 1 | 1 | 0.1 | 1 | no | 0.0102225263947240 | 0.0318911259125982 | 1.0010268009809349 | 0.0229575126660095 | 2.644570 | 0.322591 | 3 | `Python/logs/sintel_xm_no_ceres/solver/market_6_setting1_mode1_w1_sr0.1_no_ceres.log` |
| market_6 | 2 | 1 | 0.3 | 1 | no | 0.0097814464498314 | 0.0262033470892493 | 1.4549895839455975 | 0.0189246923003961 | 2.667732 | 0.359283 | 3 | `Python/logs/sintel_xm_no_ceres/solver/market_6_setting2_mode1_w1_sr0.3_no_ceres.log` |
| market_6 | 3 | 1 | 0.5 | 1 | no | 0.0093535385167399 | 0.0299220461550863 | 1.6596600192285300 | 0.0214558708320110 | 2.670893 | 0.377935 | 3 | `Python/logs/sintel_xm_no_ceres/solver/market_6_setting3_mode1_w1_sr0.5_no_ceres.log` |
| market_6 | 4 | 1 | 0.9 | 2 | no | 0.0181288821549513 | 0.1012740740241592 | 1.4653180156863417 | 0.0726670057120033 | 2.740073 | 0.510350 | 3 | `Python/logs/sintel_xm_no_ceres/solver/market_6_setting4_mode1_w2_sr0.9_no_ceres.log` |
| market_6 | 5 | 1 | 0.9 | 3 | no | 0.0065727894218635 | 0.0474586753334020 | 1.2740674526595421 | 0.0333682676822650 | 2.733993 | 0.483913 | 3 | `Python/logs/sintel_xm_no_ceres/solver/market_6_setting5_mode1_w3_sr0.9_no_ceres.log` |
| market_6 | 6 | 2 | - | 1 | no | 0.0102545413692534 | 0.0399481280150578 | 1.1389823492853164 | 0.0282953724460322 | 2.589928 | 0.252389 | 3 | `Python/logs/sintel_xm_no_ceres/solver/market_6_setting6_mode2_w1_no_ceres.log` |
| mountain_1 | 1 | 1 | 0.1 | 1 | no | 0.0127405089368735 | 0.0381510607797123 | 0.8303632285631491 | 0.0271201989065168 | 2.777582 | 0.321038 | 3 | `Python/logs/sintel_xm_no_ceres/solver/mountain_1_setting1_mode1_w1_sr0.1_no_ceres.log` |
| mountain_1 | 2 | 1 | 0.3 | 1 | no | 0.0127291683251755 | 0.0320467078274284 | 0.8344520641375162 | 0.0227772992985040 | 3.192906 | 0.359305 | 3 | `Python/logs/sintel_xm_no_ceres/solver/mountain_1_setting2_mode1_w1_sr0.3_no_ceres.log` |
| mountain_1 | 3 | 1 | 0.5 | 1 | no | 0.0127422500640313 | 0.0295994096047649 | 0.7308087652108809 | 0.0208687155005318 | 2.906627 | 0.396691 | 3 | `Python/logs/sintel_xm_no_ceres/solver/mountain_1_setting3_mode1_w1_sr0.5_no_ceres.log` |
| mountain_1 | 4 | 1 | 0.9 | 2 | no | 0.0089320869927076 | 0.0969497347519482 | 1.9680187387703183 | 0.0647402245476535 | 3.140336 | 0.432244 | 3 | `Python/logs/sintel_xm_no_ceres/solver/mountain_1_setting4_mode1_w2_sr0.9_no_ceres.log` |
| mountain_1 | 5 | 1 | 0.9 | 3 | no | 0.0128539755504981 | 0.0385649218042370 | 1.6739104417863970 | 0.0253876108928469 | 3.100897 | 0.455528 | 3 | `Python/logs/sintel_xm_no_ceres/solver/mountain_1_setting5_mode1_w3_sr0.9_no_ceres.log` |
| mountain_1 | 6 | 2 | - | 1 | no | 0.0126115646206360 | 0.0334455349646160 | 1.0432195896694696 | 0.0232535910915055 | 2.979024 | 0.249335 | 3 | `Python/logs/sintel_xm_no_ceres/solver/mountain_1_setting6_mode2_w1_no_ceres.log` |
| shaman_2 | 1 | 1 | 0.1 | 1 | no | 0.0354661809639942 | 0.0540233638666898 | 2.4841439793318316 | 0.0433490744968506 | 2.729386 | 0.328904 | 3 | `Python/logs/sintel_xm_no_ceres/solver/shaman_2_setting1_mode1_w1_sr0.1_no_ceres.log` |
| shaman_2 | 2 | 1 | 0.3 | 1 | no | 0.0353650947257675 | 0.0535053897049530 | 2.4722489014880531 | 0.0428369762213219 | 2.760211 | 0.389381 | 3 | `Python/logs/sintel_xm_no_ceres/solver/shaman_2_setting2_mode1_w1_sr0.3_no_ceres.log` |
| shaman_2 | 3 | 1 | 0.5 | 1 | no | 0.0353736510937217 | 0.0533927236339872 | 2.5858884413475471 | 0.0427462208171750 | 2.795072 | 0.465360 | 3 | `Python/logs/sintel_xm_no_ceres/solver/shaman_2_setting3_mode1_w1_sr0.5_no_ceres.log` |
| shaman_2 | 4 | 1 | 0.9 | 2 | no | 0.0118267210325838 | 0.0295245467492390 | 1.5981952370635368 | 0.0234778829357325 | 2.986568 | 0.403724 | 3 | `Python/logs/sintel_xm_no_ceres/solver/shaman_2_setting4_mode1_w2_sr0.9_no_ceres.log` |
| shaman_2 | 5 | 1 | 0.9 | 3 | no | 0.0276010951429565 | 0.0462288869436299 | 2.1186897089093235 | 0.0373053538365111 | 2.940340 | 0.516782 | 3 | `Python/logs/sintel_xm_no_ceres/solver/shaman_2_setting5_mode1_w3_sr0.9_no_ceres.log` |
| shaman_2 | 6 | 2 | - | 1 | no | 0.0362843582997709 | 0.0496728090528998 | 2.6425519998860660 | 0.0399632098520136 | 2.629693 | 0.238624 | 3 | `Python/logs/sintel_xm_no_ceres/solver/shaman_2_setting6_mode2_w1_no_ceres.log` |
| shaman_3 | 1 | 1 | 0.1 | 1 | no | 0.0266477844470951 | 0.0048016597203934 | 2.6274173669186558 | 0.0033245685765175 | 2.438815 | 0.350249 | 3 | `Python/logs/sintel_xm_no_ceres/solver/shaman_3_setting1_mode1_w1_sr0.1_no_ceres.log` |
| shaman_3 | 2 | 1 | 0.3 | 1 | no | 0.0266629981856213 | 0.0048186170597576 | 2.6242440610662889 | 0.0033367491410279 | 2.556601 | 0.414550 | 3 | `Python/logs/sintel_xm_no_ceres/solver/shaman_3_setting2_mode1_w1_sr0.3_no_ceres.log` |
| shaman_3 | 3 | 1 | 0.5 | 1 | no | 0.0266873366651828 | 0.0048972606452151 | 2.5965198562682370 | 0.0034095059249528 | 2.717366 | 0.506756 | 3 | `Python/logs/sintel_xm_no_ceres/solver/shaman_3_setting3_mode1_w1_sr0.5_no_ceres.log` |
| shaman_3 | 4 | 1 | 0.9 | 2 | no | 0.0169358176158553 | 0.0105425358932700 | 0.3960285526734330 | 0.0083469215300128 | 2.768261 | 0.627438 | 3 | `Python/logs/sintel_xm_no_ceres/solver/shaman_3_setting4_mode1_w2_sr0.9_no_ceres.log` |
| shaman_3 | 5 | 1 | 0.9 | 3 | no | 0.0214261766418841 | 0.0077608352706461 | 2.2733939303465736 | 0.0050899239341716 | 2.901538 | 0.654631 | 3 | `Python/logs/sintel_xm_no_ceres/solver/shaman_3_setting5_mode1_w3_sr0.9_no_ceres.log` |
| shaman_3 | 6 | 2 | - | 1 | no | 0.0259858858641232 | 0.0060172119486299 | 2.8473705776342335 | 0.0039381055576583 | 2.539296 | 0.271826 | 3 | `Python/logs/sintel_xm_no_ceres/solver/shaman_3_setting6_mode2_w1_no_ceres.log` |
| sleeping_1 | 1 | 1 | 0.1 | 1 | no | 0.0203043877068457 | 0.0047955178146585 | 0.9368230873467259 | 0.0031701029980967 | 2.713704 | 0.311612 | 3 | `Python/logs/sintel_xm_no_ceres/solver/sleeping_1_setting1_mode1_w1_sr0.1_no_ceres.log` |
| sleeping_1 | 2 | 1 | 0.3 | 1 | no | 0.0203417708748912 | 0.0049294524320398 | 0.8909930036423745 | 0.0032263045828359 | 2.588365 | 0.350098 | 3 | `Python/logs/sintel_xm_no_ceres/solver/sleeping_1_setting2_mode1_w1_sr0.3_no_ceres.log` |
| sleeping_1 | 3 | 1 | 0.5 | 1 | no | 0.0203569397186394 | 0.0049322477953672 | 0.8769002772977353 | 0.0032250753651824 | 2.508627 | 0.399695 | 3 | `Python/logs/sintel_xm_no_ceres/solver/sleeping_1_setting3_mode1_w1_sr0.5_no_ceres.log` |
| sleeping_1 | 4 | 1 | 0.9 | 2 | no | 0.0544072240207824 | 0.0283371590834535 | 1.0377288717265021 | 0.0239663488337528 | 2.628656 | 0.462245 | 3 | `Python/logs/sintel_xm_no_ceres/solver/sleeping_1_setting4_mode1_w2_sr0.9_no_ceres.log` |
| sleeping_1 | 5 | 1 | 0.9 | 3 | no | 0.0372539702844146 | 0.0173465351198673 | 1.3024879159665397 | 0.0110420171936794 | 2.820520 | 0.536730 | 3 | `Python/logs/sintel_xm_no_ceres/solver/sleeping_1_setting5_mode1_w3_sr0.9_no_ceres.log` |
| sleeping_1 | 6 | 2 | - | 1 | no | 0.0221965371029232 | 0.0126672772460574 | 0.9295705115384011 | 0.0073902922716595 | 2.482012 | 0.273497 | 3 | `Python/logs/sintel_xm_no_ceres/solver/sleeping_1_setting6_mode2_w1_no_ceres.log` |
| sleeping_2 | 1 | 1 | 0.1 | 1 | no | 0.0030248418656635 | 0.0022338162347702 | 0.0490453287999058 | 0.0015189898604836 | 3.011860 | 0.258784 | 3 | `Python/logs/sintel_xm_no_ceres/solver/sleeping_2_setting1_mode1_w1_sr0.1_no_ceres.log` |
| sleeping_2 | 2 | 1 | 0.3 | 1 | no | 0.0030209594858190 | 0.0021995636821832 | 0.0517973714241091 | 0.0014897573837125 | 3.217634 | 0.283949 | 3 | `Python/logs/sintel_xm_no_ceres/solver/sleeping_2_setting2_mode1_w1_sr0.3_no_ceres.log` |
| sleeping_2 | 3 | 1 | 0.5 | 1 | no | 0.0030165958043401 | 0.0021844566391713 | 0.0522527413238238 | 0.0014754164474880 | 3.230847 | 0.294517 | 3 | `Python/logs/sintel_xm_no_ceres/solver/sleeping_2_setting3_mode1_w1_sr0.5_no_ceres.log` |
| sleeping_2 | 4 | 1 | 0.9 | 2 | no | 0.0315087395064625 | 0.0168624962356047 | 0.2960169515518009 | 0.0109859179842908 | 3.259930 | 0.390386 | 3 | `Python/logs/sintel_xm_no_ceres/solver/sleeping_2_setting4_mode1_w2_sr0.9_no_ceres.log` |
| sleeping_2 | 5 | 1 | 0.9 | 3 | no | 0.0048210557065326 | 0.0042720727830855 | 0.0909937257974737 | 0.0027572060493785 | 3.279786 | 0.373700 | 3 | `Python/logs/sintel_xm_no_ceres/solver/sleeping_2_setting5_mode1_w3_sr0.9_no_ceres.log` |
| sleeping_2 | 6 | 2 | - | 1 | no | 0.0031639156011888 | 0.0038331010067123 | 0.0681598171963984 | 0.0024876859946347 | 2.938699 | 0.233785 | 3 | `Python/logs/sintel_xm_no_ceres/solver/sleeping_2_setting6_mode2_w1_no_ceres.log` |
| temple_2 | 1 | 1 | 0.1 | 1 | no | 0.0741755558069968 | 0.1608271045291013 | 0.2946116722302569 | 0.1137922807023423 | 3.277773 | 0.386646 | 3 | `Python/logs/sintel_xm_no_ceres/solver/temple_2_setting1_mode1_w1_sr0.1_no_ceres.log` |
| temple_2 | 2 | 1 | 0.3 | 1 | no | 0.0745396928185876 | 0.1501329332891960 | 0.1507331694700912 | 0.1047986699747638 | 3.210354 | 0.427797 | 3 | `Python/logs/sintel_xm_no_ceres/solver/temple_2_setting2_mode1_w1_sr0.3_no_ceres.log` |
| temple_2 | 3 | 1 | 0.5 | 1 | no | 0.0743831974844180 | 0.1579806576778839 | 0.7852981138400490 | 0.1177558390894639 | 3.159328 | 0.463106 | 3 | `Python/logs/sintel_xm_no_ceres/solver/temple_2_setting3_mode1_w1_sr0.5_no_ceres.log` |
| temple_2 | 4 | 1 | 0.9 | 2 | no | 0.0935514075527787 | 0.0780694961748650 | 0.1864483048177006 | 0.0516228665300060 | 3.745549 | 0.882205 | 3 | `Python/logs/sintel_xm_no_ceres/solver/temple_2_setting4_mode1_w2_sr0.9_no_ceres.log` |
| temple_2 | 5 | 1 | 0.9 | 3 | no | 0.0780559144692153 | 0.1064653887405060 | 0.0848410157908025 | 0.0745289845092150 | 3.673788 | 0.761314 | 3 | `Python/logs/sintel_xm_no_ceres/solver/temple_2_setting5_mode1_w3_sr0.9_no_ceres.log` |
| temple_2 | 6 | 2 | - | 1 | no | 0.1048003709805063 | 0.2772693249853905 | 2.5826362112090369 | 0.1868227992411294 | 3.141920 | 0.281871 | 3 | `Python/logs/sintel_xm_no_ceres/solver/temple_2_setting6_mode2_w1_no_ceres.log` |
| temple_3 | 1 | 1 | 0.1 | 1 | no | 0.1134341554550405 | 0.1435210488442222 | 2.9416070363800850 | 0.1044286308384843 | 3.649751 | 1.193619 | 3 | `Python/logs/sintel_xm_no_ceres/solver/temple_3_setting1_mode1_w1_sr0.1_no_ceres.log` |
| temple_3 | 2 | 1 | 0.3 | 1 | no | 0.1097853345418935 | 0.1682915872544124 | 2.5283388120618224 | 0.1200411023142005 | 3.937731 | 1.632732 | 3 | `Python/logs/sintel_xm_no_ceres/solver/temple_3_setting2_mode1_w1_sr0.3_no_ceres.log` |
| temple_3 | 3 | 1 | 0.5 | 1 | no | 0.1018986710629391 | 0.1463091356100376 | 2.3708912135712810 | 0.1092945027585631 | 4.573693 | 2.132484 | 3 | `Python/logs/sintel_xm_no_ceres/solver/temple_3_setting3_mode1_w1_sr0.5_no_ceres.log` |
| temple_3 | 4 | 1 | 0.9 | 2 | no | 0.0079506129740120 | 0.1762151663179908 | 1.1518769196625951 | 0.1285386985437560 | 3.829898 | 1.327127 | 3 | `Python/logs/sintel_xm_no_ceres/solver/temple_3_setting4_mode1_w2_sr0.9_no_ceres.log` |
| temple_3 | 5 | 1 | 0.9 | 3 | no | 0.0315606179775122 | 0.1580971931232133 | 1.5186997618389313 | 0.1102874364990628 | 4.288893 | 1.804932 | 3 | `Python/logs/sintel_xm_no_ceres/solver/temple_3_setting5_mode1_w3_sr0.9_no_ceres.log` |
| temple_3 | 6 | 2 | - | 1 | no | 0.1210334636393508 | 0.1701192567172312 | 1.4242248295111770 | 0.1140141853086703 | 3.077838 | 0.590852 | 3 | `Python/logs/sintel_xm_no_ceres/solver/temple_3_setting6_mode2_w1_no_ceres.log` |
