# Sintel XM Experiment Results

Generated: `2026-04-21T17:15:15-04:00`

Pipeline:

- Dataset root: `datasets/Sintel`
- Missing GT cameras are loaded from `datasets/MPI-Sintel-depth-training-20150305/training/camdata_left`
- Missing DA3 depth is generated with `Python/get_depths_clean.py` in the `da3` environment
- Missing `output_tapnet_da3.npz` is generated with `Python/static_tracking_online.py` in the `tapnet` environment
- Solver sweep uses the same 6 heuristics as the BAL experiment, with no added noise

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
| bamboo_1 | 50 | local | reused | generated | - | `Python/logs/sintel_xm/prep/bamboo_1_tracking.log` |
| bamboo_2 | 50 | local | generated | generated | `Python/logs/sintel_xm/prep/bamboo_2_depth.log` | `Python/logs/sintel_xm/prep/bamboo_2_tracking.log` |
| bandage_1 | 50 | local | reused | reused | - | - |
| bandage_2 | 50 | local | generated | generated | `Python/logs/sintel_xm_part2/prep/bandage_2_depth.log` | `Python/logs/sintel_xm_part2/prep/bandage_2_tracking.log` |
| cave_2 | 50 | local | reused | reused | - | - |
| cave_4 | 50 | local | generated | generated | `Python/logs/sintel_xm_part2/prep/cave_4_depth.log` | `Python/logs/sintel_xm_part2/prep/cave_4_tracking.log` |
| market_2 | 50 | local | reused | reused | - | - |
| market_5 | 50 | local | generated | generated | `Python/logs/sintel_xm_part2/prep/market_5_depth.log` | `Python/logs/sintel_xm_part2/prep/market_5_tracking.log` |
| market_6 | 40 | mpi_fallback | reused | reused | - | - |
| mountain_1 | 50 | mpi_fallback | generated | generated | `Python/logs/sintel_xm_part2/prep/mountain_1_depth.log` | `Python/logs/sintel_xm_part2/prep/mountain_1_tracking.log` |
| shaman_2 | 50 | mpi_fallback | generated | generated | `Python/logs/sintel_xm_part2/prep/shaman_2_depth.log` | `Python/logs/sintel_xm_part2/prep/shaman_2_tracking.log` |
| shaman_3 | 50 | mpi_fallback | generated | generated | `Python/logs/sintel_xm_part2/prep/shaman_3_depth.log` | `Python/logs/sintel_xm_part2/prep/shaman_3_tracking.log` |
| sleeping_1 | 50 | mpi_fallback | generated | generated | `Python/logs/sintel_xm_part2/prep/sleeping_1_depth.log` | `Python/logs/sintel_xm_part2/prep/sleeping_1_tracking.log` |
| sleeping_2 | 50 | mpi_fallback | generated | generated | `Python/logs/sintel_xm_part2/prep/sleeping_2_depth.log` | `Python/logs/sintel_xm_part2/prep/sleeping_2_tracking.log` |
| temple_2 | 50 | local | reused | reused | - | - |
| temple_3 | 50 | mpi_fallback | generated | generated | `Python/logs/sintel_xm_part2/prep/temple_3_depth.log` | `Python/logs/sintel_xm_part2/prep/temple_3_tracking.log` |

## Solver Results

| Scene | Setting | Mode | Sphere Ratio | WEIGHT_MODE | RPE-R | RPE-T | ATE-R | ATE-T | Total time (s) | Solve time (s) | Optimal rank | Log |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| alley_1 | 1 | 1 | 0.1 | 1 | 0.0252406291725374 | 0.0032760645864855 | 2.4989911043064650 | 0.0024024645255111 | 53.973447 | 0.288401 | 3 | `Python/logs/sintel_xm/solver/alley_1_setting1_mode1_w1_sr0.1.log` |
| alley_1 | 2 | 1 | 0.3 | 1 | 0.0251357232712592 | 0.0034691555807755 | 3.0849344821618763 | 0.0025337942671228 | 8.042169 | 0.339867 | 3 | `Python/logs/sintel_xm/solver/alley_1_setting2_mode1_w1_sr0.3.log` |
| alley_1 | 3 | 1 | 0.5 | 1 | 0.0255925656890788 | 0.0032189177513024 | 2.9821123750374867 | 0.0023446449032449 | 7.888055 | 0.396947 | 3 | `Python/logs/sintel_xm/solver/alley_1_setting3_mode1_w1_sr0.5.log` |
| alley_1 | 4 | 1 | 0.9 | 2 | 0.0067516710122650 | 0.0090933058137090 | 3.0497059200206427 | 0.0068261629768741 | 46.637689 | 0.399707 | 3 | `Python/logs/sintel_xm/solver/alley_1_setting4_mode1_w2_sr0.9.log` |
| alley_1 | 5 | 1 | 0.9 | 3 | 0.0255515783829816 | 0.0031688474330087 | 2.9736429432336813 | 0.0022974808344729 | 8.321475 | 0.473856 | 3 | `Python/logs/sintel_xm/solver/alley_1_setting5_mode1_w3_sr0.9.log` |
| alley_1 | 6 | 2 | - | 1 | 0.0297832401743581 | 0.0091596841183846 | 3.1096421834837424 | 0.0064900780845984 | 6.863204 | 0.248962 | 3 | `Python/logs/sintel_xm/solver/alley_1_setting6_mode2_w1.log` |
| alley_2 | 1 | 1 | 0.1 | 1 | 0.1101524215655258 | 0.0141556085872427 | 3.0896368712395175 | 0.0106135194725239 | 15.497910 | 0.371734 | 3 | `Python/logs/sintel_xm/solver/alley_2_setting1_mode1_w1_sr0.1.log` |
| alley_2 | 2 | 1 | 0.3 | 1 | 0.1103641816220384 | 0.0143537759237129 | 2.6545259964883048 | 0.0109771906168243 | 16.013114 | 0.432308 | 3 | `Python/logs/sintel_xm/solver/alley_2_setting2_mode1_w1_sr0.3.log` |
| alley_2 | 3 | 1 | 0.5 | 1 | 0.1099648333061249 | 0.0130781767309021 | 2.9676229733340227 | 0.0099921070625675 | 12.318832 | 0.489244 | 3 | `Python/logs/sintel_xm/solver/alley_2_setting3_mode1_w1_sr0.5.log` |
| alley_2 | 4 | 1 | 0.9 | 2 | 0.1084389757846860 | 0.0091112615990918 | 2.9015674514859469 | 0.0067970229944918 | 11.320384 | 1.287493 | 3 | `Python/logs/sintel_xm/solver/alley_2_setting4_mode1_w2_sr0.9.log` |
| alley_2 | 5 | 1 | 0.9 | 3 | 0.1087501304916527 | 0.0106055775035345 | 2.4945496784186663 | 0.0080234984240780 | 13.899952 | 0.871767 | 3 | `Python/logs/sintel_xm/solver/alley_2_setting5_mode1_w3_sr0.9.log` |
| alley_2 | 6 | 2 | - | 1 | 0.1083251077732383 | 0.0087021989290647 | 1.5580738679316242 | 0.0063229003733639 | 12.292626 | 0.254320 | 3 | `Python/logs/sintel_xm/solver/alley_2_setting6_mode2_w1.log` |
| ambush_2 | 1 | 1 | 0.1 | 1 | 0.2884312050150852 | 0.0000090024941586 | 2.5796504116517065 | 0.0000057058238873 | 5.114124 | 2.654206 | 3 | `Python/logs/sintel_xm/solver/ambush_2_setting1_mode1_w1_sr0.1.log` |
| ambush_2 | 2 | 1 | 0.3 | 1 | 0.4905003300031704 | 0.0000079090823513 | 0.9787618494857616 | 0.0000050624452959 | 6.443954 | 3.963848 | 3 | `Python/logs/sintel_xm/solver/ambush_2_setting2_mode1_w1_sr0.3.log` |
| ambush_2 | 3 | 1 | 0.5 | 1 | 0.4640893502662661 | 0.0000068188392820 | 0.9359532938194439 | 0.0000044725717268 | 7.318073 | 4.614384 | 3 | `Python/logs/sintel_xm/solver/ambush_2_setting3_mode1_w1_sr0.5.log` |
| ambush_2 | 4 | 1 | 0.9 | 2 | 0.0131909512810557 | 0.0000038768838962 | 2.2701251834607326 | 0.0000026854353851 | 3.568928 | 0.734491 | 3 | `Python/logs/sintel_xm/solver/ambush_2_setting4_mode1_w2_sr0.9.log` |
| ambush_2 | 5 | 1 | 0.9 | 3 | 0.1496993128685302 | 0.0000039377020300 | 0.5461083641359088 | 0.0000027191718640 | 3.605079 | 1.277213 | 3 | `Python/logs/sintel_xm/solver/ambush_2_setting5_mode1_w3_sr0.9.log` |
| ambush_2 | 6 | 2 | - | 1 | 0.4645741862670155 | 0.0000075866825256 | 0.9461587479580497 | 0.0000048816162399 | 3.886057 | 1.334058 | 3 | `Python/logs/sintel_xm/solver/ambush_2_setting6_mode2_w1.log` |
| ambush_4 | 1 | 1 | 0.1 | 1 | 0.0152883175381059 | 0.0618897516052890 | 1.7810453403648741 | 0.0441060690218139 | 8.575825 | 0.438235 | 3 | `Python/logs/sintel_xm/solver/ambush_4_setting1_mode1_w1_sr0.1.log` |
| ambush_4 | 2 | 1 | 0.3 | 1 | 0.0138947377347441 | 0.0630868570597777 | 1.7608187466811114 | 0.0447751420077075 | 8.185979 | 0.531787 | 3 | `Python/logs/sintel_xm/solver/ambush_4_setting2_mode1_w1_sr0.3.log` |
| ambush_4 | 3 | 1 | 0.5 | 1 | 0.0033332127202988 | 0.0197667235255174 | 0.0812574432123370 | 0.0122788485923693 | 17.244104 | 0.587711 | 3 | `Python/logs/sintel_xm/solver/ambush_4_setting3_mode1_w1_sr0.5.log` |
| ambush_4 | 4 | 1 | 0.9 | 2 | 0.0027437409923676 | 0.0136242228699061 | 0.0959677277755814 | 0.0089442952683529 | 11.716736 | 0.694455 | 3 | `Python/logs/sintel_xm/solver/ambush_4_setting4_mode1_w2_sr0.9.log` |
| ambush_4 | 5 | 1 | 0.9 | 3 | 0.0089038400909673 | 0.0587622686830239 | 2.9511657856697626 | 0.0419542256636283 | 8.335664 | 0.648682 | 3 | `Python/logs/sintel_xm/solver/ambush_4_setting5_mode1_w3_sr0.9.log` |
| ambush_4 | 6 | 2 | - | 1 | 0.0170590217425478 | 0.1200909767226822 | 2.6033659742461510 | 0.0779232874996289 | 4.673574 | 0.270521 | 3 | `Python/logs/sintel_xm/solver/ambush_4_setting6_mode2_w1.log` |
| ambush_5 | 1 | 1 | 0.1 | 1 | 0.0500272750971593 | 0.1536144768825550 | 2.1711742121051731 | 0.1009667455548066 | 6.374204 | 0.450194 | 3 | `Python/logs/sintel_xm/solver/ambush_5_setting1_mode1_w1_sr0.1.log` |
| ambush_5 | 2 | 1 | 0.3 | 1 | 0.0460805887985078 | 0.0773190960099906 | 1.9685942778742773 | 0.0499566846585327 | 6.593029 | 0.547123 | 3 | `Python/logs/sintel_xm/solver/ambush_5_setting2_mode1_w1_sr0.3.log` |
| ambush_5 | 3 | 1 | 0.5 | 1 | 0.0429973849022091 | 0.0463939876287229 | 1.6456028064361705 | 0.0288391106217226 | 7.124338 | 0.677409 | 3 | `Python/logs/sintel_xm/solver/ambush_5_setting3_mode1_w1_sr0.5.log` |
| ambush_5 | 4 | 1 | 0.9 | 2 | 0.0343043591810091 | 0.0491548046703386 | 1.9278793570754442 | 0.0330882543241077 | 8.227719 | 0.758152 | 3 | `Python/logs/sintel_xm/solver/ambush_5_setting4_mode1_w2_sr0.9.log` |
| ambush_5 | 5 | 1 | 0.9 | 3 | 0.0389538525934300 | 0.0463566080316732 | 1.4002855180830271 | 0.0315560154695795 | 7.080756 | 0.863155 | 3 | `Python/logs/sintel_xm/solver/ambush_5_setting5_mode1_w3_sr0.9.log` |
| ambush_5 | 6 | 2 | - | 1 | 0.0519180146649432 | 0.1027031830820949 | 0.2123299723985867 | 0.0678224497167691 | 5.481970 | 0.279294 | 3 | `Python/logs/sintel_xm/solver/ambush_5_setting6_mode2_w1.log` |
| ambush_6 | 1 | 1 | 0.1 | 1 | 0.1177887374030433 | 0.2599210402717753 | 0.7953391336905324 | 0.1787079310098066 | 3.514525 | 0.302397 | 3 | `Python/logs/sintel_xm/solver/ambush_6_setting1_mode1_w1_sr0.1.log` |
| ambush_6 | 2 | 1 | 0.3 | 1 | 0.1121488946705786 | 0.2634063705304751 | 2.7291037842923944 | 0.1764206446963696 | 3.817650 | 0.319932 | 3 | `Python/logs/sintel_xm/solver/ambush_6_setting2_mode1_w1_sr0.3.log` |
| ambush_6 | 3 | 1 | 0.5 | 1 | 0.1133291687609414 | 0.2739250649764858 | 2.8720388076594583 | 0.2139237069031955 | 3.449974 | 0.339038 | 3 | `Python/logs/sintel_xm/solver/ambush_6_setting3_mode1_w1_sr0.5.log` |
| ambush_6 | 4 | 1 | 0.9 | 2 | 0.0371792393972152 | 0.0863183399223606 | 0.8525467101150550 | 0.0545268707270543 | 3.900747 | 0.407629 | 3 | `Python/logs/sintel_xm/solver/ambush_6_setting4_mode1_w2_sr0.9.log` |
| ambush_6 | 5 | 1 | 0.9 | 3 | 0.0386686453083263 | 0.0819440045231347 | 0.7382236575277641 | 0.0514220891839506 | 3.739491 | 0.384980 | 3 | `Python/logs/sintel_xm/solver/ambush_6_setting5_mode1_w3_sr0.9.log` |
| ambush_6 | 6 | 2 | - | 1 | 0.1053578683816331 | 0.3614024251961649 | 2.5716275720907933 | 0.2678671296193031 | 4.335551 | 0.242054 | 3 | `Python/logs/sintel_xm/solver/ambush_6_setting6_mode2_w1.log` |
| ambush_7 | 1 | 1 | 0.1 | 1 | 0.0098831092717370 | 0.0000052361233342 | 1.3714801828898060 | 195.9990769694493906 | 9.680256 | 0.349834 | 3 | `Python/logs/sintel_xm/solver/ambush_7_setting1_mode1_w1_sr0.1.log` |
| ambush_7 | 2 | 1 | 0.3 | 1 | 0.0097610743759368 | 0.0000052401506231 | 1.3682799986046090 | 0.0000037082862919 | 7.864398 | 0.410859 | 3 | `Python/logs/sintel_xm/solver/ambush_7_setting2_mode1_w1_sr0.3.log` |
| ambush_7 | 3 | 1 | 0.5 | 1 | 0.0097881572189583 | 0.0000052432604743 | 1.3620443317036943 | 0.0000037384831932 | 8.199776 | 0.472409 | 3 | `Python/logs/sintel_xm/solver/ambush_7_setting3_mode1_w1_sr0.5.log` |
| ambush_7 | 4 | 1 | 0.9 | 2 | 0.0078868005004118 | 0.0000052679931097 | 1.4314447083723929 | 0.0000037745827958 | 31.012325 | 0.431062 | 3 | `Python/logs/sintel_xm/solver/ambush_7_setting4_mode1_w2_sr0.9.log` |
| ambush_7 | 5 | 1 | 0.9 | 3 | 0.0068707962486540 | 0.0000052761756742 | 1.6321691712558235 | 0.0000038149990496 | 19.429246 | 0.481478 | 3 | `Python/logs/sintel_xm/solver/ambush_7_setting5_mode1_w3_sr0.9.log` |
| ambush_7 | 6 | 2 | - | 1 | 0.0088210488633349 | 0.0000051937469064 | 1.3590998437058750 | 0.0000036870549697 | 9.018865 | 0.248487 | 3 | `Python/logs/sintel_xm/solver/ambush_7_setting6_mode2_w1.log` |
| bamboo_1 | 1 | 1 | 0.1 | 1 | 0.0019323892687612 | 0.0014313633847325 | 2.3957053572141915 | 0.0010190441731412 | 16.858996 | 0.266395 | 3 | `Python/logs/sintel_xm/solver/bamboo_1_setting1_mode1_w1_sr0.1.log` |
| bamboo_1 | 2 | 1 | 0.3 | 1 | 0.0017587740041019 | 0.0010928642508645 | 3.1060479983062965 | 0.0007715453465742 | 68.621158 | 0.305647 | 3 | `Python/logs/sintel_xm/solver/bamboo_1_setting2_mode1_w1_sr0.3.log` |
| bamboo_1 | 3 | 1 | 0.5 | 1 | 0.0016675031679045 | 0.0010259372598195 | 2.6955890750709157 | 0.0007088445014212 | 22.664635 | 0.323806 | 3 | `Python/logs/sintel_xm/solver/bamboo_1_setting3_mode1_w1_sr0.5.log` |
| bamboo_1 | 4 | 1 | 0.9 | 2 | 0.0021521179525870 | 0.0011037539256745 | 2.3225715115485563 | 0.0007875386733895 | 132.637878 | 0.329903 | 3 | `Python/logs/sintel_xm/solver/bamboo_1_setting4_mode1_w2_sr0.9.log` |
| bamboo_1 | 5 | 1 | 0.9 | 3 | 0.0015041273952465 | 0.0009657980730777 | 2.1594258100347581 | 0.0006681914341852 | 28.982831 | 0.377327 | 3 | `Python/logs/sintel_xm/solver/bamboo_1_setting5_mode1_w3_sr0.9.log` |
| bamboo_1 | 6 | 2 | - | 1 | 0.0221210584457776 | 0.0395614878395330 | 1.3699142152060348 | 0.0242421207422696 | 5.946002 | 0.236630 | 3 | `Python/logs/sintel_xm/solver/bamboo_1_setting6_mode2_w1.log` |
| bamboo_2 | 1 | 1 | 0.1 | 1 | 0.0062677544172624 | 0.0019615069747166 | 2.6192671166977641 | 0.0014257844601582 | 9.425389 | 0.280964 | 3 | `Python/logs/sintel_xm/solver/bamboo_2_setting1_mode1_w1_sr0.1.log` |
| bamboo_2 | 2 | 1 | 0.3 | 1 | 0.0083877324978398 | 0.0018496981204367 | 0.5242196722518724 | 0.0013469238329654 | 10.061150 | 0.319239 | 3 | `Python/logs/sintel_xm/solver/bamboo_2_setting2_mode1_w1_sr0.3.log` |
| bamboo_2 | 3 | 1 | 0.5 | 1 | 0.0063290727983261 | 0.0017566078148884 | 0.5244502573196319 | 0.0012841009431741 | 12.576018 | 0.329458 | 3 | `Python/logs/sintel_xm/solver/bamboo_2_setting3_mode1_w1_sr0.5.log` |
| bamboo_2 | 4 | 1 | 0.9 | 2 | 0.0077080604355000 | 0.0015016824928143 | 0.5229371559670436 | 0.0010714544283037 | 11.899651 | 0.440771 | 3 | `Python/logs/sintel_xm/solver/bamboo_2_setting4_mode1_w2_sr0.9.log` |
| bamboo_2 | 5 | 1 | 0.9 | 3 | 0.0062383654291749 | 0.0016047120103157 | 0.5244544266535862 | 0.0011689257479668 | 9.914754 | 0.394110 | 3 | `Python/logs/sintel_xm/solver/bamboo_2_setting5_mode1_w3_sr0.9.log` |
| bamboo_2 | 6 | 2 | - | 1 | 0.0000403383976593 | 0.0013191544772385 | 0.5242411952712298 | 0.0009396091492490 | 11.820381 | 0.229697 | 3 | `Python/logs/sintel_xm/solver/bamboo_2_setting6_mode2_w1.log` |
| bandage_1 | 1 | 1 | 0.1 | 1 | 0.3731602805521514 | 0.1981425917211166 | 2.0971604948491263 | 0.1550869621438087 | 4.461974 | 0.334830 | 3 | `Python/logs/sintel_xm_part2/solver/bandage_1_setting1_mode1_w1_sr0.1.log` |
| bandage_1 | 2 | 1 | 0.3 | 1 | 0.4925331061215026 | 0.3483771966155403 | 2.8475399680650302 | 0.2714744575190676 | 4.291049 | 0.391998 | 3 | `Python/logs/sintel_xm_part2/solver/bandage_1_setting2_mode1_w1_sr0.3.log` |
| bandage_1 | 3 | 1 | 0.5 | 1 | 0.4677099761757409 | 0.3057037246297292 | 2.6413038221510288 | 0.2399112359091086 | 3.939912 | 0.485635 | 3 | `Python/logs/sintel_xm_part2/solver/bandage_1_setting3_mode1_w1_sr0.5.log` |
| bandage_1 | 4 | 1 | 0.9 | 2 | 0.3900185251633387 | 0.0060976178026079 | 2.8839808382348902 | 0.0046100840154679 | 4.764042 | 0.415468 | 3 | `Python/logs/sintel_xm_part2/solver/bandage_1_setting4_mode1_w2_sr0.9.log` |
| bandage_1 | 5 | 1 | 0.9 | 3 | 0.4312743089759618 | 0.2379747942812350 | 1.1550897955925186 | 0.1875794829855397 | 4.391923 | 0.437320 | 3 | `Python/logs/sintel_xm_part2/solver/bandage_1_setting5_mode1_w3_sr0.9.log` |
| bandage_1 | 6 | 2 | - | 1 | 0.3785049887385402 | 0.3339677491455447 | 2.6821558150366123 | 0.2564035736189237 | 4.512023 | 0.256407 | 3 | `Python/logs/sintel_xm_part2/solver/bandage_1_setting6_mode2_w1.log` |
| bandage_2 | 1 | 1 | 0.1 | 1 | 0.0008456103351255 | 0.0371249827415492 | 3.0527635081402935 | 0.0280505911497685 | 4.008616 | 0.310984 | 3 | `Python/logs/sintel_xm_part2/solver/bandage_2_setting1_mode1_w1_sr0.1.log` |
| bandage_2 | 2 | 1 | 0.3 | 1 | 0.0007361525007736 | 0.0411527531919943 | 2.6517193246092119 | 0.0312739538878268 | 4.299575 | 0.352163 | 3 | `Python/logs/sintel_xm_part2/solver/bandage_2_setting2_mode1_w1_sr0.3.log` |
| bandage_2 | 3 | 1 | 0.5 | 1 | 0.0008301886987933 | 0.0431185255656559 | 2.1548077885742205 | 0.0324794796785458 | 4.038184 | 0.384980 | 3 | `Python/logs/sintel_xm_part2/solver/bandage_2_setting3_mode1_w1_sr0.5.log` |
| bandage_2 | 4 | 1 | 0.9 | 2 | 0.0008414145792081 | 0.0042294272892825 | 2.5781512254778001 | 0.0032610272271211 | 5.117499 | 0.364372 | 3 | `Python/logs/sintel_xm_part2/solver/bandage_2_setting4_mode1_w2_sr0.9.log` |
| bandage_2 | 5 | 1 | 0.9 | 3 | 0.0007608026954452 | 0.0122882687645643 | 1.1348725514710123 | 0.0093238203362787 | 4.894975 | 0.378134 | 3 | `Python/logs/sintel_xm_part2/solver/bandage_2_setting5_mode1_w3_sr0.9.log` |
| bandage_2 | 6 | 2 | - | 1 | 0.0007619181002594 | 0.0647388984036205 | 3.0523941537783896 | 0.0484534868159075 | 4.088749 | 0.244716 | 3 | `Python/logs/sintel_xm_part2/solver/bandage_2_setting6_mode2_w1.log` |
| cave_2 | 1 | 1 | 0.1 | 1 | 0.4734329891018725 | 0.5688213928276860 | 2.8401354355295871 | 0.4412557869062882 | 11.831589 | 2.567806 | 3 | `Python/logs/sintel_xm_part2/solver/cave_2_setting1_mode1_w1_sr0.1.log` |
| cave_2 | 2 | 1 | 0.3 | 1 | 0.7578978998403917 | 0.5824814371480076 | 2.5993183319540005 | 0.5075965217230466 | 8.204800 | 3.271310 | 3 | `Python/logs/sintel_xm_part2/solver/cave_2_setting2_mode1_w1_sr0.3.log` |
| cave_2 | 3 | 1 | 0.5 | 1 | 0.4870784992228625 | 0.5670130563026007 | 3.0561787558166120 | 0.4590200780672190 | 9.965357 | 2.230379 | 3 | `Python/logs/sintel_xm_part2/solver/cave_2_setting3_mode1_w1_sr0.5.log` |
| cave_2 | 4 | 1 | 0.9 | 2 | 0.5697365876194000 | 0.2092293891102895 | 2.7494632239612340 | 0.1485984763161777 | 24.706808 | 1.699647 | 3 | `Python/logs/sintel_xm_part2/solver/cave_2_setting4_mode1_w2_sr0.9.log` |
| cave_2 | 5 | 1 | 0.9 | 3 | 0.5290709903080564 | 0.2792974119988312 | 2.5769245144035606 | 0.2012063296049995 | 16.895310 | 1.854431 | 3 | `Python/logs/sintel_xm_part2/solver/cave_2_setting5_mode1_w3_sr0.9.log` |
| cave_2 | 6 | 2 | - | 1 | 0.8896435028978833 | 0.7855945503579425 | 2.5561499992681083 | 0.6165361128693453 | 5.335823 | 0.811173 | 4 | `Python/logs/sintel_xm_part2/solver/cave_2_setting6_mode2_w1.log` |
| cave_4 | 1 | 1 | 0.1 | 1 | 0.0064489037811742 | 0.0072315850279401 | 0.0491781988943372 | 0.0050827833098984 | 17.358359 | 0.322479 | 3 | `Python/logs/sintel_xm_part2/solver/cave_4_setting1_mode1_w1_sr0.1.log` |
| cave_4 | 2 | 1 | 0.3 | 1 | 0.0033776393383998 | 0.0061187849069481 | 0.0464803295659657 | 0.0043114524550262 | 6.397042 | 0.373029 | 3 | `Python/logs/sintel_xm_part2/solver/cave_4_setting2_mode1_w1_sr0.3.log` |
| cave_4 | 3 | 1 | 0.5 | 1 | 0.0014954673083550 | 0.0057637750942148 | 0.0521547291499717 | 0.0040212978714112 | 12.316287 | 0.389481 | 3 | `Python/logs/sintel_xm_part2/solver/cave_4_setting3_mode1_w1_sr0.5.log` |
| cave_4 | 4 | 1 | 0.9 | 2 | 0.0062038128054094 | 0.0119564685255163 | 0.0928575781324571 | 0.0086322897318990 | 4.087155 | 0.523194 | 3 | `Python/logs/sintel_xm_part2/solver/cave_4_setting4_mode1_w2_sr0.9.log` |
| cave_4 | 5 | 1 | 0.9 | 3 | 0.0046513548667269 | 0.0090685654916473 | 0.0734665919150020 | 0.0064602832576565 | 9.167188 | 0.454087 | 3 | `Python/logs/sintel_xm_part2/solver/cave_4_setting5_mode1_w3_sr0.9.log` |
| cave_4 | 6 | 2 | - | 1 | 0.0049502347451912 | 0.0310294543906506 | 0.0241836534827929 | 0.0182298802518021 | 3.856523 | 0.251855 | 3 | `Python/logs/sintel_xm_part2/solver/cave_4_setting6_mode2_w1.log` |
| market_2 | 1 | 1 | 0.1 | 1 | 0.0007173659834431 | 0.0083877506690898 | 0.1521326230067831 | 0.0063366896075705 | 4.381187 | 0.333822 | 3 | `Python/logs/sintel_xm_part2/solver/market_2_setting1_mode1_w1_sr0.1.log` |
| market_2 | 2 | 1 | 0.3 | 1 | 0.0007908448827283 | 0.0082070589789800 | 0.1712914198126397 | 0.0061688315446102 | 4.433890 | 0.404369 | 3 | `Python/logs/sintel_xm_part2/solver/market_2_setting2_mode1_w1_sr0.3.log` |
| market_2 | 3 | 1 | 0.5 | 1 | 0.0007937343587533 | 0.0080084041798207 | 0.1489699629150963 | 0.0060471297752610 | 4.526297 | 0.433800 | 3 | `Python/logs/sintel_xm_part2/solver/market_2_setting3_mode1_w1_sr0.5.log` |
| market_2 | 4 | 1 | 0.9 | 2 | 0.0005931280832714 | 0.0079905064726214 | 0.1486139833826051 | 0.0059253078490166 | 5.537876 | 0.430118 | 3 | `Python/logs/sintel_xm_part2/solver/market_2_setting4_mode1_w2_sr0.9.log` |
| market_2 | 5 | 1 | 0.9 | 3 | 0.0005959812027676 | 0.0083003969146422 | 0.0605165231725106 | 0.0061497340385010 | 5.684625 | 0.483596 | 3 | `Python/logs/sintel_xm_part2/solver/market_2_setting5_mode1_w3_sr0.9.log` |
| market_2 | 6 | 2 | - | 1 | 0.0006378054584083 | 0.0072922738603903 | 0.3609505980045897 | 0.0054169757389923 | 4.452606 | 0.260501 | 3 | `Python/logs/sintel_xm_part2/solver/market_2_setting6_mode2_w1.log` |
| market_5 | 1 | 1 | 0.1 | 1 | 0.0677734594557090 | 0.0042281750466437 | 1.7905620063483871 | 0.0029784348802018 | 9.642363 | 0.991741 | 3 | `Python/logs/sintel_xm_part2/solver/market_5_setting1_mode1_w1_sr0.1.log` |
| market_5 | 2 | 1 | 0.3 | 1 | 0.0775743062777305 | 0.0073847502205810 | 1.7038492767255948 | 0.0055515937596794 | 6.845854 | 1.517140 | 3 | `Python/logs/sintel_xm_part2/solver/market_5_setting2_mode1_w1_sr0.3.log` |
| market_5 | 3 | 1 | 0.5 | 1 | 0.0681989874160325 | 0.0042632606986973 | 1.7904782773118209 | 0.0029908103228926 | 6.061154 | 1.674375 | 3 | `Python/logs/sintel_xm_part2/solver/market_5_setting3_mode1_w1_sr0.5.log` |
| market_5 | 4 | 1 | 0.9 | 2 | 0.0783887390666660 | 0.0195845528085028 | 2.9642576358515775 | 0.0144949914229440 | 7.798477 | 2.027760 | 3 | `Python/logs/sintel_xm_part2/solver/market_5_setting4_mode1_w2_sr0.9.log` |
| market_5 | 5 | 1 | 0.9 | 3 | 0.0805853108161510 | 0.0049225381987306 | 2.8158693399994066 | 0.0035424112158444 | 8.012741 | 2.422414 | 3 | `Python/logs/sintel_xm_part2/solver/market_5_setting5_mode1_w3_sr0.9.log` |
| market_5 | 6 | 2 | - | 1 | 0.0675957692226655 | 0.0031328784489829 | 1.6690467985104467 | 0.0020913647706184 | 5.678586 | 0.483357 | 3 | `Python/logs/sintel_xm_part2/solver/market_5_setting6_mode2_w1.log` |
| market_6 | 1 | 1 | 0.1 | 1 | 0.0153179774240878 | 0.0197897832754392 | 2.1033953451888214 | 0.0146409390664668 | 4.109131 | 0.327237 | 3 | `Python/logs/sintel_xm_part2/solver/market_6_setting1_mode1_w1_sr0.1.log` |
| market_6 | 2 | 1 | 0.3 | 1 | 0.0173774995089928 | 0.0212826692412387 | 2.1896712799569960 | 0.0160546699255729 | 5.090848 | 0.357112 | 3 | `Python/logs/sintel_xm_part2/solver/market_6_setting2_mode1_w1_sr0.3.log` |
| market_6 | 3 | 1 | 0.5 | 1 | 0.0267161368333538 | 0.0257202616722272 | 2.3615690909871825 | 0.0187223863408414 | 5.058274 | 0.372784 | 3 | `Python/logs/sintel_xm_part2/solver/market_6_setting3_mode1_w1_sr0.5.log` |
| market_6 | 4 | 1 | 0.9 | 2 | 0.0392340180900667 | 0.0592182591527091 | 2.3831474535054920 | 0.0425386298409958 | 4.892819 | 0.520975 | 3 | `Python/logs/sintel_xm_part2/solver/market_6_setting4_mode1_w2_sr0.9.log` |
| market_6 | 5 | 1 | 0.9 | 3 | 0.0327802389450793 | 0.0263698389078215 | 2.5031144333148379 | 0.0184922056492878 | 4.747129 | 0.460725 | 3 | `Python/logs/sintel_xm_part2/solver/market_6_setting5_mode1_w3_sr0.9.log` |
| market_6 | 6 | 2 | - | 1 | 0.0199084016654955 | 0.0203232710195458 | 2.2228013158780855 | 0.0149270152866248 | 4.702069 | 0.251016 | 3 | `Python/logs/sintel_xm_part2/solver/market_6_setting6_mode2_w1.log` |
| mountain_1 | 1 | 1 | 0.1 | 1 | 0.0009585450786654 | 0.0084336960969892 | 0.0273551634616203 | 0.0059018805561337 | 5.586595 | 0.321418 | 3 | `Python/logs/sintel_xm_part2/solver/mountain_1_setting1_mode1_w1_sr0.1.log` |
| mountain_1 | 2 | 1 | 0.3 | 1 | 0.0009641532107453 | 0.0080853941463099 | 0.0326686284133566 | 0.0055779347299114 | 5.444448 | 0.355667 | 3 | `Python/logs/sintel_xm_part2/solver/mountain_1_setting2_mode1_w1_sr0.3.log` |
| mountain_1 | 3 | 1 | 0.5 | 1 | 0.0009211410272442 | 0.0078510970472268 | 0.0322954926784156 | 0.0055734998019448 | 5.533419 | 0.395469 | 3 | `Python/logs/sintel_xm_part2/solver/mountain_1_setting3_mode1_w1_sr0.5.log` |
| mountain_1 | 4 | 1 | 0.9 | 2 | 0.0011708359974697 | 0.0090022697549525 | 0.0731341705239566 | 0.0064631437593187 | 7.029169 | 0.439190 | 3 | `Python/logs/sintel_xm_part2/solver/mountain_1_setting4_mode1_w2_sr0.9.log` |
| mountain_1 | 5 | 1 | 0.9 | 3 | 0.0082881981637053 | 0.0333338718123892 | 0.6035778193798630 | 0.0215888032592614 | 5.898178 | 0.465777 | 3 | `Python/logs/sintel_xm_part2/solver/mountain_1_setting5_mode1_w3_sr0.9.log` |
| mountain_1 | 6 | 2 | - | 1 | 0.0009732924236007 | 0.0074634451058657 | 0.0584580846052275 | 0.0053935314548903 | 6.635460 | 0.238843 | 3 | `Python/logs/sintel_xm_part2/solver/mountain_1_setting6_mode2_w1.log` |
| shaman_2 | 1 | 1 | 0.1 | 1 | 0.0235078309041239 | 0.0590666536264298 | 2.8750418826931274 | 0.0458953499869918 | 7.395814 | 0.334865 | 3 | `Python/logs/sintel_xm_part2/solver/shaman_2_setting1_mode1_w1_sr0.1.log` |
| shaman_2 | 2 | 1 | 0.3 | 1 | 0.0234941737195768 | 0.0583118158816860 | 2.6460425529929137 | 0.0453021547852212 | 7.111772 | 0.389914 | 3 | `Python/logs/sintel_xm_part2/solver/shaman_2_setting2_mode1_w1_sr0.3.log` |
| shaman_2 | 3 | 1 | 0.5 | 1 | 0.0237819725702213 | 0.0568228676774866 | 1.8261709846711518 | 0.0441617019261818 | 4.926064 | 0.463050 | 3 | `Python/logs/sintel_xm_part2/solver/shaman_2_setting3_mode1_w1_sr0.5.log` |
| shaman_2 | 4 | 1 | 0.9 | 2 | 0.0248597960120527 | 0.0269173143542896 | 2.8786645754590139 | 0.0208367828136684 | 16.238194 | 0.401690 | 3 | `Python/logs/sintel_xm_part2/solver/shaman_2_setting4_mode1_w2_sr0.9.log` |
| shaman_2 | 5 | 1 | 0.9 | 3 | 0.0242901546145390 | 0.0447236980152402 | 1.9670219400324132 | 0.0347069492954857 | 5.997492 | 0.504528 | 3 | `Python/logs/sintel_xm_part2/solver/shaman_2_setting5_mode1_w3_sr0.9.log` |
| shaman_2 | 6 | 2 | - | 1 | 0.0228558921137683 | 0.0539527855560067 | 2.4777010194227249 | 0.0419201146694704 | 13.993744 | 0.240283 | 3 | `Python/logs/sintel_xm_part2/solver/shaman_2_setting6_mode2_w1.log` |
| shaman_3 | 1 | 1 | 0.1 | 1 | 0.0127214325118984 | 0.0037288684795104 | 0.8039298256752972 | 0.0027681967589873 | 6.635391 | 0.361838 | 3 | `Python/logs/sintel_xm_part2/solver/shaman_3_setting1_mode1_w1_sr0.1.log` |
| shaman_3 | 2 | 1 | 0.3 | 1 | 0.0131352101718426 | 0.0037080975676540 | 1.2074185319612432 | 0.0027305453917381 | 5.851502 | 0.417738 | 3 | `Python/logs/sintel_xm_part2/solver/shaman_3_setting2_mode1_w1_sr0.3.log` |
| shaman_3 | 3 | 1 | 0.5 | 1 | 0.0127810395146310 | 0.0038080030206381 | 0.7138398446852747 | 0.0028295964356898 | 6.187434 | 0.489480 | 3 | `Python/logs/sintel_xm_part2/solver/shaman_3_setting3_mode1_w1_sr0.5.log` |
| shaman_3 | 4 | 1 | 0.9 | 2 | 0.0109091166246241 | 0.0064214161640488 | 2.3412846537912486 | 0.0046719648519130 | 5.780686 | 0.622330 | 3 | `Python/logs/sintel_xm_part2/solver/shaman_3_setting4_mode1_w2_sr0.9.log` |
| shaman_3 | 5 | 1 | 0.9 | 3 | 0.0177217385444964 | 0.0053370551857744 | 2.7864945801488163 | 0.0038814213298214 | 5.585238 | 0.657897 | 3 | `Python/logs/sintel_xm_part2/solver/shaman_3_setting5_mode1_w3_sr0.9.log` |
| shaman_3 | 6 | 2 | - | 1 | 0.0169921263549277 | 0.0042201089716799 | 2.5376241663451768 | 0.0029307104478411 | 8.355278 | 0.264418 | 3 | `Python/logs/sintel_xm_part2/solver/shaman_3_setting6_mode2_w1.log` |
| sleeping_1 | 1 | 1 | 0.1 | 1 | 0.0458174867385237 | 0.0055697098748992 | 2.6626572489465361 | 0.0040065583790262 | 7.513967 | 0.305482 | 3 | `Python/logs/sintel_xm_part2/solver/sleeping_1_setting1_mode1_w1_sr0.1.log` |
| sleeping_1 | 2 | 1 | 0.3 | 1 | 0.0458756363844193 | 0.0056570666550633 | 2.6177340520885428 | 0.0040563863141830 | 7.071363 | 0.354912 | 3 | `Python/logs/sintel_xm_part2/solver/sleeping_1_setting2_mode1_w1_sr0.3.log` |
| sleeping_1 | 3 | 1 | 0.5 | 1 | 0.0458998716135170 | 0.0056454047461863 | 2.5842818221451744 | 0.0040535596157636 | 5.836153 | 0.401815 | 3 | `Python/logs/sintel_xm_part2/solver/sleeping_1_setting3_mode1_w1_sr0.5.log` |
| sleeping_1 | 4 | 1 | 0.9 | 2 | 0.0876703414672824 | 0.0086731946852111 | 2.9066463489700420 | 0.0056053071506918 | 4.429562 | 0.441243 | 3 | `Python/logs/sintel_xm_part2/solver/sleeping_1_setting4_mode1_w2_sr0.9.log` |
| sleeping_1 | 5 | 1 | 0.9 | 3 | 0.0212008638781247 | 0.0132136451245908 | 2.1212044859927968 | 0.0081396400636638 | 4.502293 | 0.554281 | 3 | `Python/logs/sintel_xm_part2/solver/sleeping_1_setting5_mode1_w3_sr0.9.log` |
| sleeping_1 | 6 | 2 | - | 1 | 0.0468378350285141 | 0.0061739726821249 | 2.7786210003386356 | 0.0044101830105928 | 33.988362 | 0.264169 | 3 | `Python/logs/sintel_xm_part2/solver/sleeping_1_setting6_mode2_w1.log` |
| sleeping_2 | 1 | 1 | 0.1 | 1 | 0.0004431087387020 | 0.0006520639284392 | 0.0145264676759639 | 0.0004426803601207 | 8.902128 | 0.254498 | 3 | `Python/logs/sintel_xm_part2/solver/sleeping_2_setting1_mode1_w1_sr0.1.log` |
| sleeping_2 | 2 | 1 | 0.3 | 1 | 0.0004252237410600 | 0.0006577868830908 | 0.0145545882366762 | 0.0004476344066968 | 50.065434 | 0.287381 | 3 | `Python/logs/sintel_xm_part2/solver/sleeping_2_setting2_mode1_w1_sr0.3.log` |
| sleeping_2 | 3 | 1 | 0.5 | 1 | 0.0004161727044103 | 0.0006592332385231 | 0.0146761484986842 | 0.0004483562067116 | 18.601030 | 0.288130 | 3 | `Python/logs/sintel_xm_part2/solver/sleeping_2_setting3_mode1_w1_sr0.5.log` |
| sleeping_2 | 4 | 1 | 0.9 | 2 | 0.0004371456056440 | 0.0006817193853739 | 0.0567455929020111 | 0.0004792458003362 | 48.928439 | 0.386686 | 3 | `Python/logs/sintel_xm_part2/solver/sleeping_2_setting4_mode1_w2_sr0.9.log` |
| sleeping_2 | 5 | 1 | 0.9 | 3 | 0.0003771689429098 | 0.0006868625239602 | 0.0160924201824658 | 0.0004645164942285 | 11.520516 | 0.375399 | 3 | `Python/logs/sintel_xm_part2/solver/sleeping_2_setting5_mode1_w3_sr0.9.log` |
| sleeping_2 | 6 | 2 | - | 1 | 0.0004490178796629 | 0.0007139484831809 | 0.0150374383327402 | 0.0004780664765633 | 21.207114 | 0.241888 | 3 | `Python/logs/sintel_xm_part2/solver/sleeping_2_setting6_mode2_w1.log` |
| temple_2 | 1 | 1 | 0.1 | 1 | 0.1071311469217781 | 0.1274312463423845 | 0.5266629109026727 | 0.0892500039736206 | 9.777613 | 0.388438 | 3 | `Python/logs/sintel_xm_part2/solver/temple_2_setting1_mode1_w1_sr0.1.log` |
| temple_2 | 2 | 1 | 0.3 | 1 | 0.1083234133784972 | 0.1334721445059649 | 0.4552767903265995 | 0.0948522627235327 | 9.768743 | 0.422741 | 3 | `Python/logs/sintel_xm_part2/solver/temple_2_setting2_mode1_w1_sr0.3.log` |
| temple_2 | 3 | 1 | 0.5 | 1 | 0.1083969168300822 | 0.1265364986563531 | 0.4161597478995301 | 0.0905317873176553 | 8.718434 | 0.474209 | 3 | `Python/logs/sintel_xm_part2/solver/temple_2_setting3_mode1_w1_sr0.5.log` |
| temple_2 | 4 | 1 | 0.9 | 2 | 0.0501843407490013 | 0.0365582256759603 | 0.2953109799028752 | 0.0259092175971500 | 17.639689 | 0.879815 | 3 | `Python/logs/sintel_xm_part2/solver/temple_2_setting4_mode1_w2_sr0.9.log` |
| temple_2 | 5 | 1 | 0.9 | 3 | 0.0595655569806807 | 0.0518651145785360 | 0.2912602098876534 | 0.0367237728243966 | 14.492592 | 0.752831 | 3 | `Python/logs/sintel_xm_part2/solver/temple_2_setting5_mode1_w3_sr0.9.log` |
| temple_2 | 6 | 2 | - | 1 | 0.0770718033924829 | 0.0644877167717606 | 0.4561629085407011 | 0.0455819555799982 | 10.964389 | 0.284400 | 3 | `Python/logs/sintel_xm_part2/solver/temple_2_setting6_mode2_w1.log` |
| temple_3 | 1 | 1 | 0.1 | 1 | 0.0041233589270044 | 0.2474550856822449 | 2.5307006508711529 | 0.1715180466524428 | 6.740010 | 1.201579 | 3 | `Python/logs/sintel_xm_part2/solver/temple_3_setting1_mode1_w1_sr0.1.log` |
| temple_3 | 2 | 1 | 0.3 | 1 | 0.0015978178701095 | 0.2563858047635879 | 0.3404435922971591 | 0.1739490472500639 | 9.729159 | 1.624834 | 3 | `Python/logs/sintel_xm_part2/solver/temple_3_setting2_mode1_w1_sr0.3.log` |
| temple_3 | 3 | 1 | 0.5 | 1 | 0.0061554658666632 | 0.2394520200846455 | 2.5957683241141321 | 0.1658607144583701 | 9.197801 | 2.132394 | 3 | `Python/logs/sintel_xm_part2/solver/temple_3_setting3_mode1_w1_sr0.5.log` |
| temple_3 | 4 | 1 | 0.9 | 2 | 0.0040826433023637 | 0.2135890153114079 | 2.3673460625552329 | 0.1526593682492960 | 11.473314 | 1.325938 | 3 | `Python/logs/sintel_xm_part2/solver/temple_3_setting4_mode1_w2_sr0.9.log` |
| temple_3 | 5 | 1 | 0.9 | 3 | 0.0076867215477757 | 0.4128230959966060 | 1.3254340280940105 | 0.2804043949615431 | 13.280538 | 1.790112 | 3 | `Python/logs/sintel_xm_part2/solver/temple_3_setting5_mode1_w3_sr0.9.log` |
| temple_3 | 6 | 2 | - | 1 | 0.0025542758238662 | 0.2535484867190524 | 2.2872424496217478 | 0.1729217246521908 | 6.593294 | 0.587155 | 3 | `Python/logs/sintel_xm_part2/solver/temple_3_setting6_mode2_w1.log` |
