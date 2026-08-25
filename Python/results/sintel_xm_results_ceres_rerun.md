# Sintel XM Experiment Results (with Ceres refinement)

Generated: `2026-04-21T23:11:46-04:00`

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
| alley_1 | 1 | 1 | 0.1 | 1 | yes | 0.0113963392372300 | 0.0084319641152165 | 2.7202280209447740 | 0.0058144158608932 | 11.487094 | 0.296316 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/alley_1_setting1_mode1_w1_sr0.1_ceres.log` |
| alley_1 | 2 | 1 | 0.3 | 1 | yes | 0.0047604345560703 | 0.0112633252266362 | 3.1041671799451320 | 0.0077589554889152 | 16.166857 | 0.357010 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/alley_1_setting2_mode1_w1_sr0.3_ceres.log` |
| alley_1 | 3 | 1 | 0.5 | 1 | yes | 0.0210294153481453 | 0.0067742129151300 | 2.9830527856274234 | 0.0047593373344585 | 13.908797 | 0.397898 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/alley_1_setting3_mode1_w1_sr0.5_ceres.log` |
| alley_1 | 4 | 1 | 0.9 | 2 | yes | 0.0305958400443800 | 0.0085920404713955 | 2.8952340686745037 | 0.0056972987232212 | 50.929272 | 0.408007 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/alley_1_setting4_mode1_w2_sr0.9_ceres.log` |
| alley_1 | 5 | 1 | 0.9 | 3 | yes | 0.0199211733620266 | 0.0103744232553037 | 2.8049790572118498 | 0.0071261123316430 | 33.284583 | 0.482165 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/alley_1_setting5_mode1_w3_sr0.9_ceres.log` |
| alley_1 | 6 | 2 | - | 1 | yes | 0.0264717404453319 | 0.0067777363982934 | 3.1153692008762475 | 0.0047992288993791 | 9.914633 | 0.246039 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/alley_1_setting6_mode2_w1_ceres.log` |
| alley_2 | 1 | 1 | 0.1 | 1 | yes | 0.1148573391705773 | 0.0244737011897597 | 1.8159182999626211 | 0.0177193293688120 | 21.135270 | 0.374605 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/alley_2_setting1_mode1_w1_sr0.1_ceres.log` |
| alley_2 | 2 | 1 | 0.3 | 1 | yes | 0.1150690758069241 | 0.0244871647217941 | 1.5879820292534794 | 0.0183936314181901 | 19.272411 | 0.440617 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/alley_2_setting2_mode1_w1_sr0.3_ceres.log` |
| alley_2 | 3 | 1 | 0.5 | 1 | yes | 0.1129046758230468 | 0.0235675848437453 | 1.9087132212327698 | 0.0174273572728288 | 19.072919 | 0.491064 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/alley_2_setting3_mode1_w1_sr0.5_ceres.log` |
| alley_2 | 4 | 1 | 0.9 | 2 | yes | 0.1095777826179603 | 0.0230211736171250 | 1.7689842218890910 | 0.0168781693404979 | 43.227214 | 1.285477 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/alley_2_setting4_mode1_w2_sr0.9_ceres.log` |
| alley_2 | 5 | 1 | 0.9 | 3 | yes | 0.1108607794056524 | 0.0257815368946273 | 2.8861759657623565 | 0.0190591368654730 | 18.062049 | 0.880710 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/alley_2_setting5_mode1_w3_sr0.9_ceres.log` |
| alley_2 | 6 | 2 | - | 1 | yes | 0.1058470233382572 | 0.0461739326814054 | 1.5440495798233362 | 0.0356150925423750 | 11.169651 | 0.263250 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/alley_2_setting6_mode2_w1_ceres.log` |
| ambush_2 | 1 | 1 | 0.1 | 1 | yes | 0.4356919402789986 | 0.0000097316775933 | 1.9727130408678970 | 0.0000064714756208 | 5.414404 | 2.648128 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/ambush_2_setting1_mode1_w1_sr0.1_ceres.log` |
| ambush_2 | 2 | 1 | 0.3 | 1 | yes | 0.4855577873621952 | 0.0000081832322679 | 2.1750993054401513 | 0.0000054458474081 | 6.627076 | 3.950590 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/ambush_2_setting2_mode1_w1_sr0.3_ceres.log` |
| ambush_2 | 3 | 1 | 0.5 | 1 | yes | 0.4793856703682395 | 0.0000065295734912 | 2.4831579661501260 | 0.0000044949693774 | 7.112121 | 4.610113 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/ambush_2_setting3_mode1_w1_sr0.5_ceres.log` |
| ambush_2 | 4 | 1 | 0.9 | 2 | yes | 0.0129668005753726 | 0.0000040486614525 | 1.5988851497289625 | 0.0000027148941983 | 3.700149 | 0.757447 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/ambush_2_setting4_mode1_w2_sr0.9_ceres.log` |
| ambush_2 | 5 | 1 | 0.9 | 3 | yes | 0.1254670435204166 | 0.0000041461076584 | 2.2065342428894699 | 0.0000028003020655 | 4.276833 | 1.265432 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/ambush_2_setting5_mode1_w3_sr0.9_ceres.log` |
| ambush_2 | 6 | 2 | - | 1 | yes | 0.4895163828452721 | 0.0000076713253744 | 2.0562936386571637 | 0.0000054738106603 | 3.950241 | 1.346519 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/ambush_2_setting6_mode2_w1_ceres.log` |
| ambush_4 | 1 | 1 | 0.1 | 1 | yes | 0.0136171847868344 | 0.0784049195485918 | 0.2180550815816496 | 0.0549300296478926 | 9.739322 | 0.436134 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/ambush_4_setting1_mode1_w1_sr0.1_ceres.log` |
| ambush_4 | 2 | 1 | 0.3 | 1 | yes | 0.0159798921508891 | 0.0839486126329353 | 0.1485819877415100 | 0.0580775744228481 | 10.265242 | 0.523648 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/ambush_4_setting2_mode1_w1_sr0.3_ceres.log` |
| ambush_4 | 3 | 1 | 0.5 | 1 | yes | 0.0158681927460138 | 0.0838050233895890 | 0.2715854312730039 | 0.0621442949115173 | 8.002188 | 0.591851 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/ambush_4_setting3_mode1_w1_sr0.5_ceres.log` |
| ambush_4 | 4 | 1 | 0.9 | 2 | yes | 0.0123345268466762 | 0.0764975821086326 | 2.8290400303370080 | 0.0500608124608535 | 10.790018 | 0.690462 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/ambush_4_setting4_mode1_w2_sr0.9_ceres.log` |
| ambush_4 | 5 | 1 | 0.9 | 3 | yes | 0.0104308122843522 | 0.0992277040725538 | 3.0568570426970698 | 0.0643954738394614 | 8.603638 | 0.662450 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/ambush_4_setting5_mode1_w3_sr0.9_ceres.log` |
| ambush_4 | 6 | 2 | - | 1 | yes | 0.0323368484898701 | 0.2153472490846862 | 2.0897140399711369 | 0.1582419103210596 | 5.776403 | 0.265183 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/ambush_4_setting6_mode2_w1_ceres.log` |
| ambush_5 | 1 | 1 | 0.1 | 1 | yes | 0.0558145005357243 | 0.1745109142370658 | 0.5242789782193099 | 0.1185599263235727 | 6.747106 | 0.448816 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/ambush_5_setting1_mode1_w1_sr0.1_ceres.log` |
| ambush_5 | 2 | 1 | 0.3 | 1 | yes | 0.0574115425935581 | 0.2054461951475076 | 2.5053585868617518 | 0.1361571467607847 | 6.241984 | 0.556673 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/ambush_5_setting2_mode1_w1_sr0.3_ceres.log` |
| ambush_5 | 3 | 1 | 0.5 | 1 | yes | 0.0506806486792134 | 0.2129836011297495 | 1.4891402825233586 | 0.1464106414281881 | 7.354760 | 0.681846 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/ambush_5_setting3_mode1_w1_sr0.5_ceres.log` |
| ambush_5 | 4 | 1 | 0.9 | 2 | yes | 0.0357489788373066 | 0.1571047949649292 | 1.6014044884902823 | 0.1103877746601805 | 9.349253 | 0.760044 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/ambush_5_setting4_mode1_w2_sr0.9_ceres.log` |
| ambush_5 | 5 | 1 | 0.9 | 3 | yes | 0.0471201451103356 | 0.2530093838916632 | 2.0173404884081987 | 0.1863775619809247 | 8.248238 | 0.871397 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/ambush_5_setting5_mode1_w3_sr0.9_ceres.log` |
| ambush_5 | 6 | 2 | - | 1 | yes | 0.0682414409960371 | 0.2707130329288022 | 2.6659175809053655 | 0.1781464507101151 | 6.708878 | 0.280703 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/ambush_5_setting6_mode2_w1_ceres.log` |
| ambush_6 | 1 | 1 | 0.1 | 1 | yes | 0.1168755932692460 | 0.2529653964186194 | 2.7945983809871491 | 0.1734671789226483 | 4.209843 | 0.308056 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/ambush_6_setting1_mode1_w1_sr0.1_ceres.log` |
| ambush_6 | 2 | 1 | 0.3 | 1 | yes | 0.1156675277788651 | 0.2388765865659878 | 1.1807892714712724 | 0.1810153306507707 | 4.925136 | 0.321122 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/ambush_6_setting2_mode1_w1_sr0.3_ceres.log` |
| ambush_6 | 3 | 1 | 0.5 | 1 | yes | 0.1136844477837811 | 0.2554247130187892 | 0.9704520573956611 | 0.1804536992710338 | 4.654704 | 0.346333 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/ambush_6_setting3_mode1_w1_sr0.5_ceres.log` |
| ambush_6 | 4 | 1 | 0.9 | 2 | yes | 0.0900572599712399 | 0.2641594557898248 | 1.3460761207717280 | 0.1933469981153370 | 4.331011 | 0.420390 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/ambush_6_setting4_mode1_w2_sr0.9_ceres.log` |
| ambush_6 | 5 | 1 | 0.9 | 3 | yes | 0.0893945640670699 | 0.2702113348666330 | 0.9674255058412259 | 0.1958015585487657 | 4.868900 | 0.396474 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/ambush_6_setting5_mode1_w3_sr0.9_ceres.log` |
| ambush_6 | 6 | 2 | - | 1 | yes | 0.1130778875650398 | 0.3131637429950513 | 2.1208325696058901 | 0.2115099456927989 | 5.496524 | 0.240326 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/ambush_6_setting6_mode2_w1_ceres.log` |
| ambush_7 | 1 | 1 | 0.1 | 1 | yes | 0.0030568711394903 | 0.0000045186496622 | 2.6666634638867879 | 0.0000035792335104 | 5.466179 | 0.353425 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/ambush_7_setting1_mode1_w1_sr0.1_ceres.log` |
| ambush_7 | 2 | 1 | 0.3 | 1 | yes | 0.0037169190335458 | 0.0000051635494301 | 1.4821086428789574 | 0.0000037768657617 | 6.219764 | 0.415782 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/ambush_7_setting2_mode1_w1_sr0.3_ceres.log` |
| ambush_7 | 3 | 1 | 0.5 | 1 | yes | 0.0043354096202537 | 0.0000051235513810 | 1.9002141900170113 | 0.0000037600922654 | 6.053256 | 0.464954 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/ambush_7_setting3_mode1_w1_sr0.5_ceres.log` |
| ambush_7 | 4 | 1 | 0.9 | 2 | yes | 0.0059438650972065 | 0.0000059381704312 | 2.9879988296090918 | 0.0000047318692968 | 22.391805 | 0.424315 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/ambush_7_setting4_mode1_w2_sr0.9_ceres.log` |
| ambush_7 | 5 | 1 | 0.9 | 3 | yes | 0.0040945898955785 | 0.0000052066481171 | 2.9929621257050445 | 0.0000039528989528 | 10.523996 | 0.468413 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/ambush_7_setting5_mode1_w3_sr0.9_ceres.log` |
| ambush_7 | 6 | 2 | - | 1 | yes | 0.0027618611337222 | 0.0000048700488293 | 2.6257005908575097 | 0.0000035929057552 | 6.142576 | 0.261124 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/ambush_7_setting6_mode2_w1_ceres.log` |
| bamboo_1 | 1 | 1 | 0.1 | 1 | yes | 0.0072803930030456 | 0.0156950390271561 | 3.0209587190653746 | 0.0115626684014914 | 42.053420 | 0.275225 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/bamboo_1_setting1_mode1_w1_sr0.1_ceres.log` |
| bamboo_1 | 2 | 1 | 0.3 | 1 | yes | 0.0048976593091991 | 0.0155174787669018 | 3.1084085923317231 | 0.0111812574395203 | 54.734958 | 0.324834 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/bamboo_1_setting2_mode1_w1_sr0.3_ceres.log` |
| bamboo_1 | 3 | 1 | 0.5 | 1 | yes | 0.0027160308458602 | 0.0170971885111513 | 0.5109787414099050 | 0.0122545845776312 | 11.600888 | 0.333522 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/bamboo_1_setting3_mode1_w1_sr0.5_ceres.log` |
| bamboo_1 | 4 | 1 | 0.9 | 2 | yes | 0.0031829970184154 | 0.0083357559616316 | 2.2875842978381473 | 0.0056167739116771 | 48.360773 | 0.341369 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/bamboo_1_setting4_mode1_w2_sr0.9_ceres.log` |
| bamboo_1 | 5 | 1 | 0.9 | 3 | yes | 0.0049808037839495 | 0.0121035562696991 | 0.4867514422281145 | 0.0088811749664313 | 104.280490 | 0.374337 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/bamboo_1_setting5_mode1_w3_sr0.9_ceres.log` |
| bamboo_1 | 6 | 2 | - | 1 | yes | 0.0200110595301585 | 0.0230302203119150 | 1.3642167050992564 | 0.0156212556616904 | 13.844449 | 0.243072 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/bamboo_1_setting6_mode2_w1_ceres.log` |
| bamboo_2 | 1 | 1 | 0.1 | 1 | yes | 0.0170671776121419 | 0.0166786471178988 | 2.6159695691110416 | 0.0120768090411259 | 11.269523 | 0.273826 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/bamboo_2_setting1_mode1_w1_sr0.1_ceres.log` |
| bamboo_2 | 2 | 1 | 0.3 | 1 | yes | 0.0157455534467167 | 0.0114518673390950 | 2.6174748876780076 | 0.0086392317626272 | 11.687556 | 0.327809 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/bamboo_2_setting2_mode1_w1_sr0.3_ceres.log` |
| bamboo_2 | 3 | 1 | 0.5 | 1 | yes | 0.0151973293400894 | 0.0131865435001401 | 0.5482206878393479 | 0.0107097003740635 | 11.066258 | 0.329593 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/bamboo_2_setting3_mode1_w1_sr0.5_ceres.log` |
| bamboo_2 | 4 | 1 | 0.9 | 2 | yes | 0.0153265193316906 | 0.0137692639124364 | 0.5490722768366330 | 0.0108044314930481 | 10.141806 | 0.445678 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/bamboo_2_setting4_mode1_w2_sr0.9_ceres.log` |
| bamboo_2 | 5 | 1 | 0.9 | 3 | yes | 0.0145973773617032 | 0.0098626896641455 | 2.6152939780551434 | 0.0071841713314037 | 19.860211 | 0.412059 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/bamboo_2_setting5_mode1_w3_sr0.9_ceres.log` |
| bamboo_2 | 6 | 2 | - | 1 | yes | 0.0160574511846971 | 0.0079467002544362 | 2.6176950021557261 | 0.0051069531826793 | 12.933134 | 0.227494 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/bamboo_2_setting6_mode2_w1_ceres.log` |
| bandage_1 | 1 | 1 | 0.1 | 1 | yes | 0.3985695726596743 | 0.3179162878243766 | 2.1513463723377990 | 0.2555335757841950 | 4.972697 | 0.338675 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/bandage_1_setting1_mode1_w1_sr0.1_ceres.log` |
| bandage_1 | 2 | 1 | 0.3 | 1 | yes | 0.4595833105973230 | 0.3199817548756143 | 1.3971816594302107 | 0.2576854841751940 | 4.934786 | 0.387065 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/bandage_1_setting2_mode1_w1_sr0.3_ceres.log` |
| bandage_1 | 3 | 1 | 0.5 | 1 | yes | 0.4692497274279730 | 0.3685391647868126 | 1.4214736248551012 | 0.2986746340721160 | 5.065019 | 0.467785 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/bandage_1_setting3_mode1_w1_sr0.5_ceres.log` |
| bandage_1 | 4 | 1 | 0.9 | 2 | yes | 0.3716555692702765 | 0.0323456012858091 | 2.7327850068329451 | 0.0248017638779276 | 4.496644 | 0.412503 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/bandage_1_setting4_mode1_w2_sr0.9_ceres.log` |
| bandage_1 | 5 | 1 | 0.9 | 3 | yes | 0.3916824682324636 | 0.1477597851700375 | 2.7604361983192267 | 0.1176294666774741 | 4.942456 | 0.450388 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/bandage_1_setting5_mode1_w3_sr0.9_ceres.log` |
| bandage_1 | 6 | 2 | - | 1 | yes | 0.3945921768698339 | 0.3864810892435176 | 2.6162615800662907 | 0.3056929513192356 | 7.054776 | 0.266329 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/bandage_1_setting6_mode2_w1_ceres.log` |
| bandage_2 | 1 | 1 | 0.1 | 1 | yes | 0.0003545962507216 | 0.0391530068119497 | 1.1552671121270732 | 0.0287170193855185 | 4.148395 | 0.314681 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/bandage_2_setting1_mode1_w1_sr0.1_ceres.log` |
| bandage_2 | 2 | 1 | 0.3 | 1 | yes | 0.0003521886860888 | 0.0384766525081184 | 0.7213898283082670 | 0.0280600822377305 | 4.138334 | 0.367494 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/bandage_2_setting2_mode1_w1_sr0.3_ceres.log` |
| bandage_2 | 3 | 1 | 0.5 | 1 | yes | 0.0003511358252391 | 0.0382572542264601 | 1.4555159586243702 | 0.0277065343299162 | 4.151212 | 0.388620 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/bandage_2_setting3_mode1_w1_sr0.5_ceres.log` |
| bandage_2 | 4 | 1 | 0.9 | 2 | yes | 0.0006979267957922 | 0.0065222061078356 | 1.2163365881073931 | 0.0051457197109146 | 4.499446 | 0.361391 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/bandage_2_setting4_mode1_w2_sr0.9_ceres.log` |
| bandage_2 | 5 | 1 | 0.9 | 3 | yes | 0.0005202119643835 | 0.0170823133006373 | 1.5912217158994522 | 0.0129928949510552 | 4.662124 | 0.389922 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/bandage_2_setting5_mode1_w3_sr0.9_ceres.log` |
| bandage_2 | 6 | 2 | - | 1 | yes | 0.0003154766177793 | 0.0446564029460403 | 2.0775654904158851 | 0.0318266356410501 | 4.188920 | 0.249646 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/bandage_2_setting6_mode2_w1_ceres.log` |
| cave_2 | 1 | 1 | 0.1 | 1 | yes | 0.6508485435630961 | 0.7931257004300936 | 2.8230931363129979 | 0.7989455057451924 | 7.738235 | 2.578720 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/cave_2_setting1_mode1_w1_sr0.1_ceres.log` |
| cave_2 | 2 | 1 | 0.3 | 1 | yes | 0.8709907381391513 | 0.6537173049471617 | 2.8206139798876477 | 0.7864504913501094 | 8.200559 | 3.267061 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/cave_2_setting2_mode1_w1_sr0.3_ceres.log` |
| cave_2 | 3 | 1 | 0.5 | 1 | yes | 0.8302953144308002 | 0.6120976295741886 | 2.9927500182924303 | 0.7456118515785469 | 6.627328 | 2.236327 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/cave_2_setting3_mode1_w1_sr0.5_ceres.log` |
| cave_2 | 4 | 1 | 0.9 | 2 | yes | 0.5517980375280497 | 0.1780864431633223 | 2.1889956244777582 | 0.1472384808574715 | 35.767286 | 1.710498 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/cave_2_setting4_mode1_w2_sr0.9_ceres.log` |
| cave_2 | 5 | 1 | 0.9 | 3 | yes | 0.4958950991117882 | 0.4509096938304732 | 2.8254764051468029 | 0.3906582893655951 | 10.273277 | 1.860984 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/cave_2_setting5_mode1_w3_sr0.9_ceres.log` |
| cave_2 | 6 | 2 | - | 1 | yes | 0.7470731264471253 | 1.0946036532028518 | 2.3612117203061338 | 0.8263092461709438 | 5.488219 | 0.810015 | 4 | `Python/logs/sintel_xm_ceres_rerun/solver/cave_2_setting6_mode2_w1_ceres.log` |
| cave_4 | 1 | 1 | 0.1 | 1 | yes | 0.0111851880944058 | 0.0265795564564403 | 0.2623847831900092 | 0.0185507805672435 | 11.984890 | 0.318686 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/cave_4_setting1_mode1_w1_sr0.1_ceres.log` |
| cave_4 | 2 | 1 | 0.3 | 1 | yes | 0.0103264785567629 | 0.0236065857854060 | 0.1176408282521856 | 0.0168855686491553 | 14.878333 | 0.375796 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/cave_4_setting2_mode1_w1_sr0.3_ceres.log` |
| cave_4 | 3 | 1 | 0.5 | 1 | yes | 0.0090951210657496 | 0.0226448124272923 | 0.0871877733207343 | 0.0160044745470368 | 21.013077 | 0.394716 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/cave_4_setting3_mode1_w1_sr0.5_ceres.log` |
| cave_4 | 4 | 1 | 0.9 | 2 | yes | 0.0185908980242857 | 0.0629535138657699 | 0.5354896784209358 | 0.0526876790381539 | 26.096120 | 0.534899 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/cave_4_setting4_mode1_w2_sr0.9_ceres.log` |
| cave_4 | 5 | 1 | 0.9 | 3 | yes | 0.0119053008196039 | 0.0263446822174002 | 0.1512204227235063 | 0.0191592387695929 | 8.289509 | 0.457751 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/cave_4_setting5_mode1_w3_sr0.9_ceres.log` |
| cave_4 | 6 | 2 | - | 1 | yes | 0.0209570558086401 | 0.0737354580126596 | 0.1733161615982148 | 0.0433068333381581 | 6.833510 | 0.254049 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/cave_4_setting6_mode2_w1_ceres.log` |
| market_2 | 1 | 1 | 0.1 | 1 | yes | 0.0018435403464676 | 0.0872219241189546 | 0.8115579629872992 | 0.0625908023405767 | 4.194220 | 0.347376 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/market_2_setting1_mode1_w1_sr0.1_ceres.log` |
| market_2 | 2 | 1 | 0.3 | 1 | yes | 0.0018512934353509 | 0.0848747369524164 | 0.7505380523860435 | 0.0682129207802485 | 4.309959 | 0.418027 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/market_2_setting2_mode1_w1_sr0.3_ceres.log` |
| market_2 | 3 | 1 | 0.5 | 1 | yes | 0.0018535722457471 | 0.0728499239030621 | 1.0275573881108524 | 0.0510477011041815 | 4.342358 | 0.435803 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/market_2_setting3_mode1_w1_sr0.5_ceres.log` |
| market_2 | 4 | 1 | 0.9 | 2 | yes | 0.0016460660354396 | 0.0124098818620164 | 2.9310082166974225 | 0.0084018114614434 | 5.192033 | 0.436835 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/market_2_setting4_mode1_w2_sr0.9_ceres.log` |
| market_2 | 5 | 1 | 0.9 | 3 | yes | 0.0016752316942734 | 0.0260003342326329 | 3.1284928539605037 | 0.0182536557888681 | 5.178197 | 0.491800 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/market_2_setting5_mode1_w3_sr0.9_ceres.log` |
| market_2 | 6 | 2 | - | 1 | yes | 0.0022640491623949 | 0.1105482701743284 | 2.9252733330976159 | 0.0791032917951630 | 4.364109 | 0.259521 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/market_2_setting6_mode2_w1_ceres.log` |
| market_5 | 1 | 1 | 0.1 | 1 | yes | 0.0847388611162177 | 0.0072692361488987 | 1.6936847961223096 | 0.0055396869884254 | 6.659131 | 0.996991 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/market_5_setting1_mode1_w1_sr0.1_ceres.log` |
| market_5 | 2 | 1 | 0.3 | 1 | yes | 0.0846385272807836 | 0.0074067864757737 | 1.7250403506110374 | 0.0055076378106950 | 7.487998 | 1.519224 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/market_5_setting2_mode1_w1_sr0.3_ceres.log` |
| market_5 | 3 | 1 | 0.5 | 1 | yes | 0.0770714620321535 | 0.0067134210487314 | 1.7627939217502191 | 0.0049245563795466 | 8.442534 | 1.693417 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/market_5_setting3_mode1_w1_sr0.5_ceres.log` |
| market_5 | 4 | 1 | 0.9 | 2 | yes | 0.0949275225211568 | 0.0202587521696333 | 3.1105771059818581 | 0.0152255791980397 | 11.390904 | 2.029606 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/market_5_setting4_mode1_w2_sr0.9_ceres.log` |
| market_5 | 5 | 1 | 0.9 | 3 | yes | 0.0808118112749165 | 0.0049151920505799 | 2.4909963388768683 | 0.0035983759591611 | 9.003873 | 2.428595 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/market_5_setting5_mode1_w3_sr0.9_ceres.log` |
| market_5 | 6 | 2 | - | 1 | yes | 0.1065719586955248 | 0.0082765473642501 | 1.8511505787955758 | 0.0058469525155040 | 4.547261 | 0.473771 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/market_5_setting6_mode2_w1_ceres.log` |
| market_6 | 1 | 1 | 0.1 | 1 | yes | 0.0189892059361125 | 0.0264889910003247 | 2.0732089224917929 | 0.0191153700529484 | 5.940764 | 0.319009 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/market_6_setting1_mode1_w1_sr0.1_ceres.log` |
| market_6 | 2 | 1 | 0.3 | 1 | yes | 0.0161376875298241 | 0.0259222638636006 | 2.0380644916816428 | 0.0191154173409763 | 6.071055 | 0.354669 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/market_6_setting2_mode1_w1_sr0.3_ceres.log` |
| market_6 | 3 | 1 | 0.5 | 1 | yes | 0.0188007179581253 | 0.0264595925984846 | 2.2134369671398439 | 0.0192852130688816 | 5.311876 | 0.373486 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/market_6_setting3_mode1_w1_sr0.5_ceres.log` |
| market_6 | 4 | 1 | 0.9 | 2 | yes | 0.0374509280315735 | 0.0675765559908045 | 2.8906215033642448 | 0.0481208971833512 | 5.764625 | 0.510075 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/market_6_setting4_mode1_w2_sr0.9_ceres.log` |
| market_6 | 5 | 1 | 0.9 | 3 | yes | 0.0333483228418444 | 0.0378082204714195 | 2.6684568525038244 | 0.0266127707944808 | 5.128168 | 0.473464 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/market_6_setting5_mode1_w3_sr0.9_ceres.log` |
| market_6 | 6 | 2 | - | 1 | yes | 0.0208151015695804 | 0.0311394257206721 | 2.2945998247732762 | 0.0222213057926483 | 6.231420 | 0.254340 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/market_6_setting6_mode2_w1_ceres.log` |
| mountain_1 | 1 | 1 | 0.1 | 1 | yes | 0.0120848107562447 | 0.0363963309123792 | 0.5438828487319239 | 0.0259478022947711 | 6.384930 | 0.324539 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/mountain_1_setting1_mode1_w1_sr0.1_ceres.log` |
| mountain_1 | 2 | 1 | 0.3 | 1 | yes | 0.0121142576568614 | 0.0301908927553182 | 0.5746772616017032 | 0.0213542259046682 | 6.345502 | 0.358905 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/mountain_1_setting2_mode1_w1_sr0.3_ceres.log` |
| mountain_1 | 3 | 1 | 0.5 | 1 | yes | 0.0122846976960660 | 0.0280175780808886 | 0.5666628976950584 | 0.0196941950552578 | 5.911471 | 0.392974 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/mountain_1_setting3_mode1_w1_sr0.5_ceres.log` |
| mountain_1 | 4 | 1 | 0.9 | 2 | yes | 0.0116364203265754 | 0.0973508371347397 | 1.9172662407722183 | 0.0658863928115297 | 9.385753 | 0.449624 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/mountain_1_setting4_mode1_w2_sr0.9_ceres.log` |
| mountain_1 | 5 | 1 | 0.9 | 3 | yes | 0.0104282315842757 | 0.0390759136784633 | 1.4435932002014107 | 0.0262762520621679 | 10.034123 | 0.463932 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/mountain_1_setting5_mode1_w3_sr0.9_ceres.log` |
| mountain_1 | 6 | 2 | - | 1 | yes | 0.0115606151631467 | 0.0328658246074485 | 0.7407213959986394 | 0.0226120770514042 | 7.332572 | 0.243333 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/mountain_1_setting6_mode2_w1_ceres.log` |
| shaman_2 | 1 | 1 | 0.1 | 1 | yes | 0.0231384271502701 | 0.0580241802635694 | 2.4023712886188933 | 0.0466343992453225 | 6.624136 | 0.338086 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/shaman_2_setting1_mode1_w1_sr0.1_ceres.log` |
| shaman_2 | 2 | 1 | 0.3 | 1 | yes | 0.0230648921025658 | 0.0571546204720670 | 2.6524413015569901 | 0.0458382700861990 | 6.929491 | 0.396521 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/shaman_2_setting2_mode1_w1_sr0.3_ceres.log` |
| shaman_2 | 3 | 1 | 0.5 | 1 | yes | 0.0228260214562770 | 0.0568658409363192 | 2.4531997542007149 | 0.0456015413397723 | 6.689771 | 0.473347 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/shaman_2_setting3_mode1_w1_sr0.5_ceres.log` |
| shaman_2 | 4 | 1 | 0.9 | 2 | yes | 0.0244735978875194 | 0.0269451549786594 | 1.5525532891591405 | 0.0216138555164725 | 28.360133 | 0.401392 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/shaman_2_setting4_mode1_w2_sr0.9_ceres.log` |
| shaman_2 | 5 | 1 | 0.9 | 3 | yes | 0.0232282669311397 | 0.0414992996920718 | 1.9064564547747782 | 0.0337003192625862 | 9.486147 | 0.505319 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/shaman_2_setting5_mode1_w3_sr0.9_ceres.log` |
| shaman_2 | 6 | 2 | - | 1 | yes | 0.0224846499989728 | 0.0527955361045655 | 2.6108751451241208 | 0.0424440550390041 | 12.805556 | 0.255723 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/shaman_2_setting6_mode2_w1_ceres.log` |
| shaman_3 | 1 | 1 | 0.1 | 1 | yes | 0.0231350903623577 | 0.0031264873434165 | 2.5403083969693734 | 0.0022409199559873 | 6.920348 | 0.348645 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/shaman_3_setting1_mode1_w1_sr0.1_ceres.log` |
| shaman_3 | 2 | 1 | 0.3 | 1 | yes | 0.0237508260599275 | 0.0031767446584563 | 2.5262783245984846 | 0.0022819171836902 | 5.859054 | 0.422976 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/shaman_3_setting2_mode1_w1_sr0.3_ceres.log` |
| shaman_3 | 3 | 1 | 0.5 | 1 | yes | 0.0243291332603434 | 0.0033091054361005 | 2.5226006114305490 | 0.0023880597259440 | 5.597071 | 0.494039 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/shaman_3_setting3_mode1_w1_sr0.5_ceres.log` |
| shaman_3 | 4 | 1 | 0.9 | 2 | yes | 0.0199684570535707 | 0.0055829244142870 | 3.0020947595249137 | 0.0037138254455515 | 8.112396 | 0.616317 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/shaman_3_setting4_mode1_w2_sr0.9_ceres.log` |
| shaman_3 | 5 | 1 | 0.9 | 3 | yes | 0.0247908178765373 | 0.0059698895013400 | 1.4976906638939205 | 0.0040734847661915 | 7.277577 | 0.669078 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/shaman_3_setting5_mode1_w3_sr0.9_ceres.log` |
| shaman_3 | 6 | 2 | - | 1 | yes | 0.0234316436490341 | 0.0053747285891350 | 2.7525122354422358 | 0.0035169517999748 | 9.150003 | 0.264421 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/shaman_3_setting6_mode2_w1_ceres.log` |
| sleeping_1 | 1 | 1 | 0.1 | 1 | yes | 0.0443387487156797 | 0.0064815146327079 | 0.9086870214705371 | 0.0044179007082653 | 5.314908 | 0.318409 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/sleeping_1_setting1_mode1_w1_sr0.1_ceres.log` |
| sleeping_1 | 2 | 1 | 0.3 | 1 | yes | 0.0449679619645838 | 0.0066845850194267 | 0.8232401077336158 | 0.0045663554032061 | 5.703043 | 0.362001 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/sleeping_1_setting2_mode1_w1_sr0.3_ceres.log` |
| sleeping_1 | 3 | 1 | 0.5 | 1 | yes | 0.0458294550663454 | 0.0067580446991442 | 0.8035823106022585 | 0.0046197601651508 | 5.804418 | 0.403569 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/sleeping_1_setting3_mode1_w1_sr0.5_ceres.log` |
| sleeping_1 | 4 | 1 | 0.9 | 2 | yes | 0.0551590632073909 | 0.0135918316593366 | 1.7541229613370435 | 0.0085491036376592 | 5.006420 | 0.442192 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/sleeping_1_setting4_mode1_w2_sr0.9_ceres.log` |
| sleeping_1 | 5 | 1 | 0.9 | 3 | yes | 0.0541681087328474 | 0.0074360708853496 | 0.7079208619202675 | 0.0049319186853229 | 25.897675 | 0.532832 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/sleeping_1_setting5_mode1_w3_sr0.9_ceres.log` |
| sleeping_1 | 6 | 2 | - | 1 | yes | 0.0633747852161956 | 0.0076724779317964 | 0.4221502760947469 | 0.0047351883478747 | 24.349289 | 0.272044 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/sleeping_1_setting6_mode2_w1_ceres.log` |
| sleeping_2 | 1 | 1 | 0.1 | 1 | yes | 0.0007629525602789 | 0.0016707694777144 | 0.0339383080292290 | 0.0011577360073478 | 69.707113 | 0.262406 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/sleeping_2_setting1_mode1_w1_sr0.1_ceres.log` |
| sleeping_2 | 2 | 1 | 0.3 | 1 | yes | 0.0008221202557396 | 0.0017442862299273 | 0.0491382488848183 | 0.0012022935944667 | 56.646244 | 0.285125 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/sleeping_2_setting2_mode1_w1_sr0.3_ceres.log` |
| sleeping_2 | 3 | 1 | 0.5 | 1 | yes | 0.0007743893804433 | 0.0016784548273810 | 0.0405659398207937 | 0.0011600136039736 | 29.514637 | 0.293927 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/sleeping_2_setting3_mode1_w1_sr0.5_ceres.log` |
| sleeping_2 | 4 | 1 | 0.9 | 2 | yes | 0.0031876033588699 | 0.0075785084196392 | 0.0609948873542836 | 0.0050348407473159 | 14.362597 | 0.399341 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/sleeping_2_setting4_mode1_w2_sr0.9_ceres.log` |
| sleeping_2 | 5 | 1 | 0.9 | 3 | yes | 0.0048210557078314 | 0.0042720727830855 | 0.0909937257974740 | 0.0027572060493785 | 41.687413 | 0.368774 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/sleeping_2_setting5_mode1_w3_sr0.9_ceres.log` |
| sleeping_2 | 6 | 2 | - | 1 | yes | 0.0011779285700108 | 0.0032363310346760 | 0.0403523971454917 | 0.0020711361659775 | 46.561820 | 0.238633 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/sleeping_2_setting6_mode2_w1_ceres.log` |
| temple_2 | 1 | 1 | 0.1 | 1 | yes | 0.1074904497383815 | 0.1508257212218754 | 2.8254201364121774 | 0.1210306981404493 | 10.555747 | 0.394688 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/temple_2_setting1_mode1_w1_sr0.1_ceres.log` |
| temple_2 | 2 | 1 | 0.3 | 1 | yes | 0.1074914682539223 | 0.1593348299317980 | 2.6909944092069047 | 0.1265338195059790 | 10.060318 | 0.426306 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/temple_2_setting2_mode1_w1_sr0.3_ceres.log` |
| temple_2 | 3 | 1 | 0.5 | 1 | yes | 0.1091730292666640 | 0.1583811282770480 | 2.5858128704324601 | 0.1303706808531651 | 9.534749 | 0.481874 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/temple_2_setting3_mode1_w1_sr0.5_ceres.log` |
| temple_2 | 4 | 1 | 0.9 | 2 | yes | 0.0385772515718985 | 0.0562298451519143 | 0.4245656583158947 | 0.0396696401864210 | 96.611676 | 0.890867 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/temple_2_setting4_mode1_w2_sr0.9_ceres.log` |
| temple_2 | 5 | 1 | 0.9 | 3 | yes | 0.0750905083946723 | 0.0784473143444687 | 0.4554206460037636 | 0.0557390134649816 | 13.228323 | 0.757080 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/temple_2_setting5_mode1_w3_sr0.9_ceres.log` |
| temple_2 | 6 | 2 | - | 1 | yes | 0.1117393624798709 | 0.2825147004360062 | 2.5351210934417092 | 0.1876930933314563 | 8.298217 | 0.277498 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/temple_2_setting6_mode2_w1_ceres.log` |
| temple_3 | 1 | 1 | 0.1 | 1 | yes | 0.0078460834830943 | 0.1625098437696331 | 1.2539298527705223 | 0.1259955869280095 | 7.601582 | 1.196691 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/temple_3_setting1_mode1_w1_sr0.1_ceres.log` |
| temple_3 | 2 | 1 | 0.3 | 1 | yes | 0.0072780784761645 | 0.1704248964417724 | 1.6274018384511419 | 0.1208913651492005 | 8.211765 | 1.617565 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/temple_3_setting2_mode1_w1_sr0.3_ceres.log` |
| temple_3 | 3 | 1 | 0.5 | 1 | yes | 0.0062438710297440 | 0.1619808338872177 | 2.9928311603138322 | 0.1243589588532996 | 8.917040 | 2.142007 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/temple_3_setting3_mode1_w1_sr0.5_ceres.log` |
| temple_3 | 4 | 1 | 0.9 | 2 | yes | 0.0041410637900227 | 0.1422606624847299 | 0.9598358380842774 | 0.0985853880589678 | 9.911295 | 1.322861 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/temple_3_setting4_mode1_w2_sr0.9_ceres.log` |
| temple_3 | 5 | 1 | 0.9 | 3 | yes | 0.0046399744417364 | 0.1629771840012147 | 1.4669754978500513 | 0.1125369796401896 | 8.947662 | 1.803384 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/temple_3_setting5_mode1_w3_sr0.9_ceres.log` |
| temple_3 | 6 | 2 | - | 1 | yes | 0.0102419996651847 | 0.1871227161873550 | 1.5078715363463386 | 0.1451238172225247 | 6.689856 | 0.588794 | 3 | `Python/logs/sintel_xm_ceres_rerun/solver/temple_3_setting6_mode2_w1_ceres.log` |
