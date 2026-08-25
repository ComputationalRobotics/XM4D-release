# Sintel XM Experiment Results (with Ceres refinement, strong penalty schedule)

Generated: `2026-04-22T18:25:23-04:00`

Pipeline:

- Dataset root: `datasets/Sintel`
- Missing GT cameras are loaded from `datasets/MPI-Sintel-depth-training-20150305/training/camdata_left`
- Missing DA3 depth is generated with `Python/get_depths_clean.py` in the `da3` environment
- Missing `output_tapnet_da3.npz` is generated with `Python/static_tracking_online.py` in the `tapnet` environment
- Solver sweep uses the same 6 heuristics as the BAL experiment, with no added noise
- Refinement: `with Ceres refinement`
- Chosen penalty schedule: `PENALTY_T = 15N`, `PENALTY_T_CERES_REL = 0.2 x PENALTY_T`, `PENALTY_T_CERES_WINDOW = 3`, `PENALTY_T_CERES_ACC = 20 x PENALTY_T`
- Schedule was selected from the subset comparison in `Python/sintel_penalty_subset_comparison.md`

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
| alley_1 | 1 | 1 | 0.1 | 1 | yes | 0.0251162656242745 | 0.0034961457931574 | 3.1214051852714060 | 0.0025373676831353 | 12.056923 | 0.308261 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/alley_1_setting1_mode1_w1_sr0.1_ceres.log` |
| alley_1 | 2 | 1 | 0.3 | 1 | yes | 0.0258378258753676 | 0.0030754237312767 | 3.1179553311441155 | 0.0022094909344887 | 13.257563 | 0.360662 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/alley_1_setting2_mode1_w1_sr0.3_ceres.log` |
| alley_1 | 3 | 1 | 0.5 | 1 | yes | 0.0255938956126520 | 0.0031181200251279 | 3.1056237625088277 | 0.0022355199832129 | 16.439040 | 0.421307 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/alley_1_setting3_mode1_w1_sr0.5_ceres.log` |
| alley_1 | 4 | 1 | 0.9 | 2 | yes | 0.0066869910738369 | 0.0075215035301232 | 2.4963268375633141 | 0.0055768498441074 | 28.123232 | 0.399932 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/alley_1_setting4_mode1_w2_sr0.9_ceres.log` |
| alley_1 | 5 | 1 | 0.9 | 3 | yes | 0.0334125013787683 | 0.0044006639344792 | 2.9585992443203555 | 0.0031810345687824 | 56.129332 | 0.491413 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/alley_1_setting5_mode1_w3_sr0.9_ceres.log` |
| alley_1 | 6 | 2 | - | 1 | yes | 0.0242114673770135 | 0.0307358248548364 | 2.9006574060063919 | 0.0230042591183909 | 17.317342 | 0.249641 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/alley_1_setting6_mode2_w1_ceres.log` |
| alley_2 | 1 | 1 | 0.1 | 1 | yes | 0.1062710711100172 | 0.0090238547117523 | 2.9009351576943936 | 0.0065932143859523 | 66.359816 | 0.392670 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/alley_2_setting1_mode1_w1_sr0.1_ceres.log` |
| alley_2 | 2 | 1 | 0.3 | 1 | yes | 0.1029098204840076 | 0.0239584482225931 | 0.5921300285613323 | 0.0182400660598650 | 12.271263 | 0.431772 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/alley_2_setting2_mode1_w1_sr0.3_ceres.log` |
| alley_2 | 3 | 1 | 0.5 | 1 | yes | 0.1037938036210557 | 0.0255459722279239 | 0.5038829908299297 | 0.0196717267816335 | 22.522636 | 0.500493 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/alley_2_setting3_mode1_w1_sr0.5_ceres.log` |
| alley_2 | 4 | 1 | 0.9 | 2 | yes | 0.1085275738047499 | 0.0148980279834659 | 0.3589495356614957 | 0.0109373847967126 | 9.326934 | 1.167198 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/alley_2_setting4_mode1_w2_sr0.9_ceres.log` |
| alley_2 | 5 | 1 | 0.9 | 3 | yes | 0.1051989037057518 | 0.0216272066756040 | 0.4043810396751555 | 0.0170038816562401 | 17.314741 | 0.917467 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/alley_2_setting5_mode1_w3_sr0.9_ceres.log` |
| alley_2 | 6 | 2 | - | 1 | yes | 0.1097919026886418 | 0.0523001337331688 | 2.4971010560957221 | 0.0318370340821734 | 8.682668 | 0.323581 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/alley_2_setting6_mode2_w1_ceres.log` |
| ambush_2 | 1 | 1 | 0.1 | 1 | yes | 0.1362183602429102 | 0.0000061550205291 | 1.3216237819614574 | 0.0000041208723237 | 6.313353 | 2.553908 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_2_setting1_mode1_w1_sr0.1_ceres.log` |
| ambush_2 | 2 | 1 | 0.3 | 1 | yes | 0.3026679877442434 | 0.0000067674757287 | 1.0184497304454780 | 0.0000044674013463 | 7.031172 | 3.583791 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_2_setting2_mode1_w1_sr0.3_ceres.log` |
| ambush_2 | 3 | 1 | 0.5 | 1 | yes | 0.2313586943217761 | 0.0000056988643684 | 2.4583267066754209 | 0.0000043193531038 | 8.009215 | 4.682732 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_2_setting3_mode1_w1_sr0.5_ceres.log` |
| ambush_2 | 4 | 1 | 0.9 | 2 | yes | 0.0553626528078311 | 0.0000045800317798 | 3.0725026283153705 | 0.0000035990222631 | 6.090354 | 0.803694 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_2_setting4_mode1_w2_sr0.9_ceres.log` |
| ambush_2 | 5 | 1 | 0.9 | 3 | yes | 0.0310186735288070 | 0.0000044143156162 | 2.6465532279971749 | 0.0000034012141047 | 4.748376 | 0.831639 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_2_setting5_mode1_w3_sr0.9_ceres.log` |
| ambush_2 | 6 | 2 | - | 1 | yes | 0.2571730546720442 | 0.0000070826833740 | 0.7829161392386248 | 0.0000046371996637 | 4.622015 | 1.042462 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_2_setting6_mode2_w1_ceres.log` |
| ambush_4 | 1 | 1 | 0.1 | 1 | yes | 0.0152787529750432 | 0.0533626482598752 | 1.4274279318098346 | 0.0385957642481578 | 7.452735 | 0.465048 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_4_setting1_mode1_w1_sr0.1_ceres.log` |
| ambush_4 | 2 | 1 | 0.3 | 1 | yes | 0.0266617297975062 | 0.0820955654671658 | 1.6923140890008186 | 0.0564894932033935 | 7.461857 | 0.516681 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_4_setting2_mode1_w1_sr0.3_ceres.log` |
| ambush_4 | 3 | 1 | 0.5 | 1 | yes | 0.0024454531491395 | 0.0143936564462863 | 0.0769364891161367 | 0.0098469940948802 | 8.084695 | 0.592841 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_4_setting3_mode1_w1_sr0.5_ceres.log` |
| ambush_4 | 4 | 1 | 0.9 | 2 | yes | 0.0025839277050863 | 0.0166886174004676 | 0.1584238344580814 | 0.0110214961697009 | 8.287839 | 0.701785 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_4_setting4_mode1_w2_sr0.9_ceres.log` |
| ambush_4 | 5 | 1 | 0.9 | 3 | yes | 0.0024437850959747 | 0.0111841746062484 | 0.1084343121056112 | 0.0078217579203105 | 10.123078 | 0.678077 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_4_setting5_mode1_w3_sr0.9_ceres.log` |
| ambush_4 | 6 | 2 | - | 1 | yes | 0.0136626876084416 | 0.0631923216490306 | 0.6888595208981244 | 0.0496139549064177 | 6.788711 | 0.270713 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_4_setting6_mode2_w1_ceres.log` |
| ambush_5 | 1 | 1 | 0.1 | 1 | yes | 0.0566645679924070 | 0.0537449628259942 | 0.6877754255208913 | 0.0384676370573356 | 6.789562 | 0.452054 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_5_setting1_mode1_w1_sr0.1_ceres.log` |
| ambush_5 | 2 | 1 | 0.3 | 1 | yes | 0.0849565207861482 | 0.1854377605394161 | 1.6627963301640043 | 0.1390885512337298 | 7.048051 | 0.547212 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_5_setting2_mode1_w1_sr0.3_ceres.log` |
| ambush_5 | 3 | 1 | 0.5 | 1 | yes | 0.0707162256807879 | 0.2493939790397323 | 3.1092887458853782 | 0.1655366225168784 | 7.407666 | 0.658517 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_5_setting3_mode1_w1_sr0.5_ceres.log` |
| ambush_5 | 4 | 1 | 0.9 | 2 | yes | 0.0532925788428648 | 0.1435122293808215 | 1.8606120817513891 | 0.0995475654173900 | 8.578378 | 0.797555 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_5_setting4_mode1_w2_sr0.9_ceres.log` |
| ambush_5 | 5 | 1 | 0.9 | 3 | yes | 0.0039318266908495 | 0.0092792537175290 | 1.2874859854640157 | 0.0062670326654591 | 8.851579 | 0.860683 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_5_setting5_mode1_w3_sr0.9_ceres.log` |
| ambush_5 | 6 | 2 | - | 1 | yes | 0.0755602783000388 | 0.2505544205333460 | 1.3869752255355852 | 0.1850001688659625 | 7.572545 | 0.280704 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_5_setting6_mode2_w1_ceres.log` |
| ambush_6 | 1 | 1 | 0.1 | 1 | yes | 0.1079515241865556 | 0.1761665605953859 | 0.9572762829491512 | 0.1339445697958676 | 6.047061 | 0.309311 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_6_setting1_mode1_w1_sr0.1_ceres.log` |
| ambush_6 | 2 | 1 | 0.3 | 1 | yes | 0.0506591360283399 | 0.0627401665258162 | 0.3194196965971722 | 0.0435648784322056 | 4.865806 | 0.338608 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_6_setting2_mode1_w1_sr0.3_ceres.log` |
| ambush_6 | 3 | 1 | 0.5 | 1 | yes | 0.0436736652687534 | 0.0679949619836998 | 0.7250949428235469 | 0.0414249661746120 | 4.736172 | 0.352292 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_6_setting3_mode1_w1_sr0.5_ceres.log` |
| ambush_6 | 4 | 1 | 0.9 | 2 | yes | 0.0459010244668080 | 0.1079287837536049 | 0.9019098722509487 | 0.0825543153736013 | 6.439400 | 0.408175 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_6_setting4_mode1_w2_sr0.9_ceres.log` |
| ambush_6 | 5 | 1 | 0.9 | 3 | yes | 0.0433627468065691 | 0.1115403916159408 | 1.0139287660176093 | 0.0729319928962496 | 7.465236 | 0.390152 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_6_setting5_mode1_w3_sr0.9_ceres.log` |
| ambush_6 | 6 | 2 | - | 1 | yes | 0.1191093688611741 | 0.4166673221327925 | 1.8570813746573243 | 0.2735155422256599 | 4.550281 | 0.250123 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_6_setting6_mode2_w1_ceres.log` |
| ambush_7 | 1 | 1 | 0.1 | 1 | yes | 0.0059528949766214 | 0.0000053945928790 | 2.5375831358370662 | 0.0000039369809461 | 65.532857 | 0.340693 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_7_setting1_mode1_w1_sr0.1_ceres.log` |
| ambush_7 | 2 | 1 | 0.3 | 1 | yes | 0.0056060950728824 | 0.0000053996183263 | 2.5049288264191305 | 0.0000039298347709 | 66.116748 | 0.426950 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_7_setting2_mode1_w1_sr0.3_ceres.log` |
| ambush_7 | 3 | 1 | 0.5 | 1 | yes | 0.0058947255039842 | 0.0000054016751024 | 2.4998417124793280 | 0.0000039566615557 | 65.439493 | 0.487204 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_7_setting3_mode1_w1_sr0.5_ceres.log` |
| ambush_7 | 4 | 1 | 0.9 | 2 | yes | 0.0083562910996226 | 0.0000053151066591 | 3.1261579256034269 | 0.0000038616553430 | 8.039508 | 0.417487 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_7_setting4_mode1_w2_sr0.9_ceres.log` |
| ambush_7 | 5 | 1 | 0.9 | 3 | yes | 0.0054232043605259 | 0.0000053099580057 | 2.3985328632326977 | 0.0000039936704993 | 14.406971 | 0.464672 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_7_setting5_mode1_w3_sr0.9_ceres.log` |
| ambush_7 | 6 | 2 | - | 1 | yes | 0.0069244953213823 | 0.0000051343999365 | 2.3889651855471130 | 0.0000039601099469 | 33.726386 | 0.253571 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/ambush_7_setting6_mode2_w1_ceres.log` |
| bamboo_1 | 1 | 1 | 0.1 | 1 | yes | 0.0021329966061760 | 0.0016269308042833 | 1.9073959233011279 | 0.0011241126240832 | 65.500538 | 0.266458 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bamboo_1_setting1_mode1_w1_sr0.1_ceres.log` |
| bamboo_1 | 2 | 1 | 0.3 | 1 | yes | 0.0019770014418262 | 0.0012837036687844 | 1.6324890851976324 | 0.0009083030441601 | 15.234699 | 0.326747 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bamboo_1_setting2_mode1_w1_sr0.3_ceres.log` |
| bamboo_1 | 3 | 1 | 0.5 | 1 | yes | 0.0019575877288078 | 0.0012628419426683 | 2.1404690104541007 | 0.0008861935397133 | 10.463481 | 0.334779 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bamboo_1_setting3_mode1_w1_sr0.5_ceres.log` |
| bamboo_1 | 4 | 1 | 0.9 | 2 | yes | 0.0019650969226938 | 0.0012798311345672 | 1.7393324753954198 | 0.0008975580947896 | 11.694038 | 0.352117 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bamboo_1_setting4_mode1_w2_sr0.9_ceres.log` |
| bamboo_1 | 5 | 1 | 0.9 | 3 | yes | 0.0020069813328128 | 0.0012769986436862 | 1.5486324679750891 | 0.0009007822366758 | 8.613944 | 0.372769 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bamboo_1_setting5_mode1_w3_sr0.9_ceres.log` |
| bamboo_1 | 6 | 2 | - | 1 | yes | 0.0018249128910568 | 0.0011598158330288 | 3.1171213632121533 | 0.0008010736378473 | 15.106967 | 0.230884 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bamboo_1_setting6_mode2_w1_ceres.log` |
| bamboo_2 | 1 | 1 | 0.1 | 1 | yes | 0.0005663651270135 | 0.0025034218588939 | 0.5267845706457941 | 0.0017333872304071 | 38.035202 | 0.280513 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bamboo_2_setting1_mode1_w1_sr0.1_ceres.log` |
| bamboo_2 | 2 | 1 | 0.3 | 1 | yes | 0.0015931038802508 | 0.0018274647886043 | 0.5245149390369465 | 0.0012656434334955 | 11.221753 | 0.341009 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bamboo_2_setting2_mode1_w1_sr0.3_ceres.log` |
| bamboo_2 | 3 | 1 | 0.5 | 1 | yes | 0.0014314319099285 | 0.0016203466999109 | 0.5246296144301127 | 0.0011475257847153 | 65.797452 | 0.327940 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bamboo_2_setting3_mode1_w1_sr0.5_ceres.log` |
| bamboo_2 | 4 | 1 | 0.9 | 2 | yes | 0.0017284319934831 | 0.0010503600981470 | 2.6173116808583412 | 0.0007574917428426 | 10.961304 | 0.442187 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bamboo_2_setting4_mode1_w2_sr0.9_ceres.log` |
| bamboo_2 | 5 | 1 | 0.9 | 3 | yes | 0.0016961187603470 | 0.0011947723162651 | 0.5243755545528370 | 0.0008526224879307 | 65.885874 | 0.404006 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bamboo_2_setting5_mode1_w3_sr0.9_ceres.log` |
| bamboo_2 | 6 | 2 | - | 1 | yes | 0.0143717789120872 | 0.0026447089051752 | 2.6202550461259522 | 0.0019241412408366 | 60.122640 | 0.236607 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bamboo_2_setting6_mode2_w1_ceres.log` |
| bandage_1 | 1 | 1 | 0.1 | 1 | yes | 0.3652578215158589 | 0.3132864550312141 | 1.9017388305207497 | 0.2508537984951059 | 5.883642 | 0.387227 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bandage_1_setting1_mode1_w1_sr0.1_ceres.log` |
| bandage_1 | 2 | 1 | 0.3 | 1 | yes | 0.4092128986382469 | 0.2836141767384842 | 2.0607090540216908 | 0.2264046326911180 | 5.331653 | 0.445140 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bandage_1_setting2_mode1_w1_sr0.3_ceres.log` |
| bandage_1 | 3 | 1 | 0.5 | 1 | yes | 0.4905778070673474 | 0.2702998761620797 | 2.6343322973029859 | 0.2168375034441117 | 5.345943 | 0.493238 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bandage_1_setting3_mode1_w1_sr0.5_ceres.log` |
| bandage_1 | 4 | 1 | 0.9 | 2 | yes | 0.4184286542606452 | 0.0158591417734747 | 1.2136594477499263 | 0.0120892771824891 | 7.193926 | 0.432140 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bandage_1_setting4_mode1_w2_sr0.9_ceres.log` |
| bandage_1 | 5 | 1 | 0.9 | 3 | yes | 0.4269893372507053 | 0.0918851299595494 | 2.6464310907110189 | 0.0710386714726720 | 6.626656 | 0.507302 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bandage_1_setting5_mode1_w3_sr0.9_ceres.log` |
| bandage_1 | 6 | 2 | - | 1 | yes | 0.4173454557996819 | 0.2482999325149732 | 1.6723457001426296 | 0.1883124147264787 | 16.163686 | 0.261109 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bandage_1_setting6_mode2_w1_ceres.log` |
| bandage_2 | 1 | 1 | 0.1 | 1 | yes | 0.0007767274711358 | 0.0812366993482423 | 2.4884611257161069 | 0.0621079553371695 | 9.469316 | 0.320101 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bandage_2_setting1_mode1_w1_sr0.1_ceres.log` |
| bandage_2 | 2 | 1 | 0.3 | 1 | yes | 0.0006918648104996 | 0.0626887117054093 | 3.0297001721501182 | 0.0478882124385404 | 9.443118 | 0.362689 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bandage_2_setting2_mode1_w1_sr0.3_ceres.log` |
| bandage_2 | 3 | 1 | 0.5 | 1 | yes | 0.0006087891824222 | 0.0366034861653990 | 1.9548196385437786 | 0.0270913622448997 | 9.183936 | 0.388421 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bandage_2_setting3_mode1_w1_sr0.5_ceres.log` |
| bandage_2 | 4 | 1 | 0.9 | 2 | yes | 0.0009201184453803 | 0.0027213594102466 | 1.5939684572580706 | 0.0021008468297065 | 9.247189 | 0.361757 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bandage_2_setting4_mode1_w2_sr0.9_ceres.log` |
| bandage_2 | 5 | 1 | 0.9 | 3 | yes | 0.0008717885542610 | 0.0161301652252447 | 1.6778249771768943 | 0.0127011490119228 | 21.801578 | 0.382304 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bandage_2_setting5_mode1_w3_sr0.9_ceres.log` |
| bandage_2 | 6 | 2 | - | 1 | yes | 0.0007331758571985 | 0.0666906169191597 | 2.3350835727533776 | 0.0510490198147630 | 7.294735 | 0.245217 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/bandage_2_setting6_mode2_w1_ceres.log` |
| cave_2 | 1 | 1 | 0.1 | 1 | yes | 1.2468454544488812 | 0.6198652446756278 | 2.7371799631612692 | 0.6155855489072113 | 23.682869 | 8.622233 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/cave_2_setting1_mode1_w1_sr0.1_ceres.log` |
| cave_2 | 2 | 1 | 0.3 | 1 | yes | 1.5796877200120296 | 0.7156052130084840 | 2.5109481493116741 | 0.6635077600843750 | 43.204981 | 28.259812 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/cave_2_setting2_mode1_w1_sr0.3_ceres.log` |
| cave_2 | 3 | 1 | 0.5 | 1 | yes | 0.9178424802297663 | 0.4696160422513586 | 2.8808384451657489 | 0.3853080540041759 | 21.744299 | 8.349208 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/cave_2_setting3_mode1_w1_sr0.5_ceres.log` |
| cave_2 | 4 | 1 | 0.9 | 2 | yes | 0.4979049021992650 | 0.3420394648055688 | 2.5902858002862827 | 0.2803963493326026 | 25.744018 | 2.814552 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/cave_2_setting4_mode1_w2_sr0.9_ceres.log` |
| cave_2 | 5 | 1 | 0.9 | 3 | yes | 0.5335385285342812 | 0.3643512679925200 | 2.4873166525148149 | 0.2610499997307920 | 58.826645 | 2.279332 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/cave_2_setting5_mode1_w3_sr0.9_ceres.log` |
| cave_2 | 6 | 2 | - | 1 | yes | 0.9280234887697629 | 0.7618065782426517 | 2.7657512286742629 | 0.5320434654201583 | 13.268537 | 1.373730 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/cave_2_setting6_mode2_w1_ceres.log` |
| cave_4 | 1 | 1 | 0.1 | 1 | yes | 0.0058293565371936 | 0.0078569605519593 | 0.0497599696189172 | 0.0052839519604432 | 5.518535 | 0.315729 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/cave_4_setting1_mode1_w1_sr0.1_ceres.log` |
| cave_4 | 2 | 1 | 0.3 | 1 | yes | 0.0050145383598477 | 0.0115103750623583 | 0.0997214819176467 | 0.0079180572582816 | 4.883705 | 0.355995 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/cave_4_setting2_mode1_w1_sr0.3_ceres.log` |
| cave_4 | 3 | 1 | 0.5 | 1 | yes | 0.0016484723377582 | 0.0052046635838410 | 0.0473644323833166 | 0.0036057353934296 | 6.896336 | 0.409336 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/cave_4_setting3_mode1_w1_sr0.5_ceres.log` |
| cave_4 | 4 | 1 | 0.9 | 2 | yes | 0.0018505672061050 | 0.0052149530528657 | 0.0620678725031705 | 0.0036552065732868 | 6.649494 | 0.507541 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/cave_4_setting4_mode1_w2_sr0.9_ceres.log` |
| cave_4 | 5 | 1 | 0.9 | 3 | yes | 0.0014989784574918 | 0.0050762171146503 | 0.0532909699244539 | 0.0035820901917640 | 7.763437 | 0.452124 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/cave_4_setting5_mode1_w3_sr0.9_ceres.log` |
| cave_4 | 6 | 2 | - | 1 | yes | 0.0019047350449947 | 0.0072153033338972 | 0.0495284833372902 | 0.0050749923478603 | 8.272771 | 0.254185 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/cave_4_setting6_mode2_w1_ceres.log` |
| market_2 | 1 | 1 | 0.1 | 1 | yes | 0.0089672071858380 | 0.1426777100073876 | 2.7382257913908346 | 0.0910964493771648 | 10.354903 | 0.360972 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/market_2_setting1_mode1_w1_sr0.1_ceres.log` |
| market_2 | 2 | 1 | 0.3 | 1 | yes | 0.0023520702694923 | 0.0070408202190729 | 0.5286200706546382 | 0.0052069622491171 | 5.862213 | 0.392001 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/market_2_setting2_mode1_w1_sr0.3_ceres.log` |
| market_2 | 3 | 1 | 0.5 | 1 | yes | 0.0067415205565403 | 0.0652735774482708 | 3.0149813069769773 | 0.0427951056160377 | 23.257171 | 0.448257 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/market_2_setting3_mode1_w1_sr0.5_ceres.log` |
| market_2 | 4 | 1 | 0.9 | 2 | yes | 0.0011767181387724 | 0.0081483960653578 | 0.0609950120638368 | 0.0060451390324256 | 29.561299 | 0.434808 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/market_2_setting4_mode1_w2_sr0.9_ceres.log` |
| market_2 | 5 | 1 | 0.9 | 3 | yes | 0.0013469737049261 | 0.0089959910122178 | 0.0725082726084799 | 0.0066586602369455 | 11.114515 | 0.496999 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/market_2_setting5_mode1_w3_sr0.9_ceres.log` |
| market_2 | 6 | 2 | - | 1 | yes | 0.0014904643878130 | 0.0055245494428763 | 1.5329664387117157 | 0.0037917545780851 | 17.681373 | 0.326197 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/market_2_setting6_mode2_w1_ceres.log` |
| market_5 | 1 | 1 | 0.1 | 1 | yes | 0.1228140562844199 | 0.0071091028465420 | 1.7937156978355728 | 0.0052771710960203 | 59.508432 | 1.074160 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/market_5_setting1_mode1_w1_sr0.1_ceres.log` |
| market_5 | 2 | 1 | 0.3 | 1 | yes | 0.1159276165911139 | 0.0066295775274167 | 1.7717688626242940 | 0.0049044322143086 | 36.726650 | 1.508216 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/market_5_setting2_mode1_w1_sr0.3_ceres.log` |
| market_5 | 3 | 1 | 0.5 | 1 | yes | 0.0684592177297694 | 0.0050731016489507 | 1.6918369339288184 | 0.0036424080754042 | 12.170880 | 1.666805 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/market_5_setting3_mode1_w1_sr0.5_ceres.log` |
| market_5 | 4 | 1 | 0.9 | 2 | yes | 0.0678931414652847 | 0.0055206049506531 | 1.9309639354299433 | 0.0041360039241507 | 8.971287 | 1.839253 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/market_5_setting4_mode1_w2_sr0.9_ceres.log` |
| market_5 | 5 | 1 | 0.9 | 3 | yes | 0.0678245219429474 | 0.0042472476706360 | 1.7870411123360037 | 0.0029868098257573 | 8.882025 | 2.277951 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/market_5_setting5_mode1_w3_sr0.9_ceres.log` |
| market_5 | 6 | 2 | - | 1 | yes | 0.1580004181437971 | 0.0184983348981560 | 3.0917727251925897 | 0.0115004706980182 | 7.081128 | 0.463752 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/market_5_setting6_mode2_w1_ceres.log` |
| market_6 | 1 | 1 | 0.1 | 1 | yes | 0.0178614802969760 | 0.0174224994893681 | 2.1256886680314850 | 0.0131256723797326 | 8.674309 | 0.330153 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/market_6_setting1_mode1_w1_sr0.1_ceres.log` |
| market_6 | 2 | 1 | 0.3 | 1 | yes | 0.0096988592790175 | 0.0156140001988169 | 1.8111272091778197 | 0.0112965874104202 | 4.885486 | 0.360587 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/market_6_setting2_mode1_w1_sr0.3_ceres.log` |
| market_6 | 3 | 1 | 0.5 | 1 | yes | 0.0202021662663422 | 0.0250450487923733 | 2.2823554860741009 | 0.0175971361321713 | 17.315886 | 0.390451 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/market_6_setting3_mode1_w1_sr0.5_ceres.log` |
| market_6 | 4 | 1 | 0.9 | 2 | yes | 0.0172771428572000 | 0.0222112045039229 | 2.2235040920809803 | 0.0162698953570194 | 8.802141 | 0.471606 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/market_6_setting4_mode1_w2_sr0.9_ceres.log` |
| market_6 | 5 | 1 | 0.9 | 3 | yes | 0.0169070262096413 | 0.0148986727516009 | 2.2146054910703192 | 0.0110657420993186 | 64.757260 | 0.467726 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/market_6_setting5_mode1_w3_sr0.9_ceres.log` |
| market_6 | 6 | 2 | - | 1 | yes | 0.0136093662140951 | 0.0307804706902277 | 1.9773489738413381 | 0.0214819034519996 | 5.121094 | 0.258215 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/market_6_setting6_mode2_w1_ceres.log` |
| mountain_1 | 1 | 1 | 0.1 | 1 | yes | 0.0010494831593923 | 0.0090763699099206 | 0.0219344794533233 | 0.0062985253820100 | 7.091963 | 0.318160 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/mountain_1_setting1_mode1_w1_sr0.1_ceres.log` |
| mountain_1 | 2 | 1 | 0.3 | 1 | yes | 0.0010119977275738 | 0.0087517647123871 | 0.0341495197979634 | 0.0060954981266048 | 6.795621 | 0.357607 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/mountain_1_setting2_mode1_w1_sr0.3_ceres.log` |
| mountain_1 | 3 | 1 | 0.5 | 1 | yes | 0.0012066146255299 | 0.0142319904484802 | 0.0900987141542552 | 0.0094282609686825 | 11.266563 | 0.408123 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/mountain_1_setting3_mode1_w1_sr0.5_ceres.log` |
| mountain_1 | 4 | 1 | 0.9 | 2 | yes | 0.0011882104589011 | 0.0107249336323990 | 0.0637283338289964 | 0.0075007607853361 | 20.346559 | 0.427201 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/mountain_1_setting4_mode1_w2_sr0.9_ceres.log` |
| mountain_1 | 5 | 1 | 0.9 | 3 | yes | 0.0010824568248249 | 0.0093368916516573 | 0.0403934151608210 | 0.0065257674827409 | 8.990801 | 0.427754 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/mountain_1_setting5_mode1_w3_sr0.9_ceres.log` |
| mountain_1 | 6 | 2 | - | 1 | yes | 0.0009081858574971 | 0.0074843371039810 | 0.0451201002121288 | 0.0054111333022974 | 8.318785 | 0.253499 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/mountain_1_setting6_mode2_w1_ceres.log` |
| shaman_2 | 1 | 1 | 0.1 | 1 | yes | 0.0248063198415943 | 0.0548962746148753 | 1.1555920396672448 | 0.0426585679953532 | 7.318323 | 0.367621 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/shaman_2_setting1_mode1_w1_sr0.1_ceres.log` |
| shaman_2 | 2 | 1 | 0.3 | 1 | yes | 0.0248586225845454 | 0.0521862145024292 | 0.6043335537743897 | 0.0405484781977422 | 8.601578 | 0.391960 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/shaman_2_setting2_mode1_w1_sr0.3_ceres.log` |
| shaman_2 | 3 | 1 | 0.5 | 1 | yes | 0.0247899960639192 | 0.0549394980546053 | 2.5262396374003884 | 0.0426916365427544 | 6.952648 | 0.486220 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/shaman_2_setting3_mode1_w1_sr0.5_ceres.log` |
| shaman_2 | 4 | 1 | 0.9 | 2 | yes | 0.0241518161052075 | 0.0218132704453374 | 1.2320531356623468 | 0.0168819493837827 | 7.951764 | 0.387749 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/shaman_2_setting4_mode1_w2_sr0.9_ceres.log` |
| shaman_2 | 5 | 1 | 0.9 | 3 | yes | 0.0243361861781685 | 0.0400670407440347 | 0.8728902500843430 | 0.0310295849475760 | 7.712500 | 0.439438 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/shaman_2_setting5_mode1_w3_sr0.9_ceres.log` |
| shaman_2 | 6 | 2 | - | 1 | yes | 0.0233182252284146 | 0.0334533475751542 | 1.7736983660433436 | 0.0260006678872557 | 16.840087 | 0.254327 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/shaman_2_setting6_mode2_w1_ceres.log` |
| shaman_3 | 1 | 1 | 0.1 | 1 | yes | 0.0111275312363082 | 0.0035466203327263 | 0.6686222797426423 | 0.0026987595350215 | 13.842808 | 0.354223 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/shaman_3_setting1_mode1_w1_sr0.1_ceres.log` |
| shaman_3 | 2 | 1 | 0.3 | 1 | yes | 0.0112691405010846 | 0.0035008025011579 | 0.7212448223029664 | 0.0026599934262768 | 13.831773 | 0.439271 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/shaman_3_setting2_mode1_w1_sr0.3_ceres.log` |
| shaman_3 | 3 | 1 | 0.5 | 1 | yes | 0.0113694246570570 | 0.0035271271404803 | 0.7554205929885623 | 0.0026804516605895 | 13.127880 | 0.477679 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/shaman_3_setting3_mode1_w1_sr0.5_ceres.log` |
| shaman_3 | 4 | 1 | 0.9 | 2 | yes | 0.0089130245180887 | 0.0042035973422136 | 1.1383930557283302 | 0.0029985940943932 | 5.557283 | 0.624603 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/shaman_3_setting4_mode1_w2_sr0.9_ceres.log` |
| shaman_3 | 5 | 1 | 0.9 | 3 | yes | 0.0112403485147526 | 0.0055717531415912 | 1.3969372919648242 | 0.0040350062559418 | 8.921070 | 0.594084 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/shaman_3_setting5_mode1_w3_sr0.9_ceres.log` |
| shaman_3 | 6 | 2 | - | 1 | yes | 0.0120433172475370 | 0.0036279184776693 | 1.2610347292395330 | 0.0027526921444291 | 64.083290 | 0.258728 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/shaman_3_setting6_mode2_w1_ceres.log` |
| sleeping_1 | 1 | 1 | 0.1 | 1 | yes | 0.0441716808918066 | 0.0053484000718151 | 2.4887811683128658 | 0.0038309174396277 | 9.219817 | 0.314603 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/sleeping_1_setting1_mode1_w1_sr0.1_ceres.log` |
| sleeping_1 | 2 | 1 | 0.3 | 1 | yes | 0.0450446309834292 | 0.0054319848729492 | 2.4145955619092931 | 0.0038832069829597 | 8.285745 | 0.359424 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/sleeping_1_setting2_mode1_w1_sr0.3_ceres.log` |
| sleeping_1 | 3 | 1 | 0.5 | 1 | yes | 0.0448205238981278 | 0.0053656835123897 | 2.3950309784501114 | 0.0038368094779164 | 13.478392 | 0.398394 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/sleeping_1_setting3_mode1_w1_sr0.5_ceres.log` |
| sleeping_1 | 4 | 1 | 0.9 | 2 | yes | 0.0620503197947459 | 0.0260029631836629 | 1.7160923588432759 | 0.0168141065609729 | 4.710570 | 0.398988 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/sleeping_1_setting4_mode1_w2_sr0.9_ceres.log` |
| sleeping_1 | 5 | 1 | 0.9 | 3 | yes | 0.0520517492715602 | 0.0154807295581378 | 2.9443070342113966 | 0.0099130798719846 | 15.176007 | 0.504160 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/sleeping_1_setting5_mode1_w3_sr0.9_ceres.log` |
| sleeping_1 | 6 | 2 | - | 1 | yes | 0.0447344109745067 | 0.0059110282379911 | 2.5923898219453334 | 0.0042026790689748 | 37.528890 | 0.268855 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/sleeping_1_setting6_mode2_w1_ceres.log` |
| sleeping_2 | 1 | 1 | 0.1 | 1 | yes | 0.0004140098402238 | 0.0006475426743629 | 0.0141791542814511 | 0.0004406340282965 | 7.419594 | 0.260622 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/sleeping_2_setting1_mode1_w1_sr0.1_ceres.log` |
| sleeping_2 | 2 | 1 | 0.3 | 1 | yes | 0.0004143152167541 | 0.0006488928384267 | 0.0142378888179068 | 0.0004416327285516 | 7.211372 | 0.284517 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/sleeping_2_setting2_mode1_w1_sr0.3_ceres.log` |
| sleeping_2 | 3 | 1 | 0.5 | 1 | yes | 0.0004133330003709 | 0.0006490446097015 | 0.0147686082139123 | 0.0004419523977973 | 7.252691 | 0.295516 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/sleeping_2_setting3_mode1_w1_sr0.5_ceres.log` |
| sleeping_2 | 4 | 1 | 0.9 | 2 | yes | 0.0003438172800463 | 0.0007207391978731 | 0.0603181039799991 | 0.0004933021819576 | 19.849697 | 0.377363 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/sleeping_2_setting4_mode1_w2_sr0.9_ceres.log` |
| sleeping_2 | 5 | 1 | 0.9 | 3 | yes | 0.0003558804357098 | 0.0006767580972499 | 0.0180125864587453 | 0.0004602393614242 | 10.790899 | 0.390050 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/sleeping_2_setting5_mode1_w3_sr0.9_ceres.log` |
| sleeping_2 | 6 | 2 | - | 1 | yes | 0.0004664415981625 | 0.0006737001391464 | 0.0076066596344261 | 0.0004506501610783 | 11.220200 | 0.268073 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/sleeping_2_setting6_mode2_w1_ceres.log` |
| temple_2 | 1 | 1 | 0.1 | 1 | yes | 0.0665824356729387 | 0.0461846431954396 | 0.2678704620273299 | 0.0327586217393926 | 13.073086 | 0.443722 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/temple_2_setting1_mode1_w1_sr0.1_ceres.log` |
| temple_2 | 2 | 1 | 0.3 | 1 | yes | 0.1137212296689051 | 0.0416628521005500 | 2.8983942452664548 | 0.0304377264304520 | 15.114581 | 0.467575 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/temple_2_setting2_mode1_w1_sr0.3_ceres.log` |
| temple_2 | 3 | 1 | 0.5 | 1 | yes | 0.0637710931919106 | 0.0519815282974928 | 2.9974700837589374 | 0.0368665659044259 | 36.620580 | 0.541643 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/temple_2_setting3_mode1_w1_sr0.5_ceres.log` |
| temple_2 | 4 | 1 | 0.9 | 2 | yes | 0.0732052826687416 | 0.0621399640368687 | 0.3006941522442887 | 0.0437865591670890 | 70.627855 | 1.131449 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/temple_2_setting4_mode1_w2_sr0.9_ceres.log` |
| temple_2 | 5 | 1 | 0.9 | 3 | yes | 0.0771565749840380 | 0.0410518444215670 | 2.7886339211732842 | 0.0294967865871669 | 66.647807 | 0.825375 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/temple_2_setting5_mode1_w3_sr0.9_ceres.log` |
| temple_2 | 6 | 2 | - | 1 | yes | 0.0748961852966817 | 0.0560926196876776 | 0.4372528522143436 | 0.0397804135627457 | 13.924324 | 0.352094 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/temple_2_setting6_mode2_w1_ceres.log` |
| temple_3 | 1 | 1 | 0.1 | 1 | yes | 0.0022444702185024 | 0.2177098343767482 | 1.5318648528905703 | 0.1501310319563827 | 9.198068 | 1.186441 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/temple_3_setting1_mode1_w1_sr0.1_ceres.log` |
| temple_3 | 2 | 1 | 0.3 | 1 | yes | 0.0028130079001201 | 0.2705952687188131 | 1.9282982145054846 | 0.1866645632203316 | 8.808960 | 1.530500 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/temple_3_setting2_mode1_w1_sr0.3_ceres.log` |
| temple_3 | 3 | 1 | 0.5 | 1 | yes | 0.0028431523305994 | 0.2481299487710832 | 1.4318857267651979 | 0.1679834838822336 | 9.837823 | 1.799076 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/temple_3_setting3_mode1_w1_sr0.5_ceres.log` |
| temple_3 | 4 | 1 | 0.9 | 2 | yes | 0.0045248765592398 | 0.1863547372990674 | 2.6044525123215494 | 0.1289493211467603 | 10.072533 | 1.388622 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/temple_3_setting4_mode1_w2_sr0.9_ceres.log` |
| temple_3 | 5 | 1 | 0.9 | 3 | yes | 0.0145262986818388 | 1.5666408105110721 | 0.7209898251518915 | 0.9128059642142409 | 25.800095 | 1.432676 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/temple_3_setting5_mode1_w3_sr0.9_ceres.log` |
| temple_3 | 6 | 2 | - | 1 | yes | 0.0020138142936777 | 0.3116695040669779 | 0.6560596146963420 | 0.2108569343715760 | 7.901450 | 0.560798 | 3 | `Python/logs/sintel_xm_penalty_strong/solver/temple_3_setting6_mode2_w1_ceres.log` |
