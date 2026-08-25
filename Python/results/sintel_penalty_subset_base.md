# Sintel XM Experiment Results (with Ceres refinement)

Generated: `2026-04-22T17:13:32-04:00`

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
| alley_1 | 1 | 1 | 0.1 | 1 | yes | 0.0258549793583966 | 0.0029598696530085 | 2.9826771971364390 | 0.0021195509635705 | 16.933720 | 0.287172 | 3 | `Python/logs/sintel_penalty_subset_base/solver/alley_1_setting1_mode1_w1_sr0.1_ceres.log` |
| alley_1 | 2 | 1 | 0.3 | 1 | yes | 0.0265354188640456 | 0.0029192649581104 | 2.9945603695480747 | 0.0020929469552890 | 16.352644 | 0.351590 | 3 | `Python/logs/sintel_penalty_subset_base/solver/alley_1_setting2_mode1_w1_sr0.3_ceres.log` |
| alley_1 | 3 | 1 | 0.5 | 1 | yes | 0.0264074903934475 | 0.0029085224446932 | 2.9957306246460003 | 0.0020805778391226 | 15.632726 | 0.399821 | 3 | `Python/logs/sintel_penalty_subset_base/solver/alley_1_setting3_mode1_w1_sr0.5_ceres.log` |
| alley_1 | 4 | 1 | 0.9 | 2 | yes | 0.0066825992514007 | 0.0074123019327537 | 2.4796359229185327 | 0.0054855974342684 | 25.339584 | 0.401871 | 3 | `Python/logs/sintel_penalty_subset_base/solver/alley_1_setting4_mode1_w2_sr0.9_ceres.log` |
| alley_1 | 5 | 1 | 0.9 | 3 | yes | 0.0241133102383649 | 0.0133226669510411 | 3.0497573095991704 | 0.0094911657232215 | 6.824745 | 0.487688 | 3 | `Python/logs/sintel_penalty_subset_base/solver/alley_1_setting5_mode1_w3_sr0.9_ceres.log` |
| alley_1 | 6 | 2 | - | 1 | yes | 0.0251798781341327 | 0.0034385316815449 | 3.0232428285308020 | 0.0025371276374518 | 65.815649 | 0.244871 | 3 | `Python/logs/sintel_penalty_subset_base/solver/alley_1_setting6_mode2_w1_ceres.log` |
| cave_2 | 1 | 1 | 0.1 | 1 | yes | 1.0876621531788249 | 0.5514713287071414 | 2.7098687818740719 | 0.4427695897401641 | 26.154678 | 9.933588 | 3 | `Python/logs/sintel_penalty_subset_base/solver/cave_2_setting1_mode1_w1_sr0.1_ceres.log` |
| cave_2 | 2 | 1 | 0.3 | 1 | yes | 1.2323532771918679 | 0.5183140149467469 | 2.4203712745418153 | 0.4288535330765607 | 21.926553 | 9.212259 | 3 | `Python/logs/sintel_penalty_subset_base/solver/cave_2_setting2_mode1_w1_sr0.3_ceres.log` |
| cave_2 | 3 | 1 | 0.5 | 1 | yes | 1.0271563901835150 | 0.4801206835757663 | 2.6595816850438281 | 0.3772592238828673 | 18.819476 | 5.814420 | 3 | `Python/logs/sintel_penalty_subset_base/solver/cave_2_setting3_mode1_w1_sr0.5_ceres.log` |
| cave_2 | 4 | 1 | 0.9 | 2 | yes | 0.4968446776065898 | 0.3855026092067644 | 2.5849867447908021 | 0.2692715862308944 | 70.208083 | 2.557166 | 3 | `Python/logs/sintel_penalty_subset_base/solver/cave_2_setting4_mode1_w2_sr0.9_ceres.log` |
| cave_2 | 5 | 1 | 0.9 | 3 | yes | 0.5919744959369825 | 0.5239576820331037 | 2.4612900490025864 | 0.3971169078697670 | 30.990019 | 2.128407 | 3 | `Python/logs/sintel_penalty_subset_base/solver/cave_2_setting5_mode1_w3_sr0.9_ceres.log` |
| cave_2 | 6 | 2 | - | 1 | yes | 0.7778628441207611 | 0.8658700017732944 | 2.8511475199706120 | 0.6447417591544584 | 17.645158 | 1.471878 | 3 | `Python/logs/sintel_penalty_subset_base/solver/cave_2_setting6_mode2_w1_ceres.log` |
| market_6 | 1 | 1 | 0.1 | 1 | yes | 0.0162098119378704 | 0.0159740082741808 | 2.2178070103052105 | 0.0116801660447367 | 11.547734 | 0.332908 | 3 | `Python/logs/sintel_penalty_subset_base/solver/market_6_setting1_mode1_w1_sr0.1_ceres.log` |
| market_6 | 2 | 1 | 0.3 | 1 | yes | 0.0087829280044085 | 0.0193249099806679 | 1.6610604853592872 | 0.0137037116798258 | 5.495098 | 0.366092 | 3 | `Python/logs/sintel_penalty_subset_base/solver/market_6_setting2_mode1_w1_sr0.3_ceres.log` |
| market_6 | 3 | 1 | 0.5 | 1 | yes | 0.0170203841606616 | 0.0152608284411748 | 2.2409996380974397 | 0.0110428782371365 | 64.415814 | 0.373458 | 3 | `Python/logs/sintel_penalty_subset_base/solver/market_6_setting3_mode1_w1_sr0.5_ceres.log` |
| market_6 | 4 | 1 | 0.9 | 2 | yes | 0.0204378462578658 | 0.0470784950988432 | 2.0162432401774719 | 0.0354163633687910 | 8.584056 | 0.516004 | 3 | `Python/logs/sintel_penalty_subset_base/solver/market_6_setting4_mode1_w2_sr0.9_ceres.log` |
| market_6 | 5 | 1 | 0.9 | 3 | yes | 0.0162043702600158 | 0.0162383181126285 | 2.2057844110577354 | 0.0120776075979907 | 9.990677 | 0.463694 | 3 | `Python/logs/sintel_penalty_subset_base/solver/market_6_setting5_mode1_w3_sr0.9_ceres.log` |
| market_6 | 6 | 2 | - | 1 | yes | 0.0152140379646589 | 0.0203177752308711 | 2.0532569498390236 | 0.0145729542382156 | 18.018237 | 0.261204 | 3 | `Python/logs/sintel_penalty_subset_base/solver/market_6_setting6_mode2_w1_ceres.log` |
| shaman_3 | 1 | 1 | 0.1 | 1 | yes | 0.0111885298694693 | 0.0040601449887166 | 1.2023978667652537 | 0.0030267365951376 | 9.558985 | 0.352500 | 3 | `Python/logs/sintel_penalty_subset_base/solver/shaman_3_setting1_mode1_w1_sr0.1_ceres.log` |
| shaman_3 | 2 | 1 | 0.3 | 1 | yes | 0.0111950012844783 | 0.0040685139304552 | 0.6727587325859016 | 0.0030459285838823 | 8.105798 | 0.423450 | 3 | `Python/logs/sintel_penalty_subset_base/solver/shaman_3_setting2_mode1_w1_sr0.3_ceres.log` |
| shaman_3 | 3 | 1 | 0.5 | 1 | yes | 0.0112430168748364 | 0.0040660969964462 | 0.6908687610940251 | 0.0030432925223037 | 10.588549 | 0.488433 | 3 | `Python/logs/sintel_penalty_subset_base/solver/shaman_3_setting3_mode1_w1_sr0.5_ceres.log` |
| shaman_3 | 4 | 1 | 0.9 | 2 | yes | 0.0106138328291914 | 0.0068026466173700 | 2.4335314705045734 | 0.0049783889748582 | 5.655116 | 0.618208 | 3 | `Python/logs/sintel_penalty_subset_base/solver/shaman_3_setting4_mode1_w2_sr0.9_ceres.log` |
| shaman_3 | 5 | 1 | 0.9 | 3 | yes | 0.0093077981560478 | 0.0037929429047787 | 2.6221824004242968 | 0.0027619076523841 | 5.570816 | 0.654740 | 3 | `Python/logs/sintel_penalty_subset_base/solver/shaman_3_setting5_mode1_w3_sr0.9_ceres.log` |
| shaman_3 | 6 | 2 | - | 1 | yes | 0.0127243099752482 | 0.0033709672389723 | 1.3237758249387941 | 0.0024890368503168 | 4.475761 | 0.266797 | 3 | `Python/logs/sintel_penalty_subset_base/solver/shaman_3_setting6_mode2_w1_ceres.log` |
| temple_3 | 1 | 1 | 0.1 | 1 | yes | 0.0031253629674714 | 0.3281472775894557 | 0.6697966804779532 | 0.2217497134114771 | 9.701343 | 1.219685 | 3 | `Python/logs/sintel_penalty_subset_base/solver/temple_3_setting1_mode1_w1_sr0.1_ceres.log` |
| temple_3 | 2 | 1 | 0.3 | 1 | yes | 0.0061574168460540 | 0.5172036737038337 | 0.7455675370597303 | 0.3521601664893305 | 9.107750 | 1.628526 | 3 | `Python/logs/sintel_penalty_subset_base/solver/temple_3_setting2_mode1_w1_sr0.3_ceres.log` |
| temple_3 | 3 | 1 | 0.5 | 1 | yes | 0.0019127439824346 | 0.2495257105492766 | 0.4722937293024896 | 0.1738899646637232 | 36.644323 | 2.155476 | 3 | `Python/logs/sintel_penalty_subset_base/solver/temple_3_setting3_mode1_w1_sr0.5_ceres.log` |
| temple_3 | 4 | 1 | 0.9 | 2 | yes | 0.0023595258338976 | 0.2962426640424451 | 1.3533757619065738 | 0.2017539775226528 | 67.056382 | 1.343525 | 3 | `Python/logs/sintel_penalty_subset_base/solver/temple_3_setting4_mode1_w2_sr0.9_ceres.log` |
| temple_3 | 5 | 1 | 0.9 | 3 | yes | 0.0043477865520082 | 0.3116655021646778 | 1.0961616505968661 | 0.2209222319375633 | 17.143100 | 1.809194 | 3 | `Python/logs/sintel_penalty_subset_base/solver/temple_3_setting5_mode1_w3_sr0.9_ceres.log` |
| temple_3 | 6 | 2 | - | 1 | yes | 0.0017476021771935 | 0.2608965491727602 | 2.6428223203432908 | 0.1802410313610057 | 9.113875 | 0.604449 | 3 | `Python/logs/sintel_penalty_subset_base/solver/temple_3_setting6_mode2_w1_ceres.log` |
