# Sintel XM GT-Depth Results (setting 1)

Generated: `2026-04-22T21:40:19-04:00`

Pipeline:

- Dataset root: `datasets/Sintel`
- Missing GT cameras are loaded from `datasets/MPI-Sintel-depth-training-20150305/training/camdata_left`
- GT depth is taken from `datasets/MPI-Sintel-depth-training-20150305/training/depth/<scene>` when no local `depths/` folder exists
- TAP tracking is reused from `outputs/tracks_debug.npz`; `output_tapnet_gt.npz` is rebuilt without rerunning tracking
- Solver heuristic is setting 1: `mode=1`, `sphere_ratio=0.1`, `WEIGHT_MODE=1`
- Refinement: `with Ceres refinement`

## Preparation

| Scene | Frames | Camera source | GT-depth conversion | Conversion log |
| --- | ---: | --- | --- | --- |
| alley_1 | 50 | local | reused | - |
| alley_2 | 50 | local | reused | - |
| ambush_2 | 21 | local | reused | - |
| ambush_4 | 33 | local | reused | - |
| ambush_5 | 50 | local | reused | - |
| ambush_6 | 20 | local | reused | - |
| ambush_7 | 50 | local | reused | - |
| bamboo_1 | 50 | local | reused | - |
| bamboo_2 | 50 | local | reused | - |
| bandage_1 | 50 | local | reused | - |
| bandage_2 | 50 | local | reused | - |
| cave_2 | 50 | local | reused | - |
| cave_4 | 50 | local | reused | - |
| market_2 | 50 | local | reused | - |
| market_5 | 50 | local | reused | - |
| market_6 | 40 | mpi_fallback | reused | - |
| mountain_1 | 50 | mpi_fallback | reused | - |
| shaman_2 | 50 | mpi_fallback | reused | - |
| shaman_3 | 50 | mpi_fallback | reused | - |
| sleeping_1 | 50 | mpi_fallback | reused | - |
| sleeping_2 | 50 | mpi_fallback | reused | - |
| temple_2 | 50 | local | reused | - |
| temple_3 | 50 | mpi_fallback | reused | - |

## Solver Results

| Scene | Setting | Mode | Sphere Ratio | WEIGHT_MODE | Ceres refine | RPE-R | RPE-T | ATE-R | ATE-T | Total time (s) | Solve time (s) | Optimal rank | Log |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| alley_1 | 1 | 1 | 0.1 | 1 | yes | 0.0002205537833740 | 0.0007428785343770 | 2.6387776072525333 | 0.0005189596177500 | 4.900817 | 0.473277 | 3 | `Python/logs/sintel_gt_depth_setting1/solver/alley_1_setting1_mode1_w1_sr0.1_ceres.log` |
| alley_2 | 1 | 1 | 0.1 | 1 | yes | 0.0008017413269037 | 0.0018880041082631 | 0.8609647924946000 | 0.0013850639510632 | 6.989052 | 1.392600 | 3 | `Python/logs/sintel_gt_depth_setting1/solver/alley_2_setting1_mode1_w1_sr0.1_ceres.log` |
| ambush_2 | 1 | 1 | 0.1 | 1 | yes | 0.0119641759080154 | 0.0000046817104438 | 2.3558907689710673 | 0.0000033570126226 | 5.265148 | 1.064985 | 3 | `Python/logs/sintel_gt_depth_setting1/solver/ambush_2_setting1_mode1_w1_sr0.1_ceres.log` |
| ambush_4 | 1 | 1 | 0.1 | 1 | yes | 0.0024085602670385 | 0.0237759260443953 | 0.0791239825431678 | 0.0149881243765852 | 9.749742 | 0.380223 | 3 | `Python/logs/sintel_gt_depth_setting1/solver/ambush_4_setting1_mode1_w1_sr0.1_ceres.log` |
| ambush_5 | 1 | 1 | 0.1 | 1 | yes | 0.0058935972229634 | 0.0170036385108797 | 1.6185823755351805 | 0.0115248900016584 | 10.552920 | 1.082882 | 3 | `Python/logs/sintel_gt_depth_setting1/solver/ambush_5_setting1_mode1_w1_sr0.1_ceres.log` |
| ambush_6 | 1 | 1 | 0.1 | 1 | yes | 0.0172006786262285 | 0.0396557472428689 | 0.3038274264878554 | 0.0285025720849087 | 7.277193 | 0.327988 | 3 | `Python/logs/sintel_gt_depth_setting1/solver/ambush_6_setting1_mode1_w1_sr0.1_ceres.log` |
| ambush_7 | 1 | 1 | 0.1 | 1 | yes | 0.0084349752265045 | 0.0000053895056089 | 2.0739937290421837 | 0.0000039318051557 | 5.373806 | 0.301762 | 3 | `Python/logs/sintel_gt_depth_setting1/solver/ambush_7_setting1_mode1_w1_sr0.1_ceres.log` |
| bamboo_1 | 1 | 1 | 0.1 | 1 | yes | 0.0032498314190190 | 0.0011149868214556 | 2.3539661059943615 | 0.0007710967260019 | 5.877035 | 0.276563 | 3 | `Python/logs/sintel_gt_depth_setting1/solver/bamboo_1_setting1_mode1_w1_sr0.1_ceres.log` |
| bamboo_2 | 1 | 1 | 0.1 | 1 | yes | 0.0018944611628297 | 0.0017002409725869 | 0.5236074218865899 | 0.0011942542931190 | 4.782295 | 0.288555 | 3 | `Python/logs/sintel_gt_depth_setting1/solver/bamboo_2_setting1_mode1_w1_sr0.1_ceres.log` |
| bandage_1 | 1 | 1 | 0.1 | 1 | yes | 0.3901317115961312 | 0.2513353086288944 | 1.0358082361621357 | 0.1921677099698038 | 4.452823 | 0.333928 | 3 | `Python/logs/sintel_gt_depth_setting1/solver/bandage_1_setting1_mode1_w1_sr0.1_ceres.log` |
| bandage_2 | 1 | 1 | 0.1 | 1 | yes | 0.0008407147273212 | 0.0017529716066664 | 3.1345013443269334 | 0.0013926617034375 | 3.792276 | 0.277134 | 3 | `Python/logs/sintel_gt_depth_setting1/solver/bandage_2_setting1_mode1_w1_sr0.1_ceres.log` |
| cave_2 | 1 | 1 | 0.1 | 1 | yes | 0.0089441638293704 | 0.0534631613739736 | 0.0440541071809077 | 0.0395977125798079 | 16.698493 | 2.986555 | 3 | `Python/logs/sintel_gt_depth_setting1/solver/cave_2_setting1_mode1_w1_sr0.1_ceres.log` |
| cave_4 | 1 | 1 | 0.1 | 1 | yes | 0.0026978230586054 | 0.0128306049597558 | 0.0988676368346482 | 0.0090081204992406 | 11.525653 | 0.328187 | 3 | `Python/logs/sintel_gt_depth_setting1/solver/cave_4_setting1_mode1_w1_sr0.1_ceres.log` |
| market_2 | 1 | 1 | 0.1 | 1 | yes | 1.1897739513860435 | 0.1268940165426315 | 1.7875775381432550 | 0.0874631382356379 | 42.710267 | 36.608008 | 3 | `Python/logs/sintel_gt_depth_setting1/solver/market_2_setting1_mode1_w1_sr0.1_ceres.log` |
| market_5 | 1 | 1 | 0.1 | 1 | yes | 0.0734543132573834 | 0.0047488691483758 | 1.7712359766704138 | 0.0032385044844295 | 17.333199 | 1.065151 | 3 | `Python/logs/sintel_gt_depth_setting1/solver/market_5_setting1_mode1_w1_sr0.1_ceres.log` |
| market_6 | 1 | 1 | 0.1 | 1 | yes | 0.0213453899714861 | 0.0306668140341027 | 2.1046603835851601 | 0.0215495631675609 | 4.301136 | 0.421305 | 3 | `Python/logs/sintel_gt_depth_setting1/solver/market_6_setting1_mode1_w1_sr0.1_ceres.log` |
| mountain_1 | 1 | 1 | 0.1 | 1 | yes | 0.0005562353914015 | 0.0101812605193987 | 0.1994431992637368 | 0.0061653913837005 | 33.711803 | 25.495722 | 3 | `Python/logs/sintel_gt_depth_setting1/solver/mountain_1_setting1_mode1_w1_sr0.1_ceres.log` |
| shaman_2 | 1 | 1 | 0.1 | 1 | yes | 0.0299741677968580 | 0.0031050383354489 | 2.0496509014965936 | 0.0024039418622190 | 4.944969 | 0.344992 | 3 | `Python/logs/sintel_gt_depth_setting1/solver/shaman_2_setting1_mode1_w1_sr0.1_ceres.log` |
| shaman_3 | 1 | 1 | 0.1 | 1 | yes | 0.0118242915590814 | 0.0045183339809751 | 2.2808081499544364 | 0.0033524337799592 | 4.075952 | 0.362621 | 3 | `Python/logs/sintel_gt_depth_setting1/solver/shaman_3_setting1_mode1_w1_sr0.1_ceres.log` |
| sleeping_1 | 1 | 1 | 0.1 | 1 | yes | 0.0410753992232725 | 0.0054225255246922 | 1.2773123980264500 | 0.0038456069652271 | 3.863512 | 0.308792 | 3 | `Python/logs/sintel_gt_depth_setting1/solver/sleeping_1_setting1_mode1_w1_sr0.1_ceres.log` |
| sleeping_2 | 1 | 1 | 0.1 | 1 | yes | 0.0004017313174302 | 0.0006771459890214 | 0.0141602063916036 | 0.0004625480434201 | 5.459005 | 0.286165 | 3 | `Python/logs/sintel_gt_depth_setting1/solver/sleeping_2_setting1_mode1_w1_sr0.1_ceres.log` |
| temple_2 | 1 | 1 | 0.1 | 1 | yes | 0.0010970134767884 | 0.0016254221374365 | 0.0041944113143800 | 0.0011622930589945 | 16.691638 | 7.532737 | 3 | `Python/logs/sintel_gt_depth_setting1/solver/temple_2_setting1_mode1_w1_sr0.1_ceres.log` |
| temple_3 | 1 | 1 | 0.1 | 1 | yes | 0.0013110086050393 | 0.1146660784031649 | 3.0930562793084539 | 0.0784148003968483 | 56.690349 | 11.299902 | 3 | `Python/logs/sintel_gt_depth_setting1/solver/temple_3_setting1_mode1_w1_sr0.1_ceres.log` |
