# Replica Solver Experiments (8 datasets × 1 mode(s))

Run configuration:
- Modes: depth=da3, with Ceres

## Summary table

| Dataset | Mode | Status | RPE-R | RPE-T | ATE-R | ATE-T | ATE (m) | RPE trans (m) | RPE rot (deg) | Total (s) | Solve (s) | Rank | Log |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| office1_seq1 | depth=da3, with Ceres | ok | 0.00100013 | 0.00236168 | 0.03202 | 0.00176398 | 0.002126 | 0.000644 | 0.017318 | 14.9447 | 0.550449 | 3 | `Python/logs/replica_solver_16/office1_seq1__da3_ceres.log` |
| office1_seq2 | depth=da3, with Ceres | ok | 0.00119097 | 0.0021551 | 0.0153403 | 0.00155908 | 0.001991 | 0.000776 | 0.019787 | 13.4572 | 0.675004 | 3 | `Python/logs/replica_solver_16/office1_seq2__da3_ceres.log` |
| office2_seq1 | depth=da3, with Ceres | ok | 0.000724486 | 0.00132624 | 0.0239271 | 0.000941573 | 0.001333 | 0.000582 | 0.011392 | 13.3604 | 0.582723 | 3 | `Python/logs/replica_solver_16/office2_seq1__da3_ceres.log` |
| office2_seq2 | depth=da3, with Ceres | ok | 0.00186222 | 0.00417538 | 0.0675711 | 0.00312534 | 0.004199 | 0.001233 | 0.023035 | 18.2184 | 0.949186 | 3 | `Python/logs/replica_solver_16/office2_seq2__da3_ceres.log` |
| room1_seq1 | depth=da3, with Ceres | ok | 0.00117954 | 0.00283442 | 0.0346374 | 0.00208838 | 0.002705 | 0.001038 | 0.020483 | 17.1498 | 0.588125 | 3 | `Python/logs/replica_solver_16/room1_seq1__da3_ceres.log` |
| room1_seq2 | depth=da3, with Ceres | ok | 0.00156257 | 0.00243239 | 0.066767 | 0.00171221 | 0.00248 | 0.001072 | 0.020637 | 12.8533 | 0.527711 | 3 | `Python/logs/replica_solver_16/room1_seq2__da3_ceres.log` |
| room2_seq1 | depth=da3, with Ceres | ok | 0.000730475 | 0.0012531 | 0.0484466 | 0.000886082 | 0.001414 | 0.001112 | 0.021208 | 12.7154 | 0.62125 | 3 | `Python/logs/replica_solver_16/room2_seq1__da3_ceres.log` |
| room2_seq2 | depth=da3, with Ceres | ok | 0.00135878 | 0.000937275 | 0.0597603 | 0.000662811 | 0.000936 | 0.000822 | 0.013442 | 15.294 | 0.930032 | 3 | `Python/logs/replica_solver_16/room2_seq2__da3_ceres.log` |

## Per-experiment metric blocks

### office1_seq1 — depth=da3, with Ceres

Status: **ok**

```text
RPE-R:  0.0010001337898 RPE-T:  0.00236167924953 ATE-R:  0.0320200146215 ATE-T:  0.00176397541645

============================================================
MoSca-style eval (evo)
============================================================
ATE ↓:        0.002126 m
RPE trans ↓:  0.000644 m
RPE rot ↓:    0.017318°
============================================================
Total time:  14.944653484
Solve time:  0.550449079979
Optimal rank:  3
```

### office1_seq2 — depth=da3, with Ceres

Status: **ok**

```text
RPE-R:  0.00119097307601 RPE-T:  0.00215510410195 ATE-R:  0.0153402823842 ATE-T:  0.00155908301402

============================================================
MoSca-style eval (evo)
============================================================
ATE ↓:        0.001991 m
RPE trans ↓:  0.000776 m
RPE rot ↓:    0.019787°
============================================================
Total time:  13.457186222
Solve time:  0.675004340999
Optimal rank:  3
```

### office2_seq1 — depth=da3, with Ceres

Status: **ok**

```text
RPE-R:  0.000724486166814 RPE-T:  0.00132624280057 ATE-R:  0.0239270856277 ATE-T:  0.000941572640242

============================================================
MoSca-style eval (evo)
============================================================
ATE ↓:        0.001333 m
RPE trans ↓:  0.000582 m
RPE rot ↓:    0.011392°
============================================================
Total time:  13.360410947
Solve time:  0.582722838997
Optimal rank:  3
```

### office2_seq2 — depth=da3, with Ceres

Status: **ok**

```text
RPE-R:  0.00186222210446 RPE-T:  0.00417537913634 ATE-R:  0.0675710659584 ATE-T:  0.00312534379692

============================================================
MoSca-style eval (evo)
============================================================
ATE ↓:        0.004199 m
RPE trans ↓:  0.001233 m
RPE rot ↓:    0.023035°
============================================================
Total time:  18.218425072
Solve time:  0.949185807956
Optimal rank:  3
```

### room1_seq1 — depth=da3, with Ceres

Status: **ok**

```text
RPE-R:  0.00117954090193 RPE-T:  0.00283441965929 ATE-R:  0.0346374225921 ATE-T:  0.00208838392859

============================================================
MoSca-style eval (evo)
============================================================
ATE ↓:        0.002705 m
RPE trans ↓:  0.001038 m
RPE rot ↓:    0.020483°
============================================================
Total time:  17.149790942
Solve time:  0.58812472102
Optimal rank:  3
```

### room1_seq2 — depth=da3, with Ceres

Status: **ok**

```text
RPE-R:  0.00156256598281 RPE-T:  0.00243239135787 ATE-R:  0.0667669837621 ATE-T:  0.00171221009131

============================================================
MoSca-style eval (evo)
============================================================
ATE ↓:        0.00248 m
RPE trans ↓:  0.001072 m
RPE rot ↓:    0.020637°
============================================================
Total time:  12.853326886
Solve time:  0.527710596012
Optimal rank:  3
```

### room2_seq1 — depth=da3, with Ceres

Status: **ok**

```text
RPE-R:  0.000730474578151 RPE-T:  0.00125310094602 ATE-R:  0.048446637703 ATE-T:  0.000886082227781

============================================================
MoSca-style eval (evo)
============================================================
ATE ↓:        0.001414 m
RPE trans ↓:  0.001112 m
RPE rot ↓:    0.021208°
============================================================
Total time:  12.715396464
Solve time:  0.621249609016
Optimal rank:  3
```

### room2_seq2 — depth=da3, with Ceres

Status: **ok**

```text
RPE-R:  0.00135878364731 RPE-T:  0.000937274965817 ATE-R:  0.0597602591925 ATE-T:  0.000662811214873

============================================================
MoSca-style eval (evo)
============================================================
ATE ↓:        0.000936 m
RPE trans ↓:  0.000822 m
RPE rot ↓:    0.013442°
============================================================
Total time:  15.293991669
Solve time:  0.930031585012
Optimal rank:  3
```

