# Sintel Penalty Subset Comparison

Generated: `2026-04-22T17:37:53-04:00`

Subset scenes: `alley_1`, `cave_2`, `market_6`, `shaman_3`, `temple_3`

Scoring: lower is better. `Avg rank score` is the mean per-run rank across matched `ATE`, `RPE trans`, and `RPE rot` values, so it compares candidates without mixing units.

Winner: `strong`

| Candidate | PENALTY_T | PENALTY_T_CERES_REL | WINDOW | PENALTY_T_CERES_ACC | Mean ATE (m) | Mean RPE trans (m) | Mean RPE rot (deg) | Avg rank score | Metric wins | Missing rows | MoSca report |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| base | 10.00N | 0.10 x PENALTY_T | 3 | 10.00 x PENALTY_T | 0.258694 | 0.089908 | 1.594666 | 2.077778 | 28 | 0 | `Python/sintel_penalty_subset_base_mosca.md` |
| light | 5.00N | 0.05 x PENALTY_T | 2 | 5.00 x PENALTY_T | 0.250135 | 0.107981 | 3.863415 | 1.966667 | 38 | 0 | `Python/sintel_penalty_subset_light_mosca.md` |
| strong | 15.00N | 0.20 x PENALTY_T | 3 | 20.00 x PENALTY_T | 0.246824 | 0.093029 | 1.991379 | 1.955556 | 24 | 0 | `Python/sintel_penalty_subset_strong_mosca.md` |

Chosen for full sweep:

- `strong` with `PENALTY_T=15.00N`, `PENALTY_T_CERES_REL=0.20 x PENALTY_T`, `PENALTY_T_CERES_WINDOW=3`, `PENALTY_T_CERES_ACC=20.00 x PENALTY_T`.
