# Distributed worker-island + NN policy benchmark

Generated: 2026-06-01T11:41:22.716Z

## Scope

- Runs real `DistributedManager` worker-thread islands.
- Uses a shared `MigrationHub` with atomic migration counters.
- Uses the existing `MLP` as a scorer, then applies confidence, validation, and guard-holdout veto/fallback policy.
- This is the production-runtime proof lane; the wide seed sweep is part of `release:gate`.

## Summary

- Mean single-island MAE: 2.324315
- Mean policy MAE: 0.053446
- Mean robust holdout-only MAE: 0.053446
- Mean oracle-best MAE: 0.051794
- Median lift: 8.246232x
- Median lift on non-degenerate seeds: 8.246232x
- Success rate: 100%
- Non-degenerate success rate: 100%
- Non-regression rate vs single baseline: 100%
- Completed seeds: 24/24
- Runtime timeouts: 0
- Runtime errors: 0
- Single-baseline solved seeds: 0
- Single-baseline fallbacks: 0
- Mean regret vs oracle: 0.001652
- Policy beats robust holdout-only rate: 0%
- NN holdout accuracy: 0.822574
- NN holdout Brier: 0.140804
- NN holdout AUC: 0.875337
- Hub drops: 356
- Hub pull successes: 228
- Fallbacks: 4

| Seed | Single MAE | Single val MAE | Single guard MAE | Policy MAE | Policy guard MAE | Holdout-only MAE | Oracle MAE | Lift | Island | NN p | Regret | Fallback | Hub pulls | Outcome |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.157757 | 0.142474 | 0.166682 | 0.042667 | 0.044085 | 0.042667 | 0.042667 | 3.697375x | 0 | 0.930392 | 0 | no | 11/16 | distributed-lift |
| 2 | 0.768867 | 0.673535 | 0.839236 | 0.114799 | 0.120072 | 0.114799 | 0.114799 | 6.697518x | 2 | 0.684844 | 0 | no | 10/16 | distributed-lift |
| 3 | 0.247474 | 0.212591 | 0.246134 | 0.157759 | 0.166683 | 0.157759 | 0.157759 | 1.568684x | 1 | 0.845984 | 0 | no | 10/16 | distributed-lift |
| 4 | 0.62951 | 0.618636 | 0.631109 | 1.306e-11 | 1.310e-11 | 1.306e-11 | 1.306e-11 | 4.821e+10x | 2 | 0.900157 | 0 | no | 4/12 | distributed-lift |
| 5 | 0.570535 | 0.526547 | 0.562227 | 0.122423 | 0.130066 | 0.122423 | 0.122423 | 4.660361x | 2 | 0.795704 | 0 | no | 14/16 | distributed-lift |
| 6 | 0.528679 | 0.521143 | 0.529641 | 0.153865 | 0.190968 | 0.153865 | 0.153865 | 3.435994x | 1 | 0.9375 | 0 | no | 10/16 | distributed-lift |
| 7 | 0.779231 | 0.643735 | 1.059085 | 0.055546 | 0.053682 | 0.055546 | 0.055546 | 14.028455x | 0 | 0.951052 | 0 | no | 10/16 | distributed-lift |
| 8 | 0.386966 | 0.379033 | 0.413881 | 0.015238 | 0.015653 | 0.015238 | 0.015238 | 25.395418x | 2 | 0.567707 | 0 | low-nn-confidence | 11/16 | distributed-lift |
| 9 | 0.60038 | 7.125e+28 | 0.599308 | 0.157757 | 0.166682 | 0.157757 | 0.157757 | 3.805723x | 1 | 0.883088 | 0 | no | 11/16 | distributed-lift |
| 10 | 0.188119 | 0.174087 | 0.195598 | 0.14881 | 0.157227 | 0.14881 | 0.14881 | 1.264155x | 2 | 0.653627 | 0 | validation-veto | 11/16 | distributed-lift |
| 11 | 0.270593 | 0.138341 | 0.332051 | 8.955e-8 | 8.973e-8 | 8.955e-8 | 8.955e-8 | 3.022e+6x | 3 | 0.782712 | 0 | no | 9/14 | distributed-lift |
| 12 | 0.36695 | 0.312266 | 0.391126 | 6.393e-12 | 6.416e-12 | 6.393e-12 | 6.393e-12 | 5.740e+10x | 2 | 0.171592 | 0 | low-nn-confidence | 3/12 | distributed-lift |
| 13 | 0.493314 | 0.412713 | 0.519703 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 1.655e+9x | 3 | 0.924974 | 0 | no | 4/12 | distributed-lift |
| 14 | 0.157756 | 0.142476 | 0.166675 | 8.955e-8 | 8.973e-8 | 8.955e-8 | 8.955e-8 | 1.762e+6x | 0 | 0.860773 | 0 | no | 12/14 | distributed-lift |
| 15 | 0.306881 | 0.285691 | 0.316855 | 0.122654 | 0.114333 | 0.122654 | 0.122654 | 2.502019x | 0 | 0.854165 | 0 | no | 11/16 | distributed-lift |
| 16 | 1.039911 | 0.115525 | 0.162025 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 3.489e+9x | 0 | 0.885905 | 0 | no | 7/12 | distributed-lift |
| 17 | 0.060859 | 0.045547 | 0.059957 | 0.042265 | 0.041397 | 0.042265 | 0.002626 | 1.439951x | 0 | 0.575711 | 0.039639 | low-nn-confidence | 10/16 | distributed-lift |
| 18 | 0.186365 | 0.1244 | 0.13638 | 0.034865 | 0.03463 | 0.034865 | 0.034865 | 5.345343x | 1 | 0.881248 | 0 | no | 11/16 | distributed-lift |
| 19 | 0.11987 | 0.109808 | 1.223128 | 0.064199 | 0.117955 | 0.064199 | 0.064199 | 1.86716x | 3 | 0.696677 | 0 | no | 14/16 | distributed-lift |
| 20 | 0.090597 | 0.065265 | 0.099378 | 0.002618 | 0.057959 | 0.002618 | 0.002618 | 34.608545x | 0 | 0.944862 | 0 | no | 12/16 | distributed-lift |
| 21 | 0.35326 | 0.343066 | 0.362177 | 2.588e-11 | 2.600e-11 | 2.588e-11 | 2.588e-11 | 1.365e+10x | 2 | 0.861752 | 0 | no | 4/12 | distributed-lift |
| 22 | 0.148652 | 0.138504 | 0.147318 | 0.033794 | 0.091108 | 0.033794 | 0.033794 | 4.39881x | 0 | 0.616102 | 0 | no | 12/16 | distributed-lift |
| 23 | 0.13161 | 0.124415 | 0.13997 | 0.013437 | 0.013441 | 0.013437 | 0.013437 | 9.794946x | 1 | 0.804351 | 0 | no | 14/16 | distributed-lift |
| 24 | 47.199434 | 46.101695 | 45.954595 | 1.112e-10 | 1.173e-10 | 1.112e-10 | 1.112e-10 | 4.245e+11x | 3 | 0.715611 | 0 | no | 3/12 | distributed-lift |

