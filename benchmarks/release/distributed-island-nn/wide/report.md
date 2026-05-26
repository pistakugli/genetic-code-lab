# Distributed worker-island + NN policy benchmark

Generated: 2026-05-26T11:41:16.002Z

## Scope

- Runs real `DistributedManager` worker-thread islands.
- Uses a shared `MigrationHub` with atomic migration counters.
- Uses the existing `MLP` as a scorer, then applies confidence, validation, and guard-holdout veto/fallback policy.
- This is the production-runtime proof lane; the wide seed sweep is part of `release:gate`.

## Summary

- Mean single-island MAE: 2.324315
- Mean policy MAE: 0.049837
- Mean robust holdout-only MAE: 0.049837
- Mean oracle-best MAE: 0.047367
- Median lift: 6.98898x
- Median lift on non-degenerate seeds: 6.98898x
- Success rate: 100%
- Non-degenerate success rate: 100%
- Non-regression rate vs single baseline: 100%
- Completed seeds: 24/24
- Runtime timeouts: 0
- Runtime errors: 0
- Single-baseline solved seeds: 0
- Single-baseline fallbacks: 0
- Mean regret vs oracle: 0.00247
- Policy beats robust holdout-only rate: 0%
- NN holdout accuracy: 0.776541
- NN holdout Brier: 0.156354
- NN holdout AUC: 0.858628
- Hub drops: 356
- Hub pull successes: 225
- Fallbacks: 4

| Seed | Single MAE | Single val MAE | Single guard MAE | Policy MAE | Policy guard MAE | Holdout-only MAE | Oracle MAE | Lift | Island | NN p | Regret | Fallback | Hub pulls | Outcome |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.157757 | 0.142474 | 0.166682 | 0.040127 | 0.041848 | 0.040127 | 0.040127 | 3.931477x | 0 | 0.932732 | 0 | no | 10/16 | distributed-lift |
| 2 | 0.768867 | 0.673535 | 0.839236 | 0.11464 | 0.119743 | 0.11464 | 0.11464 | 6.706815x | 2 | 0.672299 | 0 | no | 10/16 | distributed-lift |
| 3 | 0.247474 | 0.212591 | 0.246134 | 0.146995 | 0.135026 | 0.146995 | 0.146995 | 1.683548x | 3 | 0.943889 | 0 | no | 11/16 | distributed-lift |
| 4 | 0.62951 | 0.618636 | 0.631109 | 1.306e-11 | 1.310e-11 | 1.306e-11 | 1.306e-11 | 4.821e+10x | 2 | 0.89196 | 0 | no | 3/12 | distributed-lift |
| 5 | 0.570535 | 0.526547 | 0.562227 | 0.093398 | 0.092299 | 0.093398 | 0.093398 | 6.108671x | 2 | 0.837213 | 0 | no | 14/16 | distributed-lift |
| 6 | 0.528679 | 0.521143 | 0.529641 | 0.072788 | 0.140118 | 0.072788 | 0.072788 | 7.263305x | 1 | 0.794133 | 0 | no | 12/16 | distributed-lift |
| 7 | 0.779231 | 0.643735 | 1.059085 | 0.076006 | 0.078051 | 0.076006 | 0.076006 | 10.252228x | 0 | 0.94473 | 0 | no | 14/16 | distributed-lift |
| 8 | 0.386966 | 0.379033 | 0.413881 | 0.023801 | 0.024533 | 0.023801 | 0.023801 | 16.258252x | 2 | 0.576528 | 0 | low-nn-confidence | 11/16 | distributed-lift |
| 9 | 0.60038 | 7.125e+28 | 0.599308 | 0.157757 | 0.166682 | 0.157757 | 0.157757 | 3.805723x | 1 | 0.773451 | 0 | no | 11/16 | distributed-lift |
| 10 | 0.188119 | 0.174087 | 0.195598 | 0.14881 | 0.157227 | 0.14881 | 0.14881 | 1.264155x | 2 | 0.605666 | 0 | validation-veto | 10/16 | distributed-lift |
| 11 | 0.270593 | 0.138341 | 0.332051 | 1.647e-7 | 1.655e-7 | 1.647e-7 | 1.647e-7 | 1.643e+6x | 3 | 0.737485 | 0 | no | 9/14 | distributed-lift |
| 12 | 0.36695 | 0.312266 | 0.391126 | 6.393e-12 | 6.416e-12 | 6.393e-12 | 6.393e-12 | 5.740e+10x | 2 | 0.298431 | 0 | low-nn-confidence | 3/12 | distributed-lift |
| 13 | 0.493314 | 0.412713 | 0.519703 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 1.655e+9x | 3 | 0.828968 | 0 | no | 3/12 | distributed-lift |
| 14 | 0.157756 | 0.142476 | 0.166675 | 8.619e-8 | 8.559e-8 | 8.619e-8 | 8.619e-8 | 1.830e+6x | 0 | 0.850077 | 0 | no | 12/14 | distributed-lift |
| 15 | 0.306881 | 0.285691 | 0.316855 | 0.125488 | 0.119339 | 0.125488 | 0.125488 | 2.44551x | 0 | 0.944342 | 0 | no | 11/16 | distributed-lift |
| 16 | 1.039911 | 0.115525 | 0.162025 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 3.489e+9x | 0 | 0.243265 | 0 | low-nn-confidence | 7/12 | distributed-lift |
| 17 | 0.060859 | 0.045547 | 0.059957 | 0.043694 | 0.043808 | 0.043694 | 0.002654 | 1.392856x | 0 | 0.612873 | 0.04104 | no | 11/16 | distributed-lift |
| 18 | 0.186365 | 0.1244 | 0.13638 | 0.034865 | 0.03463 | 0.034865 | 0.034865 | 5.345343x | 1 | 0.914673 | 0 | no | 11/16 | distributed-lift |
| 19 | 0.11987 | 0.109808 | 1.223128 | 0.061204 | 0.11856 | 0.061204 | 0.061204 | 1.958519x | 3 | 0.653115 | 0 | no | 14/16 | distributed-lift |
| 20 | 0.090597 | 0.065265 | 0.099378 | 0.020901 | 0.023214 | 0.020901 | 0.002668 | 4.334561x | 2 | 0.728037 | 0.018233 | no | 11/16 | distributed-lift |
| 21 | 0.35326 | 0.343066 | 0.362177 | 2.588e-11 | 2.600e-11 | 2.588e-11 | 2.588e-11 | 1.365e+10x | 2 | 0.885815 | 0 | no | 4/12 | distributed-lift |
| 22 | 0.148652 | 0.138504 | 0.147318 | 0.022138 | 0.075944 | 0.022138 | 0.022138 | 6.714656x | 0 | 0.82572 | 0 | no | 10/16 | distributed-lift |
| 23 | 0.13161 | 0.124415 | 0.13997 | 0.013468 | 0.013486 | 0.013468 | 0.013468 | 9.772405x | 1 | 0.722972 | 0 | no | 10/16 | distributed-lift |
| 24 | 47.199434 | 46.101695 | 45.954595 | 1.112e-10 | 1.173e-10 | 1.112e-10 | 1.112e-10 | 4.245e+11x | 3 | 0.585183 | 0 | no | 3/12 | distributed-lift |

