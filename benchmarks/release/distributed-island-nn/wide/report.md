# Distributed worker-island + NN policy benchmark

Generated: 2026-06-06T05:52:34.818Z

## Scope

- Runs real `DistributedManager` worker-thread islands.
- Uses a shared `MigrationHub` with atomic migration counters.
- Uses the existing `MLP` as a scorer, then applies confidence, validation, and guard-holdout veto/fallback policy.
- This is the production-runtime proof lane; the wide seed sweep is part of `release:gate`.

## Summary

- Mean single-island MAE: 2.324315
- Mean policy MAE: 0.054109
- Mean robust holdout-only MAE: 0.054109
- Mean oracle-best MAE: 0.049601
- Median lift: 8.232773x
- Median lift on non-degenerate seeds: 8.232773x
- Success rate: 100%
- Non-degenerate success rate: 100%
- Non-regression rate vs single baseline: 100%
- Completed seeds: 24/24
- Runtime timeouts: 0
- Runtime errors: 0
- Single-baseline solved seeds: 0
- Single-baseline fallbacks: 0
- Mean regret vs oracle: 0.004508
- Policy beats robust holdout-only rate: 0%
- NN holdout accuracy: 0.842899
- NN holdout Brier: 0.119027
- NN holdout AUC: 0.858974
- Hub drops: 356
- Hub pull successes: 236
- Fallbacks: 2

| Seed | Single MAE | Single val MAE | Single guard MAE | Policy MAE | Policy guard MAE | Holdout-only MAE | Oracle MAE | Lift | Island | NN p | Regret | Fallback | Hub pulls | Outcome |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.157757 | 0.142474 | 0.166682 | 0.043876 | 0.045059 | 0.043876 | 0.043876 | 3.595539x | 0 | 0.932288 | 0 | no | 11/16 | distributed-lift |
| 2 | 0.768867 | 0.673535 | 0.839236 | 0.114872 | 0.120135 | 0.114872 | 0.114872 | 6.693263x | 2 | 0.742255 | 0 | no | 13/16 | distributed-lift |
| 3 | 0.247474 | 0.212591 | 0.246134 | 0.157759 | 0.166683 | 0.157759 | 0.157759 | 1.568684x | 1 | 0.717336 | 0 | no | 11/16 | distributed-lift |
| 4 | 0.62951 | 0.618636 | 0.631109 | 1.306e-11 | 1.310e-11 | 1.306e-11 | 1.306e-11 | 4.821e+10x | 2 | 0.880244 | 0 | no | 4/12 | distributed-lift |
| 5 | 0.570535 | 0.526547 | 0.562227 | 0.1279 | 0.132887 | 0.1279 | 0.081178 | 4.460783x | 2 | 0.916209 | 0.046722 | no | 13/16 | distributed-lift |
| 6 | 0.528679 | 0.521143 | 0.529641 | 0.178865 | 0.177581 | 0.178865 | 0.157787 | 2.955747x | 3 | 0.850716 | 0.021078 | no | 11/16 | distributed-lift |
| 7 | 0.779231 | 0.643735 | 1.059085 | 0.019656 | 0.020324 | 0.019656 | 0.019656 | 39.644139x | 0 | 0.920645 | 0 | no | 15/16 | distributed-lift |
| 8 | 0.386966 | 0.379033 | 0.413881 | 0.012458 | 0.013077 | 0.012458 | 0.012458 | 31.061353x | 2 | 0.762689 | 0 | no | 11/16 | distributed-lift |
| 9 | 0.60038 | 7.125e+28 | 0.599308 | 0.157757 | 0.166682 | 0.157757 | 0.157757 | 3.805723x | 1 | 0.808844 | 0 | no | 11/16 | distributed-lift |
| 10 | 0.188119 | 0.174087 | 0.195598 | 0.14881 | 0.157227 | 0.14881 | 0.14881 | 1.264155x | 2 | 0.847705 | 0 | validation-veto | 12/16 | distributed-lift |
| 11 | 0.270593 | 0.138341 | 0.332051 | 0.001168 | 0.001235 | 0.001168 | 0.001168 | 231.618143x | 3 | 0.925945 | 0 | no | 10/16 | distributed-lift |
| 12 | 0.36695 | 0.312266 | 0.391126 | 6.393e-12 | 6.416e-12 | 6.393e-12 | 6.393e-12 | 5.740e+10x | 2 | 0.306354 | 0 | low-nn-confidence | 7/12 | distributed-lift |
| 13 | 0.493314 | 0.412713 | 0.519703 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 1.655e+9x | 3 | 0.934757 | 0 | no | 3/12 | distributed-lift |
| 14 | 0.157756 | 0.142476 | 0.166675 | 8.955e-8 | 8.973e-8 | 8.955e-8 | 8.955e-8 | 1.762e+6x | 0 | 0.918228 | 0 | no | 9/14 | distributed-lift |
| 15 | 0.306881 | 0.285691 | 0.316855 | 0.125942 | 0.121401 | 0.125942 | 0.125942 | 2.43668x | 0 | 0.7795 | 0 | no | 11/16 | distributed-lift |
| 16 | 1.039911 | 0.115525 | 0.162025 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 3.489e+9x | 0 | 0.89111 | 0 | no | 7/12 | distributed-lift |
| 17 | 0.060859 | 0.045547 | 0.059957 | 0.043064 | 0.043256 | 0.043064 | 0.002668 | 1.413215x | 0 | 0.588395 | 0.040396 | no | 10/16 | distributed-lift |
| 18 | 0.186365 | 0.1244 | 0.13638 | 0.034865 | 0.03463 | 0.034865 | 0.034865 | 5.345343x | 1 | 0.911032 | 0 | no | 11/16 | distributed-lift |
| 19 | 0.11987 | 0.109808 | 1.223128 | 0.061379 | 0.118849 | 0.061379 | 0.061379 | 1.952928x | 3 | 0.892538 | 0 | no | 11/16 | distributed-lift |
| 20 | 0.090597 | 0.065265 | 0.099378 | 2.911e-11 | 2.917e-11 | 2.911e-11 | 2.911e-11 | 3.112e+9x | 2 | 0.91374 | 0 | no | 9/14 | distributed-lift |
| 21 | 0.35326 | 0.343066 | 0.362177 | 2.588e-11 | 2.600e-11 | 2.588e-11 | 2.588e-11 | 1.365e+10x | 2 | 0.800766 | 0 | no | 4/12 | distributed-lift |
| 22 | 0.148652 | 0.138504 | 0.147318 | 0.056784 | 0.111374 | 0.056784 | 0.056784 | 2.617854x | 0 | 0.912194 | 0 | no | 10/16 | distributed-lift |
| 23 | 0.13161 | 0.124415 | 0.13997 | 0.013468 | 0.013486 | 0.013468 | 0.013468 | 9.772283x | 1 | 0.788626 | 0 | no | 14/16 | distributed-lift |
| 24 | 47.199434 | 46.101695 | 45.954595 | 1.112e-10 | 1.173e-10 | 1.112e-10 | 1.112e-10 | 4.245e+11x | 3 | 0.799056 | 0 | no | 8/12 | distributed-lift |

