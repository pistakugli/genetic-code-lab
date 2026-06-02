# Distributed worker-island + NN policy benchmark

Generated: 2026-06-02T08:54:14.225Z

## Scope

- Runs real `DistributedManager` worker-thread islands.
- Uses a shared `MigrationHub` with atomic migration counters.
- Uses the existing `MLP` as a scorer, then applies confidence, validation, and guard-holdout veto/fallback policy.
- This is the production-runtime proof lane; the wide seed sweep is part of `release:gate`.

## Summary

- Mean single-island MAE: 2.324315
- Mean policy MAE: 0.050693
- Mean robust holdout-only MAE: 0.050693
- Mean oracle-best MAE: 0.048956
- Median lift: 8.235022x
- Median lift on non-degenerate seeds: 8.235022x
- Success rate: 100%
- Non-degenerate success rate: 100%
- Non-regression rate vs single baseline: 100%
- Completed seeds: 24/24
- Runtime timeouts: 0
- Runtime errors: 0
- Single-baseline solved seeds: 0
- Single-baseline fallbacks: 0
- Mean regret vs oracle: 0.001736
- Policy beats robust holdout-only rate: 0%
- NN holdout accuracy: 0.819052
- NN holdout Brier: 0.140692
- NN holdout AUC: 0.845422
- Hub drops: 356
- Hub pull successes: 236
- Fallbacks: 5

| Seed | Single MAE | Single val MAE | Single guard MAE | Policy MAE | Policy guard MAE | Holdout-only MAE | Oracle MAE | Lift | Island | NN p | Regret | Fallback | Hub pulls | Outcome |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.157757 | 0.142474 | 0.166682 | 0.041136 | 0.042801 | 0.041136 | 0.041136 | 3.835032x | 0 | 0.899496 | 0 | no | 11/16 | distributed-lift |
| 2 | 0.768867 | 0.673535 | 0.839236 | 0.114796 | 0.119979 | 0.114796 | 0.114796 | 6.697673x | 2 | 0.643476 | 0 | no | 13/16 | distributed-lift |
| 3 | 0.247474 | 0.212591 | 0.246134 | 0.157759 | 0.166683 | 0.157759 | 0.157759 | 1.568684x | 1 | 0.758413 | 0 | no | 11/16 | distributed-lift |
| 4 | 0.62951 | 0.618636 | 0.631109 | 1.306e-11 | 1.310e-11 | 1.306e-11 | 1.306e-11 | 4.821e+10x | 2 | 0.881433 | 0 | no | 4/12 | distributed-lift |
| 5 | 0.570535 | 0.526547 | 0.562227 | 0.133075 | 0.135969 | 0.133075 | 0.133075 | 4.287321x | 2 | 0.841417 | 0 | no | 14/16 | distributed-lift |
| 6 | 0.528679 | 0.521143 | 0.529641 | 0.088705 | 0.148214 | 0.088705 | 0.088705 | 5.959968x | 1 | 0.909491 | 0 | no | 10/16 | distributed-lift |
| 7 | 0.779231 | 0.643735 | 1.059085 | 0.052188 | 0.050696 | 0.052188 | 0.052188 | 14.931231x | 0 | 0.926909 | 0 | no | 15/16 | distributed-lift |
| 8 | 0.386966 | 0.379033 | 0.413881 | 0.023779 | 0.0245 | 0.023779 | 0.023779 | 16.273478x | 2 | 0.509172 | 0 | low-nn-confidence | 11/16 | distributed-lift |
| 9 | 0.60038 | 7.125e+28 | 0.599308 | 0.094635 | 0.096098 | 0.094635 | 0.094635 | 6.344187x | 0 | 0.877831 | 0 | no | 11/16 | distributed-lift |
| 10 | 0.188119 | 0.174087 | 0.195598 | 0.14881 | 0.157227 | 0.14881 | 0.14881 | 1.264155x | 2 | 0.759097 | 0 | validation-veto | 12/16 | distributed-lift |
| 11 | 0.270593 | 0.138341 | 0.332051 | 7.111e-7 | 7.160e-7 | 7.111e-7 | 7.111e-7 | 3.805e+5x | 3 | 0.887233 | 0 | no | 10/15 | distributed-lift |
| 12 | 0.36695 | 0.312266 | 0.391126 | 6.393e-12 | 6.416e-12 | 6.393e-12 | 6.393e-12 | 5.740e+10x | 2 | 0.216394 | 0 | low-nn-confidence | 7/12 | distributed-lift |
| 13 | 0.493314 | 0.412713 | 0.519703 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 1.655e+9x | 3 | 0.921177 | 0 | no | 3/12 | distributed-lift |
| 14 | 0.157756 | 0.142476 | 0.166675 | 1.936e-7 | 1.942e-7 | 1.936e-7 | 1.936e-7 | 8.147e+5x | 0 | 0.857348 | 0 | no | 12/14 | distributed-lift |
| 15 | 0.306881 | 0.285691 | 0.316855 | 0.147992 | 0.140534 | 0.147992 | 0.147992 | 2.073637x | 0 | 0.926028 | 0 | no | 10/16 | distributed-lift |
| 16 | 1.039911 | 0.115525 | 0.162025 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 3.489e+9x | 0 | 0.796416 | 0 | no | 7/12 | distributed-lift |
| 17 | 0.060859 | 0.045547 | 0.059957 | 0.044233 | 0.042542 | 0.044233 | 0.002562 | 1.375877x | 0 | 0.565922 | 0.041671 | low-nn-confidence | 10/16 | distributed-lift |
| 18 | 0.186365 | 0.1244 | 0.13638 | 0.034865 | 0.03463 | 0.034865 | 0.034865 | 5.345343x | 1 | 0.703076 | 0 | no | 12/16 | distributed-lift |
| 19 | 0.11987 | 0.109808 | 1.223128 | 0.056861 | 0.111451 | 0.056861 | 0.056861 | 2.108103x | 3 | 0.907234 | 0 | no | 11/16 | distributed-lift |
| 20 | 0.090597 | 0.065265 | 0.099378 | 1.217e-10 | 1.238e-10 | 1.217e-10 | 1.217e-10 | 7.447e+8x | 3 | 0.420678 | 0 | low-nn-confidence | 10/15 | distributed-lift |
| 21 | 0.35326 | 0.343066 | 0.362177 | 2.588e-11 | 2.600e-11 | 2.588e-11 | 2.588e-11 | 1.365e+10x | 2 | 0.843958 | 0 | no | 4/12 | distributed-lift |
| 22 | 0.148652 | 0.138504 | 0.147318 | 0.064322 | 0.117982 | 0.064322 | 0.064322 | 2.311065x | 0 | 0.712213 | 0 | no | 11/16 | distributed-lift |
| 23 | 0.13161 | 0.124415 | 0.13997 | 0.013468 | 0.013486 | 0.013468 | 0.013468 | 9.772371x | 1 | 0.83884 | 0 | no | 14/16 | distributed-lift |
| 24 | 47.199434 | 46.101695 | 45.954595 | 1.112e-10 | 1.173e-10 | 1.112e-10 | 1.112e-10 | 4.245e+11x | 3 | 0.753178 | 0 | no | 3/12 | distributed-lift |

