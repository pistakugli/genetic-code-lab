# Distributed worker-island + NN policy benchmark

Generated: 2026-06-03T08:57:47.207Z

## Scope

- Runs real `DistributedManager` worker-thread islands.
- Uses a shared `MigrationHub` with atomic migration counters.
- Uses the existing `MLP` as a scorer, then applies confidence, validation, and guard-holdout veto/fallback policy.
- This is the production-runtime proof lane; the wide seed sweep is part of `release:gate`.

## Summary

- Mean single-island MAE: 2.324315
- Mean policy MAE: 0.047187
- Mean robust holdout-only MAE: 0.047187
- Mean oracle-best MAE: 0.045448
- Median lift: 26.061338x
- Median lift on non-degenerate seeds: 26.061338x
- Success rate: 100%
- Non-degenerate success rate: 100%
- Non-regression rate vs single baseline: 100%
- Completed seeds: 24/24
- Runtime timeouts: 0
- Runtime errors: 0
- Single-baseline solved seeds: 0
- Single-baseline fallbacks: 0
- Mean regret vs oracle: 0.001739
- Policy beats robust holdout-only rate: 0%
- NN holdout accuracy: 0.841858
- NN holdout Brier: 0.12949
- NN holdout AUC: 0.902509
- Hub drops: 355
- Hub pull successes: 239
- Fallbacks: 4

| Seed | Single MAE | Single val MAE | Single guard MAE | Policy MAE | Policy guard MAE | Holdout-only MAE | Oracle MAE | Lift | Island | NN p | Regret | Fallback | Hub pulls | Outcome |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.157757 | 0.142474 | 0.166682 | 0.090237 | 0.091245 | 0.090237 | 0.090237 | 1.748245x | 0 | 0.922218 | 0 | no | 11/16 | distributed-lift |
| 2 | 0.768867 | 0.673535 | 0.839236 | 0.11504 | 0.120957 | 0.11504 | 0.11504 | 6.683457x | 2 | 0.632289 | 0 | no | 13/16 | distributed-lift |
| 3 | 0.247474 | 0.212591 | 0.246134 | 0.157757 | 0.166682 | 0.157757 | 0.157757 | 1.568701x | 1 | 0.727195 | 0 | no | 10/16 | distributed-lift |
| 4 | 0.62951 | 0.618636 | 0.631109 | 1.306e-11 | 1.310e-11 | 1.306e-11 | 1.306e-11 | 4.821e+10x | 2 | 0.839976 | 0 | no | 3/12 | distributed-lift |
| 5 | 0.570535 | 0.526547 | 0.562227 | 0.158354 | 0.16435 | 0.158354 | 0.15776 | 3.602913x | 2 | 0.813691 | 0.000594 | no | 14/16 | distributed-lift |
| 6 | 0.528679 | 0.521143 | 0.529641 | 0.145443 | 0.168272 | 0.145443 | 0.145443 | 3.634964x | 3 | 0.934222 | 0 | no | 11/16 | distributed-lift |
| 7 | 0.779231 | 0.643735 | 1.059085 | 0.018626 | 0.016756 | 0.018626 | 0.018626 | 41.835437x | 0 | 0.925147 | 0 | no | 11/16 | distributed-lift |
| 8 | 0.386966 | 0.379033 | 0.413881 | 0.021296 | 0.022224 | 0.021296 | 0.021296 | 18.170615x | 2 | 0.717241 | 0 | no | 15/16 | distributed-lift |
| 9 | 0.60038 | 7.125e+28 | 0.599308 | 0.157757 | 0.166682 | 0.157757 | 0.157757 | 3.805723x | 1 | 0.846509 | 0 | no | 11/16 | distributed-lift |
| 10 | 0.188119 | 0.174087 | 0.195598 | 5.898e-11 | 5.865e-11 | 5.898e-11 | 5.898e-11 | 3.189e+9x | 0 | 0.856331 | 0 | no | 10/14 | distributed-lift |
| 11 | 0.270593 | 0.138341 | 0.332051 | 6.576e-10 | 6.643e-10 | 6.576e-10 | 6.576e-10 | 4.115e+8x | 3 | 0.912998 | 0 | no | 10/15 | distributed-lift |
| 12 | 0.36695 | 0.312266 | 0.391126 | 6.393e-12 | 6.416e-12 | 6.393e-12 | 6.393e-12 | 5.740e+10x | 2 | 0.194114 | 0 | low-nn-confidence | 7/12 | distributed-lift |
| 13 | 0.493314 | 0.412713 | 0.519703 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 1.655e+9x | 3 | 0.949252 | 0 | no | 3/12 | distributed-lift |
| 14 | 0.157756 | 0.142476 | 0.166675 | 7.495e-7 | 7.594e-7 | 7.495e-7 | 7.495e-7 | 2.105e+5x | 0 | 0.844976 | 0 | no | 12/14 | distributed-lift |
| 15 | 0.306881 | 0.285691 | 0.316855 | 0.12621 | 0.12039 | 0.12621 | 0.12621 | 2.431511x | 0 | 0.835038 | 0 | no | 11/16 | distributed-lift |
| 16 | 1.039911 | 0.115525 | 0.162025 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 3.489e+9x | 0 | 0.207814 | 0 | low-nn-confidence | 7/12 | distributed-lift |
| 17 | 0.060859 | 0.045547 | 0.059957 | 0.043809 | 0.043275 | 0.043809 | 0.002663 | 1.389196x | 0 | 0.570645 | 0.041146 | low-nn-confidence | 15/16 | distributed-lift |
| 18 | 0.186365 | 0.1244 | 0.13638 | 0.034865 | 0.03463 | 0.034865 | 0.034865 | 5.345343x | 1 | 0.841494 | 0 | no | 10/16 | distributed-lift |
| 19 | 0.11987 | 0.109808 | 1.223128 | 0.000833 | 0.056374 | 0.000833 | 0.000833 | 143.955151x | 3 | 0.895089 | 0 | no | 10/16 | distributed-lift |
| 20 | 0.090597 | 0.065265 | 0.099378 | 0.002668 | 0.05799 | 0.002668 | 0.002668 | 33.952061x | 0 | 0.92203 | 0 | no | 11/16 | distributed-lift |
| 21 | 0.35326 | 0.343066 | 0.362177 | 2.588e-11 | 2.600e-11 | 2.588e-11 | 2.588e-11 | 1.365e+10x | 2 | 0.828438 | 0 | no | 4/12 | distributed-lift |
| 22 | 0.148652 | 0.138504 | 0.147318 | 0.046565 | 0.104517 | 0.046565 | 0.046565 | 3.192333x | 0 | 0.916705 | 0 | no | 11/16 | distributed-lift |
| 23 | 0.13161 | 0.124415 | 0.13997 | 0.01303 | 0.012853 | 0.01303 | 0.01303 | 10.100256x | 1 | 0.747026 | 0 | no | 11/16 | distributed-lift |
| 24 | 47.199434 | 46.101695 | 45.954595 | 1.112e-10 | 1.173e-10 | 1.112e-10 | 1.112e-10 | 4.245e+11x | 3 | 0.523651 | 0 | low-nn-confidence | 8/12 | distributed-lift |

