# Distributed worker-island + NN policy benchmark

Generated: 2026-05-27T15:03:05.684Z

## Scope

- Runs real `DistributedManager` worker-thread islands.
- Uses a shared `MigrationHub` with atomic migration counters.
- Uses the existing `MLP` as a scorer, then applies confidence, validation, and guard-holdout veto/fallback policy.
- This is the production-runtime proof lane; the wide seed sweep is part of `release:gate`.

## Summary

- Mean single-island MAE: 2.324315
- Mean policy MAE: 0.056545
- Mean robust holdout-only MAE: 0.056237
- Mean oracle-best MAE: 0.054475
- Median lift: 8.170324x
- Median lift on non-degenerate seeds: 8.170324x
- Success rate: 100%
- Non-degenerate success rate: 100%
- Non-regression rate vs single baseline: 100%
- Completed seeds: 24/24
- Runtime timeouts: 0
- Runtime errors: 0
- Single-baseline solved seeds: 0
- Single-baseline fallbacks: 0
- Mean regret vs oracle: 0.00207
- Policy beats robust holdout-only rate: 0%
- NN holdout accuracy: 0.839644
- NN holdout Brier: 0.139572
- NN holdout AUC: 0.867778
- Hub drops: 356
- Hub pull successes: 228
- Fallbacks: 3

| Seed | Single MAE | Single val MAE | Single guard MAE | Policy MAE | Policy guard MAE | Holdout-only MAE | Oracle MAE | Lift | Island | NN p | Regret | Fallback | Hub pulls | Outcome |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.157757 | 0.142474 | 0.166682 | 0.053103 | 0.054734 | 0.053103 | 0.053103 | 2.970766x | 0 | 0.92455 | 0 | no | 11/16 | distributed-lift |
| 2 | 0.768867 | 0.673535 | 0.839236 | 0.117056 | 0.122425 | 0.117056 | 0.117056 | 6.568344x | 2 | 0.723269 | 0 | no | 11/16 | distributed-lift |
| 3 | 0.247474 | 0.212591 | 0.246134 | 0.157759 | 0.166683 | 0.157759 | 0.157759 | 1.568684x | 1 | 0.743804 | 0 | validation-veto | 11/16 | distributed-lift |
| 4 | 0.62951 | 0.618636 | 0.631109 | 1.306e-11 | 1.310e-11 | 1.306e-11 | 1.306e-11 | 4.821e+10x | 2 | 0.8823 | 0 | no | 4/12 | distributed-lift |
| 5 | 0.570535 | 0.526547 | 0.562227 | 0.139906 | 0.13958 | 0.139906 | 0.139906 | 4.077989x | 2 | 0.820102 | 0 | no | 15/16 | distributed-lift |
| 6 | 0.528679 | 0.521143 | 0.529641 | 0.158576 | 0.197376 | 0.158576 | 0.158576 | 3.333922x | 1 | 0.901247 | 0 | no | 11/16 | distributed-lift |
| 7 | 0.779231 | 0.643735 | 1.059085 | 0.048614 | 0.049118 | 0.048614 | 0.048614 | 16.028905x | 0 | 0.921542 | 0 | no | 11/16 | distributed-lift |
| 8 | 0.386966 | 0.379033 | 0.413881 | 0.087509 | 0.076801 | 0.087509 | 0.087509 | 4.422026x | 2 | 0.710491 | 0 | no | 11/16 | distributed-lift |
| 9 | 0.60038 | 7.125e+28 | 0.599308 | 0.157757 | 0.166682 | 0.157757 | 0.157757 | 3.805723x | 1 | 0.757654 | 0 | no | 11/16 | distributed-lift |
| 10 | 0.188119 | 0.174087 | 0.195598 | 0.157757 | 0.166682 | 0.150377 | 0.150377 | 1.192454x | 1 | 0.893664 | 0.00738 | no | 12/16 | distributed-lift |
| 11 | 0.270593 | 0.138341 | 0.332051 | 8.955e-8 | 8.973e-8 | 8.955e-8 | 8.955e-8 | 3.022e+6x | 3 | 0.904726 | 0 | no | 8/14 | distributed-lift |
| 12 | 0.36695 | 0.312266 | 0.391126 | 6.393e-12 | 6.416e-12 | 6.393e-12 | 6.393e-12 | 5.740e+10x | 2 | 0.227048 | 0 | low-nn-confidence | 3/12 | distributed-lift |
| 13 | 0.493314 | 0.412713 | 0.519703 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 1.655e+9x | 3 | 0.918887 | 0 | no | 4/12 | distributed-lift |
| 14 | 0.157756 | 0.142476 | 0.166675 | 8.955e-8 | 8.973e-8 | 8.955e-8 | 8.955e-8 | 1.762e+6x | 0 | 0.877676 | 0 | no | 8/14 | distributed-lift |
| 15 | 0.306881 | 0.285691 | 0.316855 | 0.126043 | 0.12003 | 0.126043 | 0.126043 | 2.434734x | 0 | 0.81714 | 0 | no | 11/16 | distributed-lift |
| 16 | 1.039911 | 0.115525 | 0.162025 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 3.489e+9x | 0 | 0.220899 | 0 | low-nn-confidence | 7/12 | distributed-lift |
| 17 | 0.060859 | 0.045547 | 0.059957 | 0.04496 | 0.047153 | 0.04496 | 0.002668 | 1.353628x | 0 | 0.584428 | 0.042292 | no | 14/16 | distributed-lift |
| 18 | 0.186365 | 0.1244 | 0.13638 | 0.034865 | 0.03463 | 0.034865 | 0.034865 | 5.345343x | 1 | 0.853137 | 0 | no | 11/16 | distributed-lift |
| 19 | 0.11987 | 0.109808 | 1.223128 | 0.056912 | 0.111508 | 0.056912 | 0.056912 | 2.106216x | 3 | 0.91483 | 0 | no | 10/16 | distributed-lift |
| 20 | 0.090597 | 0.065265 | 0.099378 | 0.002663 | 0.057984 | 0.002663 | 0.002663 | 34.018804x | 0 | 0.810461 | 0 | no | 11/16 | distributed-lift |
| 21 | 0.35326 | 0.343066 | 0.362177 | 2.588e-11 | 2.600e-11 | 2.588e-11 | 2.588e-11 | 1.365e+10x | 2 | 0.838231 | 0 | no | 4/12 | distributed-lift |
| 22 | 0.148652 | 0.138504 | 0.147318 | 0.000129 | 0.055766 | 0.000129 | 0.000129 | 1153.892729x | 0 | 0.589916 | 0 | no | 12/16 | distributed-lift |
| 23 | 0.13161 | 0.124415 | 0.13997 | 0.013468 | 0.013486 | 0.013468 | 0.013468 | 9.772305x | 1 | 0.932149 | 0 | no | 14/16 | distributed-lift |
| 24 | 47.199434 | 46.101695 | 45.954595 | 1.112e-10 | 1.173e-10 | 1.112e-10 | 1.112e-10 | 4.245e+11x | 3 | 0.620566 | 0 | no | 3/12 | distributed-lift |

