# Distributed worker-island + NN policy benchmark

Generated: 2026-05-28T06:13:46.916Z

## Scope

- Runs real `DistributedManager` worker-thread islands.
- Uses a shared `MigrationHub` with atomic migration counters.
- Uses the existing `MLP` as a scorer, then applies confidence, validation, and guard-holdout veto/fallback policy.
- This is the production-runtime proof lane; the wide seed sweep is part of `release:gate`.

## Summary

- Mean single-island MAE: 2.324315
- Mean policy MAE: 0.06273
- Mean robust holdout-only MAE: 0.06273
- Mean oracle-best MAE: 0.060918
- Median lift: 8.28953x
- Median lift on non-degenerate seeds: 8.28953x
- Success rate: 100%
- Non-degenerate success rate: 100%
- Non-regression rate vs single baseline: 100%
- Completed seeds: 24/24
- Runtime timeouts: 0
- Runtime errors: 0
- Single-baseline solved seeds: 0
- Single-baseline fallbacks: 0
- Mean regret vs oracle: 0.001811
- Policy beats robust holdout-only rate: 0%
- NN holdout accuracy: 0.805478
- NN holdout Brier: 0.149921
- NN holdout AUC: 0.839155
- Hub drops: 356
- Hub pull successes: 220
- Fallbacks: 3

| Seed | Single MAE | Single val MAE | Single guard MAE | Policy MAE | Policy guard MAE | Holdout-only MAE | Oracle MAE | Lift | Island | NN p | Regret | Fallback | Hub pulls | Outcome |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.157757 | 0.142474 | 0.166682 | 0.108711 | 0.109314 | 0.108711 | 0.108711 | 1.451162x | 0 | 0.964598 | 0 | no | 12/16 | distributed-lift |
| 2 | 0.768867 | 0.673535 | 0.839236 | 0.114274 | 0.119383 | 0.114274 | 0.114274 | 6.728298x | 2 | 0.609471 | 0 | no | 10/16 | distributed-lift |
| 3 | 0.247474 | 0.212591 | 0.246134 | 0.157759 | 0.166683 | 0.157759 | 0.157759 | 1.568684x | 1 | 0.761888 | 0 | no | 10/16 | distributed-lift |
| 4 | 0.62951 | 0.618636 | 0.631109 | 1.306e-11 | 1.310e-11 | 1.306e-11 | 1.306e-11 | 4.821e+10x | 2 | 0.852976 | 0 | no | 4/12 | distributed-lift |
| 5 | 0.570535 | 0.526547 | 0.562227 | 0.13513 | 0.139553 | 0.13513 | 0.13513 | 4.222128x | 2 | 0.851645 | 0 | no | 11/16 | distributed-lift |
| 6 | 0.528679 | 0.521143 | 0.529641 | 0.212964 | 0.195166 | 0.212964 | 0.212964 | 2.48248x | 3 | 0.931103 | 0 | no | 10/16 | distributed-lift |
| 7 | 0.779231 | 0.643735 | 1.059085 | 0.074666 | 0.080937 | 0.074666 | 0.074666 | 10.436154x | 0 | 0.922135 | 0 | no | 11/16 | distributed-lift |
| 8 | 0.386966 | 0.379033 | 0.413881 | 0.02381 | 0.024545 | 0.02381 | 0.02381 | 16.252104x | 2 | 0.750334 | 0 | no | 10/16 | distributed-lift |
| 9 | 0.60038 | 7.125e+28 | 0.599308 | 0.157757 | 0.166682 | 0.157757 | 0.157757 | 3.805723x | 1 | 0.738424 | 0 | no | 11/16 | distributed-lift |
| 10 | 0.188119 | 0.174087 | 0.195598 | 0.149646 | 0.158112 | 0.149646 | 0.149646 | 1.25709x | 2 | 0.69475 | 0 | validation-veto | 12/16 | distributed-lift |
| 11 | 0.270593 | 0.138341 | 0.332051 | 3.471e-14 | 3.521e-14 | 3.471e-14 | 3.471e-14 | 2.706e+11x | 3 | 0.842681 | 0 | no | 9/14 | distributed-lift |
| 12 | 0.36695 | 0.312266 | 0.391126 | 6.393e-12 | 6.416e-12 | 6.393e-12 | 6.393e-12 | 5.740e+10x | 2 | 0.212355 | 0 | low-nn-confidence | 3/12 | distributed-lift |
| 13 | 0.493314 | 0.412713 | 0.519703 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 1.655e+9x | 3 | 0.96586 | 0 | no | 3/12 | distributed-lift |
| 14 | 0.157756 | 0.142476 | 0.166675 | 8.955e-8 | 8.973e-8 | 8.955e-8 | 8.955e-8 | 1.762e+6x | 0 | 0.836752 | 0 | no | 12/14 | distributed-lift |
| 15 | 0.306881 | 0.285691 | 0.316855 | 0.14838 | 0.141412 | 0.14838 | 0.14838 | 2.068209x | 0 | 0.840859 | 0 | no | 11/16 | distributed-lift |
| 16 | 1.039911 | 0.115525 | 0.162025 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 3.489e+9x | 0 | 0.144545 | 0 | low-nn-confidence | 7/12 | distributed-lift |
| 17 | 0.060859 | 0.045547 | 0.059957 | 0.046135 | 0.045234 | 0.046135 | 0.002664 | 1.319157x | 0 | 0.584322 | 0.04347 | no | 10/16 | distributed-lift |
| 18 | 0.186365 | 0.1244 | 0.13638 | 0.034865 | 0.03463 | 0.034865 | 0.034865 | 5.345343x | 1 | 0.869714 | 0 | no | 11/16 | distributed-lift |
| 19 | 0.11987 | 0.109808 | 1.223128 | 0.060997 | 0.118181 | 0.060997 | 0.060997 | 1.965163x | 3 | 0.905245 | 0 | no | 11/16 | distributed-lift |
| 20 | 0.090597 | 0.065265 | 0.099378 | 0.002668 | 0.05799 | 0.002668 | 0.002668 | 33.952061x | 0 | 0.906654 | 0 | no | 11/16 | distributed-lift |
| 21 | 0.35326 | 0.343066 | 0.362177 | 2.588e-11 | 2.600e-11 | 2.588e-11 | 2.588e-11 | 1.365e+10x | 2 | 0.884361 | 0 | no | 4/12 | distributed-lift |
| 22 | 0.148652 | 0.138504 | 0.147318 | 0.064388 | 0.118073 | 0.064388 | 0.064388 | 2.308684x | 0 | 0.883632 | 0 | no | 10/16 | distributed-lift |
| 23 | 0.13161 | 0.124415 | 0.13997 | 0.01336 | 0.013331 | 0.01336 | 0.01336 | 9.850761x | 1 | 0.804704 | 0 | no | 14/16 | distributed-lift |
| 24 | 47.199434 | 46.101695 | 45.954595 | 1.112e-10 | 1.173e-10 | 1.112e-10 | 1.112e-10 | 4.245e+11x | 3 | 0.729668 | 0 | no | 3/12 | distributed-lift |

