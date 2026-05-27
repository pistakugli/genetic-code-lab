# Distributed worker-island + NN policy benchmark

Generated: 2026-05-27T16:10:20.282Z

## Scope

- Runs real `DistributedManager` worker-thread islands.
- Uses a shared `MigrationHub` with atomic migration counters.
- Uses the existing `MLP` as a scorer, then applies confidence, validation, and guard-holdout veto/fallback policy.
- This is the production-runtime proof lane; the wide seed sweep is part of `release:gate`.

## Summary

- Mean single-island MAE: 2.324315
- Mean policy MAE: 0.040237
- Mean robust holdout-only MAE: 0.040237
- Mean oracle-best MAE: 0.038283
- Median lift: 24.380545x
- Median lift on non-degenerate seeds: 24.380545x
- Success rate: 100%
- Non-degenerate success rate: 100%
- Non-regression rate vs single baseline: 100%
- Completed seeds: 24/24
- Runtime timeouts: 0
- Runtime errors: 0
- Single-baseline solved seeds: 0
- Single-baseline fallbacks: 0
- Mean regret vs oracle: 0.001954
- Policy beats robust holdout-only rate: 0%
- NN holdout accuracy: 0.846616
- NN holdout Brier: 0.126236
- NN holdout AUC: 0.826478
- Hub drops: 356
- Hub pull successes: 229
- Fallbacks: 2

| Seed | Single MAE | Single val MAE | Single guard MAE | Policy MAE | Policy guard MAE | Holdout-only MAE | Oracle MAE | Lift | Island | NN p | Regret | Fallback | Hub pulls | Outcome |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.157757 | 0.142474 | 0.166682 | 0.051584 | 0.053607 | 0.051584 | 0.051584 | 3.058239x | 0 | 0.924701 | 0 | no | 11/16 | distributed-lift |
| 2 | 0.768867 | 0.673535 | 0.839236 | 0.115434 | 0.120151 | 0.115434 | 0.115434 | 6.660659x | 2 | 0.862443 | 0 | no | 13/16 | distributed-lift |
| 3 | 0.247474 | 0.212591 | 0.246134 | 0.157759 | 0.166683 | 0.157759 | 0.157759 | 1.568684x | 1 | 0.73062 | 0 | no | 11/16 | distributed-lift |
| 4 | 0.62951 | 0.618636 | 0.631109 | 1.306e-11 | 1.310e-11 | 1.306e-11 | 1.306e-11 | 4.821e+10x | 2 | 0.884314 | 0 | no | 4/12 | distributed-lift |
| 5 | 0.570535 | 0.526547 | 0.562227 | 0.088991 | 0.089732 | 0.088991 | 0.088991 | 6.41119x | 2 | 0.871469 | 0 | no | 14/16 | distributed-lift |
| 6 | 0.528679 | 0.521143 | 0.529641 | 0.017612 | 0.105139 | 0.017612 | 0.017612 | 30.017925x | 1 | 0.915131 | 0 | no | 11/16 | distributed-lift |
| 7 | 0.779231 | 0.643735 | 1.059085 | 0.019175 | 0.018778 | 0.019175 | 0.019175 | 40.637271x | 0 | 0.886797 | 0 | no | 14/16 | distributed-lift |
| 8 | 0.386966 | 0.379033 | 0.413881 | 0.020646 | 0.020423 | 0.020646 | 0.020646 | 18.743166x | 2 | 0.797349 | 0 | no | 14/16 | distributed-lift |
| 9 | 0.60038 | 7.125e+28 | 0.599308 | 0.157757 | 0.166682 | 0.157757 | 0.157757 | 3.805723x | 1 | 0.825165 | 0 | no | 11/16 | distributed-lift |
| 10 | 0.188119 | 0.174087 | 0.195598 | 0.061532 | 0.061062 | 0.061532 | 0.061532 | 3.057224x | 3 | 0.93064 | 0 | no | 12/16 | distributed-lift |
| 11 | 0.270593 | 0.138341 | 0.332051 | 5.889e-8 | 5.909e-8 | 5.889e-8 | 5.889e-8 | 4.595e+6x | 3 | 0.899807 | 0 | no | 9/14 | distributed-lift |
| 12 | 0.36695 | 0.312266 | 0.391126 | 6.393e-12 | 6.416e-12 | 6.393e-12 | 6.393e-12 | 5.740e+10x | 2 | 0.130984 | 0 | low-nn-confidence | 3/12 | distributed-lift |
| 13 | 0.493314 | 0.412713 | 0.519703 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 1.655e+9x | 3 | 0.949148 | 0 | no | 3/12 | distributed-lift |
| 14 | 0.157756 | 0.142476 | 0.166675 | 3.531e-7 | 3.552e-7 | 3.531e-7 | 3.531e-7 | 4.467e+5x | 0 | 0.90124 | 0 | no | 9/14 | distributed-lift |
| 15 | 0.306881 | 0.285691 | 0.316855 | 0.123672 | 0.116344 | 0.123672 | 0.123672 | 2.481406x | 0 | 0.807933 | 0 | no | 12/16 | distributed-lift |
| 16 | 1.039911 | 0.115525 | 0.162025 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 3.489e+9x | 0 | 0.88174 | 0 | no | 6/12 | distributed-lift |
| 17 | 0.060859 | 0.045547 | 0.059957 | 0.041851 | 0.041943 | 0.041851 | 0.002668 | 1.454197x | 0 | 0.568378 | 0.039182 | low-nn-confidence | 10/16 | distributed-lift |
| 18 | 0.186365 | 0.1244 | 0.13638 | 0.034865 | 0.03463 | 0.034865 | 0.034865 | 5.345343x | 1 | 0.893123 | 0 | no | 11/16 | distributed-lift |
| 19 | 0.11987 | 0.109808 | 1.223128 | 0.064304 | 0.117967 | 0.064304 | 0.064304 | 1.864108x | 3 | 0.916605 | 0 | no | 11/16 | distributed-lift |
| 20 | 0.090597 | 0.065265 | 0.099378 | 0.010371 | 0.01201 | 0.010371 | 0.002668 | 8.735853x | 2 | 0.705912 | 0.007702 | no | 10/16 | distributed-lift |
| 21 | 0.35326 | 0.343066 | 0.362177 | 2.588e-11 | 2.600e-11 | 2.588e-11 | 2.588e-11 | 1.365e+10x | 2 | 0.87067 | 0 | no | 4/12 | distributed-lift |
| 22 | 0.148652 | 0.138504 | 0.147318 | 0.000129 | 0.055766 | 0.000129 | 0.000129 | 1153.892729x | 0 | 0.930501 | 0 | no | 12/16 | distributed-lift |
| 23 | 0.13161 | 0.124415 | 0.13997 | 1.345e-9 | 1.350e-9 | 1.345e-9 | 1.345e-9 | 9.783e+7x | 0 | 0.583098 | 0 | no | 11/16 | distributed-lift |
| 24 | 47.199434 | 46.101695 | 45.954595 | 1.112e-10 | 1.173e-10 | 1.112e-10 | 1.112e-10 | 4.245e+11x | 3 | 0.657833 | 0 | no | 3/12 | distributed-lift |

