# Distributed worker-island + NN policy benchmark

Generated: 2026-05-27T17:54:30.306Z

## Scope

- Runs real `DistributedManager` worker-thread islands.
- Uses a shared `MigrationHub` with atomic migration counters.
- Uses the existing `MLP` as a scorer, then applies confidence, validation, and guard-holdout veto/fallback policy.
- This is the production-runtime proof lane; the wide seed sweep is part of `release:gate`.

## Summary

- Mean single-island MAE: 2.324315
- Mean policy MAE: 0.05956
- Mean robust holdout-only MAE: 0.05956
- Mean oracle-best MAE: 0.05778
- Median lift: 8.265716x
- Median lift on non-degenerate seeds: 8.265716x
- Success rate: 100%
- Non-degenerate success rate: 100%
- Non-regression rate vs single baseline: 100%
- Completed seeds: 24/24
- Runtime timeouts: 0
- Runtime errors: 0
- Single-baseline solved seeds: 0
- Single-baseline fallbacks: 0
- Mean regret vs oracle: 0.001781
- Policy beats robust holdout-only rate: 0%
- NN holdout accuracy: 0.839621
- NN holdout Brier: 0.144563
- NN holdout AUC: 0.895221
- Hub drops: 356
- Hub pull successes: 228
- Fallbacks: 2

| Seed | Single MAE | Single val MAE | Single guard MAE | Policy MAE | Policy guard MAE | Holdout-only MAE | Oracle MAE | Lift | Island | NN p | Regret | Fallback | Hub pulls | Outcome |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.157757 | 0.142474 | 0.166682 | 0.117213 | 0.11815 | 0.117213 | 0.117213 | 1.3459x | 0 | 0.962394 | 0 | no | 11/16 | distributed-lift |
| 2 | 0.768867 | 0.673535 | 0.839236 | 0.113751 | 0.119227 | 0.113751 | 0.113751 | 6.759242x | 2 | 0.693025 | 0 | no | 11/16 | distributed-lift |
| 3 | 0.247474 | 0.212591 | 0.246134 | 0.157759 | 0.166683 | 0.157759 | 0.157759 | 1.568684x | 1 | 0.782094 | 0 | no | 11/16 | distributed-lift |
| 4 | 0.62951 | 0.618636 | 0.631109 | 1.306e-11 | 1.310e-11 | 1.306e-11 | 1.306e-11 | 4.821e+10x | 2 | 0.832477 | 0 | no | 7/12 | distributed-lift |
| 5 | 0.570535 | 0.526547 | 0.562227 | 0.125998 | 0.132308 | 0.125998 | 0.125998 | 4.528131x | 2 | 0.835359 | 0 | no | 14/16 | distributed-lift |
| 6 | 0.528679 | 0.521143 | 0.529641 | 0.184665 | 0.188135 | 0.184665 | 0.184665 | 2.862908x | 3 | 0.927549 | 0 | no | 10/16 | distributed-lift |
| 7 | 0.779231 | 0.643735 | 1.059085 | 0.000183 | 0.000184 | 0.000183 | 0.000183 | 4269.160708x | 0 | 0.922152 | 0 | no | 10/16 | distributed-lift |
| 8 | 0.386966 | 0.379033 | 0.413881 | 0.023801 | 0.024533 | 0.023801 | 0.023801 | 16.258252x | 2 | 0.594543 | 0 | no | 11/16 | distributed-lift |
| 9 | 0.60038 | 7.125e+28 | 0.599308 | 0.157757 | 0.166682 | 0.157757 | 0.157757 | 3.805723x | 1 | 0.913461 | 0 | no | 11/16 | distributed-lift |
| 10 | 0.188119 | 0.174087 | 0.195598 | 0.085346 | 0.088417 | 0.085346 | 0.085346 | 2.20418x | 2 | 0.469464 | 0 | low-nn-confidence | 11/16 | distributed-lift |
| 11 | 0.270593 | 0.138341 | 0.332051 | 8.955e-8 | 8.973e-8 | 8.955e-8 | 8.955e-8 | 3.022e+6x | 3 | 0.922854 | 0 | no | 8/14 | distributed-lift |
| 12 | 0.36695 | 0.312266 | 0.391126 | 6.393e-12 | 6.416e-12 | 6.393e-12 | 6.393e-12 | 5.740e+10x | 2 | 0.236595 | 0 | low-nn-confidence | 3/12 | distributed-lift |
| 13 | 0.493314 | 0.412713 | 0.519703 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 1.655e+9x | 3 | 0.91348 | 0 | no | 3/12 | distributed-lift |
| 14 | 0.157756 | 0.142476 | 0.166675 | 8.955e-8 | 8.973e-8 | 8.955e-8 | 8.955e-8 | 1.762e+6x | 0 | 0.737051 | 0 | no | 12/14 | distributed-lift |
| 15 | 0.306881 | 0.285691 | 0.316855 | 0.262482 | 0.272972 | 0.262482 | 0.262482 | 1.169151x | 0 | 0.727575 | 0 | no | 11/16 | distributed-lift |
| 16 | 1.039911 | 0.115525 | 0.162025 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 3.489e+9x | 0 | 0.738471 | 0 | no | 7/12 | distributed-lift |
| 17 | 0.060859 | 0.045547 | 0.059957 | 0.045405 | 0.044955 | 0.045405 | 0.002668 | 1.340355x | 0 | 0.611722 | 0.042737 | no | 10/16 | distributed-lift |
| 18 | 0.186365 | 0.1244 | 0.13638 | 0.034865 | 0.03463 | 0.034865 | 0.034865 | 5.345343x | 1 | 0.877478 | 0 | no | 11/16 | distributed-lift |
| 19 | 0.11987 | 0.109808 | 1.223128 | 0.041064 | 0.098702 | 0.041064 | 0.041064 | 2.919112x | 3 | 0.947391 | 0 | no | 11/16 | distributed-lift |
| 20 | 0.090597 | 0.065265 | 0.099378 | 0.001387 | 0.001519 | 0.001387 | 0.001387 | 65.316118x | 2 | 0.797004 | 0 | no | 11/16 | distributed-lift |
| 21 | 0.35326 | 0.343066 | 0.362177 | 2.588e-11 | 2.600e-11 | 2.588e-11 | 2.588e-11 | 1.365e+10x | 2 | 0.868214 | 0 | no | 3/12 | distributed-lift |
| 22 | 0.148652 | 0.138504 | 0.147318 | 0.064304 | 0.117967 | 0.064304 | 0.064304 | 2.311704x | 0 | 0.849101 | 0 | no | 14/16 | distributed-lift |
| 23 | 0.13161 | 0.124415 | 0.13997 | 0.013468 | 0.013487 | 0.013468 | 0.013468 | 9.77219x | 1 | 0.841468 | 0 | no | 10/16 | distributed-lift |
| 24 | 47.199434 | 46.101695 | 45.954595 | 1.112e-10 | 1.173e-10 | 1.112e-10 | 1.112e-10 | 4.245e+11x | 3 | 0.690858 | 0 | no | 7/12 | distributed-lift |

