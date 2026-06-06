# Distributed worker-island + NN policy benchmark

Generated: 2026-06-06T07:25:13.863Z

## Scope

- Runs real `DistributedManager` worker-thread islands.
- Uses a shared `MigrationHub` with atomic migration counters.
- Uses the existing `MLP` as a scorer, then applies confidence, validation, and guard-holdout veto/fallback policy.
- This is the production-runtime proof lane; the wide seed sweep is part of `release:gate`.

## Summary

- Mean single-island MAE: 2.324315
- Mean policy MAE: 0.050331
- Mean robust holdout-only MAE: 0.050331
- Mean oracle-best MAE: 0.046627
- Median lift: 7.010939x
- Median lift on non-degenerate seeds: 7.010939x
- Success rate: 100%
- Non-degenerate success rate: 100%
- Non-regression rate vs single baseline: 100%
- Completed seeds: 24/24
- Runtime timeouts: 0
- Runtime errors: 0
- Single-baseline solved seeds: 0
- Single-baseline fallbacks: 0
- Mean regret vs oracle: 0.003704
- Policy beats robust holdout-only rate: 0%
- NN holdout accuracy: 0.779039
- NN holdout Brier: 0.152731
- NN holdout AUC: 0.800702
- Hub drops: 359
- Hub pull successes: 237
- Fallbacks: 5

| Seed | Single MAE | Single val MAE | Single guard MAE | Policy MAE | Policy guard MAE | Holdout-only MAE | Oracle MAE | Lift | Island | NN p | Regret | Fallback | Hub pulls | Outcome |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.157757 | 0.142474 | 0.166682 | 0.051299 | 0.053049 | 0.051299 | 0.051299 | 3.075279x | 0 | 0.919921 | 0 | no | 12/16 | distributed-lift |
| 2 | 0.768867 | 0.673535 | 0.839236 | 0.088615 | 0.093467 | 0.088615 | 0.088615 | 8.676535x | 2 | 0.787464 | 0 | no | 10/16 | distributed-lift |
| 3 | 0.247474 | 0.212591 | 0.246134 | 0.157759 | 0.166683 | 0.157759 | 0.157759 | 1.568684x | 1 | 0.766568 | 0 | no | 12/16 | distributed-lift |
| 4 | 0.62951 | 0.618636 | 0.631109 | 1.306e-11 | 1.310e-11 | 1.306e-11 | 1.306e-11 | 4.821e+10x | 2 | 0.858174 | 0 | no | 4/12 | distributed-lift |
| 5 | 0.570535 | 0.526547 | 0.562227 | 0.117206 | 0.121611 | 0.117206 | 0.117206 | 4.867789x | 2 | 0.875288 | 0 | no | 14/16 | distributed-lift |
| 6 | 0.528679 | 0.521143 | 0.529641 | 0.180204 | 0.166802 | 0.180204 | 0.157769 | 2.933779x | 0 | 0.358956 | 0.022435 | validation-veto | 11/16 | distributed-lift |
| 7 | 0.779231 | 0.643735 | 1.059085 | 0.001729 | 0.001784 | 0.001729 | 0.001729 | 450.733582x | 0 | 0.897601 | 0 | no | 11/16 | distributed-lift |
| 8 | 0.386966 | 0.379033 | 0.413881 | 0.015241 | 0.015717 | 0.015241 | 0.015241 | 25.390176x | 2 | 0.58227 | 0 | no | 11/16 | distributed-lift |
| 9 | 0.60038 | 7.125e+28 | 0.599308 | 0.157757 | 0.166682 | 0.157757 | 0.157757 | 3.805723x | 1 | 0.847734 | 0 | no | 11/16 | distributed-lift |
| 10 | 0.188119 | 0.174087 | 0.195598 | 0.079594 | 0.082005 | 0.079594 | 0.079594 | 2.363479x | 2 | 0.665628 | 0 | no | 11/16 | distributed-lift |
| 11 | 0.270593 | 0.138341 | 0.332051 | 0.002155 | 0.00219 | 0.002155 | 0.002155 | 125.546177x | 3 | 0.837618 | 0 | no | 10/16 | distributed-lift |
| 12 | 0.36695 | 0.312266 | 0.391126 | 6.393e-12 | 6.416e-12 | 6.393e-12 | 6.393e-12 | 5.740e+10x | 2 | 0.360337 | 0 | low-nn-confidence | 7/12 | distributed-lift |
| 13 | 0.493314 | 0.412713 | 0.519703 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 1.655e+9x | 3 | 0.935607 | 0 | no | 4/12 | distributed-lift |
| 14 | 0.157756 | 0.142476 | 0.166675 | 6.234e-11 | 6.324e-11 | 6.234e-11 | 6.234e-11 | 2.530e+9x | 0 | 0.916143 | 0 | no | 13/15 | distributed-lift |
| 15 | 0.306881 | 0.285691 | 0.316855 | 0.126615 | 0.120985 | 0.126615 | 0.126615 | 2.423733x | 0 | 0.856451 | 0 | no | 12/16 | distributed-lift |
| 16 | 1.039911 | 0.115525 | 0.162025 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 3.489e+9x | 0 | 0.512051 | 0 | low-nn-confidence | 7/12 | distributed-lift |
| 17 | 0.060859 | 0.045547 | 0.059957 | 0.048697 | 0.048687 | 0.048697 | 0.002668 | 1.249756x | 0 | 0.587418 | 0.046028 | validation-veto | 14/16 | distributed-lift |
| 18 | 0.186365 | 0.1244 | 0.13638 | 0.034865 | 0.03463 | 0.034865 | 0.034865 | 5.345343x | 1 | 0.90333 | 0 | no | 11/16 | distributed-lift |
| 19 | 0.11987 | 0.109808 | 1.223128 | 0.056912 | 0.111508 | 0.056912 | 0.056912 | 2.106216x | 3 | 0.884414 | 0 | no | 14/16 | distributed-lift |
| 20 | 0.090597 | 0.065265 | 0.099378 | 0.023105 | 0.02421 | 0.023105 | 0.002668 | 3.921161x | 2 | 0.706382 | 0.020436 | no | 11/16 | distributed-lift |
| 21 | 0.35326 | 0.343066 | 0.362177 | 2.588e-11 | 2.600e-11 | 2.588e-11 | 2.588e-11 | 1.365e+10x | 2 | 0.852709 | 0 | no | 4/12 | distributed-lift |
| 22 | 0.148652 | 0.138504 | 0.147318 | 0.052732 | 0.102515 | 0.052732 | 0.052732 | 2.818993x | 0 | 0.759951 | 0 | no | 10/16 | distributed-lift |
| 23 | 0.13161 | 0.124415 | 0.13997 | 0.013468 | 0.013486 | 0.013468 | 0.013468 | 9.772387x | 1 | 0.864504 | 0 | no | 10/16 | distributed-lift |
| 24 | 47.199434 | 46.101695 | 45.954595 | 1.112e-10 | 1.173e-10 | 1.112e-10 | 1.112e-10 | 4.245e+11x | 3 | 0.327983 | 0 | low-nn-confidence | 3/12 | distributed-lift |

