# Distributed worker-island + NN policy benchmark

Generated: 2026-06-11T20:07:28.200Z

## Scope

- Runs real `DistributedManager` worker-thread islands.
- Uses a shared `MigrationHub` with atomic migration counters.
- Uses the existing `MLP` as a scorer, then applies confidence, validation, and guard-holdout veto/fallback policy.
- This is the production-runtime proof lane; the wide seed sweep is part of `release:gate`.

## Summary

- Mean single-island MAE: 2.324315
- Mean policy MAE: 0.057724
- Mean robust holdout-only MAE: 0.057844
- Mean oracle-best MAE: 0.053537
- Median lift: 8.240907x
- Median lift on non-degenerate seeds: 8.240907x
- Success rate: 96%
- Non-degenerate success rate: 96%
- Non-regression rate vs single baseline: 100%
- Completed seeds: 24/24
- Runtime timeouts: 0
- Runtime errors: 0
- Single-baseline solved seeds: 0
- Single-baseline fallbacks: 1
- Mean regret vs oracle: 0.004366
- Policy beats robust holdout-only rate: 4%
- NN holdout accuracy: 0.791938
- NN holdout Brier: 0.148546
- NN holdout AUC: 0.827483
- Hub drops: 358
- Hub pull successes: 222
- Fallbacks: 5

| Seed | Single MAE | Single val MAE | Single guard MAE | Policy MAE | Policy guard MAE | Holdout-only MAE | Oracle MAE | Lift | Island | NN p | Regret | Fallback | Hub pulls | Outcome |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.157757 | 0.142474 | 0.166682 | 0.058097 | 0.058448 | 0.058097 | 0.058097 | 2.715403x | 0 | 0.964836 | 0 | no | 11/16 | distributed-lift |
| 2 | 0.768867 | 0.673535 | 0.839236 | 0.114589 | 0.119812 | 0.114589 | 0.114589 | 6.709772x | 2 | 0.584482 | 0 | no | 10/16 | distributed-lift |
| 3 | 0.247474 | 0.212591 | 0.246134 | 0.157759 | 0.166683 | 0.157759 | 0.157759 | 1.568684x | 1 | 0.760984 | 0 | no | 15/16 | distributed-lift |
| 4 | 0.62951 | 0.618636 | 0.631109 | 1.306e-11 | 1.310e-11 | 1.306e-11 | 1.306e-11 | 4.821e+10x | 2 | 0.85462 | 0 | no | 4/12 | distributed-lift |
| 5 | 0.570535 | 0.526547 | 0.562227 | 0.095526 | 0.096918 | 0.095526 | 0.095526 | 5.972591x | 2 | 0.848492 | 0 | no | 11/16 | distributed-lift |
| 6 | 0.528679 | 0.521143 | 0.529641 | 0.173322 | 0.173721 | 0.173322 | 0.158265 | 3.050263x | 3 | 0.894149 | 0.015057 | no | 10/16 | distributed-lift |
| 7 | 0.779231 | 0.643735 | 1.059085 | 0.018451 | 0.016867 | 0.018451 | 0.018451 | 42.233193x | 0 | 0.917206 | 0 | no | 11/16 | distributed-lift |
| 8 | 0.386966 | 0.379033 | 0.413881 | 0.023801 | 0.024533 | 0.023801 | 0.023801 | 16.258252x | 2 | 0.778184 | 0 | no | 11/16 | distributed-lift |
| 9 | 0.60038 | 7.125e+28 | 0.599308 | 0.166093 | 0.157978 | 0.166093 | 0.157757 | 3.614731x | 0 | 0.895675 | 0.008335 | no | 11/16 | distributed-lift |
| 10 | 0.188119 | 0.174087 | 0.195598 | 0.14881 | 0.157227 | 0.14881 | 0.14881 | 1.264155x | 2 | 0.573312 | 0 | validation-veto | 11/16 | distributed-lift |
| 11 | 0.270593 | 0.138341 | 0.332051 | 9.457e-5 | 9.478e-5 | 9.457e-5 | 9.457e-5 | 2861.171403x | 3 | 0.929696 | 0 | no | 11/16 | distributed-lift |
| 12 | 0.36695 | 0.312266 | 0.391126 | 6.393e-12 | 6.416e-12 | 6.393e-12 | 6.393e-12 | 5.740e+10x | 2 | 0.239875 | 0 | low-nn-confidence | 3/12 | distributed-lift |
| 13 | 0.493314 | 0.412713 | 0.519703 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 1.655e+9x | 3 | 0.916917 | 0 | no | 3/12 | distributed-lift |
| 14 | 0.157756 | 0.142476 | 0.166675 | 8.955e-8 | 8.973e-8 | 8.955e-8 | 8.955e-8 | 1.762e+6x | 0 | 0.895105 | 0 | no | 8/14 | distributed-lift |
| 15 | 0.306881 | 0.285691 | 0.316855 | 0.145028 | 0.1373 | 0.145028 | 0.145028 | 2.11601x | 0 | 0.804179 | 0 | no | 11/16 | distributed-lift |
| 16 | 1.039911 | 0.115525 | 0.162025 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 3.489e+9x | 0 | 0.066586 | 0 | low-nn-confidence | 8/12 | distributed-lift |
| 17 | 0.060859 | 0.045547 | 0.059957 | 0.042656 | 0.041505 | 0.042656 | 0.002668 | 1.426733x | 0 | 0.567645 | 0.039988 | low-nn-confidence | 10/16 | distributed-lift |
| 18 | 0.186365 | 0.1244 | 0.13638 | 0.034865 | 0.03463 | 0.034865 | 0.034865 | 5.345343x | 1 | 0.850683 | 0 | no | 10/16 | distributed-lift |
| 19 | 0.11987 | 0.109808 | 1.223128 | 0.000101 | 0.055731 | 0.000101 | 0.000101 | 1181.314629x | 3 | 0.955321 | 0 | no | 10/16 | distributed-lift |
| 20 | 0.090597 | 0.065265 | 0.099378 | 0.044065 | 0.043469 | 0.042662 | 0.002668 | 2.055984x | 2 | 0.738277 | 0.041396 | no | 11/16 | distributed-lift |
| 21 | 0.35326 | 0.343066 | 0.362177 | 2.588e-11 | 2.600e-11 | 2.588e-11 | 2.588e-11 | 1.365e+10x | 2 | 0.805821 | 0 | no | 3/12 | distributed-lift |
| 22 | 0.148652 | 0.138504 | 0.147318 | 0.148652 | 0.147318 | 0.152935 | 0.152935 | 1x | -2 | 0.670991 | 0 | single-baseline-guard-veto | 10/16 | single-baseline-fallback |
| 23 | 0.13161 | 0.124415 | 0.13997 | 0.013468 | 0.013487 | 0.013468 | 0.013468 | 9.772042x | 1 | 0.771425 | 0 | no | 11/16 | distributed-lift |
| 24 | 47.199434 | 46.101695 | 45.954595 | 1.112e-10 | 1.173e-10 | 1.112e-10 | 1.112e-10 | 4.245e+11x | 3 | 0.670116 | 0 | no | 8/12 | distributed-lift |

