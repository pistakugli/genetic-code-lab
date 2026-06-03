# Distributed worker-island + NN policy benchmark

Generated: 2026-06-03T06:12:51.755Z

## Scope

- Runs real `DistributedManager` worker-thread islands.
- Uses a shared `MigrationHub` with atomic migration counters.
- Uses the existing `MLP` as a scorer, then applies confidence, validation, and guard-holdout veto/fallback policy.
- This is the production-runtime proof lane; the wide seed sweep is part of `release:gate`.

## Summary

- Mean single-island MAE: 2.324315
- Mean policy MAE: 0.053544
- Mean robust holdout-only MAE: 0.053544
- Mean oracle-best MAE: 0.050382
- Median lift: 6.411119x
- Median lift on non-degenerate seeds: 6.411119x
- Success rate: 100%
- Non-degenerate success rate: 100%
- Non-regression rate vs single baseline: 100%
- Completed seeds: 24/24
- Runtime timeouts: 0
- Runtime errors: 0
- Single-baseline solved seeds: 0
- Single-baseline fallbacks: 0
- Mean regret vs oracle: 0.003162
- Policy beats robust holdout-only rate: 0%
- NN holdout accuracy: 0.833494
- NN holdout Brier: 0.14578
- NN holdout AUC: 0.887926
- Hub drops: 358
- Hub pull successes: 233
- Fallbacks: 4

| Seed | Single MAE | Single val MAE | Single guard MAE | Policy MAE | Policy guard MAE | Holdout-only MAE | Oracle MAE | Lift | Island | NN p | Regret | Fallback | Hub pulls | Outcome |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.157757 | 0.142474 | 0.166682 | 0.108044 | 0.104282 | 0.108044 | 0.108044 | 1.460123x | 0 | 0.928332 | 0 | no | 11/16 | distributed-lift |
| 2 | 0.768867 | 0.673535 | 0.839236 | 0.114721 | 0.120008 | 0.114721 | 0.114721 | 6.702042x | 2 | 0.634716 | 0 | no | 13/16 | distributed-lift |
| 3 | 0.247474 | 0.212591 | 0.246134 | 0.157759 | 0.166683 | 0.157759 | 0.157759 | 1.568684x | 1 | 0.741259 | 0 | no | 11/16 | distributed-lift |
| 4 | 0.62951 | 0.618636 | 0.631109 | 1.306e-11 | 1.310e-11 | 1.306e-11 | 1.306e-11 | 4.821e+10x | 2 | 0.836871 | 0 | no | 7/12 | distributed-lift |
| 5 | 0.570535 | 0.526547 | 0.562227 | 0.121143 | 0.127754 | 0.121143 | 0.121143 | 4.709607x | 2 | 0.892823 | 0 | no | 14/16 | distributed-lift |
| 6 | 0.528679 | 0.521143 | 0.529641 | 0.180421 | 0.182955 | 0.180421 | 0.157762 | 2.930256x | 3 | 0.952028 | 0.022659 | no | 12/16 | distributed-lift |
| 7 | 0.779231 | 0.643735 | 1.059085 | 0.017649 | 0.016486 | 0.017649 | 0.017649 | 44.152458x | 0 | 0.928554 | 0 | no | 11/16 | distributed-lift |
| 8 | 0.386966 | 0.379033 | 0.413881 | 0.023801 | 0.024533 | 0.023801 | 0.023801 | 16.258252x | 2 | 0.727877 | 0 | no | 10/16 | distributed-lift |
| 9 | 0.60038 | 7.125e+28 | 0.599308 | 0.157757 | 0.166682 | 0.157757 | 0.157757 | 3.805723x | 1 | 0.877592 | 0 | no | 11/16 | distributed-lift |
| 10 | 0.188119 | 0.174087 | 0.195598 | 0.081225 | 0.083824 | 0.081225 | 0.081225 | 2.31602x | 2 | 0.852394 | 0 | no | 10/16 | distributed-lift |
| 11 | 0.270593 | 0.138341 | 0.332051 | 0.006011 | 0.005824 | 0.006011 | 0.006011 | 45.01422x | 2 | 0.883954 | 0 | no | 11/16 | distributed-lift |
| 12 | 0.36695 | 0.312266 | 0.391126 | 6.393e-12 | 6.416e-12 | 6.393e-12 | 6.393e-12 | 5.740e+10x | 2 | 0.15416 | 0 | low-nn-confidence | 3/12 | distributed-lift |
| 13 | 0.493314 | 0.412713 | 0.519703 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 1.655e+9x | 3 | 0.924529 | 0 | no | 4/12 | distributed-lift |
| 14 | 0.157756 | 0.142476 | 0.166675 | 1.406e-7 | 1.415e-7 | 1.406e-7 | 1.406e-7 | 1.122e+6x | 0 | 0.754084 | 0 | no | 9/14 | distributed-lift |
| 15 | 0.306881 | 0.285691 | 0.316855 | 0.126684 | 0.121426 | 0.126684 | 0.126684 | 2.422419x | 0 | 0.796991 | 0 | no | 12/16 | distributed-lift |
| 16 | 1.039911 | 0.115525 | 0.162025 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 3.489e+9x | 0 | 0.483985 | 0 | low-nn-confidence | 7/12 | distributed-lift |
| 17 | 0.060859 | 0.045547 | 0.059957 | 0.043599 | 0.043445 | 0.043599 | 0.002668 | 1.395871x | 0 | 0.563998 | 0.040931 | low-nn-confidence | 10/16 | distributed-lift |
| 18 | 0.186365 | 0.1244 | 0.13638 | 0.034865 | 0.03463 | 0.034865 | 0.034865 | 5.345343x | 1 | 0.92299 | 0 | no | 11/16 | distributed-lift |
| 19 | 0.11987 | 0.109808 | 1.223128 | 0.049969 | 0.103459 | 0.049969 | 0.049969 | 2.398898x | 3 | 0.962006 | 0 | no | 11/16 | distributed-lift |
| 20 | 0.090597 | 0.065265 | 0.099378 | 0.014803 | 0.014714 | 0.014803 | 0.002506 | 6.120196x | 3 | 0.654658 | 0.012297 | no | 11/16 | distributed-lift |
| 21 | 0.35326 | 0.343066 | 0.362177 | 2.588e-11 | 2.600e-11 | 2.588e-11 | 2.588e-11 | 1.365e+10x | 2 | 0.775203 | 0 | no | 4/12 | distributed-lift |
| 22 | 0.148652 | 0.138504 | 0.147318 | 0.033867 | 0.091186 | 0.033867 | 0.033867 | 4.389288x | 0 | 0.641728 | 0 | no | 12/16 | distributed-lift |
| 23 | 0.13161 | 0.124415 | 0.13997 | 0.012741 | 0.012434 | 0.012741 | 0.012741 | 10.329402x | 1 | 0.835832 | 0 | no | 15/16 | distributed-lift |
| 24 | 47.199434 | 46.101695 | 45.954595 | 1.112e-10 | 1.173e-10 | 1.112e-10 | 1.112e-10 | 4.245e+11x | 3 | 0.557216 | 0 | low-nn-confidence | 3/12 | distributed-lift |

