# Distributed worker-island + NN policy benchmark

Generated: 2026-06-30T19:12:51.931Z

## Scope

- Runs real `DistributedManager` worker-thread islands.
- Uses a shared `MigrationHub` with atomic migration counters.
- Uses the existing `MLP` as a scorer, then applies confidence, validation, and guard-holdout veto/fallback policy.
- This is the production-runtime proof lane; the wide seed sweep is part of `release:gate`.

## Summary

- Mean single-island MAE: 2.324315
- Mean policy MAE: 0.053276
- Mean robust holdout-only MAE: 0.053276
- Mean oracle-best MAE: 0.049732
- Median lift: 11.397626x
- Median lift on non-degenerate seeds: 11.397626x
- Success rate: 100%
- Non-degenerate success rate: 100%
- Non-regression rate vs single baseline: 100%
- Completed seeds: 24/24
- Runtime timeouts: 0
- Runtime errors: 0
- Single-baseline solved seeds: 0
- Single-baseline fallbacks: 0
- Mean regret vs oracle: 0.003544
- Policy beats robust holdout-only rate: 0%
- NN holdout accuracy: 0.882684
- NN holdout Brier: 0.129812
- NN holdout AUC: 0.874071
- Hub drops: 354
- Hub pull successes: 228
- Fallbacks: 4

| Seed | Single MAE | Single val MAE | Single guard MAE | Policy MAE | Policy guard MAE | Holdout-only MAE | Oracle MAE | Lift | Island | NN p | Regret | Fallback | Hub pulls | Outcome |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.157757 | 0.142474 | 0.166682 | 0.049876 | 0.051992 | 0.049876 | 0.049876 | 3.162973x | 0 | 0.961425 | 0 | no | 11/16 | distributed-lift |
| 2 | 0.768867 | 0.673535 | 0.839236 | 0.116065 | 0.120759 | 0.116065 | 0.116065 | 6.624473x | 2 | 0.851878 | 0 | no | 13/16 | distributed-lift |
| 3 | 0.247474 | 0.212591 | 0.246134 | 0.157759 | 0.166683 | 0.157759 | 0.157759 | 1.568684x | 1 | 0.770372 | 0 | no | 12/16 | distributed-lift |
| 4 | 0.62951 | 0.618636 | 0.631109 | 1.306e-11 | 1.310e-11 | 1.306e-11 | 1.306e-11 | 4.821e+10x | 2 | 0.819534 | 0 | no | 7/12 | distributed-lift |
| 5 | 0.570535 | 0.526547 | 0.562227 | 0.15776 | 0.166684 | 0.15776 | 0.15776 | 3.616476x | 1 | 0.682333 | 0 | no | 14/16 | distributed-lift |
| 6 | 0.528679 | 0.521143 | 0.529641 | 0.208072 | 0.201277 | 0.208072 | 0.208072 | 2.540852x | 1 | 0.901491 | 0 | no | 11/16 | distributed-lift |
| 7 | 0.779231 | 0.643735 | 1.059085 | 0.035863 | 0.051373 | 0.035863 | 0.035863 | 21.727732x | 0 | 0.918727 | 0 | no | 11/16 | distributed-lift |
| 8 | 0.386966 | 0.379033 | 0.413881 | 0.02393 | 0.024565 | 0.02393 | 0.02393 | 16.170779x | 2 | 0.598205 | 0 | no | 10/16 | distributed-lift |
| 9 | 0.60038 | 7.125e+28 | 0.599308 | 0.157757 | 0.166682 | 0.157757 | 0.157757 | 3.805723x | 1 | 0.887487 | 0 | no | 11/16 | distributed-lift |
| 10 | 0.188119 | 0.174087 | 0.195598 | 3.302e-9 | 3.309e-9 | 3.302e-9 | 3.302e-9 | 5.696e+7x | 0 | 0.753792 | 0 | no | 9/14 | distributed-lift |
| 11 | 0.270593 | 0.138341 | 0.332051 | 0.000265 | 0.000266 | 0.000265 | 0.000265 | 1019.96125x | 3 | 0.909211 | 0 | no | 11/16 | distributed-lift |
| 12 | 0.36695 | 0.312266 | 0.391126 | 6.393e-12 | 6.416e-12 | 6.393e-12 | 6.393e-12 | 5.740e+10x | 2 | 0.275569 | 0 | low-nn-confidence | 3/12 | distributed-lift |
| 13 | 0.493314 | 0.412713 | 0.519703 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 1.655e+9x | 3 | 0.924057 | 0 | no | 3/12 | distributed-lift |
| 14 | 0.157756 | 0.142476 | 0.166675 | 1.406e-7 | 1.415e-7 | 1.406e-7 | 1.406e-7 | 1.122e+6x | 0 | 0.919293 | 0 | no | 9/14 | distributed-lift |
| 15 | 0.306881 | 0.285691 | 0.316855 | 0.125057 | 0.118815 | 0.125057 | 0.125057 | 2.45393x | 0 | 0.854053 | 0 | no | 10/16 | distributed-lift |
| 16 | 1.039911 | 0.115525 | 0.162025 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 3.489e+9x | 0 | 0.169707 | 0 | low-nn-confidence | 7/12 | distributed-lift |
| 17 | 0.060859 | 0.045547 | 0.059957 | 0.044632 | 0.043935 | 0.044632 | 0.002668 | 1.363567x | 0 | 0.562891 | 0.041964 | low-nn-confidence | 14/16 | distributed-lift |
| 18 | 0.186365 | 0.1244 | 0.13638 | 0.034865 | 0.03463 | 0.034865 | 0.034865 | 5.345343x | 1 | 0.773716 | 0 | no | 11/16 | distributed-lift |
| 19 | 0.11987 | 0.109808 | 1.223128 | 0.056912 | 0.111508 | 0.056912 | 0.056912 | 2.106216x | 3 | 0.909801 | 0 | no | 11/16 | distributed-lift |
| 20 | 0.090597 | 0.065265 | 0.099378 | 0.045758 | 0.044929 | 0.045758 | 0.002668 | 1.979889x | 3 | 0.662099 | 0.04309 | validation-veto | 11/16 | distributed-lift |
| 21 | 0.35326 | 0.343066 | 0.362177 | 2.588e-11 | 2.600e-11 | 2.588e-11 | 2.588e-11 | 1.365e+10x | 2 | 0.799609 | 0 | no | 4/12 | distributed-lift |
| 22 | 0.148652 | 0.138504 | 0.147318 | 0.064061 | 0.11796 | 0.064061 | 0.064061 | 2.320467x | 0 | 0.824701 | 0 | no | 10/16 | distributed-lift |
| 23 | 0.13161 | 0.124415 | 0.13997 | 1.105e-9 | 1.109e-9 | 1.105e-9 | 1.105e-9 | 1.191e+8x | 3 | 0.682626 | 0 | no | 12/14 | distributed-lift |
| 24 | 47.199434 | 46.101695 | 45.954595 | 1.112e-10 | 1.173e-10 | 1.112e-10 | 1.112e-10 | 4.245e+11x | 3 | 0.670938 | 0 | no | 3/12 | distributed-lift |

