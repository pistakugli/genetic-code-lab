# Distributed worker-island + NN policy benchmark

Generated: 2026-06-02T18:49:53.975Z

## Scope

- Runs real `DistributedManager` worker-thread islands.
- Uses a shared `MigrationHub` with atomic migration counters.
- Uses the existing `MLP` as a scorer, then applies confidence, validation, and guard-holdout veto/fallback policy.
- This is the production-runtime proof lane; the wide seed sweep is part of `release:gate`.

## Summary

- Mean single-island MAE: 2.324315
- Mean policy MAE: 0.051738
- Mean robust holdout-only MAE: 0.051738
- Mean oracle-best MAE: 0.04827
- Median lift: 6.510776x
- Median lift on non-degenerate seeds: 6.510776x
- Success rate: 100%
- Non-degenerate success rate: 100%
- Non-regression rate vs single baseline: 100%
- Completed seeds: 24/24
- Runtime timeouts: 0
- Runtime errors: 0
- Single-baseline solved seeds: 0
- Single-baseline fallbacks: 0
- Mean regret vs oracle: 0.003468
- Policy beats robust holdout-only rate: 0%
- NN holdout accuracy: 0.808597
- NN holdout Brier: 0.143466
- NN holdout AUC: 0.857724
- Hub drops: 359
- Hub pull successes: 229
- Fallbacks: 5

| Seed | Single MAE | Single val MAE | Single guard MAE | Policy MAE | Policy guard MAE | Holdout-only MAE | Oracle MAE | Lift | Island | NN p | Regret | Fallback | Hub pulls | Outcome |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.157757 | 0.142474 | 0.166682 | 0.048216 | 0.049905 | 0.048216 | 0.048216 | 3.271896x | 0 | 0.943507 | 0 | no | 11/16 | distributed-lift |
| 2 | 0.768867 | 0.673535 | 0.839236 | 0.114736 | 0.119877 | 0.114736 | 0.114736 | 6.701178x | 2 | 0.548722 | 0 | low-nn-confidence | 10/16 | distributed-lift |
| 3 | 0.247474 | 0.212591 | 0.246134 | 0.157759 | 0.166683 | 0.157759 | 0.157759 | 1.568684x | 1 | 0.759644 | 0 | no | 14/16 | distributed-lift |
| 4 | 0.62951 | 0.618636 | 0.631109 | 1.306e-11 | 1.310e-11 | 1.306e-11 | 1.306e-11 | 4.821e+10x | 2 | 0.899731 | 0 | no | 3/12 | distributed-lift |
| 5 | 0.570535 | 0.526547 | 0.562227 | 0.098337 | 0.100661 | 0.098337 | 0.098337 | 5.801809x | 2 | 0.858732 | 0 | no | 14/16 | distributed-lift |
| 6 | 0.528679 | 0.521143 | 0.529641 | 0.134861 | 0.168152 | 0.134861 | 0.134861 | 3.920169x | 1 | 0.936991 | 0 | no | 10/16 | distributed-lift |
| 7 | 0.779231 | 0.643735 | 1.059085 | 0.019167 | 0.018671 | 0.019167 | 0.019167 | 40.65379x | 0 | 0.859545 | 0 | no | 11/16 | distributed-lift |
| 8 | 0.386966 | 0.379033 | 0.413881 | 0.015295 | 0.015646 | 0.015295 | 0.015295 | 25.300563x | 2 | 0.592314 | 0 | no | 11/16 | distributed-lift |
| 9 | 0.60038 | 7.125e+28 | 0.599308 | 0.157757 | 0.166682 | 0.157757 | 0.157757 | 3.805723x | 1 | 0.779808 | 0 | no | 11/16 | distributed-lift |
| 10 | 0.188119 | 0.174087 | 0.195598 | 0.14881 | 0.157227 | 0.14881 | 0.14881 | 1.264155x | 2 | 0.648597 | 0 | validation-veto | 12/16 | distributed-lift |
| 11 | 0.270593 | 0.138341 | 0.332051 | 2.655e-7 | 2.661e-7 | 2.655e-7 | 2.655e-7 | 1.019e+6x | 3 | 0.91456 | 0 | no | 9/15 | distributed-lift |
| 12 | 0.36695 | 0.312266 | 0.391126 | 6.393e-12 | 6.416e-12 | 6.393e-12 | 6.393e-12 | 5.740e+10x | 2 | 0.433858 | 0 | low-nn-confidence | 3/12 | distributed-lift |
| 13 | 0.493314 | 0.412713 | 0.519703 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 1.655e+9x | 3 | 0.920484 | 0 | no | 3/12 | distributed-lift |
| 14 | 0.157756 | 0.142476 | 0.166675 | 0.000958 | 0.000973 | 0.000958 | 0.000958 | 164.595052x | 0 | 0.943434 | 0 | no | 14/16 | distributed-lift |
| 15 | 0.306881 | 0.285691 | 0.316855 | 0.124305 | 0.117273 | 0.124305 | 0.124305 | 2.468778x | 0 | 0.844651 | 0 | no | 10/16 | distributed-lift |
| 16 | 1.039911 | 0.115525 | 0.162025 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 3.489e+9x | 0 | 0.837477 | 0 | no | 7/12 | distributed-lift |
| 17 | 0.060859 | 0.045547 | 0.059957 | 0.042952 | 0.042158 | 0.042952 | 0.002668 | 1.416904x | 0 | 0.563334 | 0.040284 | low-nn-confidence | 10/16 | distributed-lift |
| 18 | 0.186365 | 0.1244 | 0.13638 | 0.034865 | 0.03463 | 0.034865 | 0.034865 | 5.345343x | 1 | 0.818517 | 0 | no | 11/16 | distributed-lift |
| 19 | 0.11987 | 0.109808 | 1.223128 | 0.061082 | 0.118345 | 0.061082 | 0.061082 | 1.962424x | 3 | 0.910432 | 0 | no | 11/16 | distributed-lift |
| 20 | 0.090597 | 0.065265 | 0.099378 | 0.045618 | 0.047902 | 0.045618 | 0.002668 | 1.98597x | 2 | 0.880298 | 0.04295 | no | 11/16 | distributed-lift |
| 21 | 0.35326 | 0.343066 | 0.362177 | 2.588e-11 | 2.600e-11 | 2.588e-11 | 2.588e-11 | 1.365e+10x | 2 | 0.807848 | 0 | no | 4/12 | distributed-lift |
| 22 | 0.148652 | 0.138504 | 0.147318 | 0.023519 | 0.080204 | 0.023519 | 0.023519 | 6.320374x | 0 | 0.752032 | 0 | no | 11/16 | distributed-lift |
| 23 | 0.13161 | 0.124415 | 0.13997 | 0.013468 | 0.013486 | 0.013468 | 0.013468 | 9.77235x | 1 | 0.644772 | 0 | no | 10/16 | distributed-lift |
| 24 | 47.199434 | 46.101695 | 45.954595 | 1.112e-10 | 1.173e-10 | 1.112e-10 | 1.112e-10 | 4.245e+11x | 3 | 0.409704 | 0 | low-nn-confidence | 8/12 | distributed-lift |

