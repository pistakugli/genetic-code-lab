# Distributed worker-island + NN policy benchmark

Generated: 2026-06-01T13:38:15.941Z

## Scope

- Runs real `DistributedManager` worker-thread islands.
- Uses a shared `MigrationHub` with atomic migration counters.
- Uses the existing `MLP` as a scorer, then applies confidence, validation, and guard-holdout veto/fallback policy.
- This is the production-runtime proof lane; the wide seed sweep is part of `release:gate`.

## Summary

- Mean single-island MAE: 2.324315
- Mean policy MAE: 0.053541
- Mean robust holdout-only MAE: 0.05052
- Mean oracle-best MAE: 0.048849
- Median lift: 9.553714x
- Median lift on non-degenerate seeds: 9.553714x
- Success rate: 96%
- Non-degenerate success rate: 96%
- Non-regression rate vs single baseline: 100%
- Completed seeds: 24/24
- Runtime timeouts: 0
- Runtime errors: 0
- Single-baseline solved seeds: 0
- Single-baseline fallbacks: 1
- Mean regret vs oracle: 0.001671
- Policy beats robust holdout-only rate: 0%
- NN holdout accuracy: 0.830479
- NN holdout Brier: 0.140223
- NN holdout AUC: 0.80811
- Hub drops: 355
- Hub pull successes: 225
- Fallbacks: 4

| Seed | Single MAE | Single val MAE | Single guard MAE | Policy MAE | Policy guard MAE | Holdout-only MAE | Oracle MAE | Lift | Island | NN p | Regret | Fallback | Hub pulls | Outcome |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.157757 | 0.142474 | 0.166682 | 0.107918 | 0.108315 | 0.107918 | 0.107918 | 1.461819x | 0 | 0.939527 | 0 | no | 11/16 | distributed-lift |
| 2 | 0.768867 | 0.673535 | 0.839236 | 0.082363 | 0.08574 | 0.082363 | 0.082363 | 9.335049x | 2 | 0.418208 | 0 | low-nn-confidence | 11/16 | distributed-lift |
| 3 | 0.247474 | 0.212591 | 0.246134 | 0.157757 | 0.166682 | 0.157757 | 0.157757 | 1.568701x | 1 | 0.638574 | 0 | no | 11/16 | distributed-lift |
| 4 | 0.62951 | 0.618636 | 0.631109 | 1.306e-11 | 1.310e-11 | 1.306e-11 | 1.306e-11 | 4.821e+10x | 2 | 0.806695 | 0 | no | 4/11 | distributed-lift |
| 5 | 0.570535 | 0.526547 | 0.562227 | 0.116272 | 0.119486 | 0.116272 | 0.116272 | 4.906916x | 2 | 0.877478 | 0 | no | 14/16 | distributed-lift |
| 6 | 0.528679 | 0.521143 | 0.529641 | 0.160164 | 0.194069 | 0.160164 | 0.160164 | 3.300857x | 1 | 0.725594 | 0 | no | 10/16 | distributed-lift |
| 7 | 0.779231 | 0.643735 | 1.059085 | 0.067793 | 0.062124 | 0.067793 | 0.067793 | 11.494324x | 0 | 0.889061 | 0 | no | 10/16 | distributed-lift |
| 8 | 0.386966 | 0.379033 | 0.413881 | 0.024109 | 0.024782 | 0.024109 | 0.024109 | 16.050532x | 2 | 0.589454 | 0 | no | 11/16 | distributed-lift |
| 9 | 0.60038 | 7.125e+28 | 0.599308 | 0.104307 | 0.106154 | 0.104307 | 0.104307 | 5.755891x | 0 | 0.923701 | 0 | no | 11/16 | distributed-lift |
| 10 | 0.188119 | 0.174087 | 0.195598 | 0.034599 | 0.035071 | 0.034599 | 0.034599 | 5.43711x | 3 | 0.929347 | 0 | no | 11/16 | distributed-lift |
| 11 | 0.270593 | 0.138341 | 0.332051 | 8.955e-8 | 8.973e-8 | 8.955e-8 | 8.955e-8 | 3.022e+6x | 3 | 0.858531 | 0 | no | 9/14 | distributed-lift |
| 12 | 0.36695 | 0.312266 | 0.391126 | 6.393e-12 | 6.416e-12 | 6.393e-12 | 6.393e-12 | 5.740e+10x | 2 | 0.471998 | 0 | low-nn-confidence | 3/12 | distributed-lift |
| 13 | 0.493314 | 0.412713 | 0.519703 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 1.655e+9x | 3 | 0.921264 | 0 | no | 3/12 | distributed-lift |
| 14 | 0.157756 | 0.142476 | 0.166675 | 8.955e-8 | 8.973e-8 | 8.955e-8 | 8.955e-8 | 1.762e+6x | 0 | 0.860368 | 0 | no | 13/14 | distributed-lift |
| 15 | 0.306881 | 0.285691 | 0.316855 | 0.126091 | 0.119986 | 0.126091 | 0.126091 | 2.433808x | 0 | 0.823445 | 0 | no | 12/16 | distributed-lift |
| 16 | 1.039911 | 0.115525 | 0.162025 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 3.489e+9x | 0 | 0.886954 | 0 | no | 7/12 | distributed-lift |
| 17 | 0.060859 | 0.045547 | 0.059957 | 0.042762 | 0.042024 | 0.042762 | 0.002668 | 1.423196x | 0 | 0.568092 | 0.040094 | low-nn-confidence | 10/16 | distributed-lift |
| 18 | 0.186365 | 0.1244 | 0.13638 | 0.034865 | 0.03463 | 0.034865 | 0.034865 | 5.345343x | 1 | 0.851115 | 0 | no | 11/16 | distributed-lift |
| 19 | 0.11987 | 0.109808 | 1.223128 | 0.061205 | 0.11856 | 0.061205 | 0.061205 | 1.958509x | 3 | 0.916246 | 0 | no | 11/16 | distributed-lift |
| 20 | 0.090597 | 0.065265 | 0.099378 | 0.002668 | 0.05799 | 0.002668 | 0.002668 | 33.952061x | 0 | 0.821039 | 0 | no | 11/16 | distributed-lift |
| 21 | 0.35326 | 0.343066 | 0.362177 | 2.588e-11 | 2.600e-11 | 2.588e-11 | 2.588e-11 | 1.365e+10x | 2 | 0.915247 | 0 | no | 3/12 | distributed-lift |
| 22 | 0.148652 | 0.138504 | 0.147318 | 0.148652 | 0.147318 | 0.076136 | 0.076136 | 1x | -2 | 0.837167 | 0 | single-baseline-guard-veto | 11/16 | single-baseline-fallback |
| 23 | 0.13161 | 0.124415 | 0.13997 | 0.013468 | 0.013486 | 0.013468 | 0.013468 | 9.77238x | 1 | 0.883736 | 0 | no | 14/16 | distributed-lift |
| 24 | 47.199434 | 46.101695 | 45.954595 | 1.112e-10 | 1.173e-10 | 1.112e-10 | 1.112e-10 | 4.245e+11x | 3 | 0.772877 | 0 | no | 3/12 | distributed-lift |

