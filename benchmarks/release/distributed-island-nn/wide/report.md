# Distributed worker-island + NN policy benchmark

Generated: 2026-06-01T16:13:27.215Z

## Scope

- Runs real `DistributedManager` worker-thread islands.
- Uses a shared `MigrationHub` with atomic migration counters.
- Uses the existing `MLP` as a scorer, then applies confidence, validation, and guard-holdout veto/fallback policy.
- This is the production-runtime proof lane; the wide seed sweep is part of `release:gate`.

## Summary

- Mean single-island MAE: 2.324315
- Mean policy MAE: 0.05799
- Mean robust holdout-only MAE: 0.058063
- Mean oracle-best MAE: 0.056252
- Median lift: 8.265086x
- Median lift on non-degenerate seeds: 8.265086x
- Success rate: 100%
- Non-degenerate success rate: 100%
- Non-regression rate vs single baseline: 100%
- Completed seeds: 24/24
- Runtime timeouts: 0
- Runtime errors: 0
- Single-baseline solved seeds: 0
- Single-baseline fallbacks: 0
- Mean regret vs oracle: 0.001738
- Policy beats robust holdout-only rate: 4%
- NN holdout accuracy: 0.838088
- NN holdout Brier: 0.14426
- NN holdout AUC: 0.86508
- Hub drops: 358
- Hub pull successes: 229
- Fallbacks: 6

| Seed | Single MAE | Single val MAE | Single guard MAE | Policy MAE | Policy guard MAE | Holdout-only MAE | Oracle MAE | Lift | Island | NN p | Regret | Fallback | Hub pulls | Outcome |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.157757 | 0.142474 | 0.166682 | 0.055917 | 0.057152 | 0.055917 | 0.055917 | 2.821254x | 0 | 0.946994 | 0 | no | 12/16 | distributed-lift |
| 2 | 0.768867 | 0.673535 | 0.839236 | 0.115638 | 0.121048 | 0.115638 | 0.115638 | 6.648911x | 2 | 0.28978 | 0 | low-nn-confidence | 14/16 | distributed-lift |
| 3 | 0.247474 | 0.212591 | 0.246134 | 0.157759 | 0.166683 | 0.157759 | 0.157759 | 1.568684x | 1 | 0.535995 | 0 | low-nn-confidence | 12/16 | distributed-lift |
| 4 | 0.62951 | 0.618636 | 0.631109 | 1.306e-11 | 1.310e-11 | 1.306e-11 | 1.306e-11 | 4.821e+10x | 2 | 0.781468 | 0 | no | 4/12 | distributed-lift |
| 5 | 0.570535 | 0.526547 | 0.562227 | 0.040896 | 0.042527 | 0.040896 | 0.040896 | 13.950975x | 2 | 0.8768 | 0 | no | 14/16 | distributed-lift |
| 6 | 0.528679 | 0.521143 | 0.529641 | 0.157762 | 0.195227 | 0.157762 | 0.157762 | 3.351108x | 1 | 0.874833 | 0 | no | 12/16 | distributed-lift |
| 7 | 0.779231 | 0.643735 | 1.059085 | 0.054244 | 0.055456 | 0.054244 | 0.054244 | 14.365236x | 0 | 0.903531 | 0 | no | 10/16 | distributed-lift |
| 8 | 0.386966 | 0.379033 | 0.413881 | 0.023872 | 0.02454 | 0.023872 | 0.023872 | 16.210352x | 2 | 0.738498 | 0 | no | 10/16 | distributed-lift |
| 9 | 0.60038 | 7.125e+28 | 0.599308 | 0.157757 | 0.166682 | 0.157757 | 0.157757 | 3.805723x | 1 | 0.852991 | 0 | no | 11/16 | distributed-lift |
| 10 | 0.188119 | 0.174087 | 0.195598 | 0.14881 | 0.157227 | 0.14881 | 0.148002 | 1.264155x | 2 | 0.614127 | 0.000808 | validation-veto | 12/16 | distributed-lift |
| 11 | 0.270593 | 0.138341 | 0.332051 | 0.158249 | 0.167764 | 0.160007 | 0.158249 | 1.709918x | 1 | 0.628952 | 0 | no | 11/16 | distributed-lift |
| 12 | 0.36695 | 0.312266 | 0.391126 | 6.393e-12 | 6.416e-12 | 6.393e-12 | 6.393e-12 | 5.740e+10x | 2 | 0.658004 | 0 | no | 3/12 | distributed-lift |
| 13 | 0.493314 | 0.412713 | 0.519703 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 1.655e+9x | 3 | 0.904471 | 0 | no | 3/12 | distributed-lift |
| 14 | 0.157756 | 0.142476 | 0.166675 | 8.955e-8 | 8.973e-8 | 8.955e-8 | 8.955e-8 | 1.762e+6x | 0 | 0.901784 | 0 | no | 8/14 | distributed-lift |
| 15 | 0.306881 | 0.285691 | 0.316855 | 0.124577 | 0.117569 | 0.124577 | 0.124577 | 2.463391x | 0 | 0.806142 | 0 | no | 11/16 | distributed-lift |
| 16 | 1.039911 | 0.115525 | 0.162025 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 3.489e+9x | 0 | 0.358611 | 0 | low-nn-confidence | 8/12 | distributed-lift |
| 17 | 0.060859 | 0.045547 | 0.059957 | 0.040436 | 0.039673 | 0.040436 | 0.002651 | 1.505057x | 0 | 0.572063 | 0.037785 | low-nn-confidence | 10/16 | distributed-lift |
| 18 | 0.186365 | 0.1244 | 0.13638 | 0.034865 | 0.03463 | 0.034865 | 0.034865 | 5.345343x | 1 | 0.826387 | 0 | no | 12/16 | distributed-lift |
| 19 | 0.11987 | 0.109808 | 1.223128 | 0.061122 | 0.118424 | 0.061122 | 0.061122 | 1.961165x | 3 | 0.895297 | 0 | no | 11/16 | distributed-lift |
| 20 | 0.090597 | 0.065265 | 0.099378 | 0.00578 | 0.006151 | 0.00578 | 0.002666 | 15.674646x | 2 | 0.684688 | 0.003114 | no | 10/16 | distributed-lift |
| 21 | 0.35326 | 0.343066 | 0.362177 | 2.588e-11 | 2.600e-11 | 2.588e-11 | 2.588e-11 | 1.365e+10x | 2 | 0.89786 | 0 | no | 4/12 | distributed-lift |
| 22 | 0.148652 | 0.138504 | 0.147318 | 0.040756 | 0.098364 | 0.040756 | 0.040756 | 3.647405x | 0 | 0.785363 | 0 | no | 10/16 | distributed-lift |
| 23 | 0.13161 | 0.124415 | 0.13997 | 0.013319 | 0.013271 | 0.013319 | 0.013319 | 9.88126x | 1 | 0.870178 | 0 | no | 14/16 | distributed-lift |
| 24 | 47.199434 | 46.101695 | 45.954595 | 1.112e-10 | 1.173e-10 | 1.112e-10 | 1.112e-10 | 4.245e+11x | 3 | 0.578955 | 0 | low-nn-confidence | 3/12 | distributed-lift |

