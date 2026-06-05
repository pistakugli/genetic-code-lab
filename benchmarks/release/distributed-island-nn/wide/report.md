# Distributed worker-island + NN policy benchmark

Generated: 2026-06-05T15:52:36.626Z

## Scope

- Runs real `DistributedManager` worker-thread islands.
- Uses a shared `MigrationHub` with atomic migration counters.
- Uses the existing `MLP` as a scorer, then applies confidence, validation, and guard-holdout veto/fallback policy.
- This is the production-runtime proof lane; the wide seed sweep is part of `release:gate`.

## Summary

- Mean single-island MAE: 2.324315
- Mean policy MAE: 0.047299
- Mean robust holdout-only MAE: 0.047299
- Mean oracle-best MAE: 0.047299
- Median lift: 14.177104x
- Median lift on non-degenerate seeds: 14.177104x
- Success rate: 100%
- Non-degenerate success rate: 100%
- Non-regression rate vs single baseline: 100%
- Completed seeds: 24/24
- Runtime timeouts: 0
- Runtime errors: 0
- Single-baseline solved seeds: 0
- Single-baseline fallbacks: 0
- Mean regret vs oracle: 0
- Policy beats robust holdout-only rate: 0%
- NN holdout accuracy: 0.805138
- NN holdout Brier: 0.136487
- NN holdout AUC: 0.855338
- Hub drops: 359
- Hub pull successes: 239
- Fallbacks: 1

| Seed | Single MAE | Single val MAE | Single guard MAE | Policy MAE | Policy guard MAE | Holdout-only MAE | Oracle MAE | Lift | Island | NN p | Regret | Fallback | Hub pulls | Outcome |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.157757 | 0.142474 | 0.166682 | 0.052885 | 0.054717 | 0.052885 | 0.052885 | 2.983045x | 0 | 0.938673 | 0 | no | 11/16 | distributed-lift |
| 2 | 0.768867 | 0.673535 | 0.839236 | 0.114976 | 0.11969 | 0.114976 | 0.114976 | 6.687199x | 2 | 0.649945 | 0 | no | 10/16 | distributed-lift |
| 3 | 0.247474 | 0.212591 | 0.246134 | 0.157759 | 0.166683 | 0.157759 | 0.157759 | 1.568684x | 1 | 0.766797 | 0 | no | 11/16 | distributed-lift |
| 4 | 0.62951 | 0.618636 | 0.631109 | 1.306e-11 | 1.310e-11 | 1.306e-11 | 1.306e-11 | 4.821e+10x | 2 | 0.879478 | 0 | no | 4/12 | distributed-lift |
| 5 | 0.570535 | 0.526547 | 0.562227 | 0.13228 | 0.134643 | 0.13228 | 0.13228 | 4.313098x | 2 | 0.89142 | 0 | no | 14/16 | distributed-lift |
| 6 | 0.528679 | 0.521143 | 0.529641 | 0.157763 | 0.196716 | 0.157763 | 0.157763 | 3.351102x | 1 | 0.906777 | 0 | no | 10/16 | distributed-lift |
| 7 | 0.779231 | 0.643735 | 1.059085 | 0.002454 | 0.002426 | 0.002454 | 0.002454 | 317.499467x | 0 | 0.904593 | 0 | no | 10/16 | distributed-lift |
| 8 | 0.386966 | 0.379033 | 0.413881 | 0.021035 | 0.021974 | 0.021035 | 0.021035 | 18.396539x | 2 | 0.676025 | 0 | no | 10/16 | distributed-lift |
| 9 | 0.60038 | 7.125e+28 | 0.599308 | 0.157757 | 0.166682 | 0.157757 | 0.157757 | 3.805723x | 1 | 0.77933 | 0 | no | 11/16 | distributed-lift |
| 10 | 0.188119 | 0.174087 | 0.195598 | 0.032726 | 0.033262 | 0.032726 | 0.032726 | 5.748313x | 3 | 0.94104 | 0 | no | 11/16 | distributed-lift |
| 11 | 0.270593 | 0.138341 | 0.332051 | 8.458e-7 | 8.508e-7 | 8.458e-7 | 8.458e-7 | 3.199e+5x | 3 | 0.652972 | 0 | no | 10/15 | distributed-lift |
| 12 | 0.36695 | 0.312266 | 0.391126 | 6.393e-12 | 6.416e-12 | 6.393e-12 | 6.393e-12 | 5.740e+10x | 2 | 0.17734 | 0 | low-nn-confidence | 3/12 | distributed-lift |
| 13 | 0.493314 | 0.412713 | 0.519703 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 1.655e+9x | 3 | 0.920884 | 0 | no | 7/12 | distributed-lift |
| 14 | 0.157756 | 0.142476 | 0.166675 | 0.002161 | 0.002168 | 0.002161 | 0.002161 | 72.997529x | 0 | 0.941923 | 0 | no | 15/16 | distributed-lift |
| 15 | 0.306881 | 0.285691 | 0.316855 | 0.125569 | 0.11997 | 0.125569 | 0.125569 | 2.443932x | 0 | 0.820903 | 0 | no | 12/16 | distributed-lift |
| 16 | 1.039911 | 0.115525 | 0.162025 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 3.489e+9x | 0 | 0.722306 | 0 | no | 8/12 | distributed-lift |
| 17 | 0.060859 | 0.045547 | 0.059957 | 0.003062 | 0.059451 | 0.003062 | 0.003062 | 19.874955x | 3 | 0.70934 | 0 | no | 11/16 | distributed-lift |
| 18 | 0.186365 | 0.1244 | 0.13638 | 0.034865 | 0.03463 | 0.034865 | 0.034865 | 5.345343x | 1 | 0.873556 | 0 | no | 11/16 | distributed-lift |
| 19 | 0.11987 | 0.109808 | 1.223128 | 0.062433 | 0.116141 | 0.062433 | 0.062433 | 1.919957x | 3 | 0.924444 | 0 | no | 14/16 | distributed-lift |
| 20 | 0.090597 | 0.065265 | 0.099378 | 0.002653 | 0.057925 | 0.002653 | 0.002653 | 34.152794x | 0 | 0.666206 | 0 | no | 11/16 | distributed-lift |
| 21 | 0.35326 | 0.343066 | 0.362177 | 2.588e-11 | 2.600e-11 | 2.588e-11 | 2.588e-11 | 1.365e+10x | 2 | 0.828547 | 0 | no | 4/12 | distributed-lift |
| 22 | 0.148652 | 0.138504 | 0.147318 | 0.061591 | 0.11913 | 0.061591 | 0.061591 | 2.41353x | 0 | 0.881753 | 0 | no | 10/16 | distributed-lift |
| 23 | 0.13161 | 0.124415 | 0.13997 | 0.013217 | 0.013305 | 0.013217 | 0.013217 | 9.95767x | 1 | 0.824474 | 0 | no | 14/16 | distributed-lift |
| 24 | 47.199434 | 46.101695 | 45.954595 | 1.112e-10 | 1.173e-10 | 1.112e-10 | 1.112e-10 | 4.245e+11x | 3 | 0.757767 | 0 | no | 7/12 | distributed-lift |

