# Distributed worker-island + NN policy benchmark

Generated: 2026-06-02T20:41:12.580Z

## Scope

- Runs real `DistributedManager` worker-thread islands.
- Uses a shared `MigrationHub` with atomic migration counters.
- Uses the existing `MLP` as a scorer, then applies confidence, validation, and guard-holdout veto/fallback policy.
- This is the production-runtime proof lane; the wide seed sweep is part of `release:gate`.

## Summary

- Mean single-island MAE: 2.324315
- Mean policy MAE: 0.052738
- Mean robust holdout-only MAE: 0.052738
- Mean oracle-best MAE: 0.051529
- Median lift: 7.378102x
- Median lift on non-degenerate seeds: 7.378102x
- Success rate: 100%
- Non-degenerate success rate: 100%
- Non-regression rate vs single baseline: 100%
- Completed seeds: 24/24
- Runtime timeouts: 0
- Runtime errors: 0
- Single-baseline solved seeds: 0
- Single-baseline fallbacks: 0
- Mean regret vs oracle: 0.00121
- Policy beats robust holdout-only rate: 0%
- NN holdout accuracy: 0.803326
- NN holdout Brier: 0.139364
- NN holdout AUC: 0.899346
- Hub drops: 358
- Hub pull successes: 238
- Fallbacks: 2

| Seed | Single MAE | Single val MAE | Single guard MAE | Policy MAE | Policy guard MAE | Holdout-only MAE | Oracle MAE | Lift | Island | NN p | Regret | Fallback | Hub pulls | Outcome |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.157757 | 0.142474 | 0.166682 | 0.111837 | 0.109387 | 0.111837 | 0.111837 | 1.410598x | 0 | 0.943152 | 0 | no | 11/16 | distributed-lift |
| 2 | 0.768867 | 0.673535 | 0.839236 | 0.0817 | 0.084613 | 0.0817 | 0.0817 | 9.41086x | 2 | 0.642669 | 0 | no | 13/16 | distributed-lift |
| 3 | 0.247474 | 0.212591 | 0.246134 | 0.095176 | 0.090599 | 0.095176 | 0.095176 | 2.60016x | 3 | 0.890248 | 0 | no | 12/16 | distributed-lift |
| 4 | 0.62951 | 0.618636 | 0.631109 | 1.306e-11 | 1.310e-11 | 1.306e-11 | 1.306e-11 | 4.821e+10x | 2 | 0.88747 | 0 | no | 4/12 | distributed-lift |
| 5 | 0.570535 | 0.526547 | 0.562227 | 0.129019 | 0.13458 | 0.129019 | 0.129019 | 4.422096x | 2 | 0.797551 | 0 | no | 14/16 | distributed-lift |
| 6 | 0.528679 | 0.521143 | 0.529641 | 0.13486 | 0.164682 | 0.13486 | 0.13486 | 3.920217x | 1 | 0.955218 | 0 | no | 11/16 | distributed-lift |
| 7 | 0.779231 | 0.643735 | 1.059085 | 0.019205 | 0.018136 | 0.019205 | 0.019205 | 40.574119x | 0 | 0.931763 | 0 | no | 11/16 | distributed-lift |
| 8 | 0.386966 | 0.379033 | 0.413881 | 0.023768 | 0.024461 | 0.023768 | 0.023768 | 16.280792x | 2 | 0.791358 | 0 | no | 10/16 | distributed-lift |
| 9 | 0.60038 | 7.125e+28 | 0.599308 | 0.157757 | 0.166682 | 0.157757 | 0.157757 | 3.805723x | 1 | 0.901751 | 0 | no | 14/16 | distributed-lift |
| 10 | 0.188119 | 0.174087 | 0.195598 | 0.149472 | 0.157927 | 0.149472 | 0.149472 | 1.258553x | 2 | 0.720575 | 0 | validation-veto | 11/16 | distributed-lift |
| 11 | 0.270593 | 0.138341 | 0.332051 | 8.955e-8 | 8.973e-8 | 8.955e-8 | 8.955e-8 | 3.022e+6x | 3 | 0.803211 | 0 | no | 9/14 | distributed-lift |
| 12 | 0.36695 | 0.312266 | 0.391126 | 6.393e-12 | 6.416e-12 | 6.393e-12 | 6.393e-12 | 5.740e+10x | 2 | 0.321015 | 0 | low-nn-confidence | 3/12 | distributed-lift |
| 13 | 0.493314 | 0.412713 | 0.519703 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 1.655e+9x | 3 | 0.822405 | 0 | no | 3/12 | distributed-lift |
| 14 | 0.157756 | 0.142476 | 0.166675 | 0.041691 | 0.042687 | 0.041691 | 0.041691 | 3.783982x | 3 | 0.958572 | 0 | no | 14/16 | distributed-lift |
| 15 | 0.306881 | 0.285691 | 0.316855 | 0.127266 | 0.121901 | 0.127266 | 0.127266 | 2.411334x | 0 | 0.831906 | 0 | no | 12/16 | distributed-lift |
| 16 | 1.039911 | 0.115525 | 0.162025 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 3.489e+9x | 0 | 0.902294 | 0 | no | 7/12 | distributed-lift |
| 17 | 0.060859 | 0.045547 | 0.059957 | 0.0317 | 0.03224 | 0.0317 | 0.002668 | 1.919863x | 0 | 0.588982 | 0.029031 | no | 11/16 | distributed-lift |
| 18 | 0.186365 | 0.1244 | 0.13638 | 0.034865 | 0.03463 | 0.034865 | 0.034865 | 5.345343x | 1 | 0.856876 | 0 | no | 11/16 | distributed-lift |
| 19 | 0.11987 | 0.109808 | 1.223128 | 0.052684 | 0.107416 | 0.052684 | 0.052684 | 2.27524x | 3 | 0.754939 | 0 | no | 11/16 | distributed-lift |
| 20 | 0.090597 | 0.065265 | 0.099378 | 8.141e-8 | 8.161e-8 | 8.141e-8 | 8.141e-8 | 1.113e+6x | 2 | 0.830049 | 0 | no | 11/16 | distributed-lift |
| 21 | 0.35326 | 0.343066 | 0.362177 | 2.588e-11 | 2.600e-11 | 2.588e-11 | 2.588e-11 | 1.365e+10x | 2 | 0.862256 | 0 | no | 4/12 | distributed-lift |
| 22 | 0.148652 | 0.138504 | 0.147318 | 0.06125 | 0.118672 | 0.06125 | 0.06125 | 2.426978x | 0 | 0.880516 | 0 | no | 14/16 | distributed-lift |
| 23 | 0.13161 | 0.124415 | 0.13997 | 0.013468 | 0.013486 | 0.013468 | 0.013468 | 9.772332x | 1 | 0.777991 | 0 | no | 14/16 | distributed-lift |
| 24 | 47.199434 | 46.101695 | 45.954595 | 1.112e-10 | 1.173e-10 | 1.112e-10 | 1.112e-10 | 4.245e+11x | 3 | 0.584107 | 0 | no | 3/12 | distributed-lift |

