# Distributed worker-island + NN policy benchmark

Generated: 2026-06-02T07:13:10.693Z

## Scope

- Runs real `DistributedManager` worker-thread islands.
- Uses a shared `MigrationHub` with atomic migration counters.
- Uses the existing `MLP` as a scorer, then applies confidence, validation, and guard-holdout veto/fallback policy.
- This is the production-runtime proof lane; the wide seed sweep is part of `release:gate`.

## Summary

- Mean single-island MAE: 2.324315
- Mean policy MAE: 0.048474
- Mean robust holdout-only MAE: 0.048975
- Mean oracle-best MAE: 0.046695
- Median lift: 8.240225x
- Median lift on non-degenerate seeds: 8.240225x
- Success rate: 100%
- Non-degenerate success rate: 100%
- Non-regression rate vs single baseline: 100%
- Completed seeds: 24/24
- Runtime timeouts: 0
- Runtime errors: 0
- Single-baseline solved seeds: 0
- Single-baseline fallbacks: 0
- Mean regret vs oracle: 0.001779
- Policy beats robust holdout-only rate: 4%
- NN holdout accuracy: 0.849277
- NN holdout Brier: 0.123987
- NN holdout AUC: 0.844415
- Hub drops: 356
- Hub pull successes: 234
- Fallbacks: 2

| Seed | Single MAE | Single val MAE | Single guard MAE | Policy MAE | Policy guard MAE | Holdout-only MAE | Oracle MAE | Lift | Island | NN p | Regret | Fallback | Hub pulls | Outcome |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.157757 | 0.142474 | 0.166682 | 0.048054 | 0.050705 | 0.048054 | 0.048054 | 3.282926x | 0 | 0.942053 | 0 | no | 12/16 | distributed-lift |
| 2 | 0.768867 | 0.673535 | 0.839236 | 0.114619 | 0.119837 | 0.114619 | 0.114619 | 6.708027x | 2 | 0.638353 | 0 | no | 14/16 | distributed-lift |
| 3 | 0.247474 | 0.212591 | 0.246134 | 0.157759 | 0.166683 | 0.157759 | 0.157759 | 1.568684x | 1 | 0.643489 | 0 | no | 12/16 | distributed-lift |
| 4 | 0.62951 | 0.618636 | 0.631109 | 1.306e-11 | 1.310e-11 | 1.306e-11 | 1.306e-11 | 4.821e+10x | 2 | 0.876503 | 0 | no | 4/12 | distributed-lift |
| 5 | 0.570535 | 0.526547 | 0.562227 | 0.054871 | 0.054178 | 0.054871 | 0.054871 | 10.397836x | 2 | 0.915777 | 0 | no | 11/16 | distributed-lift |
| 6 | 0.528679 | 0.521143 | 0.529641 | 0.210925 | 0.211237 | 0.222937 | 0.210925 | 2.506482x | 3 | 0.927357 | 0 | no | 10/16 | distributed-lift |
| 7 | 0.779231 | 0.643735 | 1.059085 | 0.031965 | 0.029783 | 0.031965 | 0.031965 | 24.377834x | 0 | 0.93794 | 0 | no | 11/16 | distributed-lift |
| 8 | 0.386966 | 0.379033 | 0.413881 | 0.023801 | 0.024533 | 0.023801 | 0.023801 | 16.258252x | 2 | 0.49276 | 0 | low-nn-confidence | 10/16 | distributed-lift |
| 9 | 0.60038 | 7.125e+28 | 0.599308 | 0.157757 | 0.166682 | 0.157757 | 0.157757 | 3.805723x | 1 | 0.800849 | 0 | no | 11/16 | distributed-lift |
| 10 | 0.188119 | 0.174087 | 0.195598 | 0.052994 | 0.055989 | 0.052994 | 0.052994 | 3.549816x | 2 | 0.710564 | 0 | no | 11/16 | distributed-lift |
| 11 | 0.270593 | 0.138341 | 0.332051 | 1.406e-7 | 1.415e-7 | 1.406e-7 | 1.406e-7 | 1.924e+6x | 3 | 0.888774 | 0 | no | 8/14 | distributed-lift |
| 12 | 0.36695 | 0.312266 | 0.391126 | 6.393e-12 | 6.416e-12 | 6.393e-12 | 6.393e-12 | 5.740e+10x | 2 | 0.508011 | 0 | low-nn-confidence | 7/12 | distributed-lift |
| 13 | 0.493314 | 0.412713 | 0.519703 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 1.655e+9x | 3 | 0.91235 | 0 | no | 4/12 | distributed-lift |
| 14 | 0.157756 | 0.142476 | 0.166675 | 1.406e-7 | 1.415e-7 | 1.406e-7 | 1.406e-7 | 1.122e+6x | 0 | 0.852993 | 0 | no | 9/14 | distributed-lift |
| 15 | 0.306881 | 0.285691 | 0.316855 | 0.147454 | 0.140745 | 0.147454 | 0.147454 | 2.081198x | 0 | 0.777195 | 0 | no | 12/16 | distributed-lift |
| 16 | 1.039911 | 0.115525 | 0.162025 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 3.489e+9x | 0 | 0.886438 | 0 | no | 7/12 | distributed-lift |
| 17 | 0.060859 | 0.045547 | 0.059957 | 0.02243 | 0.021718 | 0.02243 | 0.002668 | 2.713226x | 0 | 0.636403 | 0.019762 | no | 13/16 | distributed-lift |
| 18 | 0.186365 | 0.1244 | 0.13638 | 0.034865 | 0.03463 | 0.034865 | 0.034865 | 5.345343x | 1 | 0.873205 | 0 | no | 11/16 | distributed-lift |
| 19 | 0.11987 | 0.109808 | 1.223128 | 0.020979 | 0.077657 | 0.020979 | 0.020979 | 5.713806x | 3 | 0.947824 | 0 | no | 14/16 | distributed-lift |
| 20 | 0.090597 | 0.065265 | 0.099378 | 0.025598 | 0.026893 | 0.025598 | 0.002668 | 3.53924x | 2 | 0.768287 | 0.02293 | no | 11/16 | distributed-lift |
| 21 | 0.35326 | 0.343066 | 0.362177 | 2.588e-11 | 2.600e-11 | 2.588e-11 | 2.588e-11 | 1.365e+10x | 2 | 0.884102 | 0 | no | 4/12 | distributed-lift |
| 22 | 0.148652 | 0.138504 | 0.147318 | 0.045846 | 0.103575 | 0.045846 | 0.045846 | 3.242442x | 0 | 0.895756 | 0 | no | 10/16 | distributed-lift |
| 23 | 0.13161 | 0.124415 | 0.13997 | 0.013467 | 0.013486 | 0.013467 | 0.013467 | 9.772423x | 1 | 0.761311 | 0 | no | 15/16 | distributed-lift |
| 24 | 47.199434 | 46.101695 | 45.954595 | 1.112e-10 | 1.173e-10 | 1.112e-10 | 1.112e-10 | 4.245e+11x | 3 | 0.687161 | 0 | no | 3/12 | distributed-lift |

