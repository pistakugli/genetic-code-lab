# Distributed worker-island + NN policy benchmark

Generated: 2026-06-03T07:20:16.025Z

## Scope

- Runs real `DistributedManager` worker-thread islands.
- Uses a shared `MigrationHub` with atomic migration counters.
- Uses the existing `MLP` as a scorer, then applies confidence, validation, and guard-holdout veto/fallback policy.
- This is the production-runtime proof lane; the wide seed sweep is part of `release:gate`.

## Summary

- Mean single-island MAE: 2.324315
- Mean policy MAE: 0.052202
- Mean robust holdout-only MAE: 0.052202
- Mean oracle-best MAE: 0.050938
- Median lift: 6.893487x
- Median lift on non-degenerate seeds: 6.893487x
- Success rate: 100%
- Non-degenerate success rate: 100%
- Non-regression rate vs single baseline: 100%
- Completed seeds: 24/24
- Runtime timeouts: 0
- Runtime errors: 0
- Single-baseline solved seeds: 0
- Single-baseline fallbacks: 0
- Mean regret vs oracle: 0.001264
- Policy beats robust holdout-only rate: 0%
- NN holdout accuracy: 0.799491
- NN holdout Brier: 0.151391
- NN holdout AUC: 0.839852
- Hub drops: 359
- Hub pull successes: 229
- Fallbacks: 4

| Seed | Single MAE | Single val MAE | Single guard MAE | Policy MAE | Policy guard MAE | Holdout-only MAE | Oracle MAE | Lift | Island | NN p | Regret | Fallback | Hub pulls | Outcome |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.157757 | 0.142474 | 0.166682 | 0.043202 | 0.043736 | 0.043202 | 0.043202 | 3.651596x | 0 | 0.941638 | 0 | no | 11/16 | distributed-lift |
| 2 | 0.768867 | 0.673535 | 0.839236 | 0.11476 | 0.119933 | 0.11476 | 0.11476 | 6.699806x | 2 | 0.706204 | 0 | no | 11/16 | distributed-lift |
| 3 | 0.247474 | 0.212591 | 0.246134 | 0.157759 | 0.166683 | 0.157759 | 0.157759 | 1.568684x | 1 | 0.624865 | 0 | no | 10/16 | distributed-lift |
| 4 | 0.62951 | 0.618636 | 0.631109 | 1.306e-11 | 1.310e-11 | 1.306e-11 | 1.306e-11 | 4.821e+10x | 2 | 0.876324 | 0 | no | 4/12 | distributed-lift |
| 5 | 0.570535 | 0.526547 | 0.562227 | 0.095618 | 0.095603 | 0.095618 | 0.095618 | 5.966801x | 2 | 0.780991 | 0 | no | 14/16 | distributed-lift |
| 6 | 0.528679 | 0.521143 | 0.529641 | 0.13222 | 0.15295 | 0.13222 | 0.13222 | 3.998479x | 3 | 0.879152 | 0 | no | 11/16 | distributed-lift |
| 7 | 0.779231 | 0.643735 | 1.059085 | 0.10995 | 0.111866 | 0.10995 | 0.10995 | 7.087168x | 2 | 0.803481 | 0 | no | 11/16 | distributed-lift |
| 8 | 0.386966 | 0.379033 | 0.413881 | 3.124e-5 | 3.166e-5 | 3.124e-5 | 3.124e-5 | 12388.456672x | 3 | 0.858138 | 0 | no | 11/16 | distributed-lift |
| 9 | 0.60038 | 7.125e+28 | 0.599308 | 0.157757 | 0.166682 | 0.157757 | 0.157757 | 3.805723x | 1 | 0.791802 | 0 | validation-veto | 11/16 | distributed-lift |
| 10 | 0.188119 | 0.174087 | 0.195598 | 0.14881 | 0.157227 | 0.14881 | 0.14881 | 1.264155x | 2 | 0.557219 | 0 | validation-veto | 10/16 | distributed-lift |
| 11 | 0.270593 | 0.138341 | 0.332051 | 1.059e-6 | 1.065e-6 | 1.059e-6 | 1.059e-6 | 2.556e+5x | 3 | 0.785525 | 0 | no | 11/16 | distributed-lift |
| 12 | 0.36695 | 0.312266 | 0.391126 | 6.393e-12 | 6.416e-12 | 6.393e-12 | 6.393e-12 | 5.740e+10x | 2 | 0.15346 | 0 | low-nn-confidence | 3/12 | distributed-lift |
| 13 | 0.493314 | 0.412713 | 0.519703 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 1.655e+9x | 3 | 0.917304 | 0 | no | 3/12 | distributed-lift |
| 14 | 0.157756 | 0.142476 | 0.166675 | 1.870e-9 | 1.906e-9 | 1.870e-9 | 1.870e-9 | 8.437e+7x | 0 | 0.860592 | 0 | no | 10/15 | distributed-lift |
| 15 | 0.306881 | 0.285691 | 0.316855 | 0.127029 | 0.121604 | 0.127029 | 0.127029 | 2.415833x | 0 | 0.81734 | 0 | no | 12/16 | distributed-lift |
| 16 | 1.039911 | 0.115525 | 0.162025 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 3.489e+9x | 0 | 0.906218 | 0 | no | 7/12 | distributed-lift |
| 17 | 0.060859 | 0.045547 | 0.059957 | 0.002668 | 0.05799 | 0.002668 | 0.002668 | 22.807557x | 3 | 0.74496 | 0 | no | 10/16 | distributed-lift |
| 18 | 0.186365 | 0.1244 | 0.13638 | 0.034865 | 0.03463 | 0.034865 | 0.034865 | 5.345343x | 1 | 0.825857 | 0 | no | 11/16 | distributed-lift |
| 19 | 0.11987 | 0.109808 | 1.223128 | 0.025678 | 0.082547 | 0.025678 | 0.025678 | 4.668223x | 3 | 0.926728 | 0 | no | 11/16 | distributed-lift |
| 20 | 0.090597 | 0.065265 | 0.099378 | 0.033016 | 0.032538 | 0.033016 | 0.002668 | 2.744029x | 3 | 0.626292 | 0.030348 | validation-veto | 11/16 | distributed-lift |
| 21 | 0.35326 | 0.343066 | 0.362177 | 2.588e-11 | 2.600e-11 | 2.588e-11 | 2.588e-11 | 1.365e+10x | 2 | 0.870479 | 0 | no | 4/12 | distributed-lift |
| 22 | 0.148652 | 0.138504 | 0.147318 | 0.056912 | 0.111508 | 0.056912 | 0.056912 | 2.611945x | 0 | 0.762116 | 0 | no | 14/16 | distributed-lift |
| 23 | 0.13161 | 0.124415 | 0.13997 | 0.012576 | 0.012344 | 0.012576 | 0.012576 | 10.46551x | 1 | 0.708272 | 0 | no | 10/16 | distributed-lift |
| 24 | 47.199434 | 46.101695 | 45.954595 | 1.112e-10 | 1.173e-10 | 1.112e-10 | 1.112e-10 | 4.245e+11x | 3 | 0.587613 | 0 | no | 8/12 | distributed-lift |

