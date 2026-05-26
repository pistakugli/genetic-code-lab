# Distributed worker-island + NN policy benchmark

Generated: 2026-05-26T10:52:54.339Z

## Scope

- Runs real `DistributedManager` worker-thread islands.
- Uses a shared `MigrationHub` with atomic migration counters.
- Uses the existing `MLP` as a scorer, then applies confidence, validation, and guard-holdout veto/fallback policy.
- This is the production-runtime proof lane; the wide seed sweep is part of `release:gate`.

## Summary

- Mean single-island MAE: 2.324315
- Mean policy MAE: 0.055926
- Mean robust holdout-only MAE: 0.055926
- Mean oracle-best MAE: 0.052318
- Median lift: 8.34611x
- Median lift on non-degenerate seeds: 8.34611x
- Success rate: 100%
- Non-degenerate success rate: 100%
- Non-regression rate vs single baseline: 100%
- Completed seeds: 24/24
- Runtime timeouts: 0
- Runtime errors: 0
- Single-baseline solved seeds: 0
- Single-baseline fallbacks: 0
- Mean regret vs oracle: 0.003608
- Policy beats robust holdout-only rate: 0%
- NN holdout accuracy: 0.778548
- NN holdout Brier: 0.142572
- NN holdout AUC: 0.855275
- Hub drops: 357
- Hub pull successes: 223
- Fallbacks: 1

| Seed | Single MAE | Single val MAE | Single guard MAE | Policy MAE | Policy guard MAE | Holdout-only MAE | Oracle MAE | Lift | Island | NN p | Regret | Fallback | Hub pulls | Outcome |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.157757 | 0.142474 | 0.166682 | 0.110021 | 0.107215 | 0.110021 | 0.110021 | 1.433887x | 0 | 0.962774 | 0 | no | 11/16 | distributed-lift |
| 2 | 0.768867 | 0.673535 | 0.839236 | 0.114717 | 0.119866 | 0.114717 | 0.114717 | 6.702275x | 2 | 0.827326 | 0 | no | 13/16 | distributed-lift |
| 3 | 0.247474 | 0.212591 | 0.246134 | 0.157759 | 0.166683 | 0.157759 | 0.157759 | 1.568684x | 1 | 0.707859 | 0 | no | 11/16 | distributed-lift |
| 4 | 0.62951 | 0.618636 | 0.631109 | 1.306e-11 | 1.310e-11 | 1.306e-11 | 1.306e-11 | 4.821e+10x | 2 | 0.811504 | 0 | no | 3/12 | distributed-lift |
| 5 | 0.570535 | 0.526547 | 0.562227 | 0.123172 | 0.131035 | 0.123172 | 0.123172 | 4.632004x | 2 | 0.901728 | 0 | no | 14/16 | distributed-lift |
| 6 | 0.528679 | 0.521143 | 0.529641 | 0.201128 | 0.203299 | 0.201128 | 0.15807 | 2.628575x | 3 | 0.806238 | 0.043057 | no | 9/16 | distributed-lift |
| 7 | 0.779231 | 0.643735 | 1.059085 | 0.000578 | 0.000599 | 0.000578 | 0.000578 | 1347.347888x | 0 | 0.90722 | 0 | no | 10/16 | distributed-lift |
| 8 | 0.386966 | 0.379033 | 0.413881 | 0.023742 | 0.024474 | 0.023742 | 0.023742 | 16.29864x | 2 | 0.564744 | 0 | low-nn-confidence | 11/16 | distributed-lift |
| 9 | 0.60038 | 7.125e+28 | 0.599308 | 0.157757 | 0.166682 | 0.157757 | 0.157757 | 3.805723x | 1 | 0.641676 | 0 | no | 11/16 | distributed-lift |
| 10 | 0.188119 | 0.174087 | 0.195598 | 0.116729 | 0.12515 | 0.116729 | 0.116729 | 1.611581x | 3 | 0.924234 | 0 | no | 12/16 | distributed-lift |
| 11 | 0.270593 | 0.138341 | 0.332051 | 3.292e-14 | 3.315e-14 | 3.292e-14 | 3.292e-14 | 2.706e+11x | 3 | 0.879141 | 0 | no | 10/15 | distributed-lift |
| 12 | 0.36695 | 0.312266 | 0.391126 | 6.393e-12 | 6.416e-12 | 6.393e-12 | 6.393e-12 | 5.740e+10x | 2 | 0.742598 | 0 | no | 3/12 | distributed-lift |
| 13 | 0.493314 | 0.412713 | 0.519703 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 1.655e+9x | 3 | 0.939793 | 0 | no | 4/12 | distributed-lift |
| 14 | 0.157756 | 0.142476 | 0.166675 | 1.406e-7 | 1.415e-7 | 1.406e-7 | 1.406e-7 | 1.122e+6x | 0 | 0.936862 | 0 | no | 9/14 | distributed-lift |
| 15 | 0.306881 | 0.285691 | 0.316855 | 0.124614 | 0.118848 | 0.124614 | 0.124614 | 2.46265x | 0 | 0.810776 | 0 | no | 10/16 | distributed-lift |
| 16 | 1.039911 | 0.115525 | 0.162025 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 3.489e+9x | 0 | 0.88162 | 0 | no | 7/12 | distributed-lift |
| 17 | 0.060859 | 0.045547 | 0.059957 | 0.046192 | 0.04328 | 0.046192 | 0.002653 | 1.317514x | 0 | 0.621973 | 0.04354 | no | 14/16 | distributed-lift |
| 18 | 0.186365 | 0.1244 | 0.13638 | 0.034865 | 0.03463 | 0.034865 | 0.034865 | 5.345343x | 1 | 0.839615 | 0 | no | 11/16 | distributed-lift |
| 19 | 0.11987 | 0.109808 | 1.223128 | 0.052755 | 0.112518 | 0.052755 | 0.052755 | 2.272199x | 3 | 0.910769 | 0 | no | 11/16 | distributed-lift |
| 20 | 0.090597 | 0.065265 | 0.099378 | 0.002668 | 0.057991 | 0.002668 | 0.002668 | 33.957754x | 0 | 0.821221 | 0 | no | 11/16 | distributed-lift |
| 21 | 0.35326 | 0.343066 | 0.362177 | 2.588e-11 | 2.600e-11 | 2.588e-11 | 2.588e-11 | 1.365e+10x | 2 | 0.847819 | 0 | no | 4/12 | distributed-lift |
| 22 | 0.148652 | 0.138504 | 0.147318 | 0.062353 | 0.114368 | 0.062353 | 0.062353 | 2.384041x | 0 | 0.601582 | 0 | no | 10/16 | distributed-lift |
| 23 | 0.13161 | 0.124415 | 0.13997 | 0.013174 | 0.013061 | 0.013174 | 0.013174 | 9.989946x | 1 | 0.803245 | 0 | no | 11/16 | distributed-lift |
| 24 | 47.199434 | 46.101695 | 45.954595 | 1.112e-10 | 1.173e-10 | 1.112e-10 | 1.112e-10 | 4.245e+11x | 3 | 0.760303 | 0 | no | 3/12 | distributed-lift |

