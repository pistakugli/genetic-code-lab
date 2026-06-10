# Distributed worker-island + NN policy benchmark

Generated: 2026-06-10T19:18:32.066Z

## Scope

- Runs real `DistributedManager` worker-thread islands.
- Uses a shared `MigrationHub` with atomic migration counters.
- Uses the existing `MLP` as a scorer, then applies confidence, validation, and guard-holdout veto/fallback policy.
- This is the production-runtime proof lane; the wide seed sweep is part of `release:gate`.

## Summary

- Mean single-island MAE: 2.324315
- Mean policy MAE: 0.050999
- Mean robust holdout-only MAE: 0.050999
- Mean oracle-best MAE: 0.04966
- Median lift: 11.479105x
- Median lift on non-degenerate seeds: 11.479105x
- Success rate: 100%
- Non-degenerate success rate: 100%
- Non-regression rate vs single baseline: 100%
- Completed seeds: 24/24
- Runtime timeouts: 0
- Runtime errors: 0
- Single-baseline solved seeds: 0
- Single-baseline fallbacks: 0
- Mean regret vs oracle: 0.001339
- Policy beats robust holdout-only rate: 0%
- NN holdout accuracy: 0.824819
- NN holdout Brier: 0.143166
- NN holdout AUC: 0.862686
- Hub drops: 357
- Hub pull successes: 220
- Fallbacks: 3

| Seed | Single MAE | Single val MAE | Single guard MAE | Policy MAE | Policy guard MAE | Holdout-only MAE | Oracle MAE | Lift | Island | NN p | Regret | Fallback | Hub pulls | Outcome |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.157757 | 0.142474 | 0.166682 | 0.03638 | 0.03776 | 0.03638 | 0.03638 | 4.336426x | 0 | 0.901435 | 0 | no | 11/16 | distributed-lift |
| 2 | 0.768867 | 0.673535 | 0.839236 | 0.114757 | 0.120064 | 0.114757 | 0.114757 | 6.699959x | 2 | 0.670707 | 0 | no | 10/16 | distributed-lift |
| 3 | 0.247474 | 0.212591 | 0.246134 | 0.157759 | 0.166683 | 0.157759 | 0.157759 | 1.568684x | 1 | 0.785561 | 0 | no | 11/16 | distributed-lift |
| 4 | 0.62951 | 0.618636 | 0.631109 | 1.306e-11 | 1.310e-11 | 1.306e-11 | 1.306e-11 | 4.821e+10x | 2 | 0.885375 | 0 | no | 4/12 | distributed-lift |
| 5 | 0.570535 | 0.526547 | 0.562227 | 0.090044 | 0.094517 | 0.090044 | 0.090044 | 6.336163x | 2 | 0.893059 | 0 | no | 14/16 | distributed-lift |
| 6 | 0.528679 | 0.521143 | 0.529641 | 0.081125 | 0.134441 | 0.081125 | 0.081125 | 6.516845x | 3 | 0.904338 | 0 | no | 10/16 | distributed-lift |
| 7 | 0.779231 | 0.643735 | 1.059085 | 0.016259 | 0.01636 | 0.016259 | 0.016259 | 47.927472x | 0 | 0.928258 | 0 | no | 10/16 | distributed-lift |
| 8 | 0.386966 | 0.379033 | 0.413881 | 0.023801 | 0.024533 | 0.023801 | 0.023801 | 16.258252x | 2 | 0.60303 | 0 | no | 11/16 | distributed-lift |
| 9 | 0.60038 | 7.125e+28 | 0.599308 | 0.157757 | 0.166682 | 0.157757 | 0.157757 | 3.805723x | 1 | 0.729936 | 0 | no | 11/16 | distributed-lift |
| 10 | 0.188119 | 0.174087 | 0.195598 | 0.084677 | 0.087726 | 0.084677 | 0.084677 | 2.22161x | 2 | 0.77105 | 0 | no | 11/16 | distributed-lift |
| 11 | 0.270593 | 0.138341 | 0.332051 | 0.000307 | 0.000301 | 0.000307 | 0.000307 | 880.038085x | 3 | 0.938101 | 0 | no | 11/16 | distributed-lift |
| 12 | 0.36695 | 0.312266 | 0.391126 | 6.393e-12 | 6.416e-12 | 6.393e-12 | 6.393e-12 | 5.740e+10x | 2 | 0.274628 | 0 | low-nn-confidence | 3/12 | distributed-lift |
| 13 | 0.493314 | 0.412713 | 0.519703 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 1.655e+9x | 3 | 0.927432 | 0 | no | 3/12 | distributed-lift |
| 14 | 0.157756 | 0.142476 | 0.166675 | 8.955e-8 | 8.973e-8 | 8.955e-8 | 8.955e-8 | 1.762e+6x | 0 | 0.886823 | 0 | no | 9/14 | distributed-lift |
| 15 | 0.306881 | 0.285691 | 0.316855 | 0.289021 | 0.291921 | 0.289021 | 0.289021 | 1.061796x | 2 | 0.64543 | 0 | validation-veto | 11/16 | distributed-lift |
| 16 | 1.039911 | 0.115525 | 0.162025 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 3.489e+9x | 0 | 0.359385 | 0 | low-nn-confidence | 8/12 | distributed-lift |
| 17 | 0.060859 | 0.045547 | 0.059957 | 0.034806 | 0.034964 | 0.034806 | 0.002668 | 1.748526x | 0 | 0.616181 | 0.032138 | no | 10/16 | distributed-lift |
| 18 | 0.186365 | 0.1244 | 0.13638 | 0.034865 | 0.03463 | 0.034865 | 0.034865 | 5.345343x | 1 | 0.724387 | 0 | no | 10/16 | distributed-lift |
| 19 | 0.11987 | 0.109808 | 1.223128 | 0.064304 | 0.117967 | 0.064304 | 0.064304 | 1.864108x | 3 | 0.929458 | 0 | no | 11/16 | distributed-lift |
| 20 | 0.090597 | 0.065265 | 0.099378 | 0.002668 | 0.05799 | 0.002668 | 0.002668 | 33.952061x | 0 | 0.925321 | 0 | no | 11/16 | distributed-lift |
| 21 | 0.35326 | 0.343066 | 0.362177 | 2.588e-11 | 2.600e-11 | 2.588e-11 | 2.588e-11 | 1.365e+10x | 2 | 0.894507 | 0 | no | 4/12 | distributed-lift |
| 22 | 0.148652 | 0.138504 | 0.147318 | 0.035438 | 0.092768 | 0.035438 | 0.035438 | 4.194754x | 0 | 0.660485 | 0 | no | 10/16 | distributed-lift |
| 23 | 0.13161 | 0.124415 | 0.13997 | 1.368e-10 | 1.385e-10 | 1.368e-10 | 1.368e-10 | 9.623e+8x | 0 | 0.851637 | 0 | no | 13/15 | distributed-lift |
| 24 | 47.199434 | 46.101695 | 45.954595 | 1.112e-10 | 1.173e-10 | 1.112e-10 | 1.112e-10 | 4.245e+11x | 3 | 0.788104 | 0 | no | 3/12 | distributed-lift |

