# Distributed worker-island + NN policy benchmark

Generated: 2026-06-01T16:53:19.566Z

## Scope

- Runs real `DistributedManager` worker-thread islands.
- Uses a shared `MigrationHub` with atomic migration counters.
- Uses the existing `MLP` as a scorer, then applies confidence, validation, and guard-holdout veto/fallback policy.
- This is the production-runtime proof lane; the wide seed sweep is part of `release:gate`.

## Summary

- Mean single-island MAE: 2.324315
- Mean policy MAE: 0.054459
- Mean robust holdout-only MAE: 0.054459
- Mean oracle-best MAE: 0.052289
- Median lift: 6.65786x
- Median lift on non-degenerate seeds: 6.65786x
- Success rate: 100%
- Non-degenerate success rate: 100%
- Non-regression rate vs single baseline: 100%
- Completed seeds: 24/24
- Runtime timeouts: 0
- Runtime errors: 0
- Single-baseline solved seeds: 0
- Single-baseline fallbacks: 0
- Mean regret vs oracle: 0.00217
- Policy beats robust holdout-only rate: 0%
- NN holdout accuracy: 0.831462
- NN holdout Brier: 0.137243
- NN holdout AUC: 0.832254
- Hub drops: 360
- Hub pull successes: 246
- Fallbacks: 4

| Seed | Single MAE | Single val MAE | Single guard MAE | Policy MAE | Policy guard MAE | Holdout-only MAE | Oracle MAE | Lift | Island | NN p | Regret | Fallback | Hub pulls | Outcome |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.157757 | 0.142474 | 0.166682 | 0.043213 | 0.044677 | 0.043213 | 0.043213 | 3.650653x | 0 | 0.963544 | 0 | no | 12/16 | distributed-lift |
| 2 | 0.768867 | 0.673535 | 0.839236 | 0.120169 | 0.125159 | 0.120169 | 0.120169 | 6.398205x | 2 | 0.436911 | 0 | low-nn-confidence | 14/16 | distributed-lift |
| 3 | 0.247474 | 0.212591 | 0.246134 | 0.157759 | 0.166683 | 0.157759 | 0.157759 | 1.568684x | 1 | 0.783347 | 0 | no | 14/16 | distributed-lift |
| 4 | 0.62951 | 0.618636 | 0.631109 | 1.306e-11 | 1.310e-11 | 1.306e-11 | 1.306e-11 | 4.821e+10x | 2 | 0.880593 | 0 | no | 4/12 | distributed-lift |
| 5 | 0.570535 | 0.526547 | 0.562227 | 0.013221 | 0.013474 | 0.013221 | 0.013221 | 43.154382x | 1 | 0.897626 | 0 | no | 11/16 | distributed-lift |
| 6 | 0.528679 | 0.521143 | 0.529641 | 0.157763 | 0.196316 | 0.157763 | 0.157763 | 3.351104x | 1 | 0.935024 | 0 | no | 11/16 | distributed-lift |
| 7 | 0.779231 | 0.643735 | 1.059085 | 0.068714 | 0.070773 | 0.068714 | 0.068714 | 11.340175x | 0 | 0.930852 | 0 | no | 15/16 | distributed-lift |
| 8 | 0.386966 | 0.379033 | 0.413881 | 0.022582 | 0.024483 | 0.022582 | 0.022582 | 17.136195x | 2 | 0.664346 | 0 | no | 11/16 | distributed-lift |
| 9 | 0.60038 | 7.125e+28 | 0.599308 | 0.157757 | 0.166682 | 0.157757 | 0.157757 | 3.805723x | 1 | 0.767682 | 0 | no | 10/16 | distributed-lift |
| 10 | 0.188119 | 0.174087 | 0.195598 | 0.086571 | 0.08973 | 0.086571 | 0.086571 | 2.172997x | 2 | 0.591652 | 0 | no | 11/16 | distributed-lift |
| 11 | 0.270593 | 0.138341 | 0.332051 | 0.142447 | 0.148708 | 0.142447 | 0.142447 | 1.899603x | 1 | 0.675403 | 0 | no | 11/16 | distributed-lift |
| 12 | 0.36695 | 0.312266 | 0.391126 | 6.393e-12 | 6.416e-12 | 6.393e-12 | 6.393e-12 | 5.740e+10x | 2 | 0.221773 | 0 | low-nn-confidence | 3/12 | distributed-lift |
| 13 | 0.493314 | 0.412713 | 0.519703 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 1.655e+9x | 3 | 0.921095 | 0 | no | 3/12 | distributed-lift |
| 14 | 0.157756 | 0.142476 | 0.166675 | 0.046343 | 0.046505 | 0.046343 | 0.046343 | 3.404124x | 3 | 0.918438 | 0 | no | 15/16 | distributed-lift |
| 15 | 0.306881 | 0.285691 | 0.316855 | 0.127106 | 0.121598 | 0.127106 | 0.127106 | 2.414372x | 0 | 0.801201 | 0 | no | 12/16 | distributed-lift |
| 16 | 1.039911 | 0.115525 | 0.162025 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 3.489e+9x | 0 | 0.16657 | 0 | low-nn-confidence | 7/12 | distributed-lift |
| 17 | 0.060859 | 0.045547 | 0.059957 | 0.044148 | 0.045862 | 0.044148 | 0.002668 | 1.378507x | 0 | 0.570694 | 0.04148 | low-nn-confidence | 10/16 | distributed-lift |
| 18 | 0.186365 | 0.1244 | 0.13638 | 0.034865 | 0.03463 | 0.034865 | 0.034865 | 5.345343x | 1 | 0.799299 | 0 | no | 11/16 | distributed-lift |
| 19 | 0.11987 | 0.109808 | 1.223128 | 0.014356 | 0.069648 | 0.014356 | 0.014356 | 8.349534x | 3 | 0.92722 | 0 | no | 11/16 | distributed-lift |
| 20 | 0.090597 | 0.065265 | 0.099378 | 0.013097 | 0.012298 | 0.013097 | 0.00249 | 6.917515x | 3 | 0.661363 | 0.010607 | no | 11/16 | distributed-lift |
| 21 | 0.35326 | 0.343066 | 0.362177 | 2.588e-11 | 2.600e-11 | 2.588e-11 | 2.588e-11 | 1.365e+10x | 2 | 0.765998 | 0 | no | 4/12 | distributed-lift |
| 22 | 0.148652 | 0.138504 | 0.147318 | 0.056912 | 0.111508 | 0.056912 | 0.056912 | 2.611945x | 0 | 0.894677 | 0 | no | 14/16 | distributed-lift |
| 23 | 0.13161 | 0.124415 | 0.13997 | 1.156e-10 | 1.164e-10 | 1.156e-10 | 1.156e-10 | 1.138e+9x | 2 | 0.810856 | 0 | no | 14/16 | distributed-lift |
| 24 | 47.199434 | 46.101695 | 45.954595 | 1.112e-10 | 1.173e-10 | 1.112e-10 | 1.112e-10 | 4.245e+11x | 3 | 0.783102 | 0 | no | 7/12 | distributed-lift |

