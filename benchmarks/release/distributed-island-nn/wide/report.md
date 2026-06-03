# Distributed worker-island + NN policy benchmark

Generated: 2026-06-03T10:59:05.759Z

## Scope

- Runs real `DistributedManager` worker-thread islands.
- Uses a shared `MigrationHub` with atomic migration counters.
- Uses the existing `MLP` as a scorer, then applies confidence, validation, and guard-holdout veto/fallback policy.
- This is the production-runtime proof lane; the wide seed sweep is part of `release:gate`.

## Summary

- Mean single-island MAE: 2.324315
- Mean policy MAE: 0.056258
- Mean robust holdout-only MAE: 0.056258
- Mean oracle-best MAE: 0.050197
- Median lift: 8.238241x
- Median lift on non-degenerate seeds: 8.238241x
- Success rate: 100%
- Non-degenerate success rate: 100%
- Non-regression rate vs single baseline: 100%
- Completed seeds: 24/24
- Runtime timeouts: 0
- Runtime errors: 0
- Single-baseline solved seeds: 0
- Single-baseline fallbacks: 0
- Mean regret vs oracle: 0.006061
- Policy beats robust holdout-only rate: 0%
- NN holdout accuracy: 0.881659
- NN holdout Brier: 0.118369
- NN holdout AUC: 0.877543
- Hub drops: 353
- Hub pull successes: 228
- Fallbacks: 6

| Seed | Single MAE | Single val MAE | Single guard MAE | Policy MAE | Policy guard MAE | Holdout-only MAE | Oracle MAE | Lift | Island | NN p | Regret | Fallback | Hub pulls | Outcome |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.157757 | 0.142474 | 0.166682 | 0.047605 | 0.048984 | 0.047605 | 0.047605 | 3.313907x | 0 | 0.958733 | 0 | no | 11/16 | distributed-lift |
| 2 | 0.768867 | 0.673535 | 0.839236 | 0.114686 | 0.11993 | 0.114686 | 0.114686 | 6.704101x | 2 | 0.658992 | 0 | no | 13/16 | distributed-lift |
| 3 | 0.247474 | 0.212591 | 0.246134 | 0.157759 | 0.166683 | 0.157759 | 0.157759 | 1.568684x | 1 | 0.757673 | 0 | no | 10/16 | distributed-lift |
| 4 | 0.62951 | 0.618636 | 0.631109 | 1.306e-11 | 1.310e-11 | 1.306e-11 | 1.306e-11 | 4.821e+10x | 2 | 0.870093 | 0 | no | 4/12 | distributed-lift |
| 5 | 0.570535 | 0.526547 | 0.562227 | 0.042247 | 0.044221 | 0.042247 | 0.042247 | 13.504688x | 2 | 0.717672 | 0 | no | 14/16 | distributed-lift |
| 6 | 0.528679 | 0.521143 | 0.529641 | 0.266675 | 0.207058 | 0.266675 | 0.242918 | 1.982483x | 1 | 0.890299 | 0.023757 | no | 10/16 | distributed-lift |
| 7 | 0.779231 | 0.643735 | 1.059085 | 8.679e-12 | 8.271e-12 | 8.679e-12 | 8.679e-12 | 8.978e+10x | 3 | 0.921523 | 0 | no | 8/13 | distributed-lift |
| 8 | 0.386966 | 0.379033 | 0.413881 | 0.023811 | 0.024579 | 0.023811 | 0.023811 | 16.251256x | 2 | 0.538446 | 0 | low-nn-confidence | 11/16 | distributed-lift |
| 9 | 0.60038 | 7.125e+28 | 0.599308 | 0.157757 | 0.166682 | 0.157757 | 0.157757 | 3.805723x | 1 | 0.862411 | 0 | no | 11/16 | distributed-lift |
| 10 | 0.188119 | 0.174087 | 0.195598 | 0.14881 | 0.157227 | 0.14881 | 0.14881 | 1.264155x | 2 | 0.615749 | 0 | validation-veto | 12/16 | distributed-lift |
| 11 | 0.270593 | 0.138341 | 0.332051 | 8.955e-8 | 8.973e-8 | 8.955e-8 | 8.955e-8 | 3.022e+6x | 3 | 0.873353 | 0 | no | 9/14 | distributed-lift |
| 12 | 0.36695 | 0.312266 | 0.391126 | 6.393e-12 | 6.416e-12 | 6.393e-12 | 6.393e-12 | 5.740e+10x | 2 | 0.254158 | 0 | low-nn-confidence | 3/12 | distributed-lift |
| 13 | 0.493314 | 0.412713 | 0.519703 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 1.655e+9x | 3 | 0.155303 | 0 | low-nn-confidence | 8/12 | distributed-lift |
| 14 | 0.157756 | 0.142476 | 0.166675 | 1.406e-7 | 1.415e-7 | 1.406e-7 | 1.406e-7 | 1.122e+6x | 0 | 0.909104 | 0 | no | 9/14 | distributed-lift |
| 15 | 0.306881 | 0.285691 | 0.316855 | 0.126624 | 0.121031 | 0.126624 | 0.126624 | 2.423561x | 0 | 0.848005 | 0 | no | 12/16 | distributed-lift |
| 16 | 1.039911 | 0.115525 | 0.162025 | 2.981e-10 | 3.000e-10 | 2.981e-10 | 2.981e-10 | 3.489e+9x | 0 | 0.781002 | 0 | no | 8/12 | distributed-lift |
| 17 | 0.060859 | 0.045547 | 0.059957 | 0.041733 | 0.041422 | 0.041733 | 0.002667 | 1.458298x | 0 | 0.570239 | 0.039066 | low-nn-confidence | 10/16 | distributed-lift |
| 18 | 0.186365 | 0.1244 | 0.13638 | 0.034865 | 0.03463 | 0.034865 | 0.034865 | 5.345343x | 1 | 0.77262 | 0 | no | 12/16 | distributed-lift |
| 19 | 0.11987 | 0.109808 | 1.223128 | 0.097982 | 0.089287 | 0.097982 | 0.052819 | 1.223379x | 0 | 0.829645 | 0.045164 | no | 11/16 | distributed-lift |
| 20 | 0.090597 | 0.065265 | 0.099378 | 0.04015 | 0.040358 | 0.04015 | 0.002668 | 2.256431x | 3 | 0.478666 | 0.037482 | validation-veto | 10/16 | distributed-lift |
| 21 | 0.35326 | 0.343066 | 0.362177 | 2.588e-11 | 2.600e-11 | 2.588e-11 | 2.588e-11 | 1.365e+10x | 2 | 0.823316 | 0 | no | 3/12 | distributed-lift |
| 22 | 0.148652 | 0.138504 | 0.147318 | 0.036018 | 0.095834 | 0.036018 | 0.036018 | 4.127147x | 0 | 0.867978 | 0 | no | 11/16 | distributed-lift |
| 23 | 0.13161 | 0.124415 | 0.13997 | 0.013468 | 0.013486 | 0.013468 | 0.013468 | 9.77238x | 1 | 0.90717 | 0 | no | 10/16 | distributed-lift |
| 24 | 47.199434 | 46.101695 | 45.954595 | 1.112e-10 | 1.173e-10 | 1.112e-10 | 1.112e-10 | 4.245e+11x | 3 | 0.580976 | 0 | no | 8/12 | distributed-lift |

