# Multi-island + NN archive benchmark

Generated: 2026-06-01T11:39:49.582Z

## Scope

- Runs four real `EvolutionEngine` islands against a shared `MigrationHub`.
- Uses different island grammar roles: full, arithmetic, trig, and rational-trig.
- Uses the existing `MLP` to learn from validation labels over island champion history; validation error itself is not an input feature.
- Applies robust validation/guard-holdout selection and falls back to single baseline on guard regressions.
- The wide seed sweep is eligible for `release:gate` only when timeout/error, non-regression, and non-degenerate lift gates pass.

## Summary

- Mean single-island MAE: 2.305416
- Mean NN-selected island MAE: 0.082111
- Mean robust holdout-only MAE: 0.086682
- Mean lift vs single: 1.986e+10x
- Median lift vs single: 7.805164x
- Median eligible lift vs single: 7.805164x
- Success rate: 88%
- Non-degenerate success rate: 88%
- Non-regression rate vs single baseline: 100%
- Completed seeds: 24/24
- Runtime timeouts: 0
- Runtime errors: 0
- Single-baseline solved seeds: 0
- Single-baseline fallbacks: 3
- Archive injections: 255
- Hub pull successes: 360
- Fallbacks: 13
- Mean NN training accuracy: 0.857456

| Seed | Single MAE | Single val MAE | Single guard MAE | NN MAE | NN guard MAE | Holdout-only MAE | Lift | NN island | NN p | Archive injections | NN acc | Fallback | Outcome | Success |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | 0.157757 | 0.142474 | 0.166682 | 1.873e-11 | 1.880e-11 | 1.873e-11 | 8.422e+9x | 3 | 0.69654 | 19 | 0.842105 | no | island-nn-lift | yes |
| 2 | 0.768867 | 0.673535 | 0.839236 | 6.883e-12 | 6.876e-12 | 6.883e-12 | 1.117e+11x | 0 | 0.798503 | 17 | 0.973684 | no | island-nn-lift | yes |
| 3 | 0.247474 | 0.212591 | 0.246134 | 4.619e-12 | 4.615e-12 | 4.619e-12 | 5.358e+10x | 2 | 0.693308 | 17 | 0.960526 | no | island-nn-lift | yes |
| 4 | 0.62951 | 0.618636 | 0.631109 | 0.222917 | 0.229967 | 0.222917 | 2.82396x | 3 | 0.821822 | 4 | 0.894737 | validation-veto | island-nn-lift | yes |
| 5 | 0.570653 | 0.525361 | 0.561473 | 0.03727 | 0.052877 | 0.03727 | 15.31148x | 3 | 0.785088 | 5 | 0.907895 | no | island-nn-lift | yes |
| 6 | 0.528671 | 0.521143 | 0.52963 | 1.748e-12 | 1.776e-12 | 1.748e-12 | 3.024e+11x | 0 | 0.782675 | 7 | 0.881579 | no | island-nn-lift | yes |
| 7 | 0.779231 | 0.643735 | 1.059085 | 0.157761 | 0.166682 | 0.157761 | 4.939327x | 1 | 0.531764 | 6 | 0.75 | low-nn-confidence | island-nn-lift | yes |
| 8 | 0.386989 | 0.379161 | 0.413863 | 0.019093 | 0.016987 | 0.019093 | 20.268099x | 3 | 0.547459 | 11 | 0.697368 | low-nn-confidence | island-nn-lift | yes |
| 9 | 0.60038 | 7.125e+28 | 0.599308 | 0.130359 | 0.134466 | 0.130359 | 4.605592x | 2 | 0.519305 | 16 | 0.828947 | low-nn-confidence | island-nn-lift | yes |
| 10 | 0.188204 | 0.174163 | 0.195625 | 0.169209 | 0.217478 | 0.169209 | 1.112258x | 0 | 0.702696 | 4 | 0.855263 | no | island-nn-lift | yes |
| 11 | 0.270604 | 0.138345 | 0.33208 | 2.650e-5 | 2.578e-5 | 2.650e-5 | 10213.096579x | 3 | 0.708078 | 11 | 0.855263 | validation-veto | island-nn-lift | yes |
| 12 | 0.36695 | 0.312266 | 0.391126 | 0.034388 | 0.032446 | 0.034388 | 10.671002x | 2 | 0.616839 | 6 | 0.789474 | validation-veto | island-nn-lift | yes |
| 13 | 0.531932 | 0.474049 | 0.571011 | 0.15776 | 0.166682 | 0.15776 | 3.371767x | 1 | 0.842172 | 23 | 0.934211 | validation-veto | island-nn-lift | yes |
| 14 | 0.157756 | 0.142476 | 0.166675 | 0.012645 | 0.0466 | 0.012645 | 12.475462x | 1 | 0.758452 | 8 | 0.894737 | validation-veto | island-nn-lift | yes |
| 15 | 0.306881 | 0.285691 | 0.316855 | 0.157757 | 0.166682 | 0.157757 | 1.945275x | 1 | 0.664706 | 10 | 0.789474 | no | island-nn-lift | yes |
| 16 | 0.54185 | 0.115079 | 0.16232 | 0.003947 | 0.004104 | 0.003947 | 137.286143x | 2 | 0.631257 | 7 | 0.776316 | validation-veto | island-nn-lift | yes |
| 17 | 0.066033 | 0.050389 | 0.066556 | 0.066033 | 0.066556 | 0.132941 | 1x | -2 | 0.685435 | 16 | 0.723684 | single-baseline-guard-veto | single-baseline-fallback | no |
| 18 | 0.186029 | 0.12368 | 0.136026 | 0.07219 | 0.076472 | 0.07219 | 2.576946x | 0 | 0.892745 | 11 | 0.934211 | no | island-nn-lift | yes |
| 19 | 0.11987 | 0.109808 | 1.223128 | 0.095207 | 0.095752 | 0.095207 | 1.259042x | 0 | 0.729241 | 5 | 0.815789 | no | island-nn-lift | yes |
| 20 | 0.091352 | 1.035e+10 | 0.099841 | 2.210e-10 | 2.136e-10 | 2.210e-10 | 4.134e+8x | 0 | 0.731313 | 9 | 0.986842 | no | island-nn-lift | yes |
| 21 | 0.35326 | 0.343066 | 0.362177 | 0.196045 | 0.217613 | 0.196045 | 1.801935x | 0 | 0.713202 | 5 | 0.868421 | no | island-nn-lift | yes |
| 22 | 0.148652 | 0.138504 | 0.147318 | 0.148652 | 0.147318 | 0.181985 | 1x | -2 | 0.686911 | 16 | 0.802632 | single-baseline-guard-veto | single-baseline-fallback | no |
| 23 | 0.131634 | 0.12486 | 0.139845 | 0.131634 | 0.139845 | 0.1411 | 1x | -2 | 0.742143 | 14 | 0.947368 | single-baseline-veto | single-baseline-fallback | no |
| 24 | 47.199434 | 46.101695 | 45.954595 | 0.157763 | 0.166684 | 0.157763 | 299.179892x | 1 | 0.634947 | 8 | 0.868421 | validation-veto | island-nn-lift | yes |

