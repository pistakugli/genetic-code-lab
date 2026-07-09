# Pure GCL hard-task diagnostic

Generated: 2026-06-30T19:14:50.007Z

## Scope

- Diagnostic pure-GCL hard-task benchmark. It disables regression-basis seeding and semantic refiner. It does not use gcl-hybrid.
- Compares a single pure GCL run with role-diverse pure GCL island runs.
- Selection uses validation MAE with a guard fallback to the single baseline.
- Oracle rows are diagnostic only and are selected using final test MAE.

## Summary

- Success rate: 100.0%
- Non-regression rate vs single: 100.0%
- Completed task/seed runs: 6/6
- Runtime timeouts: 0
- Runtime errors: 0
- Mean single MAE: 1.093e-5
- Mean selected MAE: 6.976e-6
- Mean oracle MAE: 6.976e-6
- Median selected lift: 1x
- Median oracle lift: 1x

## Results

| Task | Seed | Single MAE | Selected MAE | Oracle MAE | Selected lift | Oracle lift | Selected by | Success |
|---|---:|---:|---:|---:|---:|---:|---|---|
| inverse_quadratic | 1337 | 2.371e-5 | 2.500e-17 | 2.500e-17 | 2.371e+7x | 2.371e+7x | validation | yes |
| inverse_quadratic | 2026 | 2.993e-17 | 3.659e-17 | 2.477e-17 | 1x | 1x | validation | yes |
| inverse_quadratic | 4242 | 2.649e-17 | 2.798e-17 | 2.202e-17 | 1x | 1x | validation | yes |
| trig_rational_combo | 1337 | 1.395e-5 | 1.395e-5 | 1.395e-5 | 1x | 1x | validation | yes |
| trig_rational_combo | 2026 | 1.395e-5 | 1.395e-5 | 1.395e-5 | 1x | 1x | validation | yes |
| trig_rational_combo | 4242 | 1.395e-5 | 1.395e-5 | 1.395e-5 | 1x | 1x | validation | yes |

## Claim Boundary

- Passing this lane would support a pure-GCL hard-task improvement claim for these tasks only.
- Failing this lane preserves the current boundary: hard inverse/rational tasks remain pure-GCL blockers while gcl-hybrid remains separate.
