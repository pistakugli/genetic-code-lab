# Pure GCL hard-task diagnostic

Generated: 2026-05-22T05:41:59.773Z

## Scope

- Diagnostic pure-GCL hard-task benchmark. It disables regression-basis seeding and semantic refiner. It does not use gcl-hybrid.
- Compares a single pure GCL run with role-diverse pure GCL island runs.
- Selection uses validation MAE with a guard fallback to the single baseline.
- Oracle rows are diagnostic only and are selected using final test MAE.

## Summary

- Success rate: 50.0%
- Non-regression rate vs single: 100.0%
- Mean single MAE: 0.27648
- Mean selected MAE: 0.030061
- Mean oracle MAE: 0.030061
- Median selected lift: 4.218e+3x
- Median oracle lift: 4.218e+3x

## Results

| Task | Seed | Single MAE | Selected MAE | Oracle MAE | Selected lift | Oracle lift | Selected by | Success |
|---|---:|---:|---:|---:|---:|---:|---|---|
| inverse_quadratic | 1337 | 0.236032 | 2.800e-5 | 2.800e-5 | 8.430e+3x | 8.430e+3x | validation | yes |
| trig_rational_combo | 1337 | 0.316927 | 0.060094 | 0.060094 | 5.273901x | 5.273901x | validation | no |

## Claim Boundary

- Passing this lane would support a pure-GCL hard-task improvement claim for these tasks only.
- Failing this lane preserves the current boundary: hard inverse/rational tasks remain pure-GCL blockers while gcl-hybrid remains separate.
