# Pure GCL hard-task diagnostic

Generated: 2026-05-22T05:45:41.324Z

## Scope

- Diagnostic pure-GCL hard-task benchmark. It disables regression-basis seeding and semantic refiner. It does not use gcl-hybrid.
- Compares a single pure GCL run with role-diverse pure GCL island runs.
- Selection uses validation MAE with a guard fallback to the single baseline.
- Oracle rows are diagnostic only and are selected using final test MAE.

## Summary

- Success rate: 0.0%
- Non-regression rate vs single: 0.0%
- Completed task/seed runs: 0/2
- Runtime timeouts: 2
- Runtime errors: 0
- Mean single MAE: 0
- Mean selected MAE: 0
- Mean oracle MAE: 0
- Median selected lift: 0x
- Median oracle lift: 0x

## Results

| Task | Seed | Single MAE | Selected MAE | Oracle MAE | Selected lift | Oracle lift | Selected by | Success |
|---|---:|---:|---:|---:|---:|---:|---|---|
| inverse_quadratic | 1337 | - | - | - | - | - | runtime-timeout: seed timed out after 1ms; completed after 848ms | no |
| trig_rational_combo | 1337 | - | - | - | - | - | runtime-timeout: seed timed out after 1ms; completed after 1169ms | no |

## Claim Boundary

- Passing this lane would support a pure-GCL hard-task improvement claim for these tasks only.
- Failing this lane preserves the current boundary: hard inverse/rational tasks remain pure-GCL blockers while gcl-hybrid remains separate.
