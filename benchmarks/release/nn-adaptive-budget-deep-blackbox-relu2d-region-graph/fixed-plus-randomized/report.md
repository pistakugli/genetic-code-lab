# Adaptive Budget Deep Black-Box ReLU 2D Region-Graph Benchmark

Generated: 2026-06-01T16:14:03.182Z

Pass: yes

## Scope

- GCL access: oracle calls only, `f(x,y)`
- Claim: adaptive grid-budget selection with held-out validation and compact boundary preservation
- Not claimed: full local spatial refinement, original hidden-layer identity, exact layer attribution, or arbitrary deep ND extraction

## Summary

- Teachers passed: 3/3
- Selected grid widths: 89, 89, 89
- Minimum oracle reduction: 20.102%
- Total fixed oracle calls: 153015
- Total adaptive oracle calls: 122256
- Max adaptive MAE: 2.406915e-4
- Min adaptive R2: 0.999992858794

## Results

| Teacher | Pass | Fixed Calls | Adaptive Calls | Reduction | Selected Grid | Adaptive MAE | Adaptive R2 | Boundary RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed-deep-2d-1 | yes | 51005 | 40752 | 20.102% | 89 | 2.406915e-4 | 0.999998471482 | 2.828993e-2 |
| random-deep-2d-29 | yes | 51005 | 40752 | 20.102% | 89 | 1.690140e-4 | 0.999992858794 | 2.924002e-2 |
| random-deep-2d-83 | yes | 51005 | 40752 | 20.102% | 89 | 1.642978e-4 | 0.999999236542 | 2.897477e-2 |
