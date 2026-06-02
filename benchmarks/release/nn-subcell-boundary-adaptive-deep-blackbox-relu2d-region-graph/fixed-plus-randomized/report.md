# Subcell Boundary Adaptive Deep Black-Box ReLU 2D Region-Graph Benchmark

Generated: 2026-06-02T08:54:52.529Z

Pass: yes

## Scope

- GCL access: oracle calls only, `f(x,y)`
- Claim: subcell validation selects the lowest passing boundary-aware grid and records rejected too-coarse attempts
- Not claimed: original hidden-layer identity, exact layer attribution, topology-perfect boundary localization, arbitrary deep ND extraction, or production API extraction

## Summary

- Teachers passed: 3/3
- Selected grid widths: 37, 37, 41
- Rejected too-coarse attempts: 1
- Minimum per-teacher oracle reduction vs fixed 101x101: 65.603%
- Total oracle reduction vs fixed 101x101: 78.088%
- Total oracle reduction vs boundary-refined 57x57: 35.741%
- Total fixed 101x101 oracle calls: 153015
- Total boundary-refined 57x57 oracle calls: 52176
- Total subcell-adaptive oracle calls: 33528
- Max subcell-adaptive MAE: 2.836963e-4
- Min subcell-adaptive R2: 0.999990210423
- Minimum held-out coarse-to-subcell MAE improvement: 3.550431x

## Results

| Teacher | Pass | Selected Grid | Attempts | Fixed Calls | Boundary57 Calls | Subcell Calls | Reduction vs Fixed | Reduction vs Boundary57 | Subcell MAE | Subcell R2 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| fixed-deep-2d-1 | yes | 37 | 37:pass | 51005 | 17392 | 7992 | 84.331% | 54.048% | 9.138690e-5 | 0.999999339240 |
| random-deep-2d-29 | yes | 37 | 37:pass | 51005 | 17392 | 7992 | 84.331% | 54.048% | 1.393128e-4 | 0.999990210423 |
| random-deep-2d-83 | yes | 41 | 37:fail, 41:pass | 51005 | 17392 | 17544 | 65.603% | -0.874% | 2.836963e-4 | 0.999996139739 |
