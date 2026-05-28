# Topology Boundary Deep Black-Box ReLU 2D Region-Graph Benchmark

Generated: 2026-05-28T06:14:26.537Z

Pass: yes

## Scope

- GCL access: oracle calls only, `f(x,y)`
- Claim: one topology boundary segment per recovered adjacency edge, with topology endpoints on the affine equality line
- Not claimed: original hidden-layer identity, exact layer attribution, topology-perfect true-region merging, arbitrary deep ND extraction, or production API extraction

## Summary

- Teachers passed: 3/3
- Minimum boundary compression: 10.022222x
- Max endpoint line residual: 4.440892e-16
- Max fallback segment count: 0
- Minimum topology segment length: 7.019353e-2
- Max MAE: 6.392578e-4
- Min R2: 0.999979702589
- Total raw oracle calls: 48735

## Results

| Teacher | Pass | Regions | Edges | Boundaries | Compression | Fallbacks | Max Endpoint Residual | MAE | R2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed-deep-2d-1 | yes | 21 | 43 | 43 | 12.302326x | 0 | 3.330669e-16 | 4.953795e-4 | 0.999996497384 |
| random-deep-2d-29 | yes | 19 | 38 | 38 | 13.052632x | 0 | 4.440892e-16 | 3.150531e-4 | 0.999979702589 |
| random-deep-2d-83 | yes | 20 | 45 | 45 | 10.022222x | 0 | 2.220446e-16 | 6.392578e-4 | 0.999996124999 |
