# Boundary-Refined Deep Black-Box ReLU 2D Region-Graph Benchmark

Generated: 2026-06-02T09:57:56.885Z

Pass: yes

## Scope

- GCL access: oracle calls only, `f(x,y)`
- Claim: compact-boundary local switching improves a lower-cost 57x57 output-equivalent region graph
- Not claimed: original hidden-layer identity, exact layer attribution, arbitrary deep ND extraction, or production API extraction

## Summary

- Teachers passed: 3/3
- Minimum oracle reduction vs fixed 101x101: 65.901%
- Total fixed 101x101 oracle calls: 153015
- Total boundary-refined oracle calls: 52176
- Max boundary-refined MAE: 7.994249e-5
- Min boundary-refined R2: 0.999999012853
- Minimum held-out coarse-to-refined MAE improvement: 7.996471x

## Results

| Teacher | Pass | Fixed Calls | Refined Calls | Reduction | Coarse MAE | Refined MAE | Improvement | Refined R2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed-deep-2d-1 | yes | 51005 | 17392 | 65.901% | 4.953795e-4 | 2.658264e-5 | 18.635451x | 0.999999729909 |
| random-deep-2d-29 | yes | 51005 | 17392 | 65.901% | 3.150531e-4 | 1.441883e-5 | 21.850114x | 0.999999012853 |
| random-deep-2d-83 | yes | 51005 | 17392 | 65.901% | 6.392578e-4 | 7.994249e-5 | 7.996471x | 0.999999365464 |
