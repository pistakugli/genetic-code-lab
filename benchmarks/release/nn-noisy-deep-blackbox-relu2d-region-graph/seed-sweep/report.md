# Noisy Deep Black-Box ReLU 2D Region-Graph Sweep

Generated: 2026-06-02T20:41:42.237Z

Pass: yes

## Scope

- Teachers: deterministic randomized 2-input / 4-hidden / 4-hidden / 1-output ReLU MLPs
- GCL access: repeated noisy/quantized oracle calls, `f(x,y)` only
- Claim: noise-tolerant output-equivalent affine-region graph recovery across a seed sweep
- Not claimed: original hidden-layer weights, exact layer attribution, arbitrary deep ND extraction, or production API extraction beyond this noise model

## Summary

- Teachers passed: 2/2
- Seeds: 29, 83
- Min recovered regions: 125
- Min deep-only region estimate: 120
- Min adjacency edges: 304
- Min kink-cell boundaries: 2061
- Max MAE: 0.002394670345958939
- Min R2: 0.9999458398613055
- Total raw oracle calls: 1377810
- Max raw oracle calls per teacher: 688905

## Results

| Teacher | Pass | Regions | Deep-only | Edges | Kink boundaries | MAE | R2 | First-layer MAE | Plane MAE | Raw calls |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| noisy-deep-2d-seed-29 | yes | 125 | 120 | 304 | 2061 | 9.792275e-4 | 0.999945839861 | 4.266156e-2 | 3.898374e-1 | 688905 |
| noisy-deep-2d-seed-83 | yes | 182 | 177 | 592 | 2206 | 2.394670e-3 | 0.999983897850 | 2.229795e-1 | 9.955250e-1 | 688905 |
