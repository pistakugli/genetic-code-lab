# Randomized Deep Black-Box ReLU 2D Region-Graph Sweep

Generated: 2026-06-02T09:57:47.881Z

Pass: yes

## Scope

- Teachers: deterministic randomized 2-input / 4-hidden / 4-hidden / 1-output ReLU MLPs
- GCL access: oracle calls only, `f(x,y)`
- Claim: output-equivalent affine-region graph recovery across a seed sweep
- Not claimed: original hidden-layer weights, exact layer attribution, arbitrary deep ND extraction, or noisy API recovery

## Summary

- Teachers passed: 4/4
- Seeds: 11, 29, 47, 83
- Min recovered regions: 20
- Min deep-only region estimate: 15
- Min adjacency edges: 40
- Min kink-cell boundaries: 839
- Max MAE: 0.00018896210525840208
- Min R2: 0.9999965825698692
- Total raw oracle calls: 204020
- Max raw oracle calls per teacher: 51005

## Results

| Teacher | Pass | Regions | Deep-only | Edges | Kink boundaries | MAE | R2 | First-layer MAE | Plane MAE | Oracle calls |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| deep-2d-seed-11 | yes | 29 | 24 | 57 | 1241 | 1.392588e-4 | 0.999998656482 | 1.622709e-1 | 5.481589e-1 | 51005 |
| deep-2d-seed-29 | yes | 20 | 15 | 40 | 894 | 8.806672e-5 | 0.999996582570 | 4.264412e-2 | 3.900457e-1 | 51005 |
| deep-2d-seed-47 | yes | 32 | 27 | 72 | 987 | 1.033340e-4 | 0.999999650672 | 6.867679e-2 | 4.593848e-1 | 51005 |
| deep-2d-seed-83 | yes | 21 | 16 | 44 | 839 | 1.889621e-4 | 0.999999446597 | 2.231453e-1 | 9.954581e-1 | 51005 |
