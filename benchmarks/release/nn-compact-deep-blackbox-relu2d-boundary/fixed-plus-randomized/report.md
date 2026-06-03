# Compact Deep Black-Box ReLU 2D Boundary Benchmark

Generated: 2026-06-03T06:13:26.571Z

Pass: yes

## Scope

- GCL access: oracle calls only, `f(x,y)`
- Claim: compact boundary segment extraction from the recovered output-equivalent region graph
- Not claimed: original hidden-layer identity, exact layer attribution, or topology-perfect region merging

## Summary

- Teachers passed: 3/3
- Minimum boundary compression ratio: 19.068182x
- Max boundary fit RMSE: 7.552766e-2
- Max MAE: 1.775144e-4
- Min R2: 0.999996768773
- Total raw oracle calls: 153015

## Results

| Teacher | Pass | Regions | Cell Boundaries | Compact Segments | Compression | Max Boundary RMSE | MAE | R2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed-deep-2d-1 | yes | 21 | 964 | 46 | 20.956522x | 3.240425e-2 | 1.404930e-4 | 0.999999488513 |
| random-deep-2d-29 | yes | 20 | 894 | 40 | 22.350000x | 1.968829e-2 | 9.271956e-5 | 0.999996768773 |
| random-deep-2d-83 | yes | 21 | 839 | 44 | 19.068182x | 7.552766e-2 | 1.775144e-4 | 0.999999536711 |
