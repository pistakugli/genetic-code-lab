# Deep Black-Box ReLU 2D Region-Graph Benchmark

Generated: 2026-06-01T16:13:54.181Z

Pass: yes

## Scope

- Teacher: 2-input / 4-hidden / 4-hidden / 1-output ReLU MLP
- GCL access: oracle calls only, `f(x,y)`
- Claim: output-equivalent affine-region graph recovery
- Not claimed: original hidden-layer weights or exact layer attribution

## Metrics

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Plane | 8.308133e-1 | 9.943923e-1 | 0.826346926953 |
| First-layer-only ReLU basis | 2.308757e-1 | 3.493680e-1 | 0.978564516768 |
| GCL region graph | 1.484863e-4 | 1.850454e-3 | 0.999999398655 |

## Folding Evidence

- First-layer unit count: 4
- Recovered affine region count: 21
- Deep-only region count estimate: 16
- Region adjacency edges: 46
- Kink cell boundaries: 964
- Raw oracle calls: 51005

## Recovered Structure

```
region_graph( regions=21 adjacency=46 boundaries=46 grid=101x101 )
```
