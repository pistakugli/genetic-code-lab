# Deep Black-Box ReLU 1D Decompile Benchmark

Generated: 2026-05-26T11:41:41.661Z

Pass: yes

## Scope

- Teacher: 1-input / 5-hidden / 5-hidden / 1-output ReLU MLP
- GCL access: oracle calls only, `f(x)`
- Claim: output-equivalent canonical spline recovery
- Not claimed: unique original hidden-layer weights

## Metrics

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Linear | 8.508542e-1 | 1.072668e+0 | 0.844349830638 |
| First-layer-kink-only | 2.680105e-2 | 4.176996e-2 | 0.999763980973 |
| GCL canonical deep spline | 6.677978e-14 | 1.162592e-13 | 1.000000000000 |

## Folding Evidence

- First-layer kink count: 5
- Recovered canonical kink count: 11
- Deep-only kink count: 6
- Recovered region count: 12

## Recovered Formula

```
6.220871046 + -0.06762652933 * x + 0.7931387016 * max(0, x + 1.994847744) + -2.129820311 * max(0, x + 1.653305037) + -1.386947121 * max(0, x + 1.614131925) + -1.498726168 * max(0, x + 0.7591071061) + 3.530671784 * max(0, x + 0.6649769418) + -1.731642438 * max(0, x + 0.2634789619) + -1.710765433 * max(0, x + 0.08334915738) + 1.808276155 * max(0, x - 0.7281923942) + -1.267853311 * max(0, x - 0.7944986183) + 4.604312823 * max(0, x - 0.8225264887) + -0.2614766098 * max(0, x - 0.9584286189)
```
