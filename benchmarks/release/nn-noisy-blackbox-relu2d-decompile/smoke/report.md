# NN Noisy 2D Black-Box ReLU Decompile Benchmark

- Teacher: fixed 2-input/3-hidden non-crossing non-vertical ReLU MLP
- GCL access: repeated noisy oracle calls, `f(x,y)`
- Hidden activations passed to GCL: no
- Teacher weights passed to GCL: no
- Noise std: 0.01
- Quantization step: 0.0025
- Query samples per x,y: 25
- Samples per horizontal slice: 321
- Test rows: 2745

| Model | Test MAE vs clean teacher | Test RMSE | Test R2 |
|---|---:|---:|---:|
| Plane baseline | 1.838771e-1 | 2.363100e-1 | 0.972428868 |
| GCL noisy 2D black-box ReLU decompiler | 5.994632e-4 | 6.836764e-4 | 0.999999769 |

## Gate

- Pass: yes
- Kink lines recovered within 0.08: yes
- Max delta error: 6.758327e-4
- Max alpha error: 8.991790e-4
- Max beta error: 2.571277e-4
- MAE < 0.04: yes (5.994632e-4)
- R2 > 0.998: yes (0.999999769224)

## Recovered Formula

```
0.200289327 + 0.3000990737 * x + -0.1505816989 * y + 0.8993241673 * max(0, x + 0.3502862537 * y + 1.250020355) + -0.5496739539 * max(0, x - 0.2504428012 * y - 0.1502571277) + 0.3500862165 * max(0, x + 0.150899179 * y - 1.249938713)
```
