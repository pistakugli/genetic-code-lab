# NN Noisy 2D Black-Box ReLU Decompile Benchmark

- Teacher: fixed 2-input/3-hidden crossing non-vertical ReLU MLP
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
| Plane baseline | 2.610393e-1 | 3.278511e-1 | 0.949516476 |
| GCL noisy 2D black-box ReLU decompiler | 6.050245e-3 | 8.652800e-3 | 0.999964835 |

## Gate

- Pass: yes
- Kink lines recovered within 0.08: yes
- Max delta error: 1.189245e-2
- Max alpha error: 1.382444e-3
- Max beta error: 1.014471e-3
- MAE < 0.04: yes (6.050245e-3)
- R2 > 0.998: yes (0.999964835083)

## Recovered Formula

```
0.2001832804 + 0.3000543196 * x + -0.1504279413 * y + 0.8925657387 * max(0, x + 0.3993695036 * y + 1.20084909) + -0.538107554 * max(0, x - 0.6986175559 * y - 0.1510144711) + 0.3470941384 * max(0, x + 0.1989686794 * y - 1.100989322)
```
