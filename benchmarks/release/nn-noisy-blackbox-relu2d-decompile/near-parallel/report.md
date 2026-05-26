# NN Noisy 2D Black-Box ReLU Decompile Benchmark

- Teacher: fixed 2-input/3-hidden near-parallel non-vertical ReLU MLP
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
| Plane baseline | 1.350660e-1 | 1.807823e-1 | 0.980215521 |
| GCL noisy 2D black-box ReLU decompiler | 1.682630e-4 | 2.042688e-4 | 0.999999975 |

## Gate

- Pass: yes
- Kink lines recovered within 0.08: yes
- Max delta error: 7.278496e-4
- Max alpha error: 2.779999e-4
- Max beta error: 1.025587e-3
- MAE < 0.04: yes (1.682630e-4)
- R2 > 0.998: yes (0.999999974741)

## Recovered Formula

```
0.1506411725 + 0.2802170733 * x + -0.1200228943 * y + 0.8492721504 * max(0, x + 0.2802779999 * y + 1.149789796) + -0.4995328214 * max(0, x + 0.2399979712 * y - 0.05102558696) + 0.3198333075 * max(0, x + 0.2002057543 * y - 1.279678993)
```
