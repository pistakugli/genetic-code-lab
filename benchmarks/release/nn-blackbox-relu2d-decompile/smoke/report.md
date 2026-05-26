# NN 2D Black-Box ReLU Decompile Benchmark

- Teacher: fixed 2-input/3-hidden non-vertical ReLU MLP
- GCL access: oracle calls only, `f(x,y)`
- Hidden activations passed to GCL: no
- Teacher weights passed to GCL: no
- Samples per horizontal slice: 1201
- Test rows: 2745

| Model | Test MAE | Test RMSE | Test R2 |
|---|---:|---:|---:|
| Plane baseline | 2.610393e-1 | 3.278511e-1 | 0.949516476 |
| GCL 2D black-box ReLU decompiler | 3.188150e-15 | 4.164631e-15 | 1.000000000 |

## Gate

- Pass: yes
- All kink lines recovered: yes
- Max delta-slope error: 2.331468e-15
- MAE < 1e-9: yes (3.188150e-15)
- R2 > 0.999999999: yes (1.000000000000)

## Recovered Formula

```
0.2 + 0.3 * x + -0.15 * y + 0.9 * max(0, x + 0.4 * y + 1.2) + -0.55 * max(0, x - 0.7 * y - 0.15) + 0.35 * max(0, x + 0.2 * y - 1.1)
```
