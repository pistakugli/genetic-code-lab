# NN Black-Box ReLU Decompile Benchmark

- Teacher: fixed 1-input/5-hidden ReLU MLP
- GCL access: oracle calls only, `f(x)`
- Hidden activations passed to GCL: no
- Teacher weights passed to GCL: no
- Sample rows: 1201
- Test rows: 701

| Model | Test MAE | Test RMSE | Test R2 |
|---|---:|---:|---:|
| Linear baseline | 1.512097e-1 | 1.937040e-1 | 0.979446686 |
| GCL black-box ReLU decompiler | 2.432077e-15 | 2.804578e-15 | 1.000000000 |

## Gate

- Pass: yes
- All kinks recovered: yes
- Max kink error: 1.465494e-14
- MAE < 1e-9: yes (2.432077e-15)
- R2 > 0.999999999: yes (1.000000000000)

## Recovered Formula

```
0.25 + 0.35 * x + -0.3 * max(0, x + 2.15) + 1.1 * max(0, x + 1.4) + -0.7 * max(0, x - 0.25) + 0.45 * max(0, x - 1.2) + 0.22 * max(0, x - 2.35)
```
