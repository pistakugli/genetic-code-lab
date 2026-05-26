# NN Noisy Black-Box ReLU Decompile Benchmark

- Teacher: fixed 1-input/5-hidden ReLU MLP
- GCL access: repeated noisy oracle calls, `f(x)`
- Hidden activations passed to GCL: no
- Teacher weights passed to GCL: no
- Noise std: 0.015
- Quantization step: 0.005
- Query samples per x: 41
- Sample rows: 421
- Test rows: 801

| Model | Test MAE vs clean teacher | Test RMSE | Test R2 |
|---|---:|---:|---:|
| Linear baseline | 1.512005e-1 | 1.936926e-1 | 0.979447525 |
| GCL noisy black-box ReLU decompiler | 4.549905e-4 | 5.687638e-4 | 0.999999823 |

## Gate

- Pass: yes
- Kinks recovered within 0.06: yes (4.042679e-3)
- MAE < 0.02: yes (4.549905e-4)
- R2 > 0.999: yes (0.999999822784)

## Recovered Formula

```
0.2524028607 + 0.3512515424 * x + -0.2999742315 * max(0, x + 2.149634755) + 1.098188088 * max(0, x + 1.399890118) + -0.6967671892 * max(0, x - 0.2479024708) + 0.4476665148 * max(0, x - 1.201624206) + 0.2215098351 * max(0, x - 2.354042679)
```
