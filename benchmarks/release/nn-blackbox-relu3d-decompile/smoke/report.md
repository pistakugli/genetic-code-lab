# NN 3D Black-Box ReLU Decompile Benchmark

- Teacher: fixed 3-input/3-hidden non-vertical ReLU MLP
- GCL access: oracle calls only, `f(x,y,z)`
- Hidden activations passed to GCL: no
- Teacher weights passed to GCL: no
- Z slices: 7
- Samples per 2D slice line: 801
- Test rows: 11067

| Model | Test MAE | Test RMSE | Test R2 |
|---|---:|---:|---:|
| Linear hyperplane baseline | 1.859440e-1 | 2.359625e-1 | 0.961592443 |
| GCL 3D black-box ReLU decompiler | 6.683338e-16 | 8.783213e-16 | 1.000000000 |

## Gate

- Pass: yes
- All hyperplanes recovered: yes
- Max delta-slope error: 1.443290e-15
- Max alphaY error: 1.276756e-15
- Max alphaZ error: 1.471046e-15
- Max beta error: 1.776357e-15
- MAE < 1e-9: yes (6.683338e-16)
- R2 > 0.999999999: yes (1.000000000000)

## Recovered Formula

```
0.18 + 0.22 * x + -0.11 * y + 0.07 * z + 0.82 * max(0, x + 0.25 * y - 0.2 * z + 1.1) + -0.48 * max(0, x - 0.35 * y + 0.3 * z - 0.15) + 0.31 * max(0, x + 0.1 * y + 0.22 * z - 1.2)
```
