# NN Noisy 3D Black-Box ReLU Decompile Benchmark

- Teacher: fixed 3-input/3-hidden non-vertical ReLU MLP
- GCL access: repeated noisy oracle calls, `f(x,y,z)`
- Hidden activations passed to GCL: no
- Teacher weights passed to GCL: no
- Noise std: 0.006
- Quantization step: 0.002
- Query samples per x,y,z: 17
- Z slices: 7
- Y slices per z-slice: 9
- Samples per horizontal line: 261
- Test rows: 4485

| Model | Test MAE vs clean teacher | Test RMSE | Test R2 |
|---|---:|---:|---:|
| Linear hyperplane baseline | 1.838539e-1 | 2.335083e-1 | 0.961981450 |
| GCL noisy 3D black-box ReLU decompiler | 2.823208e-4 | 3.587329e-4 | 0.999999910 |

## Gate

- Pass: yes
- All hyperplanes recovered within 0.12: yes
- Max delta-slope error: 4.460461e-4
- Max alphaY error: 1.685257e-4
- Max alphaZ error: 7.093442e-4
- Max beta error: 2.229722e-4
- MAE < 0.08: yes (2.823208e-4)
- R2 > 0.995: yes (0.999999910271)

## Recovered Formula

```
0.1798837229 + 0.2199450235 * x + -0.1102954074 * y + 0.06982643475 * z + 0.8198258924 * max(0, x + 0.2501685257 * y - 0.1998058364 * z + 1.100222972) + -0.4795539539 * max(0, x - 0.3501417463 * y + 0.2993158001 * z - 0.1500339691) + 0.30975742 * max(0, x + 0.1000938796 * y + 0.2207093442 * z - 1.200205915)
```
