# NN Noisy 2D Black-Box ReLU Decompile Benchmark

- Teacher: fixed 2-input/3-hidden near-collision non-vertical ReLU MLP
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
| Plane baseline | 2.181039e-1 | 2.629275e-1 | 0.901342182 |
| GCL noisy 2D black-box ReLU decompiler | 6.722872e-4 | 8.500950e-4 | 0.999998969 |

## Gate

- Pass: yes
- Kink lines recovered within 0.08: yes
- Max delta error: 9.292242e-4
- Max alpha error: 1.024981e-3
- Max beta error: 1.587500e-3
- MAE < 0.04: yes (6.722872e-4)
- R2 > 0.998: yes (0.999998968678)

## Recovered Formula

```
0.1202231561 + 0.2500738822 * x + -0.1002900187 * y + -0.4491656175 * max(0, x - 0.3510249813 * y + 0.2015874995) + 0.7490707758 * max(0, x + 0.3498733629 * y + 0.1992322115) + 0.2798466139 * max(0, x + 0.1001477534 * y - 1.249155271)
```
