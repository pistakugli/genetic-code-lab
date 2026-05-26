# NN Circuit Distillation Benchmark

- Teacher: fixed 3-input/3-hidden tanh MLP
- GCL access: input-output samples for each hidden neuron plus the output readout
- Train rows: 405
- Test rows: 256

| Target | Test MAE | Test RMSE | Test R2 | Stages |
|---|---:|---:|---:|---:|
| h0 | 0.000000 | 0.000000 | 1.000000 | 1/3 |
| h1 | 0.000000 | 0.000000 | 1.000000 | 1/3 |
| h2 | 0.000000 | 0.000000 | 1.000000 | 1/3 |
| output via symbolic hidden circuit | 0.000000 | 0.000000 | 1.000000 | - |

## Gate

- Pass: yes
- Max hidden MAE < 0.03: yes (0.000000)
- Output R2 >= 0.995: yes (1.000000)
- Output MAE < 0.03: yes (0.000000)

## Hidden Formulas

### h0

```
1.0000000 * ((tanh (add (add (add const(0.19999999999999993) (mul const(1.1499999999997925) x)) (mul const(-0.7499999999998646) x1)) (mul const(0.34999999999991155) x2))))
```

### h1

```
1.0000000 * ((tanh (add (add (add const(-0.14999999999999997) (mul const(-0.5499999999999008) x)) (mul const(1.1999999999997832) x1)) (mul const(-0.24999999999993727) x2))))
```

### h2

```
1.0000000 * ((tanh (add (add (add const(-5.2413491927984524e-17) (mul const(0.3499999999999368) x)) (mul const(0.44999999999991896) x1)) (mul const(1.0499999999997358) x2))))
```
