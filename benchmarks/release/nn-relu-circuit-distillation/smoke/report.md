# NN Circuit Distillation Benchmark

- Teacher: fixed 3-input/3-hidden relu MLP
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
1.0000000 * ((max const(0) (add (add (add const(0.20000000000124726) (mul const(1.1499999999988055) x)) (mul const(-0.7499999999993188) x1)) (mul const(0.3499999999996446) x2))))
```

### h1

```
1.0000000 * ((max const(0) (add (add (add const(-0.149999999997828) (mul const(-0.54999999999933) x)) (mul const(1.1999999999980637) x1)) (mul const(-0.24999999999965988) x2))))
```

### h2

```
1.0000000 * ((max const(0) (add (add (add const(1.6699974736411605e-12) (mul const(0.3499999999995639) x)) (mul const(0.44999999999943985) x1)) (mul const(1.0499999999981788) x2))))
```
