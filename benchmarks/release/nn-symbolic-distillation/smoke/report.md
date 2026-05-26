# NN Symbolic Distillation Benchmark

- Teacher: fixed 3-input/3-hidden tanh MLP
- GCL access: input-output samples only
- Train rows: 405
- Test rows: 256

| Model | Test MAE | Test RMSE | Test R2 |
|---|---:|---:|---:|
| Linear baseline | 0.324956 | 0.382150 | 0.877460 |
| GCL symbolic residual ensemble | 0.213823 | 0.311987 | 0.918326 |

## Gate

- Pass: yes
- Accepted stages: 4/4
- Beats linear by >=20%: yes
- R2 >= 0.85: yes

## Residual Stages

| Stage | Status | Coef | Validation residual MAE | Fitness |
|---:|---|---:|---:|---:|
| 1 | accepted | 0.575318 | 1.256142 -> 0.733051 | 0.921065 |
| 2 | accepted | 0.689236 | 0.733051 -> 0.460370 | 0.931897 |
| 3 | accepted | 1.107306 | 0.460370 -> 0.394485 | 0.938616 |
| 4 | accepted | 2.958486 | 0.394485 -> 0.348577 | 0.945079 |

## Expression

```
0.57531830 * ((add (add (add (add (add (add (add (mul const(0.3172760109531353) (sub x x1)) (mul const(0.3773204868457117) (add x x2))) (mul const(0.00003080244461619857) (div (mul x x1) (sqrt x1)))) (mul const(0.000027698347613278363) (div (cos x) (sqrt x1)))) (mul const(-0.000014235957392534995) (div (mul x1 x2) (sqrt (sub x x1))))) (mul const(0.2726750584394589) (sin (add (add (add (add (add (sin (add (add (add (add (add (add (add (mul const(0.3172760109531353) (sub x x1)) (mul const(-0.000023512951058430727) (add x1 x2))) (mul const(0.00003080244461619857) (div (mul x x1) (sqrt x1)))) (sub x (sin x1))) (div const(-0.000014235957392534995) (div x2 (sqrt x)))) (mul const(0.2726750584394589) (sin (sub const(0.3172760109531353) x1)))) (mul const(0.3773204868457117) (sin (add x x2)))) (mul const(-0.000023512951058430727) (div (sin x1) (sqrt x))))) (mul const(0.000027698347613278363) (div x2 (sqrt x1)))) (mul const(-0.000014235957392534995) (div (mul const(0.3773204868457117) x2) (sqrt x)))) (mul const(0.2726750584394589) (sin x))) (mul const(0.17102712163883932) (sin (add x x2)))) (mul const(-0.000023512951058430727) (div (sin x1) (sqrt x))))))) (mul const(0.17102712163883932) (sin (add x x2)))) (mul const(-0.000023512951058430727) (div (sin x1) (sqrt x))))) + 0.68923595 * ((add (add (add (add (add (add (add (mul const(-0.00002658484639095959) (div (sin x1) (sqrt x))) (mul const(0.1440008825886489) (add x x2))) (mul const(0.4757283718851092) (sin (sub x x1)))) (mul const(0.12228324515109568) (cos (add x1 (sin x2))))) (mul const(0.000029359629285642663) (div (mul x x1) (sqrt x)))) (mul const(0.18764009729765552) (cos (add const(1.0010943323667785) x)))) (mul const(0.12431514339447809) (sin (add x1 x2)))) (mul const(-0.000011695684105614602) (div (mul x1 x2) (sqrt (add const(0.12228324515109568) x)))))) + 1.1073061 * ((add (mul const(0.9942859524988662) (add (add (add (add (add (add (add (mul const(-0.000028279868062770277) (div (sin x1) (sqrt x))) (mul const(0.0000277960444390483) (div (mul x x1) (sqrt x)))) (mul const(0.12698641364827112) (sin (sub x x1)))) (mul const(-0.00001317836297536685) (div (mul x1 x2) (sqrt x)))) (mul const(0.09157578540712194) (sqrt (sqrt x2)))) (mul const(0.000010353232958282482) (div (mul x1 x2) (mul x x)))) (mul const(0.0000062975981031585324) (div (sqrt x1) (mul (sin (sub x x1)) x2)))) (mul const(0.03779739494632774) (sin (add x x2))))) (mul const(-0.003392153344614218) (sin x1)))) + 2.9584860 * ((add (mul const(0.9959325674179446) (add (add (add (add (add (add (add (mul const(0.000027267851653293964) (div (mul x x1) (sqrt x))) (mul const(-0.000015360792554476693) (div (mul x1 x2) (sqrt x)))) (mul const(-0.000019368536942012474) (div (sin x1) (sqrt x)))) (mul const(0.000012160868030459742) (div (mul x1 x2) (div x (sub const(0.000012160868030459742) (div (mul x1 x2) (div x x1))))))) (mul const(0.000005302483682172471) (div (mul x x1) (sqrt x2)))) (mul const(0.0000010987389994894852) (div (mul x1 x1) (sqrt x2)))) (mul const(0.000006162542017740873) (div x2 (sub x1 x)))) (mul const(-0.045061471136354725) (sin (add x x1))))) (mul const(-0.009270237554311698) (sub x1 x))))
```
