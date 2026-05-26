# Open-source symbolic-regression benchmark

Generated: 2026-05-24T08:01:36.788Z

## Scope

- Compares local Genetic Code Lab against open-source symbolic-regression baselines `gplearn` and `DEAP`.
- Uses identical train/test points per task and the same headline population/generation budget.
- GCL may apply train-only affine output calibration `a * expression + b` to its best genome; test points are never used to fit `a` or `b`.
- `gcl-hybrid` is a production-gated train-only sparse basis-regression lane over generic GCL math bases; it is reported separately from pure `gcl`.
- This is an engineering smoke benchmark with claim-audit accounting, not a peer-reviewed superiority claim.

## Budget

- Seeds: 1337, 2026, 4242
- Population: 300
- Generations: 80
- Python: /tmp/gcl-open-source-bench-venv/bin/python

## Summary

| Task | Baseline | Runs | Failed | Success | Mean MAE | Mean RMSE | Mean R2 | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cubic_arithmetic | deap | 3 | 0 | 0% | 0.197379 | 0.302238 | 0.982625 | 1370.590838 |
| cubic_arithmetic | gcl | 3 | 0 | 100% | 1.417e-12 | 1.702e-12 | 1 | 74.93694 |
| cubic_arithmetic | gcl-hybrid | 3 | 0 | 100% | 1.011e-12 | 1.641e-12 | 1 | 822.446849 |
| cubic_arithmetic | gplearn | 3 | 0 | 0% | 0.128287 | 0.195041 | 0.994713 | 9037.294202 |
| quadratic_arithmetic | deap | 3 | 0 | 100% | 2.056e-18 | 1.379e-17 | 1 | 1316.905055 |
| quadratic_arithmetic | gcl | 3 | 0 | 100% | 1.485e-16 | 4.094e-16 | 1 | 87.333193 |
| quadratic_arithmetic | gcl-hybrid | 3 | 0 | 100% | 4.626e-13 | 5.962e-13 | 1 | 797.049404 |
| quadratic_arithmetic | gplearn | 3 | 0 | 67% | 0.006002 | 0.006002 | 0.999996 | 855.616809 |
| rational_shifted | deap | 3 | 0 | 67% | 0.00067 | 0.00083 | 0.999986 | 2219.666684 |
| rational_shifted | gcl | 3 | 0 | 100% | 8.097e-6 | 9.010e-6 | 1 | 139.618117 |
| rational_shifted | gcl-hybrid | 3 | 0 | 100% | 3.920e-12 | 4.067e-12 | 1 | 1145.182608 |
| rational_shifted | gplearn | 3 | 0 | 33% | 0.035843 | 0.114124 | 0.772808 | 7551.404658 |
| trig_poly_mixed | deap | 3 | 0 | 0% | 0.083677 | 0.148189 | 0.980331 | 1229.475684 |
| trig_poly_mixed | gcl | 3 | 0 | 100% | 5.283e-8 | 2.864e-7 | 1 | 709.757199 |
| trig_poly_mixed | gcl-hybrid | 3 | 0 | 100% | 1.963e-12 | 2.210e-12 | 1 | 1235.61802 |
| trig_poly_mixed | gplearn | 3 | 0 | 33% | 0.091204 | 0.172337 | 0.962446 | 7934.062368 |

## Winners by Mean MAE

- cubic_arithmetic: gcl-hybrid (mean MAE 1.011e-12)
- quadratic_arithmetic: deap (mean MAE 2.056e-18)
- rational_shifted: gcl-hybrid (mean MAE 3.920e-12)
- trig_poly_mixed: gcl-hybrid (mean MAE 1.963e-12)

## Claim Audit

- Pure GCL vs external baselines: positive-smoke (win rate 75%, comparable rate 100%).
- GCL hybrid vs pure GCL: limited-pass (win rate 50%, solved rate 100%, regressions 0, success regressions 0).
- Hybrid recommendation: gcl-hybrid passes this limited task set; run the wide production gate before promoting it.
- Claim audit failures: none

## Best Expressions

- cubic_arithmetic / deap: sub(mul(mul(add(div(div(1.3380464938637502, sub(x, 0.00744275747584755)), mul(sub(x, 0.00744275747584755), 4.913624016632141)), add(div(div(1.3380464938637502, x), mul(sub(sub(x, 0.00744275747584755), 0.00744275747584755), 4.913624016632...
- cubic_arithmetic / gcl: (add (mul const(0.9999999999990848) (add const(1) (mul (mul x x) x))) (mul const(-0.4999999999986295) (add const(1) (add x x))))
- cubic_arithmetic / gcl-hybrid: 0.5*1 + -1*mul(sub(1,mul(x,x)),x)
- cubic_arithmetic / gplearn: sub(mul(mul(X0, mul(add(X0, X0), X0)), sub(3.032, 2.702)), sub(sub(add(X0, X0), add(sub(mul(mul(X0, mul(X0, X0)), sub(3.032, 2.702)), mul(mul(X0, div(0.078, add(4.661, X0))), sub(div(div(0.078, add(4.661, X0)), add(div(4.661, mul(-1.729,...
- quadratic_arithmetic / deap: add(add(add(div(div(x, div(add(1.364821443055697, x), div(sub(x, x), mul(sub(div(-1.0501598974058437, x), add(4.645031363114624, x)), div(add(-4.310406912434344, -3.913564702017683), mul(x, 1.4915669260979039)))))), sub(x, x)), x), x), m...
- quadratic_arithmetic / gcl: (mul const(1) (mul (add const(1) x) (add const(1) x)))
- quadratic_arithmetic / gcl-hybrid: 1*1 + 1*add(add(x,x),mul(x,x))
- quadratic_arithmetic / gplearn: add(add(mul(X0, X0), add(X0, X0)), div(X0, X0))
- rational_shifted / deap: div(x, sub(div(x, x), sub(sub(x, x), mul(x, x))))
- rational_shifted / gcl: add(mul(0.999961, (div x (add const(0.9999) (mul x x)))), 4.823e-17)
- rational_shifted / gcl-hybrid: 1*div(x,add(1,mul(x,x)))
- rational_shifted / gplearn: div(X0, add(div(X0, X0), mul(X0, X0)))
- trig_poly_mixed / deap: sub(cos(sin(mul(cos(sin(mul(sin(sin(x)), mul(0.9260435271845129, 0.9260435271845129)))), x))), sub(sub(sub(cos(x), 0.02503941599370485), sin(x)), 0.02503941599370485))
- trig_poly_mixed / gcl: (add (add (add (mul const(0.000004130237883419432) (add (add (add (add (mul const(0.2500395737196695) (add (add x x) (mul x x))) (mul const(0.49215833192720965) (mul (cos x) x))) (mul const(0.0882941953727949) (mul (mul x x) (sin x)))) (...
- trig_poly_mixed / gcl-hybrid: 0.25*mul(x,x) + 1*sin(x)
- trig_poly_mixed / gplearn: div(sub(div(sub(X0, X0), div(-2.509, -0.252)), sub(sub(sub(sub(sub(div(add(add(div(mul(0.831, sin(div(sub(sub(add(X0, sub(3.841, X0)), sin(sin(cos(X0)))), add(sin(add(X0, sin(X0))), sin(X0))), cos(X0)))), sub(cos(sub(sin(sub(div(sub(div(...

## GCL Calibration

- quadratic_arithmetic / seed 1337: rejected (train MAE 6.288e-17 -> 6.288e-17, validation MAE 7.852e-17 -> 7.852e-17, scale 1, offset 0)
- quadratic_arithmetic / seed 2026: rejected (train MAE 6.288e-17 -> 6.288e-17, validation MAE 7.852e-17 -> 7.852e-17, scale 1, offset 0)
- quadratic_arithmetic / seed 4242: rejected (train MAE 6.288e-17 -> 6.288e-17, validation MAE 7.852e-17 -> 7.852e-17, scale 1, offset 0)
- cubic_arithmetic / seed 1337: rejected (train MAE 1.199e-12 -> 1.199e-12, validation MAE 1.181e-12 -> 1.181e-12, scale 1, offset 0)
- cubic_arithmetic / seed 2026: rejected (train MAE 1.199e-12 -> 1.199e-12, validation MAE 1.181e-12 -> 1.181e-12, scale 1, offset 0)
- cubic_arithmetic / seed 4242: rejected (train MAE 1.199e-12 -> 1.199e-12, validation MAE 1.181e-12 -> 1.181e-12, scale 1, offset 0)
- trig_poly_mixed / seed 1337: rejected (train MAE 2.645e-9 -> 2.645e-9, validation MAE 2.984e-9 -> 2.984e-9, scale 1, offset 0)
- trig_poly_mixed / seed 2026: rejected (train MAE 2.678e-9 -> 2.678e-9, validation MAE 2.940e-9 -> 2.940e-9, scale 1, offset 0)
- trig_poly_mixed / seed 4242: rejected (train MAE 2.678e-9 -> 2.678e-9, validation MAE 2.940e-9 -> 2.940e-9, scale 1, offset 0)
- rational_shifted / seed 1337: accepted (train MAE 1.478e-5 -> 8.055e-6, validation MAE 1.506e-5 -> 8.074e-6, scale 0.999961, offset 4.823e-17)
- rational_shifted / seed 2026: accepted (train MAE 1.478e-5 -> 8.055e-6, validation MAE 1.506e-5 -> 8.074e-6, scale 0.999961, offset 4.823e-17)
- rational_shifted / seed 4242: accepted (train MAE 1.478e-5 -> 8.055e-6, validation MAE 1.506e-5 -> 8.074e-6, scale 0.999961, offset 4.823e-17)

## GCL Hybrid Sparse Selection

- quadratic_arithmetic / seed 1337: selected 2/256 terms, validation MAE 3.651 -> 7.948e-13, fit MAE 4.364e-13.
- quadratic_arithmetic / seed 2026: selected 2/256 terms, validation MAE 3.651 -> 7.948e-13, fit MAE 4.364e-13.
- quadratic_arithmetic / seed 4242: selected 2/256 terms, validation MAE 3.651 -> 7.948e-13, fit MAE 4.364e-13.
- cubic_arithmetic / seed 1337: selected 2/256 terms, validation MAE 1.245 -> 1.021e-12, fit MAE 6.797e-13.
- cubic_arithmetic / seed 2026: selected 2/256 terms, validation MAE 1.245 -> 1.021e-12, fit MAE 6.797e-13.
- cubic_arithmetic / seed 4242: selected 2/256 terms, validation MAE 1.245 -> 1.021e-12, fit MAE 6.797e-13.
- trig_poly_mixed / seed 1337: selected 3/256 terms, validation MAE 0.883998 -> 4.230e-12, fit MAE 2.085e-12.
- trig_poly_mixed / seed 2026: selected 3/256 terms, validation MAE 0.883998 -> 4.230e-12, fit MAE 2.085e-12.
- trig_poly_mixed / seed 4242: selected 3/256 terms, validation MAE 0.883998 -> 4.230e-12, fit MAE 2.085e-12.
- rational_shifted / seed 1337: selected 2/256 terms, validation MAE 0.384368 -> 7.952e-12, fit MAE 3.989e-12.
- rational_shifted / seed 2026: selected 2/256 terms, validation MAE 0.384368 -> 7.952e-12, fit MAE 3.989e-12.
- rational_shifted / seed 4242: selected 2/256 terms, validation MAE 0.384368 -> 7.952e-12, fit MAE 3.989e-12.

## Failed Runs

- None

