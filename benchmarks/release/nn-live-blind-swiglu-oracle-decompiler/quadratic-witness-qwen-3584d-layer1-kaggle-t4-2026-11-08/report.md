# SwiGLU quadratic-identity witness (oracle-only)

- pass: `True`
- model: `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` layer `1`
- oracle calls: `6145` (f(x)-only; weights never read)

| check | median rel. residual | p90 | max | gate |
| --- | --- | --- | --- | --- |
| homogeneity-t2 | 7.339e-07 | 7.724e-07 | 8.193e-07 | True |
| homogeneity-t3 | 1.307e-06 | 1.387e-06 | 1.490e-06 | True |
| polarization | 1.463e-06 | 1.550e-06 | 1.629e-06 | True |
| parallelogram | 1.045e-06 | 1.109e-06 | 1.161e-06 | True |
| originNormRelative | 0.000e+00 | - | - | True |

Basis: `silu(s) - silu(-s) = s`, so `E(x) = (f(x)+f(-x))/2` is exactly
quadratic for any bias-free SwiGLU block — including quantized
providers. A biased or non-SwiGLU-gated block fails these identities
at O(1) relative error.
