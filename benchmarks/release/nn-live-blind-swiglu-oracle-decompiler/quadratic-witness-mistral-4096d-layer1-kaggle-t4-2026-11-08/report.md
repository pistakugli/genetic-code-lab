# SwiGLU quadratic-identity witness (oracle-only)

- pass: `True`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit` layer `1`
- oracle calls: `6145` (f(x)-only; weights never read)

| check | median rel. residual | p90 | max | gate |
| --- | --- | --- | --- | --- |
| homogeneity-t2 | 8.574e-07 | 8.976e-07 | 1.528e-06 | True |
| homogeneity-t3 | 1.484e-06 | 1.582e-06 | 2.028e-06 | True |
| polarization | 1.694e-06 | 1.828e-06 | 2.598e-06 | True |
| parallelogram | 1.388e-06 | 1.483e-06 | 2.002e-06 | True |
| originNormRelative | 0.000e+00 | - | - | True |

Basis: `silu(s) - silu(-s) = s`, so `E(x) = (f(x)+f(-x))/2` is exactly
quadratic for any bias-free SwiGLU block — including quantized
providers. A biased or non-SwiGLU-gated block fails these identities
at O(1) relative error.
