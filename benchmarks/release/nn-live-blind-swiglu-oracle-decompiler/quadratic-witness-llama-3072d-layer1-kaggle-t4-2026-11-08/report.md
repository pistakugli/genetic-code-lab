# SwiGLU quadratic-identity witness (oracle-only)

- pass: `True`
- model: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit` layer `1`
- oracle calls: `6145` (f(x)-only; weights never read)

| check | median rel. residual | p90 | max | gate |
| --- | --- | --- | --- | --- |
| homogeneity-t2 | 7.329e-07 | 7.562e-07 | 7.930e-07 | True |
| homogeneity-t3 | 1.242e-06 | 1.297e-06 | 1.383e-06 | True |
| polarization | 1.404e-06 | 1.461e-06 | 1.580e-06 | True |
| parallelogram | 1.171e-06 | 1.213e-06 | 1.278e-06 | True |
| originNormRelative | 0.000e+00 | - | - | True |

Basis: `silu(s) - silu(-s) = s`, so `E(x) = (f(x)+f(-x))/2` is exactly
quadratic for any bias-free SwiGLU block — including quantized
providers. A biased or non-SwiGLU-gated block fails these identities
at O(1) relative error.
