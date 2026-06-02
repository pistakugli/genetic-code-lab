# Semantic archive composition benchmark

Generated: 2026-06-02T09:54:37.944Z

## Scope

- Measures the semantic archive as a mechanism, not the whole evolutionary loop.
- Each case gives the archive multiple elite fragments, including duplicates and distractors.
- The pass condition requires `GCL_SEMANTIC_ARCHIVE=0` to inject nothing, while the enabled archive synthesizes a lower-error composite from train data only.

## Summary

| Case | Single-best MAE | Archive MAE | Lift | Injected | Success | Retained keys |
|---|---:|---:|---:|---:|---:|---|
| delayed_bivariate_fragments | 0.836622 | 1.557e-11 | 5.373e+10x | yes | yes | mul(x1,x1), x0, sin(x0), mul(x0,x1), mul(c:2.0000000,sin(x0)), x1, cos(x0) |
| duplicate_pressure_market | 0.559954 | 6.177e-12 | 9.065e+10x | yes | yes | x0, mul(x0,x1), x1, sin(x1) |
| rational_trig_composition | 0.301938 | 1.908e-5 | 15822.059039x | yes | yes | div(sin(x0),add(c:1.0000000,mul(x1,x1))), sin(x0), x0, cos(x1) |

## Details

- delayed_bivariate_fragments: best single elite `mul(x1,x1)`, archive disabled injected=false, max error 2.224e-11.
- duplicate_pressure_market: best single elite `x0`, archive disabled injected=false, max error 8.812e-12.
- rational_trig_composition: best single elite `div(sin(x0),add(1,mul(x1,x1)))`, archive disabled injected=false, max error 8.807e-5.

Runtime: 13.742685 ms

