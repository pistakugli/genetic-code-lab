# Closed-loop archive emergence benchmark

Generated: 2026-06-02T05:39:36.672Z

## Scope

- Runs the real `EvolutionEngine`, not fixed archive fragments.
- Compares the same seeded run with `GCL_SEMANTIC_ARCHIVE=0` and with archive enabled.
- Keeps the semantic refiner enabled in both modes; the measured delta is cross-generation archive recording/injection.
- Disables `GCL_REGRESSION_BASIS_SEED` so this benchmark measures emergence from the evolutionary loop rather than the grammar-derived seed.
- This is a deterministic positive-use-case gate, not a claim that archive-on beats archive-off for every random seed.

## Task

- Formula: `0.7*sin(x0) + 1.3*x1*x1 - 0.4*x0*x1`
- Train cases: 110
- Test cases: 156

## Summary

- Mean archive-off MAE: 0.305295
- Mean archive-on MAE: 0.241591
- Mean lift: 1.830375x
- Success rate: 100%
- Archive injection successes: 17

| Seed | Archive-off MAE | Archive-on MAE | Lift | Archive injections | Success | Retained keys |
|---:|---:|---:|---:|---:|---:|---|
| 3 | 0.326397 | 0.269434 | 1.211415x | 2 | yes | mul(mul(sin(x1),add(x1,x1)),cos(div(cos(x0),add(x1,x1)))), mul(sin(x1),add(x1,x1)), cos(div(cos(x0),add(x1,x1))), x0, add(add(add(add(mul(c:1.4578462,mul(mul(sin(x1),add(x1,x1))... |
| 9 | 0.064169 | 0.020898 | 3.070521x | 3 | yes | add(mul(cos(cos(div(sub(x0,mul(sin(x1),add(mul(x1,x1),x1))),x1))),sin(add(c:6.4995037,sin(sin(x0))))),mul(x1,x1)), add(mul(cos(cos(div(sub(x0,mul(sin(x1),add(mul(x1,x1),x1))),x1... |
| 11 | 0.525319 | 0.434439 | 1.209189x | 12 | yes | mul(mul(div(x1,x0),add(sin(x1),sin(x1))),x0), mul(mul(div(x1,x0),add(sin(sub(x1,c:0.39143807)),sin(x1))),x0), mul(mul(div(x1,x0),x1),x0), x0, add(add(mul(c:1.2997902,mul(mul(div... |

