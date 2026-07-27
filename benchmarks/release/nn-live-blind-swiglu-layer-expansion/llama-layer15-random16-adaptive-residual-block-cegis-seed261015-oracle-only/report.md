# Live Blind SwiGLU Layer Expansion

- pass: `false`
- cases: `0/1`
- max expanded unit count: `16`
- min R2: `0.9963198960257796`
- max MAE: `0.011918318574182438`
- max worst error: `0.1351734945368718`
- min subspace principal cosine: `0.9999857637238745`
- total raw oracle calls: `24576`
- access invariant: `true`

- llama layer 15 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer15-16unit-expansion-oracle-only/report.json)
