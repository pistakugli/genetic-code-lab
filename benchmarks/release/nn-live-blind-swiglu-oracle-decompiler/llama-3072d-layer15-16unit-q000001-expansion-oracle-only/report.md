# Blind 3072D SwiGLU Oracle Decompiler Sweep

- pass: `true`
- cases: `1/1`
- min R2: `0.9992051540834496`
- max MAE: `0.005833540487146457`
- max worst error: `0.06428347992920569`
- min subspace principal cosine: `0.9999457028618801`
- min post-hoc paired atom joint cosine: `0.9937228873084201`
- min paired atoms above threshold: `16`
- max unit count: `16`
- max atom count: `16`
- total raw oracle calls: `142241`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- layer15-units460-618-778-1141-1279-2062-3627-3679-4186-5014-5041-5409-6455-7434-7598-7663: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer15-units460-618-778-1141-1279-2062-3627-3679-4186-5014-5041-5409-6455-7434-7598-7663-16atom-q000001-forward-coordinate-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
