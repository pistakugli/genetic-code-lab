# Blind 3072D SwiGLU Oracle Decompiler Sweep

- pass: `true`
- cases: `1/1`
- min R2: `0.9989172523599217`
- max MAE: `0.0066246793134105576`
- max worst error: `0.07162945059351677`
- min subspace principal cosine: `0.9998518967545832`
- min post-hoc paired atom joint cosine: `0.9928899385272274`
- min paired atoms above threshold: `16`
- max unit count: `16`
- max atom count: `16`
- total raw oracle calls: `142241`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- layer20-units740-1537-1584-1591-1737-1968-3484-3754-3909-4068-4357-5006-5722-5893-7187-7220: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer20-units740-1537-1584-1591-1737-1968-3484-3754-3909-4068-4357-5006-5722-5893-7187-7220-16atom-q000001-forward-coordinate-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
