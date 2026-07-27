# Blind 3072D SwiGLU Oracle Decompiler Sweep

- pass: `true`
- cases: `1/1`
- min R2: `0.9995440406106985`
- max MAE: `0.004315024051586014`
- max worst error: `0.04367946287594893`
- min subspace principal cosine: `0.9999952999253275`
- min post-hoc paired atom joint cosine: `0.992509414607041`
- min paired atoms above threshold: `16`
- max unit count: `16`
- max atom count: `16`
- total raw oracle calls: `261057`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- layer20-units740-1537-1584-1591-1737-1968-3484-3754-3909-4068-4357-5006-5722-5893-7187-7220: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer20-units740-1537-1584-1591-1737-1968-3484-3754-3909-4068-4357-5006-5722-5893-7187-7220-16atom-q000001-cubic-richardson-2048h512s0-35-0-25-forward-coordinate-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
