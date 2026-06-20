# Blind 3072D SwiGLU Oracle Decompiler Sweep

- pass: `true`
- cases: `4/4`
- min R2: `0.9999999964505302`
- max MAE: `0.000010737975340964412`
- max worst error: `0.00011067984901980132`
- min subspace principal cosine: `0.9999999979627894`
- total raw oracle calls: `57600`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- layer8-unit123: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer8-unit123-oracle-only/report.json)
- layer16-unit2048: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer16-unit2048-oracle-only/report.json)
- layer20-unit0: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer20-unit0-oracle-only/report.json)
- layer24-unit17: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer24-unit17-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
