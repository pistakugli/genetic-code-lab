# Blind 3072D SwiGLU Oracle Decompiler Sweep

- pass: `true`
- cases: `2/2`
- min R2: `0.9999880522847288`
- max MAE: `0.0012813828353731324`
- max worst error: `0.012340230224598514`
- min subspace principal cosine: `0.9999574008675185`
- max unit count: `6`
- max atom count: `6`
- total raw oracle calls: `97882`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- random0-layer1-units2704-2966-4176-5035-6675-7012: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer1-units2704-2966-4176-5035-6675-7012-6atom-forward-coordinate-oracle-only/report.json)
- random1-layer11-units159-1786-4892-5692-6670-7443: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer11-units159-1786-4892-5692-6670-7443-6atom-forward-coordinate-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
