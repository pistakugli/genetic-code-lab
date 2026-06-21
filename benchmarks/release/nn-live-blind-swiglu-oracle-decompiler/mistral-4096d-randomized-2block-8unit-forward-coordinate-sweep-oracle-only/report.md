# Blind 4096D SwiGLU Oracle Decompiler Sweep

- pass: `true`
- cases: `2/2`
- min R2: `0.9965490748792153`
- max MAE: `0.0008981839822401561`
- max worst error: `0.010140033299851102`
- min subspace principal cosine: `0.9999975326762489`
- max unit count: `8`
- max atom count: `8`
- total raw oracle calls: `162594`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- random0-layer1-units4733-5191-5754-7309-8811-11682-12272-13025: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer1-units4733-5191-5754-7309-8811-11682-12272-13025-8atom-forward-coordinate-oracle-only/report.json)
- random1-layer19-units279-3125-3222-4429-8940-9962-11673-14251: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer19-units279-3125-3222-4429-8940-9962-11673-14251-8atom-forward-coordinate-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
