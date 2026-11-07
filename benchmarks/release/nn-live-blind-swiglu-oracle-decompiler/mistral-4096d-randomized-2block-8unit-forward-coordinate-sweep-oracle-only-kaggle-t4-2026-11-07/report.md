# Blind 4096D SwiGLU Oracle Decompiler Sweep (Kaggle T4)

- pass: `True`
- cases: `2/2`
- min R2: `0.9973158790677202`
- max MAE: `0.0007405548754244283`
- max worst error: `0.01830709845547525`
- min subspace principal cosine: `0.9999216423199736`
- max unit count: `8`
- total raw oracle calls: `162594`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- random0-layer1-units4733-5191-5754-7309-8811-11682-12272-13025: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer1-units4733-5191-5754-7309-8811-11682-12272-13025-8atom-forward-coordinate-oracle-only-kaggle-t4-2026-11-07/report.json)
- random1-layer19-units279-3125-3222-4429-8940-9962-11673-14251: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer19-units279-3125-3222-4429-8940-9962-11673-14251-8atom-forward-coordinate-oracle-only-kaggle-t4-2026-11-07/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
