# Blind 3072D SwiGLU Oracle Decompiler Sweep (Kaggle T4)

- pass: `True`
- cases: `8/8`
- min R2: `0.9999988351151362`
- max MAE: `0.00014784672588079575`
- max worst error: `0.0019455556523726436`
- min subspace principal cosine: `0.9999994421367867`
- max unit count: `1`
- total raw oracle calls: `115200`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- random0-layer1-unit4176: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer1-unit4176-oracle-only-kaggle-t4-2026-11-07/report.json)
- random1-layer17-unit7012: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer17-unit7012-oracle-only-kaggle-t4-2026-11-07/report.json)
- random2-layer9-unit2966: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer9-unit2966-oracle-only-kaggle-t4-2026-11-07/report.json)
- random3-layer22-unit3288: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer22-unit3288-oracle-only-kaggle-t4-2026-11-07/report.json)
- random4-layer25-unit4892: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer25-unit4892-oracle-only-kaggle-t4-2026-11-07/report.json)
- random5-layer19-unit6670: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer19-unit6670-oracle-only-kaggle-t4-2026-11-07/report.json)
- random6-layer0-unit1786: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer0-unit1786-oracle-only-kaggle-t4-2026-11-07/report.json)
- random7-layer6-unit8143: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer6-unit8143-oracle-only-kaggle-t4-2026-11-07/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
