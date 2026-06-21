# Blind 3072D SwiGLU Oracle Decompiler Sweep

- pass: `true`
- cases: `2/2`
- min R2: `0.9995742116828891`
- max MAE: `0.010256079823764875`
- max worst error: `0.07837874541579404`
- min subspace principal cosine: `0.9992840966989263`
- max unit count: `8`
- max atom count: `8`
- total raw oracle calls: `427586`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- random0-layer1-units2704-2966-3288-4176-5035-6675-7012-7443: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-q00025-layer1-units2704-2966-3288-4176-5035-6675-7012-7443-8atom-forward-coordinate-oracle-only/report.json)
- random1-layer16-units159-1786-1841-2531-5108-5692-6670-8143: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-q00025-layer16-units159-1786-1841-2531-5108-5692-6670-8143-8atom-forward-coordinate-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
