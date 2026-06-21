# Blind 3072D SwiGLU Oracle Decompiler Sweep

- pass: `true`
- cases: `2/2`
- min R2: `0.9999781080257685`
- max MAE: `0.002327151614302204`
- max worst error: `0.02565795362142126`
- min subspace principal cosine: `0.999936352892748`
- max unit count: `8`
- max atom count: `8`
- total raw oracle calls: `136482`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- random0-layer1-units2704-2966-3288-4176-5035-6675-7012-7443: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer1-units2704-2966-3288-4176-5035-6675-7012-7443-8atom-forward-coordinate-oracle-only/report.json)
- random1-layer16-units159-1786-1841-2531-5108-5692-6670-8143: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer16-units159-1786-1841-2531-5108-5692-6670-8143-8atom-forward-coordinate-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
