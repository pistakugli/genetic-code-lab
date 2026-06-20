# Blind 3072D SwiGLU Oracle Decompiler Sweep

- pass: `true`
- cases: `2/2`
- min R2: `0.9999999419380987`
- max MAE: `0.00008572753027982732`
- max worst error: `0.0008619591998977327`
- min subspace principal cosine: `0.9999998049413197`
- max unit count: `6`
- max atom count: `6`
- total raw oracle calls: `177728`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- random0-layer1-units2704-2966-4176-5035-6675-7012: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer1-units2704-2966-4176-5035-6675-7012-6atom-oracle-only/report.json)
- random1-layer11-units159-1786-4892-5692-6670-7443: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer11-units159-1786-4892-5692-6670-7443-6atom-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
