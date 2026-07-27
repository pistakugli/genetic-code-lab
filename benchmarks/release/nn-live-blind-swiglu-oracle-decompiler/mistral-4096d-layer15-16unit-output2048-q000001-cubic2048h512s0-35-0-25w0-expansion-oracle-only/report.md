# Blind 4096D SwiGLU Oracle Decompiler Sweep

- pass: `true`
- cases: `1/1`
- min R2: `0.9995787659698843`
- max MAE: `0.00000616945597616291`
- max worst error: `0.000032305729901930614`
- min subspace principal cosine: `0.9996985570973159`
- min post-hoc paired atom joint cosine: `0.9972788963034496`
- min paired atoms above threshold: `16`
- min source cancellation fraction: `0.6887747348810841`
- minimum source pair correlation: `-0.044888687721043014`
- max unit count: `16`
- max atom count: `16`
- total raw oracle calls: `1279238`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- layer15-units7885-2037-13787-2761-5013-3496-5314-9185-2434-2487-223-5617-12804-5572-12373-10618: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer15-units7885-2037-13787-2761-5013-3496-5314-9185-2434-2487-223-5617-12804-5572-12373-10618-16atom-output2048-q000001-cubic-richardson-2048h512s0-35-0-25-symmetric-multiscale-coordinate-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
