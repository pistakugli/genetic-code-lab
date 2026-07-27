# Blind 4096D SwiGLU Oracle Decompiler Sweep

- pass: `false`
- cases: `0/1`
- min R2: `0.9976440956596437`
- max MAE: `0.000014050645102733393`
- max worst error: `0.00011631635935048469`
- min subspace principal cosine: `0.9996985570973159`
- min post-hoc paired atom joint cosine: `0.08378074120982693`
- min paired atoms above threshold: `8`
- min source cancellation fraction: `0.6935123485059438`
- minimum source pair correlation: `-0.04442603547600881`
- max unit count: `16`
- max atom count: `16`
- total raw oracle calls: `1150592`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- layer15-units7885-2037-13787-2761-5013-3496-5314-9185-2434-2487-223-5617-12804-5572-12373-10618: `fail pairedAtomMinJointCosine=0.08378074120982693<0.99; pairedAtomCount=8<16` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer15-units7885-2037-13787-2761-5013-3496-5314-9185-2434-2487-223-5617-12804-5572-12373-10618-16atom-output2048-q000001-cubic-richardson-2048h512s2-0-1-4-symmetric-multiscale-coordinate-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
