# Blind 4096D SwiGLU Oracle Decompiler Sweep

- pass: `true`
- cases: `1/1`
- min R2: `0.9995827799365143`
- max MAE: `0.000006189436115976275`
- max worst error: `0.00003509105019259542`
- min subspace principal cosine: `0.9996985570973159`
- min post-hoc paired atom joint cosine: `0.9977692945746941`
- min paired atoms above threshold: `16`
- min source cancellation fraction: `0.6935123485059438`
- minimum source pair correlation: `-0.04442603547600881`
- max unit count: `16`
- max atom count: `16`
- total raw oracle calls: `177542`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- layer15-units7885-2037-13787-2761-5013-3496-5314-9185-2434-2487-223-5617-12804-5572-12373-10618: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer15-units7885-2037-13787-2761-5013-3496-5314-9185-2434-2487-223-5617-12804-5572-12373-10618-16atom-output2048-q000001-forward-coordinate-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
