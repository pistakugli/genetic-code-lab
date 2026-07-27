# Blind 3072D SwiGLU Oracle Decompiler Sweep

- pass: `true`
- cases: `1/1`
- min R2: `0.9996515769035299`
- max MAE: `0.0038307358859897135`
- max worst error: `0.031238018542964846`
- min subspace principal cosine: `0.9999962539759294`
- min post-hoc paired atom joint cosine: `0.9929854299572484`
- min paired atoms above threshold: `16`
- max unit count: `16`
- max atom count: `16`
- total raw oracle calls: `261057`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- layer15-units662-771-1394-1591-1603-2283-2528-2967-3619-3700-4221-4583-4734-4917-5845-6203: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer15-units662-771-1394-1591-1603-2283-2528-2967-3619-3700-4221-4583-4734-4917-5845-6203-16atom-q000001-cubic-richardson-2048h512s0-35-0-25-forward-coordinate-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
