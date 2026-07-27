# Live Blind SwiGLU Layer Expansion

- pass: `false`
- cases: `0/1`
- max expanded unit count: `16`
- min R2: `0.22884734585649635`
- max MAE: `0.00026774910340945506`
- max worst error: `0.0019029208769410196`
- min subspace principal cosine: `0.006418290954695819`
- min post-hoc paired atom joint cosine: `-0.025985500941129176`
- min paired atoms above threshold: `0`
- min source cancellation fraction: `0.6760415238417381`
- minimum source pair correlation: `-0.06872174481518817`
- total raw oracle calls: `322079`
- access invariant: `true`

- mistral layer 15 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer15-16unit-output2048-q000001-expansion-oracle-only/report.json)
