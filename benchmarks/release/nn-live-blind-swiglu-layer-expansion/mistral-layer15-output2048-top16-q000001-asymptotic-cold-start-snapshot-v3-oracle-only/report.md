# Live Blind SwiGLU Layer Expansion

> Citation warning: this report's delegated citation(s) point at a shared sweep report that has since been overwritten, and predate digest pinning, so the numbers below cannot be verified against their source. See citationEvidence in report.json.

- pass: `false`
- cases: `0/1`
- max expanded unit count: `16`
- min R2: `0.9984567857307327`
- max MAE: `0.000011529334245204523`
- max worst error: `0.00006891035988992296`
- min subspace principal cosine: `0.9996985570973159`
- min post-hoc paired atom joint cosine: `0.9923730196343498`
- min paired atoms above threshold: `16`
- min source cancellation fraction: `0.6887747348810841`
- minimum source pair correlation: `-0.044888687721043014`
- total raw oracle calls: `1279238`
- access invariant: `true`

- mistral layer 15 16-unit: `fail sweep summary pass=false` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer15-16unit-output2048-q000001-cubic2048h512s0-35-0-25w0-expansion-oracle-only/report.json)
