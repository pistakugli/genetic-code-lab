# Live Blind SwiGLU Layer Expansion

> Citation warning: this report's delegated citation(s) point at a shared sweep report that has since been overwritten, and predate digest pinning, so the numbers below cannot be verified against their source. See citationEvidence in report.json.

- pass: `true`
- cases: `1/1`
- max expanded unit count: `16`
- min R2: `0.999647819803404`
- max MAE: `0.0000058047784347981145`
- max worst error: `0.000030732777192760455`
- min subspace principal cosine: `0.9996985570973159`
- min post-hoc paired atom joint cosine: `0.9991907329594327`
- min paired atoms above threshold: `16`
- min source cancellation fraction: `0.684963528861127`
- minimum source pair correlation: `-0.04406152273669756`
- total raw oracle calls: `101538`
- access invariant: `true`

- mistral layer 15 16-unit: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer15-16unit-output2048-q000001-expansion-oracle-only/report.json)
