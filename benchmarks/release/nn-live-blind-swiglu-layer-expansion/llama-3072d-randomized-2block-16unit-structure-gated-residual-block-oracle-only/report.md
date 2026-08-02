# Live Blind SwiGLU Layer Expansion

- pass: `true`
- cases: `2/2`
- max expanded unit count: `16`
- min R2: `0.9999855916796467`
- max MAE: `0.0007727211891917342`
- max worst error: `0.006249561164756855`
- min subspace principal cosine: `0.9999530973154566`
- min post-hoc paired atom joint cosine: `0.9999610978737026`
- min paired atoms above threshold: `16`
- min source cancellation fraction: `0.6810587415472`
- minimum source pair correlation: `-0.04464800348325854`
- total raw oracle calls: `461398`
- access invariant: `true`

- llama layer 1 16-unit: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer1-16unit-expansion-oracle-only/report.json)
- llama layer 15 16-unit: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer15-16unit-expansion-oracle-only/report.json)
