# Live Blind SwiGLU Layer Expansion

- pass: `true`
- cases: `1/1`
- max expanded unit count: `4`
- min R2: `0.9999989656096891`
- max MAE: `0.0003507732790537803`
- max worst error: `0.002729982480284221`
- min subspace principal cosine: `0.9999992930409601`
- min post-hoc paired atom joint cosine: `0.999999896454588`
- min paired atoms above threshold: `0`
- min source cancellation fraction: `0.3893537121368381`
- minimum source pair correlation: `-0.021986979484849047`
- total raw oracle calls: `72017`
- access invariant: `true`

- llama layer 1 4-unit: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer1-4unit-expansion-oracle-only/report.json)
