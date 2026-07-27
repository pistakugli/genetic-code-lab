# Blind 4096D SwiGLU Oracle Decompiler Sweep

- pass: `true`
- cases: `1/1`
- min R2: `0.9996824964424941`
- max MAE: `0.000005019407967445103`
- max worst error: `0.00002731915870133052`
- min subspace principal cosine: `0.9997226549029208`
- min post-hoc paired atom joint cosine: `0.9989749892979034`
- min paired atoms above threshold: `16`
- min source cancellation fraction: `0.6811694685734337`
- minimum source pair correlation: `-0.035935491081521574`
- max unit count: `16`
- max atom count: `16`
- total raw oracle calls: `107696`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- layer9-units6396-6290-3681-5380-6149-558-5485-9989-9998-11864-2097-8926-8910-2266-8108-2280: `pass` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer9-units6396-6290-3681-5380-6149-558-5485-9989-9998-11864-2097-8926-8910-2266-8108-2280-16atom-output2048-q000001-forward-coordinate-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
