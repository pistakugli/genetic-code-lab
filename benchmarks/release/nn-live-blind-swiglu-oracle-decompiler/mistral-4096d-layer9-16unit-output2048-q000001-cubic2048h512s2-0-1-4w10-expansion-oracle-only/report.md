# Blind 4096D SwiGLU Oracle Decompiler Sweep

- pass: `false`
- cases: `0/1`
- min R2: `0.9927064132277419`
- max MAE: `0.00002294618776565714`
- max worst error: `0.00015101317187022592`
- min subspace principal cosine: `0.9997226549029208`
- min post-hoc paired atom joint cosine: `0.2773761459982943`
- min paired atoms above threshold: `2`
- min source cancellation fraction: `0.6860616086677447`
- minimum source pair correlation: `-0.03201511955079522`
- max unit count: `16`
- max atom count: `16`
- total raw oracle calls: `1674880`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- layer9-units6396-6290-3681-5380-6149-558-5485-9989-9998-11864-2097-8926-8910-2266-8108-2280: `fail summary.pass=false; r2=0.9927064132277419<0.995; pairedAtomMinJointCosine=0.2773761459982943<0.99; pairedAtomCount=2<16` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer9-units6396-6290-3681-5380-6149-558-5485-9989-9998-11864-2097-8926-8910-2266-8108-2280-16atom-output2048-q000001-cubic-richardson-2048h512s2-0-1-4-symmetric-richardson-coordinate-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
