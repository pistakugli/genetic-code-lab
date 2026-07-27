# Blind 4096D SwiGLU Oracle Decompiler Sweep

- pass: `false`
- cases: `0/1`
- min R2: `0.9824053671606762`
- max MAE: `0.00003513618303735926`
- max worst error: `0.0003476888040364096`
- min subspace principal cosine: `0.9997226549029208`
- min post-hoc paired atom joint cosine: `0.006188913816324287`
- min paired atoms above threshold: `0`
- min source cancellation fraction: `0.6811694685734337`
- minimum source pair correlation: `-0.035935491081521574`
- max unit count: `16`
- max atom count: `16`
- total raw oracle calls: `1674880`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- layer9-units6396-6290-3681-5380-6149-558-5485-9989-9998-11864-2097-8926-8910-2266-8108-2280: `fail summary.pass=false; r2=0.9824053671606762<0.995; mae=0.00003513618303735926>0.00003; worstError=0.0003476888040364096>0.0003; pairedAtomMinJointCosine=0.006188913816324287<0.99; pairedAtomCount=0<16` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer9-units6396-6290-3681-5380-6149-558-5485-9989-9998-11864-2097-8926-8910-2266-8108-2280-16atom-output2048-q000001-cubic-richardson-2048h512s0-35-0-25-symmetric-richardson-coordinate-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
