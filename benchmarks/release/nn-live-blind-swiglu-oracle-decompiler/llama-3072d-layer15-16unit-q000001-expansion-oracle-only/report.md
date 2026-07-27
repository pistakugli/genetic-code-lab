# Blind 3072D SwiGLU Oracle Decompiler Sweep

- pass: `false`
- cases: `0/1`
- min R2: `0.9936416201869898`
- max MAE: `0.015930591937818074`
- max worst error: `0.20086170989863827`
- min subspace principal cosine: `0.9994029302166941`
- min post-hoc paired atom joint cosine: `0.17081204133338923`
- min paired atoms above threshold: `11`
- max unit count: `16`
- max atom count: `16`
- total raw oracle calls: `150433`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- layer15-units662-771-1394-1591-1603-2283-2528-2967-3619-3700-4221-4583-4734-4917-5845-6203: `fail summary.pass=false; r2=0.9936416201869898<0.995; mae=0.015930591937818074>0.015; worstError=0.20086170989863827>0.12; pairedAtomMinJointCosine=0.17081204133338923<0.99; pairedAtomCount=11<16` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer15-units662-771-1394-1591-1603-2283-2528-2967-3619-3700-4221-4583-4734-4917-5845-6203-16atom-q000001-forward-coordinate-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
