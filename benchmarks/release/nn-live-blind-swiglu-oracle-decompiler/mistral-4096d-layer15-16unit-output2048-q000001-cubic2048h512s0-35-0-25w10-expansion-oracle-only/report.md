# Blind 4096D SwiGLU Oracle Decompiler Sweep

- pass: `false`
- cases: `0/1`
- min R2: `0.9792319337751766`
- max MAE: `0.000040864780470485434`
- max worst error: `0.0005210991674243392`
- min subspace principal cosine: `0.9996985570973159`
- min post-hoc paired atom joint cosine: `-0.061545973396517926`
- min paired atoms above threshold: `0`
- min source cancellation fraction: `0.685456002800046`
- minimum source pair correlation: `-0.04716293159517883`
- max unit count: `16`
- max atom count: `16`
- total raw oracle calls: `1158784`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- layer15-units7885-2037-13787-2761-5013-3496-5314-9185-2434-2487-223-5617-12804-5572-12373-10618: `fail summary.pass=false; r2=0.9792319337751766<0.995; mae=0.000040864780470485434>0.00003; worstError=0.0005210991674243392>0.0003; pairedAtomMinJointCosine=-0.061545973396517926<0.99; pairedAtomCount=0<16` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/mistral-4096d-layer15-units7885-2037-13787-2761-5013-3496-5314-9185-2434-2487-223-5617-12804-5572-12373-10618-16atom-output2048-q000001-cubic-richardson-2048h512s0-35-0-25-symmetric-multiscale-coordinate-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
