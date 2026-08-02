# Blind 3072D SwiGLU Oracle Decompiler Sweep

- pass: `false`
- cases: `0/1`
- min R2: `0.9986197880720584`
- max MAE: `0.006923393240320399`
- max worst error: `0.11110972379914336`
- min subspace principal cosine: `0.9999854124669576`
- min post-hoc paired atom joint cosine: `0.23861138806479312`
- min paired atoms above threshold: `14`
- min source cancellation fraction: `0.6810587415472`
- minimum source pair correlation: `-0.04464800348325854`
- max unit count: `16`
- max atom count: `16`
- total raw oracle calls: `146337`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

- layer15-units460-618-778-1141-1279-2062-3627-3679-4186-5014-5041-5409-6455-7434-7598-7663: `fail summary.pass=false; pairedAtomMinJointCosine=0.23861138806479312<0.99; pairedAtomCount=14<16` (benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/llama-3072d-layer15-units460-618-778-1141-1279-2062-3627-3679-4186-5014-5041-5409-6455-7434-7598-7663-16atom-forward-coordinate-oracle-only/report.json)

Boundary: this is a selected-unit sweep, not full FFN or arbitrary LLM decompilation.
