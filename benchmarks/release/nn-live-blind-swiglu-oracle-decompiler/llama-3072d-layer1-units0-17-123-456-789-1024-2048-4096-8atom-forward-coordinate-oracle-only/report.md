# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`
- layer/unit: `1/[0, 17, 123, 456, 789, 1024, 2048, 4096]`
- oracle mode: `selected-unit-sum`
- output channel: `None`
- raw oracle calls: `74065`
- validation R2: `0.9999932523395926`
- validation MAE: `0.0012646223141371737`
- subspace min principal cosine: `0.9999791660156062`
- post-hoc paired atom min joint cosine: `0.9999995673189114`
- post-hoc paired atoms >= 0.99: `8/8`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
