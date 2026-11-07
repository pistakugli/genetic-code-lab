# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`
- layer/unit: `17/[7012]`
- oracle mode: `selected-unit-sum`
- output channel: `None`
- raw oracle calls: `14400`
- validation R2: `0.9999999998536622`
- validation MAE: `1.7092885884392319e-06`
- subspace min principal cosine: `0.9999999999500737`
- post-hoc paired atom min joint cosine: `0.9999999999996976`
- post-hoc paired atoms >= 0.99: `1/1`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
