# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`
- layer/unit: `0/[1786]`
- oracle mode: `selected-unit-sum`
- output channel: `None`
- raw oracle calls: `14400`
- validation R2: `0.9999999708600255`
- validation MAE: `2.478640210800991e-05`
- subspace min principal cosine: `0.9999999855967472`
- post-hoc paired atom min joint cosine: `0.9999999999918943`
- post-hoc paired atoms >= 0.99: `1/1`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
