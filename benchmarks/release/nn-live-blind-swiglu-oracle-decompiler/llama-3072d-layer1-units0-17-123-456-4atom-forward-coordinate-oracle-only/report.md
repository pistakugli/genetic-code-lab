# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`
- layer/unit: `1/[0, 17, 123, 456]`
- oracle mode: `selected-unit-sum`
- output channel: `None`
- raw oracle calls: `72017`
- validation R2: `0.9999989656096891`
- validation MAE: `0.0003507732790537803`
- subspace min principal cosine: `0.9999992930409601`
- post-hoc paired atom min joint cosine: `0.999999896454588`
- post-hoc paired atoms >= 0.99: `4/4`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
