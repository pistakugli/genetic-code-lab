# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`
- layer/unit: `1/[9659]`
- oracle mode: `selected-unit-sum`
- output channel: `None`
- raw oracle calls: `16448`
- validation R2: `0.9999999991275736`
- validation MAE: `5.085066292186929e-06`
- subspace min principal cosine: `0.9999999992629485`
- post-hoc paired atom min joint cosine: `0.9999999999995247`
- post-hoc paired atoms >= 0.99: `1/1`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
