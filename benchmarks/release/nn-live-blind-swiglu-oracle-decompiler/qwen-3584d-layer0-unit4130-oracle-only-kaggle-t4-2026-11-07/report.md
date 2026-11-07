# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`
- layer/unit: `0/[4130]`
- oracle mode: `selected-unit-sum`
- output channel: `None`
- raw oracle calls: `16448`
- validation R2: `0.9999999997547944`
- validation MAE: `1.8331322014791152e-06`
- subspace min principal cosine: `0.9999999998787482`
- post-hoc paired atom min joint cosine: `0.9999999999996926`
- post-hoc paired atoms >= 0.99: `1/1`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
