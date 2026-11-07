# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`
- layer/unit: `6/[18831]`
- oracle mode: `selected-unit-sum`
- output channel: `None`
- raw oracle calls: `16448`
- validation R2: `0.9999999997861055`
- validation MAE: `2.9298643553199316e-06`
- subspace min principal cosine: `0.9999999999278857`
- post-hoc paired atom min joint cosine: `0.9999999999996446`
- post-hoc paired atoms >= 0.99: `1/1`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
