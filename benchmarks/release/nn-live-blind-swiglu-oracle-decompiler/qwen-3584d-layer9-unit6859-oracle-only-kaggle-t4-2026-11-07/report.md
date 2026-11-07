# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`
- layer/unit: `9/[6859]`
- oracle mode: `selected-unit-sum`
- output channel: `None`
- raw oracle calls: `16448`
- validation R2: `0.9999999997961692`
- validation MAE: `2.637762128908353e-06`
- subspace min principal cosine: `0.9999999999149153`
- post-hoc paired atom min joint cosine: `0.9999999999996914`
- post-hoc paired atoms >= 0.99: `1/1`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
