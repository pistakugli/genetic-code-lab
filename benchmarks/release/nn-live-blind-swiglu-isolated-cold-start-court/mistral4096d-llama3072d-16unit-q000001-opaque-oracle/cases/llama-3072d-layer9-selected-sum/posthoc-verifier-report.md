# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`
- layer/unit: `9/[958, 1532, 1647, 1765, 2755, 3182, 3856, 4191, 4371, 4373, 5442, 7098, 7145, 7305, 7679, 7685]`
- oracle mode: `selected-unit-sum`
- output channel: `None`
- raw oracle calls: `4096`
- validation R2: `0.9999998556648195`
- validation MAE: `0.00031001468866505917`
- subspace min principal cosine: `0.9999997806246076`
- post-hoc paired atom min joint cosine: `0.9999999400198436`
- post-hoc paired atoms >= 0.99: `16/16`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
