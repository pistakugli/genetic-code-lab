# Blind 4096D SwiGLU Oracle Decompiler

- pass: `false`
- model: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`
- layer/unit: `10/[2598, 41, 7247, 7916, 6286, 6262, 5293, 1722, 6914, 8031, 7280, 2458, 327, 3740, 1827, 1612]`
- oracle mode: `selected-unit-sum`
- output channel: `None`
- raw oracle calls: `4096`
- validation R2: `0.9999990744099637`
- validation MAE: `0.0006484885415451176`
- subspace min principal cosine: `0.9999951123701174`
- post-hoc paired atom min joint cosine: `0.9999995557805313`
- post-hoc paired atoms >= 0.99: `16/16`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
