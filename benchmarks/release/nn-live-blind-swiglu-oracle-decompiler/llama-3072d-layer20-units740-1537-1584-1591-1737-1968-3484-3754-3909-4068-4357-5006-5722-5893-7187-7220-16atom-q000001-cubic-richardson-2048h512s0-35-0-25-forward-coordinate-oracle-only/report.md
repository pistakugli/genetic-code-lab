# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`
- layer/unit: `20/[740, 1537, 1584, 1591, 1737, 1968, 3484, 3754, 3909, 4068, 4357, 5006, 5722, 5893, 7187, 7220]`
- raw oracle calls: `261057`
- validation R2: `0.9995440406106985`
- validation MAE: `0.004315024051586014`
- subspace min principal cosine: `0.9999952999253275`
- post-hoc paired atom min joint cosine: `0.992509414607041`
- post-hoc paired atoms >= 0.99: `16/16`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
