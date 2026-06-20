# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `1/[7309, 8811, 12272]`
- raw oracle calls: `61968`
- validation R2: `0.9962650545162728`
- validation MAE: `0.0004227523831958203`
- subspace min principal cosine: `0.9999999911398408`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
