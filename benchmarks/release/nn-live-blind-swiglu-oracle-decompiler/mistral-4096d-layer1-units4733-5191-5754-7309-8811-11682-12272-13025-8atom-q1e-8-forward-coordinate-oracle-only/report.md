# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `1/[4733, 5191, 5754, 7309, 8811, 11682, 12272, 13025]`
- raw oracle calls: `85649`
- validation R2: `0.9957594628054579`
- validation MAE: `0.0010088005020659832`
- subspace min principal cosine: `0.9999828597286717`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
