# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `1/[4733, 5191, 5754, 7309, 8811, 11682, 12272, 13025]`
- raw oracle calls: `290754`
- validation R2: `0.9957418602955875`
- validation MAE: `0.0009919984371019835`
- subspace min principal cosine: `0.9996066410222786`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
