# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `1/[4733, 5191, 7309, 8811, 11682, 12272]`
- raw oracle calls: `115488`
- validation R2: `0.9952423306790068`
- validation MAE: `0.0009230608551877102`
- subspace min principal cosine: `0.9999999518366974`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
