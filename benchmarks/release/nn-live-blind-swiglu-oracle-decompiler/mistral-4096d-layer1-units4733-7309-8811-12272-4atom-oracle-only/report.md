# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `1/[4733, 7309, 8811, 12272]`
- raw oracle calls: `79808`
- validation R2: `0.9967551564914866`
- validation MAE: `0.0005458994858978024`
- subspace min principal cosine: `0.999999978214083`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
