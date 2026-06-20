# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `27/[4733, 5191]`
- raw oracle calls: `44128`
- validation R2: `0.9996500599502207`
- validation MAE: `0.0001245905154470415`
- subspace min principal cosine: `0.9999999823144813`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
