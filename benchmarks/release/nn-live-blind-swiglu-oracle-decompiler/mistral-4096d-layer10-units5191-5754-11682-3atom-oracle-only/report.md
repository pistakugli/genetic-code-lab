# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `10/[5191, 5754, 11682]`
- raw oracle calls: `61968`
- validation R2: `0.9962941439080226`
- validation MAE: `0.0005119471166077025`
- subspace min principal cosine: `0.9999999878109772`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
