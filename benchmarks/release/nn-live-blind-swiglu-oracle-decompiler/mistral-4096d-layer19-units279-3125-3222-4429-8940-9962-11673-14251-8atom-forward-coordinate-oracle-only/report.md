# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `19/[279, 3125, 3222, 4429, 8940, 9962, 11673, 14251]`
- raw oracle calls: `81297`
- validation R2: `0.997475549917072`
- validation MAE: `0.0007226585893503762`
- subspace min principal cosine: `0.9999998210173254`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
