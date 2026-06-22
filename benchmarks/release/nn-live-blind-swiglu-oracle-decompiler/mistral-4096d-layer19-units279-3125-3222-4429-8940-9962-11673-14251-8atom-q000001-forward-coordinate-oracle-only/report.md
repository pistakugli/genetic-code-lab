# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `19/[279, 3125, 3222, 4429, 8940, 9962, 11673, 14251]`
- raw oracle calls: `290754`
- validation R2: `0.9958676370633143`
- validation MAE: `0.0009856537193232476`
- subspace min principal cosine: `0.9994407684636628`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
