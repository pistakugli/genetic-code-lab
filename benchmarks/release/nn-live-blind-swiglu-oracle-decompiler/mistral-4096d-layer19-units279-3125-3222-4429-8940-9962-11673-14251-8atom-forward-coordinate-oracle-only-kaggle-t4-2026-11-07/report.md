# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `19/[279, 3125, 3222, 4429, 8940, 9962, 11673, 14251]`
- oracle mode: `selected-unit-sum`
- output channel: `None`
- raw oracle calls: `81297`
- validation R2: `0.9973158790677202`
- validation MAE: `0.0007405548754244283`
- subspace min principal cosine: `0.9999565318122321`
- post-hoc paired atom min joint cosine: `0.26384256847621185`
- post-hoc paired atoms >= 0.99: `1/8`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
