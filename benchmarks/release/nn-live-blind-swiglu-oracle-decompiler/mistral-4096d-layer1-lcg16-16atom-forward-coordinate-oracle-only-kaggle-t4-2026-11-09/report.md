# Blind 4096D SwiGLU Oracle Decompiler

- pass: `false`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `1/[279, 569, 3125, 3222, 4733, 5191, 5754, 7309, 8561, 8811, 9962, 11673, 11682, 12272, 13025, 14251]`
- oracle mode: `selected-unit-sum`
- output channel: `None`
- raw oracle calls: `167841`
- validation R2: `0.9993364450122835`
- validation MAE: `0.00016712912679688318`
- subspace min principal cosine: `0.9999992792391609`
- post-hoc paired atom min joint cosine: `0.28734056024778554`
- post-hoc paired atoms >= 0.99: `4/16`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
