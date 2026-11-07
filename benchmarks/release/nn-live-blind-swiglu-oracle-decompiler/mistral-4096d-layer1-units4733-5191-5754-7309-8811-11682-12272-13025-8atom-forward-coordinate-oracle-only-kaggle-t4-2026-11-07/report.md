# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `1/[4733, 5191, 5754, 7309, 8811, 11682, 12272, 13025]`
- oracle mode: `selected-unit-sum`
- output channel: `None`
- raw oracle calls: `81297`
- validation R2: `0.9977641328690762`
- validation MAE: `0.0006757192513433087`
- subspace min principal cosine: `0.9999216423199736`
- post-hoc paired atom min joint cosine: `0.39669555803489537`
- post-hoc paired atoms >= 0.99: `1/8`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
