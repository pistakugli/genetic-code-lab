# Blind 4096D SwiGLU Oracle Decompiler

- pass: `false`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `15/[7885, 2037, 13787, 2761, 5013, 3496, 5314, 9185, 2434, 2487, 223, 5617, 12804, 5572, 12373, 10618]`
- oracle mode: `down-proj-output-channel`
- output channel: `2048`
- raw oracle calls: `204705`
- validation R2: `0.17767096458876264`
- validation MAE: `0.00027325402994042477`
- subspace min principal cosine: `0.002014020544312016`
- post-hoc paired atom min joint cosine: `-0.013760827974344282`
- post-hoc paired atoms >= 0.99: `0/16`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
