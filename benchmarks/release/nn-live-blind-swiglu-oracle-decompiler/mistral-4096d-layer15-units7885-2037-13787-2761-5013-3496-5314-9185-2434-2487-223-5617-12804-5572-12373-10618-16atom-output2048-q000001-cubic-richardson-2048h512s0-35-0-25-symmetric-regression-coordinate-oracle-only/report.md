# Blind 4096D SwiGLU Oracle Decompiler

- pass: `false`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `15/[7885, 2037, 13787, 2761, 5013, 3496, 5314, 9185, 2434, 2487, 223, 5617, 12804, 5572, 12373, 10618]`
- oracle mode: `down-proj-output-channel`
- output channel: `2048`
- raw oracle calls: `634496`
- validation R2: `0.9125318082557777`
- validation MAE: `8.453385653242291e-05`
- subspace min principal cosine: `0.6622584293375741`
- post-hoc paired atom min joint cosine: `-0.047469815838226376`
- post-hoc paired atoms >= 0.99: `0/16`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
