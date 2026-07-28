# Blind 4096D SwiGLU Oracle Decompiler

- pass: `false`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `9/[14239, 1840, 6580, 4551, 5475, 12804, 724, 9560, 9801, 1760, 12603, 9156, 12075, 10781, 6129, 1540]`
- oracle mode: `down-proj-output-channel`
- output channel: `99`
- raw oracle calls: `4096`
- validation R2: `0.994055079928962`
- validation MAE: `6.28570627115881e-06`
- subspace min principal cosine: `0.9853595369461892`
- post-hoc paired atom min joint cosine: `-0.06442893806971481`
- post-hoc paired atoms >= 0.99: `11/16`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
