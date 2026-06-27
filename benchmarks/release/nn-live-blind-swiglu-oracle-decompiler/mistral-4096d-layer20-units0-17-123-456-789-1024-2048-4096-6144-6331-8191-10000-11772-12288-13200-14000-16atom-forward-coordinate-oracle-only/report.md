# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `20/[0, 17, 123, 456, 789, 1024, 2048, 4096, 6144, 6331, 8191, 10000, 11772, 12288, 13200, 14000]`
- raw oracle calls: `188193`
- validation R2: `0.9973170168506432`
- validation MAE: `0.0012842742350251879`
- subspace min principal cosine: `0.9999979444870536`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
