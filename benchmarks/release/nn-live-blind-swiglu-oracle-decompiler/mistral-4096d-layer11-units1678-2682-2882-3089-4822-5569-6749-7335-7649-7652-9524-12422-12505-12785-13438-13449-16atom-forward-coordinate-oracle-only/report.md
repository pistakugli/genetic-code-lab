# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `11/[1678, 2682, 2882, 3089, 4822, 5569, 6749, 7335, 7649, 7652, 9524, 12422, 12505, 12785, 13438, 13449]`
- raw oracle calls: `188193`
- validation R2: `0.9971375925273832`
- validation MAE: `0.0012542861266610903`
- subspace min principal cosine: `0.9999932242854456`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
