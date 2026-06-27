# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `1/[75, 1393, 2126, 2469, 2944, 4288, 6099, 6803, 7551, 7821, 8715, 9325, 9689, 13343, 14111, 14277]`
- raw oracle calls: `188193`
- validation R2: `0.9968086229936512`
- validation MAE: `0.0013585229495551043`
- subspace min principal cosine: `0.999989791877386`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
