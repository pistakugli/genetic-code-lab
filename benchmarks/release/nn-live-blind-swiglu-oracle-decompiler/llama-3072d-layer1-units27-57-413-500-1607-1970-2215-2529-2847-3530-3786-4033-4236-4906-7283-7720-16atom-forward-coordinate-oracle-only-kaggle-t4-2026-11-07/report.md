# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`
- layer/unit: `1/[27, 57, 413, 500, 1607, 1970, 2215, 2529, 2847, 3530, 3786, 4033, 4236, 4906, 7283, 7720]`
- oracle mode: `selected-unit-sum`
- output channel: `None`
- raw oracle calls: `233671`
- validation R2: `0.9999834653466863`
- validation MAE: `0.0008139637202629684`
- subspace min principal cosine: `0.999943808842831`
- post-hoc paired atom min joint cosine: `0.9999809819378254`
- post-hoc paired atoms >= 0.99: `16/16`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
