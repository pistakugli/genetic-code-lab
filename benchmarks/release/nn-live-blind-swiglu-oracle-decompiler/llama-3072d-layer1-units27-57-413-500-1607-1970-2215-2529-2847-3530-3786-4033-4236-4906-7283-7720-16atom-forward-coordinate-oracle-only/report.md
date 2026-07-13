# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`
- layer/unit: `1/[27, 57, 413, 500, 1607, 1970, 2215, 2529, 2847, 3530, 3786, 4033, 4236, 4906, 7283, 7720]`
- raw oracle calls: `142241`
- validation R2: `0.9994775776974917`
- validation MAE: `0.004589493073491326`
- subspace min principal cosine: `0.9999482868797673`
- post-hoc paired atom min joint cosine: `0.9971576723055822`
- post-hoc paired atoms >= 0.99: `16/16`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
