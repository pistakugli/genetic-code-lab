# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`
- layer/unit: `9/[192, 331, 1612, 5025, 5229, 5288, 5508, 5553, 5623, 5701, 6097, 6372, 6510, 6653, 7591, 7666]`
- raw oracle calls: `261057`
- validation R2: `0.9996022410326608`
- validation MAE: `0.0036226998892415745`
- subspace min principal cosine: `0.9999935586279501`
- post-hoc paired atom min joint cosine: `0.9943894160746654`
- post-hoc paired atoms >= 0.99: `16/16`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
