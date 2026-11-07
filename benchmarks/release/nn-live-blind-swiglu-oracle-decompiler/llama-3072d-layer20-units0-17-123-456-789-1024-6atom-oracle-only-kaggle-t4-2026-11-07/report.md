# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`
- layer/unit: `20/[0, 17, 123, 456, 789, 1024]`
- oracle mode: `selected-unit-sum`
- output channel: `None`
- raw oracle calls: `88864`
- validation R2: `0.9999999974031757`
- validation MAE: `2.6818746056832303e-05`
- subspace min principal cosine: `0.9999999941481496`
- post-hoc paired atom min joint cosine: `0.9999999997750515`
- post-hoc paired atoms >= 0.99: `6/6`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
