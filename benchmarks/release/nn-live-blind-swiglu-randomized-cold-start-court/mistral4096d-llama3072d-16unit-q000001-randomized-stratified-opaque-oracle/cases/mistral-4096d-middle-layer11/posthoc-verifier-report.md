# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- layer/unit: `11/[6707, 5889, 10938, 3561, 5613, 5173, 33, 8243, 8495, 9841, 13408, 14213, 14209, 3289, 1836, 13786]`
- oracle mode: `selected-unit-sum`
- output channel: `None`
- raw oracle calls: `4096`
- validation R2: `0.9999999498066962`
- validation MAE: `5.952318895692205e-06`
- subspace min principal cosine: `0.9999999511163822`
- post-hoc paired atom min joint cosine: `0.9999996141369001`
- post-hoc paired atoms >= 0.99: `16/16`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
