# Blind 4096D SwiGLU Oracle Decompiler

- pass: `false`
- model: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`
- layer/unit: `1/[27, 57, 413, 460, 500, 618, 778, 1141, 1607, 1970, 2062, 2215, 2529, 2847, 3530, 3627, 3679, 3786, 4033, 4186, 4236, 4624, 4906, 5014, 5041, 5409, 6455, 7283, 7434, 7598, 7663, 7720]`
- oracle mode: `selected-unit-sum`
- output channel: `None`
- raw oracle calls: `252481`
- validation R2: `0.9882781253283933`
- validation MAE: `0.03027649942396946`
- subspace min principal cosine: `0.9997704576111829`
- post-hoc paired atom min joint cosine: `0.18922121075296774`
- post-hoc paired atoms >= 0.99: `13/32`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
