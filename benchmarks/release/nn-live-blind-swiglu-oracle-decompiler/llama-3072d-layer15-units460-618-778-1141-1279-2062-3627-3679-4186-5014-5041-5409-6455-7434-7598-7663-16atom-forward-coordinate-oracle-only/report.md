# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`
- layer/unit: `15/[460, 618, 778, 1141, 1279, 2062, 3627, 3679, 4186, 5014, 5041, 5409, 6455, 7434, 7598, 7663]`
- oracle mode: `selected-unit-sum`
- output channel: `None`
- raw oracle calls: `230765`
- validation R2: `0.9999855916796467`
- validation MAE: `0.0007727211891917342`
- subspace min principal cosine: `0.9999854124669576`
- post-hoc paired atom min joint cosine: `0.9999610978737026`
- post-hoc paired atoms >= 0.99: `16/16`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
