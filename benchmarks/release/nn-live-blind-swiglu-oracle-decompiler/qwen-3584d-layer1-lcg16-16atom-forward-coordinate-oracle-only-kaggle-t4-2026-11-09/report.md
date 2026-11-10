# Blind 4096D SwiGLU Oracle Decompiler

- pass: `false`
- model: `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`
- layer/unit: `1/[368, 752, 4130, 4258, 6254, 6859, 7603, 9659, 11313, 11644, 13164, 15426, 15437, 16216, 17212, 18831]`
- oracle mode: `selected-unit-sum`
- output channel: `None`
- raw oracle calls: `150945`
- validation R2: `0.9999952005935845`
- validation MAE: `0.0003692109484069494`
- subspace min principal cosine: `0.017854628481951962`
- post-hoc paired atom min joint cosine: `-0.2676409420398668`
- post-hoc paired atoms >= 0.99: `7/16`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
