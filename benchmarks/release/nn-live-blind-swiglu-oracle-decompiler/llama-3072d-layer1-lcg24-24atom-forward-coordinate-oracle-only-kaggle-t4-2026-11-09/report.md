# Blind 4096D SwiGLU Oracle Decompiler

- pass: `false`
- model: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`
- layer/unit: `1/[159, 325, 1395, 1786, 1841, 2373, 2531, 2704, 2966, 3203, 3288, 4020, 4176, 4892, 5014, 5035, 5108, 5583, 5692, 6670, 6675, 7012, 7443, 8143]`
- oracle mode: `selected-unit-sum`
- output channel: `None`
- raw oracle calls: `232975`
- validation R2: `0.41746598284409187`
- validation MAE: `0.19178783132070099`
- subspace min principal cosine: `0.002378974260487603`
- post-hoc paired atom min joint cosine: `-0.07675133449818361`
- post-hoc paired atoms >= 0.99: `0/24`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
