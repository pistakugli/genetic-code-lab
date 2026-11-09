# Blind 4096D SwiGLU Oracle Decompiler

- pass: `false`
- model: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`
- layer/unit: `1/[113, 159, 325, 1265, 1379, 1395, 1786, 1841, 1950, 2040, 2101, 2227, 2373, 2531, 2704, 2875, 2966, 3203, 3250, 3288, 3460, 3745, 3829, 3901, 4020, 4176, 4271, 4892, 5014, 5035, 5108, 5583, 5692, 5774, 5940, 5998, 6239, 6670, 6675, 6678, 7012, 7054, 7341, 7443, 7616, 8042, 8142, 8143]`
- oracle mode: `selected-unit-sum`
- output channel: `None`
- raw oracle calls: `259841`
- validation R2: `0.010418748307168268`
- validation MAE: `0.3580148891046451`
- subspace min principal cosine: `0.0012586880504990706`
- post-hoc paired atom min joint cosine: `0.020799058106525676`
- post-hoc paired atoms >= 0.99: `0/48`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
