# Blind 4096D SwiGLU Oracle Decompiler

- pass: `true`
- model: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`
- layer/unit: `15/[662, 771, 1394, 1591, 1603, 2283, 2528, 2967, 3619, 3700, 4221, 4583, 4734, 4917, 5845, 6203]`
- raw oracle calls: `261057`
- validation R2: `0.9996515769035299`
- validation MAE: `0.0038307358859897135`
- subspace min principal cosine: `0.9999962539759294`
- post-hoc paired atom min joint cosine: `0.9929854299572484`
- post-hoc paired atoms >= 0.99: `16/16`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
