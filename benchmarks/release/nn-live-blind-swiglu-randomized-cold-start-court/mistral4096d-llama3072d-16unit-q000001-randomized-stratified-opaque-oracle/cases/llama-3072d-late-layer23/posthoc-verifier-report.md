# Blind 4096D SwiGLU Oracle Decompiler

- pass: `false`
- model: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`
- layer/unit: `23/[176, 501, 1310, 1291, 671, 4333, 1555, 1968, 6573, 2155, 2689, 6009, 3334, 2424, 3864, 7103]`
- oracle mode: `down-proj-output-channel`
- output channel: `1871`
- raw oracle calls: `4096`
- validation R2: `0.9978967623596033`
- validation MAE: `0.0009558063486473441`
- subspace min principal cosine: `0.99999864749214`
- post-hoc paired atom min joint cosine: `0.10087696558935158`
- post-hoc paired atoms >= 0.99: `12/16`

Access audit:

- solverAccess: `oracle-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.
