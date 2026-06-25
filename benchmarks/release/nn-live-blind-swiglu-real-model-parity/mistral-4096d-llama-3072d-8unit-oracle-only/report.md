# NN Live Blind SwiGLU Real-Model Parity

Status: **PASS**

This report reads existing real-model oracle-only Mistral and LLaMA-family SwiGLU decompiler artifacts. It separates real model parity from synthetic teacher claims.

| case | model family | dim | type | q | pass | R2 | MAE | worst | subspace cosine | raw calls |
| --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| mistral-clean-selected-8unit | Mistral-family | 4096 | selected | 0 | true | 0.995341399534 | 0.001122070889 | 0.012468272718 | 0.99999661638 | 85649 |
| mistral-clean-randomized-8unit | Mistral-family | 4096 | randomized | 0 | true | 0.996549074879 | 0.000898183982 | 0.0101400333 | 0.999997532676 | 162594 |
| mistral-q1e-6-randomized-8unit | Mistral-family | 4096 | randomized | 1.000000e-6 | true | 0.995867637063 | 0.000985653719 | 0.014497109405 | 0.999440768464 | 581508 |
| llama-clean-selected-8unit | LLaMA-family | 3072 | selected | 0 | true | 0.999991381826 | 0.001766439125 | 0.016726727076 | 0.999991953911 | 68241 |
| llama-clean-randomized-8unit | LLaMA-family | 3072 | randomized | 0 | true | 0.999978108026 | 0.002327151614 | 0.025657953621 | 0.999936352893 | 136482 |
| llama-q2p5e-4-selected-8unit | LLaMA-family | 3072 | selected | 0.00025 | true | 0.999597610028 | 0.011652554396 | 0.120805323663 | 0.999441573954 | 213793 |
| llama-q2p5e-4-randomized-8unit | LLaMA-family | 3072 | randomized | 0.00025 | true | 0.999574211683 | 0.010256079824 | 0.078378745416 | 0.999284096699 | 427586 |

## Summary

- model families: LLaMA-family, Mistral-family
- dimensions: 3072, 4096
- rows: 7
- min R2: 0.995341399534
- max MAE: 0.011652554396
- max worst error: 0.120805323663
- min subspace principal cosine: 0.999284096699
- access invariant: true

## Boundaries

- This is a real-model parity evidence ledger, not a new decompiler method.
- Mistral-family rows use a local Mistral-family checkpoint artifact; LLaMA-family rows use unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit artifacts.
- Rows prove selected/randomized scalar 8-unit SwiGLU block recovery only; they do not prove all-channel or full-FFN extraction.
- Solver access remains oracle-only in the source reports; source weights/hooks are post-hoc witnesses only.

