# Related work: cryptanalytic extraction of GLU feed-forward blocks

How the blind SwiGLU oracle-decompiler lane (nn-live-blind-swiglu-oracle-decompiler)
relates to the closest published result, and where the boundaries are.

## Shi & Fu (arXiv:2608.06631)

Chunhui Shi and Xinwen Fu, "Cryptanalytic Extraction of Isolated Bias-Free GLU
Feed-Forward Blocks by Antipodal Separation" (submitted 6 August 2026), gives
a constructive, multi-stage forward-query recovery primitive for isolated
bias-free GLU blocks: finite-difference curvature supplies gate-direction
candidates, and paired observations at x and -x separate gate magnitude,
orientation, and value-branch coupling (antipodal separation).

Their experiments cover six Qwen3-0.6B layers (3,072 units each), an
8,192-unit LLaMA-3.2-1B subproblem, and a full-dimensional Gemma-3-1B block
(6,912 units, full input and output dimensions). High-precision targets reach
sub-percent median validation error; four finite-precision configurations
stay below 5 percent median error; none reproduces every stored weight. For
the Qwen reconstructions the paper reports 84–97 percent of reconstructed
parameters within one BF16 ULP of the originals, with forward-query counts on
the order of 17.5–28.5 million for some configurations.

Access category: an isolated FFN block queried as a black box — only
numerical f(x) outputs during reconstruction, no weights, gradients, or
internal activations; original parameters are used only after the fact to
grade the recovery. Their own framing: these isolated-block experiments are
not an end-to-end model-API attack, and deriving internal block responses
from final model outputs remains unsolved.

## Same category, different emphasis

| dimension | Shi & Fu | GCL blind SwiGLU lane |
| --- | --- | --- |
| oracle | isolated GLU block, f(x) only | isolated real SwiGLU block, f(x) only |
| models | Qwen3-0.6B, LLaMA-3.2-1B, Gemma-3-1B | Mistral-7B (4096D), LLaMA-3.2-3B (3072D), Qwen2.5-7B (3584D) |
| precision | BF16/FP16/FP32/TF32 experiments | unsloth 4-bit quantized oracle providers |
| unit coverage | thousands of units per block, incl. full-block demos | selected units and small blocks (8–16 units replayed on Kaggle T4) |
| full-block recovery | demonstrated (Gemma) | not demonstrated on this lane |
| query volume | ~17.5–28.5M for some Qwen runs | 498,242 total across the four T4 lanes |
| cross-platform replay | not reported | RTX 3070 ↔ Kaggle T4, identical recorded oracle budgets |
| case selection | not reported in this form | deterministic Park–Miller LCG (seed 260621), asserted against recorded case IDs before any solve |
| access audit | post-hoc comparison | structured per-report audit (oracle-only; weights/hooks/source IDs unused; truth locked until recovery frozen) |

The query-volume row is not a like-for-like comparison: GCL recovers a small
number of units under tight per-case budgets, while their method jointly
recovers large blocks. A fair comparison needs the same model, the same unit
set, the same input space, and the same evaluation; neither side has run that
protocol.

## What GCL's lane adds

- Third-party hardware replay: the recorded RTX 3070 configurations re-run on
  a free Kaggle Tesla T4 with identical oracle-call budgets (162,594 /
  115,200 / 88,864 for the three replay lanes), which rules out
  hardware-specific artifacts in the original evidence.
- Larger models: 3B–7B families (Mistral-7B, LLaMA-3.2-3B, Qwen2.5-7B) under
  4-bit quantized oracle providers, complementing their 0.6B–1B
  high-precision targets.
- Evidence engineering: every case lands with a manifest (argv, environment,
  git commit), an access audit, a claim boundary, and a re-derivable
  aggregate; randomized case selection is regenerated and asserted, not
  re-picked.

## Known gaps against their results

- Coverage: no full-block or thousands-of-units demonstration on this lane;
  the largest block replayed on the T4 is 16 units (lane E). The 3070 host
  holds larger recorded blocks that have not been replayed on the T4.
- Activation families: the lane targets SwiGLU (SiLU-gated) blocks; GeGLU
  families (Gemma) are not covered by this tool.
- Precision: recovery is graded against 4-bit quantized providers, not BF16
  ULP-level precision.

## Open item

A head-to-head protocol on a shared model and unit set (for example
Qwen2.5-7B selected units, matched budgets, matched evaluation), plus a GeGLU
extension of the selected-unit oracle, would make the comparison
quantitative. Neither is in the current lane scope.
