# GPT-2 autoregressive causal computation trace

## Result

This is an `instrumented-witness` experiment on the existing local GPT-2
(`GPT2LMHeadModel`), not a black-box decompilation result. It reuses the frozen
50-fact ordered-path artifact, its exact prompts and paths, and its original
model configuration. No teacher parameters, scenario labels, or hidden
teacher data are passed to the trace runner.

The five-case pilot ran first, followed by the frozen five-case regression and
the same autoregressive protocol over all 50 passed source cases. The
independent verifier reproduced the all-50 token contexts and logits from
local GPT-2 and checked the raw activation and intervention records. The
existing five-case artifact also passed its legacy verifier and its four
pre-run SHA-256 checks.

## Frozen protocol and measured access

- Greedy autoregression, recomputing the full context at each prediction;
  maximum 12 generated tokens per case. The 12-token cap is reported as
  truncation, not as a completed natural-language answer.
- Each prediction records the unchanged baseline context, full vocabulary
  logits, top-five tokens, layerwise residual/attention/MLP computations,
  per-head and per-neuron first-order contributions, normalization, and
  component/group intervention results.
- Every intervention is evaluated on that step's unmodified baseline context.
  An altered token is never fed into later baseline steps.
- The frozen discovered path is removed, restored, and ablated component by
  component. The interaction is the group logit drop minus the sum of
  individual drops; full-vocabulary interaction vectors are also archived.
- Exactly 100 deterministic control draws are evaluated per generated token.
  Controls preserve component count, type, layer, and token positions. The
  candidate pools expand by nearest activation magnitude only when the
  tolerance-matched pool cannot supply enough distinct groups; every
  component's magnitude error and every control's match status are retained.
- The frozen strict specificity predicate uses signed target-logit drop:
  percentile at least 95, at least 95 of 100 controls within the 20% magnitude
  tolerance, at least 20 unique groups, and exact structural match. Absolute
  effect percentile is also reported as a descriptive, direction-independent
  measure; it does not replace the strict gate. Causal influence, specificity,
  and sufficiency are distinct: sufficiency was not tested.

The report stores the exact prompt token IDs and generated prefix at every
step. First-order attribution is only a local gradient-times-write
approximation; causal statements rely on intervention results, not activation
size or attention visualizations. Teacher-forced scoring of the frozen source
target is reported separately from the free-running greedy generation.

## All-50 results

| Criterion | Cases passing | Evidence |
|---|---:|---|
| A. Internal computational trace | 50/50 | All 12 transformer layers were captured per prediction. |
| B. Token-by-token greedy reconstruction | 50/50 | All 600 generated IDs matched independently recomputed greedy argmaxes. |
| C. Causal influence | 44/50 | 534/600 steps had selected-path target-logit changes over `1e-6`; six cases lost measurable influence after step 0. |
| D. Specificity against matched controls | 0/50 | Only 4/600 steps met every strict specificity condition; no case met it at every step. |
| E. Interaction measurement | 50/50 | All 600 group-vs-individual interaction values and vocabulary vectors were recorded. |
| F. Actual greedy-prediction explanation | 50/50 | Each recorded prediction has WHAT/HOW/WHY/WHAT ELSE/LIMITATIONS evidence. |
| G. Frozen-source reproducibility | 50/50 | Exact prompts, source paths, token IDs, source hash, and independent logit replay agree. |

The per-case status matrix and all 600 token-level explanations are in the
[machine-generated report](../../benchmarks/release/nn-live-gpt2-autoregressive-causal-trace/all-50/report.md).
The machine-readable measurements and tensor inventory are in
[report.json](../../benchmarks/release/nn-live-gpt2-autoregressive-causal-trace/all-50/report.json)
and [raw-tensors.npz](../../benchmarks/release/nn-live-gpt2-autoregressive-causal-trace/all-50/raw-tensors.npz).

### Causal influence failures

The six cases that fail criterion C are Russia, largest planet, nearest star,
baby dog, medical profession, and school profession. For each, the selected
path changes the first prediction but has no measurable effect above `1e-6`
on steps 1–11 at the frozen source token positions. This supports a
context-local influence claim at step 0, not a persistent circuit claim across
the generated continuation.

### Specificity and interaction

All 600 predictions had 100 unique structural control groups. Only 24/600
steps had at least 95/100 groups within the activation-magnitude tolerance,
and 9/600 met the signed-effect percentile and match-coverage requirements
together. No case met every strict condition at every step. Therefore,
specificity is **not established**. In most source paths,
the selected attention head's activation magnitude had no alternative head
within tolerance; nearest candidates were used and explicitly marked. The
result is not evidence that the frozen paths are selective relative to valid
magnitude-matched controls.

For the 600 interactions, 427 were approximately additive, 85 were
subadditive/overlap-consistent, and 88 were superadditive/synergy-consistent
under the recorded normalization rule. These are interaction measurements,
not proof that a circuit is redundant, synergistic, or sufficient.

On the original prompts, the frozen source fact token was already the greedy
next token in 7/50 cases and ranked in the top five in 23/50 (median rank 7).
The model generated the full 12-token cap for all cases; none stopped at EOS.
The traces explain the actual greedy sequence rather than treating the
source-artifact target as the model's prediction.

## Runtime and artifacts

The all-50 CUDA run used an NVIDIA GeForce RTX 3070 (8 GiB): total runtime
191.94 s, of which 142.13 s was tracing and interventions; peak allocated
VRAM was 632.1 MiB and peak reserved VRAM was 1708.0 MiB. The raw tensor
archive is about 1.38 GB and `report.json` about 266 MB. No CPU/GPU timing
comparison was run, so no GPU speedup is claimed.

The pilot produced 60 predictions across France, Germany, Italy, Portugal, and
Greece on CUDA and was independently verified. The five-case legacy regression
passed, as did 13 focused Python tests, TypeScript checking, and the project
build. The full all-50 report and source-provenance checks passed the
independent verifier.

## Mathematical and empirical boundaries

- **Mathematical/implementation fact:** hook capture exposes GPT-2's recorded
  residual, attention, MLP, normalization, and unembedding path for these
  inputs; the trace checks residual addition and intervention restoration.
- **Empirical result:** the run verifies actual greedy-token replay and local
  causal effects under the stated zeroing interventions. It establishes no
  general circuit semantics.
- **Assumption:** the verifier's installed local `gpt2` weights and tokenizer
  are the model intended by the frozen source experiment; the report records
  model metadata and all prompt token IDs.
- **Limitations:** the model is instrumented with local weights/hooks; the
  result is not oracle-only. Ablations are one-step and context-conditional.
  Discovered-path sufficiency is not tested, matched magnitude specificity
  mostly fails, and all free-running traces stop at the 12-token cap.

## Next experiment

Do not extend this into a sufficiency claim or tune the path. First test
whether a broader, independently declared control-magnitude protocol can
produce enough valid controls for the attention heads while preserving layer,
type, position, and component count. If such a pool is mathematically
unavailable, record specificity as unidentifiable under this path/control
definition rather than relaxing the threshold. Separately, a follow-up may
increase the preregistered generation cap so the answer continuation is not
truncated; it must rerun the full frozen case set and must not mix those
contexts with this 12-token experiment.
