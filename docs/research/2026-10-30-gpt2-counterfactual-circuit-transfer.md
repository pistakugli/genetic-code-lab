# GPT-2 counterfactual circuit transfer

## Status and scope

This report tests whether frozen GPT-2 ordered residual paths carry predictive
information that transfers between factually counterfactual prompts. It is an
**instrumented local-model experiment**: the runner accesses GPT-2 activations
and patches them through module hooks. It is not a black-box decompilation
result.

The source is the existing 50-passed-fact ordered-path artifact. Prompts,
answers, discovered paths, model weights, and previous experiment outputs are
not modified. Only independently recorded answers and exact prompt templates
are used to form pairs.

## Mathematical measurements

For a factual prompt context \(x\) and a teacher-forced answer token sequence
\(y=(y_1,\ldots,y_m)\), define its mean token log-likelihood as

\[
\ell(y\mid x)=\frac{1}{m}\sum_{t=1}^{m}
\log p_\theta(y_t\mid x,y_{<t}).
\]

For a donor fact \(A\) and recipient fact \(B\), the recipient answer contrast
is

\[
C_{A,B}(x_B)=\ell(y_A\mid x_B)-\ell(y_B\mid x_B).
\]

The transfer statistic is the contrast change after patching the donor's
frozen path activations into the recipient:

\[
\Delta C_{A\rightarrow B}
=C_{A,B}(x_B;\operatorname{patch}(P_A))
-C_{A,B}(x_B;\operatorname{unmodified}).
\]

A positive value indicates movement toward the donor answer relative to the
recipient answer under teacher forcing. It does **not** show that the full
donor circuit is sufficient, nor that the path is uniquely responsible.

For necessity, the runner removes the frozen path in its original factual
context and records the change in answer-sequence likelihood and greedy token.
For recovery, it replaces patched values with the recipient's original values
and compares that replay with an identity-activation intervention installed at
the same sites. The identity control exposes numerical differences caused by
the intervention hooks themselves.

Structural controls preserve donor-path component count, type, layer, and
aligned token positions while replacing units with random alternatives.
Attention magnitude is the L2 norm of the raw per-head input slice to GPT-2's
attention output projection. MLP magnitude is the absolute post-GELU neuron
activation. Both are averaged across the path's source positions. The
pre-specified relative tolerance is 20%; each direction samples 100 unique
structural groups when available. Specificity inference requires at least 95
fully magnitude-matched controls, at least 20 unique groups, and a selected
effect at or above the 95th empirical control percentile. Invalid control
coverage is reported as unresolved; it does not relax the tolerance.

Pair-level uncertainty resamples factual endpoints rather than treating both
directions or overlapping unordered pairs as independent. Benjamini-Hochberg
correction is applied across relation-family tests only when at least ten
independent facts have valid magnitude-matched controls.

## Identifiability and limitations

The experiment identifies **interventional changes in measured GPT-2 output
distributions for these prompts and these hooked component coordinates**.
Exact matching of fixed-text token IDs prevents a path value from being copied
onto a token in the variable fact slot or an unrelated text position.

The experiment does not identify a unique semantic representation for a
neuron, prove that a transferred path is complete, distinguish shared
relation-processing from generic prediction when controls are invalid, or
generalize beyond the source facts and model. A failed transfer in one
direction is not evidence that the path carries no information in the reverse
direction. A positive ablation effect is influence, not by itself necessity.

## Pair inventory

| Relation family | Facts | Eligible unordered pairs |
|---|---:|---:|
| Country → capital | 39 | 741 |
| Gendered parent term | 2 | 1 |
| Occupation description → profession | 2 | 1 |
| Planet-size extreme | 2 | 1 |
| **Total** | **45** | **744** |

Five of the 50 passed facts are excluded because they do not belong to an
independently derivable repeated prompt-template family. The 33 skipped source
rows remain excluded from the passed-fact experiment. No candidate pair failed
exact fixed-text alignment in the independently derived inventory.

## Ten-pair feasibility pilot

The pilot uses the first ten eligible pairs in source order: France paired
with Germany, Italy, Spain, Portugal, Greece, Egypt, Japan, China, Russia, and
Canada. It contains 20 transfer directions and evaluates all 45 pair-eligible
facts for original-state and path-removal measurements.

| Criterion | Pilot measurement |
|---|---:|
| Eligible pairs / tested pairs | 744 / 10 |
| Exact prompt alignments | 10 / 10 |
| Activation transplants applied | 20 / 20 directions |
| Positive A→B answer-contrast changes | 10 / 10 |
| Positive B→A answer-contrast changes | 5 / 10 |
| Facts with positive answer-likelihood drop after path removal | 44 / 45 |
| Correct baseline greedy predictions strictly changed by path removal | 3 / 45 |
| Positive donor-answer likelihood changes | 18 / 20 directions |
| Recipient recovery verified against identity-hook control | 20 / 20 directions |
| Directions with valid structural and magnitude controls | 0 / 20 |
| Directions passing the specificity criterion | 0 / 20 |

Across the 20 directed pilot observations, the mean path-transfer contrast
change was **2.529 mean-token-log-likelihood units** (95% fact-endpoint
cluster-bootstrap CI **[1.893, 3.216]**; centered bootstrap \(p=0.00062\),
Benjamini-Hochberg \(q=0.00062\) across the single estimable path-transfer
family). Mean donor-answer likelihood change was **2.011** (95% cluster CI
**[1.579, 2.481]**). Only 3,212 of 5,000 bootstrap replicates had a nonzero
pair weight. These are pilot path-transfer effects, not specificity results:
all 20 directions lacked valid amplitude-matched controls, so the matched
path-minus-control confidence interval and \(p\)-value are intentionally not
estimated.

The pilot establishes directional transfer and path-removal influence for
some tested contexts, but it does not establish path specificity. The selected
attention-head activations are often outside the 20% magnitude range of every
alternative head in their layer. The pilot therefore has no valid
path-versus-control confidence interval or significance claim. The
unvalidated mean path-minus-control contrast is descriptive only.

The pilot's CUDA run used an NVIDIA GeForce RTX 3070 with 8,214,085,632 bytes
of reported total memory and 746,502,144 bytes peak allocated GPU memory.
Model load time was 0.46 seconds, intervention time 6.98 seconds, and total
runtime 11.24 seconds for this run. These figures are measured for this
machine and run; they are not a GPU speedup comparison.

## Complete paired campaign

The frozen full campaign tested all **744/744** eligible unordered pairs
(1,488 transfer directions). Every pair passed exact fixed-text token
alignment; no candidate pair was rejected for alignment. Five passed facts
were excluded because no repeated prompt-template family could be derived.
The 33 skipped source rows were not part of this experiment.

| Criterion | Full-campaign measurement |
|---|---:|
| Successful activation transplants | 1,488 / 1,488 directions |
| Positive A→B donor-answer contrast changes | 462 / 744 |
| Positive B→A donor-answer contrast changes | 394 / 744 |
| Positive donor-answer mean-token-likelihood changes | 985 / 1,488 |
| Facts with a positive answer-likelihood drop after path removal | 44 / 45 |
| Correct baseline-greedy facts strictly changed by path removal | 3 / 45 |
| Restorations verified against recipient identity-hook controls | 1,488 / 1,488 |
| Directions with valid structural and magnitude controls | 38 / 1,488 |
| Directions passing the frozen per-direction specificity criterion | 37 / 1,488 |

Paired uncertainty resamples fact endpoints and keeps the frozen 5,000
replicate procedure. Benjamini-Hochberg adjustment is applied separately to
path-transfer and valid-control contrast tests. Values below are measured
mean-token-log-likelihood contrasts; confidence intervals are 95% endpoint
cluster-bootstrap intervals.

| Relation family | Facts / directions | Mean path-transfer contrast (95% CI; q) | Mean donor-answer likelihood change (95% CI) | Valid controls | Mean path − control (95% CI; q) |
|---|---:|---|---|---:|---|
| Country → capital | 39 / 1,482 | 1.3311 [0.6443, 2.0802]; q=0.000400 | 1.0716 [0.5393, 1.6659] | 38 | 0.2585 [0.2157, 0.3035]; q=0.000637 |
| Gendered parent term | 2 / 2 | 0.1787; not estimable with 2 facts | 0.1122; not estimable with 2 facts | 0 | not estimable |
| Occupation description → profession | 2 / 2 | 1.2077; not estimable with 2 facts | 0.7672; not estimable with 2 facts | 0 | not estimable |
| Planet-size extreme | 2 / 2 | 0.0157; not estimable with 2 facts | −0.1208; not estimable with 2 facts | 0 | not estimable |

Within the 38 country-capital directions whose controls met the frozen
structural and activation-magnitude requirements, 37 selected paths exceeded
the 95th empirical control percentile. The family-level path-minus-control
contrast is positive after correction. This is evidence of specificity only
for this small, control-valid subset; **1,444/1,482** country-capital
directions do not have valid magnitude-matched controls. It does not justify
generalizing specificity to all paths or all relation families. The three
two-fact families support descriptive observations only and have no valid
control-based inference.

The measurements distinguish several outcomes rather than collapsing them:
path activation was transplanted and restoration replayed in every direction;
transfer favored the donor on fewer than two thirds of directions; positive
path-removal answer-likelihood effects were common, but strict correct-greedy
necessity was observed for only 3 of 45 facts. Positive donor-answer likelihood
change in 985 directions is directional transfer evidence, **not** proof that
the path is a complete sufficient circuit. The independent computation
classification therefore identifies corrected positive fact-specific
transfer evidence in the country-capital family while leaving the other
families unresolved; it does not infer a semantic representation or complete
circuit.

The full run used CUDA on an NVIDIA GeForce RTX 3070. Measured model loading
was **0.477 s**, tracing and interventions **538.056 s**, and total runtime
**643.991 s**. Peak allocated VRAM was **787,323,904 bytes** (peak reserved
**970,981,376 bytes**; device memory **8,214,085,632 bytes**). No CPU-versus-GPU
speedup comparison was performed. The independent verifier replayed all
1,488 directions, and separately replayed the corrected ten-pair pilot.

The complete output directory contains the machine-readable report, a
pair-by-pair explanation report, manifest, and raw NPZ activation archive
(approximately 2.8 GB; the JSON report is approximately 308 MB). The
finalizer derived uncertainty statistics and pair-level necessity explanations
from the raw direction results; the verifier independently recomputed the
bootstrap summaries and multiple-comparison adjustments before replaying the
model. Focused tests (8/8), Python compilation, TypeScript checking, and the
project build passed.

## Reproducibility artifacts

- Runner: [gpt2_counterfactual_transfer.py](../../tools/llm_reader/gpt2_counterfactual_transfer.py)
- Independent verifier: [verify_gpt2_counterfactual_transfer.py](../../tools/llm_reader/verify_gpt2_counterfactual_transfer.py)
- Focused tests: [test_gpt2_counterfactual_transfer.py](../../tests/test_gpt2_counterfactual_transfer.py)
- Pilot report, JSON, raw NPZ, and manifest:
  `benchmarks/release/nn-live-gpt2-counterfactual-circuit-transfer/pilot-archive-corrected/`
- Complete-pair report, JSON, raw NPZ, and manifest:
  `benchmarks/release/nn-live-gpt2-counterfactual-circuit-transfer/all-pairs/`
- Complete-pair finalizer:
  [finalize_gpt2_counterfactual_transfer.py](../../tools/llm_reader/finalize_gpt2_counterfactual_transfer.py)
- The independent verifier checks the original source hash and the frozen
  hashes of the previous autoregressive experiment before accepting a result.
