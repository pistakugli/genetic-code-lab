# Unknown-Order Empirical SwiGLU Recovery

## Purpose

This is the first recorded lane of the recommended next development direction
([`GCL_NEXT_DEVELOPMENT_DIRECTION.md`](GCL_NEXT_DEVELOPMENT_DIRECTION.md)):
blind recovery of a selected SwiGLU block on the target model's **recorded
empirical MLP-input distribution**, with the selected-unit count (the
function's atom count) **hidden from the solver**. The preregistered question
was: does the ladder protocol infer a useful order from oracle behavior when
the structure count is not disclosed and the queried inputs lie on the
model's actual activation manifold?

## Process provenance and a disclosed gap

The protocol was preregistered on 2026-11-19 in kernel-side commits
(`db5f33f`, input snapshot `b3a906b`) that were never pushed: the
preregistration document itself existed only in those ephemeral commits. The
kernel campaign then recorded the empirical pools (kernel
`kuglipista/gcl-unknown-order-empirical-pool-sampler` v4, COMPLETE) and the
first ladder case (kernel `kuglipista/gcl-unknown-order-ladder-m-e1` v5,
COMPLETE), and the session ended before the outputs were downloaded,
verified, and landed.

This document is the integration of that work. The protocol below is
reconstructed from the frozen echoes recorded at run time - the kernel
sources (now under `scripts/gpu/kaggle/unknown-order-*/kernel.py`), the
sampler manifest, and the case report's protocol block - not rewritten after
the fact. The gap (document written after the first case's runs, because the
original was never pushed) is disclosed here rather than hidden: the M-E1
protocol echo, the pool, and every threshold were frozen in the kernel before
the run, and nothing in this reconstruction changes a recorded value.

## Frozen protocol

- **Corpus**: 18 templates x 12 slot values = 216 deterministic natural-language
  prompts; corpus table SHA-256
  `9042c1978c41f693c10dca648053fb9c6eb89547e9aa101252bdaf201d00a7bb`
  (recorded in the sampler manifest and checked by the sampler kernel).
- **Pools**: per case, the real MLP input vectors (residual-stream vectors
  entering `gate_proj`/`up_proj` of the target layer, i.e. the
  post-attention-layernorm hidden states) over the corpus, every token
  position, capped at 32768 points, split 60/20/20 into
  train/selection/validation with a frozen per-family shuffle seed. The
  solver-visible pool copy carries `x` and the split indices only.
- **Teachers**: Mistral-7B-Instruct v0.3 4-bit and LLaMA-3.2-3B-Instruct
  4-bit (unsloth providers), layers 1 / 14-15 / 27-28, six cases. The teacher
  function is the down-proj output-channel scalar: a weighted sum of the
  selected units' SwiGLU pair outputs, weights from the real `down_proj`
  channel row. Teacher manifests (model, layer, unit list, channel) are
  passed to the oracle server by inherited file descriptor; the solver
  command never mentions them, and the solver's argv is checked against a
  forbidden-locator list (`--model`, `--layer`, `--unit`, `--units`,
  `--output-channel`).
- **Oracle**: eval-only SwiGLU oracle over a unix socket, ready frame
  `gcl-swiglu-eval-v1-unknown-order` - **no `unitCount` field**, quantization
  step 1e-6, 1.5M oracle calls per fit, responses witnessed per request.
- **Solver**: the blind SwiGLU decompiler in unknown-order mode
  (`--unknown-order`, `--require-opaque-oracle`, `--defer-posthoc-truth`,
  frozen-fit output), asymptotic structure cold-start recovery,
  symmetric-multiscale-coordinate gradients, empirical train/selection/validation
  draws from the pool, auto input-scale calibration, seed 260723.
- **Order ladder**: candidates 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 24; at
  most 9 fits; **sufficiency = relative selection MAE <= 0.05** where the
  relative MAE is `mean(|f - f_hat|) / std(f)` over the selection split.
- **Selection rule**: the inferred order is the first sufficient candidate;
  the runner then walks two extra ladder steps; the upper region is flagged
  non-identifiable when later sufficient candidates stay within 2x the first
  sufficient r.
- **Null control** (preregistered, optional flag): the oracle permutes each
  response vector per request with a frozen seed - a structure-free teacher
  over the same points; the ladder must not find a sufficient order under
  it.
- **Independent verifier**: `tools/llm_reader/verify_unknown_order_empirical.py`
  recomputes the selection rule, the non-identifiability flag, every fit's
  grades from the frozen fits and the recorded teacher-side response tables,
  the pool SHA-256, and the teacher-secret consistency.

## Case M-E1: recorded result (Mistral-7B, layer 1)

Pool: 15998 points, dimension 4096, splits 9598/3200/3200. Teacher secret
(true m = 6, LCG unit draws seed 261119 over 14336, output channel seed
261120 over 4096 - teacher-side only). Recorded traversal:

| candidate | r_sel | r_eval | r_ood | sufficient | validation R2 |
|---:|---:|---:|---:|---|---:|
| 4 | 0.07940 | 0.07522 | 0.12257 | false | 0.98516 |
| 5 | 0.06960 | 0.06615 | 0.11763 | false | 0.99018 |
| 6 | 0.07372 | 0.06973 | 0.12297 | false | 0.98933 |
| 8 | 0.06021 | 0.05853 | 0.11740 | false | 0.99246 |
| 10 | 0.01291 | 0.01307 | 0.02952 | true | 0.99956 |
| 12 | 0.02116 | 0.02181 | 0.05995 | true | 0.99892 |
| 14 | 0.01896 | 0.01855 | 0.06182 | true | 0.99926 |

Inferred order: **10**. Hidden true order: **6**. Verdict:
`inferred-order`, with the upper region (12, 14) non-identifiable (both
sufficient, both within 2x of candidate 10's r). The preregistered
sufficiency threshold was not loosened to make the true order pass; the
artifact keeps both numbers side by side and the kernel gate is the
verifiable completeness of the evidence, not the correctness of the
inference.

### Interpretation and claim boundary

The protocol did not recover the hidden order on this case. Candidate 6
itself sits at r_selection 0.0737 against the 0.05 gate - close to the
border, but on the wrong side, and candidates 5 and 8 bracket it worse
(0.0696 / 0.0602), so this is not a single noisy rung: every sub-10
candidate's relative error hovers just above the threshold while candidate
10 drops to 0.0129. Over-atom candidates 12 and 14 also fit well (sufficient,
within 2x), which is the non-identifiable upper region. What this records
honestly:

- at threshold 0.05 the ladder rule selects 10, an over-estimate of the true
  6; the 6-atom representation is exact in principle (the teacher is a
  6-atom weighted sum) but the fit reached only r 0.0737 on the empirical
  manifold;
- the OOD relative error stays 4-9x the in-distribution error for every
  candidate, so even the sufficient fits are manifold-local (the
  preregistered direction anticipated exactly this risk);
- no claim of order recovery, structure recovery, or source alignment is
  made for this case; the null-permutation control was not run in the
  recorded kernel and remains to be exercised;
- the access-separation contract held end to end: ready frames without
  `unitCount`, teacher manifests by inherited descriptor, strace audits clean
  (`straceModelAccessRows == []`), zero truth requests during the blind
  phase, frozen fits hash-locked.

This is one teacher, one layer, one threshold. It is the first case of six
preregistered ones, and the finding that shapes the remaining five.

## Fresh preregistration: the five remaining cases

The original preregistration's per-case teacher parameters (true m, unit and
channel draw seeds) for the five unrun cases lived only in the ephemeral
commits and were not recoverable. They are re-registered here, before any
ladder run against them, with a mechanical rule chosen blind to any result:
early/middle/late layers get true m = 6/8/12, and the unit/channel LCG seed
pairs advance by two per case after M-E1's 261119/261120. The pools for all
five cases are already recorded and validated (pool-sampler kernel v4,
15998-point Mistral pools, 21524-point LLaMA pools, corpus digest matching).
Protocol, ladder, threshold, budget, and verifier are the frozen ones above,
unchanged.

| case | model | layer | pool points | true m | unit seed | channel seed |
|---|---|---:|---:|---:|---:|---:|
| M-E1 (recorded) | Mistral-7B 4-bit | 1 | 15998 | 6 | 261119 | 261120 |
| M-M15 | Mistral-7B 4-bit | 15 | 15998 | 8 | 261121 | 261122 |
| M-L28 | Mistral-7B 4-bit | 28 | 15998 | 12 | 261123 | 261124 |
| L-E1 | LLaMA-3.2-3B 4-bit | 1 | 21524 | 6 | 261125 | 261126 |
| L-M14 | LLaMA-3.2-3B 4-bit | 14 | 21524 | 8 | 261127 | 261128 |
| L-L27 | LLaMA-3.2-3B 4-bit | 27 | 21524 | 12 | 261129 | 261130 |

Stop condition unchanged from the direction document: if the ladder cannot
beat the preregistered null-permutation control, or if the upper region
stays non-identifiable across cases, the order-inference claim stops and
the recorded outcome is function fidelity plus a candid
non-identifiability statement.

## Batch results: the five remaining cases and the M-E1 null control

All six cases ran in one Kaggle T4 batch kernel
(`kuglipista/gcl-unknown-order-ladder-batch` v1, 2 h 02 m) under the frozen
protocol above; every case was independently verified locally (selection
rule, grades recompute from frozen fits and recorded response tables, pool
sha256, teacher secret, access audits) before integration.

| case | model / layer | hidden true m | inferred order | upper region | verdict |
|---|---|---:|---:|---|---|
| M-E1-null | Mistral-7B / 1 | 6 | 10 | non-identifiable (12, 14) | inferred-order; **null control: no-sufficient-order** |
| M-M15 | Mistral-7B / 15 | 8 | 18 | — | inferred-order (over) |
| M-L28 | Mistral-7B / 28 | 12 | none | — | **no-sufficient-order** |
| L-E1 | LLaMA-3.2-3B / 1 | 6 | 8 | non-identifiable (10, 12) | inferred-order (over) |
| L-M14 | LLaMA-3.2-3B / 14 | 8 | 16 | non-identifiable (18) | inferred-order (over) |
| L-L27 | LLaMA-3.2-3B / 27 | 12 | **12** | non-identifiable (14, 16) | inferred-order, **exact** |

Per-case recorded traversals (relative MAE on the selection split, r_eval
on the validation split, r_ood on the Gaussian secondary set):

- M-E1-null: identical r values to the recorded M-E1 run (deterministic
  protocol reproduced on fresh hardware) - 4: 0.0794, 5: 0.0696, 6:
  0.0737, 8: 0.0602, 10: 0.0129, 12: 0.0212, 14: 0.0190; the
  null-permutation control over the same teacher recorded
  `no-sufficient-order`, so the sufficiency gate does not fit permuted
  (structure-free) responses - the protocol's stop-condition control held.
- M-M15 (true 8): 4: 0.336, 5: 0.294, 6: 0.249, 8: 0.133, 10: 0.074, 12:
  0.098, 14: 0.061, 16: 0.065, 18: 0.047 - the first sufficient candidate
  is 18, a 2.25x over-estimate, and the r curve is non-monotone (12 worse
  than 10).
- M-L28 (true 12): 4: 0.524, 5: 0.433, 6: 0.441, 8: 0.240, 10: 0.122, 12:
  0.104, 14: 0.193, 16: 0.192, 18: 0.164 - no candidate clears the 0.05
  gate; the deep Mistral layer records an honest failure, with OOD relative
  error 1.2-4.5x on top.
- L-E1 (true 6): 4: 0.070, 5: 0.096, 6: 0.071, 8: 0.043, 10: 0.034, 12:
  0.022 - first sufficient 8, over-estimate by 2.
- L-M14 (true 8): 4: 0.402 ... 14: 0.053, 16: 0.043, 18: 0.028 - first
  sufficient 16, a 2x over-estimate.
- L-L27 (true 12): 4: 0.120, 5: 0.074, 6: 0.060, 8: 0.066, 10: 0.089, 12:
  0.036, 14: 0.029, 16: 0.025 - first sufficient **12, the hidden true
  order**, the one exact inference of the campaign; 14 and 16 are also
  sufficient (the recorded non-identifiable upper region).

### Interpretation across the campaign

One of six cases inferred the hidden order exactly (L-L27); four
over-estimated (by +2 to +10 atoms); the deepest Mistral layer failed to
find any sufficient order at the frozen threshold. The over-estimates
concentrate where the r curve is shallow or non-monotone around the true
order (M-M15, L-M14), and the exact hit occurred where the first sufficient
rung was also the sharpest drop (L-L27: 0.089 at 10 to 0.036 at 12). The
null-permutation control produced no-sufficient-order on the one teacher
it was run against, so the gate is not fitting noise - the failures are
failures of order identifiability at this threshold, not of the
experimental machinery. Per the preregistered stop condition: the
order-inference claim remains unsupported at r <= 0.05; what the campaign
records honestly is a family/layer-dependent identifiability profile
(mild over-estimate at layer 1, worse mid-layer, exact once, hard failure
at the deepest Mistral layer) plus function fidelity that stays
manifold-local everywhere (r_ood 1.2-9x r_in on every sufficient fit).

## Artifacts and evidence

- Case report: [`ladder/case-report.json`](../../benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/unknown-order-empirical-m-e1-2026-11-18/ladder/case-report.json)
- Case markdown: [`ladder/case-report.md`](../../benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/unknown-order-empirical-m-e1-2026-11-18/ladder/case-report.md)
- Lane manifest: [`manifest.json`](../../benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/unknown-order-empirical-m-e1-2026-11-18/manifest.json)
- Kernel evidence and heavy-file hashes: [`kernel-evidence.json`](../../benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/unknown-order-empirical-m-e1-2026-11-18/kernel-evidence.json)
- Pool provenance: [`pool-provenance/M-E1-sampler-manifest.json`](../../benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/unknown-order-empirical-m-e1-2026-11-18/pool-provenance/M-E1-sampler-manifest.json)
- Ladder runner: [`tools/llm_reader/unknown_order_swiglu_recovery.py`](../../tools/llm_reader/unknown_order_swiglu_recovery.py)
- Empirical sampler: [`tools/llm_reader/empirical_activation_sampler.py`](../../tools/llm_reader/empirical_activation_sampler.py)
- Synthetic sanity oracle: [`tools/llm_reader/synthetic_swiglu_oracle_server.py`](../../tools/llm_reader/synthetic_swiglu_oracle_server.py)
- Independent verifier: [`tools/llm_reader/verify_unknown_order_empirical.py`](../../tools/llm_reader/verify_unknown_order_empirical.py)
- Verifier tests: [`tests/test_unknown_order_empirical.py`](../../tests/test_unknown_order_empirical.py)
- Pool-sampler kernel source: [`scripts/gpu/kaggle/unknown-order-pool-sampler/kernel.py`](../../scripts/gpu/kaggle/unknown-order-pool-sampler/kernel.py)
- M-E1 ladder kernel source: [`scripts/gpu/kaggle/unknown-order-ladder-m-e1/kernel.py`](../../scripts/gpu/kaggle/unknown-order-ladder-m-e1/kernel.py)
- Kaggle kernels: `kuglipista/gcl-unknown-order-empirical-pool-sampler` (v4, COMPLETE), `kuglipista/gcl-unknown-order-ladder-m-e1` (v5, COMPLETE), `kuglipista/gcl-unknown-order-ladder-batch` (v1, COMPLETE, all six cases + null control); input dataset `kuglipista/gcl-unknown-order-input` (repo snapshot `b3a906b`), pools dataset `kuglipista/gcl-unknown-order-pools` (v2, all six validated pools).
- Batch case artifacts: `benchmarks/release/nn-live-blind-swiglu-oracle-decompiler/unknown-order-empirical-{m-e1-null,m-m15,l-l28,l-e1,l-m14,l-l27}-2026-11-20/` (case report, fits, run-info, teacher manifest, kernel-evidence, pool provenance each).
- Batch integration script: [`scripts/gpu/integrate-unknown-order-batch.py`](../../scripts/gpu/integrate-unknown-order-batch.py).
- Batch kernel source: [`scripts/gpu/kaggle/unknown-order-ladder-batch/kernel.py`](../../scripts/gpu/kaggle/unknown-order-ladder-batch/kernel.py).
