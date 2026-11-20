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
- Kaggle kernels: `kuglipista/gcl-unknown-order-empirical-pool-sampler` (v4, COMPLETE), `kuglipista/gcl-unknown-order-ladder-m-e1` (v5, COMPLETE); input dataset `kuglipista/gcl-unknown-order-input` (repo snapshot `b3a906b`), pools dataset `kuglipista/gcl-unknown-order-pools`.
