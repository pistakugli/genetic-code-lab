# Ordered-Path OOD Replacement Transfer Across Model Families

## Purpose

The recorded GPT-2 ordered-path OOD replacement-transfer evidence
([2026-10-30 lane](2026-10-30-ordered-path-ood-stress.md), one-case France
artifact from 2026-11-04) asked one model family one question: when an OOD
paraphrase breaks a discovered circuit, does the causal effect die, move, or
re-bind? This lane repeats that protocol on real CUDA hardware for three more
model families, so the software's answer to "why and how did the model react"
is measured per family instead of being extrapolated from GPT-2:

- Pythia-70m-deduped (GPT-NeoX parallel MLP + GELU),
- Mistral-7B-Instruct v0.3 4-bit (SwiGLU),
- Qwen2.5-1.5B-Instruct fp16 (SwiGLU).

All runs executed on Kaggle T4 GPU kernels against the repo state pinned at
commit `5fdcf36` (recorded in each `kernel-run-summary.json`). The runner
records a pass or a fail honestly; a family-level failure is a finding, not a
reason to rerun until green.

## Verifier fix recorded with this lane

Integration exposed a real defect in the independent artifact verifier
[`verify_ordered_path_ood_stress.py`](../../tools/llm_reader/verify_ordered_path_ood_stress.py):
it accumulated `replacement_transfer_records` across cases inside
`verify_case_report`, then re-checked earlier cases' held-out transfer prompts
against later cases' paraphrase prompts. On the valid Mistral artifact this
fabricated 25 `transfer prompt does not match the held-out family input`
errors. Single-family artifacts (GPT-2 France, LLaMA preflights) never
triggered it, and Pythia's replacements all ended `failed` (the check is
skipped for failed paths), which is why the defect surfaced only at
integration time.

The fix moves the reset inside the case loop, one hunk, no measurement
changed; it is the same fix the Mistral kernel v2 had already applied
on-kernel. A regression test
(`test_artifact_verifier_does_not_recheck_one_case_transfer_prompts_against_later_cases`)
covers both directions: a valid two-case report verifies clean, and a
genuinely cross-case transfer prompt is still rejected. All five OOD stress
suites in the repo (GPT-2 5-case, GPT-2 replacement-transfer, Pythia,
Mistral, Qwen) pass the fixed verifier.

## Matched protocol

For every family: replay the recorded circuit-finder interventions on the
source prompts, measure the six held-out OOD prompt families per subject
(service question, possessive question, imperative, embedded question,
sentence completion, context-prefixed question; answer-blind templates),
replay neighboring-layer controls, and for every broken paraphrase record
the full why-report chain: break classification, layer-shift rescue attempt,
replacement-path rediscovery, and held-out family transfer of the replacement
path. No surrogate is fitted; no teacher parameters or scenario labels are
supplied to the prompt-family generator.

Family-specific configuration (recorded in each artifact's `config`):

| Family | Model | Subjects (cases) | min drop | min ratio | min pass rate | layer spec | replay gate |
|---|---|---|---:|---:|---:|---:|---|
| Pythia-70m | `EleutherAI/pythia-70m-deduped` | Spain, Japan, Russia, Norway, Austria (5) | 0.05 | 0.10 | 0.66 | 0.80 | 1e-5, same GPU class |
| Mistral-7B | `unsloth/mistral-7b-instruct-v0.3-bnb-4bit` | France, Germany, Spain, Japan (4) | 0.005 | 0.05 | 0.66 | 0.75 | 0.125 cross-host |
| Qwen2.5-1.5B | `Qwen/Qwen2.5-1.5B-Instruct` | France, Germany, Japan, Spain (4) | 0.05 | 0.10 | 0.66 | 0.80 | 1e-5, same host |

The Mistral run is the only cross-host one (source artifact recorded on the
developer's CUDA machine, replay on Kaggle T4), so its replay gate admits one
4-bit logit quantum (0.125); the recorded `maxReplayError` for the accepted
run is 0.0625, within that bound. Pythia and Qwen replayed their source
artifacts with `maxReplayError` exactly 0.0.

## Results

| Family | Suite pass | Cases | Original replay | Paraphrase pass | Layer specificity | Replacement found | Held-out transfer |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pythia-70m | false | 1 / 5 | 100% | 43.3% (13/30) | 100% | 0 / 8 | not reached |
| Mistral-7B | pass | 4 / 4 | 100% | 83.3% (20/24) | 100% | 4 / 4 | 13 / 20 |
| Qwen2.5-1.5B | pass | 4 / 4 | 100% | 100% (24/24) | 100% | 0 / 0 (no breaks) | not applicable |

### Pythia-70m: an honestly recorded family-level failure

Only Russia passed the case gate. Spain, Japan, Norway, and Austria failed.
Original replay and layer specificity were 100% for every case, so the
circuits were replayed exactly and remained layer-specific; what broke was
OOD persistence itself. The break profile is bimodal: five magnitude-collapse
breaks (ablation leaves the target logit unchanged) and three sign-flips
(ablation raises the target logit, all on the service-question family for
Spain, Norway, Austria). The imperative family passed 5/5 with mean drop
ratio 5.9 — the strongest family for Pythia — while service-question passed
1/5. Eight rescue attempts were spent, layer shifts alone recovered two, and
the replacement search found no passing path in any of them (0/8). This is
the inverse of the recorded GPT-2 pilot, where imperative failed 5/5 and
service-question passed 3/5: which template family preserves a circuit is
model-relative, not template-intrinsic.

### Mistral-7B: every break was rescued, and the why-report says where the effect went

France passed with no breaks. Germany broke two paraphrases
(embedded-question and context-prefixed-question) and Spain and Japan each
broke one (sentence-completion both): all four
breaks were magnitude-collapse — the same components, rephrased, stopped
carrying the effect. For three of the four, shifting the same components
across neighboring layers alone recovered the effect (Germany: shift -1 gave
0.0625 against the source 0.4375; Germany's second break: shift -3;
Japan: shift +1) — layer misplacement. Spain's break was the exception:
layer shifts recovered nothing, and only the replacement path did (drop
0.6875 against the source 0.1875). All four replacement paths were found,
and every one was a `component-rebinding` with zero overlap in both exact
(layer, component) pairs and component ids (Germany: attention (20, 24) and
MLP (31, 9943) replaced by attention (31, 22) and MLP (31, 8967); Spain:
layers {20, 24} to {17, 18}, earlier; Japan: {20} to {19, 23}, later). The
behavioral effect survives rephrasing by re-binding to different components,
not by reusing the original ones. Held-out family transfer of the replacement
paths: 13/20 overall (Germany 4/5 and 5/5; Spain 2/5; Japan 2/5), so the
replacement paths themselves are partially family-locked.

### Qwen2.5-1.5B: the first zero-break family record

All 24 OOD replays across France, Germany, Japan, and Spain passed
(100%, mean drop ratios from 0.38 on embedded questions to 1.33 on sentence
completions). No rescue was attempted, no replacement path was searched, and
the why-report aggregate is empty by construction — a pass with nothing to
diagnose. The input circuit-finder artifact for this lane was recorded in the
same kernel run (same host, replay error exactly 0.0): 9 of 11 valid facts
passed; Greece and Netherlands failed the finder gate and were excluded
from the OOD subjects, recorded as a `fail`-status artifact rather than
silently dropped.

## Interpretation and claim boundary

Cross-family reading, bounded to what was measured: for small parallel-MLP
models (Pythia-70m) discovered capital-question circuits do not survive OOD
rephrasing, and neither narrow rescue nor replacement search within the
configured budget found the effect again; for mid-scale SwiGLU instruct
models (Mistral-7B, Qwen2.5-1.5B) the same protocol records persistence, and
where a break does occur the why-report localizes it: layer misplacement
(recoverable by shifting), or component re-binding (recoverable only by a
new path, with zero component overlap). None of this proves semantic
equivalence of circuits across families, circuit uniqueness, robustness
beyond the six capital-question templates and the selected subjects, or
anything about production deployment. The Mistral 0.125 replay gate is a
cross-host numeric allowance, not a quality relaxation of the same-host
1e-5 standard; the recorded error was 0.0625.

Runtime and VRAM were not measured in this lane; CUDA is verified from the
recorded model identity and the kernel GPU smoke evidence
(`kernel-gpu-smoke.json` per artifact). Kaggle kernel session limits, not
the software, bound the subject counts.

## Artifacts

- Pythia case report: [`ood-replacement-transfer-pythia-70m-2026-11-16/pythia/report.json`](../../benchmarks/release/nn-live-ordered-path-stability/ood-replacement-transfer-pythia-70m-2026-11-16/pythia/report.json)
- Mistral case report: [`ood-replacement-transfer-mistral-7b-2026-11-16/mistral/report.json`](../../benchmarks/release/nn-live-ordered-path-stability/ood-replacement-transfer-mistral-7b-2026-11-16/mistral/report.json)
- Qwen case report: [`ood-replacement-transfer-qwen-2.5-1.5b-2026-11-17/qwen/report.json`](../../benchmarks/release/nn-live-ordered-path-stability/ood-replacement-transfer-qwen-2.5-1.5b-2026-11-17/qwen/report.json)
- Qwen circuit-finder input: [`qwen-2.5-1.5b-11-facts-ordered-residual-path/report.json`](../../benchmarks/release/nn-live-qwen2-swiglu-ordered-residual-path-circuit-finder/qwen-2.5-1.5b-11-facts-ordered-residual-path/report.json)
- GPT-2 reference lane: [`ood-replacement-transfer-2026-11-04/report.json`](../../benchmarks/release/nn-live-ordered-path-stability/ood-replacement-transfer-2026-11-04/report.json)
- Independent verifier (fixed): [`verify_ordered_path_ood_stress.py`](../../tools/llm_reader/verify_ordered_path_ood_stress.py)

Kernel provenance: `kuglipista/gcl-pythia-ood-replacement-transfer`,
`kuglipista/gcl-mistral-ood-replacement-transfer`,
`kuglipista/gcl-qwen-ood-lane` (all pinned to repo commit `5fdcf36`);
per-artifact `kernel-run-summary.json` and `kernel-gpu-smoke.json` record
the invocation, elapsed time, pinned dependencies, and GPU identity.
