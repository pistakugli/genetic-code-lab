# 2026-11-14 — Pythia-70m OOD replacement transfer: second-family evidence for the why-report layer

## What was asked

The why-report layer (2026-10-30) had recorded evidence from exactly one model family: the GPT-2 France case in `ood-replacement-transfer-2026-11-04`. The open continuation was to give the layer evidence from a second adapter, ideally on a live model, so its break diagnoses are not single-family artifacts.

The continuation sandbox has no GPU, which rules out the LLaMA-3.2 and Mistral-7B lanes. Pythia-70m is the lane's small model, and its recorded circuit-finder artifact is CPU-replayable, so the second family is Pythia via the `gpt-neox` adapter.

## Method

- Input: `nn-live-pythia-ordered-residual-path-circuit-finder/pythia-70m-20-facts-ordered-residual-path/report.json`, the recorded France case (attention L0 h6 + MLP L1 n309, both at position 4, combined target-logit drop 2.0).
- Runner: `tools/llm-reader-ordered-path-stability.py`, adapter `gpt-neox`, model `EleutherAI/pythia-70m-deduped`, device `cpu`, one case (France), six held-out OOD paraphrase families, rescue enabled with layer shifts -3..+3, replacement search and transfer at the lane defaults.
- Artifact: `nn-live-ordered-path-stability/ood-replacement-transfer-pythia-70m-2026-11-14/` (suite report, per-model report in `pythia/`, manifest).

Replay exactness: the original replay reproduces the recorded drop exactly (absolute replay error 0.0), so the CPU run measures the same circuit the GPU pilot recorded.

## Two runner/verifier changes this run forced

1. The independent verifier rejected any artifact whose model device was not CUDA. CPU-live runs of small models are legitimate lane executions, so the check now accepts exactly `cuda` and `cpu` and still rejects unknown device strings. A test covers both directions.
2. The rescue budget defaulted to 2, but five of six OOD families broke, so three recorded breaks would have carried no why-report — contradicting the lane's north-star that every recorded break gets one. The default now covers all six OOD families, and the budget is recorded in the report config as `maxRescueAttempts`.

## Result

- Paraphrase pass rate: 1/6 (`imperative` passes with drop 17.0, ratio 8.5; the other five families replay with drop exactly 0.0).
- Break classification: five `magnitude-collapse` breaks (no sign flips).
- Layer-shift rescue: the `service-question` break is a layer misplacement — the same components shifted +1 recover drop 1.0. Negative shifts do not exist for this path (attention sits at layer 0), so the artifact records only shifts +1..+3.
- Replacement search: no passing path for any of the five breaks. Four report `ordered path requires selected MLP and attention interventions` (the rediscovery selection finds no qualifying MLP+attention path under the broken phrasing); the service-question attempt runs the best-margin-pair mode and finds nothing that beats the layer shift.
- Cross-family transfer: not attempted — transfer requires a found replacement path.
- Gate: fails honestly (`paraphrasePassRate` 0.167 < 0.66, failed subject France). The artifact records the failure; the why-report explains every recorded break.

Independent verification: `tools/llm_reader/verify_ordered_path_ood_stress.py --report .../ood-replacement-transfer-pythia-70m-2026-11-14/report.json` passes, including recomputation of every why-report block from the raw rescue measurements. The 2026-11-04 GPT-2 artifact still verifies after the verifier change.

## What the second family adds to the why-report layer

- GPT-2 gave the layer a rescued break with a found replacement path (component-rebinding, layers [9,11] -> [6]) and a transfer diagnosis with a failing family. Pythia gives the complementary cases: a break that is purely a layer misplacement (rescued by +1 shift, no replacement needed), and four breaks where neither the same components nor any rediscovered path carry the effect — so the honest why statement is "the effect did not move with those components, and no replacement path was found".
- The layer now has verified evidence for all three break kinds' neighbours: sign-flip handling exists (classification), magnitude-collapse dominates here, and the replacement/transfer branches cover both found and not-found outcomes.
- A six-layer model with a two-component path produces why statements whose attempted-shift list is asymmetric (only +1..+3 exist); the why layer reports exactly what was attempted rather than assuming symmetric budgets.

## Claim boundary

One subject (France), one model (Pythia-70m-deduped, 6 layers, 512 hidden), CPU device, six curated capital-question OOD families, thresholds as recorded in the artifact config. The failing gate is a measurement of this path's OOD robustness, not a statement about Pythia-70m generally, and the why statements are comparative diagnoses of recorded numbers, not semantic claims about what the components compute.

## Post-reconciliation regeneration (same day)

The repo carried two parallel implementations of the why-report layer: this sandbox's (rejected-candidate marking) and a richer one on origin/main (per-paraphrase break classification for every broken paraphrase, `rescueSought`, `attempted`, plus a family-level break atlas). After rebasing onto the merged layer, the recorded Pythia why blocks were regenerated by re-running `attach_why_analysis` on the recorded raw measurements: no new measurements, no model queries, and the raw fields (`originalReplay`, `paraphrases`, `paraphraseRescue`, `layerControls`, `config`) are byte-identical to the original commit. The artifact gains the every-break classification (five `magnitude-collapse` paraphrase breaks, five of five rescue-sought) and the `attempted` flag the merged verifier recomputes; `pathChangeScope` marks the service-question search's rejected candidate. The independent verifier passes on the regenerated artifact, and the 2026-11-04 GPT-2 artifact still passes against the merged verifier.
