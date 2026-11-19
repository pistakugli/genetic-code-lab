# Agent Note: Output-only OOD paraphrase profile

Status: implemented

## Context

The output/logit-only behavior rail had one recorded gate: the GPT-2 10-fact
behavior graph over standard same-relation paraphrase templates selected by
margin. The roadmap's next gates for this rail were a larger fact set, the
OOD paraphrase families, and Pythia/LLaMA/Mistral parity through the
unchanged prompt-to-logits-only solver boundary. The standard paraphrase
flow scores ten candidate templates and keeps the top-N by answer-pool
margin, which cannot measure OOD robustness: selection by margin would
silently drop any phrasing that breaks the behavior.

## Decision

`tools/llm-reader-blackbox-behavior.py` accepts
`--paraphrase-profile standard|ood` (`standard` remains the default and is
byte-identical: a regression run with the same arguments produces the same
case hashes and summary). The `ood` profile scores every prompt of the six
deterministic answer-blind OOD capital-question families from
`llm_reader/ood_paraphrases.py` and keeps all of them: no margin-based
selection, so the gate cannot cherry-pick an easy phrasing, and the runner
raises if a family prompt ever contains the answer. Each paraphrase item
records its family; the summary gains `paraphraseProfile`,
`paraphraseFamilies`, and a `paraphraseFamilySummary` table (count,
retained, retentionRate, meanMargin, minMargin, median and max target rank
per family) that restates the whole recorded population; the markdown
report gains the same family table. The solver boundary and access audit
are unchanged: prompt strings, target token ids, next-token logits only.

Two TypeScript wrappers pin the lane configurations:
`benchmark-nn-live-blackbox-llm-behavior-reader-ood.ts` (GPT-2, full 20-fact
pool, min-valid-cases 19 because Brasilia is not a single GPT-2 token - the
skipped fact is recorded in the artifact) and
`benchmark-nn-live-blackbox-pythia-behavior-reader-ood.ts` (Pythia-70m,
min-valid-cases 18 for Canberra and Brasilia). Both keep the
honest-exit wrapper discipline: a child that exits 0 without a passing
report fails the wrapper.

## Result

GPT-2 passes the full-pool OOD profile: 19 valid facts, all pass, 380
queries, family retention 0.947-1.0 per family, the only broken family
items being Austria's service-question and sentence-completion. Pythia-70m
records an honest family-level failure: 7/18 cases pass, question-style
families collapse (2-4 of 18) while possessive-question retains 17/18 - the
same family-relative template dependence the hook-based ordered-path OOD
replacement lane measured for Pythia-70m, now observed from the logit-only
boundary. Artifacts, evidence-map rows, the research note
`2026-11-20-output-only-ood-behavior-parity.md`, and the roadmap item-5
status are updated. The remaining next gate is LLaMA/Mistral
instruct-family parity on Kaggle T4 through the same boundary.

## Phase 2: instruct-family parity on Kaggle T4

The runner gained `--load-in-4bit` (CUDA-only, bitsandbytes
`BitsAndBytesConfig(load_in_4bit=True)`; the model identity records
`quantized: true` only for 4-bit runs, so unquantized artifacts are
unchanged), so the instruct families run through the same quantized unsloth
providers as the recorded hook-based family artifacts.

The parity kernel `scripts/gpu/kaggle/blackbox-ood-parity/kernel.py`
(dataset `kuglipista/gcl-blackbox-ood-input`, refreshed to repo commit
`1c98d5a` + the 4-bit extension; kernel v4) pre-downloads both providers into
HF_HOME with a hard cache pre-flight gate, runs the OOD profile over the
full 20-fact pool for both families, and recomputes the family table, case
aggregates, access audit, and model identity on-kernel from the raw case
fields: a malformed artifact is a kernel error, an honest scientific fail is
a recorded finding. Dataset-mount layout is resolved by content (the
dir-mode zip upload mounts a raw tools.zip, so the kernel unpacks any zip
under /kaggle/input and searches the whole input tree for the runner; the
full mount listing is printed for provenance).

Result (Tesla T4, torch 2.11.0+cu130, pinned
transformers 5.5.0 / accelerate 1.13.0 / bitsandbytes 0.49.2):
LLaMA-3.2-3B-Instruct 4-bit passes 19/19 valid facts - rank-1 answers
everywhere, family retention 19/19 in all six families, minimum item margin
5.125; Mistral-7B-Instruct v0.3 4-bit passes 9/9 valid facts (11 answers are
not single Mistral tokens, skipped and recorded) - rank-1 answers, family
retention 9/9, minimum item margin 5.25. The runner exited 0 for both and
the on-kernel consistency recompute found no errors. Artifacts integrated
with kernel-run-info.json provenance, manifests, evidence-map rows, the
research note's instruct sections, roadmap item-5 completion, and the
capability-matrix OOD/portability rows.
