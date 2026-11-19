# LLaMA-3.2 50-Fact Ordered-Residual-Path Circuit Discovery

## Purpose

Roadmap item 6 (scale the fact set) ran its first gate: the recorded
`nn-live-llama-swiglu-ordered-residual-path-circuit-finder` lane stopped at 20
valid facts, run on a local GPU (repo commit 5628833); this lane scales the
identical discovery protocol, layer plan, thresholds, and claim boundary to the
GPT-2 lane's 50-valid-fact pool, on a Kaggle Tesla T4, through the full
90-entry FACTS corpus. Nothing else changed: the runner, adapter, provider,
layer plan, control seed, and every threshold are the recorded 20-fact
configuration, with only `--case-count/--min-valid-cases` scaled 20 -> 50.

A tokenizer pre-flight over the provider's tokenizer.json measured 62
single-token-valid facts in the corpus, so the 50-valid gate is reachable
without weakening any threshold; the scan stops at the 50th valid fact (pool
index 78), recording 28 skipped facts.

## Protocol

The lane is the hook-based ordered residual attention->MLP->target path
discovery of `tools/llm-reader-gpt2-ordered-residual-path-sweep.py` with the
`llama-swiglu` adapter on `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`
(4-bit, pre-downloaded into HF_HOME behind a hard pre-flight gate;
`--local-files-only`): causal pair ablations, attention output-path
interventions, ordered path controls, no synthetic teacher, no fitted
surrogate. Configuration echo: mlp/attention layers 20,24,27 with fallback
16,18,20,22,24,26,27; top-neurons 8; top-logits 8; control-count 3;
control-seed 260610 (sha256-subject-derived); max-graph-nodes 3;
max-graph-heads 2; pass thresholds 0.90/0.90; min-target-drop 0.01;
min-margin-over-control-p95 0.001; min-path-activation-delta 0.001;
min-path-margin-over-control-p95 0.0001; paraphrase-count 1;
negative-fact-count 1; unrelated-logit-count 4; wrong-answer-count 2;
attention-max-target-rank 200; no component hardening; target-causal MLP
fallback allowed; device cuda.

Execution provenance: Kaggle kernel
`kuglipista/gcl-llama-50-facts-ordered-path` v2 against dataset
`kuglipista/gcl-llama-ordered-input` (tools at repo commit 81cb491, pinned
transformers 5.5.0 / accelerate 1.13.0 / bitsandbytes 0.49.2, torch
2.11.0+cu130). Kernel v1 failed fast on a dataset dependency miss
(`llm-reader-gpt2-live.py`, loaded by the gpt2 adapter module that the
factual sweep imports; the first dataset version omitted it); v2 completed
after the dataset closure fix. That v1 error and fix are part of the recorded
provenance.

On-kernel, before any download: summary aggregates, percentile statistics,
the pass verdict, model identity, config echo, and claim-boundary fields are
recomputed from the raw case fields with the runner's own percentile
algorithm - a malformed artifact is a kernel error, an honest scientific
fail is a recorded finding. After download, integration re-ran the same
recompute locally plus an independent expectation: the recorded case scan
order and skipped-subject set must match a locally recomputed tokenizer scan
of the FACTS pool. Both verifications passed.

## Results

50 valid facts, 50 passed, 0 failed, 28 skipped (skipped facts are recorded,
not dropped); 3 h 01 min on a Tesla T4; pass rate 1.0, combined graph pass
rate 1.0 against the unchanged 0.90 gates.

| metric | 20-fact local GPU run | 50-fact Kaggle T4 run |
|---|---:|---:|
| valid / passed | 20 / 20 | 50 / 50 |
| skipped | 1 | 28 |
| median selected path activation delta | 0.046875 | 0.07421875 |
| p10 selected path activation delta | 0.023046875 | 0.015625 |
| median control path p95 delta | 0.001785 | 0.001816 |
| median ordered-path lift over control p95 | 39.35 | 50.02 |
| median combined target logit drop | 0.9375 | 1.0 |

The recorded pool extends beyond capitals into the general-knowledge tail of
the FACTS corpus: the 50th valid fact is "frozen water" -> "ice", and the
lane includes "largest planet" -> "Jupiter" (layers 20/20, drop 0.125) and
"nearest star" -> "Sun" (layers 27/27, drop 1.125) as recorded cases. The
baseline target rank is 1 for the median case (max 3): the answer token is
essentially the top next-token prediction on every measured prompt.

Structural read of the 50 recorded paths: every case selects the minimal
one-attention-head + one-MLP-node prefix; attention heads land in layer 27
for 32 cases and layer 20 for 18; MLP neurons land in layer 27 for 34 cases
and layer 20 for 16; 18 of 50 cases needed an ordered-rescue attempt and all
18 still passed under the unchanged gates. Combined target logit drop spans
0.0625-1.625 (median 1.0); ordered-path lift over control p95 spans
1.28-2.9e8 with median 56 (the maximum sits where the control p95 is 0.0 and
the lift field records the ratio against the epsilon floor).

## Interpretation and claim boundary

The pass rate, gates, and causal controls held at 2.5x the recorded fact
count with zero loosening: the discovery protocol's per-fact evidence does
not degrade on the deeper pool, and the measured median path lift and
combined drop both rose slightly against the 20-fact artifact (39.4 -> 50.0;
0.9375 -> 1.0) while the p10 path delta fell (0.023 -> 0.0156) - weaker-tail
paths exist at 50 facts, and every one still cleared the unchanged margins.

What this lane does not prove is unchanged from the recorded 20-fact claim
boundary: this is not full-model circuit recovery, not exhaustive
residual-stream patching for every edge, not full FFN extraction, not
arbitrary LLM understanding, not output-only original-neuron identity, and
not a broad SOTA claim. The 28 skipped facts are a tokenizer-validity record
(multi-token answers), not a coverage claim. The run is a fresh discovery
on T4 hardware, not a bitwise replay of the local 20-fact run; within-run
causality (ablation drops, control margins) is what the gates test, and the
on-kernel and local recomputations confirm exactly that.

## Artifacts

- 50-fact report: [`llama-3.2-3b-50-facts-ordered-residual-path/report.json`](../../benchmarks/release/nn-live-llama-swiglu-ordered-residual-path-circuit-finder/llama-3.2-3b-50-facts-ordered-residual-path/report.json)
- Kernel run evidence: [`kernel-run-info.json`](../../benchmarks/release/nn-live-llama-swiglu-ordered-residual-path-circuit-finder/llama-3.2-3b-50-facts-ordered-residual-path/kernel-run-info.json), [`kernel-log.txt`](../../benchmarks/release/nn-live-llama-swiglu-ordered-residual-path-circuit-finder/llama-3.2-3b-50-facts-ordered-residual-path/kernel-log.txt), [`manifest.json`](../../benchmarks/release/nn-live-llama-swiglu-ordered-residual-path-circuit-finder/llama-3.2-3b-50-facts-ordered-residual-path/manifest.json)
- 20-fact reference (local GPU, repo commit 5628833): [`llama-3.2-3b-20-facts-ordered-residual-path/report.json`](../../benchmarks/release/nn-live-llama-swiglu-ordered-residual-path-circuit-finder/llama-3.2-3b-20-facts-ordered-residual-path/report.json)
- Kernel source: [`scripts/gpu/kaggle/llama-50-facts/kernel.py`](../../scripts/gpu/kaggle/llama-50-facts/kernel.py)
- Integration + local verification: [`scripts/gpu/integrate-llama-50-facts.py`](../../scripts/gpu/integrate-llama-50-facts.py)
- Runner: [`llm-reader-gpt2-ordered-residual-path-sweep.py`](../../tools/llm-reader-gpt2-ordered-residual-path-sweep.py)
- Recorded 20-fact lane wrapper: [`benchmark-nn-live-llama-swiglu-ordered-residual-path-circuit-finder.ts`](../../scripts/benchmark-nn-live-llama-swiglu-ordered-residual-path-circuit-finder.ts)
- GPT-2 50-fact reference artifact: [`gpt2-50-facts-ordered-residual-path/report.json`](../../benchmarks/release/nn-live-gpt2-ordered-residual-path-circuit-finder/gpt2-50-facts-ordered-residual-path/report.json)
