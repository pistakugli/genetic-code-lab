# Ordered-Path Fact-Set Scale-Up: LLaMA-3.2 50 Facts, Mistral-7B 20 Facts

## Purpose

Roadmap item 6 (scale the fact set) ran its first two gates in one session, one
kernel per family, both through the full 90-entry FACTS corpus on a Kaggle
Tesla T4:

- the recorded `nn-live-llama-swiglu-ordered-residual-path-circuit-finder`
  lane stopped at 20 valid facts (run on a local GPU, repo commit 5628833);
  this lane scales the identical discovery protocol, layer plan, thresholds,
  control seed, and claim boundary to the GPT-2 lane's 50-valid-fact pool;
- the recorded `nn-live-mistral-swiglu-ordered-residual-path-circuit-finder`
  lane stopped at 10 valid facts (local GPU); this lane scales the identical
  configuration - including the stricter 1.0/1.0 pass gates this family
  recorded - to the 20-valid-fact pool.

Nothing else changed in either lane: runner, adapter, provider, layer plan,
control seed, rescue budget, and every threshold are the recorded family
artifacts, with only `--case-count/--min-valid-cases` scaled.

Tokenizer pre-flights over each provider's tokenizer.json measured 62
single-token-valid facts for LLaMA-3.2 (50 reachable; the scan stops at the
50th valid fact, pool index 78, 28 skipped facts recorded) and 32 for
Mistral-7B (20 reachable; the scan stops at the 20th valid fact, pool index
77, 57 skipped facts recorded). Skipped facts are tokenizer multi-token
answers, recorded, not dropped; both pre-flights predicted the exact scan
order and skip set, and both artifacts match them.

## Protocol

Both lanes are the hook-based ordered residual attention->MLP->target path
discovery of `tools/llm-reader-gpt2-ordered-residual-path-sweep.py` with the
family adapter on the family's quantized unsloth provider (4-bit,
pre-downloaded into HF_HOME behind a hard pre-flight gate;
`--local-files-only`): causal pair ablations, attention output-path
interventions, ordered path controls, no synthetic teacher, no fitted
surrogate.

Per-family configuration echoes, recorded in each artifact and re-checked
by the kernel:

- LLaMA-3.2 (`llama-swiglu`, `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`):
  mlp/attention layers 20,24,27 with fallback 16,18,20,22,24,26,27;
  control-seed 260610; pass gates 0.90/0.90; rescue budget 10/32/10/24.
- Mistral-7B (`mistral-swiglu`,
  `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`): mlp/attention layers
  20,24,28,31 with fallback 16,20,24,28,31; control-seed 260611; pass gates
  1.0/1.0 (every case must pass); rescue budget 8/16/8/96.

Shared: top-neurons 8; top-logits 8; control-count 3 (sha256-subject-derived);
control prompt "The weather in Berlin is"; max-graph-nodes 3; max-graph-heads
2; min-target-drop 0.01; min-margin-over-control-p95 0.001;
min-path-activation-delta 0.001; min-path-margin-over-control-p95 0.0001;
paraphrase-count 1; negative-fact-count 1; unrelated-logit-count 4;
wrong-answer-count 2; attention-max-target-rank 200; no component hardening;
target-causal MLP fallback allowed; device cuda.

Execution provenance: one Kaggle kernel per family against its own private
input dataset carrying the repo tools at commit 81cb491
(`kuglipista/gcl-llama-ordered-input` and
`kuglipista/gcl-mistral-ordered-input`), pinned transformers 5.5.0 /
accelerate 1.13.0 / bitsandbytes 0.49.2, torch 2.11.0+cu130, python 3.13.16.
The LLaMA kernel ran as v2 (v1 failed fast on a dataset dependency miss:
`llm-reader-gpt2-live.py`, loaded by the gpt2 adapter module the factual sweep
imports; the first dataset version omitted it; v2 completed after the closure
fix - both are recorded provenance); the Mistral kernel ran as v1.

On-kernel, before any download: summary aggregates, percentile statistics,
the pass verdict, model identity, config echo, and claim-boundary fields are
recomputed from the raw case fields with the runner's own percentile
algorithm - a malformed artifact is a kernel error, an honest scientific
fail is a recorded finding. After download, integration re-ran the same
recompute locally plus the independent tokenizer-scan expectation (scan
order + skip set). All verifications passed for both families.

## Results

### LLaMA-3.2-3B 4-bit: 50 valid facts, 50 passed

3 h 01 min on a Tesla T4; pass rate 1.0, combined graph pass rate 1.0 against
the unchanged 0.90 gates; 28 skipped facts recorded.

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
the FACTS corpus: the 50th valid fact is "frozen water" -> "ice", and the lane
includes "largest planet" -> "Jupiter" (layers 20/20, drop 0.125) and
"nearest star" -> "Sun" (layers 27/27, drop 1.125) as recorded cases. The
baseline target rank is 1 for the median case (max 3). Every case selects the
minimal one-attention-head + one-MLP-node prefix; attention heads land in
layer 27 for 32 cases and layer 20 for 18; MLP neurons land in layer 27 for 34
cases and layer 20 for 16; 18 of 50 cases needed an ordered-rescue attempt and
all 18 still passed under the unchanged gates. Combined target logit drop
spans 0.0625-1.625 (median 1.0); ordered-path lift over control p95 spans
1.28-2.9e8 with median 56 (the maximum sits where the control p95 is 0.0 and
the lift field records the ratio against the epsilon floor).

### Mistral-7B-Instruct v0.3 4-bit: 20 valid facts, 20 passed under 1.0 gates

2 h 55 min of discovery on a Tesla T4; pass rate 1.0, combined graph pass rate
1.0 against this family's stricter recorded 1.0/1.0 gates (every single case
must pass); 57 skipped facts recorded.

| metric | 10-fact local GPU run | 20-fact Kaggle T4 run |
|---|---:|---:|
| valid / passed | 10 / 10 | 20 / 20 |
| skipped | 10 | 57 |
| median selected path activation delta | 0.203125 | 0.234375 |
| p10 selected path activation delta | 0.0921875 | 0.0921875 |
| median control path p95 delta | 0.045312 | 0.039929 |
| median ordered-path lift over control p95 | 5.105 | 4.967 |
| median combined target logit drop | 0.34375 | 0.3125 |

The recorded pool likewise extends beyond the capital head of the corpus: the
20th valid fact is "month after January" -> "February", and the lane includes
"nearest star" -> "Sun" and "grass color" -> "green" as recorded cases. The
baseline target rank is 1 for the median case (max 2). Every case selects the
minimal one-head + one-MLP-node prefix; MLP neurons land in layer 31 for 11
cases, layer 20 for 5, layer 24 for 3, layer 28 for 1; attention heads land
in layer 20 for 9, layer 31 for 8, layer 24 for 3; 10 of 20 cases needed an
ordered-rescue attempt and all 10 still passed. Combined target logit drop
spans 0.0625-0.75 (median 0.3125).

## Interpretation and claim boundary

Both pass rates, gates, and causal controls held at 2.5x (LLaMA) and 2x
(Mistral) the recorded fact counts with zero loosening: the discovery
protocol's per-fact evidence does not degrade on the deeper pools. LLaMA's
median path lift and combined drop both rose slightly against the 20-fact
artifact (39.4 -> 50.0; 0.9375 -> 1.0) while the p10 path delta fell
(0.023 -> 0.0156) - weaker-tail paths exist at 50 facts, and every one still
cleared the unchanged margins. Mistral's medians moved within a few percent
(lift 5.11 -> 4.97, drop 0.344 -> 0.313) while holding the stricter 1.0/1.0
gates. In both families the scan reaches past the capital head of the corpus
into world-knowledge facts, and those facts pass under the same frozen
protocol - the recorded evidence is not capital-locked.

What these lanes do not prove is unchanged from the recorded family claim
boundaries: this is not full-model circuit recovery, not exhaustive
residual-stream patching for every edge, not full FFN extraction, not
arbitrary LLM understanding, not output-only original-neuron identity, and
not a broad SOTA claim. The skipped-fact counts are tokenizer-validity
records (multi-token answers under each provider's tokenizer), not coverage
claims; Mistral's 32k vocabulary skips 57 of the 77 scanned facts and that is
part of the measurement, not a defect to engineer around. Both runs are fresh
discoveries on T4 hardware, not bitwise replays of the local artifacts;
within-run causality (ablation drops, control margins) is what the gates
test, and the on-kernel and local recomputations confirm exactly that.

## Artifacts

- LLaMA 50-fact report: [`llama-3.2-3b-50-facts-ordered-residual-path/report.json`](../../benchmarks/release/nn-live-llama-swiglu-ordered-residual-path-circuit-finder/llama-3.2-3b-50-facts-ordered-residual-path/report.json)
- Mistral 20-fact report: [`mistral-7b-instruct-v0.3-20-facts-ordered-residual-path/report.json`](../../benchmarks/release/nn-live-mistral-swiglu-ordered-residual-path-circuit-finder/mistral-7b-instruct-v0.3-20-facts-ordered-residual-path/report.json)
- LLaMA kernel evidence: [`kernel-run-info.json`](../../benchmarks/release/nn-live-llama-swiglu-ordered-residual-path-circuit-finder/llama-3.2-3b-50-facts-ordered-residual-path/kernel-run-info.json), [`kernel-log.txt`](../../benchmarks/release/nn-live-llama-swiglu-ordered-residual-path-circuit-finder/llama-3.2-3b-50-facts-ordered-residual-path/kernel-log.txt), [`manifest.json`](../../benchmarks/release/nn-live-llama-swiglu-ordered-residual-path-circuit-finder/llama-3.2-3b-50-facts-ordered-residual-path/manifest.json)
- Mistral kernel evidence: [`kernel-run-info.json`](../../benchmarks/release/nn-live-mistral-swiglu-ordered-residual-path-circuit-finder/mistral-7b-instruct-v0.3-20-facts-ordered-residual-path/kernel-run-info.json), [`kernel-log.txt`](../../benchmarks/release/nn-live-mistral-swiglu-ordered-residual-path-circuit-finder/mistral-7b-instruct-v0.3-20-facts-ordered-residual-path/kernel-log.txt), [`manifest.json`](../../benchmarks/release/nn-live-mistral-swiglu-ordered-residual-path-circuit-finder/mistral-7b-instruct-v0.3-20-facts-ordered-residual-path/manifest.json)
- LLaMA 20-fact reference (local GPU): [`llama-3.2-3b-20-facts-ordered-residual-path/report.json`](../../benchmarks/release/nn-live-llama-swiglu-ordered-residual-path-circuit-finder/llama-3.2-3b-20-facts-ordered-residual-path/report.json)
- Mistral 10-fact reference (local GPU): [`mistral-7b-instruct-v0.3-10-facts-ordered-residual-path/report.json`](../../benchmarks/release/nn-live-mistral-swiglu-ordered-residual-path-circuit-finder/mistral-7b-instruct-v0.3-10-facts-ordered-residual-path/report.json)
- LLaMA kernel source: [`scripts/gpu/kaggle/llama-50-facts/kernel.py`](../../scripts/gpu/kaggle/llama-50-facts/kernel.py)
- Mistral kernel source: [`scripts/gpu/kaggle/mistral-20-facts/kernel.py`](../../scripts/gpu/kaggle/mistral-20-facts/kernel.py)
- Integration + local verification: [`scripts/gpu/integrate-llama-50-facts.py`](../../scripts/gpu/integrate-llama-50-facts.py), [`scripts/gpu/integrate-mistral-20-facts.py`](../../scripts/gpu/integrate-mistral-20-facts.py)
- Runner: [`llm-reader-gpt2-ordered-residual-path-sweep.py`](../../tools/llm-reader-gpt2-ordered-residual-path-sweep.py)
- GPT-2 50-fact reference artifact: [`gpt2-50-facts-ordered-residual-path/report.json`](../../benchmarks/release/nn-live-gpt2-ordered-residual-path-circuit-finder/gpt2-50-facts-ordered-residual-path/report.json)
