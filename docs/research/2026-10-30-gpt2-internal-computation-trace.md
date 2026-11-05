# GPT-2 Internal Computation Trace: Five Factual Inputs

## Objective and access mode

This experiment tests the existing GCL ordered-residual-path reader in an
**instrumented-witness** mode. It reuses the exact source prompts, target token
IDs, selected attention/MLP nodes, and recorded matched controls from the
existing 50-fact GPT-2 ordered-path report. It neither changes that report nor
uses the captured tensors to rediscover or alter the selected paths.

The runner loads local `GPT2LMHeadModel` weights with `local_files_only=true`,
captures internal tensors with forward hooks, computes local first-order
attributions, and runs activation removals/restorations. This is not a
black-box solver run.

## What was measured

For each exact source prompt, the runner records:

- input token IDs, tokenizer token strings, character offsets, token and
  position embeddings, and embedding output;
- full-sequence residual inputs and outputs for all 12 transformer layers;
- each attention block's output and each head's projected residual write;
- MLP inputs, preactivations, GELU activations, MLP outputs, and
  per-neuron first-order target-logit attributions at the read position;
- final layer-normalization input/output, full vocabulary logits, target
  probability/rank, greedy next token, and the top eight predictions;
- target-logit and probability changes from selected attention, selected MLP,
  joint path, neighboring-layer, and source-recorded matched-control
  removals, followed by activation restoration;
- local runtime and CUDA memory counters.

The compressed NPZ contains all recorded activations and residual tensors.
Attention probability matrices are not captured. The per-head values are
projected residual writes and local first-order attributions, not attention
heatmaps.

## Acceptance assessment

Evidence labels in this report: **VERIFIED** means independently checked
against the regenerated machine-readable report and raw tensor archive;
**NOT ESTABLISHED** means this experiment does not support the claim.

| Criterion | Result | Evidence / boundary |
|---|---|---|
| A. Internal access | **PASS** | Real GPT-2 tensors were captured at all requested module boundaries and independently checked against the archive and model equations. |
| B. Computational trace | **PASS** | Full layerwise residual, attention, and MLP traces are recorded. The maximum measured residual-add reconstruction error is `0`. |
| C. Causal validation | **PARTIAL** | Removing selected components/path changes target logits in all five cases, and restoration returns the baseline exactly. The selected path beats its selected-MLP-plus-random-head control in all five cases; its matched random-active-neuron-plus-selected-attention control is comparable and exceeds the path effect for Italy. This does not isolate the selected head's effect from the paired path or establish unique MLP-neuron causality. |
| D. Explanatory output | **PARTIAL** | Each case has a measured WHAT/HOW/WHY/LIMITATIONS explanation. Three target facts are not the model's greedy next token, so their traces explain target-token scores, not an emitted next-token choice. Unit semantics and a complete circuit are not identified. |
| E. Reproducibility | **PASS** | Five exact prompts, source artifact hash, source path/control nodes, token IDs, full GPT-2 config, model/tokenizer identity, raw tensor inventory/hash, intervention outputs, and a verifier are included. |

**Overall conclusion: partially demonstrated, not a complete LLM reader.**
The experiment establishes instrumented internal access and quantitative
layerwise traces for these five inputs. It also measures causal score effects
under interventions and **causally** tests component removals against
restoration and controls. It does not establish a unique semantic explanation
of every output, a complete circuit, or black-box decompilation.

## Case results

`logit drop` is baseline target logit minus intervention target logit.
`probability change` is intervention target probability minus baseline target
probability. A positive logit drop means the intervention lowers the target
score; a negative probability change means it lowers the normalized target
probability.

| Subject | Target / rank | Greedy next token | Selected path | Attention removal Δlogit | MLP removal Δlogit | Joint path Δlogit | Joint Δprobability | Selected MLP + random head Δlogit | Selected attention + random active same-amplitude neuron Δlogit |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| France | ` Paris`, 5 | ` the` | L9 H8 → L11 N1369 | 1.727188 | 0.106323 | 1.818840 | -0.0138723 | 0.077339 | 1.740181 |
| Germany | ` Berlin`, 2 | ` the` | L9 H8 → L9 N687 | 1.214882 | 0.243340 | 1.388084 | -0.0104613 | 0.290451 | 1.060234 |
| Italy | ` Rome`, 1 | ` Rome` | L9 H8 → L9 N2746 | 3.036995 | 0.225784 | 3.218582 | -0.0670015 | 0.001274 | 3.250107 |
| Portugal | ` Lisbon`, 2 | ` the` | L6 H1 → L6 N2941 | 1.611923 | 0.470428 | 1.984413 | -0.0401915 | 0.759750 | 1.662148 |
| Greece | ` Athens`, 1 | ` Athens` | L6 H1 → L9 N1993 | 2.116165 | 0.459259 | 2.556099 | -0.110020 | 0.209572 | 2.416077 |

All five restorations recovered the baseline target logit and probability
exactly in the recorded float32 outputs. The selected path (selected attention
plus selected MLP neuron) produced a larger logit drop than the
selected-MLP-plus-random-attention-head control in all five cases. This is a
paired path comparison; it is **not** a head-only matched ablation. The
random-active-same-amplitude-neuron-plus-selected-attention control is close
to the selected path on France, Germany, Portugal, and Greece, and exceeds it
on Italy. The selected MLP neurons have measurable single-ablation effects,
but their unique causal specificity is **not established**.

The factual target was greedy for Italy and Greece only (`2/5`). For France,
Germany, and Portugal the target appears among the leading scored candidates
but is **not the greedy next token** the model would emit. This is why the
explanations distinguish “target-logit contribution” from “why the model
emitted that token.”

## Runtime and hardware

- Device recorded by the model: CUDA
- GPU: NVIDIA GeForce RTX 3070
- CUDA total memory: 8,214,085,632 bytes
- Peak allocated: 527,551,488 bytes
- Peak reserved: 564,133,888 bytes
- Python: 3.12.3; PyTorch: 2.10.0+cu128; Transformers: 5.5.0; NumPy: 2.4.6
- Model load: 0.480 seconds
- Five-case capture plus interventions: 0.605 seconds
- Total measured run: 1.893 seconds
- Raw tensor archive: 29,693,078 bytes

These are measurements from this run. No CPU comparison was run, so no GPU
speedup is claimed.

## Limits and next capability gate

This experiment does not identify semantic meanings for neurons or heads,
capture all attention-probability matrices, causally ablate every MLP neuron or
head, establish circuit completeness, or generalize beyond these five
capital-question prompts. It does not use a black-box/oracle-only solver.

The next capability gate should test whether the same explanations can be
produced for a broader, independently selected prompt set and whether selected
MLP components remain more informative than amplitude-matched active controls.
Thresholds or source cases should not be tuned based on this report.

## Artifacts and reproduction

- [Machine-readable report](../../benchmarks/release/nn-live-gpt2-internal-trace/five-facts/report.json)
- [Readable case explanations](../../benchmarks/release/nn-live-gpt2-internal-trace/five-facts/report.md)
- [Compressed raw tensors](../../benchmarks/release/nn-live-gpt2-internal-trace/five-facts/raw-tensors.npz)
- [Benchmark manifest](../../benchmarks/release/nn-live-gpt2-internal-trace/five-facts/manifest.json)
- [Original 50-fact ordered-path source report](../../benchmarks/release/nn-live-gpt2-ordered-residual-path-circuit-finder/gpt2-50-facts-ordered-residual-path/report.json)
- [Independent verifier](../../tools/llm_reader/verify_gpt2_internal_trace.py)
- [Focused tests](../../tests/test_gpt2_internal_trace.py)

Run the local five-case CUDA-capable experiment:

```bash
GCL_BENCHMARK_VARIANT=five-facts npm run benchmark:nn-live-gpt2-internal-trace
```

Verify the produced artifact and this claim boundary:

```bash
npm run verify:nn-live-gpt2-internal-trace
```
