# GPT-2 internal computation trace

- Model: `gpt2` (`GPT2LMHeadModel`)
- Device: `cuda`
- Source artifact SHA-256: `f24428b69b9b1fb2d750dfbf2ea55a065fc1d6cc740c415d01ac6ccb94e3715b`
- Overall instrumentation status: `measured`
- Causal validation: `measured-interventions-with-restoration-neighbor-and-source-controls`
- Explanation scope: `five configured GPT-2 capital-fact prompts only`

This is an instrumented witness run using the local model and hooks. It is not a black-box decompilation claim.

## Factual-input comparison

| Subject | Exact prompt | Target/rank | Selected attention | Selected MLP | Target probability | Joint removal logit drop |
|---|---|---|---|---|---:|---:|
| France | `The capital of France is` | ` Paris` (rank 5) | L9 H8 | L11 N1369 | 0.0322444 | 1.818840 |
| Germany | `The capital of Germany is` | ` Berlin` (rank 2) | L9 H8 | L9 N687 | 0.0527268 | 1.388084 |
| Italy | `The capital of Italy is` | ` Rome` (rank 1) | L9 H8 | L9 N2746 | 0.157348 | 3.218582 |
| Portugal | `The capital of Portugal is` | ` Lisbon` (rank 2) | L6 H1 | L6 N2941 | 0.0610085 | 1.984413 |
| Greece | `The capital of Greece is` | ` Athens` (rank 1) | L6 H1 | L9 N1993 | 0.177902 | 2.556099 |

All five cases use the same capture and intervention procedure. Different subject prompts select different MLP neurons and sometimes different attention layers; this is an observed difference in these source paths, not a semantic interpretation of those units.

## Per-case explanations

### France: The capital of France is

Target token: ` Paris` (id 6342); baseline rank 5 with logit -101.214272 and probability 0.032244433. Greedy next token is ` the` (id 262, probability 0.084592521); the target is not the greedy next token.

**WHAT?**
The source artifact selects attention head 8 at layer 9 and MLP neuron 1369 at layer 11. The selected MLP's measured activation at the read position is 1.010344; its local first-order target-logit attribution is 0.113079.

**HOW?**
The input tokenizes to 5 tokens. The recorded trace has 12 transformer layers. At the selected attention site, the head residual-write norm is 22.190819; at the selected MLP site, the selected neuron's projected residual-contribution norm is 5.312822. Each layer's residual addition was reconstructed from its block input, attention write, and MLP write with maximum absolute error 0.

**WHY?**
Removing the selected attention component changed the target logit by 1.727188; removing the selected MLP neuron changed it by 0.106323; removing both changed it by 1.818840 and changed target probability by -0.01387232. Restoring the original selected activations returned the target logit to within 0 of baseline. The selected path exceeded its source-recorded selected-mlp-plus-random-attention-head control (0.077339). The source-recorded random-active-same-amplitude-plus-selected-attention control changed the target logit by 1.740181; it did not exceed the selected path effect. Neighbor-layer removal effects were attention-head-8-layer-8: -0.137222, mlp-neuron-1369-layer-10: 0.047775, attention-head-8-layer-10: -0.455750. These comparisons measure this prompt's interventions; they do not assign semantic labels or establish a unique complete circuit.

**LIMITATIONS?**
This is an instrumented, model-specific trace using local GPT-2 weights and hooks, not black-box decompilation. First-order neuron/head attributions are local gradient approximations; the reported ablations are the causal evidence. The trace does not explain all 12 layers' learned semantics, all token-to-token attention probabilities, or the full model's behavior outside these five prompts. The target is not the greedy next token in this case, so the intervention supports a claim about its scored logit, not that it was the token actually emitted.

### Germany: The capital of Germany is

Target token: ` Berlin` (id 11307); baseline rank 2 with logit -98.473915 and probability 0.05272685. Greedy next token is ` the` (id 262, probability 0.089100756); the target is not the greedy next token.

**WHAT?**
The source artifact selects attention head 8 at layer 9 and MLP neuron 687 at layer 9. The selected MLP's measured activation at the read position is 0.850261; its local first-order target-logit attribution is 0.264045.

**HOW?**
The input tokenizes to 5 tokens. The recorded trace has 12 transformer layers. At the selected attention site, the head residual-write norm is 19.612206; at the selected MLP site, the selected neuron's projected residual-contribution norm is 3.238173. Each layer's residual addition was reconstructed from its block input, attention write, and MLP write with maximum absolute error 0.

**WHY?**
Removing the selected attention component changed the target logit by 1.214882; removing the selected MLP neuron changed it by 0.243340; removing both changed it by 1.388084 and changed target probability by -0.010461267. Restoring the original selected activations returned the target logit to within 0 of baseline. The selected path exceeded its source-recorded selected-mlp-plus-random-attention-head control (0.290451). The source-recorded random-active-same-amplitude-plus-selected-attention control changed the target logit by 1.060234; it did not exceed the selected path effect. Neighbor-layer removal effects were attention-head-8-layer-8: 0.036377, mlp-neuron-687-layer-8: -0.154167, attention-head-8-layer-10: -0.526321, mlp-neuron-687-layer-10: 0.001106. These comparisons measure this prompt's interventions; they do not assign semantic labels or establish a unique complete circuit.

**LIMITATIONS?**
This is an instrumented, model-specific trace using local GPT-2 weights and hooks, not black-box decompilation. First-order neuron/head attributions are local gradient approximations; the reported ablations are the causal evidence. The trace does not explain all 12 layers' learned semantics, all token-to-token attention probabilities, or the full model's behavior outside these five prompts. The target is not the greedy next token in this case, so the intervention supports a claim about its scored logit, not that it was the token actually emitted.

### Italy: The capital of Italy is

Target token: ` Rome` (id 10598); baseline rank 1 with logit -95.563095 and probability 0.15734816. Greedy next token is ` Rome` (id 10598, probability 0.15734816); the target is the greedy next token.

**WHAT?**
The source artifact selects attention head 8 at layer 9 and MLP neuron 2746 at layer 9. The selected MLP's measured activation at the read position is 0.895278; its local first-order target-logit attribution is 0.234307.

**HOW?**
The input tokenizes to 5 tokens. The recorded trace has 12 transformer layers. At the selected attention site, the head residual-write norm is 24.560869; at the selected MLP site, the selected neuron's projected residual-contribution norm is 3.533776. Each layer's residual addition was reconstructed from its block input, attention write, and MLP write with maximum absolute error 0.

**WHY?**
Removing the selected attention component changed the target logit by 3.036995; removing the selected MLP neuron changed it by 0.225784; removing both changed it by 3.218582 and changed target probability by -0.067001462. Restoring the original selected activations returned the target logit to within 0 of baseline. The selected path exceeded its source-recorded selected-mlp-plus-random-attention-head control (0.001274). The source-recorded random-active-same-amplitude-plus-selected-attention control changed the target logit by 3.250107; it exceeded the selected path effect. Neighbor-layer removal effects were attention-head-8-layer-8: -0.140297, mlp-neuron-2746-layer-8: -0.099419, attention-head-8-layer-10: -0.437294, mlp-neuron-2746-layer-10: 0.011971. These comparisons measure this prompt's interventions; they do not assign semantic labels or establish a unique complete circuit.

**LIMITATIONS?**
This is an instrumented, model-specific trace using local GPT-2 weights and hooks, not black-box decompilation. First-order neuron/head attributions are local gradient approximations; the reported ablations are the causal evidence. The trace does not explain all 12 layers' learned semantics, all token-to-token attention probabilities, or the full model's behavior outside these five prompts. The target is the greedy next token for this prompt, but this five-case trace does not establish general factual behavior.

### Portugal: The capital of Portugal is

Target token: ` Lisbon` (id 41898); baseline rank 2 with logit -95.585167 and probability 0.061008472. Greedy next token is ` the` (id 262, probability 0.091947228); the target is not the greedy next token.

**WHAT?**
The source artifact selects attention head 1 at layer 6 and MLP neuron 2941 at layer 6. The selected MLP's measured activation at the read position is 0.498584; its local first-order target-logit attribution is 0.467253.

**HOW?**
The input tokenizes to 5 tokens. The recorded trace has 12 transformer layers. At the selected attention site, the head residual-write norm is 10.205388; at the selected MLP site, the selected neuron's projected residual-contribution norm is 1.264254. Each layer's residual addition was reconstructed from its block input, attention write, and MLP write with maximum absolute error 0.

**WHY?**
Removing the selected attention component changed the target logit by 1.611923; removing the selected MLP neuron changed it by 0.470428; removing both changed it by 1.984413 and changed target probability by -0.040191464. Restoring the original selected activations returned the target logit to within 0 of baseline. The selected path exceeded its source-recorded selected-mlp-plus-random-attention-head control (0.759750). The source-recorded random-active-same-amplitude-plus-selected-attention control changed the target logit by 1.662148; it did not exceed the selected path effect. Neighbor-layer removal effects were attention-head-1-layer-5: -1.448959, mlp-neuron-2941-layer-5: 0.015480, attention-head-1-layer-7: 0.078262, mlp-neuron-2941-layer-7: -0.001274. These comparisons measure this prompt's interventions; they do not assign semantic labels or establish a unique complete circuit.

**LIMITATIONS?**
This is an instrumented, model-specific trace using local GPT-2 weights and hooks, not black-box decompilation. First-order neuron/head attributions are local gradient approximations; the reported ablations are the causal evidence. The trace does not explain all 12 layers' learned semantics, all token-to-token attention probabilities, or the full model's behavior outside these five prompts. The target is not the greedy next token in this case, so the intervention supports a claim about its scored logit, not that it was the token actually emitted.

### Greece: The capital of Greece is

Target token: ` Athens` (id 21891); baseline rank 1 with logit -99.606049 and probability 0.17790157. Greedy next token is ` Athens` (id 21891, probability 0.17790157); the target is the greedy next token.

**WHAT?**
The source artifact selects attention head 1 at layer 6 and MLP neuron 1993 at layer 9. The selected MLP's measured activation at the read position is 0.931496; its local first-order target-logit attribution is 0.471413.

**HOW?**
The input tokenizes to 5 tokens. The recorded trace has 12 transformer layers. At the selected attention site, the head residual-write norm is 9.566257; at the selected MLP site, the selected neuron's projected residual-contribution norm is 3.810665. Each layer's residual addition was reconstructed from its block input, attention write, and MLP write with maximum absolute error 0.

**WHY?**
Removing the selected attention component changed the target logit by 2.116165; removing the selected MLP neuron changed it by 0.459259; removing both changed it by 2.556099 and changed target probability by -0.11001961. Restoring the original selected activations returned the target logit to within 0 of baseline. The selected path exceeded its source-recorded selected-mlp-plus-random-attention-head control (0.209572). The source-recorded random-active-same-amplitude-plus-selected-attention control changed the target logit by 2.416077; it did not exceed the selected path effect. Neighbor-layer removal effects were attention-head-1-layer-5: -2.002075, mlp-neuron-1993-layer-8: -0.032204, attention-head-1-layer-7: 0.099495, mlp-neuron-1993-layer-10: 0.036430. These comparisons measure this prompt's interventions; they do not assign semantic labels or establish a unique complete circuit.

**LIMITATIONS?**
This is an instrumented, model-specific trace using local GPT-2 weights and hooks, not black-box decompilation. First-order neuron/head attributions are local gradient approximations; the reported ablations are the causal evidence. The trace does not explain all 12 layers' learned semantics, all token-to-token attention probabilities, or the full model's behavior outside these five prompts. The target is the greedy next token for this prompt, but this five-case trace does not establish general factual behavior.

## Trace fields captured

Raw arrays are stored in the adjacent compressed NPZ archive. The archive contains token and position embeddings, every block input/output residual, attention projection contexts and per-head residual writes, MLP inputs/preactivations/activations/residual writes, final normalization, full vocabulary logits, and first-order attribution arrays.

Attention head and MLP neuron first-order values are gradient-times-write local approximations. They are reported separately from causal ablation results.

## Claim boundary

This is not a black-box solver result, complete LLM decompilation, semantic interpretation of units, a complete circuit, exhaustive attention/MLP causal decomposition, generalization beyond these five facts, or proof that the factual target was emitted when it was not the greedy next token.
