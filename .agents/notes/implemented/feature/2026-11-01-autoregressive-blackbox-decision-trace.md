# Agent Note: Autoregressive blackbox decision trace

Status: implemented

## Context

The existing output-only reader measured one next-token decision and prompt
deletions, so it could not expose a generated response as a sequence of
observable decisions or test whether earlier generated tokens affected later
ones. A bounded trace makes that sequential behavior inspectable without
claiming access to hidden model state.

## Decision

The existing `explain-llm-blackbox` command accepts `--trace` and greedily
generates a bounded response. At each step it records the chosen token,
probability, rank, top alternatives, margin, and bounded deletion effects for
selected prompt and generated-prefix text units made of exact token IDs. The
parent process owns the tokenizer; a separate model worker accepts token-ID
sequences and returns logits, keeping weights outside the reader/solver
process. With
`--response-probes`, it ranks measured prompt units by their strongest observed
next-token margin effect and regenerates the full greedy response after each
selected unit deletion. Per-step input probes rotate toward unmeasured units.
With `--explanation-probes`, it searches a bounded number of oracle queries per
generated token for a whitespace-aware subset of editable text units that
retains the target argmax. Special tokens and chat headers remain protected.
With `--interaction-probes`, it measures a bounded set of pairs among the
individually probed units, prioritizing units by summed absolute single-deletion
margin changes, and reports the joint target-vs-competitor margin against the
additive prediction from their single deletions. JSON and Markdown
reports include the exact counterfactual token IDs, alternative response,
first divergence, sufficient contexts, local single-unit deletion witnesses,
pairwise interaction residuals, query budget, and oracle-call accounting.
With `--branch-probes`, it forces selected non-greedy next-token candidates
and expands their continuations with a bounded beam controlled by
`--branch-width`. Candidate beams are ranked by cumulative token
log-probability, stop at EOS, and remain within the remaining generation-token
budget. Reports include each retained continuation, its cumulative and mean
token log-probability, first divergence, and per-branch query count.

## Alternatives considered

- **Keep only a single next-token decision:** rejected because it cannot show
  the observable greedy response path or measure effects of generated-prefix
  deletions on later decisions.
- **Use hooks or hidden activations to explain each token:** rejected for this
  feature because it would change the oracle-only boundary and conflate
  instrumented circuit evidence with black-box behavior evidence.

## Consequences

The reader gains a multi-step, prompt-specific behavioral trace using the
existing CLI. Full-response counterfactuals show whether selected prompt-token
unit deletions alter the continuation, but deletion may damage formatting or
otherwise make the prompt unnatural. A one-minimal sufficient context means
only that deleting any one remaining editable unit changed the top-1 token;
the search does not prove a global minimum or a semantic explanation. Greedy
decoding does not characterize sampling policies; response changes do not
establish semantic necessity or unique internal causes. Each probe recomputes
logits from the complete token sequence, so query/runtime cost grows with trace
length and intervention budget.
Pairwise interaction residuals measure non-additivity of the selected exact
deletion interventions; they do not identify a hidden circuit, semantic
relation, or complete reasoning process.
Forced-token beams measure selected high-probability observable continuations
after an exact token-choice intervention. They do not enumerate all possible
continuations or reveal the model's hidden internal reasoning. Query cost is
bounded by the beam width and remaining generation budget, but can still be
substantial across many decisions and alternatives.

## Verification

The beam implementation is covered by 17 focused Python tests and Python
compilation. Synthetic oracle tests verify width-two continuation ranking,
EOS termination without an extra query, invalid-width rejection before oracle
access, and exact query accounting. TypeScript type-checking passes, the public
CLI help exposes `--branch-width`, and an invalid public width is rejected
before model loading. The scoped ledger passes its syntax lint.

Earlier CUDA Qwen2.5-0.5B-Instruct reports measured greedy forced-token
continuations: one run used 15 branches, 115 branch queries, and 131 total
queries; an expanded run used 45 branches, 366 branch queries, and 382 total
queries. They predate beam expansion and remain historical evidence only. A
fresh Qwen run has not yet verified the beam implementation.

An earlier CUDA
Qwen2.5-0.5B-Instruct run with an explicit chat-formatted prompt generated
`The solution to 17 + 25 is 42.` in 16 tokens. The run used 192 oracle calls:
32 context-unit interventions, 114 sufficient-context queries, and 30
full-response counterfactual queries. The report reconciled the worker and
reader counts. Deleting the measured `25` unit changed the response to
`The equation 17 + 0 equals 17.` Eight steps reached one-minimality among
editable text units and eight exhausted their per-step budget. The JSON access
contract reports token-ID-to-logits-only solver access and excludes weights,
hooks, and activations.

## Cross-references

- [Claim policy](../../../docs/claim-policy.md)
- [NN decompiler roadmap](../../../docs/nn-decompiler-roadmap.md)
