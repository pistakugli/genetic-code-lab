# Agent Note: GPT-2 counterfactual circuit transfer

Status: implemented

## Context

The existing GPT-2 path experiments establish influence but do not distinguish
fact-specific information from shared or generic predictive computation.
Activation-magnitude coverage in earlier controls was insufficient, and path
sufficiency had not been tested. Treating unmatched controls as valid would
overstate what these interventions identify.

## Decision

The experiment derives paired facts only from literal repeated prompt
templates, independently recorded answers, and frozen discovered paths in the
50-fact source artifact. It transfers source-path activations only across
identical token IDs in fixed prompt text; tokens intersecting a fact slot are
never aligned or patched. It measures the donor-versus-recipient answer
sequence mean-log-likelihood contrast under teacher forcing, path removal,
identity restoration, and structurally matched random path transplants.

Controls preserve path component count, type, layer, and token positions.
Magnitude is compared using the attention head's pre-projection activation
vector L2 norm and the MLP's post-GELU scalar absolute value. The 20% tolerance
and 100 controls per direction are frozen; insufficient magnitude overlap
invalidates specificity inference rather than relaxing the match. Pair
statistics resample fact endpoints and apply Benjamini-Hochberg correction
across analyzable relation families.

This is an instrumented local-model witness, not an oracle-only solver or a
claim of complete circuit sufficiency. Transfer, necessity, recovery, and
specificity remain distinct outcomes.

## Alternatives considered

- **Reuse greedy next-token logit change only** — rejected because the same
  factual answer may be multi-token and a changed first-token logit does not
  measure transfer of the full answer sequence.
- **Call any path ablation effect necessary or sufficient** — rejected because
  causal influence does not establish necessity, and removal in one context
  does not establish counterfactual sufficiency.
- **Expand the control tolerance until coverage passes** — rejected because
  post-hoc relaxation would turn unmatched component magnitudes into invalid
  specificity evidence.
- **Use the recipient's own discovered path as the transplant target** —
  rejected because that would silently replace the donor's frozen path
  components rather than test whether the donor path's activations transfer.

## Consequences

The test uses the source dataset's existing relation coverage; five of the
passed facts do not fit a repeated prompt family, and 33 skipped source rows
remain outside the 50 passed-fact experiment. Full pairwise transfer and 100
control groups per direction cost substantially more local inference than the
earlier five-case experiment. A rare or high-amplitude attention head may have
no adequate magnitude-matched alternatives; in that case, the experiment
reports transfer descriptively and leaves specificity unresolved. Transfer
does not imply a complete sufficient circuit.

The executed campaign covers all 744 eligible pairs and 1,488 directions.
Only 38 directions have valid structural and magnitude-matched controls; 37
meet the frozen per-direction specificity criterion. Corrected positive
path-versus-control evidence is limited to the country-capital family and
those control-valid directions. This narrow result does not establish
specificity for the other 1,444 directions or the three two-fact families.
Measured execution used an RTX 3070; no GPU speedup comparison was performed.

## Verification

The focused tests cover exact token alignment, rejection of fact-slot
positions, direct activation patching and recovery, deterministic structural
controls, and verifier rejection. The independent verifier checks the frozen
source and earlier artifact hashes, re-derives eligible pairs, validates raw
archive metadata and intervention arithmetic, and replays donor activations,
transfers, and recipient recovery against local GPT-2. It also independently
recomputes endpoint-bootstrap intervals, p-values, and Benjamini-Hochberg
adjustments from raw direction measurements. The corrected ten-pair pilot and
complete pair campaign each have separate raw archives and reports.

## Cross-references

- [Counterfactual transfer runner](../../../../tools/llm_reader/gpt2_counterfactual_transfer.py)
- [Independent verifier](../../../../tools/llm_reader/verify_gpt2_counterfactual_transfer.py)
- [Statistics finalizer](../../../../tools/llm_reader/finalize_gpt2_counterfactual_transfer.py)
- [Focused tests](../../../../tests/test_gpt2_counterfactual_transfer.py)
- [Research report](../../../../docs/research/2026-10-30-gpt2-counterfactual-circuit-transfer.md)
