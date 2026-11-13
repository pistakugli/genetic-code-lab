# Agent Note: Pythia-70m OOD replacement transfer

Status: implemented

## Context

The why-report layer had recorded evidence from exactly one model family: the
GPT-2 France case recorded on 2026-11-04. Every break diagnosis, replacement
comparison, and transfer diagnosis therefore rested on a single adapter, and
the open continuation asked for evidence from a second live model. The GPU
lanes (LLaMA-3.2, Mistral-7B) cannot run in the continuation sandbox, but
Pythia-70m is CPU-feasible and its recorded circuit-finder artifact is a valid
stress-lane input.

## Decision

Replay the recorded Pythia-70m-deduped France case (attention L0 h6 + MLP L1
n309, combined drop 2.0) live on CPU with the stability runner in OOD profile:
one case, six held-out families, layer-shift rescue, replacement search, and
cross-family transfer at the lane defaults. Record the artifact under
`ood-replacement-transfer-pythia-70m-2026-11-14` with a suite report, per-model
report, and manifest that state the CPU device and the actual python
invocation. The run needed two enabling changes: the independent verifier now
accepts recorded `cpu` live runs alongside `cuda` (unknown devices are still
rejected), and the rescue budget defaults to covering all six OOD families so
no recorded break is left without a why-report.

## Alternatives considered

- **Run the LLaMA-3.2 or Mistral-7B lane** — rejected because the continuation
  sandbox has no GPU and the four-bit adapters are not CPU-feasible.
- **Post-process a synthetic second-family artifact** — rejected because the
  layer's value is explaining measured, not constructed, breaks.
- **Keep the rescue budget at 2 and document the gap** — rejected because the
  lane's north-star is that every recorded break gets a why-report.

## Impact

The original replay reproduces the recorded drop exactly (absolute replay
error 0.0), so the CPU run measures the same circuit the GPU pilot recorded.
Five of six OOD families break as magnitude-collapse; the service-question
break is a pure layer misplacement recovered by a +1 layer shift; the
replacement search finds no passing path for any break, so transfer is not
attempted. The stress gate fails honestly at the configured thresholds. This
is a measurement of one path's OOD robustness in a six-layer model, not a
claim about Pythia-70m generally.

## Verification

The independent verifier recomputes every why-report block from the raw
rescue measurements and accepts the new artifact; the 2026-11-04 GPT-2
artifact still verifies after the verifier change. `tests/test_ordered_path_ood_stress.py`
grows to 11 tests, covering both the accepted `cpu` device and the rejection
of unknown device strings.

## Cross-references

- [Research report](../../../../docs/research/2026-11-14-pythia-70m-ood-replacement-transfer.md)
- [Why-report layer note](2026-10-30-ordered-path-ood-stress.md)
