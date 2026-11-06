# Agent Note: Ordered-path why-report layer

Status: implemented

## Context

The live ordered-path OOD stress lane recorded *that* source circuits break
under paraphrase families (rescue replay drops, replacement searches,
held-out transfer), but the artifact never said *why* each break happened or
where the measured causal effect went. North-star reader step 7 requires the
reader to explain why selected components react using recorded evidence, so
the lane needed a per-break explanation assembled from its own measurements.

## Decision

Added `tools/llm_reader/why_report.py`, a deterministic post-processing layer
over recorded ordered-path stability/OOD-stress artifacts. For every broken
paraphrase rescue it records:

- break classification: `sign-flip` (ablation raises the target logit),
  `magnitude-collapse` (positive drop below the absolute gate), or
  `relative-falloff` (clears the absolute gate but misses the ratio gate);
- layer-shift rescue outcome: best shifted-layer replay drop versus the
  broken drop, i.e. whether the effect merely moved with the components;
- replacement-path comparison against the source circuit: layer movement with
  direction, component overlap count, ablate-position movement, and a change
  kind (`component-rebinding` when overlap is below the configured threshold,
  else `component-retention`);
- held-out family transfer diagnosis for the replacement path with per
  failing family evidence;
- a why statement whose every number comes from fields already in the
  artifact, plus an explicit claim boundary.

`attach_why_analysis()` runs inside the runner's `main()` so every future
artifact carries the layer, and `render_why_markdown()` renders a
"Why the path broke" section inside the runner's own markdown output.
`verify_ordered_path_ood_stress.py` was extended to recompute the break
classification, change kind, and transfer diagnosis from the raw rescue and
transfer fields, so a fabricated or edited why block fails verification.

## Alternatives considered

- **Free-text narrative without recomputation:** rejected because prose that
  is not recomputed from raw fields cannot be verified and invites drift from
  the measurements.
- **New model queries to "explain" the break:** rejected because the
  why-report must be deterministic post-processing of recorded numbers; it
  performs no model queries and invents no values.
- **Keep the analysis in an offline demo script only:** rejected because
  step 7 belongs to the lane itself; the runner must emit it for every run,
  not just for curated demos.

## Consequences

Artifacts in the new format (rescue measurements present) must now carry
per-case and aggregate why blocks or verification fails; historical artifacts
without the layer still pass under the old rules. The markdown report grows a
why section per case with a replacement-comparison table. The why statements
are bounded by the recorded measurements: they explain where the effect went
mechanically (sign flip, layer movement, rebinding) and explicitly do not
claim semantic reasons beyond those numbers. Layer-shift rescue outcomes are
reported even when no rescue was found, which keeps negative evidence
first-class.

## Verification

`scripts/verify-why-report-cycle.py` (worktree) runs the full cycle against
the enriched real-model artifact
`benchmarks/release/nn-live-ordered-path-stability/ood-replacement-transfer-2026-11-04/gpt2/report.json`:

- positive: enriched artifact passes the extended verifier;
- negative: tampered break drop rejected ("broken drop differs");
- negative: fabricated change kind rejected ("change kind disagrees");
- negative: fabricated failing transfer family rejected ("failing family
  diagnosis differs");
- negative: stripped why block on a new-format artifact rejected;
- backward compat: historical artifact without the layer passes old rules.

All checks pass. `scripts/check-runner-markdown-integration.py` confirms
the why section renders inside the runner's own `make_markdown` output with
all anchors present. The demo copy for review lives under
`download/adaptive-circuit-reader-demo/` (enriched JSON, why-only markdown,
full runner markdown).

On the GPT-2 France imperative case the layer reports: sign-flip break
(broken replay drop -0.8927 versus 1.8188 on the source prompt), layer-shift
rescue not found (best shifted drop -0.0937 at shift -3), replacement path
found as component-rebinding (MLP L6 n834 + attention L6 h5, layers
[9, 11] -> [6], 0/2 components retained, ablate positions [4] -> [9]), and
held-out transfer 4/5 with service-question failing.

## Cross-references

- [Why-report module](../../../../tools/llm_reader/why_report.py)
- [Ordered-path stability runner](../../../../tools/llm-reader-ordered-path-stability.py)
- [Extended verifier](../../../../tools/llm_reader/verify_ordered_path_ood_stress.py)
- [Ordered-path OOD stress research note](../../research/2026-10-30-ordered-path-ood-stress.md)
- [North star](../../../../docs/gcl-north-star.md)
