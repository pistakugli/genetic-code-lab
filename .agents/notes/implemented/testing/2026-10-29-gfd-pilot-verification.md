# Agent Note: Independent verification for GFD pilot results

Status: implemented

## Context

The GFD screening pilot writes raw runs under a separate directory and uses a smaller seed count than the confirmatory campaign. Without a verifier that independently reads those raw rows, a pilot summary could be internally inconsistent or confused with the full-campaign result.

## Decision

The benchmark renders a distinct pilot report from the ten-seed summary, labels it as screening rather than confirmation, and links to the pilot raw file. The result verifier accepts `--pilot`, reads only `scratch/results/gfd/pilot`, and independently checks the paired fixtures, complete run matrix, actual oracle counts, budget ceilings, CUDA diagnostics, summary metrics, and predeclared primary-cell acceptance calculations. The existing no-argument verifier continues to target the 100-seed campaign.

## Alternatives considered

- **Trust the benchmark summary as the pilot result** - rejected because it cannot independently establish that the raw records, metrics, and acceptance verdict agree.
- **Reuse the full-campaign verifier without a pilot mode** - rejected because it requires 100 seeds and reads a different output directory, preventing verification of the early-screening artifact.

## Consequences

Pilot evidence remains separate from confirmatory evidence, and the same acceptance calculation is applied to both. The verifier intentionally checks the ten-seed pilot's completeness but does not treat its sample size as formal 100-seed confirmation.

## Verification

`npm run verify:gfd-pilot` recomputes the pilot summary and verdict from the raw JSONL records and requires the separate pilot report to exist.

## Cross-references

- [GFD-VP pilot report](../../../../docs/research/2026-10-29-gfd-vp-pilot-report.md)
- [GFD-VP verifier](../../../../scripts/verify-gfd-results.ts)
