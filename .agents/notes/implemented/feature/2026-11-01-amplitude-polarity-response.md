# Agent Note: Amplitude polarity response

Status: implemented

## Context

Prior output-only and JVP SwiGLU recovery experiments did not recover Qwen's
hidden components. The amplitude-polarity hypothesis was that large positive
and negative input responses could reveal structure hidden in ordinary
observations. The experiment needed to keep the solver oracle-only and retain
all finite-amplitude samples so lack of convergence or structural evidence
would remain visible.

## Decision

The existing `cubic_identification_probe.py` runner exposes separate
`--probe-real-swiglu-amplitude-response` and
`--evaluate-real-swiglu-amplitude-response` modes. The probe constructs
provenance-tracked natural, mixed-natural, and independent Gaussian
directions, queries both polarities over logarithmic amplitudes using the
opaque full-module oracle, and saves raw inputs, outputs, and normalized
trajectory measurements. It rejects a query budget below the exact design
count and reports query/resource accounting.

The evaluator runs separately, verifies hashes and oracle replay, then reads
teacher weights only post-hoc to compare exact limits and component-response
signatures. Neither those weights nor the teacher unit count enter probing.
The evaluator explicitly does not report a recovery result. The focused
artifact verifier checks probe accounting, raw arrays and hashes, evaluator
linkage, and the documentation's negative verdict.

## Alternatives considered

- **Extrapolate only the largest responses:** rejected because it would hide whether measured trajectories had actually stabilized and would discard the finite-amplitude information under test.
- **Build a component solver before evaluating the measurements:** rejected because post-hoc signatures cannot prove that the summed oracle outputs identify their constituent components.
- **Use teacher weights during probing:** rejected because that violates the oracle-only boundary; weights are limited to the separate evaluator.

## Consequences

The implementation provides reproducible full-width Qwen trajectories and
post-hoc convergence/signature diagnostics. The measured trajectories did not
meet the 1% stability criterion at amplitude 512, and the measurements did
not justify a recovery solver. `P(-u)=N(u)` makes the two asymptotic polarities
direction reversals, not independent limiting response families. The next
candidate is adaptive activation-boundary localization, which requires
separate synthetic identifiability evidence before a real-model pilot.

## Verification

Five synthetic tests pass, including the exact asymptotic polarity identities,
finite-amplitude controls, provenance, stability metrics, and invalid-input
handling. The real run used 2,304/2,304 opaque oracle points; its independent
replay maximum per-sample relative L2 error was `7.04e-7`. Probe and evaluator
reports are linked by the solver-report SHA-256, and the raw artifact hash
matches both reports. Measured resource figures and negative stability results
are recorded in the research report and experiment artifacts. The focused
artifact verifier has separate probe, evaluation, and documentation stages.

## Cross-references

- [Research report](../../../../docs/research/2026-11-01-qwen2.5-0.5b-swiglu-oracle-recovery.md)
- [Roadmap](../../../../docs/nn-decompiler-roadmap.md)
- [Probe report](../../../../benchmarks/release/real-qwen2.5-0.5b-layer0-swiglu/amplitude-polarity-response/probe-report.json)
- [Post-hoc evaluation](../../../../benchmarks/release/real-qwen2.5-0.5b-layer0-swiglu/amplitude-polarity-response/evaluation-report.json)
