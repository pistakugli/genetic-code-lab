# Agent Note: Swiglu split continuation

Status: implemented

## Context

The dense cubic regression rejects the real 896-dimensional Qwen SwiGLU oracle
before querying it. The prior 73,728-query student-distillation baseline ran at
full width but failed functional and structural evaluation. A new route must
fit the actual bias-free SwiGLU family while preserving the oracle-only boundary
and leaving baseline artifacts intact.

## Decision

The existing cubic probe runner exposes a separate split-continuation mode
implemented in `tools/llm_reader/swiglu_split_continuation.py`. Its student is
`W_down(SiLU(W_gate x) .* (W_up x))`. Width grows by duplicating each unit and
halving its output column, an exact function-preserving identity before
symmetric perturbations. At fixed gate/up parameters, the down projection is
solved by streamed accumulation of `H.T @ H` and `H.T @ Y`, followed by
regularized Cholesky. Failed Cholesky attempts escalate the relative ridge;
the chosen effective ridge is recorded per stage. Natural recovery inputs
define a mean-preserving shrinkage-covariance probe distribution and the
configured fraction of each optimization batch is sampled from natural
observations (0.5 by default; 0.75 in the expanded run). The solver
receives only public dimensions, saved observations, query budget, and an
opaque full-module oracle. Candidate width is a validation-selected student
width, never a teacher-count estimate.

The same runner exposes optional oracle-JVP supervision, disabled by default.
It estimates target directional derivatives with central differences at paired
oracle endpoints and computes student directional derivatives analytically.
The paired endpoints are reused as output-training observations and included
in the existing oracle-point budget. The configured JVP loss weight and query
fraction are recorded in the solver report.

The runner records successful oracle points even if fitting fails, persists
failure reports, tracks host RSS alongside CUDA memory, and saves the
reconstruction and provenance-bearing observations to a fresh output directory.

## Alternatives considered

- **Keep the cubic design as the only full-width method** — rejected because
  its explicit `R x M` float64 workspace is 308.4365 PiB for this target and
  its guard makes zero oracle queries.
- **Initialize from the teacher's 4,864 hidden units** — rejected because it
  leaks forbidden structure and cannot test measurement-driven model selection.
- **Use the selected surrogate width as a structural claim** — rejected because
  functional fitting and width selection do not establish component identity.

## Consequences

The method avoids the combinatorial cubic design and executes the real
896-dimensional oracle end to end. Exact unit duplication preserves the
current student function; perturbation and optimization are approximate. The
streamed normal-equation matrix still costs `O(k^2)` storage and can be
ill-conditioned, while fitting cost grows with width. The first 73,728-point
run selected 8,192 student components but had held-out functional NMSE
470.5759, distributional NMSE 35.0914, and no gate match at absolute cosine
0.99. An expanded run with 1,111 natural recovery observations and a 0.75
natural-batch fraction selected 2,048 student components. Its independent
natural functional NMSE was 0.3567 and distributional NMSE 0.3443; median
absolute gate cosine was 0.1371 with no matches at 0.99. The corrected
`/proc/self/status:VmHWM` parent peak in the final run was 3,098,435,584 bytes;
a race-clean process-tree monitor measured peak parent-plus-oracle PSS of
3,793,379,328 bytes (3.53 GiB) across 457 samples, below 4 GiB. The oracle
child's VmHWM was 3,710,382,080 bytes. Three JVP-supervised variants used the
same query budget and all had zero gate matches at 0.99. The best JVP variant
measured natural functional/distributional NMSE 0.4052/0.3919; the 0.01-weight
variant scored 0.4204/0.4092. Both are worse than the output-only baseline.
JVP improved held-out NMSE in two small synthetic cases at weight 0.05, but
this did not transfer to Qwen; it remains an opt-in experimental loss, not a
recovery improvement. The route is a scalable experimental fit, not a
hidden-structure recovery result.

## Verification

Eight focused continuation tests cover exact split/gauge invariance, the exact
third-derivative identity, oracle-only fitting on two synthetic architectures,
budget preflight, projection with duplicate components, and analytic JVP and
query-accounting behavior. Fourteen focused cubic/Qwen tests also pass,
including additional prompt-source SHA binding and process-memory monitoring.
The full-width solver and independent post-hoc evaluator reports preserve query
counts, artifact hashes, functional metrics, structural metrics,
host/process-tree/GPU measurements, and raw GPU-utilization samples.

## Cross-references

- [Qwen SwiGLU recovery research report](../../../../docs/research/2026-11-01-qwen2.5-0.5b-swiglu-oracle-recovery.md)
- [Qwen recovery roadmap](../../../../docs/nn-decompiler-roadmap.md)
- [Continuation solver](../../../../tools/llm_reader/swiglu_split_continuation.py)
- [Continuation tests](../../../../tests/test_swiglu_split_continuation.py)
