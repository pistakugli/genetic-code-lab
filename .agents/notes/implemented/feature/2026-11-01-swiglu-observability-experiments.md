# Agent Note: SwiGLU hidden-structure observability screen

Status: implemented

## Context

Output-only, JVP, and amplitude-polarity Qwen experiments did not recover
individual SwiGLU components. Before spending further real-model oracle calls,
the measurement design needed controlled evidence about whether changes in
finite-query input organization make hidden components more distinguishable
after their shared output projection.

## Decision

`tools/llm_reader/swiglu_observability_experiments.py` compares IID Gaussian,
coordinate-erasure, complementary-block, and sparse/dense/correlated query
families on exact synthetic full-module SwiGLU teachers. The measurement
interface receives an oracle callable, public input/output dimensions, a query
budget, and a seed. Teacher weights, width, and component labels are used only
by the separate post-measurement evaluator.

The study stores exact train/held-out inputs and outputs, derived output
readouts, and raw-file SHA-256 values under
`benchmarks/research/swiglu-hidden-structure-observability/`. Its independent
verifier replays raw observations, checks equivalence controls, recounts
queries, and enforces the real-model decision. Teacher-dependent fingerprints
are screening diagnostics only; they cannot authorize a Qwen pilot without an
oracle-only structural recovery witness.

## Alternatives considered

- **Treat evaluator-only component fingerprints as recovered structure** —
  rejected because their construction reads the hidden teacher factors and
  they are not available to the black-box solver.
- **Run Qwen after any positive fingerprint movement** — rejected because
  small-sample evaluator diagnostics do not establish information recoverable
  from summed oracle outputs.
- **Repeat amplitude, derivative, tensor, or moment methods** — rejected
  because those approaches are frozen or explicitly excluded from this
  experiment.

## Consequences

The experiment uses 21 teachers across three seeds and seven controlled
conditions, with 384 observations for each of four protocols (32,256 total).
Coordinate erasure, complementary blocks, and activation environments all
miss the predeclared median/P95 screen on the required number of seeds. No
oracle-only recovery solver is implemented and no new Qwen oracle calls are
made. Exact permutation/reciprocal-scaling, duplicate-gate GL(2), and
zero-output-column equivalences are retained as identifiability controls.

The API boundary is tested, but the synthetic harness is not a
process-isolation security boundary. Generic full-rank parameter
identifiability remains unknown. The next synthetic gate measures observable
output-span rank; it can bound `rank(D)` but cannot by itself infer hidden
component count.

## Verification

The focused unittest suite passes. Independent artifact stages validate the
hypotheses, all raw observation hashes and output replays, the preregistered
decision, and the report/roadmap. The full synthetic study records 32,256
oracle points, maximum held-out replay absolute error `0.0`, maximum tested
equivalence absolute error below `8.53e-14`, `42,168 KiB` peak process RSS,
and `0.36 s` wall time. It uses NumPy on CPU; VRAM is not measured.

## Cross-references

- [Research report](../../../docs/research/2026-11-01-qwen2.5-0.5b-swiglu-oracle-recovery.md#synthetic-hidden-structure-observability-study-2026-11-01)
- [Roadmap](../../../docs/nn-decompiler-roadmap.md)
- [Study implementation](../../../tools/llm_reader/swiglu_observability_experiments.py)
- [Focused tests](../../../tests/test_swiglu_observability_experiments.py)
- [Artifact verifier](../../../tests/verify_swiglu_observability_artifacts.py)
- [Raw study artifacts](../../../benchmarks/research/swiglu-hidden-structure-observability/)
