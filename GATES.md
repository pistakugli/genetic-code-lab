# Gates: real Qwen SwiGLU oracle recovery

OWNS: `tools/llm_reader/adapters/qwen2_swiglu.py`, `tools/llm_reader/adapters/registry.py`, `tools/llm_reader/blind_swiglu_oracle_decompiler.py`, `tools/llm_reader/cubic_identification_probe.py`, `tools/llm_reader/swiglu_split_continuation.py`, `tests/test_cubic_oracle_recovery.py`, `tests/test_swiglu_split_continuation.py`, `docs/nn-decompiler-roadmap.md`, `docs/research/2026-11-01-qwen2.5-0.5b-swiglu-oracle-recovery.md`, `benchmarks/release/real-qwen2.5-0.5b-layer0-swiglu/**`, `.agents/notes/implemented/feature/2026-11-01-real-qwen-activation-capture-verification.md`, `.agents/notes/implemented/feature/2026-11-01-adaptive-oracle-swiglu-distillation.md`, `.agents/notes/implemented/feature/2026-11-01-swiglu-split-continuation.md`, `.agents/notes/implemented/testing/2026-11-01-real-qwen-oracle-contract-tests.md`, `GATES.md`

Scope: capture and independently verify real Qwen2.5-0.5B layer-0 MLP activations, run full-width oracle-only recovery attempts, and independently score functional, structural, and distributional outcomes without claiming recovery that the evidence does not support.

- [x] G1: focused tests preserve synthetic recovery and exercise Qwen/oracle/artifact contracts
  CHECK: `./.gcl-python-gpu/bin/python tests/test_cubic_oracle_recovery.py && ./.gcl-python-gpu/bin/python tests/test_swiglu_split_continuation.py`
  EXPECT: 22 tests pass; synthetic recovery still returns 10 gates from measured span and exact SwiGLU synthetic fits pass
  CWD: repository root
  EVIDENCE: PASS (2026-11-01; 14 cubic/Qwen/artifact tests and 8 continuation tests; includes prompt-source SHA binding, process-memory metrics, synthetic equivalence, analytic student JVP checks, oracle-only query accounting, budget preflight, and artifact-integrity checks).

- [x] G2: real-model capture writes four disjoint natural-activation sets and generated oracle probes
  CHECK: `./.gcl-python-gpu/bin/python tools/llm_reader/cubic_identification_probe.py --capture-real-activations --capture-output-dir benchmarks/release/real-qwen2.5-0.5b-layer0-swiglu --live-model Qwen/Qwen2.5-0.5B-Instruct --live-layer 0 --live-local-files-only --live-max-calls 128 --device cuda --seed 941`
  EXPECT: capture report plus five provenance-separated NPZ artifacts
  CWD: repository root
  EVIDENCE: PASS; 42 recovery, 56 functional-evaluation, 39 structural-evaluation, 38 distribution-analysis, and 32 generated oracle-probe samples. Direct module/oracle delta 0.0; oracle query count 32/128. See [capture report](benchmarks/release/real-qwen2.5-0.5b-layer0-swiglu/capture-report.json).

- [x] G3: an independent verifier checks hashes/provenance and replays token IDs through the real model
  CHECK: `./.gcl-python-gpu/bin/python tools/llm_reader/cubic_identification_probe.py --verify-real-activation-capture --capture-output-dir benchmarks/release/real-qwen2.5-0.5b-layer0-swiglu --live-model Qwen/Qwen2.5-0.5B-Instruct --live-local-files-only --device cuda`
  EXPECT: `verification-report.json` records successful replay and artifact integrity
  CWD: repository root
  EVIDENCE: PASS; all natural-split input/output replay deltas and generated-probe output delta are 0.0; hashes, shapes, token positions, prompt provenance, and disjoint splits validated. The verifier reads local weights post-capture; no weights were provided to a recovery solver.

- [x] G4: research and architecture docs report the measured limits and score the three recovery objectives separately
  CHECK: `python3 -c 'from pathlib import Path; r=" ".join(Path("docs/research/2026-11-01-qwen2.5-0.5b-swiglu-oracle-recovery.md").read_text().lower().split()); d=" ".join(Path("docs/nn-decompiler-roadmap.md").read_text().lower().split()); need=("308.4365 pib","347,268,630,713,991,168","73,728 oracle points","0.9141","0.1350","0.3567","0.3443","3,098,435,584","3,793,379,328","not successful hidden-structure recovery"); assert all(x in r for x in need); assert "73,728" in d and "not hidden-structure recovery" in d; print("Qwen recovery evidence documentation present")'`
  EXPECT: `Qwen recovery evidence documentation present`
  CWD: repository root
  EVIDENCE: Updated from the final solver and independent post-hoc evaluator reports; failed recovery and resource-limit status are explicit.

- [x] G5: original synthetic measured-oracle recovery remains a regression
  CHECK: included in `./.gcl-python-gpu/bin/python tests/test_cubic_oracle_recovery.py`
  EXPECT: synthetic rank 20, 10 gates returned, 8640 oracle points
  CWD: repository root
  EVIDENCE: PASS; 2 synthetic oracle-only tests pass.

- [x] G6: live cubic access invariants remain intact
  CHECK: `npm run test:cubic-live-access`
  EXPECT: 5 live-access invariants pass
  CWD: repository root
  EVIDENCE: PASS (2026-11-01; oracle call accounting, truth lock ordering, and weight-access invariants)

- [ ] G7: scalable oracle-only recovery produces a useful reconstruction and independent functional, structural, and distributional scores
  CHECK: run the full-width recovery and separate evaluator commands in the "Full-width oracle-only recovery" section of the research report; verify the reports against the stated thresholds
  EXPECT: held-out natural NMSE < 0.1, nontrivial structural alignment modulo permutation/scaling, and host peak RSS <= 4 GiB at full 896-dimensional input/output
  CWD: repository root
  EVIDENCE: BLOCKED on fidelity and structure. The original 73,728-point run selected 256 components; functional-evaluation NMSE was 0.9575, distributional NMSE 0.9722, and no gate matched at 0.99. Its 8,605,700,096-byte RSS field came from `ru_maxrss * 1024` and is invalid in this runner: a same-process controlled allocation disagreed sharply with `/proc/self/status:VmHWM`. Actual RSS for that baseline is unknown. The exact cubic workspace estimate remains 347,268,630,713,991,168 bytes (308.4365 PiB). See the [research report](docs/research/2026-11-01-qwen2.5-0.5b-swiglu-oracle-recovery.md) and preserved `adaptive-oracle-distillation-streaming-batches/` artifacts.

- [ ] G8: function-preserving split continuation reaches the real full-width Qwen oracle within resource bounds and records independent evaluation
  CHECK: run `--recover-real-swiglu-continuation` and then `--evaluate-real-swiglu-distillation` using the expanded-final-v2 capture and a fresh output directory
  EXPECT: full 896-dimensional run; oracle budget is accounted; held-out natural NMSE < 0.1; nontrivial gate matches at absolute cosine >= 0.99; host VmHWM <= 4 GiB
  CWD: repository root
  EVIDENCE: FULL-WIDTH EXECUTION, QUERY, AND MEASURED HOST-MEMORY CONDITIONS PASS; recovery remains blocked. The race-clean Qwen2.5-0.5B layer-0 run selected a 2,048-component student from measurements, used 73,728/73,728 oracle points plus 1,111 captured natural observations, and took 44.46 s. Independent evaluation: 549 natural functional samples NMSE 0.3567; 537 distributional samples NMSE 0.3443; generated 8,192-point validation NMSE 0.5445. Structural evaluation found median absolute gate cosine 0.1371 and zero gates at 0.99 (teacher width 4,864 was read only post hoc). Parent VmHWM was 3,098,435,584 bytes; sampled parent-plus-oracle PSS peaked at 3,793,379,328 bytes over 457 samples (3.53 GiB), with oracle-child VmHWM 3,710,382,080 bytes. GPU utilization averaged 39.90% over 79 samples, max 99%; solver-sampled whole-device use peaked at 5,063,376,896 bytes. Monitor exited without exception. Solver and NPZ artifact hashes plus evaluator linkage match. See `adaptive-swiglu-split-continuation-natural75-final-tree-verified/` and the [research report](docs/research/2026-11-01-qwen2.5-0.5b-swiglu-oracle-recovery.md).
- [x] G9: optional JVP supervision is tested on exact synthetic SwiGLU and independently scored on Qwen
  CHECK: `./.gcl-python-gpu/bin/python tests/test_swiglu_split_continuation.py` plus independent evaluator reports under `benchmarks/release/real-qwen2.5-0.5b-layer0-swiglu/adaptive-swiglu-jvp-natural75-weight005/`, `adaptive-swiglu-jvp-natural75-weight005-fraction025/`, and `adaptive-swiglu-jvp-natural75-weight001-fraction025/`
  EXPECT: analytic student JVP and central-difference pairing pass; actual query count remains 73,728; functional, distributional, and structural scores are reported independently
  CWD: repository root
  EVIDENCE: Three Qwen runs used 73,728 points each; low-weight run selected 1,024 components, with natural functional/distributional NMSE 0.4204/0.4092 and zero gates at cosine 0.99. It did not beat the output-only baseline (0.3567/0.3443). Its evaluator is hash-linked to the solver report; raw query, reconstruction, and GPU-sample artifacts are preserved. Synthetic validation improved two small cases at weight 0.05, but did not predict Qwen performance.
