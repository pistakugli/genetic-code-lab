# Recommended next development direction: blind recovery on real activation distributions

## One direction

Develop and evaluate **unknown-order, real-activation-distribution function recovery for selected SwiGLU blocks**.

This is a research direction only. This audit does not implement a solver, change any thresholds, alter existing artifacts, or launch another benchmark campaign.

## Why this follows from current evidence

The current GCL implementation already has a valuable oracle-only fitting lane, explicit source-alignment diagnostics and an identifiability verdict. The narrow current problem is that the solver's task definition discloses much of the structure it is supposed to reconstruct:

- `isolated_swiglu_cold_start.py` accepts `--atom-count` and derives the default active rank from it.
- The solver receives the known SwiGLU functional family and the active-rank/atom-count configuration.
- `swiglu_oracle_server.py` includes `unitCount` in its ready metadata.
- The strict court's training inputs are Gaussian samples; the target function is a real transformer unit sum, but the query distribution is not established as the model's empirical activation distribution.
- The selected 16-atom examples pass 2/2, but the randomized opaque-oracle court passes 1/6 and fails five cases under its frozen criteria.

These are not reasons to discard the implementation. They identify the next exact capability boundary: **does the system infer a useful decomposition from oracle behavior when the structure count is hidden and the queried inputs lie on the target model's actual activation distribution?**

This direction is more informative than increasing component counts, adding more generic activation patching, or relabeling the existing solver. It tests the missing bridge between selected function fitting and actual neural computation.

## Experimental contract to preregister before implementation

### Access separation

1. **Solver-visible:** input/output query pairs, declared input dimension and numerical precision needed to call the oracle. No source weights, hooks, neuron IDs, unit count, atom count, active rank, or source-alignment metrics.
2. **Teacher/witness-only:** the real model may generate empirical input vectors and the post-fit verifier may inspect source parameters. Neither source values nor count-derived hints enter solver search, selection, stopping, or training.
3. **Post-hoc verifier:** use an independently implemented function evaluator where practical. Keep fitted parameters frozen before revealing source truth. Record any reuse of solver code as a verifier dependency.
4. **No security overclaim:** process separation and file-access logging are engineering controls, not a hostile-code OS sandbox. Define the threat model accurately.

### Data and controls

- Sample target blocks/layers with a frozen, answer-independent selection rule across available LLaMA/Mistral SwiGLU families and early/middle/late layers. Publish every selected case, including failures.
- Build query sets from empirical residual/MLP activation inputs produced by a fixed prompt corpus. Keep model parameters hidden from the solver; the sampling harness alone uses the model to produce x-values.
- Keep independent Gaussian/OOD query sets as explicit secondary distributions, not as the sole train/evaluation domain.
- Separate solver fitting, model-selection, and held-out evaluation inputs. Freeze query budgets and numerical tolerances before results.
- Report each case's function agreement on empirical and OOD inputs, source-parameter alignment after freeze, inferred model order, uncertainty/ambiguity, query count, runtime, and peak VRAM.

### Outcomes that must remain distinct

1. **Function fidelity:** held-out oracle output agreement on each declared input distribution.
2. **Structure recovery:** the solver inferred the number and form of components without count/rank hints.
3. **Source correspondence:** post-hoc match of inferred components to source units, only if the decomposition is identifiable.
4. **Identifiability:** return an explicit ambiguous/non-identifiable outcome when multiple decompositions fit or weak/cancelled units cannot be separated.
5. **Causal relevance:** only a later intervention experiment can show the recovered components explain a real model prediction; function recovery alone is not a causal explanation.

Do not treat high R² as proof of atom identity, unique parameters, full-layer recovery, or semantic meaning. Where distinct parameterizations are functionally equivalent, report the equivalence class/function and avoid claiming unique source weights.

## Falsifiable acceptance criteria

Freeze the criteria and the sample list before opening source truth:

- Solver logs demonstrate that count, rank, unit IDs and teacher weights were not available to the solver process.
- The output includes a declared inferred order or a candid non-identifiable verdict; it may not silently take the source count from metadata.
- A negative/ambiguous control produces an ambiguous or failed result rather than fabricated atom identities.
- Held-out evaluation includes both real activation-distribution and separately labeled OOD points.
- Results are per-case; every failed case remains in the denominator.
- An independent verifier recomputes held-out function metrics and, post-freeze, source alignment/identifiability diagnostics from raw query-response data and model metadata.
- Existing thresholds and historical artifacts remain immutable; any new threshold is justified before outcomes are observed.
- Any causal claim about the recovered atoms is separately tested by interventions on actual model contexts and is not inferred from correlation/source alignment.

No numeric threshold is proposed here because choosing one without a validated distinction between clean and broken controls would be arbitrary. The existing strict report thresholds should not be silently loosened.

## Why not the apparent alternatives?

- **More GPT-2 patching:** transfer methodology is already established, and the 744-pair evaluation has been completed. The current weakness is specificity validity, not a shortage of transfer observations.
- **More path search:** the ordered path is already frozen; broadening its candidate sweep does not resolve whether a recovered function's structure is identifiable or whether the oracle solver learned the target function under real inputs.
- **More selected SwiGLU successes:** curated passes do not resolve the count/family disclosure or the 1/6 randomized strict-court result.
- **General natural-language circuit explanations:** HyVE, OpenAI neuron explanations and Anthropic attribution graphs already occupy this space; explanation text without stronger causal/identifiability evidence is not the limiting capability.

## Prior-art boundary and risk

Restricted neural-network extraction and identifiability are established topics [R1–R5]; therefore, **unknown-order real-SwiGLU recovery must not be called novel in advance**. The reviewed papers do not demonstrate this exact combination of target function, hidden order, and empirical transformer activation distribution, but a broader literature search and exact baseline adaptation are still required before any priority claim.

The main technical risk is non-identifiability: cancellation, duplicate/collinear directions, weak channels, and functional symmetries may make source-atom identity unrecoverable even when function fidelity is high. A correct outcome may be an equivalence class or “non-identifiable,” not a recovered neuron list. A second risk is that empirical activation inputs occupy a narrow manifold, making off-manifold function reconstruction impossible or unnecessary for the intended use; report in-distribution and OOD results separately.

## Stop condition

If the hidden-order solver cannot beat a preregistered null/negative-control protocol on held-out empirical inputs, or if multiple decompositions remain indistinguishable, stop the structure claim. Preserve a function-only result where fidelity warrants it, report non-identifiability explicitly, and do not market the system as a neural decompiler.

## Evidence links

- Solver and oracle boundary: [`isolated_swiglu_cold_start.py`](../../tools/llm_reader/isolated_swiglu_cold_start.py), [`swiglu_oracle_server.py`](../../tools/llm_reader/swiglu_oracle_server.py), [`blind_swiglu_oracle_decompiler.py`](../../tools/llm_reader/blind_swiglu_oracle_decompiler.py).
- Current strict randomized court: [machine-readable report](../../benchmarks/release/nn-live-blind-swiglu-randomized-cold-start-court/mistral4096d-llama3072d-16unit-q000001-randomized-stratified-opaque-oracle/report.json).
- Identifiability verdict and tests: [`jacobian_boundary_recovery.py`](../../tools/llm_reader/jacobian_boundary_recovery.py), [`test_identifiability_verdict.py`](../../tests/test_identifiability_verdict.py).
- Prior-art bibliography: [GCL_PRIOR_ART_AUDIT.md](GCL_PRIOR_ART_AUDIT.md#bibliography-and-primary-sources).
