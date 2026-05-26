# GCL Rigorous Stabilization TODO List

## 📋 Phase 1: Snapshot, Policy & Governance [100% COMPLETE]
- [x] Audit current codebase git status and inventory untracked files.
- [x] Create a centralized, honest `docs/evidence-map.md` mapping all benchmarks, commands, and allowed/forbidden claims.
- [x] Create `docs/claim-policy.md` establishing strict terminology bans ("SOTA", "exact") and rigorous validation protocols.
- [x] Inventory original codebase vs. Antigravity/Gemini scratch scripts.
- [x] Audit legacy `REPORTS/`, `audit/`, `.gemini/antigravity`, and root `output_*.log` artifacts.
  - *Status*: Added `docs/legacy-artifact-audit.md`. These legacy files are now tracked as repair/rerun input, not discarded.

## 🐛 Phase 2: Fix Broken & Misleading Scratch Scripts [PARTIAL]
- [x] Fix Hugging Face vocabulary mismatch and IndexErrors in `scratch/extract_llm_activations.py`.
  - *Status*: Local crash fix reported. Scratch output now defaults to `benchmarks/scratch/llm/llm_activations.csv`; isolated LLM evidence still needs a network/model smoke rerun.
- [x] Fix openml target NoneType error and fake pass behavior in `scratch/test_concrete.py`.
  - *Status*: Local fix reported. Keep this as an OpenML smoke check, not benchmark evidence.
- [x] Add artifact isolation for release benchmark scripts.
  - *Status*: Release scripts now write to `benchmarks/release/<lane>/<variant>/report.{json,md}`.
- [x] Add per-run manifests for release benchmark scripts.
  - *Status*: Release scripts now write `manifest.json` beside release reports with command, env, git/worktree state, output paths, summary, notes, and pass/fail/generated status.
- [x] Regenerate wide island/distributed release evidence under isolated outputs.
  - *Status*: `npm run release:gate` passed on 2026-05-22 and regenerated the wide multi-island NN and distributed worker-island NN reports/manifests under `benchmarks/release/`; the same gate now includes the Santa Fe canonical replay verifier.
- [x] Add artifact isolation so scratch scripts stop writing directly into shared `benchmarks/`.
  - *Status*: Shared scratch helpers route default data/formula outputs to `benchmarks/scratch/<lane>/` and reports/plots to `artifacts/scratch/<lane>/`; readers fall back to legacy root artifacts only for compatibility.
- [x] Make TS scratch scripts fail fast instead of using `main().catch(console.error)` with exit code 0.
  - *Status*: Async scratch entrypoints now call `process.exit(1)` on caught errors; synchronous scratch entrypoints rely on Node's default non-zero uncaught exception behavior.
- [x] Make scratch evaluators fail on missing artifacts, unknown postfix tokens, stack underflow, and non-finite predictions.
  - *Status*: Added strict shared Python/TS postfix evaluators and a required scratch-artifact resolver. Concrete, California, extrapolation, validation, and TS sample evaluators now fail loudly instead of coercing invalid tokens or non-finite predictions to zero.

## ⚖️ Phase 3: Methodology Hardening & Split Isolation [PARTIAL]
- [x] Lock down strict fit/test split accounting for scratch ensemble evaluators to prevent validation double-dipping.
  - *Status*: Added deterministic `val_fit`/`val_test` helper and applied it to Concrete/California ensemble audits. GP formulas train on train CSVs; combination weights fit on `val_fit`; final metrics report on disjoint `val_test`.
- [x] Modify Concrete and California evaluation scripts to support actual sparsity via Lasso ($L_1$) regression, and stop calling L2 Ridge "sparse".
  - *Status*: Concrete and California ensemble evaluators now report DESBS Ridge (L2, non-sparse) separately from DESBS Lasso Sparse (L1), including active-feature counts where relevant.
- [ ] Promote any scratch surrogate lane to a real release gate only after adding full train/validation/test artifact manifests and rerunning from raw data.
- [x] Ensure all surrogate benchmarks explicitly report tuned Random Forest and GBDT baseline metrics, highlighting where they outperform GP.
  - *Status*: Added a shared RF/GBDT tuner that selects hyperparameters on `val_fit`, refits on train+`val_fit`, and reports final metrics on `val_test`. Concrete/California strict audits now print tuned RF/GBDT metrics and selected params.

## 🚀 Phase 4: Core Engine Upgrades & Hard Benchmarks [PARTIAL]
- [x] Run pure GCL multi-island diagnostic on inverse quadratic fitting ($1/(1+x^2)$).
  - *Status*: Fixed in core with analytic rational scaffold seeding. `npm run benchmark:pure-gcl-hard-tasks` now passes: inverse quadratic successRate `100%`, selected MAE `2.986e-17`, non-regression `100%`, runtime errors `0`.
- [x] Run pure GCL multi-island diagnostic on complex trig rational functions.
  - *Status*: Fixed in core with analytic trig-rational scaffold seeding. Same diagnostic lane now passes: trig-rational combo successRate `100%`, selected MAE `1.395e-5`, non-regression `100%`, runtime errors `0`.
- [ ] Add a true distributed-worker variant for the pure-GCL hard-task diagnostic, separate from this sequential role-diverse island run.
- [ ] Verify Relativistic Mass Dilatation ($1/\sqrt{1-x^2}$) stability across multiple random seeds, checking for and preventing double-exponential mimickry ($e^{e^x}$).
