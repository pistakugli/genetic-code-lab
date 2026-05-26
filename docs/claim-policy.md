# GCL Claim Policy & Reporting Standards

This document establishes the official terminology, validation rules, and reporting policies for all Genetic Code Lab (GCL) development, testing, and academic publications. The goal is to enforce strict scientific integrity and prevent hyperbole or selective bias.

## 1. Classification of Project Stages

Every feature, benchmark, or model in GCL must be explicitly categorized into one of these four categories in all reports and documentation:

1.  **PROVEN (Stable Core)**:
    *   *Definition*: Systematically passes non-regression gates, has deterministic or highly stable statistical convergence, and outperforms basic baseline models across multiple independent seeds.
    *   *Standard*: Must have a unit test or release gate script that runs automatically on CI/CD and exits non-zero on any deviation.
2.  **PROMISING**:
    *   *Definition*: Shows clear, verified performance benefits under specific, controlled scenarios, but has known limitations, sensitivity to hyper-parameters, or convergence stability issues across seeds.
    *   *Standard*: Must document exactly which configurations or seeds fail, and the exact delta in comparison to standard baselines.
3.  **EXPERIMENTAL (Sandbox)**:
    *   *Definition*: Active research script or hypothesis testing. May crash, fail, or be heavily outperformed by basic models.
    *   *Standard*: Experimental code must remain strictly inside `scratch/` or `trails/` directories. Experimental artifacts must not be treated as release evidence unless promoted by a release gate with a manifest.
4.  **NEEDS REPAIR / BLOCKED**:
    *   *Definition*: Empirical evaluations show the approach is systematically outperformed by standard baselines (e.g. Random Forest), suffers from numerical collapse, or has extreme overfitting.
    *   *Standard*: Must be openly documented in the `evidence-map.md` with clear explanation of the blocking mechanism.

---

## 2. Terminology Restrictions (The Banned List)

To maintain a sober, engineering-focused tone, the following words and phrases are strictly banned or restricted:

| Banned Phrase | Permitted Alternative | Reason |
| :--- | :--- | :--- |
| **"SOTA" / "State of the Art"** | "High-fidelity local approximation" / "Competitive baseline" | High-dimensional GP is rarely SOTA compared to deep ensembles or large GBDTs. |
| **"Exact reverse-engineering"** | "Approximate parameter recovery" / "Structural alignment" | Stochastic optimization results in visible numerical errors ($\pm 0.01 - 0.05$). |
| **"Zero double-dipping"** | "Strict out-of-sample partition" | Double-dipping can only be claimed if full train/validation/test isolation is programmatically locked. |
| **"Ultimate proof" / "Perfect fit"** | "Verified empirical validation" | Extravagant claims harm academic and industrial credibility. |
| **"Universal solver"** | "Targeted Symbolic Regression" | GP scales poorly to hundreds of dimensions without massive resources. |

---

## 3. Mandatory Benchmarking Rules

Every benchmark run and report must adhere to the following rules:

1.  **Strict Split Protocol**: You must partition validation/test sets. Never fit hyper-parameters (such as Ridge/Lasso combination weights) on the same dataset used for reporting final generalization error.
2.  **Baselines Must Be Strong**: When comparing GCL to standard machine learning models, always include a tuned **Random Forest Regressor** or a **Gradient Boosting (XGBoost/LightGBM)** baseline. Never downplay or hide their scores when they beat GCL.
3.  **Physical Feasibility Checks**: Every evolved formula must be audited for "numerical mimickry" (such as high-order nested exponential terms $e^{e^x}$ that fit training data but explode OOD).
4.  **Artifact Hygiene**: Release benchmark scripts write to `benchmarks/release/<lane>/<variant>/report.{json,md}`. A file under `benchmarks/` stays outside the current release lane until its generating command, run date, pass/fail status, and claim boundary are listed in `docs/evidence-map.md`.
5.  **Scratch Isolation**: Scratch scripts may exist as research tools. Default scratch data/formula outputs belong under `benchmarks/scratch/<lane>/`, reports and plots belong under `artifacts/scratch/<lane>/`, and any scratch output left in the legacy `benchmarks/` root is legacy repair/rerun input until regenerated under the isolated layout with a manifest.
