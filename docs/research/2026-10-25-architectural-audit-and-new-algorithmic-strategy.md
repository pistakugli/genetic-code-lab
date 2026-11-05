# Architectural Audit & Algorithmic Strategy: Cross-System Neural Discovery in GCL

**Date:** 2026-10-25  
**Document Path:** `docs/research/2026-10-25-architectural-audit-and-new-algorithmic-strategy.md`  
**Status:** Research Specification & Experimental Design (Phase 1–4 Complete)  
**Constraint:** Zero production code changes; zero unsubstantiated claims; verified against live codebase.

---

## Executive Summary

This report establishes a forensic architectural baseline of the **Genetic Code Lab (GCL)** codebase across its 89 source modules in `src/`, 76 neural benchmarks in `scripts/`, and underlying PyTorch/Transformers adapters in `tools/`. 

Crucially, it diagnoses why previous isolated line-sweep experiments in `scratch/` reached a performance ceiling and reveals a fundamental, cross-subsystem discovery: **GCL's five major capabilities (Active Subspace, Region Adjacency, Smooth FFN Solvers, Evolution Engine, and Live Circuit Readers) currently operate as disconnected, one-way pipelines.** 

We formulate three fundamentally new algorithmic hypotheses that bridge these isolated modules. We select the strongest hypothesis—**Hypothesis 1: Residual Polyhedral Curvature Incompleteness (RPCI)**—and specify an exact, controlled falsification experiment.

---

## 1. Architectural Map of GCL (Codebase Reality)

An automated import-graph and symbol audit reveals five distinct architectural pillars within GCL:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             GCL ARCHITECTURAL STACK                              │
├───────────────────────────────────────┬──────────────────────────────────────────┤
│ 1. Active Subspace & Dimensionality   │ 2. Smooth FFN Decompilers                │
│    `src/solvers/active-subspace-*.ts` │    `src/solvers/gelu-ffn-*.ts`           │
│    (4096D -> kD via Grad Covariance)  │    `src/solvers/swiglu-ffn-*.ts`         │
├───────────────────────────────────────┼──────────────────────────────────────────┤
│ 3. Region Adjacency Geometry          │ 4. Live Causal Circuit Reader            │
│    `src/solvers/deep-blackbox-relu-*` │    `scripts/benchmark-nn-live-*.ts`      │
│    (Polyhedral Cell Contact Witnesses)│    `tools/llm-reader-gpt2-factual-*.py`  │
├───────────────────────────────────────┴──────────────────────────────────────────┤
│ 5. Core Evolutionary & Symbolic Optimization Engine                              │
│    `src/core/evolution.ts`, `super-engine.ts`, `alps.ts`, `lexicase.ts`          │
│    `src/core/semantic-archive.ts`, `src/solvers/symbolic-residual-ensemble.ts`   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Forensic Inventory: What Each Subsystem Exposes, Discards, and Ignores

| Subsystem | What It Discovers / Calculates | What It Discards (Information Loss) | What It Ignores from Other Subsystems |
|---|---|---|---|
| **Active Subspace** (`active-subspace-ffn-decompiler.ts`) | Leading $k$ directions $V_k$ of gradient variance via power iteration on $\frac{1}{N} \sum \nabla f \nabla f^\top$. | Discards low-energy subspace directions ($V_\perp$). | Ignores discrete boundary normals discovered by Region Adjacency. Discards rare neurons with small active volumes. |
| **Region Adjacency** (`deep-blackbox-relu-nd-adjacency-geometry.ts`) | Polyhedral cell contacts, transition agreement ($>0.55$), projection residuals, witness flags. | Discards regions with low support ($< \text{minSupportScore}$). | Does NOT use dimensionality reduction. Runs in full input dimension, leading to neighbor sparsity in $>6$D. |
| **Smooth FFN Solvers** (`gelu-ffn-*.ts`, `swiglu-ffn-*.ts`) | Asymptotic channel directions, shared-gate canonical signatures, atom strengths under gauge groups. | Discards non-smooth boundary artifacts; prunes atoms below adaptive thresholds. | Ignores symbolic evolutionary refinements from `EvolutionEngine`. Does not cross-validate against residual graph paths. |
| **Evolution Engine** (`EvolutionEngine`, `ALPS`, `Lexicase`, `SemanticArchive`) | Pareto-optimal symbolic trees, semantic age layers (ALPS), non-dominated sorting (SPEA2). | Discards structural gradients; treats the problem as blackbox fitness. | Only consumed by `symbolic-residual-ensemble.ts`. Completely disconnected from neural solvers and LLM circuit finders! |
| **Live Circuit Reader** (`llm-reader-gpt2-factual-sweep.py`) | Target-specific logit drops, paraphrase-robust causal paths, attention/MLP attribution. | Discards intermediate sub-threshold activations; treats individual MLPs as blackbox nodes. | Does not invoke GCL's FFN decompilers to explain *why* an identified MLP node fires on the fact token. |

---

## 2. The Core Research Question

> **Can the internal structure already discovered by one GCL subsystem predict exactly where its current model representation must be incomplete, even when standard goodness-of-fit metrics ($R^2 \approx 1.0$, Low Validation MAE) report success?**

### The Blind Spot of Classical Metrics
In all previous tests, a solver achieves $R^2 = 0.9999$ because linear compensation parameters absorb unmodeled neurons. Classical verification (random test-point sampling) evaluates average volume:
$$\mathbb{E}_{x \sim U}[|f(x) - \hat{f}(x)|] \le \epsilon$$
Because a missing neuron's activation support may occupy only a tiny fraction of the input space ($<1\%$), random test points almost never land inside it. 

**However, the geometric structure of the already-recovered neurons mathematically dictates the boundary boundaries of the missing neurons.**

---

## 3. Three Fundamentally New Algorithmic Hypotheses

### Hypothesis 1: Residual Polyhedral Curvature Incompleteness (RPCI)
* **New Principle:** **Polyhedral Boundary Incompleteness Propagation.**  
  In a piecewise linear/ReLU network, every true hyperplane $H_i: w_i^\top x + b_i = 0$ partitions the affine regions of other neurons. When a subset of $k$ neurons is recovered, their intersection lines form a 1-skeleton inside the bounding box. If an unrecovered neuron $H^*$ exists, it must cut through at least one edge of this 1-skeleton.
* **Mechanism:** Instead of uniform random probing or blind line-sweeps, the solver extracts the polyhedral 1-skeleton (the edges formed by the intersection of $(d-1)$ recovered hyperplanes) and shoots secant probes *exclusively along these 1D intersection lines*.
* **Mathematical Assumption:** If the current $k$-neuron model is complete, the directional derivative of the oracle along any 1D skeleton edge $e_{ij} = H_i \cap H_j \cap \dots$ must be piecewise constant with zero slope discontinuities outside the known vertex set.
* **Falsification Condition:** If secant probes along the 1-skeleton of an incomplete model encounter zero slope discontinuities, or if the 1-skeleton probes fail to detect an unrecovered neuron that uniform sampling detects, the hypothesis is FALSIFIED.
* **Existing Modules Involved:** `src/solvers/deep-blackbox-relu-nd-adjacency-geometry.ts` (skeleton & witness tracking) + `src/solvers/blackbox-relu-nd-line-probe.ts` (analytic secant solver).

---

### Hypothesis 2: Dual Active-Subspace Gradient Curvature Deficiency (AS-GCD)
* **New Principle:** **Subspace Orthogonality Defect.**  
  Active Subspace PCA on $\nabla f(x) \nabla f(x)^\top$ computes the subspace $V_k \in \mathbb{R}^{d \times k}$. The orthogonal complement $V_\perp \in \mathbb{R}^{d \times (d-k)}$ is assumed to have zero gradient energy. If the model is incomplete, directional derivatives along $V_\perp$ evaluated at random points are near zero, but evaluated *near the boundaries of recovered neurons* they exhibit sharp non-zero spikes.
* **Mechanism:** Evaluate the gradient of the residual error $r(x) = f(x) - \hat{f}_k(x)$ projected onto $V_\perp$, restricted to the contact centroids identified by Region Adjacency.
* **Mathematical Assumption:** For a complete model, $\sup_{x \in \mathcal{C}} \|\text{Proj}_{V_\perp} \nabla f(x)\|_2 = 0$. For an incomplete model, $\sup_{x \in \mathcal{C}} \|\text{Proj}_{V_\perp} \nabla f(x)\|_2 > \alpha > 0$.
* **Falsification Condition:** If the projected gradient on $V_\perp$ along region boundaries does not correlate with missing structural rank, the hypothesis is FALSIFIED.
* **Existing Modules Involved:** `src/solvers/active-subspace-ffn-decompiler.ts` + `src/solvers/deep-blackbox-relu-nd-region-sampler.ts`.

---

### Hypothesis 3: Semantic Symbolic Lexicase Residual Distillation (SS-LRD)
* **New Principle:** **Lexicase Residual Discontinuity Decomposition.**  
  Instead of numerical least squares absorbing residual errors into global linear bias $w_0^\top x + b_0$, the unmodeled residual $r(x) = f(x) - \hat{f}(x)$ is fed directly into GCL's `EvolutionEngine` with `Lexicase` selection. Lexicase selection does not average errors; it filters candidate formulas by individual test cases.
* **Mechanism:** The symbolic engine searches for small activation atoms (e.g., $\operatorname{ReLU}(w^\top x + b)$) that solve specific clusters of residual outliers, bypassing gradient descent.
* **Mathematical Assumption:** Lexicase selection isolates candidate genomes that solve isolated subsets of sample points without degrading the global fit, identifying localized neural atoms that continuous regression blurs.
* **Falsification Condition:** If Lexicase symbolic evolution generates expressions that do not align with the true missing hyperplane normals within tolerance, the hypothesis is FALSIFIED.
* **Existing Modules Involved:** `src/core/lexicase.ts` + `src/core/evolution.ts` + `src/solvers/symbolic-residual-ensemble.ts`.

---

## 4. Selected Hypothesis for Immediate Research: Hypothesis 1 (RPCI)

We select **Hypothesis 1 (Residual Polyhedral Curvature Incompleteness)** because:
1. It directly addresses the scaling failure observed in our 8D / 20-neuron benchmark (where 19 of 20 neurons were found, but the 20th remained missing under uniform query limits).
2. It has an unambiguous geometric falsification criterion (a slope discontinuity must occur on the 1-skeleton if and only if a neuron is missing).
3. It creates an explicit synergy between `deep-blackbox-relu-nd-adjacency-geometry.ts` and `blackbox-relu-nd-line-probe.ts`.

### Mathematical Formulation of Hypothesis 1
Let $f: \mathbb{R}^d \to \mathbb{R}$ be a single-layer ReLU network:
$$f(x) = b_0 + w_0^\top x + \sum_{i=1}^m a_i \operatorname{ReLU}(w_i^\top x + b_i)$$
Let $\mathcal{H}_k = \{H_1, \dots, H_k\}$ be the set of $k < m$ hyperplanes currently recovered.
The 1-skeleton $\mathcal{S}_1(\mathcal{H}_k)$ is the set of 1-dimensional affine lines formed by the intersection of $(d-1)$ hyperplanes from $\mathcal{H}_k$:
$$L_{I} = \bigcap_{i \in I} H_i, \quad |I| = d - 1, \quad \operatorname{rank}(\{w_i\}_{i \in I}) = d - 1$$
Let $e \subset L_I \cap [-1, 1]^d$ be an edge segment of the skeleton.
Parametrize $e$ as $x(t) = x_0 + t v$, where $v \in \ker(W_I)$.

**Theorem (Incompleteness Detection via 1-Skeleton):**
1. If the model is structurally complete ($k = m$), then for every skeleton edge $e$, the restriction $f(x(t))$ has constant derivative:
   $$\frac{d}{dt} f(x(t)) = \text{const}, \quad \forall t \text{ along interior of } e$$
2. If there exists an unrecovered neuron $H^*: w^{*\top} x + b^* = 0$ whose boundary intersects the interior of $[-1, 1]^d$ and is not parallel to all edges, then there exists at least one skeleton edge $e$ intersecting $H^*$ at $t^* \in (0, 1)$ such that:
   $$\lim_{t \to t^{*+}} \frac{d}{dt} f(x(t)) - \lim_{t \to t^{*-}} \frac{d}{dt} f(x(t)) = a^* (w^{*\top} v) \ne 0$$
3. By querying along the 1-skeleton $\mathcal{S}_1(\mathcal{H}_k)$, GCL detects missing neurons with $O(\binom{k}{d-1})$ targeted line probes, without any blind uniform sampling!

---

## 5. Controlled Falsification Experiment Design

### Benchmark Setup
- **Teacher Models:** 10 independent seeds from `scripts/benchmark-nn-nd-adjacency-geometry-6d-adaptive-probe.ts` configuration (6D, 8 units and 8D, 12 units).
- **Oracle Interface:** Strict `query(X)` counting oracle, tracking total queries.
- **Control Method:** Existing GCL Static Line Probing (`blackbox-relu-nd-line-probe.ts`) using equal query budget ($B = 10,000$ calls).
- **Experimental Method (RPCI):**
  1. Run Stage A to recover initial $k$ hyperplanes.
  2. Compute 1-skeleton edges between recovered candidates using vertex adjacency from `deep-blackbox-relu-nd-adjacency-geometry.ts`.
  3. Probe along the 1-skeleton edges. Detect any secant jumps.
  4. If a jump is found, compute normal jump $\Delta g = a^* w^*$ directly on the edge.
  5. Add verified normal to pool and refit.
- **Metrics Tracked:**
  - `missing_neuron_recall`: Does RPCI discover the missing neuron when $k = m - 1$?
  - `query_efficiency`: Number of oracle queries needed to isolate the missing neuron vs uniform baseline.
  - `false_positive_rate`: Does RPCI claim an incompleteness jump on a complete model ($k = m$)?
- **Falsification Thresholds:**
  - If RPCI reports a false jump on a complete model ($k = m$) in $> 5\%$ of cases $\to$ **FALSIFIED**.
  - If RPCI fails to detect a missing neuron that intersects the active region $\to$ **FALSIFIED**.
  - If query cost of skeleton probing exceeds uniform random probing $\to$ **REJECTED AS IMPRACTICAL**.

---

## 6. Implementation Plan & Affected Source Files

When proceeding to implementation, changes will be restricted to the experimental solver interface without touching production paths:

1. **New Solver Extension:**  
   `src/solvers/polyhedral-incompleteness-solver.ts`  
   - Implements `compute1SkeletonEdges(planes: Hyperplane[]): LineSegment[]`.
   - Implements `auditSkeletonCurvature(oracle: CountingOracle, skeleton: LineSegment[])`.
2. **Existing Interface Touched:**  
   `src/solvers/deep-blackbox-relu-nd-adjacency-geometry.ts`  
   - Exposes `extractContactSkeleton()` to export verified cell intersection lines.
3. **Benchmark Harness:**  
   `benchmarks/research/benchmark-polyhedral-incompleteness.ts`  
   - Implements the strict falsification protocol comparing RPCI vs uniform line probes under equal budget.

---

## 7. Risks and Alternative Explanations

1. **Combinatorial Explosion of 1-Skeleton in High Dimensions:**  
   $\binom{k}{d-1}$ grows rapidly if $k \gg d$. In 8D with 20 neurons, $\binom{20}{7} = 77,520$ lines, which exceeds reasonable query budgets!  
   *Mitigation:* Only test skeleton edges formed by *adjacent* cells already witnessed by `deep-blackbox-relu-nd-adjacency-geometry.ts` (the dual Delaunay/Voronoi graph), reducing candidate edges from $77,520$ to $O(\text{edges}) \approx 300 - 500$.
2. **Boundary Clipping Outside Domain:**  
   Intersection lines might exit the $[-1, 1]^d$ domain before hitting the missing neuron.  
   *Mitigation:* Convex polyhedral bounding box clipping (Cyrus-Beck / Cohen-Sutherland in $N$-D).
3. **Alternative Explanation for Non-Zero Jumps:**  
   Numerical noise in finite-difference gradient evaluation could be mistaken for a missing neuron.  
   *Mitigation:* Enforce the machine-precision finite difference parameters verified in our audit ($\delta = 0.015, \varepsilon = 0.0001$).
