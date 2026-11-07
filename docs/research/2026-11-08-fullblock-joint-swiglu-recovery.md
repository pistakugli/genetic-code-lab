# Full-block joint SwiGLU recovery: investigation notes (2026-11-08)

Research note for the blind SwiGLU oracle-decompiler lane. Goal: recover an
entire FFN block (all intermediate units, thousands of atoms) from `f(x)`-only
oracle access, going beyond the selected-unit / small-block lanes. Status:
path identified, joint-solver barrier characterized, sub-block architecture
proposed; lane F (128-unit block, 8x the previous record) landed as the
incremental step. This note records the analysis so the next session does not
re-derive it.

## The exact quadratic identity

For a bias-free SwiGLU block

    f(x) = sum_j d_j * silu(g_j . x) * (u_j . x)

the SiLU function itself satisfies `silu(s) - silu(-s) = s` (exact algebra, no
approximation), so splitting f into even and odd parts around the origin gives

    E(x) = (f(x) + f(-x)) / 2 = (1/2) sum_j d_j (g_j . x)(u_j . x)
    O(x) = (f(x) - f(-x)) / 2 = sum_j d_j (u_j . x) h(g_j . x),  h(s) = silu(s) - s/2

with h even. Three consequences:

1. **E is exactly a vector-valued quadratic map** — for any weights, including
   4-bit-quantized providers, because the identity is a property of the
   activation function, not of the parameters. This is the basis of the
   oracle-only structural witness (`tools/llm_reader/swiglu_quadratic_witness.py`):
   homogeneity `E(2x) = 4E(x)`, three-point polarization, and the
   parallelogram identity certify "bias-free SiLU-gated block" from queries
   alone. A biased or non-SwiGLU-gated block fails at O(1) relative error.
2. **O breaks the per-unit gauge**: the even part alone cannot distinguish
   `(g, u)` from the one-parameter rotation family
   `(g cos t + u sin t, -g sin t + u cos t)` (same `sym(g u^T)`), nor the
   sign flip `(-g, -u)`, nor the gate/up swap. The odd part breaks all three.
3. Query lower bound: spurious rank-<=2m quadratic maps vanish on R generic
   points when R is small, so probe counts must scale like `R >= ~6m` fibers
   (m = atom count) before the fiber samples even pin the map; `R = 3-4m`
   demonstrably admits exact-fitting wrong factorizations.

## What was tried and failed (all on noiseless synthetic oracles)

- **CP-ALS on the even-map fibers** (D-update exact m x m normal equations;
  G/U-updates exact CG on normal equations; verified the true factors are a
  zero-loss point and each block update is a true minimizer): from random
  init the loss plateaus at 0.4-35% relative error with weight cosines
  ~0.25-0.35 at every regime m <= d, m = 2d, m = 4d, R = 4m..10m, up to 3000
  sweeps and 8 restarts. At R = 4m it finds *exact* even-fits (loss ~1e-15)
  with wrong factors — those are the rotation-gauge / spurious-manifold
  solutions, not convergence errors.
- **Full-f Adam (torch autograd, cosine LR, 3000-8000 steps)**: also plateaus
  (loss ~1e-9 relative ~1e-2) with wrong factors. The gauge is broken in the
  objective, but the landscape still has near-zero spurious basins; joint
  factor recovery from random init does not descend to the truth at any
  tested scale.
- **Matrix-pencil / Jennrich-style exact recovery**: the observable bilinear
  operators `B(z, .)` are sums over *both* families `{d_j g_j^T}` and
  `{d_j u_j^T}` with z-dependent weights; a pencil between two such sums has
  2m rank-1 components, and its null vectors are polluted by the other
  family, so eigenvectors are not atom directions. Single-family spans would
  require breaking the gate/up symmetry *before* decomposition, which the
  quadratic part cannot do (point 2 above).

## What this implies

- **m <= d (with margin m <= d/2) is the regime where the pencil literature
  and the existing per-case solver operate.** The existing tool's lanes
  (8-16 atoms, activeRank = 2 * units) sit deep inside it.
- Full-block coverage should be attacked by **recorded sub-block chunks**
  (m_c <= d/2 per chunk, LCG-drawn unit partition, no cherry-picking) with
  per-chunk recovery and a final assembled full-block validation against the
  `full-mlp-vector` oracle (fresh-sample R2 + paired-atom gates). The oracle
  server already supports arbitrary unit subsets and the whole-module vector
  mode, so no protocol changes are needed.
- The per-unit query cost of the existing adaptive solver (~10-15k
  calls/unit) is the scaling bottleneck for thousands of units; the
  published antipodal-separation work reports ~2.4-3.5k queries/unit at
  0.6B-1B scale with BF16 providers. Closing that gap is the open solver
  problem; the quadratic identity above is the shared mathematical
  foundation both sides build on.

## Artifacts from this session

- `tools/llm_reader/swiglu_quadratic_witness.py` — oracle-only structural
  witness (homogeneity / polarization / parallelogram / origin-norm gates).
- Lane F: LLaMA-3.2-3B layer-1, 128 LCG-drawn units (seed 260621), 128 atoms,
  activeRank 256 — 8x the previous largest replayed block (lane E, 16 units).
- Prototype scripts (session sandbox, not repo):
  `scripts/fullblock-prototype.py` (numpy ALS + identifiability matrix),
  `scripts/fullblock-torch.py` (full-f Adam).

## Claim boundary

Nothing here demonstrates full-block joint recovery or weight-exact
decompilation; lane F extends the recorded selected-unit-block evidence to
128 units under the existing gates, and the witness certifies target
structure only. The joint-solver barrier and the sub-block proposal are
recorded as research direction, not as results.
