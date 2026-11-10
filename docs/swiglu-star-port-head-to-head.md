# Star-port head-to-head bench: local mechanism validation

Status note (2026-11-10). This documents `tools/llm_reader/swiglu_head_to_head.py`
- the local, pre-Kaggle bench for the recorded failure point (lane F v2:
48 atoms, joint fit fails at R2 0.0104). Nothing here is recorded evidence;
it decides which mechanism earns a T4 kernel slot.

## The three arms (one oracle, one call accountant, one gate set)

- **baseline**: multi-start joint nonlinear least squares over all atoms -
  the recorded solver's method class (the thing that failed at 48).
- **curvature**: Shi&Fu-class port - finite-difference curvature gate
  candidates + antipodal (+/-x) separation, sequential peel.
- **starport**: GCL star-port - structural extraction, see below.

## Verified mathematical identities (bias-free SiLU, f(0) = 0)

1. **Odd part is exactly linear**: f(x) - f(-x) = 2 L x with
   L = 1/2 sum_i u_i g_i^T. Local check: odd-map regression error 1.8e-15
   (relative Frobenius). SVD of L gives the active spans U, V - but its
   singular pairs are *mixtures* (best joint cosine 0.17-0.38 against true
   atoms): second-order structure is rotation-degenerate, which is exactly
   why loss-driven joint fits mix atoms.
2. **Three-scale quartic form**: with kappa_s(d) = (r(sd) + r(-sd))/s^2,
   Delta3(d) = kappa_s(d) - 17 kappa_{s/2}(d) + 16 kappa_{s/4}(d)
   = -(3 s^2/32) sum_i u_i (g_i . d)^4 + O(s^8): kills the degenerate
   quadratic term AND the sixth-order term exactly. Gradient ascent on
   |Delta3| in the dominant channel isolates the strongest remaining atom
   (isolated extraction test: cosine 0.97-0.98 with the true atom,
   next-best 0.21-0.27).
3. **Up given a gate is exact linear regression** (closed form).

## Bench results (small preset: 16 atoms, dim 32, noiseless, seed 260621)

| arm | R2 | subspace | paired >= 0.99 | oracle calls |
| --- | --- | --- | --- | --- |
| baseline (joint, 4 restarts) | 0.9738 | 0.9742 | 8/16 | 6,256 |
| curvature (Shi&Fu class) | 0.24-0.90 (unstable) | 0.10-0.9997 | 0/16 | ~52k-133k |
| starport | 0.39 | 0.9900 | 3/16 (0.992/0.991/0.990) | ~198k |

Starport accepted 4 atoms; the top 3 are true atoms at joint cosine
>= 0.99 - atom-level recovery by structural extraction, no joint fit.
Subspace recovered to 0.99.

## Bugs found and fixed during development (kept for the record)

- analytic gradient sign inversion (ratio -1 vs finite differences) -
  L-BFGS was ascending;
- atom model used psi (even part) instead of full SiLU - R2 stuck at 0;
- unconstrained gate norm explodes (up to ||g|| ~ 9.6e3): the atom
  degenerates into a saturated hinge that explains energy while matching
  no atom. Fix: gate reparameterized on the unit sphere (amplitude lives
  in the up vector - the canonical form);
- loss-driven single-atom refine *degrades* good quartic gates
  (0.98 -> 0.86): the optimal single-atom approximator of an N-atom
  residual is a mixture, not the strongest atom. Fix: structural
  extraction anchors the gate; only anchored polish (penalty
  lam|b - b_anchor|^2, drift caps) is allowed;
- duplicate re-extraction: an imperfectly peeled atom leaves a remnant
  that dominates the quartic form again. Fix: cross-round dedup
  (|cos| > 0.9 with an accepted gate = rejection).

## Open items (the frontier of this mechanism)

1. **Weak-atom extraction**: when amplitudes are close, the quartic ascent
   lands on mixtures (one round measured `explained 0.066`); the accounting
   catches it, but recovery of the weak tail needs a sharper extraction
   (multi-channel quartic, alternating least squares on Delta3, or
   larger-scale probes with the O(s^8) error controlled).
2. **Peel completeness**: subtraction with a 0.99-cosine gate leaves a
   remnant that can re-dominate; the completion polish helps but does not
   close it. Candidate: iterate extraction-subtraction to convergence per
   atom before moving on.
3. **Stopping rule**: "no candidates after 3 duplicate extractions" is too
   harsh; needs retries with fresh seeds before giving up.
4. The curvature arm (Shi&Fu class port) is unstable at this scale
   (R2 0.24-0.90 across iterations) - its eigvec candidates are mixtures
   and it has no structural anchor.

## Verdict for the T4 kernel slot

The star-port mechanism already recovers true atoms individually where the
joint-fit method class mixes them (8/16 with degenerate directions vs 3
clean atoms + correct subspace at 16 atoms, noiseless). It is NOT yet at
recorded-gate strength (16/16 with R2 >= 0.995), so the honest next step is
closing the open items on this bench BEFORE any Kaggle push - the failure
modes are measured and specific, and the bench runs in seconds, not hours.
