# Global Spectral-Moment Inversion (GSMI) pilot

**Run:** 2026-10-30T06-05-17-466Z  
**Verdict:** not-supported  
**Seeds per cell:** 10  
**Dimensions:** 16, 32  
**Scenarios:** near-parallel-low-rank, multi-crossing-low-rank, small-amplitude-low-rank, contaminated-local-probes, orthogonal-full-rank  
**Methods:** asad, ransac, star-probe, adaptive, bcad, mbcd-full, gfd-profiled, gsmi-full, gsmi-second-only  
**Raw runs:** [gsmi-2026-10-30T06-05-17-466Z-raw.jsonl](../../scratch/results/gsmi/pilot/gsmi-2026-10-30T06-05-17-466Z-raw.jsonl)

## Mathematical derivation

Consider the globally defined scalar function

`f(x)=c+b^T x+Σ_j a_j (n_j^T x+t_j)_+`, with `||n_j||₂=1`.
In the distributional sense,
`∂²f/∂x∂x^T=Σ_j a_j n_j n_j^T δ(n_j^T x+t_j)`.
Use the Fourier convention `F[g](ω)=∫g(x)e^{-iω^Tx}dx`. A plane term has Fourier support on the line parallel to `n_j`; along `ω=λ n_j`, its phase is `e^{i λ t_j}`. Thus exact unwindowed distributional Fourier data separates nonparallel plane orientations by support and encodes offsets in phase. Finite windows broaden each line, so directions separated by less than their reciprocal spatial width are poorly conditioned.

GSMI uses Gaussian-windowed spatial moments, equivalent to derivatives at zero frequency. Let `ρσ(x)` be the `N(0,σ²I)` density and define `M⁽⁰⁾_{pq}=⟨∂pq f,ρσ⟩`, `M⁽¹⁾_{pqr}=⟨∂pq f,x_rρσ⟩`. Integration by parts gives estimators from values alone:

```
M⁽⁰⁾pq = E[f(X)(Xp Xq/σ⁴ − δpq/σ²)]
M⁽¹⁾pqr = E[f(X)(Xp Xq Xr/σ⁴ −
                   (δpq Xr + δpr Xq + δqr Xp)/σ²)]
```

For a plane, Gaussian conditioning gives `∫δ(nᵀx+t)ρσ(x)dx=φσ(t)` and `∫xδ(nᵀx+t)ρσ(x)dx=−t n φσ(t)`. Therefore

```
M⁽⁰⁾ = N diag(wj) Nᵀ,       wj = aj φσ(tj)
M⁽¹⁾[:,:,v] = N diag(wj (qjᵀv)) Nᵀ,   qj = −tj nj
```

On the signal subspace, a matrix pencil `(M⁽¹⁾[:,:,v], M⁽⁰⁾)` has eigenvalues `qjᵀv`; repeating contractions in coordinate directions yields `qj`. For `qj ≠ 0`, `nj=qj/||qj||` and the gauge-invariant offset is `tj=−||qj||`. The plane Hessian weight then determines `aj=wj/φσ(tj)`. A final linear least-squares readout fits the affine part and amplitudes from the same global observations; it does not choose geometry.

## Identifiability and falsifiable predictions

- **Mathematical claim:** with distinct components, nonzero `wj`, linearly independent normals, nonzero offsets, and a non-colliding generic pencil spectrum, exact `M⁽⁰⁾,M⁽¹⁾` identify the unoriented planes and Hessian amplitudes up to component permutation. The ReLU affine gauge remains unidentifiable, so the final affine readout is not uniquely attributable to individual units.
- **Non-identifiable cases:** `t=0` makes `q=0`; third moments contain no normal direction for that component. A rank-deficient or cancelling signed `M⁽⁰⁾`, repeated pencil eigenvalues, more components than identifiable signal rank, and exactly coincident planes likewise defeat this inversion. The second-moment-only ablation cannot separate non-orthogonal components from their aggregate matrix.
- **Near-parallel prediction:** normals with small separation make `N` ill-conditioned and pencil eigenvectors unstable; finite-sample error should increase before exact orientation information disappears. GSMI is not expected to resolve arbitrarily close planes at a fixed sample budget.
- **Contamination prediction:** a smooth local bump has a continuous Hessian spectrum, not a finite sum of plane atoms; a finite-rank pencil may create false components or leave a large residual.
- **Sampling approximation:** the worker samples a Gaussian with σ=0.2, rejecting points outside the cube [-1,1]^d. At the configured five-sigma bound, the omitted Gaussian tail is small but nonzero; the derivation is exact for an untruncated Gaussian, so this finite-domain implementation has a stated tail approximation.

## Paired pilot results

All methods use the same deterministic teacher, solver seed, oracle ceiling, and independent 192-point test set per seed/dimension/scenario. GSMI receives only oracle values, dimension, query budget, and fixed sampling policy; scenario construction and hidden terms are used only by the benchmark evaluator. GFD-VP remains an unchanged comparator. The pilot has 10 seeds per cell and is screening evidence, not 100-seed confirmation.

| D | Scenario | Best existing baseline | GSMI MAE | Baseline MAE | GSMI complete rate | Oracle calls | False-accepted run rate |
|---:|---|---|---:|---:|---:|---:|---:|
| 16 | near-parallel-low-rank | adaptive | 2.2109e-1 | 1.2652e-12 | 0.0% | 4096 | 70.0% |
| 16 | multi-crossing-low-rank | adaptive | 2.6313e-1 | 1.3391e-12 | 0.0% | 4096 | 100.0% |
| 16 | small-amplitude-low-rank | asad | 2.0512e-3 | 1.7708e-10 | 0.0% | 4096 | 100.0% |
| 32 | near-parallel-low-rank | adaptive | 1.9899e-1 | 1.0599e-12 | 0.0% | 4096 | 50.0% |
| 32 | multi-crossing-low-rank | adaptive | 2.8546e-1 | 1.7438e-12 | 0.0% | 4096 | 100.0% |
| 32 | small-amplitude-low-rank | asad | 2.9227e-3 | 5.2606e-10 | 0.0% | 4096 | 100.0% |

## All-seed method metrics

| D | Scenario | Method | Scored/runs | MAE | Complete | Oracle calls | Normal error ° | Offset error | False planes/run | Runtime ms | VRAM MiB | GPU moment ms | Estimated rank | Pencil residual | Threw |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | contaminated-local-probes | adaptive | 10/10 | 2.9497e-12 | 100.0% | 10204 | 0.00 | 2.476e-13 | 0.00 | 3.7 | n/a | n/a | n/a | n/a | 0 |
| 16 | contaminated-local-probes | asad | 10/10 | 6.5847e-5 | 100.0% | 891 | 0.00 | 7.607e-6 | 0.00 | 9.3 | n/a | n/a | n/a | n/a | 0 |
| 16 | contaminated-local-probes | bcad | 10/10 | 1.5863e-2 | 90.0% | 1188 | 9.01 | 1.000e-1 | 0.00 | 16.1 | n/a | n/a | n/a | n/a | 0 |
| 16 | contaminated-local-probes | gfd-profiled | 10/10 | 1.0248e-2 | 90.0% | 4096 | 9.00 | 1.000e-1 | 0.10 | 155.6 | 20.59 | 130.9 | n/a | n/a | 0 |
| 16 | contaminated-local-probes | gsmi-full | 10/10 | 6.5908e-2 | 0.0% | 4096 | 90.00 | 1.000e+0 | 1.00 | 54.8 | 35.19 | 24.8 | 1.00 | 8.788e-1 | 0 |
| 16 | contaminated-local-probes | gsmi-second-only | 10/10 | 4.3650e-2 | 0.0% | 4096 | 47.63 | 5.401e-1 | 0.50 | 49.8 | 35.08 | 16.7 | 1.00 | 0.000e+0 | 0 |
| 16 | contaminated-local-probes | mbcd-full | 10/10 | 6.9164e-2 | 0.0% | 2640 | 37.30 | 4.168e-1 | 0.40 | 21.6 | n/a | n/a | n/a | n/a | 0 |
| 16 | contaminated-local-probes | ransac | 0/10 | n/a | 0.0% | 8448 | n/a | n/a | 0.00 | 2.4 | n/a | n/a | n/a | n/a | 10 |
| 16 | contaminated-local-probes | star-probe | 10/10 | 2.9497e-12 | 100.0% | 9874 | 0.00 | 2.476e-13 | 0.00 | 3.6 | n/a | n/a | n/a | n/a | 0 |
| 16 | multi-crossing-low-rank | adaptive | 10/10 | 1.3391e-12 | 100.0% | 11321 | 0.00 | 8.556e-13 | 0.00 | 4.7 | n/a | n/a | n/a | n/a | 0 |
| 16 | multi-crossing-low-rank | asad | 10/10 | 1.0062e+0 | 0.0% | 3168 | 84.00 | 9.333e-1 | 0.00 | 63.4 | n/a | n/a | n/a | n/a | 0 |
| 16 | multi-crossing-low-rank | bcad | 10/10 | 2.8771e-1 | 0.0% | 3168 | 87.00 | 9.667e-1 | 0.20 | 62.6 | n/a | n/a | n/a | n/a | 0 |
| 16 | multi-crossing-low-rank | gfd-profiled | 10/10 | 5.7938e-4 | 0.0% | 4096 | 0.11 | 5.857e-4 | 0.00 | 160.7 | 20.84 | 136.1 | n/a | n/a | 0 |
| 16 | multi-crossing-low-rank | gsmi-full | 10/10 | 2.6313e-1 | 0.0% | 4096 | 90.00 | 1.000e+0 | 2.80 | 56.8 | 35.42 | 26.9 | 2.80 | 9.592e-1 | 0 |
| 16 | multi-crossing-low-rank | gsmi-second-only | 10/10 | 9.6316e-2 | 0.0% | 4096 | 42.77 | 4.607e-1 | 1.10 | 46.7 | 35.31 | 16.8 | 2.80 | 0.000e+0 | 0 |
| 16 | multi-crossing-low-rank | mbcd-full | 10/10 | 1.3850e-1 | 0.0% | 2640 | 36.00 | 4.000e-1 | 1.20 | 38.4 | n/a | n/a | n/a | n/a | 0 |
| 16 | multi-crossing-low-rank | ransac | 0/10 | n/a | 0.0% | 8448 | n/a | n/a | 0.00 | 3.8 | n/a | n/a | n/a | n/a | 10 |
| 16 | multi-crossing-low-rank | star-probe | 10/10 | 1.3391e-12 | 100.0% | 10338 | 0.00 | 8.556e-13 | 0.00 | 4.5 | n/a | n/a | n/a | n/a | 0 |
| 16 | near-parallel-low-rank | adaptive | 10/10 | 1.2652e-12 | 100.0% | 10726 | 0.00 | 5.681e-13 | 0.00 | 4.3 | n/a | n/a | n/a | n/a | 0 |
| 16 | near-parallel-low-rank | asad | 10/10 | 1.1747e+0 | 10.0% | 2950 | 63.05 | 7.060e-1 | 0.00 | 58.1 | n/a | n/a | n/a | n/a | 0 |
| 16 | near-parallel-low-rank | bcad | 10/10 | 2.7151e-2 | 50.0% | 2356 | 22.52 | 2.523e-1 | 0.00 | 41.0 | n/a | n/a | n/a | n/a | 0 |
| 16 | near-parallel-low-rank | gfd-profiled | 10/10 | 1.0310e-2 | 0.0% | 4096 | 32.26 | 3.710e-1 | 0.70 | 269.6 | 20.64 | 244.5 | n/a | n/a | 0 |
| 16 | near-parallel-low-rank | gsmi-full | 10/10 | 2.2109e-1 | 0.0% | 4096 | 90.00 | 1.000e+0 | 0.70 | 87.3 | 35.12 | 55.7 | 0.70 | 9.341e-1 | 0 |
| 16 | near-parallel-low-rank | gsmi-second-only | 10/10 | 8.3161e-2 | 0.0% | 4096 | 90.00 | 1.000e+0 | 0.70 | 46.7 | 35.02 | 16.6 | 0.70 | 0.000e+0 | 0 |
| 16 | near-parallel-low-rank | mbcd-full | 10/10 | 1.4779e-1 | 0.0% | 2640 | 43.04 | 4.703e-1 | 0.80 | 28.0 | n/a | n/a | n/a | n/a | 0 |
| 16 | near-parallel-low-rank | ransac | 0/10 | n/a | 0.0% | 8448 | n/a | n/a | 0.00 | 4.3 | n/a | n/a | n/a | n/a | 10 |
| 16 | near-parallel-low-rank | star-probe | 10/10 | 1.2652e-12 | 100.0% | 10066 | 0.00 | 5.681e-13 | 0.00 | 5.3 | n/a | n/a | n/a | n/a | 0 |
| 16 | orthogonal-full-rank | adaptive | 10/10 | 1.3198e+0 | 0.0% | 12848 | 35.44 | 3.938e-1 | 1.80 | 11.2 | n/a | n/a | n/a | n/a | 0 |
| 16 | orthogonal-full-rank | asad | 10/10 | 7.3810e-1 | 0.0% | 3168 | 90.00 | 1.000e+0 | 0.00 | 68.1 | n/a | n/a | n/a | n/a | 0 |
| 16 | orthogonal-full-rank | bcad | 10/10 | 6.6499e-1 | 0.0% | 3168 | 90.00 | 1.000e+0 | 0.10 | 111.8 | n/a | n/a | n/a | n/a | 0 |
| 16 | orthogonal-full-rank | gfd-profiled | 10/10 | 2.0615e-1 | 0.0% | 4096 | 78.68 | 8.659e-1 | 13.80 | 164.9 | 24.01 | 138.9 | n/a | n/a | 0 |
| 16 | orthogonal-full-rank | gsmi-full | 10/10 | 7.4428e-1 | 0.0% | 4096 | 90.00 | 1.000e+0 | 1.10 | 54.9 | 35.16 | 23.1 | 1.10 | 9.947e-1 | 0 |
| 16 | orthogonal-full-rank | gsmi-second-only | 10/10 | 7.2413e-1 | 0.0% | 4096 | 90.00 | 1.000e+0 | 1.10 | 48.3 | 35.07 | 16.1 | 1.10 | 0.000e+0 | 0 |
| 16 | orthogonal-full-rank | mbcd-full | 10/10 | 7.4060e-1 | 0.0% | 2640 | 90.00 | 1.000e+0 | 16.00 | 53.6 | n/a | n/a | n/a | n/a | 0 |
| 16 | orthogonal-full-rank | ransac | 0/10 | n/a | 0.0% | 8448 | n/a | n/a | 0.00 | 15.2 | n/a | n/a | n/a | n/a | 10 |
| 16 | orthogonal-full-rank | star-probe | 10/10 | 1.3198e+0 | 0.0% | 12848 | 5.63 | 6.250e-2 | 0.00 | 7.2 | n/a | n/a | n/a | n/a | 0 |
| 16 | small-amplitude-low-rank | adaptive | 0/10 | n/a | 0.0% | 9095 | n/a | n/a | 0.00 | 2.0 | n/a | n/a | n/a | n/a | 10 |
| 16 | small-amplitude-low-rank | asad | 10/10 | 1.7708e-10 | 100.0% | 891 | 0.00 | 4.098e-12 | 0.00 | 8.6 | n/a | n/a | n/a | n/a | 0 |
| 16 | small-amplitude-low-rank | bcad | 10/10 | 9.5641e-4 | 80.0% | 1749 | 18.00 | 2.000e-1 | 0.00 | 23.8 | n/a | n/a | n/a | n/a | 0 |
| 16 | small-amplitude-low-rank | gfd-profiled | 10/10 | 8.0073e-9 | 100.0% | 4096 | 0.00 | 3.015e-7 | 0.00 | 153.8 | 20.59 | 129.9 | n/a | n/a | 0 |
| 16 | small-amplitude-low-rank | gsmi-full | 10/10 | 2.0512e-3 | 0.0% | 4096 | 90.00 | 1.000e+0 | 1.00 | 54.2 | 35.19 | 24.8 | 1.00 | 8.652e-1 | 0 |
| 16 | small-amplitude-low-rank | gsmi-second-only | 10/10 | 1.5105e-3 | 0.0% | 4096 | 55.92 | 6.297e-1 | 0.60 | 45.9 | 35.08 | 16.5 | 1.00 | 0.000e+0 | 0 |
| 16 | small-amplitude-low-rank | mbcd-full | 10/10 | 1.6368e-3 | 0.0% | 2640 | 22.31 | 2.269e-1 | 0.20 | 20.1 | n/a | n/a | n/a | n/a | 0 |
| 16 | small-amplitude-low-rank | ransac | 0/10 | n/a | 0.0% | 8448 | n/a | n/a | 0.00 | 1.8 | n/a | n/a | n/a | n/a | 10 |
| 16 | small-amplitude-low-rank | star-probe | 0/10 | n/a | 0.0% | 9095 | n/a | n/a | 0.00 | 1.9 | n/a | n/a | n/a | n/a | 10 |
| 32 | contaminated-local-probes | adaptive | 10/10 | 3.4911e-13 | 100.0% | 18012 | 0.00 | 8.743e-13 | 0.00 | 12.6 | n/a | n/a | n/a | n/a | 0 |
| 32 | contaminated-local-probes | asad | 10/10 | 8.8310e-10 | 100.0% | 3315 | 0.00 | 2.665e-11 | 0.00 | 38.2 | n/a | n/a | n/a | n/a | 0 |
| 32 | contaminated-local-probes | bcad | 10/10 | 1.4927e-2 | 90.0% | 3692 | 9.00 | 1.000e-1 | 0.00 | 53.3 | n/a | n/a | n/a | n/a | 0 |
| 32 | contaminated-local-probes | gfd-profiled | 10/10 | 1.1062e-2 | 90.0% | 4096 | 9.00 | 1.000e-1 | 0.10 | 189.8 | 24.55 | 143.8 | n/a | n/a | 0 |
| 32 | contaminated-local-probes | gsmi-full | 10/10 | 9.0150e-2 | 0.0% | 4096 | 90.00 | 1.000e+0 | 1.00 | 121.3 | 85.89 | 63.8 | 1.00 | 9.813e-1 | 0 |
| 32 | contaminated-local-probes | gsmi-second-only | 10/10 | 3.5522e-2 | 0.0% | 4096 | 23.44 | 2.435e-1 | 0.20 | 100.4 | 85.10 | 42.7 | 1.00 | 0.000e+0 | 0 |
| 32 | contaminated-local-probes | mbcd-full | 10/10 | 7.4215e-2 | 0.0% | 5200 | 29.02 | 3.180e-1 | 0.30 | 33.4 | n/a | n/a | n/a | n/a | 0 |
| 32 | contaminated-local-probes | ransac | 0/10 | n/a | 0.0% | 14592 | n/a | n/a | 0.00 | 5.2 | n/a | n/a | n/a | n/a | 10 |
| 32 | contaminated-local-probes | star-probe | 10/10 | 3.4911e-13 | 100.0% | 17362 | 0.00 | 8.743e-13 | 0.00 | 12.4 | n/a | n/a | n/a | n/a | 0 |
| 32 | multi-crossing-low-rank | adaptive | 5/10 | 1.7438e-12 | 50.0% | 19294 | 0.00 | 1.284e-12 | 0.00 | 11.2 | n/a | n/a | n/a | n/a | 5 |
| 32 | multi-crossing-low-rank | asad | 10/10 | 1.9177e+0 | 0.0% | 6240 | 90.00 | 1.000e+0 | 0.00 | 114.1 | n/a | n/a | n/a | n/a | 0 |
| 32 | multi-crossing-low-rank | bcad | 10/10 | 3.0228e-1 | 0.0% | 6240 | 87.00 | 9.667e-1 | 0.00 | 103.2 | n/a | n/a | n/a | n/a | 0 |
| 32 | multi-crossing-low-rank | gfd-profiled | 10/10 | 3.6205e-4 | 0.0% | 4096 | 0.05 | 1.133e-4 | 0.00 | 189.9 | 24.76 | 143.5 | n/a | n/a | 0 |
| 32 | multi-crossing-low-rank | gsmi-full | 10/10 | 2.8546e-1 | 0.0% | 4096 | 90.00 | 1.000e+0 | 2.20 | 122.1 | 86.05 | 64.7 | 2.20 | 9.950e-1 | 0 |
| 32 | multi-crossing-low-rank | gsmi-second-only | 10/10 | 1.3827e-1 | 0.0% | 4096 | 57.45 | 6.180e-1 | 1.00 | 100.7 | 85.26 | 42.5 | 2.20 | 0.000e+0 | 0 |
| 32 | multi-crossing-low-rank | mbcd-full | 10/10 | 1.0662e-1 | 0.0% | 5200 | 27.00 | 3.000e-1 | 0.90 | 66.7 | n/a | n/a | n/a | n/a | 0 |
| 32 | multi-crossing-low-rank | ransac | 0/10 | n/a | 0.0% | 14592 | n/a | n/a | 0.00 | 7.1 | n/a | n/a | n/a | n/a | 10 |
| 32 | multi-crossing-low-rank | star-probe | 5/10 | 1.7438e-12 | 50.0% | 17864 | 0.00 | 1.284e-12 | 0.00 | 10.8 | n/a | n/a | n/a | n/a | 5 |
| 32 | near-parallel-low-rank | adaptive | 8/10 | 1.0599e-12 | 80.0% | 18627 | 0.00 | 7.608e-13 | 0.00 | 12.5 | n/a | n/a | n/a | n/a | 2 |
| 32 | near-parallel-low-rank | asad | 10/10 | 9.1873e-1 | 0.0% | 6006 | 67.54 | 7.544e-1 | 0.10 | 99.6 | n/a | n/a | n/a | n/a | 0 |
| 32 | near-parallel-low-rank | bcad | 10/10 | 3.4677e-2 | 50.0% | 5213 | 27.00 | 3.000e-1 | 0.00 | 70.5 | n/a | n/a | n/a | n/a | 0 |
| 32 | near-parallel-low-rank | gfd-profiled | 10/10 | 2.1048e-2 | 0.0% | 4096 | 50.78 | 5.768e-1 | 1.10 | 188.8 | 24.65 | 142.6 | n/a | n/a | 0 |
| 32 | near-parallel-low-rank | gsmi-full | 10/10 | 1.9899e-1 | 0.0% | 4096 | 90.00 | 1.000e+0 | 0.50 | 122.3 | 85.78 | 64.3 | 0.50 | 9.850e-1 | 0 |
| 32 | near-parallel-low-rank | gsmi-second-only | 10/10 | 1.0066e-1 | 0.0% | 4096 | 90.00 | 1.000e+0 | 0.50 | 102.7 | 85.14 | 44.7 | 0.50 | 0.000e+0 | 0 |
| 32 | near-parallel-low-rank | mbcd-full | 10/10 | 9.4271e-2 | 0.0% | 5200 | 50.82 | 5.661e-1 | 0.90 | 41.7 | n/a | n/a | n/a | n/a | 0 |
| 32 | near-parallel-low-rank | ransac | 0/10 | n/a | 0.0% | 14592 | n/a | n/a | 0.00 | 5.4 | n/a | n/a | n/a | n/a | 10 |
| 32 | near-parallel-low-rank | star-probe | 8/10 | 1.0599e-12 | 80.0% | 17457 | 0.00 | 7.608e-13 | 0.00 | 12.0 | n/a | n/a | n/a | n/a | 2 |
| 32 | orthogonal-full-rank | adaptive | 10/10 | 1.9960e+0 | 0.0% | 25168 | 63.28 | 7.031e-1 | 3.30 | 118.0 | n/a | n/a | n/a | n/a | 0 |
| 32 | orthogonal-full-rank | asad | 10/10 | 1.3069e+0 | 0.0% | 6240 | 90.00 | 1.000e+0 | 0.00 | 199.0 | n/a | n/a | n/a | n/a | 0 |
| 32 | orthogonal-full-rank | bcad | 10/10 | 1.1058e+0 | 0.0% | 6240 | 90.00 | 1.000e+0 | 0.10 | 232.8 | n/a | n/a | n/a | n/a | 0 |
| 32 | orthogonal-full-rank | gfd-profiled | 10/10 | 5.0887e-1 | 0.0% | 4096 | 90.00 | 1.000e+0 | 32.00 | 212.0 | 31.04 | 159.5 | n/a | n/a | 0 |
| 32 | orthogonal-full-rank | gsmi-full | 10/10 | 1.1438e+0 | 0.0% | 4096 | 90.00 | 1.000e+0 | 0.00 | 121.2 | 85.68 | 57.7 | 0.00 | 0.000e+0 | 0 |
| 32 | orthogonal-full-rank | gsmi-second-only | 10/10 | 1.1438e+0 | 0.0% | 4096 | 90.00 | 1.000e+0 | 0.00 | 106.3 | 85.18 | 42.1 | 0.00 | 0.000e+0 | 0 |
| 32 | orthogonal-full-rank | mbcd-full | 10/10 | 2.2333e+0 | 0.0% | 5200 | 90.00 | 1.000e+0 | 32.00 | 110.7 | n/a | n/a | n/a | n/a | 0 |
| 32 | orthogonal-full-rank | ransac | 0/10 | n/a | 0.0% | 14592 | n/a | n/a | 0.00 | 149.7 | n/a | n/a | n/a | n/a | 10 |
| 32 | orthogonal-full-rank | star-probe | 10/10 | 1.9960e+0 | 0.0% | 25168 | 51.19 | 5.688e-1 | 0.00 | 45.9 | n/a | n/a | n/a | n/a | 0 |
| 32 | small-amplitude-low-rank | adaptive | 0/10 | n/a | 0.0% | 15840 | n/a | n/a | 0.00 | 4.5 | n/a | n/a | n/a | n/a | 10 |
| 32 | small-amplitude-low-rank | asad | 10/10 | 5.2606e-10 | 100.0% | 3315 | 0.00 | 3.057e-12 | 0.00 | 37.5 | n/a | n/a | n/a | n/a | 0 |
| 32 | small-amplitude-low-rank | bcad | 10/10 | 5.4870e-4 | 90.0% | 3608 | 9.00 | 1.000e-1 | 0.00 | 51.7 | n/a | n/a | n/a | n/a | 0 |
| 32 | small-amplitude-low-rank | gfd-profiled | 10/10 | 9.2595e-9 | 100.0% | 4096 | 0.00 | 1.781e-7 | 0.00 | 188.9 | 24.55 | 142.6 | n/a | n/a | 0 |
| 32 | small-amplitude-low-rank | gsmi-full | 10/10 | 2.9227e-3 | 0.0% | 4096 | 90.00 | 1.000e+0 | 1.00 | 121.0 | 85.89 | 64.4 | 1.00 | 9.793e-1 | 0 |
| 32 | small-amplitude-low-rank | gsmi-second-only | 10/10 | 1.2076e-3 | 0.0% | 4096 | 23.32 | 2.445e-1 | 0.20 | 100.5 | 85.10 | 43.1 | 1.00 | 0.000e+0 | 0 |
| 32 | small-amplitude-low-rank | mbcd-full | 10/10 | 5.8979e-4 | 0.0% | 5200 | 9.96 | 1.114e-1 | 0.10 | 32.7 | n/a | n/a | n/a | n/a | 0 |
| 32 | small-amplitude-low-rank | ransac | 0/10 | n/a | 0.0% | 14592 | n/a | n/a | 0.00 | 4.5 | n/a | n/a | n/a | n/a | 10 |
| 32 | small-amplitude-low-rank | star-probe | 0/10 | n/a | 0.0% | 15840 | n/a | n/a | 0.00 | 4.6 | n/a | n/a | n/a | n/a | 10 |

**Run status totals:** 230 complete, 516 incomplete, 154 threw.  
**Budget overruns:** 0; **oracle diagnostic mismatches:** 0.

## Ablations

`gsmi-second-only` removes the third Gaussian-windowed moment and its matrix-pencil inversion. It uses only eigenvectors of the second moment, which are aggregate directions rather than guaranteed individual planes. Comparing it with `gsmi-full` tests whether offset-bearing third moments and pencil inversion add identifiable geometry. Both variants use the same queried samples and CUDA worker.

## Pilot failure analysis

The full GSMI method completed 0/100 reconstructions overall and 0/60 in the three primary low-rank scenarios. Across those primary cells, the mean relative moment-reconstruction residual ranged from 0.865–0.995. The full-rank controls reported 16D rank 1.10, 1.10 false planes/run, 0% complete; 32D rank 0.00, 0.00 false planes/run, 0% complete. The second-moment-only ablation completed 0/100; it had lower mean MAE than the full method in 9/10 cells, but did not complete the geometry in those cases.

These observations locate the empirical failure in turning estimated low-order moments into plane geometry: the full method's reconstructed moments disagree strongly with the sampled moments, and omitting the pencil improves some function-value errors without recovering complete geometry. This is consistent with finite-sample moment error amplified by the ill-conditioned inversion, but this pilot did not isolate sampling variance, bounded-domain truncation, and structural non-identifiability with separate ablations. The full-rank controls also do not show reliable rank recovery. Do not treat the low oracle count as evidence of successful geometric reconstruction.

## Conclusion and next experiment

The pilot does not pass the required pre-campaign screen: no qualifying hard-case geometry/control pair is demonstrated. Do not start a larger seed campaign unchanged. The next experiment should change the mathematical information source—for example, test finite-frequency Fourier-ridge support and phase estimates directly against the same global-query controls—rather than tune rank or acceptance thresholds. First determine whether that representation separates sampling error from the observed pencil-conditioning failure.

The pilot verdict is computed from all raw runs and the declared screen; tests establish implementation/accounting behavior, not reconstruction accuracy. No raw results from earlier campaigns are modified.
