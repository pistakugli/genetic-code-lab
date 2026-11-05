# Global Spectral-Moment Inversion (GSMI)

## Research question

Can the distributional Hessian of a piecewise-affine scalar function be recovered from global value queries, and can its low-order Gaussian moments identify the ReLU kink hyperplanes without probing local gradients?

## Function class and access boundary

The target class is a globally defined finite ReLU sum

\[
f(x)=c+b^\top x+\sum_{j=1}^{m} a_j(n_j^\top x+t_j)_+,\qquad
\|n_j\|_2=1.
\]

GSMI receives only function values from an oracle, the dimension, a query budget, and fixed sampling settings. Teacher normals, offsets, amplitudes, term count, and scenario labels are not solver inputs. Benchmark code uses those hidden values only after fitting to score the result. The implementation uses a CUDA worker to form moment tensors and solve their matrix pencil; oracle calls themselves are counted on the TypeScript side and run independently of GPU fitting.

## Distributional Hessian

For a one-dimensional hinge \(h(s)=(s)_+\), \(h''=\delta_0\) as a distribution. By the chain rule,

\[
\nabla^2 f(x)=\sum_j a_j n_jn_j^\top\delta(n_j^\top x+t_j).
\]

The affine term vanishes. Under the Fourier convention \(\widehat{T}(\omega)=\langle T,e^{-i\omega^\top x}\rangle\), write \(x=sn+y\), where \(y\perp n\). Then

\[
\widehat{\delta(n^\top x+t)}(\omega)
= e^{i\lambda t}(2\pi)^{d-1}\delta(P_{n^\perp}\omega),
\quad \omega=\lambda n,
\]

up to the normalization convention for the subspace delta. Thus a plane contributes a frequency-space ridge supported along its normal direction; its phase along that ridge encodes the offset. Distinct nonparallel normals have distinct ridge supports in the exact infinite-domain transform. A finite observation window convolves this transform with the window spectrum and broadens the ridges.

## Gaussian spectral moments

Let \(\rho_\sigma(x)\) be the density of \(X\sim\mathcal N(0,\sigma^2I)\). Derivatives at zero frequency of the windowed Fourier transform correspond to moments against \(x^\alpha\rho_\sigma\). Define

\[
M^{(0)}_{pq}=\langle \partial_{pq}f,\rho_\sigma\rangle,\qquad
M^{(1)}_{pqr}=\langle \partial_{pq}f,x_r\rho_\sigma\rangle.
\]

Two integrations by parts yield value-only estimators:

\[
M^{(0)}_{pq}=
\mathbb E\left[f(X)\left(\frac{X_pX_q}{\sigma^4}
-\frac{\delta_{pq}}{\sigma^2}\right)\right],
\]

\[
M^{(1)}_{pqr}=
\mathbb E\left[f(X)\left(
\frac{X_pX_qX_r}{\sigma^4}
-\frac{\delta_{pq}X_r+\delta_{pr}X_q+\delta_{qr}X_p}{\sigma^2}
\right)\right].
\]

For a unit normal \(n\), Gaussian conditioning gives

\[
\int\delta(n^\top x+t)\rho_\sigma(x)\,dx
=\phi_\sigma(t),\qquad
\int x\,\delta(n^\top x+t)\rho_\sigma(x)\,dx
=-t n\phi_\sigma(t),
\]

where \(\phi_\sigma(t)=(\sqrt{2\pi}\sigma)^{-1}e^{-t^2/(2\sigma^2)}\). Therefore, with \(N=[n_1,\ldots,n_m]\), \(w_j=a_j\phi_\sigma(t_j)\), and \(q_j=-t_jn_j\),

\[
M^{(0)}=N\operatorname{diag}(w)N^\top,\qquad
M^{(1)}[:,:,v]=N\operatorname{diag}(w_j(q_j^\top v))N^\top.
\]

On the signal subspace, a generic matrix pencil \((M^{(1)}[:,:,v],M^{(0)})\) has eigenvalues \(q_j^\top v\). Repeating contractions in coordinate directions recovers \(q_j\), after which \(n_j=q_j/\|q_j\|\), \(t_j=-\|q_j\|\), and \(a_j=w_j/\phi_\sigma(t_j)\). The sign convention is gauge-invariant: \((n,t)\) and \((-n,-t)\) describe the same plane and produce the same \(q\) and Hessian. GSMI then fits the affine output layer using the sampled values and recovered hinges.

## Identifiability conditions

Exact second and third moments identify a set of distinct planes and their Hessian amplitudes under these sufficient conditions:

1. The normals are linearly independent and the \(w_j\) are nonzero, so \(M^{(0)}\) has signal rank \(m\).
2. The rank-one matrices \(n_jn_j^\top\) are linearly independent, allowing the \(w_j\) to be recovered after the normals are known.
3. Every offset is nonzero, so \(q_j\ne0\) and determines both the normal direction and offset.
4. The generic pencil eigenvalues \(q_j^\top v\) are pairwise distinct for at least one contraction vector \(v\).
5. The Gaussian moment integrals exist and the sampled oracle corresponds to the stated global ReLU-sum class.

These are sufficient, not necessary, conditions. The Hessian does not identify the affine gauge of a ReLU representation. It also cannot split multiple terms on the same plane whose Hessian amplitudes aggregate. If normals are linearly dependent, signed \(M^{(0)}\) can be rank deficient or cancel. If \(t_j=0\), the third moment contributes no orientation signal for that component. Repeated pencil eigenvalues, nearly cancelling weights, and finite-sample error make the algebraic inversion ambiguous or ill-conditioned.

For nearly parallel planes, the matrix \(N\) becomes ill-conditioned: small moment perturbations can cause large changes in pencil eigenvectors. They remain theoretically distinct for nonzero angular separation and exact moments, but no finite budget can guarantee resolution as separation tends to zero. A smooth contaminant contributes a non-atomic Hessian distribution; a finite plane model is then misspecified and can produce false components.

## Experimental predictions fixed before the pilot

1. On a single nonzero-offset plane, the full second/third-moment pencil should recover the plane up to sampling error using only values.
2. Removing the third moment should lose offset information and make nonorthogonal component directions indistinguishable as individual planes.
3. Near-parallel planes should show larger angular error and/or moment-pencil residual than well-separated, equally budgeted cases.
4. Exact zero-offset terms should be unresolved by this third-moment method, even if the second moment detects their aggregate rank.
5. The orthogonal full-rank controls should not be mislabeled as low-rank; component false-acceptance and completeness are measured separately.
6. Smooth contamination should increase the plane-model residual or false-component rate.

These are falsifiable predictions, not claims that the pilot will meet them.

## Sampling and computation

The pilot samples \(X\sim\mathcal N(0,0.2^2I)\) and rejects any vector outside \([-1,1]^d\). This preserves the benchmark oracle domain and makes the estimator a truncated-Gaussian approximation to the displayed exact identities. At five standard deviations per coordinate the omitted tail is small but nonzero; the derivation is exact for an untruncated Gaussian, so this finite-domain implementation has a stated tail approximation.

The full method estimates \(M^{(0)}\) and \(M^{(1)}\); the `second-only` ablation reuses the same samples and GPU but omits the third moment and pencil inversion. Runtime reports separate end-to-end solver time from worker GPU moment/inversion time and peak allocated VRAM. Gaussian query generation, teacher-oracle calls, and benchmark scoring do not count as GPU work.

## Pilot outcome

See the paired ten-seed [GSMI pilot report](./2026-10-30-global-spectral-inversion-pilot.md) and its raw JSONL file linked there. The pilot report owns the measured result and conclusion. This document states the mathematical model and predictions without promoting the method beyond that evidence.
