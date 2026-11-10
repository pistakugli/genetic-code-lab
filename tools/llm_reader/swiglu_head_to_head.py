"""Head-to-head solver comparison on a synthetic bias-free SiLU block.

This is the local, pre-Kaggle mechanism bench for the recorded failure point
(lane F v2: 48 atoms, joint fit fails). Three arms share one oracle, one
query accountant, and one gate set:

  arm baseline - multi-start joint nonlinear least squares over all atoms
    (the recorded solver's method class: structured + random starts, all
    atoms fitted jointly - the thing that failed at 48 atoms on the T4).
  arm curvature - Shi&Fu-class port: finite-difference curvature supplies
    gate-direction candidates, antipodal (+/-x) observations separate
    magnitude and orientation. Sequential peel with per-atom verification.
  arm starport - GCL star-port: the residual odd map (exactly linear for
    bias-free SiLU) supplies the active spans and the up estimate; the
    quartic scale-difference form
      kappa_s(d) - kappa_{s/2}(d) = -(s^2/32) sum_i u_i (g_i . d)^4 + O(s^6)
    kills the rotation-degenerate quadratic term and isolates ONE gate via
    gradient ascent; per-atom verification then peels it. Rejection
    accounting as in the LSPNR star-probe.

The synthetic teacher replicates the lane structure: bias-free SiLU atoms
u_i SiLU(g_i^T x), gate span = active rank = 2 * atom count, lognormal
amplitudes (weak atoms exist), optional output-grid noise as a quantization
proxy. Nothing here is recorded evidence - it decides whether the peel
mechanism earns a T4 kernel slot.

Mathematical identities used (both exact for bias-free SiLU):
  SiLU(z) - SiLU(-z) = z            -> odd part of f is exactly L x,
                                       L = 1/2 sum_i u_i g_i^T
  SiLU(z) + SiLU(-z) = z tanh(z/2)  -> even part is sum_i u_i psi(g_i^T x)
                                       with psi(z) = z tanh(z/2)/2
                                       (psi even, psi(z) ~ z^2/4 near 0)
The even part's quadratic regime is rotation-degenerate (why joint fits
mix atoms); the peel attacks one atom at a time on the residual and
verifies each before accepting.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


# ---------------------------------------------------------------------------
# teacher and oracle
# ---------------------------------------------------------------------------


def make_teacher(atom_count: int, seed: int, amplitude_sigma: float = 0.7):
    """D = 2 * atom_count mirrors the recorded activeRank = 2 * units."""
    dim = 2 * atom_count
    rng = np.random.default_rng(seed)
    gates = rng.normal(size=(atom_count, dim))
    gates /= np.linalg.norm(gates, axis=1, keepdims=True)
    ups = rng.normal(size=(atom_count, dim))
    ups /= np.linalg.norm(ups, axis=1, keepdims=True)
    amplitudes = rng.lognormal(mean=0.0, sigma=amplitude_sigma, size=atom_count)
    ups = ups * amplitudes[:, None]
    return dim, gates, ups


def silu(z: np.ndarray) -> np.ndarray:
    return z * expit(z)


def silu_prime(z: np.ndarray) -> np.ndarray:
    s = expit(z)
    return s * (1.0 - z * (1.0 - s))


def psi(z: np.ndarray) -> np.ndarray:
    """Even part of SiLU: (SiLU(z)+SiLU(-z))/2 = z tanh(z/2)/2."""
    return 0.5 * z * np.tanh(0.5 * z)


class Oracle:
    """Vector-valued block oracle with call accounting and optional grid noise."""

    def __init__(self, gates: np.ndarray, ups: np.ndarray, noise_step: float = 0.0):
        self.gates = gates
        self.ups = ups
        self.noise_step = noise_step
        self.calls = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.calls += x.shape[0]
        z = x @ self.gates.T
        y = silu(z) @ self.ups
        if self.noise_step > 0.0:
            y = np.round(y / self.noise_step) * self.noise_step
        return y


def sample_inputs(rng: np.random.Generator, count: int, dim: int) -> np.ndarray:
    return rng.normal(size=(count, dim))


# ---------------------------------------------------------------------------
# shared blind machinery
# ---------------------------------------------------------------------------


def estimate_odd_map(oracle: Oracle, x: np.ndarray) -> tuple[np.ndarray, int]:
    """Least-squares fit of L in  y_odd(x) = (f(x)-f(-x))/2 = L x  (exact identity)."""
    y_plus = oracle(x)
    y_minus = oracle(-x)
    y_odd = 0.5 * (y_plus - y_minus)
    coeff, *_ = np.linalg.lstsq(x, y_odd, rcond=None)
    return coeff.T, 0


def active_spans(odd_map: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """U (output span), V (input span) from the odd map's top singular block."""
    left, singular, right_t = np.linalg.svd(odd_map, full_matrices=False)
    keep = min(rank, left.shape[1], right_t.shape[0])
    return left[:, :keep], right_t[:keep, :].T


def refine_single_atom(
    x: np.ndarray,
    residual: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    a_init: np.ndarray,
    b_init: np.ndarray,
    maxiter: int = 200,
):
    """L-BFGS fit of one atom u=Ua, g=V*(c/|c|) against the residual.

    The gate is reparameterized onto the unit sphere (the amplitude lives in
    the up vector - the canonical form). Without this constraint the gate
    norm blows up and the atom degenerates into a saturated hinge that fits
    energy while matching no true atom.
    """
    r = U.shape[1]

    def unpack(theta):
        return theta[:r], theta[r:]

    def objective(theta):
        a, c = unpack(theta)
        cn = float(np.linalg.norm(c))
        if cn < 1e-12:
            cn = 1e-12
        b = c / cn
        z = x @ (V @ b)
        phi = silu(z)
        up = U @ a
        pred = phi[:, None] * up[None, :]
        err = pred - residual
        loss = float(np.mean(err * err))
        d_up = (2.0 / err.size) * (phi @ err)
        grad_a = U.T @ d_up
        dphi = silu_prime(z)
        coeff = (2.0 / err.size) * (err @ up)
        grad_b = V.T @ (x.T @ (dphi * coeff))
        # chain rule through c -> b = c/|c|:  db/dc = (I - b b^T) / |c|
        grad_c = (grad_b - b * float(b @ grad_b)) / cn
        return loss, np.concatenate([grad_a, grad_c])

    b_unit = b_init / max(float(np.linalg.norm(b_init)), 1e-12)
    theta0 = np.concatenate([a_init, b_unit])
    result = minimize(
        objective,
        theta0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-10},
    )
    a, c = unpack(result.x)
    b = c / max(float(np.linalg.norm(c)), 1e-12)
    return a, b, float(result.fun)


def polish_anchored_atom(
    x: np.ndarray,
    residual: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    a_init: np.ndarray,
    b_anchor: np.ndarray,
    lam: float = 100.0,
    maxiter: int = 120,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Anchored polish of one atom: L-BFGS on (a, c) with a penalty
    lam * |b - b_anchor|^2 keeping the gate near the structural extraction.

    The anchor lets the gate correct the last percent of direction error
    without drifting into the mixture basin that an unanchored loss-driven
    fit prefers. Returns (up, gate, drift) with drift = |b - b_anchor|.
    """
    r = U.shape[1]

    def unpack(theta):
        return theta[:r], theta[r:]

    def objective(theta):
        a, c = unpack(theta)
        cn = float(np.linalg.norm(c))
        if cn < 1e-12:
            cn = 1e-12
        b = c / cn
        z = x @ (V @ b)
        phi = silu(z)
        up = U @ a
        err = phi[:, None] * up[None, :] - residual
        penalty = lam * float(np.dot(b - b_anchor, b - b_anchor))
        loss = float(np.mean(err * err)) + penalty
        d_up = (2.0 / err.size) * (phi @ err)
        grad_a = U.T @ d_up
        dphi = silu_prime(z)
        coeff = (2.0 / err.size) * (err @ up)
        grad_b = V.T @ (x.T @ (dphi * coeff)) + 2.0 * lam * (b - b_anchor)
        grad_c = (grad_b - b * float(b @ grad_b)) / cn
        return loss, np.concatenate([grad_a, grad_c])

    theta0 = np.concatenate([a_init, b_anchor])
    result = minimize(
        objective,
        theta0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-10},
    )
    a, c = unpack(result.x)
    b = c / max(float(np.linalg.norm(c)), 1e-12)
    drift = float(np.linalg.norm(b - b_anchor))
    return U @ a, V @ b, drift


def curvature_candidates(
    oracle: Oracle,
    U: np.ndarray,
    V: np.ndarray,
    count: int,
    rng: np.random.Generator,
    scale: float,
    probe_points: int = 6,
) -> list[np.ndarray]:
    """Shi&Fu-class gate candidates: finite-difference curvature near zero.

    Second differences of the even part along coordinate-pair directions in
    the active span estimate the channel-aggregated Hessian; its top
    eigenvectors are the gate candidates (curvature concentrates where the
    strong atoms live).
    """
    rank = V.shape[1]
    rows, cols = np.triu_indices(rank)
    # probes: e_i alone gives Q_ii; e_i + e_j gives Q_ii + Q_jj + 2 Q_ij
    coords = []
    labels = []
    for i in range(rank):
        d = np.zeros(rank)
        d[i] = 1.0
        coords.append(d)
        labels.append((i, i))
    for i, j in zip(rows, cols):
        if i == j:
            continue
        d = np.zeros(rank)
        d[i] += 1.0
        d[j] += 1.0
        coords.append(d)
        labels.append((i, j))
    coords = np.array(coords)
    world_dirs = (V @ coords.T).T * scale
    y_plus = oracle(world_dirs)
    y_minus = oracle(-world_dirs)
    y_zero = oracle(np.zeros((1, world_dirs.shape[1])))
    kappa = (y_plus + y_minus - 2.0 * y_zero) / (scale * scale)
    channel = int(np.argmax(np.var(kappa, axis=0)))
    kappa_channel = kappa[:, channel]
    quadratic = np.zeros((rank, rank))
    diag = {}
    for k, (i, j) in enumerate(labels):
        if i == j:
            diag[i] = float(kappa_channel[k])
    for k, (i, j) in enumerate(labels):
        if i == j:
            quadratic[i, i] = diag[i]
        else:
            quadratic[i, j] = quadratic[j, i] = 0.5 * (
                float(kappa_channel[k]) - diag[i] - diag[j]
            )
    values, vectors = np.linalg.eigh(quadratic)
    order = np.argsort(np.abs(values))[::-1][:count]
    return [vectors[:, index] for index in order]


def oddsvd_candidates(odd_map: np.ndarray, count: int) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Top singular (up, gate) pairs of the residual odd map (mixture - start points only)."""
    left, singular, right_t = np.linalg.svd(odd_map, full_matrices=False)
    out = []
    for index in range(min(count, singular.shape[0])):
        if singular[index] <= 1e-12:
            break
        out.append((left[:, index].copy(), right_t[index, :].copy(), float(singular[index])))
    return out


def quartic_gate(
    residual_fn,
    V: np.ndarray,
    s0: float,
    rng: np.random.Generator,
    steps: int = 16,
    epsilon: float = 0.03,
    step_start: float = 0.25,
    step_end: float = 0.03,
) -> np.ndarray:
    """Extract ONE gate direction from the residual via the quartic form.

    For a bias-free SiLU residual r (r(0) = 0), with
        kappa_s(d) = (r(s d) + r(-s d)) / s^2,        (odd part cancels exactly)
        kappa_s(d) = c2(d)/2 - s^2 c4(d)/24 + s^4 c6(d)/240 - ...
    where ck(d) = sum_i u_i (g_i . d)^k, the three-scale combination
        Delta3(d) = kappa_s(d) - 17 kappa_{s/2}(d) + 16 kappa_{s/4}(d)
    kills BOTH the quadratic and the sixth-order terms exactly, leaving
        Delta3(d) = -(3 s^2 / 32) c4(d) + O(s^8).
    The quadratic term is the rotation-degenerate part (why joint fits mix
    atoms); the quartic concentrates on the strongest atom aligned with d,
    so gradient ascent on |Delta3| isolates a single gate - the piece the
    odd-map SVD cannot give. Returns unit coordinates in the V-span.
    """
    rank = V.shape[1]

    def kappa(points_world: np.ndarray, s: float) -> np.ndarray:
        y_plus = residual_fn(points_world * s)
        y_minus = residual_fn(-points_world * s)
        return (y_plus + y_minus) / (s * s)

    def delta3_at(coords: np.ndarray, channel: int) -> float:
        world = (V @ coords)[None, :]
        diff = kappa(world, s0) - 17.0 * kappa(world, s0 * 0.5) + 16.0 * kappa(world, s0 * 0.25)
        return float(diff[0, channel])

    def delta3_many(world: np.ndarray) -> np.ndarray:
        return kappa(world, s0) - 17.0 * kappa(world, s0 * 0.5) + 16.0 * kappa(world, s0 * 0.25)

    # channel selection + ascent seed from random unit probes
    probe_count = max(4 * rank, 64)
    probes = rng.normal(size=(probe_count, rank))
    probes /= np.linalg.norm(probes, axis=1, keepdims=True)
    world = probes @ V.T
    delta = delta3_many(world)
    channel = int(np.argmax(np.var(delta, axis=0)))
    seed_index = int(np.argmax(np.abs(delta[:, channel])))
    d = probes[seed_index].copy()

    for step_index in range(steps):
        sign = np.sign(delta3_at(d, channel))
        if sign == 0.0:
            break
        grad = np.zeros(rank)
        for j in range(rank):
            dp = d.copy()
            dp[j] += epsilon
            dm = d.copy()
            dm[j] -= epsilon
            grad[j] = (delta3_at(dp, channel) - delta3_at(dm, channel)) / (2.0 * epsilon)
        grad *= sign
        norm = float(np.linalg.norm(grad))
        if norm < 1e-14:
            break
        t = step_index / max(steps - 1, 1)
        step_size = step_start + (step_end - step_start) * t
        d = d + step_size * grad / norm
        d /= float(np.linalg.norm(d))
    return d


def backfit(
    oracle: Oracle,
    atoms: list,
    U: np.ndarray,
    V: np.ndarray,
    dim: int,
    rng: np.random.Generator,
    rows: int = 8000,
    rounds: int = 3,
    drift_cap: float = 0.2,
) -> list:
    """Coordinate-descent polish: refit each atom against the residual with
    all OTHER atoms already subtracted (backfitting for additive models).

    During the peel sweep a single-atom fit faces an N-atom residual and the
    loss pulls toward mixtures; here the target contains essentially one
    true atom, so the anchored optimum IS that atom. Oracle cost: one f()
    batch per round (models are local).
    """
    for _bf_round in range(rounds):
        x_bf = sample_inputs(rng, rows, dim)
        y_bf = oracle(x_bf)
        model_all = np.zeros_like(y_bf)
        for up, gate in atoms:
            model_all += silu(x_bf @ gate)[:, None] * up[None, :]
        improved = False
        for k in range(len(atoms)):
            up_k, gate_k = atoms[k]
            own = silu(x_bf @ gate_k)[:, None] * up_k[None, :]
            r_k = y_bf - (model_all - own)
            up_p, gate_p, drift = polish_anchored_atom(
                x_bf, r_k, U, V, U.T @ up_k, V.T @ gate_k, lam=100.0
            )
            if drift > drift_cap:
                continue
            phi = silu(x_bf @ gate_p)
            gram = float(phi @ phi)
            if gram < 1e-14:
                continue
            up_p = (phi @ r_k) / gram
            pred_old = own
            pred_new = phi[:, None] * up_p[None, :]
            if float(np.mean((pred_new - r_k) ** 2)) < float(np.mean((pred_old - r_k) ** 2)):
                atoms[k] = (up_p, gate_p)
                model_all += pred_new - pred_old
                improved = True
        if not improved:
            break
    return atoms


# ---------------------------------------------------------------------------
# the peel loop shared by both sequential arms
# ---------------------------------------------------------------------------


@dataclass
class PeelAccounting:
    accepted: int = 0
    rejected_contamination: int = 0
    rejected_low_amplitude: int = 0
    rejected_low_support: int = 0
    rejected_budget: int = 0
    notes: list = field(default_factory=list)


def peel(
    oracle: Oracle,
    dim: int,
    rank: int,
    atom_count: int,
    arm: str,
    rng: np.random.Generator,
    train_rows: int = 8000,
    polish_rows: int = 8000,
    verify_rows: int = 4000,
    candidate_count: int = 6,
    min_explained: float = 0.6,
    min_amplitude: float = 1e-4,
    max_rounds: int | None = None,
    curvature_scale: float = 0.35,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict]:
    """Sequential atom peeling with per-atom verification.

    Returns recovered atoms as (up, gate) in world coordinates plus a
    bookkeeping dict (accounting, oracle calls, timings).
    """
    accounting = PeelAccounting()
    # spans from the full-block odd map
    x_span = sample_inputs(rng, min(4 * rank, 3000), dim)
    odd_map, _ = estimate_odd_map(oracle, x_span)
    U, V = active_spans(odd_map, rank)
    r = V.shape[1]

    atoms: list[tuple[np.ndarray, np.ndarray]] = []

    def model_eval(x: np.ndarray) -> np.ndarray:
        if not atoms:
            return np.zeros((x.shape[0], U.shape[0]))
        out = np.zeros((x.shape[0], U.shape[0]))
        for up, gate in atoms:
            z = x @ gate
            out += silu(z)[:, None] * up[None, :]
        return out

    def residual_at(x: np.ndarray) -> np.ndarray:
        return oracle(x) - model_eval(x)

    rounds = atom_count if max_rounds is None else min(max_rounds, atom_count)
    t0 = time.time()
    x_verify = sample_inputs(rng, verify_rows, dim)
    y_verify = residual_at(x_verify)
    energy = float(np.mean(y_verify * y_verify))
    total_energy = energy
    consecutive_rejects = 0

    def residual_fn(x: np.ndarray) -> np.ndarray:
        return oracle(x) - model_eval(x)

    for round_index in range(rounds):
        if consecutive_rejects >= 3:
            accounting.notes.append(
                f"round {round_index}: stopping after {consecutive_rejects} consecutive rejections"
            )
            break
        # the per-round odd-map regression is only needed by the oddsvd arm
        # (spans were estimated once above; up inits come from closed_form_up)
        residual_odd_map = None
        if arm == "oddsvd":
            x_odd = sample_inputs(rng, min(2 * rank, 1500), dim)
            r_plus = residual_at(x_odd)
            r_minus = residual_at(-x_odd)
            r_odd = 0.5 * (r_plus - r_minus)
            odd_coeff, *_ = np.linalg.lstsq(x_odd, r_odd, rcond=None)
            residual_odd_map = odd_coeff.T

        # fresh polish set for this round (residual targets)
        x_fit = sample_inputs(rng, polish_rows, dim)
        r_fit = residual_at(x_fit)

        def closed_form_up(gate_world: np.ndarray) -> np.ndarray | None:
            """Optimal up for a fixed gate: linear least squares on the round's residual."""
            phi = silu(x_fit @ gate_world)
            gram = float(phi @ phi)
            if gram < 1e-14:
                return None
            return (phi @ r_fit) / gram

        # candidate generation per arm
        accepted_gates = [gate for _, gate in atoms]

        def is_duplicate(gate_world: np.ndarray, threshold: float = 0.9) -> bool:
            for g_acc in accepted_gates:
                if abs(float(np.dot(g_acc, gate_world))) > threshold:
                    return True
            return False

        if arm == "starport":
            candidates = []
            for _attempt in range(3):
                gate_coords = quartic_gate(residual_fn, V, s0=0.8, rng=rng)
                gate_world = V @ gate_coords
                if is_duplicate(gate_world):
                    accounting.rejected_contamination += 1
                    continue
                up_world = closed_form_up(gate_world)
                if up_world is None:
                    continue
                candidates.append((up_world, gate_world, 1.0))
            # de-duplicate near-identical extractions
            deduped = []
            for up_w, gate_w, w in candidates:
                if all(
                    float(np.linalg.norm(gate_w - g2)) > 0.05
                    for _u2, g2, _w2 in deduped
                ):
                    deduped.append((up_w, gate_w, w))
            candidates = deduped
        elif arm == "curvature":
            gate_dirs = curvature_candidates(
                oracle if not atoms else _ResidualOracle(oracle, model_eval),
                U,
                V,
                candidate_count,
                rng,
                scale=curvature_scale,
            )
            candidates = []
            for g_coord in gate_dirs:
                g_world = V @ (g_coord / np.linalg.norm(g_coord))
                if is_duplicate(g_world):
                    continue
                up_world = closed_form_up(g_world)
                if up_world is None or np.linalg.norm(up_world) < 1e-12:
                    continue
                candidates.append((up_world, g_world, float(np.linalg.norm(up_world))))
        else:  # oddsvd (mixture starts - kept for comparison)
            pairs = oddsvd_candidates(residual_odd_map, candidate_count)
            candidates = []
            for up_dir, gate_dir, sigma in pairs:
                up_world = U @ up_dir
                gate_world = V @ gate_dir
                candidates.append((up_world, gate_world, sigma))

        if not candidates:
            accounting.notes.append(f"round {round_index}: no candidates")
            break

        best = None
        for cand_index, (up_w, gate_w, weight) in enumerate(candidates):
            gate_norm = float(np.linalg.norm(gate_w))
            if gate_norm < 1e-12:
                continue
            if arm == "starport":
                # structural extraction: the quartic direction anchors the gate;
                # an anchored polish may correct the last percent of direction
                # but must stay within a small ball of the anchor.
                gate = gate_w / gate_norm
                phi = silu(x_fit @ gate)
                gram = float(phi @ phi)
                if gram < 1e-14:
                    continue
                up = (phi @ r_fit) / gram
                pred = phi[:, None] * up[None, :]
                loss = float(np.mean((pred - r_fit) ** 2))
                up_p, gate_p, drift = polish_anchored_atom(
                    x_fit, r_fit, U, V, U.T @ up, V.T @ gate, lam=100.0
                )
                if drift <= 0.1:
                    phi_p = silu(x_fit @ gate_p)
                    gram_p = float(phi_p @ phi_p)
                    if gram_p > 1e-14:
                        up_p = (phi_p @ r_fit) / gram_p
                    pred_p = phi_p[:, None] * up_p[None, :]
                    loss_p = float(np.mean((pred_p - r_fit) ** 2))
                    if loss_p <= loss:
                        up, gate, loss = up_p, gate_p, loss_p
            else:
                a_init = U.T @ up_w
                b_init = V.T @ (gate_w / gate_norm)
                a, b, loss = refine_single_atom(x_fit, r_fit, U, V, a_init, b_init)
                up = U @ a
                gate = V @ b  # unit norm by construction
                # closed-form optimal up for the final gate direction
                phi = silu(x_fit @ gate)
                gram = float(phi @ phi)
                if gram > 1e-14:
                    up = (phi @ r_fit) / gram
            if np.linalg.norm(up) < 1e-12:
                continue
            if best is None or loss < best[0]:
                best = (loss, up, gate)

        if best is None:
            accounting.rejected_contamination += 1
            accounting.notes.append(f"round {round_index}: all candidates degenerate")
            consecutive_rejects += 1
            continue

        _, up, gate = best
        # verification on the held-out set: the atom must explain a large
        # fraction of its own predicted contribution (contamination gate)
        atom_pred = silu(x_verify @ gate)[:, None] * up[None, :]
        predicted_energy = float(np.mean(atom_pred * atom_pred))
        residual_after = y_verify - atom_pred
        energy_after = float(np.mean(residual_after * residual_after))
        explained = (energy - energy_after) / max(predicted_energy, 1e-30)
        amplitude_ratio = predicted_energy / max(total_energy, 1e-30)

        if amplitude_ratio < min_amplitude:
            accounting.rejected_low_amplitude += 1
            accounting.notes.append(
                f"round {round_index}: low amplitude {amplitude_ratio:.2e}"
            )
            consecutive_rejects += 1
            continue
        if explained < min_explained:
            accounting.rejected_contamination += 1
            accounting.notes.append(
                f"round {round_index}: explained {explained:.3f} below {min_explained}"
            )
            consecutive_rejects += 1
            continue

        consecutive_rejects = 0
        atoms.append((up, gate))
        accounting.accepted += 1
        # completion: polish the just-accepted atom against the residual that
        # remains after its own subtraction (one fewer atom to mix with)
        x_post = sample_inputs(rng, min(4000, polish_rows), dim)
        r_post = residual_at(x_post)
        up_c, gate_c, drift_c = polish_anchored_atom(
            x_post, r_post, U, V, U.T @ up, V.T @ gate, lam=100.0
        )
        if drift_c <= 0.15:
            phi_c = silu(x_post @ gate_c)
            gram_c = float(phi_c @ phi_c)
            if gram_c > 1e-14:
                up_c = (phi_c @ r_post) / gram_c
                pred_c = phi_c[:, None] * up_c[None, :]
                base = float(np.mean(r_post * r_post))
                after = float(np.mean((r_post - pred_c) ** 2))
                if after < base:
                    atoms[-1] = (up_c, gate_c)
                    up, gate = up_c, gate_c
        y_verify = residual_after
        energy = energy_after

    if atoms and arm in {"starport", "curvature"}:
        atoms = backfit(oracle, atoms, U, V, dim, rng, rows=min(polish_rows, 8000), rounds=3)

    elapsed = time.time() - t0
    book = {
        "arm": arm,
        "accounting": {
            "accepted": accounting.accepted,
            "rejectedContamination": accounting.rejected_contamination,
            "rejectedLowAmplitude": accounting.rejected_low_amplitude,
            "rejectedLowSupport": accounting.rejected_low_support,
            "rejectedBudget": accounting.rejected_budget,
        },
        "notes": accounting.notes[:12],
        "elapsedSeconds": elapsed,
    }
    return atoms, book


class _ResidualOracle:
    """Wraps (oracle, model) so curvature probes see the residual function."""

    def __init__(self, oracle: Oracle, model_eval):
        self.oracle = oracle
        self.model_eval = model_eval

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.oracle(x) - self.model_eval(x)


# ---------------------------------------------------------------------------
# arm baseline: multi-start joint nonlinear least squares (recorded method class)
# ---------------------------------------------------------------------------


def joint_fit(
    oracle: Oracle,
    dim: int,
    rank: int,
    atom_count: int,
    rng: np.random.Generator,
    train_rows: int = 8000,
    restarts: int = 4,
    maxiter: int = 400,
) -> list[tuple[np.ndarray, np.ndarray]]:
    x_span = sample_inputs(rng, min(4 * rank, 3000), dim)
    odd_map, _ = estimate_odd_map(oracle, x_span)
    U, V = active_spans(odd_map, rank)
    r = V.shape[1]

    x_fit = sample_inputs(rng, train_rows, dim)
    y_fit = oracle(x_fit)

    def make_starts():
        starts = []
        pairs = oddsvd_candidates(odd_map, atom_count)
        structured = []
        for index in range(atom_count):
            if index < len(pairs):
                up_dir, gate_dir, _ = pairs[index % len(pairs)]
                structured.append((U @ up_dir, V @ gate_dir))
            else:
                structured.append(
                    (U @ rng.normal(size=r), V @ rng.normal(size=r))
                )
        starts.append(structured)
        for _ in range(restarts - 1):
            starts.append(
                [(U @ rng.normal(size=r), V @ rng.normal(size=r)) for _ in range(atom_count)]
            )
        return starts

    best_loss = None
    best_atoms = None
    for start_index, start in enumerate(make_starts()):
        a0 = np.stack([U.T @ up for up, _ in start])
        b0 = np.stack([V.T @ gate for _, gate in start])

        def unpack(theta):
            A = theta[: atom_count * r].reshape(atom_count, r)
            B = theta[atom_count * r:].reshape(atom_count, r)
            return A, B

        def objective(theta):
            A, B = unpack(theta)
            G = (V @ B.T).T  # atom_count x dim gates
            Ups = (U @ A.T).T  # atom_count x dim ups
            z = x_fit @ G.T
            phi = silu(z)
            pred = phi @ Ups
            err = pred - y_fit
            loss = float(np.mean(err * err))
            d_ups = (2.0 / err.size) * (phi.T @ err)
            grad_A = d_ups @ U
            dphi = silu_prime(z)
            coeff = (2.0 / err.size) * (err @ Ups.T)
            grad_B = (dphi * coeff).T @ x_fit @ V
            return loss, np.concatenate([grad_A.ravel(), grad_B.ravel()])

        theta0 = np.concatenate([a0.ravel(), b0.ravel()])
        result = minimize(
            objective,
            theta0,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-10},
        )
        A, B = unpack(result.x)
        atoms = [(U @ A[index], V @ B[index]) for index in range(atom_count)]
        if best_loss is None or result.fun < best_loss:
            best_loss = float(result.fun)
            best_atoms = atoms
    return best_atoms


# ---------------------------------------------------------------------------
# gates (same family as the recorded T4 gates)
# ---------------------------------------------------------------------------


def grade_atoms(
    atoms: list[tuple[np.ndarray, np.ndarray]],
    true_gates: np.ndarray,
    true_ups: np.ndarray,
) -> dict:
    if not atoms:
        return {
            "pairedAtoms": 0,
            "subspaceMinPrincipalCosine": 0.0,
            "gateMaxAbsCosine": 0.0,
        }
    rec_gates = np.stack([g for _, g in atoms])
    rec_ups = np.stack([u for u, _ in atoms])
    true_gate_span, *_ = np.linalg.svd(true_gates.T, full_matrices=False)
    rec_gate_span, *_ = np.linalg.svd(rec_gates.T, full_matrices=False)
    M = true_gate_span.T @ rec_gate_span
    singular = np.linalg.svd(M, compute_uv=False)
    subspace = float(np.min(singular))
    paired = 0
    best_cosines = []
    for index in range(true_gates.shape[0]):
        cos_g = np.abs(rec_gates @ true_gates[index]) / np.maximum(
            1e-30, np.linalg.norm(rec_gates, axis=1)
        )
        cos_u = np.abs(rec_ups @ true_ups[index]) / np.maximum(
            1e-30, np.linalg.norm(rec_ups, axis=1)
        )
        joint = np.minimum(cos_g, cos_u)
        best = int(np.argmax(joint))
        best_cosines.append(float(joint[best]))
        if joint[best] >= 0.99:
            paired += 1
    return {
        "pairedAtoms": paired,
        "subspaceMinPrincipalCosine": subspace,
        "gateMaxAbsCosine": float(np.max(best_cosines)) if best_cosines else 0.0,
        "bestJointCosines": [round(c, 4) for c in sorted(best_cosines, reverse=True)[:8]],
    }


def functional_metrics(
    oracle: Oracle,
    atoms: list[tuple[np.ndarray, np.ndarray]],
    x_val: np.ndarray,
) -> dict:
    y_true = oracle(x_val)
    y_pred = np.zeros_like(y_true)
    for up, gate in atoms:
        y_pred += silu(x_val @ gate)[:, None] * up[None, :]
    err = y_pred - y_true
    denom = np.mean((y_true - y_true.mean()) ** 2)
    return {
        "r2": float(1.0 - np.mean(err * err) / max(denom, 1e-30)),
        "mae": float(np.mean(np.abs(err))),
        "worstError": float(np.max(np.abs(err))),
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


PRESETS = {
    "small": {
        "atom_count": 16,
        "seed": 260621,
        "train_rows": 6000,
        "val_rows": 2000,
        "noise": 0.0,
    },
    "lanef": {
        "atom_count": 48,
        "seed": 260621,
        "train_rows": 20000,
        "val_rows": 4000,
        "noise": 0.0,
    },
    "lanef-noise": {
        "atom_count": 48,
        "seed": 260621,
        "train_rows": 20000,
        "val_rows": 4000,
        "noise": 2 ** -10,
    },
}


def run_arm(arm: str, cfg: dict, quiet: bool = False) -> dict:
    atom_count = cfg["atom_count"]
    dim, gates, ups = make_teacher(atom_count, cfg["seed"])
    scale_ref = float(np.max(np.abs(silu(sample_inputs(np.random.default_rng(7), 512, dim) @ gates.T) @ ups)))
    noise_step = cfg["noise"] * scale_ref if cfg["noise"] > 0 else 0.0
    oracle = Oracle(gates, ups, noise_step=noise_step)
    rng = np.random.default_rng(cfg["seed"] + 1)
    x_val = sample_inputs(rng, cfg["val_rows"], dim)
    t0 = time.time()

    if arm == "baseline":
        atoms = joint_fit(
            oracle,
            dim,
            rank=dim,
            atom_count=atom_count,
            rng=rng,
            train_rows=cfg["train_rows"],
            restarts=cfg.get("restarts", 4),
        )
        book = {"arm": arm, "elapsedSeconds": time.time() - t0, "accounting": {}}
    else:
        atoms, book = peel(
            oracle,
            dim,
            rank=dim,
            atom_count=atom_count,
            arm=arm,
            rng=rng,
            train_rows=cfg["train_rows"],
            polish_rows=min(cfg["train_rows"], 8000),
            verify_rows=cfg["val_rows"],
        )

    result = {
        "arm": arm,
        "atomCount": atom_count,
        "oracleCalls": oracle.calls,
        "functional": functional_metrics(oracle, atoms, x_val),
        "structure": grade_atoms(atoms, gates, ups),
        "book": book,
    }
    if not quiet:
        f = result["functional"]
        s = result["structure"]
        print(
            f"  {arm:>10} | R2 {f['r2']:.6f} | MAE {f['mae']:.2e} | worst {f['worstError']:.2e} "
            f"| subspace {s['subspaceMinPrincipalCosine']:.4f} | paired {s['pairedAtoms']}/{atom_count} "
            f"| calls {result['oracleCalls']:,} | {book['elapsedSeconds']:.1f}s"
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="small")
    parser.add_argument("--arms", default="baseline,curvature,starport")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    cfg = PRESETS[args.preset]
    print(f"preset {args.preset}: atoms={cfg['atom_count']}, dim={2 * cfg['atom_count']}, "
          f"rows={cfg['train_rows']}, noise={cfg['noise']}")
    results = [run_arm(arm.strip(), cfg) for arm in args.arms.split(",") if arm.strip()]
    if args.output:
        with open(args.output, "w") as handle:
            json.dump({"preset": args.preset, "results": results}, handle, indent=1)
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
