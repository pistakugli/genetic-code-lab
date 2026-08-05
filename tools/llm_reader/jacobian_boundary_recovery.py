#!/usr/bin/env python3
"""Recover a SwiGLU block from activation boundaries in a vector-valued oracle.

This is the live-model form of the route measured in `jacobian_jump_probe.py`. Everything it
needs from the model arrives through `OracleClient`, so the access invariant is the same as
every other lane here: batch outputs in, no weight ever read by the solver.

The existing recovery path spends most of its budget before it starts. Building the active
subspace from scalar gradients costs one finite-difference pass per probe point over the full
input dimension - measured at 117409 calls of a 230633-call run on a 16-unit LLaMA block -
and then a nonlinear fit searches for gate directions that a boundary already determines.

Three observations remove both costs.

Jacobian row spaces answer the subspace question. With a vector-valued oracle the Jacobian is
`J(x) = sum_i d_i (x) da_i/dx`, and `da_i/dx` lies in `span{g_i, u_i}` for every `i`, so the
row space of a Jacobian sits inside the active subspace. A few stacked Jacobians give a basis
for a few finite-difference passes - `2 * input_dim` calls each - instead of one pass per
probe point. One is not enough: at a single point each unit contributes only its own mixture
of `g_i` and `u_i`, so the rank is `k` rather than `2k`.

A boundary crossing hands over the output direction for two calls. The jump is
`(u_i.x) * d_i (x) g_i`, so any directional derivative of it, `dJ.v = c(g_i.v) d_i`, is
parallel to `d_i`. One probe direction is enough to read `d_i` off the peak; there is no need
to assemble the jump matrix at all.

The gate direction is a null space, not a fit. Every crossing of unit `i` sits on
`g_i.x = 0`. Collecting crossings that share an output direction gives points on one
hyperplane, and in the recovered basis that hyperplane needs `2k` points rather than
`input_dim` of them. `g_i` is the smallest singular vector of those points - no optimisation,
no restarts, nothing to get stuck in.

What this does not do: it recovers `g_i` and `d_i`. `u_i` follows by least squares once the
gates are known, which `jacobian_jump_probe.close_layer_from_known_gates` measures, and is
not repeated here.
"""
from __future__ import annotations

import os

# Before numpy, because the thread pool is sized at import. Most of the host work here is
# memory bound - differences and norms over `samples x output_dim` blocks - and spreading that
# over every core buys nothing. Measured end to end on a 64-unit block: 8 threads is the
# fastest at 30.4 seconds, 16 is both slower at 31.7 and twice the CPU, 4 costs one second and
# a third of the CPU. What looked like the route hammering the processor was two thirds waste.
# An explicit setting from the caller wins; this only fills in a default.
for _pool in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_pool, str(min(8, os.cpu_count() or 8)))

import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class BoundaryCrossing:
    line: int
    coordinate: float
    output_direction: np.ndarray
    basis_point: np.ndarray
    peak_sharpness: float
    # The line this crossing was found on. The amplitude sweep needs to step along it, and
    # stepping along the point's own direction instead is not the same thing at all. Held by
    # index into the caller's list rather than copied per crossing: every crossing on a line
    # shares it, and at 1024 units a scan finds 60000 crossings, so a private copy is 1 GB of
    # the same few hundred vectors written over and over.
    line_directions: list[np.ndarray] | None = None


@dataclass
class RecoveryTelemetry:
    oracleCalls: int = 0
    basisCalls: int = 0
    scanCalls: int = 0
    linesScanned: int = 0
    crossingsFound: int = 0
    clusters: int = 0
    unitsRecovered: int = 0
    amplitudeCalls: int = 0
    # Sweeps actually spent against the maximum allowed. The gap is what the per-unit stopping
    # rule saves, and it is the term that decides whether a full layer is affordable.
    amplitudeSweeps: int = 0
    amplitudeSweepsAllowed: int = 0
    duplicatesDropped: int = 0
    clustersSkippedAsSmall: int = 0
    rejectedByResidual: int = 0
    upSolveCalls: int = 0
    upsRecovered: int = 0
    # Wall seconds per phase. The route spends most of its time outside the oracle, and
    # without this the only visible number is the total, which says nothing about which part
    # to fix.
    phaseSeconds: dict[str, dict[str, float]] = field(default_factory=dict)
    # How close the line scan is to its own resolution limit. A `silu` knee has a width along
    # the line of `1/(S ||g|| |g.d|)`, and boundaries are spaced about `2/k` apart, so the
    # scan stops separating them once `k` approaches `2 S ||g|| / sqrt(dim)`. Peaks are then
    # thinned to one per `separation` samples, which caps what a line can report at
    # `samples / separation` regardless. Reported rather than gated: the threshold is not yet
    # measured, and a number with an invented cutoff is worse than a number.
    crossingsPerLine: float = 0.0
    resolvableSlotsPerLine: int = 0
    notes: list[str] = field(default_factory=list)


def recover_basis_from_jacobian(
    evaluate: Callable[[np.ndarray], np.ndarray],
    input_dim: int,
    active_rank: int,
    *,
    point: np.ndarray,
    step: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Basis of the active subspace from the row space of one Jacobian.

    `da_i/dx = silu'(g_i.x)(u_i.x) g_i + silu(g_i.x) u_i` lies in `span{g_i, u_i}`, so every
    row of `J = sum_i d_i (x) da_i/dx` does too.

    One Jacobian is not enough, and assuming it was cost a full debugging cycle. At a single
    point each unit contributes one direction - its own mixture of `g_i` and `u_i` with the
    coefficients that point happens to produce - so a `k`-unit block gives a rank-`k` Jacobian,
    not rank `2k`. Taking `2k` singular vectors from it returns `k` real directions and `k`
    of noise: measured, the recovered basis held 46% of a gate vector's energy and every gate
    solved out of it came back at cosine 0.46.

    Stacking Jacobians from several points fixes it, because the coefficients differ from
    point to point and the two directions of each unit separate. The cost is one
    finite-difference pass per point, and the returned spectrum says whether the rank assumed
    was actually reached.

    Accumulated as a Gram matrix rather than a stack. The right singular vectors of the
    stacked Jacobians are the eigenvectors of `sum_p J_p^T J_p`, and that sum is
    `input_dim x input_dim` however many points are used, while the stack grows with them:
    at 512 units the basis wanted 68 points, which is 68 dense `3072 x 3072` blocks - 5 GB
    before `vstack` copies it to 10, on a machine with 30. The kernel killed that run. The
    Gram is 75 MB and the eigendecomposition is smaller than the SVD it replaces.
    """
    axes = np.eye(input_dim)
    torch = _torch_cuda()
    # `J^T J` is `input_dim` cubed per probe point - 29 GFLOP at 3072, once per point. The
    # card is already answering the oracle calls that produce `J`; there is no reason for the
    # host to do the multiply.
    # Double precision, on the card. `J^T J` squares the condition number, and the directions
    # that matter here are the small ones - the active subspace is what the basis is for. In
    # single precision the held-out reconstruction of a synthetic block fell from 0.998 to
    # 0.888, which is the whole recovery quietly degraded by a dtype.
    gram = (torch.zeros((input_dim, input_dim), dtype=torch.float64, device="cuda")
            if torch is not None else np.zeros((input_dim, input_dim), dtype=np.float64))
    calls = 0
    for probe_point in np.atleast_2d(point):
        plus = evaluate(probe_point[None, :] + step * axes)
        minus = evaluate(probe_point[None, :] - step * axes)
        jacobian = ((plus - minus) / (2.0 * step)).T
        if torch is not None:
            block = torch.as_tensor(jacobian, dtype=torch.float64, device="cuda")
            gram += block.T @ block
        else:
            gram += jacobian.T @ jacobian
        calls += 2 * input_dim
    # Symmetric by construction, so `eigh` rather than `eig`, and its ascending order is
    # reversed to match the descending convention of `svd`.
    if torch is not None:
        gram = gram.cpu().numpy()
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    singular = np.sqrt(np.clip(eigenvalues[order], 0.0, None))
    return eigenvectors[:, order].T[:active_rank], singular, calls


def scan_line_for_crossings(
    evaluate: Callable[[np.ndarray], np.ndarray],
    basis: np.ndarray,
    *,
    origin: np.ndarray,
    direction: np.ndarray,
    probe: np.ndarray,
    scale: float,
    samples: int,
    step: float,
    peak_fraction: float,
    separation: int,
    min_sharpness: float,
    line_index: int,
    line_directions: list[np.ndarray] | None = None,
    timings: dict[str, float] | None = None,
) -> tuple[list[BoundaryCrossing], int]:
    """Walk one line and read an output direction off every boundary it crosses.

    The probe stays inside the recovered basis, so the line lives in the same `2k` dimensions
    the units do. Two calls per sample: the boundaries show up as peaks in how fast the
    directional derivative turns, and the derivative at a peak is parallel to that unit's
    output direction.
    """
    if line_directions is None:
        line_directions = []
    while len(line_directions) <= line_index:
        line_directions.append(np.zeros_like(direction))
    line_directions[line_index] = direction

    entered = time.perf_counter()
    coordinates = np.linspace(-1.0, 1.0, samples)
    basis_points = origin[None, :] + coordinates[:, None] * direction[None, :]
    line = scale * (basis_points @ basis)
    probe_vector = probe @ basis

    built = time.perf_counter()
    plus = line + step * probe_vector[None, :]
    minus = line - step * probe_vector[None, :]
    asked = time.perf_counter()
    forward = evaluate(plus)
    backward = evaluate(minus)
    answered = time.perf_counter()
    calls = 2 * samples
    # The difference, its derivative along the line and the row norms, on the card. These are
    # three passes over a `samples x output_dim` block - about 20 MB each at 801 by 3072 - and
    # numpy spreads memory-bound work across every core it can find: measured at 131 seconds
    # of CPU for 11 of wall clock over 80 lines, while the card that had just produced the
    # block sat idle. Only the peaks are brought back.
    #
    # Double precision, because this is where the cancellation is: `forward` and `backward`
    # differ by something of order `step`, so single precision keeps three digits of a
    # difference the whole route is built on. Cast on the card, after the transfer - these are
    # elementwise and bandwidth-bound, so the width costs a factor of two and the wrong dtype
    # costs the recovery. Measured in single: a synthetic block's held-out reconstruction fell
    # to 0.56 and half the gates pointed the wrong way.
    torch = _torch_cuda()
    if torch is not None:
        left = torch.as_tensor(forward, device="cuda").double()
        right = torch.as_tensor(backward, device="cuda").double()
        derivative_gpu = (left - right) / (2.0 * step)
        turning_gpu = torch.gradient(
            derivative_gpu,
            spacing=(torch.as_tensor(coordinates, device="cuda").double(),),
            dim=0,
        )[0]
        magnitude = torch.linalg.vector_norm(turning_gpu, dim=1).cpu().numpy()
        turning = None
    else:
        derivative = (forward - backward) / (2.0 * step)
        turning = np.gradient(derivative, coordinates, axis=0)
        magnitude = np.linalg.norm(turning, axis=1)

    floor = float(np.max(magnitude)) * peak_fraction
    candidates = [
        index
        for index in range(2, magnitude.size - 2)
        if magnitude[index] >= floor
        and magnitude[index] >= magnitude[index - 1]
        and magnitude[index] > magnitude[index + 1]
    ]
    # One boundary produces a band of samples above the floor, not a single one. Without
    # thinning, the same crossing is entered many times and every one of those entries carries
    # a slightly different direction, which is what collapsed clustering into one bucket of
    # 523 against fourteen of fewer than twenty.
    candidates.sort(key=lambda index: -magnitude[index])
    kept: list[int] = []
    for index in candidates:
        if all(abs(index - other) >= separation for other in kept):
            kept.append(index)

    if timings is not None:
        finished = time.perf_counter()
        timings["buildPoints"] = timings.get("buildPoints", 0.0) + (asked - entered)
        timings["oracleWait"] = timings.get("oracleWait", 0.0) + (answered - asked)
        timings["postProcess"] = timings.get("postProcess", 0.0) + (finished - answered)

    crossings: list[BoundaryCrossing] = []
    # Only the kept peaks come back from the card - a few dozen rows out of `samples`.
    vectors = (
        turning_gpu[sorted(kept)].cpu().numpy() if turning is None and kept
        else None
    )
    for position, index in enumerate(sorted(kept)):
        vector = vectors[position] if vectors is not None else turning[index]
        norm = float(np.linalg.norm(vector))
        if norm <= 0.0:
            continue
        window = magnitude[max(0, index - 4 * separation):index + 4 * separation + 1]
        background = float(np.median(window))
        sharpness = float(magnitude[index] / background) if background > 0 else 0.0
        # A peak that barely stands out from its own neighbourhood is a crossing read through
        # someone else's transition, and its direction is a mixture rather than one unit's.
        if sharpness < min_sharpness:
            continue
        crossings.append(BoundaryCrossing(
            line=line_index,
            coordinate=float(coordinates[index]),
            output_direction=(vector / norm).astype(np.float32),
            basis_point=basis_points[index].astype(np.float32),
            peak_sharpness=sharpness,
            line_directions=line_directions,
        ))
    return crossings, calls


def cluster_by_output_direction(
    crossings: list[BoundaryCrossing],
    *,
    min_cosine: float,
) -> list[list[BoundaryCrossing]]:
    """Group crossings that belong to the same unit.

    Two crossings of the same boundary, on different lines, hand back the same output
    direction up to sign. Nothing else in the block shares it, which is what makes the output
    direction usable as an identity where a scalar oracle offers no identity at all.
    """
    clusters: list[list[BoundaryCrossing]] = []
    centroids: list[np.ndarray] = []
    for crossing in sorted(crossings, key=lambda item: -item.peak_sharpness):
        placed = False
        for index, centroid in enumerate(centroids):
            similarity = abs(float(crossing.output_direction @ centroid))
            if similarity >= min_cosine:
                sign = np.sign(float(crossing.output_direction @ centroid)) or 1.0
                clusters[index].append(crossing)
                updated = centroid * len(clusters[index]) + sign * crossing.output_direction
                centroids[index] = updated / np.linalg.norm(updated)
                placed = True
                break
        if not placed:
            clusters.append([crossing])
            centroids.append(crossing.output_direction.copy())

    # A second pass over centroids rather than members. One unit produced about two clusters
    # on the first pass, because a single crossing's direction carries enough noise to miss
    # the threshold against another crossing of the same boundary, while the two clusters'
    # averages do not: purity was already 1.000, so nothing here risks merging real units, it
    # only stops one unit being counted twice and its points being split between halves too
    # small to determine a hyperplane.
    merged = True
    while merged and len(clusters) > 1:
        merged = False
        for left in range(len(clusters)):
            for right in range(left + 1, len(clusters)):
                similarity = abs(float(centroids[left] @ centroids[right]))
                if similarity < min_cosine:
                    continue
                sign = np.sign(float(centroids[left] @ centroids[right])) or 1.0
                clusters[left].extend(clusters[right])
                combined = centroids[left] * len(clusters[left]) + sign * centroids[right] * len(clusters[right])
                centroids[left] = combined / np.linalg.norm(combined)
                del clusters[right]
                del centroids[right]
                merged = True
                break
            if merged:
                break
    return clusters


def gate_from_cluster(cluster: list[BoundaryCrossing], active_rank: int) -> tuple[np.ndarray, float] | None:
    """The gate direction is the null space of the points on its boundary.

    Every crossing satisfies `g.x = 0`, so in basis coordinates the points span a hyperplane
    and `g` is its normal: the smallest right singular vector. The ratio of the last two
    singular values says how well determined it is, and a cluster that does not span enough
    directions is rejected rather than guessed at.
    """
    if len(cluster) < active_rank:
        return None
    points = np.stack([crossing.basis_point for crossing in cluster]).astype(np.float64)
    _, singular, right = np.linalg.svd(points, full_matrices=False)
    if singular.size < 2 or singular[-1] <= 0.0:
        return None
    determination = float(singular[-2] / singular[-1])
    return right[-1], determination


def identifiability_verdict(
    *,
    basis_spectrum: np.ndarray,
    active_rank: int,
    gates: np.ndarray,
    output_directions: np.ndarray,
    clusters_found: int,
    duplicate_gate_cosine: float,
    collinear_threshold: float,
    weak_share_threshold: float,
    rank_energy_threshold: float,
    unit_budget: int,
    norm_diagnostics: list[dict[str, float]] | None = None,
    knee_share_threshold: float = 0.5,
    ray_spread_threshold: float = 0.25,
) -> dict[str, Any]:
    """Decide whether this recovery may be reported as structure at all.

    `docs/neural-decompiler-formalization.md` makes this mandatory, and names the outcome
    that must be a hard failure: "fails by inventing unsupported structure". A route that
    always returns something cannot distinguish a recovered block from an unrecoverable one,
    and every number it prints is then unfalsifiable.

    Each check below maps to a required negative-case family from that document, and each
    resolves to one of three honest outcomes - recovered, canonicalised to a smaller
    structure, or non-identifiable - never to a structure the evidence does not support.
    """
    findings: list[dict[str, Any]] = []

    # Mis-specified activation family, and the full-slice insufficient-rank case. If the
    # requested rank does not capture the Jacobians' energy, the block is not a `rank`-sized
    # SwiGLU sum in the first place, and everything downstream is fitted to noise.
    spectrum = np.asarray(basis_spectrum, dtype=np.float64)
    if spectrum.size:
        captured = float(np.sum(spectrum[:active_rank] ** 2) / max(np.sum(spectrum ** 2), 1e-30))
        if captured < rank_energy_threshold:
            findings.append({
                "family": "insufficient-rank-or-wrong-activation-family",
                "detail": f"rank {active_rank} captures {captured:.6f} of Jacobian energy",
                "verdict": "non-identifiable",
            })

    # Exact duplicate split: one true unit returned as several. Canonicalisation to the
    # smaller structure is correct; reporting both as units is not.
    duplicate_pairs = 0
    collinear_pairs = 0
    if gates.shape[0] > 1:
        normalised = gates / np.maximum(np.linalg.norm(gates, axis=1, keepdims=True), 1e-30)
        similarity = np.abs(normalised @ normalised.T)
        np.fill_diagonal(similarity, 0.0)
        duplicate_pairs = int(np.sum(similarity >= duplicate_gate_cosine) // 2)
        collinear_pairs = int(np.sum(similarity >= collinear_threshold) // 2) - duplicate_pairs
    if duplicate_pairs:
        findings.append({
            "family": "exact-duplicate-split",
            "detail": f"{duplicate_pairs} gate pair(s) at or above cosine {duplicate_gate_cosine}",
            "verdict": "canonicalised",
        })
    # Near-collinear directions: several bases explain the same oracle about equally well, so
    # which one is returned is not determined by the evidence.
    if collinear_pairs:
        findings.append({
            "family": "near-collinear-directions",
            "detail": f"{collinear_pairs} gate pair(s) between cosine {collinear_threshold} and {duplicate_gate_cosine}",
            "verdict": "ambiguous",
        })

    # Equal/opposite cancellation, seen from the output side: two units whose output
    # directions are anti-parallel cancel in the sum, and their split is not observable.
    cancelling_pairs = 0
    if output_directions.shape[0] > 1:
        out = output_directions / np.maximum(np.linalg.norm(output_directions, axis=1, keepdims=True), 1e-30)
        overlap = out @ out.T
        np.fill_diagonal(overlap, 0.0)
        cancelling_pairs = int(np.sum(overlap <= -collinear_threshold) // 2)
    if cancelling_pairs:
        findings.append({
            "family": "equal-opposite-cancellation",
            "detail": f"{cancelling_pairs} output pair(s) anti-parallel beyond {collinear_threshold}",
            "verdict": "ambiguous",
        })

    # Overcomplete budget: more clusters survived than the block can hold. Returning the
    # extras as units is the invented-structure failure this whole check exists to stop.
    if unit_budget > 0 and gates.shape[0] > unit_budget:
        findings.append({
            "family": "overcomplete-budget",
            "detail": f"{gates.shape[0]} solutions returned for a budget of {unit_budget}",
            "verdict": "non-identifiable",
        })

    # Weak channel below the noise floor: a solution whose share of the recovered energy is
    # negligible is not evidence of a unit, it is evidence of a probe that found nothing.
    weak = 0
    if gates.shape[0]:
        strength = np.linalg.norm(gates, axis=1)
        total = float(np.sum(strength))
        if total > 0:
            weak = int(np.sum((strength / total) < weak_share_threshold))
    if weak:
        findings.append({
            "family": "weak-channel-below-floor",
            "detail": f"{weak} solution(s) below {weak_share_threshold} of recovered gate energy",
            "verdict": "canonicalised",
        })

    # A gate norm that was not identified is a unit reported at the wrong scale, and `silu`
    # is not scale free, so that is a claim about behaviour the evidence does not carry. The
    # ray already measures both ways this can happen - a knee that never appeared in the
    # probed range, and independent rays that disagree about where it is - and computing that
    # without acting on it is the same as not measuring it.
    if norm_diagnostics:
        unidentified = [
            index for index, row in enumerate(norm_diagnostics)
            if row.get("kneeShare", 1.0) < knee_share_threshold
            or row.get("raySpread", 0.0) > ray_spread_threshold
        ]
        if unidentified:
            findings.append({
                "family": "gate-norm-not-identified",
                "detail": (
                    f"{len(unidentified)} solution(s) with knee share below "
                    f"{knee_share_threshold} or ray spread above {ray_spread_threshold}"
                ),
                "verdict": "ambiguous",
            })

    blocking = [row for row in findings if row["verdict"] == "non-identifiable"]
    return {
        "structureClaimAllowed": not blocking,
        "verdict": "non-identifiable" if blocking else ("qualified" if findings else "recovered"),
        "findings": findings,
        "clustersFound": int(clusters_found),
        "thresholds": {
            "duplicateGateCosine": float(duplicate_gate_cosine),
            "collinear": float(collinear_threshold),
            "weakShare": float(weak_share_threshold),
            "rankEnergy": float(rank_energy_threshold),
            "kneeShare": float(knee_share_threshold),
            "raySpread": float(ray_spread_threshold),
        },
    }


def solve_least_squares(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Least squares on whichever device can actually do it.

    The oracle runs on the GPU and the solve did not, which put the whole route on the wrong
    side of the machine. At 256 units the up design is 120000 by 1024 per unit and there are
    256 of them: measured at fourteen cores of numpy, that is hours of `lstsq` while the GPU
    sits idle behind an oracle that answers in seconds.

    Single precision is enough here and the reason is worth stating rather than assuming. The
    designs are strongly overdetermined and well conditioned - a hundred rows per unknown -
    so the solve returns about six digits, while the bar downstream is a direction cosine of
    0.99. Anything that needed the other seven digits would not be a recovery.

    Falls back to numpy when there is no GPU, when the problem is small enough that moving it
    would cost more than it saves - measured, the crossover sits near ten million entries:
    30000 by 512 runs 7.5 times faster on the card and 120000 by 1024 fourteen times, while
    30000 by 256 is a wash - and when the CUDA driver refuses full-rank `gels` - a
    rank-deficient design is a real case here, since near-collinear units are one of the
    negative cases this file is required to handle.
    """
    torch = _torch_cuda()
    # Anything past a hundred thousand entries. The earlier cut at ten million left the up
    # solve of a 64-unit block - 30000 by 256 - on the host, which is the shape this route
    # runs most often.
    if torch is not None and design.size >= 100_000:
        try:
            left = torch.as_tensor(design, dtype=torch.float32, device="cuda")
            right = torch.as_tensor(target, dtype=torch.float32, device="cuda")
            solution = torch.linalg.lstsq(left, right.unsqueeze(1)).solution
            if torch.isfinite(solution).all():
                return solution.squeeze(1).double().cpu().numpy()
        except Exception:
            pass
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    return solution


class _Phase:
    """Wall seconds and cores held, for one stage.

    The route alternates: build points here, wait for the card, process the answer here, and
    the only visible number was the total, which cannot say which half to fix. `coresHeld` is
    the ratio of this process's CPU time to wall time, so a phase reading 26 is holding
    twenty-six cores while the card idles - which is how the reconstruction turned out to be
    accumulating outer products where one matrix multiply belonged.

    Solver CPU only, deliberately. `os.times()` credits a child's time when it is reaped and
    the oracle outlives every phase, so an oracle column read a flat zero - a number that
    looks like an answer and is not one.
    """

    def __init__(self, telemetry: "RecoveryTelemetry", name: str) -> None:
        self.telemetry = telemetry
        self.name = name

    def __enter__(self) -> "_Phase":
        self.wall = time.perf_counter()
        self.cpu = time.process_time()
        return self

    def __exit__(self, *_exc: object) -> None:
        wall = time.perf_counter() - self.wall
        cpu = time.process_time() - self.cpu
        self.telemetry.phaseSeconds[self.name] = {
            "wall": round(wall, 3),
            "solverCpu": round(cpu, 3),
            "coresHeld": round(cpu / wall, 1) if wall > 0 else 0.0,
        }


def _torch_cuda():
    """The CUDA device if there is a usable one, else `None`.

    Cached on the function so the import and the availability probe happen once. The host's
    kernel module and userspace library sit on different point releases here, so NVML refuses
    while compute is fine - `PYTORCH_NVML_BASED_CUDA_CHECK=0` is what gets past that, and it
    is set by the callers that need it.
    """
    if not hasattr(_torch_cuda, "cached"):
        try:
            import torch

            _torch_cuda.cached = torch if torch.cuda.is_available() else None
        except Exception:
            _torch_cuda.cached = None
    return _torch_cuda.cached


def _batched_normal_equations(
    grid: np.ndarray,
    window: np.ndarray,
    background: np.ndarray,
    projected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares error for every candidate knee scale, in one shot.

    The model is `silu(c s)(a s + b)` plus a quadratic background, so each candidate `c` gives
    a five-column design and the only thing that changes between them is the gated pair. Built
    as one `(C, P, 5)` tensor and reduced through five-by-five normal equations, which are
    exact enough here because the columns are a knee and a polynomial rather than an
    ill-conditioned basis.

    Runs on the GPU when there is one. The card was already answering oracle calls while this
    loop ran on the host, which is the wrong side of the machine for an operation that is the
    same arithmetic 240 times over.
    """
    torch = _torch_cuda()
    if torch is not None:
        device = "cuda"
        candidates = torch.as_tensor(grid, dtype=torch.float32, device=device)
        positions = torch.as_tensor(window, dtype=torch.float32, device=device)
        base = torch.as_tensor(background, dtype=torch.float32, device=device)
        target = torch.as_tensor(projected, dtype=torch.float32, device=device)
        scaled = candidates[:, None] * positions[None, :]
        activation = scaled * torch.sigmoid(scaled)
        design = torch.cat((
            torch.stack((activation * positions[None, :], activation), dim=2),
            base[None, :, :].expand(candidates.shape[0], -1, -1),
        ), dim=2)
        # Column norms first. The design holds a knee beside `s^2` over five decades of `s`,
        # so its columns differ by six orders of magnitude, and normal equations square that.
        # In single precision the unnormalised form put the worst synthetic norm 38% out
        # against 7% for the loop it replaced. Least squares is invariant to column scaling -
        # the solution scales inversely and the residual does not move - so this costs nothing
        # but conditioning.
        scale = torch.linalg.norm(design, dim=1, keepdim=True).clamp_min(1e-30)
        design = design / scale
        gram = design.transpose(1, 2) @ design
        moment = design.transpose(1, 2) @ target[None, :, None].expand(candidates.shape[0], -1, -1)
        # A ridge of one part in a million of the trace. Some candidates put every knee outside
        # the window, which makes two columns identically zero and the Gram singular; without
        # it the solve raises on inputs the scan is meant to reject by scoring them badly.
        ridge = 1e-6 * torch.diagonal(gram, dim1=1, dim2=2).sum(dim=1)
        gram = gram + ridge[:, None, None] * torch.eye(gram.shape[1], device=device)[None, :, :]
        solution = torch.linalg.solve(gram, moment)
        residual = design @ solution - target[None, :, None]
        return (residual.squeeze(2) ** 2).sum(dim=1).double().cpu().numpy(), solution.squeeze(2).double().cpu().numpy()

    scaled = grid[:, None] * window[None, :]
    activation = scaled / (1.0 + np.exp(-np.clip(scaled, -60.0, 60.0)))
    design = np.concatenate((
        np.stack((activation * window[None, :], activation), axis=2),
        np.broadcast_to(background, (grid.size,) + background.shape),
    ), axis=2)
    scale = np.maximum(np.linalg.norm(design, axis=1, keepdims=True), 1e-30)
    design = design / scale
    gram = np.einsum('cpi,cpj->cij', design, design)
    moment = np.einsum('cpi,p->ci', design, projected)
    ridge = 1e-6 * np.trace(gram, axis1=1, axis2=2)
    gram = gram + ridge[:, None, None] * np.eye(gram.shape[1])[None, :, :]
    solution = np.linalg.solve(gram, moment)
    residual = np.einsum('cpi,ci->cp', design, solution) - projected[None, :]
    return (residual ** 2).sum(axis=1), solution


def calibrate_gate_norms_on_rays(
    evaluate,
    gate_directions: np.ndarray,
    output_directions: np.ndarray,
    *,
    points_per_ray: int,
    min_magnitude: float,
    max_magnitude: float,
    offset_scale: float,
    request_chunk: int,
    repeats: int = 3,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Measure each gate's norm on a ray along its own direction.

    The boundary scan returns `g_i` as a direction; the norm is a separate number and it is
    the one that decides what the unit does on real inputs. `silu` is not scale free, so a
    norm that is wrong by a factor puts the unit in the wrong regime: on real hidden states
    of this model, `‖x‖ ~ 15` lands in the curved part of `silu`, which is exactly where the
    norm is felt.

    Estimating it from the isotropic probe cloud does not work, and the reason is structural
    rather than a matter of sample count. There the candidate norm enters as `silu(c z)`, and
    as `c -> 0` that tends to `c z / 2`, so the design collapses to `z * x` and the free up
    direction absorbs `c` outright. Every small candidate scores the same residual, and on a
    unit whose own projection sits near the cross-talk floor the noise picks the bottom of the
    grid. Measured on 64 units of layer 15: 25 norms landed outside +-10% of truth and several
    at the grid floor of 2e-4, which predicts silence - `silu(2e-4 z) ~ 0` says the unit
    contributes nothing at all.

    Probing along `ĝ_i` removes the degeneracy instead of fighting it. On `x(s) = s ĝ_i + p`
    with `p` orthogonal to `ĝ_i`, the pre-activation is exactly `‖g_i‖ s`, so the response
    projected on `d_i` is `silu(‖g_i‖ s)(a s + b)` and the norm is no longer entangled with
    the up direction: it sets *where the curve turns on* along the ray, which no choice of
    `a`, `b` can imitate. Every other unit sees `‖g_j‖ (ĝ_j . ĝ_i) s`, and for directions in
    3072 dimensions that inner product is around 0.018, so their transitions sit two decades
    away in `s` and do not overlap the one being measured.

    Cost is one ray per unit, which is `units * points_per_ray` oracle calls - 12k for a
    64-unit block against the 230k the block already spends, so this is not where the budget
    goes. Access is unchanged: the ray is built from the recovered direction and read through
    the same oracle.

    Returns the norms and, per unit, the share of ray variance the fitted curve explains. A
    low share means the ray did not identify that unit and the caller should not treat its
    norm as measured.
    """
    rng = np.random.default_rng(seed)
    units, input_dim = gate_directions.shape
    magnitudes = np.geomspace(min_magnitude, max_magnitude, max(points_per_ray // 2, 2))
    positions = np.concatenate((-magnitudes[::-1], magnitudes))

    # Several rays per unit, differing only in the offset, reduced by the median. The knee is
    # a property of the unit, so independent offsets have to agree on it - and a single ray
    # can be certain and wrong: on a six-unit teacher in 64 dimensions one norm came back 51%
    # high while its fit explained 99.8% of the ray, so residual quality alone certifies
    # nothing. Three rays take the worst synthetic case from 51% out to 30% and the share of
    # units outside 10% from 8.3% to 6.2%, and more than three buys almost nothing. Cost is
    # two extra passes per unit against the 440k calls a 64-unit block already spends.
    #
    # Built a batch of units at a time rather than all at once. Every ray is `points_per_ray`
    # rows of `input_dim`, so holding the whole set is `units * repeats * points_per_ray *
    # input_dim`: at 512 units that is 7 GB for the probe points, as much again for the
    # responses, and a third copy when they are concatenated. The kernel killed a 512-unit run
    # at 22 GB and this was where it went. Nothing about the measurement is joint - each unit's
    # knee is read off its own ray - so the batch only needs to be large enough to keep the
    # oracle requests near `request_chunk`.
    per_unit = repeats * positions.size
    batch_units = max(1, request_chunk // per_unit)
    rays = np.empty((batch_units * per_unit, input_dim), dtype=np.float64)

    def build(index: int, slot: int) -> None:
        direction = gate_directions[index] / np.linalg.norm(gate_directions[index])
        for repeat in range(repeats):
            offset = rng.normal(0.0, 1.0, size=input_dim)
            offset -= (offset @ direction) * direction
            offset *= offset_scale * np.sqrt(input_dim) / np.linalg.norm(offset)
            start = (slot * repeats + repeat) * positions.size
            rays[start:start + positions.size] = positions[:, None] * direction[None, :] + offset[None, :]

    # Wide enough to cover a transition anywhere in the probed range of `s`, since the whole
    # point is that the norm is unknown before the ray is read.
    candidates = np.geomspace(1.0 / max_magnitude, 4.0 / min_magnitude, 240)
    # Far from its own boundary `silu(n s)(a s + b)` is just `n a s^2 + n b s`, and every
    # other unit contributes a curve of the same shape through `ĝ_j . ĝ_i`. Fitting the ray
    # without saying so hands the whole comparison to the largest `|s|`, where every candidate
    # looks quadratic and the residual is decided by cross-talk: the first measurement on 64
    # units of layer 15 came back a factor of 90 low, locked onto the scale at which the
    # *neighbours* turn on rather than this unit. Carrying the polynomial explicitly leaves
    # only the knee, which is the one feature no background term and no other unit can
    # reproduce - `silu` is flat on one side of it and linear on the other.
    def fit_knee(window: np.ndarray, projected: np.ndarray, grid: np.ndarray) -> tuple[float, float, float]:
        background = np.stack((window ** 2, window, np.ones_like(window)), axis=1)
        flat, *_ = np.linalg.lstsq(background, projected, rcond=None)
        background_error = float(np.sum((background @ flat - projected) ** 2))
        best_norm, best_error = 1.0, float("inf")
        # Magnitude only. The ray cannot see which way the unit fires, and that is algebra
        # rather than noise: `silu(z) - silu(-z) = z`, so along a ray the two orientations
        # differ by a term linear in `s` - exactly what the background is there to absorb.
        # Scanning both signs here returned the right one for 2 of 6 synthetic units, which is
        # a coin toss reported as a measurement. The sign is settled in the up solve instead,
        # where the difference is a quadratic form in the full input space and nothing in the
        # design can take it.
        #
        # Every candidate at once rather than one per iteration. The scan is 240 candidates
        # deep and runs three times per ray, three rays per unit, so a 64-unit block used to
        # walk a Python loop 115000 times over five-column systems - the route reported one
        # minute of wall clock against twenty-four of CPU, and this was most of it. The design
        # for all candidates is a single `(C, P, 5)` tensor, and five-by-five normal equations
        # are exact here because the columns are a knee and a quadratic, not an ill-conditioned
        # basis.
        errors, solutions = _batched_normal_equations(grid, window, background, projected)
        best = int(np.argmin(errors))
        return float(grid[best]), float(errors[best]), background_error

    def measure_ray(projected: np.ndarray) -> tuple[float, float, float]:
        norm, error, reference = fit_knee(positions, projected, candidates)
        # Then again on the stretch of ray around the knee it just found. The neighbours' own
        # transitions sit at `1 / (n_j |ĝ_j.ĝ_i|)`, so how far away they are is set by the
        # input dimension: at 3072 the inner product is about 0.018 and they are two decades
        # out, but at 24 dimensions it is 0.2 and they are a factor of five away, close enough
        # to bend the fit - measured on a six-unit teacher in 24 dimensions, the worst norm
        # came back 50% high. Narrowing to a few multiples of the knee scale drops them
        # without needing to know they are there, and the knee itself decides the window, so
        # nothing here assumes what is being measured.
        for _ in range(3):
            window = np.abs(positions) <= 3.0 / abs(norm)
            if int(window.sum()) < 24:
                break
            local = np.geomspace(abs(norm) / 20.0, abs(norm) * 20.0, 120)
            norm, error, reference = fit_knee(positions[window], projected[window], local)
        return norm, error, reference

    norms = np.ones(units, dtype=np.float64)
    diagnostics: list[dict[str, float]] = []
    for first in range(0, units, batch_units):
        batch = list(range(first, min(first + batch_units, units)))
        for slot, index in enumerate(batch):
            build(index, slot)
        rows = len(batch) * per_unit
        responses = np.concatenate(
            [evaluate(rays[start:min(start + request_chunk, rows)])
             for start in range(0, rows, request_chunk)],
            axis=0,
        )
        for slot, index in enumerate(batch):
            measured, shares = [], []
            for repeat in range(repeats):
                start = (slot * repeats + repeat) * positions.size
                norm, error, reference = measure_ray(
                    responses[start:start + positions.size] @ output_directions[index]
                )
                measured.append(norm)
                shares.append(float(1.0 - error / reference) if reference > 0 else 0.0)
            # Median, not mean. One ray can land on a neighbour's knee instead of this unit's,
            # and averaging in log space lets that single answer drag the result: with four
            # rays and a geometric mean the worst synthetic norm went from 33% out to a factor
            # of ten, because a mean has no defence against one bad member. The median has
            # one, which is the whole reason for repeating the measurement.
            norms[index] = float(np.median(measured))
            diagnostics.append({
                "norm": float(norms[index]),
                # Share of the ray the knee explains that a smooth background cannot. A unit
                # whose own transition never showed up in the probed range scores near zero
                # here, and its norm is a guess rather than a measurement.
                "kneeShare": float(np.min(shares)),
                # How far apart independent rays put the same knee, as a ratio. This is the
                # measure that catches a confident wrong answer, which `kneeShare` does not.
                "raySpread": float(max(np.abs(measured)) / min(np.abs(measured)) - 1.0),
            })
    return norms, diagnostics


def _decide_gate_signs(
    raw_response: np.ndarray,
    unmixed: np.ndarray,
    coordinates: np.ndarray,
    gate_norms: np.ndarray,
) -> np.ndarray:
    """Which side of its boundary each unit fires on, decided on the card.

    Two candidate orientations per unit, each a least squares over `rows x 2*rank`, and the
    lower residual wins. Written as a host loop it built a 24 MB design a hundred and
    twenty-eight times and shipped each one across: 100 seconds of CPU for 3 of wall clock,
    31 cores held. The design is assembled where it is solved, and `coordinates` - the half
    that does not change - is uploaded once.

    Falls back to the host when there is no card, with the same arithmetic.
    """
    torch = _torch_cuda()
    units = raw_response.shape[1]
    signs = np.ones(units, dtype=np.float64)
    if torch is not None:
        coords = torch.as_tensor(coordinates, dtype=torch.float32, device="cuda")
        raw = torch.as_tensor(raw_response, dtype=torch.float32, device="cuda")
        targets = torch.as_tensor(unmixed, dtype=torch.float32, device="cuda")
        norms = torch.as_tensor(gate_norms, dtype=torch.float32, device="cuda")
        for index in range(units):
            best_error, best_sign = float("inf"), 1.0
            for candidate in (1.0, -1.0):
                z = raw[:, index] * (norms[index] * candidate)
                design = torch.cat((z.mul(torch.sigmoid(z))[:, None] * coords, coords), dim=1)
                solution = torch.linalg.lstsq(design, targets[:, index, None]).solution
                error = float(((design @ solution).squeeze(1) - targets[:, index]).pow(2).sum())
                if error < best_error:
                    best_error, best_sign = error, candidate
            signs[index] = best_sign
        return signs

    for index in range(units):
        best_error, best_sign = float("inf"), 1.0
        for candidate in (1.0, -1.0):
            z = raw_response[:, index] * (gate_norms[index] * candidate)
            activation = z / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))
            design = np.hstack((activation[:, None] * coordinates, coordinates))
            solution = solve_least_squares(design, unmixed[:, index])
            error = float(np.sum((design @ solution - unmixed[:, index]) ** 2))
            if error < best_error:
                best_error, best_sign = error, candidate
        signs[index] = best_sign
    return signs


def recover_block(
    evaluate: Callable[[np.ndarray], np.ndarray],
    *,
    input_dim: int,
    active_rank: int,
    line_count: int,
    samples: int,
    scale: float,
    step: float,
    peak_fraction: float,
    separation: int,
    min_sharpness: float,
    cluster_min_cosine: float,
    basis_points_count: int,
    basis_scale: float,
    gate_mode: str,
    amplitude_span: float,
    amplitude_min_alignment: float,
    duplicate_gate_cosine: float,
    gate_residual_tolerance: float,
    amplitude_crossings: int,
    min_cluster_size: int,
    gate_stability_cosine: float = 0.99999,
    progress: bool = False,
    up_sample_count: int,
    up_sample_scale: float,
    up_request_chunk: int = 4096,
    norm_ray_points: int = 192,
    norm_ray_min: float = 1e-2,
    norm_ray_max: float = 30.0,
    norm_ray_offset: float = 0.25,
    norm_ray_repeats: int = 3,
    reconstruction_points: int = 4000,
    up_refinement_sweeps: int = 0,
    collinear_threshold: float = 0.9,
    weak_share_threshold: float = 0.002,
    rank_energy_threshold: float = 0.98,
    unit_budget: int = 0,
    seed: int,
) -> tuple[dict[str, Any], RecoveryTelemetry]:
    rng = np.random.default_rng(seed)
    telemetry = RecoveryTelemetry()

    def unit(vector: np.ndarray) -> np.ndarray:
        return vector / np.linalg.norm(vector)

    # The basis probe and the boundary scan want opposite things from the scale. The scan
    # needs the transitions sharp, which means large. The Jacobian needs its finite difference
    # to survive the subtraction, and at scale 2e4 the block's outputs are of order 4e8 in
    # float32, where a step of 1e-4 is below the noise: measured on the live model, basis
    # energy fell from 0.596 to 0.445 as points were added, because every added Jacobian was
    # mostly noise and the SVD kept spending rank on it.
    basis_points = np.stack([unit(rng.normal(size=input_dim)) * basis_scale for _ in range(basis_points_count)])
    with _Phase(telemetry, "basis"):
        basis, spectrum, basis_calls = recover_basis_from_jacobian(
            evaluate, input_dim, active_rank, point=basis_points, step=step
        )
        telemetry.basisCalls = basis_calls
        telemetry.oracleCalls += basis_calls
        energy = float(np.sum(spectrum[:active_rank] ** 2) / max(np.sum(spectrum ** 2), 1e-30))
        telemetry.notes.append(f"basis energy in rank {active_rank}: {energy:.12f}")

    scan_timings: dict[str, float] = {}
    with _Phase(telemetry, "lineScan"):
        crossings: list[BoundaryCrossing] = []
        line_directions: list[np.ndarray] = []
        for line_index in range(line_count):
            found, calls = scan_line_for_crossings(
                evaluate,
                basis,
                origin=unit(rng.normal(size=active_rank)),
                direction=unit(rng.normal(size=active_rank)),
                probe=unit(rng.normal(size=active_rank)),
                scale=scale,
                samples=samples,
                step=step,
                peak_fraction=peak_fraction,
                separation=separation,
                min_sharpness=min_sharpness,
                line_index=line_index,
                line_directions=line_directions,
                timings=scan_timings,
            )
            crossings.extend(found)
            telemetry.scanCalls += calls
            telemetry.oracleCalls += calls
            telemetry.linesScanned += 1
        telemetry.crossingsFound = len(crossings)

    telemetry.phaseSeconds["lineScanBreakdown"] = {k: round(v, 3) for k, v in scan_timings.items()}
    with _Phase(telemetry, "clustering"):
        clusters = cluster_by_output_direction(crossings, min_cosine=cluster_min_cosine)
    telemetry.clusters = len(clusters)

    gates: list[np.ndarray] = []
    outputs: list[np.ndarray] = []
    determinations: list[float] = []
    amplitude_probe = np.eye(active_rank) @ basis if gate_mode == "amplitude" else None
    # Amplitude sweeps cost `rank * 8` calls each, so spending them on clusters that are not
    # units is where the budget goes: a 128-unit block produced 289 clusters, and sweeping all
    # of them at four crossings would cost 2.4M calls to recover 128 directions. A cluster too
    # small to be a real boundary is skipped before any call is made, and the number skipped
    # is reported so the cut is visible rather than implicit.
    ordered_clusters = [group for group in sorted(clusters, key=len, reverse=True) if len(group) >= min_cluster_size]
    telemetry.clustersSkippedAsSmall = len(clusters) - len(ordered_clusters)
    for position, cluster in enumerate(ordered_clusters):
        if progress and position % 10 == 0:
            # Long runs with no output are indistinguishable from hung ones, and a 128-unit
            # block spends most of its time in this loop.
            print(
                json.dumps({
                    "type": "progress", "phase": "gate-solve",
                    "cluster": position, "of": len(ordered_clusters),
                    "solved": len(gates), "oracleCalls": telemetry.oracleCalls,
                }),
                flush=True,
            )
        if gate_mode == "amplitude":
            # One crossing is enough. The null-space route needs `active_rank` crossings of
            # the same unit, which is why a 32-unit block recovered 17 of 32 while every one
            # it did recover came back above cosine 0.999: the misses were clusters too small
            # to determine a hyperplane, not gates read wrongly.
            ranked = sorted(cluster, key=lambda item: -item.peak_sharpness)[:amplitude_crossings]
            telemetry.amplitudeSweepsAllowed += len(ranked)
            best = ranked[0]
            centroid_direction = np.mean([
                crossing.output_direction * (1.0 if crossing.output_direction @ best.output_direction >= 0 else -1.0)
                for crossing in cluster
            ], axis=0)
            centroid_direction = centroid_direction / np.linalg.norm(centroid_direction)
            # Averaged over several crossings of the same unit, not read off one. A gate at
            # cosine 0.9985 is not good enough downstream: `g.x` is a small number on this
            # model, so a 3-degree error in `g` makes the error in `silu(g.x)` larger than
            # the signal, and the up direction solved from it came back at 15 of 32. Each
            # crossing is an independent read of the same direction, so averaging them cuts
            # that error without introducing a fit.
            # The check that decides how many crossings this unit needs. Every point in the
            # cluster satisfies `g.x = 0` for the true gate, so the residual over the cluster
            # says whether the direction read so far is already consistent with every point
            # that produced it - and it costs no calls and needs no truth.
            cluster_points = np.stack([crossing.basis_point for crossing in cluster]).astype(np.float64)
            cluster_norms = np.linalg.norm(cluster_points, axis=1)

            # Which crossings feed the estimate, so the rest can judge it.
            ranked_ids = {id(crossing) for crossing in ranked}
            held_out_mask = np.array([id(c) not in ranked_ids for c in cluster], dtype=bool)

            estimates: list[np.ndarray] = []
            alignments: list[float] = []
            gate_full = np.zeros(active_rank)
            residual, alignment = float("inf"), 0.0
            for crossing in ranked:
                point = scale * (crossing.basis_point.astype(np.float64) @ basis)
                if crossing.line_directions is None:
                    continue
                span_full = crossing.line_directions[crossing.line].astype(np.float64) @ basis
                span = amplitude_span * scale * span_full / max(float(np.linalg.norm(span_full)), 1e-30)
                estimate, single_alignment, spent = gate_from_peak_amplitudes(
                    evaluate,
                    peak_point=point,
                    step_along_line=span,
                    output_direction=centroid_direction,
                    probe_basis=amplitude_probe,
                    step=step,
                )
                telemetry.oracleCalls += spent
                telemetry.amplitudeCalls += spent
                telemetry.amplitudeSweeps += 1
                if single_alignment < amplitude_min_alignment:
                    continue
                if estimates and float(estimate @ estimates[0]) < 0.0:
                    estimate = -estimate
                estimates.append(estimate)
                alignments.append(single_alignment)
                averaged = np.mean(np.stack(estimates), axis=0)
                gate_full = averaged / max(float(np.linalg.norm(averaged)), 1e-30)
                alignment = float(np.mean(alignments))
                residual = float(
                    np.median(np.abs(cluster_points @ gate_full) / np.maximum(cluster_norms, 1e-30))
                )
                # Stop when the direction already explains the crossings it was *not* built
                # from. How many crossings a unit needs is set by how crowded its neighbourhood
                # is, and that varies unit to unit: one read is enough for every unit of a
                # 64-unit block but leaves four units of a 128-unit block with no solution at
                # all, while the ranked list lets those retry. A fixed count either overpays
                # for the easy units or fails the hard ones, and this is the term that decides
                # the budget for a full layer - `8 * rank` calls per sweep, per unit.
                #
                # Audit finding F19 was that this test and the rejection filter below were the
                # same expression against the same threshold, so a unit that stopped early
                # could not be rejected. Measurement then contradicted the severity: the
                # residual was already evaluated over the whole cluster while the estimate came
                # from one to three of its crossings, so it was a held-out check in substance -
                # and the 128-unit run with one fixed crossing returned no bad gate at all
                # (every solution above cosine 0.99886), it simply missed four units. The
                # defect was the appearance of circularity, not its consequence.
                #
                # It is made explicit rather than argued: stopping is decided on the crossings
                # that did not enter the estimate, rejection on all of them. Two quantities,
                # so neither can vouch for the other. Stability of the direction between
                # successive readings was tried instead and is sound but wasteful - it cannot
                # stop before two readings, so every easy unit pays for a second, 261 sweeps
                # against 132 for identical quality.
                if held_out_mask.any():
                    held_out = float(np.median(
                        np.abs(cluster_points[held_out_mask] @ gate_full)
                        / np.maximum(cluster_norms[held_out_mask], 1e-30)
                    ))
                    if held_out <= gate_residual_tolerance and alignment >= amplitude_min_alignment:
                        break
            if not estimates:
                continue
            if alignment < amplitude_min_alignment:
                continue
            # Solutions that match no unit fail the same check, which is what removes them
            # from the returned set instead of leaving a caller to filter them.
            if residual > gate_residual_tolerance:
                telemetry.rejectedByResidual += 1
                continue
            gate, determination = gate_full, alignment
        else:
            solved = gate_from_cluster(cluster, active_rank)
            if solved is None:
                continue
            gate, determination = solved
        # Two clusters of one unit both solve to the same gate, and the second adds nothing
        # but drags the reported minimum down: measured, 42 solutions for 32 units took the
        # minimum cosine from 0.999 to 0.789 while the median stayed at 0.99997. Dropping a
        # duplicate is not dropping a unit - the direction is already held.
        if any(abs(float(gate @ existing)) >= duplicate_gate_cosine for existing in gates):
            telemetry.duplicatesDropped += 1
            continue
        gates.append(gate)
        centroid = np.mean([
            crossing.output_direction * (1.0 if crossing.output_direction @ cluster[0].output_direction >= 0 else -1.0)
            for crossing in cluster
        ], axis=0)
        outputs.append(centroid / np.linalg.norm(centroid))
        determinations.append(determination)
    telemetry.unitsRecovered = len(gates)
    telemetry.crossingsPerLine = (
        float(telemetry.crossingsFound / telemetry.linesScanned) if telemetry.linesScanned else 0.0
    )
    telemetry.resolvableSlotsPerLine = int(samples // max(separation, 1))

    # The up directions. With every gate known, silu(g_i.x) is a known function of the probe
    # point and y(x) = sum_i silu(g_i.x) * (d_i (x) u_i) x is linear in the matrices
    # M_i = d_i (x) u_i. One least-squares solve returns them all and each is rank-1, so its
    # SVD splits it into the output direction and the up direction. The gates were the only
    # hard part; this closes the block rather than leaving a caller with two thirds of it.
    ups: list[np.ndarray] = []
    reconstruction: dict[str, float] | None = None
    if gates and up_sample_count > 0:
        gate_matrix = np.stack(gates) @ basis
        sample_rng = np.random.default_rng(seed + 9137)
        samples_in = sample_rng.normal(0.0, up_sample_scale, size=(up_sample_count, input_dim))
        # Chunked. One call for 30000 points at 3072 wide is a 368 MB array through a file
        # handoff, which is where a 64-unit run stopped making progress after the gates were
        # already solved. The chunk size bounds the transfer without changing the solve.
        response_chunks = []
        for start in range(0, up_sample_count, up_request_chunk):
            response_chunks.append(evaluate(samples_in[start:start + up_request_chunk]))
        responses = np.concatenate(response_chunks, axis=0)
        telemetry.oracleCalls += up_sample_count
        telemetry.upSolveCalls = up_sample_count
        # The recovered gate is a direction, so its norm is 1 while the real one is around
        # 1.07 on this layer. `silu` is not scale free, so a wrong norm evaluates the unit in
        # the wrong regime - which is why the up solve returned 14 of 32 before the norm was
        # recovered at all, and did not move when samples, scale or gate quality changed.
        raw_response = samples_in @ gate_matrix.T
        coordinates = samples_in @ basis.T
        # The probe points are not needed past these two products, and at 512 units they are
        # two gigabytes held for the rest of the solve.
        del samples_in
        output_directions = np.stack(outputs)
        # Unmix the output directions before solving, rather than tolerating the overlap.
        # `R = A D` with `A` the activations and `D` the recovered output directions, so
        # projecting on `d_i` returns `sum_j (d_j.d_i) a_j` - the unit plus every neighbour
        # scaled by an inner product that is about `1/sqrt(dim)`. Those errors add in
        # quadrature, so the leak grows as `sqrt(k/dim)`: 0.2 at 128 units in 3072 dimensions,
        # 1.0 at 3072, which is larger than the signal being solved for. Sweeping and
        # subtracting fitted estimates was the previous answer and it injected each
        # neighbour's error while removing its contribution.
        #
        # `D` is already known, so the mixing is a matrix that can be inverted instead:
        # `A = R D^+`. One `k x k` solve removes the whole term at once, exactly, with no
        # iteration and no estimate of anything standing in for a measurement. The pseudo
        # inverse is what handles output directions that are close to collinear, which is the
        # case `identifiability_verdict` reports as ambiguous.
        mixing = np.linalg.pinv(output_directions)
        torch_backend = _torch_cuda()
        if torch_backend is not None:
            unmixed = (
                torch_backend.as_tensor(responses, dtype=torch_backend.float32, device="cuda")
                @ torch_backend.as_tensor(mixing, dtype=torch_backend.float32, device="cuda")
            ).double().cpu().numpy()
        else:
            unmixed = responses @ mixing
        # One ray per unit along its own gate direction, where the norm is what decides when
        # the curve turns on and nothing else can imitate it. See
        # `calibrate_gate_norms_on_rays` for why the isotropic cloud cannot settle this.
        with _Phase(telemetry, "normCalibration"):
            gate_norms, norm_diagnostics = calibrate_gate_norms_on_rays(
                evaluate,
                gate_matrix / np.linalg.norm(gate_matrix, axis=1, keepdims=True),
                output_directions,
                points_per_ray=norm_ray_points,
                min_magnitude=norm_ray_min,
                max_magnitude=norm_ray_max,
                offset_scale=norm_ray_offset,
                request_chunk=up_request_chunk,
                repeats=norm_ray_repeats,
                seed=seed + 4451,
            )
            telemetry.oracleCalls += len(gates) * (norm_ray_points // 2) * 2 * norm_ray_repeats
        # Which way the unit fires. The gate arrives as a null-space vector, so its sign is
        # free, and every direction measure in this file takes an absolute cosine and cannot
        # see it. That is not cosmetic: `silu` is flat on one side of the knee and linear on
        # the other, so a flipped gate describes a unit that fires exactly when the real one
        # is silent. It passed every cosine bar on 25 of 64 units of layer 15 and then ranked
        # inputs backwards - rank correlation -0.74 against the real unit, with the predicted
        # optimum landing where the real activation is 2e-5 of its peak.
        #
        # It is decided here rather than on the calibration ray because the ray cannot decide
        # it. `silu(z) - silu(-z) = z`, so along a ray the two orientations differ by a term
        # linear in `s`, which is precisely what the background absorbs. Over the full input
        # space that same difference is `(g_i.x)(u_i.x)`, a quadratic form no column of this
        # design can take, so the residual separates them.
        with _Phase(telemetry, "signDecision"):
            sign_rows = min(up_sample_count, 6000)
            signs = _decide_gate_signs(
                raw_response[:sign_rows],
                unmixed[:sign_rows],
                coordinates[:sign_rows],
                gate_norms,
            )
        gates = [row * sign for row, sign in zip(gates, signs)]
        gate_matrix = gate_matrix * signs[:, None]
        raw_response = raw_response * signs[None, :]
        gate_response = raw_response * gate_norms[None, :]
        smooth = gate_response / (1.0 + np.exp(-np.clip(gate_response, -60.0, 60.0)))
        # Solved in basis coordinates, not in the input space. `u_i` lies in the active
        # subspace by construction, so it has `active_rank` free parameters rather than
        # `input_dim`: for 32 units in 3072 dimensions that is 2048 unknowns instead of
        # 98304, and 6000 samples go from hopelessly underdetermined - which returned zero
        # usable up directions on the live model - to comfortably overdetermined.
        # Per unit, through its own output direction, rather than one joint solve over all of
        # them. Projecting the response onto `d_i` leaves that unit's contribution with
        # coefficient 1 and everyone else's scaled by `d_j . d_i`, which for directions in
        # 3072 dimensions is around 0.018. A joint solve instead lets a small error in one
        # gate be absorbed by another unit's column: with gates at per-unit cosine 0.9985 it
        # returned 15 of 32 up directions above 0.99 and adding samples did not move it,
        # because the limit was cross-talk and not conditioning.
        # Cross-talk is what limits this, not conditioning. Projecting onto `d_i` leaves every
        # other unit scaled by `d_j . d_i`, around 0.018 for directions in 3072 dimensions,
        # and those errors add: at 32 units the residual noise is about 0.10 of the signal and
        # every up direction came back above 0.99, at 128 units it is 0.20 and 22 of them did
        # not. Sweeping repeatedly and subtracting the units already solved removes that term
        # instead of tolerating it - each pass sees a cleaner residual than the last.
        # Subtracting the units already solved was tried and makes things worse: 64 units went
        # from 64/64 up directions to 60/64 with two sweeps. The cross-talk term is real, but
        # so is the error in every not-yet-correct neighbour, and subtracting an estimate
        # injects the second while removing the first. Default is zero sweeps, which is the
        # plain per-unit solve; the option stays because at some unit count the trade may
        # reverse, and because a measured dead end is worth keeping runnable.
        #
        # The running model is kept and adjusted rather than rebuilt: recomputing every other
        # unit's contribution inside the loop is quadratic in the unit count and made a
        # 128-unit sweep slower than the entire oracle phase.
        with _Phase(telemetry, "upSolve"):
            solved_coefficients = [np.zeros(active_rank) for _ in gates]
            activations = [np.zeros(up_sample_count) for _ in gates]
            # Only the refinement sweeps need a running model, and it is the same size as the
            # whole response block - two gigabytes at 512 units that the default path never reads.
            explained = np.zeros_like(responses) if up_refinement_sweeps > 0 else None
            # One design buffer for the whole loop. Only the gated half changes from unit to unit,
            # and rebuilding both halves each time allocates `rows x 2*active_rank` plus a
            # same-sized temporary for the product - 5 GB per unit at 1024 units, a thousand times
            # over. The ungated half is written once.
            design = np.empty((up_sample_count, 2 * active_rank), dtype=np.float64)
            design[:, active_rank:] = coordinates
            for _sweep in range(max(1, up_refinement_sweeps)):
                subtract = up_refinement_sweeps > 0
                for index in range(len(gates)):
                    if subtract:
                        explained -= np.outer(activations[index], output_directions[index])
                    projected = (unmixed[:, index] if not subtract
                                 else (responses - explained) @ output_directions[index])
                    # `silu` does not average to zero, and that is what leaks the neighbours into
                    # `u_i`. Every other unit enters the projection as
                    # `(d_j.d_i) silu(z_j)(u_j.x)`, whose leading term is
                    # `(d_j.d_i) E[silu] (u_j.x)` - a plain linear function of `x` - and the
                    # regressor `silu(z_i) x` carries the same non-zero mean, so the solve pays
                    # for that overlap out of `u_i`. Carrying an ungated linear block absorbs it
                    # instead. This stayed hidden while the gate norms were wrong: a norm near
                    # zero makes `silu(c z) ~ c z / 2`, which is mean-zero and has no overlap to
                    # leak, so the up solve scored 64 of 64 on a model that was wrong about every
                    # unit's scale. With the norms measured, the same solve returned 39.
                    np.multiply(coordinates, smooth[:, index][:, None], out=design[:, :active_rank])
                    solution = solve_least_squares(design, projected)[:active_rank]
                    solved_coefficients[index] = solution
                    if subtract:
                        activations[index] = smooth[:, index] * (coordinates @ solution)
                        explained += np.outer(activations[index], output_directions[index])
            for index in range(len(gates)):
                solution = solved_coefficients[index]
                norm = float(np.linalg.norm(solution))
                ups.append((solution / norm if norm > 0 else solution) @ basis)
            telemetry.upsRecovered = len(ups)

        # What the recovery is worth without the answer key. Everything else this file reports
        # per unit is a cosine against the model's own weights, which is a post-hoc witness a
        # caller in the black-box setting does not have. The recovered block is a complete
        # function, so it can be run on points it was never fitted on and compared against the
        # oracle - and that single number moves for every failure at once: a unit that was
        # missed, a norm at the wrong scale, a gate pointing the wrong way, an up direction
        # bent by a neighbour. It is the only measure here that a user without weights gets.
        with _Phase(telemetry, "reconstruction"):
            held_out = np.random.default_rng(seed + 8821).normal(
                0.0, up_sample_scale, size=(max(reconstruction_points, 0), input_dim)
            )
            if held_out.shape[0]:
                truth_chunks = [
                    evaluate(held_out[start:start + up_request_chunk])
                    for start in range(0, held_out.shape[0], up_request_chunk)
                ]
                observed = np.concatenate(truth_chunks, axis=0)
                telemetry.oracleCalls += held_out.shape[0]
                held_gate = (held_out @ gate_matrix.T) * gate_norms[None, :]
                held_smooth = held_gate / (1.0 + np.exp(-np.clip(held_gate, -60.0, 60.0)))
                held_coordinates = held_out @ basis.T
                # One product, not one per unit. `sum_i a_i (x) d_i` is `A D` with `A` the
                # activations and `D` the output directions, and writing it as an accumulation
                # of outer products allocates a full response block per unit - 98 MB times the
                # unit count. Measured at 111 seconds of CPU against 4 of wall clock, 26 cores
                # held, for arithmetic one matrix multiply does in a moment.
                coefficients = np.stack(solved_coefficients)
                predicted = (held_smooth * (held_coordinates @ coefficients.T)) @ output_directions
                residual = float(np.sum((observed - predicted) ** 2))
                total = float(np.sum((observed - observed.mean(axis=0)) ** 2))
                reconstruction = {
                    "points": int(held_out.shape[0]),
                    "rSquared": float(1.0 - residual / total) if total > 0 else 0.0,
                    "relativeResidual": float(
                        np.linalg.norm(observed - predicted) / max(np.linalg.norm(observed), 1e-30)
                    ),
                }
        telemetry.upsRecovered = len(ups)

    solved_gates = np.stack(gates) if gates else np.zeros((0, active_rank))
    solved_outputs = np.stack(outputs) if outputs else np.zeros((0, 1))
    verdict = identifiability_verdict(
        basis_spectrum=spectrum,
        active_rank=active_rank,
        gates=solved_gates,
        output_directions=solved_outputs,
        clusters_found=len(clusters),
        duplicate_gate_cosine=duplicate_gate_cosine,
        collinear_threshold=collinear_threshold,
        weak_share_threshold=weak_share_threshold,
        rank_energy_threshold=rank_energy_threshold,
        unit_budget=unit_budget,
        norm_diagnostics=norm_diagnostics if gates and up_sample_count > 0 else None,
    )
    telemetry.notes.append(f"identifiability verdict: {verdict['verdict']}")

    return {
        "identifiability": verdict,
        "reconstruction": reconstruction,
        "basis": basis,
        "upDirections": np.stack(ups) if ups else np.zeros((0, input_dim)),
        "gateNorms": gate_norms if gates and up_sample_count > 0 else np.zeros(0),
        "gateNormDiagnostics": norm_diagnostics if gates and up_sample_count > 0 else [],
        "basisSpectrum": spectrum,
        "gates": np.stack(gates) if gates else np.zeros((0, active_rank)),
        "outputDirections": np.stack(outputs) if outputs else np.zeros((0, 1)),
        "gateDeterminations": determinations,
        "clusterSizes": [len(cluster) for cluster in sorted(clusters, key=len, reverse=True)],
    }, telemetry


def count_fully_recovered(
    *,
    recovered_gates: np.ndarray,
    truth_gates: np.ndarray,
    recovered_ups: np.ndarray,
    truth_ups: np.ndarray,
    gate_norms: np.ndarray,
    recovered_outputs: np.ndarray | None = None,
    truth_outputs: np.ndarray | None = None,
) -> int:
    """Units whose gate direction, up direction and gate norm all came back, matched to one source.

    Lifted out of the reporting flow so it can be tested at all. It was written inline, which
    meant the measure that certifies every recovery in this route had no test of its own, and
    the defect below survived a day of work because nothing could reach it.

    The gate cosine is **signed**. `silu` is flat on one side of the knee and linear on the
    other, so a unit whose gate points the wrong way fires exactly when the real one is silent.
    The recovery was taught to get that right earlier the same day - 39 of 64 to 64 of 64 - and
    this measure, which signs off on the result, was still taking an absolute value and could
    not tell the two apart.

    The up cosine is **not** signed, and that is physics rather than laxity: a unit contributes
    `d_i silu(g_i.x)(u_i.x)`, so flipping `u_i` and `d_i` together leaves it unchanged. The
    sign of `u_i` alone is a gauge choice and nothing can observe it. What is observable is
    that the two flip together, so when the output directions are available the check is that
    `sign(u.u_true) == sign(d.d_true)`.
    """
    def unit(rows: np.ndarray) -> np.ndarray:
        return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-30)

    gate_cosine = unit(recovered_gates) @ unit(truth_gates).T
    up_cosine = unit(recovered_ups) @ unit(truth_ups).T
    assigned = np.abs(gate_cosine).argmax(axis=1)
    truth_norms = np.linalg.norm(truth_gates, axis=1)

    pair_ok = None
    if recovered_outputs is not None and truth_outputs is not None:
        pair_ok = np.sign(up_cosine) == np.sign(unit(recovered_outputs) @ unit(truth_outputs).T)

    counted: set[int] = set()
    for solution in range(recovered_gates.shape[0]):
        source = int(assigned[solution])
        if gate_cosine[solution, source] <= 0.99:
            continue
        if abs(up_cosine[solution, source]) <= 0.99:
            continue
        if abs(gate_norms[solution] / truth_norms[source] - 1.0) > 0.1:
            continue
        if pair_ok is not None and not bool(pair_ok[solution, source]):
            continue
        counted.add(source)
    return len(counted)


def summarise(result: dict[str, Any], telemetry: RecoveryTelemetry) -> str:
    return json.dumps({
        "oracleCalls": telemetry.oracleCalls,
        "basisCalls": telemetry.basisCalls,
        "scanCalls": telemetry.scanCalls,
        "linesScanned": telemetry.linesScanned,
        "crossingsFound": telemetry.crossingsFound,
        "clusters": telemetry.clusters,
        "unitsRecovered": telemetry.unitsRecovered,
        "amplitudeCalls": telemetry.amplitudeCalls,
        "phaseSeconds": telemetry.phaseSeconds,
        "amplitudeSweeps": telemetry.amplitudeSweeps,
        "amplitudeSweepsAllowed": telemetry.amplitudeSweepsAllowed,
        "crossingsPerLine": telemetry.crossingsPerLine,
        "resolvableSlotsPerLine": telemetry.resolvableSlotsPerLine,
        "duplicatesDropped": telemetry.duplicatesDropped,
        "clustersSkippedAsSmall": telemetry.clustersSkippedAsSmall,
        "rejectedByResidual": telemetry.rejectedByResidual,
        "upSolveCalls": telemetry.upSolveCalls,
        "upsRecovered": telemetry.upsRecovered,
        "clusterSizes": result["clusterSizes"][:16],
        "notes": telemetry.notes,
    }, indent=2)


def gate_from_peak_amplitudes(
    evaluate: Callable[[np.ndarray], np.ndarray],
    *,
    peak_point: np.ndarray,
    step_along_line: np.ndarray,
    output_direction: np.ndarray,
    probe_basis: np.ndarray,
    step: float,
    request_chunk: int = 4096,
) -> tuple[np.ndarray, float, int]:
    """Read a gate direction off the amplitudes at one crossing, not from many crossings.

    The null-space route needs `rank` points on a unit's hyperplane, so a full 8192-unit layer
    in 3072 dimensions would need 3072 crossings of every single unit. This needs one.

    At a crossing the Jacobian jumps by `(u_i.x) * d_i (x) g_i`. Probing it along `v` gives
    `c * (g_i.v) * d_i`, so projecting onto the already-known output direction leaves the
    scalar `c * (g_i.v)`. Sweeping `v` over a basis returns `c * g_i` component by component,
    and normalising drops the unknown `c`. Every other factor - the transition's second
    derivative, `u_i.x`, the line's own direction - is identical across the sweep because it
    is the same crossing, which is exactly why the ratios are clean.

    Cost is `8` calls per basis vector: two points a side of the crossing, extrapolated so the
    other units' smooth drift cancels, the same correction that took the rank-1 share from
    0.14 to 0.54 elsewhere in this route.

    Returns the direction, the share of the sweep's energy that lay along `output_direction`
    as a confidence signal, and the calls spent.
    """
    probes = np.atleast_2d(probe_basis)
    left_near = peak_point - 0.5 * step_along_line
    left_far = peak_point - step_along_line
    right_near = peak_point + 0.5 * step_along_line
    right_far = peak_point + step_along_line

    # One batch for the whole sweep, not one per probe direction. The arithmetic is the same
    # either way; the difference is round trips. Probing a rank-256 basis one direction at a
    # time is 2048 separate oracle requests per crossing, each with its own file write and
    # subprocess exchange, which is what made a 128-unit run take over half an hour of
    # overhead rather than of computation.
    anchors = np.stack([left_near, left_far, right_near, right_far])
    offsets = np.concatenate([
        anchors[None, :, :] + step * probes[:, None, :],
        anchors[None, :, :] - step * probes[:, None, :],
    ], axis=1).reshape(-1, probes.shape[1])
    # Chunked for the same reason the up solve is: batching removes round trips, but an
    # unbounded batch turns into a multi-hundred-megabyte file handoff per crossing.
    chunks = [evaluate(offsets[start:start + request_chunk]) for start in range(0, offsets.shape[0], request_chunk)]
    values = np.concatenate(chunks, axis=0)
    calls = offsets.shape[0]
    reshaped = values.reshape(probes.shape[0], 8, -1)
    derivatives = (reshaped[:, :4, :] - reshaped[:, 4:, :]) / (2.0 * step)
    jumps = (2.0 * derivatives[:, 2, :] - derivatives[:, 3, :]) - (
        2.0 * derivatives[:, 0, :] - derivatives[:, 1, :]
    )
    components = jumps @ output_direction
    parallel = np.abs(components)
    total = np.linalg.norm(jumps, axis=1)

    norm = float(np.linalg.norm(components))
    if norm <= 0.0:
        return np.zeros(probes.shape[0]), 0.0, calls
    alignment = float(np.sum(parallel) / max(np.sum(total), 1e-30))
    return components / norm, alignment, calls


def _oracle_namespace(
    *,
    model: str,
    layer: int,
    units: str,
    device: str,
    max_oracle_calls: int,
    batch_size: int,
) -> Any:
    """The fields `OracleClient` reads, and nothing else.

    The decompiler's own namespace carries well over a hundred settings for a solver this
    module does not use. Constructing only what the transport needs keeps it obvious that no
    fitting knob reaches the oracle, and that the access path is the same one every other
    lane uses.
    """
    import argparse as _argparse

    return _argparse.Namespace(
        model=model,
        layer=int(layer),
        unit=int(units.split(",")[0]),
        units=units,
        device=device,
        oracle_batch_size=int(batch_size),
        max_oracle_calls=int(max_oracle_calls),
        output_channel=-1,
        oracle_output_mode="selected-unit-ffn-vector",
        oracle_quantization_step=0.0,
        local_files_only=True,
        oracle_read_fd=-1,
        oracle_write_fd=-1,
        oracle_ready_mode="full",
        oracle_eval_only=False,
        defer_posthoc_truth=False,
        require_opaque_oracle=False,
        expected_dimension=0,
        oracle_manifest_fd=-1,
    )


def main() -> int:
    import argparse
    import tempfile
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Recover a SwiGLU block from boundary crossings.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--units", required=True, help="comma- or plus-separated unit indices")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-oracle-calls", type=int, default=400000)
    parser.add_argument("--oracle-batch-size", type=int, default=512)
    parser.add_argument("--line-count", type=int, default=120)
    parser.add_argument("--samples", type=int, default=1001)
    parser.add_argument("--scale", type=float, default=2e4)
    parser.add_argument("--step", type=float, default=1e-4)
    parser.add_argument("--peak-fraction", type=float, default=0.05)
    parser.add_argument("--separation", type=int, default=6)
    parser.add_argument("--min-sharpness", type=float, default=3.0)
    parser.add_argument("--cluster-min-cosine", type=float, default=0.99)
    parser.add_argument("--basis-points", type=int, default=8)
    parser.add_argument("--basis-scale", type=float, default=0.0, help="0 reuses --scale")
    parser.add_argument("--gate-mode", choices=["null-space", "amplitude"], default="amplitude")
    parser.add_argument("--amplitude-span", type=float, default=0.01)
    parser.add_argument("--amplitude-min-alignment", type=float, default=0.5)
    parser.add_argument("--duplicate-gate-cosine", type=float, default=0.98)
    parser.add_argument("--gate-residual-tolerance", type=float, default=1e-3)
    parser.add_argument("--amplitude-crossings", type=int, default=4)
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--progress", action="store_true")
    # Measured on 64 units: 30000 samples give 64/64 up directions, 20000 give 58/64. The
    # per-unit solve is well conditioned either way - 256 unknowns against either count - so
    # what improves is the signal against the cross-talk from the other units, which falls as
    # the square root of the sample count. Budget accordingly rather than by conditioning.
    parser.add_argument("--up-sample-count", type=int, default=30000)
    parser.add_argument("--up-sample-scale", type=float, default=0.6)
    parser.add_argument("--up-request-chunk", type=int, default=4096)
    parser.add_argument("--gate-stability-cosine", type=float, default=0.99999)
    parser.add_argument("--norm-ray-points", type=int, default=192)
    parser.add_argument("--reconstruction-points", type=int, default=4000)
    parser.add_argument("--collinear-threshold", type=float, default=0.9)
    parser.add_argument("--weak-share-threshold", type=float, default=0.002)
    parser.add_argument("--rank-energy-threshold", type=float, default=0.98)
    parser.add_argument("--unit-budget", type=int, default=0, help="0 uses the requested unit count")
    parser.add_argument("--up-refinement-sweeps", type=int, default=0,
                        help="subtract already-solved units before each solve; measured worse, kept runnable")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-json", default="")
    parser.add_argument(
        "--output-directions-npz",
        default="",
        help="write recovered gate and output directions, so a claim about them can be checked",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.llm_reader.blind_swiglu_oracle_decompiler import OracleClient

    # The oracle protocol takes a comma-separated list; accepting `+` too keeps this usable
    # from a shell that would otherwise need the list quoted.
    unit_list = [int(value) for value in args.units.replace("+", ",").split(",") if value.strip()]
    unit_spec = ",".join(str(value) for value in unit_list)
    with tempfile.TemporaryDirectory(prefix="gcl-jacobian-recovery-") as work:
        oracle = OracleClient(
            _oracle_namespace(
                model=args.model,
                layer=args.layer,
                units=unit_spec,
                device=args.device,
                max_oracle_calls=args.max_oracle_calls,
                batch_size=args.oracle_batch_size,
            ),
            Path(work),
        )
        try:
            width = int(oracle.meta.get("oracleResponseWidth", 1))
            if width <= 1:
                raise RuntimeError(
                    "this recovery needs a vector-valued oracle; "
                    f"the one that answered reports response width {width}"
                )
            dimension = int(oracle.meta["hiddenSize"])
            result, telemetry = recover_block(
                lambda points: oracle.eval(points.astype(np.float32), stage="jacobian-boundary-recovery"),
                input_dim=dimension,
                active_rank=2 * len(unit_list),
                line_count=args.line_count,
                samples=args.samples,
                scale=args.scale,
                step=args.step,
                peak_fraction=args.peak_fraction,
                separation=args.separation,
                min_sharpness=args.min_sharpness,
                cluster_min_cosine=args.cluster_min_cosine,
                basis_points_count=args.basis_points,
                basis_scale=args.basis_scale if args.basis_scale > 0 else args.scale,
                gate_mode=args.gate_mode,
                amplitude_span=args.amplitude_span,
                amplitude_min_alignment=args.amplitude_min_alignment,
                duplicate_gate_cosine=args.duplicate_gate_cosine,
                gate_residual_tolerance=args.gate_residual_tolerance,
                amplitude_crossings=args.amplitude_crossings,
                min_cluster_size=args.min_cluster_size,
                progress=args.progress,
                up_sample_count=args.up_sample_count,
                up_sample_scale=args.up_sample_scale,
                up_request_chunk=args.up_request_chunk,
                gate_stability_cosine=args.gate_stability_cosine,
                norm_ray_points=args.norm_ray_points,
                reconstruction_points=args.reconstruction_points,
                up_refinement_sweeps=args.up_refinement_sweeps,
                collinear_threshold=args.collinear_threshold,
                weak_share_threshold=args.weak_share_threshold,
                rank_energy_threshold=args.rank_energy_threshold,
                unit_budget=args.unit_budget if args.unit_budget > 0 else len(unit_list),
                seed=args.seed,
            )
            report: dict[str, Any] = {
                # First, because a report whose structure claim is not allowed must not be
                # read metric-first. The formalization's fourth outcome - inventing
                # unsupported structure - is exactly what a reader does when the numbers
                # come before the verdict.
                "identifiability": result["identifiability"],
                # Second, because it is the only number here a caller without the weights can
                # act on. Everything under `posthoc` is measured against the model's own
                # parameters and exists to validate the method, not to be available in use.
                "reconstruction": result["reconstruction"],
                "model": args.model,
                "layer": args.layer,
                "units": unit_list,
                "oracleResponseWidth": width,
                "telemetry": json.loads(summarise(result, telemetry)),
                "accessAudit": {
                    "solverAccess": "oracle-only",
                    "weightsUsedBySolver": False,
                    "hooksUsedBySolver": False,
                    "postHocWitnessOnly": True,
                },
            }

            # Post-hoc witness, read only after the recovery above is complete and frozen.
            oracle.unlock_posthoc_truth("jacobian-boundary-recovery-frozen")
            truth_gate, truth_up = oracle.truth()
            basis = result["basis"]
            recovered = result["gates"] @ basis
            if recovered.shape[0]:
                recovered = recovered / np.linalg.norm(recovered, axis=1, keepdims=True)
                reference = truth_gate / np.linalg.norm(truth_gate, axis=1, keepdims=True)
                similarity = np.abs(recovered @ reference.T)
                best = similarity.max(axis=1)
                # Per source unit, not per solution. `distinctUnitsMatched` counts how many
                # units are some solution's nearest neighbour, which is not the same as how
                # many units were actually recovered: a bad solution still has a nearest
                # neighbour, and reporting the two together reads as though every unit came
                # back clean when some may be covered only by a weak match.
                per_unit_best = similarity.max(axis=0)
                truth_down = oracle.truth_down()
                up_per_unit = None
                joint_per_unit = None
                if result["upDirections"].shape[0]:
                    reference_up = truth_up / np.linalg.norm(truth_up, axis=1, keepdims=True)
                    recovered_up = result["upDirections"]
                    recovered_up = recovered_up / np.linalg.norm(recovered_up, axis=1, keepdims=True)
                    up_similarity = np.abs(recovered_up @ reference_up.T)
                    up_per_unit = up_similarity.max(axis=0)
                    # The joint bar the fitted lanes report: a unit counts only when its gate
                    # and its up direction are both right, matched to the same source.
                    joint = np.minimum(similarity, up_similarity)
                    joint_per_unit = joint.max(axis=0)
                # Directions alone are not the unit. `silu` is not scale free, so a solution
                # whose gate points the right way with the wrong norm evaluates the unit in
                # the wrong regime, and a norm near zero describes a unit that never fires.
                # The joint direction bar reported 64 of 64 on a run where 25 gate norms were
                # wrong by up to four orders of magnitude - the same shape of mistake as
                # `distinctUnitsMatched`, a measure a bad solution also satisfies. This one
                # requires the gate direction, the up direction and the norm together, matched
                # to one source unit.
                fully_recovered = None
                norms = result.get("gateNorms", np.zeros(0))
                if up_per_unit is not None and norms.shape[0] == recovered.shape[0]:
                    fully_recovered = count_fully_recovered(
                        recovered_gates=recovered,
                        truth_gates=truth_gate,
                        recovered_ups=result["upDirections"],
                        truth_ups=truth_up,
                        gate_norms=norms,
                        recovered_outputs=result["outputDirections"] if truth_down is not None else None,
                        truth_outputs=truth_down.T if truth_down is not None else None,
                    )
                output_per_unit = None
                output_per_solution = None
                if truth_down is not None and result["outputDirections"].shape[0]:
                    reference_down = truth_down / np.linalg.norm(truth_down, axis=0, keepdims=True)
                    output_similarity = np.abs(result["outputDirections"] @ reference_down)
                    output_per_unit = output_similarity.max(axis=0)
                    output_per_solution = output_similarity.max(axis=1)
                report["posthoc"] = {
                    # Per source unit. This is the question - was the unit recovered - and it
                    # is not what `distinctUnitsMatched` answers: a solution that matches
                    # nothing well still has a nearest unit, so that count can read 32 of 32
                    # while units are covered only by weak matches.
                    "unitsWithGateAbove099": int(np.sum(per_unit_best > 0.99)),
                    "perUnitGateCosineMin": float(np.min(per_unit_best)),
                    "perUnitGateCosineMedian": float(np.median(per_unit_best)),
                    "unitsWithUpAbove099": None if up_per_unit is None else int(np.sum(up_per_unit > 0.99)),
                    "perUnitUpCosineMin": None if up_per_unit is None else float(np.min(up_per_unit)),
                    "unitsFullyRecovered": fully_recovered,
                    "unitsWithJointAbove099": None if joint_per_unit is None else int(np.sum(joint_per_unit > 0.99)),
                    "perUnitJointCosineMin": None if joint_per_unit is None else float(np.min(joint_per_unit)),
                    "unitsWithOutputAbove099": None if output_per_unit is None else int(np.sum(output_per_unit > 0.99)),
                    "perUnitOutputCosineMin": None if output_per_unit is None else float(np.min(output_per_unit)),
                    "outputCosineMedian": None if output_per_solution is None else float(np.median(output_per_solution)),
                    # Solutions that match no unit well. They are in the returned set and a
                    # caller has to drop them, so they are counted here rather than left out
                    # of a summary that would otherwise read as a clean recovery.
                    "solutionCount": int(recovered.shape[0]),
                    "spuriousSolutions": int(np.sum(best <= 0.99)),
                    "gateCosineMedian": float(np.median(best)),
                    "gateCosineMin": float(np.min(best)),
                    "gateCosineShareAbove099": float(np.mean(best > 0.99)),
                    "distinctUnitsMatchedNotProofOfRecovery": int(len(set(similarity.argmax(axis=1).tolist()))),
                    "unitCount": len(unit_list),
                    "basisGateEnergy": float(np.mean([
                        np.linalg.norm(basis @ row) / np.linalg.norm(row) for row in truth_gate
                    ])),
                    "basisUpEnergy": float(np.mean([
                        np.linalg.norm(basis @ row) / np.linalg.norm(row) for row in truth_up
                    ])),
                }
            else:
                report["posthoc"] = {"gateCosineMedian": None, "unitCount": len(unit_list)}

            if args.output_directions_npz:
                # Without this the output directions exist only inside the run. Claiming the
                # route "returns d_i" while the artifact holds no d_i to check is a claim
                # about something nobody can inspect.
                Path(args.output_directions_npz).parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    args.output_directions_npz,
                    gates_in_basis=result["gates"],
                    basis=result["basis"],
                    output_directions=result["outputDirections"],
                    up_directions=result["upDirections"],
                    gate_norms=result.get("gateNorms", np.zeros(0)),
                    units=np.asarray(unit_list),
                )

            rendered = json.dumps(report, indent=2)
            print(rendered)
            if args.output_json:
                Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output_json).write_text(f"{rendered}\n", encoding="utf-8")
        finally:
            oracle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
