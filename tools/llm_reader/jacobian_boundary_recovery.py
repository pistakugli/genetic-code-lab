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
# An explicit setting from the caller wins; this only fills in a default. It also only takes
# effect when this module is the entry point, because the pools are sized when numpy imports:
# a test file that imports numpy first gets the machine default, and the commit that added
# this said otherwise. Audit finding F16.
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
    lineScanBreakdown: dict[str, float] = field(default_factory=dict)
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
        if torch is not None:
            # Differenced on the card. The oracle answers `input_dim x output_dim` per point -
            # 3072 by 3072 - and building the difference, dividing it and transposing it on the
            # host is three arrays of 75 MB in double before the card sees any of it.
            left = torch.as_tensor(plus, device="cuda").double()
            right = torch.as_tensor(minus, device="cuda").double()
            block = ((left - right) / (2.0 * step)).T
            gram += block.T @ block
            del left, right, block
        else:
            jacobian = ((plus - minus) / (2.0 * step)).T
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
    peak_floor_mode: str = "max",
    sharpness_mode: str = "ratio",
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
    # Built on the card. The line is `samples x rank` against `rank x output_dim` and the two
    # probe offsets are copies of it, which measured as the largest single piece of host work
    # left in the route: 3.3 seconds of the line scan's 12.4 on a 128-unit block, at six and a
    # half cores.
    #
    # Double, and the first attempt at single is why this comment exists. The line carries
    # values of order `scale` - 2e4 - while the probe offset is of order `step`. In single
    # precision a small step falls below the resolution of the number it is added to, the two
    # sides of the difference become the same array, and the derivative the whole route is
    # built on becomes noise: gate cosine 0.38. That is the third time today that single
    # precision in the wrong place has silently destroyed a recovery, after the basis Gram and
    # the scan difference, and the pattern is always the same - it is fine where quantities are
    # compared and fatal where they are subtracted.
    torch_backend = _torch_cuda()
    if torch_backend is not None:
        device_basis = torch_backend.as_tensor(basis, dtype=torch_backend.float64, device="cuda")
        device_line = scale * (
            torch_backend.as_tensor(basis_points, dtype=torch_backend.float64, device="cuda") @ device_basis
        )
        device_probe = torch_backend.as_tensor(probe, dtype=torch_backend.float64, device="cuda") @ device_basis
        line = None
        probe_vector = device_probe.cpu().numpy()
    else:
        line = scale * (basis_points @ basis)
        probe_vector = probe @ basis

    built = time.perf_counter()
    if line is None:
        offset = step * device_probe
        plus = (device_line + offset).cpu().numpy()
        minus = (device_line - offset).cpu().numpy()
        del device_line, device_probe, device_basis
    else:
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

    # The pre-filter that decides which samples are even considered as peaks. It used to be a
    # fraction of the single largest turning magnitude on the line, which is a constant compared
    # against a quantity that grows with the block: one unit with a large `(u_i.x)` sets the
    # scale, and every other crossing on that line is measured against it. The bigger the block,
    # the likelier such an outlier, so the floor climbs and ordinary crossings drop out.
    #
    # Measured across the ladder, crossings found per line: 19.4 at 64 units, 47.6 at 512, 36.3
    # at 1024, 14.1 at 2048, 4.8 at 4096 - falling where more boundaries should give more, with
    # occupancy of the resolvable slots down to 3.6%. The crossings were not colliding; they
    # were being thrown away before anything looked at them. That is why 4096 returned 962
    # solutions and 2048 missed 460 units. Same shape as F31, where an absolute floor was
    # compared against a share that goes as 1/k. Audit finding F34.
    #
    # The median of the line's own turning magnitude is a background that one extreme peak does
    # not move. It carries no multiplier, so no new cutoff is introduced, and the real filter is
    # unchanged - the per-candidate sharpness test below, which compares each peak against its
    # own local neighbourhood rather than against the line's loudest event.
    if peak_floor_mode == "background":
        floor = float(np.median(magnitude))
    else:
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
        # A peak that barely stands out from its own neighbourhood is a crossing read through
        # someone else's transition, and its direction is a mixture rather than one unit's.
        #
        # How far it has to stand out is the question, and `peak / background >= 3` was the
        # wrong shape for it. The background is the incoherent sum of every other unit's smooth
        # contribution, so it grows about as `sqrt(k)` while one unit's jump does not, and a
        # fixed ratio therefore demands more of a crossing the larger the block gets. Measured
        # on the 2048-unit block at 40 lines: the ratio at 3.0 keeps 14.2 crossings per line,
        # at 1.5 keeps 56.4, at 1.05 keeps 74.0. Three quarters of the real crossings were being
        # discarded before anything looked at them, which is why 2048 missed 460 units. Audit
        # finding F34, the same shape as F31.
        #
        # The distance of the peak from its own neighbourhood measured in that neighbourhood's
        # own robust spread does not carry that bias: the numerator and the denominator grow
        # together, so the same number means the same selectivity at every block size. The
        # median absolute deviation is used rather than a standard deviation because the window
        # contains the peak being tested and its shoulders.
        if sharpness_mode == "sigma":
            spread = float(np.median(np.abs(window - background))) * 1.4826
            sharpness = (
                float((magnitude[index] - background) / spread) if spread > 0 else float("inf")
            )
        else:
            sharpness = float(magnitude[index] / background) if background > 0 else 0.0
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
    chunk: int = 4096,
) -> list[list[BoundaryCrossing]]:
    """Group crossings that belong to the same unit.

    Two crossings of the same boundary, on different lines, hand back the same output
    direction up to sign. Nothing else in the block shares it, which is what makes the output
    direction usable as an identity where a scalar oracle offers no identity at all.

    This was the wall between the route and a whole layer - not the oracle, not the algebra.
    As a pair of Python loops it measured 0.7 seconds at 128 units and 3129.8 at 1024, which
    is 62% of that entire run and a growth of 4471x for an eightfold block, because the first
    pass is `O(crossings x clusters)` and the second restarts its `O(clusters^2)` scan after
    every merge. A full layer would have been days of it.

    Two changes, and the second is the one that matters. The comparisons move to the card,
    where they are matrix products against the whole centroid block rather than a walk over
    it. And the loop that stays in Python runs over *clusters* rather than crossings: a chunk
    of crossings is matched against every existing centroid in one product, and only the ones
    that match nothing - which is at most one per unit - need sequential treatment. Python
    work drops from `O(crossings)` to `O(clusters)`.

    The merge pass is not sequential at all. It is connected components over the centroid
    cosine graph, blocked on the card and resolved by union-find, which also removes the
    rescan-from-scratch behaviour that turned the quadratic into a cubic.
    """
    if not crossings:
        return []
    ordered = sorted(crossings, key=lambda item: -item.peak_sharpness)
    stacked = np.stack([item.output_direction for item in ordered]).astype(np.float32)
    total, width = stacked.shape

    torch = _torch_cuda()
    if torch is not None:
        device_points = torch.as_tensor(stacked, device="cuda")
        centroids = torch.zeros((max(1024, total // 4), width), dtype=torch.float32, device="cuda")
    else:
        device_points = stacked
        centroids = np.zeros((max(1024, total // 4), width), dtype=np.float32)

    sizes: list[int] = []
    members: list[list[BoundaryCrossing]] = []
    count = 0

    def grow(needed: int):
        nonlocal centroids
        if needed <= centroids.shape[0]:
            return
        capacity = centroids.shape[0]
        while capacity < needed:
            capacity *= 2
        if torch is not None:
            grown = torch.zeros((capacity, width), dtype=torch.float32, device="cuda")
        else:
            grown = np.zeros((capacity, width), dtype=np.float32)
        grown[:count] = centroids[:count]
        centroids = grown

    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        block = device_points[start:stop]
        if count:
            overlap = block @ centroids[:count].T
            magnitude = overlap.abs()
            if torch is not None:
                best_value, best_index = magnitude.max(dim=1)
                best_value = best_value.cpu().numpy()
                best_index = best_index.cpu().numpy()
                signs = torch.sign(overlap.gather(1, torch.as_tensor(best_index, device="cuda").unsqueeze(1))).squeeze(1).cpu().numpy()
            else:
                best_index = magnitude.argmax(axis=1)
                best_value = magnitude[np.arange(magnitude.shape[0]), best_index]
                signs = np.sign(overlap[np.arange(overlap.shape[0]), best_index])
        else:
            best_value = np.zeros(stop - start)
            best_index = np.zeros(stop - start, dtype=int)
            signs = np.ones(stop - start)

        # Everything that already has a home is placed without touching Python's opinion of
        # geometry; only the strays below go one at a time, and there is at most one stray per
        # unit in the whole run.
        for offset in range(stop - start):
            crossing = ordered[start + offset]
            if count and best_value[offset] >= min_cosine:
                home = int(best_index[offset])
                sign = 1.0 if signs[offset] >= 0 else -1.0
                members[home].append(crossing)
                sizes[home] += 1
                updated = centroids[home] * (sizes[home] - 1) + sign * device_points[start + offset]
                norm = float(updated.norm()) if torch is not None else float(np.linalg.norm(updated))
                centroids[home] = updated / max(norm, 1e-30)
                continue
            grow(count + 1)
            centroids[count] = device_points[start + offset]
            sizes.append(1)
            members.append([crossing])
            count += 1

    if count < 2:
        return members

    # One unit produced about two clusters on the first pass, because a single crossing's
    # direction carries enough noise to miss the threshold against another crossing of the same
    # boundary while the two clusters' averages do not. Purity was already 1.000, so this only
    # stops a unit being counted twice and its points split between halves too small to
    # determine a hyperplane.
    parent = list(range(count))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    live = centroids[:count]
    rows = max(1, (1 << 25) // max(count, 1))
    for start in range(0, count, rows):
        stop = min(start + rows, count)
        if torch is not None:
            overlap = (live[start:stop] @ live.T).abs()
            # `as_tuple`, because torch returns an (N, 2) index matrix where numpy returns a
            # tuple of index arrays, and reading one as the other silently pairs a row with a
            # column - which merged two orthogonal directions into one cluster.
            rows_hit, cols_hit = torch.nonzero(overlap >= min_cosine, as_tuple=True)
            pairs = zip(rows_hit.cpu().numpy(), cols_hit.cpu().numpy())
        else:
            overlap = np.abs(live[start:stop] @ live.T)
            pairs = zip(*np.nonzero(overlap >= min_cosine))
        for row, column in pairs:
            left, right = find(start + int(row)), find(int(column))
            if left != right:
                parent[max(left, right)] = min(left, right)

    grouped: dict[int, list[BoundaryCrossing]] = {}
    for index in range(count):
        grouped.setdefault(find(index), []).extend(members[index])
    return list(grouped.values())


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

    # More units than the oracle's answer is wide. This finding was first written as
    # `non-identifiable` on the reasoning that the unmixing `A = R D+` becomes underdetermined,
    # and that reasoning does not survive reading the path that actually runs: the up solve has
    # been joint since it was measured to beat the per-unit route everywhere, and the joint
    # solve never forms `D+` at all. Above the boundary the per-unit route recovers no up
    # direction above cosine 0.99 while the joint solve returns 1.000000, which is why it is the
    # default.
    #
    # What genuinely degrades is narrower: the gate-norm rays are read through `pinv(D)`, whose
    # row `i` cancels the other units exactly, and above the boundary that reader falls back to
    # the raw transpose where the leak grows as `sqrt(k/width)` and passes one at 4096 in 3072.
    # A contaminated norm is a unit evaluated in the wrong regime of `silu`, which is real
    # damage - but it is one degraded reader, not an impossible problem, so this is `ambiguous`
    # and no longer blocks the structure claim by itself.
    #
    # Worth stating precisely, because the obvious reading is wrong and would misdirect the
    # next attempt: `k > w` does not mean directions become near-parallel. 8192 random
    # directions in 3072 dimensions still have typical pairwise cosine around 0.018, and the
    # measurements agree - the 4096-unit run found 14503 clusters and the 2048-unit run 16008,
    # so the grouping was over-segmenting rather than merging. The collapse is downstream.
    #
    # No threshold and no probe decides this, which is what makes it worth checking first: it
    # is knowable before the run rather than after. The 4096-unit block spent 45 million oracle
    # calls against a 3072-wide response to arrive at 962 solutions, held-out R2 0.360, zero
    # units matched above cosine 0.99 - and a verdict of `recovered`. Audit finding F32.
    response_width = int(output_directions.shape[1]) if output_directions.ndim == 2 else 0
    if unit_budget > 0 and response_width and unit_budget > response_width:
        findings.append({
            "family": "units-exceed-output-rank",
            "detail": (
                f"{unit_budget} units requested against a {response_width}-wide oracle response; "
                "gate norms are read through a projection that no longer cancels neighbours"
            ),
            "verdict": "ambiguous",
        })

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

    # The positive side of the same overlap, and the more dangerous one, because the output
    # direction is this route's *identity*: crossings are grouped by it, so two units whose
    # outputs nearly coincide either merge into one cluster upstream or come back as two
    # near-parallel columns whose pseudo-inverse amplifies noise by the condition number.
    # Measured on a six-unit teacher with one pair at cosine 0.999: cond(D) 6968, one of six
    # up directions recovered - and the verdict said `recovered` with no findings, because
    # only the gates were being checked for collinearity. Audit finding F28.
    collinear_output_pairs = 0
    if output_directions.shape[0] > 1:
        collinear_output_pairs = int(np.sum(overlap >= collinear_threshold) // 2)
    if collinear_output_pairs:
        findings.append({
            "family": "near-collinear-output-directions",
            "detail": f"{collinear_output_pairs} output pair(s) parallel beyond {collinear_threshold}",
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

    # The other side of that count, which had no check at all. Every family here examines the
    # solutions that came back - their collinearity, their energy share, whether there are too
    # many of them. A block whose units mostly merged upstream returns few solutions, and the
    # few that survive are exactly the well-separated ones, so it reads as a small clean block
    # and passes every test above. That is how 962 solutions for a 4096-unit block and 1565 for
    # a 2048-unit one were both reported as `recovered` with an empty findings list.
    #
    # The evidence in that case supports a smaller structure than the block asked about, which
    # is what `canonicalised` already means here. No cutoff is invented: the comparison is a
    # count against the count requested. Audit finding F32.
    if unit_budget > 0 and gates.shape[0] < unit_budget:
        findings.append({
            "family": "undercomplete-recovery",
            "detail": f"{gates.shape[0]} solution(s) returned for a block of {unit_budget}",
            "verdict": "canonicalised",
        })

    # Weak channel below the noise floor: a solution whose share of the recovered energy is
    # negligible is not evidence of a unit, it is evidence of a probe that found nothing.
    #
    # Relative to what an average unit carries, not to a constant. A unit's share of the
    # block's gate energy is about `1/k` by construction, so an absolute floor means something
    # different at every block size and nothing at all past `1/floor`: at the 0.002 this used
    # to compare against, every unit of a block larger than five hundred is below it. The
    # first live firing of this family said exactly that - 1023 of 1023 solutions flagged on a
    # 1024-unit block that had in fact recovered 1022 of them. Audit finding F31.
    #
    # The question the check is for is whether a solution carries far less than its peers,
    # which is `share * k` well under one.
    weak = 0
    if gates.shape[0]:
        strength = np.linalg.norm(gates, axis=1)
        total = float(np.sum(strength))
        if total > 0:
            relative = (strength / total) * gates.shape[0]
            weak = int(np.sum(relative < weak_share_threshold))
    if weak:
        findings.append({
            "family": "weak-channel-below-floor",
            "detail": f"{weak} solution(s) below {weak_share_threshold} of an average unit's gate energy",
            "verdict": "canonicalised",
        })

    # A gate norm that was not identified is a unit reported at the wrong scale, and `silu`
    # is not scale free, so that matters. The ray measures both ways it can happen - a knee
    # that never appeared in the probed range, and independent rays that disagree about where
    # it sits - and those counts are reported below.
    #
    # They do not raise a finding, and that is deliberate. The two cutoffs were invented, not
    # measured, in a file where I had written that "a number with an invented cutoff is worse
    # than a number" - audit finding F04. Reporting a measurement without a threshold is
    # honest; letting an invented threshold qualify a recovery is not. They return to the
    # verdict when a run that is known to be broken shows a different distribution from one
    # that is known to be clean.
    norm_flags = {"kneeShareBelowThreshold": 0, "raySpreadAboveThreshold": 0}
    if norm_diagnostics:
        norm_flags["kneeShareBelowThreshold"] = sum(
            1 for row in norm_diagnostics if row.get("kneeShare", 1.0) < knee_share_threshold
        )
        norm_flags["raySpreadAboveThreshold"] = sum(
            1 for row in norm_diagnostics if row.get("raySpread", 0.0) > ray_spread_threshold
        )

    blocking = [row for row in findings if row["verdict"] == "non-identifiable"]
    return {
        "structureClaimAllowed": not blocking,
        "verdict": "non-identifiable" if blocking else ("qualified" if findings else "recovered"),
        "findings": findings,
        "clustersFound": int(clusters_found),
        "gateNormDiagnosticCounts": norm_flags,
        "thresholds": {
            "duplicateGateCosine": float(duplicate_gate_cosine),
            "collinear": float(collinear_threshold),
            "weakShare": float(weak_share_threshold),
            "rankEnergy": float(rank_energy_threshold),
            "kneeShare": float(knee_share_threshold),
            "raySpread": float(ray_spread_threshold),
        },
    }


def _record_fallback(notes: list[str] | None, reason: str) -> None:
    """Leave a trace when the card is not used for a reason other than not being there.

    A silent fallback is a run that is merely slow and says nothing, which is how a broken
    card looks exactly like a working one. The caller passes its telemetry notes; when it does
    not, the message still reaches stderr rather than disappearing.
    """
    if notes is not None:
        notes.append(reason)
    else:
        print(f"solve_least_squares: {reason}", file=sys.stderr)


def _chunked_normal_equations(
    torch,
    design: np.ndarray,
    target: np.ndarray,
    chunk_rows: int = 16_384,
) -> np.ndarray:
    """The same least squares, without ever holding the design on the card.

    A 1024-unit block asks for a 100000 by 4096 design - 1.6 GB before the QR workspace - on
    a card where the oracle already holds 2.5 of 7.65 GB. `A^T A` is `cols x cols` however
    many rows there are, so the Gram is accumulated a chunk at a time and only 4096 squared
    ever lives on the device.

    Precision is handled rather than hoped for, because normal equations square the condition
    number and that has burned this file twice (F15). Columns are normalised first, so the
    Gram's conditioning comes from geometry and not from scale; each chunk product is fp32 -
    that is where the flops are, and fp64 on this card is thirty times slower - but the
    accumulator is fp64, so summation error does not grow with the row count; and the final
    `cols x cols` solve is fp64 with a relative ridge for degenerate columns.
    """
    rows, cols = design.shape
    gram = torch.zeros((cols, cols), dtype=torch.float64, device="cuda")
    moment = torch.zeros((cols,), dtype=torch.float64, device="cuda")
    sumsq = torch.zeros((cols,), dtype=torch.float64, device="cuda")
    for start in range(0, rows, chunk_rows):
        block = torch.as_tensor(design[start:start + chunk_rows], dtype=torch.float32, device="cuda")
        sumsq += (block * block).sum(dim=0).double()
    scale = sumsq.sqrt().clamp_min(1e-30).float()
    for start in range(0, rows, chunk_rows):
        block = torch.as_tensor(design[start:start + chunk_rows], dtype=torch.float32, device="cuda") / scale
        rhs = torch.as_tensor(target[start:start + chunk_rows], dtype=torch.float32, device="cuda")
        gram += (block.T @ block).double()
        moment += (block.T @ rhs).double()
    ridge = 1e-10 * torch.diagonal(gram).sum()
    gram += ridge * torch.eye(cols, dtype=torch.float64, device="cuda")
    factor = torch.linalg.cholesky(gram)
    solution = torch.cholesky_solve(moment.unsqueeze(1), factor).squeeze(1)

    # Two rounds of iterative refinement, and they are what make fp32 Gram products honest.
    # Those products carry about 1e-7 of relative error and normal equations hand it to the
    # solution multiplied by the squared condition number: on a design with columns six orders
    # of magnitude apart the unrefined answer missed ill-determined coefficients by 15%. Each
    # round multiplies the error by (Gram error x condition), so two take that to 1e-7.
    #
    # The residual is computed the same way the Gram was - a chunk at a time, nothing full
    # sized on either side - and it is a matvec where the Gram is a matmul, so it is linear in
    # the rows against the Gram's quadratic and costs almost nothing next to what it fixes.
    # Loaded double here, unlike the Gram pass. Promoting an fp32 block with `.double()`
    # does not recover the digits it never had, and refinement built that way corrects the
    # error it just introduced - measured, it stalled at 4e-3 against 1e-7 when the residual
    # is computed from fp64 data. Half the chunk, because the rows now cost eight bytes; this
    # is a matvec, so it is bandwidth-bound and the width is the whole cost.
    refine_rows = max(chunk_rows // 2, 1)
    scale64 = scale.double()
    for _ in range(2):
        correction = torch.zeros((cols,), dtype=torch.float64, device="cuda")
        for start in range(0, rows, refine_rows):
            block = torch.as_tensor(
                design[start:start + refine_rows], dtype=torch.float64, device="cuda"
            ) / scale64
            rhs = torch.as_tensor(target[start:start + refine_rows], dtype=torch.float64, device="cuda")
            correction += block.T @ (rhs - block @ solution)
        solution = solution + torch.cholesky_solve(correction.unsqueeze(1), factor).squeeze(1)
    return (solution / scale.double()).cpu().numpy()


def solve_least_squares(
    design: np.ndarray,
    target: np.ndarray,
    notes: list[str] | None = None,
) -> np.ndarray:
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
        # Only the one condition that is a legitimate answer rather than a breakdown: CUDA's
        # `gels` driver requires full rank, and a rank-deficient design is a real case here -
        # near-collinear units are one of the negative cases this file has to handle. Anything
        # else - out of memory, a driver fault, a shape error - is a failure and leaves by the
        # front door. It used to be `except Exception: pass`, which turned a broken card into
        # a run that was merely slow and said nothing, and `operator-integrity-lock` forbids
        # exactly that. Audit finding F01.
        # Whether the direct solve fits. The design, the target and the QR workspace all
        # live on the card at once - about two and a half times the design - and a 1024-unit
        # block was killed here asking for 1.69 GB with 1.53 free, while the oracle held its
        # 2.5. Checked against the device rather than assumed, and routed to the chunked
        # Gram path when it does not fit; the failure used to leave by the front door, which
        # was correct, and now the work also gets done.
        free_bytes, _total = torch.cuda.mem_get_info()
        needed = int(2.5 * design.size * 4)
        # The override exists because the chunked path is otherwise unreachable on any block
        # small enough to have a known-good answer: it only engages where the direct solve does
        # not fit, and the only block that big had never been recovered at all. With it, the
        # two paths can be run against the same 128-unit recovery and compared, which is the
        # difference between "agrees with numpy on synthetic data" and "produces the same
        # block". It is also how a machine with a larger card can exercise it in CI.
        forced = os.environ.get("GCL_FORCE_CHUNKED_SOLVE") == "1"
        if forced or needed > free_bytes - (256 << 20):
            _record_fallback(notes, (
                f"design of {design.size * 4 >> 20} MB exceeds free device memory; "
                "solved by chunked normal equations"
            ))
            return _chunked_normal_equations(torch, design, target)
        try:
            left = torch.as_tensor(design, dtype=torch.float32, device="cuda")
            right = torch.as_tensor(target, dtype=torch.float32, device="cuda")
            solution = torch.linalg.lstsq(left, right.unsqueeze(1)).solution
        except torch.linalg.LinAlgError as rank_deficient:
            _record_fallback(notes, f"rank-deficient design on the card: {rank_deficient}")
        except torch.OutOfMemoryError:
            # The estimate above missed - fragmentation, or another process grew. Same
            # answer by the bounded path, and recorded, never silent.
            _record_fallback(notes, "direct solve hit device OOM; solved by chunked normal equations")
            return _chunked_normal_equations(torch, design, target)
        except RuntimeError as fault:
            # The host's NVIDIA kernel module and userspace library are on different point
            # releases. `PYTORCH_NVML_BASED_CUDA_CHECK=0` gets past the availability probe, but
            # the caching allocator calls `nvmlInit_v2` itself when it has to grow, and that
            # assert fires - which is why a 1024-unit block dies here while 512 does not.
            #
            # Translated rather than swallowed or left as a PyTorch internal assert. Falling
            # back would finish the run on the host over many hours and hand back a result
            # whose provenance is "the card was broken", so it stops and says what to do.
            if "nvml" not in str(fault).lower():
                raise
            raise RuntimeError(
                "the CUDA allocator could not initialise NVML, which means the loaded kernel "
                "module and the userspace library are on different releases. Reboot to load "
                "the module that matches, then re-run. Blocks up to 512 units do not reach "
                "the allocation size that triggers it."
            ) from fault
        else:
            # A non-finite solution is a failure too, not a reason to quietly try elsewhere.
            # Falling back is right, saying nothing about it is not. Audit finding F07.
            if torch.isfinite(solution).all():
                return solution.squeeze(1).double().cpu().numpy()
            _record_fallback(notes, "card returned a non-finite solution; solved on the host")
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
        except ImportError:
            # No torch is a legitimate state, not a fault. Audit finding F08: one blanket
            # `except Exception` made a missing package and a broken driver look identical.
            _torch_cuda.cached = None
        else:
            _torch_cuda.cached = torch if torch.cuda.is_available() else None
    return _torch_cuda.cached


def _batched_normal_equations(
    grid: np.ndarray,
    window: np.ndarray,
    background: np.ndarray,
    projected: np.ndarray,
) -> np.ndarray:
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
        # Errors only. The solution was returned alongside and never read, which in the GPU
        # branch meant a `(C, 5)` transfer back to the host on every call - four calls per ray,
        # three rays per unit. Audit finding F12: a synchronisation added inside the function
        # that had just been moved to the card to avoid them.
        return (residual.squeeze(2) ** 2).sum(dim=1).double().cpu().numpy()

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
    return (residual ** 2).sum(axis=1)


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

    # The outer products that build the rays, and the projection that reads them back, on the
    # card. Each batch is `batch_units * repeats * points` rows of `input_dim` and there is one
    # batch per seven units, so at a full layer this is a thousand of them - built, filled and
    # projected on the host while the oracle answering them runs on the card.
    torch_rays = _torch_cuda()
    device_positions = (
        torch_rays.as_tensor(positions, dtype=torch_rays.float64, device="cuda")
        if torch_rays is not None else None
    )
    device_rays = (
        torch_rays.zeros((batch_units * per_unit, input_dim), dtype=torch_rays.float64, device="cuda")
        if torch_rays is not None else None
    )

    def build(index: int, slot: int) -> None:
        direction = gate_directions[index] / np.linalg.norm(gate_directions[index])
        for repeat in range(repeats):
            offset = rng.normal(0.0, 1.0, size=input_dim)
            offset -= (offset @ direction) * direction
            offset *= offset_scale * np.sqrt(input_dim) / np.linalg.norm(offset)
            start = (slot * repeats + repeat) * positions.size
            if torch_rays is not None:
                device_rays[start:start + positions.size] = (
                    device_positions[:, None]
                    * torch_rays.as_tensor(direction, dtype=torch_rays.float64, device="cuda")[None, :]
                    + torch_rays.as_tensor(offset, dtype=torch_rays.float64, device="cuda")[None, :]
                )
            else:
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
        errors = _batched_normal_equations(grid, window, background, projected)
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
        if torch_rays is not None:
            rays[:rows] = device_rays[:rows].cpu().numpy()
        responses = np.concatenate(
            [evaluate(rays[start:min(start + request_chunk, rows)])
             for start in range(0, rows, request_chunk)],
            axis=0,
        )
        if torch_rays is not None:
            device_responses = torch_rays.as_tensor(responses, dtype=torch_rays.float64, device="cuda")
            device_readers = torch_rays.as_tensor(
                output_directions[list(batch)], dtype=torch_rays.float64, device="cuda"
            )
            projected_batch = (device_responses @ device_readers.T).cpu().numpy()
            del device_responses, device_readers
        for slot, index in enumerate(batch):
            measured, shares = [], []
            for repeat in range(repeats):
                start = (slot * repeats + repeat) * positions.size
                projection = (
                    projected_batch[start:start + positions.size, slot] if torch_rays is not None
                    else responses[start:start + positions.size] @ output_directions[index]
                )
                norm, error, reference = measure_ray(projection)
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
            # Both orientations scored on the card and compared there. Pulling each residual
            # across with `float(...)` cost a synchronisation per candidate per unit - audit
            # finding F13, the same mistake as F12 and in the same commit.
            scored = []
            for candidate in (1.0, -1.0):
                z = raw[:, index] * (norms[index] * candidate)
                design = torch.cat((z.mul(torch.sigmoid(z))[:, None] * coords, coords), dim=1)
                solution = torch.linalg.lstsq(design, targets[:, index, None]).solution
                scored.append(((design @ solution).squeeze(1) - targets[:, index]).pow(2).sum())
            signs[index] = 1.0 if bool(scored[0] <= scored[1]) else -1.0
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


def solve_up_directions_jointly(
    smooth: np.ndarray,
    coordinates: np.ndarray,
    responses: np.ndarray,
    output_directions: np.ndarray,
    iterations: int = 200,
    chunk_rows: int = 8192,
) -> np.ndarray:
    """Every unit's up direction at once, by conjugate gradient on a matrix never formed.

    The per-unit solve reads a unit through its own output direction, which carries every
    other unit scaled by `d_j . d_i`. Inverting that mixing removes it exactly, but only while
    there are more output channels than units - above that the system is wide and the
    pseudo-inverse fabricates. A whole layer is 8192 units into 3072 channels, so the route
    had nothing left but the raw projection, where the leak grows as `sqrt(k/dim)` and reaches
    1.63 - larger than the signal it is trying to read.

    The residual cross-talk is only removable by solving for every `u_j` together, and that
    system has `k * rank` unknowns: 25 million for a layer, which is why it was written off.
    But conjugate gradient never needs the matrix, only its action, and the action is two
    matrix products:

        apply(U)  = (smooth * (coordinates @ U^T)) @ D
        adjoint(Y) = (smooth * (Y @ D^T))^T @ coordinates

    Measured against the per-unit projection on synthetic blocks above the boundary, share of
    up directions above cosine 0.99 and their median:

        43 units in 32 dims    0/43  at 0.9761    ->  43/43  at 1.000000
        86 units in 32 dims    0/86  at 0.9489    ->  86/86  at 1.000000
       200 units in 64 dims    0/200 at 0.9556    -> 200/200 at 1.000000

    Rows are streamed because `smooth` alone is `n by k` - 3.3 GB at a hundred thousand
    samples and a full layer - and nothing here needs it resident.
    """
    torch = _torch_cuda()
    rows, rank = coordinates.shape
    units = smooth.shape[1]
    if torch is None:
        activation, coords, out = smooth, coordinates, output_directions
        apply_op = lambda U: (activation * (coords @ U.T)) @ out
        adjoint = lambda Y: (activation * (Y @ out.T)).T @ coords
        moment = adjoint(responses)
        solution = np.zeros((units, rank))
        residual = moment - adjoint(apply_op(solution))
        direction = residual.copy()
        squared = float((residual * residual).sum())
        for _ in range(iterations):
            curvature = adjoint(apply_op(direction))
            step = squared / max(float((direction * curvature).sum()), 1e-30)
            solution += step * direction
            residual -= step * curvature
            updated = float((residual * residual).sum())
            if updated < 1e-24 * max(squared, 1e-30):
                break
            direction = residual + (updated / squared) * direction
            squared = updated
        return solution

    device_out = torch.as_tensor(output_directions, dtype=torch.float32, device="cuda")

    # Resident when they fit, streamed when they do not. Conjugate gradient walks the whole
    # dataset twice per iteration and takes a couple of hundred iterations, so streaming from
    # the host means four hundred uploads of everything - measured at 29 seconds of host CPU
    # against 3.7 of wall clock on a 128-unit block, which is copying, not arithmetic. At a
    # full layer `smooth` alone is 3.3 GB and streaming is the only option, so both paths stay.
    free_bytes, _total = torch.cuda.mem_get_info()
    resident = (smooth.size + coordinates.size) * 4 + (1 << 28) < free_bytes
    if resident:
        device_smooth = torch.as_tensor(smooth, dtype=torch.float32, device="cuda")
        device_coords = torch.as_tensor(coordinates, dtype=torch.float32, device="cuda")
        device_responses = torch.as_tensor(responses, dtype=torch.float32, device="cuda")

    def normal_operator(candidate, target=None):
        """`adjoint(apply(candidate))`, or `adjoint(target)` when no candidate is given."""
        accumulated = torch.zeros((units, rank), dtype=torch.float64, device="cuda")
        for start in range(0, rows, chunk_rows):
            stop = min(start + chunk_rows, rows)
            if resident:
                activation = device_smooth[start:stop]
                coords = device_coords[start:stop]
            else:
                activation = torch.as_tensor(smooth[start:stop], dtype=torch.float32, device="cuda")
                coords = torch.as_tensor(coordinates[start:stop], dtype=torch.float32, device="cuda")
            if candidate is None:
                block = (
                    device_responses[start:stop] if resident
                    else torch.as_tensor(target[start:stop], dtype=torch.float32, device="cuda")
                )
            else:
                block = (activation * (coords @ candidate.float().T)) @ device_out
            accumulated += ((activation * (block @ device_out.T)).T @ coords).double()
        return accumulated

    moment = normal_operator(None, responses)
    solution = torch.zeros((units, rank), dtype=torch.float64, device="cuda")
    residual = moment - normal_operator(solution)
    direction = residual.clone()
    squared = float((residual * residual).sum())
    for _ in range(iterations):
        curvature = normal_operator(direction)
        step = squared / max(float((direction * curvature).sum()), 1e-30)
        solution += step * direction
        residual -= step * curvature
        updated = float((residual * residual).sum())
        # Relative, because the scale of `moment` varies by orders of magnitude with the block.
        if updated < 1e-24 * max(squared, 1e-30):
            break
        direction = residual + (updated / squared) * direction
        squared = updated
    return solution.cpu().numpy()


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
    peak_floor_mode: str = "max",
    sharpness_mode: str = "ratio",
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
    weak_share_threshold: float = 0.05,
    rank_energy_threshold: float = 0.98,
    unit_budget: int = 0,
    seed: int,
) -> tuple[dict[str, Any], RecoveryTelemetry]:
    rng = np.random.default_rng(seed)
    telemetry = RecoveryTelemetry()

    # The active subspace cannot be wider than the space it sits in. Callers ask for `2k`
    # because each unit contributes a gate and an up direction, and that is right until `2k`
    # passes `input_dim`, where the subspace is simply everything and the request is
    # meaningless. Unclamped it does not degrade, it crashes: the basis comes back with
    # `input_dim` rows while the probe lines are drawn with `2k` coordinates, and the first
    # scan dies on a shape mismatch. That is every block above 1536 units on this model, which
    # is to say every block on the way to a layer.
    if active_rank > input_dim:
        telemetry.notes.append(
            f"active rank {active_rank} exceeds input dimension {input_dim}; the active "
            "subspace is the whole space"
        )
        active_rank = input_dim

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
                peak_floor_mode=peak_floor_mode,
                sharpness_mode=sharpness_mode,
                line_index=line_index,
                line_directions=line_directions,
                timings=scan_timings,
            )
            crossings.extend(found)
            telemetry.scanCalls += calls
            telemetry.oracleCalls += calls
            telemetry.linesScanned += 1
        telemetry.crossingsFound = len(crossings)

    # Its own field, not an entry in `phaseSeconds`. It has different keys from every other
    # entry there, so anything iterating the phases hit a `KeyError` - which I did, in my own
    # reporting script. Audit finding F05.
    telemetry.lineScanBreakdown = {k: round(v, 3) for k, v in scan_timings.items()}
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
        # Both products on the card. `samples_in` is `rows x output_dim` and these turn it into
        # `rows x units` and `rows x rank`, which at a hundred thousand samples and a full
        # layer is 2.5e12 and 9.4e11 floating point operations - on the host, while the card
        # that had just answered every one of those samples sat idle. Double throughout: these
        # feed `silu` and a least squares, and single precision in this file has destroyed a
        # recovery three times.
        torch_probe = _torch_cuda()
        if torch_probe is not None:
            device_samples = torch_probe.as_tensor(samples_in, dtype=torch_probe.float64, device="cuda")
            # Single on the way back, double in the product. These are inner products of
            # probe points with recovered directions, not differences, so the width that
            # matters is the one the multiply is done in - and everything downstream, the
            # conjugate gradient and the sign design alike, casts to single anyway. At a full
            # layer `rows x units` in double is 7.9 GB per copy and there are three of them.
            # Chunked over rows, and the chunk is a memory budget rather than a row count -
            # the note above says `rows x units` in double is 7.9 GB per copy at a full layer
            # and that there are three of them, and then formed them whole anyway. That is what
            # the 8192-unit run died on: 1.38 GiB requested with 973 MiB free. The products are
            # row-independent, so a chunk goes to the host before the next is formed and the
            # device holds only `chunk x units`. A row count would have had to be retuned at
            # every block size; a budget shrinks the rows as the block grows, and leaves small
            # blocks in one piece exactly as before.
            del device_samples
            gate_device = torch_probe.as_tensor(gate_matrix, dtype=torch_probe.float64, device="cuda")
            basis_device = torch_probe.as_tensor(basis, dtype=torch_probe.float64, device="cuda")
            widest = max(int(gate_matrix.shape[0]), int(basis.shape[0]), 1)
            row_chunk = max(1, min(up_sample_count, (512 * 1024 * 1024) // (widest * 12)))
            raw_parts: list[np.ndarray] = []
            coordinate_parts: list[np.ndarray] = []
            for start in range(0, up_sample_count, row_chunk):
                block = torch_probe.as_tensor(
                    samples_in[start:start + row_chunk], dtype=torch_probe.float64, device="cuda"
                )
                raw_parts.append((block @ gate_device.T).float().cpu().numpy())
                coordinate_parts.append((block @ basis_device.T).float().cpu().numpy())
                del block
            raw_response = np.concatenate(raw_parts, axis=0) if len(raw_parts) > 1 else raw_parts[0]
            coordinates = (
                np.concatenate(coordinate_parts, axis=0) if len(coordinate_parts) > 1 else coordinate_parts[0]
            )
            del raw_parts, coordinate_parts, gate_device, basis_device
            torch_probe.cuda.empty_cache()
            telemetry.notes.append(f"up-solve products chunked at {row_chunk} rows for {widest} columns")
        else:
            raw_response = (samples_in @ gate_matrix.T).astype(np.float32)
            coordinates = (samples_in @ basis.T).astype(np.float32)
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
        # Whether the mixing can be inverted at all. `R = A D` with `D` shaped units by
        # output_dim: when there are more units than output channels the system is wide, `A`
        # has more unknowns per row than there are equations, and `pinv` does not recover the
        # activations - it fabricates the minimum-norm member of an infinite family. Measured
        # across the boundary on a 32-dimensional teacher, held-out R2:
        #
        #     12 units   unmixed 0.9999   projected 0.9893
        #     43 units   unmixed 0.9076   projected 0.9253
        #     86 units   unmixed 0.6154   projected 0.8456
        #
        # So it wins where it is defined and loses badly where it is not, and a full layer -
        # 8192 units in 3072 channels - is firmly on the wrong side. The condition is not a
        # tuned threshold but the one under which the operation means anything, and the rank
        # is taken from `D` rather than from the count, because near-collinear outputs make a
        # narrow matrix behave like a wide one.
        invertible = output_directions.shape[0] <= output_directions.shape[1] and (
            np.linalg.matrix_rank(output_directions) == output_directions.shape[0]
        )
        # The pseudo-inverse either way. Above the boundary it does not recover the activations
        # - nothing linear can - but the transpose does not either, and it is the worse of the
        # two failures rather than the safer one. The measurement above that said otherwise was
        # taken on a 32-dimensional teacher, where random directions sit at pairwise cosine
        # around 0.18 and `D` is badly conditioned; at 256 that is 0.06 and at the real 3072 it
        # is 0.018, which is a different geometry, and the conclusion did not carry.
        #
        # Re-measured on real SwiGLU structure at matched `k/dim` ratios, reading correlation
        # and the held-out reconstruction the pipeline actually cares about:
        #
        #     k/dim 0.67   transpose r 0.769  R2 0.862   |  pinv r 1.000  R2 1.000
        #     k/dim 1.33   transpose r 0.650  R2 0.703   |  pinv r 0.866  R2 0.930
        #     k/dim 2.67   transpose r 0.519  R2 0.292   |  pinv r 0.608  R2 0.700
        #
        # 2.67 is 8192 units into 3072 channels - a whole layer - where the projection loses
        # two thirds of the reconstruction and the pseudo-inverse loses a third.
        telemetry.notes.append(
            "output mixing pseudo-inverted"
            + ("" if invertible else "; wide system, reader is least squares rather than exact")
        )
        mixing = np.linalg.pinv(output_directions)
        torch_backend = _torch_cuda()
        if torch_backend is not None:
            # Chunked over rows, like the solve. `responses` is rows by output_dim and
            # `unmixed` is rows by units, so a whole-layer block puts 3.3 GB of the second
            # on a card that has already given 2.5 to the oracle - the same unbounded
            # allocation the direct solve was killed by, one function away and unguarded.
            # Only as many rows as anything actually reads. The joint solve works from
            # `responses` directly and never touches this; the sign decision needs the first
            # few thousand. Materialising all of them is `rows x units` - 7.9 GB at a full
            # layer for an array with one small consumer.
            needed = responses.shape[0] if up_refinement_sweeps > 0 else min(responses.shape[0], 6000)
            device_mixing = torch_backend.as_tensor(mixing, dtype=torch_backend.float32, device="cuda")
            unmixed = np.empty((needed, mixing.shape[1]), dtype=np.float64)
            for start in range(0, needed, up_request_chunk):
                stop = min(start + up_request_chunk, needed)
                block = torch_backend.as_tensor(
                    responses[start:stop], dtype=torch_backend.float32, device="cuda"
                )
                unmixed[start:stop] = (block @ device_mixing).double().cpu().numpy()
            del device_mixing
        else:
            needed = responses.shape[0] if up_refinement_sweeps > 0 else min(responses.shape[0], 6000)
            unmixed = responses[:needed] @ mixing
        # One ray per unit along its own gate direction, where the norm is what decides when
        # the curve turns on and nothing else can imitate it. See
        # `calibrate_gate_norms_on_rays` for why the isotropic cloud cannot settle this.
        with _Phase(telemetry, "normCalibration"):
            gate_norms, norm_diagnostics = calibrate_gate_norms_on_rays(
                evaluate,
                gate_matrix / np.linalg.norm(gate_matrix, axis=1, keepdims=True),
                # The unmixed reader, not the raw output direction, wherever the mixing can
                # be inverted. The ray measures the knee on the response projected onto this
                # unit's channel, so a neighbour that overlaps it leaks straight into the
                # measurement. Row `i` of `pinv(D)` reads unit `i` and cancels the others
                # exactly, which is the same correction the up solve already gets.
                #
                # It matters for exactly the units it should. In the 512-unit block one norm
                # came back 10.16% out against a 10% bar - unit 140, whose output norm is in
                # the 11th percentile of the block and whose nearest neighbour sits at cosine
                # 0.320, the second highest there. Weak channel, close neighbour, contaminated
                # projection. Above the channel boundary `mixing` is already the transpose and
                # this reduces to what it was.
                mixing.T,
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
        raw_response *= signs[None, :].astype(raw_response.dtype)
        # In place, and without a separate `gate_response`. Each of these is `rows x units`,
        # which is 3.9 GB at a full layer even in single, and the intermediate was never read
        # again after `smooth` was built from it.
        smooth = raw_response * gate_norms[None, :].astype(raw_response.dtype)
        np.divide(smooth, 1.0 + np.exp(-np.clip(smooth, -60.0, 60.0)), out=smooth)
        # Solved in basis coordinates, not in the input space. `u_i` lies in the active
        # subspace by construction, so it has `active_rank` free parameters rather than
        # `input_dim`: for 32 units in 3072 dimensions that is 2048 unknowns instead of
        # 98304, and 6000 samples go from hopelessly underdetermined - which returned zero
        # usable up directions on the live model - to comfortably overdetermined.
        # Per unit, through its own output direction, rather than one joint solve over all of
        # them. A joint solve lets a small error in one gate be absorbed by another unit's
        # column: with gates at per-unit cosine 0.9985 it returned 15 of 32 up directions and
        # adding samples did not move it.
        #
        # Cross-talk between units is removed before this point, not tolerated here - see the
        # pseudo-inverse above. The description that used to sit in this comment, of projecting
        # onto `d_i` and living with `d_j . d_i`, described the route as it was before that,
        # and the sweep-and-subtract dead end it recommended is off by default: subtracting
        # fitted estimates injects each neighbour's error while removing its contribution, and
        # took 64 units from 64 up directions to 60. The option stays runnable because a
        # measured dead end is worth keeping, and because the trade may reverse at some unit
        # count. Audit finding F10 was that this comment outlived the code it described.
        #
        # The running model is kept and adjusted rather than rebuilt: recomputing every other
        # unit's contribution inside the loop is quadratic in the unit count and made a
        # 128-unit sweep slower than the entire oracle phase.
        with _Phase(telemetry, "upSolve"):
            # Every unit at once, which turns out to be better everywhere rather than only
            # where the per-unit route breaks down.
            #
            # Reading a unit through its own output direction carries every other unit scaled
            # by `d_j . d_i`. Inverting that mixing removes it exactly, but only while there
            # are more output channels than units; above that boundary - and a layer is 8192
            # units into 3072 channels - the per-unit path has nothing but the raw projection,
            # where the leak grows as `sqrt(k/dim)` and passes one. There it recovers no up
            # direction above cosine 0.99 at all against the joint solve's 1.000000.
            #
            # Below the boundary, where the per-unit path is at its best, the joint solve is
            # still ahead on both axes. Measured on the live 128-unit block: held-out R2
            # 0.9994409 against 0.9994177, and 3.7 seconds against 13.0. It also removes the
            # per-unit design entirely, which at 1024 units was 1.6 GB built and solved a
            # thousand times over.
            joint_solved = up_refinement_sweeps == 0 and os.environ.get("GCL_PER_UNIT_UP") != "1"
            if joint_solved:
                telemetry.notes.append("up directions solved jointly by conjugate gradient")
                joint = solve_up_directions_jointly(
                    smooth, coordinates, responses, output_directions,
                    chunk_rows=up_request_chunk,
                )
                solved_coefficients = [joint[index] for index in range(len(gates))]
            else:
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
            for _sweep in range(0 if joint_solved else max(1, up_refinement_sweeps)):
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
                # On the card. It is one product either way, but at a full layer the operands
                # are `points x units` and `units x output_dim` and the host was holding three
                # cores for it while the oracle that produced the held-out block idled.
                torch_reco = _torch_cuda()
                if torch_reco is not None:
                    as_t = lambda a: torch_reco.as_tensor(a, dtype=torch_reco.float32, device="cuda")
                    predicted = (
                        (as_t(held_smooth) * (as_t(held_coordinates) @ as_t(coefficients).T))
                        @ as_t(output_directions)
                    ).double().cpu().numpy()
                else:
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
        "lineScanBreakdown": telemetry.lineScanBreakdown,
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
    parser.add_argument("--weak-share-threshold", type=float, default=0.05)
    parser.add_argument("--rank-energy-threshold", type=float, default=0.98)
    parser.add_argument("--unit-budget", type=int, default=0, help="0 uses the requested unit count")
    parser.add_argument("--up-refinement-sweeps", type=int, default=0,
                        help="subtract already-solved units before each solve; measured worse, kept runnable")
    parser.add_argument("--sharpness-mode", choices=("sigma", "ratio"), default="ratio",
                        help="how far a peak must stand out from its neighbourhood. 'sigma' is "
                             "distance above the local median in robust spreads, which means the "
                             "same thing at every block size; 'ratio' is the previous peak over "
                             "background, which demands more the larger the block gets")
    parser.add_argument("--peak-floor-mode", choices=("background", "max"), default="max",
                        help="what the peak pre-filter is measured against. 'background' is the "
                             "line's median turning magnitude, which one loud crossing does not "
                             "move; 'max' is the previous behaviour, a fraction of the line's "
                             "largest peak, kept so the two can be compared on the same block")
    parser.add_argument("--allow-unidentifiable-block", action="store_true",
                        help="run even when the unit count exceeds the oracle response width; "
                             "the result cannot be a structure claim, but measuring the failure "
                             "is how the verdict's blind spot to it was found")
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
    # In RAM when there is a tmpfs to put it in. Every oracle exchange is two `.npy` files -
    # about 10 MB each way for a line scan - and on a disk-backed `/tmp` that measured 19.9 ms
    # per call, an effective 0.99 GB/s, with the card at 2% utilisation the whole time. The
    # work is neither on the host nor on the card, it is waiting on the filesystem.
    shared = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else None
    with tempfile.TemporaryDirectory(prefix="gcl-jacobian-recovery-", dir=shared) as work:
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
            # The counting argument, before the first probe rather than after forty-five
            # million of them. Past the response width the unmixing `A = R D+` is
            # underdetermined and its fallback leaks more than the signal, so the block cannot
            # be separated however well the crossings are found. The 4096-unit block spent an
            # hour and a half establishing that, and reported `recovered` at the end of it.
            # The flag exists because measuring a known-impossible configuration is how the
            # blind spot was found, and that has to stay possible on purpose.
            if len(unit_list) > width and not args.allow_unidentifiable_block:
                raise SystemExit(
                    f"{len(unit_list)} units against a {width}-wide oracle response was measured at "
                    "0 of 4096 recovered, held-out R2 0.360. Above the width the gate-norm rays "
                    "lose the reader that cancels neighbours exactly, and the leak passes one. "
                    "This is a measured failure rather than a proof of impossibility - the up "
                    "solve is joint and never forms the inverse - so pass "
                    "--allow-unidentifiable-block to run it and find out."
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
                peak_floor_mode=args.peak_floor_mode,
                sharpness_mode=args.sharpness_mode,
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
            # Evidence rather than declaration. `accessAudit` above is four literals the tool
            # writes about itself, and the lane asserting that those literals hold their
            # expected values proves only that the tool spelled them correctly - audit finding
            # F17. The oracle already keeps what is needed: it refuses `truth()` until the
            # unlock, and it records the query count at the moment of unlocking.
            #
            # Two independent counters have to agree. `truthUnlockQueryCount` is what the
            # oracle had served when truth was unlocked; `solverOracleCalls` is what the solver
            # counted spending. If they differ, a query was made outside the accounted path,
            # which is the thing the invariant is actually about.
            report["accessAudit"].update({
                "truthUnlockStage": oracle.truth_unlock_stage,
                "truthUnlockQueryCount": oracle.truth_unlock_query_count,
                "solverOracleCalls": int(telemetry.oracleCalls),
                "queryCountAtReport": int(oracle.query_count),
            })
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
