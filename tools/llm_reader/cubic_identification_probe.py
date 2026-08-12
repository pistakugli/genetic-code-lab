#!/usr/bin/env python3
"""Identify SwiGLU units algebraically from the third directional derivative.

Why this exists. The boundary route in `jacobian_boundary_recovery.py` reads a unit off the
kink its activation boundary puts in the oracle response, and it groups crossings by *output
direction*. That identity caps the block at the response width: 4096 units cannot have
independent directions in a 3072-wide answer, so the run refuses the block up front. The
whole-layer rung needs an identity that is not a direction in the answer.

What the third derivative gives. Along a line `t -> y(t d)` through the origin,

    silu(t g.d) * (t u.d) = 0.5 t^2 (g.d)(u.d) + 0.25 t^3 (g.d)^2 (u.d) + O(t^5)

so the third derivative at zero is `1.5 * sum_i (g_i.d)^2 (u_i.d) d_i`, and reading it through
a fixed reader `w` gives a cubic form in `d` whose symmetric tensor is

    S = sum_i lambda_i * sym(g_i (x) g_i (x) u_i),   lambda_i = 1.5 * (w.d_i)

The second derivative cannot do this job. It gives `sum_i (w.d_i) sym(g_i u_i^T)`, which is
symmetric in `g` and `u`: nothing in it says which of the two recovered directions is the gate.
Squaring the gate in the third derivative is what breaks that tie, and the break is algebraic
rather than numerical - it survives as an exact collinearity, not as a fitted preference.

The identification, and the route to it that did not work. Apolarity writes each unit as three
symmetric rank-1 terms, `sym(g g u) = [(g+u)^3 - (g-u)^3 - 2u^3]/6`, which suggests symmetric
rank `3k` and a grouping rule on the triple. It is wrong, and the harness said so: those three
vectors span the *plane* `span{g,u}`, not a 3-space, so the factor matrix has rank `2k` and
Jennrich's full-column-rank hypothesis fails by construction - reconstruction error `0.79` with
weights of order `1e15`.

What the pencil does return is better than the triple. `S(.,.,x) S(.,.,y)^+` has `k` eigenvalues
of multiplicity two, and their eigenvectors are the gate directions at cosine `1.000000`. The
ups then need no second nonlinear step at all: with the gates known, `phi(d) = sum_i (g_i.d)^2
(v_i.d)` is *linear* in `v_i = lambda_i u_i`, so they come out of one least squares over
directions already probed. Gate algebraically, up linearly, nothing iterative anywhere.

Two numbers this measured, both on a synthetic teacher, both on the card:

  - The wall is `2k <= n`, on the *input* dimension, not the response width. At `n=16` recovery
    is `k/k` through `k=8` and collapses at `k=9`; at `n=24` it is `k/k` through `k=12` and
    collapses at `k=13`. The unfolding rank saturates at `n` and the recovered unit count stops
    with it. For a 3072-wide input that is 1536 units, so this route does not reach 8192 either
    - it moves the wall, it does not remove it.
  - The unit count is not assumed. It is read off as `rank/2` and was correct at every point
    below the wall.

What this costs. Fitting every degree-3 coefficient costs `4n(n+1)(n+2)` oracle points and buys
`n/6` times more tensor than the decomposition consumes: the pencil reads two slices. Measuring
just those by polarisation costs `8*(1.5n(n+1) + 3n + 2)`, verified to return the same gates to
the same cosine - `5.34x` cheaper at `n=16` and `8.01x` at `n=24`, a saving of `n/3`. Run with
`--project-dim` for what those two laws say about a live layer.

The response width is not the fundamental bound it looked like, and that is the result here that
matters most. At the origin every reader weights unit `i` by `1.5 (w.d_i)`, so all readers see
the `k` unit tensors through the down projection and the accessible dimension is `rank(D) <= m` -
measured exactly, `rank span{S_w} = min(k, m)` at six settings. That is the arithmetic behind
"4096 units cannot be separated in a 3072-wide answer", and it is arithmetic about *probing at
one point*.

Off the origin the weight is `beta_i = 3 (w.d_i) silu''(g_i.x0)`, and the second factor is a
nonlinear function of the unit rather than a linear read through `D`. So base points buy rank
that readers cannot: the profile rank is `min(k, m*P)` for `P` base points, verified at every
sweep point. With `k=40` units behind an `m=8` answer - five times wider than the oracle - eight
base points reach all 40, and the smallest-to-largest singular value stays between `1.2e-02` and
`4.5e-01` across two seeds, so this is a conditioned separation and not a rank counted at a
threshold. For a 3072-wide answer over 8192 units the same law asks for three base points.

What that is and is not. It is the necessary condition - the units' contributions are separable
in what the oracle can be made to show - measured, conditioned, and reproducible through
`--base-point-sweep`. It is not a recovery run: the pencil above still needs `2k <= n`, and no
decomposition here yet recovers `k > n` units.

And the wall the next decomposition has to beat is not `n` either. The pencil stops at `2k <= n`
because it diagonalises in `R^n`, where each unit costs two of `n` directions. The `(ab)`
unfolding of the same tensor puts each unit in `Sym^2(R^n)` instead, as `g (x) g` and
`sym(g (x) u)`, and there the room is `n(n+1)/2` - quadratic where it was linear. Measured
(`--lifted-sweep`): the span's rank is `min(2k, n(n+1)/2)` at every setting, so the ceiling is
`k <= n(n+1)/4`. At `n=10` that is 27 units against the pencil's 5, holding at 1.13e-03 for
`k=27` and collapsing to 8.9e-17 at `k=28`, which is exactly where `2k` passes the room. For a
3072-wide input it is **2360064** units against the pencil's 1536, and 8192 is nowhere near it.

So the whole layer is not blocked by identifiability at any of the three walls this file
measured - response width, input dimension, or lifted room. What is missing is the extraction:
recovering the `g_i` means finding the rank-1 elements of that lifted span, a FOOBI-class step
that is not implemented here. That is the next gate, and it is now the only one left standing.

This is a measurement harness against a synthetic teacher, in the idiom of
`jacobian_jump_probe.py`. It is not wired into a lane. Run it directly.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from itertools import combinations_with_replacement

import torch


# The whole probe is a cancellation measurement: a third derivative is a difference of
# differences of differences, and in single precision the stencil's numerator loses every digit
# the signal has. Double everywhere, on the card.
DTYPE = torch.float64


@dataclass
class OracleCallCounter:
    points: int = 0
    batches: int = 0

    def charge(self, count: int) -> None:
        self.points += int(count)
        self.batches += 1


@dataclass
class Teacher:
    """`y(x) = D * (silu(G x) . (U x))`, the same shape the live block has."""

    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor
    counter: OracleCallCounter = field(default_factory=OracleCallCounter)

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        self.counter.charge(points.shape[0])
        z = points @ self.gate.T
        activation = torch.nn.functional.silu(z)
        return (activation * (points @ self.up.T)) @ self.down.T


def make_teacher(units: int, input_dim: int, output_dim: int, seed: int, device: torch.device) -> Teacher:
    generator = torch.Generator(device=device).manual_seed(seed)

    def normal(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, device=device, dtype=DTYPE)

    return Teacher(
        gate=normal(units, input_dim) / math.sqrt(input_dim),
        up=normal(units, input_dim) / math.sqrt(input_dim),
        down=normal(output_dim, units),
    )


def analytic_cubic_form(teacher: Teacher, reader: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """`1.5 * sum_i (w.d_i) (g_i.d)^2 (u_i.d)`, the value the stencil has to reproduce."""
    weights = 1.5 * (teacher.down.T @ reader)
    gate_projection = directions @ teacher.gate.T
    up_projection = directions @ teacher.up.T
    return (gate_projection ** 2 * up_projection * weights[None, :]).sum(dim=1)


def measure_cubic_form(
    teacher: Teacher,
    reader: torch.Tensor,
    directions: torch.Tensor,
    steps: list[float],
) -> torch.Tensor:
    """Central third-derivative stencil, Richardson-extrapolated in `h^2`.

    The stencil `[f(2h) - 2f(h) + 2f(-h) - f(-2h)] / (2h^3)` is `D3 + c h^2 + O(h^4)`, so two
    steps and a straight line in `h^2` remove the leading term. Every point is charged to the
    oracle counter, because the cost of this route is the result this file is here to report.
    """
    # Probe on the unit sphere and put the length back afterwards. `phi` is homogeneous of
    # degree three, so `phi(v) = |v|^3 phi(v/|v|)`, and a step of `h` along a vector of length
    # `L` is a step of `hL` in the function - which makes the stencil's `O(h^2)` truncation
    # term grow with `L^2` while the answer is only wanted to fixed relative accuracy.
    # Polarisation probes vectors of length up to about 2.4 where the direction fit probes unit
    # vectors, and that alone cost the slice route two decimal digits: measured, slice error
    # 5.5e-07 against 1.6e-08 for the same block read as a whole tensor.
    lengths = directions.norm(dim=1, keepdim=True).clamp_min(1e-300)
    unit_directions = directions / lengths
    cube = lengths.squeeze(1) ** 3

    estimates = []
    for step in steps:
        offsets = torch.tensor([2.0 * step, step, -step, -2.0 * step], device=directions.device, dtype=DTYPE)
        # One batch per stencil node, all directions at once: the counter charges points, so
        # batching changes the wall clock and not the reported cost.
        values = [(teacher(unit_directions * offset) @ reader) for offset in offsets]
        derivative = (values[0] - 2.0 * values[1] + 2.0 * values[2] - values[3]) / (2.0 * step ** 3)
        estimates.append(derivative * cube)
    stacked = torch.stack(estimates, dim=0)
    if len(steps) == 1:
        return stacked[0]
    step_squares = torch.tensor([[1.0, s ** 2] for s in steps], device=directions.device, dtype=DTYPE)
    coefficients = torch.linalg.lstsq(step_squares, stacked).solution
    return coefficients[0]


def monomial_index(input_dim: int) -> list[tuple[int, int, int]]:
    return list(combinations_with_replacement(range(input_dim), 3))


def recover_symmetric_tensor(
    directions: torch.Tensor,
    values: torch.Tensor,
    input_dim: int,
) -> torch.Tensor:
    """Least squares over degree-3 monomials, then undo the multinomial multiplicity.

    `phi(d) = sum_{a<=b<=c} C_abc d_a d_b d_c` while `phi(d) = sum_abc S_abc d_a d_b d_c` over
    every ordered triple, so `C_abc = mult * S_abc` and the tensor is the fitted coefficient
    divided by how many orderings that index has.
    """
    index = monomial_index(input_dim)
    design = torch.stack(
        [directions[:, a] * directions[:, b] * directions[:, c] for a, b, c in index],
        dim=1,
    )
    coefficients = torch.linalg.lstsq(design, values.unsqueeze(1)).solution.squeeze(1)
    tensor = torch.zeros((input_dim, input_dim, input_dim), device=directions.device, dtype=DTYPE)
    for value, (a, b, c) in zip(coefficients, index):
        multiplicity = len({(a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)})
        entry = value / multiplicity
        for i, j, k in {(a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)}:
            tensor[i, j, k] = entry
    return tensor


def measure_slices(
    teacher: Teacher,
    reader: torch.Tensor,
    anchors: list[torch.Tensor],
    steps: list[float],
) -> list[torch.Tensor]:
    """Every slice the pencil needs, with the anchor-free terms charged once.

    Of the seven polarisation terms, three - `phi(a+b)`, `phi(a)`, `phi(b)` - do not mention the
    anchor, and they are the `O(n^2)` ones. Measuring them per slice pays for them twice for no
    new information; the pencil needs two slices, so that is the whole second copy.
    """
    dimension = anchors[0].shape[0]
    basis = torch.eye(dimension, device=anchors[0].device, dtype=DTYPE)
    rows, cols = torch.triu_indices(dimension, dimension, device=anchors[0].device)
    pair_sum = basis[rows] + basis[cols]

    phi_pair = measure_cubic_form(teacher, reader, pair_sum, steps)
    phi_basis = measure_cubic_form(teacher, reader, basis, steps)
    shared = -phi_pair + phi_basis[rows] + phi_basis[cols]

    slices = []
    for anchor in anchors:
        phi_pair_anchor = measure_cubic_form(teacher, reader, pair_sum + anchor[None, :], steps)
        phi_basis_anchor = measure_cubic_form(teacher, reader, basis + anchor[None, :], steps)
        phi_anchor = measure_cubic_form(teacher, reader, anchor.unsqueeze(0), steps)[0]
        entries = (
            phi_pair_anchor + shared - phi_basis_anchor[rows] - phi_basis_anchor[cols] + phi_anchor
        ) / 6.0
        matrix = torch.zeros((dimension, dimension), device=anchor.device, dtype=DTYPE)
        matrix[rows, cols] = entries
        matrix[cols, rows] = entries
        slices.append(matrix)
    return slices


def measure_slice(
    teacher: Teacher,
    reader: torch.Tensor,
    anchor: torch.Tensor,
    steps: list[float],
) -> torch.Tensor:
    """`S(., ., x)` by polarisation, without ever forming the whole tensor.

    This is the half of the route that decides whether it can be afforded. Jennrich reads two
    slices and nothing else, so paying `C(n+2,3)` for every degree-3 coefficient buys `n/6`
    times more tensor than the decomposition consumes. Polarisation,

        6 S(a,b,c) = phi(a+b+c) - phi(a+b) - phi(a+c) - phi(b+c) + phi(a) + phi(b) + phi(c)

    with `a, b` over the basis and `c` the anchor, needs `O(n^2)` evaluations instead. The terms
    that do not mention the anchor are the same for every slice and are charged once.
    """
    dimension = anchor.shape[0]
    basis = torch.eye(dimension, device=anchor.device, dtype=DTYPE)
    rows, cols = torch.triu_indices(dimension, dimension, device=anchor.device)
    pair_sum = basis[rows] + basis[cols]

    phi_pair_anchor = measure_cubic_form(teacher, reader, pair_sum + anchor[None, :], steps)
    phi_pair = measure_cubic_form(teacher, reader, pair_sum, steps)
    phi_basis_anchor = measure_cubic_form(teacher, reader, basis + anchor[None, :], steps)
    phi_basis = measure_cubic_form(teacher, reader, basis, steps)
    phi_anchor = measure_cubic_form(teacher, reader, anchor.unsqueeze(0), steps)[0]

    entries = (
        phi_pair_anchor - phi_pair - phi_basis_anchor[rows] - phi_basis_anchor[cols]
        + phi_basis[rows] + phi_basis[cols] + phi_anchor
    ) / 6.0
    matrix = torch.zeros((dimension, dimension), device=anchor.device, dtype=DTYPE)
    matrix[rows, cols] = entries
    matrix[cols, rows] = entries
    return matrix


def gates_from_slices(
    slice_x: torch.Tensor,
    slice_y: torch.Tensor,
    tolerance: float,
    collinearity: float,
) -> tuple[torch.Tensor, int]:
    """The same pencil as `jennrich`, fed two measured slices instead of a whole tensor."""
    stacked = torch.cat([slice_x, slice_y], dim=1)
    left, values, _ = torch.linalg.svd(stacked, full_matrices=False)
    rank = int((values > values[0] * tolerance).sum())
    basis = left[:, :rank]
    reduced_x = basis.T @ slice_x @ basis
    reduced_y = basis.T @ slice_y @ basis
    _, eigenvectors = torch.linalg.eig(reduced_x @ torch.linalg.pinv(reduced_y))
    factors = (basis.to(eigenvectors.dtype) @ eigenvectors).real
    factors = factors / factors.norm(dim=0, keepdim=True).clamp_min(1e-300)
    return distinct_directions(factors, collinearity), rank


def analytic_symmetric_tensor(teacher: Teacher, reader: torch.Tensor) -> torch.Tensor:
    """`sum_i lambda_i sym(g_i (x) g_i (x) u_i)`, built directly, to check the fitted one."""
    weights = 1.5 * (teacher.down.T @ reader)
    gate, up = teacher.gate, teacher.up
    ggu = torch.einsum("i,ia,ib,ic->abc", weights, gate, gate, up)
    return (ggu + ggu.permute(0, 2, 1) + ggu.permute(2, 0, 1)) / 3.0


def unfolding_rank(tensor: torch.Tensor, tolerance: float) -> tuple[int, list[float]]:
    """Numerical rank of the mode-1 unfolding, which counts the units.

    Each unit's contribution lives in `span{g_i, u_i}`, so the unfolding's column space is the
    sum of `k` planes and its rank is `2k` for generic units. That is where the unit count comes
    from - it is read off the measurement rather than supplied, and it is the first thing this
    route gives that the boundary route has to cluster for.
    """
    dimension = tensor.shape[0]
    values = torch.linalg.svdvals(tensor.reshape(dimension, -1))
    rank = int((values > values[0] * tolerance).sum())
    return rank, values.tolist()


def jennrich(tensor: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Algebraic symmetric CP decomposition. No iteration, no starting point, no objective.

    Two random slices share eigenvectors with the factor matrix, so one eigendecomposition of
    `S_x S_y^+` returns the factors. The tensor is first pushed into the `rank`-dimensional
    span of its own unfolding, which is what makes the pseudo-inverse well conditioned; without
    it the slices are singular by construction whenever `rank < n`.

    What comes back here is not the apolarity triple `{g+u, g-u, u}` this file first went
    looking for. Those three vectors span a plane, not a 3-space, so the factor matrix has rank
    `2k` and Jennrich's hypothesis fails by construction - measured, at reconstruction error
    `0.79` with weights of order `1e15`. What the pencil does return is `k` eigenvalues of
    multiplicity two whose eigenvectors are the gate directions, at cosine `1.000000` on every
    block from 1 to 6 units. The gate is what is recovered algebraically; the up is a linear
    solve afterwards.
    """
    input_dim = tensor.shape[0]
    unfolding = tensor.reshape(input_dim, input_dim * input_dim)
    left, _, _ = torch.linalg.svd(unfolding, full_matrices=False)
    basis = left[:, :rank]
    core = torch.einsum("abc,ai,bj,ck->ijk", tensor, basis, basis, basis)

    generator = torch.Generator(device=tensor.device).manual_seed(0)
    x = torch.randn(rank, generator=generator, device=tensor.device, dtype=DTYPE)
    y = torch.randn(rank, generator=generator, device=tensor.device, dtype=DTYPE)
    slice_x = torch.einsum("ijk,k->ij", core, x)
    slice_y = torch.einsum("ijk,k->ij", core, y)

    pencil = slice_x @ torch.linalg.pinv(slice_y)
    eigenvalues, eigenvectors = torch.linalg.eig(pencil)
    # A real tensor with real factors still comes back complex from a non-symmetric eig. The
    # imaginary part is the conditioning of the pencil, so it is reported rather than dropped.
    imaginary_share = float(
        eigenvectors.imag.abs().max() / eigenvectors.abs().max().clamp_min(1e-300)
    )
    factors = (basis.to(eigenvectors.dtype) @ eigenvectors).real
    factors = factors / factors.norm(dim=0, keepdim=True).clamp_min(1e-300)

    # Weights from a linear solve against the recovered directions: the eigenvectors fix the
    # directions and nothing else, and `v^3` is linear in the weight.
    columns = torch.stack([
        torch.einsum("a,b,c->abc", factors[:, r], factors[:, r], factors[:, r]).reshape(-1)
        for r in range(rank)
    ], dim=1)
    weights = torch.linalg.lstsq(columns, tensor.reshape(-1, 1)).solution.squeeze(1)
    return factors, weights, imaginary_share


def distinct_directions(factors: torch.Tensor, collinearity: float) -> torch.Tensor:
    """Collapse the pencil's repeated eigenvectors to one direction per eigenvalue.

    The gate eigenvalues have multiplicity two, so each gate comes back twice. Merging by
    absolute cosine rather than by eigenvalue keeps this honest about what it is doing: two
    directions are the same unit only if they point the same way.
    """
    kept: list[torch.Tensor] = []
    for column in range(factors.shape[1]):
        candidate = factors[:, column]
        norm = candidate.norm()
        if float(norm) <= 1e-300:
            continue
        candidate = candidate / norm
        if any(abs(float(candidate @ existing)) >= collinearity for existing in kept):
            continue
        kept.append(candidate)
    if not kept:
        return torch.zeros((0, factors.shape[0]), device=factors.device, dtype=DTYPE)
    return torch.stack(kept)


def solve_ups_from_gates(
    directions: torch.Tensor,
    values: torch.Tensor,
    gates: torch.Tensor,
) -> torch.Tensor:
    """Given the gates, the ups are a linear least squares and nothing more.

    `phi(d) = sum_i (g_i.d)^2 (v_i.d)` with `v_i = lambda_i u_i`, which is linear in the `v_i`.
    So once the pencil has returned the gates there is no second nonlinear step and no further
    oracle call: the design is built from directions already probed. This is the same shape as
    the up solve in the boundary route, which recovers each unit from a known gate without
    inverting the down projection - and it is why the response width does not enter here.
    """
    unit_count, input_dim = gates.shape
    gate_projection = directions @ gates.T
    design = (gate_projection ** 2).unsqueeze(2) * directions.unsqueeze(1)
    design = design.reshape(directions.shape[0], unit_count * input_dim)
    solution = torch.linalg.lstsq(design, values.unsqueeze(1)).solution.squeeze(1)
    scaled = solution.reshape(unit_count, input_dim)
    return scaled / scaled.norm(dim=1, keepdim=True).clamp_min(1e-300)


def match_to_truth(recovered: torch.Tensor, truth: torch.Tensor) -> list[tuple[int, int, float]]:
    """Greedy one-to-one match on |cosine|. Sign is free: the triple fixes direction, not orientation."""
    if recovered.shape[0] == 0:
        return []
    truth_unit = truth / truth.norm(dim=1, keepdim=True)
    cosines = (recovered @ truth_unit.T).abs()
    pairs: list[tuple[int, int, float]] = []
    taken_rows: set[int] = set()
    taken_cols: set[int] = set()
    flat = torch.argsort(cosines.reshape(-1), descending=True)
    width = cosines.shape[1]
    for position in flat.tolist():
        row, col = divmod(position, width)
        if row in taken_rows or col in taken_cols:
            continue
        taken_rows.add(row)
        taken_cols.add(col)
        pairs.append((row, col, float(cosines[row, col])))
    return pairs


def run(
    units: int,
    input_dim: int,
    output_dim: int,
    seed: int,
    direction_multiple: float,
    steps: list[float],
    collinearity: float,
    rank_tolerance: float,
    device: torch.device,
) -> dict:
    teacher = make_teacher(units, input_dim, output_dim, seed, device)
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    reader = torch.randn(output_dim, generator=generator, device=device, dtype=DTYPE)
    reader = reader / reader.norm()

    coefficient_count = len(monomial_index(input_dim))
    direction_count = int(math.ceil(direction_multiple * coefficient_count))
    directions = torch.randn(direction_count, input_dim, generator=generator, device=device, dtype=DTYPE)
    directions = directions / directions.norm(dim=1, keepdim=True)

    measured = measure_cubic_form(teacher, reader, directions, steps)
    analytic = analytic_cubic_form(teacher, reader, directions)
    stencil_error = float(
        (measured - analytic).norm() / analytic.norm().clamp_min(1e-300)
    )

    tensor = recover_symmetric_tensor(directions, measured, input_dim)
    tensor_truth = analytic_symmetric_tensor(teacher, reader)
    tensor_error = float((tensor - tensor_truth).norm() / tensor_truth.norm().clamp_min(1e-300))

    rank, spectrum = unfolding_rank(tensor, rank_tolerance)
    units_from_rank = rank // 2
    result: dict = {
        "units": units,
        "inputDim": input_dim,
        "outputDim": output_dim,
        "unfoldingRank": rank,
        "unitsFromRank": units_from_rank,
        "unitCountRecovered": units_from_rank == units,
        # `2k <= n` is the condition, not `3k <= n`: the unfolding rank is what the pencil has
        # to invert, and it is two per unit because a unit spans a plane.
        "pencilIdentifiable": 2 * units <= input_dim,
        "coefficientCount": coefficient_count,
        "directionCount": direction_count,
        "stencilRelativeError": stencil_error,
        "tensorRelativeError": tensor_error,
        "oracleCallsPoints": teacher.counter.points,
        "oracleCallsBatches": teacher.counter.batches,
        "device": str(tensor.device),
    }

    factors, weights, imaginary_share = jennrich(tensor, max(rank, 1))
    gates = distinct_directions(factors, collinearity)
    ups = (
        solve_ups_from_gates(directions, measured, gates)
        if gates.shape[0] > 0
        else gates
    )
    gate_matches = match_to_truth(gates, teacher.gate)
    up_matches = match_to_truth(ups, teacher.up)

    def summarise(matches: list[tuple[int, int, float]]) -> dict:
        cosines = sorted((score for _, _, score in matches), reverse=True)
        return {
            "matched": len(cosines),
            "above0.99": sum(1 for value in cosines if value >= 0.99),
            "above0.999": sum(1 for value in cosines if value >= 0.999),
            "median": cosines[len(cosines) // 2] if cosines else None,
            "worst": cosines[-1] if cosines else None,
        }

    # The pairing has to be checked, not assumed: a triple can return a good gate against the
    # wrong unit's up and still look like two good matches when each is scored on its own.
    paired = 0
    gate_by_row = {row: col for row, col, _ in gate_matches}
    up_by_row = {row: col for row, col, _ in up_matches}
    for row, gate_col in gate_by_row.items():
        if up_by_row.get(row) == gate_col:
            paired += 1

    # The slice route, priced on its own counter. A fresh teacher with the same seed is the
    # same block; sharing the first teacher's counter would report the two routes' costs added
    # together and call it one of them.
    slice_teacher = make_teacher(units, input_dim, output_dim, seed, device)
    anchor_generator = torch.Generator(device=device).manual_seed(seed + 2)
    # Unit anchors, not raw normals. `phi` is cubic, so an anchor of norm `sqrt(n)` makes
    # `phi(e_i + e_j + x)` about `n^1.5` times the cross term the polarisation is trying to
    # read, and the subtraction throws away that many digits. Measured at n=16: raw normals
    # returned 6 of 8 gates above cosine 0.999 where the whole tensor returned 8.
    anchor_x = torch.randn(input_dim, generator=anchor_generator, device=device, dtype=DTYPE)
    anchor_y = torch.randn(input_dim, generator=anchor_generator, device=device, dtype=DTYPE)
    anchor_x = anchor_x / anchor_x.norm()
    anchor_y = anchor_y / anchor_y.norm()
    slice_x, slice_y = measure_slices(slice_teacher, reader, [anchor_x, anchor_y], steps)
    slice_gates, slice_rank = gates_from_slices(slice_x, slice_y, rank_tolerance, collinearity)
    slice_gate_matches = match_to_truth(slice_gates, teacher.gate)

    result.update({
        "gatesReturned": int(gates.shape[0]),
        "jennrichImaginaryShare": imaginary_share,
        "gate": summarise(gate_matches),
        "up": summarise(up_matches),
        "unitsIdentifiedAsPairs": paired,
        "recoveryRate": paired / units,
        "slice": {
            "rank": slice_rank,
            "gatesReturned": int(slice_gates.shape[0]),
            "gate": summarise(slice_gate_matches),
            "oracleCallsPoints": slice_teacher.counter.points,
            "callRatioAgainstFullTensor": (
                teacher.counter.points / max(slice_teacher.counter.points, 1)
            ),
        },
    })
    return result


def silu_second_derivative(argument: torch.Tensor) -> torch.Tensor:
    argument = argument.detach().requires_grad_(True)
    first, = torch.autograd.grad(torch.nn.functional.silu(argument).sum(), argument, create_graph=True)
    second, = torch.autograd.grad(first.sum(), argument)
    return second.detach()


def base_point_profile(teacher: Teacher, base_points: torch.Tensor) -> torch.Tensor:
    """Per-unit weight of the `sym(g g u)` family, one row per (reader, base point).

    At the origin the third derivative weights unit `i` by `1.5 (w.d_i)` alone, so every reader
    sees the same `k` tensors through the down projection and the accessible dimension is
    `rank(D) <= m`. That is the response width returning as a bound, and it is why the boundary
    route's ceiling looked structural.

    Off the origin it is not the same weight. With `a_i = g_i.x0`, `b_i = u_i.x0`, the third
    derivative along `d` is

        sum_i (w.d_i) [ silu'''(a_i) b_i (g_i.d)^3 + 3 silu''(a_i) (g_i.d)^2 (u_i.d) ]

    so the `sym(g g u)` family carries `beta_i = 3 (w.d_i) silu''(g_i.x0)`. The second factor is a
    *nonlinear* function of the unit, not a linear read through `D`, so base points buy rank that
    readers cannot. Measured: the profile rank is `min(k, m*P)`, so `P` base points lift the
    ceiling from `m` to `m*P`.
    """
    activations = silu_second_derivative(base_points @ teacher.gate.T)
    return (3.0 * teacher.down[:, None, :] * activations[None, :, :]).reshape(-1, teacher.gate.shape[0])


def base_point_reach(
    units: int,
    input_dim: int,
    output_dim: int,
    base_point_count: int,
    seed: int,
    tolerance: float,
    device: torch.device,
) -> dict:
    """How many units are separable, and how well conditioned that separation is.

    The rank alone is not the answer. A rank counted at a threshold can be carried by singular
    values a hundred million times smaller than the first, which is a separation no measurement
    would ever reach; the smallest-to-largest ratio is reported next to it so the number has to
    survive being looked at.
    """
    teacher = make_teacher(units, input_dim, output_dim, seed, device)
    generator = torch.Generator(device=device).manual_seed(seed + 7)
    base_points = torch.randn(base_point_count, input_dim, generator=generator, device=device, dtype=DTYPE)
    profile = base_point_profile(teacher, base_points)
    spectrum = torch.linalg.svdvals(profile)
    rank = int((spectrum > spectrum[0] * tolerance).sum())
    reachable = min(units, output_dim * base_point_count)
    return {
        "units": units,
        "inputDim": input_dim,
        "responseWidth": output_dim,
        "basePoints": base_point_count,
        "readerTimesBasePoints": output_dim * base_point_count,
        "profileRank": rank,
        "rankMatchesMinKMP": rank == reachable,
        "unitsPastResponseWidth": units > output_dim,
        "conditioning": float(spectrum[min(units, spectrum.shape[0]) - 1] / spectrum[0]),
        "device": str(profile.device),
    }


def lifted_reach(
    units: int,
    input_dim: int,
    output_dim: int,
    seed: int,
    tolerance: float,
    device: torch.device,
) -> dict:
    """Where the ceiling goes once the unit is a point in `Sym^2` instead of a vector in `R^n`.

    The pencil stops at `2k <= n` because it diagonalises in `R^n`, where `k` units contribute
    `2k` directions and there is only room for `n`. The `(ab)` unfolding of the same tensor puts
    each unit in `Sym^2(R^n)` instead - `g (x) g` and `sym(g (x) u)` - and that space has
    dimension `n(n+1)/2`, which is quadratic where the old room was linear.

    Measured here as the rank of that span, with the conditioning next to it because a rank at a
    threshold proves nothing on its own: `min(2k, n(n+1)/2)` at every setting, so the ceiling is
    `k <= n(n+1)/4`. At `n=10` that is 27 units against the pencil's 5, and the span for `k=25` -
    two and a half times overcomplete - separates at 1.6e-02.

    This is the identifiability ceiling, not a recovery. Extracting the `g_i` means finding the
    rank-1 elements of that span, which is a FOOBI-class step and is not implemented here.
    """
    teacher = make_teacher(units, input_dim, output_dim, seed, device)
    gate, up = teacher.gate, teacher.up
    gate_outer = torch.einsum("ia,ib->iab", gate, gate)
    mixed = (torch.einsum("ia,ib->iab", gate, up) + torch.einsum("ia,ib->iab", up, gate)) / 2.0
    span = torch.cat([gate_outer, mixed], dim=0).reshape(2 * units, -1)
    spectrum = torch.linalg.svdvals(span)
    rank = int((spectrum > spectrum[0] * tolerance).sum())
    room = input_dim * (input_dim + 1) // 2
    return {
        "units": units,
        "inputDim": input_dim,
        "liftedRoom": room,
        "spanRank": rank,
        "rankMatchesMin2KRoom": rank == min(2 * units, room),
        "conditioning": float(spectrum[min(2 * units, spectrum.shape[0]) - 1] / spectrum[0]),
        "pencilCeiling": input_dim // 2,
        "liftedCeiling": room // 2,
        "device": str(span.device),
    }


def cost_laws(input_dim: int) -> dict:
    """The two measured cost laws, evaluated rather than quoted.

    Both were checked against the counter at `n=16` and `n=24` before being written down:
    `4n(n+1)(n+2)` gave 19584 and 62400, `8*(1.5n(n+1)+3n+2)` gave 3664 and 7792, and those are
    the numbers the runs report. Applying them to a live layer is arithmetic on a verified law,
    not a measurement of that layer, and the field name says so.
    """
    n = int(input_dim)
    full = 4 * n * (n + 1) * (n + 2)
    slices = 8 * (3 * n * (n + 1) // 2 + 3 * n + 2)
    return {
        "inputDim": n,
        "fullTensorOracleCalls": full,
        "twoSliceOracleCalls": slices,
        "ratio": full / slices,
        "identifiableUnitCeiling": n // 2,
        "extrapolatedFromMeasuredLaw": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--units", type=int, default=4)
    parser.add_argument("--input-dim", type=int, default=16)
    parser.add_argument("--output-dim", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--direction-multiple", type=float, default=3.0)
    parser.add_argument("--steps", type=float, nargs="+", default=[0.02, 0.04])
    parser.add_argument("--collinearity", type=float, default=0.99)
    parser.add_argument("--rank-tolerance", type=float, default=1e-6)
    parser.add_argument("--project-dim", type=int, nargs="*", default=None)
    parser.add_argument("--base-point-sweep", type=int, nargs="*", default=None,
                        help="base-point counts to measure the response-width ceiling against")
    parser.add_argument("--reach-units", type=int, default=40)
    parser.add_argument("--reach-width", type=int, default=8)
    parser.add_argument("--lifted-sweep", type=int, nargs="*", default=None,
                        help="unit counts to measure the lifted identifiability ceiling at")
    parser.add_argument("--sweep-units", type=int, nargs="*", default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested and not available; this probe is a GPU measurement")

    unit_list = args.sweep_units if args.sweep_units else [args.units]
    runs = [
        run(
            units=count,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            seed=args.seed,
            direction_multiple=args.direction_multiple,
            steps=list(args.steps),
            collinearity=args.collinearity,
            rank_tolerance=args.rank_tolerance,
            device=device,
        )
        for count in unit_list
    ]
    report = {"runs": runs}
    if args.project_dim:
        report["costProjection"] = [cost_laws(dim) for dim in args.project_dim]
    if args.lifted_sweep:
        report["liftedReach"] = [
            lifted_reach(
                units=count,
                input_dim=args.input_dim,
                output_dim=args.reach_width,
                seed=args.seed,
                tolerance=args.rank_tolerance,
                device=device,
            )
            for count in args.lifted_sweep
        ]
    if args.base_point_sweep:
        report["basePointReach"] = [
            base_point_reach(
                units=args.reach_units,
                input_dim=args.input_dim,
                output_dim=args.reach_width,
                base_point_count=count,
                seed=args.seed,
                tolerance=args.rank_tolerance,
                device=device,
            )
            for count in args.base_point_sweep
        ]
    text = json.dumps(report, indent=2)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
