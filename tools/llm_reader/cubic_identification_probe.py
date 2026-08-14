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
recovering the `g_i` means finding the rank-1 elements of that lifted span.

A correction to the paragraph above, from trying it. Full span rank says the units are
*distinguishable*; it does not say the extraction has isolated answers. Rank-1 matrices form a
variety of projective dimension `n-1`, and membership in `W` imposes `n(n+1)/2 - 2k` conditions,
so solutions are isolated only for `k <= (n^2-n+2)/4` - 23 at `n=10` where the rank ceiling said
27, and 2358528 rather than 2360064 at `n=3072`. The difference is small and the reason is not:
at `k=25`, `n=10`, the search returned 61 distinct directions and none of them was a gate.

Three routes to the extraction were tried and none of them recovers `k > n`. The objective is
right - `f(v) = ||(I - P_W) vec(v v^T)||^2` reads `1e-16` on true gates against `0.5` on random
directions, and `g g^T` sits in `W` to `2e-16` - so what is missing is a solver, not a
formulation. Projected gradient descent on that quartic returns 2 to 5 of `k` from thousands of
restarts; the landscape is flat and most restarts stall above zero. Rayleigh iteration on
`A(v) = mat((I - P_W) vec(v v^T))` fails for a structural reason worth keeping: at a solution
*every* eigenvalue of `A` is `~1e-16`, so the smallest eigenvector is noise and the fixed point
carries no direction - measured, cosine 0.397 against the gate it had converged to. And
localising with a distant base point, which does thin the active set - participation ratio falls
from 64 to 9.2 as the base point norm goes 0 to 50 at `k=64`, `n=16` - does not drop the
*unfolding rank*, which stays saturated at `n` because the quiet units are quiet and not absent.
That route returned 2 of 6 at its best setting.

The solver that works, and why the three above could not. All of them searched for `v` directly,
which makes rank-1-ness a quartic condition with a flat landscape and, at the solution, no
second-order information left to steer by. Rank is a statement about minors, and minors are
bilinear:

    Phi(M,N)_ijkl = 1/2 (M_ij N_kl + N_ij M_kl - M_il N_kj - N_il M_kj)

vanishes identically at `M = N = v v^T`. So `Phi(M(c), M(c)) = 0` is *linear* in `c (x) c`, and
the search is a null space rather than an optimisation. The null space is spanned by
`c^(i) (x) c^(i)`, which are rank-1 in the `2k`-dimensional coefficient space - so the same
pencil that failed in `R^n` separates them there, where the objects really are rank-1. Each `c`
rebuilds a matrix whose top eigenvector is a gate, and a purity test drops anything that did not
come back rank-1. Nothing iterates.

Measured at `n=8`, where the pencil ceiling is 4 units:

| units `k` | `k/n` | recovered | median cosine |
| --- | --- | --- | --- |
| 8 | 1.00 | 8/8 | 1.000000000000000 |
| 10 | 1.25 | 10/10 | 0.999999999999999 |
| 12 | 1.50 | 12/12 | 1.000000000000000 |
| 13 | 1.63 | 13/13 | 0.999999999999999 |
| 14 | 1.75 | 0/14 | - |

Three seeds each, and the seeds agree at every row including the failure.

That last sentence used to read that the solver runs out exactly where the geometry does, and
scanning `n=10` showed it does not. There the last working count is 20 and 21 fails - three
seeds each way - while the isolation bound allows 23. So the solver stops one below the bound at
`n=8` and three below it at `n=10`: near the geometry, not at it, and two points are not a law.

What says when to trust a run, without any truth to compare against, is the count: the span has
rank `2k` when the units are separable, so a healthy run returns exactly that many halves.
Measured, it is `True` at every recovering row and `False` at every failing one - 13 of 13
returned at `n=8, k=13` against 0 at `k=14`, 20 against 0 at `n=10`, and 30 and 40 returned at
`n=16`.

Where it stops now, and the way past it. The Gram is `(2k)^2` square, so `k=50` peaks at 4.49
GiB and returns 50 of 50 while `k=64` runs out of card in the eigendecomposition. Pushing `k`
further is the wrong move: the answer is fewer units per solve. A distant base point weights unit
`i` by `silu''(g_i.x0)`, which decays, so units below the measurement floor drop out of the
*measured* span and the rank falls with them. Measured at `n=16`, `k=64`, four base points and
four readers: rank 128 at scales 1 through 25 with all 64 units loud, then 123 at scale 50 where
61 are, and 99 at scale 100 where 44 are. The rank tracks the active count exactly at scale 50.
That is the subset mechanism a full layer needs, and it is now the pipeline: `--tile` walks base
points, solves whatever subset each one presents, and merges by cosine between recovered
directions with no truth involved. At `n=16` with `k=64` - four times the input dimension and
eight times the response width, a size no single solve can hold - twenty base points and 391680
oracle points cover 64 of 64, merging 67 distinct directions of which one is spurious. Coverage
climbs 15, 22, 32, 41, 47, 52 over the first six and closes from there.

How that scales, and where it does not. The distance is chosen by probe rather than by hand,
since the loud count at a given distance grows with the layer. Measured at `n=16`, rank cap 45:

| units | scale | base points | covered | spurious | oracle points |
| --- | --- | --- | --- | --- | --- |
| 16 | 4 | 10 | 16/16 | 0 | 215424 |
| 32 | 4 | 15 | 32/32 | 0 | 313344 |
| 64 | 60 | 20 | 63/64 | 2 | 450432 |
| 128 | 120 | 30 | 120/128 | 4 | 665856 |

`k=128` is eight times the input dimension and sixteen times the response width. Coverage decays
slowly while base points grow about `k/4`. The probe rule is not optimal and the table says so:
at `k=64` it takes 60 and covers 63, where the hand-tuned 120 covered all 64 - the first distance
that fits is not the one that separates best.

The per-base-point cost is now the slice route, and the saving is measured rather than promised.
Fitting every degree-3 coefficient is `4n^3`; the span only needs enough slices to reach
`2k_loud`, and each by polarisation is `O(n^2)`, so the ratio is linear in `n` and crosses at 15
for this anchor count. Measured against the same span, subspace angle `1.000000000` at each:

| n | slices | tensor | ratio |
| --- | --- | --- | --- |
| 24 | 38992 | 62400 | 1.60x |
| 32 | 67312 | 143616 | 2.13x |
| 48 | 146992 | 470400 | 3.20x |
| 64 | 257392 | 1098240 | 4.27x |
| 96 | 570352 | 3650304 | 6.40x |

At `n=3072` that is `204.8x` - `5.7e8` points a base point instead of `1.16e11`. The tensor route
also runs out of card at `n=24` where the slice route does not, so this is not only cheaper.

Anchors are the expensive axis and readers are free, which inverts what the response width means
here. The oracle answers `m` numbers per call whether one is read or all of them, so one anchor
already yields `R` slices - but only `k_loud` independent ones, since they share an anchor. Two
anchors are the floor and `A * R >= 2 k_loud` is the condition. Measured at `n=32`, `k=12`: `A=2`
with 24 readers reaches rank 24 for 13456 points where `A=14` with 3 readers needs 67312 for the
same rank, and `A=1` stops at 12 however many readers it gets. Through the whole pipeline at
`n=32`, `k=32`, covering 32 of 32 each time:

| response width | readers | anchors | oracle points |
| --- | --- | --- | --- |
| 8 | 8 | 8 | 915456 |
| 24 | 24 | 3 | 646176 |
| 48 | 48 | 2 | 592320 |

A wider answer now makes the read *cheaper*, where in the boundary route it was the ceiling.

Probing inside a subspace cuts the per-solve cost again, and hands back something less than a
unit. Restricting the input to a random `d`-dimensional `V` leaves a SwiGLU with gates `P_V g_i`,
so everything above applies with `n` replaced by `d` and a base point costs `12 d^2`. Measured at
`n=64`, `k=64`, `m=32`, base scale 150: `d=24` returns 12 gates all matching the projected truth
at median cosine `0.999966` for 10392 points, and `d=16` returns 12 at `0.999849` for 4888 -
`4.7x` and `10.1x` against probing the full input.

What comes back is `P_V g_i`, so the subspaces have to be stitched, and that is now implemented.
Give every `V_t = A + W_t` the same core `A` with the `W_t` an orthogonal partition of the rest:
`P_A P_{V_t} = P_A`, so the core block of a recovery is the same vector for every `t` up to
scale - the signature that groups recoveries into units, and the length that puts blocks on one
scale. The partition makes the assembly exact rather than a fit. Measured at `n=64`, `k=64`,
`d=16`, `e=8`, seven blocks: 5 groups cover every block and all 5 assemble to full-dimension
gates above cosine 0.99, median `0.999802`, for 34216 points - 6843 a gate.

Blocks merge across core points, which is what turns a product into a union. A unit's block
component `P_{W_t} g` does not depend on where the base point sits, and neither does its
signature `P_A g`, so block 3 recovered at one core point and block 5 at another are two pieces
of the same vector. Requiring all blocks at a single core point multiplies a per-block recall of
about 0.7 seven times over; drawing from several core points and grouping everything by signature
does not. Measured at `n=64`, `k=64`, `d=16`, `e=8`:

| core points | assembled | correct | median cosine | oracle points | per gate |
| --- | --- | --- | --- | --- | --- |
| 1 | 3 | 3 | 0.999745 | 34216 | 11405 |
| 3 | 12 | 12 | 0.999800 | 102648 | 8554 |
| 6 | 27 | 27 | 0.999795 | 205296 | 7603 |

Every assembled direction is correct at every setting, and the cost per gate *falls* as core
points are added, because the union grows faster than the linear cost of collecting it.

What limits this is per-block recall, and the measurement says where it is not. At a shared base
point 19 of 64 units are loud, the measured span in each block has rank 45 to 50 - so the span
holds them - and every direction returned is correct. The solve returns 8 to 11 of about 23. So
the loss is the solver's recall, not the localisation and not the span. Drawing several random
pencils instead of one recovers strictly more of the same span at no cost in precision: 10
directions all correct from one draw, 15 all correct from five. Lowering the purity test buys
recall with precision instead, which is not a trade worth making - 0.7 gives 23 at precision
0.870 and 0.3 gives 77 at 0.338. Through the pipeline that is 67 local recoveries to 91, five
assembled gates to six, and 6843 points a gate to 5702.

The base point has to live in the core, and that was the difference between working and not.
With a fresh base point per subspace each one localises onto a different subset, so a unit is
almost never seen twice: 52 groups from 67 recoveries, nothing assembled above 0.85. In the core
it is the same point in the full space for every solve and the loud set is shared.

Whether this is a saving depends on `n`. Carrying the `n/d` subspaces a unit must appear in,
subspace probing is about `k n d` against `0.27 k n^2` for the full input - a ratio of `3.7 d/n`,
so `0.93` at `n=64` where it buys nothing and the measured 6843 a gate against about 4096 for
full-input probing says the same, then `0.23` at 256 and `0.019` at 3072. A large-`n` mechanism,
now built rather than argued.

It is still not a layer, and the arithmetic should be read rather than rounded off. At `n=3072`
the response width pins anchors at the floor of two, so a base point is about `1.1e8` points, and
base points grow about `k/4` - 2048 of them for 8192 units, so `1.9e11`. The boundary route spent
`4.5e7` to fail at 4096. So what is measured is that the identification holds to sixteen times
the response width and that a base point costs `12 n^2`; not that a layer of 8192 has been read,
and not that this could afford to.

Scale, which is what kept this at toy sizes. The minor map has `n^4` rows - 1.8 GiB at `n=16`
and 15.8 GiB at `n=24`, where the card refused it. Only the null space is wanted and it lives in
the Gram matrix, which contracts to traces:
`<Phi(A,B),Phi(C,D)> = <A,C><B,D> + <A,D><B,C> - tr(ACBD) - tr(ADBC)`. Checked against the
explicit product at `n=8`: relative difference `5.9e-16`, nullity identical. Memory stops
depending on `n` - `n=24, k=30` now runs in 0.59 GiB and returns 30 of 30, and `n=16, k=40`
returns 40 of 40 at two and a half times overcomplete.

From the oracle, not from a span. The paragraph above used to end by admitting that `W` was
built analytically from the true gates and ups. It is now measured: third derivatives along
lines through several base points, every probe point charged to the counter, no weight read
anywhere. At `n=8` with four base points and two readers, 11520 oracle points return 13 of 13
gates at median cosine `0.999998`, and 10 units - already overcomplete - come back 10/10 for
8640. One bug had to be fixed to get there: the measured span and the minor null space were
sharing a rank threshold, and at `1e-8` the span admitted three noise directions as real ones,
reporting rank 27 where `2k = 24` and recovering nothing. They are separate numbers now, and the
span's is set by the measurement error rather than by arithmetic.

This is a measurement harness against a synthetic teacher, in the idiom of
`jacobian_jump_probe.py`. It is not wired into a lane. Run it directly.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any

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


class LiveTeacher:
    """The same call the synthetic teacher answers, served by the real block through the oracle.

    Everything above this line was measured against a teacher whose weights this file wrote. That
    makes every number here a statement about the method and none of them a statement about a
    layer. This class is the only place the two meet: it presents `__call__(points) -> values` and
    a `counter`, which is the whole surface `measure_cubic_forms_at` uses, and answers it by
    spending real oracle calls.

    Two things are not the same as the synthetic path and both matter. The transport is float32 -
    the client saves inputs as float32 and the block answers in it - while the third-derivative
    stencil divides by `2h^3`, so the relative error on a value is amplified by about `1/h^3`
    before anything downstream sees it. And the truth is locked: `gate` and `up` are unavailable
    until `unlock` is called, which is what keeps a comparison against them post-hoc.
    """

    def __init__(self, client: Any, device: torch.device) -> None:
        self.client = client
        self.device = device
        self.counter = OracleCallCounter()

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        self.counter.charge(int(points.shape[0]))
        values = self.client.eval(points.detach().cpu().numpy(), stage="cubic-probe")
        return torch.as_tensor(values, dtype=DTYPE, device=self.device)


def open_live_oracle(
    model: str,
    layer: int,
    units: str,
    device: str,
    max_calls: int,
    work_dir: Path,
    local_files_only: bool,
) -> Any:
    """Start the real block behind the same protocol the lanes use, and refuse a host answer."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import blind_swiglu_oracle_decompiler as live

    work_dir.mkdir(parents=True, exist_ok=True)
    arguments = argparse.Namespace(
        max_oracle_calls=int(max_calls),
        oracle_read_fd=-1,
        oracle_write_fd=-1,
        model=model,
        layer=int(layer),
        unit=0,
        units=units,
        device=device,
        oracle_batch_size=2048,
        output_channel=-1,
        oracle_output_mode="selected-unit-ffn-vector",
        oracle_quantization_step=0.0,
        local_files_only=bool(local_files_only),
        require_opaque_oracle=False,
    )
    return live.OracleClient(arguments, work_dir)


def live_form_check(
    model: str,
    layer: int,
    units: str,
    direction_count: int,
    subspace_dim: int,
    base_scale: float,
    steps: list[float],
    seed: int,
    max_calls: int,
    work_dir: Path,
    local_files_only: bool,
    device: torch.device,
) -> dict:
    """Does the third derivative survive the trip through a real, float32 oracle.

    This is the first gate that has to pass before any of the recovery above means anything about
    a layer, and it is cheap: `8` points a direction. It measures the cubic form the way the span
    builder does, then unlocks the post-hoc truth and computes what the form should have been.
    Agreement is the claim; the query count and the device the block answered from are recorded
    beside it so neither can be assumed later.
    """
    client = open_live_oracle(model, layer, units, str(device), max_calls, work_dir, local_files_only)
    try:
        hidden = int(client.meta["hiddenSize"])
        generator = torch.Generator(device=device).manual_seed(seed)
        frame, _ = torch.linalg.qr(
            torch.randn(hidden, subspace_dim, generator=generator, device=device, dtype=DTYPE)
        )
        base_point = torch.randn(subspace_dim, generator=generator, device=device, dtype=DTYPE)
        base_point = base_point / base_point.norm() * base_scale
        directions = torch.randn(direction_count, subspace_dim, generator=generator, device=device, dtype=DTYPE)
        directions = directions / directions.norm(dim=1, keepdim=True)

        teacher = LiveTeacher(client, device)
        lifted_base = frame @ base_point
        lifted_directions = directions @ frame.T

        # The step is the whole experiment, not a detail. Transport is float32, so a value
        # carries about `1e-7` relative error and the stencil divides by `2h^3`: the noise on the
        # estimate is roughly `eps * |y| / h^3` while the signal does not move with `h` at all.
        # At `h=0.02` that is 0.025 against a form of size 3e-5, which is why the first live check
        # came back at relative error 0.945 - not a broken bridge, a step three orders too small.
        # Every candidate is measured before the truth is unlocked, so the choice cannot have
        # been informed by the answer.
        step_sets = [list(steps)] if len(steps) > 1 else [[float(steps[0]), 2.0 * float(steps[0])]]
        step_sets = step_sets + [[0.5, 1.0], [2.0, 4.0], [8.0, 16.0], [32.0, 64.0]]
        measurements = [
            measure_cubic_forms_at(teacher, lifted_base, lifted_directions, candidate)
            for candidate in step_sets
        ]
        measured = measurements[0]
        solver_calls = int(client.query_count)

        # Only now, and the oracle enforces it rather than this file promising it.
        client.unlock_posthoc_truth("live-form-check")
        gate_truth, up_truth = client.truth()
        down_truth = client.truth_down()
        gate = torch.as_tensor(gate_truth, dtype=DTYPE, device=device)
        up = torch.as_tensor(up_truth, dtype=DTYPE, device=device)
        down = (
            torch.as_tensor(down_truth, dtype=DTYPE, device=device)
            if down_truth is not None
            else torch.eye(gate.shape[0], device=device, dtype=DTYPE)
        )
        truth_teacher = Teacher(gate=gate, up=up, down=down)
        analytic = torch.stack([
            analytic_cubic_form(truth_teacher, reader, lifted_directions)
            for reader in torch.eye(int(down.shape[0]), device=device, dtype=DTYPE)
        ], dim=1)

        scale = analytic.norm(dim=1).clamp_min(1e-300)
        step_report = []
        for candidate, estimate in zip(step_sets, measurements):
            errors = (estimate - analytic).norm(dim=1) / scale
            step_report.append({
                "steps": candidate,
                "relativeErrorMedian": float(errors.median()),
                "relativeErrorMin": float(errors.min()),
            })
        best = min(step_report, key=lambda entry: entry["relativeErrorMedian"])
        relative = (measured - analytic).norm(dim=1) / scale
        return {
            "stepSweep": step_report,
            "bestSteps": best["steps"],
            "bestRelativeErrorMedian": best["relativeErrorMedian"],
            "model": model,
            "layer": int(layer),
            "units": units,
            "hiddenSize": hidden,
            "blockDevice": client.meta.get("blockDevice"),
            "oracleResponseWidth": client.meta.get("oracleResponseWidth"),
            "subspaceDim": int(subspace_dim),
            "directions": int(direction_count),
            "baseScale": float(base_scale),
            "steps": list(steps),
            "solverOracleCalls": solver_calls,
            "truthUnlockStage": client.truth_unlock_stage,
            "truthUnlockQueryCount": client.truth_unlock_query_count,
            "unlockedAfterAllSolverCalls": client.truth_unlock_query_count == solver_calls,
            "analyticFormNormMedian": float(scale.median()),
            "relativeErrorMedian": float(relative.median()),
            "relativeErrorMax": float(relative.max()),
            "relativeErrorMin": float(relative.min()),
            "device": str(measured.device),
        }
    finally:
        client.close()


def live_subspace_solve(
    model: str,
    layer: int,
    units: str,
    subspace_dim: int,
    base_scale: float,
    reader_count: int,
    anchor_count: int,
    rank_target: int,
    steps: list[float],
    seed: int,
    tolerance: float,
    span_tolerance: float,
    purity: float,
    collinearity: float,
    pencil_draws: int,
    max_calls: int,
    work_dir: Path,
    local_files_only: bool,
    device: torch.device,
) -> dict:
    """One subspace of the real layer, solved: measured span, FOOBI, then the truth.

    The form check proved the third derivative survives the trip. This is the next thing that has
    to be true before a full stitch is worth its oracle bill: that the span measured through a
    live oracle still has the structure the extraction needs, and that the directions coming back
    are the layer's gates projected into the subspace rather than noise that happens to be
    rank-1.

    Nothing about the truth is touched until the recovery is frozen. The comparison afterwards
    scores what the solver already decided; it cannot steer it.
    """
    client = open_live_oracle(model, layer, units, str(device), max_calls, work_dir, local_files_only)
    try:
        hidden = int(client.meta["hiddenSize"])
        generator = torch.Generator(device=device).manual_seed(seed + 31)
        frame, _ = torch.linalg.qr(
            torch.randn(hidden, subspace_dim, generator=generator, device=device, dtype=DTYPE)
        )
        base_point = torch.randn(subspace_dim, generator=generator, device=device, dtype=DTYPE)
        base_point = base_point / base_point.norm() * base_scale
        readers = torch.randn(reader_count, hidden, generator=generator, device=device, dtype=DTYPE)
        readers = readers / readers.norm(dim=1, keepdim=True)

        # `anchors * readers` is a ceiling on the measured rank, and the span needs `2k`. Three
        # anchors against eight readers is 24, which is under the 32 a 16-unit block asks for, so
        # the first live run at that size returned a rank-24 span and zero directions - a
        # rank-deficient measurement that looked like the method failing. The count is derived
        # here for the same reason `stitch_recovery` derives it, and the ceiling is reported so a
        # span that ran out of slices says so instead of being read as a small block.
        anchor_count = max(anchor_count, -(-2 * rank_target // max(reader_count, 1)))
        slice_rank_ceiling = anchor_count * reader_count
        anchors = torch.randn(anchor_count, subspace_dim, generator=generator, device=device, dtype=DTYPE)
        anchors = anchors / anchors.norm(dim=1, keepdim=True)

        teacher = LiveTeacher(client, device)
        oracle = SubspaceOracle(teacher, frame)
        span_basis, rank = measured_lifted_span_by_slices(
            oracle, base_point, readers, anchors, subspace_dim, steps, span_tolerance
        )
        gates, nullity = foobi_gates(
            span_basis, subspace_dim, tolerance, purity, collinearity, seed + 7, attempts=pencil_draws
        )
        solver_calls = int(client.query_count)
        recovered = int(gates.shape[0])

        client.unlock_posthoc_truth("live-subspace-solve")
        gate_truth, _ = client.truth()
        truth = torch.as_tensor(gate_truth, dtype=DTYPE, device=device)
        projected = truth @ frame
        projected = projected / projected.norm(dim=1, keepdim=True).clamp_min(1e-300)
        if recovered:
            cosines = (gates @ projected.T).abs()
            per_unit = cosines.max(dim=0).values
            per_direction = cosines.max(dim=1).values
            covered = int((per_unit >= 0.99).sum())
            clean = int((per_direction >= 0.99).sum())
            median = float(per_direction.median())
        else:
            covered, clean, median = 0, 0, 0.0
        return {
            "model": model,
            "layer": int(layer),
            "units": units,
            "unitCount": int(truth.shape[0]),
            "hiddenSize": hidden,
            "blockDevice": client.meta.get("blockDevice"),
            "subspaceDim": int(subspace_dim),
            "baseScale": float(base_scale),
            "readers": int(reader_count),
            "anchors": int(anchor_count),
            "sliceRankCeiling": int(slice_rank_ceiling),
            "spanSaturatedAtCeiling": int(rank) >= int(slice_rank_ceiling),
            "steps": list(steps),
            "measuredSpanRank": int(rank),
            "spanRankIsTwoK": int(rank) == 2 * int(truth.shape[0]),
            "minorNullity": int(nullity),
            "directionsReturned": recovered,
            "directionsMatchingAUnit": clean,
            "unitsRecovered": covered,
            "unitCoverage": covered / max(int(truth.shape[0]), 1),
            "medianCosine": median,
            "solverOracleCalls": solver_calls,
            "truthUnlockQueryCount": client.truth_unlock_query_count,
            "unlockedAfterAllSolverCalls": client.truth_unlock_query_count == solver_calls,
            "device": str(span_basis.device),
        }
    finally:
        client.close()


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
        (eigenvectors.imag.abs().max() / eigenvectors.abs().max().clamp_min(1e-300)).detach()
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

    Two ceilings, and the difference between them is a correction to what this function used to
    report. The span having full rank `2k` says the units are *distinguishable*. It does not say
    the extraction has isolated answers. Rank-1 matrices form a variety of projective dimension
    `n-1` in `Sym^2`, and asking one to lie in `W` imposes `codim(W) = n(n+1)/2 - 2k` conditions,
    so the solutions are isolated only when `n - 1 <= n(n+1)/2 - 2k`, that is `k <= (n^2-n+2)/4`.
    Below that the two ceilings differ by one or two units; above it the search returns a
    continuum. Measured at `n=10`, where identifiability allows 27 and isolation allows 23: at
    `k=25` the extraction returned 61 distinct directions and none of them was a gate.
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
        # What the span can tell apart, and what a search can pin down, are different numbers.
        "isolatedSolutionCeiling": (input_dim * input_dim - input_dim + 2) // 4,
        "solutionsIsolated": 2 * units <= (input_dim * input_dim - input_dim + 2) // 2,
        "device": str(span.device),
    }


def measure_cubic_forms_at(
    teacher: Teacher,
    base_point: torch.Tensor,
    directions: torch.Tensor,
    steps: list[float],
) -> torch.Tensor:
    """Third derivative along lines through `base_point`, returned as a vector.

    The whole output vector comes back rather than one reader's projection, because readers are
    free once the points are paid for - the oracle answers `m` numbers per call whether one of
    them is read or all of them.

    The length rescale applies here too, and a comment in this file used to claim it did not. `y`
    is not homogeneous around a base point but its third *directional derivative* is: along
    `t -> y(x0 + t d)` the third derivative is a cubic form in `d`. So a step `h` along a vector
    of length `L` is a step `hL` in the function and pays `L^2` in the truncation term for
    nothing. Probe the unit direction and put `L^3` back, which is what the origin version does
    and what polarisation needs, since it probes vectors of length up to about 2.4.
    """
    lengths = directions.norm(dim=1, keepdim=True).clamp_min(1e-300)
    unit_directions = directions / lengths
    cube = (lengths.squeeze(1) ** 3).unsqueeze(1)

    estimates = []
    for step in steps:
        offsets = [2.0 * step, step, -step, -2.0 * step]
        values = [teacher(base_point[None, :] + unit_directions * offset) for offset in offsets]
        derivative = (values[0] - 2.0 * values[1] + 2.0 * values[2] - values[3]) / (2.0 * step ** 3)
        estimates.append(derivative * cube)
    stacked = torch.stack(estimates, dim=0)
    if len(steps) == 1:
        return stacked[0]
    # Two unknowns, and the right-hand side is the whole measurement. `lstsq` fits
    # `value(h) = a0 + a1 h^2` against `directions x width` right-hand columns, and its workspace
    # scales with that count: at 48 dimensions against a 3072-wide live response it is 1176 x 3072
    # columns and a 13.86 GiB request on a 7.65 GiB card, which is where the first 16-unit live
    # solve died.
    #
    # The columns are independent, so the fit is chunked rather than replaced. Normal equations
    # would also fit in memory and were tried first, but they square the condition number and the
    # run came back with different numbers - same recovery, different intermediates - and a
    # substitution that moves results is not a repair. Chunking keeps the same solver and the same
    # answer, and only bounds the workspace.
    step_squares = torch.tensor([[1.0, s ** 2] for s in steps], device=directions.device, dtype=DTYPE)
    flattened = stacked.reshape(len(steps), -1)
    columns = int(flattened.shape[1])
    chunk = max(1, min(columns, 1 << 20))
    pieces = [
        torch.linalg.lstsq(step_squares, flattened[:, start:start + chunk]).solution[0]
        for start in range(0, columns, chunk)
    ]
    solution = torch.cat(pieces) if len(pieces) > 1 else pieces[0]
    return solution.reshape(stacked.shape[1], stacked.shape[2])


def measured_lifted_span_by_slices(
    teacher: Teacher,
    base_point: torch.Tensor,
    readers: torch.Tensor,
    anchors: torch.Tensor,
    input_dim: int,
    steps: list[float],
    tolerance: float,
) -> tuple[torch.Tensor, int]:
    """The same span, paid for by polarisation instead of by fitting the whole tensor.

    Fitting every degree-3 coefficient costs `4n(n+1)(n+2)` points and then throws most of the
    tensor away: the span only needs enough slices to reach rank `2k_loud`, which is tens, not
    `n`. Each slice by polarisation costs `O(n^2)`, and the terms that do not mention the anchor
    are shared across all of them.

    Which route is cheaper is a function of `n` and was measured rather than assumed: the tensor
    is `4n^3` and slices are about `4 A n^2` for `A` anchors, so slices win only past `n ~ 90 A/45`.
    At `n=16` the tensor route is the cheap one and this would be a pessimisation; at `n=3072` it
    is the difference between `1.16e11` points per base point and something a run could finish.
    """
    basis = torch.eye(input_dim, device=base_point.device, dtype=DTYPE)
    rows, cols = torch.triu_indices(input_dim, input_dim, device=base_point.device)
    pair_sum = basis[rows] + basis[cols]

    shared_pair = measure_cubic_forms_at(teacher, base_point, pair_sum, steps)
    shared_basis = measure_cubic_forms_at(teacher, base_point, basis, steps)

    slices: list[torch.Tensor] = []
    for anchor in anchors:
        pair_anchor = measure_cubic_forms_at(teacher, base_point, pair_sum + anchor[None, :], steps)
        basis_anchor = measure_cubic_forms_at(teacher, base_point, basis + anchor[None, :], steps)
        anchor_only = measure_cubic_forms_at(teacher, base_point, anchor.unsqueeze(0), steps)[0]
        for reader in readers:
            entries = (
                pair_anchor @ reader
                - shared_pair @ reader
                - (basis_anchor @ reader)[rows]
                - (basis_anchor @ reader)[cols]
                + (shared_basis @ reader)[rows]
                + (shared_basis @ reader)[cols]
                + (anchor_only @ reader)
            ) / 6.0
            matrix = torch.zeros((input_dim, input_dim), device=base_point.device, dtype=DTYPE)
            matrix[rows, cols] = entries
            matrix[cols, rows] = entries
            slices.append(matrix.reshape(-1))
    stacked = torch.stack(slices, dim=0)
    left, spectrum, _ = torch.linalg.svd(stacked.T, full_matrices=False)
    rank = int((spectrum > spectrum[0] * tolerance).sum())
    return left[:, :rank], rank


def measured_lifted_span(
    teacher: Teacher,
    base_points: torch.Tensor,
    readers: torch.Tensor,
    input_dim: int,
    steps: list[float],
    direction_multiple: float,
    tolerance: float,
) -> tuple[torch.Tensor, int]:
    """Build `W` from what the oracle answers, not from the weights.

    The rank threshold here is not the one the minor null space uses, and conflating them is a
    bug this file had: the span is built from *measured* tensors carrying about `1e-8` relative
    error, so a cutoff at `1e-8` admits noise directions as real ones. Measured at `n=8`, `k=12`:
    the span came back rank 27 where `2k = 24`, and those three noise directions took the
    recovery from 12/12 to 0/12. The null space, by contrast, is computed in exact arithmetic
    from an orthonormal basis and wants a tight cutoff. Two numbers, two names.

    Every measured tensor `S` at a base point satisfies
    `S(.,.,e_c) = sum_i [alpha_i g_c + (beta_i/3) u_c] g_i g_i^T + (beta_i/3) g_c sym(g_i u_i^T)`,
    so each of its `n` slices already lies inside `span{g_i (x) g_i, sym(g_i (x) u_i)}`. Collecting
    slices across base points and readers therefore fills `W` from below, and the rank stops at
    `2k` when it is full. This is the step that turns the recovery above from a statement about a
    span into a statement about an oracle.
    """
    coefficient_count = len(monomial_index(input_dim))
    direction_count = int(math.ceil(direction_multiple * coefficient_count))
    generator = torch.Generator(device=base_points.device).manual_seed(17)
    directions = torch.randn(direction_count, input_dim, generator=generator, device=base_points.device, dtype=DTYPE)
    directions = directions / directions.norm(dim=1, keepdim=True)

    slices: list[torch.Tensor] = []
    for index in range(base_points.shape[0]):
        forms = measure_cubic_forms_at(teacher, base_points[index], directions, steps)
        for reader in readers:
            tensor = recover_symmetric_tensor(directions, forms @ reader, input_dim)
            slices.append(tensor.permute(2, 0, 1).reshape(input_dim, -1))
    stacked = torch.cat(slices, dim=0)
    left, spectrum, _ = torch.linalg.svd(stacked.T, full_matrices=False)
    rank = int((spectrum > spectrum[0] * tolerance).sum())
    return left[:, :rank], rank


def minor_map(basis_matrices: torch.Tensor) -> torch.Tensor:
    """The 2x2 minors of `M = sum_r c_r B_r`, as a linear map on `c (x) c`.

    This is what the three dead routes were missing. Searching for rank-1 elements of `W`
    directly is a quartic in `v` with a flat landscape and no usable second-order information at
    the solution. But rank is a statement about minors, and minors are *bilinear*:

        Phi(M,N)_ijkl = 1/2 (M_ij N_kl + N_ij M_kl - M_il N_kj - N_il M_kj)

    vanishes identically when `M = N = v v^T`, because both halves are `v_i v_j v_k v_l`. So
    `Phi(M(c), M(c)) = sum_rs c_r c_s Phi(B_r, B_s) = 0` is *linear* in `c (x) c`, and the search
    becomes a null space instead of an optimisation. Nothing here iterates.
    """
    count = basis_matrices.shape[0]
    symmetric = (basis_matrices + basis_matrices.transpose(1, 2)) / 2
    minors = (
        torch.einsum("rij,skl->rsijkl", symmetric, symmetric)
        + torch.einsum("sij,rkl->rsijkl", symmetric, symmetric)
        - torch.einsum("ril,skj->rsijkl", symmetric, symmetric)
        - torch.einsum("sil,rkj->rsijkl", symmetric, symmetric)
    ) / 2.0
    return minors.reshape(count * count, -1).T


class SubspaceOracle:
    """The same block seen through a `d`-dimensional slice of its input.

    Restricting `x` to a subspace `V` leaves a SwiGLU with gates `P_V g_i` and ups `P_V u_i`, so
    everything above applies with `n` replaced by `d`. A base point costs `12 d^2` instead of
    `12 n^2`, and the counter still charges the real oracle, because the restriction is a choice
    of where to probe rather than a different machine.

    What comes back is the projection, not the unit. Recovering `g_i` from projections means
    stitching across overlapping subspaces, which is not implemented here.
    """

    def __init__(self, teacher: Teacher, basis: torch.Tensor) -> None:
        self.teacher = teacher
        self.basis = basis
        self.counter = teacher.counter

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        return self.teacher(points @ self.basis.T)


def subspace_probe(
    units: int,
    input_dim: int,
    output_dim: int,
    subspace_dim: int,
    base_scale: float,
    rank_cap: int,
    seed: int,
    tolerance: float,
    span_tolerance: float,
    purity: float,
    collinearity: float,
    steps: list[float],
    device: torch.device,
) -> dict:
    """One solve inside a random subspace, scored against the projected gates."""
    teacher = make_teacher(units, input_dim, output_dim, seed, device)
    generator = torch.Generator(device=device).manual_seed(seed + 9)
    basis, _ = torch.linalg.qr(
        torch.randn(input_dim, subspace_dim, generator=generator, device=device, dtype=DTYPE)
    )
    oracle = SubspaceOracle(teacher, basis)

    base_point = torch.randn(subspace_dim, generator=generator, device=device, dtype=DTYPE)
    base_point = base_point / base_point.norm() * base_scale
    readers = torch.randn(output_dim, output_dim, generator=generator, device=device, dtype=DTYPE)
    readers = readers / readers.norm(dim=1, keepdim=True)
    anchors = torch.randn(3, subspace_dim, generator=generator, device=device, dtype=DTYPE)
    anchors = anchors / anchors.norm(dim=1, keepdim=True)

    before = teacher.counter.points
    span_basis, rank = measured_lifted_span_by_slices(
        oracle, base_point, readers, anchors, subspace_dim, steps, span_tolerance
    )
    calls = teacher.counter.points - before
    if rank // 2 > rank_cap:
        del span_basis
        torch.cuda.empty_cache()
        return {
            "subspaceDim": subspace_dim, "baseScale": base_scale, "spanRank": rank,
            "tooLargeForSolve": True, "oracleCallsPoints": calls, "device": str(basis.device),
        }
    gates, _ = foobi_gates(span_basis, subspace_dim, tolerance, purity, collinearity, seed + 7)
    del span_basis
    torch.cuda.empty_cache()

    projected = teacher.gate @ basis
    projected = projected / projected.norm(dim=1, keepdim=True)
    best = (
        (gates @ projected.T).abs().max(dim=1).values
        if gates.shape[0] > 0
        else torch.zeros(1, device=device, dtype=DTYPE)
    )
    return {
        "units": units,
        "inputDim": input_dim,
        "subspaceDim": subspace_dim,
        "baseScale": base_scale,
        "spanRank": rank,
        "tooLargeForSolve": False,
        "gatesReturned": int(gates.shape[0]),
        "gatesMatchingProjection": int((best >= 0.99).sum()),
        "medianCosine": float(best.median()),
        "oracleCallsPoints": calls,
        "callsAgainstFullDim": (12 * input_dim * input_dim) / max(calls, 1),
        "recoversProjectionsOnly": True,
        "device": str(basis.device),
    }


def stitch_recovery(
    units: int,
    input_dim: int,
    output_dim: int,
    subspace_dim: int,
    core_dim: int,
    base_scale: float,
    rank_cap: int,
    core_points: int,
    pencil_draws: int,
    seed: int,
    tolerance: float,
    span_tolerance: float,
    purity: float,
    collinearity: float,
    steps: list[float],
    device: torch.device,
) -> dict:
    """Full-dimension gates from subspace solves, by giving every subspace the same core.

    A subspace solve returns `P_V g`, which is not a unit. Stitching needs two things and one
    core supplies both. Let every `V_t = A + W_t` share a core `A` and take the `W_t` as an
    orthogonal partition of the rest. Then `P_A P_{V_t} = P_A`, so the core block of a recovered
    projection is the same vector for every `t` up to scale: it is the signature that says which
    recoveries are one unit, and its length is what puts the blocks on a common scale. The
    orthogonal partition then makes the assembly exact rather than a fit.

    The base point has to live in the core too, and that was the difference between working and
    not. With a fresh base point per subspace, each one localises onto a different subset and a
    unit is almost never seen twice: 52 groups from 67 recoveries, 11 with two subspaces, nothing
    assembled above cosine 0.85. Put the base point in `A` and it is the same point in the full
    space for every solve, so the loud set is the same one - 20 groups, and every group that
    covers all blocks assembles.
    """
    # Core points follow the unit count, not the input dimension, and that was worth measuring
    # because it decides whether a live layer is reachable at all. Every earlier sweep held
    # `k = n`, so a requirement that grows with `k` and one that grows with `n` fit the same
    # numbers. Separating them at fixed `n=256`, `d=48`, 96 core points: `k=64` covers 64 of 64,
    # `k=128` covers 110 of 128, `k=256` covers 135 of 256. At `1.5 k` core points instead, the
    # three land on 100%, 99% and 97%, and `n=512` with `k=64` at 96 points gives 95%.
    #
    # The oracle cost does not depend on `k` at all - it is `blocks * core_points * c(d)`, and
    # `k=64` and `k=256` at the same geometry both charge 23 318 184 points. So the bill for a
    # live block is set by how wide the layer's input is and how many units are wanted, not by
    # how many units the layer has.
    core_points_derived = core_points <= 0
    core_points = core_points if core_points > 0 else max(1, (3 * units + 1) // 2)

    teacher = make_teacher(units, input_dim, output_dim, seed, device)
    generator = torch.Generator(device=device).manual_seed(seed + 31)
    frame, _ = torch.linalg.qr(torch.randn(input_dim, input_dim, generator=generator, device=device, dtype=DTYPE))
    core = frame[:, :core_dim]
    block_width = subspace_dim - core_dim
    blocks = [frame[:, index:index + block_width] for index in range(core_dim, input_dim, block_width)]
    # The tail block used to be dropped for being narrow, and dropping it drops coordinates: with
    # `core_dim=8` and `block_width=16` at `n=64` the last 8 directions of the frame are in no
    # block at all, so an assembled gate is missing that part of the space and cannot reach the
    # 0.99 cosine however well the solves went. Nothing reported it, because every field is
    # computed from what was assembled. A short block is not a problem for the solve - the
    # subspace is `core + block` and its width is whatever it is - so keep it and carry the width
    # per block instead of assuming one.
    blocks = [block for block in blocks if block.shape[1] > 0]
    covered_directions = core_dim + sum(int(block.shape[1]) for block in blocks)

    readers = torch.randn(output_dim, output_dim, generator=generator, device=device, dtype=DTYPE)
    readers = readers / readers.norm(dim=1, keepdim=True)

    # Anchors were a fixed 3, and that number - not the geometry, not the solver - was the per
    # block ceiling. Slices at one anchor see the same span through one down projection, so the
    # measured rank stops at `anchors * readers` however many units are loud, and `2 k_loud` of
    # them is what a solve needs. Three anchors against eight readers is rank 24, which is twelve
    # units, and the recall that saturated at 0.7 was a solve being handed half the block.
    # `tile_recovery` already carried the rule `A * R >= 2 k_loud`; this path did not.
    #
    # It shows up as a flat probe: rank 24 at every scale from 15 to 4000, including the two that
    # recover nothing, because a saturated measurement reports its own ceiling and says nothing
    # about the layer. The count now comes from the cap, and `sliceRankCeiling` is reported so a
    # saturated span is visible rather than silent.
    anchor_count = max(3, -(-2 * rank_cap // max(int(readers.shape[0]), 1)))
    slice_rank_ceiling = anchor_count * int(readers.shape[0])

    # The scale was a flag, and it decides everything: measured at n=64, k=64, scales 4 and 15
    # recover nothing because the base point sits near the origin where the rank is `min(k, m)`,
    # 500 covers the layer, and 4000 and above starve it again. The peak is not a constant either
    # - it moves with `k`, so a number tuned once is wrong at the next block size.
    #
    # Probe for it, and read the rank the other way round from the obvious one. Argmax of the
    # measured rank picks the worst scale: at 3 anchors it returned 24 at every scale from 15 to
    # 4000, chose 15.0 on the tie, and covered nothing. A rank sitting at `anchors * readers` is
    # the measurement reporting its own ceiling - the loud set is larger than the slices can
    # separate - and both ends of the scale range do that, the near end because every unit is
    # loud and the far end because what is left is noise.
    #
    # Reading the rank instead - smallest rank strictly below the ceiling - reproduced the
    # operating point at n=64 and then failed outright at n=256, where every candidate from 15 to
    # 4000 saturates at 24 and nothing is strictly below anything. It fell back to scale 1.0 and
    # covered 0 of 256, while a hand-set 1000 covered 27 with the same three anchors. A saturated
    # rank does not mean the solve is useless; it means the rank cannot rank the scales.
    #
    # So the probe runs the solve it is choosing for and counts what comes back. One block, one
    # candidate, `foobi_gates` on the measured span, and the score is the number of directions
    # that pass the purity test. That is the quantity coverage actually follows: at n=256, scales
    # 500, 1000 and 2000 give 1.2, 4.9 and 3.2 directions a solve and 1, 27 and 25 units covered.
    # Eight extra solves against `blocks * core_points` of them is a rounding error in the cost.
    #
    # The scale itself moves with `n`, which is why a fixed number could not survive the jump:
    # the base point lives in the `e`-dimensional core, so the drive `g.x0` carries a factor
    # `sqrt(e/n)` for a gate of unit norm, and the same drive needs `S` proportional to `sqrt(n)`.
    # Measured: 500 at n=64 and 1000 at n=256, which is the ratio the law asks for.
    scale_probe: list[dict] = []
    if base_scale <= 0:
        probe_subspace = torch.cat([core, blocks[0]], dim=1)
        probe_width = int(blocks[0].shape[1])
        probe_dim = core_dim + probe_width
        probe_oracle = SubspaceOracle(teacher, probe_subspace)
        probe_anchors = torch.randn(anchor_count, probe_dim, generator=generator, device=device, dtype=DTYPE)
        probe_anchors = probe_anchors / probe_anchors.norm(dim=1, keepdim=True)
        best_score = -1
        base_scale = 0.0
        reference = 500.0 * math.sqrt(input_dim / 64.0)
        for step_index in range(-3, 4):
            candidate = reference * (2.0 ** step_index)
            probe_core = torch.randn(core_dim, generator=generator, device=device, dtype=DTYPE)
            probe_core = probe_core / probe_core.norm() * candidate
            probe_point = torch.cat([probe_core, torch.zeros(probe_width, device=device, dtype=DTYPE)])
            probe_basis, probe_rank = measured_lifted_span_by_slices(
                probe_oracle, probe_point, readers, probe_anchors, probe_dim, steps, span_tolerance
            )
            if probe_rank // 2 > rank_cap:
                recovered = 0
            else:
                probe_gates, _ = foobi_gates(
                    probe_basis, probe_dim, tolerance, purity, collinearity, seed + 7, attempts=pencil_draws,
                )
                recovered = int(probe_gates.shape[0])
                del probe_gates
            del probe_basis
            torch.cuda.empty_cache()
            scale_probe.append({
                "scale": candidate,
                "measuredRank": probe_rank,
                "saturated": probe_rank >= slice_rank_ceiling,
                "recoveredDirections": recovered,
            })
            if recovered > best_score:
                best_score, base_scale = recovered, candidate
        # A probe that recovered nothing anywhere has not chosen a scale, and returning one that
        # was never seen to work is how the rank criterion covered 0 of 256 without saying so.
        if best_score <= 0:
            raise RuntimeError(
                f"base-scale probe recovered no directions at any candidate around {reference:.1f}; "
                f"ranks {[entry['measuredRank'] for entry in scale_probe]}"
            )

    # Several core points, and the blocks merge across all of them. A unit's block component
    # `P_{W_t} g` does not depend on where the base point sits, and neither does its signature
    # `P_A g`, so a recovery of block 3 at one core point and block 5 at another are two pieces
    # of the same vector. Per-block recall saturates at about 0.7 - measured, drawing 1, 5, 15,
    # 40 and 100 pencils gives 0.435, 0.652, 0.696, 0.739, 0.783 with precision falling from
    # 1.000 to 0.783 past 15 - so requiring all blocks at one core point discards most units.
    # Independent core points turn that product into a union.
    recoveries: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []
    oversized = 0
    for core_index in range(max(1, core_points)):
        core_point = torch.randn(core_dim, generator=generator, device=device, dtype=DTYPE)
        core_point = core_point / core_point.norm() * base_scale
        for block_index, block in enumerate(blocks):
            # Width per block, not one width for all of them: the tail block is narrower and the
            # base point, the anchors and the solve all have to agree with the subspace actually
            # being probed.
            width = int(block.shape[1])
            solve_dim = core_dim + width
            base_point = torch.cat([core_point, torch.zeros(width, device=device, dtype=DTYPE)])
            subspace = torch.cat([core, block], dim=1)
            oracle = SubspaceOracle(teacher, subspace)
            anchors = torch.randn(anchor_count, solve_dim, generator=generator, device=device, dtype=DTYPE)
            anchors = anchors / anchors.norm(dim=1, keepdim=True)
            span_basis, rank = measured_lifted_span_by_slices(
                oracle, base_point, readers, anchors, solve_dim, steps, span_tolerance
            )
            if rank // 2 > rank_cap:
                oversized += 1
                del span_basis
                torch.cuda.empty_cache()
                continue
            gates, _ = foobi_gates(
                span_basis, solve_dim, tolerance, purity, collinearity,
                seed + 7 + 101 * core_index, attempts=pencil_draws,
            )
            del span_basis
            torch.cuda.empty_cache()
            for index in range(gates.shape[0]):
                recovered = gates[index]
                head, tail = recovered[:core_dim], recovered[core_dim:]
                if float(head.norm()) < 1e-9:
                    continue
                recoveries.append((head / head.norm(), head, tail, block_index))

    groups: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]] = []
    for record in recoveries:
        for group in groups:
            if abs(float(record[0] @ group[0][0])) >= 0.999:
                group.append(record)
                break
        else:
            groups.append([record])

    # A group is discarded whole when it misses one block, and that is a product: at per-block
    # recall `p` a unit needs all `len(blocks)` of them. Whether the loss sits one block short or
    # far from complete decides what is worth building next, so count it rather than assume it.
    # The histogram is over groups, indexed by how many distinct blocks each one covers.
    block_coverage = [0] * (len(blocks) + 1)
    for group in groups:
        block_coverage[len({item[3] for item in group})] += 1

    complete = [group for group in groups if len({item[3] for item in group}) == len(blocks)]
    assembled: list[torch.Tensor] = []
    for group in complete:
        _, reference, _, _ = group[0]
        vector = core @ reference
        used: set[int] = set()
        for _, head, tail, block_index in group:
            if block_index in used:
                continue
            used.add(block_index)
            factor = float(reference.norm() / head.norm())
            if float(head @ reference) < 0:
                factor = -factor
            vector = vector + blocks[block_index] @ (tail * factor)
        assembled.append(vector / vector.norm())

    # One unit can come back as several assembled directions: the signature `P_A g` is only
    # `core_dim` long, so two recoveries of one unit at different core points can miss the 0.999
    # grouping test on core-block noise and assemble twice. Deduplicate in the full space, where
    # the vectors are `input_dim` long and the comparison is not noise-limited.
    deduplicated: list[torch.Tensor] = []
    for vector in assembled:
        if any(abs(float(vector @ kept)) >= collinearity for kept in deduplicated):
            continue
        deduplicated.append(vector)

    truth = teacher.gate / teacher.gate.norm(dim=1, keepdim=True)
    if deduplicated:
        cosines = (torch.stack(deduplicated) @ truth.T).abs()
        # Per direction: is it a real unit. Per unit: was it covered. The first is precision and
        # the second is coverage, and only the second can be read against the unit count - an
        # earlier field counted correct directions and reported 65 of 64 units on a duplicate.
        best = cosines.max(dim=1).values
        matched = int((best >= 0.99).sum())
        covered = int((cosines.max(dim=0).values >= 0.99).sum())
        median = float(best.median())
    else:
        matched, covered, median = 0, 0, 0.0
    return {
        "units": units,
        "inputDim": input_dim,
        "subspaceDim": subspace_dim,
        "coreDim": core_dim,
        "blocks": len(blocks),
        "corePoints": core_points,
        "corePointsDerivedFromUnits": core_points_derived,
        "pencilDraws": pencil_draws,
        "baseScale": base_scale,
        "baseScaleAuto": bool(scale_probe),
        "scaleProbe": scale_probe,
        "anchorsPerSolve": anchor_count,
        "readersUsed": int(readers.shape[0]),
        "sliceRankCeiling": slice_rank_ceiling,
        "localRecoveries": len(recoveries),
        "groups": len(groups),
        "groupsCoveringAllBlocks": len(complete),
        "solvesOversized": oversized,
        # If the blocks do not tile the space, an assembled gate is missing coordinates and no
        # other field can say so - they are all computed from what came back. This one can fail.
        "directionsInBlocks": covered_directions,
        "blocksTileInputSpace": covered_directions == input_dim,
        "blockWidths": [int(block.shape[1]) for block in blocks],
        "groupsByBlocksCovered": block_coverage,
        "groupsOneBlockShort": block_coverage[len(blocks) - 1] if len(blocks) else 0,
        "fullDimensionGates": len(deduplicated),
        "assembledBeforeDeduplication": len(assembled),
        "duplicateDirections": len(assembled) - len(deduplicated),
        "gatesAbove0.99": matched,
        "spuriousDirections": len(deduplicated) - matched,
        "unitsCovered": covered,
        "coverage": covered / units,
        "medianCosine": median,
        "oracleCallsPoints": teacher.counter.points,
        "callsPerCoveredUnit": teacher.counter.points // max(covered, 1),
        "callsPerAssembledGate": teacher.counter.points // max(len(deduplicated), 1),
        "device": str(truth.device),
    }


def minor_gram(basis_matrices: torch.Tensor) -> torch.Tensor:
    """`Psi^T Psi` without ever forming `Psi`, whose row count is `n^4`.

    The explicit minor map is what stopped this method at toy sizes: `n^4` rows against `(2k)^2`
    columns is 1.8 GiB at `n=16` and 15.8 GiB at `n=24`, so the card refused the second one. Only
    the null space is wanted, and that lives in the Gram matrix, which contracts to traces:

        <Phi(A,B), Phi(C,D)> = <A,C><B,D> + <A,D><B,C> - tr(ACBD) - tr(ADBC)

    Checked against the explicit product at `n=8`: relative difference `5.9e-16` for `k=6` and
    `1.6e-15` for `k=10`, with the nullity identical at 78 and 210. Memory drops from 4.5 MiB to
    0.16 MiB there, and stops depending on `n` at all - the cost is now `(2k)^4` and the input
    dimension only enters through matrix products of size `n`.
    """
    symmetric = (basis_matrices + basis_matrices.transpose(1, 2)) / 2
    count = symmetric.shape[0]
    inner = torch.einsum("rij,sij->rs", symmetric, symmetric)
    products = torch.einsum("rab,sbc->rsac", symmetric, symmetric)
    transposed = products.transpose(2, 3).contiguous()

    # Blocked over the first index. The result is `(2k)^2` square and unavoidable, but forming
    # the two trace terms whole needs two more tensors of that size at once, which is what put a
    # `k=64` run 2 GiB over a 7.65 GiB card. A block of rows costs `block * (2k)^3` instead.
    gram = torch.empty((count, count, count, count), device=symmetric.device, dtype=DTYPE)
    block = max(1, min(count, int(2 ** 26 // max(count ** 3, 1)) or 1))
    for start in range(0, count, block):
        stop = min(start + block, count)
        head = products[start:stop]
        first = torch.einsum("ruab,svba->rsuv", head, products)
        second = torch.einsum("rvab,suba->rsuv", head, products)
        gram[start:stop] = (
            torch.einsum("ru,sv->rsuv", inner[start:stop], inner)
            + torch.einsum("rv,su->rsuv", inner[start:stop], inner)
            - first
            - second
        )
        del first, second
    del products, transposed
    return gram.reshape(count * count, count * count)


def foobi_gates(
    span_basis: torch.Tensor,
    input_dim: int,
    tolerance: float,
    purity: float,
    collinearity: float,
    seed: int,
    attempts: int = 5,
) -> tuple[torch.Tensor, int]:
    """Recover the gate directions from the lifted span, including when `k > n`.

    The null space of `minor_map` is spanned by `c^(i) (x) c^(i)`, the coefficient vectors of the
    rank-1 elements. Those are genuinely rank-1 matrices in the `2k`-dimensional coefficient
    space, so a pencil separates them there - the same pencil that could not work in `R^n`, now
    applied where the objects actually are rank-1. Each recovered `c` rebuilds a matrix whose top
    eigenvector is a gate; the purity test rejects any that did not come back rank-1.
    """
    count = span_basis.shape[1]
    basis_matrices = span_basis.T.reshape(count, input_dim, input_dim)
    gram = minor_gram(basis_matrices)
    values, vectors = torch.linalg.eigh(gram)
    # `gram` is `Psi^T Psi`, so its eigenvalues are the squared singular values of `Psi` and the
    # cutoff has to be squared with them.
    nullity = int((values < values.max() * tolerance * tolerance).sum())
    if nullity < 2:
        return torch.zeros((0, input_dim), device=span_basis.device, dtype=DTYPE), nullity
    null_space = vectors[:, :nullity].T.reshape(-1, count, count)

    # One random pencil does not separate every eigenvalue it could. Which units come back
    # depends on the mixing vectors, so several draws recover strictly more of the same span at
    # no cost in precision - measured on a span of rank 46 from a subspace solve: one draw
    # returns 10 directions, all correct; five return 15, all correct. Lowering the purity test
    # instead buys recall with precision, which is not a trade worth making: 0.7 gives 23 at
    # 0.870, and 0.3 gives 77 at 0.338.
    kept: list[torch.Tensor] = []
    for attempt in range(max(1, attempts)):
        generator = torch.Generator(device=span_basis.device).manual_seed(seed + 13 * attempt)
        left_mix = torch.randn(nullity, generator=generator, device=span_basis.device, dtype=DTYPE)
        right_mix = torch.randn(nullity, generator=generator, device=span_basis.device, dtype=DTYPE)
        first = torch.einsum("q,qrs->rs", left_mix, null_space)
        second = torch.einsum("q,qrs->rs", right_mix, null_space)
        first = (first + first.T) / 2
        second = (second + second.T) / 2
        _, coefficients = torch.linalg.eig(first @ torch.linalg.pinv(second))

        for column in range(coefficients.real.shape[1]):
            candidate = torch.einsum("r,rij->ij", coefficients.real[:, column], basis_matrices)
            candidate = (candidate + candidate.T) / 2
            values, vectors = torch.linalg.eigh(candidate)
            magnitude = values.abs()
            if float(magnitude.max()) <= 0:
                continue
            if float(magnitude.max() / magnitude.sum().clamp_min(1e-300)) <= purity:
                continue
            direction = vectors[:, int(magnitude.argmax())]
            if any(abs(float(direction @ existing)) >= collinearity for existing in kept):
                continue
            kept.append(direction)
    if not kept:
        return torch.zeros((0, input_dim), device=span_basis.device, dtype=DTYPE), nullity
    return torch.stack(kept), nullity


def overcomplete_recovery(
    units: int,
    input_dim: int,
    output_dim: int,
    seed: int,
    tolerance: float,
    purity: float,
    collinearity: float,
    device: torch.device,
) -> dict:
    """End to end: lift, take the minor null space, and match the gates against truth."""
    teacher = make_teacher(units, input_dim, output_dim, seed, device)
    gate, up = teacher.gate, teacher.up
    gate_outer = torch.einsum("ia,ib->iab", gate, gate).reshape(units, -1)
    mixed = ((torch.einsum("ia,ib->iab", gate, up) + torch.einsum("ia,ib->iab", up, gate)) / 2.0).reshape(units, -1)
    span_basis, _ = torch.linalg.qr(torch.cat([gate_outer, mixed], dim=0).T)
    gates, nullity = foobi_gates(span_basis, input_dim, tolerance, purity, collinearity, seed + 3)
    truth = gate / gate.norm(dim=1, keepdim=True)
    if gates.shape[0] == 0:
        best = torch.zeros(units, device=device, dtype=DTYPE)
    else:
        best = (gates @ truth.T).abs().max(dim=0).values
    return {
        "units": units,
        "inputDim": input_dim,
        "unitsPerInputDim": units / input_dim,
        "overcomplete": units > input_dim,
        "pencilCeiling": input_dim // 2,
        "isolatedSolutionCeiling": (input_dim * input_dim - input_dim + 2) // 4,
        "nullity": nullity,
        "gatesMatchUnitCount": int(gates.shape[0]) == units,
        "gatesReturned": int(gates.shape[0]),
        "gatesAbove0.99": int((best >= 0.99).sum()),
        "gatesAbove0.999": int((best >= 0.999).sum()),
        "medianCosine": float(best.median()),
        "device": str(span_basis.device),
    }


def oracle_recovery(
    units: int,
    input_dim: int,
    output_dim: int,
    base_point_count: int,
    reader_count: int,
    base_scale: float,
    seed: int,
    tolerance: float,
    span_tolerance: float,
    purity: float,
    collinearity: float,
    steps: list[float],
    direction_multiple: float,
    device: torch.device,
) -> dict:
    """Oracle to gates, with nothing read off the weights and every point charged."""
    teacher = make_teacher(units, input_dim, output_dim, seed, device)
    generator = torch.Generator(device=device).manual_seed(seed + 23)
    base_points = torch.randn(base_point_count, input_dim, generator=generator, device=device, dtype=DTYPE)
    base_points = base_points / base_points.norm(dim=1, keepdim=True) * base_scale
    readers = torch.randn(reader_count, output_dim, generator=generator, device=device, dtype=DTYPE)
    readers = readers / readers.norm(dim=1, keepdim=True)

    span_basis, span_rank = measured_lifted_span(
        teacher, base_points, readers, input_dim, steps, direction_multiple, span_tolerance
    )
    gates, nullity = foobi_gates(span_basis, input_dim, tolerance, purity, collinearity, seed + 3)
    truth = teacher.gate / teacher.gate.norm(dim=1, keepdim=True)
    best = (
        (gates @ truth.T).abs().max(dim=0).values
        if gates.shape[0] > 0
        else torch.zeros(units, device=device, dtype=DTYPE)
    )
    return {
        "units": units,
        "inputDim": input_dim,
        "responseWidth": output_dim,
        "unitsPerInputDim": units / input_dim,
        "overcomplete": units > input_dim,
        "basePoints": base_point_count,
        "readers": reader_count,
        "measuredSpanRank": span_rank,
        "spanRankIsTwoK": span_rank == 2 * units,
        "spanTolerance": span_tolerance,
        "nullity": nullity,
        # A check that needs no truth. The span's rank is `2k` when the units are separable, so
        # a healthy run returns exactly `spanRank / 2` distinct rank-1 directions. A run that has
        # gone past the geometry returns some other number, and says so without being told.
        "gatesMatchSpanRank": int(gates.shape[0]) == span_rank // 2,
        "gatesReturned": int(gates.shape[0]),
        "gatesAbove0.99": int((best >= 0.99).sum()),
        "gatesAbove0.999": int((best >= 0.999).sum()),
        "medianCosine": float(best.median()),
        "oracleCallsPoints": teacher.counter.points,
        "device": str(span_basis.device),
    }


def tile_recovery(
    units: int,
    input_dim: int,
    output_dim: int,
    base_scale: float,
    trials: int,
    reader_count: int,
    rank_cap: int,
    seed: int,
    tolerance: float,
    span_tolerance: float,
    purity: float,
    collinearity: float,
    steps: list[float],
    direction_multiple: float,
    slice_crossover: int,
    device: torch.device,
) -> dict:
    """Cover a layer wider than one solve can hold, by solving subsets of it.

    The extraction cost is `(2k)^4` in the coefficient space, so a layer is never one solve: at
    `k=64` the Gram alone asks for more card than exists. It does not have to be. A base point at
    scale `S` weights unit `i` by `silu''(g_i.x0)`, which decays, so units under the measurement
    floor leave the *measured* span and take rank with them - the rank tracks the loud count. Each
    base point therefore presents a different small subset, and enough of them cover the layer.

    Merging uses no truth. Directions from different base points are the same unit when their
    cosine says so, and the truth match afterwards only scores what the merge already decided.
    """
    teacher = make_teacher(units, input_dim, output_dim, seed, device)
    generator = torch.Generator(device=device).manual_seed(seed + 101)

    # Scale is not a constant to guess. The loud count at a given distance grows with the layer,
    # so a scale tuned at k=64 presents oversized subsets at k=256 and starved ones at k=16.
    # Probe geometrically and take the nearest scale whose subset fits the solve: the cheapest
    # honest search, and the cost is one span measurement per candidate.
    chosen_scale = base_scale
    if base_scale <= 0:
        probe_readers = torch.randn(reader_count, output_dim, generator=generator, device=device, dtype=DTYPE)
        probe_readers = probe_readers / probe_readers.norm(dim=1, keepdim=True)
        chosen_scale = 1.0
        for candidate in [4.0, 15.0, 60.0, 120.0, 250.0, 500.0, 1000.0, 2000.0]:
            probe_point = torch.randn(1, input_dim, generator=generator, device=device, dtype=DTYPE)
            probe_point = probe_point / probe_point.norm() * candidate
            _, probe_rank = measured_lifted_span(
                teacher, probe_point, probe_readers, input_dim, steps, direction_multiple, span_tolerance
            )
            torch.cuda.empty_cache()
            chosen_scale = candidate
            if probe_rank // 2 <= rank_cap:
                break

    merged: list[torch.Tensor] = []
    growth: list[int] = []
    skipped = 0

    for _ in range(trials):
        base_point = torch.randn(1, input_dim, generator=generator, device=device, dtype=DTYPE)
        base_point = base_point / base_point.norm() * chosen_scale
        # Readers are free and anchors are not. The oracle answers `m` numbers per call whether
        # one is read or all of them, so a single anchor already yields `R` slices - but only
        # `k_loud` independent ones, because they all sit at the same anchor. Two anchors are the
        # floor and `A * R >= 2 k_loud` is the condition. Measured at n=32, k=12: A=2 with 24
        # readers reaches rank 24 for 13456 points where A=14 with 3 readers needs 67312 for the
        # same rank, and A=1 stops at 12 however many readers it is given.
        effective_readers = max(1, min(output_dim, max(reader_count, 2 * rank_cap)))
        anchor_count = max(2, -(-2 * rank_cap // effective_readers))
        readers = torch.randn(effective_readers, output_dim, generator=generator, device=device, dtype=DTYPE)
        readers = readers / readers.norm(dim=1, keepdim=True)
        try:
            if input_dim >= slice_crossover:
                # Measured, not assumed: the tensor route is `4n^3` and slices are `4 A n^2`, so
                # the ratio is linear in `n` and crosses at 15 for the anchor count this uses -
                # 1.60x at n=24, 2.13x at 32, 3.20x at 48, 4.27x at 64, 6.40x at 96, all against
                # the same span to a subspace angle of 1.000000000.
                anchors = torch.randn(
                    anchor_count, input_dim, generator=generator, device=device, dtype=DTYPE,
                )
                anchors = anchors / anchors.norm(dim=1, keepdim=True)
                span_basis, rank = measured_lifted_span_by_slices(
                    teacher, base_point[0], readers, anchors, input_dim, steps, span_tolerance
                )
            else:
                span_basis, rank = measured_lifted_span(
                    teacher, base_point, readers, input_dim, steps, direction_multiple, span_tolerance
                )
            if rank // 2 > rank_cap:
                skipped += 1
                del span_basis
                torch.cuda.empty_cache()
                growth.append(len(merged))
                continue
            gates, _ = foobi_gates(span_basis, input_dim, tolerance, purity, collinearity, seed + 7)
            del span_basis
        except torch.OutOfMemoryError:
            skipped += 1
            torch.cuda.empty_cache()
            growth.append(len(merged))
            continue
        torch.cuda.empty_cache()
        for index in range(gates.shape[0]):
            direction = gates[index]
            if any(abs(float(direction @ existing)) >= collinearity for existing in merged):
                continue
            merged.append(direction)
        growth.append(len(merged))

    truth = teacher.gate / teacher.gate.norm(dim=1, keepdim=True)
    if merged:
        cosines = (torch.stack(merged) @ truth.T).abs()
        matched = int((cosines.max(dim=0).values >= 0.99).sum())
        spurious = int((cosines.max(dim=1).values < 0.99).sum())
    else:
        matched, spurious = 0, 0
    return {
        "units": units,
        "inputDim": input_dim,
        "unitsPerInputDim": units / input_dim,
        "responseWidth": output_dim,
        "baseScale": chosen_scale,
        "baseScaleAuto": base_scale <= 0,
        "basePoints": trials,
        "solvesSkippedTooLarge": skipped,
        "distinctDirectionsMerged": len(merged),
        "unitsCovered": matched,
        "coverage": matched / units,
        "spuriousDirections": spurious,
        "spanRoute": "slices" if input_dim >= slice_crossover else "tensor",
        "readersUsed": max(1, min(output_dim, max(reader_count, 2 * rank_cap))),
        "anchorsPerBasePoint": max(2, -(-2 * rank_cap // max(1, min(output_dim, max(reader_count, 2 * rank_cap))))),
        "oracleCallsPoints": teacher.counter.points,
        "growth": growth,
        "device": str(truth.device),
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
    parser.add_argument("--recover-sweep", type=int, nargs="*", default=None,
                        help="unit counts to run the overcomplete recovery at")
    parser.add_argument("--purity", type=float, default=0.9)
    parser.add_argument("--oracle-recover", type=int, nargs="*", default=None,
                        help="unit counts to run oracle-to-gates recovery at")
    parser.add_argument("--base-scale", type=float, default=1.0)
    parser.add_argument("--readers", type=int, default=2)
    parser.add_argument("--base-count", type=int, default=3)
    parser.add_argument("--span-tolerance", type=float, default=1e-5)
    parser.add_argument("--tile", type=int, nargs="*", default=None,
                        help="unit counts to cover by tiling subsets over base points")
    parser.add_argument("--tile-trials", type=int, default=20)
    parser.add_argument("--rank-cap", type=int, default=52)
    parser.add_argument("--slice-crossover", type=int, default=16)
    parser.add_argument("--lifted-sweep", type=int, nargs="*", default=None,
                        help="unit counts to measure the lifted identifiability ceiling at")
    parser.add_argument("--sweep-units", type=int, nargs="*", default=None)
    parser.add_argument("--stitch", type=int, nargs="*", default=None,
                        help="unit counts to assemble full-dimension gates for from subspace solves")
    parser.add_argument("--subspace-dim", type=int, default=16)
    parser.add_argument("--core-dim", type=int, default=8)
    parser.add_argument("--core-points", type=int, default=0,
                        help="core points to spend; 0 derives 1.5k from the unit count")
    parser.add_argument("--pencil-draws", type=int, default=5)
    parser.add_argument("--live-form-check", action="store_true",
                        help="measure the cubic form through the real oracle and check it post-hoc")
    parser.add_argument("--live-model", type=str, default="unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit")
    parser.add_argument("--live-layer", type=int, default=15)
    parser.add_argument("--live-units", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--live-directions", type=int, default=40)
    parser.add_argument("--live-subspace-dim", type=int, default=8)
    parser.add_argument("--live-base-scale", type=float, default=1.0)
    parser.add_argument("--live-max-calls", type=int, default=200000)
    parser.add_argument("--live-work-dir", type=str, default=None)
    parser.add_argument("--live-subspace-solve", action="store_true",
                        help="solve one subspace of the real layer and score it post-hoc")
    parser.add_argument("--live-readers", type=int, default=8)
    parser.add_argument("--live-anchors", type=int, default=3)
    parser.add_argument("--live-local-files-only", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested and not available; this probe is a GPU measurement")

    # The pencil run is the default mode, not a preamble to the others. It fits the full symmetric
    # tensor, whose design is `4n(n+1)(n+2)` rows against `C(n+2,3)` monomials, so at `n=64` it
    # allocates 6.89 GiB before any requested mode is reached and the card has nothing left: a
    # `--stitch` run at that dimension died on 20 MiB it should never have had to ask for. Run it
    # when it was asked for - explicitly by `--sweep-units`, or by default when no mode was named.
    selected_modes = [
        args.stitch, args.tile, args.oracle_recover, args.recover_sweep,
        args.lifted_sweep, args.base_point_sweep, args.project_dim,
        args.live_form_check or None, args.live_subspace_solve or None,
    ]
    runs: list[dict] = []
    if args.sweep_units or not any(selected_modes):
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
    if args.live_form_check:
        work_dir = Path(args.live_work_dir) if args.live_work_dir else Path("/tmp/cubic-live-check")
        report["liveFormCheck"] = live_form_check(
            model=args.live_model, layer=args.live_layer, units=args.live_units,
            direction_count=args.live_directions, subspace_dim=args.live_subspace_dim,
            base_scale=args.live_base_scale, steps=list(args.steps), seed=args.seed,
            max_calls=args.live_max_calls, work_dir=work_dir,
            local_files_only=args.live_local_files_only, device=device,
        )
    if args.live_subspace_solve:
        work_dir = Path(args.live_work_dir) if args.live_work_dir else Path("/tmp/cubic-live-check")
        report["liveSubspaceSolve"] = live_subspace_solve(
            model=args.live_model, layer=args.live_layer, units=args.live_units,
            subspace_dim=args.live_subspace_dim, base_scale=args.live_base_scale,
            reader_count=args.live_readers, anchor_count=args.live_anchors,
            rank_target=args.rank_cap,
            steps=list(args.steps), seed=args.seed, tolerance=args.rank_tolerance,
            span_tolerance=args.span_tolerance, purity=args.purity,
            collinearity=args.collinearity, pencil_draws=args.pencil_draws,
            max_calls=args.live_max_calls, work_dir=work_dir,
            local_files_only=args.live_local_files_only, device=device,
        )
    if args.project_dim:
        report["costProjection"] = [cost_laws(dim) for dim in args.project_dim]
    if args.tile:
        report["tileRecovery"] = [
            tile_recovery(
                units=count, input_dim=args.input_dim, output_dim=args.reach_width,
                base_scale=args.base_scale, trials=args.tile_trials, reader_count=args.readers,
                rank_cap=args.rank_cap, seed=args.seed, tolerance=args.rank_tolerance,
                span_tolerance=args.span_tolerance, purity=args.purity,
                collinearity=args.collinearity, steps=list(args.steps),
                direction_multiple=args.direction_multiple,
                slice_crossover=args.slice_crossover, device=device,
            )
            for count in args.tile
        ]
    if args.oracle_recover:
        report["oracleRecovery"] = [
            oracle_recovery(
                units=count, input_dim=args.input_dim, output_dim=args.reach_width,
                base_point_count=args.base_count, reader_count=args.readers,
                base_scale=args.base_scale, seed=args.seed, tolerance=args.rank_tolerance,
                span_tolerance=args.span_tolerance,
                purity=args.purity, collinearity=args.collinearity, steps=list(args.steps),
                direction_multiple=args.direction_multiple, device=device,
            )
            for count in args.oracle_recover
        ]
    if args.stitch:
        report["stitchRecovery"] = [
            stitch_recovery(
                units=count, input_dim=args.input_dim, output_dim=args.reach_width,
                subspace_dim=args.subspace_dim, core_dim=args.core_dim,
                base_scale=args.base_scale, rank_cap=args.rank_cap,
                core_points=args.core_points, pencil_draws=args.pencil_draws,
                seed=args.seed, tolerance=args.rank_tolerance,
                span_tolerance=args.span_tolerance, purity=args.purity,
                collinearity=args.collinearity, steps=list(args.steps), device=device,
            )
            for count in args.stitch
        ]
    if args.recover_sweep:
        report["overcompleteRecovery"] = [
            overcomplete_recovery(
                units=count, input_dim=args.input_dim, output_dim=args.reach_width,
                seed=args.seed, tolerance=args.rank_tolerance, purity=args.purity,
                collinearity=args.collinearity, device=device,
            )
            for count in args.recover_sweep
        ]
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
