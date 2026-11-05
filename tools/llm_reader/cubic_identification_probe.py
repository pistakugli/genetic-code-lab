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
import gc
import hashlib
import json
import math
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any, Callable

import torch
import numpy as np


# The whole probe is a cancellation measurement: a third derivative is a difference of
# differences of differences, and in single precision the stencil's numerator loses every digit
# the signal has. Double everywhere, on the card.
DTYPE = torch.float64
OracleCallable = Callable[[torch.Tensor], torch.Tensor]


def _proc_status_memory_bytes(pid: int | str, field: str) -> int | None:
    status_path = Path(f"/proc/{pid}/status")
    if not status_path.is_file():
        return None
    try:
        lines = status_path.read_text(encoding="ascii").splitlines()
    except (FileNotFoundError, ProcessLookupError):
        return None
    for line in lines:
        if line.startswith(f"{field}:"):
            fields = line.split()
            if len(fields) != 3 or fields[2] != "kB":
                raise RuntimeError(f"unexpected {field} format: {line!r}")
            return int(fields[1]) * 1024
    return None


def _proc_smaps_pss_bytes(pid: int | str) -> int | None:
    smaps_path = Path(f"/proc/{pid}/smaps_rollup")
    if not smaps_path.is_file():
        return None
    try:
        lines = smaps_path.read_text(encoding="ascii").splitlines()
    except (FileNotFoundError, ProcessLookupError):
        return None
    for line in lines:
        if line.startswith("Pss:"):
            fields = line.split()
            if len(fields) != 3 or fields[2] != "kB":
                raise RuntimeError(f"unexpected Pss format: {line!r}")
            return int(fields[1]) * 1024
    return None


def _proc_vm_hwm_bytes() -> int | None:
    return _proc_status_memory_bytes("self", "VmHWM")


class OracleQueryBudgetExceeded(RuntimeError):
    def __init__(
        self,
        *,
        requested_points: int,
        spent_points: int,
        max_oracle_points: int,
    ) -> None:
        self.requested_points = requested_points
        self.spent_points = spent_points
        self.max_oracle_points = max_oracle_points
        super().__init__(
            "oracle query budget exceeded: "
            f"spent={spent_points}, requested={requested_points}, budget={max_oracle_points}"
        )


class OracleWorkspaceLimitExceeded(RuntimeError):
    def __init__(
        self,
        *,
        input_dim: int,
        direction_count: int,
        coefficient_count: int,
        minimum_workspace_bytes: int,
        max_workspace_bytes: int,
    ) -> None:
        self.input_dim = input_dim
        self.direction_count = direction_count
        self.coefficient_count = coefficient_count
        self.minimum_workspace_bytes = minimum_workspace_bytes
        self.max_workspace_bytes = max_workspace_bytes
        super().__init__(
            "measured cubic recovery requires at least "
            f"{minimum_workspace_bytes} workspace bytes at input_dim={input_dim} "
            f"(directions={direction_count}, cubic_coefficients={coefficient_count}); "
            f"configured limit is {max_workspace_bytes}"
        )


def estimate_measured_cubic_workspace(
    input_dim: int,
    direction_multiple: float,
) -> dict[str, int]:
    if input_dim <= 0:
        raise ValueError("input_dim must be positive")
    if not math.isfinite(direction_multiple) or direction_multiple <= 0.0:
        raise ValueError("direction_multiple must be finite and positive")
    coefficient_count = math.comb(input_dim + 2, 3)
    direction_count = int(math.ceil(direction_multiple * coefficient_count))
    element_size = torch.empty((), dtype=DTYPE).element_size()
    direction_tensor_bytes = direction_count * input_dim * element_size
    cubic_design_bytes = direction_count * coefficient_count * element_size
    return {
        "coefficientCount": coefficient_count,
        "directionCount": direction_count,
        "directionTensorBytes": direction_tensor_bytes,
        "cubicDesignBytes": cubic_design_bytes,
        "minimumWorkspaceBytes": direction_tensor_bytes + cubic_design_bytes,
    }


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

    def __init__(self, client: Any, device: torch.device, projection: torch.Tensor | None = None) -> None:
        self.client = client
        self.device = device
        self.counter = OracleCallCounter()
        # Readers are free in oracle calls and expensive in transport. The block answers 3072
        # numbers a point, of which a solve reads eight, and every one of them crosses the bus
        # twice and the disk twice: 29 MB out, 14 MB written, 14 MB read, 29 MB back, per eval
        # call, 88 calls a solve. Projecting is linear and commutes with the difference stencil,
        # so doing it here is the same arithmetic against 384x less traffic. The oracle charge is
        # identical - the points are what is billed, not the width of the answer.
        self.projection = projection
        # What a live call costs in seconds, separated from what the solve costs. A stitched
        # recovery is `blocks * core_points` solves, so whether the route is hours or days is
        # decided by this number and not by anything on the card - and the transport writes both
        # the request and the answer to disk as `.npy`, at 3072 floats each way per point.
        self.eval_seconds = 0.0
        self.eval_calls = 0

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        self.counter.charge(int(points.shape[0]))
        started = time.monotonic()
        # float32 on the way out because the transport casts to it anyway; sending float64 pays
        # twice the bus for numbers the client immediately narrows.
        request = points.detach().to(torch.float32).cpu().numpy()
        values = self.client.eval(request, stage="cubic-probe")
        answer = torch.as_tensor(values, dtype=DTYPE, device=self.device)
        self.eval_seconds += time.monotonic() - started
        self.eval_calls += 1
        if self.projection is None:
            return answer
        return answer @ self.projection.T


def open_live_oracle(
    model: str,
    layer: int,
    units: str,
    device: str,
    max_calls: int,
    work_dir: Path,
    local_files_only: bool,
    oracle_output_mode: str = "selected-unit-ffn-vector",
    opaque_eval_only: bool = False,
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
        units=None if oracle_output_mode == "full-mlp-vector" else units,
        device=device,
        oracle_batch_size=2048,
        output_channel=-1,
        oracle_output_mode=oracle_output_mode,
        oracle_quantization_step=0.0,
        local_files_only=bool(local_files_only),
        require_opaque_oracle=bool(opaque_eval_only),
        oracle_ready_mode="opaque" if opaque_eval_only else "full",
        oracle_eval_only=bool(opaque_eval_only),
    )
    return live.OracleClient(arguments, work_dir)


def _load_extra_prompts(
    source_path: Path,
    existing_ids: set[str],
    required_prefix: str,
) -> tuple[list[tuple[str, str]], bytes]:
    source_bytes = source_path.read_bytes()
    prompt_data = json.loads(source_bytes)
    if not isinstance(prompt_data, list):
        raise ValueError("extra recovery prompts must be a JSON array")
    extra_prompts: list[tuple[str, str]] = []
    for index, item in enumerate(prompt_data):
        if not isinstance(item, dict) or set(item) != {"promptId", "text"}:
            raise ValueError(
                f"extra recovery prompt {index} must contain exactly promptId and text"
            )
        prompt_id, prompt = item["promptId"], item["text"]
        if (
            not isinstance(prompt_id, str)
            or not prompt_id.startswith(required_prefix)
            or prompt_id in existing_ids
            or not isinstance(prompt, str)
            or not prompt.strip()
        ):
            raise ValueError(f"invalid or duplicate extra prompt at index {index}")
        existing_ids.add(prompt_id)
        extra_prompts.append((prompt_id, prompt))
    if not extra_prompts:
        raise ValueError("extra prompt file must contain at least one prompt")
    return extra_prompts, source_bytes


def capture_qwen_real_activations(
    model: str,
    layer: int,
    output_dir: Path,
    device: torch.device,
    local_files_only: bool,
    seed: int,
    oracle_query_budget: int,
    extra_recovery_prompts_file: Path | None = None,
    extra_functional_prompts_file: Path | None = None,
    extra_distributional_prompts_file: Path | None = None,
) -> dict[str, Any]:
    """Capture provenance-separated Qwen activations and smoke-test its opaque full-MLP oracle."""
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("real Qwen activation capture requires an available CUDA device")
    if layer < 0:
        raise ValueError("layer must be non-negative")
    if oracle_query_budget <= 0:
        raise ValueError("oracle_query_budget must be positive")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing pilot artifacts: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_splits = {
        "recovery_observations": [
            ("recovery-01", "Explain why leaves change color during autumn."),
            ("recovery-02", "Write a short Python function that checks whether an integer is prime."),
            ("recovery-03", "Objasni razliku između memorije i skladišta podataka."),
        ],
        "functional_evaluation": [
            ("functional-01", "A train leaves at 09:20 and travels for 95 minutes. When does it arrive?"),
            ("functional-02", "Summarize how a hash table resolves collisions in two sentences."),
            ("functional-03", "Napiši tri koraka za bezbedno čuvanje rezervne kopije."),
        ],
        "structural_evaluation": [
            ("structural-01", "What is the geometric meaning of a matrix's null space?"),
            ("structural-02", "Give a concise example of recursion using a tree traversal."),
            ("structural-03", "Kako se razlikuju korelacija i uzročnost?"),
        ],
        "distributional_analysis": [
            ("distribution-01", "Describe the water cycle from evaporation to precipitation."),
            ("distribution-02", "Compare a compiler with an interpreter for a beginner."),
            ("distribution-03", "Predloži jednostavan plan za organizovanje radnog dana."),
        ],
    }
    prompt_sources: dict[str, dict[str, Any]] = {}
    for split_name, source_file, required_prefix in (
        (
            "recovery_observations",
            extra_recovery_prompts_file,
            "recovery-extra-",
        ),
        (
            "functional_evaluation",
            extra_functional_prompts_file,
            "functional-extra-",
        ),
        (
            "distributional_analysis",
            extra_distributional_prompts_file,
            "distribution-extra-",
        ),
    ):
        if source_file is None:
            continue
        existing_ids = {
            prompt_id
            for prompts in prompt_splits.values()
            for prompt_id, _ in prompts
        }
        extra_prompts, source_bytes = _load_extra_prompts(
            source_file.resolve(), existing_ids, required_prefix
        )
        prompt_splits[split_name].extend(extra_prompts)
        copied_name = f"additional-{split_name.replace('_', '-')}-prompts.json"
        copied_path = output_dir / copied_name
        copied_path.write_bytes(source_bytes)
        prompt_sources[split_name] = {
            "file": copied_name,
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "promptCount": len(extra_prompts),
        }

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from tools.llm_reader.adapters.registry import create_adapter

    adapter = create_adapter("qwen2-swiglu")
    load_started = time.perf_counter()
    tokenizer, model_instance = adapter.load_model_and_tokenizer(
        model, local_files_only=local_files_only, device=str(device)
    )
    load_seconds = time.perf_counter() - load_started
    mlp_modules = adapter.mlp_modules(model_instance, [layer])
    mlp_view = mlp_modules[layer]
    mlp = mlp_view._mlp
    input_dim = int(mlp.gate_proj.in_features)
    intermediate_dim = int(mlp.gate_proj.out_features)
    output_dim = int(mlp.down_proj.out_features)
    if input_dim != output_dim:
        raise RuntimeError(
            f"Qwen pilot expects a square residual-stream MLP, got {input_dim}->{output_dim}"
        )
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    capture_started = time.perf_counter()

    dataset_records: dict[str, dict[str, Any]] = {}
    split_summaries: dict[str, dict[str, Any]] = {}
    for split_name, prompts in prompt_splits.items():
        input_rows: list[np.ndarray] = []
        output_rows: list[np.ndarray] = []
        token_rows: list[np.ndarray] = []
        position_rows: list[np.ndarray] = []
        prompt_id_rows: list[np.ndarray] = []
        prompt_hash_rows: list[np.ndarray] = []
        prompt_records: list[dict[str, Any]] = []
        split_max_direct_delta = 0.0
        split_started = time.perf_counter()
        for prompt_id, prompt in prompts:
            encoded = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=128
            )
            encoded = {name: value.to(device) for name, value in encoded.items()}
            captured: dict[str, torch.Tensor] = {}

            def capture_input(_module: Any, inputs: tuple[torch.Tensor, ...]) -> None:
                captured["input"] = inputs[0].detach()

            def capture_output(
                _module: Any, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor
            ) -> None:
                captured["output"] = output.detach()

            pre_handle = mlp.register_forward_pre_hook(capture_input)
            post_handle = mlp.register_forward_hook(capture_output)
            try:
                with torch.inference_mode():
                    model_instance(**encoded)
            finally:
                pre_handle.remove()
                post_handle.remove()
            if "input" not in captured or "output" not in captured:
                raise RuntimeError(f"Qwen MLP hooks did not capture prompt {prompt_id}")
            hidden = captured["input"]
            observed_output = captured["output"]
            with torch.inference_mode():
                direct_output = mlp(hidden)
            direct_delta = float((direct_output - observed_output).abs().max().item())
            split_max_direct_delta = max(split_max_direct_delta, direct_delta)
            hidden_rows = hidden[0].float().cpu().numpy()
            output_values = observed_output[0].float().cpu().numpy()
            token_ids = encoded["input_ids"][0].detach().cpu().numpy().astype(np.int64)
            sequence_length = int(hidden_rows.shape[0])
            if sequence_length != len(token_ids) or output_values.shape != hidden_rows.shape:
                raise RuntimeError(f"captured activation shape mismatch for prompt {prompt_id}")
            input_rows.append(hidden_rows)
            output_rows.append(output_values)
            token_rows.append(token_ids)
            position_rows.append(np.arange(sequence_length, dtype=np.int32))
            prompt_id_rows.append(np.full(sequence_length, prompt_id))
            prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            prompt_hash_rows.append(np.full(sequence_length, prompt_digest))
            prompt_records.append({
                "promptId": prompt_id,
                "promptSha256": prompt_digest,
                "tokenCount": sequence_length,
            })

        dataset_path = output_dir / f"{split_name}.npz"
        inputs_array = np.concatenate(input_rows, axis=0).astype(np.float32, copy=False)
        outputs_array = np.concatenate(output_rows, axis=0).astype(np.float32, copy=False)
        np.savez_compressed(
            dataset_path,
            inputs=inputs_array,
            outputs=outputs_array,
            token_ids=np.concatenate(token_rows),
            token_positions=np.concatenate(position_rows),
            prompt_ids=np.concatenate(prompt_id_rows),
            prompt_sha256=np.concatenate(prompt_hash_rows),
        )
        dataset_records[split_name] = {
            "file": dataset_path.name,
            "sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "samples": int(inputs_array.shape[0]),
            "inputShape": list(inputs_array.shape),
            "outputShape": list(outputs_array.shape),
            "activationProvenance": "natural_observed_model_inference",
            "accessMode": "instrumented_capture",
            "prompts": prompt_records,
        }
        split_summaries[split_name] = {
            "inputRms": float(np.sqrt(np.mean(np.square(inputs_array)))),
            "inputMean": float(np.mean(inputs_array)),
            "inputStd": float(np.std(inputs_array)),
            "inputAbsQuantiles": np.quantile(
                np.abs(inputs_array), [0.5, 0.9, 0.99, 1.0]
            ).tolist(),
            "directReplayMaxAbsDelta": split_max_direct_delta,
            "captureSeconds": time.perf_counter() - split_started,
        }

    torch.cuda.synchronize(device)
    capture_seconds = time.perf_counter() - capture_started
    probe_generator = torch.Generator(device="cpu").manual_seed(seed)
    generated_probes = (
        torch.randn((32, input_dim), generator=probe_generator, dtype=torch.float32)
        / math.sqrt(input_dim)
    )
    with torch.inference_mode():
        direct_probe_outputs = mlp(generated_probes.to(device)).float().cpu().numpy()

    del mlp_modules, mlp_view, mlp, model_instance, tokenizer, adapter
    gc.collect()
    torch.cuda.empty_cache()

    oracle_started = time.perf_counter()
    client = open_live_oracle(
        model=model,
        layer=layer,
        units="",
        device=str(device),
        max_calls=oracle_query_budget,
        work_dir=output_dir / "oracle-work",
        local_files_only=local_files_only,
        oracle_output_mode="full-mlp-vector",
        opaque_eval_only=True,
    )
    try:
        expected_metadata = {
            "hiddenSize": input_dim,
            "oracleResponseWidth": output_dim,
            "oracleOutputMode": "full-mlp-vector",
        }
        for key, expected in expected_metadata.items():
            if client.meta.get(key) != expected:
                raise RuntimeError(
                    f"full-module oracle metadata mismatch for {key}: "
                    f"{client.meta.get(key)!r} != {expected!r}"
                )
        forbidden = {
            "model", "modelType", "layer", "unit", "units", "unitCount",
            "intermediateSize", "device", "blockDevice", "outputChannel",
        }
        exposed = sorted(forbidden.intersection(client.meta))
        if exposed:
            raise RuntimeError(f"opaque full-module oracle exposed {exposed}")
        oracle_probe_outputs = client.eval(
            generated_probes.numpy(), stage="generated-full-module-smoke-probes"
        ).astype(np.float32)
        oracle_query_count = int(client.query_count)
        oracle_stderr = client.stderr_tail()
    finally:
        client.close()
    torch.cuda.synchronize(device)
    oracle_seconds = time.perf_counter() - oracle_started
    probe_delta = float(np.max(np.abs(oracle_probe_outputs - direct_probe_outputs)))
    probe_path = output_dir / "generated_oracle_probes.npz"
    np.savez_compressed(
        probe_path,
        inputs=generated_probes.numpy(),
        outputs=oracle_probe_outputs,
        provenance=np.full(
            len(generated_probes), "generated_oracle_probe", dtype="<U32"
        ),
    )
    dataset_records["generated_oracle_probes"] = {
        "file": probe_path.name,
        "sha256": hashlib.sha256(probe_path.read_bytes()).hexdigest(),
        "samples": int(len(generated_probes)),
        "inputShape": list(generated_probes.shape),
        "outputShape": list(oracle_probe_outputs.shape),
        "activationProvenance": "deliberately_generated_oracle_probes",
        "accessMode": "opaque_eval_only_oracle",
        "distribution": "iid_standard_normal_scaled_to_unit_expected_l2",
    }
    if oracle_query_count != len(generated_probes):
        raise RuntimeError(
            f"oracle query count {oracle_query_count} does not match probe count "
            f"{len(generated_probes)}"
        )

    workspace = estimate_measured_cubic_workspace(input_dim, 3.0)
    report = {
        "schemaVersion": 1,
        "kind": "gcl-real-qwen-activation-capture",
        "model": model,
        "requestedModel": model,
        "module": f"model.layers.{layer}.mlp",
        "architecture": {
            "modelType": "qwen2",
            "inputDim": input_dim,
            "outputDim": output_dim,
            "intermediateDim": intermediate_dim,
            "activation": "silu",
        },
        "accessMode": {
            "activationCapture": "instrumented",
            "fullModuleOracle": "opaque_eval_only",
            "solverReceivedWeights": False,
            "solverReceivedHooks": False,
            "solverReceivedHiddenUnitCount": False,
            "postHocGroundTruthUsed": False,
        },
        "datasets": dataset_records,
        "additionalPromptSources": prompt_sources,
        "activationDistribution": split_summaries,
        "oracleSmoke": {
            "queryCount": oracle_query_count,
            "queryBudget": oracle_query_budget,
            "unitCountExposedToOracleClient": False,
            "maxAbsDeltaAgainstIndependentDirectMlp": probe_delta,
            "elapsedSeconds": oracle_seconds,
            "oracleStderrTail": oracle_stderr[-1000:],
        },
        "recovery": {
            "status": "blocked_before_oracle_queries_by_workspace_guard",
            "reconstructionProduced": False,
            "functionalRecovery": "not_evaluated_no_reconstruction",
            "structuralRecovery": "not_evaluated_no_reconstruction",
            "distributionalGeneralization": "not_evaluated_no_reconstruction",
            "measuredCubicWorkspaceEstimate": workspace,
            "directionMultiple": 3.0,
            "workspaceBytesAtFourGiBLimit": 4 * 1024**3,
            "oracleCallsSpentByRecoverySolver": 0,
        },
        "performance": {
            "device": str(device),
            "cudaDeviceName": torch.cuda.get_device_name(device),
            "cudaTotalMemoryBytes": int(torch.cuda.get_device_properties(device).total_memory),
            "cudaPeakAllocatedBytesDuringActivationCapture": int(
                torch.cuda.max_memory_allocated(device)
            ),
            "modelLoadSeconds": load_seconds,
            "activationCaptureSeconds": capture_seconds,
            "activationSamplesPerSecond": (
                sum(int(record["samples"]) for name, record in dataset_records.items()
                    if name != "generated_oracle_probes") / capture_seconds
            ),
            "fullModuleOracleProbesPerSecond": (
                oracle_query_count / oracle_seconds if oracle_seconds > 0 else None
            ),
            "torchVersion": torch.__version__,
            "torchCudaVersion": torch.version.cuda,
        },
    }
    report_path = output_dir / "capture-report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def validate_qwen_real_activation_artifacts(
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    """Validate capture hashes, shapes, token provenance, and disjoint prompt splits."""
    output_dir = output_dir.resolve()
    report_path = output_dir / "capture-report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"capture report not found: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schemaVersion") != 1 or report.get("kind") != "gcl-real-qwen-activation-capture":
        raise ValueError("unsupported real-Qwen activation capture schema")
    architecture = report.get("architecture", {})
    input_dim = int(architecture.get("inputDim", 0))
    output_dim = int(architecture.get("outputDim", 0))
    if architecture.get("modelType") != "qwen2" or input_dim <= 0 or output_dim <= 0:
        raise ValueError("capture report has invalid Qwen architecture dimensions")
    access = report.get("accessMode", {})
    expected_access = {
        "activationCapture": "instrumented",
        "fullModuleOracle": "opaque_eval_only",
        "solverReceivedWeights": False,
        "solverReceivedHooks": False,
        "solverReceivedHiddenUnitCount": False,
        "postHocGroundTruthUsed": False,
    }
    if any(access.get(key) != value for key, value in expected_access.items()):
        raise ValueError("capture report violates the declared solver/oracle access boundary")

    natural_splits = (
        "recovery_observations",
        "functional_evaluation",
        "structural_evaluation",
        "distributional_analysis",
    )
    expected_splits = {*natural_splits, "generated_oracle_probes"}
    records = report.get("datasets", {})
    if set(records) != expected_splits:
        raise ValueError(f"capture datasets must be exactly {sorted(expected_splits)}")
    loaded: dict[str, dict[str, np.ndarray]] = {}
    seen_prompt_ids: set[str] = set()
    seen_prompt_hashes: set[str] = set()

    for split_name in sorted(expected_splits):
        record = records[split_name]
        filename = record.get("file")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"invalid artifact filename for {split_name}")
        artifact_path = output_dir / filename
        if artifact_path.resolve().parent != output_dir or not artifact_path.is_file():
            raise FileNotFoundError(f"dataset artifact not found: {artifact_path}")
        digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if digest != record.get("sha256"):
            raise ValueError(f"SHA-256 mismatch for {split_name}")
        with np.load(artifact_path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        required = (
            {"inputs", "outputs", "token_ids", "token_positions", "prompt_ids", "prompt_sha256"}
            if split_name in natural_splits
            else {"inputs", "outputs", "provenance"}
        )
        if set(arrays) != required:
            raise ValueError(f"unexpected array keys in {split_name}: {sorted(arrays)}")
        inputs = arrays["inputs"]
        outputs = arrays["outputs"]
        samples = int(record.get("samples", -1))
        if inputs.shape != (samples, input_dim) or outputs.shape != (samples, output_dim):
            raise ValueError(f"activation shape mismatch in {split_name}")
        if record.get("inputShape") != list(inputs.shape) or record.get("outputShape") != list(outputs.shape):
            raise ValueError(f"manifest shape mismatch in {split_name}")
        if not np.isfinite(inputs).all() or not np.isfinite(outputs).all():
            raise ValueError(f"non-finite activation value in {split_name}")

        if split_name in natural_splits:
            if record.get("activationProvenance") != "natural_observed_model_inference":
                raise ValueError(f"natural-activation provenance mismatch in {split_name}")
            if record.get("accessMode") != "instrumented_capture":
                raise ValueError(f"natural-activation access mode mismatch in {split_name}")
            token_ids = arrays["token_ids"]
            token_positions = arrays["token_positions"]
            prompt_ids = arrays["prompt_ids"].astype(str)
            prompt_hashes = arrays["prompt_sha256"].astype(str)
            if any(len(arrays[key]) != samples for key in ("token_ids", "token_positions", "prompt_ids", "prompt_sha256")):
                raise ValueError(f"token provenance length mismatch in {split_name}")
            manifest_prompts = record.get("prompts", [])
            manifest_by_id = {item["promptId"]: item for item in manifest_prompts}
            actual_prompt_ids = set(prompt_ids.tolist())
            if len(manifest_by_id) != len(manifest_prompts) or actual_prompt_ids != set(manifest_by_id):
                raise ValueError(f"prompt manifest mismatch in {split_name}")
            for prompt_id, prompt_record in manifest_by_id.items():
                prompt_hash = prompt_record.get("promptSha256")
                if prompt_id in seen_prompt_ids or prompt_hash in seen_prompt_hashes:
                    raise ValueError("natural-activation prompt splits are not disjoint")
                seen_prompt_ids.add(prompt_id)
                seen_prompt_hashes.add(prompt_hash)
                mask = prompt_ids == prompt_id
                positions = token_positions[mask]
                if len(positions) != int(prompt_record.get("tokenCount", -1)):
                    raise ValueError(f"token count mismatch for {prompt_id}")
                if not np.array_equal(positions, np.arange(len(positions), dtype=positions.dtype)):
                    raise ValueError(f"token positions are not contiguous for {prompt_id}")
                if not np.all(prompt_hashes[mask] == prompt_hash):
                    raise ValueError(f"prompt hash mismatch for {prompt_id}")
        else:
            if record.get("activationProvenance") != "deliberately_generated_oracle_probes":
                raise ValueError("generated probe provenance mismatch")
            if record.get("accessMode") != "opaque_eval_only_oracle":
                raise ValueError("generated probe access mode mismatch")
            if not np.all(arrays["provenance"].astype(str) == "generated_oracle_probe"):
                raise ValueError("generated probes contain non-generated provenance")
        loaded[split_name] = arrays

    prompt_sources = report.get("additionalPromptSources", {})
    if not isinstance(prompt_sources, dict):
        raise ValueError("additional prompt sources must be a mapping")
    required_prefixes = {
        "recovery_observations": "recovery-extra-",
        "functional_evaluation": "functional-extra-",
        "distributional_analysis": "distribution-extra-",
    }
    for split_name, source in prompt_sources.items():
        if split_name not in required_prefixes or not isinstance(source, dict):
            raise ValueError(f"invalid additional prompt source for {split_name!r}")
        filename = source.get("file")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"invalid additional prompt source filename for {split_name}")
        source_path = output_dir / filename
        if source_path.resolve().parent != output_dir or not source_path.is_file():
            raise FileNotFoundError(f"additional prompt source not found: {source_path}")
        source_bytes = source_path.read_bytes()
        if hashlib.sha256(source_bytes).hexdigest() != source.get("sha256"):
            raise ValueError(f"additional prompt source SHA-256 mismatch for {split_name}")
        source_rows = json.loads(source_bytes)
        if not isinstance(source_rows, list) or len(source_rows) != source.get("promptCount"):
            raise ValueError(f"additional prompt count mismatch for {split_name}")
        manifest_prompts = {
            item["promptId"]: item
            for item in records[split_name].get("prompts", [])
        }
        for item in source_rows:
            if (
                not isinstance(item, dict)
                or set(item) != {"promptId", "text"}
                or not isinstance(item["promptId"], str)
                or not item["promptId"].startswith(required_prefixes[split_name])
                or not isinstance(item["text"], str)
                or item["promptId"] not in manifest_prompts
            ):
                raise ValueError(f"additional prompt manifest mismatch for {split_name}")
            expected_hash = hashlib.sha256(item["text"].encode("utf-8")).hexdigest()
            if manifest_prompts[item["promptId"]].get("promptSha256") != expected_hash:
                raise ValueError(f"additional prompt hash mismatch for {item['promptId']}")

    oracle_smoke = report.get("oracleSmoke", {})
    generated_count = int(records["generated_oracle_probes"]["samples"])
    if (
        oracle_smoke.get("queryCount") != generated_count
        or oracle_smoke.get("queryCount", 0) > oracle_smoke.get("queryBudget", -1)
        or oracle_smoke.get("unitCountExposedToOracleClient") is not False
        or not math.isfinite(float(oracle_smoke.get("maxAbsDeltaAgainstIndependentDirectMlp", float("nan"))))
        or float(oracle_smoke["maxAbsDeltaAgainstIndependentDirectMlp"]) > 1e-5
    ):
        raise ValueError("oracle smoke result or query-budget record is invalid")
    recovery = report.get("recovery", {})
    if (
        recovery.get("reconstructionProduced") is not False
        or recovery.get("oracleCallsSpentByRecoverySolver") != 0
        or recovery.get("status") != "blocked_before_oracle_queries_by_workspace_guard"
    ):
        raise ValueError("capture report does not accurately describe the blocked recovery attempt")
    return report, loaded


def verify_qwen_real_activation_capture(
    model: str,
    output_dir: Path,
    device: torch.device,
    local_files_only: bool,
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Independently replay saved token sequences against the real Qwen module."""
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("real Qwen artifact verification requires an available CUDA device")
    started = time.perf_counter()
    report, datasets = validate_qwen_real_activation_artifacts(output_dir)
    if report.get("model") != model:
        raise ValueError(
            f"verification model {model!r} does not match capture model {report.get('model')!r}"
        )
    module = report.get("module", "")
    try:
        layer = int(module.split(".")[2])
    except (IndexError, ValueError) as error:
        raise ValueError(f"invalid module identity in capture report: {module!r}") from error
    if module != f"model.layers.{layer}.mlp":
        raise ValueError(f"unsupported module identity in capture report: {module!r}")

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from tools.llm_reader.adapters.registry import create_adapter

    adapter = create_adapter("qwen2-swiglu")
    tokenizer, model_instance = adapter.load_model_and_tokenizer(
        model, local_files_only=local_files_only, device=str(device)
    )
    del tokenizer
    mlp = adapter.mlp_modules(model_instance, [layer])[layer]._mlp
    natural_splits = (
        "recovery_observations",
        "functional_evaluation",
        "structural_evaluation",
        "distributional_analysis",
    )
    replay_results: dict[str, dict[str, Any]] = {}
    try:
        for split_name in natural_splits:
            arrays = datasets[split_name]
            prompt_ids = arrays["prompt_ids"].astype(str)
            prompt_results: dict[str, dict[str, float]] = {}
            for prompt_id in dict.fromkeys(prompt_ids.tolist()):
                mask = prompt_ids == prompt_id
                token_ids = arrays["token_ids"][mask].astype(np.int64, copy=False)
                expected_inputs = arrays["inputs"][mask]
                expected_outputs = arrays["outputs"][mask]
                input_ids = torch.as_tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
                attention_mask = torch.ones_like(input_ids)
                captured: dict[str, torch.Tensor] = {}

                def capture_input(_module: Any, inputs: tuple[torch.Tensor, ...]) -> None:
                    captured["input"] = inputs[0].detach()

                def capture_output(
                    _module: Any, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor
                ) -> None:
                    captured["output"] = output.detach()

                pre_handle = mlp.register_forward_pre_hook(capture_input)
                post_handle = mlp.register_forward_hook(capture_output)
                try:
                    with torch.inference_mode():
                        model_instance(input_ids=input_ids, attention_mask=attention_mask)
                finally:
                    pre_handle.remove()
                    post_handle.remove()
                if "input" not in captured or "output" not in captured:
                    raise RuntimeError(f"independent replay did not capture {prompt_id}")
                replay_input = captured["input"][0].float().cpu().numpy()
                replay_output = captured["output"][0].float().cpu().numpy()
                if replay_input.shape != expected_inputs.shape or replay_output.shape != expected_outputs.shape:
                    raise ValueError(f"independent replay shape mismatch for {prompt_id}")
                input_delta = float(np.max(np.abs(replay_input - expected_inputs)))
                output_delta = float(np.max(np.abs(replay_output - expected_outputs)))
                if input_delta > tolerance or output_delta > tolerance:
                    raise ValueError(
                        f"independent replay exceeded tolerance for {prompt_id}: "
                        f"input={input_delta}, output={output_delta}, tolerance={tolerance}"
                    )
                prompt_results[prompt_id] = {
                    "inputMaxAbsDelta": input_delta,
                    "outputMaxAbsDelta": output_delta,
                }
            replay_results[split_name] = {
                "samples": int(arrays["inputs"].shape[0]),
                "prompts": prompt_results,
                "maxInputAbsDelta": max(
                    values["inputMaxAbsDelta"] for values in prompt_results.values()
                ),
                "maxOutputAbsDelta": max(
                    values["outputMaxAbsDelta"] for values in prompt_results.values()
                ),
            }

        generated = datasets["generated_oracle_probes"]
        with torch.inference_mode():
            expected_generated = mlp(torch.as_tensor(generated["inputs"], device=device)).float().cpu().numpy()
        generated_delta = float(np.max(np.abs(expected_generated - generated["outputs"])))
        if generated_delta > tolerance:
            raise ValueError(
                f"independent oracle-probe replay exceeded tolerance: "
                f"{generated_delta} > {tolerance}"
            )
    finally:
        del mlp, model_instance, adapter
        gc.collect()
        torch.cuda.empty_cache()

    verification = {
        "schemaVersion": 1,
        "kind": "gcl-real-qwen-activation-verification",
        "status": "verified",
        "captureReportSha256": hashlib.sha256(
            (Path(output_dir).resolve() / "capture-report.json").read_bytes()
        ).hexdigest(),
        "model": model,
        "module": module,
        "verificationMode": "independent_posthoc_white_box_replay",
        "groundTruthWeightsReadByVerifier": True,
        "groundTruthWeightsProvidedToRecoverySolver": False,
        "tolerance": tolerance,
        "artifactIntegrity": {
            split_name: {
                "sha256": report["datasets"][split_name]["sha256"],
                "samples": report["datasets"][split_name]["samples"],
            }
            for split_name in sorted(report["datasets"])
        },
        "naturalDatasetReplays": replay_results,
        "generatedOracleProbeMaxAbsDelta": generated_delta,
        "verificationSeconds": time.perf_counter() - started,
    }
    verification_path = Path(output_dir).resolve() / "verification-report.json"
    verification_path.write_text(
        json.dumps(verification, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return verification


class _SwiGLUDistillationStudent(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        components: int,
        generator: torch.Generator,
        device: torch.device,
        gate: torch.Tensor | None = None,
        up: torch.Tensor | None = None,
        down: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.components = components
        if gate is None:
            gate = torch.randn(
                components, input_dim, generator=generator, dtype=torch.float32
            ) * (0.5 / math.sqrt(input_dim))
        if up is None:
            up = torch.randn(
                components, input_dim, generator=generator, dtype=torch.float32
            ) * (0.5 / math.sqrt(input_dim))
        if down is None:
            down = torch.randn(
                output_dim, components, generator=generator, dtype=torch.float32
            ) * (0.3 / math.sqrt(components))
        self.gate = torch.nn.Parameter(gate.to(device=device, dtype=torch.float32))
        self.up = torch.nn.Parameter(up.to(device=device, dtype=torch.float32))
        self.down = torch.nn.Parameter(down.to(device=device, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gate_projection = torch.nn.functional.linear(inputs, self.gate)
        up_projection = torch.nn.functional.linear(inputs, self.up)
        return torch.nn.functional.linear(
            torch.nn.functional.silu(gate_projection) * up_projection,
            self.down,
        )

    def grow(self, components: int, generator: torch.Generator) -> None:
        if components <= self.components:
            raise ValueError("student growth must increase the candidate component count")
        additional = components - self.components
        device = self.gate.device
        new_gate = torch.randn(
            additional, self.input_dim, generator=generator, dtype=torch.float32
        ) * (0.5 / math.sqrt(self.input_dim))
        new_up = torch.randn(
            additional, self.input_dim, generator=generator, dtype=torch.float32
        ) * (0.5 / math.sqrt(self.input_dim))
        new_down = torch.randn(
            self.output_dim, additional, generator=generator, dtype=torch.float32
        ) * (0.3 / math.sqrt(components))
        self.gate = torch.nn.Parameter(torch.cat([self.gate.detach(), new_gate.to(device)], dim=0))
        self.up = torch.nn.Parameter(torch.cat([self.up.detach(), new_up.to(device)], dim=0))
        self.down = torch.nn.Parameter(torch.cat([self.down.detach(), new_down.to(device)], dim=1))
        self.components = components


def _swiglu_candidate_widths(max_components: int) -> list[int]:
    if max_components <= 0:
        raise ValueError("max_components must be positive")
    widths = [1]
    while widths[-1] < max_components:
        widths.append(min(max_components, widths[-1] * 2))
    return widths


def _functional_metrics(
    reference: np.ndarray,
    predicted: np.ndarray,
    inputs: np.ndarray | None = None,
) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if reference.shape != predicted.shape or reference.ndim != 2:
        raise ValueError("functional metrics require matching 2D reference and prediction arrays")
    residual = predicted - reference
    mse = float(np.mean(np.square(residual)))
    reference_variance = float(np.mean(np.square(reference - reference.mean(axis=0, keepdims=True))))
    reference_energy = float(np.mean(np.square(reference)))
    sample_cosine = np.sum(reference * predicted, axis=1) / np.maximum(
        np.linalg.norm(reference, axis=1) * np.linalg.norm(predicted, axis=1), 1e-30
    )
    result: dict[str, Any] = {
        "samples": int(reference.shape[0]),
        "relativeError": float(np.linalg.norm(residual) / max(np.linalg.norm(reference), 1e-30)),
        "normalizedMse": mse / max(reference_variance, 1e-30),
        "meanSquaredError": mse,
        "referenceRms": math.sqrt(reference_energy),
        "cosineSimilarityMean": float(np.mean(sample_cosine)),
        "cosineSimilarityMedian": float(np.median(sample_cosine)),
    }
    if inputs is not None and len(inputs) == len(reference):
        norms = np.linalg.norm(np.asarray(inputs, dtype=np.float64), axis=1)
        quantile_edges = np.quantile(norms, [0.0, 0.25, 0.5, 0.75, 1.0])
        density_rows = []
        for index in range(4):
            mask = (
                (norms >= quantile_edges[index])
                & (norms <= quantile_edges[index + 1])
                if index == 3
                else (norms >= quantile_edges[index])
                & (norms < quantile_edges[index + 1])
            )
            if mask.any():
                density_rows.append({
                    "inputNormLow": float(quantile_edges[index]),
                    "inputNormHigh": float(quantile_edges[index + 1]),
                    "metrics": _functional_metrics(reference[mask], predicted[mask]),
                })
        result["inputDensityQuartiles"] = density_rows
    return result


def recover_swiglu_by_adaptive_oracle_distillation(
    oracle: Callable[[np.ndarray], np.ndarray],
    input_dim: int,
    output_dim: int,
    recovery_inputs: np.ndarray,
    recovery_outputs: np.ndarray,
    *,
    max_components: int,
    oracle_query_budget: int,
    initial_train_queries: int,
    validation_queries: int,
    adaptive_batch_queries: int,
    steps_per_stage: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
    recovery_metadata: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Fit an adaptive-width SwiGLU student from oracle observations only.

    Candidate width is selected by held-out prediction, not copied from the teacher.
    Oracle inputs cover radial shells calibrated from recovery observations; additional
    batches target the shell with the largest normalized validation residual.
    """
    if min(input_dim, output_dim, max_components, oracle_query_budget, batch_size) <= 0:
        raise ValueError("dimensions, component cap, query budget, and batch size must be positive")
    if initial_train_queries < 4 or validation_queries < 4 or adaptive_batch_queries < 0:
        raise ValueError("oracle sample counts must be non-negative and initial/validation counts >= 4")
    if steps_per_stage <= 0:
        raise ValueError("steps_per_stage must be positive")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if recovery_inputs.ndim != 2 or recovery_inputs.shape[1] != input_dim:
        raise ValueError("recovery_inputs have the wrong shape")
    if recovery_outputs.shape != (recovery_inputs.shape[0], output_dim):
        raise ValueError("recovery_outputs have the wrong shape")
    if recovery_metadata is not None and any(
        values.shape[0] != recovery_inputs.shape[0]
        for values in recovery_metadata.values()
    ):
        raise ValueError("recovery metadata must have one row per natural observation")
    if not np.isfinite(recovery_inputs).all() or not np.isfinite(recovery_outputs).all():
        raise ValueError("recovery observations must be finite")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("adaptive Qwen SwiGLU distillation requires an available CUDA device")
    initial_queries = initial_train_queries + validation_queries
    if initial_queries > oracle_query_budget:
        raise OracleQueryBudgetExceeded(
            requested_points=initial_queries,
            spent_points=0,
            max_oracle_points=oracle_query_budget,
        )

    import resource

    query_count = 0
    rng = np.random.default_rng(seed)
    rms_by_coordinate = np.sqrt(np.mean(np.square(recovery_inputs.astype(np.float64)), axis=0))
    global_rms = float(np.sqrt(np.mean(np.square(recovery_inputs.astype(np.float64)))))
    if not math.isfinite(global_rms) or global_rms <= 0:
        raise ValueError("recovery observations do not define a positive activation scale")
    rms_by_coordinate = np.maximum(rms_by_coordinate, global_rms * 0.05)
    shell_multipliers = np.asarray([0.45, 0.75, 1.0, 1.4], dtype=np.float64)

    def draw_shell(shell_index: int, count: int) -> np.ndarray:
        return (
            rng.normal(size=(count, input_dim))
            * rms_by_coordinate[None, :]
            * shell_multipliers[shell_index]
        ).astype(np.float32)

    def query(points: np.ndarray, stage: str) -> np.ndarray:
        nonlocal query_count
        requested = int(points.shape[0])
        if query_count + requested > oracle_query_budget:
            raise OracleQueryBudgetExceeded(
                requested_points=requested,
                spent_points=query_count,
                max_oracle_points=oracle_query_budget,
            )
        answers: list[np.ndarray] = []
        for start in range(0, requested, 2048):
            stop = min(start + 2048, requested)
            values = np.asarray(oracle(points[start:stop]), dtype=np.float32)
            expected_shape = (stop - start, output_dim)
            if values.shape != expected_shape:
                raise ValueError(
                    f"oracle output shape {values.shape} does not match {expected_shape} at {stage}"
                )
            if not np.isfinite(values).all():
                raise ValueError(f"oracle returned non-finite values at {stage}")
            query_count += stop - start
            answers.append(values)
        return np.concatenate(answers, axis=0)

    per_shell_train = [initial_train_queries // 4] * 4
    for index in range(initial_train_queries % 4):
        per_shell_train[index] += 1
    train_generated_inputs = np.concatenate([
        draw_shell(index, count) for index, count in enumerate(per_shell_train)
    ])
    train_generated_targets = query(train_generated_inputs, "initial-radial-shell-training")
    per_shell_validation = [validation_queries // 4] * 4
    for index in range(validation_queries % 4):
        per_shell_validation[index] += 1
    validation_inputs = np.concatenate([
        draw_shell(index, count) for index, count in enumerate(per_shell_validation)
    ])
    validation_shells = np.concatenate([
        np.full(count, index, dtype=np.int16)
        for index, count in enumerate(per_shell_validation)
    ])
    validation_targets = query(validation_inputs, "radial-shell-validation")

    train_inputs = np.concatenate([
        recovery_inputs.astype(np.float32, copy=False),
        train_generated_inputs,
    ])
    train_targets = np.concatenate([
        recovery_outputs.astype(np.float32, copy=False),
        train_generated_targets,
    ])
    generator = torch.Generator(device="cpu").manual_seed(seed + 501)
    student: _SwiGLUDistillationStudent | None = None
    stage_results: list[dict[str, Any]] = []
    selected_state: dict[str, torch.Tensor] | None = None
    selected_score = float("inf")
    selected_stage = -1
    adaptive_queries: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats(device)
    train_started = time.perf_counter()
    widths = _swiglu_candidate_widths(max_components)
    for stage_index, width in enumerate(widths):
        x_train = torch.from_numpy(train_inputs)
        y_train = torch.from_numpy(train_targets)
        target_energy = float(
            np.einsum("ij,ij->", train_targets, train_targets, dtype=np.float64)
            / train_targets.size
        )
        training_loss_scale = torch.tensor(
            max(target_energy, 1e-8),
            dtype=torch.float32,
            device=device,
        )
        student = _SwiGLUDistillationStudent(
            input_dim, output_dim, width, generator, device
        )
        with torch.no_grad():
            calibration_count = min(512, int(x_train.shape[0]))
            calibration_inputs = x_train[:calibration_count].to(device)
            calibration_targets = y_train[:calibration_count].to(device)
            initial_output_rms = student(calibration_inputs).square().mean().sqrt()
            target_output_rms = calibration_targets.square().mean().sqrt()
            if (
                not torch.isfinite(initial_output_rms)
                or not torch.isfinite(target_output_rms)
                or float(initial_output_rms) <= 1e-12
                or float(target_output_rms) <= 1e-12
            ):
                raise RuntimeError(f"invalid student output scale at candidate width {width}")
            output_scale = target_output_rms / initial_output_rms
            student.down.mul_(output_scale)
        del calibration_inputs, calibration_targets
        optimizer = torch.optim.AdamW(
            student.parameters(), lr=learning_rate, weight_decay=1e-6
        )
        for _step in range(steps_per_stage):
            indices = torch.randint(
                x_train.shape[0], (min(batch_size, x_train.shape[0]),), generator=generator
            )
            batch_x = x_train[indices].to(device, non_blocking=True)
            batch_y = y_train[indices].to(device, non_blocking=True)
            prediction = student(batch_x)
            loss = (prediction - batch_y).square().mean() / training_loss_scale
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite distillation loss at candidate width {width}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=10.0)
            optimizer.step()

        with torch.inference_mode():
            x_validation = torch.as_tensor(validation_inputs, dtype=torch.float32, device=device)
            y_validation = torch.as_tensor(validation_targets, dtype=torch.float32, device=device)
            validation_prediction = student(x_validation)
            residual = validation_prediction - y_validation
            normalized_mse = float(
                (residual.square().mean() / y_validation.var(unbiased=False).clamp_min(1e-12)).item()
            )
            per_shell_error = []
            for shell_index in range(4):
                mask = torch.as_tensor(
                    validation_shells == shell_index, device=device, dtype=torch.bool
                )
                if mask.any():
                    shell_residual = residual[mask]
                    shell_target = y_validation[mask]
                    per_shell_error.append(
                        float(
                            (
                                shell_residual.square().mean()
                                / shell_target.var(unbiased=False).clamp_min(1e-12)
                            ).item()
                        )
                    )
                else:
                    per_shell_error.append(None)
        # BIC's sample count is the training set that estimated the parameters, not
        # the held-out set used to compare predictive error.
        observations = max(train_targets.size, 2)
        parameter_count = width * (2 * input_dim + output_dim)
        bic_penalty = parameter_count * math.log(observations) / observations
        bic_proxy_score = math.log(max(normalized_mse, 1e-12)) + bic_penalty
        selection_score = normalized_mse
        candidate = {
            "candidateComponents": width,
            "effectiveParameterCount": parameter_count,
            "validationNormalizedMse": normalized_mse,
            "validationShellNormalizedMse": per_shell_error,
            "bicProxyPenalty": bic_penalty,
            "bicProxyScore": bic_proxy_score,
            "selectionScore": selection_score,
            "initialOutputScale": float(output_scale.item()),
            "initialOutputRms": float(initial_output_rms.item()),
            "trainingTargetOutputRms": float(target_output_rms.item()),
            "trainingObservationCount": int(train_inputs.shape[0]),
        }
        stage_results.append(candidate)
        if selection_score < selected_score:
            selected_score = selection_score
            selected_stage = stage_index
            selected_state = {
                name: value.detach().cpu().clone()
                for name, value in student.state_dict().items()
            }

        if (
            adaptive_batch_queries > 0
            and stage_index in {2, 4, 6, 8}
            and query_count + adaptive_batch_queries <= oracle_query_budget
        ):
            valid_errors = [
                (value, shell)
                for shell, value in enumerate(per_shell_error)
                if value is not None
            ]
            selected_shell = max(valid_errors)[1] if valid_errors else 0
            active_inputs = draw_shell(selected_shell, adaptive_batch_queries)
            active_targets = query(active_inputs, f"adaptive-residual-shell-{selected_shell}")
            train_inputs = np.concatenate([train_inputs, active_inputs])
            train_targets = np.concatenate([train_targets, active_targets])
            adaptive_queries.append({
                "afterCandidateComponents": width,
                "selectedShell": selected_shell,
                "selectedShellMultiplier": float(shell_multipliers[selected_shell]),
                "validationNormalizedMseByShell": per_shell_error,
                "oraclePoints": adaptive_batch_queries,
            })
        del x_train, y_train

    if student is None or selected_state is None:
        raise RuntimeError("adaptive distillation produced no candidate model")
    stage_selected_width = widths[selected_stage]
    selected_student = _SwiGLUDistillationStudent(
        input_dim, output_dim, stage_selected_width, generator, device
    )
    selected_student.load_state_dict(selected_state)
    student = selected_student.eval()
    with torch.inference_mode():
        predicted_validation = student(
            torch.as_tensor(validation_inputs, dtype=torch.float32, device=device)
        ).float().cpu().numpy()
        validation_coordinates = torch.nn.functional.linear(
            torch.as_tensor(validation_inputs, dtype=torch.float32, device=device),
            student.gate,
        )
        validation_up = torch.nn.functional.linear(
            torch.as_tensor(validation_inputs, dtype=torch.float32, device=device),
            student.up,
        )
        component_strength = (
            torch.nn.functional.silu(validation_coordinates)
            * validation_up
        ).square().mean(dim=0).sqrt() * student.down.norm(dim=0)
    selected_component_count = student.components

    training_seconds = time.perf_counter() - train_started
    torch.cuda.synchronize(device)
    peak_vram = int(torch.cuda.max_memory_allocated(device))
    max_rss = _proc_vm_hwm_bytes()
    return {
        "schemaVersion": 1,
        "kind": "gcl-oracle-only-adaptive-swiglu-distillation",
        "solverAccess": "oracle-only",
        "weightsUsedBySolver": False,
        "hooksUsedBySolver": False,
        "hiddenUnitCountProvidedToSolver": False,
        "teacherComponentIdentitiesProvidedToSolver": False,
        "inputDim": input_dim,
        "outputDim": output_dim,
        "publicActivationFamily": "silu-swiglu",
        "biases": False,
        "candidateWidthPolicy": "geometric-growth-selected-by-held-out-normalized-MSE",
        "studentOptimizer": {
            "name": "AdamW",
            "learningRate": learning_rate,
            "weightDecay": 1e-6,
        },
        "initializationCalibration": "match_student_output_RMS_to_measured_training_output_RMS",
        "maxCandidateComponents": max_components,
        "selectedCandidateComponents": selected_component_count,
        "studentWidthSelectedByMeasurements": True,
        "teacherComponentCountInferred": False,
        "selectionScore": selected_score,
        "selectedCandidateValidationNormalizedMse": stage_results[selected_stage]["validationNormalizedMse"],
        "componentStrengthRms": component_strength.detach().cpu().numpy().astype(np.float32),
        "candidateStages": stage_results,
        "adaptiveQueryRounds": adaptive_queries,
        "queryAccounting": {
            "oraclePoints": query_count,
            "oracleQueryBudget": oracle_query_budget,
            "reusedRecoveryObservations": int(recovery_inputs.shape[0]),
            "initialGeneratedTrainingQueries": int(initial_train_queries),
            "validationQueries": int(validation_queries),
            "adaptiveQueries": int(sum(item["oraclePoints"] for item in adaptive_queries)),
        },
        "validation": _functional_metrics(validation_targets, predicted_validation, validation_inputs),
        "trainingSeconds": training_seconds,
        "performance": {
            "device": str(device),
            "cudaDeviceName": torch.cuda.get_device_name(device),
            "peakAllocatedVramBytes": peak_vram,
            "solverProcessPeakRssBytes": max_rss,
            "rssMeasurementSource": "/proc/self/status:VmHWM",
            "torchVersion": torch.__version__,
            "torchCudaVersion": torch.version.cuda,
        },
        "trainingData": {
            "inputs": train_inputs.astype(np.float32, copy=False),
            "outputs": train_targets.astype(np.float32, copy=False),
            "provenance": np.concatenate([
                np.full(recovery_inputs.shape[0], "natural_observed_model_inference", dtype="<U48"),
                np.full(initial_train_queries, "generated_radial_shell_oracle_probe", dtype="<U48"),
                np.full(
                    sum(item["oraclePoints"] for item in adaptive_queries),
                    "adaptive_residual_shell_oracle_probe",
                    dtype="<U48",
                ),
            ]),
            **{
                f"natural_{name}": np.concatenate([
                    values,
                    np.full(
                        (initial_train_queries + sum(
                            item["oraclePoints"] for item in adaptive_queries
                        ), *values.shape[1:]),
                        -1 if np.issubdtype(values.dtype, np.number) else "",
                        dtype=values.dtype,
                    ),
                ])
                for name, values in (recovery_metadata or {}).items()
            },
        },
        "validationData": {
            "inputs": validation_inputs.astype(np.float32, copy=False),
            "outputs": validation_targets.astype(np.float32, copy=False),
            "shells": validation_shells,
            "provenance": np.full(
                validation_queries, "generated_radial_shell_validation_probe", dtype="<U48"
            ),
        },
        "reconstruction": {
            "gate": student.gate.detach().cpu().numpy().astype(np.float32),
            "up": student.up.detach().cpu().numpy().astype(np.float32),
            "down": student.down.detach().cpu().numpy().astype(np.float32),
        },
    }


def run_qwen_oracle_distillation(
    capture_dir: Path,
    output_dir: Path,
    model: str,
    layer: int,
    device: torch.device,
    local_files_only: bool,
    seed: int,
    max_components: int,
    oracle_query_budget: int,
    initial_train_queries: int,
    validation_queries: int,
    adaptive_batch_queries: int,
    steps_per_stage: int,
    batch_size: int,
    learning_rate: float,
) -> dict[str, Any]:
    """Run the adaptive-width solver through the opaque real-module oracle."""
    capture_report, datasets = validate_qwen_real_activation_artifacts(capture_dir)
    if capture_report.get("architecture", {}).get("modelType") != "qwen2":
        raise ValueError("the adaptive SwiGLU route currently requires a Qwen2 capture")
    captured_model = capture_report.get("requestedModel", capture_report.get("model"))
    if captured_model != model or capture_report.get("module") != f"model.layers.{layer}.mlp":
        raise ValueError(
            "capture model/module must match the requested oracle target: "
            f"capture=({captured_model!r}, {capture_report.get('module')!r}) "
            f"requested=({model!r}, 'model.layers.{layer}.mlp')"
        )
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite distillation artifacts: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    recovery = datasets["recovery_observations"]
    memory_samples: list[tuple[int, int]] = []
    monitor_stop = threading.Event()

    def monitor_device_memory() -> None:
        while not monitor_stop.is_set():
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            memory_samples.append((int(total_bytes - free_bytes), int(total_bytes)))
            monitor_stop.wait(0.05)

    memory_monitor = threading.Thread(target=monitor_device_memory, daemon=True)
    memory_monitor.start()
    client = None
    try:
        client = open_live_oracle(
            model=model,
            layer=layer,
            units="",
            device=str(device),
            max_calls=oracle_query_budget,
            work_dir=output_dir / "oracle-work",
            local_files_only=local_files_only,
            oracle_output_mode="full-mlp-vector",
            opaque_eval_only=True,
        )
        expected = {
            "hiddenSize": int(capture_report["architecture"]["inputDim"]),
            "oracleResponseWidth": int(capture_report["architecture"]["outputDim"]),
            "oracleOutputMode": "full-mlp-vector",
        }
        for key, value in expected.items():
            if client.meta.get(key) != value:
                raise RuntimeError(
                    f"opaque oracle metadata mismatch for {key}: "
                    f"{client.meta.get(key)!r} != {value!r}"
                )
        forbidden = {
            "model", "modelType", "layer", "unit", "units", "unitCount",
            "intermediateSize", "device", "blockDevice", "outputChannel",
        }
        exposed = sorted(forbidden.intersection(client.meta))
        if exposed:
            raise RuntimeError(f"opaque oracle exposed forbidden solver metadata: {exposed}")
        solver = recover_swiglu_by_adaptive_oracle_distillation(
            oracle=lambda points: client.eval(points, stage="adaptive-swiglu-distillation"),
            input_dim=expected["hiddenSize"],
            output_dim=expected["oracleResponseWidth"],
            recovery_inputs=recovery["inputs"],
            recovery_outputs=recovery["outputs"],
            max_components=max_components,
            oracle_query_budget=oracle_query_budget,
            initial_train_queries=initial_train_queries,
            validation_queries=validation_queries,
            adaptive_batch_queries=adaptive_batch_queries,
            steps_per_stage=steps_per_stage,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            device=device,
            recovery_metadata={
                key: recovery[key]
                for key in ("token_ids", "token_positions", "prompt_ids", "prompt_sha256")
            },
        )
        if int(solver["queryAccounting"]["oraclePoints"]) != client.query_count:
            raise RuntimeError(
                "distillation and opaque-oracle query counters disagree: "
                f"{solver['queryAccounting']['oraclePoints']} != {client.query_count}"
            )
        oracle_stderr = client.stderr_tail()[-1000:]
    finally:
        if client is not None:
            client.close()
        monitor_stop.set()
        memory_monitor.join()

    if not memory_samples:
        raise RuntimeError("CUDA memory monitor produced no samples")
    training = solver.pop("trainingData")
    validation = solver.pop("validationData")
    reconstruction = solver.pop("reconstruction")
    component_strength = solver.pop("componentStrengthRms")
    artifact_arrays = {
        "reconstruction.npz": reconstruction,
        "distillation-training-observations.npz": training,
        "distillation-validation-probes.npz": validation,
        "component-strength.npz": {"rms": component_strength},
    }
    artifact_records: dict[str, dict[str, Any]] = {}
    for filename, arrays in artifact_arrays.items():
        path = output_dir / filename
        np.savez_compressed(path, **arrays)
        artifact_records[filename] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "arrays": {
                name: list(value.shape) for name, value in arrays.items()
            },
        }
    solver["oracleMetadataKeys"] = sorted(client.meta)
    solver["opaqueOracleStderrTail"] = oracle_stderr
    solver["sourceCaptureReportSha256"] = hashlib.sha256(
        (Path(capture_dir).resolve() / "capture-report.json").read_bytes()
    ).hexdigest()
    solver["artifacts"] = artifact_records
    solver["teacherArchitectureKnownToSolver"] = {
        "inputDim": solver["inputDim"],
        "outputDim": solver["outputDim"],
        "activationFamily": "silu-swiglu",
        "hiddenComponents": "not provided",
    }
    solver["targetModel"] = model
    solver["targetModule"] = f"model.layers.{layer}.mlp"
    solver["captureNaturalRecoveryProvenancePreserved"] = True
    solver["performance"]["peakWholeDeviceUsedBytesDuringOracleAndTraining"] = max(
        used for used, _ in memory_samples
    )
    solver["performance"]["minimumWholeDeviceFreeBytesDuringOracleAndTraining"] = min(
        total - used for used, total in memory_samples
    )
    solver["performance"]["wholeDeviceMemoryMonitorSamples"] = len(memory_samples)
    report_path = output_dir / "distillation-report.json"
    report_path.write_text(
        json.dumps(solver, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return solver


def run_qwen_oracle_split_continuation(
    capture_dir: Path,
    output_dir: Path,
    model: str,
    layer: int,
    device: torch.device,
    local_files_only: bool,
    seed: int,
    max_components: int,
    query_budget: int,
    training_queries: int,
    validation_queries: int,
    adaptive_batch_queries: int,
    steps_per_stage: int,
    batch_size: int,
    learning_rate: float,
    projection_interval: int,
    projection_batch_size: int,
    split_perturbation: float,
    projection_ridge: float,
    natural_training_batch_fraction: float,
    directional_derivative_weight: float,
    directional_difference_step: float,
    directional_derivative_query_fraction: float,
) -> dict[str, Any]:
    """Run function-preserving SwiGLU continuation against the opaque live oracle."""
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from tools.llm_reader.swiglu_split_continuation import recover_by_split_continuation

    capture_report, datasets = validate_qwen_real_activation_artifacts(capture_dir)
    if capture_report.get("architecture", {}).get("modelType") != "qwen2":
        raise ValueError("the split-continuation route currently requires a Qwen2 capture")
    captured_model = capture_report.get("requestedModel", capture_report.get("model"))
    if captured_model != model or capture_report.get("module") != f"model.layers.{layer}.mlp":
        raise ValueError("capture model/module does not match the requested oracle target")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite continuation artifacts: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    recovery = datasets["recovery_observations"]
    memory_samples: list[tuple[int, int]] = []
    host_rss_samples: list[int] = []
    oracle_rss_samples: list[int] = []
    oracle_hwm_samples: list[int] = []
    process_tree_pss_samples: list[int] = []
    import resource

    statm_path = Path("/proc/self/statm")
    page_size = resource.getpagesize()
    monitor_stop = threading.Event()
    client = None

    def monitor_device_memory() -> None:
        while not monitor_stop.is_set():
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            memory_samples.append((int(total_bytes - free_bytes), int(total_bytes)))
            if statm_path.is_file():
                resident_pages = int(statm_path.read_text(encoding="ascii").split()[1])
                host_rss_samples.append(resident_pages * page_size)
            parent_pss = _proc_smaps_pss_bytes("self")
            oracle_process = (
                None
                if client is None
                else getattr(client, "process", None)
            )
            oracle_pid = None if oracle_process is None else oracle_process.pid
            if oracle_pid is not None:
                oracle_rss = _proc_status_memory_bytes(oracle_pid, "VmRSS")
                oracle_hwm = _proc_status_memory_bytes(oracle_pid, "VmHWM")
                oracle_pss = _proc_smaps_pss_bytes(oracle_pid)
                if oracle_rss is not None:
                    oracle_rss_samples.append(oracle_rss)
                if oracle_hwm is not None:
                    oracle_hwm_samples.append(oracle_hwm)
                if parent_pss is not None and oracle_pss is not None:
                    process_tree_pss_samples.append(parent_pss + oracle_pss)
            monitor_stop.wait(0.05)

    memory_monitor = threading.Thread(target=monitor_device_memory, daemon=True)
    memory_monitor.start()
    oracle_query_points = 0
    try:
        client = open_live_oracle(
            model=model,
            layer=layer,
            units="",
            device=str(device),
            max_calls=query_budget,
            work_dir=output_dir / "oracle-work",
            local_files_only=local_files_only,
            oracle_output_mode="full-mlp-vector",
            opaque_eval_only=True,
        )
        input_dim = int(capture_report["architecture"]["inputDim"])
        output_dim = int(capture_report["architecture"]["outputDim"])
        expected = {
            "hiddenSize": input_dim,
            "oracleResponseWidth": output_dim,
            "oracleOutputMode": "full-mlp-vector",
        }
        for key, value in expected.items():
            if client.meta.get(key) != value:
                raise RuntimeError(
                    f"opaque oracle metadata mismatch for {key}: "
                    f"{client.meta.get(key)!r} != {value!r}"
                )
        forbidden = {
            "model", "modelType", "layer", "unit", "units", "unitCount",
            "intermediateSize", "device", "blockDevice", "outputChannel",
        }
        exposed = sorted(forbidden.intersection(client.meta))
        if exposed:
            raise RuntimeError(f"opaque oracle exposed forbidden solver metadata: {exposed}")

        def counted_oracle(points: np.ndarray) -> np.ndarray:
            nonlocal oracle_query_points
            result = client.eval(points, stage="swiglu-split-continuation")
            oracle_query_points += len(points)
            return result

        try:
            solver = recover_by_split_continuation(
                oracle=counted_oracle,
                input_dim=input_dim,
                output_dim=output_dim,
                recovery_inputs=recovery["inputs"],
                recovery_outputs=recovery["outputs"],
                max_components=max_components,
                query_budget=query_budget,
                training_queries=training_queries,
                validation_queries=validation_queries,
                adaptive_batch_queries=adaptive_batch_queries,
                steps_per_stage=steps_per_stage,
                batch_size=batch_size,
                learning_rate=learning_rate,
                projection_interval=projection_interval,
                projection_batch_size=projection_batch_size,
                split_perturbation=split_perturbation,
                projection_ridge=projection_ridge,
                seed=seed,
                device=device,
                natural_training_batch_fraction=natural_training_batch_fraction,
                directional_derivative_weight=directional_derivative_weight,
                directional_difference_step=directional_difference_step,
                directional_derivative_query_fraction=(
                    directional_derivative_query_fraction
                ),
                recovery_metadata={
                    key: recovery[key]
                    for key in ("token_ids", "token_positions", "prompt_ids", "prompt_sha256")
                },
            )
        except Exception as error:
            (output_dir / "solver-failure-report.json").write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "kind": "gcl-oracle-only-swiglu-split-continuation-failure",
                        "status": "failed",
                        "errorType": type(error).__name__,
                        "error": str(error),
                        "solverAccess": "oracle-only",
                        "weightsUsedBySolver": False,
                        "hooksUsedBySolver": False,
                        "hiddenUnitCountProvidedToSolver": False,
                        "inputDim": input_dim,
                        "outputDim": output_dim,
                        "queryAccounting": {
                            "oraclePointsSuccessfullyReturned": oracle_query_points,
                            "opaqueOracleReportedQueries": client.query_count,
                            "configuredBudget": query_budget,
                        },
                        "trainingConfig": {
                            "maxComponents": max_components,
                            "trainingQueries": training_queries,
                            "validationQueries": validation_queries,
                            "adaptiveBatchQueries": adaptive_batch_queries,
                            "stepsPerStage": steps_per_stage,
                            "batchSize": batch_size,
                            "learningRate": learning_rate,
                            "splitPerturbation": split_perturbation,
                            "projectionRidge": projection_ridge,
                            "naturalTrainingBatchFraction": natural_training_batch_fraction,
                            "directionalDerivativeWeight": directional_derivative_weight,
                            "directionalDifferenceStep": directional_difference_step,
                            "directionalDerivativeQueryFraction": (
                                directional_derivative_query_fraction
                            ),
                        },
                    },
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            raise
        if solver["queryAccounting"]["oraclePoints"] != client.query_count:
            raise RuntimeError(
                "continuation and opaque-oracle query counters disagree: "
                f"{solver['queryAccounting']['oraclePoints']} != {client.query_count}"
            )
        oracle_meta = client.meta.copy()
        oracle_stderr = client.stderr_tail()[-1000:]
    finally:
        if client is not None:
            client.close()
        monitor_stop.set()
        memory_monitor.join()

    if not memory_samples:
        raise RuntimeError("CUDA memory monitor produced no samples")
    training = solver.pop("trainingData")
    validation = solver.pop("validationData")
    reconstruction = solver.pop("reconstruction")
    component_strength = solver.pop("componentStrengthRms")
    directional_derivative_data = solver.pop("directionalDerivativeData")
    artifact_arrays = {
        "reconstruction.npz": reconstruction,
        "distillation-training-observations.npz": training,
        "distillation-validation-probes.npz": validation,
        "component-strength.npz": {"rms": component_strength},
    }
    if directional_derivative_data is not None:
        artifact_arrays["distillation-directional-derivatives.npz"] = (
            directional_derivative_data
        )
    artifact_records: dict[str, dict[str, Any]] = {}
    for filename, arrays in artifact_arrays.items():
        path = output_dir / filename
        np.savez_compressed(path, **arrays)
        artifact_records[filename] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "arrays": {name: list(value.shape) for name, value in arrays.items()},
        }
    solver["oracleMetadataKeys"] = sorted(oracle_meta)
    solver["opaqueOracleStderrTail"] = oracle_stderr
    solver["sourceCaptureReportSha256"] = hashlib.sha256(
        (Path(capture_dir).resolve() / "capture-report.json").read_bytes()
    ).hexdigest()
    solver["artifacts"] = artifact_records
    solver["teacherArchitectureKnownToSolver"] = {
        "inputDim": input_dim,
        "outputDim": output_dim,
        "activationFamily": "silu-swiglu",
        "hiddenComponents": "not provided",
    }
    solver["targetModel"] = model
    solver["targetModule"] = f"model.layers.{layer}.mlp"
    solver["captureNaturalRecoveryProvenancePreserved"] = True
    solver["performance"]["peakWholeDeviceUsedBytesDuringOracleAndTraining"] = max(
        used for used, _ in memory_samples
    )
    solver["performance"]["minimumWholeDeviceFreeBytesDuringOracleAndTraining"] = min(
        total - used for used, total in memory_samples
    )
    solver["performance"]["wholeDeviceMemoryMonitorSamples"] = len(memory_samples)
    solver["performance"]["sampledHostRssPeakBytes"] = (
        max(host_rss_samples) if host_rss_samples else None
    )
    solver["performance"]["hostRssMonitorSamples"] = len(host_rss_samples)
    vm_hwm = _proc_vm_hwm_bytes()
    solver["performance"]["procVmHwmBytes"] = vm_hwm
    solver["performance"]["solverProcessPeakRssBytes"] = vm_hwm
    solver["performance"]["rssMeasurementSource"] = "/proc/self/status:VmHWM"
    solver["performance"]["oracleProcessPeakRssBytes"] = (
        max(oracle_hwm_samples) if oracle_hwm_samples else None
    )
    solver["performance"]["sampledOracleProcessRssPeakBytes"] = (
        max(oracle_rss_samples) if oracle_rss_samples else None
    )
    solver["performance"]["sampledParentAndOraclePssPeakBytes"] = (
        max(process_tree_pss_samples) if process_tree_pss_samples else None
    )
    solver["performance"]["processTreePssMonitorSamples"] = len(process_tree_pss_samples)
    solver["performance"]["processTreeMemoryMeasurementSource"] = (
        "/proc/<pid>/smaps_rollup:Pss plus /proc/<pid>/status:VmRSS,VmHWM"
    )
    (output_dir / "distillation-report.json").write_text(
        json.dumps(solver, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return solver


def evaluate_qwen_oracle_distillation(
    capture_dir: Path,
    distillation_dir: Path,
    model: str,
    layer: int,
    device: torch.device,
    local_files_only: bool,
) -> dict[str, Any]:
    """Post-hoc white-box functional and structural evaluation after solver artifacts freeze."""
    target_replay_absolute_tolerance = 2e-5
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Qwen student evaluation requires an available CUDA device")
    started = time.perf_counter()
    capture_report, datasets = validate_qwen_real_activation_artifacts(capture_dir)
    distillation_dir = distillation_dir.resolve()
    report_path = distillation_dir / "distillation-report.json"
    reconstruction_path = distillation_dir / "reconstruction.npz"
    if not report_path.is_file() or not reconstruction_path.is_file():
        raise FileNotFoundError("distillation report or reconstruction artifact is missing")
    evaluation_path = distillation_dir / "evaluation-report.json"
    if evaluation_path.exists():
        raise FileExistsError(f"refusing to overwrite post-hoc evaluation: {evaluation_path}")
    solver_report = json.loads(report_path.read_text(encoding="utf-8"))
    if solver_report.get("solverAccess") != "oracle-only" or solver_report.get(
        "hiddenUnitCountProvidedToSolver"
    ) is not False:
        raise ValueError("solver report violates the oracle-only/no-hidden-count contract")
    with np.load(reconstruction_path, allow_pickle=False) as archive:
        student = {key: archive[key] for key in archive.files}
    gate_student = student["gate"].astype(np.float32, copy=False)
    up_student = student["up"].astype(np.float32, copy=False)
    down_student = student["down"].astype(np.float32, copy=False)
    candidate_count = int(gate_student.shape[0])
    input_dim = int(capture_report["architecture"]["inputDim"])
    output_dim = int(capture_report["architecture"]["outputDim"])
    if (
        gate_student.shape != (candidate_count, input_dim)
        or up_student.shape != gate_student.shape
        or down_student.shape != (output_dim, candidate_count)
        or candidate_count != solver_report.get("selectedCandidateComponents")
    ):
        raise ValueError("student reconstruction dimensions disagree with its report")

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from tools.llm_reader.adapters.registry import create_adapter

    adapter = create_adapter("qwen2-swiglu")
    tokenizer, model_instance = adapter.load_model_and_tokenizer(
        model, local_files_only=local_files_only, device=str(device)
    )
    del tokenizer
    mlp = adapter.mlp_modules(model_instance, [layer])[layer]._mlp
    if (
        int(mlp.gate_proj.in_features) != input_dim
        or int(mlp.down_proj.out_features) != output_dim
    ):
        raise ValueError("post-hoc evaluator loaded a target with incompatible dimensions")

    student_module = _SwiGLUDistillationStudent(
        input_dim,
        output_dim,
        candidate_count,
        torch.Generator(device="cpu").manual_seed(0),
        device,
        gate=torch.from_numpy(gate_student),
        up=torch.from_numpy(up_student),
        down=torch.from_numpy(down_student),
    ).eval()

    def run_student(inputs: np.ndarray) -> np.ndarray:
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(inputs), 256):
                block = torch.as_tensor(
                    inputs[start:start + 256], dtype=torch.float32, device=device
                )
                chunks.append(student_module(block).float().cpu().numpy())
        return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, output_dim), np.float32)

    def run_target(inputs: np.ndarray) -> np.ndarray:
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(inputs), 256):
                block = torch.as_tensor(
                    inputs[start:start + 256], dtype=torch.float32, device=device
                )
                chunks.append(mlp(block).float().cpu().numpy())
        return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, output_dim), np.float32)

    functional_results: dict[str, Any] = {}
    for split_name in (
        "recovery_observations",
        "functional_evaluation",
        "structural_evaluation",
        "distributional_analysis",
    ):
        arrays = datasets[split_name]
        prediction = run_student(arrays["inputs"])
        independent_target = run_target(arrays["inputs"])
        target_delta = float(np.max(np.abs(independent_target - arrays["outputs"])))
        if target_delta > target_replay_absolute_tolerance:
            raise RuntimeError(
                f"post-hoc target replay disagrees with capture in {split_name}: {target_delta}"
            )
        functional_results[split_name] = _functional_metrics(
            independent_target, prediction, arrays["inputs"]
        )
        functional_results[split_name]["independentTargetReplayMaxAbsDelta"] = target_delta

    validation_path = distillation_dir / "distillation-validation-probes.npz"
    with np.load(validation_path, allow_pickle=False) as archive:
        validation = {key: archive[key] for key in archive.files}
    validation_target = run_target(validation["inputs"])
    validation_prediction = run_student(validation["inputs"])
    validation_target_delta = float(np.max(np.abs(validation_target - validation["outputs"])))
    if validation_target_delta > target_replay_absolute_tolerance:
        raise RuntimeError(
            f"post-hoc target replay disagrees with solver validation labels: {validation_target_delta}"
        )
    generated_validation_metrics = _functional_metrics(
        validation_target, validation_prediction, validation["inputs"]
    )
    generated_validation_metrics["independentTargetReplayMaxAbsDelta"] = validation_target_delta

    gate_truth = mlp.gate_proj.weight.detach().float()
    up_truth = mlp.up_proj.weight.detach().float()
    down_truth = mlp.down_proj.weight.detach().float()
    gate_candidate = torch.as_tensor(gate_student, device=device)
    gate_truth_norm = torch.nn.functional.normalize(gate_truth, dim=1)
    gate_candidate_norm = torch.nn.functional.normalize(gate_candidate, dim=1)
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as error:
        raise RuntimeError("post-hoc structural matching requires the installed scipy package") from error

    if min(candidate_count, int(gate_truth.shape[0])) <= 512:
        similarity = (gate_candidate_norm @ gate_truth_norm.T).abs().cpu().numpy()
        candidate_indices, truth_indices = linear_sum_assignment(-similarity)
        matching_method = "exact_hungarian_max_abs_gate_cosine"
        matched_cosines = similarity[candidate_indices, truth_indices]
    else:
        top_candidates: list[tuple[float, int, int, float]] = []
        top_k = min(16, int(gate_truth.shape[0]))
        for start in range(0, candidate_count, 32):
            scores = gate_candidate_norm[start:start + 32] @ gate_truth_norm.T
            values, indices = scores.abs().topk(top_k, dim=1)
            signed_values = scores.gather(1, indices)
            for local_index in range(values.shape[0]):
                for rank_index in range(top_k):
                    top_candidates.append((
                        float(values[local_index, rank_index].item()),
                        start + local_index,
                        int(indices[local_index, rank_index].item()),
                        float(signed_values[local_index, rank_index].item()),
                    ))
        used_candidate: set[int] = set()
        used_truth: set[int] = set()
        accepted: list[tuple[int, int, float]] = []
        for cosine, candidate_index, truth_index, signed in sorted(top_candidates, reverse=True):
            if candidate_index in used_candidate or truth_index in used_truth:
                continue
            used_candidate.add(candidate_index)
            used_truth.add(truth_index)
            accepted.append((candidate_index, truth_index, signed))
        candidate_indices = np.asarray([item[0] for item in accepted], dtype=np.int64)
        truth_indices = np.asarray([item[1] for item in accepted], dtype=np.int64)
        matched_cosines = np.asarray([abs(item[2]) for item in accepted], dtype=np.float64)
        matching_method = "greedy_one_to_one_from_top16_gate_cosines"

    candidate_index_t = torch.as_tensor(candidate_indices, device=device, dtype=torch.long)
    truth_index_t = torch.as_tensor(truth_indices, device=device, dtype=torch.long)
    signed_gate_cosines = (
        gate_candidate_norm[candidate_index_t] * gate_truth_norm[truth_index_t]
    ).sum(dim=1).cpu().numpy()
    up_candidate = torch.as_tensor(up_student, device=device)
    down_candidate = torch.as_tensor(down_student, device=device)
    matched_up_cosines: list[float] = []
    matched_down_cosines: list[float] = []
    matched_up_scaled_errors: list[float] = []
    matched_down_scaled_errors: list[float | None] = []
    matched_gate_norm_relative_errors: list[float] = []
    for candidate_index, truth_index in zip(candidate_indices, truth_indices):
        student_up = up_candidate[candidate_index]
        true_up = up_truth[truth_index]
        student_down = down_candidate[:, candidate_index]
        true_down = down_truth[:, truth_index]
        student_up_norm = student_up.norm().clamp_min(1e-30)
        true_up_norm = true_up.norm().clamp_min(1e-30)
        student_down_norm = student_down.norm().clamp_min(1e-30)
        true_down_norm = true_down.norm().clamp_min(1e-30)
        matched_up_cosines.append(float(torch.dot(student_up, true_up).abs() / (student_up_norm * true_up_norm)))
        matched_down_cosines.append(float(torch.dot(student_down, true_down).abs() / (student_down_norm * true_down_norm)))
        # U -> cU and D -> D/c is a genuine reciprocal scale symmetry.
        scale = torch.dot(student_up, true_up) / student_up.square().sum().clamp_min(1e-30)
        matched_up_scaled_errors.append(
            float((scale * student_up - true_up).norm() / true_up_norm)
        )
        matched_down_scaled_errors.append(
            float((student_down / scale - true_down).norm() / true_down_norm)
            if abs(float(scale.item())) > 1e-20 else None
        )
        candidate_gate_norm = gate_candidate[candidate_index].norm().clamp_min(1e-30)
        truth_gate_norm = gate_truth[truth_index].norm().clamp_min(1e-30)
        matched_gate_norm_relative_errors.append(
            float((candidate_gate_norm - truth_gate_norm).abs() / truth_gate_norm)
        )
    structural = {
        "teacherComponentsObservedPosthoc": int(gate_truth.shape[0]),
        "studentCandidateComponents": candidate_count,
        "oneToOneMatches": int(len(candidate_indices)),
        "matchingMethod": matching_method,
        "gateAbsCosineMedian": float(np.median(matched_cosines)) if len(matched_cosines) else None,
        "gateSignedCosineMedian": float(np.median(signed_gate_cosines)) if len(signed_gate_cosines) else None,
        "gateCorrectOrientationCountAtCosine0.99": int(
            np.sum(signed_gate_cosines >= 0.99)
        ),
        "gateMatchedCountAtAbsCosine0.99": int(np.sum(np.asarray(matched_cosines) >= 0.99)),
        "gateNormRelativeErrorMedian": float(np.median(matched_gate_norm_relative_errors)) if matched_gate_norm_relative_errors else None,
        "upAbsCosineMedian": float(np.median(matched_up_cosines)) if matched_up_cosines else None,
        "downAbsCosineMedian": float(np.median(matched_down_cosines)) if matched_down_cosines else None,
        "upScaleAlignedRelativeErrorMedian": float(np.median(matched_up_scaled_errors)) if matched_up_scaled_errors else None,
        "downReciprocalScaleAlignedRelativeErrorMedian": (
            float(np.median([value for value in matched_down_scaled_errors if value is not None]))
            if any(value is not None for value in matched_down_scaled_errors) else None
        ),
        "downReciprocalScaleAlignmentUndefinedCount": sum(
            value is None for value in matched_down_scaled_errors
        ),
        "legitimateSymmetries": [
            "permutation of hidden component rows/columns",
            "nonzero reciprocal scaling of up-projection row and matching down-projection column",
        ],
        "gateScaleSymmetry": False,
        "completeStructuralRecovery": bool(
            candidate_count == int(gate_truth.shape[0])
            and len(candidate_indices) == int(gate_truth.shape[0])
            and np.sum(signed_gate_cosines >= 0.99) == int(gate_truth.shape[0])
        ),
    }
    result = {
        "schemaVersion": 1,
        "kind": "gcl-qwen-swiglu-distillation-evaluation",
        "status": "evaluated_posthoc",
        "verificationMode": "independent_posthoc_white_box_evaluator",
        "model": model,
        "module": f"model.layers.{layer}.mlp",
        "groundTruthWeightsReadByEvaluator": True,
        "groundTruthWeightsUsedBySolver": False,
        "targetReplayAbsoluteTolerance": target_replay_absolute_tolerance,
        "solverReportSha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "functional": functional_results,
        "generatedOracleValidation": generated_validation_metrics,
        "structural": structural,
        "identifiability": {
            "finiteOracleObservations": int(
                solver_report["queryAccounting"]["oraclePoints"]
                + solver_report["queryAccounting"]["reusedRecoveryObservations"]
            ),
            "teacherComponentsWereHiddenFromSolver": True,
            "teacherParametersWereHiddenFromSolver": True,
            "functionalFitDoesNotProveStructuralIdentifiability": True,
            "countEstimateIsStudentModelSelectionNotTeacherCountProof": True,
            "identifiedByModelFamily": [
                "input dimension",
                "output dimension",
                "SwiGLU activation family",
                "bias-free Qwen2 feed-forward form",
            ],
            "unresolved": [
                "global uniqueness of the 4864-component finite-sample factorization",
                "whether omitted or weak student components are functionally redundant on the sampled domain",
            ],
        },
        "evaluationSeconds": time.perf_counter() - started,
    }
    evaluation_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    del student_module, mlp, model_instance, adapter
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _nvidia_smi_memory_bytes(device_index: int) -> int:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            f"--id={device_index}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(
            f"expected one nvidia-smi memory row for GPU {device_index}, got {rows!r}"
        )
    try:
        return int(rows[0]) * 1024**2
    except ValueError as error:
        raise RuntimeError(f"invalid nvidia-smi memory value: {rows[0]!r}") from error


def _component_signature_summary(gram: torch.Tensor) -> dict[str, Any]:
    norms = gram.diagonal().clamp_min(0.0).sqrt()
    active = norms > 1e-20
    active_count = int(active.sum().item())
    gram.div_(norms[:, None].clamp_min(1e-20))
    gram.div_(norms[None, :].clamp_min(1e-20))
    gram.abs_()
    gram.fill_diagonal_(0.0)
    nearest = gram.max(dim=1).values[active]
    if active_count:
        nearest_cpu = nearest.float().cpu().numpy()
        return {
            "activeComponents": active_count,
            "inactiveComponents": int(len(norms) - active_count),
            "nearestOtherComponentAbsoluteCosineMedian": float(
                np.median(nearest_cpu)
            ),
            "nearestOtherComponentAbsoluteCosineP95": float(
                np.quantile(nearest_cpu, 0.95)
            ),
            "componentsWithNearestCosineAtLeast0.99": int(
                np.sum(nearest_cpu >= 0.99)
            ),
        }
    return {
        "activeComponents": 0,
        "inactiveComponents": int(len(norms)),
        "nearestOtherComponentAbsoluteCosineMedian": None,
        "nearestOtherComponentAbsoluteCosineP95": None,
        "componentsWithNearestCosineAtLeast0.99": 0,
    }


def run_qwen_amplitude_response(
    capture_dir: Path,
    output_dir: Path,
    model: str,
    layer: int,
    device: torch.device,
    local_files_only: bool,
    seed: int,
    max_oracle_points: int,
) -> dict[str, Any]:
    """Measure exact positive/negative amplitude trajectories using only the opaque oracle."""
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Qwen amplitude probing requires an available CUDA device")
    if max_oracle_points <= 0:
        raise ValueError("max_oracle_points must be positive")
    capture_report, datasets = validate_qwen_real_activation_artifacts(capture_dir)
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing amplitude artifacts: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from tools.llm_reader.swiglu_amplitude_response import (
        DEFAULT_AMPLITUDES,
        amplitude_inputs,
        analyze_amplitude_outputs,
        make_amplitude_directions,
    )

    capture_digest = hashlib.sha256(
        (Path(capture_dir).resolve() / "capture-report.json").read_bytes()
    ).hexdigest()
    natural = datasets["recovery_observations"]
    directions = make_amplitude_directions(
        natural["inputs"],
        prompt_ids=natural["prompt_ids"],
        prompt_sha256=natural["prompt_sha256"],
        token_positions=natural["token_positions"],
        natural_count=32,
        combination_count=32,
        independent_count=32,
        seed=seed,
    )
    amplitudes = DEFAULT_AMPLITUDES.copy()
    exact_inputs = amplitude_inputs(directions["directions"], amplitudes)
    direction_count, amplitude_count, polarity_count, input_dim = exact_inputs.shape
    output_dim = int(capture_report["architecture"]["outputDim"])
    requested_points = int(direction_count * amplitude_count * polarity_count)
    if requested_points > max_oracle_points:
        raise ValueError(
            f"amplitude design needs {requested_points} oracle points, "
            f"budget is {max_oracle_points}"
        )
    if input_dim != int(capture_report["architecture"]["inputDim"]):
        raise ValueError("amplitude direction width disagrees with the verified capture")

    device_index = int(device.index if device.index is not None else torch.cuda.current_device())
    stop_sampling = threading.Event()
    memory_samples: list[dict[str, int | None]] = []
    sampling_errors: list[Exception] = []
    oracle_pid: int | None = None

    def sample_resources() -> None:
        try:
            while not stop_sampling.is_set():
                gpu_used = _nvidia_smi_memory_bytes(device_index)
                memory_samples.append({
                    "gpuUsedBytes": gpu_used,
                    "parentPssBytes": _proc_smaps_pss_bytes("self"),
                    "oraclePssBytes": (
                        _proc_smaps_pss_bytes(oracle_pid)
                        if oracle_pid is not None
                        else None
                    ),
                })
                stop_sampling.wait(0.1)
        except Exception as error:
            sampling_errors.append(error)
            stop_sampling.set()

    started = time.perf_counter()
    memory_thread = threading.Thread(
        target=sample_resources,
        name="qwen-amplitude-resource-sampler",
        daemon=True,
    )
    memory_thread.start()
    client = None
    oracle_outputs_flat: list[np.ndarray] = []
    oracle_requests = 0
    try:
        client = open_live_oracle(
            model=model,
            layer=layer,
            units="",
            device=str(device),
            max_calls=max_oracle_points,
            work_dir=output_dir / "oracle-work",
            local_files_only=local_files_only,
            oracle_output_mode="full-mlp-vector",
            opaque_eval_only=True,
        )
        if client.meta.get("hiddenSize") != input_dim or client.meta.get(
            "oracleResponseWidth"
        ) != output_dim:
            raise RuntimeError("opaque amplitude oracle dimensions disagree with the capture")
        forbidden = {
            "model", "modelType", "layer", "unit", "units", "unitCount",
            "intermediateSize", "device", "blockDevice", "outputChannel",
        }
        exposed = sorted(forbidden.intersection(client.meta))
        if exposed:
            raise RuntimeError(f"opaque amplitude oracle exposed {exposed}")
        if client.meta.get("protocolCapabilities", {}).get("truth") is not False:
            raise RuntimeError("amplitude solver oracle unexpectedly exposes truth capability")
        if client.process is not None:
            oracle_pid = int(client.process.pid)

        flat_inputs = exact_inputs.reshape(-1, input_dim)
        for start_index in range(0, len(flat_inputs), 256):
            batch = flat_inputs[start_index:start_index + 256]
            oracle_outputs_flat.append(
                client.eval(batch, stage="amplitude-polarity-trajectories").astype(
                    np.float32, copy=False
                )
            )
            oracle_requests += 1
        oracle_points = int(client.query_count)
        oracle_stderr = client.stderr_tail()
        if oracle_points != requested_points:
            raise RuntimeError(
                f"opaque oracle used {oracle_points} points, expected {requested_points}"
            )
        outputs = np.concatenate(oracle_outputs_flat, axis=0).reshape(
            direction_count, amplitude_count, polarity_count, output_dim
        )
        analysis = analyze_amplitude_outputs(outputs, amplitudes)
        raw_path = output_dir / "amplitude-responses.npz"
        np.savez_compressed(
            raw_path,
            directions=directions["directions"],
            direction_ids=directions["direction_ids"],
            direction_kinds=directions["direction_kinds"],
            source_natural_indices=directions["source_natural_indices"],
            source_coefficients=directions["source_coefficients"],
            source_prompt_ids=directions["source_prompt_ids"],
            source_prompt_sha256=directions["source_prompt_sha256"],
            source_token_positions=directions["source_token_positions"],
            amplitudes=amplitudes,
            polarities=np.asarray([1.0, -1.0], dtype=np.float32),
            exact_inputs=exact_inputs,
            oracle_outputs=outputs,
            normalized_responses=analysis["normalized_responses"],
            consecutive_relative_changes=analysis["consecutive_relative_changes"],
            consecutive_cosines=analysis["consecutive_cosines"],
            consecutive_cosine_defined=analysis["consecutive_cosine_defined"],
            log2_amplitude_slopes=analysis["log2_amplitude_slopes"],
            even_responses=analysis["even_responses"],
            odd_responses=analysis["odd_responses"],
            stable_at_final_two_steps=analysis["stable_at_final_two_steps"],
        )
        raw_sha256 = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        result = {
            "schemaVersion": 1,
            "kind": "gcl-qwen-swiglu-amplitude-polarity-probe",
            "status": "captured",
            "model": model,
            "module": f"model.layers.{layer}.mlp",
            "architecture": {
                "inputDim": input_dim,
                "outputDim": output_dim,
                "activation": "silu",
            },
            "solverAccess": "oracle-only",
            "teacherParametersUsedBySolver": False,
            "hiddenUnitCountProvidedToSolver": False,
            "accessMode": {
                "fullModuleOracle": "opaque_eval_only",
                "capturedNaturalActivationVectorsUsedOnlyAsDirections": True,
                "oracleTruthCapability": False,
            },
            "captureReportSha256": capture_digest,
            "directionDesign": {
                "count": direction_count,
                "naturalObserved": int(
                    np.sum(directions["direction_kinds"] == "natural_observed")
                ),
                "naturalCombinations": int(
                    np.sum(directions["direction_kinds"] == "natural_combination")
                ),
                "independentGaussian": int(
                    np.sum(directions["direction_kinds"] == "independent_gaussian")
                ),
                "seed": int(seed),
                "provenanceStoredPerDirection": True,
            },
            "amplitudes": amplitudes.tolist(),
            "polarityOrder": [1.0, -1.0],
            "queryAccounting": {
                "oraclePoints": oracle_points,
                "oracleQueryBudget": int(max_oracle_points),
                "requestedPoints": requested_points,
                "evalRequests": oracle_requests,
                "pointsPerDirection": int(amplitude_count * polarity_count),
                "rawArtifact": raw_path.name,
                "rawArtifactSha256": raw_sha256,
            },
            "responseAnalysis": {
                "stabilityRelativeChangeThreshold": 0.01,
                "stableBothPolaritiesAtFinalTwoSteps": int(
                    np.sum(analysis["stable_at_final_two_steps"])
                ),
                "stableFraction": float(
                    np.mean(analysis["stable_at_final_two_steps"])
                ),
                "medianFinalTwoStepRelativeChangeByPolarity": np.median(
                    analysis["consecutive_relative_changes"][:, -2:, :], axis=(0, 1)
                ).tolist(),
                "p95FinalTwoStepRelativeChangeByPolarity": np.quantile(
                    analysis["consecutive_relative_changes"][:, -2:, :],
                    0.95,
                    axis=(0, 1),
                ).tolist(),
                "maximumFinalTwoStepRelativeChangeByPolarity": np.max(
                    analysis["consecutive_relative_changes"][:, -2:, :], axis=(0, 1)
                ).tolist(),
            },
            "performance": {
                "runtimeSeconds": time.perf_counter() - started,
                "parentVmHwmBytes": _proc_vm_hwm_bytes(),
                "oracleVmHwmBytes": (
                    _proc_status_memory_bytes(oracle_pid, "VmHWM")
                    if oracle_pid is not None
                    else None
                ),
                "sampledPeakCombinedPssBytes": max(
                    (
                        int(sample["parentPssBytes"] or 0)
                        + int(sample["oraclePssBytes"] or 0)
                        for sample in memory_samples
                    ),
                    default=None,
                ),
                "sampledPeakDeviceWideVramBytes": max(
                    (int(sample["gpuUsedBytes"]) for sample in memory_samples),
                    default=None,
                ),
                "resourceSampleCount": len(memory_samples),
                "resourceSampleIntervalSeconds": 0.1,
                "vramScope": (
                    "nvidia-smi device-wide usage; includes concurrent GPU processes"
                ),
                "parentPssMeasurementSource": "/proc/self/smaps_rollup:Pss",
                "oraclePssMeasurementSource": "/proc/<pid>/smaps_rollup:Pss",
                "processHighWaterMarkSource": "/proc/<pid>/status:VmHWM",
                "oracleStderrTail": oracle_stderr[-1000:],
            },
            "interpretation": {
                "functionalRecovery": "not_measured_no_reconstruction",
                "structuralRecovery": "not_measured_no_reconstruction",
                "distributionalGeneralization": "not_measured_no_reconstruction",
            },
        }
    finally:
        if client is not None:
            client.close()
        stop_sampling.set()
        memory_thread.join()
    if sampling_errors:
        raise RuntimeError("Qwen amplitude resource sampling failed") from sampling_errors[0]
    report_path = output_dir / "probe-report.json"
    report_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def evaluate_qwen_amplitude_response(
    capture_dir: Path,
    output_dir: Path,
    model: str,
    layer: int,
    device: torch.device,
    local_files_only: bool,
) -> dict[str, Any]:
    """Use a separate white-box reload only to evaluate frozen oracle measurements."""
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Qwen amplitude evaluation requires an available CUDA device")
    started = time.perf_counter()
    capture_report, _ = validate_qwen_real_activation_artifacts(capture_dir)
    output_dir = output_dir.resolve()
    report_path = output_dir / "probe-report.json"
    raw_path = output_dir / "amplitude-responses.npz"
    evaluation_path = output_dir / "evaluation-report.json"
    if not report_path.is_file() or not raw_path.is_file():
        raise FileNotFoundError("Qwen amplitude probe report or raw responses are missing")
    if evaluation_path.exists():
        raise FileExistsError(f"refusing to overwrite amplitude evaluation: {evaluation_path}")
    solver_report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        solver_report.get("status") != "captured"
        or solver_report.get("solverAccess") != "oracle-only"
        or solver_report.get("teacherParametersUsedBySolver") is not False
        or solver_report.get("hiddenUnitCountProvidedToSolver") is not False
    ):
        raise ValueError("amplitude probe report violates its oracle-only contract")
    if hashlib.sha256(raw_path.read_bytes()).hexdigest() != solver_report[
        "queryAccounting"
    ].get("rawArtifactSha256"):
        raise ValueError("raw amplitude response SHA-256 mismatch")
    if hashlib.sha256(
        (Path(capture_dir).resolve() / "capture-report.json").read_bytes()
    ).hexdigest() != solver_report.get("captureReportSha256"):
        raise ValueError("activation-capture report changed after the amplitude probe")
    with np.load(raw_path, allow_pickle=False) as archive:
        raw = {key: archive[key] for key in archive.files}
    directions = raw["directions"].astype(np.float32, copy=False)
    amplitudes = raw["amplitudes"].astype(np.float32, copy=False)
    exact_inputs = raw["exact_inputs"].astype(np.float32, copy=False)
    oracle_outputs = raw["oracle_outputs"].astype(np.float32, copy=False)
    if exact_inputs.ndim != 4 or exact_inputs.shape[:3] != oracle_outputs.shape[:3]:
        raise ValueError("raw amplitude inputs and outputs have inconsistent dimensions")
    expected_points = int(np.prod(exact_inputs.shape[:3]))
    if expected_points != solver_report["queryAccounting"].get("oraclePoints"):
        raise ValueError("raw amplitude query count disagrees with the solver report")
    if not np.isfinite(exact_inputs).all() or not np.isfinite(oracle_outputs).all():
        raise ValueError("raw amplitude observations contain non-finite values")

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from tools.llm_reader.adapters.registry import create_adapter
    from tools.llm_reader.swiglu_amplitude_response import (
        analyze_amplitude_outputs,
        asymptotic_swiglu_responses,
    )

    adapter = create_adapter("qwen2-swiglu")
    tokenizer, model_instance = adapter.load_model_and_tokenizer(
        model, local_files_only=local_files_only, device=str(device)
    )
    del tokenizer
    mlp = adapter.mlp_modules(model_instance, [layer])[layer]._mlp
    input_dim = int(mlp.gate_proj.in_features)
    output_dim = int(mlp.down_proj.out_features)
    if exact_inputs.shape[-1] != input_dim or oracle_outputs.shape[-1] != output_dim:
        raise ValueError("amplitude artifacts do not match the independently loaded Qwen module")
    if any(
        getattr(projection, "bias", None) is not None
        for projection in (mlp.gate_proj, mlp.up_proj, mlp.down_proj)
    ):
        raise ValueError("asymptotic evaluator expects the bias-free Qwen SwiGLU module")
    torch.cuda.reset_peak_memory_stats(device)

    def replay(inputs: np.ndarray) -> np.ndarray:
        rows = inputs.reshape(-1, input_dim)
        chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for start_index in range(0, len(rows), 128):
                block = torch.as_tensor(
                    rows[start_index:start_index + 128],
                    dtype=torch.float32,
                    device=device,
                )
                chunks.append(mlp(block).float().cpu().numpy())
        return np.concatenate(chunks, axis=0).reshape(
            *inputs.shape[:-1], output_dim
        )

    target_outputs = replay(exact_inputs)
    max_abs_delta = float(np.max(np.abs(target_outputs - oracle_outputs)))
    replay_delta = (
        target_outputs.astype(np.float64) - oracle_outputs.astype(np.float64)
    ).reshape(-1, output_dim)
    oracle_rows = oracle_outputs.astype(np.float64).reshape(-1, output_dim)
    replay_relative_by_row = np.linalg.norm(replay_delta, axis=1) / np.maximum(
        np.linalg.norm(oracle_rows, axis=1), 1e-30
    )
    max_relative_l2_delta = float(np.max(replay_relative_by_row))
    target_replay_relative_tolerance = 5e-6
    if max_relative_l2_delta > target_replay_relative_tolerance:
        raise RuntimeError(
            "post-hoc module replay relative L2 error "
            f"{max_relative_l2_delta} exceeds {target_replay_relative_tolerance}; "
            f"max absolute delta was {max_abs_delta}"
        )
    flat_inputs = exact_inputs.reshape(-1, input_dim)
    replay_metrics = _functional_metrics(
        target_outputs.reshape(-1, output_dim),
        oracle_outputs.reshape(-1, output_dim),
        flat_inputs,
    )

    gate = mlp.gate_proj.weight.detach().float().cpu().numpy()
    up = mlp.up_proj.weight.detach().float().cpu().numpy()
    down = mlp.down_proj.weight.detach().float().cpu().numpy()
    positive_limit, negative_limit = asymptotic_swiglu_responses(
        gate, up, down, directions
    )
    reversed_positive, reversed_negative = asymptotic_swiglu_responses(
        gate, up, down, -directions
    )
    polarity_identity_error = max(
        float(np.max(np.abs(positive_limit - reversed_negative))),
        float(np.max(np.abs(negative_limit - reversed_positive))),
    )
    if polarity_identity_error > 1e-10:
        raise RuntimeError("exact asymptotic polarity-reversal identity failed")
    analysis = analyze_amplitude_outputs(oracle_outputs, amplitudes)
    limits = np.stack([positive_limit, negative_limit], axis=1)
    final_normalized = analysis["normalized_responses"][:, -1]
    limit_norms = np.linalg.norm(limits, axis=-1)
    limit_errors = np.linalg.norm(final_normalized - limits, axis=-1) / np.maximum(
        limit_norms, 1e-30
    )

    gate_t = mlp.gate_proj.weight.detach().float()
    up_t = mlp.up_proj.weight.detach().float()
    down_t = mlp.down_proj.weight.detach().float()
    down_gram = down_t.T @ down_t
    signature_results: dict[str, Any] = {}
    direction_kinds = raw["direction_kinds"].astype(str)
    for kind in (
        "natural_observed",
        "natural_combination",
        "independent_gaussian",
    ):
        selected = np.flatnonzero(direction_kinds == kind)
        direction_tensor = torch.as_tensor(
            directions[selected], dtype=torch.float32, device=device
        )
        gate_values = direction_tensor @ gate_t.T
        up_values = direction_tensor @ up_t.T
        signatures: dict[str, Any] = {}
        for polarity_name, coefficients in (
            ("positive", gate_values.clamp_min(0.0) * up_values),
            ("negative", gate_values.clamp_max(0.0) * up_values),
        ):
            coefficient_rows = coefficients.T.contiguous()
            activation_gram = coefficient_rows @ coefficient_rows.T
            projected_gram = activation_gram * down_gram
            signatures[polarity_name] = {
                "beforeOutputProjection": _component_signature_summary(
                    activation_gram
                ),
                "afterOutputProjection": _component_signature_summary(
                    projected_gram
                ),
            }
        signature_results[kind] = signatures
        del direction_tensor, gate_values, up_values, signatures
    del down_gram, gate_t, up_t

    stable = analysis["stable_at_final_two_steps"]
    stability_by_kind: dict[str, Any] = {}
    asymptotic_error_by_kind: dict[str, Any] = {}
    for kind in (
        "natural_observed",
        "natural_combination",
        "independent_gaussian",
    ):
        selected = direction_kinds == kind
        stability_by_kind[kind] = {
            "directions": int(np.sum(selected)),
            "stableBothPolaritiesAtFinalTwoSteps": int(np.sum(stable[selected])),
            "stableFraction": float(np.mean(stable[selected])),
        }
        kind_errors = limit_errors[selected]
        asymptotic_error_by_kind[kind] = {
            "positiveMedianRelativeError": float(np.median(kind_errors[:, 0])),
            "positiveP95RelativeError": float(np.quantile(kind_errors[:, 0], 0.95)),
            "negativeMedianRelativeError": float(np.median(kind_errors[:, 1])),
            "negativeP95RelativeError": float(np.quantile(kind_errors[:, 1], 0.95)),
            "nearZeroAsymptoticComponents": int(np.sum(limit_norms[selected] <= 1e-20)),
        }

    result = {
        "schemaVersion": 1,
        "kind": "gcl-qwen-swiglu-amplitude-polarity-evaluation",
        "status": "evaluated_posthoc",
        "verificationMode": "independent_posthoc_white_box_evaluator",
        "model": model,
        "module": f"model.layers.{layer}.mlp",
        "groundTruthWeightsReadByEvaluator": True,
        "groundTruthWeightsUsedBySolver": False,
        "solverReportSha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "rawArtifactSha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "captureReportSha256": solver_report["captureReportSha256"],
        "independentOracleReplay": {
            "maxAbsoluteDelta": max_abs_delta,
            "maxPerSampleRelativeL2Delta": max_relative_l2_delta,
            "relativeL2Tolerance": target_replay_relative_tolerance,
            "absoluteDeltaIsInformationalAtLargeAmplitudes": True,
            "metrics": replay_metrics,
        },
        "responseStability": {
            "criterion": (
                "relative L2 change of f(±t u)/t² at each of the final two "
                "amplitude transitions is <= 1% for both polarities"
            ),
            "byDirectionKind": stability_by_kind,
            "overallStableDirectionFraction": float(np.mean(stable)),
            "byPolarityFinalTwoStepP95RelativeChange": np.quantile(
                analysis["consecutive_relative_changes"][:, -2:, :],
                0.95,
                axis=(0, 1),
            ).tolist(),
        },
        "asymptoticResponse": {
            "formula": (
                "P(u)=D sum_i max(g_i·u,0)(v_i·u); "
                "N(u)=D sum_i min(g_i·u,0)(v_i·u)"
            ),
            "exactPolarityReversalIdentityMaxAbsoluteError": polarity_identity_error,
            "positiveNegativeLimitsAreDirectionReversals": True,
            "finalAmplitudeRelativeErrorByDirectionKind": asymptotic_error_by_kind,
        },
        "posthocComponentSignatureAnalysis": {
            "teacherComponentCountReadOnlyAfterProbe": int(gate.shape[0]),
            "teacherWeightsUsedOnlyByEvaluator": True,
            "signatureDefinition": (
                "per-component asymptotic response across directions, measured "
                "before and after its down-projection column"
            ),
            "cosinesAreAbsoluteAndCompareNearestOtherComponent": True,
            "byDirectionKindAndPolarity": signature_results,
            "isSolverRecoveryEvidence": False,
        },
        "conclusion": {
            "structuralRecoveryAlgorithmImplemented": False,
            "recoveryJustifiedByMeasurements": False,
            "reason": (
                "this evaluator reports measured convergence and post-hoc "
                "component signatures; it does not invert their summed oracle "
                "responses into an oracle-only structure"
            ),
        },
        "performance": {
            "evaluationSeconds": time.perf_counter() - started,
            "peakHostVmHwmBytes": _proc_vm_hwm_bytes(),
            "peakCudaAllocatorBytes": int(torch.cuda.max_memory_allocated(device)),
            "cudaDeviceWideUsedBytesAfterEvaluation": _nvidia_smi_memory_bytes(
                int(device.index if device.index is not None else torch.cuda.current_device())
            ),
        },
    }
    evaluation_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    del mlp, model_instance, adapter
    gc.collect()
    torch.cuda.empty_cache()
    return result


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

        # `anchors * readers` is a ceiling on the measured rank, and the span needs `2k`. Three
        # anchors against eight readers is 24, which is under the 32 a 16-unit block asks for, so
        # the first live run at that size returned a rank-24 span and zero directions - a
        # rank-deficient measurement that looked like the method failing. The count is derived
        # here for the same reason `stitch_recovery` derives it, and the ceiling is reported so a
        # span that ran out of slices says so instead of being read as a small block.
        # Anchors cost points and readers do not. A slice needs one of each, and the ceiling on
        # the measured rank is their product - but an anchor multiplies the whole polarisation
        # design, `d(d+1)/2 + d + 1` directions of it, while a reader is another column of an
        # answer already paid for. Measured on 16 live units at subspace 48, same ceiling 48
        # either way: 8 readers and 6 anchors give 16/16 at cosine 0.9999953 for 68592 calls, and
        # 24 readers with 2 anchors give 16/16 at 0.9999996 for 29392. Same recovery, better
        # agreement, 2.33x fewer calls.
        #
        # So anchors sit at the floor - one anchor cannot separate the span whatever it is given,
        # two is the least that can, which `tile_recovery` measured first - and the readers carry
        # the ceiling. The margin over `2k` is deliberate: a rank read at its own ceiling says
        # nothing about the layer, which is the trap the scale probe fell into.
        anchor_count = max(2, anchor_count) if anchor_count > 0 else 2
        reader_count = max(reader_count, -(-3 * rank_target // anchor_count))
        readers = torch.randn(reader_count, hidden, generator=generator, device=device, dtype=DTYPE)
        readers = readers / readers.norm(dim=1, keepdim=True)
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
            "oracleEvalSeconds": round(teacher.eval_seconds, 3),
            "oracleEvalCalls": teacher.eval_calls,
            "pointsPerSecond": round(solver_calls / max(teacher.eval_seconds, 1e-9), 1),
            "secondsPerEvalCall": round(teacher.eval_seconds / max(teacher.eval_calls, 1), 4),
            "truthUnlockQueryCount": client.truth_unlock_query_count,
            "unlockedAfterAllSolverCalls": client.truth_unlock_query_count == solver_calls,
            "device": str(span_basis.device),
        }
    finally:
        client.close()


def live_stitch(
    model: str,
    layer: int,
    units: str,
    subspace_dim: int,
    core_dim: int,
    base_scale: float,
    reader_count: int,
    rank_cap: int,
    core_points: int,
    pencil_draws: int,
    steps: list[float],
    seed: int,
    tolerance: float,
    span_tolerance: float,
    purity: float,
    collinearity: float,
    max_calls: int,
    work_dir: Path,
    local_files_only: bool,
    device: torch.device,
) -> dict:
    """The stitched recovery, with the real layer answering.

    Nothing about the assembly changes for a live block. The solve only ever calls
    `oracle(points)`, so the whole difference is who answers, and the truth arrives through a
    provider that the oracle keeps locked until the recovery is already frozen. What comes out is
    a full 3072-dimensional gate direction, not a subspace projection.
    """
    client = open_live_oracle(model, layer, units, str(device), max_calls, work_dir, local_files_only)
    try:
        hidden = int(client.meta["hiddenSize"])
        unit_count = len([piece for piece in units.split(",") if piece.strip()])
        # The readers are drawn here and handed to the bridge, so the answer that crosses the bus
        # is already `reader_count` wide. `stitch_recovery` then reads them as coordinates, which
        # is what identity readers mean, and the arithmetic downstream is untouched.
        reader_count = max(reader_count, -(-3 * rank_cap // 2))
        reader_generator = torch.Generator(device=device).manual_seed(seed + 31)
        projection = torch.randn(
            reader_count, hidden, generator=reader_generator, device=device, dtype=DTYPE
        )
        projection = projection / projection.norm(dim=1, keepdim=True)
        teacher = LiveTeacher(client, device, projection=projection)

        def truth_gate_provider() -> torch.Tensor:
            client.unlock_posthoc_truth("live-stitch")
            gate_truth, _ = client.truth()
            return torch.as_tensor(gate_truth, dtype=DTYPE, device=device)

        report = stitch_recovery(
            units=unit_count, input_dim=hidden, output_dim=reader_count,
            subspace_dim=subspace_dim, core_dim=core_dim, base_scale=base_scale,
            rank_cap=rank_cap, core_points=core_points, pencil_draws=pencil_draws,
            seed=seed, tolerance=tolerance, span_tolerance=span_tolerance,
            purity=purity, collinearity=collinearity, steps=steps, device=device,
            teacher=teacher, truth_gate_provider=truth_gate_provider,
            reader_count=reader_count, identity_readers=True, min_anchors=2,
        )
        report.update({
            "model": model,
            "layer": int(layer),
            "unitList": units,
            "blockDevice": client.meta.get("blockDevice"),
            "solverOracleCalls": int(client.query_count),
            "oracleEvalSeconds": round(teacher.eval_seconds, 3),
            "pointsPerSecond": round(teacher.counter.points / max(teacher.eval_seconds, 1e-9), 1),
            "truthUnlockQueryCount": client.truth_unlock_query_count,
            "unlockedAfterAllSolverCalls": client.truth_unlock_query_count == int(client.query_count),
        })
        return report
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
    teacher: OracleCallable,
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
    teacher: OracleCallable,
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
    teacher: OracleCallable,
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
    teacher: Any = None,
    truth_gate_provider: Any = None,
    reader_count: int = 0,
    identity_readers: bool = False,
    min_anchors: int = 3,
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

    # The teacher is injected when the block is real. Everything below is unchanged by that:
    # the solve only ever calls `oracle(points)`, so a synthetic block and a live one differ in
    # who answers and in nothing else. The truth arrives the same way - through a provider that
    # the live path only calls after every solver query is spent.
    teacher = teacher if teacher is not None else make_teacher(units, input_dim, output_dim, seed, device)
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

    # One reader is one slice, and the slices are what the span SVD runs on. On the synthetic
    # path `output_dim` is 8 and taking all of them costs nothing; on a live block it is 3072,
    # which would build 3072 slices an anchor for a span whose rank is `2k`. The oracle charge is
    # unchanged either way - it answers the whole vector per point regardless - so this bounds
    # the linear algebra, not the bill.
    reader_total = reader_count if reader_count > 0 else output_dim
    if identity_readers:
        # The bridge already applied them, so here they are coordinates.
        readers = torch.eye(reader_total, device=device, dtype=DTYPE)
    else:
        readers = torch.randn(reader_total, output_dim, generator=generator, device=device, dtype=DTYPE)
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
    # The floor is a parameter because the two paths measured different ones. On the synthetic
    # teacher three anchors is what every result above was taken at; on a live block two is the
    # floor and it is 2.33x cheaper at equal recovery, because an anchor multiplies the whole
    # polarisation design while a reader is another column of an answer already paid for.
    anchor_count = max(min_anchors, -(-2 * rank_cap // max(int(readers.shape[0]), 1)))
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

    truth_raw = truth_gate_provider() if truth_gate_provider is not None else teacher.gate
    truth = truth_raw / truth_raw.norm(dim=1, keepdim=True)
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


def recover_gates_from_measured_oracle(
    oracle: OracleCallable,
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
    max_oracle_points: int | None = None,
    max_workspace_bytes: int | None = None,
    measured_base_points: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Recover gates from an oracle under explicit query and workspace limits."""
    if min(input_dim, output_dim, base_point_count, reader_count) <= 0:
        raise ValueError("input_dim, output_dim, base_point_count, and reader_count must be positive")
    if not steps or any(not math.isfinite(step) or step <= 0.0 for step in steps):
        raise ValueError("steps must contain only finite positive values")
    if not math.isfinite(base_scale):
        raise ValueError("base_scale must be finite")
    if not math.isfinite(direction_multiple) or direction_multiple <= 0.0:
        raise ValueError("direction_multiple must be finite and positive")
    if max_oracle_points is not None and max_oracle_points < 0:
        raise ValueError("max_oracle_points must be non-negative")
    if max_workspace_bytes is not None and max_workspace_bytes <= 0:
        raise ValueError("max_workspace_bytes must be positive")

    workspace = estimate_measured_cubic_workspace(input_dim, direction_multiple)
    coefficient_count = workspace["coefficientCount"]
    direction_count = workspace["directionCount"]
    minimum_workspace_bytes = workspace["minimumWorkspaceBytes"]
    if max_workspace_bytes is not None and minimum_workspace_bytes > max_workspace_bytes:
        raise OracleWorkspaceLimitExceeded(
            input_dim=input_dim,
            direction_count=direction_count,
            coefficient_count=coefficient_count,
            minimum_workspace_bytes=minimum_workspace_bytes,
            max_workspace_bytes=max_workspace_bytes,
        )
    initial_oracle_points = (
        base_point_count * direction_count * len(steps) * 4
    )
    if (
        max_oracle_points is not None
        and initial_oracle_points > max_oracle_points
    ):
        raise OracleQueryBudgetExceeded(
            requested_points=initial_oracle_points,
            spent_points=0,
            max_oracle_points=max_oracle_points,
        )

    if measured_base_points is not None:
        if measured_base_points.ndim != 2 or tuple(measured_base_points.shape) != (
            base_point_count,
            input_dim,
        ):
            raise ValueError(
                "measured_base_points must have shape "
                f"({base_point_count}, {input_dim})"
            )
        if not torch.isfinite(measured_base_points).all().item():
            raise ValueError("measured_base_points must contain only finite values")

    generator = torch.Generator(device=device).manual_seed(seed + 23)
    if measured_base_points is None:
        base_points = torch.randn(
            base_point_count, input_dim, generator=generator, device=device, dtype=DTYPE
        )
        base_points = base_points / base_points.norm(dim=1, keepdim=True) * base_scale
    else:
        base_points = measured_base_points.to(device=device, dtype=DTYPE)
    readers = torch.randn(
        reader_count, output_dim, generator=generator, device=device, dtype=DTYPE
    )
    readers = readers / readers.norm(dim=1, keepdim=True)

    oracle_calls_points = 0

    def checked_oracle(points: torch.Tensor) -> torch.Tensor:
        nonlocal oracle_calls_points
        if points.ndim != 2 or points.shape[1] != input_dim:
            raise ValueError(
                f"oracle input shape {tuple(points.shape)} does not match input dimension {input_dim}"
            )
        requested_points = int(points.shape[0])
        if (
            max_oracle_points is not None
            and oracle_calls_points + requested_points > max_oracle_points
        ):
            raise OracleQueryBudgetExceeded(
                requested_points=requested_points,
                spent_points=oracle_calls_points,
                max_oracle_points=max_oracle_points,
            )
        oracle_calls_points += requested_points
        values = oracle(points)
        if not isinstance(values, torch.Tensor):
            raise TypeError("oracle must return a torch.Tensor")
        expected_shape = (int(points.shape[0]), output_dim)
        if tuple(values.shape) != expected_shape:
            raise ValueError(
                f"oracle output shape {tuple(values.shape)} does not match expected {expected_shape}"
            )
        return values

    span_basis, span_rank = measured_lifted_span(
        checked_oracle,
        base_points,
        readers,
        input_dim,
        steps,
        direction_multiple,
        span_tolerance,
    )
    gates, nullity = foobi_gates(
        span_basis, input_dim, tolerance, purity, collinearity, seed + 3
    )
    span_implied_unit_count = (
        span_rank // 2 if span_rank > 0 and span_rank % 2 == 0 else None
    )
    return {
        "gates": gates,
        "measuredSpanRank": span_rank,
        "spanImpliedUnitCount": span_implied_unit_count,
        "spanRankIsEven": span_rank > 0 and span_rank % 2 == 0,
        "nullity": nullity,
        "gatesMatchSpanRank": (
            span_implied_unit_count is not None
            and int(gates.shape[0]) == span_implied_unit_count
        ),
        "gatesReturned": int(gates.shape[0]),
        "oracleCallsPoints": oracle_calls_points,
    }


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
    """Run oracle-only recovery, then score against synthetic truth post hoc."""
    synthetic_teacher = make_teacher(units, input_dim, output_dim, seed, device)

    def eval_only(points: torch.Tensor) -> torch.Tensor:
        return synthetic_teacher(points)

    recovery = recover_gates_from_measured_oracle(
        oracle=eval_only,
        input_dim=input_dim,
        output_dim=output_dim,
        base_point_count=base_point_count,
        reader_count=reader_count,
        base_scale=base_scale,
        seed=seed,
        tolerance=tolerance,
        span_tolerance=span_tolerance,
        purity=purity,
        collinearity=collinearity,
        steps=steps,
        direction_multiple=direction_multiple,
        device=device,
    )
    gates = recovery["gates"]
    span_rank = int(recovery["measuredSpanRank"])
    truth = synthetic_teacher.gate / synthetic_teacher.gate.norm(dim=1, keepdim=True)
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
        "spanRankIsEven": recovery["spanRankIsEven"],
        "spanImpliedUnitCount": recovery["spanImpliedUnitCount"],
        "unitCountInferenceMatchesTruth": recovery["spanImpliedUnitCount"] == units,
        "spanTolerance": span_tolerance,
        "nullity": recovery["nullity"],
        # A check that needs no truth. The span's rank is `2k` when the units are separable, so
        # a healthy run returns exactly `spanRank / 2` distinct rank-1 directions. A run that has
        # gone past the geometry returns some other number, and says so without being told.
        "gatesMatchSpanRank": recovery["gatesMatchSpanRank"],
        "gatesReturned": recovery["gatesReturned"],
        "gatesAbove0.99": int((best >= 0.99).sum()),
        "gatesAbove0.999": int((best >= 0.999).sum()),
        "medianCosine": float(best.median()),
        "oracleCallsPoints": recovery["oracleCallsPoints"],
        "device": str(gates.device),
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
    parser.add_argument("--live-anchors", type=int, default=2,
                        help="anchors per solve; 2 is the measured floor and anchors are what cost points")
    parser.add_argument("--live-stitch", action="store_true",
                        help="stitch full-dimension gates out of the real layer")
    parser.add_argument("--live-local-files-only", action="store_true")
    parser.add_argument(
        "--capture-real-activations",
        action="store_true",
        help="capture provenance-separated real Qwen MLP activation datasets and smoke-test its opaque full-module oracle",
    )
    parser.add_argument(
        "--capture-output-dir",
        type=str,
        default=None,
        help="new, empty directory for activation-capture datasets and report",
    )
    parser.add_argument(
        "--capture-extra-recovery-prompts-file",
        type=str,
        default=None,
        help="JSON array of {promptId, text} records added only to recovery observations",
    )
    parser.add_argument(
        "--capture-extra-functional-prompts-file",
        type=str,
        default=None,
        help="JSON array of {promptId, text} records added only to functional evaluation",
    )
    parser.add_argument(
        "--capture-extra-distributional-prompts-file",
        type=str,
        default=None,
        help="JSON array of {promptId, text} records added only to distributional analysis",
    )
    parser.add_argument(
        "--verify-real-activation-capture",
        action="store_true",
        help="independently replay captured token sequences against the real Qwen MLP and verify artifact integrity",
    )
    parser.add_argument(
        "--recover-real-swiglu-distillation",
        action="store_true",
        help="fit an adaptive-width SwiGLU student from the opaque full-module oracle and saved natural recovery activations",
    )
    parser.add_argument(
        "--evaluate-real-swiglu-distillation",
        action="store_true",
        help="run a separate post-hoc Qwen reload to measure natural/generalization and structural recovery",
    )
    parser.add_argument(
        "--recover-real-swiglu-continuation",
        action="store_true",
        help="fit the exact SwiGLU function using function-preserving width splits and a streamed output-projection solve",
    )
    parser.add_argument(
        "--probe-real-swiglu-amplitude-response",
        action="store_true",
        help="measure positive/negative amplitude trajectories through the opaque real Qwen SwiGLU oracle",
    )
    parser.add_argument(
        "--evaluate-real-swiglu-amplitude-response",
        action="store_true",
        help="independently replay frozen Qwen amplitude measurements and analyze asymptotic component signatures post hoc",
    )
    parser.add_argument(
        "--distillation-output-dir",
        type=str,
        default=None,
        help="new, empty directory for solver artifacts or the post-hoc evaluation report",
    )
    parser.add_argument("--distillation-max-components", type=int, default=8192)
    parser.add_argument("--distillation-query-budget", type=int, default=73728)
    parser.add_argument("--distillation-initial-train-queries", type=int, default=32768)
    parser.add_argument("--distillation-validation-queries", type=int, default=8192)
    parser.add_argument("--distillation-adaptive-batch-queries", type=int, default=8192)
    parser.add_argument("--distillation-steps-per-stage", type=int, default=1000)
    parser.add_argument("--distillation-batch-size", type=int, default=128)
    parser.add_argument("--distillation-learning-rate", type=float, default=0.003)
    parser.add_argument("--continuation-max-components", type=int, default=8192)
    parser.add_argument("--continuation-query-budget", type=int, default=73728)
    parser.add_argument("--continuation-training-queries", type=int, default=32768)
    parser.add_argument("--continuation-validation-queries", type=int, default=8192)
    parser.add_argument("--continuation-adaptive-batch-queries", type=int, default=8192)
    parser.add_argument("--continuation-steps-per-stage", type=int, default=1000)
    parser.add_argument("--continuation-batch-size", type=int, default=128)
    parser.add_argument("--continuation-learning-rate", type=float, default=0.001)
    parser.add_argument("--continuation-projection-interval", type=int, default=250)
    parser.add_argument("--continuation-projection-batch-size", type=int, default=512)
    parser.add_argument("--continuation-split-perturbation", type=float, default=0.06)
    parser.add_argument("--continuation-projection-ridge", type=float, default=1e-6)
    parser.add_argument(
        "--continuation-natural-training-batch-fraction",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--continuation-directional-derivative-weight",
        type=float,
        default=0.0,
        help="enable oracle finite-difference JVP supervision and weight its normalized loss",
    )
    parser.add_argument(
        "--continuation-directional-difference-step",
        type=float,
        default=0.02,
        help="unit-direction step for paired central-difference oracle queries",
    )
    parser.add_argument(
        "--continuation-directional-derivative-query-fraction",
        type=float,
        default=0.25,
        help="fraction of training/validation oracle points paired for JVP supervision",
    )
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
        args.live_stitch or None, args.capture_real_activations or None,
        args.verify_real_activation_capture or None,
        args.recover_real_swiglu_distillation or None,
        args.recover_real_swiglu_continuation or None,
        args.evaluate_real_swiglu_distillation or None,
        args.probe_real_swiglu_amplitude_response or None,
        args.evaluate_real_swiglu_amplitude_response or None,
    ]
    if args.capture_real_activations and args.verify_real_activation_capture:
        parser.error("capture and independent verification must run as separate invocations")
    if args.recover_real_swiglu_distillation and args.evaluate_real_swiglu_distillation:
        parser.error("solver and post-hoc evaluation must run as separate invocations")
    if args.recover_real_swiglu_continuation and (
        args.recover_real_swiglu_distillation or args.evaluate_real_swiglu_distillation
    ):
        parser.error("continuation recovery must run separately from other solver/evaluator modes")
    if (
        args.probe_real_swiglu_amplitude_response
        and args.evaluate_real_swiglu_amplitude_response
    ):
        parser.error("amplitude probing and post-hoc evaluation must run separately")
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
    if args.capture_real_activations:
        if not args.capture_output_dir:
            parser.error("--capture-real-activations requires --capture-output-dir")
        report["realActivationCapture"] = capture_qwen_real_activations(
            model=args.live_model,
            layer=args.live_layer,
            output_dir=Path(args.capture_output_dir),
            device=device,
            local_files_only=args.live_local_files_only,
            seed=args.seed,
            oracle_query_budget=args.live_max_calls,
            extra_recovery_prompts_file=(
                Path(args.capture_extra_recovery_prompts_file)
                if args.capture_extra_recovery_prompts_file
                else None
            ),
            extra_functional_prompts_file=(
                Path(args.capture_extra_functional_prompts_file)
                if args.capture_extra_functional_prompts_file
                else None
            ),
            extra_distributional_prompts_file=(
                Path(args.capture_extra_distributional_prompts_file)
                if args.capture_extra_distributional_prompts_file
                else None
            ),
        )
    if args.verify_real_activation_capture:
        if not args.capture_output_dir:
            parser.error("--verify-real-activation-capture requires --capture-output-dir")
        report["realActivationVerification"] = verify_qwen_real_activation_capture(
            model=args.live_model,
            output_dir=Path(args.capture_output_dir),
            device=device,
            local_files_only=args.live_local_files_only,
        )
    if args.recover_real_swiglu_distillation:
        if not args.capture_output_dir or not args.distillation_output_dir:
            parser.error(
                "--recover-real-swiglu-distillation requires both --capture-output-dir "
                "and --distillation-output-dir"
            )
        report["realSwiGLUDistillation"] = run_qwen_oracle_distillation(
            capture_dir=Path(args.capture_output_dir),
            output_dir=Path(args.distillation_output_dir),
            model=args.live_model,
            layer=args.live_layer,
            device=device,
            local_files_only=args.live_local_files_only,
            seed=args.seed,
            max_components=args.distillation_max_components,
            oracle_query_budget=args.distillation_query_budget,
            initial_train_queries=args.distillation_initial_train_queries,
            validation_queries=args.distillation_validation_queries,
            adaptive_batch_queries=args.distillation_adaptive_batch_queries,
            steps_per_stage=args.distillation_steps_per_stage,
            batch_size=args.distillation_batch_size,
            learning_rate=args.distillation_learning_rate,
        )
    if args.recover_real_swiglu_continuation:
        if not args.capture_output_dir or not args.distillation_output_dir:
            parser.error(
                "--recover-real-swiglu-continuation requires both --capture-output-dir "
                "and --distillation-output-dir"
            )
        report["realSwiGLUSplitContinuation"] = run_qwen_oracle_split_continuation(
            capture_dir=Path(args.capture_output_dir),
            output_dir=Path(args.distillation_output_dir),
            model=args.live_model,
            layer=args.live_layer,
            device=device,
            local_files_only=args.live_local_files_only,
            seed=args.seed,
            max_components=args.continuation_max_components,
            query_budget=args.continuation_query_budget,
            training_queries=args.continuation_training_queries,
            validation_queries=args.continuation_validation_queries,
            adaptive_batch_queries=args.continuation_adaptive_batch_queries,
            steps_per_stage=args.continuation_steps_per_stage,
            batch_size=args.continuation_batch_size,
            learning_rate=args.continuation_learning_rate,
            projection_interval=args.continuation_projection_interval,
            projection_batch_size=args.continuation_projection_batch_size,
            split_perturbation=args.continuation_split_perturbation,
            projection_ridge=args.continuation_projection_ridge,
            natural_training_batch_fraction=(
                args.continuation_natural_training_batch_fraction
            ),
            directional_derivative_weight=(
                args.continuation_directional_derivative_weight
            ),
            directional_difference_step=(
                args.continuation_directional_difference_step
            ),
            directional_derivative_query_fraction=(
                args.continuation_directional_derivative_query_fraction
            ),
        )
    if args.evaluate_real_swiglu_distillation:
        if not args.capture_output_dir or not args.distillation_output_dir:
            parser.error(
                "--evaluate-real-swiglu-distillation requires both --capture-output-dir "
                "and --distillation-output-dir"
            )
        report["realSwiGLUDistillationEvaluation"] = evaluate_qwen_oracle_distillation(
            capture_dir=Path(args.capture_output_dir),
            distillation_dir=Path(args.distillation_output_dir),
            model=args.live_model,
            layer=args.live_layer,
            device=device,
            local_files_only=args.live_local_files_only,
        )
    if args.probe_real_swiglu_amplitude_response:
        if not args.capture_output_dir or not args.distillation_output_dir:
            parser.error(
                "--probe-real-swiglu-amplitude-response requires both --capture-output-dir "
                "and --distillation-output-dir"
            )
        report["realSwiGLUAmplitudeResponse"] = run_qwen_amplitude_response(
            capture_dir=Path(args.capture_output_dir),
            output_dir=Path(args.distillation_output_dir),
            model=args.live_model,
            layer=args.live_layer,
            device=device,
            local_files_only=args.live_local_files_only,
            seed=args.seed,
            max_oracle_points=args.live_max_calls,
        )
    if args.evaluate_real_swiglu_amplitude_response:
        if not args.capture_output_dir or not args.distillation_output_dir:
            parser.error(
                "--evaluate-real-swiglu-amplitude-response requires both "
                "--capture-output-dir and --distillation-output-dir"
            )
        report["realSwiGLUAmplitudeResponseEvaluation"] = (
            evaluate_qwen_amplitude_response(
                capture_dir=Path(args.capture_output_dir),
                output_dir=Path(args.distillation_output_dir),
                model=args.live_model,
                layer=args.live_layer,
                device=device,
                local_files_only=args.live_local_files_only,
            )
        )
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
    if args.live_stitch:
        work_dir = Path(args.live_work_dir) if args.live_work_dir else Path("/tmp/cubic-live-check")
        report["liveStitch"] = live_stitch(
            model=args.live_model, layer=args.live_layer, units=args.live_units,
            subspace_dim=args.subspace_dim, core_dim=args.core_dim,
            base_scale=args.base_scale, reader_count=args.live_readers,
            rank_cap=args.rank_cap, core_points=args.core_points,
            pencil_draws=args.pencil_draws, steps=list(args.steps), seed=args.seed,
            tolerance=args.rank_tolerance, span_tolerance=args.span_tolerance,
            purity=args.purity, collinearity=args.collinearity,
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
