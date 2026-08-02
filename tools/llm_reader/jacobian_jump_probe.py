#!/usr/bin/env python3
"""Read a SwiGLU unit off the rank-1 jump in the Jacobian at its activation boundary.

Why this exists. The recovery path in `blind_swiglu_oracle_decompiler.py` reaches 16/16 atom
identity on a 16-unit block, but it reads boundaries as kinks in a *scalar* oracle. The width
of a SiLU transition along a probe line is 1/(S*|g.d|) for probe scale S, while the density of
boundaries along that line grows with the unit count, so a full 8192-unit layer buries every
kink under its neighbours. Measured here: with the scale held fixed, the share of boundaries
read correctly falls from 94% at 32 units to 17% at 2048.

What changes with a vector-valued oracle. The full FFN output is y = D a(x), so the Jacobian
is J(x) = sum_i d_i (x) da_i/dx. Crossing unit i's boundary switches its activation indicator
while silu(g_i.x) is zero there, so J changes by (u_i.x) * d_i (x) g_i - a rank-1 matrix. One
SVD of that jump returns the gate direction and the output direction of that single unit,
with no fitting anywhere. Boundaries that overlap in *time* along the line no longer collide,
because their jumps live in different rank-1 subspaces.

Two details decide whether it works, both measured:

  - The smooth drift of the other units must be removed. Differencing the Jacobian either
    side of the peak leaves a rank-1 share of 0.14; extrapolating each side linearly from two
    points first takes it to 0.54 and the median gate cosine to 0.99991.
  - The rank-1 share is itself the quality filter, and it needs no access to truth. At a
    threshold of 0.7, every accepted crossing returned both directions at cosine 0.99999
    (10 of 40 crossings on one line); unfiltered, 68% cleared 0.99.

This module is the measurement harness for that claim against a synthetic teacher. It is not
wired into a lane: the live-model version needs a vector-valued oracle, which the current
oracle protocol does not expose. Run it directly to reproduce the numbers above.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class Teacher:
    gate: np.ndarray
    up: np.ndarray
    down: np.ndarray

    def __call__(self, points: np.ndarray) -> np.ndarray:
        z = points @ self.gate.T
        activation = z / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))
        return (activation * (points @ self.up.T)) @ self.down.T


def make_teacher(units: int, input_dim: int, output_dim: int, seed: int) -> Teacher:
    rng = np.random.default_rng(seed)
    return Teacher(
        gate=rng.normal(0.0, 1.0, size=(units, input_dim)) / np.sqrt(input_dim),
        up=rng.normal(0.0, 1.0, size=(units, input_dim)) / np.sqrt(input_dim),
        down=rng.normal(0.0, 1.0, size=(output_dim, units)),
    )


def find_peaks(magnitude: np.ndarray, height: float, distance: int) -> np.ndarray:
    """Local maxima above `height`, thinned to one per `distance` samples.

    Written out rather than taken from scipy so the probe has no dependency the solver
    does not already carry.
    """
    candidates = [
        index
        for index in range(1, magnitude.size - 1)
        if magnitude[index] >= magnitude[index - 1]
        and magnitude[index] > magnitude[index + 1]
        and magnitude[index] >= height
    ]
    candidates.sort(key=lambda index: -magnitude[index])
    kept: list[int] = []
    for index in candidates:
        if all(abs(index - other) >= distance for other in kept):
            kept.append(index)
    return np.array(sorted(kept), dtype=int)


def probe_line(
    teacher: Teacher,
    *,
    scale: float,
    samples: int,
    step: float,
    rank1_threshold: float,
    max_crossings: int,
    seed: int,
) -> dict:
    input_dim = teacher.gate.shape[1]
    rng = np.random.default_rng(seed)

    def unit(vector: np.ndarray) -> np.ndarray:
        return vector / np.linalg.norm(vector)

    origin = unit(rng.normal(size=input_dim))
    direction = unit(rng.normal(size=input_dim))
    probe = unit(rng.normal(size=input_dim))

    coordinates = np.linspace(-1.0, 1.0, samples)
    line = scale * (origin[None, :] + coordinates[:, None] * direction[None, :])

    # One directional derivative along the line is enough to locate the boundaries.
    scan = (teacher(line + step * probe) - teacher(line - step * probe)) / (2.0 * step)
    calls = 2 * samples
    slope = np.gradient(scan, coordinates, axis=0)
    magnitude = np.linalg.norm(slope, axis=1)
    peaks = find_peaks(magnitude, height=float(np.max(magnitude)) * 0.05, distance=3)

    truth_crossings = np.array([
        -(teacher.gate[index] @ origin) / (teacher.gate[index] @ direction)
        for index in range(teacher.gate.shape[0])
    ])

    basis = np.eye(input_dim)

    def jacobian(point: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 2 * input_dim
        return (teacher(point[None, :] + step * basis) - teacher(point[None, :] - step * basis)) / (2.0 * step)

    accepted: list[dict] = []
    for position, peak in enumerate(peaks):
        previous = peaks[position - 1] if position > 0 else 0
        following = peaks[position + 1] if position + 1 < peaks.size else samples - 1
        left_gap, right_gap = (peak - previous) // 2, (following - peak) // 2
        if min(left_gap, right_gap) < 6:
            continue
        # Two points a side, extrapolated onto the peak, so the other units' smooth drift
        # cancels and what remains is the one boundary's jump.
        left = 2.0 * jacobian(line[peak - left_gap // 2]) - jacobian(line[peak - left_gap])
        right = 2.0 * jacobian(line[peak + right_gap // 2]) - jacobian(line[peak + right_gap])
        left_vectors, singular, right_vectors = np.linalg.svd(right - left, full_matrices=False)
        rank1_share = float(singular[0] / singular.sum())

        nearest = int(np.argmin(np.abs(truth_crossings - coordinates[peak])))
        accepted.append({
            "rank1Share": rank1_share,
            "accepted": rank1_share >= rank1_threshold,
            "gateCosine": abs(float(left_vectors[:, 0] @ (teacher.gate[nearest] / np.linalg.norm(teacher.gate[nearest])))),
            "outputCosine": abs(float(right_vectors[0] @ (teacher.down[:, nearest] / np.linalg.norm(teacher.down[:, nearest])))),
        })
        if len(accepted) >= max_crossings:
            break

    kept = [row for row in accepted if row["accepted"]]
    summarise = lambda rows, field: (
        float(np.median([row[field] for row in rows])) if rows else None,
        float(np.mean([row[field] > 0.99 for row in rows])) if rows else None,
    )
    gate_median, gate_share = summarise(kept, "gateCosine")
    output_median, output_share = summarise(kept, "outputCosine")
    all_gate_median, all_gate_share = summarise(accepted, "gateCosine")
    return {
        "units": int(teacher.gate.shape[0]),
        "inputDim": int(input_dim),
        "scale": float(scale),
        "rank1Threshold": float(rank1_threshold),
        "oracleCalls": int(calls),
        "crossingsExamined": len(accepted),
        "crossingsAccepted": len(kept),
        "acceptedGateCosineMedian": gate_median,
        "acceptedGateCosineShareAbove099": gate_share,
        "acceptedOutputCosineMedian": output_median,
        "acceptedOutputCosineShareAbove099": output_share,
        "unfilteredGateCosineMedian": all_gate_median,
        "unfilteredGateCosineShareAbove099": all_gate_share,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=int, default=512)
    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--output-dim", type=int, default=512)
    parser.add_argument("--scale", type=float, default=2e4)
    parser.add_argument("--samples", type=int, default=8001)
    parser.add_argument("--step", type=float, default=1e-4)
    parser.add_argument("--rank1-threshold", type=float, default=0.7)
    parser.add_argument("--max-crossings", type=int, default=40)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    teacher = make_teacher(args.units, args.input_dim, args.output_dim, args.seed)
    report = probe_line(
        teacher,
        scale=args.scale,
        samples=args.samples,
        step=args.step,
        rank1_threshold=args.rank1_threshold,
        max_crossings=args.max_crossings,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
