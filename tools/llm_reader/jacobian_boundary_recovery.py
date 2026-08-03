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

import json
import sys
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


@dataclass
class RecoveryTelemetry:
    oracleCalls: int = 0
    basisCalls: int = 0
    scanCalls: int = 0
    linesScanned: int = 0
    crossingsFound: int = 0
    clusters: int = 0
    unitsRecovered: int = 0
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
    """
    axes = np.eye(input_dim)
    rows: list[np.ndarray] = []
    calls = 0
    for probe_point in np.atleast_2d(point):
        plus = evaluate(probe_point[None, :] + step * axes)
        minus = evaluate(probe_point[None, :] - step * axes)
        rows.append(((plus - minus) / (2.0 * step)).T)
        calls += 2 * input_dim
    stacked = np.vstack(rows)
    _, singular, right = np.linalg.svd(stacked, full_matrices=False)
    return right[:active_rank], singular, calls


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
) -> tuple[list[BoundaryCrossing], int]:
    """Walk one line and read an output direction off every boundary it crosses.

    The probe stays inside the recovered basis, so the line lives in the same `2k` dimensions
    the units do. Two calls per sample: the boundaries show up as peaks in how fast the
    directional derivative turns, and the derivative at a peak is parallel to that unit's
    output direction.
    """
    coordinates = np.linspace(-1.0, 1.0, samples)
    basis_points = origin[None, :] + coordinates[:, None] * direction[None, :]
    line = scale * (basis_points @ basis)
    probe_vector = probe @ basis

    forward = evaluate(line + step * probe_vector[None, :])
    backward = evaluate(line - step * probe_vector[None, :])
    calls = 2 * samples
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

    crossings: list[BoundaryCrossing] = []
    for index in sorted(kept):
        vector = turning[index]
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
            output_direction=vector / norm,
            basis_point=basis_points[index].copy(),
            peak_sharpness=sharpness,
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
    points = np.stack([crossing.basis_point for crossing in cluster])
    _, singular, right = np.linalg.svd(points, full_matrices=False)
    if singular.size < 2 or singular[-1] <= 0.0:
        return None
    determination = float(singular[-2] / singular[-1])
    return right[-1], determination


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
    basis, spectrum, basis_calls = recover_basis_from_jacobian(
        evaluate, input_dim, active_rank, point=basis_points, step=step
    )
    telemetry.basisCalls = basis_calls
    telemetry.oracleCalls += basis_calls
    energy = float(np.sum(spectrum[:active_rank] ** 2) / max(np.sum(spectrum ** 2), 1e-30))
    telemetry.notes.append(f"basis energy in rank {active_rank}: {energy:.12f}")

    crossings: list[BoundaryCrossing] = []
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
        )
        crossings.extend(found)
        telemetry.scanCalls += calls
        telemetry.oracleCalls += calls
        telemetry.linesScanned += 1
    telemetry.crossingsFound = len(crossings)

    clusters = cluster_by_output_direction(crossings, min_cosine=cluster_min_cosine)
    telemetry.clusters = len(clusters)

    gates: list[np.ndarray] = []
    outputs: list[np.ndarray] = []
    determinations: list[float] = []
    for cluster in sorted(clusters, key=len, reverse=True):
        solved = gate_from_cluster(cluster, active_rank)
        if solved is None:
            continue
        gate, determination = solved
        gates.append(gate)
        centroid = np.mean([
            crossing.output_direction * (1.0 if crossing.output_direction @ cluster[0].output_direction >= 0 else -1.0)
            for crossing in cluster
        ], axis=0)
        outputs.append(centroid / np.linalg.norm(centroid))
        determinations.append(determination)
    telemetry.unitsRecovered = len(gates)

    return {
        "basis": basis,
        "basisSpectrum": spectrum,
        "gates": np.stack(gates) if gates else np.zeros((0, active_rank)),
        "outputDirections": np.stack(outputs) if outputs else np.zeros((0, 1)),
        "gateDeterminations": determinations,
        "clusterSizes": [len(cluster) for cluster in sorted(clusters, key=len, reverse=True)],
    }, telemetry


def summarise(result: dict[str, Any], telemetry: RecoveryTelemetry) -> str:
    return json.dumps({
        "oracleCalls": telemetry.oracleCalls,
        "basisCalls": telemetry.basisCalls,
        "scanCalls": telemetry.scanCalls,
        "linesScanned": telemetry.linesScanned,
        "crossingsFound": telemetry.crossingsFound,
        "clusters": telemetry.clusters,
        "unitsRecovered": telemetry.unitsRecovered,
        "clusterSizes": result["clusterSizes"][:16],
        "notes": telemetry.notes,
    }, indent=2)


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
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-json", default="")
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
                seed=args.seed,
            )
            report: dict[str, Any] = {
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
                report["posthoc"] = {
                    "gateCosineMedian": float(np.median(best)),
                    "gateCosineMin": float(np.min(best)),
                    "gateCosineShareAbove099": float(np.mean(best > 0.99)),
                    "distinctUnitsMatched": int(len(set(similarity.argmax(axis=1).tolist()))),
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
