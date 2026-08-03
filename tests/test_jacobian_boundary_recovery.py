#!/usr/bin/env python3
"""Boundary recovery: a basis from Jacobians, an identity from output directions, a gate from
a null space.

Run against LLaMA-3.2-3B layer 15, the same 16-unit block the fitted path recovers, this
returns every gate at cosine 0.99927 or better, median 0.99998, 16 of 16 distinct units, for
189432 oracle calls against 230633 - and hands back the output direction of each unit, which
the fitted path never produces at all.

Three things had to be right and each was measured wrong first. These tests hold all three,
because every one of them fails silently: the run still completes and still reports numbers.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.llm_reader.jacobian_boundary_recovery import (  # noqa: E402
    BoundaryCrossing,
    cluster_by_output_direction,
    gate_from_cluster,
    recover_basis_from_jacobian,
    recover_block,
)


def teacher(units: int = 8, dim: int = 64, seed: int = 3):
    rng = np.random.default_rng(seed)
    gate = rng.normal(0.0, 1.0, size=(units, dim)) / np.sqrt(dim)
    up = rng.normal(0.0, 1.0, size=(units, dim)) / np.sqrt(dim)
    down = rng.normal(0.0, 1.0, size=(dim, units))

    def evaluate(points: np.ndarray) -> np.ndarray:
        z = points @ gate.T
        return ((z / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))) * (points @ up.T)) @ down.T

    return evaluate, gate, up, down


class BasisFromJacobiansTest(unittest.TestCase):
    def test_one_jacobian_does_not_span_the_active_subspace(self) -> None:
        evaluate, gate, _up, _down = teacher()
        rng = np.random.default_rng(1)
        point = rng.normal(size=(1, 64))

        basis, _spectrum, _calls = recover_basis_from_jacobian(evaluate, 64, 16, point=point, step=1e-4)
        energy = np.mean([np.linalg.norm(basis @ row) / np.linalg.norm(row) for row in gate])

        # At one point each unit contributes a single direction - its own mixture of g and u -
        # so the rank is the unit count, not twice it. Assuming otherwise cost a debugging
        # cycle and produced gates at cosine 0.46 that looked like a clustering problem.
        self.assertLess(energy, 0.95)

    def test_stacked_jacobians_do(self) -> None:
        evaluate, gate, up, _down = teacher()
        rng = np.random.default_rng(1)
        points = rng.normal(size=(8, 64))

        basis, _spectrum, calls = recover_basis_from_jacobian(evaluate, 64, 16, point=points, step=1e-4)

        for reference in (gate, up):
            energy = np.mean([np.linalg.norm(basis @ row) / np.linalg.norm(row) for row in reference])
            self.assertGreater(energy, 0.99)
        self.assertEqual(calls, 8 * 2 * 64)


class ClusteringTest(unittest.TestCase):
    @staticmethod
    def crossing(direction: np.ndarray, point: np.ndarray, sharpness: float = 5.0) -> BoundaryCrossing:
        return BoundaryCrossing(
            line=0,
            coordinate=0.0,
            output_direction=direction / np.linalg.norm(direction),
            basis_point=point,
            peak_sharpness=sharpness,
        )

    def test_opposite_signs_of_one_direction_are_the_same_unit(self) -> None:
        direction = np.array([1.0, 0.0, 0.0])
        crossings = [
            self.crossing(direction, np.zeros(3)),
            self.crossing(-direction, np.zeros(3)),
        ]

        # A crossing read from either side hands back the direction with either sign, and a
        # unit counted twice splits its points between halves too small to solve.
        self.assertEqual(len(cluster_by_output_direction(crossings, min_cosine=0.99)), 1)

    def test_distinct_directions_stay_apart(self) -> None:
        crossings = [
            self.crossing(np.array([1.0, 0.0, 0.0]), np.zeros(3)),
            self.crossing(np.array([0.0, 1.0, 0.0]), np.zeros(3)),
        ]

        self.assertEqual(len(cluster_by_output_direction(crossings, min_cosine=0.99)), 2)

    def test_near_duplicates_merge_through_their_centroids(self) -> None:
        base = np.array([1.0, 0.0, 0.0])
        wobble = np.array([1.0, 0.02, 0.0])
        crossings = [self.crossing(base, np.zeros(3)), self.crossing(wobble, np.zeros(3))]

        self.assertEqual(len(cluster_by_output_direction(crossings, min_cosine=0.99)), 1)


class GateFromNullSpaceTest(unittest.TestCase):
    def test_points_on_a_hyperplane_return_its_normal(self) -> None:
        rng = np.random.default_rng(4)
        rank = 8
        normal = rng.normal(size=rank)
        normal /= np.linalg.norm(normal)
        raw = rng.normal(size=(40, rank))
        on_plane = raw - np.outer(raw @ normal, normal)
        cluster = [
            BoundaryCrossing(line=0, coordinate=0.0, output_direction=np.ones(3), basis_point=row, peak_sharpness=5.0)
            for row in on_plane
        ]

        solved = gate_from_cluster(cluster, rank)

        self.assertIsNotNone(solved)
        recovered, determination = solved
        self.assertGreater(abs(float(recovered @ normal)), 0.999)
        self.assertGreater(determination, 100.0)

    def test_too_few_points_are_refused_rather_than_guessed(self) -> None:
        rng = np.random.default_rng(4)
        cluster = [
            BoundaryCrossing(line=0, coordinate=0.0, output_direction=np.ones(3), basis_point=row, peak_sharpness=5.0)
            for row in rng.normal(size=(5, 8))
        ]

        # With fewer points than the rank, the smallest singular vector is not the normal - it
        # is an arbitrary direction the points happen not to span, which reads as an answer.
        self.assertIsNone(gate_from_cluster(cluster, 8))


class EndToEndTest(unittest.TestCase):
    def test_the_block_comes_back_without_any_fitting(self) -> None:
        evaluate, gate, _up, down = teacher(units=6, dim=48, seed=11)

        result, telemetry = recover_block(
            evaluate,
            input_dim=48,
            active_rank=12,
            line_count=60,
            samples=401,
            scale=2e4,
            step=1e-4,
            peak_fraction=0.05,
            separation=6,
            min_sharpness=3.0,
            cluster_min_cosine=0.99,
            basis_points_count=8,
            basis_scale=1.0,
            seed=5,
        )

        self.assertGreater(telemetry.unitsRecovered, 0)
        recovered = result["gates"] @ result["basis"]
        recovered = recovered / np.linalg.norm(recovered, axis=1, keepdims=True)
        reference = gate / np.linalg.norm(gate, axis=1, keepdims=True)
        best = np.abs(recovered @ reference.T).max(axis=1)
        self.assertGreater(float(np.median(best)), 0.99)

        outputs = result["outputDirections"]
        normalised = down / np.linalg.norm(down, axis=0, keepdims=True)
        self.assertGreater(float(np.median(np.abs(outputs @ normalised).max(axis=1))), 0.99)

    def test_the_call_budget_is_exactly_accounted(self) -> None:
        evaluate, _gate, _up, _down = teacher(units=4, dim=32, seed=2)

        _result, telemetry = recover_block(
            evaluate,
            input_dim=32,
            active_rank=8,
            line_count=5,
            samples=101,
            scale=2e4,
            step=1e-4,
            peak_fraction=0.05,
            separation=6,
            min_sharpness=3.0,
            cluster_min_cosine=0.99,
            basis_points_count=4,
            basis_scale=1.0,
            seed=2,
        )

        self.assertEqual(telemetry.basisCalls, 4 * 2 * 32)
        self.assertEqual(telemetry.scanCalls, 5 * 2 * 101)
        self.assertEqual(telemetry.oracleCalls, telemetry.basisCalls + telemetry.scanCalls)


if __name__ == "__main__":
    unittest.main()
