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
    gate_from_peak_amplitudes,
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
            gate_mode="null-space",
            amplitude_span=0.01,
            amplitude_min_alignment=0.5,
            duplicate_gate_cosine=0.98,
            gate_residual_tolerance=1.0,
            amplitude_crossings=4,
            up_sample_count=0,
            up_sample_scale=0.6,
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
            gate_mode="null-space",
            amplitude_span=0.01,
            amplitude_min_alignment=0.5,
            duplicate_gate_cosine=0.98,
            gate_residual_tolerance=1.0,
            amplitude_crossings=4,
            up_sample_count=0,
            up_sample_scale=0.6,
            seed=2,
        )

        self.assertEqual(telemetry.basisCalls, 4 * 2 * 32)
        self.assertEqual(telemetry.scanCalls, 5 * 2 * 101)
        self.assertEqual(telemetry.oracleCalls, telemetry.basisCalls + telemetry.scanCalls)


class AmplitudeGateTest(unittest.TestCase):
    """One crossing is enough, which is what removes the dependence on boundary density.

    The null-space route needs `rank` crossings of the same unit, so a 32-unit block on the
    live model recovered 17 of 32 until the scan was made dense enough to cost 689792 calls.
    Reading the gate off the amplitudes at a single crossing gave all 32 for 130776.
    """

    @staticmethod
    def setup(units: int = 10, dim: int = 40, seed: int = 5):
        rng = np.random.default_rng(seed)
        gate = rng.normal(0.0, 1.0, size=(units, dim)) / np.sqrt(dim)
        up = rng.normal(0.0, 1.0, size=(units, dim)) / np.sqrt(dim)
        down = rng.normal(0.0, 1.0, size=(dim, units))

        def evaluate(points: np.ndarray) -> np.ndarray:
            z = points @ gate.T
            return ((z / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))) * (points @ up.T)) @ down.T

        return evaluate, gate, down

    def on_boundary(self, gate: np.ndarray, target: int, dim: int, seed: int = 5) -> np.ndarray:
        rng = np.random.default_rng(seed + 1)
        point = rng.normal(size=dim)
        point -= (gate[target] @ point) / (gate[target] @ gate[target]) * gate[target]
        return point / np.linalg.norm(point) * 2e4

    def test_a_single_crossing_returns_the_gate(self) -> None:
        evaluate, gate, down = self.setup()
        dim = gate.shape[1]
        point = self.on_boundary(gate, 0, dim)
        rng = np.random.default_rng(7)
        along = rng.normal(size=dim)
        along = 0.01 * np.linalg.norm(point) * along / np.linalg.norm(along)

        recovered, alignment, calls = gate_from_peak_amplitudes(
            evaluate,
            peak_point=point,
            step_along_line=along,
            output_direction=down[:, 0] / np.linalg.norm(down[:, 0]),
            probe_basis=np.eye(dim),
            step=1.0,
        )

        similarity = abs(float(recovered / np.linalg.norm(recovered) @ (gate[0] / np.linalg.norm(gate[0]))))
        self.assertGreater(similarity, 0.99)
        self.assertGreater(alignment, 0.5)
        self.assertEqual(calls, 8 * dim)

    def test_the_wrong_output_direction_is_reported_as_low_alignment(self) -> None:
        evaluate, gate, down = self.setup()
        dim = gate.shape[1]
        point = self.on_boundary(gate, 0, dim)
        rng = np.random.default_rng(7)
        along = rng.normal(size=dim)
        along = 0.01 * np.linalg.norm(point) * along / np.linalg.norm(along)

        _recovered, alignment, _calls = gate_from_peak_amplitudes(
            evaluate,
            peak_point=point,
            step_along_line=along,
            output_direction=down[:, 3] / np.linalg.norm(down[:, 3]),
            probe_basis=np.eye(dim),
            step=1.0,
        )

        # The projection is onto a direction this crossing has nothing to do with, so the
        # sweep's energy does not lie along it. Without this the method would return a
        # confident-looking vector for a mismatched cluster.
        self.assertLess(alignment, 0.5)


class UpRecoveryTest(unittest.TestCase):
    """The up direction, and the gate norm that has to be found before it can be solved.

    A recovered gate is a direction: its norm is 1, while the real one is around 0.02 on
    LLaMA layer 15. `silu` is not scale free, so evaluating it 50 times too far out puts every
    unit in saturation. That single mistake held the up recovery at 14 of 32 and did not move
    when samples, sample scale, cross-talk or gate accuracy were addressed in turn. With the
    norm chosen per unit it is 32 of 32 at a joint gate/up cosine of 0.9926.
    """

    @staticmethod
    def teacher_with_small_gates(units: int = 6, dim: int = 32, gate_norm: float = 0.02, seed: int = 3):
        rng = np.random.default_rng(seed)
        gate = rng.normal(0.0, 1.0, size=(units, dim))
        gate = gate_norm * gate / np.linalg.norm(gate, axis=1, keepdims=True)
        up = rng.normal(0.0, 1.0, size=(units, dim)) / np.sqrt(dim)
        down = rng.normal(0.0, 1.0, size=(dim, units))

        def evaluate(points: np.ndarray) -> np.ndarray:
            z = points @ gate.T
            return ((z / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))) * (points @ up.T)) @ down.T

        return evaluate, gate, up, down

    def test_a_unit_norm_gate_puts_silu_in_the_wrong_regime(self) -> None:
        _evaluate, gate, _up, _down = self.teacher_with_small_gates()
        rng = np.random.default_rng(9)
        points = rng.normal(0.0, 2.0, size=(500, gate.shape[1]))

        true_argument = np.abs(points @ gate[0])
        unit_norm_argument = np.abs(points @ (gate[0] / np.linalg.norm(gate[0])))

        # Two orders of magnitude apart, which is the difference between the linear part of
        # silu and its saturated tail. Nothing downstream can recover from being handed the
        # wrong one.
        self.assertGreater(float(np.median(unit_norm_argument) / np.median(true_argument)), 10.0)

    def test_up_directions_come_back_when_the_norm_is_searched(self) -> None:
        evaluate, gate, up, _down = self.teacher_with_small_gates()

        result, telemetry = recover_block(
            evaluate,
            input_dim=gate.shape[1],
            active_rank=2 * gate.shape[0],
            line_count=50,
            samples=401,
            scale=2e4,
            step=1e-4,
            peak_fraction=0.05,
            separation=6,
            min_sharpness=3.0,
            cluster_min_cosine=0.95,
            basis_points_count=8,
            basis_scale=1.0,
            gate_mode="amplitude",
            amplitude_span=0.01,
            amplitude_min_alignment=0.5,
            duplicate_gate_cosine=0.98,
            gate_residual_tolerance=1.0,
            amplitude_crossings=4,
            up_sample_count=4000,
            up_sample_scale=2.0,
            seed=5,
        )

        self.assertEqual(telemetry.upsRecovered, telemetry.unitsRecovered)
        recovered = result["upDirections"]
        self.assertGreater(recovered.shape[0], 0)
        recovered = recovered / np.linalg.norm(recovered, axis=1, keepdims=True)
        reference = up / np.linalg.norm(up, axis=1, keepdims=True)
        per_unit = np.abs(recovered @ reference.T).max(axis=0)
        self.assertGreater(float(np.median(per_unit)), 0.9)

    def test_a_solution_off_its_own_hyperplane_is_rejected(self) -> None:
        rng = np.random.default_rng(2)
        rank = 6
        normal = rng.normal(size=rank)
        normal /= np.linalg.norm(normal)
        off_plane = rng.normal(size=(20, rank))

        residual = float(np.median(np.abs(off_plane @ normal) / np.linalg.norm(off_plane, axis=1)))

        # This is the quantity the residual filter thresholds. Points that genuinely lie on
        # the hyperplane give near zero; a spurious solution's points do not, which is what
        # removes it from the returned set instead of leaving a caller to filter it.
        self.assertGreater(residual, 1e-3)


if __name__ == "__main__":
    unittest.main()
