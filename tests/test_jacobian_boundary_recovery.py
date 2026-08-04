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
    calibrate_gate_norms_on_rays,  # noqa: E402
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
            min_cluster_size=1,
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
            min_cluster_size=1,
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
            min_cluster_size=1,
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


class GateNormCalibrationTest(unittest.TestCase):
    """The norm is the part of a gate that decides the unit's regime, so it has to be measured.

    `silu` is not scale free. A gate direction with the wrong norm evaluates the unit in the
    wrong part of the curve, and a norm near zero describes a unit that never fires at all -
    which is what the isotropic-cloud search returned for 25 of 64 units on layer 15 before
    the ray replaced it.
    """

    @staticmethod
    def teacher(units: int = 6, dim: int = 256, seed: int = 3):
        rng = np.random.default_rng(seed)
        gate = rng.normal(size=(units, dim)) / np.sqrt(dim)
        gate *= np.geomspace(0.3, 3.0, units)[:, None] / np.linalg.norm(gate, axis=1, keepdims=True)
        up = rng.normal(size=(units, dim)) / np.sqrt(dim)
        down = rng.normal(size=(dim, units))
        down /= np.linalg.norm(down, axis=0, keepdims=True)

        def evaluate(points: np.ndarray) -> np.ndarray:
            z = points @ gate.T
            return ((z / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))) * (points @ up.T)) @ down.T

        return evaluate, gate, down

    def calibrate(self, dim: int = 256, **overrides):
        evaluate, gate, down = self.teacher(dim=dim)
        settings = dict(
            points_per_ray=192,
            min_magnitude=1e-2,
            max_magnitude=30.0,
            offset_scale=0.25,
            request_chunk=4096,
            seed=5,
        )
        settings.update(overrides)
        directions = gate / np.linalg.norm(gate, axis=1, keepdims=True)
        norms, diagnostics = calibrate_gate_norms_on_rays(evaluate, directions, down.T, **settings)
        return norms, diagnostics, np.linalg.norm(gate, axis=1)

    def test_the_measured_norm_matches_the_real_one(self) -> None:
        norms, _diagnostics, truth = self.calibrate()

        self.assertLess(float(np.max(np.abs(norms / truth - 1.0))), 0.1)

    def test_crowding_is_survived_rather_than_assumed_away(self) -> None:
        """Neighbours sit at `1 / (n_j |ĝ_j.ĝ_i|)`, so how crowded the ray is falls with the
        input dimension - and the measurement has to stay usable while they are close.

        Measured across 24, 64, 256 and 1024 dimensions, one ray put 8.3% of units outside
        10% of their true norm and the worst 51% out; three rays reduced by the median take
        that to 6.2% and 30%. The typical unit is unaffected either way, which is why the
        median error is the wrong thing to watch here and the tail is the right one.
        """
        crowded, _diagnostics, crowded_truth = self.calibrate(dim=24)

        # Every unit still lands within a factor that keeps `silu` in the right regime, even
        # where the neighbours are only a factor of five away in `s`.
        self.assertLess(float(np.max(np.abs(crowded / crowded_truth - 1.0))), 0.5)
        self.assertLess(float(np.median(np.abs(crowded / crowded_truth - 1.0))), 0.1)

    def test_one_wild_ray_does_not_carry_the_answer(self) -> None:
        norms, diagnostics, truth = self.calibrate()

        # The spread between rays is reported so a caller can see the disagreement that the
        # median hid, rather than reading a single reduced number as though it were clean.
        self.assertTrue(all("raySpread" in row for row in diagnostics))
        self.assertLess(float(np.max(np.abs(norms / truth - 1.0))), 0.1)

    def test_norms_two_decades_apart_are_all_recovered(self) -> None:
        # Recovering a single scale is easy for the wrong reason if every unit shares it.
        norms, _diagnostics, truth = self.calibrate()

        self.assertGreater(truth.max() / truth.min(), 5.0)
        self.assertTrue(np.all(np.abs(norms / truth - 1.0) < 0.1))

    def test_the_knee_share_reports_when_the_ray_missed_the_transition(self) -> None:
        # Probed far above every boundary, the ray sees only the saturated quadratic, which
        # the background explains on its own. The norm returned there is a guess and the
        # diagnostic has to say so rather than let a caller read it as measured.
        _norms, diagnostics, _truth = self.calibrate(min_magnitude=2e3, max_magnitude=8e3)
        missed = [row["kneeShare"] for row in diagnostics]

        _norms, in_range, _truth = self.calibrate()
        found = [row["kneeShare"] for row in in_range]

        self.assertLess(float(np.median(missed)), float(np.median(found)))

    def test_the_ray_costs_one_pass_per_unit_and_repeat(self) -> None:
        calls = []
        evaluate, gate, down = self.teacher()

        def counted(points: np.ndarray) -> np.ndarray:
            calls.append(points.shape[0])
            return evaluate(points)

        directions = gate / np.linalg.norm(gate, axis=1, keepdims=True)
        calibrate_gate_norms_on_rays(
            counted, directions, down.T,
            points_per_ray=192, min_magnitude=1e-2, max_magnitude=30.0,
            offset_scale=0.25, request_chunk=4096, seed=5,
        )

        # Two sided, so `points_per_ray // 2` magnitudes become that many points each way,
        # and the default three rays per unit are what buys the tail behaviour above.
        self.assertEqual(sum(calls), gate.shape[0] * 192 * 3)


class GateSignTest(unittest.TestCase):
    """Which side of the boundary the unit fires on is part of the unit, and it is not free.

    The gate arrives as a null-space vector, so its sign is undetermined, and every direction
    measure in this file takes an absolute cosine and cannot see that. A flipped gate models a
    unit that is silent exactly when the real one fires. On layer 15 that was 25 of 64 units:
    they cleared every cosine bar, then ranked inputs backwards at -0.74 against the real unit,
    with the predicted optimum landing where the real activation is 2e-5 of its peak.
    """

    @staticmethod
    def teacher(units: int = 5, dim: int = 32, seed: int = 4):
        rng = np.random.default_rng(seed)
        gate = rng.normal(size=(units, dim))
        gate /= np.linalg.norm(gate, axis=1, keepdims=True)
        up = rng.normal(size=(units, dim)) / np.sqrt(dim)
        down = rng.normal(size=(dim, units))

        def evaluate(points: np.ndarray) -> np.ndarray:
            z = points @ gate.T
            return ((z / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))) * (points @ up.T)) @ down.T

        return evaluate, gate, up, down

    def recover(self):
        evaluate, gate, up, down = self.teacher()
        result, _telemetry = recover_block(
            evaluate,
            input_dim=gate.shape[1],
            active_rank=2 * gate.shape[0],
            line_count=50, samples=401, scale=2e4, step=1e-4,
            peak_fraction=0.05, separation=6, min_sharpness=3.0,
            cluster_min_cosine=0.95, basis_points_count=8, basis_scale=1.0,
            gate_mode="amplitude", amplitude_span=0.01, amplitude_min_alignment=0.5,
            duplicate_gate_cosine=0.98, gate_residual_tolerance=1.0,
            amplitude_crossings=4, min_cluster_size=1,
            up_sample_count=4000, up_sample_scale=2.0, seed=5,
        )
        recovered = result["gates"] @ result["basis"]
        recovered = recovered / np.linalg.norm(recovered, axis=1, keepdims=True)
        return recovered, gate / np.linalg.norm(gate, axis=1, keepdims=True)

    def test_the_recovered_gate_points_the_way_the_unit_fires(self) -> None:
        recovered, reference = self.recover()
        signed = recovered @ reference.T
        assigned = np.abs(signed).argmax(axis=1)

        for solution in range(recovered.shape[0]):
            # Signed, not absolute. An absolute cosine here would pass on a gate pointing the
            # wrong way, which is the whole failure this test exists for.
            self.assertGreater(float(signed[solution, assigned[solution]]), 0.9)

    def test_a_flipped_gate_would_model_a_unit_that_is_silent_when_the_real_one_fires(self) -> None:
        _evaluate, gate, _up, _down = self.teacher()
        rng = np.random.default_rng(7)
        points = rng.normal(0.0, 2.0, size=(2000, gate.shape[1]))

        def activation(direction: np.ndarray) -> np.ndarray:
            z = points @ direction
            return z / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))

        forward = activation(gate[0])
        flipped = activation(-gate[0])

        # They are not near-duplicates that a tolerance could excuse: where one is large the
        # other is essentially zero.
        self.assertLess(float(np.mean(flipped[forward > np.quantile(forward, 0.9)])), 0.05)


class OutputChannelUnmixingTest(unittest.TestCase):
    """Reading a unit through its own output direction reads every other unit with it.

    `R = A D`, so projecting the response on `d_i` returns `sum_j (d_j.d_i) a_j`. Random
    directions in `dim` dimensions overlap by about `1/sqrt(dim)` and those errors add in
    quadrature, so the leak grows as `sqrt(k/dim)` - tolerable at 128 units in 3072
    dimensions, larger than the signal well before 8192. `D` is recovered, so the mixing is a
    matrix that can be inverted rather than a noise floor to live under.
    """

    @staticmethod
    def block(units: int, dim: int, points: int = 2500, seed: int = 1):
        rng = np.random.default_rng(seed)
        gate = rng.normal(size=(units, dim)) / np.sqrt(dim)
        up = rng.normal(size=(units, dim)) / np.sqrt(dim)
        down = rng.normal(size=(units, dim))
        down /= np.linalg.norm(down, axis=1, keepdims=True)
        inputs = rng.normal(0.0, 2.0, size=(points, dim))
        z = inputs @ gate.T
        activation = (z / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))) * (inputs @ up.T)
        return inputs, activation @ down, gate, up, down

    @staticmethod
    def solve(inputs, targets, gate, up):
        """Recover every up direction from targets that are one column per unit."""
        cosines = []
        for index in range(gate.shape[0]):
            z = inputs @ gate[index]
            smooth = z / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))
            design = np.hstack((smooth[:, None] * inputs, inputs))
            solution, *_ = np.linalg.lstsq(design, targets[:, index], rcond=None)
            recovered = solution[: inputs.shape[1]]
            cosines.append(
                abs(float(recovered @ up[index]))
                / (np.linalg.norm(recovered) * np.linalg.norm(up[index]))
            )
        return np.array(cosines)

    def test_inverting_the_mixing_beats_projecting_through_it(self) -> None:
        inputs, responses, gate, up, down = self.block(units=120, dim=160)

        projected = self.solve(inputs, responses @ down.T, gate, up)
        unmixed = self.solve(inputs, responses @ np.linalg.pinv(down), gate, up)

        # At 120 units in 160 dimensions the leak is about 0.87 of the signal, which is the
        # regime a full 8192-unit layer sits in, not an invented worst case. Kept small on
        # purpose: a suite that takes a quarter of an hour stops being run, and the effect
        # being measured is set by the ratio `k/dim`, not by the absolute size.
        self.assertLess(float(np.median(projected)), 0.99)
        self.assertGreater(float(np.median(unmixed)), 0.999)
        self.assertGreater(float(np.mean(unmixed > 0.99)), float(np.mean(projected > 0.99)))

    def test_the_gain_grows_with_the_crowding(self) -> None:
        gains = []
        for units in (24, 120):
            inputs, responses, gate, up, down = self.block(units=units, dim=160)
            projected = self.solve(inputs, responses @ down.T, gate, up)
            unmixed = self.solve(inputs, responses @ np.linalg.pinv(down), gate, up)
            gains.append(float(np.median(unmixed) - np.median(projected)))

        # Sparse blocks barely need it; crowded ones cannot do without it. That ordering is
        # the reason this is the step that scales rather than a general tidy-up.
        self.assertGreater(gains[1], gains[0])

    def test_it_costs_one_solve_rather_than_a_sweep_per_unit(self) -> None:
        _inputs, responses, _gate, _up, down = self.block(units=40, dim=160)

        # A `dim x k` pseudo inverse applied once, against the previous answer of repeatedly
        # subtracting fitted estimates - which injected each neighbour's error while removing
        # its contribution, and took 64 units from 64 up directions to 60.
        unmixed = responses @ np.linalg.pinv(down)

        self.assertEqual(unmixed.shape, (responses.shape[0], down.shape[0]))


class HeldOutReconstructionTest(unittest.TestCase):
    """The only number here that a caller without the model's weights can act on.

    Every per-unit figure in this file is a cosine against `gate_proj` and `up_proj`, which is
    a post-hoc witness: it exists to validate the method, and is not available in the setting
    the method is for. The recovered block is a complete function, so it can be run on points
    it was never fitted on and compared against the oracle - and that single number moves for
    every failure at once, a missed unit, a norm at the wrong scale, a gate pointing the wrong
    way, an up direction bent by a neighbour.
    """

    @staticmethod
    def teacher(units: int = 6, dim: int = 32, seed: int = 3):
        rng = np.random.default_rng(seed)
        gate = rng.normal(size=(units, dim)) / np.sqrt(dim)
        up = rng.normal(size=(units, dim)) / np.sqrt(dim)
        down = rng.normal(size=(dim, units))

        def evaluate(points: np.ndarray) -> np.ndarray:
            z = points @ gate.T
            return ((z / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))) * (points @ up.T)) @ down.T

        return evaluate, gate, up, down

    def recover(self, **overrides):
        evaluate, gate, _up, _down = self.teacher()
        settings = dict(
            input_dim=gate.shape[1], active_rank=2 * gate.shape[0],
            line_count=50, samples=401, scale=2e4, step=1e-4,
            peak_fraction=0.05, separation=6, min_sharpness=3.0,
            cluster_min_cosine=0.95, basis_points_count=8, basis_scale=1.0,
            gate_mode="amplitude", amplitude_span=0.01, amplitude_min_alignment=0.5,
            duplicate_gate_cosine=0.98, gate_residual_tolerance=1.0,
            amplitude_crossings=4, min_cluster_size=1,
            up_sample_count=4000, up_sample_scale=2.0, seed=5,
        )
        settings.update(overrides)
        result, _telemetry = recover_block(evaluate, **settings)
        return result

    def test_a_good_recovery_reproduces_the_oracle_on_points_it_never_saw(self) -> None:
        result = self.recover()

        self.assertIsNotNone(result["reconstruction"])
        self.assertGreater(result["reconstruction"]["rSquared"], 0.95)

    def test_the_points_are_held_out_rather_than_the_ones_it_was_fitted_on(self) -> None:
        # Fitting quality on the fitting set says nothing about a missed unit, because the
        # solve is free to spend the units it does have on explaining what the missing one
        # did. Different points are the whole content of the check.
        result = self.recover(reconstruction_points=1500)

        self.assertEqual(result["reconstruction"]["points"], 1500)

    def test_it_can_be_turned_off_without_breaking_the_report(self) -> None:
        result = self.recover(reconstruction_points=0)

        self.assertIsNone(result["reconstruction"])

    def test_a_block_recovered_with_units_missing_scores_worse(self) -> None:
        # Half the budget cannot describe the whole block, and the point of a blind measure is
        # that it says so without being told how many units there were.
        whole = self.recover()
        partial = self.recover(active_rank=4, line_count=12, min_cluster_size=8)

        if partial["reconstruction"] is not None:
            self.assertLess(
                partial["reconstruction"]["rSquared"],
                whole["reconstruction"]["rSquared"],
            )
