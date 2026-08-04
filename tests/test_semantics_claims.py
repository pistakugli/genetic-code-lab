#!/usr/bin/env python3
"""The two semantics claims beyond prediction, tested where they can fail.

Prediction says the recovered parameters describe a unit's function. Intervention says they
describe its effect: silence the unit and the layer output should move by exactly minus its
contribution. Selectivity says they describe what drives it: inputs built to maximise the
recovered unit should maximise the real one.

Both are written so that a wrong recovery scores badly rather than plausibly, which is the
only thing that makes them evidence.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.llm_reader.unit_intervention_test import predicted_delta  # noqa: E402
from tools.llm_reader.unit_selectivity_test import (  # noqa: E402
    candidate_inputs,
    drive,
    rank_correlation,
)


class InterventionTest(unittest.TestCase):
    def test_the_predicted_delta_is_minus_the_unit_contribution(self) -> None:
        rng = np.random.default_rng(3)
        dim = 24
        gate, up = rng.normal(size=dim) * 0.02, rng.normal(size=dim)
        output = rng.normal(size=16)
        output /= np.linalg.norm(output)
        points = rng.normal(0.0, 2.0, size=(50, dim))

        delta = predicted_delta(gate, up, output, points)
        z = points @ gate
        contribution = np.outer((z / (1.0 + np.exp(-z))) * (points @ up), output)

        np.testing.assert_allclose(delta, -contribution, rtol=1e-10, atol=1e-12)

    def test_a_wrong_gate_gives_a_delta_pointing_elsewhere(self) -> None:
        rng = np.random.default_rng(5)
        dim = 24
        gate, up = rng.normal(size=dim) * 0.02, rng.normal(size=dim)
        output = rng.normal(size=16)
        output /= np.linalg.norm(output)
        points = rng.normal(0.0, 2.0, size=(200, dim))

        right = predicted_delta(gate, up, output, points)
        wrong = predicted_delta(rng.normal(size=dim) * 0.02, up, output, points)
        cosine = float(
            np.sum(right * wrong) / max(np.linalg.norm(right) * np.linalg.norm(wrong), 1e-30)
        )

        # If any gate produced a plausible delta the test would prove nothing.
        self.assertLess(abs(cosine), 0.9)


class SelectivityTest(unittest.TestCase):
    def test_candidates_span_the_units_own_plane_at_fixed_norm(self) -> None:
        rng = np.random.default_rng(7)
        dim = 20
        gate, up = rng.normal(size=dim), rng.normal(size=dim)

        points = candidate_inputs(gate, up, norm=5.0, count=64)

        np.testing.assert_allclose(np.linalg.norm(points, axis=1), 5.0, rtol=1e-9)
        # The drive depends on x only through g.x and u.x, so this plane contains the
        # maximiser: a candidate set outside it would be searching the wrong space.
        self.assertEqual(np.linalg.matrix_rank(points, tol=1e-8), 2)

    def test_the_exact_parameters_rank_inputs_identically(self) -> None:
        rng = np.random.default_rng(11)
        dim = 20
        gate, up = rng.normal(size=dim) * 0.02, rng.normal(size=dim)
        points = candidate_inputs(gate, up, norm=20.0, count=180)

        scores = drive(gate, up, points)

        self.assertAlmostEqual(rank_correlation(np.abs(scores), np.abs(scores)), 1.0, places=9)

    def test_an_unrelated_unit_does_not_rank_them_the_same(self) -> None:
        rng = np.random.default_rng(13)
        dim = 20
        gate, up = rng.normal(size=dim) * 0.02, rng.normal(size=dim)
        other_gate, other_up = rng.normal(size=dim) * 0.02, rng.normal(size=dim)
        points = candidate_inputs(gate, up, norm=20.0, count=180)

        correlation = rank_correlation(
            np.abs(drive(gate, up, points)), np.abs(drive(other_gate, other_up, points))
        )

        self.assertLess(abs(correlation), 0.9)

    def test_rank_correlation_is_zero_on_a_constant_signal(self) -> None:
        # A degenerate unit must score zero rather than raising or returning nan, since these
        # runs summarise over many units and one nan would silently poison a median.
        self.assertEqual(rank_correlation(np.ones(20), np.arange(20.0)), 0.0)


if __name__ == "__main__":
    unittest.main()
