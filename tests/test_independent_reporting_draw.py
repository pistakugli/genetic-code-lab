#!/usr/bin/env python3
"""A repair may not be scored on the draw that decided to keep it.

Four validation-repair stages accepted a candidate when a fresh oracle draw cleared the
thresholds, then published that same draw's numbers. Evaluation is not on training data, so
this is not leakage, but it is a max-of-two selection: the figure is reported only because
it looked good, which biases it upward and biases worstError hardest, being a maximum over
thousands of samples.

`independent_reporting_metrics` draws again, with a different seed, after the decision is
made. When the query budget cannot cover that draw it returns None, and the caller declines
the repair rather than publishing a number selected on itself.
"""
from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.llm_reader.blind_swiglu_oracle_decompiler import (  # noqa: E402
    independent_reporting_metrics,
    silu_np,
)


class RecordingOracle:
    """Counts queries and records the stage each batch was charged to."""

    def __init__(self, gates: np.ndarray, ups: np.ndarray, scales: np.ndarray, basis: np.ndarray):
        self.gates, self.ups, self.scales, self.basis = gates, ups, scales, basis
        self.query_count = 0
        self.stages: list[str] = []
        self.seen: list[np.ndarray] = []

    def eval(self, inputs: np.ndarray, *, stage: str = "oracle-eval") -> np.ndarray:
        self.query_count += int(inputs.shape[0])
        self.stages.append(stage)
        self.seen.append(np.asarray(inputs, dtype=np.float64).copy())
        coords = np.asarray(inputs, dtype=np.float64) @ self.basis.T
        return (silu_np(coords @ self.gates.T) * (coords @ self.ups.T)) @ self.scales


def fixture(max_oracle_calls: int = 10_000, validation_count: int = 256):
    rank, dimension = 2, 6
    rng = np.random.default_rng(3)
    basis = np.linalg.qr(rng.normal(0.0, 1.0, size=(dimension, rank)))[0].T
    gates = rng.normal(0.0, 1.0, size=(2, rank))
    ups = rng.normal(0.0, 1.0, size=(2, rank))
    scales = np.array([1.0, -0.5])
    oracle = RecordingOracle(gates, ups, scales, basis)
    args = argparse.Namespace(
        seed=17,
        validation_count=validation_count,
        validation_scale=0.6,
        max_oracle_calls=max_oracle_calls,
    )
    candidate = {"gates": gates, "ups": ups, "scales": scales, "bias": 0.0}
    return oracle, args, basis, dimension, candidate


class IndependentReportingDrawTest(unittest.TestCase):
    def test_the_reporting_draw_is_charged_to_the_budget_and_named(self) -> None:
        oracle, args, basis, dimension, candidate = fixture()

        metrics = independent_reporting_metrics(
            oracle, args, basis, dimension, candidate, seed_offset=70428, stage="a-stage"
        )

        self.assertIsNotNone(metrics)
        self.assertEqual(oracle.query_count, 256)
        self.assertEqual(oracle.stages, ["a-stage"])

    def test_a_different_seed_offset_draws_different_points(self) -> None:
        oracle, args, basis, dimension, candidate = fixture()

        independent_reporting_metrics(oracle, args, basis, dimension, candidate, seed_offset=70427, stage="selection")
        independent_reporting_metrics(oracle, args, basis, dimension, candidate, seed_offset=70428, stage="reporting")

        # The whole point: the second draw shares no point with the one that decided.
        self.assertFalse(np.allclose(oracle.seen[0], oracle.seen[1]))

    def test_an_exhausted_budget_returns_none_instead_of_a_biased_number(self) -> None:
        oracle, args, basis, dimension, candidate = fixture(max_oracle_calls=100, validation_count=256)

        metrics = independent_reporting_metrics(
            oracle, args, basis, dimension, candidate, seed_offset=70428, stage="a-stage"
        )

        self.assertIsNone(metrics)
        # Refusing must be free: a declined repair may not silently spend the budget.
        self.assertEqual(oracle.query_count, 0)

    def test_a_budget_that_exactly_fits_is_allowed(self) -> None:
        oracle, args, basis, dimension, candidate = fixture(max_oracle_calls=256, validation_count=256)

        self.assertIsNotNone(
            independent_reporting_metrics(
                oracle, args, basis, dimension, candidate, seed_offset=70428, stage="a-stage"
            )
        )

    def test_an_exact_candidate_scores_exactly_on_the_fresh_draw(self) -> None:
        oracle, args, basis, dimension, candidate = fixture()

        metrics = independent_reporting_metrics(
            oracle, args, basis, dimension, candidate, seed_offset=70428, stage="a-stage"
        )

        assert metrics is not None
        self.assertAlmostEqual(metrics["mae"], 0.0, places=12)
        self.assertAlmostEqual(metrics["worstError"], 0.0, places=12)

    def test_the_draw_is_reproducible_for_a_given_seed(self) -> None:
        first, args, basis, dimension, candidate = fixture()
        second, args2, basis2, dimension2, candidate2 = fixture()

        independent_reporting_metrics(first, args, basis, dimension, candidate, seed_offset=70428, stage="s")
        independent_reporting_metrics(second, args2, basis2, dimension2, candidate2, seed_offset=70428, stage="s")

        np.testing.assert_allclose(first.seen[0], second.seen[0])


if __name__ == "__main__":
    unittest.main()
