#!/usr/bin/env python3
"""The rank-1 jump at a boundary must return the unit, and must know when it has not.

The scalar recovery path reads boundaries as kinks in one number, and the share of
boundaries it reads correctly falls from 94% at 32 units to 17% at 2048, because SiLU's
transition width is fixed while boundary density grows with the unit count. A vector-valued
oracle changes the geometry: crossing unit i's boundary moves the Jacobian by
(u_i.x) * d_i (x) g_i, a rank-1 matrix, and one SVD returns both directions with no fitting.

These tests pin the two things that make it usable rather than merely true: removing the
other units' smooth drift before differencing, and the rank-1 share as a filter that works
without access to the teacher.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.llm_reader.jacobian_jump_probe import (  # noqa: E402
    find_peaks,
    make_teacher,
    probe_line,
)


def small_probe(**overrides):
    settings = dict(
        scale=2e4,
        samples=2001,
        step=1e-4,
        rank1_threshold=0.7,
        max_crossings=12,
        seed=11,
    )
    settings.update(overrides)
    teacher = make_teacher(units=96, input_dim=96, output_dim=96, seed=11)
    return probe_line(teacher, **settings)


class JacobianJumpProbeTest(unittest.TestCase):
    def test_an_accepted_crossing_returns_both_directions(self) -> None:
        report = small_probe()

        self.assertGreater(report["crossingsAccepted"], 0)
        self.assertGreater(report["acceptedGateCosineMedian"], 0.99)
        self.assertGreater(report["acceptedOutputCosineMedian"], 0.99)

    def test_the_rank1_filter_is_worth_its_cost(self) -> None:
        report = small_probe()

        # The filter uses only the measured jump, never the teacher, so this is the property
        # that lets the method run blind: accepted crossings are cleaner than the raw ones.
        self.assertGreaterEqual(
            report["acceptedGateCosineShareAbove099"],
            report["unfilteredGateCosineShareAbove099"],
        )

    def test_a_higher_threshold_never_lowers_accepted_quality(self) -> None:
        loose = small_probe(rank1_threshold=0.2)
        strict = small_probe(rank1_threshold=0.7)

        self.assertLessEqual(strict["crossingsAccepted"], loose["crossingsAccepted"])
        self.assertGreaterEqual(
            strict["acceptedGateCosineShareAbove099"],
            loose["acceptedGateCosineShareAbove099"],
        )

    def test_the_oracle_cost_is_reported_and_dominated_by_the_jacobians(self) -> None:
        report = small_probe()

        # Scanning is two calls a sample; each examined crossing costs four Jacobians of
        # 2*input_dim calls each. The cost model has to be visible, since the whole question
        # for a full layer is how many calls it takes.
        scan_cost = 2 * 2001
        self.assertGreater(report["oracleCalls"], scan_cost)
        self.assertEqual(
            report["oracleCalls"],
            scan_cost + report["crossingsExamined"] * 4 * 2 * report["inputDim"],
        )

    def test_peak_finding_keeps_one_index_per_neighbourhood(self) -> None:
        signal = np.zeros(50)
        signal[10] = 1.0
        signal[11] = 0.9
        signal[30] = 0.8

        found = find_peaks(signal, height=0.5, distance=5)

        self.assertEqual(list(found), [10, 30])

    def test_a_flat_signal_yields_no_peaks(self) -> None:
        self.assertEqual(list(find_peaks(np.ones(50), height=0.5, distance=3)), [])


if __name__ == "__main__":
    unittest.main()
