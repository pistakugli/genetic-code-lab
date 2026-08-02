#!/usr/bin/env python3
"""Per-source amplitude has to be readable from the artifact, not inferred from pair rows.

Two of sixteen atoms in the randomized LLaMA layer-15 block pair with their source unit at
0.24 and 0.30 while the other fourteen clear 0.99. The first thing to check is whether those
units contribute enough to the sum for the fit to have any reason to resolve them - and the
cancellation audit published mean absolute contribution only for units appearing in its eight
most negatively correlated pairs, so for one of the two failing units it was simply absent.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.llm_reader.blind_swiglu_oracle_decompiler import (  # noqa: E402
    audit_swiglu_source_cancellation,
)


def audit(up_scales: list[float], seed: int = 5, rank: int = 4, count: int = 2000):
    rng = np.random.default_rng(seed)
    coords = rng.normal(0.0, 0.6, size=(count, rank))
    gates = rng.normal(0.0, 1.0, size=(len(up_scales), rank))
    ups = rng.normal(0.0, 1.0, size=(len(up_scales), rank))
    for index, scale in enumerate(up_scales):
        ups[index] *= scale
    units = [10 * (index + 1) for index in range(len(up_scales))]
    return audit_swiglu_source_cancellation(coords, gates, ups, units)


class SourceContributionAuditTest(unittest.TestCase):
    def test_every_source_is_reported_not_only_those_in_negative_pairs(self) -> None:
        result = audit([1.0, 1.0, 1.0, 1.0, 1.0])

        self.assertEqual(len(result["perSourceContribution"]), 5)
        self.assertEqual(
            [row["sourceUnit"] for row in result["perSourceContribution"]],
            [10, 20, 30, 40, 50],
        )

    def test_a_weak_unit_is_visible_as_a_small_share_of_the_gross(self) -> None:
        result = audit([1.0, 1.0, 0.05])
        weak = result["perSourceContribution"][2]

        self.assertLess(weak["shareOfGross"], 0.05)
        self.assertGreater(result["sourceAmplitudeRatio"], 10.0)

    def test_shares_are_measured_against_the_gross_sum(self) -> None:
        result = audit([1.0, 1.0, 1.0])
        total = sum(row["shareOfGross"] for row in result["perSourceContribution"])

        # Gross is the mean of the summed absolute contributions, so the shares partition it.
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_scaling_one_unit_down_widens_the_amplitude_ratio(self) -> None:
        # Random gate/up directions already spread the contributions by several times over,
        # so the ratio is only meaningful as a comparison: weakening one unit has to move it.
        even = audit([1.0, 1.0, 1.0, 1.0])["sourceAmplitudeRatio"]
        uneven = audit([1.0, 1.0, 1.0, 0.05])["sourceAmplitudeRatio"]

        self.assertGreaterEqual(even, 1.0)
        self.assertGreater(uneven, even * 3.0)

    def test_the_extremes_match_the_per_source_rows(self) -> None:
        result = audit([1.0, 0.4, 0.05, 2.0])
        values = [row["meanAbsContribution"] for row in result["perSourceContribution"]]

        self.assertAlmostEqual(result["minSourceMeanAbsContribution"], min(values), places=12)
        self.assertAlmostEqual(result["maxSourceMeanAbsContribution"], max(values), places=12)

    def test_the_existing_cancellation_fields_are_unchanged(self) -> None:
        result = audit([1.0, 1.0, 1.0])

        for field in ("meanGrossContribution", "meanNetContribution", "cancellationFraction", "mostNegativePairs"):
            self.assertIn(field, result)


if __name__ == "__main__":
    unittest.main()
