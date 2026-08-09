#!/usr/bin/env python3
"""A structure finder that always returns something cannot be wrong, and is therefore useless.

`docs/neural-decompiler-formalization.md` makes negative cases mandatory and names the outcome
that must be a hard failure: "fails by inventing unsupported structure". Each test below pins
one required family from that document's table, and each asserts the honest outcome -
recovered, canonicalised to a smaller structure, or non-identifiable - rather than a structure
the evidence does not support.

These are cheap by design. They run on the verdict function directly, with no model and no
oracle, because a check that only runs during an expensive live recovery is a check that stops
being run.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.llm_reader.jacobian_boundary_recovery import identifiability_verdict  # noqa: E402


def verdict(**overrides):
    settings = dict(
        basis_spectrum=np.concatenate([np.ones(4), np.full(4, 1e-3)]),
        active_rank=4,
        gates=np.eye(4),
        output_directions=np.eye(4),
        clusters_found=4,
        duplicate_gate_cosine=0.98,
        collinear_threshold=0.9,
        weak_share_threshold=0.002,
        rank_energy_threshold=0.98,
        unit_budget=4,
    )
    settings.update(overrides)
    return identifiability_verdict(**settings)


def families(result) -> set[str]:
    return {row["family"] for row in result["findings"]}


class IdentifiabilityVerdictTest(unittest.TestCase):
    def test_a_clean_recovery_is_allowed_to_claim_structure(self) -> None:
        result = verdict()

        self.assertEqual(result["verdict"], "recovered")
        self.assertTrue(result["structureClaimAllowed"])
        self.assertEqual(result["findings"], [])

    def test_insufficient_rank_blocks_the_claim(self) -> None:
        # A rank that does not capture the Jacobians' energy means the block is not a sum of
        # that many units, so everything downstream is fitted to what the basis missed.
        result = verdict(basis_spectrum=np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.9]), active_rank=2)

        self.assertEqual(result["verdict"], "non-identifiable")
        self.assertFalse(result["structureClaimAllowed"])
        self.assertIn("insufficient-rank-or-wrong-activation-family", families(result))

    def test_an_overcomplete_budget_blocks_the_claim(self) -> None:
        # More solutions than the block can hold is the invented-structure failure itself.
        result = verdict(unit_budget=2)

        self.assertEqual(result["verdict"], "non-identifiable")
        self.assertFalse(result["structureClaimAllowed"])
        self.assertIn("overcomplete-budget", families(result))

    def test_a_duplicate_split_is_reported_as_canonicalised_not_as_two_units(self) -> None:
        gates = np.array([[1.0, 0.0, 0.0], [0.999, 0.001, 0.0], [0.0, 1.0, 0.0]])
        result = verdict(gates=gates, output_directions=np.eye(3), basis_spectrum=np.concatenate([np.ones(3), np.full(3, 1e-3)]), active_rank=3, unit_budget=3)

        self.assertIn("exact-duplicate-split", families(result))
        # Duplicates collapse to a smaller structure; they do not invalidate the recovery.
        self.assertEqual(result["verdict"], "qualified")
        self.assertTrue(result["structureClaimAllowed"])

    def test_near_collinear_directions_are_reported_as_ambiguous(self) -> None:
        gates = np.array([[1.0, 0.0, 0.0], [0.95, 0.312, 0.0], [0.0, 0.0, 1.0]])
        result = verdict(gates=gates, output_directions=np.eye(3), basis_spectrum=np.concatenate([np.ones(3), np.full(3, 1e-3)]), active_rank=3, unit_budget=3)

        self.assertIn("near-collinear-directions", families(result))
        self.assertEqual(
            [row["verdict"] for row in result["findings"] if row["family"] == "near-collinear-directions"],
            ["ambiguous"],
        )

    def test_anti_parallel_output_directions_are_flagged_as_cancellation(self) -> None:
        outputs = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        result = verdict(gates=np.eye(3), output_directions=outputs, basis_spectrum=np.concatenate([np.ones(3), np.full(3, 1e-3)]), active_rank=3, unit_budget=3)

        # Two units whose outputs cancel in the sum cannot have their split observed from
        # output access, whatever the gates look like.
        self.assertIn("equal-opposite-cancellation", families(result))

    def test_a_channel_below_the_energy_floor_is_reported_not_counted(self) -> None:
        gates = np.array([[1.0, 0.0], [1.0, 0.0], [1e-6, 0.0]])
        result = verdict(gates=gates, output_directions=np.eye(3)[:, :2], basis_spectrum=np.concatenate([np.ones(2), np.full(2, 1e-3)]), active_rank=2, unit_budget=3)

        self.assertIn("weak-channel-below-floor", families(result))

    def test_norm_diagnostics_are_counted_and_reported(self) -> None:
        # `silu` is not scale free, so a unit reported at the wrong scale is a claim about
        # behaviour the evidence does not carry, and both ways that happens are measured: a
        # knee that never appeared in the probed range, and independent rays that disagree
        # about where it sits.
        result = verdict(norm_diagnostics=[
            {"norm": 1.0, "kneeShare": 1.0, "raySpread": 0.001},
            {"norm": 0.5, "kneeShare": 0.02, "raySpread": 0.001},
            {"norm": 2.0, "kneeShare": 1.0, "raySpread": 0.9},
            {"norm": 1.2, "kneeShare": 1.0, "raySpread": 0.001},
        ])

        counts = result["gateNormDiagnosticCounts"]
        self.assertEqual(counts["kneeShareBelowThreshold"], 1)
        self.assertEqual(counts["raySpreadAboveThreshold"], 1)

    def test_the_counts_do_not_qualify_the_recovery(self) -> None:
        # The two cutoffs were invented rather than measured, in a file where I had written
        # that a number with an invented cutoff is worse than a number. Reporting the count is
        # honest; letting it qualify a recovery is not, until a broken run and a clean one are
        # shown to separate on it.
        noisy = verdict(norm_diagnostics=[
            {"norm": 1.0, "kneeShare": 0.0, "raySpread": 9.0} for _ in range(4)
        ])

        self.assertEqual(noisy["verdict"], "recovered")
        self.assertEqual(noisy["findings"], [])
        self.assertEqual(noisy["gateNormDiagnosticCounts"]["kneeShareBelowThreshold"], 4)

    def test_every_finding_carries_one_of_the_three_honest_outcomes(self) -> None:
        allowed = {"non-identifiable", "canonicalised", "ambiguous"}
        gates = np.array([[1.0, 0.0, 0.0], [0.999, 0.001, 0.0], [1e-7, 0.0, 0.0]])
        result = verdict(gates=gates, output_directions=np.eye(3), basis_spectrum=np.array([5.0, 1.0, 0.5]), active_rank=1, unit_budget=1)

        self.assertTrue(result["findings"])
        for row in result["findings"]:
            self.assertIn(row["verdict"], allowed)
            # A finding with no detail cannot be acted on, which makes it decoration.
            self.assertTrue(row["detail"])


if __name__ == "__main__":
    unittest.main()
