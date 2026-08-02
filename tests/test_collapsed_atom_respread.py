#!/usr/bin/env python3
"""Two atoms on one direction must be pulled apart, not left to the surface loss.

Measured on the randomized LLaMA layer-15 block: recovered atoms 12 and 14 both sit at gate
cosine 0.62 to 0.70 against both of two source units, and the principal cosines between their
span and the sources' span are 0.9907 and 0.02265. One direction covered twice, the second
not represented at all, while R2 holds at 0.9986 because the missing direction is absorbed by
the other fourteen atoms. Nothing in the loss objects to that, so it has to be detected as
geometry and repaired by re-seeding.

`detect_swiglu_cancellation_pairs` does not see it: that filter requires a strongly negative
contribution correlation, and a collapsed pair can have either sign depending on how
least-squares last rescaled it.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.llm_reader.blind_swiglu_oracle_decompiler import (  # noqa: E402
    detect_collapsed_swiglu_atoms,
    predict_swiglu,
    regression_metrics,
    respread_collapsed_atoms_swiglu_candidate,
    rescore_swiglu_candidate_on_data,
    silu_np,
)

try:
    import torch

    CUDA_AVAILABLE = torch.cuda.is_available()
except Exception:  # pragma: no cover
    CUDA_AVAILABLE = False

THRESHOLDS = dict(gate_cosine_threshold=0.55, up_cosine_threshold=0.55, contribution_correlation_threshold=0.30)


def teacher(rank: int = 4, count: int = 4000, seed: int = 11):
    rng = np.random.default_rng(seed)
    gates = rng.normal(0.0, 1.0, size=(3, rank))
    ups = rng.normal(0.0, 1.0, size=(3, rank))
    coords = rng.normal(0.0, 0.6, size=(count, rank))
    targets = (silu_np(coords @ gates.T) * (coords @ ups.T)).sum(axis=1)
    return coords, targets, gates, ups, rng


def collapsed_candidate():
    coords, targets, gates, ups, rng = teacher()
    g, u = gates.copy(), ups.copy()
    # Atom 2 lands on atom 1's direction: the third source is left unrepresented.
    g[2] = g[1] * 1.02 + 0.02 * rng.normal(0.0, 1.0, size=g.shape[1])
    u[2] = u[1] * 0.5
    candidate = {"gates": g, "ups": u, "scales": np.ones(3), "bias": 0.0}
    return coords, targets, gates, candidate


def max_gate_cosine(direction: np.ndarray, truth: np.ndarray) -> tuple[float, int]:
    unit = direction / max(float(np.linalg.norm(direction)), 1e-12)
    cosines = [abs(float(unit @ (row / np.linalg.norm(row)))) for row in truth]
    return max(cosines), int(np.argmax(cosines))


class CollapsedAtomDetectionTest(unittest.TestCase):
    def test_a_collapsed_pair_is_detected(self) -> None:
        coords, _targets, _truth, candidate = collapsed_candidate()

        pairs = detect_collapsed_swiglu_atoms(coords, candidate, **THRESHOLDS)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(sorted(pairs[0]["atoms"]), [1, 2])

    def test_the_weaker_atom_of_the_pair_is_the_one_named(self) -> None:
        coords, _targets, _truth, candidate = collapsed_candidate()

        pair = detect_collapsed_swiglu_atoms(coords, candidate, **THRESHOLDS)[0]

        left, right = pair["meanAbsContribution"]
        expected = pair["atoms"][0] if left <= right else pair["atoms"][1]
        self.assertEqual(pair["weakerAtom"], expected)

    def test_detection_survives_the_rescaling_that_flips_the_correlation_sign(self) -> None:
        coords, targets, _truth, candidate = collapsed_candidate()
        rescaled = rescore_swiglu_candidate_on_data(coords, targets, candidate, recalibrate_scales=True)

        before = detect_collapsed_swiglu_atoms(coords, candidate, **THRESHOLDS)
        after = detect_collapsed_swiglu_atoms(coords, rescaled, **THRESHOLDS)

        # Least squares turned scales of 1.0 into -5.49 and 10.51 and the correlation from
        # +0.22 into -0.999, while gate and up cosines stayed at 1.000. A signed test loses
        # the pair here; the magnitude test keeps it.
        self.assertEqual(len(before), 1)
        self.assertEqual(len(after), 1)
        self.assertLess(after[0]["contributionCorrelation"], 0.0)
        self.assertGreater(after[0]["absContributionCorrelation"], 0.9)

    def test_well_spread_atoms_are_not_flagged(self) -> None:
        coords, _targets, truth, _candidate = collapsed_candidate()
        rng = np.random.default_rng(3)
        healthy = {
            "gates": truth,
            "ups": rng.normal(0.0, 1.0, size=truth.shape),
            "scales": np.ones(truth.shape[0]),
            "bias": 0.0,
        }

        self.assertEqual(detect_collapsed_swiglu_atoms(coords, healthy, **THRESHOLDS), [])


@unittest.skipUnless(CUDA_AVAILABLE, "respread requires CUDA")
class CollapsedAtomRespreadTest(unittest.TestCase):
    def setUp(self) -> None:
        self._previous = os.environ.get("GCL_BLIND_SWIGLU_RESPREAD_COLLAPSED_ATOMS")
        os.environ["GCL_BLIND_SWIGLU_RESPREAD_COLLAPSED_ATOMS"] = "1"

    def tearDown(self) -> None:
        if self._previous is None:
            os.environ.pop("GCL_BLIND_SWIGLU_RESPREAD_COLLAPSED_ATOMS", None)
        else:
            os.environ["GCL_BLIND_SWIGLU_RESPREAD_COLLAPSED_ATOMS"] = self._previous

    def test_the_re_seeded_atom_lands_on_the_source_nothing_covered(self) -> None:
        coords, targets, truth, candidate = collapsed_candidate()

        repaired = respread_collapsed_atoms_swiglu_candidate(coords, targets, candidate)

        self.assertEqual(repaired["collapsedAtomRespreadAtoms"], [2])
        cosine, source = max_gate_cosine(np.asarray(repaired["gates"])[2], truth)
        # Before the repair this atom sat at 0.9996 against source 1, the direction atom 1
        # already held.
        self.assertEqual(source, 2)
        self.assertGreater(cosine, 0.9)

    def test_the_pair_is_gone_and_the_fit_improves(self) -> None:
        coords, targets, _truth, candidate = collapsed_candidate()
        before = regression_metrics(targets, predict_swiglu(coords, candidate))

        repaired = respread_collapsed_atoms_swiglu_candidate(coords, targets, candidate)
        after = regression_metrics(targets, predict_swiglu(coords, repaired))

        self.assertEqual(repaired["collapsedPairsAfter"], 0)
        self.assertLess(after["mae"], before["mae"])
        self.assertLess(after["worstError"], before["worstError"])

    def test_a_healthy_candidate_is_returned_untouched(self) -> None:
        coords, targets, truth, _candidate = collapsed_candidate()
        rng = np.random.default_rng(3)
        healthy = {
            "gates": truth,
            "ups": rng.normal(0.0, 1.0, size=truth.shape),
            "scales": np.ones(truth.shape[0]),
            "bias": 0.0,
        }

        result = respread_collapsed_atoms_swiglu_candidate(coords, targets, healthy)

        # Nothing is changed, but the pass records that it looked. Returning the candidate
        # untouched left no way to tell "no collapsed pair here" from "this never ran", and
        # two live runs came back bit-identical with nothing in the artifact to say which.
        self.assertFalse(result["collapsedAtomsRespread"])
        self.assertEqual(result["collapsedPairsBefore"], 0)
        self.assertIsNotNone(result["collapsedClosestPair"])
        for field in ("gates", "ups", "scales", "bias"):
            np.testing.assert_array_equal(np.asarray(result[field]), np.asarray(healthy[field]))

    def test_the_pass_is_off_unless_asked_for(self) -> None:
        os.environ.pop("GCL_BLIND_SWIGLU_RESPREAD_COLLAPSED_ATOMS", None)
        coords, targets, _truth, candidate = collapsed_candidate()

        self.assertIs(respread_collapsed_atoms_swiglu_candidate(coords, targets, candidate), candidate)


if __name__ == "__main__":
    unittest.main()
