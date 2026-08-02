#!/usr/bin/env python3
"""An atom that failed to pair has to say what it is aligned to instead.

The Hungarian assignment reports one number per atom. Two different failures produce the same
number and need opposite repairs: an atom spread thinly over several units has resolved
nothing, while two atoms that each hold part of the other's unit have recovered the right
subspace and only mixed the split inside it. Measured on the randomized LLaMA layer-15 block,
atoms 12 and 14 sit at 0.30 and 0.24 against units 3627 and 3679, and per-source amplitude
ruled out the weak-atom explanation: unit 778 has all but the same contribution and pairs at
0.9959.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.llm_reader.blind_swiglu_oracle_decompiler import diagnose_unpaired_atoms  # noqa: E402


def orthonormal(count: int, rank: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.linalg.qr(rng.normal(0.0, 1.0, size=(rank, rank)))[0][:count]


class UnpairedAtomDiagnosticTest(unittest.TestCase):
    def test_a_fully_paired_run_reports_nothing(self) -> None:
        truth = orthonormal(3, 3, 4)
        joint = truth @ truth.T

        result = diagnose_unpaired_atoms(joint, joint, joint, np.array([0, 1, 2]), [1, 2, 3], truth, truth)

        self.assertEqual(result["unpairedAtomCount"], 0)
        self.assertEqual(result["atoms"], [])
        self.assertIsNone(result["confusedSubspace"])

    def test_two_rotated_atoms_are_reported_as_span_recovered_but_unresolved(self) -> None:
        truth = orthonormal(4, 4, 2)
        mixed = truth.copy()
        angle = np.pi / 4
        mixed[0] = np.cos(angle) * truth[0] + np.sin(angle) * truth[1]
        mixed[1] = -np.sin(angle) * truth[0] + np.cos(angle) * truth[1]
        joint = mixed @ truth.T

        result = diagnose_unpaired_atoms(joint, joint, joint, np.array([0, 1, 2, 3]), [100, 200, 300, 400], mixed, truth)

        self.assertEqual(result["unpairedAtomCount"], 2)
        subspace = result["confusedSubspace"]
        self.assertAlmostEqual(subspace["minPrincipalCosine"], 1.0, places=9)
        # The atoms cover the right two directions; only the split between them is wrong.
        self.assertTrue(subspace["spanRecoveredButUnresolved"])

    def test_a_genuine_miss_is_not_reported_as_a_rotation(self) -> None:
        truth = orthonormal(4, 4, 7)
        recovered = truth.copy()
        # Point two atoms somewhere the sources do not span at all.
        stray = orthonormal(2, 4, 11)
        recovered[0], recovered[1] = stray[0], stray[1]
        joint = recovered @ truth.T

        result = diagnose_unpaired_atoms(joint, joint, joint, np.array([0, 1, 2, 3]), [100, 200, 300, 400], recovered, truth)

        subspace = result["confusedSubspace"]
        self.assertLess(subspace["minPrincipalCosine"], 0.9)
        self.assertFalse(subspace["spanRecoveredButUnresolved"])

    def test_an_atom_names_the_source_it_actually_matches(self) -> None:
        truth = orthonormal(3, 3, 5)
        recovered = truth.copy()
        recovered[0], recovered[1] = truth[1].copy(), truth[0].copy()
        joint = recovered @ truth.T

        # Assignment forced to the wrong pairing, which is what the diagnostic has to expose.
        result = diagnose_unpaired_atoms(joint, joint, joint, np.array([0, 1, 2]), [10, 20, 30], recovered, truth)

        first = next(row for row in result["atoms"] if row["recoveredAtom"] == 0)
        self.assertEqual(first["assignedSourceUnit"], 10)
        self.assertEqual(first["bestAlternatives"][0]["sourceUnit"], 20)
        self.assertFalse(first["bestAlternatives"][0]["isAssigned"])

    def test_a_single_unpaired_atom_reports_no_subspace(self) -> None:
        truth = orthonormal(3, 3, 9)
        recovered = truth.copy()
        recovered[0] = orthonormal(1, 3, 13)[0]
        joint = recovered @ truth.T

        result = diagnose_unpaired_atoms(joint, joint, joint, np.array([0, 1, 2]), [10, 20, 30], recovered, truth)

        # One atom does not span anything to compare, so the field stays absent rather than
        # being filled with a one-dimensional number that would read like agreement.
        self.assertEqual(result["unpairedAtomCount"], 1)
        self.assertIsNone(result["confusedSubspace"])

    def test_the_threshold_decides_which_atoms_are_examined(self) -> None:
        truth = orthonormal(3, 3, 3)
        recovered = truth.copy()
        recovered[0] = 0.995 * truth[0] + np.sqrt(1 - 0.995 ** 2) * truth[1]
        joint = recovered @ truth.T

        self.assertEqual(diagnose_unpaired_atoms(joint, joint, joint, np.array([0, 1, 2]), [1, 2, 3], recovered, truth, threshold=0.99)["unpairedAtomCount"], 0)
        self.assertEqual(diagnose_unpaired_atoms(joint, joint, joint, np.array([0, 1, 2]), [1, 2, 3], recovered, truth, threshold=0.999)["unpairedAtomCount"], 1)


if __name__ == "__main__":
    unittest.main()
