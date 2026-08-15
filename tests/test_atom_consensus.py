#!/usr/bin/env python3
"""The consensus score decides which atoms the solver is allowed to trust.

Nothing acts on it yet, but the plan for it is to silence the atoms it picks so the next fit sees
fewer atoms in a smaller subspace - and a silencing direction that is wrong does not fail loudly,
it removes the wrong part of the space and quietly corrupts everything fitted afterwards. So the
score has to be wrong in the safe direction: agreement must mean agreement, disagreement must show
up as a low number, and "nothing to compare against" must not read as "everything scored zero".

These check the three claims the score makes about itself: it is invariant to how the candidates
happen to order their atoms, it separates an atom the candidates agree on from one they do not,
and it reports absence as absence.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("GCL_BLIND_SWIGLU_FIT_DEVICE", "cpu")

import numpy as np  # noqa: E402

from tools.llm_reader.blind_swiglu_oracle_decompiler import (  # noqa: E402
    atom_consensus_scores,
    greedy_gate_assignment,
)

ATOMS = 6
RANK = 5


def _candidate(gates: np.ndarray) -> dict:
    return {"gates": gates}


class AtomConsensus(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(4)
        self.gates = rng.normal(size=(ATOMS, RANK))
        self.best = _candidate(self.gates)

    def test_identical_candidates_agree_completely(self) -> None:
        scores = atom_consensus_scores(self.best, [self.best, _candidate(self.gates.copy())])
        assert scores is not None
        for value in scores:
            self.assertAlmostEqual(value, 1.0, places=12)

    def test_agreement_survives_a_reordered_candidate(self) -> None:
        """Restarts have no shared atom order, so a score that depended on one would be noise."""
        permutation = np.array([3, 0, 5, 1, 4, 2])
        scores = atom_consensus_scores(
            self.best, [self.best, _candidate(self.gates[permutation] * 2.5)]
        )
        assert scores is not None
        for value in scores:
            self.assertAlmostEqual(value, 1.0, places=12)

    def test_a_disputed_atom_scores_below_the_agreed_ones(self) -> None:
        rng = np.random.default_rng(11)
        disputed = 2
        others = []
        for index in range(4):
            gates = self.gates.copy()
            gates[disputed] = rng.normal(size=RANK)
            others.append(_candidate(gates))
        scores = atom_consensus_scores(self.best, [self.best, *others])
        assert scores is not None
        agreed = [value for position, value in enumerate(scores) if position != disputed]
        self.assertLess(scores[disputed], 0.9)
        self.assertGreater(min(agreed), 0.99)
        self.assertLess(scores[disputed], min(agreed))

    def test_nothing_to_compare_against_is_reported_as_absent(self) -> None:
        """An empty list would be read downstream as every atom scoring zero."""
        self.assertIsNone(atom_consensus_scores(self.best, [self.best]))
        self.assertIsNone(atom_consensus_scores(self.best, []))

    def test_a_candidate_of_the_wrong_shape_is_ignored_rather_than_crashing(self) -> None:
        wrong = _candidate(np.zeros((ATOMS + 1, RANK)))
        self.assertIsNone(atom_consensus_scores(self.best, [self.best, wrong]))

    def test_the_assignment_is_a_permutation(self) -> None:
        """A repeated column would score one candidate atom twice and drop another."""
        rng = np.random.default_rng(7)
        similarity = np.abs(rng.normal(size=(ATOMS, ATOMS)))
        assignment = greedy_gate_assignment(similarity)
        self.assertEqual(sorted(assignment), list(range(ATOMS)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
