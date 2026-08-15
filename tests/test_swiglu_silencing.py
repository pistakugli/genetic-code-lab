#!/usr/bin/env python3
"""The identity the silencing loop is built on, checked instead of assumed.

Everything in `swiglu_slice_stitch_probe.py` past the plain stitch rests on one claim: on a
subspace orthogonal to `gate_i`, atom i contributes exactly nothing, because `silu(0) = 0`. That is
what lets the loop freeze an atom and then fit the remainder in a smaller rank without subtracting
anything, and it is what makes a wrong silencing direction dangerous - the atom does not disappear,
it leaks back in at a size nobody is watching.

So this measures the leak rather than trusting the algebra. It also checks the two supporting
pieces the same way: that the complement really is the complement including when the directions
handed to it are dependent, and that a slice of the complement is orthogonal to every silenced
gate rather than to most of them.
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
os.environ.setdefault("GCL_BLIND_SWIGLU_STRUCTURED_STARTS", "0")

import numpy as np  # noqa: E402

from tools.llm_reader.swiglu_slice_stitch_probe import (  # noqa: E402
    SliceOracle,
    orthogonal_complement,
)

ATOMS = 10
RANK = 24
SILENCED = 4


class SilencingIsExact(unittest.TestCase):
    def setUp(self) -> None:
        self.oracle = SliceOracle(ATOMS, RANK, seed=13)
        self.gates, self.ups = self.oracle.truth()
        self.rng = np.random.default_rng(5)

    def _evaluate_subset(self, points: np.ndarray, indices) -> np.ndarray:
        z = points @ self.gates[list(indices)].T
        return ((z / (1.0 + np.exp(-z))) * (points @ self.ups[list(indices)].T)).sum(axis=1)

    def test_silenced_atoms_contribute_nothing_on_the_complement(self) -> None:
        """The whole architecture in one assertion."""
        basis = orthogonal_complement(self.gates[:SILENCED], RANK)
        points = self.rng.normal(size=(600, int(basis.shape[1]))) @ basis.T
        full = self.oracle.evaluate(points)
        kept = self._evaluate_subset(points, range(SILENCED, ATOMS))
        scale = float(np.std(full)) or 1.0
        self.assertLess(float(np.max(np.abs(full - kept))) / scale, 1e-12)

    def test_the_leak_is_visible_when_a_direction_is_wrong(self) -> None:
        """A test that only ever passes proves nothing, so this is the same check made to fail.

        Silencing a direction that is not the gate leaves the atom in the slice, and the point of
        the first test is that the difference is zero rather than merely small.
        """
        wrong = self.gates[:SILENCED].copy()
        wrong[0] = self.rng.normal(size=RANK)
        basis = orthogonal_complement(wrong, RANK)
        points = self.rng.normal(size=(600, int(basis.shape[1]))) @ basis.T
        full = self.oracle.evaluate(points)
        kept = self._evaluate_subset(points, range(SILENCED, ATOMS))
        scale = float(np.std(full)) or 1.0
        self.assertGreater(float(np.max(np.abs(full - kept))) / scale, 1e-3)

    def test_the_complement_is_orthogonal_to_every_silenced_gate(self) -> None:
        basis = orthogonal_complement(self.gates[:SILENCED], RANK)
        self.assertEqual(int(basis.shape[1]), RANK - SILENCED)
        self.assertLess(float(np.max(np.abs(self.gates[:SILENCED] @ basis))), 1e-12)

    def test_dependent_directions_do_not_eat_extra_dimensions(self) -> None:
        """Two frozen atoms can end up nearly parallel, and the complement must not shrink twice."""
        directions = self.gates[:3].copy()
        directions[2] = directions[0] * -2.0
        basis = orthogonal_complement(directions, RANK)
        self.assertEqual(int(basis.shape[1]), RANK - 2)
        self.assertLess(float(np.max(np.abs(directions @ basis))), 1e-12)

    def test_the_basis_is_orthonormal(self) -> None:
        """A slice built on a non-orthonormal basis is a different model, not a restriction."""
        basis = orthogonal_complement(self.gates[:SILENCED], RANK)
        gram = basis.T @ basis
        self.assertLess(float(np.max(np.abs(gram - np.eye(gram.shape[0])))), 1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
