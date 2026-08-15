#!/usr/bin/env python3
"""Which way round the recovered basis goes, as a check rather than a habit.

`recover_basis` returns shape `(rank, hidden)`, and nothing in its signature says so. Getting it
backwards does not raise: a probe point multiplied by the wrong side still produces a matrix of
plausible numbers, the oracle still answers, the fit still converges to something, and the first
symptom is a recovered basis reported as rank 3072 - which is only noticed if somebody is reading
that field. It cost a live run to find.

So the orientation is pinned here, together with the two operations built on it: sending a point
out as `coords @ basis`, and projecting the post-hoc truth back with `weights @ basis.T`. Both are
checked against what the subspace is supposed to do rather than against a remembered convention -
a point sent out and projected back must return unchanged when it started inside the subspace.
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

from tools.llm_reader.blind_swiglu_oracle_decompiler import recover_basis  # noqa: E402

HIDDEN = 96
RANK = 8
WITNESSES = 40


def _gradients(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Witnesses that genuinely live in a rank-`RANK` subspace of `HIDDEN` dimensions."""
    rng = np.random.default_rng(seed)
    true_basis = np.linalg.qr(rng.normal(size=(HIDDEN, RANK)))[0]
    coefficients = rng.normal(size=(WITNESSES, RANK))
    return coefficients @ true_basis.T, true_basis


class ActiveBasisOrientation(unittest.TestCase):
    def setUp(self) -> None:
        self.gradients, self.true_basis = _gradients(seed=21)
        self.basis, _values, self.energy_ratio = recover_basis(self.gradients, RANK, "raw")

    def test_the_basis_is_rank_by_hidden(self) -> None:
        self.assertEqual(self.basis.shape, (RANK, HIDDEN))

    def test_it_found_the_subspace_the_witnesses_live_in(self) -> None:
        """Without this the shape test would pass on any orthonormal set of the right size."""
        self.assertGreater(self.energy_ratio, 1.0 - 1e-9)
        principal = np.linalg.svd(self.basis @ self.true_basis, compute_uv=False)
        self.assertGreater(float(np.min(principal)), 1.0 - 1e-9)

    def test_the_rows_are_orthonormal(self) -> None:
        gram = self.basis @ self.basis.T
        self.assertLess(float(np.max(np.abs(gram - np.eye(RANK)))), 1e-9)

    def test_a_point_survives_the_round_trip_through_the_subspace(self) -> None:
        """`coords @ basis` goes out to the model, `hidden @ basis.T` comes back."""
        rng = np.random.default_rng(3)
        coords = rng.normal(size=(50, RANK))
        hidden = coords @ self.basis
        self.assertEqual(hidden.shape, (50, HIDDEN))
        self.assertLess(float(np.max(np.abs(hidden @ self.basis.T - coords))), 1e-9)

    def test_weights_project_into_the_subspace_with_the_transpose(self) -> None:
        """The post-hoc comparison: source rows are `(units, hidden)` and must land `(units, rank)`."""
        rng = np.random.default_rng(4)
        units = 5
        inside = rng.normal(size=(units, RANK)) @ self.basis
        projected = inside @ self.basis.T
        self.assertEqual(projected.shape, (units, RANK))
        self.assertLess(float(np.max(np.abs(projected @ self.basis - inside))), 1e-9)

    def test_the_wrong_side_does_not_merely_fail_loudly(self) -> None:
        """Why this file exists: the mistake produces a valid array, not an exception."""
        rng = np.random.default_rng(6)
        coords = rng.normal(size=(50, RANK))
        with self.assertRaises(ValueError):
            _ = coords @ self.basis.T
        hidden_shaped = rng.normal(size=(50, HIDDEN))
        wrong = hidden_shaped @ self.basis.T
        self.assertEqual(wrong.shape, (50, RANK))


if __name__ == "__main__":
    unittest.main(verbosity=2)
