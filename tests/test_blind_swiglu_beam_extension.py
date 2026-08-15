#!/usr/bin/env python3
"""The dictionary beam's shortcut, checked against the solve it replaced.

`extend_beam_residuals` claims an identity: the residual of a design extended by one block equals
the residual of regressing the current residual on the part of that block orthogonal to the base
span. If that is true the beam can skip re-solving `atoms * rank + 1` columns for every candidate;
if it is even slightly false, every score the beam compares is wrong in a direction that is
invisible from the outside, because a wrong residual is still a plausible-looking number.

So these tests do not assert that the function returns something. They compare it, candidate by
candidate, against `np.linalg.lstsq` on the full extended design - the code that was there before -
and they cover the two ways the shortcut can quietly over-project: a rank-deficient base, and a new
block that adds nothing because it already lies in the base span.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The card is not required to check an identity, and pinning the host keeps the comparison to one
# implementation rather than two.
os.environ.setdefault("GCL_BLIND_SWIGLU_FIT_DEVICE", "cpu")

import numpy as np  # noqa: E402

from tools.llm_reader.blind_swiglu_oracle_decompiler import (  # noqa: E402
    extend_beam_residuals,
    orthonormal_span_basis,
    silu_np,
)

ROWS = 900
RANK = 6
POOL = 24


def _problem(seed: int):
    rng = np.random.default_rng(seed)
    xs = rng.normal(size=(ROWS, RANK))
    candidates = np.stack([rng.normal(size=RANK) for _ in range(POOL)])
    targets = rng.normal(size=ROWS)
    return xs, candidates, targets


def _block(xs: np.ndarray, gate: np.ndarray) -> np.ndarray:
    return silu_np(xs @ gate)[:, None] * xs


def _design(xs: np.ndarray, candidates: np.ndarray, selected: list[int]) -> np.ndarray:
    return np.column_stack(
        [_block(xs, candidates[index]) for index in selected] + [np.ones(xs.shape[0])]
    )


def _full_solve(xs, candidates, targets, selected: list[int]):
    design = _design(xs, candidates, selected)
    coefficients, *_ = np.linalg.lstsq(design, targets, rcond=None)
    residual = targets - design @ coefficients
    return residual, float(np.dot(residual, residual))


class BeamExtensionMatchesTheFullSolve(unittest.TestCase):
    def test_scores_and_residuals_match_at_several_base_sizes(self) -> None:
        xs, candidates, targets = _problem(seed=17)
        for base_size in (0, 1, 4, 9):
            selected = list(range(base_size))
            base_residual = (
                targets - float(np.mean(targets))
                if base_size == 0
                else _full_solve(xs, candidates, targets, selected)[0]
            )
            branch = list(range(12, 18))
            extended = extend_beam_residuals(
                xs, base_residual, _design(xs, candidates, selected), candidates, branch
            )
            self.assertEqual(
                [index for index, _score, _residual in extended], branch,
                "the beam relies on every branch candidate coming back, in order",
            )
            for index, score, residual in extended:
                reference_residual, reference_score = _full_solve(
                    xs, candidates, targets, selected + [index]
                )
                self.assertAlmostEqual(
                    score / reference_score, 1.0, places=10,
                    msg=f"base {base_size}, candidate {index}",
                )
                self.assertLess(
                    float(np.max(np.abs(residual - reference_residual))), 1e-9,
                    f"base {base_size}, candidate {index}",
                )

    def test_a_block_already_in_the_base_span_changes_nothing(self) -> None:
        """The over-projection trap, as a test rather than a comment.

        Re-selecting a gate the base already holds adds no column space, so the extended residual
        must equal the base residual exactly. Judged against its own largest singular value the
        rounding noise left by such a block looks full rank, and the score came back better than
        the truth - which in a beam means the duplicate wins.
        """
        xs, candidates, targets = _problem(seed=23)
        selected = [0, 1, 2]
        base_residual, base_score = _full_solve(xs, candidates, targets, selected)
        extended = extend_beam_residuals(
            xs, base_residual, _design(xs, candidates, selected), candidates, [1]
        )
        self.assertEqual(len(extended), 1)
        _index, score, residual = extended[0]
        self.assertAlmostEqual(score / base_score, 1.0, places=12)
        self.assertLess(float(np.max(np.abs(residual - base_residual))), 1e-9)

    def test_the_span_basis_reports_the_real_rank_of_a_deficient_design(self) -> None:
        """Reduced QR would return `columns` orthonormal vectors here regardless of rank."""
        rng = np.random.default_rng(5)
        independent = rng.normal(size=(ROWS, 4))
        design = np.column_stack([independent, independent[:, :2] @ rng.normal(size=(2, 3))])
        basis = orthonormal_span_basis(design)
        self.assertIsNotNone(basis)
        self.assertEqual(int(basis.shape[1]), 4, "expected the rank, not the column count")
        residual = design - basis.cpu().numpy() @ (basis.cpu().numpy().T @ design)
        self.assertLess(float(np.max(np.abs(residual))), 1e-9, "the basis must span the design")


if __name__ == "__main__":
    unittest.main(verbosity=2)
