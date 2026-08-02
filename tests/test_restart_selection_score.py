#!/usr/bin/env python3
"""Restart selection must rank on a finite score, or say it cannot rank at all.

`cubic_relative_rmse` averaged over the cubic witness arrays after checking only that
they were not None. With cubic witnesses disabled the probe returns empty arrays, not
None, so the mean was taken over nothing and every candidate's selectionScore became
NaN. Ranking used a bare `<`, and every comparison against NaN is False, so `best` kept
whichever candidate was evaluated first and a 530-restart search selected restart 0.

Measured on the randomized LLaMA 16-unit block that this hid in: train MAE 0.030 instead
of 0.005, and 0 of 16 recovered atoms matching a source unit instead of 16 of 16.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.llm_reader.blind_swiglu_oracle_decompiler import (  # noqa: E402
    fit_swiglu_sum_batched,
    paired_atom_identity_passes,
    silu_np,
)

try:
    import torch

    CUDA_AVAILABLE = torch.cuda.is_available()
except Exception:  # pragma: no cover - torch is a hard dependency of the solver
    CUDA_AVAILABLE = False


def swiglu_sum(coords: np.ndarray, gates: np.ndarray, ups: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return (silu_np(coords @ gates.T) * (coords @ ups.T)) @ scales


def teacher(rank: int = 3, atom_count: int = 2, count: int = 512, seed: int = 11):
    rng = np.random.default_rng(seed)
    gates = rng.normal(0.0, 1.0, size=(atom_count, rank))
    ups = rng.normal(0.0, 1.0, size=(atom_count, rank))
    scales = np.array([1.0, -0.7][:atom_count], dtype=np.float64)
    coords = rng.normal(0.0, 0.6, size=(count, rank))
    return coords, swiglu_sum(coords, gates, ups, scales)


@unittest.skipUnless(CUDA_AVAILABLE, "batched SwiGLU fit requires CUDA")
class RestartSelectionScoreTest(unittest.TestCase):
    def _fit(self, **overrides):
        coords, targets = teacher()
        kwargs = dict(
            coords=coords,
            targets=targets,
            atom_count=2,
            seed=7,
            steps=40,
            learning_rate=0.05,
            restart_count=8,
            structured_starts=[],
            gradient_coords=None,
            gradient_targets=None,
            gradient_weight=0.0,
            restart_chunk_size=0,
        )
        kwargs.update(overrides)
        return fit_swiglu_sum_batched(**kwargs)

    def test_empty_cubic_witness_leaves_the_selection_score_finite(self) -> None:
        fit = self._fit(
            cubic_directions=np.zeros((0, 3), dtype=np.float64),
            cubic_targets=np.zeros(0, dtype=np.float64),
        )

        self.assertIsNone(fit["cubicRelativeRmse"])
        self.assertTrue(math.isfinite(float(fit["selectionScore"])))
        for kind, value in fit["bestScoreByStartKind"].items():
            self.assertTrue(math.isfinite(float(value)), f"{kind} start kind scored non-finite")

    def test_absent_and_empty_cubic_witnesses_select_the_same_restart(self) -> None:
        absent = self._fit()
        empty = self._fit(
            cubic_directions=np.zeros((0, 3), dtype=np.float64),
            cubic_targets=np.zeros(0, dtype=np.float64),
        )

        # An empty witness array carries the same information as no witness at all, so it
        # must not change which restart wins. Before the fix the empty case ranked on NaN
        # and kept restart 0 whatever the search found.
        self.assertEqual(absent["restart"], empty["restart"])
        self.assertAlmostEqual(float(absent["selectionScore"]), float(empty["selectionScore"]), places=9)

    def test_selection_beats_the_first_evaluated_restart(self) -> None:
        fit = self._fit()

        # Ranking that silently degrades to "keep the first candidate" is the failure this
        # guards: the winner has to be at least as good as restart 0 on the same score.
        first_restart_score = float(fit["bestMetricsByStartKind"]["random"]["selectionScore"])
        self.assertTrue(math.isfinite(first_restart_score))
        self.assertLessEqual(float(fit["selectionScore"]), first_restart_score + 1e-12)

    def test_non_finite_cubic_witness_fails_loudly_instead_of_ranking_on_nan(self) -> None:
        with self.assertRaises(RuntimeError) as raised:
            self._fit(
                cubic_directions=np.ones((4, 3), dtype=np.float64),
                cubic_targets=np.full(4, np.nan, dtype=np.float64),
                selection_cubic_weight_override=1.0,
                cubic_weight_override=0.0,
            )

        self.assertIn("no finite selection score", str(raised.exception))




class PairedAtomVerdictTest(unittest.TestCase):
    """The report's own verdict must include the identity bar the gate applies.

    The sweep failed a case on `pairedAtomMinJointCosine` while the solver, never told
    about that bar, wrote `pass: true` into the same run's report. The manifest then read
    `status: fail` beside `summary.pass: true`, which is the artifact disagreeing with
    itself about whether it passed.
    """

    @staticmethod
    def alignment(minimum: float, above99: int, above95: int) -> dict:
        return {
            "pairedAtomMinJointCosine": minimum,
            "pairedAtomCountAbove099": above99,
            "pairedAtomCountAbove095": above95,
        }

    def test_bar_is_off_when_no_threshold_is_requested(self) -> None:
        self.assertTrue(paired_atom_identity_passes(self.alignment(0.1, 0, 0), 0.0, 16))

    def test_all_atoms_above_the_bar_pass(self) -> None:
        self.assertTrue(paired_atom_identity_passes(self.alignment(0.9909, 16, 16), 0.99, 16))

    def test_two_unidentified_atoms_fail_even_with_a_high_minimum_count(self) -> None:
        # The measured layer15 case: 14 of 16 atoms matched, the other two at 0.24.
        self.assertFalse(paired_atom_identity_passes(self.alignment(0.2386, 14, 14), 0.99, 16))

    def test_a_bar_that_cannot_be_measured_fails(self) -> None:
        self.assertFalse(paired_atom_identity_passes({}, 0.99, 16))

    def test_095_bar_reads_the_095_count(self) -> None:
        self.assertTrue(paired_atom_identity_passes(self.alignment(0.96, 12, 16), 0.95, 16))
        self.assertFalse(paired_atom_identity_passes(self.alignment(0.96, 12, 15), 0.95, 16))


if __name__ == "__main__":
    unittest.main()
