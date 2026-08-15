#!/usr/bin/env python3
"""Every path out of `fit_swiglu_sum` has to report the same things about itself.

There are three of them - a single-atom branch, a sequential fit for two and three atoms, and the
batched fit above that - and callers do not choose which one they get, the atom count does. A
peeling loop starts at 32 atoms in the batched path and ends at one in the single-atom branch
without ever asking for the change, so a field that only one path fills is a field that disappears
partway through a run.

That is not hypothetical. The single-atom branch returned neither `mse` nor `r2` nor
`atomConsensus`, and a live loop reading `float(fit.get("r2", nan))` got a NaN on its last round
while the fit underneath had reached MAE 6.45e-09. Worse than the NaN: `atomConsensus` came back
`None`, which a caller cannot tell from a measured zero, so a correctly recovered atom looked
exactly like an atom nobody could vouch for.

These tests fit real data at each atom count and check the report, not the arithmetic.
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

from tools.llm_reader.blind_swiglu_oracle_decompiler import fit_swiglu_sum  # noqa: E402

ROWS = 500
RANK = 4


def _fit(atoms: int, restarts: int, seed: int = 1) -> dict:
    rng = np.random.default_rng(7)
    gate = rng.normal(size=(atoms, RANK)) / RANK
    up = rng.normal(size=(atoms, RANK)) / RANK
    coords = rng.normal(size=(ROWS, RANK)) * 0.4
    z = coords @ gate.T
    targets = ((z / (1.0 + np.exp(-z))) * (coords @ up.T)).sum(axis=1)
    return fit_swiglu_sum(
        coords, targets, atoms, seed=seed, steps=60, learning_rate=0.05,
        restart_override=restarts, fit_mode="batched",
    )


class EveryPathReportsTheSameFields(unittest.TestCase):
    def test_each_atom_count_reports_error_and_consensus(self) -> None:
        for atoms in (1, 2, 3, 5):
            with self.subTest(atoms=atoms):
                fit = _fit(atoms, restarts=8)
                self.assertIsNotNone(fit.get("mse"), "mse missing")
                self.assertIsNotNone(fit.get("r2"), "r2 missing")
                self.assertTrue(np.isfinite(float(fit["r2"])), "r2 is not a number")
                consensus = fit.get("atomConsensus")
                self.assertIsNotNone(consensus, "atomConsensus missing")
                self.assertEqual(len(consensus), atoms)
                for value in consensus:
                    self.assertTrue(0.0 <= float(value) <= 1.0 + 1e-9)

    def test_the_three_paths_are_actually_all_exercised(self) -> None:
        """Without this the loop above could be testing one path four times."""
        modes = {atoms: str(_fit(atoms, restarts=8).get("fitMode", "")) for atoms in (1, 2, 5)}
        self.assertEqual(modes[1], "single-atom")
        self.assertTrue(modes[2].startswith("sequential"), modes[2])
        self.assertTrue(modes[5].startswith("batched"), modes[5])

    def test_the_single_atom_branch_honours_the_restart_override(self) -> None:
        """It dropped it, so a loop asking for 128 restarts got 16 on its last round."""
        few = _fit(1, restarts=8)
        many = _fit(1, restarts=128)
        self.assertGreater(int(many["restartCount"]), int(few["restartCount"]))
        self.assertGreaterEqual(int(many["restartCount"]), 128)

    def test_r2_agrees_with_the_returned_parameters(self) -> None:
        """A reported r2 that does not come from the reported fit is worse than none."""
        rng = np.random.default_rng(7)
        atoms = 1
        gate = rng.normal(size=(atoms, RANK)) / RANK
        up = rng.normal(size=(atoms, RANK)) / RANK
        coords = rng.normal(size=(ROWS, RANK)) * 0.4
        z = coords @ gate.T
        targets = ((z / (1.0 + np.exp(-z))) * (coords @ up.T)).sum(axis=1)
        fit = fit_swiglu_sum(
            coords, targets, atoms, seed=1, steps=60, learning_rate=0.05,
            restart_override=8, fit_mode="batched",
        )
        gates = np.asarray(fit["gates"], dtype=np.float64)
        ups = np.asarray(fit["ups"], dtype=np.float64)
        scales = np.asarray(fit["scales"], dtype=np.float64)
        inner = coords @ gates.T
        prediction = ((inner / (1.0 + np.exp(-inner))) * (coords @ ups.T) * scales).sum(axis=1)
        prediction = prediction + float(fit["bias"])
        mse = float(np.mean((targets - prediction) ** 2))
        variance = float(np.mean((targets - np.mean(targets)) ** 2))
        self.assertAlmostEqual(float(fit["mse"]), mse, places=10)
        self.assertAlmostEqual(float(fit["r2"]), 1.0 - mse / variance, places=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
