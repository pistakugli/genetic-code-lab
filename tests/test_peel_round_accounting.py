#!/usr/bin/env python3
"""A peel round that freezes the whole remainder still has to be scored.

The loop only reached its scoring step on rounds that left atoms over to refit. A round that
froze everything remaining left through the `remaining <= 0` exit instead, so the complete block
it had just produced was never assembled, never compared, and never kept.

That is what the live 32-unit Llama lane did: round 0 froze 6 of 32 atoms, round 1 returned all 26
remaining as confident and froze them, and the run finished holding round 0's six-atom assembly at
R2 0.99166. The round-1 entry in the artifact carries no `assembledMse` at all, which is the
signature of the discarded block rather than of a block that failed to solve.

These tests run the real peel loop on a small block and check the round log, not the arithmetic.
The rank is kept below the slice threshold so no round asks the oracle for anything.
"""
from __future__ import annotations

import argparse
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

from tools.llm_reader.blind_swiglu_oracle_decompiler import (  # noqa: E402
    OracleBudgetExceeded,
    fit_swiglu_sum,
    peel_and_silence_refine,
)

ROWS = 900
RANK = 6
ATOMS = 4
# Above the slice threshold, so a round actually reaches the oracle instead of skipping it.
SLICE_RANK = 12


def _block(seed: int = 7, rank: int = RANK) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    gate = rng.normal(size=(ATOMS, rank)) / rank
    up = rng.normal(size=(ATOMS, rank)) / rank
    coords = rng.normal(size=(ROWS, rank)) * 0.4
    z = coords @ gate.T
    targets = ((z / (1.0 + np.exp(-z))) * (coords @ up.T)).sum(axis=1)
    return coords, targets


class _BrokeOracle:
    """An oracle with nothing left in the lane's budget."""

    def __init__(self) -> None:
        self.calls = 0

    def eval(self, points, stage=None):  # noqa: ANN001
        self.calls += 1
        raise OracleBudgetExceeded(
            stage=str(stage), requested=int(np.asarray(points).shape[0]), consumed=1, limit=1
        )


def _peel(rounds: int, threshold: float, oracle=None, rank: int = RANK) -> dict:  # noqa: ANN001
    coords, targets = _block(rank=rank)
    fit = fit_swiglu_sum(
        coords, targets, ATOMS, seed=1, steps=120, learning_rate=0.05,
        restart_override=16, fit_mode="batched",
    )
    previous = {
        "GCL_BLIND_SWIGLU_PEEL_ROUNDS": os.environ.get("GCL_BLIND_SWIGLU_PEEL_ROUNDS"),
        "GCL_BLIND_SWIGLU_PEEL_THRESHOLD": os.environ.get("GCL_BLIND_SWIGLU_PEEL_THRESHOLD"),
    }
    os.environ["GCL_BLIND_SWIGLU_PEEL_ROUNDS"] = str(rounds)
    os.environ["GCL_BLIND_SWIGLU_PEEL_THRESHOLD"] = str(threshold)
    try:
        return peel_and_silence_refine(
            oracle=oracle,
            basis=np.eye(rank),
            coords=coords,
            targets=targets,
            fit=fit,
            atom_count=ATOMS,
            seed=1,
            steps=120,
            learning_rate=0.05,
            restart_count=16,
            input_scale=0.4,
            args=argparse.Namespace(fit_mode="batched", fit_restart_chunk_size=None),
        )
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class PeelRoundsAreAccountedFor(unittest.TestCase):
    def test_a_round_that_freezes_the_remainder_is_still_scored(self) -> None:
        refined = _peel(rounds=3, threshold=0.0)
        log = refined.get("peelRounds")
        self.assertIsNotNone(log, "the peel loop reported no rounds")
        saturated = [
            entry for entry in log
            if int(entry.get("frozenThisRound", 0)) == int(entry["fittedAtoms"])
        ]
        self.assertTrue(saturated, f"no round froze its whole remainder: {log}")
        for entry in saturated:
            self.assertTrue(
                "assembledMse" in entry or "assembleFailed" in entry,
                f"round {entry['round']} froze every remaining atom and was never scored: {entry}",
            )

    def test_a_scored_complete_round_can_win_the_run(self) -> None:
        """Scoring it is only half the repair - it also has to be reachable as the best block."""
        refined = _peel(rounds=3, threshold=0.0)
        log = refined.get("peelRounds")
        complete = [entry for entry in log if entry.get("assembledAllRemaining")]
        if not complete:
            self.skipTest("this block never froze its whole remainder in one round")
        best = min(
            float(entry["assembledMse"]) for entry in log if entry.get("assembledMse") is not None
        )
        if float(complete[0]["assembledMse"]) <= best and refined.get("peelAccepted"):
            self.assertAlmostEqual(
                float(refined["mse"]), float(complete[0]["assembledMse"]), places=12,
                msg="the winning round's block is not the one that was returned",
            )

    def test_no_round_freezes_more_atoms_than_it_was_offered(self) -> None:
        refined = _peel(rounds=3, threshold=0.0)
        for entry in refined.get("peelRounds", []):
            if "frozenThisRound" not in entry:
                continue
            self.assertLessEqual(
                int(entry["frozenThisRound"]), int(entry["confidentOffered"]),
                f"round {entry['round']} froze more than it was offered: {entry}",
            )
            self.assertLessEqual(
                int(entry["frozenThisRound"]), int(entry["fittedAtoms"]),
                f"round {entry['round']} froze more atoms than remained: {entry}",
            )

    def test_a_round_stopped_by_the_budget_still_scores_what_it_froze(self) -> None:
        """The budget stops the round; it must not also erase the atoms the round froze.

        The exception left through the loop with the freezes sitting in `frozen_gates` and no
        assembly ever built from them, so the run returned an earlier round's block and every
        oracle call the round had spent bought nothing.
        """
        oracle = _BrokeOracle()
        refined = _peel(rounds=3, threshold=0.0, oracle=oracle, rank=SLICE_RANK)
        log = refined.get("peelRounds")
        self.assertIsNotNone(
            log,
            "the run came back as the untouched starting fit: with nothing scored there was no "
            "best block, so the rounds and their frozen atoms were dropped whole",
        )
        self.assertGreater(oracle.calls, 0, "no round reached the oracle, so the path is untested")
        stopped = [entry for entry in log if "budget" in str(entry.get("stopped", ""))]
        self.assertTrue(stopped, f"no round recorded the budget stop: {log}")
        for entry in stopped:
            self.assertTrue(
                "assembledMse" in entry or "assembleFailed" in entry,
                f"round {entry['round']} froze atoms and lost them to the budget: {entry}",
            )

    def test_the_prefix_search_reports_what_it_chose_from(self) -> None:
        """A round log without the offered count cannot say whether a threshold discriminated."""
        refined = _peel(rounds=3, threshold=0.0)
        scoring = [entry for entry in refined.get("peelRounds", []) if entry["confident"] > 0]
        self.assertTrue(scoring, "no round found a confident atom at threshold 0.0")
        for entry in scoring:
            if entry.get("stopped"):
                continue
            self.assertIn("confidentOffered", entry)
            self.assertIn("frozenThisRound", entry)


if __name__ == "__main__":
    unittest.main(verbosity=2)
