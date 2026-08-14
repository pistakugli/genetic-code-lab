#!/usr/bin/env python3
"""The access invariant of the cubic route, as a check rather than a promise.

Every claim this route makes rests on one thing: the solver sees `oracle(points)` and nothing
else, and the truth is unavailable until the recovery is already frozen. Until now that was
enforced by the live oracle and *recorded* by the reports - `truthUnlockQueryCount`,
`unlockedAfterAllSolverCalls` - which is the same shape as `accessAudit` was before F17: fields
that agree with the request by construction. A field written by the thing it describes is not a
check.

So this drives the real `stitch_recovery` through the real `LiveTeacher` against an oracle whose
weights count every read. A solver that peeked would leave a non-zero count before the unlock,
and these tests would fail. They are written to fail that way: `test_weights_are_untouched_...`
fails if the count moves, and `test_truth_is_locked_...` fails if the lock is decorative.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from tools.llm_reader.cubic_identification_probe import (  # noqa: E402
    DTYPE,
    LiveTeacher,
    stitch_recovery,
)


class CountingOracle:
    """A SwiGLU block behind the same surface the live client presents.

    The weights live behind properties that count their own reads, which is the whole point: the
    solver's access is not asserted, it is measured. `truth` raises while locked, the same way
    `OracleClient.truth` does, so a path that tried to score early would raise rather than
    silently succeed.
    """

    def __init__(self, units: int, hidden: int, seed: int, device: torch.device) -> None:
        generator = torch.Generator(device=device).manual_seed(seed)
        self._gate = torch.randn(units, hidden, generator=generator, device=device, dtype=DTYPE)
        self._gate = self._gate / (hidden ** 0.5)
        self._up = torch.randn(units, hidden, generator=generator, device=device, dtype=DTYPE)
        self._up = self._up / (hidden ** 0.5)
        self._down = torch.randn(hidden, units, generator=generator, device=device, dtype=DTYPE)
        self.weight_reads = 0
        self.query_count = 0
        self.truth_unlock_stage: str | None = None
        self.truth_unlock_query_count: int | None = None
        self._unlocked = False
        self.meta = {"hiddenSize": hidden, "blockDevice": str(device)}

    @property
    def gate(self) -> torch.Tensor:
        self.weight_reads += 1
        return self._gate

    @property
    def up(self) -> torch.Tensor:
        self.weight_reads += 1
        return self._up

    @property
    def down(self) -> torch.Tensor:
        self.weight_reads += 1
        return self._down

    def eval(self, inputs, stage: str = "eval"):
        del stage
        points = torch.as_tensor(inputs, dtype=DTYPE, device=self._gate.device)
        self.query_count += int(points.shape[0])
        # Reading the weights to answer a call is the oracle doing its job, so the counter is
        # rolled back here. Only reads from outside this method are the solver's.
        before = self.weight_reads
        activation = torch.nn.functional.silu(points @ self.gate.T)
        values = (activation * (points @ self.up.T)) @ self.down.T
        self.weight_reads = before
        return values.cpu().numpy()

    def unlock_posthoc_truth(self, stage: str) -> None:
        if self._unlocked:
            raise RuntimeError("post-hoc truth already unlocked")
        self._unlocked = True
        self.truth_unlock_stage = stage
        self.truth_unlock_query_count = int(self.query_count)

    def truth(self):
        if not self._unlocked:
            raise RuntimeError("oracle truth is locked until recovery is frozen after validation")
        return self.gate, self.up


def _run_stitch(oracle: CountingOracle, hidden: int, units: int, device: torch.device) -> dict:
    teacher = LiveTeacher(oracle, device)
    unlock_state: dict[str, int] = {}

    def truth_gate_provider() -> torch.Tensor:
        # What the solver had spent at the moment scoring began. If any solve ran afterwards the
        # two counts would differ, which is the thing the live reports assert about themselves.
        unlock_state["queries_before_unlock"] = int(oracle.query_count)
        unlock_state["weight_reads_before_unlock"] = int(oracle.weight_reads)
        oracle.unlock_posthoc_truth("test-stitch")
        gate, _ = oracle.truth()
        return gate

    report = stitch_recovery(
        units=units, input_dim=hidden, output_dim=hidden, subspace_dim=8, core_dim=4,
        base_scale=1.0, rank_cap=4, core_points=2, pencil_draws=3, seed=0,
        tolerance=1e-6, span_tolerance=1e-6, purity=0.9, collinearity=0.99,
        steps=[0.02, 0.04], device=device, teacher=teacher,
        truth_gate_provider=truth_gate_provider, reader_count=6, min_anchors=2,
    )
    report.update(unlock_state)
    report["finalQueryCount"] = int(oracle.query_count)
    report["finalWeightReads"] = int(oracle.weight_reads)
    return report


class LiveAccessInvariant(unittest.TestCase):
    HIDDEN = 12
    UNITS = 2

    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.oracle = CountingOracle(self.UNITS, self.HIDDEN, seed=3, device=self.device)
        self.report = _run_stitch(self.oracle, self.HIDDEN, self.UNITS, self.device)

    def test_solver_spends_oracle_calls(self) -> None:
        """A run that recovered nothing because it asked nothing would pass every other test."""
        self.assertGreater(self.report["finalQueryCount"], 0)
        self.assertEqual(self.report["oracleCallsPoints"], self.report["finalQueryCount"])

    def test_weights_are_untouched_before_the_truth_is_unlocked(self) -> None:
        """The invariant: `weightsUsedBySolver = false`, measured rather than declared."""
        self.assertEqual(self.report["weight_reads_before_unlock"], 0)

    def test_truth_is_locked_until_every_solver_call_is_spent(self) -> None:
        self.assertEqual(
            self.report["queries_before_unlock"], self.report["finalQueryCount"],
            "the truth was unlocked before the solver had finished spending calls",
        )
        self.assertEqual(self.oracle.truth_unlock_query_count, self.report["finalQueryCount"])

    def test_the_lock_actually_raises(self) -> None:
        """A lock nobody tested is a comment. This one refuses a second unlock and an early read."""
        fresh = CountingOracle(self.UNITS, self.HIDDEN, seed=4, device=self.device)
        with self.assertRaises(RuntimeError):
            fresh.truth()
        fresh.unlock_posthoc_truth("first")
        with self.assertRaises(RuntimeError):
            fresh.unlock_posthoc_truth("second")

    def test_blocks_tile_the_input_space(self) -> None:
        """An assembled gate missing coordinates cannot be seen in any other field (F51)."""
        self.assertTrue(self.report["blocksTileInputSpace"])
        self.assertEqual(self.report["directionsInBlocks"], self.HIDDEN)


if __name__ == "__main__":
    unittest.main(verbosity=2)
