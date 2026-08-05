#!/usr/bin/env python3
"""Tests that must fail before the repair, and pass after.

Written from the audit register in `~/.claude/plans/wondrous-coalescing-cookie.md`. Each one
pins a defect that survived a full day of work because nothing could see it, and each is
written to fail against the code as it stood when the audit was made. A test added after a fix
proves the fix; a test that fails first proves the defect was real.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.llm_reader.jacobian_boundary_recovery import count_fully_recovered  # noqa: E402


class GateSignBlindSpotTest(unittest.TestCase):
    """F20 — the measure that certifies a recovery cannot see a reversed gate.

    `silu` is flat on one side of the knee and linear on the other, so a unit whose gate points
    the wrong way fires exactly when the real one is silent. The recovery was taught to get the
    sign right earlier the same day - 39 of 64 to 64 of 64 - but the measure that signs off on
    it takes an absolute cosine and cannot tell the two apart.
    """

    @staticmethod
    def block(units: int = 4, reverse_gate_of: int | None = None, flip_up_of: int | None = None):
        """A perfect recovery, optionally with one unit's gate or up direction reversed."""
        rng = np.random.default_rng(3)
        truth_gate = rng.normal(size=(units, 16))
        truth_up = rng.normal(size=(units, 16))
        truth_out = rng.normal(size=(units, 16))
        gate, up, out = truth_gate.copy(), truth_up.copy(), truth_out.copy()
        if reverse_gate_of is not None:
            gate[reverse_gate_of] *= -1.0
        if flip_up_of is not None:
            # Both together, which is the gauge freedom, not an error.
            up[flip_up_of] *= -1.0
            out[flip_up_of] *= -1.0
        return dict(
            recovered_gates=gate, truth_gates=truth_gate,
            recovered_ups=up, truth_ups=truth_up,
            recovered_outputs=out, truth_outputs=truth_out,
            gate_norms=np.linalg.norm(truth_gate, axis=1),
        )

    def test_a_clean_block_counts_every_unit(self) -> None:
        self.assertEqual(count_fully_recovered(**self.block()), 4)

    def test_a_reversed_gate_is_not_counted_as_recovered(self) -> None:
        # `silu` is flat on one side of the knee, so this unit is silent exactly when the real
        # one fires. Counting it is the failure the audit found.
        self.assertEqual(count_fully_recovered(**self.block(reverse_gate_of=0)), 3)

    def test_flipping_up_and_output_together_is_not_an_error(self) -> None:
        # `d_i silu(g.x)(u.x)` is unchanged when `u` and `d` flip together, so this is a gauge
        # choice and nothing can observe it. Refusing it would be a false negative.
        self.assertEqual(count_fully_recovered(**self.block(flip_up_of=1)), 4)


if __name__ == "__main__":
    unittest.main()


class SwallowedFailureTest(unittest.TestCase):
    """F01, F07, F08 — a broken card must not look like a working one.

    `solve_least_squares` used to wrap the whole GPU path in `except Exception: pass`. An out
    of memory, a driver fault, a shape error - all of them turned into a run that was merely
    slower and said nothing at all. `operator-integrity-lock` forbids exactly this: a failure
    that is silenced rather than repaired.

    One condition is a legitimate answer rather than a breakdown: CUDA's `gels` driver needs
    full rank, and a rank-deficient design is a real case here because near-collinear units are
    one of the negative cases this route must handle. That one is caught by name, and it still
    leaves a trace.
    """

    @staticmethod
    def source() -> str:
        return (REPO_ROOT / "tools/llm_reader/jacobian_boundary_recovery.py").read_text()

    def test_no_blanket_exception_swallows_a_failure(self) -> None:
        body = self.source()

        # Comments explaining the removed anti-pattern are allowed; code doing it is not.
        offending = [
            line for line in body.splitlines()
            if "except Exception" in line and not line.lstrip().startswith("#")
        ]

        self.assertEqual(offending, [])

    def test_a_fallback_leaves_a_trace(self) -> None:
        from tools.llm_reader.jacobian_boundary_recovery import _record_fallback

        notes: list[str] = []
        _record_fallback(notes, "card returned a non-finite solution")

        # Without this the run reports a clean result produced by a path nobody chose.
        self.assertEqual(len(notes), 1)
        self.assertIn("non-finite", notes[0])

    def test_absence_of_a_card_is_not_treated_as_a_fault(self) -> None:
        import ast

        tree = ast.parse(self.source())
        probe = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_torch_cuda"
        )
        caught = [
            ast.unparse(handler.type) if handler.type else "bare"
            for block in ast.walk(probe) if isinstance(block, ast.Try)
            for handler in block.handlers
        ]

        # "No torch installed" is a state, not a failure, and it is the only thing this may
        # absorb. A driver that raises anything else has to surface, because a broken card and
        # an absent one are different situations that used to look identical.
        self.assertEqual(caught, ["ImportError"])
