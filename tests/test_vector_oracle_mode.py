#!/usr/bin/env python3
"""A vector-valued oracle returns the block's contribution to the residual stream.

The scalar modes collapse a block to one number per point. That is what makes a unit's
activation boundary visible only as a kink competing with every other unit's kink for the
same scalar, and it is why the share of boundaries read correctly falls from 94% at 32 units
to 17% at 2048. With the whole 3072-wide contribution returned instead, the Jacobian is a
matrix and a boundary crossing is a rank-1 jump in it, separable by linear algebra rather
than by resolving overlaps along the probe line.

These tests pin the arithmetic of the new mode against the two scalar modes it sits beside,
using the same tensor algebra the oracle server runs, so they hold without a model download.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.llm_reader.blind_swiglu_oracle_decompiler import silu_np  # noqa: E402


def block(seed: int = 5, units: int = 6, hidden: int = 12, points: int = 40):
    rng = np.random.default_rng(seed)
    return {
        "gate": rng.normal(0.0, 1.0, size=(units, hidden)),
        "up": rng.normal(0.0, 1.0, size=(units, hidden)),
        "down": rng.normal(0.0, 1.0, size=(hidden, units)),
        "x": rng.normal(0.0, 0.6, size=(points, hidden)),
    }


def activations(b) -> np.ndarray:
    return silu_np(b["x"] @ b["gate"].T) * (b["x"] @ b["up"].T)


def scalar_sum(b) -> np.ndarray:
    return activations(b) @ np.ones(b["gate"].shape[0])


def scalar_channel(b, channel: int) -> np.ndarray:
    return activations(b) @ b["down"][channel]


def ffn_vector(b) -> np.ndarray:
    return activations(b) @ b["down"].T


class VectorOracleModeTest(unittest.TestCase):
    def test_the_vector_answer_is_one_row_per_point(self) -> None:
        b = block()

        self.assertEqual(ffn_vector(b).shape, (b["x"].shape[0], b["down"].shape[0]))
        self.assertEqual(scalar_sum(b).shape, (b["x"].shape[0],))

    def test_a_single_channel_of_the_vector_is_the_scalar_channel_mode(self) -> None:
        b = block()

        # The vector mode has to be a strict superset of what the channel mode already
        # publishes, or the two lanes are measuring different teachers.
        for channel in (0, 3, 11):
            np.testing.assert_allclose(ffn_vector(b)[:, channel], scalar_channel(b, channel), rtol=1e-12, atol=1e-12)

    def test_the_summed_mode_is_not_recoverable_from_one_channel(self) -> None:
        b = block()

        # Guards against reading the modes as interchangeable: the sum weights every unit by
        # one, a channel weights it by its down_proj entry.
        self.assertFalse(np.allclose(scalar_sum(b), scalar_channel(b, 0)))

    def test_the_jacobian_of_the_vector_output_has_full_structure(self) -> None:
        b = block()
        step = 1e-5
        point = b["x"][0]
        hidden = b["down"].shape[0]
        columns = []
        for axis in range(hidden):
            shift = np.zeros(hidden)
            shift[axis] = step
            plus = silu_np((point + shift) @ b["gate"].T) * ((point + shift) @ b["up"].T) @ b["down"].T
            minus = silu_np((point - shift) @ b["gate"].T) * ((point - shift) @ b["up"].T) @ b["down"].T
            columns.append((plus - minus) / (2 * step))
        jacobian = np.stack(columns, axis=1)

        # The scalar oracle's "Jacobian" is a single row; the vector oracle's is a matrix of
        # rank up to the unit count, which is the room the rank-1 jump argument needs.
        self.assertEqual(jacobian.shape, (hidden, hidden))
        self.assertGreaterEqual(np.linalg.matrix_rank(jacobian, tol=1e-6), 2)

    def test_a_boundary_crossing_moves_the_vector_jacobian_by_a_near_rank_one_matrix(self) -> None:
        b = block(units=3, hidden=8)
        gate = b["gate"]
        step = 1e-5
        hidden = b["down"].shape[0]

        def jacobian(point: np.ndarray) -> np.ndarray:
            columns = []
            for axis in range(hidden):
                shift = np.zeros(hidden)
                shift[axis] = step
                plus = silu_np((point + shift) @ gate.T) * ((point + shift) @ b["up"].T) @ b["down"].T
                minus = silu_np((point - shift) @ gate.T) * ((point - shift) @ b["up"].T) @ b["down"].T
                columns.append((plus - minus) / (2 * step))
            return np.stack(columns, axis=1)

        # Walk along unit 0's gate direction, far enough out that the other two units are
        # saturated and only unit 0's indicator changes across the crossing.
        direction = gate[0] / np.linalg.norm(gate[0])
        offset = np.linalg.qr(np.column_stack([direction, np.eye(hidden)[:, :2]]))[0][:, 1]
        scale = 400.0

        def at(distance: float) -> np.ndarray:
            return jacobian(scale * (offset + distance * direction))

        naive = at(0.02) - at(-0.02)
        # The other units are not perfectly saturated, so what a plain difference returns is
        # the jump plus their smooth drift across the same interval. Extrapolating each side
        # from two points cancels that drift to first order, which is the step that took the
        # rank-1 share from 0.14 to 0.54 on the 512-unit probe.
        extrapolated = (2 * at(0.02) - at(0.04)) - (2 * at(-0.02) - at(-0.04))

        share = lambda matrix: float(np.linalg.svd(matrix, compute_uv=False)[0] / np.linalg.svd(matrix, compute_uv=False).sum())
        self.assertGreater(share(extrapolated), share(naive))
        self.assertGreater(share(extrapolated), 0.9)


if __name__ == "__main__":
    unittest.main()
