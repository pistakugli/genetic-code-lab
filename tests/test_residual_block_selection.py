#!/usr/bin/env python3
from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.llm_reader.blind_swiglu_oracle_decompiler import (
    detect_swiglu_cancellation_pairs,
    maximum_weight_assignment,
    select_residual_block_atoms,
)


def diagnostic(scores: list[float]) -> dict[str, object]:
    return {
        "topAtoms": [
            {"atom": index, "tailResponsibilityScore": score}
            for index, score in enumerate(scores)
        ]
    }


class ResidualBlockSelectionTest(unittest.TestCase):
    def test_detects_source_blind_cancellation_pair(self) -> None:
        rng = np.random.default_rng(260914)
        coords = rng.normal(size=(2048, 3))
        candidate = {
            "gates": np.array([
                [1.0, 0.0, 0.0],
                [0.99, 0.08, 0.0],
                [0.0, 0.0, 1.0],
            ]),
            "ups": np.array([
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
            ]),
            "scales": np.ones(3),
        }

        pairs = detect_swiglu_cancellation_pairs(coords, candidate)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["atoms"], [0, 1])
        self.assertGreater(pairs[0]["gateCosine"], 0.99)
        self.assertGreater(pairs[0]["upCosine"], 0.99)
        self.assertLess(pairs[0]["contributionCorrelation"], -0.99)

    def test_does_not_flag_independent_swiglu_atoms(self) -> None:
        rng = np.random.default_rng(260915)
        coords = rng.normal(size=(2048, 3))
        candidate = {
            "gates": np.eye(3),
            "ups": np.roll(np.eye(3), shift=1, axis=1),
            "scales": np.ones(3),
        }

        self.assertEqual(detect_swiglu_cancellation_pairs(coords, candidate), [])

    def test_maximum_weight_assignment_avoids_greedy_collision(self) -> None:
        assignment = maximum_weight_assignment(np.array([
            [10.0, 9.0, 0.0],
            [9.0, 0.0, 0.0],
            [0.0, 8.0, 8.0],
        ]))

        self.assertEqual(assignment, [1, 0, 2])

    def test_selects_four_atom_elbow_for_layer9_profile(self) -> None:
        gate_norms = np.array([
            0.8015, 0.7818, 0.7371, 0.7090,
            1.4019, 1.4661, 0.9854, 1.2493,
            0.8726, 0.9577, 0.9654, 1.0947,
            1.1311, 1.1587, 1.1801, 1.1990,
        ])
        atoms, telemetry = select_residual_block_atoms(
            diagnostic([
                0.007986804745446708,
                0.006593001315381546,
                0.005089776389715272,
                0.004277377561020204,
                0.0031239879326643736,
                0.0029614117320949287,
                0.0023178812924602435,
                0.0022521091873103413,
            ]),
            16,
            adaptive=True,
            fixed_top_k=4,
            min_atoms=4,
            max_atoms=8,
            gate_norms=gate_norms,
        )

        self.assertEqual(atoms, [0, 1, 2, 3])
        self.assertEqual(telemetry["method"], "gate-norm-tail-consensus-strong-cut")
        self.assertEqual(telemetry["selectedCount"], 4)

    def test_selects_five_atom_elbow_for_layer1_profile(self) -> None:
        gate_norms = np.array([
            0.7996, 0.8166, 0.8175, 0.8367,
            0.8653, 1.0026, 1.0066, 1.0139,
            1.0305, 1.0339, 1.0359, 1.0372,
            1.0519, 1.0549, 1.1038, 1.2050,
        ])
        atoms, telemetry = select_residual_block_atoms(
            diagnostic([
                0.0060454970020228345,
                0.0056014182367720294,
                0.0047109801927608845,
                0.0041296070497757725,
                0.0038404367088938256,
                0.0029525908340678627,
                0.002527195018048271,
                0.0024455954478864498,
            ]),
            16,
            adaptive=True,
            fixed_top_k=4,
            min_atoms=4,
            max_atoms=8,
            gate_norms=gate_norms,
        )

        self.assertEqual(atoms, [0, 1, 2, 3, 4])
        self.assertEqual(telemetry["method"], "gate-norm-weak-block")
        self.assertEqual(telemetry["selectedCount"], 5)

    def test_selects_eight_atom_weak_block_for_layer15_profile(self) -> None:
        atoms, telemetry = select_residual_block_atoms(
            {
                "topAtoms": [
                    {"atom": 5, "tailResponsibilityScore": 0.003935445789807495},
                    {"atom": 15, "tailResponsibilityScore": 0.0036477064686118448},
                    {"atom": 2, "tailResponsibilityScore": 0.0036222140433447994},
                    {"atom": 14, "tailResponsibilityScore": 0.0033467926530600643},
                    {"atom": 7, "tailResponsibilityScore": 0.003313670692545136},
                    {"atom": 10, "tailResponsibilityScore": 0.0033036917893401233},
                    {"atom": 9, "tailResponsibilityScore": 0.003178097972462981},
                    {"atom": 8, "tailResponsibilityScore": 0.0031146366536028186},
                ],
            },
            16,
            adaptive=True,
            fixed_top_k=4,
            min_atoms=4,
            max_atoms=8,
            gate_norms=np.array([
                0.9755, 1.3610, 0.6762, 1.0905,
                1.2928, 0.7918, 1.2027, 0.6748,
                0.7317, 0.6808, 1.0909, 1.1888,
                0.8714, 1.1895, 0.8147, 0.6777,
            ]),
        )

        self.assertEqual(atoms, [5, 15, 2, 14, 7, 9, 8, 12])
        self.assertEqual(telemetry["method"], "gate-norm-weak-block")
        self.assertEqual(telemetry["selectedCount"], 8)

    def test_fixed_mode_remains_available_for_diagnostics(self) -> None:
        atoms, telemetry = select_residual_block_atoms(
            diagnostic([0.8, 0.4, 0.1, 0.01]),
            4,
            adaptive=False,
            fixed_top_k=2,
            min_atoms=1,
            max_atoms=3,
        )

        self.assertEqual(atoms, [0, 1])
        self.assertEqual(telemetry["method"], "fixed-top-k")


if __name__ == "__main__":
    unittest.main()
