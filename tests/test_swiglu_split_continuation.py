#!/usr/bin/env python3
from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.llm_reader.swiglu_split_continuation import (  # noqa: E402
    _Student,
    _grow_by_splitting,
    _normalize_up_gauge,
    _project_down,
    recover_by_split_continuation,
)


def _synthetic_oracle(
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    device: torch.device,
):
    gate_t = torch.as_tensor(gate, dtype=torch.float32, device=device)
    up_t = torch.as_tensor(up, dtype=torch.float32, device=device)
    down_t = torch.as_tensor(down, dtype=torch.float32, device=device)

    def oracle(inputs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(inputs, dtype=torch.float32, device=device)
        with torch.inference_mode():
            return (
                torch.nn.functional.linear(
                    torch.nn.functional.silu(torch.nn.functional.linear(x, gate_t))
                    * torch.nn.functional.linear(x, up_t),
                    down_t,
                )
                .cpu()
                .numpy()
            )

    return oracle


class SplitContinuationAlgebra(unittest.TestCase):
    def test_student_directional_derivative_matches_autograd_and_central_difference(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(74)
        student = _Student(6, 4, 3, generator, torch.device("cpu"))
        with torch.no_grad():
            student.down.normal_()
        inputs = torch.randn(5, 6, generator=generator)
        directions = torch.randn(5, 6, generator=generator)
        actual = student.directional_derivative(inputs, directions)
        _, autograd_jvp = torch.autograd.functional.jvp(
            student, inputs, directions, create_graph=False
        )
        self.assertTrue(torch.allclose(actual, autograd_jvp, rtol=2e-5, atol=2e-6))
        step = 0.01
        finite_difference = (
            student(inputs + step * directions)
            - student(inputs - step * directions)
        ) / (2.0 * step)
        self.assertTrue(torch.allclose(actual, finite_difference, rtol=2e-2, atol=2e-4))

    def test_streamed_projection_handles_duplicate_student_components(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(29)
        student = _Student(4, 3, 8, generator, torch.device("cpu"))
        with torch.no_grad():
            student.gate[1::2].copy_(student.gate[::2])
            student.up[1::2].copy_(student.up[::2])
        rng = np.random.default_rng(29)
        inputs = rng.normal(size=(64, 4)).astype(np.float32)
        targets = rng.normal(size=(64, 3)).astype(np.float32)
        effective_ridge = _project_down(
            student,
            inputs,
            targets,
            torch.device("cpu"),
            batch_size=16,
            ridge=1e-8,
        )
        self.assertGreaterEqual(effective_ridge, 1e-8)
        self.assertTrue(torch.isfinite(student.down).all())

    def test_exact_swiglu_third_derivative_matches_cubic_taylor_coefficient(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(12)
        gate = torch.randn(4, 6, generator=generator, dtype=torch.float64)
        up = torch.randn(4, 6, generator=generator, dtype=torch.float64)
        down = torch.randn(5, 4, generator=generator, dtype=torch.float64)
        direction = torch.randn(6, generator=generator, dtype=torch.float64)
        reader = torch.randn(5, generator=generator, dtype=torch.float64)
        scalar = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        x = scalar * direction
        value = (
            torch.nn.functional.silu(x @ gate.T)
            * (x @ up.T)
        ) @ down.T @ reader
        first = torch.autograd.grad(value, scalar, create_graph=True)[0]
        second = torch.autograd.grad(first, scalar, create_graph=True)[0]
        third = torch.autograd.grad(second, scalar)[0]
        expected = 1.5 * torch.sum(
            (down.T @ reader) * (gate @ direction).square() * (up @ direction)
        )
        self.assertTrue(torch.allclose(third, expected, rtol=1e-12, atol=1e-12))

    @unittest.skipUnless(torch.cuda.is_available(), "split continuation requires CUDA")
    def test_component_split_and_up_down_gauge_preserve_the_function(self) -> None:
        device = torch.device("cuda")
        generator = torch.Generator(device="cpu").manual_seed(4)
        student = _Student(7, 5, 3, generator, device)
        with torch.no_grad():
            student.down.normal_()
        inputs = torch.randn(19, 7, device=device)
        before = student(inputs)
        _grow_by_splitting(student, generator, 0.0)
        after_split = student(inputs)
        self.assertTrue(torch.allclose(before, after_split, rtol=2e-6, atol=2e-6))
        _normalize_up_gauge(student)
        after_gauge = student(inputs)
        self.assertTrue(torch.allclose(after_split, after_gauge, rtol=2e-6, atol=2e-6))

    @unittest.skipUnless(torch.cuda.is_available(), "split continuation requires CUDA")
    def test_oracle_only_recovery_on_multiple_exact_swiglu_architectures(self) -> None:
        device = torch.device("cuda")
        cases = ((4, 3, 2, 7, 0.07), (8, 5, 3, 17, 0.20))
        for input_dim, output_dim, teacher_width, seed, max_nmse in cases:
            with self.subTest(input_dim=input_dim, teacher_width=teacher_width):
                rng = np.random.default_rng(seed)
                gate = (rng.normal(size=(teacher_width, input_dim)) * 0.7).astype(np.float32)
                up = (rng.normal(size=(teacher_width, input_dim)) * 0.7).astype(np.float32)
                down = (rng.normal(size=(output_dim, teacher_width)) * 0.5).astype(np.float32)
                oracle = _synthetic_oracle(gate, up, down, device)
                natural_inputs = rng.normal(size=(96, input_dim)).astype(np.float32)
                natural_outputs = oracle(natural_inputs)
                calls = 0

                def counted_oracle(points: np.ndarray) -> np.ndarray:
                    nonlocal calls
                    calls += len(points)
                    return oracle(points)

                parameters = inspect.signature(recover_by_split_continuation).parameters
                self.assertFalse({"teacher", "weights", "unit_count", "hidden_unit_count"} & parameters.keys())
                result = recover_by_split_continuation(
                    counted_oracle,
                    input_dim,
                    output_dim,
                    natural_inputs,
                    natural_outputs,
                    max_components=8,
                    query_budget=2048,
                    training_queries=1024,
                    validation_queries=512,
                    adaptive_batch_queries=0,
                    steps_per_stage=1000,
                    batch_size=64,
                    learning_rate=0.003,
                    projection_interval=250,
                    projection_batch_size=256,
                    split_perturbation=0.06,
                    projection_ridge=1e-6,
                    seed=seed + 1,
                    device=device,
                )
                self.assertLess(result["validation"]["normalizedMse"], max_nmse)
                self.assertEqual(result["queryAccounting"]["oraclePoints"], calls)
                self.assertEqual(calls, 1536)
                self.assertFalse(result["teacherComponentCountInferred"])
                self.assertEqual(result["solverAccess"], "oracle-only")

    @unittest.skipUnless(torch.cuda.is_available(), "split continuation requires CUDA")
    def test_query_budget_rejects_before_oracle_call(self) -> None:
        device = torch.device("cuda")
        called = False

        def oracle(inputs: np.ndarray) -> np.ndarray:
            nonlocal called
            called = True
            return np.zeros((len(inputs), 2), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "exceed query budget"):
            recover_by_split_continuation(
                oracle,
                2,
                2,
                np.ones((4, 2), dtype=np.float32),
                np.ones((4, 2), dtype=np.float32),
                max_components=1,
                query_budget=7,
                training_queries=4,
                validation_queries=4,
                adaptive_batch_queries=0,
                steps_per_stage=1,
                batch_size=2,
                learning_rate=0.001,
                projection_interval=1,
                projection_batch_size=2,
                split_perturbation=0.01,
                projection_ridge=1e-5,
                seed=1,
                device=device,
            )
        self.assertFalse(called)

    @unittest.skipUnless(torch.cuda.is_available(), "split continuation requires CUDA")
    def test_oracle_jvp_supervision_preserves_query_accounting(self) -> None:
        device = torch.device("cuda")
        rng = np.random.default_rng(75)
        gate = rng.normal(size=(2, 3)).astype(np.float32) * 0.5
        up = rng.normal(size=(2, 3)).astype(np.float32) * 0.5
        down = rng.normal(size=(2, 2)).astype(np.float32) * 0.5
        oracle = _synthetic_oracle(gate, up, down, device)
        natural_inputs = rng.normal(size=(16, 3)).astype(np.float32)
        natural_outputs = oracle(natural_inputs)
        calls = 0

        def counted_oracle(points: np.ndarray) -> np.ndarray:
            nonlocal calls
            calls += len(points)
            return oracle(points)

        result = recover_by_split_continuation(
            counted_oracle,
            3,
            2,
            natural_inputs,
            natural_outputs,
            max_components=2,
            query_budget=96,
            training_queries=64,
            validation_queries=32,
            adaptive_batch_queries=0,
            steps_per_stage=4,
            batch_size=8,
            learning_rate=0.001,
            projection_interval=2,
            projection_batch_size=16,
            split_perturbation=0.03,
            projection_ridge=1e-6,
            seed=76,
            device=device,
            directional_derivative_weight=0.2,
            directional_difference_step=0.02,
        )
        self.assertEqual(calls, 96)
        self.assertEqual(result["queryAccounting"]["oraclePoints"], calls)
        self.assertEqual(result["queryAccounting"]["directionalDerivativePairs"], 8)
        self.assertEqual(
            result["queryAccounting"][
                "directionalDerivativeProbePointsReusedForOutputTraining"
            ],
            16,
        )
        self.assertTrue(
            result["queryAccounting"][
                "directionalDerivativeQueriesAreSubsetOfTrainingQueries"
            ]
        )
        self.assertIsNotNone(result["directionalDerivativeValidation"])
        derivative_data = result["directionalDerivativeData"]
        self.assertEqual(derivative_data["trainingDirections"].shape, (8, 3))
        self.assertEqual(derivative_data["validationDirections"].shape, (4, 3))
        self.assertEqual(len(result["trainingData"]["provenance"]), 80)
        self.assertEqual(len(result["validationData"]["provenance"]), 32)
        self.assertEqual(
            np.count_nonzero(
                result["trainingData"]["provenance"]
                == "generated_finite_difference_oracle_probe"
            ),
            16,
        )
        self.assertEqual(
            np.count_nonzero(
                result["validationData"]["provenance"]
                == "generated_finite_difference_validation_probe"
            ),
            8,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "split continuation requires CUDA")
    def test_jvp_mode_rejects_insufficient_pairs_before_oracle_call(self) -> None:
        device = torch.device("cuda")
        called = False

        def oracle(inputs: np.ndarray) -> np.ndarray:
            nonlocal called
            called = True
            return np.zeros((len(inputs), 2), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "needs at least one pair"):
            recover_by_split_continuation(
                oracle,
                2,
                2,
                np.ones((4, 2), dtype=np.float32),
                np.ones((4, 2), dtype=np.float32),
                max_components=1,
                query_budget=9,
                training_queries=5,
                validation_queries=4,
                adaptive_batch_queries=0,
                steps_per_stage=1,
                batch_size=2,
                learning_rate=0.001,
                projection_interval=1,
                projection_batch_size=2,
                split_perturbation=0.01,
                projection_ridge=1e-5,
                seed=1,
                device=device,
                directional_derivative_weight=0.1,
                directional_derivative_query_fraction=0.1,
            )
        self.assertFalse(called)


if __name__ == "__main__":
    unittest.main(verbosity=2)
