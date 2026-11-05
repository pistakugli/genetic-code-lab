from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.llm_reader.swiglu_amplitude_response import (  # noqa: E402
    DEFAULT_AMPLITUDES,
    analyze_amplitude_outputs,
    amplitude_inputs,
    asymptotic_swiglu_responses,
    make_amplitude_directions,
)


def _synthetic_forward(
    inputs: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
) -> np.ndarray:
    x = torch.as_tensor(inputs, dtype=torch.float64)
    gate_t = torch.as_tensor(gate, dtype=torch.float64)
    up_t = torch.as_tensor(up, dtype=torch.float64)
    down_t = torch.as_tensor(down, dtype=torch.float64)
    with torch.inference_mode():
        return (
            F.silu(F.linear(x, gate_t)) * F.linear(x, up_t)
        ).matmul(down_t.T).numpy()


class SwiGLUAmplitudePolarity(unittest.TestCase):
    def test_exact_positive_negative_asymptotes_and_polarity_swap(self) -> None:
        rng = np.random.default_rng(814)
        input_dim, output_dim, width = 9, 5, 7
        gate = rng.normal(0.0, 0.5, (width, input_dim))
        up = rng.normal(0.0, 0.5, (width, input_dim))
        down = rng.normal(0.0, 0.4, (output_dim, width))
        directions = rng.normal(size=(11, input_dim))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        positive, negative = asymptotic_swiglu_responses(
            gate, up, down, directions
        )
        reversed_positive, reversed_negative = asymptotic_swiglu_responses(
            gate, up, down, -directions
        )
        np.testing.assert_allclose(positive, reversed_negative, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(negative, reversed_positive, rtol=1e-12, atol=1e-12)
        quadratic = (
            (directions @ gate.T) * (directions @ up.T)
        ) @ down.T
        np.testing.assert_allclose(positive + negative, quadratic, rtol=1e-12, atol=1e-12)

        amplitude = 32768.0
        positive_limit = _synthetic_forward(amplitude * directions, gate, up, down)
        negative_limit = _synthetic_forward(-amplitude * directions, gate, up, down)
        np.testing.assert_allclose(
            positive_limit / amplitude**2, positive, rtol=1e-7, atol=2e-8
        )
        np.testing.assert_allclose(
            negative_limit / amplitude**2, negative, rtol=1e-7, atol=2e-8
        )

    def test_polarity_reversal_is_exact_direction_reversal_at_finite_amplitudes(self) -> None:
        rng = np.random.default_rng(91)
        inputs = rng.normal(size=(4, 8)).astype(np.float32)
        gate = rng.normal(size=(6, 8)).astype(np.float32)
        up = rng.normal(size=(6, 8)).astype(np.float32)
        down = rng.normal(size=(5, 6)).astype(np.float32)
        points = amplitude_inputs(inputs)
        reversed_points = amplitude_inputs(-inputs)
        forward = _synthetic_forward(points.reshape(-1, 8), gate, up, down)
        reverse = _synthetic_forward(reversed_points.reshape(-1, 8), gate, up, down)
        forward = forward.reshape(4, len(DEFAULT_AMPLITUDES), 2, 5)
        reverse = reverse.reshape(4, len(DEFAULT_AMPLITUDES), 2, 5)
        np.testing.assert_array_equal(forward[:, :, 1], reverse[:, :, 0])
        np.testing.assert_array_equal(forward[:, :, 0], reverse[:, :, 1])

    def test_direction_generation_preserves_sources_and_separates_provenance(self) -> None:
        rng = np.random.default_rng(31)
        natural = rng.normal(size=(20, 6)).astype(np.float32)
        prompt_ids = np.asarray([f"p-{i}" for i in range(len(natural))])
        hashes = np.asarray([f"{i:064x}" for i in range(len(natural))])
        positions = np.arange(len(natural), dtype=np.int64)
        first = make_amplitude_directions(
            natural,
            prompt_ids=prompt_ids,
            prompt_sha256=hashes,
            token_positions=positions,
            natural_count=8,
            combination_count=9,
            independent_count=7,
            seed=12,
        )
        second = make_amplitude_directions(
            natural,
            prompt_ids=prompt_ids,
            prompt_sha256=hashes,
            token_positions=positions,
            natural_count=8,
            combination_count=9,
            independent_count=7,
            seed=12,
        )
        self.assertEqual(first["directions"].shape, (24, 6))
        np.testing.assert_array_equal(first["directions"], second["directions"])
        np.testing.assert_allclose(
            np.linalg.norm(first["directions"], axis=1), 1.0, atol=1e-6
        )
        self.assertEqual(
            first["direction_kinds"].tolist(),
            ["natural_observed"] * 8
            + ["natural_combination"] * 9
            + ["independent_gaussian"] * 7,
        )
        self.assertTrue(np.all(first["source_natural_indices"][:8, 1] == -1))
        self.assertTrue(np.all(first["source_natural_indices"][-7:] == -1))
        self.assertTrue(np.all(first["source_prompt_ids"][-7:] == ""))

    def test_response_analysis_marks_stability_and_keeps_polarity_responses(self) -> None:
        amplitudes = np.asarray([1.0, 2.0, 4.0, 8.0], dtype=np.float32)
        limit = np.asarray([[1.0, 2.0], [-1.0, 0.5]], dtype=np.float32)
        outputs = np.empty((2, len(amplitudes), 2, 2), dtype=np.float32)
        for index, amplitude in enumerate(amplitudes):
            outputs[:, index, 0] = amplitude**2 * limit
            outputs[:, index, 1] = amplitude**2 * -limit
        analysis = analyze_amplitude_outputs(outputs, amplitudes)
        np.testing.assert_allclose(analysis["normalized_responses"][:, -1, 0], limit)
        np.testing.assert_allclose(analysis["even_responses"], 0.0, atol=1e-7)
        np.testing.assert_allclose(analysis["odd_responses"][:, -1], limit)
        self.assertTrue(np.all(analysis["stable_at_final_two_steps"]))
        self.assertTrue(np.all(analysis["consecutive_cosine_defined"]))

    def test_invalid_zero_direction_and_nonmonotonic_amplitudes_fail_loudly(self) -> None:
        natural = np.ones((3, 4), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "zero-length"):
            make_amplitude_directions(
                np.vstack([natural, np.zeros(4, dtype=np.float32)]),
                prompt_ids=np.asarray(["a", "b", "c", "d"]),
                prompt_sha256=np.asarray(["a", "b", "c", "d"]),
                token_positions=np.arange(4),
                natural_count=4,
                combination_count=0,
                independent_count=0,
            )
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            amplitude_inputs(np.ones((1, 4)), np.asarray([1.0, 0.5]))


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(
        unittest.defaultTestLoader.loadTestsFromTestCase(SwiGLUAmplitudePolarity)
    )
    if not result.wasSuccessful():
        raise SystemExit(1)
    print("SwiGLU amplitude-polarity tests passed")
