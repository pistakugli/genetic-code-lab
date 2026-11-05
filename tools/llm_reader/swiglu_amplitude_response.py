from __future__ import annotations

import math
from typing import Any

import numpy as np


DEFAULT_AMPLITUDES = np.asarray(
    [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0],
    dtype=np.float32,
)


def _unit_rows(values: np.ndarray, name: str) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float64)
    if rows.ndim != 2 or not np.isfinite(rows).all():
        raise ValueError(f"{name} must be a finite rank-2 array")
    norms = np.linalg.norm(rows, axis=1)
    if np.any(norms <= np.finfo(np.float64).tiny):
        raise ValueError(f"{name} contains a zero-length direction")
    return (rows / norms[:, None]).astype(np.float32)


def make_amplitude_directions(
    natural_inputs: np.ndarray,
    *,
    prompt_ids: np.ndarray,
    prompt_sha256: np.ndarray,
    token_positions: np.ndarray,
    natural_count: int = 32,
    combination_count: int = 32,
    independent_count: int = 32,
    seed: int = 941,
) -> dict[str, np.ndarray]:
    """Build unit directions with source rows retained as provenance."""
    natural = np.asarray(natural_inputs)
    if natural.ndim != 2 or natural.shape[0] == 0:
        raise ValueError("natural_inputs must be a non-empty matrix")
    if natural_count <= 0 or combination_count < 0 or independent_count < 0:
        raise ValueError("natural_count must be positive and other counts non-negative")
    if natural_count > len(natural):
        raise ValueError("natural_count exceeds the number of captured natural inputs")
    metadata = {
        "prompt_ids": np.asarray(prompt_ids).astype(str),
        "prompt_sha256": np.asarray(prompt_sha256).astype(str),
        "token_positions": np.asarray(token_positions),
    }
    if any(values.shape != (len(natural),) for values in metadata.values()):
        raise ValueError("natural activation provenance lengths do not match the inputs")
    if not np.isfinite(natural).all():
        raise ValueError("natural_inputs contains non-finite values")

    rng = np.random.default_rng(seed)
    natural_indices = np.linspace(
        0, len(natural) - 1, num=natural_count, dtype=np.int64
    )
    directions = [_unit_rows(natural[natural_indices], "natural inputs")]
    kinds = ["natural_observed"] * natural_count
    source_indices = np.full((natural_count, 2), -1, dtype=np.int64)
    source_indices[:, 0] = natural_indices
    source_coefficients = np.zeros((natural_count, 2), dtype=np.float32)
    source_coefficients[:, 0] = 1.0

    if combination_count:
        first = rng.integers(0, len(natural), size=combination_count, dtype=np.int64)
        offset = rng.integers(1, len(natural), size=combination_count, dtype=np.int64)
        second = (first + offset) % len(natural)
        coefficients = rng.normal(size=(combination_count, 2)).astype(np.float32)
        mixtures = (
            coefficients[:, :1] * natural[first].astype(np.float32)
            + coefficients[:, 1:] * natural[second].astype(np.float32)
        )
        directions.append(_unit_rows(mixtures, "natural combinations"))
        kinds.extend(["natural_combination"] * combination_count)
        source_indices = np.concatenate(
            [source_indices, np.stack([first, second], axis=1)], axis=0
        )
        source_coefficients = np.concatenate(
            [source_coefficients, coefficients], axis=0
        )

    if independent_count:
        independent = rng.normal(
            size=(independent_count, natural.shape[1])
        ).astype(np.float32)
        directions.append(_unit_rows(independent, "independent directions"))
        kinds.extend(["independent_gaussian"] * independent_count)
        source_indices = np.concatenate(
            [source_indices, np.full((independent_count, 2), -1, dtype=np.int64)],
            axis=0,
        )
        source_coefficients = np.concatenate(
            [source_coefficients, np.zeros((independent_count, 2), dtype=np.float32)],
            axis=0,
        )

    direction_matrix = np.concatenate(directions, axis=0)
    direction_count = len(direction_matrix)
    source_prompt_ids = np.full((direction_count, 2), "", dtype="<U64")
    source_prompt_hashes = np.full((direction_count, 2), "", dtype="<U64")
    source_token_positions = np.full((direction_count, 2), -1, dtype=np.int64)
    for direction_index, indices in enumerate(source_indices):
        for source_slot, natural_index in enumerate(indices):
            if natural_index >= 0:
                source_prompt_ids[direction_index, source_slot] = metadata[
                    "prompt_ids"
                ][natural_index]
                source_prompt_hashes[direction_index, source_slot] = metadata[
                    "prompt_sha256"
                ][natural_index]
                source_token_positions[direction_index, source_slot] = int(
                    metadata["token_positions"][natural_index]
                )

    return {
        "directions": direction_matrix,
        "direction_ids": np.asarray(
            [f"amplitude-direction-{index:04d}" for index in range(direction_count)],
            dtype="<U32",
        ),
        "direction_kinds": np.asarray(kinds, dtype="<U24"),
        "source_natural_indices": source_indices,
        "source_coefficients": source_coefficients,
        "source_prompt_ids": source_prompt_ids,
        "source_prompt_sha256": source_prompt_hashes,
        "source_token_positions": source_token_positions,
    }


def amplitude_inputs(
    directions: np.ndarray,
    amplitudes: np.ndarray = DEFAULT_AMPLITUDES,
) -> np.ndarray:
    vectors = np.asarray(directions, dtype=np.float32)
    scales = np.asarray(amplitudes, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] == 0 or vectors.shape[1] == 0:
        raise ValueError("directions must be a non-empty matrix")
    if scales.ndim != 1 or scales.size == 0 or not np.isfinite(scales).all():
        raise ValueError("amplitudes must be a non-empty finite vector")
    if np.any(scales <= 0.0) or np.any(np.diff(scales) <= 0.0):
        raise ValueError("amplitudes must be positive and strictly increasing")
    if not np.isfinite(vectors).all():
        raise ValueError("directions contain non-finite values")
    polarities = np.asarray([1.0, -1.0], dtype=np.float32)
    return (
        vectors[:, None, None, :]
        * scales[None, :, None, None]
        * polarities[None, None, :, None]
    )


def analyze_amplitude_outputs(
    outputs: np.ndarray,
    amplitudes: np.ndarray = DEFAULT_AMPLITUDES,
    *,
    stability_tolerance: float = 1e-2,
) -> dict[str, np.ndarray]:
    values = np.asarray(outputs, dtype=np.float32)
    scales = np.asarray(amplitudes, dtype=np.float64)
    if values.ndim != 4 or values.shape[2] != 2:
        raise ValueError("outputs must have shape [directions, amplitudes, 2, output_dim]")
    if values.shape[1] != len(scales) or len(scales) < 3:
        raise ValueError("outputs must have at least three matching amplitude levels")
    if not np.isfinite(values).all():
        raise ValueError("oracle outputs contain non-finite values")
    if not math.isfinite(stability_tolerance) or stability_tolerance <= 0:
        raise ValueError("stability_tolerance must be finite and positive")

    normalized64 = values.astype(np.float64) / (scales[None, :, None, None] ** 2)
    if not np.isfinite(normalized64).all() or np.max(np.abs(normalized64)) > np.finfo(
        np.float32
    ).max:
        raise FloatingPointError("amplitude normalization produced non-finite float32 values")
    normalized = normalized64.astype(np.float32)
    differences = np.diff(normalized64, axis=1)
    previous_norms = np.linalg.norm(normalized64[:, :-1], axis=-1)
    current_norms = np.linalg.norm(normalized64[:, 1:], axis=-1)
    difference_norms = np.linalg.norm(differences, axis=-1)
    denominator = np.maximum(previous_norms, current_norms)
    relative_changes = np.divide(
        difference_norms,
        denominator,
        out=np.zeros_like(difference_norms),
        where=denominator > 1e-30,
    )
    cosine_denominator = previous_norms * current_norms
    cosine_values = np.einsum(
        "dapo,dapo->dap", normalized64[:, :-1], normalized64[:, 1:]
    )
    cosines = np.divide(
        cosine_values,
        cosine_denominator,
        out=np.zeros_like(cosine_values),
        where=cosine_denominator > 1e-30,
    )
    cosine_defined = cosine_denominator > 1e-30
    log2_steps = np.diff(np.log2(scales))
    slopes = differences / log2_steps[None, :, None, None]
    stable = np.all(relative_changes[:, -2:, :] <= stability_tolerance, axis=(1, 2))
    return {
        "normalized_responses": normalized,
        "consecutive_relative_changes": relative_changes.astype(np.float32),
        "consecutive_cosines": cosines.astype(np.float32),
        "consecutive_cosine_defined": cosine_defined,
        "log2_amplitude_slopes": slopes.astype(np.float32),
        "even_responses": (normalized[:, :, 0] + normalized[:, :, 1]) * 0.5,
        "odd_responses": (normalized[:, :, 0] - normalized[:, :, 1]) * 0.5,
        "stable_at_final_two_steps": stable,
    }


def asymptotic_swiglu_responses(
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact positive and negative large-amplitude limits."""
    gate_weights = np.asarray(gate, dtype=np.float64)
    up_weights = np.asarray(up, dtype=np.float64)
    down_weights = np.asarray(down, dtype=np.float64)
    vectors = np.asarray(directions, dtype=np.float64)
    if (
        gate_weights.ndim != 2
        or up_weights.shape != gate_weights.shape
        or down_weights.ndim != 2
        or down_weights.shape[1] != gate_weights.shape[0]
        or vectors.ndim != 2
        or vectors.shape[1] != gate_weights.shape[1]
    ):
        raise ValueError("incompatible bias-free SwiGLU parameter/direction shapes")
    if not all(
        np.isfinite(value).all()
        for value in (gate_weights, up_weights, down_weights, vectors)
    ):
        raise ValueError("SwiGLU parameters and directions must be finite")
    gate_values = vectors @ gate_weights.T
    up_values = vectors @ up_weights.T
    positive = (np.maximum(gate_values, 0.0) * up_values) @ down_weights.T
    negative = (np.minimum(gate_values, 0.0) * up_values) @ down_weights.T
    return positive, negative
