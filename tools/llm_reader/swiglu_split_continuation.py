from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.nn import functional as F


Oracle = Callable[[np.ndarray], np.ndarray]


def _proc_vm_hwm_bytes() -> int | None:
    status_path = Path("/proc/self/status")
    if not status_path.is_file():
        return None
    for line in status_path.read_text(encoding="ascii").splitlines():
        if line.startswith("VmHWM:"):
            fields = line.split()
            if len(fields) != 3 or fields[2] != "kB":
                raise RuntimeError(f"unexpected VmHWM format: {line!r}")
            return int(fields[1]) * 1024
    return None


class _Student(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        components: int,
        generator: torch.Generator,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.components = components
        self.gate = torch.nn.Parameter(
            torch.randn(components, input_dim, generator=generator)
            * (0.5 / math.sqrt(input_dim))
        )
        self.up = torch.nn.Parameter(
            torch.randn(components, input_dim, generator=generator)
            * (1.0 / math.sqrt(input_dim))
        )
        self.down = torch.nn.Parameter(torch.zeros(output_dim, components))
        self.to(device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(F.linear(inputs, self.gate)) * F.linear(inputs, self.up)
        return F.linear(hidden, self.down)

    def directional_derivative(
        self, inputs: torch.Tensor, directions: torch.Tensor
    ) -> torch.Tensor:
        gate_values = F.linear(inputs, self.gate)
        up_values = F.linear(inputs, self.up)
        gate_directions = F.linear(directions, self.gate)
        up_directions = F.linear(directions, self.up)
        sigmoid = torch.sigmoid(gate_values)
        silu = F.silu(gate_values)
        silu_derivative = sigmoid * (1.0 + gate_values * (1.0 - sigmoid))
        hidden_derivative = (
            silu_derivative * gate_directions * up_values
            + silu * up_directions
        )
        return F.linear(hidden_derivative, self.down)


def _widths(max_components: int) -> list[int]:
    widths = [1]
    while widths[-1] < max_components:
        widths.append(min(max_components, widths[-1] * 2))
    return widths


def _grow_by_splitting(
    student: _Student,
    generator: torch.Generator,
    relative_perturbation: float,
) -> None:
    old_width = student.components
    gate = student.gate.detach().cpu().repeat_interleave(2, dim=0)
    up = student.up.detach().cpu().repeat_interleave(2, dim=0)
    signs = torch.tensor([1.0, -1.0], dtype=gate.dtype).repeat(old_width)
    gate_scale = gate.norm(dim=1, keepdim=True).clamp_min(1e-6)
    up_scale = up.norm(dim=1, keepdim=True).clamp_min(1e-6)
    gate_noise = torch.randn(gate.shape, generator=generator, dtype=gate.dtype)
    up_noise = torch.randn(up.shape, generator=generator, dtype=up.dtype)
    gate = gate + signs[:, None] * gate_noise * (
        relative_perturbation * gate_scale / math.sqrt(student.input_dim)
    )
    up = up + signs[:, None] * up_noise * (
        relative_perturbation * up_scale / math.sqrt(student.input_dim)
    )
    down = student.down.detach().cpu().repeat_interleave(2, dim=1) * 0.5
    student.gate = torch.nn.Parameter(gate.to(student.gate.device))
    student.up = torch.nn.Parameter(up.to(student.up.device))
    student.down = torch.nn.Parameter(down.to(student.down.device))
    student.components = old_width * 2


def _forward_chunks(
    student: _Student,
    inputs: np.ndarray,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    results: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(inputs), chunk_size):
            batch = torch.as_tensor(
                inputs[start : start + chunk_size], dtype=torch.float32, device=device
            )
            results.append(student(batch).float().cpu().numpy())
    return np.concatenate(results) if results else np.empty((0, student.output_dim), np.float32)


def _directional_derivative_chunks(
    student: _Student,
    inputs: np.ndarray,
    directions: np.ndarray,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    results: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(inputs), chunk_size):
            batch = torch.as_tensor(
                inputs[start : start + chunk_size], dtype=torch.float32, device=device
            )
            vectors = torch.as_tensor(
                directions[start : start + chunk_size],
                dtype=torch.float32,
                device=device,
            )
            results.append(student.directional_derivative(batch, vectors).cpu().numpy())
    return np.concatenate(results) if results else np.empty((0, student.output_dim), np.float32)


def _normalize_up_gauge(student: _Student) -> None:
    """Fix the exact reciprocal up/down scaling freedom to improve conditioning."""
    with torch.no_grad():
        scales = student.up.norm(dim=1).clamp_min(1e-8)
        student.up.div_(scales[:, None])
        student.down.mul_(scales[None, :])


def _project_down(
    student: _Student,
    inputs: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
    *,
    batch_size: int,
    ridge: float,
) -> float:
    _normalize_up_gauge(student)
    width = student.components
    gram = torch.zeros((width, width), dtype=torch.float32, device=device)
    cross = torch.zeros((width, student.output_dim), dtype=torch.float32, device=device)
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            x = torch.as_tensor(
                inputs[start : start + batch_size], dtype=torch.float32, device=device
            )
            y = torch.as_tensor(
                targets[start : start + batch_size], dtype=torch.float32, device=device
            )
            features = F.silu(F.linear(x, student.gate)) * F.linear(x, student.up)
            gram.addmm_(features.T, features)
            cross.addmm_(features.T, y)
    scale = gram.diagonal().mean().clamp_min(1e-12)
    effective_ridge = max(float(ridge), float(torch.finfo(gram.dtype).eps))
    factor = None
    info_value = 0
    for retry in range(8):
        regularized_gram = gram.clone()
        regularized_gram.diagonal().add_(effective_ridge * scale)
        candidate_factor, info = torch.linalg.cholesky_ex(regularized_gram)
        info_value = int(info.max().item())
        if info_value == 0:
            factor = candidate_factor
            break
        effective_ridge *= 10.0
    if factor is None:
        raise RuntimeError(
            f"output-projection solve failed at width {width}: "
            f"cholesky info={info_value} after adaptive ridge escalation"
        )
    solution = torch.cholesky_solve(cross, factor).T
    if not torch.isfinite(solution).all().item():
        raise RuntimeError(f"output-projection solve produced non-finite values at width {width}")
    with torch.no_grad():
        student.down.copy_(solution)
    return effective_ridge


def _metrics(reference: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    ref = np.asarray(reference, dtype=np.float64)
    pred = np.asarray(predicted, dtype=np.float64)
    residual = pred - ref
    mse = float(np.mean(residual * residual))
    variance = float(np.mean((ref - ref.mean(axis=0, keepdims=True)) ** 2))
    return {
        "relativeError": float(np.linalg.norm(residual) / max(np.linalg.norm(ref), 1e-30)),
        "normalizedMse": mse / max(variance, 1e-30),
        "meanSquaredError": mse,
        "cosineSimilarityMean": float(np.mean(
            np.sum(ref * pred, axis=1)
            / np.maximum(np.linalg.norm(ref, axis=1) * np.linalg.norm(pred, axis=1), 1e-30)
        )),
    }


def recover_by_split_continuation(
    oracle: Oracle,
    input_dim: int,
    output_dim: int,
    recovery_inputs: np.ndarray,
    recovery_outputs: np.ndarray,
    *,
    max_components: int,
    query_budget: int,
    training_queries: int,
    validation_queries: int,
    adaptive_batch_queries: int,
    steps_per_stage: int,
    batch_size: int,
    learning_rate: float,
    projection_interval: int,
    projection_batch_size: int,
    split_perturbation: float,
    projection_ridge: float,
    seed: int,
    device: torch.device,
    natural_training_batch_fraction: float = 0.5,
    recovery_metadata: dict[str, np.ndarray] | None = None,
    directional_derivative_weight: float = 0.0,
    directional_difference_step: float = 0.02,
    directional_derivative_query_fraction: float = 0.25,
) -> dict[str, Any]:
    """Fit the complete SwiGLU oracle by function-preserving width continuation.

    The output projection is linear conditional on gate/up parameters. It is
    eliminated at stage boundaries by streamed ridge least squares. Width growth
    duplicates each existing unit and divides its down-column, preserving the
    represented function before a small symmetric perturbation unlocks new degrees
    of freedom.
    """
    if min(input_dim, output_dim, max_components, query_budget, batch_size) <= 0:
        raise ValueError("dimensions, component cap, query budget, and batch size must be positive")
    if training_queries < 4 or validation_queries < 4 or adaptive_batch_queries < 0:
        raise ValueError("training and validation queries must be >= 4")
    if steps_per_stage <= 0 or projection_interval <= 0 or projection_batch_size <= 0:
        raise ValueError("step and projection batch counts must be positive")
    if not 0.0 < split_perturbation < 0.1:
        raise ValueError("split_perturbation must be in (0, 0.1)")
    if projection_ridge <= 0.0 or learning_rate <= 0.0:
        raise ValueError("projection_ridge and learning_rate must be positive")
    if not 0.0 < natural_training_batch_fraction < 1.0:
        raise ValueError("natural_training_batch_fraction must be in (0, 1)")
    if not math.isfinite(directional_derivative_weight) or directional_derivative_weight < 0.0:
        raise ValueError("directional_derivative_weight must be finite and non-negative")
    if directional_derivative_weight > 0.0 and (
        not math.isfinite(directional_difference_step)
        or directional_difference_step <= 0.0
    ):
        raise ValueError("directional_difference_step must be finite and positive for JVP fitting")
    use_directional_derivatives = directional_derivative_weight > 0.0
    if not math.isfinite(directional_derivative_query_fraction) or not (
        0.0 < directional_derivative_query_fraction <= 1.0
    ):
        raise ValueError("directional_derivative_query_fraction must be in (0, 1]")
    if use_directional_derivatives:
        training_pair_points = (
            int(training_queries * directional_derivative_query_fraction / 2.0) * 2
        )
        validation_pair_points = (
            int(validation_queries * directional_derivative_query_fraction / 2.0) * 2
        )
        if training_pair_points < 2 or validation_pair_points < 2:
            raise ValueError("JVP supervision needs at least one pair in training and validation")
    else:
        training_pair_points = 0
        validation_pair_points = 0
    x_natural = np.asarray(recovery_inputs, dtype=np.float32)
    y_natural = np.asarray(recovery_outputs, dtype=np.float32)
    if x_natural.ndim != 2 or x_natural.shape[1] != input_dim:
        raise ValueError("recovery_inputs have the wrong shape")
    if y_natural.shape != (len(x_natural), output_dim):
        raise ValueError("recovery_outputs have the wrong shape")
    if not np.isfinite(x_natural).all() or not np.isfinite(y_natural).all():
        raise ValueError("recovery observations must be finite")
    if recovery_metadata is not None and any(len(value) != len(x_natural) for value in recovery_metadata.values()):
        raise ValueError("recovery metadata must have one row per natural observation")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("split-continuation recovery requires an available CUDA device")
    mandatory_queries = training_queries + validation_queries
    if mandatory_queries > query_budget:
        raise ValueError("initial training and validation points exceed query budget")

    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    natural_mean = x_natural.astype(np.float64).mean(axis=0)
    centered_natural = x_natural.astype(np.float64) - natural_mean
    _, singular_values, right_vectors = np.linalg.svd(
        centered_natural, full_matrices=False
    )
    covariance_eigenvalues = np.square(singular_values) / max(len(x_natural) - 1, 1)
    total_variance = float(covariance_eigenvalues.sum())
    if not math.isfinite(total_variance) or total_variance <= 0.0:
        raise ValueError("recovery observations do not define positive activation variance")
    isotropic_variance = total_variance / input_dim
    covariance_rank = len(covariance_eigenvalues)
    eigen_scales = np.sqrt(np.maximum(covariance_eigenvalues, 0.0))
    shells = np.asarray([0.45, 0.75, 1.0, 1.4], dtype=np.float64)
    spent = 0

    def draw_shell(index: int, count: int) -> np.ndarray:
        samples: list[np.ndarray] = []
        scale = shells[index]
        for start in range(0, count, 2048):
            batch_count = min(2048, count - start)
            low_rank_noise = rng.normal(size=(batch_count, covariance_rank))
            correlated = (low_rank_noise * eigen_scales[None, :]) @ right_vectors
            isotropic = rng.normal(size=(batch_count, input_dim)) * math.sqrt(
                0.15 * isotropic_variance
            )
            samples.append(
                (natural_mean[None, :] + scale * (math.sqrt(0.85) * correlated + isotropic))
                .astype(np.float32)
            )
        return np.concatenate(samples) if samples else np.empty((0, input_dim), np.float32)

    def query(points: np.ndarray, label: str) -> np.ndarray:
        nonlocal spent
        if spent + len(points) > query_budget:
            raise RuntimeError(
                f"query budget exceeded at {label}: spent={spent}, "
                f"requested={len(points)}, budget={query_budget}"
            )
        parts: list[np.ndarray] = []
        for start in range(0, len(points), 512):
            output = np.asarray(oracle(points[start : start + 512]), dtype=np.float32)
            expected = (min(512, len(points) - start), output_dim)
            if output.shape != expected or not np.isfinite(output).all():
                raise ValueError(f"invalid oracle response at {label}: {output.shape}, expected {expected}")
            spent += len(output)
            parts.append(output)
        return np.concatenate(parts)

    if use_directional_derivatives:
        training_pair_count = training_pair_points // 2
        per_shell_train_pairs = [training_pair_count // 4] * 4
        for index in range(training_pair_count % 4):
            per_shell_train_pairs[index] += 1
        train_centers = np.concatenate([
            draw_shell(index, count)
            for index, count in enumerate(per_shell_train_pairs)
        ])
        train_directions = rng.normal(
            size=(training_pair_count, input_dim)
        ).astype(np.float32)
        train_directions /= np.maximum(
            np.linalg.norm(train_directions, axis=1, keepdims=True), 1e-30
        )
        paired_train_x = np.empty((training_pair_points, input_dim), dtype=np.float32)
        paired_train_x[0::2] = (
            train_centers + directional_difference_step * train_directions
        )
        paired_train_x[1::2] = (
            train_centers - directional_difference_step * train_directions
        )
        training_standard_queries = training_queries - training_pair_points
        per_shell_train_standard = [training_standard_queries // 4] * 4
        for index in range(training_standard_queries % 4):
            per_shell_train_standard[index] += 1
        training_standard_x = np.concatenate([
            draw_shell(index, count)
            for index, count in enumerate(per_shell_train_standard)
        ])
        train_generated_x = np.concatenate([paired_train_x, training_standard_x])
        del train_centers
    else:
        training_pair_count = 0
        train_directions = np.empty((0, input_dim), dtype=np.float32)
        training_standard_queries = training_queries
        per_shell_train = [training_queries // 4] * 4
        for index in range(training_queries % 4):
            per_shell_train[index] += 1
        train_generated_x = np.concatenate([
            draw_shell(index, count) for index, count in enumerate(per_shell_train)
        ])
    train_generated_y = query(train_generated_x, "initial-training")
    if use_directional_derivatives:
        validation_pair_count = validation_pair_points // 2
        per_shell_valid_pairs = [validation_pair_count // 4] * 4
        for index in range(validation_pair_count % 4):
            per_shell_valid_pairs[index] += 1
        validation_centers = np.concatenate([
            draw_shell(index, count)
            for index, count in enumerate(per_shell_valid_pairs)
        ])
        validation_directions = rng.normal(
            size=(validation_pair_count, input_dim)
        ).astype(np.float32)
        validation_directions /= np.maximum(
            np.linalg.norm(validation_directions, axis=1, keepdims=True), 1e-30
        )
        paired_validation_x = np.empty((validation_pair_points, input_dim), dtype=np.float32)
        paired_validation_x[0::2] = (
            validation_centers + directional_difference_step * validation_directions
        )
        paired_validation_x[1::2] = (
            validation_centers - directional_difference_step * validation_directions
        )
        validation_standard_queries = validation_queries - validation_pair_points
        per_shell_valid_standard = [validation_standard_queries // 4] * 4
        for index in range(validation_standard_queries % 4):
            per_shell_valid_standard[index] += 1
        validation_standard_x = np.concatenate([
            draw_shell(index, count)
            for index, count in enumerate(per_shell_valid_standard)
        ])
        validation_x = np.concatenate([paired_validation_x, validation_standard_x])
        validation_shells = np.concatenate([
            np.repeat(
                np.concatenate([
                    np.full(count, index, dtype=np.int8)
                    for index, count in enumerate(per_shell_valid_pairs)
                ]),
                2,
            ),
            np.concatenate([
                np.full(count, index, dtype=np.int8)
                for index, count in enumerate(per_shell_valid_standard)
            ]),
        ])
        del validation_centers
    else:
        validation_pair_count = 0
        validation_directions = np.empty((0, input_dim), dtype=np.float32)
        validation_standard_queries = validation_queries
        per_shell_valid = [validation_queries // 4] * 4
        for index in range(validation_queries % 4):
            per_shell_valid[index] += 1
        validation_shells = np.concatenate([
            np.full(count, index, dtype=np.int8)
            for index, count in enumerate(per_shell_valid)
        ])
        validation_x = np.concatenate([
            draw_shell(index, count) for index, count in enumerate(per_shell_valid)
        ])
    validation_y = query(validation_x, "validation")
    train_x = np.concatenate([x_natural, train_generated_x])
    train_y = np.concatenate([y_natural, train_generated_y])
    train_pair_row_offset = len(x_natural)
    if use_directional_derivatives:
        derivative_sum = 0.0
        derivative_sum_squares = 0.0
        derivative_value_count = 0
        for start in range(0, training_pair_count, 512):
            stop = min(start + 512, training_pair_count)
            values = (
                train_generated_y[2 * start : 2 * stop : 2].astype(np.float64)
                - train_generated_y[2 * start + 1 : 2 * stop : 2].astype(np.float64)
            ) / (2.0 * directional_difference_step)
            derivative_sum += float(values.sum())
            derivative_sum_squares += float(np.square(values).sum())
            derivative_value_count += values.size
        derivative_mean = derivative_sum / max(derivative_value_count, 1)
        derivative_target_scale = max(
            derivative_sum_squares / max(derivative_value_count, 1)
            - derivative_mean * derivative_mean,
            1e-8,
        )
    else:
        derivative_target_scale = 1.0
    del train_generated_x, train_generated_y
    train_provenance = np.concatenate([
        np.full(len(x_natural), "natural_observed_model_inference", dtype="<U48"),
        np.full(
            training_pair_points,
            (
                "generated_finite_difference_oracle_probe"
                if use_directional_derivatives
                else "generated_radial_shell_oracle_probe"
            ),
            dtype="<U48",
        ),
        np.full(
            training_standard_queries,
            "generated_radial_shell_oracle_probe",
            dtype="<U48",
        ),
    ])
    generated_adaptive_count = 0
    adaptive_rounds: list[dict[str, Any]] = []
    cpu_generator = torch.Generator(device="cpu").manual_seed(seed)
    student: _Student | None = None
    selected_state: dict[str, torch.Tensor] | None = None
    selected_width = 0
    selected_score = float("inf")
    selected_validation_nmse = float("inf")
    selected_directional_validation_nmse: float | None = None
    stage_results: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats(device)

    for stage_index, width in enumerate(_widths(max_components)):
        if student is None:
            student = _Student(input_dim, output_dim, width, cpu_generator, device)
        else:
            _grow_by_splitting(student, cpu_generator, split_perturbation)
        if student.components != width:
            raise RuntimeError(f"continuation width mismatch: {student.components} != {width}")

        x_train = torch.from_numpy(train_x)
        y_train = torch.from_numpy(train_y)
        target_scale = torch.tensor(
            max(float(np.var(train_y.astype(np.float64))), 1e-8),
            dtype=torch.float32,
            device=device,
        )
        derivative_scale = torch.tensor(
            derivative_target_scale, dtype=torch.float32, device=device
        )
        stage_ridges = [_project_down(
            student, train_x, train_y, device,
            batch_size=projection_batch_size, ridge=projection_ridge,
        )]
        optimizer = torch.optim.Adam(
            [student.gate, student.up, student.down], lr=learning_rate
        )
        for step in range(steps_per_stage):
            if step and step % projection_interval == 0:
                stage_ridges.append(_project_down(
                    student, train_x, train_y, device,
                    batch_size=projection_batch_size, ridge=projection_ridge,
                ))
                optimizer = torch.optim.Adam(
                    [student.gate, student.up, student.down], lr=learning_rate
                )
            sample_count = min(batch_size, len(x_train))
            natural_batch = min(
                len(x_natural),
                max(1, min(sample_count - 1, round(
                    sample_count * natural_training_batch_fraction
                ))),
            )
            generated_batch = sample_count - natural_batch
            natural_indices = torch.randint(
                len(x_natural), (natural_batch,), generator=cpu_generator
            )
            generated_indices = torch.randint(
                len(x_train) - len(x_natural),
                (generated_batch,),
                generator=cpu_generator,
            ) + len(x_natural)
            indices = torch.cat([natural_indices, generated_indices])
            indices = indices[torch.randperm(len(indices), generator=cpu_generator)]
            x = x_train.index_select(0, indices).to(device, non_blocking=True)
            y = y_train.index_select(0, indices).to(device, non_blocking=True)
            prediction = student(x)
            loss = (prediction - y).square().mean() / target_scale
            if use_directional_derivatives:
                derivative_batch_size = min(
                    max(1, batch_size // 2), training_pair_count
                )
                pair_indices = torch.randint(
                    training_pair_count,
                    (derivative_batch_size,),
                    generator=cpu_generator,
                )
                row_indices = pair_indices * 2 + train_pair_row_offset
                plus_x = x_train.index_select(0, row_indices).to(
                    device, non_blocking=True
                )
                minus_x = x_train.index_select(0, row_indices + 1).to(
                    device, non_blocking=True
                )
                plus_y = y_train.index_select(0, row_indices).to(
                    device, non_blocking=True
                )
                minus_y = y_train.index_select(0, row_indices + 1).to(
                    device, non_blocking=True
                )
                derivative_inputs = (plus_x + minus_x) * 0.5
                derivative_targets = (
                    plus_y - minus_y
                ) / (2.0 * directional_difference_step)
                directions = torch.from_numpy(train_directions).index_select(
                    0, pair_indices
                ).to(device, non_blocking=True)
                derivative_prediction = student.directional_derivative(
                    derivative_inputs, directions
                )
                loss = loss + directional_derivative_weight * (
                    derivative_prediction - derivative_targets
                ).square().mean() / derivative_scale
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite continuation loss at width {width}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [student.gate, student.up, student.down], max_norm=5.0
            )
            optimizer.step()

        stage_ridges.append(_project_down(
            student, train_x, train_y, device,
            batch_size=projection_batch_size, ridge=projection_ridge,
        ))
        validation_prediction = _forward_chunks(student, validation_x, device, 512)
        validation_metrics = _metrics(validation_y, validation_prediction)
        if use_directional_derivatives:
            validation_centers = (
                validation_x[:validation_pair_points:2]
                + validation_x[1:validation_pair_points:2]
            ) * 0.5
            validation_derivative_targets = (
                validation_y[:validation_pair_points:2]
                - validation_y[1:validation_pair_points:2]
            ) / (2.0 * directional_difference_step)
            directional_validation_metrics = _metrics(
                validation_derivative_targets,
                _directional_derivative_chunks(
                    student,
                    validation_centers,
                    validation_directions,
                    device,
                    512,
                ),
            )
            stage_selection_score = (
                validation_metrics["normalizedMse"]
                + directional_derivative_weight
                * directional_validation_metrics["normalizedMse"]
            )
        else:
            directional_validation_metrics = None
            stage_selection_score = validation_metrics["normalizedMse"]
        shell_errors = []
        for shell_index in range(4):
            mask = validation_shells == shell_index
            shell_errors.append(
                _metrics(validation_y[mask], validation_prediction[mask])["normalizedMse"]
                if mask.any() else None
            )
        stage_results.append({
            "candidateComponents": width,
            "validation": validation_metrics,
            "directionalDerivativeValidation": directional_validation_metrics,
            "selectionScore": stage_selection_score,
            "validationShellNormalizedMse": shell_errors,
            "trainingObservations": int(len(train_x)),
            "maximumEffectiveProjectionRidge": max(stage_ridges),
        })
        if stage_selection_score < selected_score:
            selected_score = stage_selection_score
            selected_validation_nmse = validation_metrics["normalizedMse"]
            selected_directional_validation_nmse = (
                None
                if directional_validation_metrics is None
                else directional_validation_metrics["normalizedMse"]
            )
            selected_width = width
            selected_state = {
                name: value.detach().cpu().clone()
                for name, value in student.state_dict().items()
            }

        if (
            adaptive_batch_queries
            and stage_index in {2, 4, 6, 8}
            and spent + adaptive_batch_queries <= query_budget
        ):
            valid_shells = [
                (float(error), index)
                for index, error in enumerate(shell_errors)
                if error is not None
            ]
            selected_shell = max(valid_shells)[1] if valid_shells else 0
            adaptive_x = draw_shell(selected_shell, adaptive_batch_queries)
            adaptive_y = query(adaptive_x, f"adaptive-shell-{selected_shell}")
            train_x = np.concatenate([train_x, adaptive_x])
            train_y = np.concatenate([train_y, adaptive_y])
            train_provenance = np.concatenate([
                train_provenance,
                np.full(adaptive_batch_queries, "adaptive_residual_shell_oracle_probe", dtype="<U48"),
            ])
            generated_adaptive_count += adaptive_batch_queries
            adaptive_rounds.append({
                "afterCandidateComponents": width,
                "selectedShell": selected_shell,
                "oraclePoints": adaptive_batch_queries,
                "validationNormalizedMseByShell": shell_errors,
            })
        del x_train, y_train

    if selected_state is None:
        raise RuntimeError("continuation produced no candidate model")
    selected = _Student(
        input_dim,
        output_dim,
        selected_width,
        cpu_generator,
        device,
    )
    selected.load_state_dict(selected_state)
    selected.eval()
    prediction = _forward_chunks(selected, validation_x, device, 512)
    if use_directional_derivatives:
        validation_centers = (
            validation_x[:validation_pair_points:2]
            + validation_x[1:validation_pair_points:2]
        ) * 0.5
        validation_derivative_targets = (
            validation_y[:validation_pair_points:2]
            - validation_y[1:validation_pair_points:2]
        ) / (2.0 * directional_difference_step)
        directional_validation = _metrics(
            validation_derivative_targets,
            _directional_derivative_chunks(
                selected, validation_centers, validation_directions, device, 512
            ),
        )
    else:
        directional_validation = None
    torch.cuda.synchronize(device)
    peak_vram = int(torch.cuda.max_memory_allocated(device))
    component_strength = (
        F.silu(F.linear(
            torch.as_tensor(validation_x[: min(2048, len(validation_x))], device=device),
            selected.gate,
        ))
        * F.linear(
            torch.as_tensor(validation_x[: min(2048, len(validation_x))], device=device),
            selected.up,
        )
    ).square().mean(dim=0).sqrt() * selected.down.norm(dim=0)
    result: dict[str, Any] = {
        "schemaVersion": 1,
        "kind": "gcl-oracle-only-swiglu-split-continuation",
        "algorithm": (
            "function-preserving-unit-splitting-with-streamed-variable-projection"
            + (
                "-and-oracle-jvp-supervision"
                if use_directional_derivatives
                else ""
            )
        ),
        "solverAccess": "oracle-only",
        "weightsUsedBySolver": False,
        "hooksUsedBySolver": False,
        "hiddenUnitCountProvidedToSolver": False,
        "teacherComponentIdentitiesProvidedToSolver": False,
        "teacherComponentCountInferred": False,
        "inputDim": input_dim,
        "outputDim": output_dim,
        "publicActivationFamily": "silu-swiglu",
        "biases": False,
        "candidateWidthPolicy": "geometric-function-preserving-splitting-selected-by-held-out-oracle-nmse",
        "trainingConfig": {
            "stepsPerStage": steps_per_stage,
            "batchSize": batch_size,
            "learningRate": learning_rate,
            "splitPerturbation": split_perturbation,
            "projectionRidge": projection_ridge,
            "projectionInterval": projection_interval,
            "projectionBatchSize": projection_batch_size,
            "targetNaturalTrainingBatchFraction": natural_training_batch_fraction,
            "directionalDerivativeWeight": directional_derivative_weight,
            "directionalDifferenceStep": directional_difference_step,
            "directionalDerivativeQueryFraction": (
                directional_derivative_query_fraction
            ),
        },
        "probeDistribution": {
            "sampler": "natural-mean-preserving-shrinkage-covariance-gaussian",
            "naturalRecoverySamples": int(len(x_natural)),
            "empiricalCovarianceRankBound": int(covariance_rank),
            "isotropicShrinkage": 0.15,
            "radialShellMultipliers": shells.tolist(),
        },
        "selectedCandidateComponents": selected_width,
        "studentWidthSelectedByMeasurements": True,
        "selectionScore": selected_score,
        "selectionScoreMetric": (
            "functional_nmse_plus_weighted_directional_derivative_nmse"
            if use_directional_derivatives
            else "functional_nmse"
        ),
        "selectedCandidateValidationNormalizedMse": selected_validation_nmse,
        "selectedCandidateValidationDirectionalDerivativeNormalizedMse": (
            selected_directional_validation_nmse
        ),
        "candidateStages": stage_results,
        "adaptiveQueryRounds": adaptive_rounds,
        "queryAccounting": {
            "oraclePoints": spent,
            "oracleQueryBudget": query_budget,
            "reusedRecoveryObservations": int(len(x_natural)),
            "initialGeneratedTrainingQueries": training_queries,
            "validationQueries": validation_queries,
            "adaptiveQueries": generated_adaptive_count,
            "directionalDerivativePairs": training_pair_count,
            "directionalDerivativeProbePointsReusedForOutputTraining": (
                training_pair_points
            ),
            "directionalDerivativeQueriesAreSubsetOfTrainingQueries": (
                use_directional_derivatives
            ),
        },
        "validation": _metrics(validation_y, prediction),
        "directionalDerivativeValidation": directional_validation,
        "trainingSeconds": time.perf_counter() - started,
        "performance": {
            "device": str(device),
            "cudaDeviceName": torch.cuda.get_device_name(device),
            "peakAllocatedVramBytes": peak_vram,
            "solverProcessPeakRssBytes": _proc_vm_hwm_bytes(),
            "rssMeasurementSource": "/proc/self/status:VmHWM",
            "torchVersion": torch.__version__,
            "torchCudaVersion": torch.version.cuda,
            "projectionBatchSize": projection_batch_size,
            "projectionInterval": projection_interval,
        },
        "trainingData": {
            "inputs": train_x.astype(np.float32, copy=False),
            "outputs": train_y.astype(np.float32, copy=False),
            "provenance": train_provenance,
            **{
                f"natural_{key}": np.concatenate([
                    value,
                    np.full(
                        (training_queries + generated_adaptive_count, *value.shape[1:]),
                        -1 if np.issubdtype(value.dtype, np.number) else "",
                        dtype=value.dtype,
                    ),
                ])
                for key, value in (recovery_metadata or {}).items()
            },
        },
        "validationData": {
            "inputs": validation_x.astype(np.float32, copy=False),
            "outputs": validation_y.astype(np.float32, copy=False),
            "shells": validation_shells,
            "provenance": (
                np.concatenate([
                        np.full(
                            validation_pair_points,
                            "generated_finite_difference_validation_probe",
                            dtype="<U48",
                        ),
                        np.full(
                            validation_standard_queries,
                            "generated_radial_shell_validation_probe",
                            dtype="<U48",
                        ),
                    ])
                if use_directional_derivatives
                else np.full(
                    validation_queries,
                    "generated_radial_shell_validation_probe",
                    dtype="<U48",
                )
            ),
        },
        "directionalDerivativeData": (
            {
                "trainingDirections": train_directions,
                "trainingPlusRowIndices": (
                    train_pair_row_offset + 2 * np.arange(training_pair_count)
                ).astype(np.int64),
                "trainingMinusRowIndices": (
                    train_pair_row_offset + 2 * np.arange(training_pair_count) + 1
                ).astype(np.int64),
                "validationDirections": validation_directions,
                "validationPlusRowIndices": (
                    2 * np.arange(validation_pair_count)
                ).astype(np.int64),
                "validationMinusRowIndices": (
                    2 * np.arange(validation_pair_count) + 1
                ).astype(np.int64),
                "finiteDifferenceStep": np.asarray(
                    [directional_difference_step], dtype=np.float32
                ),
                "provenance": np.asarray(
                    ["generated_oracle_central_difference_jvp"], dtype="<U48"
                ),
            }
            if use_directional_derivatives
            else None
        ),
        "reconstruction": {
            "gate": selected.gate.detach().cpu().numpy().astype(np.float32),
            "up": selected.up.detach().cpu().numpy().astype(np.float32),
            "down": selected.down.detach().cpu().numpy().astype(np.float32),
        },
        "componentStrengthRms": component_strength.detach().cpu().numpy().astype(np.float32),
    }
    return result
