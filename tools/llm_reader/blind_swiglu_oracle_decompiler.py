#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def silu_np(values: np.ndarray) -> np.ndarray:
    return values / (1.0 + np.exp(-values))


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise RuntimeError("cannot normalize near-zero vector")
    return vector / norm


def parse_units(unit: int, units: str | None) -> list[int]:
    if units is None or not units.strip():
        return [unit]
    parsed = [int(item.strip()) for item in units.split(",") if item.strip()]
    if not parsed:
        raise RuntimeError("--units did not contain any unit ids")
    return parsed


def dequantized_weight(module: Any) -> Any:
    weight = module.weight
    quant_state = getattr(weight, "quant_state", None)
    if quant_state is None:
        return weight.detach()
    import bitsandbytes.functional as bnb_functional

    return bnb_functional.dequantize_4bit(weight.data, quant_state).detach()


def oracle_server(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.utils import logging as transformers_logging

    transformers_logging.disable_progress_bar()
    config = AutoConfig.from_pretrained(args.model, local_files_only=args.local_files_only)
    model_type = str(getattr(config, "model_type", ""))
    if model_type not in {"mistral", "llama"}:
        raise RuntimeError(f"blind SwiGLU oracle expected mistral/llama model_type, got {model_type!r}")

    if getattr(config, "quantization_config", None):
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=args.local_files_only,
            device_map=args.device,
        )
    else:
        dtype = torch.float16 if args.device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=args.local_files_only,
            dtype=dtype,
        )
        model.to(args.device)
    model.eval()

    layers = model.model.layers
    if args.layer < 0 or args.layer >= len(layers):
        raise RuntimeError(f"layer {args.layer} outside model range 0..{len(layers) - 1}")
    mlp = layers[args.layer].mlp
    gate_weight = dequantized_weight(mlp.gate_proj).detach().float().cpu()
    up_weight = dequantized_weight(mlp.up_proj).detach().float().cpu()
    units = parse_units(args.unit, args.units)
    for unit in units:
        if unit < 0 or unit >= gate_weight.shape[0]:
            raise RuntimeError(f"unit {unit} outside SwiGLU intermediate range 0..{gate_weight.shape[0] - 1}")
    gate = gate_weight[units].contiguous()
    up = up_weight[units].contiguous()
    hidden_size = int(gate.shape[1])
    gate_norm = torch.linalg.vector_norm(gate, dim=1, keepdim=True).clamp_min(1e-12)
    up_norm = torch.linalg.vector_norm(up, dim=1, keepdim=True).clamp_min(1e-12)
    gate_unit = (gate / gate_norm).numpy().astype(np.float32)
    up_unit = (up / up_norm).numpy().astype(np.float32)

    print(json.dumps({
        "type": "ready",
        "model": args.model,
        "modelType": model_type,
        "layer": args.layer,
        "unit": args.unit,
        "units": units,
        "unitCount": len(units),
        "hiddenSize": hidden_size,
        "intermediateSize": int(gate_weight.shape[0]),
        "device": args.device,
        "oracleQuantizationStep": args.oracle_quantization_step,
        "oracleOwnsWeights": True,
        "solverAccess": "oracle-only"
    }), flush=True)

    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        command = request.get("cmd")
        if command == "eval_npy":
            input_path = Path(request["inputPath"])
            output_path = Path(request["outputPath"])
            values = np.load(input_path).astype(np.float32, copy=False)
            if values.ndim != 2 or values.shape[1] != hidden_size:
                raise RuntimeError(f"oracle input shape {values.shape} does not match hidden size {hidden_size}")
            outputs: list[np.ndarray] = []
            batch_size = max(1, int(request.get("batchSize", args.batch_size)))
            with torch.no_grad():
                for start in range(0, values.shape[0], batch_size):
                    batch = torch.from_numpy(values[start:start + batch_size])
                    gate_projection = batch @ gate.T
                    up_projection = batch @ up.T
                    outputs.append((torch.nn.functional.silu(gate_projection) * up_projection).sum(dim=1).cpu().numpy())
            result = np.concatenate(outputs).astype(np.float64)
            if args.oracle_quantization_step > 0.0:
                result = np.round(result / args.oracle_quantization_step) * args.oracle_quantization_step
            np.save(output_path, result.astype(np.float32))
            print(json.dumps({"type": "ok", "count": int(values.shape[0])}), flush=True)
        elif command == "truth_npy":
            gate_path = Path(request["gatePath"])
            up_path = Path(request["upPath"])
            np.save(gate_path, gate_unit)
            np.save(up_path, up_unit)
            print(json.dumps({"type": "ok", "truth": True}), flush=True)
        elif command == "stop":
            print(json.dumps({"type": "bye"}), flush=True)
            return 0
        else:
            raise RuntimeError(f"unknown oracle command {command!r}")
    return 0


class OracleClient:
    def __init__(self, args: argparse.Namespace, work_dir: Path) -> None:
        self.stderr_path = work_dir / "oracle-stderr.log"
        self.stderr_file = self.stderr_path.open("w", encoding="utf-8", buffering=1)
        command = [
            sys.executable,
            __file__,
            "--oracle-server",
            "--model",
            args.model,
            "--layer",
            str(args.layer),
            "--unit",
            str(args.unit),
            "--device",
            args.device,
            "--batch-size",
            str(args.oracle_batch_size),
        ]
        if args.units:
            command.extend(["--units", args.units])
        if args.oracle_quantization_step > 0.0:
            command.extend(["--oracle-quantization-step", str(args.oracle_quantization_step)])
        if args.local_files_only:
            command.append("--local-files-only")
        self.process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[2],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_file,
            text=True,
            bufsize=1,
        )
        assert self.process.stdout is not None
        ready_line = self.process.stdout.readline()
        if not ready_line:
            raise RuntimeError(f"oracle did not become ready; exit={self.process.poll()} stderr={self.stderr_tail()}")
        ready = json.loads(ready_line)
        if ready.get("type") != "ready":
            raise RuntimeError(f"oracle did not become ready: {ready}")
        self.meta = ready
        self.work_dir = work_dir
        self.query_count = 0
        self._counter = 0

    def stderr_tail(self, max_chars: int = 4000) -> str:
        try:
            self.stderr_file.flush()
            text = self.stderr_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return text[-max_chars:]

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        if self.process.poll() is not None:
            raise RuntimeError(f"oracle process exited before request {payload.get('cmd')!r}; exit={self.process.returncode} stderr={self.stderr_tail()}")
        try:
            self.process.stdin.write(json.dumps(payload) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise RuntimeError(f"oracle pipe failed during request {payload.get('cmd')!r}; exit={self.process.poll()} stderr={self.stderr_tail()}") from error
        response_line = self.process.stdout.readline()
        if not response_line:
            raise RuntimeError(f"oracle closed stdout during request {payload.get('cmd')!r}; exit={self.process.poll()} stderr={self.stderr_tail()}")
        response = json.loads(response_line)
        if response.get("type") not in {"ok", "bye"}:
            raise RuntimeError(f"oracle request failed: {response}")
        return response

    def eval(self, inputs: np.ndarray) -> np.ndarray:
        self._counter += 1
        input_path = self.work_dir / f"oracle-input-{self._counter}.npy"
        output_path = self.work_dir / f"oracle-output-{self._counter}.npy"
        np.save(input_path, inputs.astype(np.float32, copy=False))
        response = self.request({
            "cmd": "eval_npy",
            "inputPath": str(input_path),
            "outputPath": str(output_path),
        })
        self.query_count += int(response["count"])
        return np.load(output_path).astype(np.float64)

    def truth(self) -> tuple[np.ndarray, np.ndarray]:
        gate_path = self.work_dir / "truth-gate.npy"
        up_path = self.work_dir / "truth-up.npy"
        self.request({
            "cmd": "truth_npy",
            "gatePath": str(gate_path),
            "upPath": str(up_path),
        })
        return np.load(gate_path).astype(np.float64), np.load(up_path).astype(np.float64)

    def close(self) -> None:
        try:
            if self.process.poll() is None:
                self.request({"cmd": "stop"})
        finally:
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.stderr_file.close()


def estimate_gradient(
    oracle: OracleClient,
    rng: np.random.Generator,
    base: np.ndarray,
    direction_count: int,
    step: float,
) -> np.ndarray:
    dimension = base.shape[0]
    directions = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(direction_count, dimension))
    directions /= math.sqrt(dimension)
    plus = base[None, :] + step * directions
    minus = base[None, :] - step * directions
    values = oracle.eval(np.vstack([plus, minus]))
    directional = (values[:direction_count] - values[direction_count:]) / (2.0 * step)
    return dimension * np.mean(directional[:, None] * directions, axis=0)


def estimate_coordinate_gradient(
    oracle: OracleClient,
    base: np.ndarray,
    step: float,
) -> np.ndarray:
    dimension = base.shape[0]
    eye = np.eye(dimension, dtype=np.float32)
    plus = base[None, :] + step * eye
    minus = base[None, :] - step * eye
    values = oracle.eval(np.vstack([plus, minus]))
    return (values[:dimension] - values[dimension:]) / (2.0 * step)


def estimate_forward_coordinate_gradient(
    oracle: OracleClient,
    base: np.ndarray,
    step: float,
) -> np.ndarray:
    dimension = base.shape[0]
    eye = np.eye(dimension, dtype=np.float32)
    shifted = base[None, :] + step * eye
    values = oracle.eval(np.vstack([base[None, :], shifted]))
    return (values[1:] - values[0]) / step


def parse_step_list(raw: str | None, fallback: float) -> list[float]:
    if raw is None or not raw.strip():
        return [float(fallback)]
    steps = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if not steps:
        return [float(fallback)]
    if any((not np.isfinite(step)) or step <= 0.0 for step in steps):
        raise RuntimeError(f"finite-difference steps must be positive finite values: {raw!r}")
    return steps


def parse_float_list(raw: str | None) -> list[float]:
    if raw is None or not raw.strip():
        return []
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if any((not np.isfinite(value)) or value <= 0.0 for value in values):
        raise RuntimeError(f"expected positive finite float list, got {raw!r}")
    return values


def parse_init_scale_schedule(raw: str | None, fallback: float) -> list[float]:
    values = parse_float_list(raw)
    return values if values else [fallback]


def sample_scale_mixture_inputs(
    rng: np.random.Generator,
    count: int,
    dimension: int,
    fallback_scale: float,
    scale_mixture: list[float],
) -> np.ndarray:
    scales = scale_mixture if scale_mixture else [fallback_scale]
    chunks: list[np.ndarray] = []
    remaining = count
    for index, scale in enumerate(scales):
        chunk_count = remaining if index == len(scales) - 1 else count // len(scales)
        remaining -= chunk_count
        if chunk_count > 0:
            chunks.append(rng.normal(0.0, scale, size=(chunk_count, dimension)).astype(np.float32))
    if not chunks:
        return np.zeros((0, dimension), dtype=np.float32)
    values = np.vstack(chunks)
    rng.shuffle(values, axis=0)
    return values.astype(np.float32, copy=False)


def recover_basis(gradients: np.ndarray, rank: int, moment: str = "auto") -> tuple[np.ndarray, np.ndarray, float]:
    # Centering n gradient witnesses caps the covariance rank at n - 1. For the
    # low-query path where n == rank, use raw second moments so a rank-2 SwiGLU
    # channel can still expose two independent active directions.
    if moment == "raw" or (moment == "auto" and gradients.shape[0] <= rank):
        basis_source = gradients
    elif moment in {"auto", "centered"}:
        basis_source = gradients - np.mean(gradients, axis=0, keepdims=True)
    else:
        raise RuntimeError(f"unknown basis moment mode {moment!r}")
    matrix = basis_source.T @ basis_source / max(1, basis_source.shape[0])
    values, vectors = np.linalg.eigh(matrix)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    basis = vectors[:, :rank].T
    for index in range(rank):
        basis[index] = normalize(basis[index])
    total = float(np.sum(np.maximum(values, 0.0)))
    selected = float(np.sum(np.maximum(values[:rank], 0.0)))
    return basis, values[:rank], selected / max(total, 1e-12)


def estimate_active_forward_gradients(
    oracle: OracleClient,
    rng: np.random.Generator,
    basis: np.ndarray,
    count: int,
    input_scale: float,
    steps: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    rank = basis.shape[0]
    if count <= 0:
        return np.zeros((0, rank), dtype=np.float64), np.zeros((0, rank), dtype=np.float64)
    if not steps:
        raise RuntimeError("active-gradient estimation requires at least one positive step")
    coords = rng.normal(0.0, input_scale, size=(count, rank)).astype(np.float32)
    base_inputs = coords @ basis
    step_gradients = []
    for step in steps:
        shifted_coords = np.repeat(coords[:, None, :], rank, axis=1)
        for index in range(rank):
            shifted_coords[:, index, index] += step
        shifted_inputs = shifted_coords.reshape(count * rank, rank) @ basis
        values = oracle.eval(np.vstack([base_inputs, shifted_inputs]).astype(np.float32))
        base_values = values[:count]
        shifted_values = values[count:].reshape(count, rank)
        step_gradients.append((shifted_values - base_values[:, None]) / step)
    active_gradients = np.mean(np.stack(step_gradients, axis=0), axis=0)
    return coords.astype(np.float64), active_gradients.astype(np.float64)


def estimate_active_cubic_directions(
    oracle: OracleClient,
    rng: np.random.Generator,
    basis: np.ndarray,
    count: int,
    step: float,
) -> tuple[np.ndarray, np.ndarray]:
    if count <= 0:
        return np.zeros((0, basis.shape[0]), dtype=np.float64), np.zeros(0, dtype=np.float64)
    if step <= 0.0 or not np.isfinite(step):
        raise RuntimeError("cubic direction probe step must be positive and finite")
    rank = basis.shape[0]
    directions = rng.normal(0.0, 1.0, size=(count, rank)).astype(np.float64)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True).clip(min=1e-12)
    multipliers = np.array([2.0, 1.0, -1.0, -2.0], dtype=np.float64)
    active_points = (multipliers[:, None, None] * step * directions[None, :, :]).reshape(count * 4, rank)
    values = oracle.eval((active_points @ basis).astype(np.float32)).reshape(4, count)
    # Central third directional derivative stencil:
    # f'''(0) ~= (f(2h) - 2f(h) + 2f(-h) - f(-2h)) / (2h^3)
    cubic_scores = (values[0] - 2.0 * values[1] + 2.0 * values[2] - values[3]) / (2.0 * step ** 3)
    order = np.argsort(np.abs(cubic_scores))[::-1]
    return directions[order].astype(np.float64), cubic_scores[order].astype(np.float64)


def make_quadratic_atom_starts(
    coords: np.ndarray,
    targets: np.ndarray,
    seed: int,
    max_starts: int,
) -> list[tuple[np.ndarray, np.ndarray, float, float]]:
    rank = coords.shape[1]
    if rank < 3 or coords.shape[0] < rank * 8:
        return []
    rng = np.random.default_rng(seed)
    sample_count = min(coords.shape[0], max(2048, rank * 96))
    if sample_count < coords.shape[0]:
        sample_indices = np.sort(rng.choice(coords.shape[0], size=sample_count, replace=False))
        xs = coords[sample_indices]
        ys = targets[sample_indices]
    else:
        xs = coords
        ys = targets
    ys = ys.astype(np.float64) - float(np.mean(ys))
    columns: list[np.ndarray] = []
    pairs: list[tuple[int, int]] = []
    for left in range(rank):
        for right in range(left, rank):
            feature = xs[:, left] * xs[:, right]
            if left != right:
                feature = 2.0 * feature
            columns.append(feature)
            pairs.append((left, right))
    design = np.column_stack(columns)
    ridge = 1e-8 * np.eye(design.shape[1], dtype=np.float64)
    try:
        coeffs = np.linalg.solve(design.T @ design + ridge, design.T @ ys)
    except np.linalg.LinAlgError:
        try:
            coeffs, *_ = np.linalg.lstsq(design, ys, rcond=None)
        except np.linalg.LinAlgError:
            return []
    if not np.all(np.isfinite(coeffs)):
        return []
    quadratic = np.zeros((rank, rank), dtype=np.float64)
    for value, (left, right) in zip(coeffs, pairs):
        quadratic[left, right] = value
        quadratic[right, left] = value
    try:
        values, vectors = np.linalg.eigh(quadratic)
    except np.linalg.LinAlgError:
        return []
    order = np.argsort(np.abs(values))[::-1]
    basis_vectors = [normalize(vectors[:, index]) for index in order[:min(rank, 10)] if abs(values[index]) > 1e-12]
    if len(basis_vectors) < 2:
        return []
    gate_scales = np.array([0.2, 0.35, 0.55, 0.85, 1.25, 1.8, 2.6], dtype=np.float64)
    starts: list[tuple[float, np.ndarray, np.ndarray, float, float]] = []
    ones = np.ones(xs.shape[0], dtype=np.float64)
    raw_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for i, first in enumerate(basis_vectors[:8]):
        for second in basis_vectors[i + 1:8]:
            for sign in (1.0, -1.0):
                raw_pairs.append((first + sign * second, first - sign * second))
                raw_pairs.append((first - sign * second, first + sign * second))
    for vector in basis_vectors[:8]:
        for other in basis_vectors[:8]:
            if vector is other:
                continue
            raw_pairs.append((vector, other))
    seen: set[tuple[float, ...]] = set()
    for gate_raw, up_raw in raw_pairs:
        try:
            gate_dir = normalize(gate_raw)
            up_dir = normalize(up_raw)
        except RuntimeError:
            continue
        signature = tuple(np.round(np.concatenate([gate_dir, up_dir]), 3).tolist())
        if signature in seen:
            continue
        seen.add(signature)
        up_projection = xs @ up_dir
        for gate_scale in gate_scales:
            gate = gate_dir * gate_scale
            feature = silu_np(xs @ gate) * up_projection
            design_one = np.column_stack([feature, ones])
            try:
                solved, *_ = np.linalg.lstsq(design_one, targets[sample_indices] if sample_count < coords.shape[0] else targets, rcond=None)
            except np.linalg.LinAlgError:
                continue
            if not np.all(np.isfinite(solved)):
                continue
            predicted = design_one @ solved
            target_values = targets[sample_indices] if sample_count < coords.shape[0] else targets
            mae = float(np.mean(np.abs(predicted - target_values)))
            starts.append((mae, gate, up_dir, float(solved[0]), float(solved[1])))
    starts.sort(key=lambda item: item[0])
    return [(gate, up, scale, bias) for _mae, gate, up, scale, bias in starts[:max_starts]]


def make_quadratic_multi_structured_starts(
    coords: np.ndarray,
    targets: np.ndarray,
    atom_count: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    if atom_count < 4:
        return []
    atom_starts = make_quadratic_atom_starts(
        coords,
        targets,
        seed + 811,
        max_starts=max(atom_count * 16, 128),
    )
    if len(atom_starts) < atom_count:
        return []
    rng = np.random.default_rng(seed + 812)
    gate_pool = [gate for gate, _up, _scale, _bias in atom_starts]
    starts: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []

    def add_selected(selected_gates: list[np.ndarray]) -> None:
        if len(selected_gates) != atom_count:
            return
        solved = solve_fixed_gate_start(coords, targets, np.vstack(selected_gates))
        if solved is None:
            return
        gates, ups, scales, bias, _mae = solved
        starts.append((gates, ups, scales, bias))

    for max_abs_cosine in (0.999, 0.995, 0.985, 0.965):
        selected: list[np.ndarray] = []
        for gate in gate_pool:
            direction = canonical_direction(gate)
            if direction is None:
                continue
            if any(abs(float(np.dot(direction, existing / max(1e-9, np.linalg.norm(existing))))) >= max_abs_cosine for existing in selected):
                continue
            selected.append(gate)
            if len(selected) >= atom_count:
                break
        add_selected(selected)

    top_pool = gate_pool[:min(len(gate_pool), max(atom_count * 8, 96))]
    for stride in (2, 3, 5, 7):
        selected = []
        for index in range(stride - 1, len(top_pool), stride):
            gate = top_pool[index]
            direction = canonical_direction(gate)
            if direction is None:
                continue
            if any(abs(float(np.dot(direction, existing / max(1e-9, np.linalg.norm(existing))))) >= 0.985 for existing in selected):
                continue
            selected.append(gate)
            if len(selected) >= atom_count:
                break
        add_selected(selected)

    for _restart_index in range(8):
        selected = []
        for gate in rng.permutation(top_pool):
            direction = canonical_direction(gate)
            if direction is None:
                continue
            if any(abs(float(np.dot(direction, existing / max(1e-9, np.linalg.norm(existing))))) >= 0.99 for existing in selected):
                continue
            selected.append(gate)
            if len(selected) >= atom_count:
                break
        add_selected(selected)

    if not starts:
        return []
    scored: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, float]] = []
    for gates, ups, scales, bias in starts:
        prediction = predict_swiglu(coords, {
            "gates": gates,
            "ups": ups,
            "scales": scales,
            "bias": bias,
        })
        mae = float(np.mean(np.abs(prediction - targets)))
        scored.append((mae, gates, ups, scales, bias))
    scored.sort(key=lambda item: item[0])
    unique: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    signatures: set[tuple[float, ...]] = set()
    for _mae, gates, ups, scales, bias in scored:
        signature_values = []
        for gate in gates:
            direction = canonical_direction(gate)
            if direction is not None:
                signature_values.extend(np.round(direction, 3).tolist())
        signature = tuple(signature_values)
        if signature in signatures:
            continue
        signatures.add(signature)
        unique.append((gates, ups, scales, bias))
        if len(unique) >= 8:
            break
    return unique


def fit_swiglu_atom(
    coords: np.ndarray,
    targets: np.ndarray,
    seed: int,
    steps: int,
    learning_rate: float,
) -> dict[str, Any]:
    import torch

    torch.manual_seed(seed)
    x = torch.tensor(coords, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)
    rank = coords.shape[1]
    target_scale = float(np.std(targets)) or 1.0
    target_variance = float(np.mean((targets - np.mean(targets)) ** 2))
    best: dict[str, Any] | None = None
    structured_starts: list[tuple[np.ndarray, np.ndarray, float, float]] = []
    if rank == 2:
        angles = np.linspace(0.0, math.pi, 24, endpoint=False)
        gate_scales = np.array([0.15, 0.25, 0.4, 0.65, 1.0, 1.6, 2.5, 4.0], dtype=np.float64)
        scored: list[tuple[float, np.ndarray, np.ndarray, float, float]] = []
        ones = np.ones(coords.shape[0], dtype=np.float64)
        for gate_angle in angles:
            gate_dir = np.array([math.cos(gate_angle), math.sin(gate_angle)], dtype=np.float64)
            for up_angle in angles:
                up_dir = np.array([math.cos(up_angle), math.sin(up_angle)], dtype=np.float64)
                up_projection = coords @ up_dir
                for gate_scale in gate_scales:
                    feature = silu_np(gate_scale * (coords @ gate_dir)) * up_projection
                    design = np.column_stack([feature, ones])
                    try:
                        coeffs, *_ = np.linalg.lstsq(design, targets, rcond=None)
                    except np.linalg.LinAlgError:
                        continue
                    predicted = design @ coeffs
                    mae = float(np.mean(np.abs(predicted - targets)))
                    scored.append((mae, gate_scale * gate_dir, up_dir, float(coeffs[0]), float(coeffs[1])))
        scored.sort(key=lambda item: item[0])
        structured_starts = [(gate, up, scale, bias) for _, gate, up, scale, bias in scored[:12]]
    elif rank > 2:
        structured_starts = make_quadratic_atom_starts(
            coords,
            targets,
            seed + 707,
            max_starts=16,
        )
    base_starts = [
        (np.array([1.0, 0.0]), np.array([0.0, 1.0])),
        (np.array([0.0, 1.0]), np.array([1.0, 0.0])),
        (np.array([1.0, 1.0]), np.array([1.0, -1.0])),
        (np.array([1.0, -1.0]), np.array([1.0, 1.0])),
    ]
    restart_count = len(structured_starts) + 16
    for restart in range(restart_count):
        if restart < len(structured_starts):
            gate_init, up_init, scale_init, bias_init = structured_starts[restart]
            gate = torch.nn.Parameter(torch.tensor(gate_init, dtype=torch.float32))
            up = torch.nn.Parameter(torch.tensor(up_init, dtype=torch.float32))
            scale = torch.nn.Parameter(torch.tensor(scale_init, dtype=torch.float32))
            bias = torch.nn.Parameter(torch.tensor(bias_init, dtype=torch.float32))
        elif restart - len(structured_starts) < len(base_starts) and rank == 2:
            base_index = restart - len(structured_starts)
            gate_init, up_init = base_starts[base_index]
            gate = torch.nn.Parameter(torch.tensor(normalize(gate_init) * 0.75, dtype=torch.float32))
            up = torch.nn.Parameter(torch.tensor(normalize(up_init) * 0.75, dtype=torch.float32))
            scale = torch.nn.Parameter(torch.tensor(max(0.1, target_scale)))
            bias = torch.nn.Parameter(torch.tensor(float(np.mean(targets))))
        else:
            gate = torch.nn.Parameter(torch.randn(rank) * 0.8)
            up = torch.nn.Parameter(torch.randn(rank) * 0.8)
            scale = torch.nn.Parameter(torch.tensor(max(0.1, target_scale)))
            bias = torch.nn.Parameter(torch.tensor(float(np.mean(targets))))
        optimizer = torch.optim.Adam([gate, up, scale, bias], lr=learning_rate)
        for _ in range(steps):
            optimizer.zero_grad()
            prediction = scale * torch.nn.functional.silu(x @ gate) * (x @ up) + bias
            loss = torch.mean((prediction - y) ** 2)
            loss.backward()
            optimizer.step()
        lbfgs = torch.optim.LBFGS([gate, up, scale, bias], lr=0.8, max_iter=160, line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            lbfgs.zero_grad()
            prediction = scale * torch.nn.functional.silu(x @ gate) * (x @ up) + bias
            loss = torch.mean((prediction - y) ** 2)
            loss.backward()
            return loss

        try:
            lbfgs.step(closure)
        except RuntimeError:
            pass
        with torch.no_grad():
            feature = (torch.nn.functional.silu(x @ gate) * (x @ up)).detach().cpu().numpy().astype(np.float64)
            design = np.column_stack([feature, np.ones(coords.shape[0], dtype=np.float64)])
            try:
                coeffs, *_ = np.linalg.lstsq(design, targets, rcond=None)
                if np.all(np.isfinite(coeffs)):
                    scale_value = float(coeffs[0])
                    bias_value = float(coeffs[1])
                    prediction_np = design @ coeffs
                    mae = float(np.mean(np.abs(prediction_np - targets)))
                else:
                    raise np.linalg.LinAlgError("non-finite least-squares coefficients")
            except np.linalg.LinAlgError:
                prediction = scale * torch.nn.functional.silu(x @ gate) * (x @ up) + bias
                mae = torch.mean(torch.abs(prediction - y)).item()
                scale_value = float(scale.detach().cpu().item())
                bias_value = float(bias.detach().cpu().item())
            candidate = {
                "restart": restart,
                "mae": mae,
                "gate": gate.detach().cpu().numpy().astype(np.float64),
                "up": up.detach().cpu().numpy().astype(np.float64),
                "scale": scale_value,
                "bias": bias_value,
            }
            if best is None or candidate["mae"] < best["mae"]:
                best = candidate
    assert best is not None
    best["restartCount"] = restart_count
    return best


def fit_swiglu_sum_batched(
    coords: np.ndarray,
    targets: np.ndarray,
    atom_count: int,
    seed: int,
    steps: int,
    learning_rate: float,
    restart_count: int,
    structured_starts: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    gradient_coords: np.ndarray | None,
    gradient_targets: np.ndarray | None,
    gradient_weight: float,
    restart_chunk_size: int,
) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("batched SwiGLU fit requires CUDA")

    torch.manual_seed(seed)
    device = torch.device("cuda")
    fit_dtype = torch.float32
    x = torch.tensor(coords, dtype=fit_dtype, device=device)
    y = torch.tensor(targets, dtype=fit_dtype, device=device)
    rank = coords.shape[1]
    target_scale = float(np.std(targets)) or 1.0
    target_variance = float(np.mean((targets - np.mean(targets)) ** 2))
    has_gradient_witness = (
        gradient_coords is not None
        and gradient_targets is not None
        and gradient_weight > 0.0
        and gradient_coords.shape[0] > 0
    )
    if has_gradient_witness:
        gx = torch.tensor(gradient_coords, dtype=fit_dtype, device=device)
        gy = torch.tensor(gradient_targets, dtype=fit_dtype, device=device)
        target_grad_scale = torch.mean(gy ** 2).detach().clamp_min(1e-12)
    else:
        gx = None
        gy = None
        target_grad_scale = None

    best: dict[str, Any] | None = None
    top_candidates: list[dict[str, Any]] = []
    gate_candidate_pool: list[tuple[float, np.ndarray]] = []
    best_score_by_start_kind: dict[str, float] = {}
    best_metrics_by_start_kind: dict[str, dict[str, Any]] = {}
    selection_tail_weight = float(os.environ.get("GCL_BLIND_SWIGLU_SELECTION_TAIL_WEIGHT", "0.0") or "0.0")
    selection_tail_quantile = float(os.environ.get("GCL_BLIND_SWIGLU_SELECTION_TAIL_QUANTILE", "0.995") or "0.995")
    if not np.isfinite(selection_tail_weight) or selection_tail_weight < 0.0:
        raise RuntimeError("GCL_BLIND_SWIGLU_SELECTION_TAIL_WEIGHT must be non-negative and finite")
    if not np.isfinite(selection_tail_quantile) or selection_tail_quantile <= 0.0 or selection_tail_quantile >= 1.0:
        raise RuntimeError("GCL_BLIND_SWIGLU_SELECTION_TAIL_QUANTILE must be in (0, 1)")
    random_init_scales = parse_init_scale_schedule(
        os.environ.get("GCL_BLIND_SWIGLU_RANDOM_INIT_SCALES"),
        float(os.environ.get("GCL_BLIND_SWIGLU_RANDOM_INIT_SCALE", "0.6") or "0.6"),
    )
    structured_start_count = len(structured_starts)
    total_restart_count = restart_count + structured_start_count
    for gate_start, _up_start, _scale_start, _bias_start in structured_starts:
        for gate in gate_start:
            if np.all(np.isfinite(gate)) and np.linalg.norm(gate) > 1e-9:
                gate_candidate_pool.append((float("inf"), np.asarray(gate, dtype=np.float64)))
    ones = np.ones(coords.shape[0], dtype=np.float64)
    selection_fit_indices: np.ndarray | None = None
    selection_holdout_indices: np.ndarray | None = None
    if coords.shape[0] >= 1536:
        selection_rng = np.random.default_rng(seed + 90917)
        holdout_count = min(coords.shape[0] // 4, max(512, atom_count * 64))
        holdout_count = min(holdout_count, coords.shape[0] - max(512, atom_count * 64))
        if holdout_count > 0:
            permutation = selection_rng.permutation(coords.shape[0])
            selection_holdout_indices = np.sort(permutation[:holdout_count])
            selection_fit_indices = np.sort(permutation[holdout_count:])
    chunk_size = total_restart_count if restart_chunk_size <= 0 else max(1, min(total_restart_count, restart_chunk_size))
    chunk_specs: list[tuple[str, int, int]] = []
    for structured_start in range(0, structured_start_count, chunk_size):
        chunk_specs.append(("structured", structured_start, min(chunk_size, structured_start_count - structured_start)))
    for random_start in range(0, restart_count, chunk_size):
        chunk_specs.append(("random", random_start, min(chunk_size, restart_count - random_start)))

    def candidate_rank_score(candidate: dict[str, Any]) -> float:
        return float(candidate.get("selectionScore", candidate.get("score", candidate["mae"])))

    def residual_selection_score(prediction: np.ndarray, actual: np.ndarray) -> float:
        residual = np.abs(prediction - actual)
        mae = float(np.mean(residual))
        if selection_tail_weight <= 0.0 or residual.shape[0] == 0:
            return mae
        tail = float(np.quantile(residual, selection_tail_quantile))
        return mae + selection_tail_weight * tail

    def remember_top_candidate(candidate: dict[str, Any]) -> None:
        top_candidates.append(dict(candidate))
        top_candidates.sort(key=candidate_rank_score)
        keep_count = max(4, min(12, atom_count if atom_count >= 8 else 4))
        del top_candidates[keep_count:]

    def rescore_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
        gates_np = np.asarray(candidate["gates"], dtype=np.float64)
        ups_np = np.asarray(candidate["ups"], dtype=np.float64)
        scales_np = np.asarray(candidate["scales"], dtype=np.float64)
        bias_value = float(candidate["bias"])
        activation_np = silu_np(coords @ gates_np.T) * (coords @ ups_np.T)
        design = np.column_stack([activation_np, np.ones(coords.shape[0], dtype=np.float64)])
        try:
            coeffs, *_ = np.linalg.lstsq(design, targets, rcond=None)
            if np.all(np.isfinite(coeffs)):
                scales_np = coeffs[:-1].astype(np.float64)
                bias_value = float(coeffs[-1])
        except np.linalg.LinAlgError:
            pass
        prediction_np = activation_np @ scales_np + bias_value
        mae = float(np.mean(np.abs(prediction_np - targets)))
        mse = float(np.mean((prediction_np - targets) ** 2))
        selection_score = residual_selection_score(prediction_np, targets)
        if selection_fit_indices is not None and selection_holdout_indices is not None:
            try:
                selection_coeffs, *_ = np.linalg.lstsq(
                    design[selection_fit_indices],
                    targets[selection_fit_indices],
                    rcond=None,
                )
                if np.all(np.isfinite(selection_coeffs)):
                    selection_prediction = design[selection_holdout_indices] @ selection_coeffs
                    selection_score = residual_selection_score(selection_prediction, targets[selection_holdout_indices])
            except np.linalg.LinAlgError:
                selection_score = residual_selection_score(prediction_np, targets)
        rescored = dict(candidate)
        rescored.update({
            "mae": mae,
            "mse": mse,
            "r2": 1.0 - mse / max(target_variance, 1e-12),
            "worstError": float(np.max(np.abs(prediction_np - targets))),
            "score": mae,
            "selectionScore": selection_score,
            "gates": gates_np,
            "ups": ups_np,
            "scales": scales_np,
            "bias": bias_value,
        })
        return rescored

    for chunk_kind, chunk_start, current_count in chunk_specs:
        if chunk_kind == "random":
            generator = torch.Generator(device=device)
            generator.manual_seed(seed + chunk_start)
            gates_init = torch.randn(current_count, atom_count, rank, dtype=fit_dtype, device=device, generator=generator)
            ups_init = torch.randn(current_count, atom_count, rank, dtype=fit_dtype, device=device, generator=generator)
            init_scale_values = torch.tensor(
                [random_init_scales[(chunk_start + index) % len(random_init_scales)] for index in range(current_count)],
                dtype=fit_dtype,
                device=device,
            ).view(current_count, 1, 1)
            gates_init = gates_init * init_scale_values
            ups_init = ups_init * init_scale_values
            scales_init = torch.ones(current_count, atom_count, dtype=fit_dtype, device=device) * max(0.1, target_scale / max(1, atom_count))
            bias_init = torch.ones(current_count, dtype=fit_dtype, device=device) * float(np.mean(targets))
        else:
            gates_init = torch.empty(current_count, atom_count, rank, dtype=fit_dtype, device=device)
            ups_init = torch.empty(current_count, atom_count, rank, dtype=fit_dtype, device=device)
            scales_init = torch.empty(current_count, atom_count, dtype=fit_dtype, device=device)
            bias_init = torch.empty(current_count, dtype=fit_dtype, device=device)
            for local_index in range(current_count):
                gate_start, up_start, scale_start, bias_start = structured_starts[chunk_start + local_index]
                gates_init[local_index] = torch.tensor(gate_start, dtype=fit_dtype, device=device)
                ups_init[local_index] = torch.tensor(up_start, dtype=fit_dtype, device=device)
                scales_init[local_index] = torch.tensor(scale_start, dtype=fit_dtype, device=device)
                bias_init[local_index] = float(bias_start)

        gates = torch.nn.Parameter(gates_init)
        ups = torch.nn.Parameter(ups_init)
        scales = torch.nn.Parameter(scales_init)
        bias = torch.nn.Parameter(bias_init)
        optimizer = torch.optim.Adam([gates, ups, scales, bias], lr=learning_rate)
        for _ in range(steps):
            optimizer.zero_grad()
            gate_projection = torch.einsum("nr,bkr->bnk", x, gates)
            up_projection = torch.einsum("nr,bkr->bnk", x, ups)
            activations = torch.nn.functional.silu(gate_projection) * up_projection
            prediction = torch.sum(activations * scales[:, None, :], dim=2) + bias[:, None]
            per_restart_loss = torch.mean((prediction - y[None, :]) ** 2, dim=1)
            if has_gradient_witness and gx is not None and gy is not None and target_grad_scale is not None:
                grad_gate_projection = torch.einsum("mr,bkr->bmk", gx, gates)
                grad_up_projection = torch.einsum("mr,bkr->bmk", gx, ups)
                sigmoid = torch.sigmoid(grad_gate_projection)
                silu_values = torch.nn.functional.silu(grad_gate_projection)
                silu_prime = sigmoid * (1.0 + grad_gate_projection * (1.0 - sigmoid))
                gradient_terms = scales[:, None, :, None] * (
                    silu_prime[:, :, :, None] * grad_up_projection[:, :, :, None] * gates[:, None, :, :]
                    + silu_values[:, :, :, None] * ups[:, None, :, :]
                )
                predicted_gradients = torch.sum(gradient_terms, dim=2)
                gradient_loss = torch.mean((predicted_gradients - gy[None, :, :]) ** 2, dim=(1, 2)) / target_grad_scale
                per_restart_loss = per_restart_loss + gradient_weight * gradient_loss
            loss = torch.mean(per_restart_loss)
            loss.backward()
            optimizer.step()

        gates_np = gates.detach().cpu().numpy().astype(np.float64)
        ups_np = ups.detach().cpu().numpy().astype(np.float64)
        for local_restart in range(current_count):
            if chunk_kind == "structured":
                restart = chunk_start + local_restart
                start_kind = "structured"
                random_restart_index = None
            else:
                random_restart_index = chunk_start + local_restart
                restart = structured_start_count + random_restart_index
                start_kind = "random"
            activation_np = silu_np(coords @ gates_np[local_restart].T) * (coords @ ups_np[local_restart].T)
            design = np.column_stack([activation_np, ones])
            try:
                coeffs, *_ = np.linalg.lstsq(design, targets, rcond=None)
                if not np.all(np.isfinite(coeffs)):
                    continue
            except np.linalg.LinAlgError:
                continue
            prediction_np = design @ coeffs
            mae = float(np.mean(np.abs(prediction_np - targets)))
            mse = float(np.mean((prediction_np - targets) ** 2))
            r2 = 1.0 - mse / max(target_variance, 1e-12)
            score = mae
            selection_score = residual_selection_score(prediction_np, targets)
            if selection_fit_indices is not None and selection_holdout_indices is not None:
                try:
                    selection_coeffs, *_ = np.linalg.lstsq(
                        design[selection_fit_indices],
                        targets[selection_fit_indices],
                        rcond=None,
                    )
                    if np.all(np.isfinite(selection_coeffs)):
                        selection_prediction = design[selection_holdout_indices] @ selection_coeffs
                        selection_score = residual_selection_score(selection_prediction, targets[selection_holdout_indices])
                except np.linalg.LinAlgError:
                    selection_score = residual_selection_score(prediction_np, targets)
            candidate = {
                "restart": restart,
                "mae": mae,
                "mse": mse,
                "r2": r2,
                "score": score,
                "selectionScore": selection_score,
                "startKind": start_kind,
                "randomRestartIndex": random_restart_index,
                "gates": gates_np[local_restart],
                "ups": ups_np[local_restart],
                "scales": coeffs[:-1].astype(np.float64),
                "bias": float(coeffs[-1]),
            }
            for gate in gates_np[local_restart]:
                if np.all(np.isfinite(gate)) and np.linalg.norm(gate) > 1e-9:
                    gate_candidate_pool.append((score, gate.copy()))
            rank_score = candidate_rank_score(candidate)
            previous_kind_score = best_score_by_start_kind.get(start_kind)
            if previous_kind_score is None or rank_score < previous_kind_score:
                best_score_by_start_kind[start_kind] = rank_score
                best_metrics_by_start_kind[start_kind] = {
                    "restart": int(restart),
                    "randomRestartIndex": None if random_restart_index is None else int(random_restart_index),
                    "mae": mae,
                    "mse": mse,
                    "r2": r2,
                    "selectionScore": selection_score,
                }
            remember_top_candidate(candidate)
            if best is None or candidate_rank_score(candidate) < candidate_rank_score(best):
                best = candidate
        if len(gate_candidate_pool) > 4096:
            gate_candidate_pool.sort(key=lambda item: item[0])
            gate_candidate_pool = gate_candidate_pool[:2048]
        del gates, ups, scales, bias, optimizer, gates_np, ups_np
        torch.cuda.empty_cache()
    if best is None:
        raise RuntimeError("batched SwiGLU fit did not produce a finite candidate")
    if atom_count >= 8 and gate_candidate_pool:
        dictionary_fit = fit_swiglu_gate_dictionary(
            coords,
            targets,
            [gate for _score, gate in sorted(gate_candidate_pool, key=lambda item: item[0])],
            atom_count,
            seed + 424242,
        )
        if dictionary_fit is not None:
            dictionary_fit = rescore_candidate(dictionary_fit)
            remember_top_candidate(dictionary_fit)
        if dictionary_fit is not None and candidate_rank_score(dictionary_fit) < candidate_rank_score(best):
            best = dictionary_fit
    polish_top_k = int(os.environ.get(
        "GCL_BLIND_SWIGLU_POLISH_TOP_K",
        "5" if atom_count >= 12 else "3",
    ))
    polish_candidates = top_candidates[:max(1, polish_top_k)]
    if not any(candidate.get("restart") == best.get("restart") and candidate.get("startKind") == best.get("startKind") for candidate in polish_candidates):
        polish_candidates.append(best)
    polished_candidates: list[dict[str, Any]] = []
    for candidate in polish_candidates:
        polished = polish_swiglu_candidate(
            coords,
            targets,
            candidate,
            learning_rate,
            gradient_coords,
            gradient_targets,
            gradient_weight,
        )
        polished_candidates.append(rescore_candidate(polished))
    best = min(polished_candidates, key=candidate_rank_score)
    best["polishedCandidateCount"] = len(polished_candidates)
    backfit_cycles = int(os.environ.get(
        "GCL_BLIND_SWIGLU_BACKFIT_CYCLES",
        "0",
    ))
    if backfit_cycles > 0:
        best = rescore_candidate(coordinate_backfit_swiglu_candidate(
            coords,
            targets,
            best,
            backfit_cycles,
        ))
    tail_polish_steps = int(os.environ.get(
        "GCL_BLIND_SWIGLU_TAIL_POLISH_STEPS",
        "420" if atom_count >= 12 else "0",
    ))
    if tail_polish_steps > 0:
        tail_polished = tail_error_polish_swiglu_candidate(
            coords,
            targets,
            best,
            tail_polish_steps,
        )
        if os.environ.get("GCL_BLIND_SWIGLU_TAIL_POLISH_RESCORE", "1").strip() == "0":
            best = tail_polished
        else:
            best = rescore_candidate(tail_polished)
    if os.environ.get("GCL_BLIND_SWIGLU_MINIMAX_BIAS_POLISH", "").strip() == "1":
        best = minimax_bias_polish_swiglu_candidate(
            coords,
            targets,
            best,
        )
    if int(os.environ.get("GCL_BLIND_SWIGLU_FINAL_LBFGS_STEPS", "0") or "0") > 0:
        best = final_lbfgs_polish_swiglu_candidate(
            coords,
            targets,
            best,
        )
    if os.environ.get("GCL_BLIND_SWIGLU_CHEBYSHEV_CALIBRATION", "").strip() == "1":
        best = chebyshev_calibrate_swiglu_candidate(
            coords,
            targets,
            best,
        )
    if os.environ.get("GCL_BLIND_SWIGLU_ROBUST_SCALAR_CALIBRATION", "").strip() == "1":
        best = robust_scalar_calibrate_swiglu_candidate(
            coords,
            targets,
            best,
        )
    best["restartCount"] = total_restart_count
    best["randomRestartCount"] = restart_count
    best["structuredStartCount"] = structured_start_count
    best["selectionTailWeight"] = selection_tail_weight
    best["selectionTailQuantile"] = selection_tail_quantile
    best["randomInitScales"] = [float(value) for value in random_init_scales]
    best["bestScoreByStartKind"] = {key: float(value) for key, value in best_score_by_start_kind.items()}
    best["bestMetricsByStartKind"] = best_metrics_by_start_kind
    best["restartChunkSize"] = chunk_size
    batched_mode = "batched-cuda" if chunk_size >= total_restart_count else "batched-cuda-chunked"
    suffixes = []
    if structured_start_count:
        suffixes.append("structured-starts")
    if best.get("dictionarySelectedCount"):
        suffixes.append("gate-dictionary-beam" if best.get("dictionaryBeamWidth") else "gate-dictionary")
    if best.get("adamRefined"):
        suffixes.append("adam-refine")
    if best.get("directionalGateValuePolished"):
        suffixes.append("directional-gate-value-polish")
    if best.get("fixedGateDirectionValuePolished"):
        suffixes.append("fixed-gate-value-polish")
    if best.get("minimaxBiasPolished"):
        suffixes.append("minimax-bias-polish")
    if best.get("finalLbfgsPolished"):
        suffixes.append("final-lbfgs-polish")
    if best.get("chebyshevCalibrated"):
        suffixes.append("chebyshev-calibration")
    if best.get("robustScalarCalibrated"):
        suffixes.append("robust-scalar-calibration")
    if best.get("polished"):
        suffixes.append("lbfgs-polish")
    best["fitMode"] = "+".join([batched_mode] + suffixes)
    return best


def unique_gate_candidates(gates: list[np.ndarray], max_candidates: int) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    signatures: set[tuple[float, ...]] = set()
    scales = (0.5, 1.0, 2.0)
    for raw_gate in gates:
        gate = np.asarray(raw_gate, dtype=np.float64)
        norm = float(np.linalg.norm(gate))
        if norm < 1e-9 or not np.all(np.isfinite(gate)):
            continue
        direction = gate / norm
        if direction[np.argmax(np.abs(direction))] < 0.0:
            direction = -direction
        for scale in scales:
            candidate = direction * norm * scale
            candidate_norm = max(1e-9, np.linalg.norm(candidate))
            signature = (
                round(float(np.log(candidate_norm)), 3),
                *np.round(candidate / candidate_norm, 4).tolist(),
            )
            if signature in signatures:
                continue
            signatures.add(signature)
            unique.append(candidate)
            if len(unique) >= max_candidates:
                return unique
    return unique


def fit_swiglu_gate_dictionary(
    coords: np.ndarray,
    targets: np.ndarray,
    gate_candidates: list[np.ndarray],
    atom_count: int,
    seed: int,
) -> dict[str, Any] | None:
    if not gate_candidates:
        return None
    rng = np.random.default_rng(seed)
    candidates = unique_gate_candidates(gate_candidates, max_candidates=512 if atom_count >= 12 else 256)
    if len(candidates) < atom_count:
        return None
    sample_count = min(coords.shape[0], max(2048, atom_count * 384))
    if sample_count < coords.shape[0]:
        sample_indices = np.sort(rng.choice(coords.shape[0], size=sample_count, replace=False))
        xs = coords[sample_indices]
        ys = targets[sample_indices]
    else:
        xs = coords
        ys = targets
    beam_width = int(os.environ.get("GCL_BLIND_SWIGLU_DICTIONARY_BEAM_WIDTH", "1") or "1")
    beam_branch = int(os.environ.get("GCL_BLIND_SWIGLU_DICTIONARY_BEAM_BRANCH", "12") or "12")
    beam_pool_limit = int(os.environ.get("GCL_BLIND_SWIGLU_DICTIONARY_BEAM_POOL_LIMIT", "192") or "192")
    if beam_width < 1 or beam_branch < 1 or beam_pool_limit < atom_count:
        raise RuntimeError("GCL_BLIND_SWIGLU_DICTIONARY_BEAM_* values must be positive and pool_limit >= atom_count")

    def solve_sample_residual(selected_indices: list[int]) -> tuple[np.ndarray, float] | None:
        design_blocks = [silu_np(xs @ candidates[index])[:, None] * xs for index in selected_indices]
        design = np.column_stack(design_blocks + [np.ones(xs.shape[0], dtype=np.float64)])
        try:
            solved, *_ = np.linalg.lstsq(design, ys, rcond=None)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(solved)):
            return None
        residual_value = ys - design @ solved
        score_value = float(np.dot(residual_value, residual_value))
        return residual_value, score_value

    if beam_width > 1:
        pool_count = min(len(candidates), beam_pool_limit)
        beams: list[tuple[float, list[int], np.ndarray]] = [
            (
                float(np.dot(ys.astype(np.float64) - float(np.mean(ys)), ys.astype(np.float64) - float(np.mean(ys)))),
                [],
                ys.astype(np.float64) - float(np.mean(ys)),
            )
        ]
        for _atom_index in range(atom_count):
            next_beams: list[tuple[float, list[int], np.ndarray]] = []
            for _score, selected_indices, residual in beams:
                selected_directions = [
                    candidates[index] / max(1e-9, np.linalg.norm(candidates[index]))
                    for index in selected_indices
                ]
                local_choices: list[tuple[float, int]] = []
                selected_set = set(selected_indices)
                for index in range(pool_count):
                    if index in selected_set:
                        continue
                    gate = candidates[index]
                    gate_direction = gate / max(1e-9, np.linalg.norm(gate))
                    if any(abs(float(np.dot(gate_direction, selected_direction))) > 0.995 for selected_direction in selected_directions):
                        continue
                    block = silu_np(xs @ gate)[:, None] * xs
                    try:
                        coeffs, *_ = np.linalg.lstsq(block, residual, rcond=None)
                    except np.linalg.LinAlgError:
                        continue
                    if not np.all(np.isfinite(coeffs)):
                        continue
                    projection = block @ coeffs
                    score_value = float(np.dot(residual - projection, residual - projection))
                    local_choices.append((score_value, index))
                local_choices.sort(key=lambda item: item[0])
                for _candidate_score, index in local_choices[:beam_branch]:
                    new_selected = selected_indices + [index]
                    solved_residual = solve_sample_residual(new_selected)
                    if solved_residual is None:
                        continue
                    new_residual, new_score = solved_residual
                    next_beams.append((new_score, new_selected, new_residual))
            if not next_beams:
                break
            next_beams.sort(key=lambda item: item[0])
            deduped: list[tuple[float, list[int], np.ndarray]] = []
            signatures: set[tuple[int, ...]] = set()
            for item in next_beams:
                signature = tuple(sorted(item[1]))
                if signature in signatures:
                    continue
                signatures.add(signature)
                deduped.append(item)
                if len(deduped) >= beam_width:
                    break
            beams = deduped
        complete_beams = [beam for beam in beams if len(beam[1]) == atom_count]
        if complete_beams:
            complete_beams.sort(key=lambda item: item[0])
            selected = complete_beams[0][1]
            full_blocks = [silu_np(coords @ candidates[index])[:, None] * coords for index in selected]
            full_design = np.column_stack(full_blocks + [np.ones(coords.shape[0], dtype=np.float64)])
            try:
                solved, *_ = np.linalg.lstsq(full_design, targets, rcond=None)
            except np.linalg.LinAlgError:
                solved = None
            if solved is not None and np.all(np.isfinite(solved)):
                prediction = full_design @ solved
                mae = float(np.mean(np.abs(prediction - targets)))
                mse = float(np.mean((prediction - targets) ** 2))
                rank = coords.shape[1]
                up_vectors = []
                scales = []
                for atom_index in range(atom_count):
                    coeff_slice = solved[atom_index * rank:(atom_index + 1) * rank].astype(np.float64)
                    coeff_norm = float(np.linalg.norm(coeff_slice))
                    if coeff_norm < 1e-12:
                        up_vectors.append(np.zeros(rank, dtype=np.float64))
                        scales.append(0.0)
                    else:
                        up_vectors.append(coeff_slice / coeff_norm)
                        scales.append(coeff_norm)
                return {
                    "restart": -1,
                    "mae": mae,
                    "mse": mse,
                    "score": mae,
                    "selectionScore": mae,
                    "startKind": "dictionary-beam",
                    "randomRestartIndex": None,
                    "gates": np.vstack([candidates[index] for index in selected]).astype(np.float64),
                    "ups": np.vstack(up_vectors).astype(np.float64),
                    "scales": np.array(scales, dtype=np.float64),
                    "bias": float(solved[-1]),
                    "dictionarySelectedCount": len(selected),
                    "dictionaryBeamWidth": int(beam_width),
                    "dictionaryBeamBranch": int(beam_branch),
                    "dictionaryBeamPoolLimit": int(pool_count),
                }

    selected: list[int] = []
    selected_set: set[int] = set()
    selected_directions: list[np.ndarray] = []
    residual = ys.astype(np.float64) - float(np.mean(ys))
    for _ in range(atom_count):
        best_choice: tuple[float, int, np.ndarray] | None = None
        for index, gate in enumerate(candidates):
            if index in selected_set:
                continue
            gate_direction = gate / max(1e-9, np.linalg.norm(gate))
            if any(abs(float(np.dot(gate_direction, selected_direction))) > 0.995 for selected_direction in selected_directions):
                continue
            block = silu_np(xs @ gate)[:, None] * xs
            try:
                coeffs, *_ = np.linalg.lstsq(block, residual, rcond=None)
            except np.linalg.LinAlgError:
                continue
            if not np.all(np.isfinite(coeffs)):
                continue
            projection = block @ coeffs
            improvement = float(np.dot(residual, residual) - np.dot(residual - projection, residual - projection))
            if best_choice is None or improvement > best_choice[0]:
                best_choice = (improvement, index, coeffs.astype(np.float64))
        if best_choice is None or best_choice[0] <= 1e-12:
            break
        selected.append(best_choice[1])
        selected_set.add(best_choice[1])
        selected_gate = candidates[best_choice[1]]
        selected_directions.append(selected_gate / max(1e-9, np.linalg.norm(selected_gate)))
        design_blocks = [silu_np(xs @ candidates[index])[:, None] * xs for index in selected]
        design = np.column_stack(design_blocks + [np.ones(xs.shape[0], dtype=np.float64)])
        try:
            solved, *_ = np.linalg.lstsq(design, ys, rcond=None)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(solved)):
            break
        residual = ys - design @ solved
    if len(selected) != atom_count:
        return None
    full_blocks = [silu_np(coords @ candidates[index])[:, None] * coords for index in selected]
    full_design = np.column_stack(full_blocks + [np.ones(coords.shape[0], dtype=np.float64)])
    try:
        solved, *_ = np.linalg.lstsq(full_design, targets, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(solved)):
        return None
    prediction = full_design @ solved
    mae = float(np.mean(np.abs(prediction - targets)))
    mse = float(np.mean((prediction - targets) ** 2))
    rank = coords.shape[1]
    up_vectors = []
    scales = []
    for atom_index in range(atom_count):
        coeff_slice = solved[atom_index * rank:(atom_index + 1) * rank].astype(np.float64)
        coeff_norm = float(np.linalg.norm(coeff_slice))
        if coeff_norm < 1e-12:
            up_vectors.append(np.zeros(rank, dtype=np.float64))
            scales.append(0.0)
        else:
            up_vectors.append(coeff_slice / coeff_norm)
            scales.append(coeff_norm)
    return {
        "restart": -1,
        "mae": mae,
        "mse": mse,
        "score": mae,
        "selectionScore": mae,
        "startKind": "dictionary",
        "randomRestartIndex": None,
        "gates": np.vstack([candidates[index] for index in selected]).astype(np.float64),
        "ups": np.vstack(up_vectors).astype(np.float64),
        "scales": np.array(scales, dtype=np.float64),
        "bias": float(solved[-1]),
        "dictionarySelectedCount": len(selected),
    }


def polish_swiglu_candidate(
    coords: np.ndarray,
    targets: np.ndarray,
    candidate: dict[str, Any],
    learning_rate: float,
    gradient_coords: np.ndarray | None,
    gradient_targets: np.ndarray | None,
    gradient_weight: float,
) -> dict[str, Any]:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fit_dtype = torch.float32
    x = torch.tensor(coords, dtype=fit_dtype, device=device)
    y = torch.tensor(targets, dtype=fit_dtype, device=device)
    gates = torch.nn.Parameter(torch.tensor(candidate["gates"], dtype=fit_dtype, device=device))
    ups = torch.nn.Parameter(torch.tensor(candidate["ups"], dtype=fit_dtype, device=device))
    scales = torch.nn.Parameter(torch.tensor(candidate["scales"], dtype=fit_dtype, device=device))
    bias = torch.nn.Parameter(torch.tensor(float(candidate["bias"]), dtype=fit_dtype, device=device))
    has_gradient_witness = (
        gradient_coords is not None
        and gradient_targets is not None
        and gradient_weight > 0.0
        and gradient_coords.shape[0] > 0
    )
    if has_gradient_witness:
        gx = torch.tensor(gradient_coords, dtype=fit_dtype, device=device)
        gy = torch.tensor(gradient_targets, dtype=fit_dtype, device=device)
        target_grad_scale = torch.mean(gy ** 2).detach().clamp_min(1e-12)
    else:
        gx = None
        gy = None
        target_grad_scale = None

    lbfgs = torch.optim.LBFGS(
        [gates, ups, scales, bias],
        lr=max(0.2, min(0.8, learning_rate * 28.0)),
        max_iter=520 if gates.shape[0] >= 12 else 360,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        lbfgs.zero_grad()
        gate_projection = x @ gates.T
        up_projection = x @ ups.T
        activations = torch.nn.functional.silu(gate_projection) * up_projection
        prediction = activations @ scales + bias
        loss = torch.mean((prediction - y) ** 2)
        if has_gradient_witness and gx is not None and gy is not None and target_grad_scale is not None:
            grad_gate_projection = gx @ gates.T
            grad_up_projection = gx @ ups.T
            sigmoid = torch.sigmoid(grad_gate_projection)
            silu_values = torch.nn.functional.silu(grad_gate_projection)
            silu_prime = sigmoid * (1.0 + grad_gate_projection * (1.0 - sigmoid))
            gradient_terms = scales[None, :, None] * (
                silu_prime[:, :, None] * grad_up_projection[:, :, None] * gates[None, :, :]
                + silu_values[:, :, None] * ups[None, :, :]
            )
            predicted_gradients = torch.sum(gradient_terms, dim=1)
            loss = loss + gradient_weight * torch.mean((predicted_gradients - gy) ** 2) / target_grad_scale
        loss.backward()
        return loss

    try:
        lbfgs.step(closure)
    except RuntimeError:
        return candidate

    gates_np = gates.detach().cpu().numpy().astype(np.float64)
    ups_np = ups.detach().cpu().numpy().astype(np.float64)
    activation_np = silu_np(coords @ gates_np.T) * (coords @ ups_np.T)
    design = np.column_stack([activation_np, np.ones(coords.shape[0], dtype=np.float64)])
    try:
        coeffs, *_ = np.linalg.lstsq(design, targets, rcond=None)
        if not np.all(np.isfinite(coeffs)):
            return candidate
    except np.linalg.LinAlgError:
        return candidate

    prediction_np = design @ coeffs
    mae = float(np.mean(np.abs(prediction_np - targets)))
    mse = float(np.mean((prediction_np - targets) ** 2))
    enable_experimental_polish = os.environ.get("GCL_BLIND_SWIGLU_EXPERIMENTAL_POLISH", "").strip() == "1"
    if mae >= float(candidate["score"]):
        directional = candidate
        if enable_experimental_polish:
            refined = adam_refine_swiglu_candidate(
                coords,
                targets,
                candidate,
                gradient_coords,
                gradient_targets,
                gradient_weight,
            )
            directional = directional_gate_value_polish(
                coords,
                targets,
                refined,
                gradient_coords,
                gradient_targets,
                gradient_weight,
            )
        return fixed_gate_direction_value_polish(
            coords,
            targets,
            directional,
            gradient_coords,
            gradient_targets,
            gradient_weight,
        )
    polished = dict(candidate)
    polished.update({
        "mae": mae,
        "mse": mse,
        "score": mae,
        "gates": gates_np,
        "ups": ups_np,
        "scales": coeffs[:-1].astype(np.float64),
        "bias": float(coeffs[-1]),
        "polished": True,
        "polishMaeBefore": float(candidate["mae"]),
    })
    directional = polished
    if enable_experimental_polish:
        refined = adam_refine_swiglu_candidate(
            coords,
            targets,
            polished,
            gradient_coords,
            gradient_targets,
            gradient_weight,
        )
        directional = directional_gate_value_polish(
            coords,
            targets,
            refined,
            gradient_coords,
            gradient_targets,
            gradient_weight,
        )
    return fixed_gate_direction_value_polish(
        coords,
        targets,
        directional,
        gradient_coords,
        gradient_targets,
        gradient_weight,
    )


def adam_refine_swiglu_candidate(
    coords: np.ndarray,
    targets: np.ndarray,
    candidate: dict[str, Any],
    gradient_coords: np.ndarray | None,
    gradient_targets: np.ndarray | None,
    gradient_weight: float,
) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return candidate
    gates_np = np.asarray(candidate["gates"], dtype=np.float64)
    if gates_np.ndim != 2 or gates_np.shape[0] < 2:
        return candidate
    device = torch.device("cuda")
    fit_dtype = torch.float32
    x = torch.tensor(coords, dtype=fit_dtype, device=device)
    y = torch.tensor(targets, dtype=fit_dtype, device=device)
    gates = torch.nn.Parameter(torch.tensor(gates_np, dtype=fit_dtype, device=device))
    ups = torch.nn.Parameter(torch.tensor(candidate["ups"], dtype=fit_dtype, device=device))
    scales = torch.nn.Parameter(torch.tensor(candidate["scales"], dtype=fit_dtype, device=device))
    bias = torch.nn.Parameter(torch.tensor(float(candidate["bias"]), dtype=fit_dtype, device=device))
    has_gradient_witness = (
        gradient_coords is not None
        and gradient_targets is not None
        and gradient_weight > 0.0
        and gradient_coords.shape[0] > 0
    )
    if has_gradient_witness:
        gx = torch.tensor(gradient_coords, dtype=fit_dtype, device=device)
        gy = torch.tensor(gradient_targets, dtype=fit_dtype, device=device)
        target_grad_scale = torch.mean(gy ** 2).detach().clamp_min(1e-12)
    else:
        gx = None
        gy = None
        target_grad_scale = None
    optimizer = torch.optim.AdamW([gates, ups, scales, bias], lr=0.006, weight_decay=2e-6)
    best_mae = float(candidate["mae"])
    best_state: tuple[np.ndarray, np.ndarray, np.ndarray, float] | None = None
    step_count = 2600 if gates_np.shape[0] >= 12 else 1600
    for step in range(step_count):
        optimizer.zero_grad()
        gate_projection = x @ gates.T
        up_projection = x @ ups.T
        activations = torch.nn.functional.silu(gate_projection) * up_projection
        prediction = activations @ scales + bias
        loss = torch.mean((prediction - y) ** 2)
        if has_gradient_witness and gx is not None and gy is not None and target_grad_scale is not None:
            grad_gate_projection = gx @ gates.T
            grad_up_projection = gx @ ups.T
            sigmoid = torch.sigmoid(grad_gate_projection)
            silu_values = torch.nn.functional.silu(grad_gate_projection)
            silu_prime = sigmoid * (1.0 + grad_gate_projection * (1.0 - sigmoid))
            gradient_terms = scales[None, :, None] * (
                silu_prime[:, :, None] * grad_up_projection[:, :, None] * gates[None, :, :]
                + silu_values[:, :, None] * ups[None, :, :]
            )
            predicted_gradients = torch.sum(gradient_terms, dim=1)
            loss = loss + gradient_weight * torch.mean((predicted_gradients - gy) ** 2) / target_grad_scale
        loss.backward()
        torch.nn.utils.clip_grad_norm_([gates, ups, scales, bias], 8.0)
        optimizer.step()
        if step == 0 or (step + 1) % 160 == 0 or step + 1 == step_count:
            with torch.no_grad():
                pred = (torch.nn.functional.silu(x @ gates.T) * (x @ ups.T)) @ scales + bias
                mae = float(torch.mean(torch.abs(pred - y)).detach().cpu().item())
                if mae < best_mae:
                    best_mae = mae
                    best_state = (
                        gates.detach().cpu().numpy().astype(np.float64),
                        ups.detach().cpu().numpy().astype(np.float64),
                        scales.detach().cpu().numpy().astype(np.float64),
                        float(bias.detach().cpu().item()),
                    )
    if best_state is None:
        return candidate
    gates_np, ups_np, scales_np, bias_value = best_state
    activation_np = silu_np(coords @ gates_np.T) * (coords @ ups_np.T)
    design = np.column_stack([activation_np, np.ones(coords.shape[0], dtype=np.float64)])
    try:
        coeffs, *_ = np.linalg.lstsq(design, targets, rcond=None)
    except np.linalg.LinAlgError:
        coeffs = None
    if coeffs is not None and np.all(np.isfinite(coeffs)):
        scales_np = coeffs[:-1].astype(np.float64)
        bias_value = float(coeffs[-1])
        prediction_np = design @ coeffs
    else:
        prediction_np = activation_np @ scales_np + bias_value
    mae = float(np.mean(np.abs(prediction_np - targets)))
    mse = float(np.mean((prediction_np - targets) ** 2))
    if mae >= float(candidate["score"]):
        return candidate
    refined = dict(candidate)
    refined.update({
        "mae": mae,
        "mse": mse,
        "score": mae,
        "gates": gates_np.astype(np.float64),
        "ups": ups_np.astype(np.float64),
        "scales": scales_np,
        "bias": bias_value,
        "adamRefined": True,
        "adamRefineMaeBefore": float(candidate["mae"]),
    })
    if candidate.get("polished"):
        refined["polished"] = True
    return refined


def directional_gate_value_polish(
    coords: np.ndarray,
    targets: np.ndarray,
    candidate: dict[str, Any],
    gradient_coords: np.ndarray | None,
    gradient_targets: np.ndarray | None,
    gradient_weight: float,
) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return candidate
    gates = np.asarray(candidate["gates"], dtype=np.float64)
    if gates.ndim != 2 or gates.shape[0] < 2:
        return candidate
    gate_norms = np.linalg.norm(gates, axis=1).clip(min=1e-6)
    gate_dirs = gates / gate_norms[:, None]
    base_ups = np.asarray(candidate["ups"], dtype=np.float64) * np.asarray(candidate["scales"], dtype=np.float64)[:, None]

    device = torch.device("cuda")
    fit_dtype = torch.float32
    x = torch.tensor(coords, dtype=fit_dtype, device=device)
    y = torch.tensor(targets, dtype=fit_dtype, device=device)
    gate_raw = torch.nn.Parameter(torch.tensor(gate_dirs, dtype=fit_dtype, device=device))
    log_gate_norms = torch.nn.Parameter(torch.tensor(np.log(gate_norms), dtype=fit_dtype, device=device))
    up_vectors = torch.nn.Parameter(torch.tensor(base_ups, dtype=fit_dtype, device=device))
    bias = torch.nn.Parameter(torch.tensor(float(candidate["bias"]), dtype=fit_dtype, device=device))
    has_gradient_witness = (
        gradient_coords is not None
        and gradient_targets is not None
        and gradient_weight > 0.0
        and gradient_coords.shape[0] > 0
    )
    if has_gradient_witness:
        gx = torch.tensor(gradient_coords, dtype=fit_dtype, device=device)
        gy = torch.tensor(gradient_targets, dtype=fit_dtype, device=device)
        target_grad_scale = torch.mean(gy ** 2).detach().clamp_min(1e-12)
    else:
        gx = None
        gy = None
        target_grad_scale = None

    optimizer = torch.optim.AdamW([gate_raw, log_gate_norms, up_vectors, bias], lr=0.012, weight_decay=2e-6)
    best_mae = float(candidate["mae"])
    best_state: tuple[np.ndarray, np.ndarray, float] | None = None
    step_count = 3200 if gates.shape[0] >= 12 else 2200
    for step in range(step_count):
        optimizer.zero_grad()
        gate_dirs_t = torch.nn.functional.normalize(gate_raw, dim=1, eps=1e-8)
        gate_vectors = gate_dirs_t * torch.exp(log_gate_norms).clamp(1e-5, 20.0)[:, None]
        gate_projection = x @ gate_vectors.T
        up_projection = x @ up_vectors.T
        prediction = torch.sum(torch.nn.functional.silu(gate_projection) * up_projection, dim=1) + bias
        loss = torch.mean((prediction - y) ** 2)
        if has_gradient_witness and gx is not None and gy is not None and target_grad_scale is not None:
            grad_gate_projection = gx @ gate_vectors.T
            grad_up_projection = gx @ up_vectors.T
            sigmoid = torch.sigmoid(grad_gate_projection)
            silu_values = torch.nn.functional.silu(grad_gate_projection)
            silu_prime = sigmoid * (1.0 + grad_gate_projection * (1.0 - sigmoid))
            gradient_terms = (
                silu_prime[:, :, None] * grad_up_projection[:, :, None] * gate_vectors[None, :, :]
                + silu_values[:, :, None] * up_vectors[None, :, :]
            )
            predicted_gradients = torch.sum(gradient_terms, dim=1)
            loss = loss + gradient_weight * torch.mean((predicted_gradients - gy) ** 2) / target_grad_scale
        if gates.shape[0] >= 4:
            cosine = gate_dirs_t @ gate_dirs_t.T
            offdiag = cosine - torch.eye(gates.shape[0], dtype=fit_dtype, device=device)
            loss = loss + 1e-4 * torch.mean(torch.relu(torch.abs(offdiag) - 0.995) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([gate_raw, log_gate_norms, up_vectors, bias], 8.0)
        optimizer.step()
        if step == 0 or (step + 1) % 160 == 0 or step + 1 == step_count:
            with torch.no_grad():
                gate_dirs_t = torch.nn.functional.normalize(gate_raw, dim=1, eps=1e-8)
                gate_vectors = gate_dirs_t * torch.exp(log_gate_norms).clamp(1e-5, 20.0)[:, None]
                prediction = torch.sum(torch.nn.functional.silu(x @ gate_vectors.T) * (x @ up_vectors.T), dim=1) + bias
                mae = float(torch.mean(torch.abs(prediction - y)).detach().cpu().item())
                if mae < best_mae:
                    best_mae = mae
                    best_state = (
                        gate_vectors.detach().cpu().numpy().astype(np.float64),
                        up_vectors.detach().cpu().numpy().astype(np.float64),
                        float(bias.detach().cpu().item()),
                    )
    if best_state is None:
        return candidate
    gates_np, ups_np, bias_value = best_state
    activation_np = silu_np(coords @ gates_np.T) * (coords @ ups_np.T)
    design = np.column_stack([activation_np, np.ones(coords.shape[0], dtype=np.float64)])
    try:
        coeffs, *_ = np.linalg.lstsq(design, targets, rcond=None)
    except np.linalg.LinAlgError:
        coeffs = None
    if coeffs is not None and np.all(np.isfinite(coeffs)):
        ups_np = ups_np * coeffs[:-1, None]
        bias_value = float(coeffs[-1])
        prediction_np = design @ coeffs
    else:
        prediction_np = activation_np.sum(axis=1) + bias_value
    mae = float(np.mean(np.abs(prediction_np - targets)))
    mse = float(np.mean((prediction_np - targets) ** 2))
    if mae >= float(candidate["score"]):
        return candidate
    polished = dict(candidate)
    polished.update({
        "mae": mae,
        "mse": mse,
        "score": mae,
        "gates": gates_np.astype(np.float64),
        "ups": ups_np.astype(np.float64),
        "scales": np.ones(gates.shape[0], dtype=np.float64),
        "bias": bias_value,
        "directionalGateValuePolished": True,
        "directionalGateValueMaeBefore": float(candidate["mae"]),
    })
    if candidate.get("polished"):
        polished["polished"] = True
    return polished


def fixed_gate_direction_value_polish(
    coords: np.ndarray,
    targets: np.ndarray,
    candidate: dict[str, Any],
    gradient_coords: np.ndarray | None,
    gradient_targets: np.ndarray | None,
    gradient_weight: float,
) -> dict[str, Any]:
    import torch

    gates = np.asarray(candidate["gates"], dtype=np.float64)
    gate_norms = np.linalg.norm(gates, axis=1).clip(min=1e-6)
    gate_dirs = gates / gate_norms[:, None]
    base_ups = np.asarray(candidate["ups"], dtype=np.float64) * np.asarray(candidate["scales"], dtype=np.float64)[:, None]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fit_dtype = torch.float32
    x = torch.tensor(coords, dtype=fit_dtype, device=device)
    y = torch.tensor(targets, dtype=fit_dtype, device=device)
    gate_dirs_t = torch.tensor(gate_dirs, dtype=fit_dtype, device=device)
    log_gate_norms = torch.nn.Parameter(torch.tensor(np.log(gate_norms), dtype=fit_dtype, device=device))
    up_vectors = torch.nn.Parameter(torch.tensor(base_ups, dtype=fit_dtype, device=device))
    bias = torch.nn.Parameter(torch.tensor(float(candidate["bias"]), dtype=fit_dtype, device=device))
    has_gradient_witness = (
        gradient_coords is not None
        and gradient_targets is not None
        and gradient_weight > 0.0
        and gradient_coords.shape[0] > 0
    )
    if has_gradient_witness:
        gx = torch.tensor(gradient_coords, dtype=fit_dtype, device=device)
        gy = torch.tensor(gradient_targets, dtype=fit_dtype, device=device)
        target_grad_scale = torch.mean(gy ** 2).detach().clamp_min(1e-12)
    else:
        gx = None
        gy = None
        target_grad_scale = None

    optimizer = torch.optim.AdamW([log_gate_norms, up_vectors, bias], lr=0.03, weight_decay=1e-6)
    best_state: tuple[float, np.ndarray, np.ndarray, float] | None = None
    step_count = 2800 if gates.shape[0] >= 12 else 1800
    for step in range(step_count):
        optimizer.zero_grad()
        gate_norms_t = torch.exp(log_gate_norms).clamp(1e-5, 20.0)
        gate_vectors = gate_dirs_t * gate_norms_t[:, None]
        gate_projection = x @ gate_vectors.T
        up_projection = x @ up_vectors.T
        prediction = torch.sum(torch.nn.functional.silu(gate_projection) * up_projection, dim=1) + bias
        loss = torch.mean((prediction - y) ** 2)
        if has_gradient_witness and gx is not None and gy is not None and target_grad_scale is not None:
            grad_gate_projection = gx @ gate_vectors.T
            grad_up_projection = gx @ up_vectors.T
            sigmoid = torch.sigmoid(grad_gate_projection)
            silu_values = torch.nn.functional.silu(grad_gate_projection)
            silu_prime = sigmoid * (1.0 + grad_gate_projection * (1.0 - sigmoid))
            gradient_terms = (
                silu_prime[:, :, None] * grad_up_projection[:, :, None] * gate_vectors[None, :, :]
                + silu_values[:, :, None] * up_vectors[None, :, :]
            )
            predicted_gradients = torch.sum(gradient_terms, dim=1)
            loss = loss + gradient_weight * torch.mean((predicted_gradients - gy) ** 2) / target_grad_scale
        loss.backward()
        torch.nn.utils.clip_grad_norm_([log_gate_norms, up_vectors, bias], 10.0)
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0:
            with torch.no_grad():
                plain_prediction = torch.sum(
                    torch.nn.functional.silu((x @ (gate_dirs_t * torch.exp(log_gate_norms).clamp(1e-5, 20.0)[:, None]).T))
                    * (x @ up_vectors.T),
                    dim=1,
                ) + bias
                plain_loss = float(torch.mean((plain_prediction - y) ** 2).detach().cpu().item())
                if best_state is None or plain_loss < best_state[0]:
                    best_state = (
                        plain_loss,
                        torch.exp(log_gate_norms).clamp(1e-5, 20.0).detach().cpu().numpy().astype(np.float64),
                        up_vectors.detach().cpu().numpy().astype(np.float64),
                        float(bias.detach().cpu().item()),
                    )
    if best_state is None:
        return candidate
    _loss, norm_values, up_values, bias_value = best_state
    gate_values = gate_dirs * norm_values[:, None]
    activation_np = silu_np(coords @ gate_values.T) * (coords @ up_values.T)
    design = np.column_stack([activation_np, np.ones(coords.shape[0], dtype=np.float64)])
    try:
        coeffs, *_ = np.linalg.lstsq(design, targets, rcond=None)
    except np.linalg.LinAlgError:
        coeffs = None
    if coeffs is not None and np.all(np.isfinite(coeffs)):
        up_values = up_values * coeffs[:-1, None]
        bias_value = float(coeffs[-1])
        prediction_np = design @ coeffs
    else:
        prediction_np = activation_np.sum(axis=1) + bias_value
    mae = float(np.mean(np.abs(prediction_np - targets)))
    mse = float(np.mean((prediction_np - targets) ** 2))
    if mae >= float(candidate["score"]):
        return candidate
    polished = dict(candidate)
    polished.update({
        "mae": mae,
        "mse": mse,
        "score": mae,
        "gates": gate_values.astype(np.float64),
        "ups": up_values.astype(np.float64),
        "scales": np.ones(gates.shape[0], dtype=np.float64),
        "bias": bias_value,
        "fixedGateDirectionValuePolished": True,
        "fixedGateDirectionMaeBefore": float(candidate["mae"]),
    })
    if candidate.get("polished"):
        polished["polished"] = True
    return polished


def coordinate_backfit_swiglu_candidate(
    coords: np.ndarray,
    targets: np.ndarray,
    candidate: dict[str, Any],
    cycles: int,
) -> dict[str, Any]:
    if cycles <= 0:
        return candidate
    import torch

    gates_np = np.asarray(candidate["gates"], dtype=np.float64)
    ups_np = np.asarray(candidate["ups"], dtype=np.float64)
    scales_np = np.asarray(candidate["scales"], dtype=np.float64)
    if gates_np.ndim != 2 or ups_np.shape != gates_np.shape or scales_np.shape[0] != gates_np.shape[0]:
        return candidate
    atom_count, rank = gates_np.shape
    if atom_count < 2:
        return candidate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fit_dtype = torch.float32
    x = torch.tensor(coords, dtype=fit_dtype, device=device)
    y = torch.tensor(targets, dtype=fit_dtype, device=device)
    gates = torch.tensor(gates_np, dtype=fit_dtype, device=device)
    up_vectors = torch.tensor(ups_np * scales_np[:, None], dtype=fit_dtype, device=device)
    bias_value = float(candidate["bias"])
    step_count = 420 if atom_count >= 12 else 300

    for _cycle in range(cycles):
        with torch.no_grad():
            contributions = torch.nn.functional.silu(x @ gates.T) * (x @ up_vectors.T)
        for atom_index in range(atom_count):
            with torch.no_grad():
                residual = y - float(bias_value) - torch.sum(contributions, dim=1) + contributions[:, atom_index]
            gate = torch.nn.Parameter(gates[atom_index].detach().clone())
            up = torch.nn.Parameter(up_vectors[atom_index].detach().clone())
            optimizer = torch.optim.AdamW([gate, up], lr=0.018, weight_decay=1e-7)
            for _step in range(step_count):
                optimizer.zero_grad()
                prediction = torch.nn.functional.silu(x @ gate) * (x @ up)
                loss = torch.mean((prediction - residual) ** 2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([gate, up], 8.0)
                optimizer.step()
            lbfgs = torch.optim.LBFGS([gate, up], lr=0.45, max_iter=80, line_search_fn="strong_wolfe")

            def closure() -> torch.Tensor:
                lbfgs.zero_grad()
                prediction = torch.nn.functional.silu(x @ gate) * (x @ up)
                loss = torch.mean((prediction - residual) ** 2)
                loss.backward()
                return loss

            try:
                lbfgs.step(closure)
            except RuntimeError:
                pass
            with torch.no_grad():
                gates[atom_index] = gate.detach()
                up_vectors[atom_index] = up.detach()
                contributions[:, atom_index] = torch.nn.functional.silu(x @ gates[atom_index]) * (x @ up_vectors[atom_index])

        activation_np = contributions.detach().cpu().numpy().astype(np.float64)
        design = np.column_stack([activation_np, np.ones(coords.shape[0], dtype=np.float64)])
        try:
            coeffs, *_ = np.linalg.lstsq(design, targets, rcond=None)
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(coeffs)):
            continue
        with torch.no_grad():
            up_vectors *= torch.tensor(coeffs[:-1], dtype=fit_dtype, device=device)[:, None]
            bias_value = float(coeffs[-1])

    gates_final = gates.detach().cpu().numpy().astype(np.float64)
    ups_final = up_vectors.detach().cpu().numpy().astype(np.float64)
    activation_np = silu_np(coords @ gates_final.T) * (coords @ ups_final.T)
    design = np.column_stack([activation_np, np.ones(coords.shape[0], dtype=np.float64)])
    try:
        coeffs, *_ = np.linalg.lstsq(design, targets, rcond=None)
    except np.linalg.LinAlgError:
        coeffs = None
    if coeffs is not None and np.all(np.isfinite(coeffs)):
        ups_final = ups_final * coeffs[:-1, None]
        bias_value = float(coeffs[-1])
        prediction_np = design @ coeffs
    else:
        prediction_np = activation_np.sum(axis=1) + bias_value
    mae = float(np.mean(np.abs(prediction_np - targets)))
    mse = float(np.mean((prediction_np - targets) ** 2))
    if mae >= float(candidate["score"]):
        return candidate
    backfit = dict(candidate)
    backfit.update({
        "mae": mae,
        "mse": mse,
        "score": mae,
        "gates": gates_final,
        "ups": ups_final,
        "scales": np.ones(atom_count, dtype=np.float64),
        "bias": bias_value,
        "coordinateBackfit": True,
        "coordinateBackfitCycles": int(cycles),
        "coordinateBackfitMaeBefore": float(candidate["mae"]),
    })
    return backfit


def tail_error_polish_swiglu_candidate(
    coords: np.ndarray,
    targets: np.ndarray,
    candidate: dict[str, Any],
    steps: int,
) -> dict[str, Any]:
    if steps <= 0:
        return candidate
    import torch

    gates_np = np.asarray(candidate["gates"], dtype=np.float64)
    ups_np = np.asarray(candidate["ups"], dtype=np.float64)
    scales_np = np.asarray(candidate["scales"], dtype=np.float64)
    if gates_np.ndim != 2 or ups_np.shape != gates_np.shape or scales_np.shape[0] != gates_np.shape[0]:
        return candidate
    atom_count = gates_np.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fit_dtype = torch.float32
    x = torch.tensor(coords, dtype=fit_dtype, device=device)
    y = torch.tensor(targets, dtype=fit_dtype, device=device)
    gates = torch.nn.Parameter(torch.tensor(gates_np, dtype=fit_dtype, device=device))
    up_vectors = torch.nn.Parameter(torch.tensor(ups_np * scales_np[:, None], dtype=fit_dtype, device=device))
    bias = torch.nn.Parameter(torch.tensor(float(candidate["bias"]), dtype=fit_dtype, device=device))
    tail_weight = float(os.environ.get("GCL_BLIND_SWIGLU_TAIL_POLISH_WEIGHT", "0.35") or "0.35")
    tail_fraction = float(os.environ.get("GCL_BLIND_SWIGLU_TAIL_POLISH_FRACTION", "0.125") or "0.125")
    tail_lr = float(os.environ.get("GCL_BLIND_SWIGLU_TAIL_POLISH_LR", "0.006") or "0.006")
    tail_loss_kind = os.environ.get("GCL_BLIND_SWIGLU_TAIL_POLISH_LOSS", "topk").strip().lower() or "topk"
    tail_smoothness = float(os.environ.get("GCL_BLIND_SWIGLU_TAIL_POLISH_SMOOTHNESS", "40.0") or "40.0")
    tail_calibration = os.environ.get("GCL_BLIND_SWIGLU_TAIL_POLISH_CALIBRATION", "ls").strip().lower() or "ls"
    if not np.isfinite(tail_weight) or tail_weight < 0.0:
        raise RuntimeError("GCL_BLIND_SWIGLU_TAIL_POLISH_WEIGHT must be non-negative and finite")
    if not np.isfinite(tail_fraction) or tail_fraction <= 0.0 or tail_fraction > 1.0:
        raise RuntimeError("GCL_BLIND_SWIGLU_TAIL_POLISH_FRACTION must be in (0, 1]")
    if not np.isfinite(tail_lr) or tail_lr <= 0.0:
        raise RuntimeError("GCL_BLIND_SWIGLU_TAIL_POLISH_LR must be positive and finite")
    if tail_loss_kind not in {"topk", "smoothmax"}:
        raise RuntimeError("GCL_BLIND_SWIGLU_TAIL_POLISH_LOSS must be topk or smoothmax")
    if tail_calibration not in {"ls", "direct", "best-worst"}:
        raise RuntimeError("GCL_BLIND_SWIGLU_TAIL_POLISH_CALIBRATION must be ls, direct, or best-worst")
    if not np.isfinite(tail_smoothness) or tail_smoothness <= 0.0:
        raise RuntimeError("GCL_BLIND_SWIGLU_TAIL_POLISH_SMOOTHNESS must be positive and finite")
    optimizer = torch.optim.AdamW([gates, up_vectors, bias], lr=tail_lr, weight_decay=1e-7)
    tail_count = max(64, min(coords.shape[0], int(coords.shape[0] * tail_fraction), atom_count * 128))
    best_state: tuple[float, np.ndarray, np.ndarray, float] | None = None
    for _step in range(steps):
        optimizer.zero_grad()
        prediction = torch.sum(torch.nn.functional.silu(x @ gates.T) * (x @ up_vectors.T), dim=1) + bias
        residual = prediction - y
        mse_loss = torch.mean(residual ** 2)
        tail_values = torch.topk(torch.abs(residual), k=tail_count, largest=True).values
        if tail_loss_kind == "smoothmax":
            tail_loss = torch.logsumexp(tail_values * tail_smoothness, dim=0) / tail_smoothness
        else:
            tail_loss = torch.mean(tail_values ** 2)
        loss = mse_loss + tail_weight * tail_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([gates, up_vectors, bias], 8.0)
        optimizer.step()
        if _step == 0 or (_step + 1) % 80 == 0:
            with torch.no_grad():
                current_prediction = torch.sum(torch.nn.functional.silu(x @ gates.T) * (x @ up_vectors.T), dim=1) + bias
                score = float(torch.max(torch.abs(current_prediction - y)).detach().cpu().item())
                if best_state is None or score < best_state[0]:
                    best_state = (
                        score,
                        gates.detach().cpu().numpy().astype(np.float64),
                        up_vectors.detach().cpu().numpy().astype(np.float64),
                        float(bias.detach().cpu().item()),
                    )
    if best_state is None:
        return candidate
    _tail, gate_values, up_values, bias_value = best_state
    activation_np = silu_np(coords @ gate_values.T) * (coords @ up_values.T)
    direct_prediction = activation_np.sum(axis=1) + bias_value
    direct_metrics = regression_metrics(targets, direct_prediction)
    selected_up_values = up_values
    selected_bias_value = bias_value
    prediction_np = direct_prediction
    calibration_mode = "direct"
    if tail_calibration in {"ls", "best-worst"}:
        design = np.column_stack([activation_np, np.ones(coords.shape[0], dtype=np.float64)])
        try:
            coeffs, *_ = np.linalg.lstsq(design, targets, rcond=None)
        except np.linalg.LinAlgError:
            coeffs = None
        if coeffs is not None and np.all(np.isfinite(coeffs)):
            ls_up_values = up_values * coeffs[:-1, None]
            ls_bias_value = float(coeffs[-1])
            ls_prediction = design @ coeffs
            ls_metrics = regression_metrics(targets, ls_prediction)
            use_ls = tail_calibration == "ls" or (
                tail_calibration == "best-worst"
                and (
                    ls_metrics["worstError"] < direct_metrics["worstError"]
                    or (
                        abs(ls_metrics["worstError"] - direct_metrics["worstError"]) <= 1e-12
                        and ls_metrics["mae"] <= direct_metrics["mae"]
                    )
                )
            )
            if use_ls:
                selected_up_values = ls_up_values
                selected_bias_value = ls_bias_value
                prediction_np = ls_prediction
                calibration_mode = "ls"
    mae = float(np.mean(np.abs(prediction_np - targets)))
    mse = float(np.mean((prediction_np - targets) ** 2))
    worst = float(np.max(np.abs(prediction_np - targets)))
    previous_worst = float(candidate.get("worstError", float("inf")))
    if mae >= float(candidate["score"]) and worst >= previous_worst:
        return candidate
    polished = dict(candidate)
    polished.update({
        "mae": mae,
        "mse": mse,
        "score": mae,
        "worstError": worst,
        "gates": gate_values,
        "ups": selected_up_values,
        "scales": np.ones(atom_count, dtype=np.float64),
        "bias": selected_bias_value,
        "tailErrorPolished": True,
        "tailErrorPolishSteps": int(steps),
        "tailErrorPolishWeight": float(tail_weight),
        "tailErrorPolishFraction": float(tail_fraction),
        "tailErrorPolishCount": int(tail_count),
        "tailErrorPolishLearningRate": float(tail_lr),
        "tailErrorPolishLoss": tail_loss_kind,
        "tailErrorPolishSmoothness": float(tail_smoothness),
        "tailErrorPolishCalibration": calibration_mode,
        "tailErrorPolishMaeBefore": float(candidate["mae"]),
    })
    return polished


def final_lbfgs_polish_swiglu_candidate(
    coords: np.ndarray,
    targets: np.ndarray,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    import torch

    gates_np = np.asarray(candidate["gates"], dtype=np.float64)
    ups_np = np.asarray(candidate["ups"], dtype=np.float64)
    scales_np = np.asarray(candidate["scales"], dtype=np.float64)
    if gates_np.ndim != 2 or ups_np.shape != gates_np.shape or scales_np.shape[0] != gates_np.shape[0]:
        return candidate
    max_iter = int(os.environ.get("GCL_BLIND_SWIGLU_FINAL_LBFGS_STEPS", "0") or "0")
    if max_iter <= 0:
        return candidate
    tail_weight = float(os.environ.get("GCL_BLIND_SWIGLU_FINAL_LBFGS_TAIL_WEIGHT", "0.0") or "0.0")
    tail_fraction = float(os.environ.get("GCL_BLIND_SWIGLU_FINAL_LBFGS_TAIL_FRACTION", "0.20") or "0.20")
    if not np.isfinite(tail_weight) or tail_weight < 0.0:
        raise RuntimeError("GCL_BLIND_SWIGLU_FINAL_LBFGS_TAIL_WEIGHT must be non-negative and finite")
    if not np.isfinite(tail_fraction) or tail_fraction <= 0.0 or tail_fraction > 1.0:
        raise RuntimeError("GCL_BLIND_SWIGLU_FINAL_LBFGS_TAIL_FRACTION must be in (0, 1]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fit_dtype = torch.float32
    x = torch.tensor(coords, dtype=fit_dtype, device=device)
    y = torch.tensor(targets, dtype=fit_dtype, device=device)
    gates = torch.nn.Parameter(torch.tensor(gates_np, dtype=fit_dtype, device=device))
    up_vectors = torch.nn.Parameter(torch.tensor(ups_np * scales_np[:, None], dtype=fit_dtype, device=device))
    bias = torch.nn.Parameter(torch.tensor(float(candidate["bias"]), dtype=fit_dtype, device=device))
    tail_count = max(64, min(coords.shape[0], int(coords.shape[0] * tail_fraction), gates_np.shape[0] * 192))
    optimizer = torch.optim.LBFGS(
        [gates, up_vectors, bias],
        lr=float(os.environ.get("GCL_BLIND_SWIGLU_FINAL_LBFGS_LR", "0.35") or "0.35"),
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-8,
        tolerance_change=1e-10,
        history_size=int(os.environ.get("GCL_BLIND_SWIGLU_FINAL_LBFGS_HISTORY", "80") or "80"),
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        prediction = torch.sum(torch.nn.functional.silu(x @ gates.T) * (x @ up_vectors.T), dim=1) + bias
        residual = prediction - y
        loss = torch.mean(residual ** 2)
        if tail_weight > 0.0:
            tail_values = torch.topk(torch.abs(residual), k=tail_count, largest=True).values
            loss = loss + tail_weight * torch.mean(tail_values ** 2)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except RuntimeError:
        return candidate

    gate_values = gates.detach().cpu().numpy().astype(np.float64)
    up_values = up_vectors.detach().cpu().numpy().astype(np.float64)
    bias_value = float(bias.detach().cpu().item())
    activation_np = silu_np(coords @ gate_values.T) * (coords @ up_values.T)
    direct_prediction = activation_np.sum(axis=1) + bias_value
    direct_metrics = regression_metrics(targets, direct_prediction)
    selected_up_values = up_values
    selected_bias_value = bias_value
    prediction_np = direct_prediction
    calibration_mode = "direct"

    design = np.column_stack([activation_np, np.ones(coords.shape[0], dtype=np.float64)])
    try:
        coeffs, *_ = np.linalg.lstsq(design, targets, rcond=None)
    except np.linalg.LinAlgError:
        coeffs = None
    if coeffs is not None and np.all(np.isfinite(coeffs)):
        ls_up_values = up_values * coeffs[:-1, None]
        ls_bias_value = float(coeffs[-1])
        ls_prediction = design @ coeffs
        ls_metrics = regression_metrics(targets, ls_prediction)
        if (
            ls_metrics["mae"] < direct_metrics["mae"]
            or (
                abs(ls_metrics["mae"] - direct_metrics["mae"]) <= 1e-12
                and ls_metrics["worstError"] < direct_metrics["worstError"]
            )
        ):
            selected_up_values = ls_up_values
            selected_bias_value = ls_bias_value
            prediction_np = ls_prediction
            calibration_mode = "ls"

    metrics = regression_metrics(targets, prediction_np)
    previous_mae = float(candidate.get("mae", candidate.get("score", float("inf"))))
    previous_worst = float(candidate.get("worstError", float("inf")))
    if metrics["mae"] >= previous_mae and metrics["worstError"] >= previous_worst:
        return candidate
    polished = dict(candidate)
    polished.update({
        "mae": float(metrics["mae"]),
        "mse": float(metrics["mse"]),
        "score": float(metrics["mae"]),
        "worstError": float(metrics["worstError"]),
        "gates": gate_values,
        "ups": selected_up_values,
        "scales": np.ones(gates_np.shape[0], dtype=np.float64),
        "bias": selected_bias_value,
        "finalLbfgsPolished": True,
        "finalLbfgsSteps": int(max_iter),
        "finalLbfgsTailWeight": float(tail_weight),
        "finalLbfgsTailFraction": float(tail_fraction),
        "finalLbfgsTailCount": int(tail_count),
        "finalLbfgsCalibration": calibration_mode,
        "finalLbfgsMaeBefore": previous_mae,
        "finalLbfgsWorstBefore": previous_worst,
    })
    return polished


def canonical_direction(vector: np.ndarray) -> np.ndarray | None:
    direction = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9 or not np.all(np.isfinite(direction)):
        return None
    direction = direction / norm
    pivot = int(np.argmax(np.abs(direction)))
    if direction[pivot] < 0.0:
        direction = -direction
    return direction


def append_diverse_direction(
    directions: list[np.ndarray],
    candidate: np.ndarray,
    max_abs_cosine: float,
) -> None:
    direction = canonical_direction(candidate)
    if direction is None:
        return
    if any(abs(float(np.dot(direction, existing))) >= max_abs_cosine for existing in directions):
        return
    directions.append(direction)


def make_gradient_gate_directions(
    gradient_targets: np.ndarray | None,
    atom_count: int,
    seed: int,
    max_candidates: int,
) -> list[np.ndarray]:
    if gradient_targets is None or gradient_targets.shape[0] < max(4, atom_count):
        return []
    rng = np.random.default_rng(seed)
    gradients = np.asarray(gradient_targets, dtype=np.float64)
    norms = np.linalg.norm(gradients, axis=1)
    valid = np.where(np.isfinite(norms) & (norms > 1e-9))[0]
    if valid.size == 0:
        return []
    directions: list[np.ndarray] = []

    centered = gradients[valid] - np.mean(gradients[valid], axis=0, keepdims=True)
    try:
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
        for row in vh[:min(vh.shape[0], atom_count * 2)]:
            append_diverse_direction(directions, row, 0.995)
            append_diverse_direction(directions, -row, 0.995)
    except np.linalg.LinAlgError:
        pass

    for index in valid[np.argsort(norms[valid])[::-1]]:
        append_diverse_direction(directions, gradients[index], 0.995)
        if len(directions) >= max_candidates:
            return directions[:max_candidates]

    pair_count = min(max_candidates * 8, max(0, valid.size * 3))
    for _ in range(pair_count):
        left, right = rng.choice(valid, size=2, replace=False)
        append_diverse_direction(directions, gradients[left] - gradients[right], 0.995)
        append_diverse_direction(directions, gradients[left] + gradients[right], 0.995)
        if len(directions) >= max_candidates:
            break
    return directions[:max_candidates]


def solve_fixed_gate_start(
    coords: np.ndarray,
    targets: np.ndarray,
    gates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float] | None:
    if gates.shape[0] == 0:
        return None
    blocks = [silu_np(coords @ gate)[:, None] * coords for gate in gates]
    design = np.column_stack(blocks + [np.ones(coords.shape[0], dtype=np.float64)])
    try:
        coeffs, *_ = np.linalg.lstsq(design, targets, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(coeffs)):
        return None
    rank = coords.shape[1]
    raw_up_values = coeffs[:-1].reshape(gates.shape[0], rank).astype(np.float64)
    up_values = np.zeros_like(raw_up_values)
    scales = np.zeros(gates.shape[0], dtype=np.float64)
    for index, raw_up in enumerate(raw_up_values):
        up_norm = float(np.linalg.norm(raw_up))
        if up_norm > 1e-12:
            up_values[index] = raw_up / up_norm
            scales[index] = up_norm
    bias = float(coeffs[-1])
    prediction = design @ coeffs
    mae = float(np.mean(np.abs(prediction - targets)))
    return gates.astype(np.float64), up_values, scales, bias, mae


def make_gradient_structured_starts(
    coords: np.ndarray,
    targets: np.ndarray,
    gradient_targets: np.ndarray | None,
    gate_seed_directions: np.ndarray | None,
    atom_count: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    if atom_count < 4:
        return []
    rank = coords.shape[1]
    if rank <= 0:
        return []
    rng = np.random.default_rng(seed + 331)
    directions: list[np.ndarray] = []

    # Geometry-cover starts are deliberately oracle-only: they use only active
    # coordinates, finite-difference witnesses, and oracle-scored residuals.
    basis = np.eye(rank, dtype=np.float64)
    for row in basis:
        append_diverse_direction(directions, row, 0.998)
    for i in range(rank):
        for j in range(i + 1, rank):
            append_diverse_direction(directions, basis[i] + basis[j], 0.998)
            append_diverse_direction(directions, basis[i] - basis[j], 0.998)

    for _ in range(max(atom_count * 6, rank * 8, 64)):
        append_diverse_direction(directions, rng.normal(size=rank), 0.998)
        append_diverse_direction(directions, rng.choice([-1.0, 1.0], size=rank), 0.998)
    for _ in range(max(2, atom_count // 2)):
        random_matrix = rng.normal(size=(rank, rank))
        try:
            q, _r = np.linalg.qr(random_matrix)
            for row in q.T:
                append_diverse_direction(directions, row, 0.998)
        except np.linalg.LinAlgError:
            pass

    centered_coords = coords - np.mean(coords, axis=0, keepdims=True)
    try:
        _u, _s, vh = np.linalg.svd(centered_coords, full_matrices=False)
        for row in vh[:rank]:
            append_diverse_direction(directions, row, 0.998)
    except np.linalg.LinAlgError:
        pass

    target_centered = targets.astype(np.float64) - float(np.mean(targets))
    try:
        linear_coeffs, *_ = np.linalg.lstsq(coords, target_centered, rcond=None)
        append_diverse_direction(directions, linear_coeffs, 0.998)
    except np.linalg.LinAlgError:
        pass

    directions = make_gradient_gate_directions(
        gradient_targets,
        atom_count,
        seed,
        max_candidates=max(atom_count * 12, 96),
    ) + directions
    if gate_seed_directions is not None and gate_seed_directions.shape[0] > 0:
        for row in gate_seed_directions:
            append_diverse_direction(directions, row, 0.995)
    if len(directions) < atom_count:
        return []
    sample_count = min(coords.shape[0], max(4096, atom_count * 384))
    if sample_count < coords.shape[0]:
        sample_indices = np.sort(rng.choice(coords.shape[0], size=sample_count, replace=False))
        xs = coords[sample_indices]
        ys = targets[sample_indices]
    else:
        xs = coords
        ys = targets
    gate_norms = np.array([0.18, 0.28, 0.42, 0.65, 0.95, 1.35, 1.9, 2.7], dtype=np.float64)
    scored: list[tuple[float, np.ndarray]] = []
    ones = np.ones(xs.shape[0], dtype=np.float64)
    for direction in directions:
        for norm_value in gate_norms:
            gate = direction * norm_value
            block = silu_np(xs @ gate)[:, None] * xs
            design = np.column_stack([block, ones])
            try:
                coeffs, *_ = np.linalg.lstsq(design, ys, rcond=None)
            except np.linalg.LinAlgError:
                continue
            if not np.all(np.isfinite(coeffs)):
                continue
            predicted = design @ coeffs
            scored.append((float(np.mean(np.abs(predicted - ys))), gate))
    if len(scored) < atom_count:
        return []
    scored.sort(key=lambda item: item[0])

    def solve_selected(selected_gates: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
        if len(selected_gates) != atom_count:
            return None
        solved_start = solve_fixed_gate_start(coords, targets, np.vstack(selected_gates))
        if solved_start is None:
            return None
        gates, ups, scales, bias, _mae = solved_start
        return gates, ups, scales, bias

    def greedy_residual_selected(
        candidate_pool: list[np.ndarray],
        max_abs_cosine: float,
        pool_limit: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
        pool = candidate_pool[:min(len(candidate_pool), pool_limit)]
        if len(pool) < atom_count:
            return None
        selected_gates: list[np.ndarray] = []
        selected_indices: set[int] = set()
        residual = ys.astype(np.float64) - float(np.mean(ys))
        solved: np.ndarray | None = None
        for _atom_index in range(atom_count):
            best_choice: tuple[float, int] | None = None
            for index, gate in enumerate(pool):
                if index in selected_indices:
                    continue
                direction = canonical_direction(gate)
                if direction is None:
                    continue
                if any(abs(float(np.dot(direction, existing / max(1e-9, np.linalg.norm(existing))))) >= max_abs_cosine for existing in selected_gates):
                    continue
                block = silu_np(xs @ gate)[:, None] * xs
                try:
                    coeffs, *_ = np.linalg.lstsq(block, residual, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                if not np.all(np.isfinite(coeffs)):
                    continue
                projection = block @ coeffs
                improvement = float(np.dot(residual, residual) - np.dot(residual - projection, residual - projection))
                if best_choice is None or improvement > best_choice[0]:
                    best_choice = (improvement, index)
            if best_choice is None or best_choice[0] <= 1e-12:
                return None
            selected_indices.add(best_choice[1])
            selected_gates.append(pool[best_choice[1]])
            blocks = [silu_np(xs @ gate)[:, None] * xs for gate in selected_gates]
            design = np.column_stack(blocks + [np.ones(xs.shape[0], dtype=np.float64)])
            try:
                solved, *_ = np.linalg.lstsq(design, ys, rcond=None)
            except np.linalg.LinAlgError:
                return None
            if not np.all(np.isfinite(solved)):
                return None
            residual = ys - design @ solved
        return solve_selected(selected_gates)

    selected: list[np.ndarray] = []
    for _score, gate in scored:
        direction = canonical_direction(gate)
        if direction is None:
            continue
        if any(abs(float(np.dot(direction, existing / max(1e-9, np.linalg.norm(existing))))) >= 0.992 for existing in selected):
            continue
        selected.append(gate)
        if len(selected) >= atom_count:
            break
    if len(selected) < atom_count:
        return []

    starts: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    scored_pool = [gate for _score, gate in scored]
    for max_abs_cosine, pool_limit in ((0.999, 192), (0.995, 256), (0.985, 320), (0.965, 384)):
        greedy_start = greedy_residual_selected(scored_pool, max_abs_cosine, pool_limit)
        if greedy_start is not None:
            starts.append(greedy_start)

    for norm_value in (0.42, 0.65, 0.95, 1.35):
        for _cover_index in range(4):
            try:
                q, _r = np.linalg.qr(rng.normal(size=(rank, rank)))
            except np.linalg.LinAlgError:
                continue
            gates_cover = q.T[:atom_count] * norm_value
            solved_cover = solve_selected([gate for gate in gates_cover])
            if solved_cover is not None:
                starts.append(solved_cover)
            if rank >= atom_count * 2:
                mixed = []
                for atom_index in range(atom_count):
                    direction = q[:, atom_index] + q[:, atom_index + atom_count]
                    mixed.append(normalize(direction) * norm_value)
                solved_mixed = solve_selected(mixed)
                if solved_mixed is not None:
                    starts.append(solved_mixed)

    solved = solve_selected(selected)
    if solved is not None:
        starts.append(solved)

    for scale in (0.55, 0.75, 1.15, 1.45):
        scaled = np.vstack([gate * scale for gate in selected])
        solved_scaled = solve_selected([gate for gate in scaled])
        if solved_scaled is not None:
            starts.append(solved_scaled)

    axes_pool = [gate for _score, gate in scored[:min(len(scored), max(atom_count * 12, 96))]]
    for stride in (1, 2, 3, 5, 7):
        chosen: list[np.ndarray] = []
        for index in range(stride - 1, len(axes_pool), stride):
            gate = axes_pool[index]
            direction = canonical_direction(gate)
            if direction is None:
                continue
            if any(abs(float(np.dot(direction, existing / max(1e-9, np.linalg.norm(existing))))) >= 0.985 for existing in chosen):
                continue
            chosen.append(gate)
            if len(chosen) >= atom_count:
                break
        solved_stride = solve_selected(chosen)
        if solved_stride is not None:
            starts.append(solved_stride)

    if len(scored) >= atom_count * 2:
        pool = [gate for _score, gate in scored[:min(len(scored), atom_count * 8)]]
        for _restart_index in range(8):
            chosen: list[np.ndarray] = []
            for gate in rng.permutation(pool):
                direction = canonical_direction(gate)
                if direction is None:
                    continue
                if any(abs(float(np.dot(direction, existing / max(1e-9, np.linalg.norm(existing))))) >= 0.99 for existing in chosen):
                    continue
                chosen.append(gate)
                if len(chosen) >= atom_count:
                    break
            solved_random = solve_selected(chosen)
            if solved_random is not None:
                starts.append(solved_random)

    unique_starts: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    signatures: set[tuple[float, ...]] = set()
    for gates, ups, scales, bias in starts:
        gate_dirs = []
        for gate in gates:
            direction = canonical_direction(gate)
            if direction is not None:
                gate_dirs.extend(np.round(direction, 3).tolist())
        signature = tuple(gate_dirs[:min(len(gate_dirs), atom_count * rank)])
        if signature in signatures:
            continue
        signatures.add(signature)
        unique_starts.append((gates, ups, scales, bias))
    return unique_starts[:18]


def expand_structured_start_variants(
    coords: np.ndarray,
    targets: np.ndarray,
    starts: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    atom_count: int,
    seed: int,
    max_starts: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    if not starts or max_starts <= 0:
        return starts
    rng = np.random.default_rng(seed)
    scored: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, float]] = []
    signatures: set[tuple[float, ...]] = set()

    def add_gates(raw_gates: np.ndarray) -> None:
        gates = np.asarray(raw_gates, dtype=np.float64)
        if gates.shape != (atom_count, coords.shape[1]) or not np.all(np.isfinite(gates)):
            return
        normalized_rows = []
        for gate in gates:
            direction = canonical_direction(gate)
            if direction is None:
                return
            normalized_rows.extend(np.round(direction, 3).tolist())
        signature = tuple(normalized_rows)
        if signature in signatures:
            return
        signatures.add(signature)
        solved = solve_fixed_gate_start(coords, targets, gates)
        if solved is None:
            return
        solved_gates, solved_ups, solved_scales, solved_bias, mae = solved
        scored.append((mae, solved_gates, solved_ups, solved_scales, solved_bias))

    for gates, _ups, _scales, _bias in starts:
        gates = np.asarray(gates, dtype=np.float64)
        add_gates(gates)
        for scale in (0.72, 0.88, 1.12, 1.35):
            add_gates(gates * scale)
        gate_norms = np.linalg.norm(gates, axis=1, keepdims=True).clip(min=1e-9)
        gate_dirs = gates / gate_norms
        for jitter in (0.025, 0.055, 0.09, 0.14):
            noise = rng.normal(size=gates.shape)
            noise -= np.sum(noise * gate_dirs, axis=1, keepdims=True) * gate_dirs
            noise_norm = np.linalg.norm(noise, axis=1, keepdims=True).clip(min=1e-9)
            perturbed = gate_dirs + jitter * (noise / noise_norm)
            perturbed /= np.linalg.norm(perturbed, axis=1, keepdims=True).clip(min=1e-9)
            add_gates(perturbed * gate_norms)
        for _ in range(2):
            shuffled = gates.copy()
            rng.shuffle(shuffled, axis=0)
            add_gates(shuffled)

    if not scored:
        return starts
    scored.sort(key=lambda item: item[0])
    return [(gates, ups, scales, bias) for _mae, gates, ups, scales, bias in scored[:max_starts]]


def fit_swiglu_sum(
    coords: np.ndarray,
    targets: np.ndarray,
    atom_count: int,
    seed: int,
    steps: int,
    learning_rate: float,
    restart_override: int | None = None,
    gradient_coords: np.ndarray | None = None,
    gradient_targets: np.ndarray | None = None,
    gate_seed_directions: np.ndarray | None = None,
    gradient_weight: float = 0.0,
    fit_mode: str = "auto",
    restart_chunk_size: int = 0,
) -> dict[str, Any]:
    if atom_count == 1:
        single = fit_swiglu_atom(coords, targets, seed, steps, learning_rate)
        return {
            "mae": single["mae"],
            "restart": single["restart"],
            "gates": np.expand_dims(single["gate"], axis=0),
            "ups": np.expand_dims(single["up"], axis=0),
            "scales": np.array([single["scale"]], dtype=np.float64),
            "bias": single["bias"],
            "restartCount": int(single.get("restartCount", 1)),
            "restartChunkSize": None,
            "fitMode": "single-atom",
        }

    import torch

    torch.manual_seed(seed)
    fit_dtype = torch.float64 if atom_count >= 4 else torch.float32
    x = torch.tensor(coords, dtype=fit_dtype)
    y = torch.tensor(targets, dtype=fit_dtype)
    has_gradient_witness = (
        gradient_coords is not None
        and gradient_targets is not None
        and gradient_weight > 0.0
        and gradient_coords.shape[0] > 0
    )
    if has_gradient_witness:
        gx = torch.tensor(gradient_coords, dtype=fit_dtype)
        gy = torch.tensor(gradient_targets, dtype=fit_dtype)
    else:
        gx = None
        gy = None
    rank = coords.shape[1]
    target_scale = float(np.std(targets)) or 1.0
    target_variance = float(np.mean((targets - np.mean(targets)) ** 2))
    restart_count = max(64, atom_count * 32)
    if atom_count >= 4:
        restart_count = max(restart_count, atom_count * 48)
    if restart_override is not None:
        restart_count = max(1, restart_override)
    best: dict[str, Any] | None = None
    structured_starts: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    if atom_count > 1:
        residual = targets.astype(np.float64).copy()
        residual_bias = float(np.mean(residual))
        residual -= residual_bias
        residual_gates: list[np.ndarray] = []
        residual_ups: list[np.ndarray] = []
        residual_scales: list[float] = []
        residual_steps = max(240, min(700, steps // 4))
        try:
            for atom_index in range(atom_count):
                single = fit_swiglu_atom(
                    coords,
                    residual,
                    seed + 1009 + atom_index * 131,
                    residual_steps,
                    learning_rate,
                )
                component = predict_swiglu(coords, single)
                residual_bias += float(single["bias"])
                residual_gates.append(single["gate"])
                residual_ups.append(single["up"])
                residual_scales.append(float(single["scale"]))
                residual -= component
            structured_starts.append((
                np.vstack(residual_gates),
                np.vstack(residual_ups),
                np.array(residual_scales, dtype=np.float64),
                residual_bias,
            ))
        except Exception:
            structured_starts = []

    gradient_starts = make_gradient_structured_starts(
        coords,
        targets,
        gradient_targets,
        gate_seed_directions,
        atom_count,
        seed + 2029,
    )
    quadratic_multi_starts = make_quadratic_multi_structured_starts(
        coords,
        targets,
        atom_count,
        seed + 3037,
    )
    structured_starts.extend(quadratic_multi_starts)
    structured_starts.extend(gradient_starts)
    if structured_starts and atom_count >= 4:
        structured_starts = expand_structured_start_variants(
            coords,
            targets,
            structured_starts,
            atom_count,
            seed + 4051,
            max_starts=max(12, min(24, atom_count + 2)),
        )

    if fit_mode in {"auto", "batched"} and atom_count >= 4:
        try:
            return fit_swiglu_sum_batched(
                coords,
                targets,
                atom_count,
                seed,
                steps,
                learning_rate,
                restart_count,
                structured_starts,
                gradient_coords,
                gradient_targets,
                gradient_weight,
                restart_chunk_size,
            )
        except Exception:
            if fit_mode == "batched":
                raise

    structured_start_count = len(structured_starts)
    total_restart_count = restart_count + structured_start_count
    best_score_by_start_kind: dict[str, float] = {}
    best_metrics_by_start_kind: dict[str, dict[str, Any]] = {}
    for restart in range(total_restart_count):
        torch.manual_seed(seed + restart)
        if restart < structured_start_count:
            gate_init, up_init, scale_init, bias_init = structured_starts[restart]
            gates = torch.nn.Parameter(torch.tensor(gate_init, dtype=fit_dtype))
            ups = torch.nn.Parameter(torch.tensor(up_init, dtype=fit_dtype))
            scales = torch.nn.Parameter(torch.tensor(scale_init, dtype=fit_dtype))
            bias = torch.nn.Parameter(torch.tensor(bias_init, dtype=fit_dtype))
        else:
            gates = torch.nn.Parameter(torch.randn(atom_count, rank, dtype=fit_dtype) * 0.6)
            ups = torch.nn.Parameter(torch.randn(atom_count, rank, dtype=fit_dtype) * 0.6)
            scales = torch.nn.Parameter(torch.ones(atom_count, dtype=fit_dtype) * max(0.1, target_scale / max(1, atom_count)))
            bias = torch.nn.Parameter(torch.tensor(float(np.mean(targets)), dtype=fit_dtype))
        optimizer = torch.optim.Adam([gates, ups, scales, bias], lr=learning_rate)
        for _ in range(steps):
            optimizer.zero_grad()
            activations = torch.nn.functional.silu(x @ gates.T) * (x @ ups.T)
            prediction = activations @ scales + bias
            loss = torch.mean((prediction - y) ** 2)
            if has_gradient_witness and gx is not None and gy is not None:
                gate_projection = gx @ gates.T
                up_projection = gx @ ups.T
                sigmoid = torch.sigmoid(gate_projection)
                silu_values = torch.nn.functional.silu(gate_projection)
                silu_prime = sigmoid * (1.0 + gate_projection * (1.0 - sigmoid))
                gradient_terms = scales[None, :, None] * (
                    silu_prime[:, :, None] * up_projection[:, :, None] * gates[None, :, :]
                    + silu_values[:, :, None] * ups[None, :, :]
                )
                predicted_gradients = torch.sum(gradient_terms, dim=1)
                target_grad_scale = torch.mean(gy ** 2).detach().clamp_min(torch.tensor(1e-12, dtype=fit_dtype))
                loss = loss + gradient_weight * torch.mean((predicted_gradients - gy) ** 2) / target_grad_scale
            loss.backward()
            optimizer.step()

        lbfgs = torch.optim.LBFGS(
            [gates, ups, scales, bias],
            lr=0.7,
            max_iter=420 if atom_count >= 4 else 260,
            line_search_fn="strong_wolfe"
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad()
            activations = torch.nn.functional.silu(x @ gates.T) * (x @ ups.T)
            prediction = activations @ scales + bias
            loss = torch.mean((prediction - y) ** 2)
            if has_gradient_witness and gx is not None and gy is not None:
                gate_projection = gx @ gates.T
                up_projection = gx @ ups.T
                sigmoid = torch.sigmoid(gate_projection)
                silu_values = torch.nn.functional.silu(gate_projection)
                silu_prime = sigmoid * (1.0 + gate_projection * (1.0 - sigmoid))
                gradient_terms = scales[None, :, None] * (
                    silu_prime[:, :, None] * up_projection[:, :, None] * gates[None, :, :]
                    + silu_values[:, :, None] * ups[None, :, :]
                )
                predicted_gradients = torch.sum(gradient_terms, dim=1)
                target_grad_scale = torch.mean(gy ** 2).detach().clamp_min(torch.tensor(1e-12, dtype=fit_dtype))
                loss = loss + gradient_weight * torch.mean((predicted_gradients - gy) ** 2) / target_grad_scale
            loss.backward()
            return loss

        try:
            lbfgs.step(closure)
        except RuntimeError:
            pass

        with torch.no_grad():
            activations = torch.nn.functional.silu(x @ gates.T) * (x @ ups.T)
            activation_np = activations.detach().cpu().numpy().astype(np.float64)
            design = np.column_stack([activation_np, np.ones(coords.shape[0], dtype=np.float64)])
            try:
                coeffs, *_ = np.linalg.lstsq(design, targets, rcond=None)
                if np.all(np.isfinite(coeffs)):
                    scale_values = coeffs[:-1].astype(np.float64)
                    bias_value = float(coeffs[-1])
                    prediction_np = design @ coeffs
                    mae = float(np.mean(np.abs(prediction_np - targets)))
                    mse = float(np.mean((prediction_np - targets) ** 2))
                else:
                    raise np.linalg.LinAlgError("non-finite least-squares coefficients")
            except np.linalg.LinAlgError:
                prediction = activations @ scales + bias
                mae = torch.mean(torch.abs(prediction - y)).item()
                mse = torch.mean((prediction - y) ** 2).item()
                scale_values = scales.detach().cpu().numpy().astype(np.float64)
                bias_value = float(bias.detach().cpu().item())
            score = mae
            r2 = 1.0 - mse / max(target_variance, 1e-12)
            candidate = {
                "restart": restart,
                "mae": mae,
                "mse": mse,
                "r2": r2,
                "score": score,
                "startKind": "structured" if restart < structured_start_count else "random",
                "randomRestartIndex": restart - structured_start_count if restart >= structured_start_count else None,
                "gates": gates.detach().cpu().numpy().astype(np.float64),
                "ups": ups.detach().cpu().numpy().astype(np.float64),
                "scales": scale_values,
                "bias": bias_value,
            }
            start_kind = str(candidate["startKind"])
            previous_kind_score = best_score_by_start_kind.get(start_kind)
            if previous_kind_score is None or score < previous_kind_score:
                best_score_by_start_kind[start_kind] = score
                best_metrics_by_start_kind[start_kind] = {
                    "restart": int(restart),
                    "randomRestartIndex": None if candidate["randomRestartIndex"] is None else int(candidate["randomRestartIndex"]),
                    "mae": mae,
                    "mse": mse,
                    "r2": r2,
                }
            if best is None or candidate["score"] < best["score"]:
                best = candidate
    assert best is not None
    best["restartCount"] = total_restart_count
    best["randomRestartCount"] = restart_count
    best["structuredStartCount"] = structured_start_count
    best["bestScoreByStartKind"] = {key: float(value) for key, value in best_score_by_start_kind.items()}
    best["bestMetricsByStartKind"] = best_metrics_by_start_kind
    best["fitMode"] = "sequential+structured-starts" if structured_start_count else "sequential"
    return best


def predict_swiglu(coords: np.ndarray, fit: dict[str, Any]) -> np.ndarray:
    if "gates" in fit:
        activations = silu_np(coords @ fit["gates"].T) * (coords @ fit["ups"].T)
        return activations @ fit["scales"] + fit["bias"]
    return fit["scale"] * silu_np(coords @ fit["gate"]) * (coords @ fit["up"]) + fit["bias"]


def minimax_bias_polish_swiglu_candidate(
    coords: np.ndarray,
    targets: np.ndarray,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    prediction = predict_swiglu(coords, candidate)
    residual = targets - prediction
    if residual.shape[0] == 0 or not np.all(np.isfinite(residual)):
        return candidate
    shift = (float(np.max(residual)) + float(np.min(residual))) / 2.0
    if not np.isfinite(shift) or abs(shift) < 1e-15:
        return candidate
    polished = dict(candidate)
    polished["bias"] = float(candidate["bias"]) + shift
    polished_prediction = prediction + shift
    polished_residual = targets - polished_prediction
    mae = float(np.mean(np.abs(polished_residual)))
    mse = float(np.mean(polished_residual ** 2))
    worst = float(np.max(np.abs(polished_residual)))
    previous_worst = float(candidate.get("worstError", float(np.max(np.abs(residual)))))
    if worst > previous_worst:
        return candidate
    polished.update({
        "mae": mae,
        "mse": mse,
        "score": mae,
        "worstError": worst,
        "minimaxBiasPolished": True,
        "minimaxBiasShift": float(shift),
        "minimaxBiasWorstBefore": previous_worst,
    })
    return polished


def chebyshev_calibrate_swiglu_candidate(
    coords: np.ndarray,
    targets: np.ndarray,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    try:
        from scipy.optimize import linprog
    except Exception:
        return candidate
    gates_np = np.asarray(candidate["gates"], dtype=np.float64)
    ups_np = np.asarray(candidate["ups"], dtype=np.float64)
    if gates_np.ndim != 2 or ups_np.shape != gates_np.shape:
        return candidate
    features = silu_np(coords @ gates_np.T) * (coords @ ups_np.T)
    if features.shape[0] == 0 or not np.all(np.isfinite(features)):
        return candidate
    sample_limit = int(os.environ.get("GCL_BLIND_SWIGLU_CHEBYSHEV_SAMPLE_LIMIT", "12000") or "12000")
    if sample_limit > 0 and features.shape[0] > sample_limit:
        rng = np.random.default_rng(int(os.environ.get("GCL_BLIND_SWIGLU_CHEBYSHEV_SEED", "91703") or "91703"))
        indices = np.sort(rng.choice(features.shape[0], size=sample_limit, replace=False))
        fit_features = features[indices]
        fit_targets = targets[indices]
    else:
        fit_features = features
        fit_targets = targets
    row_count, atom_count = fit_features.shape
    # Variables are atom scales, bias, and non-negative max error t.
    objective = np.zeros(atom_count + 2, dtype=np.float64)
    objective[-1] = 1.0
    upper = np.zeros((row_count * 2, atom_count + 2), dtype=np.float64)
    upper[:row_count, :atom_count] = fit_features
    upper[:row_count, atom_count] = 1.0
    upper[:row_count, -1] = -1.0
    upper[row_count:, :atom_count] = -fit_features
    upper[row_count:, atom_count] = -1.0
    upper[row_count:, -1] = -1.0
    bounds = [(None, None)] * (atom_count + 1) + [(0.0, None)]
    rhs = np.concatenate([fit_targets, -fit_targets]).astype(np.float64)
    try:
        result = linprog(
            objective,
            A_ub=upper,
            b_ub=rhs,
            bounds=bounds,
            method="highs",
            options={"presolve": True},
        )
    except Exception:
        return candidate
    if not result.success or result.x is None or not np.all(np.isfinite(result.x)):
        return candidate
    scales = result.x[:atom_count].astype(np.float64)
    bias = float(result.x[atom_count])
    prediction = features @ scales + bias
    metrics = regression_metrics(targets, prediction)
    previous_worst = float(candidate.get("worstError", float("inf")))
    previous_mae = float(candidate.get("mae", float("inf")))
    # Chebyshev calibration is allowed to trade tiny MAE for lower max error,
    # but not to materially degrade average fit.
    max_mae_ratio = float(os.environ.get("GCL_BLIND_SWIGLU_CHEBYSHEV_MAX_MAE_RATIO", "1.08") or "1.08")
    if metrics["worstError"] > previous_worst or metrics["mae"] > previous_mae * max_mae_ratio:
        return candidate
    calibrated = dict(candidate)
    calibrated.update({
        "mae": float(metrics["mae"]),
        "mse": float(metrics["mse"]),
        "score": float(metrics["mae"]),
        "worstError": float(metrics["worstError"]),
        "scales": scales,
        "bias": bias,
        "chebyshevCalibrated": True,
        "chebyshevObjective": float(result.fun),
        "chebyshevSampleCount": int(row_count),
        "chebyshevWorstBefore": previous_worst,
    })
    return calibrated


def robust_scalar_calibrate_swiglu_candidate(
    coords: np.ndarray,
    targets: np.ndarray,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    import torch

    gates_np = np.asarray(candidate["gates"], dtype=np.float64)
    ups_np = np.asarray(candidate["ups"], dtype=np.float64)
    scales_np = np.asarray(candidate["scales"], dtype=np.float64)
    if gates_np.ndim != 2 or ups_np.shape != gates_np.shape or scales_np.shape[0] != gates_np.shape[0]:
        return candidate
    features_np = silu_np(coords @ gates_np.T) * (coords @ ups_np.T)
    if features_np.shape[0] == 0 or not np.all(np.isfinite(features_np)):
        return candidate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    features = torch.tensor(features_np, dtype=dtype, device=device)
    y = torch.tensor(targets, dtype=dtype, device=device)
    scales = torch.nn.Parameter(torch.tensor(scales_np, dtype=dtype, device=device))
    bias = torch.nn.Parameter(torch.tensor(float(candidate["bias"]), dtype=dtype, device=device))
    steps = int(os.environ.get("GCL_BLIND_SWIGLU_SCALAR_CALIBRATION_STEPS", "900") or "900")
    lr = float(os.environ.get("GCL_BLIND_SWIGLU_SCALAR_CALIBRATION_LR", "0.01") or "0.01")
    tail_weight = float(os.environ.get("GCL_BLIND_SWIGLU_SCALAR_CALIBRATION_TAIL_WEIGHT", "1.0") or "1.0")
    tail_fraction = float(os.environ.get("GCL_BLIND_SWIGLU_SCALAR_CALIBRATION_TAIL_FRACTION", "0.20") or "0.20")
    if steps <= 0 or not np.isfinite(lr) or lr <= 0.0:
        return candidate
    tail_count = max(64, min(features_np.shape[0], int(features_np.shape[0] * tail_fraction)))
    optimizer = torch.optim.AdamW([scales, bias], lr=lr, weight_decay=1e-8)
    best_state: tuple[float, np.ndarray, float] | None = None
    for step_index in range(steps):
        optimizer.zero_grad()
        prediction = features @ scales + bias
        residual = prediction - y
        mse_loss = torch.mean(residual ** 2)
        tail_values = torch.topk(torch.abs(residual), k=tail_count, largest=True).values
        loss = mse_loss + tail_weight * torch.mean(tail_values ** 2)
        loss.backward()
        optimizer.step()
        if step_index == 0 or (step_index + 1) % 60 == 0:
            with torch.no_grad():
                current_prediction = features @ scales + bias
                score = float(torch.max(torch.abs(current_prediction - y)).detach().cpu().item())
                if best_state is None or score < best_state[0]:
                    best_state = (
                        score,
                        scales.detach().cpu().numpy().astype(np.float64),
                        float(bias.detach().cpu().item()),
                    )
    if best_state is None:
        return candidate
    _score, calibrated_scales, calibrated_bias = best_state
    prediction_np = features_np @ calibrated_scales + calibrated_bias
    metrics = regression_metrics(targets, prediction_np)
    previous_worst = float(candidate.get("worstError", float("inf")))
    previous_mae = float(candidate.get("mae", float("inf")))
    max_mae_ratio = float(os.environ.get("GCL_BLIND_SWIGLU_SCALAR_CALIBRATION_MAX_MAE_RATIO", "1.08") or "1.08")
    if metrics["worstError"] > previous_worst or metrics["mae"] > previous_mae * max_mae_ratio:
        return candidate
    calibrated = dict(candidate)
    calibrated.update({
        "mae": float(metrics["mae"]),
        "mse": float(metrics["mse"]),
        "score": float(metrics["mae"]),
        "worstError": float(metrics["worstError"]),
        "scales": calibrated_scales,
        "bias": calibrated_bias,
        "robustScalarCalibrated": True,
        "robustScalarCalibrationSteps": int(steps),
        "robustScalarCalibrationTailWeight": float(tail_weight),
        "robustScalarCalibrationTailFraction": float(tail_fraction),
        "robustScalarCalibrationWorstBefore": previous_worst,
    })
    return calibrated


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = actual - predicted
    mae = float(np.mean(np.abs(residual)))
    mse = float(np.mean(residual ** 2))
    variance = float(np.mean((actual - np.mean(actual)) ** 2))
    r2 = 1.0 - mse / max(variance, 1e-12)
    worst = float(np.max(np.abs(residual)))
    return {"mae": mae, "mse": mse, "r2": r2, "worstError": worst}


def passes_metric_thresholds(
    metrics: dict[str, float],
    min_r2: float,
    max_mae: float,
    max_worst_error: float,
) -> bool:
    return (
        metrics["r2"] >= min_r2
        and metrics["mae"] <= max_mae
        and metrics["worstError"] <= max_worst_error
    )


def log_progress(args: argparse.Namespace, phase: str, **fields: Any) -> None:
    if not getattr(args, "progress", False):
        return
    print(json.dumps({"type": "progress", "phase": phase, **fields}), file=sys.stderr, flush=True)


def validation_counterexample_inputs(
    rng: np.random.Generator,
    validation_inputs: np.ndarray,
    validation_coords: np.ndarray,
    validation_targets: np.ndarray,
    validation_prediction: np.ndarray,
    basis: np.ndarray,
    counterexample_count: int,
    repeat_count: int,
    jitter_count: int,
    jitter_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    if counterexample_count <= 0:
        return (
            np.zeros((0, validation_inputs.shape[1]), dtype=np.float32),
            np.zeros(0, dtype=np.float64),
            np.zeros((0, validation_inputs.shape[1]), dtype=np.float32),
            0,
            0,
        )
    residual = validation_targets - validation_prediction
    order = np.argsort(np.abs(residual))[::-1]
    selected_count = min(counterexample_count, validation_inputs.shape[0])
    selected = order[:selected_count]
    repeats = max(1, repeat_count)
    hard_inputs = np.repeat(validation_inputs[selected], repeats, axis=0).astype(np.float32, copy=False)
    hard_targets = np.repeat(validation_targets[selected], repeats, axis=0).astype(np.float64, copy=False)

    if jitter_count <= 0 or jitter_scale <= 0.0:
        return hard_inputs, hard_targets, np.zeros((0, validation_inputs.shape[1]), dtype=np.float32), selected_count, 0
    base_indices = np.resize(selected, jitter_count)
    jitter_coords = validation_coords[base_indices] + rng.normal(
        0.0,
        jitter_scale,
        size=(jitter_count, validation_coords.shape[1]),
    )
    jitter_inputs = (jitter_coords @ basis).astype(np.float32)
    return (
        hard_inputs,
        hard_targets,
        jitter_inputs,
        selected_count,
        jitter_count,
    )


def make_tail_probe_polish_set(
    rng: np.random.Generator,
    oracle: OracleClient,
    basis: np.ndarray,
    fit: dict[str, Any],
    dimension: int,
    count: int,
    scale: float,
    repeat: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    if count <= 0:
        return (
            np.zeros((0, basis.shape[0]), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            0,
            0,
        )
    repeats = max(1, repeat)
    probe_inputs = rng.normal(0.0, scale, size=(count, dimension)).astype(np.float32)
    probe_targets = oracle.eval(probe_inputs)
    probe_coords = probe_inputs @ basis.T
    probe_prediction = predict_swiglu(probe_coords, fit)
    residual = np.abs(probe_targets - probe_prediction)
    selected_count = min(max(32, int(np.sqrt(count)) * 4), count)
    selected = np.argsort(residual)[-selected_count:]
    hard_coords = np.repeat(probe_coords[selected], repeats, axis=0).astype(np.float64, copy=False)
    hard_targets = np.repeat(probe_targets[selected], repeats, axis=0).astype(np.float64, copy=False)
    return hard_coords, hard_targets, selected_count, count


def principal_cosines(basis: np.ndarray, truth_vectors: np.ndarray) -> np.ndarray:
    q1, _ = np.linalg.qr(basis.T)
    q2, _ = np.linalg.qr(truth_vectors.T)
    _, singular_values, _ = np.linalg.svd(q1.T @ q2, full_matrices=False)
    return singular_values


def run_solver(args: argparse.Namespace) -> int:
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    if output_md:
        output_md.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="gcl-blind-swiglu-") as raw_tmp:
        work_dir = Path(raw_tmp)
        oracle = OracleClient(args, work_dir)
        try:
            dimension = int(oracle.meta["hiddenSize"])
            unit_count = int(oracle.meta.get("unitCount", 1))
            log_progress(args, "oracle-ready", dimension=dimension, unitCount=unit_count, layer=int(oracle.meta["layer"]))
            if args.atom_count is None:
                args.atom_count = unit_count
            if args.atom_count != unit_count:
                raise RuntimeError(f"atom_count={args.atom_count} does not match oracle unit_count={unit_count}")
            rng = np.random.default_rng(args.seed)
            independent_rng = os.environ.get("GCL_BLIND_SWIGLU_INDEPENDENT_RNG", "").strip() == "1"
            basis_rng = np.random.default_rng(args.seed + 101) if independent_rng else rng
            cubic_rng = np.random.default_rng(args.seed + 202) if independent_rng else rng
            train_rng = np.random.default_rng(args.seed + 303) if independent_rng else rng
            fit_gradient_rng = np.random.default_rng(args.seed + 404) if independent_rng else rng
            validation_rng = np.random.default_rng(args.seed + 505) if independent_rng else rng
            finite_difference_steps = parse_step_list(args.finite_difference_steps, args.finite_difference_step)
            fit_gradient_steps = parse_step_list(
                args.fit_gradient_steps,
                args.fit_gradient_step if args.fit_gradient_step > 0 else args.finite_difference_step,
            )
            train_scale_mixture = parse_float_list(args.train_scale_mixture)
            gradients = []
            for _ in range(args.probe_points):
                base = basis_rng.normal(0.0, args.input_scale, size=dimension).astype(np.float32)
                for step in finite_difference_steps:
                    if args.gradient_mode == "coordinate":
                        gradients.append(estimate_coordinate_gradient(oracle, base, step))
                    elif args.gradient_mode == "forward-coordinate":
                        gradients.append(estimate_forward_coordinate_gradient(oracle, base, step))
                    else:
                        gradients.append(estimate_gradient(oracle, basis_rng, base, args.direction_count, step))
            gradient_matrix = np.vstack(gradients)
            basis_moment = args.basis_moment
            if basis_moment == "auto" and args.gradient_mode == "forward-coordinate":
                basis_moment = "raw"
            basis, eigenvalues, energy_ratio = recover_basis(gradient_matrix, args.active_rank, basis_moment)
            gradient_witness_count = int(gradient_matrix.shape[0])
            log_progress(
                args,
                "basis-recovered",
                activeRank=int(args.active_rank),
                gradientWitnessCount=gradient_witness_count,
                energyRatio=float(energy_ratio),
            )
            cubic_gate_directions, cubic_scores = estimate_active_cubic_directions(
                oracle,
                cubic_rng,
                basis,
                args.cubic_direction_count,
                args.cubic_step,
            )

            train_inputs = sample_scale_mixture_inputs(
                train_rng,
                args.train_count,
                dimension,
                args.input_scale,
                train_scale_mixture,
            )
            train_targets = oracle.eval(train_inputs)
            train_coords = train_inputs @ basis.T
            log_progress(args, "train-sampled", trainCount=int(train_coords.shape[0]), rawOracleCalls=int(oracle.query_count))
            fit_gradient_coords, fit_gradient_targets = estimate_active_forward_gradients(
                oracle,
                fit_gradient_rng,
                basis,
                args.fit_gradient_count,
                args.input_scale,
                fit_gradient_steps,
            )
            fit = fit_swiglu_sum(
                train_coords,
                train_targets,
                args.atom_count,
                args.seed + 17,
                args.fit_steps,
                args.fit_learning_rate,
                args.fit_restarts if args.fit_restarts > 0 else None,
                fit_gradient_coords,
                fit_gradient_targets,
                cubic_gate_directions,
                args.fit_gradient_weight,
                args.fit_mode,
                args.fit_restart_chunk_size,
            )
            log_progress(
                args,
                "initial-fit-complete",
                fitTrainMae=float(fit["mae"]),
                fitRestarts=int(fit["restartCount"]),
                selectedStartKind=str(fit.get("startKind", "unknown")),
            )
            tail_probe_selected = 0
            tail_probe_count = 0
            if args.tail_probe_count > 0:
                tail_probe_scale = args.tail_probe_scale if args.tail_probe_scale > 0.0 else args.validation_scale
                tail_probe_rng = np.random.default_rng(args.seed + 99017)
                tail_probe_coords, tail_probe_targets, tail_probe_selected, tail_probe_count = make_tail_probe_polish_set(
                    tail_probe_rng,
                    oracle,
                    basis,
                    fit,
                    dimension,
                    args.tail_probe_count,
                    tail_probe_scale,
                    args.tail_probe_repeat,
                )
                if tail_probe_coords.shape[0] > 0:
                    polish_coords = np.vstack([train_coords, tail_probe_coords])
                    polish_targets = np.concatenate([train_targets, tail_probe_targets])
                    fit = tail_error_polish_swiglu_candidate(
                        polish_coords,
                        polish_targets,
                        fit,
                        max(1, int(os.environ.get("GCL_BLIND_SWIGLU_TAIL_PROBE_POLISH_STEPS", "420") or "420")),
                    )
                    log_progress(
                        args,
                        "tail-probe-polish-complete",
                        probeCount=int(tail_probe_count),
                        selected=int(tail_probe_selected),
                        trainCount=int(polish_coords.shape[0]),
                        rawOracleCalls=int(oracle.query_count),
                    )

            validation_inputs = validation_rng.normal(0.0, args.validation_scale, size=(args.validation_count, dimension)).astype(np.float32)
            validation_targets = oracle.eval(validation_inputs)
            validation_coords = validation_inputs @ basis.T
            validation_prediction = predict_swiglu(validation_coords, fit)
            metrics = regression_metrics(validation_targets, validation_prediction)
            log_progress(
                args,
                "validation-complete",
                r2=float(metrics["r2"]),
                mae=float(metrics["mae"]),
                worstError=float(metrics["worstError"]),
                rawOracleCalls=int(oracle.query_count),
            )
            fit_attempts = [{
                "stage": "initial",
                "trainCount": int(train_coords.shape[0]),
                "fitGradientCount": int(fit_gradient_coords.shape[0]),
                "fitSteps": int(args.fit_steps),
                "fitRestarts": int(fit["restartCount"]),
                "fitRandomRestarts": int(fit.get("randomRestartCount", fit["restartCount"])),
                "fitStructuredStarts": int(fit.get("structuredStartCount", 0)),
                "selectedStartKind": str(fit.get("startKind", "unknown")),
                "selectedRandomRestartIndex": fit.get("randomRestartIndex"),
                "bestScoreByStartKind": fit.get("bestScoreByStartKind", {}),
                "bestMetricsByStartKind": fit.get("bestMetricsByStartKind", {}),
                "fitMode": str(fit.get("fitMode", "unknown")),
                "fitTrainMae": float(fit["mae"]),
                "fitSelectionScore": float(fit.get("selectionScore", fit.get("score", fit["mae"]))),
                "tailProbeCount": int(tail_probe_count),
                "tailProbeSelected": int(tail_probe_selected),
                "tailProbeRepeat": int(args.tail_probe_repeat),
                "validationR2": float(metrics["r2"]),
                "validationMae": float(metrics["mae"]),
                "validationWorstError": float(metrics["worstError"]),
                "rawOracleCalls": int(oracle.query_count),
            }]

            if (
                not passes_metric_thresholds(metrics, args.min_r2, args.max_mae, args.max_worst_error)
                and not args.disable_adaptive_rescue
                and args.atom_count >= 8
                and oracle.query_count < args.max_oracle_calls
            ):
                log_progress(args, "adaptive-rescue-start", rawOracleCalls=int(oracle.query_count))
                rescue_train_default = max(args.train_count, args.atom_count * 2000)
                rescue_gradient_default = max(args.fit_gradient_count, args.atom_count * 32)
                rescue_steps_default = max(args.fit_steps, 5200 if args.atom_count >= 16 else 4200)
                rescue_restarts_default = max(args.fit_restarts if args.fit_restarts > 0 else 0, 1024 if args.atom_count >= 16 else 768)
                rescue_train_target = int(os.environ.get(
                    "GCL_BLIND_SWIGLU_RESCUE_TRAIN_TARGET",
                    str(args.rescue_train_target if args.rescue_train_target > 0 else rescue_train_default),
                ) or rescue_train_default)
                rescue_gradient_target = int(os.environ.get(
                    "GCL_BLIND_SWIGLU_RESCUE_GRADIENT_TARGET",
                    str(args.rescue_gradient_target if args.rescue_gradient_target > 0 else rescue_gradient_default),
                ) or rescue_gradient_default)
                rescue_steps = int(os.environ.get(
                    "GCL_BLIND_SWIGLU_RESCUE_STEPS",
                    str(args.rescue_steps if args.rescue_steps > 0 else rescue_steps_default),
                ) or rescue_steps_default)
                rescue_restarts = int(os.environ.get(
                    "GCL_BLIND_SWIGLU_RESCUE_RESTARTS",
                    str(args.rescue_restarts if args.rescue_restarts > 0 else rescue_restarts_default),
                ) or rescue_restarts_default)
                rescue_train_target = max(args.train_count, rescue_train_target)
                rescue_gradient_target = max(args.fit_gradient_count, rescue_gradient_target)
                rescue_steps = max(1, rescue_steps)
                rescue_restarts = max(1, rescue_restarts)
                counterexample_selected = 0
                counterexample_jittered = 0
                counterexample_repeat = 0
                if args.counterexample_count > 0:
                    hard_inputs, hard_targets, jitter_inputs, counterexample_selected, counterexample_jittered = validation_counterexample_inputs(
                        train_rng,
                        validation_inputs,
                        validation_coords,
                        validation_targets,
                        validation_prediction,
                        basis,
                        args.counterexample_count,
                        args.counterexample_repeat,
                        args.counterexample_jitter_count,
                        args.counterexample_jitter_scale,
                    )
                    counterexample_repeat = max(1, args.counterexample_repeat)
                    cex_inputs = hard_inputs
                    cex_targets = hard_targets
                    if jitter_inputs.shape[0] > 0:
                        jitter_targets = oracle.eval(jitter_inputs)
                        cex_inputs = np.vstack([cex_inputs, jitter_inputs])
                        cex_targets = np.concatenate([cex_targets, jitter_targets])
                    if cex_inputs.shape[0] > 0:
                        train_inputs = np.vstack([train_inputs, cex_inputs])
                        train_targets = np.concatenate([train_targets, cex_targets])
                        train_coords = train_inputs @ basis.T
                        log_progress(
                            args,
                            "counterexamples-added",
                            selected=int(counterexample_selected),
                            repeat=int(counterexample_repeat),
                            jittered=int(counterexample_jittered),
                            trainCount=int(train_coords.shape[0]),
                            rawOracleCalls=int(oracle.query_count),
                        )
                extra_train = max(0, rescue_train_target - train_coords.shape[0])
                if extra_train > 0:
                    extra_inputs = sample_scale_mixture_inputs(
                        train_rng,
                        extra_train,
                        dimension,
                        args.input_scale,
                        train_scale_mixture,
                    )
                    extra_targets = oracle.eval(extra_inputs)
                    train_inputs = np.vstack([train_inputs, extra_inputs])
                    train_targets = np.concatenate([train_targets, extra_targets])
                    train_coords = train_inputs @ basis.T
                    log_progress(args, "rescue-train-expanded", trainCount=int(train_coords.shape[0]), rawOracleCalls=int(oracle.query_count))
                extra_gradient = max(0, rescue_gradient_target - fit_gradient_coords.shape[0])
                if extra_gradient > 0:
                    extra_coords, extra_targets = estimate_active_forward_gradients(
                        oracle,
                        fit_gradient_rng,
                        basis,
                        extra_gradient,
                        args.input_scale,
                        fit_gradient_steps,
                    )
                    fit_gradient_coords = np.vstack([fit_gradient_coords, extra_coords])
                    fit_gradient_targets = np.vstack([fit_gradient_targets, extra_targets])
                    log_progress(args, "rescue-gradients-expanded", fitGradientCount=int(fit_gradient_coords.shape[0]), rawOracleCalls=int(oracle.query_count))
                log_progress(
                    args,
                    "adaptive-rescue-fit-start",
                    trainCount=int(train_coords.shape[0]),
                    fitGradientCount=int(fit_gradient_coords.shape[0]),
                    fitSteps=int(rescue_steps),
                    fitRestarts=int(rescue_restarts),
                    rawOracleCalls=int(oracle.query_count),
                )
                rescue_fit = fit_swiglu_sum(
                    train_coords,
                    train_targets,
                    args.atom_count,
                    args.seed + 7919,
                    rescue_steps,
                    args.fit_learning_rate,
                    rescue_restarts,
                    fit_gradient_coords,
                    fit_gradient_targets,
                    cubic_gate_directions,
                    args.fit_gradient_weight,
                    args.fit_mode,
                    args.fit_restart_chunk_size,
                )
                rescue_prediction = predict_swiglu(validation_coords, rescue_fit)
                rescue_metrics = regression_metrics(validation_targets, rescue_prediction)
                log_progress(
                    args,
                    "adaptive-rescue-complete",
                    r2=float(rescue_metrics["r2"]),
                    mae=float(rescue_metrics["mae"]),
                    worstError=float(rescue_metrics["worstError"]),
                    rawOracleCalls=int(oracle.query_count),
                )
                fit_attempts.append({
                    "stage": "adaptive-rescue",
                    "trainCount": int(train_coords.shape[0]),
                    "fitGradientCount": int(fit_gradient_coords.shape[0]),
                    "fitSteps": int(rescue_steps),
                    "fitRestarts": int(rescue_fit["restartCount"]),
                    "fitRandomRestarts": int(rescue_fit.get("randomRestartCount", rescue_fit["restartCount"])),
                    "fitStructuredStarts": int(rescue_fit.get("structuredStartCount", 0)),
                    "selectedStartKind": str(rescue_fit.get("startKind", "unknown")),
                    "selectedRandomRestartIndex": rescue_fit.get("randomRestartIndex"),
                    "bestScoreByStartKind": rescue_fit.get("bestScoreByStartKind", {}),
                    "bestMetricsByStartKind": rescue_fit.get("bestMetricsByStartKind", {}),
                    "fitMode": str(rescue_fit.get("fitMode", "unknown")),
                    "fitTrainMae": float(rescue_fit["mae"]),
                    "fitSelectionScore": float(rescue_fit.get("selectionScore", rescue_fit.get("score", rescue_fit["mae"]))),
                    "validationR2": float(rescue_metrics["r2"]),
                    "validationMae": float(rescue_metrics["mae"]),
                    "validationWorstError": float(rescue_metrics["worstError"]),
                    "counterexampleSelected": int(counterexample_selected),
                    "counterexampleRepeat": int(counterexample_repeat),
                    "counterexampleJittered": int(counterexample_jittered),
                    "rawOracleCalls": int(oracle.query_count),
                })
                if (
                    passes_metric_thresholds(rescue_metrics, args.min_r2, args.max_mae, args.max_worst_error)
                    or rescue_metrics["mae"] < metrics["mae"]
                ):
                    fit = rescue_fit
                    metrics = rescue_metrics

            truth_gate, truth_up = oracle.truth()
            if truth_gate.ndim == 1:
                truth_gate = np.expand_dims(truth_gate, axis=0)
            if truth_up.ndim == 1:
                truth_up = np.expand_dims(truth_up, axis=0)
            truth_basis = np.vstack([normalize(row) for row in truth_gate] + [normalize(row) for row in truth_up])
            cosines = principal_cosines(basis, truth_basis)
            recovered_gates = np.vstack([normalize(row @ basis) for row in fit["gates"]])
            recovered_ups = np.vstack([normalize(row @ basis) for row in fit["ups"]])
            source_alignment = {
                "gateMaxAbsCosine": float(max(abs(float(np.dot(recovered, truth))) for recovered in recovered_gates for truth in truth_basis)),
                "upMaxAbsCosine": float(max(abs(float(np.dot(recovered, truth))) for recovered in recovered_ups for truth in truth_basis)),
                "subspaceMinPrincipalCosine": float(np.min(cosines)),
                "subspaceCosines": [float(value) for value in cosines],
            }

            pass_gate = (
                metrics["r2"] >= args.min_r2
                and metrics["mae"] <= args.max_mae
                and metrics["worstError"] <= args.max_worst_error
                and source_alignment["subspaceMinPrincipalCosine"] >= args.min_subspace_cosine
                and oracle.query_count <= args.max_oracle_calls
                and args.atom_count == unit_count
                and oracle.meta.get("solverAccess") == "oracle-only"
            )
            report = {
                "summary": {
                    "pass": pass_gate,
                    "model": oracle.meta["model"],
                    "modelType": oracle.meta["modelType"],
                    "dimension": dimension,
                    "layer": oracle.meta["layer"],
                    "unit": oracle.meta["unit"],
                    "units": oracle.meta.get("units", [oracle.meta["unit"]]),
                    "unitCount": unit_count,
                    "activeRank": args.active_rank,
                    "atomCount": args.atom_count,
                    "rawOracleCalls": oracle.query_count,
                    "metrics": metrics,
                    "sourceAlignment": source_alignment,
                    "selectedGradientEnergyRatio": float(energy_ratio),
                },
                "accessAudit": {
                    "solverAccess": "oracle-only",
                    "weightsUsedBySolver": False,
                    "hooksUsedBySolver": False,
                    "sourceIdsUsedBySolver": False,
                    "oracleOwnsWeights": True,
                    "postHocWitnessOnly": True,
                    "processBoundary": "solver parent communicates with oracle subprocess via npy batch files and JSONL commands",
                },
                "solver": {
                    "kind": "blind-active-subspace-explicit-swiglu-sum",
                    "probePoints": args.probe_points,
                    "gradientMode": args.gradient_mode,
                    "basisMoment": basis_moment,
                    "directionCount": args.direction_count,
                    "gradientWitnessCount": gradient_witness_count,
                    "cubicDirectionCount": int(cubic_gate_directions.shape[0]),
                    "cubicStep": float(args.cubic_step),
                    "cubicMaxAbsScore": float(np.max(np.abs(cubic_scores))) if cubic_scores.size else 0.0,
                    "subspaceCount": args.subspace_count if args.subspace_count > 0 else None,
                    "finiteDifferenceStep": args.finite_difference_step,
                    "finiteDifferenceSteps": [float(step) for step in finite_difference_steps],
                    "independentRng": bool(independent_rng),
                    "trainScaleMixture": [float(value) for value in train_scale_mixture] if train_scale_mixture else None,
                    "trainCount": args.train_count,
                    "validationCount": args.validation_count,
                    "validationScale": args.validation_scale,
                    "gradientEigenvalues": [float(value) for value in eigenvalues],
                    "fitTrainMae": float(fit["mae"]),
                    "fitRestarts": int(fit["restartCount"]),
                    "fitRandomRestarts": int(fit.get("randomRestartCount", fit["restartCount"])),
                    "fitStructuredStarts": int(fit.get("structuredStartCount", 0)),
                    "selectedStartKind": str(fit.get("startKind", "unknown")),
                    "selectedRandomRestartIndex": fit.get("randomRestartIndex"),
                    "bestScoreByStartKind": fit.get("bestScoreByStartKind", {}),
                    "bestMetricsByStartKind": fit.get("bestMetricsByStartKind", {}),
                    "fitRestartChunkSize": None if fit.get("restartChunkSize") is None else int(fit["restartChunkSize"]),
                    "fitMode": str(fit.get("fitMode", "unknown")),
                    "randomInitScales": fit.get("randomInitScales", [0.6]),
                    "fitGradientCount": int(fit_gradient_coords.shape[0]),
                    "fitGradientSteps": [float(step) for step in fit_gradient_steps],
                    "fitGradientWeight": float(args.fit_gradient_weight),
                    "selectionTailWeight": float(fit.get("selectionTailWeight", 0.0)),
                    "selectionTailQuantile": float(fit.get("selectionTailQuantile", 0.995)),
                    "counterexampleCount": int(args.counterexample_count),
                    "counterexampleRepeat": int(args.counterexample_repeat),
                    "counterexampleJitterCount": int(args.counterexample_jitter_count),
                    "counterexampleJitterScale": float(args.counterexample_jitter_scale),
                    "tailProbeCount": int(args.tail_probe_count),
                    "tailProbeRepeat": int(args.tail_probe_repeat),
                    "tailProbeScale": float(args.tail_probe_scale if args.tail_probe_scale > 0.0 else args.validation_scale),
                    "tailProbeSelected": int(tail_probe_selected),
                    "fitSelectionScore": float(fit.get("selectionScore", fit.get("score", fit["mae"]))),
                    "chebyshevCalibrated": bool(fit.get("chebyshevCalibrated", False)),
                    "chebyshevSampleCount": int(fit.get("chebyshevSampleCount", 0)),
                    "chebyshevObjective": None if fit.get("chebyshevObjective") is None else float(fit["chebyshevObjective"]),
                    "robustScalarCalibrated": bool(fit.get("robustScalarCalibrated", False)),
                    "robustScalarCalibrationSteps": int(fit.get("robustScalarCalibrationSteps", 0)),
                    "fitScales": [float(value) for value in fit["scales"]],
                    "fitBias": float(fit["bias"]),
                    "fitAttempts": fit_attempts,
                },
                "oracleResponse": {
                    "quantizationStep": float(args.oracle_quantization_step),
                },
                "claimBoundary": [
                    "This gate proves selected real local SwiGLU unit outputs can be queried as a black-box oracle and recovered as an explicit sum of rank-2 SwiGLU atoms under the recorded budget.",
                    "It does not prove full FFN extraction, all-channel decompilation, exhaustive LLM circuit recovery, or semantic original-neuron identity from output-only access.",
                ],
            }
            output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            if output_md:
                output_md.write_text(
                    "\n".join([
                        "# Blind 4096D SwiGLU Oracle Decompiler",
                        "",
                        f"- pass: `{str(pass_gate).lower()}`",
                        f"- model: `{oracle.meta['model']}`",
                        f"- layer/unit: `{oracle.meta['layer']}/{oracle.meta.get('units', [oracle.meta['unit']])}`",
                        f"- raw oracle calls: `{oracle.query_count}`",
                        f"- validation R2: `{metrics['r2']}`",
                        f"- validation MAE: `{metrics['mae']}`",
                        f"- subspace min principal cosine: `{source_alignment['subspaceMinPrincipalCosine']}`",
                        "",
                        "Access audit:",
                        "",
                        "- solverAccess: `oracle-only`",
                        "- weightsUsedBySolver: `false`",
                        "- hooksUsedBySolver: `false`",
                        "- postHocWitnessOnly: `true`",
                        "",
                        "Boundary: this is a selected real SwiGLU-unit black-box gate, not full LLM decompilation.",
                        "",
                    ]),
                    encoding="utf-8",
                )
            print(json.dumps(report["summary"], indent=2))
            return 0 if pass_gate else 1
        finally:
            oracle.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Blind oracle-only recovery for one real 4096D SwiGLU unit.")
    parser.add_argument("--oracle-server", action="store_true")
    parser.add_argument("--model", default="unsloth/mistral-7b-instruct-v0.3-bnb-4bit")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--unit", type=int, default=0)
    parser.add_argument("--units", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--oracle-batch-size", type=int, default=2048)
    parser.add_argument("--oracle-quantization-step", type=float, default=0.0)
    parser.add_argument("--active-rank", type=int, default=2)
    parser.add_argument("--atom-count", type=int, default=None)
    parser.add_argument("--gradient-mode", choices=["coordinate", "forward-coordinate", "random-direction"], default="coordinate")
    parser.add_argument("--basis-moment", choices=["auto", "centered", "raw"], default="auto")
    parser.add_argument("--probe-points", type=int, default=2)
    parser.add_argument("--direction-count", type=int, default=384)
    parser.add_argument("--subspace-count", type=int, default=0)
    parser.add_argument("--cubic-direction-count", type=int, default=0)
    parser.add_argument("--cubic-step", type=float, default=0.35)
    parser.add_argument("--finite-difference-step", type=float, default=0.08)
    parser.add_argument("--finite-difference-steps", default="")
    parser.add_argument("--input-scale", type=float, default=0.45)
    parser.add_argument("--train-scale-mixture", default="")
    parser.add_argument("--validation-scale", type=float, default=0.65)
    parser.add_argument("--train-count", type=int, default=1600)
    parser.add_argument("--validation-count", type=int, default=512)
    parser.add_argument("--fit-steps", type=int, default=1200)
    parser.add_argument("--fit-learning-rate", type=float, default=0.025)
    parser.add_argument("--fit-restarts", type=int, default=0)
    parser.add_argument("--fit-restart-chunk-size", type=int, default=0)
    parser.add_argument("--fit-mode", choices=["auto", "sequential", "batched"], default="auto")
    parser.add_argument("--fit-gradient-count", type=int, default=0)
    parser.add_argument("--fit-gradient-step", type=float, default=0.0)
    parser.add_argument("--fit-gradient-steps", default="")
    parser.add_argument("--fit-gradient-weight", type=float, default=0.2)
    parser.add_argument("--counterexample-count", type=int, default=0)
    parser.add_argument("--counterexample-repeat", type=int, default=4)
    parser.add_argument("--counterexample-jitter-count", type=int, default=0)
    parser.add_argument("--counterexample-jitter-scale", type=float, default=0.025)
    parser.add_argument("--tail-probe-count", type=int, default=0)
    parser.add_argument("--tail-probe-repeat", type=int, default=4)
    parser.add_argument("--tail-probe-scale", type=float, default=0.0)
    parser.add_argument("--rescue-train-target", type=int, default=0)
    parser.add_argument("--rescue-gradient-target", type=int, default=0)
    parser.add_argument("--rescue-steps", type=int, default=0)
    parser.add_argument("--rescue-restarts", type=int, default=0)
    parser.add_argument("--disable-adaptive-rescue", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--seed", type=int, default=260620)
    parser.add_argument("--min-r2", type=float, default=0.985)
    parser.add_argument("--max-mae", type=float, default=0.015)
    parser.add_argument("--max-worst-error", type=float, default=1e30)
    parser.add_argument("--min-subspace-cosine", type=float, default=0.93)
    parser.add_argument("--max-oracle-calls", type=int, default=20000)
    parser.add_argument("--output-json", required=False)
    parser.add_argument("--output-md", required=False)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.oracle_server:
        return oracle_server(args)
    if not args.output_json:
        parser.error("--output-json is required outside --oracle-server mode")
    return run_solver(args)


if __name__ == "__main__":
    raise SystemExit(main())
