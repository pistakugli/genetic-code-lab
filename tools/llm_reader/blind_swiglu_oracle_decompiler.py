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

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    gates_init = torch.randn(restart_count, atom_count, rank, dtype=fit_dtype, device=device, generator=generator) * 0.6
    ups_init = torch.randn(restart_count, atom_count, rank, dtype=fit_dtype, device=device, generator=generator) * 0.6
    scales_init = torch.ones(restart_count, atom_count, dtype=fit_dtype, device=device) * max(0.1, target_scale / max(1, atom_count))
    bias_init = torch.ones(restart_count, dtype=fit_dtype, device=device) * float(np.mean(targets))
    for index, (gate_start, up_start, scale_start, bias_start) in enumerate(structured_starts[:restart_count]):
        gates_init[index] = torch.tensor(gate_start, dtype=fit_dtype, device=device)
        ups_init[index] = torch.tensor(up_start, dtype=fit_dtype, device=device)
        scales_init[index] = torch.tensor(scale_start, dtype=fit_dtype, device=device)
        bias_init[index] = float(bias_start)

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
    best: dict[str, Any] | None = None
    ones = np.ones(coords.shape[0], dtype=np.float64)
    for restart in range(restart_count):
        activation_np = silu_np(coords @ gates_np[restart].T) * (coords @ ups_np[restart].T)
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
        candidate = {
            "restart": restart,
            "mae": mae,
            "mse": mse,
            "gates": gates_np[restart],
            "ups": ups_np[restart],
            "scales": coeffs[:-1].astype(np.float64),
            "bias": float(coeffs[-1]),
        }
        if best is None or candidate["mae"] < best["mae"]:
            best = candidate
    if best is None:
        raise RuntimeError("batched SwiGLU fit did not produce a finite candidate")
    best["restartCount"] = restart_count
    best["fitMode"] = "batched-cuda"
    return best


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
    gradient_weight: float = 0.0,
    fit_mode: str = "auto",
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
            )
        except Exception:
            if fit_mode == "batched":
                raise

    for restart in range(restart_count):
        torch.manual_seed(seed + restart)
        if restart < len(structured_starts):
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
            candidate = {
                "restart": restart,
                "mae": mae,
                "mse": mse,
                "gates": gates.detach().cpu().numpy().astype(np.float64),
                "ups": ups.detach().cpu().numpy().astype(np.float64),
                "scales": scale_values,
                "bias": bias_value,
            }
            if best is None or candidate["mae"] < best["mae"]:
                best = candidate
    assert best is not None
    best["restartCount"] = restart_count
    best["fitMode"] = "sequential"
    return best


def predict_swiglu(coords: np.ndarray, fit: dict[str, Any]) -> np.ndarray:
    if "gates" in fit:
        activations = silu_np(coords @ fit["gates"].T) * (coords @ fit["ups"].T)
        return activations @ fit["scales"] + fit["bias"]
    return fit["scale"] * silu_np(coords @ fit["gate"]) * (coords @ fit["up"]) + fit["bias"]


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = actual - predicted
    mae = float(np.mean(np.abs(residual)))
    mse = float(np.mean(residual ** 2))
    variance = float(np.mean((actual - np.mean(actual)) ** 2))
    r2 = 1.0 - mse / max(variance, 1e-12)
    worst = float(np.max(np.abs(residual)))
    return {"mae": mae, "mse": mse, "r2": r2, "worstError": worst}


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
            if args.atom_count is None:
                args.atom_count = unit_count
            if args.atom_count != unit_count:
                raise RuntimeError(f"atom_count={args.atom_count} does not match oracle unit_count={unit_count}")
            rng = np.random.default_rng(args.seed)
            finite_difference_steps = parse_step_list(args.finite_difference_steps, args.finite_difference_step)
            fit_gradient_steps = parse_step_list(
                args.fit_gradient_steps,
                args.fit_gradient_step if args.fit_gradient_step > 0 else args.finite_difference_step,
            )
            gradients = []
            for _ in range(args.probe_points):
                base = rng.normal(0.0, args.input_scale, size=dimension).astype(np.float32)
                for step in finite_difference_steps:
                    if args.gradient_mode == "coordinate":
                        gradients.append(estimate_coordinate_gradient(oracle, base, step))
                    elif args.gradient_mode == "forward-coordinate":
                        gradients.append(estimate_forward_coordinate_gradient(oracle, base, step))
                    else:
                        gradients.append(estimate_gradient(oracle, rng, base, args.direction_count, step))
            gradient_matrix = np.vstack(gradients)
            basis_moment = args.basis_moment
            if basis_moment == "auto" and args.gradient_mode == "forward-coordinate":
                basis_moment = "raw"
            basis, eigenvalues, energy_ratio = recover_basis(gradient_matrix, args.active_rank, basis_moment)
            gradient_witness_count = int(gradient_matrix.shape[0])

            train_inputs = rng.normal(0.0, args.input_scale, size=(args.train_count, dimension)).astype(np.float32)
            train_targets = oracle.eval(train_inputs)
            train_coords = train_inputs @ basis.T
            fit_gradient_coords, fit_gradient_targets = estimate_active_forward_gradients(
                oracle,
                rng,
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
                args.fit_gradient_weight,
                args.fit_mode,
            )

            validation_inputs = rng.normal(0.0, args.validation_scale, size=(args.validation_count, dimension)).astype(np.float32)
            validation_targets = oracle.eval(validation_inputs)
            validation_coords = validation_inputs @ basis.T
            validation_prediction = predict_swiglu(validation_coords, fit)
            metrics = regression_metrics(validation_targets, validation_prediction)

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
                    "subspaceCount": args.subspace_count if args.subspace_count > 0 else None,
                    "finiteDifferenceStep": args.finite_difference_step,
                    "finiteDifferenceSteps": [float(step) for step in finite_difference_steps],
                    "trainCount": args.train_count,
                    "validationCount": args.validation_count,
                    "validationScale": args.validation_scale,
                    "gradientEigenvalues": [float(value) for value in eigenvalues],
                    "fitTrainMae": float(fit["mae"]),
                    "fitRestarts": int(fit["restartCount"]),
                    "fitMode": str(fit.get("fitMode", "unknown")),
                    "fitGradientCount": int(fit_gradient_coords.shape[0]),
                    "fitGradientSteps": [float(step) for step in fit_gradient_steps],
                    "fitGradientWeight": float(args.fit_gradient_weight),
                    "fitScales": [float(value) for value in fit["scales"]],
                    "fitBias": float(fit["bias"]),
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
    parser.add_argument("--finite-difference-step", type=float, default=0.08)
    parser.add_argument("--finite-difference-steps", default="")
    parser.add_argument("--input-scale", type=float, default=0.45)
    parser.add_argument("--validation-scale", type=float, default=0.65)
    parser.add_argument("--train-count", type=int, default=1600)
    parser.add_argument("--validation-count", type=int, default=512)
    parser.add_argument("--fit-steps", type=int, default=1200)
    parser.add_argument("--fit-learning-rate", type=float, default=0.025)
    parser.add_argument("--fit-restarts", type=int, default=0)
    parser.add_argument("--fit-mode", choices=["auto", "sequential", "batched"], default="auto")
    parser.add_argument("--fit-gradient-count", type=int, default=0)
    parser.add_argument("--fit-gradient-step", type=float, default=0.0)
    parser.add_argument("--fit-gradient-steps", default="")
    parser.add_argument("--fit-gradient-weight", type=float, default=0.2)
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
