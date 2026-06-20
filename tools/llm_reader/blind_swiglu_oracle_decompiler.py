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
    if args.unit < 0 or args.unit >= gate_weight.shape[0]:
        raise RuntimeError(f"unit {args.unit} outside SwiGLU intermediate range 0..{gate_weight.shape[0] - 1}")
    gate = gate_weight[args.unit].contiguous()
    up = up_weight[args.unit].contiguous()
    hidden_size = int(gate.numel())
    gate_norm = torch.linalg.vector_norm(gate).item()
    up_norm = torch.linalg.vector_norm(up).item()
    gate_unit = (gate / max(gate_norm, 1e-12)).numpy().astype(np.float32)
    up_unit = (up / max(up_norm, 1e-12)).numpy().astype(np.float32)

    print(json.dumps({
        "type": "ready",
        "model": args.model,
        "modelType": model_type,
        "layer": args.layer,
        "unit": args.unit,
        "hiddenSize": hidden_size,
        "intermediateSize": int(gate_weight.shape[0]),
        "device": args.device,
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
                    gate_projection = batch @ gate
                    up_projection = batch @ up
                    outputs.append((torch.nn.functional.silu(gate_projection) * up_projection).cpu().numpy())
            np.save(output_path, np.concatenate(outputs).astype(np.float32))
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
        if args.local_files_only:
            command.append("--local-files-only")
        self.process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[2],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            bufsize=1,
        )
        assert self.process.stdout is not None
        ready = json.loads(self.process.stdout.readline())
        if ready.get("type") != "ready":
            raise RuntimeError(f"oracle did not become ready: {ready}")
        self.meta = ready
        self.work_dir = work_dir
        self.query_count = 0
        self._counter = 0

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()
        response = json.loads(self.process.stdout.readline())
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
            self.request({"cmd": "stop"})
        finally:
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()


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


def recover_basis(gradients: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray, float]:
    # Centering n gradient witnesses caps the covariance rank at n - 1. For the
    # low-query path where n == rank, use raw second moments so a rank-2 SwiGLU
    # channel can still expose two independent active directions.
    if gradients.shape[0] <= rank:
        basis_source = gradients
    else:
        basis_source = gradients - np.mean(gradients, axis=0, keepdims=True)
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
            prediction = scale * torch.nn.functional.silu(x @ gate) * (x @ up) + bias
            mae = torch.mean(torch.abs(prediction - y)).item()
            candidate = {
                "restart": restart,
                "mae": mae,
                "gate": gate.detach().cpu().numpy().astype(np.float64),
                "up": up.detach().cpu().numpy().astype(np.float64),
                "scale": float(scale.detach().cpu().item()),
                "bias": float(bias.detach().cpu().item()),
            }
            if best is None or candidate["mae"] < best["mae"]:
                best = candidate
    assert best is not None
    return best


def predict_swiglu(coords: np.ndarray, fit: dict[str, Any]) -> np.ndarray:
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
            rng = np.random.default_rng(args.seed)
            gradients = []
            for _ in range(args.probe_points):
                base = rng.normal(0.0, args.input_scale, size=dimension).astype(np.float32)
                if args.gradient_mode == "coordinate":
                    gradients.append(estimate_coordinate_gradient(oracle, base, args.finite_difference_step))
                else:
                    gradients.append(estimate_gradient(oracle, rng, base, args.direction_count, args.finite_difference_step))
            gradient_matrix = np.vstack(gradients)
            basis, eigenvalues, energy_ratio = recover_basis(gradient_matrix, args.active_rank)

            train_inputs = rng.normal(0.0, args.input_scale, size=(args.train_count, dimension)).astype(np.float32)
            train_targets = oracle.eval(train_inputs)
            train_coords = train_inputs @ basis.T
            fit = fit_swiglu_atom(train_coords, train_targets, args.seed + 17, args.fit_steps, args.fit_learning_rate)

            validation_inputs = rng.normal(0.0, args.validation_scale, size=(args.validation_count, dimension)).astype(np.float32)
            validation_targets = oracle.eval(validation_inputs)
            validation_coords = validation_inputs @ basis.T
            validation_prediction = predict_swiglu(validation_coords, fit)
            metrics = regression_metrics(validation_targets, validation_prediction)

            truth_gate, truth_up = oracle.truth()
            truth_basis = np.vstack([normalize(truth_gate), normalize(truth_up)])
            cosines = principal_cosines(basis, truth_basis)
            recovered_gate = normalize(fit["gate"] @ basis)
            recovered_up = normalize(fit["up"] @ basis)
            source_alignment = {
                "gateMaxAbsCosine": float(max(abs(np.dot(recovered_gate, truth_gate)), abs(np.dot(recovered_gate, truth_up)))),
                "upMaxAbsCosine": float(max(abs(np.dot(recovered_up, truth_gate)), abs(np.dot(recovered_up, truth_up)))),
                "subspaceMinPrincipalCosine": float(np.min(cosines)),
                "subspaceCosines": [float(value) for value in cosines],
            }

            pass_gate = (
                metrics["r2"] >= args.min_r2
                and metrics["mae"] <= args.max_mae
                and source_alignment["subspaceMinPrincipalCosine"] >= args.min_subspace_cosine
                and oracle.query_count <= args.max_oracle_calls
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
                    "activeRank": args.active_rank,
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
                    "kind": "blind-active-subspace-explicit-swiglu-atom",
                    "probePoints": args.probe_points,
                    "gradientMode": args.gradient_mode,
                    "directionCount": args.direction_count,
                    "finiteDifferenceStep": args.finite_difference_step,
                    "trainCount": args.train_count,
                    "validationCount": args.validation_count,
                    "validationScale": args.validation_scale,
                    "gradientEigenvalues": [float(value) for value in eigenvalues],
                    "fitTrainMae": float(fit["mae"]),
                    "fitScale": float(fit["scale"]),
                    "fitBias": float(fit["bias"]),
                },
                "claimBoundary": [
                    "This gate proves one real local SwiGLU unit can be queried as a black-box oracle and recovered as a rank-2 explicit SwiGLU atom under the recorded budget.",
                    "It does not prove full Mistral FFN extraction, all-channel 4096D decompilation, exhaustive LLM circuit recovery, or semantic original-neuron identity from output-only access.",
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
                        f"- layer/unit: `{oracle.meta['layer']}/{oracle.meta['unit']}`",
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
                        "Boundary: this is a single real SwiGLU-unit black-box gate, not full LLM decompilation.",
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--oracle-batch-size", type=int, default=2048)
    parser.add_argument("--active-rank", type=int, default=2)
    parser.add_argument("--gradient-mode", choices=["coordinate", "random-direction"], default="coordinate")
    parser.add_argument("--probe-points", type=int, default=2)
    parser.add_argument("--direction-count", type=int, default=384)
    parser.add_argument("--finite-difference-step", type=float, default=0.08)
    parser.add_argument("--input-scale", type=float, default=0.45)
    parser.add_argument("--validation-scale", type=float, default=0.65)
    parser.add_argument("--train-count", type=int, default=1600)
    parser.add_argument("--validation-count", type=int, default=512)
    parser.add_argument("--fit-steps", type=int, default=1200)
    parser.add_argument("--fit-learning-rate", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=260620)
    parser.add_argument("--min-r2", type=float, default=0.985)
    parser.add_argument("--max-mae", type=float, default=0.015)
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
